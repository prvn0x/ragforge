"""Regression tests for the round-6 review: schema errors are not ValueErrors, every request has an
output-token ceiling, a per-day quota 429 is not a retryable rate limit, transport errors carry
no reachable httpx exception (whose request holds the key header), thought signatures are size
capped, and an enum with no type is inferred as STRING only when it lists strings."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest
import respx
from pydantic import SecretStr

from module_g1.g1_support import (
    CANARY_KEY,
    MODEL,
    STREAM_URL,
    USER,
    answered,
    body,
    call_chunk,
    collect,
    consume_into,
    finish_chunk,
    google_error,
    ok,
    sse,
    text_chunk,
)
from ragforge_core.adapters.llm import chat_gemini
from ragforge_core.adapters.llm.chat_gemini import (
    ANSWER_MAX_OUTPUT_TOKENS,
    SKIP_SIGNATURE,
    STRUCTURED_MAX_OUTPUT_TOKENS,
    GeminiChatModel,
    ModelConfigurationError,
    OutputLimitReached,
    QuotaExhausted,
    StreamInterrupted,
    UnsupportedSchema,
    to_gemini_schema,
)
from ragforge_core.adapters.llm.chat_ollama import ChatModelError, OutputTruncated
from ragforge_core.domain.errors import RagforgeError, UpstreamRateLimited
from ragforge_core.domain.models import (
    ChatMessage,
    ChatStreamEvent,
    Role,
    ToolCall,
    ToolSpec,
)

SEARCH = ToolSpec(
    name="search_notes",
    description="Search the notes.",
    parameters={"type": "object", "properties": {"query": {"type": "string"}}},
)
VERDICT: dict[str, object] = {
    "type": "object",
    "properties": {"verdict": {"type": "string", "enum": ["yes", "no"]}},
}


def _adapter(http: httpx.AsyncClient, budget: int | None) -> GeminiChatModel:
    return GeminiChatModel(http, api_key=SecretStr(CANARY_KEY), model=MODEL, thinking_budget=budget)


# ---------------------------------------------------------------------- UnsupportedSchema
def test_unsupported_schema_is_a_loud_programming_error_not_a_value_error() -> None:
    """Grounding catches ValueError as "the model sent unparseable JSON": a schema bug must not
    be swallowed into a malformed-verdict retry, nor read as a provider failure."""
    with pytest.raises(UnsupportedSchema) as caught:
        to_gemini_schema({"type": "object", "properties": {"x": {"oneOf": []}}})
    assert not isinstance(caught.value, ValueError)
    assert not isinstance(caught.value, ChatModelError)
    assert isinstance(caught.value, RagforgeError)
    assert caught.value.status == 500


@respx.mock
async def test_unsupported_schema_escapes_an_except_value_error_before_any_request(
    model: GeminiChatModel,
) -> None:
    route = respx.post(STREAM_URL).mock(return_value=answered())
    with pytest.raises(UnsupportedSchema):
        await _grounding_style_call(model)
    assert route.call_count == 0


async def _grounding_style_call(model: GeminiChatModel) -> None:
    """The handler shape of agent/grounding.py: ValueError means "unparseable JSON"."""
    try:
        await model.complete_json("", USER, {"type": "object"})  # a map: not expressible
    except ValueError:  # pragma: no cover - reaching here is the bug under test
        pytest.fail("schema error was swallowed as a JSON parse error")


# ---------------------------------------------------------------------- output ceiling
@pytest.mark.parametrize(
    ("budget", "expected"),
    [
        pytest.param(0, ANSWER_MAX_OUTPUT_TOKENS, id="no-thinking"),
        pytest.param(1024, ANSWER_MAX_OUTPUT_TOKENS + 1024, id="fixed-budget"),
        pytest.param(-1, ANSWER_MAX_OUTPUT_TOKENS + 24_576, id="dynamic"),
        pytest.param(None, ANSWER_MAX_OUTPUT_TOKENS + 24_576, id="model-default"),
    ],
)
@respx.mock
async def test_stream_sends_an_output_ceiling_with_thinking_headroom(
    client: httpx.AsyncClient, budget: int | None, expected: int
) -> None:
    route = respx.post(STREAM_URL).mock(return_value=answered())
    await collect(_adapter(client, budget).stream("", USER))
    assert body(route)["generationConfig"]["maxOutputTokens"] == expected


@respx.mock
async def test_complete_json_has_its_own_larger_ceiling(model: GeminiChatModel) -> None:
    route = respx.post(STREAM_URL).mock(
        return_value=ok([text_chunk('{"verdict": "yes"}'), finish_chunk()])
    )
    await model.complete_json("", USER, VERDICT)
    assert body(route)["generationConfig"]["maxOutputTokens"] == STRUCTURED_MAX_OUTPUT_TOKENS


@respx.mock
async def test_refused_zero_budget_retry_gets_headroom_for_the_default_thinking(
    model: GeminiChatModel,
) -> None:
    rejected = httpx.Response(
        400, json=google_error(400, "INVALID_ARGUMENT", "This model only works in thinking mode.")
    )
    route = respx.post(STREAM_URL).mock(side_effect=[rejected, answered()])
    await collect(model.stream("", USER))
    assert body(route, 0)["generationConfig"]["maxOutputTokens"] == ANSWER_MAX_OUTPUT_TOKENS
    retried = body(route, 1)["generationConfig"]
    assert "thinkingConfig" not in retried
    assert retried["maxOutputTokens"] == ANSWER_MAX_OUTPUT_TOKENS + 2048


@respx.mock
async def test_hitting_the_ceiling_says_output_limit_not_context_window(
    model: GeminiChatModel,
) -> None:
    respx.post(STREAM_URL).mock(
        return_value=ok([text_chunk("loop loop"), finish_chunk("MAX_TOKENS", output=2048)])
    )
    with pytest.raises(OutputLimitReached) as caught:
        await collect(model.stream("", USER))
    assert isinstance(caught.value, OutputTruncated)  # provider-neutral handling still applies
    assert str(caught.value) == "model output truncated: output token limit reached"
    assert caught.value.usage.completion_tokens == 2048


# ---------------------------------------------------------------------- per-day quota
def _quota_429(quota_id: str, *, delay: str = "30s") -> httpx.Response:
    details: list[dict[str, Any]] = [
        {
            "@type": "type.googleapis.com/google.rpc.QuotaFailure",
            "violations": [
                {
                    "quotaMetric": "generativelanguage.googleapis.com/generate_content_requests",
                    "quotaId": quota_id,
                }
            ],
        },
        {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": delay},
    ]
    return httpx.Response(429, json=google_error(429, "RESOURCE_EXHAUSTED", "quota", details))


@respx.mock
async def test_daily_quota_is_exhausted_not_rate_limited(model: GeminiChatModel) -> None:
    respx.post(STREAM_URL).mock(
        return_value=_quota_429("GenerateRequestsPerDayPerProjectPerModel-FreeTier")
    )
    with pytest.raises(QuotaExhausted) as caught:
        await collect(model.stream("", USER))
    assert isinstance(caught.value, ModelConfigurationError)  # permanent: operator action
    assert not isinstance(caught.value, UpstreamRateLimited)
    assert str(caught.value) == "model provider daily quota exhausted"
    assert "FreeTier" not in str(caught.value)


@respx.mock
async def test_per_minute_quota_stays_a_retryable_rate_limit(model: GeminiChatModel) -> None:
    respx.post(STREAM_URL).mock(
        return_value=_quota_429("GenerateRequestsPerMinutePerProjectPerModel-FreeTier", delay="7s")
    )
    with pytest.raises(UpstreamRateLimited) as caught:
        await collect(model.stream("", USER))
    assert caught.value.retry_after_s == 7.0


@respx.mock
async def test_quota_id_that_is_not_an_identifier_is_ignored(model: GeminiChatModel) -> None:
    respx.post(STREAM_URL).mock(return_value=_quota_429("PerDay but free text " + CANARY_KEY))
    with pytest.raises(UpstreamRateLimited):
        await collect(model.stream("", USER))


@respx.mock
async def test_daily_quota_reported_mid_stream_is_exhausted_too(model: GeminiChatModel) -> None:
    error = google_error(
        429,
        "RESOURCE_EXHAUSTED",
        "quota",
        [{"violations": [{"quotaId": "GenerateContentInputTokensPerModelPerDay-FreeTier"}]}],
    )
    respx.post(STREAM_URL).mock(return_value=ok([error]))
    with pytest.raises(QuotaExhausted):
        await collect(model.stream("", USER))


# ---------------------------------------------------------------------- no reachable context
def _chain(exc: BaseException) -> list[BaseException]:
    """Every exception reachable through __cause__ / __context__, suppressed or not."""
    seen: list[BaseException] = []
    current: BaseException | None = exc
    while current is not None and current not in seen:
        seen.append(current)
        current = current.__cause__ or current.__context__
    return seen


def _assert_clean_chain(exc: BaseException) -> None:
    for link in _chain(exc):
        assert not isinstance(link, httpx.HTTPError), type(link)
        assert CANARY_KEY not in repr(link) + repr(link.args)


class _DropsAfter(httpx.AsyncByteStream):
    def __init__(self, first: bytes) -> None:
        self._first = first

    async def __aiter__(self) -> AsyncIterator[bytes]:
        yield self._first
        raise httpx.ReadError("connection reset by gemini.test")


@pytest.mark.parametrize(
    "failure",
    [httpx.ConnectError("dns failed"), httpx.ReadTimeout("slow")],
    ids=["connect", "timeout"],
)
@respx.mock
async def test_transport_error_before_output_has_no_context(
    model: GeminiChatModel, failure: httpx.HTTPError
) -> None:
    respx.post(STREAM_URL).mock(side_effect=failure)
    with pytest.raises(ChatModelError) as caught:
        await collect(model.stream("", USER))
    assert caught.value.__context__ is None
    assert caught.value.__cause__ is None
    _assert_clean_chain(caught.value)


@respx.mock
async def test_transport_error_after_output_has_no_reachable_httpx_exception(
    model: GeminiChatModel,
) -> None:
    upstream = _DropsAfter(sse([text_chunk("Kafka is")]))
    respx.post(STREAM_URL).mock(return_value=httpx.Response(200, stream=upstream))
    seen: list[ChatStreamEvent] = []
    with pytest.raises(StreamInterrupted) as caught:
        await consume_into(model.stream("", USER), seen)
    _assert_clean_chain(caught.value)


@respx.mock
async def test_malformed_event_does_not_carry_the_event_text(model: GeminiChatModel) -> None:
    """A JSONDecodeError keeps the whole document in `.doc`: it must not ride along."""
    raw = b"data: {not json " + CANARY_KEY.encode() + b"\r\n\r\n"
    respx.post(STREAM_URL).mock(return_value=httpx.Response(200, content=raw))
    with pytest.raises(ChatModelError) as caught:
        await collect(model.stream("", USER))
    assert caught.value.__context__ is None
    _assert_clean_chain(caught.value)


@respx.mock
async def test_invalid_structured_output_does_not_carry_the_model_text(
    model: GeminiChatModel,
) -> None:
    respx.post(STREAM_URL).mock(return_value=ok([text_chunk("{nope secret"), finish_chunk()]))
    with pytest.raises(ChatModelError, match="not valid JSON") as caught:
        await model.complete_json("", USER, VERDICT)
    assert caught.value.__context__ is None


# ---------------------------------------------------------------------- signature size cap
async def _call_and_replay(model: GeminiChatModel, route: respx.Route) -> str:
    events = await collect(model.stream("", USER, [SEARCH]))
    [call] = [e for e in events if isinstance(e, ToolCall)]
    history = [
        *USER,
        ChatMessage(Role.assistant, "", tool_calls=(call,)),
        ChatMessage(Role.tool, "r", tool_call_id=call.id),
    ]
    await collect(model.stream("", history, [SEARCH]))
    sent: Any = body(route)["contents"][1]["parts"][0]["thoughtSignature"]
    assert isinstance(sent, str)
    return sent


@pytest.mark.parametrize(
    ("size", "kept"),
    [(chat_gemini._MAX_SIGNATURE_CHARS, True), (chat_gemini._MAX_SIGNATURE_CHARS + 1, False)],
    ids=["at-cap", "over-cap"],
)
@respx.mock
async def test_oversize_signature_is_dropped_and_replayed_as_the_placeholder(
    model: GeminiChatModel, size: int, kept: bool
) -> None:
    signature = "s" * size
    route = respx.post(STREAM_URL).mock(
        side_effect=[
            ok([call_chunk("search_notes", {"query": "q"}, signature=signature), finish_chunk()]),
            answered(),
        ]
    )
    sent = await _call_and_replay(model, route)
    assert sent == (signature if kept else SKIP_SIGNATURE)
    if not kept:
        assert model._calls == {}  # nothing cached for the oversize one


# ---------------------------------------------------------------------- enum inference
@pytest.mark.parametrize("enum", [[1, 2], ["a", 1], [True]], ids=["ints", "mixed", "bools"])
def test_untyped_enum_of_non_strings_is_rejected_not_told_string(enum: list[object]) -> None:
    with pytest.raises(UnsupportedSchema, match="only strings"):
        to_gemini_schema({"enum": enum})


def test_typed_non_string_enum_keeps_its_type() -> None:
    assert to_gemini_schema({"type": "integer", "enum": [1, 2]}) == {"type": "INTEGER"}
    assert to_gemini_schema({"enum": ["a", "b"]}) == {"type": "STRING", "enum": ["a", "b"]}
