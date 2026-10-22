"""GeminiChatModel against a respx-mocked streamGenerateContent (no network, no model)."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest
import respx
from structlog.testing import capture_logs

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
    ok,
    sse,
    text_chunk,
)
from ragforge_core.adapters.llm.chat_gemini import (
    API_KEY_HEADER,
    SKIP_SIGNATURE,
    GeminiChatModel,
    ModelConfigurationError,
    ModelRequestRejected,
    ResponseBlocked,
    StreamInterrupted,
)
from ragforge_core.adapters.llm.chat_ollama import ChatModelError, OutputTruncated
from ragforge_core.domain.models import (
    ChatMessage,
    ChatStreamEvent,
    ChatTurnUsage,
    Role,
    TextDelta,
    ToolCall,
    ToolSpec,
    UsageReport,
)
from ragforge_core.domain.ports import ChatModel

SEARCH = ToolSpec(
    "search_notes",
    "Search the notes",
    {
        "type": "object",
        "additionalProperties": False,
        "properties": {"query": {"type": "string", "minLength": 1, "title": "Query"}},
        "required": ["query"],
    },
)
NO_ARGS = ToolSpec("list_sources", "List sources", {"type": "object", "properties": {}})


def test_implements_the_chat_model_port(model: GeminiChatModel) -> None:
    assert isinstance(model, ChatModel)
    assert model.provider == "gemini"
    assert model.model == MODEL


# ---------------------------------------------------------------------- streaming text
@respx.mock
async def test_streams_text_then_exactly_one_usage_report_last(model: GeminiChatModel) -> None:
    respx.post(STREAM_URL).mock(
        return_value=ok([text_chunk("Hel"), text_chunk("lo"), finish_chunk(prompt=9, output=2)])
    )
    events = await collect(model.stream("be brief", USER))
    assert events == [
        TextDelta("Hel"),
        TextDelta("lo"),
        UsageReport(ChatTurnUsage("gemini", MODEL, prompt_tokens=9, completion_tokens=2)),
    ]


@respx.mock
async def test_request_shape_system_instruction_roles_and_thinking(
    model: GeminiChatModel,
) -> None:
    route = respx.post(STREAM_URL).mock(return_value=answered())
    history = [
        ChatMessage(Role.user, "q1"),
        ChatMessage(Role.assistant, "a1"),
        ChatMessage(Role.system, "summary so far"),
        ChatMessage(Role.user, "q2"),
    ]
    await collect(model.stream("rules", history))
    request = route.calls.last.request
    assert request.url.params["alt"] == "sse"
    sent = body(route)
    assert sent["systemInstruction"] == {"parts": [{"text": "rules"}, {"text": "summary so far"}]}
    assert sent["contents"] == [
        {"role": "user", "parts": [{"text": "q1"}]},
        {"role": "model", "parts": [{"text": "a1"}]},
        {"role": "user", "parts": [{"text": "q2"}]},
    ]
    # default thinking budget 0: latency first (frozen setting default)
    assert sent["generationConfig"]["thinkingConfig"] == {"thinkingBudget": 0}
    assert "tools" not in sent


@respx.mock
async def test_thought_parts_are_never_surfaced(model: GeminiChatModel) -> None:
    thought = {"candidates": [{"content": {"parts": [{"text": "secret plan", "thought": True}]}}]}
    respx.post(STREAM_URL).mock(return_value=ok([thought, text_chunk("answer"), finish_chunk()]))
    events = await collect(model.stream("", USER))
    assert [e for e in events if isinstance(e, TextDelta)] == [TextDelta("answer")]


@respx.mock
async def test_usage_counts_thinking_as_output_and_uses_the_last_report(
    model: GeminiChatModel,
) -> None:
    early = {**text_chunk("x"), "usageMetadata": {"promptTokenCount": 1}}
    respx.post(STREAM_URL).mock(
        return_value=ok([early, finish_chunk(prompt=20, output=4, thoughts=6)])
    )
    events = await collect(model.stream("", USER))
    reports = [e for e in events if isinstance(e, UsageReport)]
    assert len(reports) == 1
    assert events[-1] is reports[0]
    assert reports[0].usage.prompt_tokens == 20
    assert reports[0].usage.completion_tokens == 10


@respx.mock
async def test_sse_framing_comments_multiline_and_no_trailing_blank_line(
    model: GeminiChatModel,
) -> None:
    raw = (
        b": keep-alive\n\n"
        b'event: message\ndata: {"candidates": [{"content":\n'
        b'data: {"parts": [{"text": "hi"}]}}]}\n\n'
        b'data: {"candidates": [{"content": {"parts": []}, "finishReason": "STOP"}]}'
    )
    respx.post(STREAM_URL).mock(return_value=httpx.Response(200, content=raw))
    events = await collect(model.stream("", USER))
    assert events[0] == TextDelta("hi")
    assert isinstance(events[-1], UsageReport)


# ---------------------------------------------------------------------- tools
@respx.mock
async def test_function_call_round_trip_with_fresh_unique_ids(model: GeminiChatModel) -> None:
    route = respx.post(STREAM_URL).mock(
        side_effect=[
            ok([call_chunk("search_notes", {"query": "kafka"}, call_id="p0"), finish_chunk()]),
            ok([call_chunk("search_notes", {"query": "redis"}, call_id="p0"), finish_chunk()]),
            ok([text_chunk("done"), finish_chunk()]),
        ]
    )
    first = [e for e in await collect(model.stream("", USER, [SEARCH])) if isinstance(e, ToolCall)]
    assert first[0].name == "search_notes"
    assert first[0].arguments == {"query": "kafka"}
    history = [
        *USER,
        ChatMessage(Role.assistant, "", tool_calls=(first[0],)),
        ChatMessage(Role.tool, '{"hits": 1}', tool_call_id=first[0].id),
    ]
    second = [
        e for e in await collect(model.stream("", history, [SEARCH])) if isinstance(e, ToolCall)
    ]
    # The provider reused "p0"; our ids must differ across turns and never be the provider's.
    assert first[0].id != second[0].id
    assert {first[0].id, second[0].id}.isdisjoint({"p0"})
    assert len(first[0].id) == 32

    sent = body(route, 1)
    assert sent["tools"] == [
        {
            "functionDeclarations": [
                {
                    "name": "search_notes",
                    "description": "Search the notes",
                    "parameters": {
                        "type": "OBJECT",
                        "properties": {"query": {"type": "STRING", "minLength": 1}},
                        "propertyOrdering": ["query"],
                        "required": ["query"],
                    },
                }
            ]
        }
    ]
    assert sent["contents"][1:] == [
        {
            "role": "model",
            "parts": [
                {
                    # Google's own call id goes back to Google (only), next to the call...
                    "functionCall": {
                        "name": "search_notes",
                        "args": {"query": "kafka"},
                        "id": "p0",
                    },
                    # the mocked model sent no signature, so the placeholder goes back
                    "thoughtSignature": SKIP_SIGNATURE,
                }
            ],
        },
        {
            "role": "user",
            "parts": [
                {
                    "functionResponse": {
                        "name": "search_notes",
                        "response": {"output": '{"hits": 1}'},
                        "id": "p0",  # ...and pairs the result with it
                    }
                }
            ],
        },
    ]
    assert first[0].id not in route.calls[1].request.content.decode()  # our ids stay local


@respx.mock
async def test_parallel_calls_and_results_keep_order_in_one_turn(model: GeminiChatModel) -> None:
    two = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {"functionCall": {"name": "search_notes", "args": {"query": "a"}}},
                        {"functionCall": {"name": "search_notes", "args": {"query": "b"}}},
                    ]
                }
            }
        ]
    }
    route = respx.post(STREAM_URL).mock(
        side_effect=[ok([two, finish_chunk()]), ok([text_chunk("x"), finish_chunk()])]
    )
    calls = [e for e in await collect(model.stream("", USER, [SEARCH])) if isinstance(e, ToolCall)]
    assert [c.arguments["query"] for c in calls] == ["a", "b"]
    assert len({c.id for c in calls}) == 2
    history = [
        *USER,
        ChatMessage(Role.assistant, "", tool_calls=tuple(calls)),
        ChatMessage(Role.tool, "ra", tool_call_id=calls[0].id),
        ChatMessage(Role.tool, "rb", tool_call_id=calls[1].id),
    ]
    await collect(model.stream("", history, [SEARCH]))
    results = body(route)["contents"][2]
    assert results["role"] == "user"
    assert [p["functionResponse"]["response"]["output"] for p in results["parts"]] == ["ra", "rb"]


@respx.mock
async def test_thought_signature_is_replayed_with_its_call(model: GeminiChatModel) -> None:
    route = respx.post(STREAM_URL).mock(
        side_effect=[
            ok([call_chunk("search_notes", {"query": "q"}, signature="c2lnLTE="), finish_chunk()]),
            ok([text_chunk("x"), finish_chunk()]),
        ]
    )
    [call] = [e for e in await collect(model.stream("", USER, [SEARCH])) if isinstance(e, ToolCall)]
    history = [
        *USER,
        ChatMessage(Role.assistant, "", tool_calls=(call,)),
        ChatMessage(Role.tool, "r", tool_call_id=call.id),
    ]
    await collect(model.stream("", history, [SEARCH]))
    assert body(route)["contents"][1]["parts"][0]["thoughtSignature"] == "c2lnLTE="


@respx.mock
async def test_tool_without_arguments_is_declared_without_parameters(
    model: GeminiChatModel,
) -> None:
    route = respx.post(STREAM_URL).mock(return_value=answered())
    await collect(model.stream("", USER, [NO_ARGS]))
    declaration = body(route)["tools"][0]["functionDeclarations"][0]
    assert declaration == {"name": "list_sources", "description": "List sources"}


async def test_tool_result_without_a_matching_call_is_rejected_before_sending(
    model: GeminiChatModel,
) -> None:
    orphan = [*USER, ChatMessage(Role.tool, "r", tool_call_id="nope")]
    with pytest.raises(ModelRequestRejected, match="tool result"):
        await collect(model.stream("", orphan))


@pytest.mark.parametrize(
    "call",
    [{"args": {}}, {"name": "", "args": {}}, {"name": "x", "args": [1]}, "not-an-object"],
)
@respx.mock
async def test_malformed_function_calls_raise(model: GeminiChatModel, call: object) -> None:
    chunk = {"candidates": [{"content": {"parts": [{"functionCall": call}]}}]}
    respx.post(STREAM_URL).mock(return_value=ok([chunk, finish_chunk()]))
    with pytest.raises(ChatModelError, match="tool call"):
        await collect(model.stream("", USER, [SEARCH]))


# ---------------------------------------------------------------------- stream integrity
@respx.mock
async def test_stream_without_finish_reason_is_an_error_not_an_answer(
    model: GeminiChatModel,
) -> None:
    respx.post(STREAM_URL).mock(return_value=ok([{"candidates": []}]))
    with pytest.raises(ChatModelError, match="ended before completion"):
        await collect(model.stream("", USER))
    # Cut off after output: not an answer, and not blindly retryable either.
    respx.post(STREAM_URL).mock(return_value=ok([text_chunk("half an ans")]))
    seen: list[ChatStreamEvent] = []
    with pytest.raises(StreamInterrupted) as caught:
        await consume_into(model.stream("", USER), seen)
    assert caught.value.cause == "transient"
    assert not any(isinstance(e, UsageReport) for e in seen)


@pytest.mark.parametrize(
    "raw",
    [
        b"data: {not json\n\n",
        b"data: [1, 2]\n\n",
        b'data: {"candidates": "nope"}\n\n',
        b'data: {"candidates": [{"content": {"parts": "nope"}}]}\n\n',
        b'data: {"candidates": [{"finishReason": 7}]}\n\n',
        b'data: {"promptFeedback": "nope"}\n\n',
    ],
)
@respx.mock
async def test_malformed_events_raise(model: GeminiChatModel, raw: bytes) -> None:
    respx.post(STREAM_URL).mock(return_value=httpx.Response(200, content=raw))
    with pytest.raises(ChatModelError, match="malformed"):
        await collect(model.stream("", USER))


@respx.mock
async def test_error_event_mid_stream_is_mapped(model: GeminiChatModel) -> None:
    error = {"error": {"code": 500, "status": "INTERNAL", "message": "echo: hello"}}
    respx.post(STREAM_URL).mock(return_value=ok([error]))
    with pytest.raises(ChatModelError) as caught:
        await collect(model.stream("", USER))
    assert type(caught.value) is ChatModelError  # nothing emitted yet: plainly transient
    assert str(caught.value) == "model provider returned HTTP 500: INTERNAL"

    respx.post(STREAM_URL).mock(return_value=ok([text_chunk("a"), error]))
    with pytest.raises(StreamInterrupted) as interrupted:
        await collect(model.stream("", USER))
    assert str(interrupted.value) == "model stream interrupted after output: transient"
    assert interrupted.value.__cause__ is None
    assert "hello" not in repr(interrupted.value.args)


@respx.mock
async def test_transport_failures_have_fixed_messages(model: GeminiChatModel) -> None:
    respx.post(STREAM_URL).mock(side_effect=httpx.ConnectError("dns for gemini.test failed"))
    with pytest.raises(ChatModelError) as caught:
        await collect(model.stream("", USER))
    assert str(caught.value) == "model provider unreachable"
    respx.post(STREAM_URL).mock(side_effect=httpx.ReadTimeout("slow"))
    with pytest.raises(ChatModelError, match=r"^model provider timed out$"):
        await collect(model.stream("", USER))


@respx.mock
async def test_max_tokens_raises_truncated_with_usage_instead_of_report(
    model: GeminiChatModel,
) -> None:
    respx.post(STREAM_URL).mock(
        return_value=ok([text_chunk("cut"), finish_chunk("MAX_TOKENS", prompt=3, output=8)])
    )
    seen: list[ChatStreamEvent] = []
    with pytest.raises(OutputTruncated) as caught:
        await consume_into(model.stream("", USER), seen)
    assert caught.value.usage.completion_tokens == 8
    assert not any(isinstance(e, UsageReport) for e in seen)


@pytest.mark.parametrize("reason", ["MALFORMED_FUNCTION_CALL", "TOO_MANY_TOOL_CALLS"])
@respx.mock
async def test_bad_tool_finish_reasons_raise(model: GeminiChatModel, reason: str) -> None:
    respx.post(STREAM_URL).mock(return_value=ok([finish_chunk(reason)]))
    with pytest.raises(ChatModelError, match=reason.lower()):
        await collect(model.stream("", USER, [SEARCH]))


@pytest.mark.parametrize("reason", ["NEW_REASON", "FINISH_REASON_UNSPECIFIED"])
@respx.mock
async def test_unknown_or_unspecified_finish_reason_fails_closed(
    model: GeminiChatModel, reason: str
) -> None:
    respx.post(STREAM_URL).mock(return_value=ok([finish_chunk(reason)]))
    with pytest.raises(ChatModelError, match="unexpected finish reason"):
        await collect(model.stream("", USER))
    # After output it still never passes as an answer (and is not blindly retryable).
    respx.post(STREAM_URL).mock(return_value=ok([text_chunk("x"), finish_chunk(reason)]))
    seen: list[ChatStreamEvent] = []
    with pytest.raises(StreamInterrupted):
        await consume_into(model.stream("", USER), seen)
    assert not any(isinstance(e, UsageReport) for e in seen)


# ---------------------------------------------------------------------- blocked responses
@pytest.mark.parametrize(
    "reason", ["SAFETY", "RECITATION", "OTHER", "BLOCKLIST", "PROHIBITED_CONTENT", "SPII"]
)
@respx.mock
async def test_blocked_answer_raises_typed_error_never_an_empty_answer(
    model: GeminiChatModel, reason: str
) -> None:
    respx.post(STREAM_URL).mock(return_value=ok([text_chunk("partial"), finish_chunk(reason)]))
    seen: list[ChatStreamEvent] = []
    with pytest.raises(ResponseBlocked) as caught:
        await consume_into(model.stream("", USER), seen)
    assert caught.value.stage == "response"
    assert caught.value.reason == reason.lower()
    assert caught.value.usage is not None
    assert not any(isinstance(e, UsageReport) for e in seen)
    assert isinstance(caught.value, ChatModelError)  # callers catching provider errors see it


@pytest.mark.parametrize(
    ("block", "category"),
    [("SAFETY", "safety"), ("PROHIBITED_CONTENT", "prohibited_content"), ("WHO_KNOWS", "other")],
)
@respx.mock
async def test_blocked_prompt_raises_typed_error(
    model: GeminiChatModel, block: str, category: str
) -> None:
    chunk = {
        "promptFeedback": {"blockReason": block},
        "usageMetadata": {"promptTokenCount": 4, "totalTokenCount": 4},
    }
    respx.post(STREAM_URL).mock(return_value=ok([chunk]))
    with pytest.raises(ResponseBlocked) as caught:
        await collect(model.stream("", USER))
    assert (caught.value.stage, caught.value.reason) == ("prompt", category)
    assert caught.value.usage is not None
    assert caught.value.usage.prompt_tokens == 4


@respx.mock
async def test_complete_json_blocked_raises(model: GeminiChatModel) -> None:
    respx.post(STREAM_URL).mock(return_value=ok([finish_chunk("SAFETY")]))
    with pytest.raises(ResponseBlocked):
        await model.complete_json("", USER, GROUND)


# ---------------------------------------------------------------------- complete_json
GROUND: dict[str, object] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["verdict", "unsupported_claims"],
    "properties": {
        "verdict": {"type": "string", "enum": ["supported", "unsupported"]},
        "unsupported_claims": {"type": "array", "items": {"type": "string"}, "maxItems": 20},
    },
}


@respx.mock
async def test_complete_json_uses_json_mode_and_returns_parsed_object(
    model: GeminiChatModel,
) -> None:
    route = respx.post(STREAM_URL).mock(
        return_value=ok(
            [
                text_chunk('{"verdict": "supp'),
                text_chunk('orted", "unsupported_claims": []}'),
                finish_chunk(prompt=30, output=12),
            ]
        )
    )
    payload, usage = await model.complete_json("grade", USER, GROUND)
    assert payload == {"verdict": "supported", "unsupported_claims": []}
    assert usage == ChatTurnUsage("gemini", MODEL, 30, 12)
    config = body(route)["generationConfig"]
    assert config["responseMimeType"] == "application/json"
    assert config["temperature"] == 0
    assert config["responseSchema"] == {
        "type": "OBJECT",
        "properties": {
            "verdict": {"type": "STRING", "enum": ["supported", "unsupported"]},
            "unsupported_claims": {"type": "ARRAY", "items": {"type": "STRING"}, "maxItems": 20},
        },
        "propertyOrdering": ["verdict", "unsupported_claims"],
        "required": ["verdict", "unsupported_claims"],
    }


@pytest.mark.parametrize(
    ("text", "message"),
    [("", "missing"), ("{nope", "not valid JSON"), ("[1]", "must be a JSON object")],
)
@respx.mock
async def test_complete_json_rejects_bad_output(
    model: GeminiChatModel, text: str, message: str
) -> None:
    chunks: list[dict[str, Any]] = [text_chunk(text)] if text else []
    respx.post(STREAM_URL).mock(return_value=ok([*chunks, finish_chunk()]))
    with pytest.raises(ChatModelError, match=message):
        await model.complete_json("", USER, GROUND)


@respx.mock
async def test_complete_json_truncation_raises(model: GeminiChatModel) -> None:
    respx.post(STREAM_URL).mock(return_value=ok([text_chunk('{"a": '), finish_chunk("MAX_TOKENS")]))
    with pytest.raises(OutputTruncated):
        await model.complete_json("", USER, GROUND)


@respx.mock
async def test_complete_json_rejects_a_tool_call(model: GeminiChatModel) -> None:
    respx.post(STREAM_URL).mock(return_value=ok([call_chunk("x", {}), finish_chunk()]))
    with pytest.raises(ChatModelError, match="unexpected tool call"):
        await model.complete_json("", USER, GROUND)


# ---------------------------------------------------------------------- thinking fallback
@respx.mock
async def test_thinking_config_rejected_retries_once_without_it_and_remembers(
    model: GeminiChatModel,
) -> None:
    rejected = httpx.Response(
        400,
        json={
            "error": {
                "code": 400,
                "status": "INVALID_ARGUMENT",
                "message": "Budget 0 is invalid. This model only works in thinking mode.",
            }
        },
    )
    route = respx.post(STREAM_URL).mock(
        side_effect=[rejected, ok([text_chunk("a"), finish_chunk()]), answered()]
    )
    events = await collect(model.stream("", USER))
    assert events[0] == TextDelta("a")
    assert "thinkingConfig" in body(route, 0)["generationConfig"]
    assert "thinkingConfig" not in body(route, 1)["generationConfig"]
    await collect(model.stream("", USER))
    assert route.call_count == 3
    assert "thinkingConfig" not in body(route, 2)["generationConfig"]


@respx.mock
async def test_bare_invalid_argument_is_treated_as_a_thinking_rejection(
    model: GeminiChatModel,
) -> None:
    """gemini-3.5-flash-lite rejects budget 0 with Google's bare, detail-less message (seen live
    2026-10-17). The retry without thinkingConfig is accepted, so the budget is dropped."""
    bare = {
        "code": 400,
        "status": "INVALID_ARGUMENT",
        "message": "Request contains an invalid argument.",
    }
    route = respx.post(STREAM_URL).mock(
        side_effect=[
            httpx.Response(400, json={"error": bare}),
            ok([text_chunk("a"), finish_chunk()]),
        ]
    )
    with capture_logs() as logs:
        events = await collect(model.stream("", USER))
    assert events[0] == TextDelta("a")
    assert "thinkingConfig" in body(route, 0)["generationConfig"]
    assert "thinkingConfig" not in body(route, 1)["generationConfig"]
    dropped = [e for e in logs if e["event"] == "gemini_thinking_config_dropped"]
    assert dropped == [
        {
            "event": "gemini_thinking_config_dropped",
            "log_level": "warning",
            "model": model.model,
            "configured_budget": 0,
        }
    ]


@respx.mock
async def test_bare_invalid_argument_with_details_or_a_failed_retry_keeps_the_budget(
    model: GeminiChatModel,
) -> None:
    bare = {
        "code": 400,
        "status": "INVALID_ARGUMENT",
        "message": "Request contains an invalid argument.",
    }
    with_details = {**bare, "details": [{"@type": "type.googleapis.com/google.rpc.BadRequest"}]}
    route = respx.post(STREAM_URL).mock(
        side_effect=[
            httpx.Response(
                400, json={"error": with_details}
            ),  # details: not the bare form, no retry
            httpx.Response(400, json={"error": bare}),  # bare: one retry ...
            httpx.Response(400, json={"error": bare}),  # ... which also fails: budget kept
            answered(),
        ]
    )
    for _ in range(2):
        with pytest.raises(ModelRequestRejected):
            await collect(model.stream("", USER))
    assert route.call_count == 3
    await collect(model.stream("", USER))
    assert "thinkingConfig" in body(route, 3)["generationConfig"]


# ---------------------------------------------------------------------- cancellation
class _HangingBody(httpx.AsyncByteStream):
    """Sends one event, then blocks like a model that keeps generating; records aclose()."""

    def __init__(self) -> None:
        self.closed = asyncio.Event()
        self.first_sent = asyncio.Event()

    async def __aiter__(self) -> AsyncIterator[bytes]:
        yield sse([text_chunk("first")])
        self.first_sent.set()
        await asyncio.Event().wait()  # never finishes on its own
        yield b""  # pragma: no cover

    async def aclose(self) -> None:
        self.closed.set()


@respx.mock
async def test_aclose_cancels_the_upstream_request(model: GeminiChatModel) -> None:
    upstream = _HangingBody()
    respx.post(STREAM_URL).mock(return_value=httpx.Response(200, stream=upstream))
    stream = model.stream("", USER)
    assert await anext(stream) == TextDelta("first")
    await asyncio.wait_for(stream.aclose(), timeout=2)  # type: ignore[attr-defined]
    assert upstream.closed.is_set(), "closing the iterator must close the upstream response"


@respx.mock
async def test_task_cancellation_closes_the_upstream_request(model: GeminiChatModel) -> None:
    upstream = _HangingBody()
    respx.post(STREAM_URL).mock(return_value=httpx.Response(200, stream=upstream))

    async def consume() -> None:
        async for _ in model.stream("", USER):
            pass

    task = asyncio.create_task(consume())
    await asyncio.wait_for(upstream.first_sent.wait(), timeout=2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert upstream.closed.is_set()


# ---------------------------------------------------------------------- construction
def test_model_name_that_could_alter_the_url_is_rejected(client: httpx.AsyncClient) -> None:
    from pydantic import SecretStr

    for bad in ("../files", "m?alt=json", "a/b", "", "x y"):
        with pytest.raises(ModelConfigurationError):
            GeminiChatModel(client, api_key=SecretStr(CANARY_KEY), model=bad)


def test_models_prefix_is_accepted(client: httpx.AsyncClient) -> None:
    from pydantic import SecretStr

    built = GeminiChatModel(client, api_key=SecretStr(CANARY_KEY), model="models/gemini-x")
    assert built.model == "gemini-x"


def test_blank_key_is_a_configuration_error(client: httpx.AsyncClient) -> None:
    from pydantic import SecretStr

    with pytest.raises(ModelConfigurationError, match="not configured"):
        GeminiChatModel(client, api_key=SecretStr("  \n"), model=MODEL)


@respx.mock
async def test_key_is_stripped_before_use(client: httpx.AsyncClient) -> None:
    from pydantic import SecretStr

    route = respx.post(STREAM_URL).mock(return_value=answered())
    padded = GeminiChatModel(client, api_key=SecretStr(f" {CANARY_KEY}\n"), model=MODEL)
    await collect(padded.stream("", USER))
    assert route.calls.last.request.headers[API_KEY_HEADER] == CANARY_KEY


async def test_the_owned_client_retries_connection_failures_only() -> None:
    """httpx transport retries cover connect errors/timeouts (nothing sent yet), never a
    request that reached the server."""
    from pydantic import SecretStr

    from ragforge_core.adapters.llm.chat_gemini import CONNECT_RETRIES
    from ragforge_core.config import Settings

    settings = Settings.model_construct(
        gemini_model="gemini-3.5-flash-lite",
        gemini_api_key=SecretStr(CANARY_KEY),
        gemini_thinking_budget=0,
    )
    model = GeminiChatModel.from_settings(settings)
    try:
        transport = model._client._transport
        assert isinstance(transport, httpx.AsyncHTTPTransport)
        assert transport._pool._retries == CONNECT_RETRIES == 2
    finally:
        await model.aclose()


@respx.mock
async def test_after_two_failed_bare_retries_a_bare_400_costs_one_request(
    model: GeminiChatModel,
) -> None:
    """Review rounds 6 and 8: the bare 400 also covers unrelated invalid requests. After TWO
    consecutive definitive retry rejections, stop paying a second request for each one."""
    bare = {
        "code": 400,
        "status": "INVALID_ARGUMENT",
        "message": "Request contains an invalid argument.",
    }
    route = respx.post(STREAM_URL).mock(
        side_effect=[httpx.Response(400, json={"error": bare}) for _ in range(5)]
    )
    with capture_logs() as logs:
        for _ in range(3):
            with pytest.raises(ModelRequestRejected):
                await collect(model.stream("", USER))
    assert route.call_count == 5  # 2 + 2 + 1
    assert "thinkingConfig" in body(route, 4)["generationConfig"]  # the budget is kept
    assert [e["event"] for e in logs].count("gemini_bare_400_not_thinking") == 1


@respx.mock
async def test_one_unrelated_bare_400_does_not_latch_the_fallback_off(
    model: GeminiChatModel,
) -> None:
    """Review round 8, P2: one request that is invalid for another reason, then normal turns
    on a model that needs the fallback: they must still be answered."""
    bare = {
        "code": 400,
        "status": "INVALID_ARGUMENT",
        "message": "Request contains an invalid argument.",
    }
    route = respx.post(STREAM_URL).mock(
        side_effect=[
            httpx.Response(400, json={"error": bare}),  # bad request, thinking sent
            httpx.Response(400, json={"error": bare}),  # bad without thinking too
            httpx.Response(400, json={"error": bare}),  # next turn: thinking rejected
            ok([text_chunk("a"), finish_chunk()]),  # retried without it: accepted
        ]
    )
    with pytest.raises(ModelRequestRejected):
        await collect(model.stream("", USER))
    events = await collect(model.stream("", USER))
    assert events[0] == TextDelta("a")
    assert route.call_count == 4


@pytest.mark.parametrize(
    "retry_failure",
    [
        httpx.ConnectTimeout("t"),
        httpx.Response(500, json={"error": {"code": 500, "status": "INTERNAL"}}),
        httpx.Response(503, json={"error": {"code": 503, "status": "UNAVAILABLE"}}),
    ],
)
@respx.mock
async def test_a_transient_retry_failure_keeps_the_bare_400_fallback(
    model: GeminiChatModel, retry_failure: httpx.Response | Exception
) -> None:
    """Review round 7, P1: only a definitive 400 on the retry clears trust. A timeout or 5xx on
    the retry must not disable the fallback for good."""
    bare = {
        "code": 400,
        "status": "INVALID_ARGUMENT",
        "message": "Request contains an invalid argument.",
    }
    route = respx.post(STREAM_URL).mock(
        side_effect=[
            httpx.Response(400, json={"error": bare}),
            retry_failure,
            httpx.Response(400, json={"error": bare}),
            ok([text_chunk("a"), finish_chunk()]),
        ]
    )
    with pytest.raises(Exception):  # noqa: B017, PT011 - the transient error class varies
        await collect(model.stream("", USER))
    events = await collect(model.stream("", USER))  # the bare 400 is still retried
    assert events[0] == TextDelta("a")
    assert route.call_count == 4
    assert "thinkingConfig" not in body(route, 3)["generationConfig"]


@respx.mock
async def test_bare_misses_must_be_consecutive(model: GeminiChatModel) -> None:
    """Review round 9, P2: an accepted response between two misses resets the count."""
    bare = {
        "code": 400,
        "status": "INVALID_ARGUMENT",
        "message": "Request contains an invalid argument.",
    }
    no = httpx.Response(400, json={"error": bare})
    route = respx.post(STREAM_URL).mock(
        side_effect=[no, no, ok([text_chunk("a"), finish_chunk()]), no, no, no, no]
    )
    with pytest.raises(ModelRequestRejected):
        await collect(model.stream("", USER))  # miss 1 (2 requests)
    await collect(model.stream("", USER))  # accepted with thinking: resets
    with pytest.raises(ModelRequestRejected):
        await collect(model.stream("", USER))  # miss 1 again (2 requests)
    with pytest.raises(ModelRequestRejected):
        await collect(model.stream("", USER))  # miss 2: still retried (2 requests)
    assert route.call_count == 7


class _Drip(httpx.AsyncByteStream):
    """An endless line in small reads: the oversize guard must trip in linear time."""

    def __init__(self, total: int, step: int) -> None:
        self.total, self.step = total, step

    async def __aiter__(self) -> AsyncIterator[bytes]:
        sent = 0
        while sent < self.total:
            yield b"x" * self.step
            sent += self.step

    async def aclose(self) -> None:
        return None


@respx.mock
async def test_an_endless_line_is_rejected_in_linear_time(model: GeminiChatModel) -> None:
    """Final review, P2: re-splitting the whole tail per read was quadratic (36 s of CPU)."""
    import time

    respx.post(STREAM_URL).mock(return_value=httpx.Response(200, stream=_Drip(8 << 20, 1024)))
    started = time.perf_counter()
    with pytest.raises(ChatModelError, match="oversized event"):
        await collect(model.stream("", USER))
    assert time.perf_counter() - started < 2.0
