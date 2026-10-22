"""Regression tests for the round-2 review: the read timeout grows with the thinking budget,
failures after output are marked so callers cannot blindly replay (and duplicate) it,
FINISH_REASON_UNSPECIFIED fails closed (see test_g1_chat_gemini), and Google's own
functionCall.id is echoed back to Google while our minted ids stay local."""

from __future__ import annotations

from collections.abc import AsyncIterator

import httpx
import pytest
import respx
from pydantic import SecretStr

from module_g1.g1_support import (
    BASE,
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
    MAX_THINKING_READ_S,
    SKIP_SIGNATURE,
    GeminiChatModel,
    ResponseBlocked,
    StreamInterrupted,
)
from ragforge_core.adapters.llm.chat_ollama import ChatModelError, OutputTruncated
from ragforge_core.domain.errors import UpstreamRateLimited
from ragforge_core.domain.models import (
    ChatMessage,
    ChatStreamEvent,
    Role,
    ToolCall,
    ToolSpec,
    UsageReport,
)

SEARCH = ToolSpec(
    "search_notes",
    "Search the notes",
    {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
)
_BASE_TIMEOUT = httpx.Timeout(connect=5.0, read=60.0, write=30.0, pool=5.0)


def _adapter(http: httpx.AsyncClient, budget: int | None) -> GeminiChatModel:
    return GeminiChatModel(http, api_key=SecretStr(CANARY_KEY), model=MODEL, thinking_budget=budget)


def _sent_timeout(route: respx.Route) -> dict[str, float | None]:
    timeout = route.calls.last.request.extensions["timeout"]
    assert isinstance(timeout, dict)
    return timeout


# ---------------------------------------------------------------------- read timeout vs thinking
@pytest.mark.parametrize(
    ("budget", "read"),
    [
        (0, 60.0),  # no thinking: the client's own timeout, untouched
        (8_000, 60.0 + 80.0),  # budget / 100 tokens per second on top
        (32_768, MAX_THINKING_READ_S),  # clamped: a hung connection still fails in minutes
        (-1, MAX_THINKING_READ_S),  # dynamic: no stated size, the full allowance
        (None, MAX_THINKING_READ_S),  # model default (may think): same
    ],
)
@respx.mock
async def test_read_timeout_grows_with_the_thinking_budget(budget: int | None, read: float) -> None:
    route = respx.post(STREAM_URL).mock(return_value=answered())
    async with httpx.AsyncClient(base_url=BASE, timeout=_BASE_TIMEOUT) as http:
        await collect(_adapter(http, budget).stream("", USER))
    sent = _sent_timeout(route)
    assert sent["read"] == pytest.approx(read)
    # Only the read timeout moves: connecting and writing are not slowed by thinking.
    assert (sent["connect"], sent["write"], sent["pool"]) == (5.0, 30.0, 5.0)


@respx.mock
async def test_read_timeout_is_never_shortened_or_invented() -> None:
    route = respx.post(STREAM_URL).mock(return_value=answered())
    long_wait = httpx.Timeout(10.0, read=900.0)
    async with httpx.AsyncClient(base_url=BASE, timeout=long_wait) as http:
        await collect(_adapter(http, 1_000).stream("", USER))
    assert _sent_timeout(route)["read"] == 900.0
    async with httpx.AsyncClient(base_url=BASE, timeout=httpx.Timeout(10.0, read=None)) as http:
        await collect(_adapter(http, -1).stream("", USER))
    assert _sent_timeout(route)["read"] is None


@respx.mock
async def test_thinking_fallback_from_zero_gets_a_short_allowance() -> None:
    rejected = httpx.Response(
        400, json=google_error(400, "INVALID_ARGUMENT", "This model only works in thinking mode.")
    )
    route = respx.post(STREAM_URL).mock(side_effect=[rejected, answered()])
    async with httpx.AsyncClient(base_url=BASE, timeout=_BASE_TIMEOUT) as http:
        await collect(_adapter(http, 0).stream("", USER))
    assert route.calls[0].request.extensions["timeout"]["read"] == 60.0
    # Round 4: the model's default (minimal) level, not the 300 s dynamic allowance.
    assert route.calls[1].request.extensions["timeout"]["read"] == 60.0 + 30.0


# ---------------------------------------------------------------------- failures after output
class _DropsAfterFirstEvent(httpx.AsyncByteStream):
    """One complete event, then the connection breaks (a proxy reset, a pod restart)."""

    def __init__(self, first: bytes) -> None:
        self._first = first

    async def __aiter__(self) -> AsyncIterator[bytes]:
        yield self._first
        raise httpx.ReadError("connection reset by gemini.test")


@pytest.mark.parametrize(
    "first",
    [text_chunk("Kafka is"), call_chunk("search_notes", {"query": "kafka"})],
    ids=["after-text", "after-tool-call"],
)
@respx.mock
async def test_dropped_connection_after_output_is_not_retryable(
    model: GeminiChatModel, first: dict[str, object]
) -> None:
    upstream = _DropsAfterFirstEvent(sse([first]))
    respx.post(STREAM_URL).mock(return_value=httpx.Response(200, stream=upstream))
    seen: list[ChatStreamEvent] = []
    with pytest.raises(StreamInterrupted) as caught:
        await consume_into(model.stream("", USER, [SEARCH]), seen)
    assert len(seen) == 1  # the consumer already has output a replay would duplicate
    assert caught.value.cause == "transient"
    assert type(caught.value) is not ChatModelError  # i.e. not the transient class
    assert caught.value.__cause__ is None
    assert caught.value.__suppress_context__
    assert "gemini.test" not in str(caught.value)


@respx.mock
async def test_dropped_connection_before_output_stays_transient(model: GeminiChatModel) -> None:
    upstream = _DropsAfterFirstEvent(sse([{"candidates": []}]))  # an event, but no output
    respx.post(STREAM_URL).mock(return_value=httpx.Response(200, stream=upstream))
    with pytest.raises(ChatModelError) as caught:
        await collect(model.stream("", USER))
    assert type(caught.value) is ChatModelError
    assert str(caught.value) == "model provider unreachable"


@pytest.mark.parametrize(
    ("finish", "error"),
    [("MAX_TOKENS", OutputTruncated), ("SAFETY", ResponseBlocked)],
)
@respx.mock
async def test_permanent_errors_after_output_keep_their_class(
    model: GeminiChatModel, finish: str, error: type[Exception]
) -> None:
    respx.post(STREAM_URL).mock(return_value=ok([text_chunk("x"), finish_chunk(finish)]))
    with pytest.raises(error) as caught:
        await collect(model.stream("", USER))
    assert not isinstance(caught.value, StreamInterrupted)


@respx.mock
async def test_complete_json_keeps_retry_classes_because_it_emits_nothing_early(
    model: GeminiChatModel,
) -> None:
    schema: dict[str, object] = {"type": "object", "properties": {"a": {"type": "string"}}}
    respx.post(STREAM_URL).mock(
        return_value=ok([text_chunk('{"a": '), google_error(429, "RESOURCE_EXHAUSTED")])
    )
    with pytest.raises(UpstreamRateLimited):
        await model.complete_json("", USER, schema)


# ---------------------------------------------------------------------- Google's call ids
def _two_calls(*ids: object) -> dict[str, object]:
    parts = [
        {"functionCall": {"name": "search_notes", "args": {"query": q}, "id": i}}
        for q, i in zip(("a", "b"), ids, strict=True)
    ]
    return {"candidates": [{"content": {"role": "model", "parts": parts}}]}


@respx.mock
async def test_parallel_calls_to_one_tool_pair_results_by_provider_id(
    model: GeminiChatModel,
) -> None:
    route = respx.post(STREAM_URL).mock(
        side_effect=[ok([_two_calls("g-1", "g-2"), finish_chunk()]), answered()]
    )
    calls = [e for e in await collect(model.stream("", USER, [SEARCH])) if isinstance(e, ToolCall)]
    assert {c.id for c in calls}.isdisjoint({"g-1", "g-2"})
    # Results arrive in the opposite order: the ids still pair each with its call.
    history = [
        *USER,
        ChatMessage(Role.assistant, "", tool_calls=tuple(calls)),
        ChatMessage(Role.tool, "rb", tool_call_id=calls[1].id),
        ChatMessage(Role.tool, "ra", tool_call_id=calls[0].id),
    ]
    await collect(model.stream("", history, [SEARCH]))
    sent = body(route)["contents"]
    assert [p["functionCall"]["id"] for p in sent[1]["parts"]] == ["g-1", "g-2"]
    responses = [p["functionResponse"] for p in sent[2]["parts"]]
    assert [(r["id"], r["response"]["output"]) for r in responses] == [("g-2", "rb"), ("g-1", "ra")]
    wire = route.calls.last.request.content.decode()
    assert all(c.id not in wire for c in calls)  # our minted ids never leave the process


@respx.mock
async def test_unusable_provider_ids_are_not_echoed(model: GeminiChatModel) -> None:
    route = respx.post(STREAM_URL).mock(
        side_effect=[ok([_two_calls("x" * 300, 7), finish_chunk()]), answered()]
    )
    calls = [e for e in await collect(model.stream("", USER, [SEARCH])) if isinstance(e, ToolCall)]
    history = [
        *USER,
        ChatMessage(Role.assistant, "", tool_calls=tuple(calls)),
        *(ChatMessage(Role.tool, "r", tool_call_id=c.id) for c in calls),
    ]
    await collect(model.stream("", history, [SEARCH]))
    sent = body(route)["contents"]
    assert all("id" not in p["functionCall"] for p in sent[1]["parts"])
    assert all("id" not in p["functionResponse"] for p in sent[2]["parts"])


@respx.mock
async def test_evicted_call_metadata_degrades_to_placeholder_and_order(
    model: GeminiChatModel, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(chat_gemini, "_SIGNATURE_CACHE_SIZE", 1)
    route = respx.post(STREAM_URL).mock(
        side_effect=[
            ok(
                [
                    call_chunk("search_notes", {"query": "a"}, signature="s", call_id="g-1"),
                    finish_chunk(),
                ]
            ),
            ok([call_chunk("search_notes", {"query": "b"}, call_id="g-2"), finish_chunk()]),
            answered(),
        ]
    )
    [old] = [e for e in await collect(model.stream("", USER, [SEARCH])) if isinstance(e, ToolCall)]
    await collect(model.stream("", USER, [SEARCH]))  # evicts the first call's metadata
    history = [
        *USER,
        ChatMessage(Role.assistant, "", tool_calls=(old,)),
        ChatMessage(Role.tool, "r", tool_call_id=old.id),
    ]
    events = await collect(model.stream("", history, [SEARCH]))
    assert isinstance(events[-1], UsageReport)
    part = body(route)["contents"][1]["parts"][0]
    assert part["thoughtSignature"] == SKIP_SIGNATURE
    assert "id" not in part["functionCall"]
