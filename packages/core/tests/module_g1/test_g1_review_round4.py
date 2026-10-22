"""Regression tests for the round-3 review: after the thinking fallback from budget 0 the read
timeout stays short for the life of the adapter, the SSE size cap holds while bytes arrive (not
after a whole line is buffered), our own line splitter frames events exactly like the SSE spec
whatever the network read boundaries, and an orphan tool result is a typed, fixed-text error."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Iterable

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
    collect,
    finish_chunk,
    google_error,
    sse,
    text_chunk,
)
from ragforge_core.adapters.llm.chat_gemini import (
    MAX_THINKING_READ_S,
    GeminiChatModel,
    ModelRequestRejected,
)
from ragforge_core.adapters.llm.chat_ollama import ChatModelError
from ragforge_core.domain.errors import ValidationFailed
from ragforge_core.domain.models import ChatMessage, Role, TextDelta, UsageReport

_BASE_TIMEOUT = httpx.Timeout(connect=5.0, read=60.0, write=30.0, pool=5.0)
_REJECTED_THINKING = google_error(
    400, "INVALID_ARGUMENT", "This model only works in thinking mode."
)


def _adapter(http: httpx.AsyncClient, budget: int | None) -> GeminiChatModel:
    return GeminiChatModel(http, api_key=SecretStr(CANARY_KEY), model=MODEL, thinking_budget=budget)


def _read_timeouts(route: respx.Route) -> list[float | None]:
    return [call.request.extensions["timeout"]["read"] for call in route.calls]


# ---------------------------------------------------------------------- timeout after fallback
@respx.mock
async def test_after_a_refused_zero_budget_every_later_request_keeps_a_short_timeout() -> None:
    """The bug: the fallback set the budget to None, which the timeout read as 'dynamic', so
    every later request of the process waited 300 s for a byte from a hung connection."""
    route = respx.post(STREAM_URL).mock(
        side_effect=[
            httpx.Response(400, json=_REJECTED_THINKING),
            answered(),
            answered(),
            answered(),
        ]
    )
    async with httpx.AsyncClient(base_url=BASE, timeout=_BASE_TIMEOUT) as http:
        model = _adapter(http, 0)
        for _ in range(3):
            await collect(model.stream("", USER))
    assert _read_timeouts(route) == [60.0, 90.0, 90.0, 90.0]
    # The fallback itself is unchanged: later requests no longer send thinkingConfig.
    assert all("thinkingConfig" not in body(route, i)["generationConfig"] for i in (1, 2, 3))


@pytest.mark.parametrize(
    ("budget", "read"),
    [
        (8_000, 60.0 + 80.0),  # a stated budget keeps its own bound after a fallback
        (-1, MAX_THINKING_READ_S),  # dynamic was asked for: the full allowance stays
    ],
)
@respx.mock
async def test_a_fallback_from_a_nonzero_budget_keeps_the_configured_allowance(
    budget: int, read: float
) -> None:
    route = respx.post(STREAM_URL).mock(
        side_effect=[
            httpx.Response(400, json=google_error(400, "INVALID_ARGUMENT", "thinking_budget bad")),
            answered(),
            answered(),
        ]
    )
    async with httpx.AsyncClient(base_url=BASE, timeout=_BASE_TIMEOUT) as http:
        model = _adapter(http, budget)
        await collect(model.stream("", USER))
        await collect(model.stream("", USER))
    assert _read_timeouts(route) == [read, read, read]
    assert "thinkingConfig" not in body(route, 2)["generationConfig"]


@respx.mock
async def test_a_failed_fallback_keeps_sending_the_budget_with_no_allowance() -> None:
    """The retry is rejected too (the 400 was not about thinking): nothing sticks."""
    rejected = httpx.Response(400, json=_REJECTED_THINKING)
    route = respx.post(STREAM_URL).mock(side_effect=[rejected, rejected, answered()])
    async with httpx.AsyncClient(base_url=BASE, timeout=_BASE_TIMEOUT) as http:
        model = _adapter(http, 0)
        with pytest.raises(ModelRequestRejected):
            await collect(model.stream("", USER))
        await collect(model.stream("", USER))
    assert _read_timeouts(route) == [60.0, 90.0, 60.0]
    assert body(route, 2)["generationConfig"]["thinkingConfig"] == {"thinkingBudget": 0}


# ---------------------------------------------------------------------- SSE size cap
class _Counted(httpx.AsyncByteStream):
    """Yields the given pieces in order and records how many the adapter pulled."""

    def __init__(self, pieces: Iterable[bytes]) -> None:
        self._pieces = list(pieces)
        self.pulled = 0

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for piece in self._pieces:
            self.pulled += 1
            yield piece


@respx.mock
async def test_an_endless_line_fails_while_it_arrives_not_after_it_is_buffered(
    model: GeminiChatModel,
) -> None:
    """16 MiB with no line break. `aiter_lines()` held all of it before the cap was checked
    (it needs a newline to hand over a line); the adapter now stops at the first read past
    4 MiB, so memory stays bounded however long the line is."""
    mib = b"x" * (1024 * 1024)
    upstream = _Counted([b"data: " + mib, *([mib] * 15)])
    respx.post(STREAM_URL).mock(return_value=httpx.Response(200, stream=upstream))
    with pytest.raises(ChatModelError, match=r"^oversized event from model provider$"):
        await collect(model.stream("", USER))
    assert upstream.pulled == 4  # 4 MiB + 6 bytes is over the cap; nothing after it is read


@respx.mock
async def test_an_endless_comment_line_is_capped_too(model: GeminiChatModel) -> None:
    """Comment and unknown-field lines are ignored by SSE, but still cannot grow without limit."""
    upstream = _Counted([b": keep-alive", *([b"z" * (1024 * 1024)] * 8)])
    respx.post(STREAM_URL).mock(return_value=httpx.Response(200, stream=upstream))
    with pytest.raises(ChatModelError, match="oversized event"):
        await collect(model.stream("", USER))
    assert upstream.pulled == 5


# ---------------------------------------------------------------------- SSE framing
def _texts(events: Iterable[object]) -> str:
    return "".join(e.text for e in events if isinstance(e, TextDelta))


@pytest.mark.parametrize("terminator", [b"\r\n", b"\n", b"\r"])
@respx.mock
async def test_events_frame_identically_whatever_the_read_boundaries(
    model: GeminiChatModel, terminator: bytes
) -> None:
    """Every byte arrives in its own read: CRLF split between reads must not invent a blank
    line (which would end an event early), and a multi-byte character split between reads must
    decode intact. U+2028 is legal unescaped inside a JSON string and must not split a line."""
    text = (
        "caf" + chr(0xE9) + " " + chr(0x2028) + " " + chr(0x65E5) + chr(0x672C) + " " + chr(0x1F600)
    )
    events = [text_chunk(text), finish_chunk()]
    raw = b"".join(
        b"data: " + json.dumps(c, ensure_ascii=False).encode() + terminator * 2 for c in events
    )
    upstream = _Counted(raw[i : i + 1] for i in range(len(raw)))
    respx.post(STREAM_URL).mock(return_value=httpx.Response(200, stream=upstream))
    seen = await collect(model.stream("", USER))
    assert _texts(seen) == text
    assert isinstance(seen[-1], UsageReport)


@respx.mock
async def test_multi_line_data_split_across_reads_is_one_event(model: GeminiChatModel) -> None:
    """One event whose JSON spans two `data:` lines, with the CRLF between them split."""
    payload = json.dumps(text_chunk("joined")).encode()
    half = payload.index(b"[")  # between JSON tokens: the joining newline is whitespace
    pieces = [
        b"data: " + payload[:half] + b"\r",
        b"\ndata: " + payload[half:] + b"\r",
        b"\n\r\n" + sse([finish_chunk()]),
    ]
    respx.post(STREAM_URL).mock(return_value=httpx.Response(200, stream=_Counted(pieces)))
    assert _texts(await collect(model.stream("", USER))) == "joined"


@respx.mock
async def test_a_last_event_without_a_terminator_still_counts(model: GeminiChatModel) -> None:
    raw = sse([text_chunk("a")]) + b"data: " + json.dumps(finish_chunk()).encode() + b"\r"
    respx.post(STREAM_URL).mock(return_value=httpx.Response(200, stream=_Counted([raw])))
    seen = await collect(model.stream("", USER))
    assert _texts(seen) == "a"
    assert isinstance(seen[-1], UsageReport)


@respx.mock
async def test_invalid_utf8_is_a_malformed_event_not_a_crash(model: GeminiChatModel) -> None:
    raw = b'data: {"candidates": [\xff]}\r\n\r\n'
    respx.post(STREAM_URL).mock(return_value=httpx.Response(200, stream=_Counted([raw])))
    with pytest.raises(ChatModelError, match=r"^malformed event from model provider$"):
        await collect(model.stream("", USER))


# ---------------------------------------------------------------------- orphan tool result
@respx.mock
async def test_an_orphan_tool_result_is_a_typed_permanent_error_before_any_request(
    model: GeminiChatModel,
) -> None:
    route = respx.post(STREAM_URL).mock(return_value=answered())
    secret_result = "orphan result text that must not leak"
    orphan = [*USER, ChatMessage(Role.tool, secret_result, tool_call_id="never-issued")]
    with pytest.raises(ModelRequestRejected) as caught:
        await collect(model.stream("", orphan))
    error = caught.value
    # Inside the ChatModelError contract (callers' `except ChatModelError` covers it) and
    # permanent (not the transient base class), never a bare ValueError / HTTP 500.
    assert isinstance(error, ChatModelError)
    assert isinstance(error, ValidationFailed)
    assert type(error) is not ChatModelError
    assert not isinstance(error, ValueError)
    assert str(error) == "tool result does not answer a tool call in this history"
    assert secret_result not in repr(error)
    assert "never-issued" not in str(error)
    assert not route.called
