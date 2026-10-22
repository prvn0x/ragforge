"""Regression tests for the round-5 review: a STOP with no text and no tool call is a typed,
transient error (never a silent empty answer), and the thought-signature cache is a true LRU (a
replay refreshes an entry, so an active conversation keeps its signature)."""

from __future__ import annotations

from typing import Any

import httpx
import pytest
import respx

from module_g1.g1_support import (
    CANARY_KEY,
    STREAM_URL,
    USER,
    answered,
    body,
    call_chunk,
    collect,
    consume_into,
    finish_chunk,
    ok,
)
from ragforge_core.adapters.llm import chat_gemini
from ragforge_core.adapters.llm.chat_gemini import SKIP_SIGNATURE, GeminiChatModel
from ragforge_core.adapters.llm.chat_ollama import ChatModelError
from ragforge_core.domain.models import (
    ChatMessage,
    ChatStreamEvent,
    Role,
    ToolCall,
    ToolSpec,
    UsageReport,
)

SEARCH = ToolSpec(
    name="search_notes",
    description="Search the notes.",
    parameters={"type": "object", "properties": {"query": {"type": "string"}}},
)


def _thought_only() -> dict[str, Any]:
    parts = [{"text": "pondering", "thought": True}]
    return {"candidates": [{"content": {"role": "model", "parts": parts}}]}


# ---------------------------------------------------------------------- empty STOP
@pytest.mark.parametrize(
    "chunks",
    [
        pytest.param([finish_chunk()], id="stop-without-parts"),
        pytest.param([_thought_only(), finish_chunk(thoughts=40)], id="thinking-ate-the-output"),
    ],
)
@respx.mock
async def test_stop_without_output_is_a_transient_error_not_an_empty_answer(
    model: GeminiChatModel, chunks: list[dict[str, Any]]
) -> None:
    respx.post(STREAM_URL).mock(return_value=ok(chunks))
    seen: list[ChatStreamEvent] = []
    with pytest.raises(ChatModelError) as caught:
        await consume_into(model.stream("", USER), seen)
    # Exactly the base class: transient and retryable, because nothing was shown.
    assert type(caught.value) is ChatModelError
    assert str(caught.value) == "model returned an empty answer"
    assert seen == []  # no UsageReport: the turn did not succeed
    assert CANARY_KEY not in str(caught.value)
    assert "pondering" not in str(caught.value)


@respx.mock
async def test_empty_answer_after_a_tool_result_raises_and_a_retry_can_succeed(
    model: GeminiChatModel,
) -> None:
    """The production shape: the model goes silent right after a functionResponse turn."""
    respx.post(STREAM_URL).mock(
        side_effect=[
            ok([call_chunk("search_notes", {"query": "kafka"}), finish_chunk()]),
            ok([finish_chunk()]),
            answered("Kafka is a log."),
        ]
    )
    [call] = [e for e in await collect(model.stream("", USER, [SEARCH])) if isinstance(e, ToolCall)]
    history = [
        *USER,
        ChatMessage(Role.assistant, "", tool_calls=(call,)),
        ChatMessage(Role.tool, "Kafka is a log.", tool_call_id=call.id),
    ]
    with pytest.raises(ChatModelError, match="empty answer"):
        await collect(model.stream("", history, [SEARCH]))
    events = await collect(model.stream("", history, [SEARCH]))
    assert isinstance(events[-1], UsageReport)


@respx.mock
async def test_a_tool_call_alone_is_output_not_an_empty_answer(model: GeminiChatModel) -> None:
    respx.post(STREAM_URL).mock(return_value=ok([call_chunk("search_notes", {}), finish_chunk()]))
    events = await collect(model.stream("", USER, [SEARCH]))
    assert [type(e) for e in events] == [ToolCall, UsageReport]


# ---------------------------------------------------------------------- LRU signature cache
def _many_signed_calls(count: int) -> dict[str, Any]:
    parts = [
        {
            "functionCall": {"name": "search_notes", "args": {"query": str(i)}},
            "thoughtSignature": f"n{i}",
        }
        for i in range(count)
    ]
    return {"candidates": [{"content": {"role": "model", "parts": parts}}]}


def _signed(signature: str) -> httpx.Response:
    return ok(
        [call_chunk("search_notes", {"query": signature}, signature=signature), finish_chunk()]
    )


def _tool_turn(call: ToolCall) -> list[ChatMessage]:
    return [
        ChatMessage(Role.assistant, "", tool_calls=(call,)),
        ChatMessage(Role.tool, "r", tool_call_id=call.id),
    ]


async def _one_call(model: GeminiChatModel) -> ToolCall:
    [call] = [e for e in await collect(model.stream("", USER, [SEARCH])) if isinstance(e, ToolCall)]
    return call


@respx.mock
async def test_a_replayed_entry_survives_a_full_cache_of_newer_inserts(
    model: GeminiChatModel,
) -> None:
    """A then B are cached; replaying A refreshes it; 1023 newer calls then fill the cache.
    FIFO would evict A (the oldest insert); the LRU evicts B (the least recently used)."""
    size = chat_gemini._SIGNATURE_CACHE_SIZE
    route = respx.post(STREAM_URL).mock(
        side_effect=[
            _signed("sig-A"),
            _signed("sig-B"),
            answered(),  # the replay of A
            ok([_many_signed_calls(size - 1), finish_chunk()]),
            answered(),  # the final replay of A and B
        ]
    )
    call_a = await _one_call(model)
    call_b = await _one_call(model)
    await collect(model.stream("", [*USER, *_tool_turn(call_a)], [SEARCH]))
    newer = await collect(model.stream("", USER, [SEARCH]))
    assert sum(isinstance(e, ToolCall) for e in newer) == size - 1

    await collect(model.stream("", [*USER, *_tool_turn(call_a), *_tool_turn(call_b)], [SEARCH]))
    contents = body(route)["contents"]
    assert contents[1]["parts"][0]["thoughtSignature"] == "sig-A"
    assert contents[3]["parts"][0]["thoughtSignature"] == SKIP_SIGNATURE


@respx.mock
async def test_an_entry_never_replayed_is_evicted_by_insertion_age(
    model: GeminiChatModel, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The contrast case: without a replay, the oldest entry is the one that goes."""
    monkeypatch.setattr(chat_gemini, "_SIGNATURE_CACHE_SIZE", 2)
    route = respx.post(STREAM_URL).mock(
        side_effect=[_signed("sig-A"), _signed("sig-B"), _signed("sig-C"), answered()]
    )
    call_a = await _one_call(model)
    call_b = await _one_call(model)
    await _one_call(model)
    await collect(model.stream("", [*USER, *_tool_turn(call_a), *_tool_turn(call_b)], [SEARCH]))
    contents = body(route)["contents"]
    assert contents[1]["parts"][0]["thoughtSignature"] == SKIP_SIGNATURE
    assert contents[3]["parts"][0]["thoughtSignature"] == "sig-B"
