"""OllamaChatModel against a mocked /api/chat (respx; no network, no model)."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Iterable
from typing import Any

import httpx
import pytest
import respx

from ragforge_core.adapters.llm.chat_ollama import ChatModelError, OllamaChatModel, OutputTruncated
from ragforge_core.config import Settings
from ragforge_core.domain.errors import UpstreamRateLimited
from ragforge_core.domain.models import (
    ChatMessage,
    ChatStreamEvent,
    Role,
    TextDelta,
    ToolCall,
    ToolSpec,
    UsageReport,
)
from ragforge_core.domain.ports import ChatModel

BASE = "http://ollama.test"
CHAT = f"{BASE}/api/chat"
USER = [ChatMessage(Role.user, "hello")]
SEARCH = ToolSpec("search_notes", "Search the notes", {"type": "object", "properties": {}})


def _ndjson(records: Iterable[dict[str, Any]]) -> bytes:
    return b"".join(json.dumps(r).encode() + b"\n" for r in records)


def _delta(text: str) -> dict[str, Any]:
    return {"message": {"role": "assistant", "content": text}, "done": False}


def _done(prompt: int = 7, completion: int = 3) -> dict[str, Any]:
    return {
        "message": {"role": "assistant", "content": ""},
        "done": True,
        "done_reason": "stop",
        "prompt_eval_count": prompt,
        "eval_count": completion,
    }


def _tool(name: str, arguments: object, provider_id: str = "call_0") -> dict[str, Any]:
    call = {"id": provider_id, "function": {"name": name, "arguments": arguments}}
    return {"message": {"role": "assistant", "content": "", "tool_calls": [call]}, "done": False}


@pytest.fixture
async def model() -> AsyncIterator[OllamaChatModel]:
    async with httpx.AsyncClient(base_url=BASE) as client:
        yield OllamaChatModel(client, model="qwen3:8b", num_ctx=8192, keep_alive="30m")


async def _collect(stream: AsyncIterator[ChatStreamEvent]) -> list[ChatStreamEvent]:
    return [event async for event in stream]


async def _consume_into(
    stream: AsyncIterator[ChatStreamEvent], seen: list[ChatStreamEvent]
) -> None:
    """Collects into `seen` so a test can inspect what arrived before the stream raised."""
    async for event in stream:
        seen.append(event)


def _body(route: respx.Route, call: int = -1) -> dict[str, Any]:
    body = json.loads(route.calls[call].request.content)
    assert isinstance(body, dict)
    return body


def test_adapter_satisfies_the_port(model: OllamaChatModel) -> None:
    assert isinstance(model, ChatModel)
    assert (model.provider, model.model) == ("ollama", "qwen3:8b")


# ---------------------------------------------------------------------- stream: happy path
@respx.mock
async def test_stream_yields_text_then_exactly_one_usage_last(model: OllamaChatModel) -> None:
    respx.post(CHAT).respond(200, content=_ndjson([_delta("Hel"), _delta("lo"), _done(11, 2)]))
    events = await _collect(model.stream("sys", USER))
    assert events[:2] == [TextDelta("Hel"), TextDelta("lo")]
    usages = [e for e in events if isinstance(e, UsageReport)]
    assert len(usages) == 1
    assert events[-1] is usages[0]
    assert usages[0].usage.provider == "ollama"
    assert (usages[0].usage.prompt_tokens, usages[0].usage.completion_tokens) == (11, 2)


@respx.mock
async def test_request_pins_context_keep_alive_and_thinking(model: OllamaChatModel) -> None:
    route = respx.post(CHAT).respond(200, content=_ndjson([_done()]))
    history = [
        ChatMessage(Role.user, "find x"),
        ChatMessage(
            Role.assistant, "", tool_calls=(ToolCall("our-id", "search_notes", {"q": "x"}),)
        ),
        ChatMessage(Role.tool, "result", tool_call_id="our-id"),
    ]
    await _collect(model.stream("be precise", history, tools=[SEARCH]))
    body = _body(route)
    assert body["model"] == "qwen3:8b"
    assert body["stream"] is True
    assert body["think"] is False
    assert body["options"] == {"num_ctx": 8192}
    assert body["keep_alive"] == "30m"
    assert body["messages"][0] == {"role": "system", "content": "be precise"}
    assert body["messages"][2]["tool_calls"] == [
        {"function": {"name": "search_notes", "arguments": {"q": "x"}}}
    ]
    assert body["messages"][3] == {"role": "tool", "content": "result", "tool_name": "search_notes"}
    assert "our-id" not in json.dumps(body), "our tool ids never leave the process"
    assert body["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "search_notes",
                "description": "Search the notes",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]


# ---------------------------------------------------------------------- tool calls
@respx.mock
async def test_tool_call_ids_are_minted_and_unique_across_turns(model: OllamaChatModel) -> None:
    """The provider reuses "call_0" every turn; reusing it would collide with an old approval."""
    respx.post(CHAT).respond(
        200, content=_ndjson([_tool("search_notes", {"q": "a"}), _tool("get_page", "{}"), _done()])
    )
    ids: list[str] = []
    for _ in range(3):
        events = await _collect(model.stream("", USER, tools=[SEARCH]))
        calls = [e for e in events if isinstance(e, ToolCall)]
        assert [c.name for c in calls] == ["search_notes", "get_page"]
        assert calls[0].arguments == {"q": "a"}
        assert calls[1].arguments == {}
        assert isinstance(events[-1], UsageReport)
        ids.extend(c.id for c in calls)
    assert len(set(ids)) == len(ids) == 6
    assert "call_0" not in ids
    assert all(len(i) == 32 and int(i, 16) >= 0 for i in ids)


@respx.mock
async def test_string_arguments_are_parsed_as_json(model: OllamaChatModel) -> None:
    respx.post(CHAT).respond(200, content=_ndjson([_tool("t", '{"k": [1, 2]}'), _done()]))
    events = await _collect(model.stream("", USER))
    assert isinstance(events[0], ToolCall)
    assert events[0].arguments == {"k": [1, 2]}


@pytest.mark.parametrize(
    "bad_call",
    [
        {"function": {"name": "t", "arguments": "{not json"}},
        {"function": {"name": "t", "arguments": [1, 2]}},
        {"function": {"arguments": {}}},
        "not-an-object",
    ],
)
@respx.mock
async def test_malformed_tool_calls_raise(model: OllamaChatModel, bad_call: object) -> None:
    record = {"message": {"role": "assistant", "content": "", "tool_calls": [bad_call]}}
    respx.post(CHAT).respond(200, content=_ndjson([record, _done()]))
    with pytest.raises(ChatModelError):
        await _collect(model.stream("", USER))


# ---------------------------------------------------------------------- malformed streams
@pytest.mark.parametrize(
    "content",
    [
        b'{"message": {"content": "hi"}, "done": false}\n{"message": \n',  # broken JSON
        b"[1, 2]\n",  # not an object
        b'{"message": "text", "done": false}\n',  # message not an object
        b'{"error": "model crashed"}\n',  # in-band error
        _ndjson([_delta("partial")]),  # truncated: no final done record
        b"",  # empty body
    ],
)
@respx.mock
async def test_malformed_streams_raise_and_never_report_usage(
    model: OllamaChatModel, content: bytes
) -> None:
    respx.post(CHAT).respond(200, content=content)
    seen: list[ChatStreamEvent] = []

    async def consume() -> None:
        async for event in model.stream("", USER):
            seen.append(event)

    with pytest.raises(ChatModelError):
        await consume()
    assert not any(isinstance(e, UsageReport) for e in seen)


@respx.mock
async def test_blank_lines_and_trailing_records_after_done_are_ignored(
    model: OllamaChatModel,
) -> None:
    content = b"\n" + _ndjson([_delta("a")]) + b"\n\n" + _ndjson([_done(), _delta("late")])
    respx.post(CHAT).respond(200, content=content)
    events = await _collect(model.stream("", USER))
    assert events[0] == TextDelta("a")
    assert isinstance(events[-1], UsageReport)
    assert TextDelta("late") not in events


@respx.mock
async def test_thinking_text_is_never_surfaced(model: OllamaChatModel) -> None:
    record = {"message": {"role": "assistant", "content": "", "thinking": "secret plan"}}
    respx.post(CHAT).respond(200, content=_ndjson([record, _delta("answer"), _done()]))
    events = await _collect(model.stream("", USER))
    assert [e for e in events if isinstance(e, TextDelta)] == [TextDelta("answer")]


# ---------------------------------------------------------------------- HTTP errors
@respx.mock
async def test_rate_limited_maps_to_upstream_rate_limited(model: OllamaChatModel) -> None:
    respx.post(CHAT).respond(503, headers={"retry-after": "2"}, json={"error": "busy"})
    with pytest.raises(UpstreamRateLimited) as exc:
        await _collect(model.stream("", USER))
    assert exc.value.retry_after_s == 2.0


@respx.mock
async def test_server_error_is_a_truncated_chat_model_error(model: OllamaChatModel) -> None:
    respx.post(CHAT).respond(500, text="x" * 5000)
    with pytest.raises(ChatModelError) as exc:
        await _collect(model.stream("", USER))
    assert "HTTP 500" in str(exc.value)
    assert len(str(exc.value)) < 300


@respx.mock
async def test_models_without_thinking_get_one_retry_without_the_field(
    model: OllamaChatModel,
) -> None:
    route = respx.post(CHAT).mock(
        side_effect=[
            httpx.Response(400, json={"error": '"llama3.2:3b" does not support thinking'}),
            httpx.Response(200, content=_ndjson([_delta("ok"), _done()])),
            httpx.Response(200, content=_ndjson([_done()])),
        ]
    )
    events = await _collect(model.stream("", USER))
    assert events[0] == TextDelta("ok")
    assert _body(route, 0)["think"] is False
    assert "think" not in _body(route, 1)
    await _collect(model.stream("", USER))  # remembered: no second failed attempt
    assert route.call_count == 3
    assert "think" not in _body(route, 2)


@respx.mock
async def test_other_bad_requests_are_not_retried(model: OllamaChatModel) -> None:
    route = respx.post(CHAT).respond(400, json={"error": "invalid tools"})
    with pytest.raises(ChatModelError):
        await _collect(model.stream("", USER))
    assert route.call_count == 1


# ---------------------------------------------------------------------- cancellation
class _HangingBody(httpx.AsyncByteStream):
    """Sends one record, then blocks like a model that keeps generating; records aclose()."""

    def __init__(self) -> None:
        self.closed = asyncio.Event()
        self.first_sent = asyncio.Event()

    async def __aiter__(self) -> AsyncIterator[bytes]:
        yield _ndjson([_delta("first")])
        self.first_sent.set()
        await asyncio.Event().wait()  # never finishes on its own
        yield b""  # pragma: no cover

    async def aclose(self) -> None:
        self.closed.set()


@respx.mock
async def test_aclose_cancels_the_upstream_request(model: OllamaChatModel) -> None:
    body = _HangingBody()
    respx.post(CHAT).mock(return_value=httpx.Response(200, stream=body))
    stream = model.stream("", USER)
    assert await anext(stream) == TextDelta("first")
    await asyncio.wait_for(stream.aclose(), timeout=2)  # type: ignore[attr-defined]
    assert body.closed.is_set(), "closing the iterator must close the upstream response"


@respx.mock
async def test_task_cancellation_closes_the_upstream_request(model: OllamaChatModel) -> None:
    body = _HangingBody()
    respx.post(CHAT).mock(return_value=httpx.Response(200, stream=body))

    async def consume() -> None:
        async for _ in model.stream("", USER):
            pass

    task = asyncio.create_task(consume())
    await asyncio.wait_for(body.first_sent.wait(), timeout=2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert body.closed.is_set()


# ---------------------------------------------------------------------- complete_json
SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {"verdict": {"type": "string"}},
    "required": ["verdict"],
}


@respx.mock
async def test_complete_json_uses_schema_format(model: OllamaChatModel) -> None:
    record = {
        "message": {"role": "assistant", "content": '{"verdict": "supported"}'},
        "done": True,
        "prompt_eval_count": 20,
        "eval_count": 5,
    }
    route = respx.post(CHAT).respond(200, json=record)
    value, usage = await model.complete_json("grade", USER, SCHEMA)
    assert value == {"verdict": "supported"}
    assert (usage.prompt_tokens, usage.completion_tokens, usage.model) == (20, 5, "qwen3:8b")
    body = _body(route)
    assert body["format"] == SCHEMA
    assert body["stream"] is True, "streamed so the read timeout bounds chunk gaps only"
    assert body["think"] is False
    assert body["options"] == {"num_ctx": 8192, "temperature": 0}
    assert body["keep_alive"] == "30m"
    assert "tools" not in body


@pytest.mark.parametrize(
    "record",
    [
        {"message": {"content": "not json"}, "done": True},
        {"message": {"content": "[1]"}, "done": True},
        {"message": {}, "done": True},
        {"message": {"content": 5}, "done": True},
        {"error": "boom"},
        {"message": {"content": '{"verdict": "x"}'}, "done": False},  # no done record
    ],
)
@respx.mock
async def test_complete_json_rejects_invalid_output(
    model: OllamaChatModel, record: dict[str, Any]
) -> None:
    respx.post(CHAT).respond(200, json=record)
    with pytest.raises(ChatModelError):
        await model.complete_json("grade", USER, SCHEMA)


@respx.mock
async def test_from_settings_uses_configured_model_and_owns_its_client() -> None:
    settings = Settings(
        _env_file=None, ollama_base_url=BASE, chat_model="llama3.2:3b", chat_num_ctx=4096
    )
    route = respx.post(CHAT).respond(200, content=_ndjson([_done()]))
    chat = OllamaChatModel.from_settings(settings)
    await _collect(chat.stream("", USER))
    await chat.aclose()
    body = _body(route)
    assert (body["model"], body["options"]["num_ctx"]) == ("llama3.2:3b", 4096)
    assert body["keep_alive"] == settings.ollama_keep_alive


@respx.mock
async def test_complete_json_assembles_streamed_chunks(model: OllamaChatModel) -> None:
    chunks = [_delta('{"verd'), _delta('ict": "supp'), _delta('orted"}'), _done(9, 4)]
    respx.post(CHAT).respond(200, content=_ndjson(chunks))
    value, usage = await model.complete_json("grade", USER, SCHEMA)
    assert value == {"verdict": "supported"}
    assert (usage.prompt_tokens, usage.completion_tokens) == (9, 4)


# ---------------------------------------------------------------------- truncation
def _truncated_done() -> dict[str, Any]:
    return {**_done(4000, 96), "done_reason": "length"}


@respx.mock
async def test_context_window_cut_off_is_an_error_not_a_finished_answer(
    model: OllamaChatModel,
) -> None:
    respx.post(CHAT).respond(200, content=_ndjson([_delta("The answer is"), _truncated_done()]))
    seen: list[ChatStreamEvent] = []
    with pytest.raises(OutputTruncated) as exc:
        await _consume_into(model.stream("", USER), seen)
    assert seen == [TextDelta("The answer is")]
    assert (exc.value.usage.prompt_tokens, exc.value.usage.completion_tokens) == (4000, 96)
    assert exc.value.status == 502


@respx.mock
async def test_complete_json_cut_off_raises_truncated(model: OllamaChatModel) -> None:
    respx.post(CHAT).respond(200, content=_ndjson([_delta('{"verdict": "sup'), _truncated_done()]))
    with pytest.raises(OutputTruncated):
        await model.complete_json("grade", USER, SCHEMA)


# ---------------------------------------------------------------------- transport failures
class _FailingBody(httpx.AsyncByteStream):
    """Sends one record, then the connection fails the way a dropped/stalled upstream does."""

    def __init__(self, error: Exception) -> None:
        self.error = error

    async def __aiter__(self) -> AsyncIterator[bytes]:
        yield _ndjson([_delta("partial")])
        raise self.error


TRANSPORT_ERRORS = [
    pytest.param(httpx.ConnectError("[Errno 61] connect to ollama.test:11434"), id="connect"),
    pytest.param(httpx.ReadTimeout("read timed out ollama.test"), id="read-timeout"),
    pytest.param(httpx.ConnectTimeout("connect timed out ollama.test"), id="connect-timeout"),
]
MID_STREAM_ERRORS = [
    pytest.param(httpx.ReadTimeout("read timed out ollama.test"), id="read-timeout"),
    pytest.param(httpx.RemoteProtocolError("peer closed ollama.test"), id="dropped"),
    pytest.param(httpx.ReadError("reset by ollama.test"), id="read-error"),
]


def _assert_sanitized(exc: BaseException) -> None:
    assert isinstance(exc, ChatModelError)
    assert "ollama.test" not in str(exc), "internal host names never reach clients"
    assert exc.__cause__ is None
    assert exc.__suppress_context__


@pytest.mark.parametrize("error", TRANSPORT_ERRORS)
@respx.mock
async def test_transport_failure_before_first_byte_is_a_chat_model_error(
    model: OllamaChatModel, error: Exception
) -> None:
    respx.post(CHAT).mock(side_effect=error)
    with pytest.raises(ChatModelError) as exc:
        await _collect(model.stream("", USER))
    _assert_sanitized(exc.value)
    with pytest.raises(ChatModelError) as exc:
        await model.complete_json("grade", USER, SCHEMA)
    _assert_sanitized(exc.value)


@pytest.mark.parametrize("error", MID_STREAM_ERRORS)
@respx.mock
async def test_transport_failure_mid_stream_is_a_chat_model_error(
    model: OllamaChatModel, error: Exception
) -> None:
    respx.post(CHAT).mock(return_value=httpx.Response(200, stream=_FailingBody(error)))
    seen: list[ChatStreamEvent] = []
    with pytest.raises(ChatModelError) as exc:
        await _consume_into(model.stream("", USER), seen)
    _assert_sanitized(exc.value)
    assert seen == [TextDelta("partial")], "no usage after a failed stream"
    with pytest.raises(ChatModelError) as exc:
        await model.complete_json("grade", USER, SCHEMA)
    _assert_sanitized(exc.value)


@respx.mock
async def test_timeout_message_says_timed_out(model: OllamaChatModel) -> None:
    respx.post(CHAT).mock(side_effect=httpx.ReadTimeout("x"))
    with pytest.raises(ChatModelError, match="timed out"):
        await _collect(model.stream("", USER))


# ---------------------------------------------------------------------- tool results order
@respx.mock
async def test_same_name_tool_results_keep_call_order(model: OllamaChatModel) -> None:
    """Ollama has no tool-call ids: two results of the same tool are told apart only by
    position, so the request must carry them exactly in call order."""
    route = respx.post(CHAT).respond(200, content=_ndjson([_done()]))
    first = ToolCall("id-1", "search_notes", {"q": "alpha"})
    second = ToolCall("id-2", "search_notes", {"q": "beta"})
    history = [
        ChatMessage(Role.user, "compare"),
        ChatMessage(Role.assistant, "", tool_calls=(first, second)),
        ChatMessage(Role.tool, "alpha result", tool_call_id="id-1"),
        ChatMessage(Role.tool, "beta result", tool_call_id="id-2"),
    ]
    await _collect(model.stream("", history, tools=[SEARCH]))
    messages = _body(route)["messages"]
    assert [c["function"]["arguments"] for c in messages[1]["tool_calls"]] == [
        {"q": "alpha"},
        {"q": "beta"},
    ]
    assert messages[2:] == [
        {"role": "tool", "content": "alpha result", "tool_name": "search_notes"},
        {"role": "tool", "content": "beta result", "tool_name": "search_notes"},
    ]
