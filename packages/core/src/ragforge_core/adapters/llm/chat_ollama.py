"""Ollama ChatModel adapter: `/api/chat` over httpx (ADR-0004).

Request discipline (ADR-0004): a fixed `num_ctx` and `keep_alive` on every request (a changed
`num_ctx` makes Ollama reload the model), and `think: false` for models that support thinking.
Models that reject the field get one retry without it; the outcome is remembered per instance.

Stream contract (domain.ports.ChatModel):
- yields TextDelta / ToolCall (complete, one per call) / exactly one UsageReport, LAST;
- tool-call ids are minted here (uuid4 hex). Provider ids ("call_0", ...) repeat across turns and
  would collide with old rows in pending_approvals (unique per conversation + tool_call_id);
- a stream that ends without Ollama's final `done` record, or carries a malformed record, raises
  `ChatModelError` (never a silent partial answer with made-up usage);
- transport failures (Ollama not running, connection dropped mid-stream, read timeout) raise
  `ChatModelError` with a fixed message: no URL or exception text, which would name internal
  hosts. `CancelledError`/`GeneratorExit` are never caught;
- a generation cut off by the context window (`done_reason == "length"`) raises
  `OutputTruncated` (a `ChatModelError` carrying the spent usage) INSTEAD of the UsageReport, so a
  cut-off answer can never pass as a finished one while its tokens can still be accounted;
- `aclose()` on the iterator exits the `client.stream(...)` block, which closes the response and
  its connection; Ollama aborts generation when the client disconnects.

`complete_json` streams too: the read timeout then bounds the gap between chunks, not the whole
generation, so a long grading call is not killed by a fixed 120 s wall clock.

Tool results: Ollama has no tool-call ids; it matches a tool result to a call by `tool_name` and
position. Results are sent in the order of the messages given (callers must keep call order),
which is what disambiguates two parallel calls to the same tool.

Thinking text (`message.thinking`) is never surfaced. Upstream error bodies are truncated before
they reach exceptions, so prompts echoed back by a proxy cannot flood logs.
"""

from __future__ import annotations

import json
from collections.abc import AsyncGenerator, AsyncIterator, Mapping, Sequence
from contextlib import aclosing
from typing import Any, Final, Self
from uuid import uuid4

import httpx

from ragforge_core.config import Settings
from ragforge_core.domain.errors import RagforgeError, UpstreamRateLimited
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

PROVIDER: Final = "ollama"
_CHAT_PATH: Final = "/api/chat"
_MAX_ERROR_CHARS: Final = 200
_DEFAULT_TIMEOUT: Final = httpx.Timeout(connect=5.0, read=120.0, write=30.0, pool=5.0)


class ChatModelError(RagforgeError):
    """The model provider failed or answered with something that is not a valid response."""

    status = 502
    title = "Model provider error"


class OutputTruncated(ChatModelError):
    """The model stopped because the context window (`num_ctx`) ran out: the answer is cut off.
    `usage` is what the provider billed, so the caller can still account the turn."""

    def __init__(self, usage: ChatTurnUsage) -> None:
        super().__init__("model output truncated: context window exhausted")
        self.usage = usage


class _ThinkUnsupported(Exception):
    """Ollama rejected the `think` field for this model."""


class OllamaChatModel:
    """Implements `domain.ports.ChatModel` against an Ollama server."""

    provider: str = PROVIDER

    def __init__(
        self,
        client: httpx.AsyncClient,
        *,
        model: str,
        num_ctx: int,
        keep_alive: str,
        think: bool | None = False,
        owns_client: bool = False,
    ) -> None:
        """`client` carries the base URL and timeouts. `think=None` never sends the field."""
        if num_ctx < 1:
            raise ValueError("num_ctx must be positive")
        self.model = model
        self._client = client
        self._num_ctx = num_ctx
        self._keep_alive = keep_alive
        self._think = think
        self._owns_client = owns_client

    @classmethod
    def from_settings(cls, settings: Settings, client: httpx.AsyncClient | None = None) -> Self:
        owned = client is None
        http = client or httpx.AsyncClient(
            base_url=settings.ollama_base_url, timeout=_DEFAULT_TIMEOUT
        )
        return cls(
            http,
            model=settings.chat_model,
            num_ctx=settings.chat_num_ctx,
            keep_alive=settings.ollama_keep_alive,
            owns_client=owned,
        )

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    # ------------------------------------------------------------------ ChatModel
    async def stream(
        self,
        system: str,
        messages: Sequence[ChatMessage],
        tools: Sequence[ToolSpec] = (),
    ) -> AsyncIterator[ChatStreamEvent]:
        body: dict[str, Any] = {
            **self._base_body(system, messages),
            "stream": True,
        }
        if tools:
            body["tools"] = [_tool_spec(t) for t in tools]
        for attempt_think in self._think_attempts():
            final: Mapping[str, Any] = {}
            try:
                # aclosing: when the consumer calls aclose() on THIS generator, the inner one
                # (which owns the HTTP response) must be closed now, not whenever it is GC'd.
                async with aclosing(self._records(body, attempt_think)) as records:
                    async for record in records:
                        for event in _events(record):
                            yield event
                        final = record
            except _ThinkUnsupported:  # raised before any record, so nothing was yielded yet
                self._think = None
                continue
            yield UsageReport(self._finished_usage(final))
            return

    async def complete_json(
        self,
        system: str,
        messages: Sequence[ChatMessage],
        schema: dict[str, object],
    ) -> tuple[dict[str, object], ChatTurnUsage]:
        """JSON-schema constrained output (`format`). Temperature 0: these calls grade and
        verify, so they should be repeatable. Schema validation/repair is the caller's policy."""
        body: dict[str, Any] = {
            **self._base_body(system, messages),
            "stream": True,
            "format": schema,
        }
        body["options"] = {**body["options"], "temperature": 0}
        for attempt_think in self._think_attempts():
            parts: list[str] = []
            final: Mapping[str, Any] = {}
            try:
                async with aclosing(self._records(body, attempt_think)) as records:
                    async for record in records:
                        parts.append(_content(record))
                        final = record
            except _ThinkUnsupported:
                self._think = None
                continue
            usage = self._finished_usage(final)
            return _parse_json_content("".join(parts)), usage
        raise ChatModelError("no response")  # pragma: no cover - attempts always yield once

    # ------------------------------------------------------------------ transport
    def _think_attempts(self) -> list[bool | None]:
        """Try with the configured `think`; fall back to omitting it if the model rejects it."""
        return [self._think] if self._think is None else [self._think, None]

    def _base_body(self, system: str, messages: Sequence[ChatMessage]) -> dict[str, Any]:
        return {
            "model": self.model,
            "messages": _to_ollama_messages(system, messages),
            "options": {"num_ctx": self._num_ctx},
            "keep_alive": self._keep_alive,
        }

    async def _records(
        self, body: dict[str, Any], think: bool | None
    ) -> AsyncGenerator[dict[str, Any]]:
        """Decoded NDJSON records, up to and including the final `done` record (later bytes are
        ignored). Exactly one place maps transport failures, before or mid-stream."""
        try:
            async with self._client.stream(
                "POST", _CHAT_PATH, json=_with_think(body, think)
            ) as resp:
                if resp.status_code != httpx.codes.OK:
                    await _raise_for_status(resp, think)
                async for line in resp.aiter_lines():
                    if not line.strip():
                        continue
                    record = _decode_record(line)
                    yield record
                    if record.get("done") is True:
                        return
        except httpx.TimeoutException:
            raise ChatModelError("model provider timed out") from None
        except httpx.HTTPError:  # ConnectError, ReadError, RemoteProtocolError, DecodingError...
            raise ChatModelError("model provider unreachable") from None
        raise ChatModelError("model stream ended before completion")

    def _finished_usage(self, final: Mapping[str, Any]) -> ChatTurnUsage:
        """Usage of a completed generation; a context-window cut-off is an error, not an end."""
        usage = self._usage(final)
        if final.get("done_reason") == "length":
            raise OutputTruncated(usage)
        return usage

    def _usage(self, record: Mapping[str, Any]) -> ChatTurnUsage:
        return ChatTurnUsage(
            provider=PROVIDER,
            model=self.model,
            prompt_tokens=_count(record.get("prompt_eval_count")),
            completion_tokens=_count(record.get("eval_count")),
        )


# ---------------------------------------------------------------------- request mapping
def _with_think(body: dict[str, Any], think: bool | None) -> dict[str, Any]:
    return body if think is None else {**body, "think": think}


def _tool_spec(tool: ToolSpec) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters,
        },
    }


def _to_ollama_messages(system: str, messages: Sequence[ChatMessage]) -> list[dict[str, Any]]:
    """Our ids never leave the process: Ollama correlates a tool result by `tool_name`, which we
    recover from the assistant message that requested the call."""
    out: list[dict[str, Any]] = [{"role": "system", "content": system}] if system else []
    names: dict[str, str] = {}
    for message in messages:
        item: dict[str, Any] = {"role": message.role.value, "content": message.content}
        if message.tool_calls:
            item["tool_calls"] = [
                {"function": {"name": c.name, "arguments": c.arguments}} for c in message.tool_calls
            ]
            names.update({c.id: c.name for c in message.tool_calls})
        if message.role is Role.tool and message.tool_call_id in names:
            item["tool_name"] = names[message.tool_call_id]
        out.append(item)
    return out


# ---------------------------------------------------------------------- response mapping
def _decode_record(raw: str) -> dict[str, Any]:
    try:
        record = json.loads(raw)
    except json.JSONDecodeError:
        raise ChatModelError("malformed record from model provider") from None
    if not isinstance(record, dict):
        raise ChatModelError("malformed record from model provider")
    if "error" in record:
        raise ChatModelError(f"model provider error: {_truncate(str(record['error']))}")
    return record


def _events(record: Mapping[str, Any]) -> list[ChatStreamEvent]:
    message = record.get("message")
    if message is None:
        return []
    if not isinstance(message, dict):
        raise ChatModelError("malformed message from model provider")
    events: list[ChatStreamEvent] = []
    content = message.get("content")
    if isinstance(content, str) and content:
        events.append(TextDelta(content))
    calls = message.get("tool_calls") or []
    if not isinstance(calls, list):
        raise ChatModelError("malformed tool_calls from model provider")
    events.extend(_tool_call(call) for call in calls)
    return events


def _tool_call(call: object) -> ToolCall:
    function = call.get("function") if isinstance(call, dict) else None
    if not isinstance(function, dict) or not isinstance(function.get("name"), str):
        raise ChatModelError("malformed tool call from model provider")
    arguments = function.get("arguments", {})
    if isinstance(arguments, str):  # some models emit the arguments as a JSON string
        try:
            arguments = json.loads(arguments) if arguments.strip() else {}
        except json.JSONDecodeError:
            raise ChatModelError("tool call arguments are not valid JSON") from None
    if not isinstance(arguments, dict):
        raise ChatModelError("tool call arguments must be a JSON object")
    return ToolCall(id=uuid4().hex, name=function["name"], arguments=arguments)


def _content(record: Mapping[str, Any]) -> str:
    message = record.get("message")
    if message is None:
        return ""
    if not isinstance(message, dict):
        raise ChatModelError("malformed message from model provider")
    content = message.get("content", "")
    if not isinstance(content, str):
        raise ChatModelError("malformed message from model provider")
    return content


def _parse_json_content(content: str) -> dict[str, object]:
    if not content.strip():
        raise ChatModelError("structured output missing")
    try:
        value = json.loads(content)
    except json.JSONDecodeError:
        raise ChatModelError("structured output is not valid JSON") from None
    if not isinstance(value, dict):
        raise ChatModelError("structured output must be a JSON object")
    return value


def _count(value: object) -> int:
    return value if isinstance(value, int) and value >= 0 else 0


def _truncate(text: str) -> str:
    return text if len(text) <= _MAX_ERROR_CHARS else text[:_MAX_ERROR_CHARS] + "..."


async def _raise_for_status(resp: httpx.Response, think: bool | None) -> None:
    detail = _truncate((await resp.aread()).decode("utf-8", errors="replace"))
    if resp.status_code in (httpx.codes.TOO_MANY_REQUESTS, httpx.codes.SERVICE_UNAVAILABLE):
        raise UpstreamRateLimited(_retry_after(resp))
    if think is not None and resp.status_code == httpx.codes.BAD_REQUEST and "think" in detail:
        raise _ThinkUnsupported
    raise ChatModelError(f"model provider returned HTTP {resp.status_code}: {detail}")


def _retry_after(resp: httpx.Response) -> float | None:
    try:
        return max(0.0, float(resp.headers["retry-after"]))
    except (KeyError, ValueError):
        return None
