"""Shared builders for module G1 tests: Gemini SSE bodies and request inspection."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Iterable
from typing import Any, Final

import httpx
import respx

from ragforge_core.domain.models import ChatMessage, ChatStreamEvent, Role

BASE: Final = "https://gemini.test"
MODEL: Final = "gemini-test-flash"
STREAM_URL: Final = f"{BASE}/v1beta/models/{MODEL}:streamGenerateContent"
# A recognisable fake key: every security test greps for it in URLs, logs and exception text.
CANARY_KEY: Final = "AIzaCANARY-g1-do-not-leak-0123456789"
USER: Final = (ChatMessage(Role.user, "hello"),)


def sse(chunks: Iterable[dict[str, Any]]) -> bytes:
    """One SSE event per chunk, framed the way the Gemini API frames them."""
    return b"".join(b"data: " + json.dumps(c).encode() + b"\r\n\r\n" for c in chunks)


def text_chunk(text: str) -> dict[str, Any]:
    return {"candidates": [{"content": {"role": "model", "parts": [{"text": text}]}}]}


def finish_chunk(
    reason: str = "STOP", *, prompt: int = 11, output: int = 5, thoughts: int | None = None
) -> dict[str, Any]:
    usage: dict[str, Any] = {
        "promptTokenCount": prompt,
        "candidatesTokenCount": output,
        "totalTokenCount": prompt + output + (thoughts or 0),
    }
    if thoughts is not None:
        usage["thoughtsTokenCount"] = thoughts
    return {
        "candidates": [{"content": {"role": "model", "parts": []}, "finishReason": reason}],
        "usageMetadata": usage,
    }


def call_chunk(
    name: str, args: object, *, signature: str | None = None, call_id: str | None = None
) -> dict[str, Any]:
    call: dict[str, Any] = {"name": name, "args": args}
    if call_id is not None:
        call["id"] = call_id
    part: dict[str, Any] = {"functionCall": call}
    if signature is not None:
        part["thoughtSignature"] = signature
    return {"candidates": [{"content": {"role": "model", "parts": [part]}}]}


def ok(chunks: Iterable[dict[str, Any]]) -> httpx.Response:
    return httpx.Response(200, content=sse(chunks), headers={"content-type": "text/event-stream"})


def answered(text: str = "ok") -> httpx.Response:
    """A minimal successful generation. A bare STOP with no parts is an empty answer, which the
    adapter rejects, so tests that only inspect the request still need some output."""
    return ok([text_chunk(text), finish_chunk()])


def google_error(
    code: int, status: str, message: str = "boom", details: list[dict[str, Any]] | None = None
) -> dict[str, Any]:
    error: dict[str, Any] = {"code": code, "status": status, "message": message}
    if details is not None:
        error["details"] = details
    return {"error": error}


def body(route: respx.Route, call: int = -1) -> dict[str, Any]:
    payload = json.loads(route.calls[call].request.content)
    assert isinstance(payload, dict)
    return payload


async def collect(stream: AsyncIterator[ChatStreamEvent]) -> list[ChatStreamEvent]:
    return [event async for event in stream]


async def consume_into(stream: AsyncIterator[ChatStreamEvent], seen: list[ChatStreamEvent]) -> None:
    """Collects into `seen` so a test can inspect what arrived before the stream raised."""
    async for event in stream:
        seen.append(event)
