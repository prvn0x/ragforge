"""Minimal, spec-following Server-Sent Events parser (WHATWG HTML, "event stream" section).

Only what the RAGForge streams use: `event:` and `data:` fields, multi-line data, comments
(`: keepalive`), and dispatch on a blank line. `id:`/`retry:` are ignored (no reconnection: a
dropped chat turn is re-asked, and job streams can simply be reopened).
"""

from __future__ import annotations

from collections.abc import AsyncIterable, AsyncIterator
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SseMessage:
    event: str
    data: str


async def iter_sse(lines: AsyncIterable[str]) -> AsyncIterator[SseMessage]:
    event = ""
    data: list[str] = []
    async for raw in lines:
        line = raw.rstrip("\r\n")
        if line == "":
            if data:
                yield SseMessage(event=event or "message", data="\n".join(data))
            event, data = "", []
            continue
        if line.startswith(":"):
            continue  # comment / keepalive
        name, _, value = line.partition(":")
        value = value.removeprefix(" ")
        if name == "event":
            event = value
        elif name == "data":
            data.append(value)
    if data:  # a well-formed stream ends with a blank line; tolerate a missing final one
        yield SseMessage(event=event or "message", data="\n".join(data))
