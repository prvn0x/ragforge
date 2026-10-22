"""Fakes for the MCP module tests: a respx-mocked RAGForge API, SSE bodies, and a connected
in-memory MCP client. No network, no model: every API response is built here."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Iterable, Sequence
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

import httpx
import respx
from mcp.client import Client
from mcp.types import CallToolResult, TextContent

from ragforge_client.client import RagforgeClient
from ragforge_contracts.api import Citation
from ragforge_contracts.sse import ChatEvent, encode_sse
from ragforge_mcp.errors import Redactor
from ragforge_mcp.server import build_server
from ragforge_mcp.service import RagforgeService

API = "http://rf.test"
UI = "http://ui.test"
# A recognisable secret: every test that can surface text asserts it never appears.
KEY = "rf_live_SUPERSECRET_9f8e7d6c5b4a"
NOW = datetime(2026, 10, 18, 12, 0, tzinfo=UTC)

COLL_A = UUID("00000000-0000-4000-8000-0000000000a1")
COLL_B = UUID("00000000-0000-4000-8000-0000000000b2")


def collection(cid: UUID, name: str) -> dict[str, Any]:
    return {"id": str(cid), "name": name, "active_space_id": 1, "created_at": NOW.isoformat()}


def page(items: Sequence[dict[str, Any]], next_cursor: str | None = None) -> dict[str, Any]:
    return {"items": list(items), "next_cursor": next_cursor}


def connector(
    cid: UUID, name: str, *, kind: str = "notion", status: str = "active", coll: UUID = COLL_A
) -> dict[str, Any]:
    return {
        "id": str(cid),
        "kind": kind,
        "target_collection_id": str(coll),
        "display_name": name,
        "status": status,
        "last_error": None,
        "created_at": NOW.isoformat(),
    }


def document(did: UUID, title: str, coll: UUID = COLL_A) -> dict[str, Any]:
    return {
        "id": str(did),
        "collection_id": str(coll),
        "title": title,
        "external_url": "https://notion.so/page",
        "external_updated_at": NOW.isoformat(),
        "fetched_at": None,
    }


def cite(label: int | None, title: str = "Kafka", *, inferred: bool = False) -> Citation:
    return Citation(
        anchor=f"blk-{label}",
        document_id=uuid4(),
        title=title,
        heading_path=["Consumers", "Rebalance"],
        url=None,
        snippet="max.poll.interval.ms bounds processing time",
        inferred=inferred,
        label=label,
    )


def problem(status: int, title: str, detail: str | None = None, **extra: str) -> httpx.Response:
    body: dict[str, Any] = {"title": title, "status": status, "detail": detail, **extra}
    return httpx.Response(status, json=body, headers={"content-type": "application/problem+json"})


def sse(events: Iterable[ChatEvent]) -> httpx.Response:
    body = b"".join(encode_sse(e) for e in events)
    return httpx.Response(200, content=body, headers={"content-type": "text/event-stream"})


class BrokenStream(httpx.AsyncByteStream):
    """An SSE body that delivers `head`, then the connection drops."""

    def __init__(self, head: bytes) -> None:
        self._head = head

    async def __aiter__(self) -> AsyncIterator[bytes]:
        yield self._head
        raise httpx.ReadError("connection reset by peer")


def mock_api(collections: Sequence[dict[str, Any]] | None = None) -> respx.MockRouter:
    """A router with the collection list in place (every tool resolves its scope from it)."""
    router = respx.mock(base_url=API, assert_all_called=False)
    items = [collection(COLL_A, "notes")] if collections is None else collections
    router.get("/v1/collections", name="collections").mock(
        return_value=httpx.Response(200, json=page(items))
    )
    return router


def make_service(collections: Sequence[str] = ()) -> tuple[RagforgeService, RagforgeClient]:
    client = RagforgeClient(API, KEY)
    service = RagforgeService(
        client, api_url=API, ui_url=UI, redact=Redactor(KEY), collections=collections
    )
    return service, client


@asynccontextmanager
async def connected(collections: Sequence[str] = ()) -> AsyncIterator[Client]:
    """An MCP client talking to a fresh server in-process (the SDK's own in-memory path)."""
    service, rf = make_service(collections)
    async with rf, Client(build_server(service)) as client:
        yield client


def text_of(result: CallToolResult) -> str:
    return "\n".join(c.text for c in result.content if isinstance(c, TextContent))


def structured(result: CallToolResult) -> dict[str, Any]:
    assert not result.is_error, text_of(result)
    content: dict[str, Any] | None = result.structured_content
    assert content is not None
    return content


def everything(result: CallToolResult) -> str:
    """All text a tool result carries to the agent (for leak checks)."""
    return text_of(result) + json.dumps(result.structured_content or {})
