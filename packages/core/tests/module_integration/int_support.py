"""Shared helpers for the composed (real container) integration tests: scripted models and a
seeded document. Only models are scripted; everything else in these tests is production wiring."""

from __future__ import annotations

import dataclasses
import hashlib
from collections.abc import AsyncIterator, Sequence
from uuid import UUID, uuid4

import psycopg

from module_b.b_support import add_embedding, unit_vector
from ragforge_core.domain.models import (
    ChatMessage,
    ChatStreamEvent,
    ChatTurnUsage,
    RankedChunk,
    TextDelta,
    ToolCall,
    ToolSpec,
    UsageReport,
)

Conn = psycopg.Connection[tuple[object, ...]]
QUERY_VECTOR = unit_vector("composed-query")
SUPPORTED: dict[str, object] = {"verdict": "supported", "unsupported_claims": []}


# ---------------------------------------------------------------- scripted models
class ScriptedChat:
    """ChatModel port: each `stream` / `complete_json` call pops the next scripted item."""

    provider = "scripted"
    model = "scripted-chat"

    def __init__(self) -> None:
        self.script: list[list[ChatStreamEvent] | dict[str, object]] = []
        self.calls: list[str] = []

    async def stream(
        self, system: str, messages: Sequence[ChatMessage], tools: Sequence[ToolSpec] = ()
    ) -> AsyncIterator[ChatStreamEvent]:
        self.calls.append("stream")
        item = self.script.pop(0)
        assert isinstance(item, list), f"expected a stream, script has {item!r}"
        for event in item:
            yield event

    async def complete_json(
        self, system: str, messages: Sequence[ChatMessage], schema: dict[str, object]
    ) -> tuple[dict[str, object], ChatTurnUsage]:
        self.calls.append("json")
        item = self.script.pop(0)
        assert isinstance(item, dict), f"expected json, script has {item!r}"
        return item, ChatTurnUsage("scripted", "scripted-chat", 5, 5)


def say(text: str, *calls: ToolCall) -> list[ChatStreamEvent]:
    events: list[ChatStreamEvent] = [TextDelta(text)] if text else []
    return [*events, *calls, UsageReport(ChatTurnUsage("scripted", "scripted-chat", 10, 5))]


class FixedEmbedder:
    model = "fixed"
    dim = len(QUERY_VECTOR)

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [QUERY_VECTOR for _ in texts]

    async def embed_query(self, text: str) -> list[float]:
        return QUERY_VECTOR


class FixedEmbedders:
    async def ensure(self, space_id: int) -> None:
        return None

    def for_space(self, space_id: int) -> FixedEmbedder:
        return FixedEmbedder()


class ScoringReranker:
    async def rerank(self, query: str, candidates: Sequence[RankedChunk]) -> list[RankedChunk]:
        return [dataclasses.replace(c, rerank_score=0.9) for c in candidates]


# ---------------------------------------------------------------- seed
def sha(s: str) -> bytes:
    return hashlib.sha256(s.encode()).digest()


def seed_document(su: Conn, tenant: UUID, coll: UUID) -> UUID:
    upload, doc = uuid4(), uuid4()
    su.execute(
        "INSERT INTO rf.uploads (id, tenant_id, collection_id, blob_key, filename, sha256, "
        "size_bytes) VALUES (%s, %s, %s, %s, 'handbook.md', %s, 1)",
        (upload, tenant, coll, f"k/{upload}", sha(str(upload))),
    )
    su.execute(
        "INSERT INTO rf.documents (id, tenant_id, collection_id, upload_id, source_kind, "
        "external_id, title) VALUES (%s, %s, %s, %s, 'file', %s, 'Handbook')",
        (doc, tenant, coll, upload, str(upload)),
    )
    embed = sha(f"embed-{doc}")
    su.execute(
        "INSERT INTO rf.chunks (id, tenant_id, collection_id, document_id, anchor, ordinal, "
        "display_text, display_sha256, embed_sha256, token_count) "
        "VALUES (%s, %s, %s, %s, 'a0', 0, %s, %s, %s, 12)",
        (uuid4(), tenant, coll, doc, "On-call rotations last one week.", sha(f"d{doc}"), embed),
    )
    add_embedding(su, tenant, coll, embed, QUERY_VECTOR)
    return doc
