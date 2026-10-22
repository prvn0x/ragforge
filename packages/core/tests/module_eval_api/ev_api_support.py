"""Module EV-API test support (not a test module): D0 loaders, a scripted verifier and the
database builders for file and Notion pages.

Everything here is synthetic or the committed D0 demo set (evals/demo/): never private data.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from typing import Any
from uuid import NAMESPACE_URL, UUID, uuid4, uuid5

import psycopg

from ragforge_client.evals.schema import VerifierItem, VerifierSet
from ragforge_contracts.api import ChunkKey
from ragforge_core.domain.models import ChatMessage, ChatTurnUsage, RankedChunk

Conn = psycopg.Connection[tuple[object, ...]]
Reply = dict[str, object]

DEMO = Path(__file__).resolve().parents[4] / "evals" / "demo"
D0_COLLECTION = UUID(int=7000)


# ----------------------------------------------------------------- D0
def _key_id(key: dict[str, Any]) -> str:
    return json.dumps(key, sort_keys=True)


@cache
def _d0_chunks() -> dict[str, dict[str, Any]]:
    chunks: list[dict[str, Any]] = json.loads((DEMO / "chunks.json").read_text("utf-8"))
    return {_key_id(c["key"]): c for c in chunks}


@cache
def d0_verifier_set() -> VerifierSet:
    return VerifierSet.model_validate_json((DEMO / "verifier.json").read_text("utf-8"))


def d0_item(item_id: str) -> VerifierItem:
    return next(i for i in d0_verifier_set().items if i.id == item_id)


def d0_note(key: ChunkKey) -> RankedChunk:
    """The D0 chunk with this key as the server would hold it (ids derived from the key, so
    they are stable across runs, like the snapshot's)."""
    chunk = _d0_chunks()[_key_id(key.model_dump(mode="json"))]
    text: str = chunk["text"]
    page = key.page.key
    return RankedChunk(
        chunk_id=uuid5(NAMESPACE_URL, _key_id(chunk["key"])),
        document_id=uuid5(NAMESPACE_URL, f"file:{page}"),
        collection_id=D0_COLLECTION,
        anchor=f"{page}#{len(key.heading_path)}",
        title=page.removesuffix(".md"),
        heading_path=tuple(key.heading_path),
        display_text=text,
        display_sha256=hashlib.sha256(text.encode()).digest(),
        external_url=None,
        fused_score=0.0,
    )


def d0_notes(item: VerifierItem) -> list[RankedChunk]:
    return [d0_note(k) for k in item.shown]


def mutated_text(item: VerifierItem) -> str:
    assert item.span is not None
    return item.answer[item.span.start : item.span.end]


# ----------------------------------------------------------------- the verifier
def supported() -> Reply:
    return {"verdict": "supported", "unsupported_claims": []}


def objects(*claims: str, verdict: str = "partially_supported") -> Reply:
    return {"verdict": verdict, "unsupported_claims": list(claims)}


@dataclass
class ScriptedVerifier:
    """A ChatModel whose JSON replies are scripted in order (the last one repeats). Only the
    ground check calls it here; `stream` is never used. `usage_model` is the model the provider
    REPORTS, which may differ from `model` (an alias that moved): verify must report it."""

    replies: list[Reply | Exception | Callable[[str], Reply]] = field(
        default_factory=lambda: [supported()]
    )
    provider: str = "fake"
    model: str = "fake-verifier"
    usage_model: str = "fake-verifier-2026"
    calls: list[list[ChatMessage]] = field(default_factory=list)

    def stream(self, *args: object, **kwargs: object) -> Any:
        raise AssertionError("the verifier never streams")

    async def complete_json(
        self, system: str, messages: Sequence[ChatMessage], schema: dict[str, object]
    ) -> tuple[dict[str, object], ChatTurnUsage]:
        self.calls.append(list(messages))
        reply = self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]
        if isinstance(reply, Exception):
            raise reply
        payload = reply(messages[-1].content) if callable(reply) else reply
        return payload, ChatTurnUsage(self.provider, self.usage_model, 7, 3)


# ----------------------------------------------------------------- database builders
@dataclass
class DbPage:
    document: UUID
    chunks: list[UUID]


def _chunks(
    superuser: Conn,
    tenant: UUID,
    collection: UUID,
    document: UUID,
    sections: Sequence[tuple[Sequence[str], str]],
) -> list[UUID]:
    ids: list[UUID] = []
    for ordinal, (headings, text) in enumerate(sections):
        chunk = uuid4()
        display = hashlib.sha256(text.encode()).digest()  # equal text -> equal digest
        embed = hashlib.sha256(f"{chunk}".encode()).digest()
        superuser.execute(
            "INSERT INTO rf.chunks (id, tenant_id, collection_id, document_id, anchor, ordinal, "
            "heading_path, display_text, display_sha256, embed_sha256, token_count) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 10)",
            (chunk, tenant, collection, document, f"b-{ordinal}", ordinal, list(headings), text,
             display, embed),
        )  # fmt: skip
        ids.append(chunk)
    return ids


def file_page(
    superuser: Conn,
    tenant: UUID,
    collection: UUID,
    filename: str,
    sections: Sequence[tuple[Sequence[str], str]],
    *,
    deleted: bool = False,
    with_filename: bool = True,
) -> DbPage:
    """An uploaded file, stored as the upload handler stores it: external_id = the upload id,
    metadata.filename = the file name (services/sync/handlers.py)."""
    upload, document = uuid4(), uuid4()
    superuser.execute(
        "INSERT INTO rf.uploads (id, tenant_id, collection_id, blob_key, filename, sha256, "
        "size_bytes) VALUES (%s, %s, %s, %s, %s, %s, 1)",
        (upload, tenant, collection, f"blob-{upload.hex}", filename, b"\x00" * 32),
    )
    metadata = {"mime": "text/markdown", **({"filename": filename} if with_filename else {})}
    superuser.execute(
        "INSERT INTO rf.documents (id, tenant_id, collection_id, upload_id, source_kind, "
        "external_id, title, metadata, deleted_at) VALUES (%s, %s, %s, %s, 'file', %s, %s, "
        "%s, CASE WHEN %s THEN now() END)",
        (document, tenant, collection, upload, str(upload), filename, json.dumps(metadata),
         deleted),
    )  # fmt: skip
    return DbPage(document, _chunks(superuser, tenant, collection, document, sections))


def notion_page(
    superuser: Conn,
    tenant: UUID,
    collection: UUID,
    connector: UUID,
    page_id: str,
    sections: Sequence[tuple[Sequence[str], str]],
) -> DbPage:
    document = uuid4()
    superuser.execute(
        "INSERT INTO rf.documents (id, tenant_id, collection_id, connector_id, source_kind, "
        "external_id, title) VALUES (%s, %s, %s, %s, 'notion', %s, 'Page')",
        (document, tenant, collection, connector, page_id),
    )
    return DbPage(document, _chunks(superuser, tenant, collection, document, sections))


def sha_hex(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()
