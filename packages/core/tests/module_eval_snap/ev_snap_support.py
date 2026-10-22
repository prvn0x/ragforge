"""EV-SNAP test support: the D0 demo pages chunked by the SERVER's own parser and chunker, a
source tenant seeded with them (as if a Notion connector had synced them), deterministic fake
embedders, and a helper that runs one queued job through its real handler.

Imported as `module_eval_snap.ev_snap_support` (packages/core/tests is on sys.path and on
mypy_path). Only evals/demo/** and synthetic data: never the private set.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import struct
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from uuid import UUID, uuid4

import psycopg

from ragforge_core.adapters.connectors.files.parser import SandboxedDocumentParser
from ragforge_core.adapters.db.jobs import PgJobQueue
from ragforge_core.bootstrap import _chunker  # the exact chunker the server builds
from ragforge_core.domain import ids
from ragforge_core.domain.models import Chunk, JobLease
from ragforge_core.services.auth.keys import generate_key

Conn = psycopg.Connection[tuple[object, ...]]
DEMO_PAGES = Path(__file__).resolve().parents[4] / "evals" / "demo" / "pages"
SOURCE_SPACE = 1  # the base space from migration 0001 (nomic-embed-text, 768 dims)
SOURCE_DIM = 768


@dataclass(frozen=True)
class DemoPage:
    filename: str
    title: str
    chunks: tuple[Chunk, ...]

    @property
    def page_id(self) -> str:
        """A stable fake Notion page id (the snapshot's page_key for notion pages)."""
        return str(UUID(hashlib.md5(self.filename.encode()).hexdigest()))  # noqa: S324


def load_demo_pages() -> list[DemoPage]:
    """The D0 pages through the server's parser and chunker (the ingest path's chunks)."""

    async def run() -> list[DemoPage]:
        parser, chunker = SandboxedDocumentParser(), _chunker()
        pages = []
        for path in sorted(DEMO_PAGES.glob("*.md")):
            doc = await parser.parse(path.name, path.read_bytes())
            chunks = tuple(sorted(chunker.chunk(doc), key=lambda c: c.ordinal))
            pages.append(DemoPage(path.name, doc.title, chunks))
        return pages

    return asyncio.run(run())


def vector(seed: bytes, dim: int) -> list[float]:
    """Deterministic pseudo-embedding of `seed` (no model)."""
    out: list[float] = []
    counter = 0
    while len(out) < dim:
        block = hashlib.sha256(seed + counter.to_bytes(4, "big")).digest()
        out.extend((v / 2**32) - 0.5 for v in struct.unpack(">8I", block))
        counter += 1
    return out[:dim]


def vector_literal(values: Sequence[float]) -> str:
    return "[" + ",".join(repr(float(v)) for v in values) + "]"


@dataclass
class Source:
    """A personal tenant whose collection holds the D0 pages as Notion-synced documents."""

    tenant: UUID
    user: UUID
    collection: UUID
    connector: UUID
    admin_key: str
    pages: list[DemoPage]
    document_ids: dict[str, UUID] = field(default_factory=dict)  # page_id -> document id


def issue_key(superuser: Conn, tenant: UUID, user: UUID, scopes: Sequence[str]) -> str:
    key = generate_key()
    superuser.execute(
        "INSERT INTO rf.api_keys (tenant_id, user_id, name, prefix, key_sha256, scopes) "
        "VALUES (%s, %s, 'ui:test', %s, %s, %s)",
        (tenant, user, key.prefix, key.sha256, list(scopes)),
    )
    return key.plaintext


def seed_source(
    superuser: Conn,
    tenant: UUID,
    user: UUID,
    collection: UUID,
    pages: Sequence[DemoPage],
    *,
    admin_scopes: Sequence[str] = ("read", "write", "chat", "approvals:write", "admin"),
) -> Source:
    connector = uuid4()
    ex = superuser.execute
    ex(
        "INSERT INTO rf.connectors (id, tenant_id, kind, target_collection_id, display_name) "
        "VALUES (%s, %s, 'notion', %s, 'notion')",
        (connector, tenant, collection),
    )
    source = Source(
        tenant=tenant,
        user=user,
        collection=collection,
        connector=connector,
        admin_key=issue_key(superuser, tenant, user, admin_scopes),
        pages=list(pages),
    )
    hashes: set[bytes] = set()
    for i, page in enumerate(pages):
        doc = ids.document_id(tenant, collection, connector, page.page_id)
        source.document_ids[page.page_id] = doc
        ex(
            "INSERT INTO rf.documents (id, tenant_id, collection_id, connector_id, source_kind, "
            "external_id, external_url, title, fetched_at, page_sha256, metadata) "
            "VALUES (%s, %s, %s, %s, 'notion', %s, %s, %s, now(), %s, %s)",
            (
                doc, tenant, collection, connector, page.page_id,
                f"https://notion.so/{page.page_id.replace('-', '')}", page.title,
                hashlib.sha256(page.filename.encode()).digest(),
                json.dumps({"filename": page.filename}),
            ),
        )  # fmt: skip
        for chunk in page.chunks:
            ex(
                "INSERT INTO rf.chunks (id, tenant_id, collection_id, document_id, anchor, "
                "ordinal, heading_path, display_text, display_sha256, embed_sha256, "
                "token_count, metadata) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                (
                    ids.chunk_id(doc, chunk.ordinal), tenant, collection, doc, chunk.anchor,
                    chunk.ordinal, list(chunk.heading_path), chunk.display_text,
                    chunk.display_sha256, chunk.embed_sha256, chunk.token_count,
                    json.dumps(chunk.metadata),
                ),
            )  # fmt: skip
            hashes.add(chunk.embed_sha256)
        if i + 1 < len(pages):  # a link to the next page, so document_links are exercised
            ex(
                "INSERT INTO rf.document_links (tenant_id, from_document_id, to_external_id) "
                "VALUES (%s, %s, %s)",
                (tenant, doc, pages[i + 1].page_id),
            )
    for sha in sorted(hashes):
        add_vector(superuser, tenant, collection, SOURCE_SPACE, sha, SOURCE_DIM)
    return source


def add_vector(
    superuser: Conn, tenant: UUID, collection: UUID, space: int, sha: bytes, dim: int
) -> None:
    superuser.execute(
        "INSERT INTO rf.embeddings (tenant_id, collection_id, space_id, embed_sha256, embedding) "
        "VALUES (%s, %s, %s, %s, %s::ext.vector)",
        (tenant, collection, space, sha, vector_literal(vector(sha, dim))),
    )
    superuser.execute(
        "INSERT INTO rf.embedding_touches (tenant_id, collection_id, space_id, embed_sha256) "
        "VALUES (%s, %s, %s, %s)",
        (tenant, collection, space, sha),
    )


def make_eval_tenant(superuser: Conn, source_tenant: UUID) -> tuple[UUID, UUID]:
    """An eval tenant paired with `source_tenant`, created as the superuser (fixtures only)."""
    tenant, user = uuid4(), uuid4()
    superuser.execute(
        "INSERT INTO rf.tenants (id, name, kind, eval_of_tenant_id) VALUES (%s, %s, 'eval', %s)",
        (tenant, f"eval-{tenant.hex[:6]}", source_tenant),
    )
    superuser.execute(
        "INSERT INTO rf.users (id, tenant_id, display_name) VALUES (%s, %s, 'eval')",
        (user, tenant),
    )
    return tenant, user


# ----------------------------------------------------------------- fakes
class FakeEmbedder:
    def __init__(self, model: str, dim: int) -> None:
        self.model = model
        self.dim = dim
        self.inputs: list[str] = []

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        self.inputs.extend(texts)
        return [vector(hashlib.sha256(t.encode()).digest(), self.dim) for t in texts]

    async def embed_query(self, text: str) -> list[float]:
        return vector(text.encode(), self.dim)


class FakeEmbedders:
    """EmbedderRegistry port: one FakeEmbedder per space, dims given up front."""

    def __init__(self, dims: dict[int, int]) -> None:
        self.embedders = {space: FakeEmbedder(f"fake-{space}", dim) for space, dim in dims.items()}
        self.ensured: list[int] = []

    async def ensure(self, space_id: int) -> None:
        self.ensured.append(space_id)

    def for_space(self, space_id: int) -> FakeEmbedder:
        return self.embedders[space_id]


# ----------------------------------------------------------------- running a queued job
async def run_job(
    queue: PgJobQueue, job_id: UUID, handler: Callable[[JobLease], Awaitable[None]]
) -> JobLease:
    """Claim the queued job `job_id` exactly as a worker would (other claimed jobs are released
    untouched), then run `handler` on its lease. Returns the lease."""
    for _ in range(20):
        leases = await queue.claim("ev-snap-test", 100)
        mine = [lease for lease in leases if lease.job_id == job_id]
        for lease in leases:
            if lease.job_id != job_id:
                await queue.release(lease)
        if mine:
            await handler(mine[0])
            return mine[0]
    raise AssertionError(f"job {job_id} was never claimable")
