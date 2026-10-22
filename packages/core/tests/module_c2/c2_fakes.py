"""In-memory fakes of OTHER modules' ports (A: JobQueue, B: DocumentIndex/Chunker/Embedder,
C1: SourceConnector) plus the sync store, for the C2 handler tests.

The fakes model transactions faithfully enough to prove the guarantees: every write is STAGED
on the FakeTx and applied only when the `async with tx` block exits cleanly, so a raised
LeaseLost (or any error) rolls back everything written in that transaction. `World.open_tx`
counts open transactions so the fake embedder/connector can assert none is open.
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

from ragforge_core.adapters.crypto.envelope import EnvelopeSecretBox
from ragforge_core.domain.errors import LeaseLost, NotFound
from ragforge_core.domain.ids import document_id
from ragforge_core.domain.models import (
    Chunk,
    DocumentRecord,
    JobLease,
    JobSpec,
    NormalizedBlock,
    SealedSecret,
    SourceDocument,
    SourceItemRef,
    SourceKind,
    UpsertOutcome,
)
from ragforge_core.domain.ports import SourceConnector, Tx
from ragforge_core.services.sync.deps import SyncDeps
from ragforge_core.services.sync.handlers import associated_data
from ragforge_core.services.sync.policy import SyncPolicy

TOKEN = b"secret_notion_token_value"


def ts(minute: int, second: int = 0) -> datetime:
    return datetime(2026, 10, 16, 12, minute, second, tzinfo=UTC)


# ================================================================= transactions
@dataclass
class FakeTx:
    tenant_id: UUID | None
    staged: list[Callable[[], None]] = field(default_factory=list)

    def stage(self, fn: Callable[[], None]) -> None:
        self.staged.append(fn)


def as_fake(tx: Tx) -> FakeTx:
    assert isinstance(tx, FakeTx)
    return tx


@dataclass
class ConnectorRec:
    id: UUID
    kind: str
    target_collection_id: UUID
    config: Mapping[str, Any]
    status: str
    sealed: SealedSecret | None
    last_error: str | None = None


@dataclass(frozen=True)
class UploadRec:
    id: UUID
    collection_id: UUID
    blob_key: str
    filename: str
    sha256: bytes
    size_bytes: int
    created_at: datetime


@dataclass(frozen=True)
class KnownDoc:
    id: UUID
    external_id: str
    fetched_at: datetime | None
    synced_source_ids: Sequence[str]


@dataclass
class World:
    open_tx: int = 0
    commits: int = 0
    # store
    connectors: dict[tuple[UUID, UUID], ConnectorRec] = field(default_factory=dict)
    cursors: dict[tuple[UUID, UUID, str], datetime] = field(default_factory=dict)
    cursor_requests: list[datetime] = field(default_factory=list)
    exclusions: set[tuple[UUID, str, str]] = field(default_factory=set)
    uploads: dict[tuple[UUID, UUID], UploadRec] = field(default_factory=dict)
    audit: list[tuple[UUID, str, str | None, dict[str, Any]]] = field(default_factory=list)
    locks: list[tuple[str, str]] = field(default_factory=list)
    # index
    spaces: list[int] = field(default_factory=lambda: [1])
    documents: dict[UUID, DocumentRecord] = field(default_factory=dict)
    doc_tenant: dict[UUID, UUID] = field(default_factory=dict)
    tombstoned: set[UUID] = field(default_factory=set)
    chunks: dict[UUID, list[Chunk]] = field(default_factory=dict)
    replace_calls: int = 0
    embeddings: set[tuple[UUID, int, bytes]] = field(default_factory=set)
    touches: set[tuple[UUID, int, bytes]] = field(default_factory=set)
    # queue
    lease_valid: bool = True
    heartbeat_ok: bool = True
    enqueued: list[tuple[UUID | None, JobSpec]] = field(default_factory=list)
    completed: list[UUID] = field(default_factory=list)
    progress: list[dict[str, object]] = field(default_factory=list)
    failed: list[tuple[UUID, str, float | None, bool]] = field(default_factory=list)


class FakeTxManager:
    def __init__(self, world: World) -> None:
        self.w = world

    @asynccontextmanager
    async def _tx(self, tenant: UUID | None) -> AsyncIterator[FakeTx]:
        tx = FakeTx(tenant)
        self.w.open_tx += 1
        try:
            yield tx
        finally:
            self.w.open_tx -= 1
        for fn in tx.staged:  # only reached when the block exited without an exception
            fn()
        self.w.commits += 1

    def tenant(self, tenant_id: UUID) -> AbstractAsyncContextManager[Tx]:
        return self._tx(tenant_id)

    def system(self) -> AbstractAsyncContextManager[Tx]:
        return self._tx(None)


# ================================================================= A: job queue
class FakeQueue:
    def __init__(self, world: World) -> None:
        self.w = world
        self.on_heartbeat: Callable[[], None] | None = None

    async def enqueue(self, tx: Tx, spec: JobSpec) -> UUID:
        t = as_fake(tx)
        t.stage(lambda: self.w.enqueued.append((t.tenant_id, spec)))
        return uuid4()

    async def claim(self, worker_id: str, n: int) -> Sequence[JobLease]:
        return []

    async def assert_lease(self, tx: Tx, lease: JobLease) -> None:
        as_fake(tx)
        if not self.w.lease_valid:
            raise LeaseLost("fake lease lost")

    async def report_progress(self, tx: Tx, lease: JobLease, progress: dict[str, object]) -> None:
        as_fake(tx).stage(lambda: self.w.progress.append(dict(progress)))

    async def complete(self, tx: Tx, lease: JobLease) -> None:
        as_fake(tx).stage(lambda: self.w.completed.append(lease.job_id))

    async def heartbeat(self, lease: JobLease) -> bool:
        if self.on_heartbeat:
            self.on_heartbeat()
        return self.w.heartbeat_ok and self.w.lease_valid

    async def release(self, lease: JobLease) -> None:
        return None

    async def fail(
        self, lease: JobLease, error: str, retry_after_s: float | None, *, permanent: bool = False
    ) -> None:
        self.w.failed.append((lease.job_id, error, retry_after_s, permanent))


# ================================================================= B: index, chunker, embedder
class FakeIndex:
    def __init__(self, world: World) -> None:
        self.w = world
        self.forced_outcome: UpsertOutcome | None = None

    async def target_spaces(self, tx: Tx, collection_id: UUID) -> Sequence[int]:
        return list(self.w.spaces)

    async def touch_and_find_missing(
        self, tx: Tx, collection_id: UUID, space_id: int, embed_sha256s: Sequence[bytes]
    ) -> set[bytes]:
        keys = {(collection_id, space_id, h) for h in embed_sha256s}
        as_fake(tx).stage(lambda: self.w.touches.update(keys))
        return {h for h in embed_sha256s if (collection_id, space_id, h) not in self.w.embeddings}

    async def insert_embeddings(
        self,
        tx: Tx,
        collection_id: UUID,
        space_id: int,
        vectors: Mapping[bytes, Sequence[float]],
    ) -> int:
        keys = {(collection_id, space_id, h) for h in vectors}
        as_fake(tx).stage(lambda: self.w.embeddings.update(keys))
        return len(keys)

    async def upsert_document(self, tx: Tx, record: DocumentRecord) -> UpsertOutcome:
        t = as_fake(tx)
        old = self.w.documents.get(record.id)
        outcome = self.forced_outcome or _outcome(old, record)
        if outcome is UpsertOutcome.stale:
            return outcome
        new = record if outcome is not UpsertOutcome.unchanged or old is None else old
        new = replace(new, fetched_at=record.fetched_at)

        def apply() -> None:
            self.w.documents[record.id] = new
            assert t.tenant_id is not None
            self.w.doc_tenant[record.id] = t.tenant_id
            self.w.tombstoned.discard(record.id)

        t.stage(apply)
        return outcome

    async def replace_chunks(
        self, tx: Tx, document_id: UUID, collection_id: UUID, chunks: Sequence[Chunk]
    ) -> None:
        def apply() -> None:
            self.w.chunks[document_id] = list(chunks)
            self.w.replace_calls += 1

        as_fake(tx).stage(apply)

    async def tombstone(self, tx: Tx, document_id: UUID) -> None:
        as_fake(tx).stage(lambda: self.w.tombstoned.add(document_id))

    async def is_excluded(self, tx: Tx, source_kind: str, external_id: str) -> bool:
        raise AssertionError("C2 uses its own batched exclusion lookup")


def _outcome(old: DocumentRecord | None, new: DocumentRecord) -> UpsertOutcome:
    if old is None:
        return UpsertOutcome.inserted
    if old.fetched_at >= new.fetched_at:
        return UpsertOutcome.stale
    if old.page_sha256 == new.page_sha256:
        return UpsertOutcome.unchanged
    return UpsertOutcome.updated


class FakeChunker:
    def chunk(self, doc: SourceDocument) -> list[Chunk]:
        out = []
        for i, block in enumerate(doc.blocks):
            embedded = f"{doc.title} › \n\n{block.text}"
            out.append(
                Chunk(
                    anchor=block.anchor,
                    ordinal=i,
                    heading_path=(),
                    display_text=block.text,
                    embedded_text=embedded,
                    display_sha256=hashlib.sha256(block.text.encode()).digest(),
                    embed_sha256=hashlib.sha256(embedded.encode()).digest(),
                    token_count=len(block.text.split()),
                )
            )
        return out


class FakeEmbedder:
    """Fails loudly if called while ANY transaction is open (the core no-model-in-tx rule)."""

    def __init__(self, world: World, dim: int = 4) -> None:
        self.w = world
        self.model = "fake"
        self.dim = dim
        self.calls: list[list[str]] = []
        self.on_embed: Callable[[], None] | None = None

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        assert self.w.open_tx == 0, "embedder called while a transaction is open"
        self.calls.append(list(texts))
        if self.on_embed:
            self.on_embed()
        return [[0.5] * self.dim for _ in texts]

    async def embed_query(self, text: str) -> list[float]:
        return [0.5] * self.dim


class FakeRegistry:
    def __init__(self, world: World) -> None:
        self.by_space: dict[int, FakeEmbedder] = {}
        self.w = world

    async def ensure(self, space_id: int) -> None:
        return None

    def for_space(self, space_id: int) -> FakeEmbedder:
        return self.by_space.setdefault(space_id, FakeEmbedder(self.w))


# ================================================================= C1: source connector
@dataclass
class FakePage:
    last_edited: datetime | None
    blocks: tuple[str, ...] = ("hello world",)
    title: str = "Page"
    fetched_at: datetime | None = None
    synced_source_ids: tuple[str, ...] = ()


class FakeSource:
    def __init__(self, world: World, pages: dict[str, FakePage]) -> None:
        self.w = world
        self.pages = pages
        self.since_seen: list[datetime | None] = []
        self.fetched: list[str] = []
        self.confirm: dict[str, bool | Exception] = {}
        self.confirm_calls: list[str] = []
        self.fail_listing_after: int | None = None

    def _no_tx(self) -> None:
        assert self.w.open_tx == 0, "network call while a transaction is open"

    def _ref(self, ext: str) -> SourceItemRef:
        return SourceItemRef(SourceKind.notion, ext, self.pages[ext].last_edited)

    async def list_changed(self, since: datetime | None) -> AsyncIterator[SourceItemRef]:
        self.since_seen.append(since)
        for n, ext in enumerate(sorted(self.pages)):
            if self.fail_listing_after is not None and n >= self.fail_listing_after:
                raise ConnectionError("listing broke")
            last = self.pages[ext].last_edited
            if since is None or last is None or last >= since:
                yield self._ref(ext)

    async def list_all(self) -> AsyncIterator[SourceItemRef]:
        for ext in sorted(self.pages):
            yield self._ref(ext)

    async def fetch(self, ref: SourceItemRef) -> SourceDocument:
        self._no_tx()
        self.fetched.append(ref.external_id)
        page = self.pages[ref.external_id]
        return SourceDocument(
            ref=self._ref(ref.external_id),
            title=page.title,
            external_url=f"https://notion.so/{ref.external_id}",
            blocks=tuple(
                NormalizedBlock(anchor=f"a{i}", kind="paragraph", text=t)
                for i, t in enumerate(page.blocks)
            ),
            synced_source_ids=page.synced_source_ids,
            fetched_at=page.fetched_at or ts(30),
        )

    async def confirm_deleted(self, external_id: str) -> bool:
        self._no_tx()
        self.confirm_calls.append(external_id)
        answer = self.confirm.get(external_id, False)
        if isinstance(answer, Exception):
            raise answer
        return answer


class FakeConnectorFactory:
    def __init__(self, source: FakeSource) -> None:
        self.source = source
        self.tokens: list[bytes] = []

    def open(
        self, *, connector_id: UUID, kind: str, token: bytes, config: Mapping[str, Any]
    ) -> AbstractAsyncContextManager[SourceConnector]:
        self.tokens.append(token)

        @asynccontextmanager
        async def _cm() -> AsyncIterator[SourceConnector]:
            yield self.source

        return _cm()


# ================================================================= sync store
class FakeStore:
    def __init__(self, world: World) -> None:
        self.w = world

    @staticmethod
    def _t(tx: Tx) -> UUID:
        tenant = as_fake(tx).tenant_id
        assert tenant is not None, "store used outside a tenant transaction"
        return tenant

    async def load_connector(self, tx: Tx, connector_id: UUID) -> ConnectorRec | None:
        return self.w.connectors.get((self._t(tx), connector_id))

    async def mark_connector_synced(self, tx: Tx, connector_id: UUID, next_poll_in_s: int) -> None:
        rec = self.w.connectors.get((self._t(tx), connector_id))
        if rec is not None:
            as_fake(tx).stage(lambda: setattr(rec, "last_error", None))

    async def mark_connector_error(self, tx: Tx, connector_id: UUID, message: str) -> None:
        rec = self.w.connectors.get((self._t(tx), connector_id))

        def apply() -> None:
            if rec is not None and rec.status == "active":
                rec.status, rec.last_error = "error", message

        as_fake(tx).stage(apply)

    async def destroy_secret(self, tx: Tx, connector_id: UUID) -> None:
        rec = self.w.connectors.get((self._t(tx), connector_id))

        def apply() -> None:
            if rec is not None:
                rec.sealed, rec.status = None, "deleting"

        as_fake(tx).stage(apply)

    async def delete_connector_documents(self, tx: Tx, connector_id: UUID, limit: int) -> int:
        tenant = self._t(tx)
        ids = [
            d.id
            for d in self.w.documents.values()
            if d.connector_id == connector_id and self.w.doc_tenant.get(d.id) == tenant
        ][:limit]

        def apply() -> None:
            for i in ids:
                self.w.documents.pop(i, None)

        as_fake(tx).stage(apply)
        return len(ids)

    async def delete_connector(self, tx: Tx, connector_id: UUID) -> bool:
        key = (self._t(tx), connector_id)
        existed = key in self.w.connectors

        def apply() -> None:
            self.w.connectors.pop(key, None)

        as_fake(tx).stage(apply)
        return existed

    async def read_cursor(self, tx: Tx, connector_id: UUID, scope: str) -> datetime | None:
        return self.w.cursors.get((self._t(tx), connector_id, scope))

    async def advance_cursor(
        self, tx: Tx, connector_id: UUID, scope: str, value: datetime
    ) -> datetime:
        key = (self._t(tx), connector_id, scope)
        self.w.cursor_requests.append(value)
        current = self.w.cursors.get(key)
        stored = value if current is None else max(current, value)
        as_fake(tx).stage(lambda: self.w.cursors.__setitem__(key, stored))
        return stored

    async def excluded_ids(
        self, tx: Tx, source_kind: SourceKind, external_ids: Sequence[str]
    ) -> set[str]:
        tenant = self._t(tx)
        return {e for e in external_ids if (tenant, source_kind.value, e) in self.w.exclusions}

    async def lock_source_item(self, tx: Tx, source_kind: SourceKind, external_id: str) -> None:
        self.w.locks.append((source_kind.value, external_id))

    async def add_exclusion(
        self, tx: Tx, source_kind: SourceKind, external_id: str, reason: str | None
    ) -> None:
        key = (self._t(tx), source_kind.value, external_id)
        as_fake(tx).stage(lambda: self.w.exclusions.add(key))

    def _live(self, tenant: UUID, connector_id: UUID) -> list[DocumentRecord]:
        return [
            d
            for d in self.w.documents.values()
            if d.connector_id == connector_id
            and self.w.doc_tenant.get(d.id) == tenant
            and d.id not in self.w.tombstoned
        ]

    async def fetched_at_by_external_id(
        self, tx: Tx, connector_id: UUID, external_ids: Sequence[str]
    ) -> dict[str, datetime | None]:
        wanted = set(external_ids)
        return {
            d.external_id: d.fetched_at
            for d in self._live(self._t(tx), connector_id)
            if d.external_id in wanted
        }

    async def live_documents(self, tx: Tx, connector_id: UUID) -> list[KnownDoc]:
        return [
            KnownDoc(d.id, d.external_id, d.fetched_at, d.synced_source_ids)
            for d in self._live(self._t(tx), connector_id)
        ]

    async def insert_upload(
        self,
        tx: Tx,
        *,
        upload_id: UUID,
        collection_id: UUID,
        blob_key: str,
        filename: str,
        sha256: bytes,
        size_bytes: int,
    ) -> None:
        rec = UploadRec(
            upload_id, collection_id, blob_key, filename, sha256, size_bytes, datetime.now(UTC)
        )
        key = (self._t(tx), upload_id)
        as_fake(tx).stage(lambda: self.w.uploads.__setitem__(key, rec))

    async def load_upload(self, tx: Tx, upload_id: UUID) -> UploadRec | None:
        return self.w.uploads.get((self._t(tx), upload_id))

    async def audit(
        self, tx: Tx, *, actor: str, action: str, target: str | None, details: Mapping[str, Any]
    ) -> None:
        entry = (self._t(tx), action, target, dict(details))
        as_fake(tx).stage(lambda: self.w.audit.append(entry))


# ================================================================= blobs + parser
class FakeBlobs:
    def __init__(self) -> None:
        self.data: dict[str, bytes] = {}
        self.deleted: list[str] = []

    async def put(self, key: str, data: bytes) -> None:
        self.data[key] = data

    async def get(self, key: str) -> bytes:
        if key not in self.data:
            raise NotFound("blob")
        return self.data[key]

    async def delete(self, key: str) -> None:
        self.deleted.append(key)
        self.data.pop(key, None)


class FakeParser:
    async def parse(self, filename: str, data: bytes) -> SourceDocument:
        paras = [p for p in data.decode().split("\n\n") if p.strip()]
        return SourceDocument(
            ref=SourceItemRef(SourceKind.file, filename, None),
            title=filename,
            external_url=None,
            blocks=tuple(
                NormalizedBlock(anchor=f"b{i}", kind="paragraph", text=p)
                for i, p in enumerate(paras)
            ),
        )


# ================================================================= harness
@dataclass
class Harness:
    world: World
    deps: SyncDeps
    source: FakeSource
    factory: FakeConnectorFactory
    queue: FakeQueue
    index: FakeIndex
    registry: FakeRegistry
    blobs: FakeBlobs
    box: EnvelopeSecretBox
    tenant: UUID
    collection: UUID
    connector: UUID

    def lease(self, kind: str, payload: dict[str, Any], tenant: UUID | None = None) -> JobLease:
        return JobLease(
            job_id=uuid4(),
            tenant_id=tenant or self.tenant,
            kind=kind,
            payload=payload,
            lease_version=1,
            attempts=1,
        )

    def add_connector(
        self, *, tenant: UUID | None = None, status: str = "active", sealed: bool = True
    ) -> UUID:
        tenant = tenant or self.tenant
        cid = uuid4()
        secret = (
            self.box.seal(TOKEN, associated_data=associated_data(tenant, cid)) if sealed else None
        )
        self.world.connectors[(tenant, cid)] = ConnectorRec(
            cid, "notion", self.collection, {}, status, secret
        )
        return cid

    def seed_document(
        self, external_id: str, fetched_at: datetime, synced: tuple[str, ...] = ()
    ) -> DocumentRecord:
        rec = DocumentRecord(
            id=document_id(self.tenant, self.collection, self.connector, external_id),
            collection_id=self.collection,
            connector_id=self.connector,
            upload_id=None,
            source_kind=SourceKind.notion,
            external_id=external_id,
            external_url=None,
            title="seeded",
            external_updated_at=None,
            fetched_at=fetched_at,
            page_sha256=b"\x00" * 32,
            synced_source_ids=synced,
        )
        self.world.documents[rec.id] = rec
        self.world.doc_tenant[rec.id] = self.tenant
        return rec


def make_harness(pages: dict[str, FakePage] | None = None, **policy: Any) -> Harness:
    world = World()
    box = EnvelopeSecretBox(os.urandom(32))
    source = FakeSource(world, pages or {})
    factory = FakeConnectorFactory(source)
    queue, index, registry, blobs = (
        FakeQueue(world),
        FakeIndex(world),
        FakeRegistry(world),
        FakeBlobs(),
    )
    deps = SyncDeps(
        tx=FakeTxManager(world),
        queue=queue,
        store=FakeStore(world),
        index=index,
        chunker=FakeChunker(),
        embedders=registry,
        secrets=box,
        connectors=factory,
        blobs=blobs,
        parser=FakeParser(),
        policy=SyncPolicy(**policy),
    )
    h = Harness(
        world, deps, source, factory, queue, index, registry, blobs, box,
        tenant=uuid4(), collection=uuid4(), connector=uuid4(),
    )  # fmt: skip
    h.connector = h.add_connector()
    return h
