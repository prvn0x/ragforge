"""Pure EV-SNAP rules (no database): the digest reference, the embed-input mirror, payloads,
refusals, and the job handlers' failure semantics with fake ports."""

from __future__ import annotations

import hashlib
import random
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from dataclasses import replace
from importlib.resources import files
from typing import Any, cast
from uuid import UUID, uuid4

import pytest
from hypothesis import given
from hypothesis import strategies as st

from module_eval_snap.ev_snap_support import DemoPage, FakeEmbedder, FakeEmbedders
from ragforge_core.apps.worker.main import register_eval_snapshot_handlers
from ragforge_core.domain import ids
from ragforge_core.domain.errors import Conflict, Forbidden, LeaseLost, NotFound, ValidationFailed
from ragforge_core.domain.models import JobLease, JobPriority
from ragforge_core.domain.ports import DocumentIndex, JobQueue, TransactionManager, Tx
from ragforge_core.rag.chunker import HEADER_SEPARATOR as CHUNKER_SEPARATOR
from ragforge_core.services.auth.scopes import ClientKind, client_kind_for_name
from ragforge_core.services.eval_snapshot import (
    EVAL_KEY_NAME,
    EVAL_KEY_SCOPES,
    KIND_SNAPSHOT_COPY,
    KIND_SNAPSHOT_REEMBED,
    MAX_RUNTIME_S,
    CopyOutcome,
    EmbedSource,
    EvalSnapshotDeps,
    EvalTenantExists,
    ReembedOutcome,
    SnapshotNotReady,
    copy_job,
    reembed_job,
    refusal_for,
    run_copy_job,
    run_reembed_job,
)
from ragforge_core.services.eval_snapshot.digests import (
    SnapshotChunk,
    SnapshotPage,
    content_digest,
    index_digest,
    netstring,
    occurrences,
    page_key,
)
from ragforge_core.services.eval_snapshot.embed_input import (
    HEADER_SEPARATOR,
    EmbedInputMismatch,
    checked_embed_input,
    embedded_text,
)
from ragforge_core.services.eval_snapshot.jobs import EmbeddingContractError
from ragforge_core.services.jobs import JobHandlers, PermanentJobError
from ragforge_core.services.sync.policy import SYNC_KINDS

# ----------------------------------------------------------------- digests (reference)
texts = st.text(min_size=0, max_size=12)
chunks = st.lists(st.tuples(st.lists(texts, max_size=2), texts), min_size=0, max_size=4).map(
    lambda raw: tuple(_chunk(i, tuple(h), t) for i, (h, t) in enumerate(raw))
)
pages = st.lists(
    st.tuples(st.sampled_from(["notion", "file"]), st.text(min_size=1, max_size=8), chunks),
    max_size=4,
).map(lambda raw: [SnapshotPage(k, key, c) for k, key, c in raw])


def _chunk(ordinal: int, heading: tuple[str, ...], text: str) -> SnapshotChunk:
    display = hashlib.sha256(text.encode()).digest()
    embed = hashlib.sha256(("h" + "/".join(heading) + text).encode()).digest()
    return SnapshotChunk(ordinal, heading, text, display, embed)


@given(pages, st.randoms())
def test_both_digests_ignore_page_and_row_order(
    corpus: list[SnapshotPage], rnd: random.Random
) -> None:
    shuffled = [replace(p, chunks=tuple(rnd.sample(p.chunks, len(p.chunks)))) for p in corpus]
    rnd.shuffle(shuffled)
    assert content_digest(shuffled) == content_digest(corpus)
    assert index_digest(7, shuffled) == index_digest(7, corpus)


@given(pages.filter(lambda c: any(p.chunks for p in c)), texts)
def test_any_text_edit_moves_the_content_digest(corpus: list[SnapshotPage], extra: str) -> None:
    i = next(n for n, p in enumerate(corpus) if p.chunks)
    first = corpus[i].chunks[0]
    edited = replace(first, display_text=first.display_text + extra + "!")
    changed = [*corpus[:i], replace(corpus[i], chunks=(edited, *corpus[i].chunks[1:])),
               *corpus[i + 1 :]]  # fmt: skip
    assert content_digest(changed) != content_digest(corpus)


def test_netstrings_make_field_boundaries_unambiguous() -> None:
    one = [SnapshotPage("notion", "a", (_chunk(0, (), "bc"),))]
    two = [SnapshotPage("notion", "ab", (_chunk(0, (), "c"),))]
    assert content_digest(one) != content_digest(two)
    assert netstring("é") == "2:é,"  # bytes, not characters


def test_the_index_digest_tracks_the_space_but_not_the_text() -> None:
    corpus = [SnapshotPage("notion", "p", (_chunk(0, ("H",), "x"),))]
    assert index_digest(1, corpus) != index_digest(2, corpus)
    same_hashes = [replace(corpus[0], chunks=(replace(corpus[0].chunks[0], display_text="y"),))]
    assert index_digest(1, same_hashes) == index_digest(1, corpus)


def test_occurrence_ranks_equal_chunks_in_ordinal_order() -> None:
    dup = [_chunk(2, ("H",), "same"), _chunk(0, ("H",), "same"), _chunk(1, ("G",), "same")]
    assert occurrences(dup) == [1, 0, 0]


def test_page_keys_follow_principle_6() -> None:
    assert page_key("notion", "page-id", {"filename": "x.md"}) == "page-id"
    assert page_key("file", "upload-id", {"filename": "x.md"}) == "x.md"
    assert page_key("file", "upload-id", {}) == "upload-id"


# ----------------------------------------------------------------- embed input
def test_the_embed_input_is_the_chunkers_own_on_every_demo_chunk(
    demo_pages: list[DemoPage],
) -> None:
    assert HEADER_SEPARATOR == CHUNKER_SEPARATOR
    for page in demo_pages:
        for chunk in page.chunks:
            assert embedded_text(page.title, chunk.heading_path, chunk.display_text) == (
                chunk.embedded_text
            )
            assert (
                checked_embed_input(
                    page.title, chunk.heading_path, chunk.display_text, chunk.embed_sha256
                )
                == chunk.embedded_text
            )


def test_a_mismatched_embed_input_is_refused() -> None:
    with pytest.raises(EmbedInputMismatch):
        checked_embed_input("T", ("H",), "text", hashlib.sha256(b"other").digest())
    assert embedded_text("  ", (), "body") == "body"  # no header: the display text alone


# ----------------------------------------------------------------- policy and refusals
def test_job_specs_are_pure_functions_of_their_dedupe_keys() -> None:
    snap, coll, target = uuid4(), uuid4(), uuid4()
    spec = copy_job(snap, coll, target)
    assert spec.kind == KIND_SNAPSHOT_COPY
    assert spec.priority is JobPriority.interactive
    assert spec.dedupe_key == f"{KIND_SNAPSHOT_COPY}:{snap}"
    assert spec.payload == copy_job(snap, coll, target).payload
    again = reembed_job(coll, 3)
    assert again.kind == KIND_SNAPSHOT_REEMBED
    assert again.priority is JobPriority.maintenance
    assert again.payload == {"collection_id": str(coll), "space_id": 3}


def test_the_eval_key_is_exactly_eval_chat_read_and_never_an_mcp_key() -> None:
    assert {"eval", "chat", "read"} == EVAL_KEY_SCOPES
    assert client_kind_for_name(EVAL_KEY_NAME) is ClientKind.user


@pytest.mark.parametrize(
    ("state", "error"),
    [("42501", Forbidden), ("P0002", NotFound), ("22023", ValidationFailed),
     ("55000", SnapshotNotReady)],
)  # fmt: skip
def test_refusals_map_to_domain_errors(state: str, error: type[Exception]) -> None:
    refusal = refusal_for(state, "the message")
    assert isinstance(refusal, error)
    assert str(refusal) == "the message"


def test_only_the_pairing_unique_violation_is_a_refusal() -> None:
    exists = refusal_for("23505", "duplicate key", constraint="tenants_eval_of_tenant_id_key")
    assert isinstance(exists, EvalTenantExists)
    assert "duplicate key" not in str(exists)  # a fixed message, never the driver's
    # an id collision inside the copy is a bug: it must surface as one, not as a 409
    assert refusal_for("23505", "duplicate key", constraint="documents_pkey") is None
    assert refusal_for("23505", "duplicate key") is None


def test_unknown_states_are_not_refusals() -> None:
    assert refusal_for("40001", "serialization failure") is None
    assert refusal_for(None, None) is None
    assert issubclass(SnapshotNotReady, Conflict)


def test_the_migration_spells_the_frozen_namespace() -> None:
    source = (
        files("ragforge_core.db.migrations") / "versions" / "0012_eval_snapshot.py"
    ).read_text()
    assert f'RAGFORGE_NAMESPACE = "{ids.RAGFORGE_NAMESPACE}"' in source
    assert f"'{ids.RAGFORGE_NAMESPACE}'::uuid" in source


# ----------------------------------------------------------------- job handlers (fake ports)
LEASE_TENANT = uuid4()


def lease(kind: str, payload: dict[str, Any]) -> JobLease:
    return JobLease(uuid4(), LEASE_TENANT, kind, payload, lease_version=1, attempts=1)


class FakeTxManager:
    def __init__(self) -> None:
        self.opened: list[UUID | None] = []

    @asynccontextmanager
    async def _open(self, tenant: UUID | None) -> AsyncIterator[Tx]:
        self.opened.append(tenant)
        yield object()

    def tenant(self, tenant_id: UUID) -> Any:
        return self._open(tenant_id)

    def system(self) -> Any:
        return self._open(None)


class FakeQueue:
    def __init__(self, *, beats: bool = True) -> None:
        self.beats = beats
        self.progress: list[dict[str, object]] = []
        self.completed = 0

    async def assert_lease(self, tx: Tx, lease: JobLease) -> None:
        return None

    async def report_progress(self, tx: Tx, lease: JobLease, progress: dict[str, object]) -> None:
        self.progress.append(progress)

    async def complete(self, tx: Tx, lease: JobLease) -> None:
        self.completed += 1

    async def heartbeat(self, lease: JobLease) -> bool:
        return self.beats


class FakeStore:
    def __init__(self, *, error: Exception | None = None, sources: Sequence[EmbedSource] = ()):
        self.error = error
        self.sources = list(sources)
        self.finished: list[int] = []

    async def copy(
        self, tx: Tx, *, source_collection_id: UUID, target_tenant_id: UUID, snapshot_id: UUID
    ) -> CopyOutcome:
        if self.error is not None:
            raise self.error
        return CopyOutcome(2, 3, 3, 1, 0, "c" * 64, "i" * 64)

    async def embed_sources(self, tx: Tx, collection_id: UUID) -> Sequence[EmbedSource]:
        if self.error is not None:
            raise self.error
        return self.sources

    async def finish_reembed(self, tx: Tx, collection_id: UUID, space_id: int) -> ReembedOutcome:
        self.finished.append(space_id)
        return ReembedOutcome("c" * 64, "j" * 64, len(self.sources))


class FakeIndex:
    def __init__(self, missing: set[bytes] | None = None) -> None:
        self.missing = missing
        self.inserted: dict[bytes, Sequence[float]] = {}

    async def touch_and_find_missing(
        self, tx: Tx, collection_id: UUID, space_id: int, embed_sha256s: Sequence[bytes]
    ) -> set[bytes]:
        return set(embed_sha256s) if self.missing is None else self.missing

    async def insert_embeddings(
        self, tx: Tx, collection_id: UUID, space_id: int, vectors: dict[bytes, Sequence[float]]
    ) -> int:
        self.inserted.update(vectors)
        return len(vectors)


def source(text: str, *, title: str = "Page", heading: tuple[str, ...] = ("H",)) -> EmbedSource:
    sha = hashlib.sha256(embedded_text(title, heading, text).encode()).digest()
    return EmbedSource(sha, title, heading, text)


def deps(
    store: FakeStore, *, queue: FakeQueue | None = None, index: FakeIndex | None = None,
    dims: dict[int, int] | None = None,
) -> tuple[EvalSnapshotDeps, FakeQueue, FakeIndex, FakeEmbedders]:  # fmt: skip
    q, i, e = queue or FakeQueue(), index or FakeIndex(), FakeEmbedders(dims or {5: 4})
    built = EvalSnapshotDeps(
        tx=cast(TransactionManager, FakeTxManager()),
        queue=cast(JobQueue, q),
        store=store,
        index=cast(DocumentIndex, i),
        embedders=e,
        embed_batch=2,
    )
    return built, q, i, e


def copy_lease() -> JobLease:
    return lease(KIND_SNAPSHOT_COPY, copy_job(uuid4(), uuid4(), uuid4()).payload)


async def test_a_malformed_payload_is_permanent_and_touches_nothing() -> None:
    store = FakeStore()
    d, q, _, _ = deps(store)
    with pytest.raises(PermanentJobError, match="malformed"):
        await run_copy_job(lease(KIND_SNAPSHOT_COPY, {"snapshot_id": "nope"}), d)
    with pytest.raises(PermanentJobError, match="malformed"):
        await run_reembed_job(lease(KIND_SNAPSHOT_REEMBED, {"collection_id": str(uuid4()),
                                                            "space_id": 0}), d)  # fmt: skip
    assert q.completed == 0
    assert q.progress == []


@pytest.mark.parametrize("refusal", [Forbidden("guard"), NotFound("gone"), ValidationFailed("x")])
async def test_guard_refusals_are_permanent(refusal: Exception) -> None:
    d, q, _, _ = deps(FakeStore(error=refusal))
    with pytest.raises(PermanentJobError) as caught:
        await run_copy_job(copy_lease(), d)
    assert caught.value.__cause__ is refusal
    assert q.completed == 0


async def test_a_source_still_indexing_is_retried_not_buried() -> None:
    d, q, _, _ = deps(FakeStore(error=SnapshotNotReady("still indexing")))
    with pytest.raises(SnapshotNotReady):
        await run_copy_job(copy_lease(), d)
    assert q.completed == 0


async def test_a_copy_reports_its_counts_and_digests_and_completes_once() -> None:
    d, q, _, _ = deps(FakeStore())
    await run_copy_job(copy_lease(), d)
    assert q.completed == 1
    assert q.progress == [{"stage": "done", "pages": 2, "chunks": 3, "vectors": 3, "links": 1,
                           "excluded": 0, "content_digest": "c" * 64,
                           "index_digest": "i" * 64}]  # fmt: skip


def reembed_lease(space: int = 5) -> JobLease:
    return lease(KIND_SNAPSHOT_REEMBED, reembed_job(uuid4(), space).payload)


async def test_reembed_embeds_only_missing_hashes_in_bounded_batches() -> None:
    sources = [source(f"text {n}") for n in range(5)]
    missing = {s.embed_sha256 for s in sources[:3]}
    store = FakeStore(sources=sources)
    d, q, index, embedders = deps(store, index=FakeIndex(missing))
    await run_reembed_job(reembed_lease(), d)
    assert set(index.inserted) == missing
    assert sorted(embedders.for_space(5).inputs) == sorted(
        embedded_text(s.title, s.heading_path, s.display_text) for s in sources[:3]
    )
    assert store.finished == [5]
    assert q.completed == 1
    assert q.progress[0] == {"stage": "embedding", "space_id": 5, "total": 3}


async def test_reembed_with_nothing_missing_only_flips() -> None:
    store = FakeStore(sources=[source("a")])
    d, _, index, embedders = deps(store, index=FakeIndex(set()))
    await run_reembed_job(reembed_lease(), d)
    assert embedders.ensured == []
    assert index.inserted == {}
    assert store.finished == [5]


async def test_reembed_refuses_a_drifted_input_before_any_embed_call() -> None:
    drifted = replace(source("a"), display_text="b")
    d, q, index, embedders = deps(FakeStore(sources=[drifted]))
    with pytest.raises(PermanentJobError, match="embed_sha256"):
        await run_reembed_job(reembed_lease(), d)
    assert embedders.for_space(5).inputs == []
    assert index.inserted == {}
    assert q.completed == 0


async def test_reembed_stops_when_the_lease_is_lost() -> None:
    d, q, index, _ = deps(FakeStore(sources=[source("a")]), queue=FakeQueue(beats=False))
    with pytest.raises(LeaseLost):
        await run_reembed_job(reembed_lease(), d)
    assert index.inserted == {}
    assert q.completed == 0


class ShortEmbedder(FakeEmbedder):
    """Claims `dim` but returns shorter vectors (a misconfigured model)."""

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [v[:2] for v in await super().embed_documents(texts)]


async def test_reembed_rejects_vectors_of_the_wrong_shape() -> None:
    d, q, index, embedders = deps(FakeStore(sources=[source("a")]))
    embedders.embedders[5] = ShortEmbedder("short", 4)
    with pytest.raises(EmbeddingContractError):
        await run_reembed_job(reembed_lease(), d)
    assert index.inserted == {}
    assert q.completed == 0


def test_the_worker_registers_both_kinds_with_their_bounds() -> None:
    handlers = JobHandlers()
    d, _, _, _ = deps(FakeStore())
    register_eval_snapshot_handlers(handlers, d)
    assert handlers.kinds() == {KIND_SNAPSHOT_COPY, KIND_SNAPSHOT_REEMBED}
    assert not handlers.kinds() & SYNC_KINDS
    for kind in handlers.kinds():
        assert handlers.max_runtime_s(kind) == MAX_RUNTIME_S[kind]
        assert not handlers.reports_outcome(kind)  # they complete in their fenced tx
