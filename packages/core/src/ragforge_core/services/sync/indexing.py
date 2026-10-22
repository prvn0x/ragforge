"""The 3-step indexing flow shared by sync_page and ingest_file (DocumentIndex docstring,
DESIGN section 3 Flow A steps 6-7).

  1. short tx:  target spaces; touch every referenced hash and find the missing ones (commits,
                so the touches protect the hashes from GC while we embed)
  2. no tx:     embed only the missing texts, per space
  3. fenced tx: assert_lease -> exclusion re-check -> insert_embeddings -> upsert_document ->
                replace_chunks (only if inserted/updated) -> audit -> progress -> complete

No model call ever happens while a transaction is open: a slow embedder must not hold row
locks or a pooled connection, and step 3 is the only transaction that writes the document.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from ragforge_core.domain.errors import LeaseLost
from ragforge_core.domain.models import Chunk, DocumentRecord, JobLease, UpsertOutcome
from ragforge_core.services.sync.deps import SyncDeps

log = logging.getLogger(__name__)

_CHANGED = frozenset({UpsertOutcome.inserted, UpsertOutcome.updated})


@dataclass(frozen=True, slots=True)
class IndexResult:
    outcome: UpsertOutcome | None  # None: excluded while we worked, nothing written
    chunks: int
    embedded: int


class EmbeddingContractError(Exception):
    """The embedder returned the wrong number or size of vectors (retryable: likely transient
    or a misconfigured model that an operator will fix)."""


def _texts_by_hash(chunks: Sequence[Chunk]) -> dict[bytes, str]:
    """One text per distinct embed hash (identical text in two chunks is embedded once)."""
    texts: dict[bytes, str] = {}
    for chunk in chunks:
        texts.setdefault(chunk.embed_sha256, chunk.embedded_text)
    return texts


async def _find_missing(
    deps: SyncDeps, lease: JobLease, record: DocumentRecord, hashes: Sequence[bytes]
) -> dict[int, set[bytes]]:
    """Step 1. Touches commit here on purpose: they must outlive a failed step 3."""
    async with deps.tx.tenant(lease.tenant_id) as tx:
        spaces = list(await deps.index.target_spaces(tx, record.collection_id))
        missing: dict[int, set[bytes]] = {}
        for space_id in spaces:
            missing[space_id] = set(
                await deps.index.touch_and_find_missing(tx, record.collection_id, space_id, hashes)
            )
        return missing


async def _ensure_lease(deps: SyncDeps, lease: JobLease) -> None:
    if not await deps.queue.heartbeat(lease):
        raise LeaseLost("lease lost while embedding")


async def _embed_space(
    deps: SyncDeps, lease: JobLease, space_id: int, texts: Mapping[bytes, str]
) -> dict[bytes, list[float]]:
    """Step 2 for one space, in bounded batches with a heartbeat between them so a long embed
    either keeps its lease or stops early instead of finishing work a zombie would discard."""
    await deps.embedders.ensure(space_id)  # no tx open; picks up a space created after startup
    embedder = deps.embedders.for_space(space_id)
    items = sorted(texts.items())  # deterministic batch composition
    vectors: dict[bytes, list[float]] = {}
    size = max(1, deps.policy.embed_batch)
    for start in range(0, len(items), size):
        batch = items[start : start + size]
        out = await embedder.embed_documents([text for _, text in batch])
        if len(out) != len(batch) or any(len(v) != embedder.dim for v in out):
            # last_error is user-visible: no space id or dimensions in it (they are in the log)
            log.warning("embedding contract violation: space %s, dim %s", space_id, embedder.dim)
            raise EmbeddingContractError("the embedder returned vectors of the wrong shape")
        vectors.update(zip((h for h, _ in batch), out, strict=True))
        await _ensure_lease(deps, lease)
    return vectors


async def index_document(
    deps: SyncDeps, lease: JobLease, record: DocumentRecord, chunks: Sequence[Chunk]
) -> IndexResult:
    """Run steps 1-3 and complete the job. Raises LeaseLost if the lease is gone (nothing from
    step 3 is committed then)."""
    texts = _texts_by_hash(chunks)
    missing = await _find_missing(deps, lease, record, list(texts)) if texts else {}

    vectors: dict[int, dict[bytes, list[float]]] = {}
    for space_id, hashes in missing.items():
        wanted = {h: texts[h] for h in hashes if h in texts}
        vectors[space_id] = await _embed_space(deps, lease, space_id, wanted) if wanted else {}

    return await _commit(deps, lease, record, chunks, vectors)


async def _commit(
    deps: SyncDeps,
    lease: JobLease,
    record: DocumentRecord,
    chunks: Sequence[Chunk],
    vectors: Mapping[int, Mapping[bytes, Sequence[float]]],
) -> IndexResult:
    """Step 3: one fenced transaction, so a zombie (lost lease) commits nothing at all."""
    embedded = sum(len(v) for v in vectors.values())
    async with deps.tx.tenant(lease.tenant_id) as tx:
        await deps.queue.assert_lease(tx, lease)
        # Serialize against a concurrent exclusion of this item, then re-check: an exclusion
        # that committed while we fetched/embedded must win (no zombie re-ingestion).
        await deps.store.lock_source_item(tx, record.source_kind, record.external_id)
        if await deps.store.excluded_ids(tx, record.source_kind, [record.external_id]):
            await deps.queue.report_progress(tx, lease, {"outcome": "excluded"})
            await deps.queue.complete(tx, lease)
            return IndexResult(outcome=None, chunks=0, embedded=0)

        for space_id, space_vectors in vectors.items():
            if space_vectors:
                await deps.index.insert_embeddings(
                    tx, record.collection_id, space_id, space_vectors
                )
        outcome = await deps.index.upsert_document(tx, record)
        if outcome in _CHANGED:
            await deps.index.replace_chunks(tx, record.id, record.collection_id, chunks)
        if outcome is not UpsertOutcome.stale:
            # A stale upsert affected no rows: the job completes as a no-op (DESIGN 3, step 7).
            await deps.store.audit(
                tx,
                actor=deps.actor,
                action="document.synced",
                target=str(record.id),
                details={"outcome": outcome.value, "source_kind": record.source_kind.value},
            )
        await deps.queue.report_progress(
            tx,
            lease,
            {"outcome": outcome.value, "chunks": len(chunks), "embedded": embedded},
        )
        await deps.queue.complete(tx, lease)
    log.info("indexed document %s outcome=%s", record.id, outcome.value)
    return IndexResult(outcome=outcome, chunks=len(chunks), embedded=embedded)
