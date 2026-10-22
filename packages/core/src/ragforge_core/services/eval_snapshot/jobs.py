"""The two snapshot job handlers (registry contract: services/jobs/registry.py).

copy (runs in the SOURCE tenant, which is how the copy function learns its source):
    one fenced transaction: assert_lease -> copy_collection_to_eval -> progress -> complete.
    No model call, so one transaction is right, and the digests it records describe exactly
    what committed.
re-embed (runs in the EVAL tenant), the same 3 steps as sync indexing (services/sync/indexing):
    1. short tx: the snapshot's distinct embed hashes; touch them in the new space and find the
       missing ones (touches commit, protecting the hashes from GC while we embed);
    2. no tx: rebuild each missing hash's input, re-check it against the stored embed_sha256,
       embed in bounded batches with a heartbeat between them;
    3. fenced tx: insert the vectors -> finish_eval_reembed (refuses a partial flip) ->
       progress -> complete.

Failures: the r4 guard's refusals, an unknown snapshot, a malformed payload or a rebuilt input
that does not match its hash can never succeed, so they are permanent (`dead`, audited);
`SnapshotNotReady` (the source is still indexing) and everything else retry with backoff.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass

from pydantic import BaseModel, ValidationError

from ragforge_core.domain.errors import Forbidden, LeaseLost, NotFound, ValidationFailed
from ragforge_core.domain.models import JobLease
from ragforge_core.domain.ports import DocumentIndex, EmbedderRegistry, JobQueue, TransactionManager
from ragforge_core.services.eval_snapshot.embed_input import EmbedInputMismatch, checked_embed_input
from ragforge_core.services.eval_snapshot.policy import CopyJobPayload, ReembedJobPayload
from ragforge_core.services.eval_snapshot.ports import EmbedSource, SnapshotStore
from ragforge_core.services.jobs.registry import PermanentJobError

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class EvalSnapshotDeps:
    tx: TransactionManager
    queue: JobQueue
    store: SnapshotStore
    index: DocumentIndex
    embedders: EmbedderRegistry
    embed_batch: int = 64  # texts per embed_documents call (SyncPolicy.embed_batch)


class EmbeddingContractError(Exception):
    """The embedder returned the wrong number or size of vectors (retryable)."""


def _payload[P: BaseModel](model: type[P], lease: JobLease) -> P:
    try:
        return model.model_validate(lease.payload)
    except ValidationError:
        raise PermanentJobError(f"malformed {lease.kind} payload") from None


@contextmanager
def _refusals_are_permanent() -> Iterator[None]:
    """Wrap OUTSIDE the transaction, so it has rolled back before the job is buried."""
    try:
        yield
    except (Forbidden, NotFound, ValidationFailed, EmbedInputMismatch) as exc:
        raise PermanentJobError(str(exc)) from exc


async def run_copy_job(lease: JobLease, deps: EvalSnapshotDeps) -> None:
    payload = _payload(CopyJobPayload, lease)
    with _refusals_are_permanent():
        async with deps.tx.tenant(lease.tenant_id) as tx:
            await deps.queue.assert_lease(tx, lease)
            outcome = await deps.store.copy(
                tx,
                source_collection_id=payload.source_collection_id,
                target_tenant_id=payload.target_tenant_id,
                snapshot_id=payload.snapshot_id,
            )
            await deps.queue.report_progress(
                tx,
                lease,
                {
                    "stage": "done",
                    "pages": outcome.pages,
                    "chunks": outcome.chunks,
                    "vectors": outcome.vectors,
                    "links": outcome.links,
                    "excluded": outcome.excluded,
                    "content_digest": outcome.content_digest,
                    "index_digest": outcome.index_digest,
                },
            )
            await deps.queue.complete(tx, lease)
    log.info("eval snapshot %s copied: %s pages", payload.snapshot_id, outcome.pages)


async def run_reembed_job(lease: JobLease, deps: EvalSnapshotDeps) -> None:
    payload = _payload(ReembedJobPayload, lease)
    with _refusals_are_permanent():
        sources, missing = await _plan(deps, lease, payload)
        texts = _inputs(sources, missing)
        vectors = await _embed(deps, lease, payload.space_id, texts)
        async with deps.tx.tenant(lease.tenant_id) as tx:
            await deps.queue.assert_lease(tx, lease)
            if vectors:
                await deps.index.insert_embeddings(
                    tx, payload.collection_id, payload.space_id, vectors
                )
            outcome = await deps.store.finish_reembed(tx, payload.collection_id, payload.space_id)
            await deps.queue.report_progress(
                tx,
                lease,
                {
                    "stage": "done",
                    "space_id": payload.space_id,
                    "embedded": len(vectors),
                    "vectors": outcome.vectors,
                    "index_digest": outcome.index_digest,
                },
            )
            await deps.queue.complete(tx, lease)


async def _plan(
    deps: EvalSnapshotDeps, lease: JobLease, payload: ReembedJobPayload
) -> tuple[Sequence[EmbedSource], set[bytes]]:
    """Step 1. Touches commit here on purpose: they must outlive a failed step 3."""
    async with deps.tx.tenant(lease.tenant_id) as tx:
        sources = await deps.store.embed_sources(tx, payload.collection_id)
        missing = await deps.index.touch_and_find_missing(
            tx, payload.collection_id, payload.space_id, [s.embed_sha256 for s in sources]
        )
        await deps.queue.report_progress(
            tx, lease, {"stage": "embedding", "space_id": payload.space_id, "total": len(missing)}
        )
    return sources, missing


def _inputs(sources: Sequence[EmbedSource], missing: set[bytes]) -> dict[bytes, str]:
    """Every rebuilt input is verified before the first embed call (all or nothing)."""
    return {
        s.embed_sha256: checked_embed_input(s.title, s.heading_path, s.display_text, s.embed_sha256)
        for s in sources
        if s.embed_sha256 in missing
    }


async def _embed(
    deps: EvalSnapshotDeps, lease: JobLease, space_id: int, texts: Mapping[bytes, str]
) -> dict[bytes, list[float]]:
    """Step 2, no transaction open. Bounded batches with a heartbeat between them, so a long
    re-embed either keeps its lease or stops early instead of finishing work a zombie would
    discard."""
    if not texts:
        return {}
    await deps.embedders.ensure(space_id)  # picks up a space created after startup
    embedder = deps.embedders.for_space(space_id)
    items = sorted(texts.items())  # deterministic batch composition
    vectors: dict[bytes, list[float]] = {}
    size = max(1, deps.embed_batch)
    for start in range(0, len(items), size):
        batch = items[start : start + size]
        out = await embedder.embed_documents([text for _, text in batch])
        if len(out) != len(batch) or any(len(v) != embedder.dim for v in out):
            log.warning("embedding contract violation: space %s, dim %s", space_id, embedder.dim)
            raise EmbeddingContractError("the embedder returned vectors of the wrong shape")
        vectors.update(zip((h for h, _ in batch), out, strict=True))
        if not await deps.queue.heartbeat(lease):
            raise LeaseLost("lease lost while re-embedding")
    return vectors
