"""Sync job handlers (module C2): sync_connector, sync_page, reconcile_deletes, delete_connector,
ingest_file. Each takes a JobLease plus injected ports (`SyncDeps`).

Rules every handler follows (ADR-0007, ADR-0010, DESIGN section 3 Flow A):
- Network and model calls happen outside any transaction; reads use short transactions.
- Every transaction with side effects starts with `JobQueue.assert_lease` and the job's last
  one ends with `JobQueue.complete`, so a worker that lost its lease commits nothing.
- The tenant always comes from the lease (set by the claim), never from the payload.
- Handlers raise; `run_sync_job` maps exceptions to `JobQueue.fail` (permanent or retryable).
"""

from __future__ import annotations

import dataclasses
import hashlib
import logging
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable, Sequence
from contextlib import AbstractAsyncContextManager
from datetime import datetime
from uuid import UUID

from ragforge_core.domain.errors import (
    LeaseLost,
    NotFound,
    UpstreamRateLimited,
    ValidationFailed,
)
from ragforge_core.domain.ids import document_id
from ragforge_core.domain.models import (
    DocumentRecord,
    JobLease,
    JobPriority,
    JobSpec,
    SourceDocument,
    SourceItemRef,
    SourceKind,
)
from ragforge_core.domain.ports import SourceConnector
from ragforge_core.services.sync.deps import PermanentJobError, SyncDeps
from ragforge_core.services.sync.indexing import index_document
from ragforge_core.services.sync.policy import (
    CURSOR_SCOPE,
    KIND_DELETE_CONNECTOR,
    KIND_INGEST_FILE,
    KIND_RECONCILE,
    KIND_SYNC_CONNECTOR,
    KIND_SYNC_PAGE,
    is_fresh,
    listing_since,
    next_cursor,
    page_sha256,
    sync_page_key,
    sync_page_payload,
)
from ragforge_core.services.sync.ports import ConnectorState, KnownDocument

log = logging.getLogger(__name__)

Handler = Callable[[JobLease, SyncDeps], Awaitable[None]]


# ================================================================= shared helpers
def _payload_uuid(lease: JobLease, key: str) -> UUID:
    raw = lease.payload.get(key)
    try:
        return UUID(str(raw))
    except (TypeError, ValueError):
        raise PermanentJobError(f"invalid payload: {key}") from None


def _payload_str(lease: JobLease, key: str) -> str:
    raw = lease.payload.get(key)
    if not isinstance(raw, str) or not raw or len(raw) > 512:
        raise PermanentJobError(f"invalid payload: {key}")
    return raw


def associated_data(tenant_id: UUID, connector_id: UUID) -> bytes:
    """Binds the sealed token to its row (PHASE1 seam contract with module E)."""
    return f"connector:{tenant_id}:{connector_id}".encode()


def _source_kind(conn: ConnectorState) -> SourceKind:
    try:
        return SourceKind(conn.kind)
    except ValueError:
        raise PermanentJobError(f"unsupported connector kind: {conn.kind}") from None


async def _complete_noop(deps: SyncDeps, lease: JobLease, reason: str) -> None:
    async with deps.tx.tenant(lease.tenant_id) as tx:
        await deps.queue.assert_lease(tx, lease)
        await deps.queue.report_progress(tx, lease, {"skipped": reason})
        await deps.queue.complete(tx, lease)


async def _heartbeat(deps: SyncDeps, lease: JobLease) -> None:
    if not await deps.queue.heartbeat(lease):
        raise LeaseLost("lease lost")


async def _load_active_connector(
    deps: SyncDeps, lease: JobLease, connector_id: UUID
) -> ConnectorState | None:
    async with deps.tx.tenant(lease.tenant_id) as tx:
        conn = await deps.store.load_connector(tx, connector_id)
    return conn if conn is not None and conn.status == "active" else None


async def _mark_connector_error(
    deps: SyncDeps, lease: JobLease, connector_id: UUID, message: str
) -> None:
    """Fenced but NOT completed: the caller fails the job right after."""
    async with deps.tx.tenant(lease.tenant_id) as tx:
        await deps.queue.assert_lease(tx, lease)
        await deps.store.mark_connector_error(tx, connector_id, message)
        await deps.store.audit(
            tx,
            actor=deps.actor,
            action="connector.error",
            target=str(connector_id),
            details={"error": message},
        )


async def _open_token(deps: SyncDeps, lease: JobLease, conn: ConnectorState) -> bytes:
    """Decrypt the connector token. Any failure is permanent and never echoes key material:
    the original exception is dropped (`from None`) so nothing about it reaches last_error."""
    message = "connector secret is missing or cannot be decrypted"
    if conn.sealed is None:
        await _mark_connector_error(deps, lease, conn.id, message)
        raise PermanentJobError(message)
    try:
        return deps.secrets.open(
            conn.sealed, associated_data=associated_data(lease.tenant_id, conn.id)
        )
    except Exception:
        await _mark_connector_error(deps, lease, conn.id, message)
        raise PermanentJobError(message) from None


def _open_source(
    deps: SyncDeps, conn: ConnectorState, token: bytes
) -> AbstractAsyncContextManager[SourceConnector]:
    return deps.connectors.open(
        connector_id=conn.id, kind=conn.kind, token=token, config=conn.config
    )


def _batches[T](items: Sequence[T], size: int) -> Iterable[Sequence[T]]:
    step = max(1, size)
    for start in range(0, len(items), step):
        yield items[start : start + step]


async def _collect(
    deps: SyncDeps, lease: JobLease, refs: AsyncIterator[SourceItemRef]
) -> dict[str, SourceItemRef]:
    """Drain a listing, keeping the newest ref per id, heartbeating on long listings."""
    seen: dict[str, SourceItemRef] = {}
    count = 0
    async for ref in refs:
        prior = seen.get(ref.external_id)
        if prior is None or _newer(ref.last_edited_time, prior.last_edited_time):
            seen[ref.external_id] = ref
        count += 1
        if count % max(1, deps.policy.heartbeat_every) == 0:
            await _heartbeat(deps, lease)
    return seen


def _newer(candidate: datetime | None, current: datetime | None) -> bool:
    return candidate is not None and (current is None or candidate > current)


async def _stale_candidates(
    deps: SyncDeps,
    lease: JobLease,
    connector_id: UUID,
    kind: SourceKind,
    refs: dict[str, SourceItemRef],
) -> list[str]:
    """Listed ids that are neither excluded nor fresh (ADR-0007), in short read transactions."""
    ids = sorted(refs)
    keep: list[str] = []
    for batch in _batches(ids, deps.policy.lookup_batch):
        async with deps.tx.tenant(lease.tenant_id) as tx:
            excluded = await deps.store.excluded_ids(tx, kind, batch)
            fetched = await deps.store.fetched_at_by_external_id(tx, connector_id, batch)
        keep.extend(
            ext
            for ext in batch
            if ext not in excluded and not is_fresh(refs[ext].last_edited_time, fetched.get(ext))
        )
    return keep


def _page_job(connector_id: UUID, external_id: str, priority: JobPriority) -> JobSpec:
    return JobSpec(
        kind=KIND_SYNC_PAGE,
        priority=priority,
        payload=sync_page_payload(connector_id, external_id),
        dedupe_key=sync_page_key(connector_id, external_id),
        connector_id=connector_id,
    )


# ================================================================= sync_connector
async def handle_sync_connector(lease: JobLease, deps: SyncDeps) -> None:
    """List changed pages since the cursor, enqueue sync_page for stale ones, advance cursor.

    Enqueues and the cursor advance commit in ONE fenced transaction: the cursor can never move
    past a page whose job was not enqueued. The cursor advances when the listing completes,
    independent of page-job outcomes (reconcile repairs dead page jobs).
    """
    connector_id = _payload_uuid(lease, "connector_id")
    async with deps.tx.tenant(lease.tenant_id) as tx:
        conn = await deps.store.load_connector(tx, connector_id)
        cursor = await deps.store.read_cursor(tx, connector_id, CURSOR_SCOPE)
    if conn is None or conn.status != "active":
        await _complete_noop(deps, lease, "connector not active")
        return
    kind = _source_kind(conn)
    token = await _open_token(deps, lease, conn)

    async with _open_source(deps, conn, token) as source:
        since = listing_since(cursor, deps.policy.cursor_overlap)
        refs = await _collect(deps, lease, source.list_changed(since))
    observed = max((r.last_edited_time for r in refs.values() if r.last_edited_time), default=None)
    candidates = await _stale_candidates(deps, lease, connector_id, kind, refs)
    new_cursor = next_cursor(cursor, observed)

    async with deps.tx.tenant(lease.tenant_id) as tx:
        await deps.queue.assert_lease(tx, lease)
        for external_id in candidates:
            await deps.queue.enqueue(
                tx, _page_job(connector_id, external_id, JobPriority.incremental)
            )
        if new_cursor is not None:
            await deps.store.advance_cursor(tx, connector_id, CURSOR_SCOPE, new_cursor)
        await deps.store.mark_connector_synced(tx, connector_id, deps.policy.poll_interval_s)
        await deps.queue.report_progress(
            tx, lease, {"listed": len(refs), "enqueued": len(candidates)}
        )
        await deps.queue.complete(tx, lease)


# ================================================================= sync_page
async def handle_sync_page(lease: JobLease, deps: SyncDeps) -> None:
    connector_id = _payload_uuid(lease, "connector_id")
    external_id = _payload_str(lease, "external_id")
    conn = await _load_active_connector(deps, lease, connector_id)
    if conn is None:
        await _complete_noop(deps, lease, "connector not active")
        return
    kind = _source_kind(conn)
    async with deps.tx.tenant(lease.tenant_id) as tx:
        excluded = await deps.store.excluded_ids(tx, kind, [external_id])
    if excluded:
        await _complete_noop(deps, lease, "excluded")
        return

    doc_id = document_id(lease.tenant_id, conn.target_collection_id, connector_id, external_id)
    token = await _open_token(deps, lease, conn)
    try:
        async with _open_source(deps, conn, token) as source:
            doc = await source.fetch(SourceItemRef(kind, external_id, None))
    except NotFound:
        # A definitive "gone" from the connector (Notion: trashed, archived or object_not_found;
        # the same evidence reconcile's confirm_deleted accepts) is an answer, not a failure:
        # record it, so an undo of a user delete can never revive what the owner deleted.
        await _tombstone_gone(deps, lease, kind, external_id, doc_id)
        return
    _check_fetched(doc, kind, external_id)
    assert doc.fetched_at is not None  # narrowed by _check_fetched

    record = DocumentRecord(
        id=doc_id,
        collection_id=conn.target_collection_id,
        connector_id=connector_id,
        upload_id=None,
        source_kind=kind,
        external_id=external_id,
        external_url=doc.external_url,
        title=doc.title,
        external_updated_at=doc.ref.last_edited_time,
        fetched_at=doc.fetched_at,
        page_sha256=page_sha256(doc),
        metadata=dict(doc.metadata),
        links_to=doc.links_to,
        synced_source_ids=doc.synced_source_ids,
    )
    await index_document(deps, lease, record, deps.chunker.chunk(doc))


async def _tombstone_gone(
    deps: SyncDeps, lease: JobLease, kind: SourceKind, external_id: str, doc_id: UUID
) -> None:
    """Fenced commit in sync's lock order (job lease, source-item lock, document row): mark the
    copy 'source_deleted' (live or 'excluded'; a missing row is a no-op) and complete the job."""
    async with deps.tx.tenant(lease.tenant_id) as tx:
        await deps.queue.assert_lease(tx, lease)
        await deps.store.lock_source_item(tx, kind, external_id)
        await deps.index.tombstone(tx, doc_id)
        await deps.store.audit(
            tx,
            actor=deps.actor,
            action="document.tombstoned",
            target=str(doc_id),
            details={"reason": "source_gone_on_fetch"},
        )
        await deps.queue.complete(tx, lease)


def _check_fetched(doc: SourceDocument, kind: SourceKind, external_id: str) -> None:
    """Connector contract guards: the document must be the one asked for, and carry the
    Notion-clock fetched_at that the monotonic upsert depends on."""
    if doc.ref.external_id != external_id or doc.ref.source_kind is not kind:
        raise PermanentJobError("connector returned a different document than requested")
    if doc.fetched_at is None:
        raise PermanentJobError("connector returned a document without fetched_at")


# ================================================================= reconcile_deletes
async def handle_reconcile(lease: JobLease, deps: SyncDeps) -> None:
    """Nightly repair (ADR-0007): re-enqueue every non-fresh listed page (dead jobs, missed
    polls, same-minute edits), re-enqueue pages whose synced-block sources changed, and
    tombstone missing pages ONLY when the connector confirms the deletion."""
    connector_id = _payload_uuid(lease, "connector_id")
    conn = await _load_active_connector(deps, lease, connector_id)
    if conn is None:
        await _complete_noop(deps, lease, "connector not active")
        return
    kind = _source_kind(conn)
    token = await _open_token(deps, lease, conn)

    async with _open_source(deps, conn, token) as source:
        listing = await _collect(deps, lease, source.list_all())
        async with deps.tx.tenant(lease.tenant_id) as tx:
            known = list(await deps.store.live_documents(tx, connector_id))
        stale = set(await _stale_candidates(deps, lease, connector_id, kind, listing))
        stale |= await _synced_source_stale(deps, lease, kind, known, listing)
        confirmed = await _confirm_missing(deps, lease, source, known, listing)

    async with deps.tx.tenant(lease.tenant_id) as tx:
        await deps.queue.assert_lease(tx, lease)
        for external_id in sorted(stale):
            await deps.queue.enqueue(tx, _page_job(connector_id, external_id, JobPriority.backfill))
        # Same lock order as every other tombstone writer and sync's commit: the source-item
        # advisory lock first, then the document row (taken by tombstone). Sorted, so two
        # reconciles of overlapping connectors take the item locks in the same order.
        for doc in sorted(confirmed, key=lambda d: d.external_id):
            await deps.store.lock_source_item(tx, kind, doc.external_id)
            await deps.index.tombstone(tx, doc.id)
            await deps.store.audit(
                tx,
                actor=deps.actor,
                action="document.tombstoned",
                target=str(doc.id),
                details={"reason": "confirmed_deleted"},
            )
        missing = sum(1 for d in known if d.external_id not in listing)
        await deps.queue.report_progress(
            tx,
            lease,
            {
                "listed": len(listing),
                "requeued": len(stale),
                "tombstoned": len(confirmed),
                "unconfirmed": missing - len(confirmed),
            },
        )
        await deps.queue.complete(tx, lease)


async def _synced_source_stale(
    deps: SyncDeps,
    lease: JobLease,
    kind: SourceKind,
    known: Sequence[KnownDocument],
    listing: dict[str, SourceItemRef],
) -> set[str]:
    """Pages embedding synced blocks are stale when any source changed since their fetch.

    Editing an original never bumps the embedding page's last_edited_time, so the page's own
    freshness says nothing. A source we cannot see in the listing has an unknown edit time and
    therefore counts as changed (is_fresh(None, ...) is False): converging daily beats missing
    an edit forever.
    """
    dependents = [d for d in known if d.synced_source_ids and d.external_id in listing]
    if not dependents:
        return set()
    async with deps.tx.tenant(lease.tenant_id) as tx:
        excluded = await deps.store.excluded_ids(tx, kind, [d.external_id for d in dependents])
    stale: set[str] = set()
    for doc in dependents:
        if doc.external_id in excluded:
            continue
        for source_id in doc.synced_source_ids:
            ref = listing.get(source_id)
            if not is_fresh(ref.last_edited_time if ref else None, doc.fetched_at):
                stale.add(doc.external_id)
                break
    return stale


async def _confirm_missing(
    deps: SyncDeps,
    lease: JobLease,
    source: SourceConnector,
    known: Sequence[KnownDocument],
    listing: dict[str, SourceItemRef],
) -> list[KnownDocument]:
    """Absence from a listing is NOT evidence (search is not exhaustive): every candidate needs
    a definitive confirmation. Errors count as "not confirmed" and are retried next night."""
    confirmed: list[KnownDocument] = []
    for n, doc in enumerate(d for d in known if d.external_id not in listing):
        try:
            gone = await source.confirm_deleted(doc.external_id)
        except (LeaseLost, UpstreamRateLimited):
            raise
        except Exception:
            log.warning("confirm_deleted failed for document %s; retry next run", doc.id)
            gone = False
        if gone is True:
            confirmed.append(doc)
        if (n + 1) % max(1, deps.policy.heartbeat_every) == 0:
            await _heartbeat(deps, lease)
    return confirmed


# ================================================================= delete_connector
async def handle_delete_connector(lease: JobLease, deps: SyncDeps) -> None:
    """Destroy the secret, delete documents in short fenced batches, then the connector row.

    Batches commit independently (each fenced): deletion is idempotent, so a retry after a lost
    lease simply continues. Only the final transaction completes the job.
    """
    connector_id = _payload_uuid(lease, "connector_id")
    async with deps.tx.tenant(lease.tenant_id) as tx:
        await deps.queue.assert_lease(tx, lease)
        await deps.store.destroy_secret(tx, connector_id)

    deleted = 0
    while True:
        async with deps.tx.tenant(lease.tenant_id) as tx:
            await deps.queue.assert_lease(tx, lease)
            n = await deps.store.delete_connector_documents(
                tx, connector_id, deps.policy.delete_batch
            )
            deleted += n
            if n:
                await deps.queue.report_progress(tx, lease, {"documents_deleted": deleted})
        if n == 0:
            break
        await _heartbeat(deps, lease)

    async with deps.tx.tenant(lease.tenant_id) as tx:
        await deps.queue.assert_lease(tx, lease)
        existed = await deps.store.delete_connector(tx, connector_id)
        await deps.store.audit(
            tx,
            actor=deps.actor,
            action="connector.deleted",
            target=str(connector_id),
            details={"documents_deleted": deleted, "existed": existed},
        )
        await deps.queue.report_progress(tx, lease, {"documents_deleted": deleted})
        await deps.queue.complete(tx, lease)


# ================================================================= ingest_file
async def handle_ingest_file(lease: JobLease, deps: SyncDeps) -> None:
    """Upload -> blob -> sandboxed parse -> the same indexing flow as Notion pages.

    external_id is the upload id (one document per upload, excludable by id), and fetched_at is
    the upload's created_at: re-ingesting the same upload (reindex) is idempotent.
    """
    upload_id = _payload_uuid(lease, "upload_id")
    external_id = str(upload_id)
    async with deps.tx.tenant(lease.tenant_id) as tx:
        upload = await deps.store.load_upload(tx, upload_id)
        excluded = await deps.store.excluded_ids(tx, SourceKind.file, [external_id])
    if upload is None:
        await _complete_noop(deps, lease, "upload not found")
        return
    if excluded:
        await _complete_noop(deps, lease, "excluded")
        return

    try:
        data = await deps.blobs.get(upload.blob_key)
    except NotFound:
        raise PermanentJobError("upload blob is missing") from None
    if len(data) != upload.size_bytes or hashlib.sha256(data).digest() != upload.sha256:
        raise PermanentJobError("upload blob failed its integrity check")

    parsed = await deps.parser.parse(upload.filename, data)
    doc = dataclasses.replace(
        parsed,
        ref=SourceItemRef(SourceKind.file, external_id, upload.created_at),
        external_url=None,
        fetched_at=upload.created_at,
        metadata={**parsed.metadata, "filename": upload.filename},
    )
    record = DocumentRecord(
        id=document_id(lease.tenant_id, upload.collection_id, None, external_id),
        collection_id=upload.collection_id,
        connector_id=None,
        upload_id=upload.id,
        source_kind=SourceKind.file,
        external_id=external_id,
        external_url=None,
        title=doc.title,
        external_updated_at=upload.created_at,
        fetched_at=upload.created_at,
        page_sha256=page_sha256(doc),
        metadata=dict(doc.metadata),
    )
    await index_document(deps, lease, record, deps.chunker.chunk(doc))


# ================================================================= dispatch + failure policy
HANDLERS: dict[str, Handler] = {
    KIND_SYNC_CONNECTOR: handle_sync_connector,
    KIND_SYNC_PAGE: handle_sync_page,
    KIND_RECONCILE: handle_reconcile,
    KIND_DELETE_CONNECTOR: handle_delete_connector,
    KIND_INGEST_FILE: handle_ingest_file,
}

_PERMANENT = (PermanentJobError, NotFound, ValidationFailed)


def _safe_error(exc: BaseException, limit: int) -> str:
    """last_error is user-visible: class name plus a bounded message. Handler-raised messages
    are written to be safe; foreign exceptions are truncated (connectors must keep tokens out
    of messages, see ConnectorFactory)."""
    text = f"{type(exc).__name__}: {exc}" if str(exc) else type(exc).__name__
    return text[:limit]


async def run_sync_job(lease: JobLease, deps: SyncDeps) -> None:
    """Run one sync job and translate its outcome into JobQueue calls. Registered with
    `reports_outcome=True`: every path ends the job here, so the runtime does not complete it.

    - LeaseLost: another worker owns the job now; never call fail. Re-raised so the runtime
      logs `job_lease_lost` (the one signal for real fencing losses).
    - UpstreamRateLimited: retry honoring Retry-After.
    - Permanent errors (bad input, definitive 404, corrupt upload): straight to dead.
    - Anything else: retry with backoff (dead after max_attempts; reconcile repairs pages).
    asyncio.CancelledError (shutdown) propagates: the worker releases the job.
    """
    handler = HANDLERS.get(lease.kind)
    limit = deps.policy.max_error_chars
    if handler is None:
        await deps.queue.fail(
            lease, f"unknown job kind: {lease.kind}"[:limit], None, permanent=True
        )
        return
    try:
        await handler(lease, deps)
    except LeaseLost:
        raise
    except UpstreamRateLimited as exc:
        await deps.queue.fail(lease, "upstream rate limited", exc.retry_after_s)
    except _PERMANENT as exc:
        await deps.queue.fail(lease, _safe_error(exc, limit), None, permanent=True)
    except Exception as exc:
        log.warning("job %s (%s) failed: %s", lease.job_id, lease.kind, type(exc).__name__)
        await deps.queue.fail(lease, _safe_error(exc, limit), None)
