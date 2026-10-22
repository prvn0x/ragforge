"""Documents, exclusions and the audit log."""

from __future__ import annotations

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Query, Response
from sqlalchemy import RowMapping
from sqlalchemy.ext.asyncio import AsyncConnection

from ragforge_contracts.api import AuditEntry, Document, Exclusion, Page
from ragforge_core.adapters.db.tx import connection
from ragforge_core.apps.api.deps import (
    Jobs,
    KeyManagerKey,
    ReadKey,
    SyncStores,
    TxManager,
    WriteKey,
)
from ragforge_core.apps.api.pagination import (
    DEFAULT_LIMIT,
    CursorParam,
    Limit,
    int_id_after,
    int_id_cursor,
    page,
    time_id_after,
)
from ragforge_core.apps.api.routes.sources import require_collections
from ragforge_core.apps.api.sql import audit, execute, fetch_all, fetch_one
from ragforge_core.domain.errors import Conflict, NotFound
from ragforge_core.domain.models import JobPriority, JobSpec, SourceKind
from ragforge_core.domain.ports import JobQueue, Tx
from ragforge_core.services.sync.policy import (
    KIND_INGEST_FILE,
    KIND_SYNC_PAGE,
    sync_page_key,
    sync_page_payload,
)
from ragforge_core.services.sync.ports import SyncStore

router = APIRouter(prefix="/v1")

DOCUMENT_COLUMNS = (
    "id, collection_id, title, external_url, external_updated_at, fetched_at, created_at"
)
EXCLUSION_COLUMNS = "id, source_kind, external_id, reason, created_at"


# ----------------------------------------------------------------- documents
@router.get("/documents", tags=["documents"])
async def list_documents(
    key: ReadKey,
    txm: TxManager,
    collection_id: UUID | None = None,
    cursor: CursorParam = None,
    limit: Limit = DEFAULT_LIMIT,
) -> Page[Document]:
    after, params = time_id_after(cursor)
    # Explicit tenant predicate (redundant with RLS): matches the partial index
    # documents_live_tenant_created (migration 0006) so each page is an index range scan.
    where = "tenant_id = :tenant AND deleted_at IS NULL" + after
    params["tenant"] = key.principal.tenant_id
    if collection_id is not None:
        where += " AND collection_id = :coll"
        params["coll"] = collection_id
    async with txm.tenant(key.principal.tenant_id) as tx:
        conn = connection(tx)
        if collection_id is not None:
            await require_collections(conn, [collection_id])
        rows = await fetch_all(
            conn,
            f"SELECT {DOCUMENT_COLUMNS} FROM rf.documents WHERE {where} "  # noqa: S608
            "ORDER BY created_at DESC, id DESC LIMIT :n",
            {**params, "n": limit + 1},
        )
    return page(rows, limit, lambda r: Document.model_validate(dict(r)))


_DOCUMENT_REF = "SELECT source_kind, external_id, deleted_at FROM rf.documents WHERE id = :id"
# Every copy of one source item in the tenant (RLS), row-locked in id order. Taken only AFTER the
# source-item advisory lock (same order as sync's commit).
_ITEM_DOCUMENTS = (
    "SELECT id, connector_id, upload_id, collection_id, deleted_at, deleted_reason "
    "FROM rf.documents WHERE source_kind = :kind AND external_id = :external_id "
    "ORDER BY id FOR UPDATE"
)
_EXCLUDE_COPIES = (
    "UPDATE rf.documents SET deleted_at = now(), deleted_reason = 'excluded', updated_at = now() "
    "WHERE id = ANY(CAST(:ids AS uuid[])) AND deleted_at IS NULL"
)
_REVIVE = (
    "UPDATE rf.documents SET deleted_at = NULL, updated_at = now() "
    "WHERE id = :id AND deleted_reason = 'excluded'"
)


@router.delete("/documents/{document_id}", status_code=204, tags=["documents"])
async def delete_document(
    document_id: UUID,
    key: WriteKey,
    txm: TxManager,
    store: SyncStores,
    jobs: Jobs,
    forget_exclusion: Annotated[bool, Query()] = False,
) -> Response:
    """Default: exclude the source item and tombstone the document in ONE transaction, so a sync
    racing this request can never resurrect it (sync skips excluded items, DESIGN section 3).

    The exclusion is tenant-wide (unique on tenant, source_kind, external_id), so EVERY live copy
    of the item in the tenant is tombstoned with it (e.g. one Notion page synced into two
    collections). Tombstoning only this copy would leave the others searchable but frozen:
    every later sync of them is skipped as excluded. The audit row records `copies`.

    Lock order matches sync's fenced commit (services/sync/indexing.py): the source-item
    advisory lock FIRST, then the document rows. Taking a row first would deadlock against a
    page sync committing at the same moment.

    `?forget_exclusion=true` undoes that and RESTORES the item (see `_restore_source_item`).
    """
    async with txm.tenant(key.principal.tenant_id) as tx:
        conn = connection(tx)
        doc = await fetch_one(conn, _DOCUMENT_REF, {"id": document_id})
        if doc is None or (doc["deleted_at"] is not None and not forget_exclusion):
            raise NotFound("document not found")
        kind, external_id = SourceKind(doc["source_kind"]), str(doc["external_id"])
        ref = {"source_kind": kind.value, "external_id": external_id}
        if forget_exclusion:
            restored = await _restore_source_item(tx, store, jobs, kind, external_id)
            await audit(
                conn, key, "document.forget_exclusion", str(document_id), {**ref, **restored}
            )
            return Response(status_code=204)
        await store.lock_source_item(tx, kind, external_id)
        copies = await fetch_all(
            conn, _ITEM_DOCUMENTS, {"kind": kind.value, "external_id": external_id}
        )
        live = [d["id"] for d in copies if d["deleted_at"] is None]
        if document_id not in live:
            raise NotFound("document not found")  # deleted by a concurrent request
        await store.add_exclusion(tx, kind, external_id, "deleted via API")
        await execute(conn, _EXCLUDE_COPIES, {"ids": live})
        await audit(conn, key, "document.delete", str(document_id), {**ref, "copies": len(live)})
    return Response(status_code=204)


# ----------------------------------------------------------------- exclusions
@router.get("/exclusions", tags=["exclusions"])
async def list_exclusions(
    key: ReadKey, txm: TxManager, cursor: CursorParam = None, limit: Limit = DEFAULT_LIMIT
) -> Page[Exclusion]:
    after, params = time_id_after(cursor)
    async with txm.tenant(key.principal.tenant_id) as tx:
        rows = await fetch_all(
            connection(tx),
            f"SELECT {EXCLUSION_COLUMNS} FROM rf.exclusions WHERE true{after} "  # noqa: S608
            "ORDER BY created_at DESC, id DESC LIMIT :n",
            {**params, "n": limit + 1},
        )
    return page(rows, limit, lambda r: Exclusion.model_validate(dict(r)))


@router.delete("/exclusions/{exclusion_id}", status_code=204, tags=["exclusions"])
async def delete_exclusion(
    exclusion_id: UUID, key: WriteKey, txm: TxManager, store: SyncStores, jobs: Jobs
) -> Response:
    """Forget an exclusion and restore the item, like `DELETE /documents/{id}?forget_exclusion`."""
    async with txm.tenant(key.principal.tenant_id) as tx:
        conn = connection(tx)
        row = await fetch_one(
            conn,
            "SELECT source_kind, external_id FROM rf.exclusions WHERE id = :id",
            {"id": exclusion_id},
        )
        if row is None:
            raise NotFound("exclusion not found")
        kind, external_id = SourceKind(row["source_kind"]), str(row["external_id"])
        restored = await _restore_source_item(tx, store, jobs, kind, external_id)
        await audit(
            conn,
            key,
            "exclusion.delete",
            str(exclusion_id),
            {"source_kind": kind.value, "external_id": external_id, **restored},
        )
    return Response(status_code=204)


# ----------------------------------------------------------------- restore (undo delete)
async def _upload_row(conn: AsyncConnection, external_id: str) -> RowMapping | None:
    """A file's external_id is its upload id (handoff C2); anything else has no upload."""
    try:
        upload_id = UUID(external_id)
    except ValueError:
        return None
    return await fetch_one(conn, "SELECT id FROM rf.uploads WHERE id = :id", {"id": upload_id})


async def _restore_source_item(
    tx: Tx, store: SyncStore, jobs: JobQueue, kind: SourceKind, external_id: str
) -> dict[str, object]:
    """Remove the exclusion of (kind, external_id) and bring the item back, all or nothing.

    Lock order matches sync's commit and DELETE /documents: source-item advisory lock, then the
    document rows. Only copies tombstoned BECAUSE of the exclusion (deleted_reason 'excluded',
    slot 0009) are revived; a copy that sync/reconcile confirmed deleted at the source stays
    deleted (undoing a user delete must not resurrect what the owner deleted in Notion).

    - notion: nothing is revived here. Notion is the source of truth and the page may have been
      deleted, unshared or redacted while it was excluded, so each excluded copy only gets a
      `sync_page` (its connector's own page job, interactive). A successful fetch clears the
      'excluded' tombstone whatever its fetched_at (the sticky rule is for 'source_deleted'
      only); a definitive "gone" turns it into 'source_deleted'. The page is back within
      seconds, once the job runs. No copy left (all purged): one `sync_page` per active notion
      connector of the tenant (the page id is global; a connector that cannot see the page
      records nothing).
    - file: each excluded copy is un-tombstoned and `ingest_file` is re-enqueued from the upload,
      which re-embeds anything GC removed. A file document cannot outlive its upload row
      (documents.upload_id is ON DELETE CASCADE), so no upload row means nothing to revive:
      409, and the exclusion is kept.

    Returns identifiers for the audit row.
    """
    conn = connection(tx)
    await store.lock_source_item(tx, kind, external_id)
    exists = await fetch_one(
        conn,
        "SELECT 1 FROM rf.exclusions WHERE source_kind = :kind AND external_id = :external_id",
        {"kind": kind.value, "external_id": external_id},
    )
    if exists is None:
        raise NotFound("exclusion not found")
    docs = await fetch_all(conn, _ITEM_DOCUMENTS, {"kind": kind.value, "external_id": external_id})
    excluded = [d for d in docs if d["deleted_reason"] == "excluded"]
    kept_deleted = sum(1 for d in docs if d["deleted_reason"] == "source_deleted")
    jobs_enqueued = 0
    revived: list[str] = []

    if kind is SourceKind.file:
        upload = await _upload_row(conn, external_id)
        if upload is None:
            raise Conflict("nothing left to restore: the upload was purged")
        for d in excluded:
            await execute(conn, _REVIVE, {"id": d["id"]})
            revived.append(str(d["id"]))
        await jobs.enqueue(
            tx,
            JobSpec(
                kind=KIND_INGEST_FILE,
                priority=JobPriority.interactive,
                payload={"upload_id": str(upload["id"])},
                dedupe_key=f"{KIND_INGEST_FILE}:{upload['id']}",
            ),
        )
        jobs_enqueued += 1
    else:
        connectors = {d["connector_id"] for d in excluded if d["connector_id"] is not None}
        if not docs:  # every copy was purged: ask each active connector of this kind
            rows = await fetch_all(
                conn,
                "SELECT id FROM rf.connectors WHERE kind = :kind AND status = 'active'",
                {"kind": kind.value},
            )
            connectors = {r["id"] for r in rows}
        for connector_id in sorted(connectors):
            await jobs.enqueue(
                tx,
                JobSpec(
                    kind=KIND_SYNC_PAGE,
                    priority=JobPriority.interactive,
                    payload=sync_page_payload(connector_id, external_id),
                    dedupe_key=sync_page_key(connector_id, external_id),
                    connector_id=connector_id,
                ),
            )
            jobs_enqueued += 1

    await execute(
        conn,
        "DELETE FROM rf.exclusions WHERE source_kind = :kind AND external_id = :external_id",
        {"kind": kind.value, "external_id": external_id},
    )
    return {
        "revived_documents": revived,
        "jobs_enqueued": jobs_enqueued,
        "kept_source_deleted": kept_deleted,
    }


# ----------------------------------------------------------------- audit
@router.get("/audit", tags=["audit"])
async def list_audit(
    key: KeyManagerKey, txm: TxManager, cursor: CursorParam = None, limit: Limit = DEFAULT_LIMIT
) -> Page[AuditEntry]:
    """The explicit tenant predicate (redundant with RLS) lets the planner walk the
    (tenant_id, id DESC) index from migration 0006 instead of the whole table's primary key."""
    after, params = int_id_after(cursor)
    params["tenant"] = key.principal.tenant_id
    async with txm.tenant(key.principal.tenant_id) as tx:
        rows = await fetch_all(
            connection(tx),
            "SELECT id, actor, action, target, details, created_at FROM rf.audit_log "  # noqa: S608
            f"WHERE tenant_id = :tenant{after} ORDER BY id DESC LIMIT :n",
            {**params, "n": limit + 1},
        )
    return page(rows, limit, lambda r: AuditEntry.model_validate(dict(r)), int_id_cursor)
