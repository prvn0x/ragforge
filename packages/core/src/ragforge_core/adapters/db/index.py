"""DocumentIndex (domain port) on Postgres. Every method runs in the caller's tenant `tx`.

Sync flow (ports.DocumentIndex): step 1 `target_spaces` + `touch_and_find_missing` in a short
tx; step 2 embed with NO tx; step 3 the fenced tx runs `insert_embeddings` -> `upsert_document`
-> `replace_chunks`. The tenant is never a parameter: rows are written with rf.current_tenant()
and RLS WITH CHECK rejects anything else.

Concurrency notes (why the SQL looks the way it does):
  * upsert_document locks the row (SELECT ... FOR UPDATE) before deciding, so two sync_page
    runs for one page serialise; the later-committing one sees the earlier one's fetched_at and
    the monotonic guard COALESCE(fetched_at, '-infinity') < new decides, in either commit
    order. A first insert races via ON CONFLICT DO NOTHING + one retry of the locked path.
  * Hash arrays are de-duplicated and sorted, so concurrent syncs touching overlapping hashes
    take row locks in the same order (no deadlock), and ON CONFLICT DO UPDATE never sees the
    same key twice in one statement.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection

from ragforge_core.adapters.db.search import vector_literal
from ragforge_core.adapters.db.tx import connection
from ragforge_core.domain import ids
from ragforge_core.domain.errors import NotFound
from ragforge_core.domain.models import Chunk, DocumentRecord, UpsertOutcome
from ragforge_core.domain.ports import Tx
from ragforge_core.observability.logging import get_logger

log = get_logger(__name__)

SHA256_LEN = 32

_TARGET_SPACES = text(
    "SELECT active_space_id, building_space_id FROM rf.collections "
    "WHERE id = :collection AND tenant_id = rf.current_tenant()"
)
_TOUCH = text(
    """
    INSERT INTO rf.embedding_touches (tenant_id, collection_id, space_id, embed_sha256, touched_at)
    SELECT rf.current_tenant(), :collection, :space, h.sha, now()
      FROM unnest(CAST(:hashes AS bytea[])) AS h(sha)
     ORDER BY h.sha
    ON CONFLICT (tenant_id, collection_id, space_id, embed_sha256)
    DO UPDATE SET touched_at = excluded.touched_at
    """
)
_MISSING = text(
    """
    SELECT h.sha FROM unnest(CAST(:hashes AS bytea[])) AS h(sha)
     WHERE NOT EXISTS (
           SELECT 1 FROM rf.embeddings e
            WHERE e.tenant_id = rf.current_tenant() AND e.collection_id = :collection
              AND e.space_id = :space AND e.embed_sha256 = h.sha)
    """
)
_SPACE = text("SELECT dim, status FROM rf.embedding_spaces WHERE id = :space")
_INSERT_EMBEDDINGS = text(
    """
    INSERT INTO rf.embeddings (tenant_id, collection_id, space_id, embed_sha256, embedding)
    SELECT rf.current_tenant(), :collection, :space, u.sha, CAST(u.vec AS ext.vector)
      FROM unnest(CAST(:hashes AS bytea[]), CAST(:vectors AS text[])) AS u(sha, vec)
     ORDER BY u.sha
    ON CONFLICT DO NOTHING
    """
)
_BUMP_STATS = text(
    """
    INSERT INTO rf.collection_space_stats (tenant_id, collection_id, space_id, vector_count)
    VALUES (rf.current_tenant(), :collection, :space, :n)
    ON CONFLICT (collection_id, space_id) DO UPDATE
       SET vector_count = rf.collection_space_stats.vector_count + excluded.vector_count,
           updated_at = now()
    """
)
_CURRENT_TENANT = text("SELECT rf.current_tenant()")
_LOCK_DOCUMENT = text(
    "SELECT fetched_at, page_sha256, deleted_reason FROM rf.documents "
    "WHERE id = :id AND tenant_id = rf.current_tenant() FOR UPDATE"
)
_INSERT_DOCUMENT = text(
    """
    INSERT INTO rf.documents (id, tenant_id, collection_id, connector_id, upload_id, source_kind,
                              external_id, external_url, title, external_updated_at, fetched_at,
                              page_sha256, synced_source_ids, metadata)
    VALUES (:id, rf.current_tenant(), :collection_id, :connector_id, :upload_id, :source_kind,
            :external_id, :external_url, :title, :external_updated_at, :fetched_at,
            :page_sha256, CAST(:synced_source_ids AS text[]), CAST(:metadata AS jsonb))
    ON CONFLICT DO NOTHING
    RETURNING id
    """
)
# The fetched_at guard is repeated in both UPDATEs even though the row is locked: it is the
# documented invariant (never move fetched_at backwards), so it must hold by construction.
#
# Tombstones confirmed at the source (deleted_reason = 'source_deleted') are sticky against older
# fetches: a sync revives such a page ONLY if its fetch started after the tombstone. Otherwise a
# sync_page that fetched the page before reconcile confirmed it deleted/unshared, but committed
# after, would make the owner's deleted content searchable again (and reset the 7-day hard
# delete). fetched_at is the source's timestamp minus a safety margin, so ties and clock skew err
# toward staying deleted.
# An 'excluded' tombstone is different: sync only reaches this upsert for an item that is NOT
# excluded (it checks under the source-item lock), so the exclusion was forgotten and the user
# asked for the item back; the timing of the fetch is irrelevant (slot 0009). The trigger on
# rf.documents clears deleted_reason whenever deleted_at becomes NULL. (Same CASE in both.)
_UPDATE_DOCUMENT = text(
    """
    UPDATE rf.documents
       SET external_url = :external_url, title = :title,
           external_updated_at = :external_updated_at, fetched_at = :fetched_at,
           page_sha256 = :page_sha256, synced_source_ids = CAST(:synced_source_ids AS text[]),
           metadata = CAST(:metadata AS jsonb), upload_id = :upload_id,
           deleted_at = CASE WHEN deleted_reason = 'source_deleted' AND deleted_at >= :fetched_at
                             THEN deleted_at ELSE NULL END,
           updated_at = now()
     WHERE id = :id AND COALESCE(fetched_at, '-infinity') < :fetched_at
    """
)
_BUMP_FETCHED = text(
    """
    UPDATE rf.documents
       SET fetched_at = :fetched_at,
           deleted_at = CASE WHEN deleted_reason = 'source_deleted' AND deleted_at >= :fetched_at
                             THEN deleted_at ELSE NULL END,
           updated_at = now()
     WHERE id = :id AND COALESCE(fetched_at, '-infinity') < :fetched_at
    """
)
_DELETE_LINKS = text("DELETE FROM rf.document_links WHERE from_document_id = :id")
_INSERT_LINKS = text(
    """
    INSERT INTO rf.document_links (tenant_id, from_document_id, to_external_id)
    SELECT rf.current_tenant(), :id, l FROM unnest(CAST(:links AS text[])) AS l
    ON CONFLICT DO NOTHING
    """
)
_DOCUMENT_COLLECTION = text(
    "SELECT collection_id FROM rf.documents WHERE id = :id AND tenant_id = rf.current_tenant()"
)
_UPSERT_CHUNK = text(
    """
    INSERT INTO rf.chunks (id, tenant_id, collection_id, document_id, anchor, ordinal,
                           heading_path, display_text, display_sha256, embed_sha256,
                           token_count, metadata)
    VALUES (:id, rf.current_tenant(), :collection_id, :document_id, :anchor, :ordinal,
            CAST(:heading_path AS text[]), :display_text, :display_sha256, :embed_sha256,
            :token_count, CAST(:metadata AS jsonb))
    ON CONFLICT (id) DO UPDATE
       SET anchor = excluded.anchor, heading_path = excluded.heading_path,
           display_text = excluded.display_text, display_sha256 = excluded.display_sha256,
           embed_sha256 = excluded.embed_sha256, token_count = excluded.token_count,
           metadata = excluded.metadata
     WHERE (rf.chunks.anchor, rf.chunks.heading_path, rf.chunks.display_text,
            rf.chunks.embed_sha256, rf.chunks.token_count, rf.chunks.metadata)
           IS DISTINCT FROM
           (excluded.anchor, excluded.heading_path, excluded.display_text,
            excluded.embed_sha256, excluded.token_count, excluded.metadata)
    """
)
_TRIM_CHUNKS = text("DELETE FROM rf.chunks WHERE document_id = :id AND ordinal >= :n")
_TOMBSTONE = text(
    # An 'excluded' tombstone is upgraded too: the source has now CONFIRMED the deletion, so an
    # undo of the exclusion must not revive it. deleted_at is reset to the CONFIRMATION time:
    # the sticky rule compares later fetches against it, and keeping the older exclusion time
    # would let a fetch that started after the exclusion but before the source deletion (an
    # undo's refresh that commits late) revive owner-deleted content. Cost: the 7-day purge
    # clock restarts at the confirmation, which only delays the hard delete.
    "UPDATE rf.documents SET deleted_at = now(), "
    "deleted_reason = 'source_deleted', updated_at = now() "
    "WHERE id = :id AND (deleted_at IS NULL OR deleted_reason = 'excluded')"
)
# Undo of an exclusion when the stored fetch is as new as (or newer than) this one: the content
# on file is already current, so only the tombstone is lifted (slot 0009 trigger clears reason).
_REVIVE_EXCLUDED = text(
    "UPDATE rf.documents SET deleted_at = NULL, updated_at = now() "
    "WHERE id = :id AND deleted_reason = 'excluded'"
)
_IS_EXCLUDED = text(
    """
    SELECT EXISTS (SELECT 1 FROM rf.exclusions
                    WHERE tenant_id = rf.current_tenant() AND source_kind = :kind
                      AND external_id = :external_id)
    """
)


class PgDocumentIndex:
    """Implements the `DocumentIndex` port. Stateless; share one instance."""

    async def target_spaces(self, tx: Tx, collection_id: UUID) -> Sequence[int]:
        row = (await connection(tx).execute(_TARGET_SPACES, {"collection": collection_id})).first()
        if row is None:
            raise NotFound(f"collection {collection_id} not found")
        active, building = row
        return [int(active)] if building is None else [int(active), int(building)]

    async def touch_and_find_missing(
        self, tx: Tx, collection_id: UUID, space_id: int, embed_sha256s: Sequence[bytes]
    ) -> set[bytes]:
        hashes = _sorted_hashes(embed_sha256s)
        if not hashes:
            return set()
        conn = connection(tx)
        params = {"collection": collection_id, "space": int(space_id), "hashes": hashes}
        await conn.execute(_TOUCH, params)
        # A separate statement, so it gets a fresh snapshot AFTER the touch upsert (which may
        # have waited on a concurrent gc_embeddings that just deleted the embedding).
        result = await conn.execute(_MISSING, params)
        return {bytes(row[0]) for row in result}

    async def insert_embeddings(
        self,
        tx: Tx,
        collection_id: UUID,
        space_id: int,
        vectors: Mapping[bytes, Sequence[float]],
    ) -> int:
        if not vectors:
            return 0
        conn = connection(tx)
        dim = await _writable_space_dim(conn, space_id)
        hashes = _sorted_hashes(list(vectors))
        literals = []
        for sha in hashes:
            vector = vectors[sha]
            if len(vector) != dim:
                # jobs.last_error is user-visible: the space id and sizes go to the log only.
                log.error(
                    "embedding_dimension_mismatch",
                    space_id=int(space_id),
                    expected=dim,
                    got=len(vector),
                )
                raise ValueError("embedding has the wrong number of dimensions for its space")
            literals.append(vector_literal(vector))
        params = {"collection": collection_id, "space": int(space_id)}
        result = await conn.execute(
            _INSERT_EMBEDDINGS, {**params, "hashes": hashes, "vectors": literals}
        )
        inserted = max(int(result.rowcount or 0), 0)
        if inserted:
            await conn.execute(_BUMP_STATS, {**params, "n": inserted})
        return inserted

    async def upsert_document(self, tx: Tx, record: DocumentRecord) -> UpsertOutcome:
        conn = connection(tx)
        await _check_document_id(conn, record)
        _check_sha(record.page_sha256, "page_sha256")
        for _attempt in range(2):
            locked = (await conn.execute(_LOCK_DOCUMENT, {"id": record.id})).first()
            if locked is not None:
                return await _update_locked(conn, record, locked[0], locked[1], locked[2])
            inserted = (await conn.execute(_INSERT_DOCUMENT, _document_params(record))).first()
            if inserted is not None:
                await _replace_links(conn, record)
                return UpsertOutcome.inserted
            # A concurrent transaction inserted the row first; it is committed now (ON CONFLICT
            # waited for it), so the locked path sees it on the retry.
        raise RuntimeError(f"document {record.id} vanished during upsert")  # pragma: no cover

    async def replace_chunks(
        self, tx: Tx, document_id: UUID, collection_id: UUID, chunks: Sequence[Chunk]
    ) -> None:
        conn = connection(tx)
        row = (await conn.execute(_DOCUMENT_COLLECTION, {"id": document_id})).first()
        if row is None:
            raise NotFound(f"document {document_id} not found")
        if row[0] != collection_id:
            raise ValueError("document belongs to a different collection")
        _check_ordinals(chunks)
        if chunks:
            await conn.execute(
                _UPSERT_CHUNK, [_chunk_params(document_id, collection_id, c) for c in chunks]
            )
        await conn.execute(_TRIM_CHUNKS, {"id": document_id, "n": len(chunks)})

    async def tombstone(self, tx: Tx, document_id: UUID) -> None:
        """Soft delete confirmed at the source (deleted_reason 'source_deleted'), also over an
        'excluded' tombstone (whose deleted_at is reset to now: see _TOMBSTONE). Search skips it
        at once and C's purge_tombstones hard-deletes it 7 days after the confirmation. Only a
        fetch that STARTED after the confirmation (fetched_at > deleted_at) un-tombstones it; a
        sync that fetched earlier and commits later leaves it deleted."""
        await connection(tx).execute(_TOMBSTONE, {"id": document_id})

    async def is_excluded(self, tx: Tx, source_kind: str, external_id: str) -> bool:
        result = await connection(tx).execute(
            _IS_EXCLUDED, {"kind": str(source_kind), "external_id": external_id}
        )
        return bool(result.scalar())


# ---------------------------------------------------------------------- helpers
async def _update_locked(
    conn: AsyncConnection,
    record: DocumentRecord,
    stored_fetched_at: datetime | None,
    stored_sha: bytes | None,
    deleted_reason: str | None,
) -> UpsertOutcome:
    if stored_fetched_at is not None and stored_fetched_at >= record.fetched_at:
        if deleted_reason == "excluded":
            # Sync only reaches the upsert for an item that is no longer excluded (it re-checks
            # under the source-item lock), so this is an undo whose refresh fetched no newer
            # copy than the one on file (e.g. the same Date-header second). The stored content
            # is current; lift the tombstone instead of discarding the undo as stale.
            await conn.execute(_REVIVE_EXCLUDED, {"id": record.id})
            return UpsertOutcome.unchanged
        return UpsertOutcome.stale
    if stored_sha is not None and bytes(stored_sha) == record.page_sha256:
        await conn.execute(_BUMP_FETCHED, {"id": record.id, "fetched_at": record.fetched_at})
        return UpsertOutcome.unchanged
    await conn.execute(_UPDATE_DOCUMENT, _document_params(record))
    await _replace_links(conn, record)
    return UpsertOutcome.updated


async def _replace_links(conn: AsyncConnection, record: DocumentRecord) -> None:
    await conn.execute(_DELETE_LINKS, {"id": record.id})
    links = sorted(set(record.links_to))
    if links:
        await conn.execute(_INSERT_LINKS, {"id": record.id, "links": links})


async def _check_document_id(conn: AsyncConnection, record: DocumentRecord) -> None:
    """record.id must be THE id domain.ids derives for this tenant; anything else would break
    deep links and restores (ids are frozen)."""
    tenant = (await conn.execute(_CURRENT_TENANT)).scalar()
    if tenant is None:
        raise RuntimeError("upsert_document requires a tenant transaction")
    expected = ids.document_id(
        tenant, record.collection_id, record.connector_id, record.external_id
    )
    if record.id != expected:
        raise ValueError("DocumentRecord.id does not match domain.ids.document_id(...)")


async def _writable_space_dim(conn: AsyncConnection, space_id: int) -> int:
    row = (await conn.execute(_SPACE, {"space": int(space_id)})).first()
    if row is None:
        log.error("embedding_space_missing", space_id=int(space_id))
        raise NotFound("embedding space not found")
    if row[1] == "retired":
        log.warning("embedding_space_retired", space_id=int(space_id))
        raise ValueError("embedding space is retired")
    return int(row[0])


def _document_params(record: DocumentRecord) -> dict[str, Any]:
    if record.fetched_at.tzinfo is None:
        raise ValueError("fetched_at must be timezone-aware")
    return {
        "id": record.id,
        "collection_id": record.collection_id,
        "connector_id": record.connector_id,
        "upload_id": record.upload_id,
        "source_kind": str(record.source_kind),
        "external_id": record.external_id,
        "external_url": record.external_url,
        "title": record.title,
        "external_updated_at": record.external_updated_at,
        "fetched_at": record.fetched_at,
        "page_sha256": record.page_sha256,
        "synced_source_ids": list(record.synced_source_ids),
        "metadata": _jsonb(record.metadata),
    }


def _chunk_params(document_id: UUID, collection_id: UUID, chunk: Chunk) -> dict[str, Any]:
    _check_sha(chunk.display_sha256, "display_sha256")
    _check_sha(chunk.embed_sha256, "embed_sha256")
    return {
        "id": ids.chunk_id(document_id, chunk.ordinal),
        "collection_id": collection_id,
        "document_id": document_id,
        "anchor": chunk.anchor,
        "ordinal": chunk.ordinal,
        "heading_path": list(chunk.heading_path),
        "display_text": chunk.display_text,
        "display_sha256": chunk.display_sha256,
        "embed_sha256": chunk.embed_sha256,
        "token_count": chunk.token_count,
        "metadata": _jsonb(chunk.metadata),
    }


def _jsonb(value: Mapping[str, Any]) -> str:
    """jsonb rejects NaN/Infinity; fail here with a clear error instead of in Postgres."""
    return json.dumps(value, sort_keys=True, default=str, allow_nan=False)


def _check_ordinals(chunks: Sequence[Chunk]) -> None:
    if sorted(c.ordinal for c in chunks) != list(range(len(chunks))):
        raise ValueError("chunk ordinals must be exactly 0..n-1")


def _check_sha(value: bytes, name: str) -> None:
    if not isinstance(value, bytes) or len(value) != SHA256_LEN:
        raise ValueError(f"{name} must be a raw 32-byte sha256 digest")


def _sorted_hashes(hashes: Sequence[bytes]) -> list[bytes]:
    for sha in hashes:
        _check_sha(sha, "embed_sha256")
    return sorted(set(hashes))
