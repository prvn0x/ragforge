"""SQL implementation of the sync service's module-local `SyncStore` port (module C2).

It satisfies `ragforge_core.services.sync.ports.SyncStore` structurally: adapters may not import
services (import-linter), so the row types below are local dataclasses with the same read-only
attributes as the port's Protocols.

Every method runs in the caller's tenant transaction. Tenant isolation is RLS (fail closed:
without app.tenant_id no row is visible and every insert fails its WITH CHECK); inserts take
tenant_id from rf.current_tenant(), never from a parameter.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

from ragforge_core.adapters.db.tx import connection
from ragforge_core.domain.errors import NotFound, ValidationFailed
from ragforge_core.domain.models import SealedSecret, SourceKind
from ragforge_core.domain.ports import Tx

_MAX_ERROR_CHARS = 500
_FK_VIOLATION = "23503"


@dataclass(frozen=True, slots=True)
class ConnectorRow:
    id: UUID
    kind: str
    target_collection_id: UUID
    config: Mapping[str, Any]
    status: str
    sealed: SealedSecret | None = field(repr=False)  # never in logs/reprs


@dataclass(frozen=True, slots=True)
class UploadRow:
    id: UUID
    collection_id: UUID
    blob_key: str
    filename: str
    sha256: bytes
    size_bytes: int
    created_at: datetime


@dataclass(frozen=True, slots=True)
class KnownDocumentRow:
    id: UUID
    external_id: str
    fetched_at: datetime | None
    synced_source_ids: Sequence[str]


def _cursor_text(value: datetime) -> str:
    """Cursors are text in the schema; store a fixed-offset ISO-8601 so the SQL comparison
    (cast to timestamptz) is exact and independent of the session TimeZone."""
    if value.tzinfo is None:
        raise ValueError("cursor must be timezone-aware")
    return value.astimezone(UTC).isoformat()


class SqlSyncStore:
    # ------------------------------------------------------------------ connectors
    async def load_connector(self, tx: Tx, connector_id: UUID) -> ConnectorRow | None:
        row = (
            await connection(tx).execute(
                text(
                    "SELECT id, kind, target_collection_id, config, status, "
                    "secret_ciphertext, wrapped_dek FROM rf.connectors WHERE id = :id"
                ),
                {"id": connector_id},
            )
        ).first()
        if row is None:
            return None
        sealed = (
            SealedSecret(
                ciphertext=bytes(row.secret_ciphertext), wrapped_dek=bytes(row.wrapped_dek)
            )
            if row.secret_ciphertext is not None and row.wrapped_dek is not None
            else None
        )
        return ConnectorRow(
            id=row.id,
            kind=row.kind,
            target_collection_id=row.target_collection_id,
            config=dict(row.config or {}),
            status=row.status,
            sealed=sealed,
        )

    async def mark_connector_synced(self, tx: Tx, connector_id: UUID, next_poll_in_s: int) -> None:
        await connection(tx).execute(
            text(
                "UPDATE rf.connectors SET last_error = NULL, "
                "next_poll_at = now() + make_interval(secs => :s) "
                "WHERE id = :id AND status = 'active'"
            ),
            {"id": connector_id, "s": max(0, int(next_poll_in_s))},
        )

    async def mark_connector_error(self, tx: Tx, connector_id: UUID, message: str) -> None:
        await connection(tx).execute(
            text(
                "UPDATE rf.connectors SET status = 'error', last_error = :m "
                "WHERE id = :id AND status = 'active'"
            ),
            {"id": connector_id, "m": message[:_MAX_ERROR_CHARS]},
        )

    async def destroy_secret(self, tx: Tx, connector_id: UUID) -> None:
        await connection(tx).execute(
            text(
                "UPDATE rf.connectors SET secret_ciphertext = NULL, wrapped_dek = NULL, "
                "status = 'deleting' WHERE id = :id"
            ),
            {"id": connector_id},
        )

    async def delete_connector_documents(self, tx: Tx, connector_id: UUID, limit: int) -> int:
        result = await connection(tx).execute(
            text(
                "DELETE FROM rf.documents WHERE id IN ("
                " SELECT id FROM rf.documents WHERE connector_id = :id LIMIT :n)"
            ),
            {"id": connector_id, "n": max(1, int(limit))},
        )
        return int(result.rowcount)

    async def delete_connector(self, tx: Tx, connector_id: UUID) -> bool:
        result = await connection(tx).execute(
            text("DELETE FROM rf.connectors WHERE id = :id"), {"id": connector_id}
        )
        return bool(result.rowcount)

    # ------------------------------------------------------------------ cursors
    async def read_cursor(self, tx: Tx, connector_id: UUID, scope: str) -> datetime | None:
        value = (
            await connection(tx).execute(
                text("SELECT cursor FROM rf.sync_cursors WHERE connector_id = :id AND scope = :s"),
                {"id": connector_id, "s": scope},
            )
        ).scalar_one_or_none()
        return None if value is None else datetime.fromisoformat(value)

    async def advance_cursor(
        self, tx: Tx, connector_id: UUID, scope: str, value: datetime
    ) -> datetime:
        """GREATEST semantics in one statement: concurrent advances (a manual sync racing the
        scheduled one) serialize on the row and the larger value always wins."""
        stored = (
            await connection(tx).execute(
                text(
                    "INSERT INTO rf.sync_cursors (connector_id, tenant_id, scope, cursor) "
                    "VALUES (:id, rf.current_tenant(), :s, :v) "
                    "ON CONFLICT (connector_id, scope) DO UPDATE SET "
                    " cursor = CASE WHEN excluded.cursor::timestamptz"
                    "   > rf.sync_cursors.cursor::timestamptz"
                    "   THEN excluded.cursor ELSE rf.sync_cursors.cursor END,"
                    " updated_at = now() "
                    "RETURNING cursor"
                ),
                {"id": connector_id, "s": scope, "v": _cursor_text(value)},
            )
        ).scalar_one()
        return datetime.fromisoformat(stored)

    # ------------------------------------------------------------------ exclusions
    async def excluded_ids(
        self, tx: Tx, source_kind: SourceKind, external_ids: Sequence[str]
    ) -> set[str]:
        if not external_ids:
            return set()
        rows = await connection(tx).execute(
            text(
                "SELECT external_id FROM rf.exclusions "
                "WHERE source_kind = :k AND external_id = ANY(CAST(:ids AS text[]))"
            ),
            {"k": source_kind.value, "ids": list(external_ids)},
        )
        return {r[0] for r in rows}

    async def lock_source_item(self, tx: Tx, source_kind: SourceKind, external_id: str) -> None:
        """pg_advisory_xact_lock on a 64-bit hash of (tenant, kind, external_id). Released at
        commit/rollback; a hash collision only serializes two unrelated items (harmless)."""
        await connection(tx).execute(
            text(
                "SELECT pg_advisory_xact_lock(hashtextextended("
                "'rf.source:' || rf.current_tenant()::text || ':' || :k || ':' || :e, 0))"
            ),
            {"k": source_kind.value, "e": external_id},
        )

    async def add_exclusion(
        self, tx: Tx, source_kind: SourceKind, external_id: str, reason: str | None
    ) -> None:
        await self.lock_source_item(tx, source_kind, external_id)
        await connection(tx).execute(
            text(
                "INSERT INTO rf.exclusions (tenant_id, source_kind, external_id, reason) "
                "VALUES (rf.current_tenant(), :k, :e, :r) "
                "ON CONFLICT (tenant_id, source_kind, external_id) DO NOTHING"
            ),
            {"k": source_kind.value, "e": external_id, "r": reason},
        )

    # ------------------------------------------------------------------ documents (read-only)
    async def fetched_at_by_external_id(
        self, tx: Tx, connector_id: UUID, external_ids: Sequence[str]
    ) -> dict[str, datetime | None]:
        if not external_ids:
            return {}
        rows = await connection(tx).execute(
            text(
                "SELECT external_id, fetched_at FROM rf.documents "
                "WHERE connector_id = :id AND deleted_at IS NULL "
                "AND external_id = ANY(CAST(:ids AS text[]))"
            ),
            {"id": connector_id, "ids": list(external_ids)},
        )
        return {r.external_id: r.fetched_at for r in rows}

    async def live_documents(self, tx: Tx, connector_id: UUID) -> list[KnownDocumentRow]:
        rows = await connection(tx).execute(
            text(
                "SELECT id, external_id, fetched_at, synced_source_ids FROM rf.documents "
                "WHERE connector_id = :id AND deleted_at IS NULL ORDER BY external_id"
            ),
            {"id": connector_id},
        )
        return [
            KnownDocumentRow(
                id=r.id,
                external_id=r.external_id,
                fetched_at=r.fetched_at,
                synced_source_ids=tuple(r.synced_source_ids or ()),
            )
            for r in rows
        ]

    # ------------------------------------------------------------------ uploads
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
        if len(sha256) != 32:
            raise ValidationFailed("sha256 must be a raw 32-byte digest")
        conn = connection(tx)
        try:
            # Savepoint: a FK violation must not poison the caller's transaction.
            async with conn.begin_nested():
                await conn.execute(
                    text(
                        "INSERT INTO rf.uploads "
                        "(id, tenant_id, collection_id, blob_key, filename, sha256, size_bytes) "
                        "VALUES (:id, rf.current_tenant(), :c, :k, :f, :h, :n)"
                    ),
                    {
                        "id": upload_id,
                        "c": collection_id,
                        "k": blob_key,
                        "f": filename,
                        "h": sha256,
                        "n": size_bytes,
                    },
                )
        except IntegrityError as exc:
            if getattr(exc.orig, "sqlstate", None) == _FK_VIOLATION:
                # The composite FK includes tenant_id, so another tenant's collection id lands
                # here too: indistinguishable from "does not exist" (no existence oracle).
                raise NotFound("collection not found") from None
            raise

    async def load_upload(self, tx: Tx, upload_id: UUID) -> UploadRow | None:
        row = (
            await connection(tx).execute(
                text(
                    "SELECT id, collection_id, blob_key, filename, sha256, size_bytes, created_at "
                    "FROM rf.uploads WHERE id = :id"
                ),
                {"id": upload_id},
            )
        ).first()
        if row is None:
            return None
        return UploadRow(
            id=row.id,
            collection_id=row.collection_id,
            blob_key=row.blob_key,
            filename=row.filename,
            sha256=bytes(row.sha256),
            size_bytes=int(row.size_bytes),
            created_at=row.created_at,
        )

    # ------------------------------------------------------------------ audit
    async def audit(
        self, tx: Tx, *, actor: str, action: str, target: str | None, details: Mapping[str, Any]
    ) -> None:
        await connection(tx).execute(
            text(
                "INSERT INTO rf.audit_log (tenant_id, actor, action, target, details) "
                "VALUES (rf.current_tenant(), :a, :act, :t, CAST(:d AS jsonb))"
            ),
            {"a": actor, "act": action, "t": target, "d": _json(details)},
        )


def _json(details: Mapping[str, Any]) -> str:
    return json.dumps(dict(details), default=str, sort_keys=True)
