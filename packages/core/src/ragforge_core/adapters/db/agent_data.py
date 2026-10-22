"""SQL adapters for the agent's module-local ports (`agent/ports.py`, handoff D1 request 1).

`SqlConversationStore` implements `ConversationStore` and `SqlKnowledgeBase` implements
`KnowledgeBase`, structurally: adapters may not import `agent` (import-linter), so the port
signatures use only domain types and primitives.

Every method runs in the CALLER's tenant transaction, as rf_api, under RLS: the tenant is
ambient (`rf.current_tenant()`), never a parameter, so nothing a model emits can widen it. Rows
handed to the model (`list_sources`, `sync_status`) are JSON-safe and never include a
connector's `config` or secret columns.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any
from uuid import UUID

from sqlalchemy import RowMapping, text

from ragforge_core.adapters.connectors.store import SqlSyncStore
from ragforge_core.adapters.db.tx import connection
from ragforge_core.domain.models import ApprovalChoice, RankedChunk, SourceKind, ToolCall
from ragforge_core.domain.ports import Tx

# ================================================================ conversations
_CREATE = text(
    "INSERT INTO rf.conversations (tenant_id, user_id, collection_ids) "
    "VALUES (rf.current_tenant(), :user_id, CAST(:collections AS uuid[])) RETURNING id"
)
_GET = text("SELECT user_id, collection_ids FROM rf.conversations WHERE id = :id")
_FOR_RUN = text(  # no row lock: the lease (conversation row) is taken first, then approvals
    "SELECT conversation_id FROM rf.pending_approvals WHERE run_id = :run_id LIMIT 1"
)
_RECORD = text(
    """
    INSERT INTO rf.pending_approvals (tenant_id, conversation_id, run_id, tool_call_id, tool, args)
    VALUES (rf.current_tenant(), :conversation_id, :run_id, :tool_call_id, :tool,
            CAST(:args AS jsonb))
    ON CONFLICT (conversation_id, tool_call_id) DO NOTHING
    """
)
_DECIDE = text(
    """
    UPDATE rf.pending_approvals
       SET status = :status, decided_at = clock_timestamp()
     WHERE run_id = :run_id AND tool_call_id = :tool_call_id AND status = 'pending'
    RETURNING id, conversation_id, tool
    """
)
_CLAIM = text(
    """
    UPDATE rf.pending_approvals
       SET executed_at = clock_timestamp()
     WHERE conversation_id = :conversation_id AND tool_call_id = :tool_call_id
       AND status = 'approved' AND executed_at IS NULL
    RETURNING 1
    """
)
_SUMMARY = text(
    "SELECT summary FROM rf.conversation_summaries WHERE conversation_id = :conversation_id"
)
_AUDIT = text(
    "INSERT INTO rf.audit_log (tenant_id, actor, action, target, details) "
    "VALUES (rf.current_tenant(), :actor, :action, :target, CAST(:details AS jsonb))"
)


class SqlConversationStore:
    """Implements `agent.ports.ConversationStore`."""

    async def create(self, tx: Tx, user_id: UUID, collection_ids: Sequence[UUID]) -> UUID:
        row = (
            await connection(tx).execute(
                _CREATE, {"user_id": user_id, "collections": list(collection_ids)}
            )
        ).one()
        conversation_id: UUID = row[0]
        return conversation_id

    async def get(self, tx: Tx, conversation_id: UUID) -> tuple[UUID, tuple[UUID, ...]] | None:
        row = (await connection(tx).execute(_GET, {"id": conversation_id})).one_or_none()
        if row is None:
            return None
        return row[0], tuple(row[1])

    async def conversation_for_run(self, tx: Tx, run_id: UUID) -> UUID | None:
        value = (await connection(tx).execute(_FOR_RUN, {"run_id": run_id})).scalar_one_or_none()
        return value if isinstance(value, UUID) else None

    async def record_pending(
        self, tx: Tx, conversation_id: UUID, run_id: UUID, calls: Sequence[ToolCall]
    ) -> None:
        conn = connection(tx)
        for call in calls:
            await conn.execute(
                _RECORD,
                {
                    "conversation_id": conversation_id,
                    "run_id": run_id,
                    "tool_call_id": call.id,
                    "tool": call.name,
                    "args": json.dumps(dict(call.arguments)),
                },
            )

    async def decide_pending(
        self, tx: Tx, run_id: UUID, decisions: Mapping[str, ApprovalChoice]
    ) -> None:
        """Only `pending` rows change, so re-applying a recorded decision is a no-op. One audit
        row per decided call, identifiers only (tool arguments may hold user content)."""
        conn = connection(tx)
        for tool_call_id, choice in decisions.items():
            status = "approved" if choice == ApprovalChoice.approve else "rejected"
            decided = (
                await conn.execute(
                    _DECIDE, {"status": status, "run_id": run_id, "tool_call_id": tool_call_id}
                )
            ).one_or_none()
            if decided is None:
                continue
            await conn.execute(
                _AUDIT,
                {
                    # The decision is a human's; the port carries no Principal, so the actor names
                    # the path. The API's `approvals.submit` row records the deciding key.
                    "actor": "user-approval",
                    "action": f"approval.{status}",
                    "target": f"pending_approval:{decided.id}",
                    "details": json.dumps(
                        {
                            "conversation_id": str(decided.conversation_id),
                            "run_id": str(run_id),
                            "tool": decided.tool,
                            "tool_call_id": tool_call_id,
                        }
                    ),
                },
            )

    async def claim_execution(self, tx: Tx, conversation_id: UUID, tool_call_id: str) -> bool:
        result = await connection(tx).execute(
            _CLAIM, {"conversation_id": conversation_id, "tool_call_id": tool_call_id}
        )
        return result.first() is not None

    async def summary(self, tx: Tx, conversation_id: UUID) -> str | None:
        value = (
            await connection(tx).execute(_SUMMARY, {"conversation_id": conversation_id})
        ).scalar_one_or_none()
        return value if isinstance(value, str) else None


# ================================================================ knowledge base
_TAUS = text("SELECT id, tau FROM rf.collections WHERE id = ANY(CAST(:ids AS uuid[]))")
_PAGE = text(
    """
    SELECT c.id, c.document_id, c.collection_id, c.anchor, d.title, c.heading_path,
           c.display_text, c.display_sha256, d.external_url
      FROM rf.documents d
      JOIN rf.chunks c ON c.document_id = d.id AND c.tenant_id = d.tenant_id
     WHERE d.id = :document_id
       AND d.collection_id = ANY(CAST(:collections AS uuid[]))
       AND d.deleted_at IS NULL
     ORDER BY c.ordinal
     LIMIT :limit
    """
)
# rf_api has no privilege on rf.sync_cursors (0001 grants), so "last synced" is not offered.
_CONNECTORS = text(
    """
    SELECT c.id, c.kind, c.display_name, c.target_collection_id, c.status, c.last_error,
           c.poll_failures, c.next_poll_at, c.created_at
      FROM rf.connectors c
     WHERE c.target_collection_id = ANY(CAST(:collections AS uuid[]))
     ORDER BY c.created_at, c.id
     LIMIT 100
    """
)
_UPLOADS = text(
    """
    SELECT u.id, u.filename, u.collection_id, u.size_bytes, u.created_at
      FROM rf.uploads u
     WHERE u.collection_id = ANY(CAST(:collections AS uuid[]))
       -- A deleted upload keeps its row and blob (undo restores from them), and the purge
       -- removes only its document after 7 days, so the exclusion is the durable signal.
       AND NOT EXISTS (SELECT 1 FROM rf.exclusions x
                        WHERE x.source_kind = 'file' AND x.external_id = u.id::text)
       AND NOT EXISTS (SELECT 1 FROM rf.documents d
                        WHERE d.upload_id = u.id AND d.deleted_at IS NOT NULL)
     ORDER BY u.created_at DESC, u.id
     LIMIT 100
    """
)
_DOC_FOR_REMOVAL = """
    SELECT source_kind, external_id
      FROM rf.documents
     WHERE id = :document_id AND collection_id = ANY(CAST(:collections AS uuid[]))
       AND deleted_at IS NULL
"""
_DOC_REF = text(_DOC_FOR_REMOVAL)
# Every copy of the item in the tenant, row-locked in id order after the advisory lock.
_ITEM_COPIES = text(
    "SELECT id, deleted_at FROM rf.documents "
    "WHERE source_kind = :kind AND external_id = :external_id ORDER BY id FOR UPDATE"
)
_EXCLUDE_COPIES = text(
    "UPDATE rf.documents SET deleted_at = now(), deleted_reason = 'excluded', updated_at = now() "
    "WHERE id = ANY(CAST(:ids AS uuid[])) AND deleted_at IS NULL"
)


def _json_safe(row: RowMapping) -> dict[str, object]:
    out: dict[str, object] = {}
    for key, value in row.items():
        if isinstance(value, UUID):
            out[key] = str(value)
        elif isinstance(value, datetime):
            out[key] = value.isoformat()
        else:
            out[key] = value
    return out


class SqlKnowledgeBase:
    """Implements `agent.ports.KnowledgeBase`."""

    def __init__(self, sync_store: SqlSyncStore | None = None) -> None:
        self._sync_store = sync_store or SqlSyncStore()

    async def collection_taus(
        self, tx: Tx, collection_ids: Sequence[UUID]
    ) -> Mapping[UUID, float | None]:
        rows = await connection(tx).execute(_TAUS, {"ids": list(collection_ids)})
        return {row.id: (None if row.tau is None else float(row.tau)) for row in rows}

    async def get_page(
        self, tx: Tx, document_id: UUID, collection_ids: Sequence[UUID], limit: int
    ) -> Sequence[RankedChunk]:
        rows = await connection(tx).execute(
            _PAGE,
            {"document_id": document_id, "collections": list(collection_ids), "limit": limit},
        )
        return [
            RankedChunk(
                chunk_id=r[0],
                document_id=r[1],
                collection_id=r[2],
                anchor=r[3],
                title=r[4],
                heading_path=tuple(r[5] or ()),
                display_text=r[6],
                display_sha256=bytes(r[7]),
                external_url=r[8],
                fused_score=0.0,
            )
            for r in rows
        ]

    async def list_sources(
        self, tx: Tx, collection_ids: Sequence[UUID]
    ) -> Sequence[Mapping[str, object]]:
        conn = connection(tx)
        params = {"collections": list(collection_ids)}
        connectors = [
            {"type": "connector", **_json_safe(r)}
            for r in (await conn.execute(_CONNECTORS, params)).mappings()
        ]
        uploads = [
            {"type": "upload", **_json_safe(r)}
            for r in (await conn.execute(_UPLOADS, params)).mappings()
        ]
        keep = ("type", "id", "kind", "display_name", "filename", "status", "created_at")
        return [{k: row[k] for k in keep if k in row} for row in (*connectors, *uploads)]

    async def sync_status(
        self, tx: Tx, collection_ids: Sequence[UUID]
    ) -> Sequence[Mapping[str, object]]:
        """`poll_failures` + `last_error` make a backed-off connector visible: after failed
        listings slot 0008 pushes `next_poll_at` out (up to a day) and records the error."""
        rows = await connection(tx).execute(_CONNECTORS, {"collections": list(collection_ids)})
        keep = ("id", "display_name", "status", "last_error", "poll_failures", "next_poll_at")
        out = []
        for row in rows.mappings():
            safe = _json_safe(row)
            out.append({k: safe[k] for k in keep})
        return out

    async def remove_from_index(
        self, tx: Tx, document_id: UUID, collection_ids: Sequence[UUID], actor: str
    ) -> bool:
        """Exclusion + tombstone + audit in the caller's tx (the one holding the execution
        claim). Lock order matches sync's fenced commit (services/sync/indexing.py): the
        source-item advisory lock FIRST, then the document rows, so a removal racing a page
        sync serializes on the advisory lock instead of deadlocking on a row.

        The document must be visible in the conversation's collections, but the exclusion it
        writes is tenant-wide (unique on tenant, source_kind, external_id), so every live copy
        of the item is tombstoned too ('excluded', slot 0009); a copy left live would be frozen
        (every later sync skips it as excluded). The audit row records `copies`."""
        conn = connection(tx)
        params = {"document_id": document_id, "collections": list(collection_ids)}
        ref = (await conn.execute(_DOC_REF, params)).one_or_none()
        if ref is None:
            return False
        kind = SourceKind(ref.source_kind)
        await self._sync_store.lock_source_item(tx, kind, ref.external_id)
        copies = (
            await conn.execute(_ITEM_COPIES, {"kind": kind.value, "external_id": ref.external_id})
        ).all()
        live = [c.id for c in copies if c.deleted_at is None]
        if document_id not in live:
            return False  # tombstoned by a concurrent removal while we waited
        await self._sync_store.add_exclusion(tx, kind, ref.external_id, "removed by agent")
        await conn.execute(_EXCLUDE_COPIES, {"ids": live})
        details: dict[str, Any] = {
            "source_kind": kind.value,
            "external_id": ref.external_id,
            "copies": len(live),
        }
        await conn.execute(
            _AUDIT,
            {
                "actor": actor,
                "action": "document.removed",
                "target": str(document_id),
                "details": json.dumps(details),
            },
        )
        return True
