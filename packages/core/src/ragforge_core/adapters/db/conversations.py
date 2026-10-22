"""ConversationRuns adapter: the conversation run lease + pending-approval bookkeeping (module D).

The lease is `conversations.active_run_id/run_expires_at` (DESIGN section 3 Flow B, section 8):

- `acquire` is a CAS UPDATE in the CALLER's tenant transaction, so "take the lease" composes with
  "lock/expire the run's approvals" atomically (resume and new-turn are serialized: both acquire
  first, then lock pending_approvals FOR UPDATE in the same transaction).
- `heartbeat`/`release` open their OWN short tenant transaction: they run from a background task
  while the turn streams, and must never extend the caller's transaction.

Every lease comparison uses `clock_timestamp()`: `now()` is frozen at transaction start, so a
caller whose transaction began before the lease expired would wrongly see it as live (or set an
expiry in the past). All statements run under RLS (tenant in the transaction), so a foreign
conversation is simply invisible.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from datetime import timedelta
from typing import Final
from uuid import UUID, uuid4

from sqlalchemy import RowMapping, text

from ragforge_core.adapters.db.tx import connection
from ragforge_core.domain.errors import NotFound, TurnInProgress
from ragforge_core.domain.models import PendingApproval, ToolCall
from ragforge_core.domain.ports import TransactionManager, Tx

DEFAULT_LEASE_TTL: Final = timedelta(seconds=60)
"""Heartbeat every 15 s (DESIGN section 8) against a 60 s lease: three missed beats to lose it."""

_ACQUIRE = text(
    """
    UPDATE rf.conversations
       SET active_run_id = :run_id,
           run_expires_at = clock_timestamp() + make_interval(secs => :ttl)
     WHERE id = :conversation_id
       AND (active_run_id IS NULL OR run_expires_at <= clock_timestamp())
    RETURNING id
    """
)
_EXISTS = text("SELECT 1 FROM rf.conversations WHERE id = :conversation_id")
_HEARTBEAT = text("SELECT rf.heartbeat_run(:conversation_id, :run_id, :ttl)")
_RELEASE = text(
    """
    UPDATE rf.conversations
       SET active_run_id = NULL, run_expires_at = NULL
     WHERE id = :conversation_id AND active_run_id = :run_id
    """
)
_LOCK_PENDING = text(
    """
    SELECT id, conversation_id, run_id, tool_call_id, tool, args, status
      FROM rf.pending_approvals
     WHERE run_id = :run_id
     ORDER BY created_at, id
       FOR UPDATE
    """
)
_EXPIRE_PENDING = text(
    """
    UPDATE rf.pending_approvals
       SET status = 'expired', decided_at = clock_timestamp()
     WHERE conversation_id = :conversation_id AND status = 'pending'
    RETURNING id, tenant_id, run_id, tool_call_id, tool
    """
)
_CHAT_ACTIVE = text("SELECT rf.chat_active()")
_AUDIT = text(
    """
    INSERT INTO rf.audit_log (tenant_id, actor, action, target, details)
    VALUES (:tenant_id, 'agent', 'approval.expired', :target, CAST(:details AS jsonb))
    """
)


class SqlConversationRuns:
    """Implements `domain.ports.ConversationRuns` with SQL under RLS."""

    def __init__(
        self, transactions: TransactionManager, *, lease_ttl: timedelta = DEFAULT_LEASE_TTL
    ) -> None:
        ttl = int(lease_ttl.total_seconds())
        if not 5 <= ttl <= 600:  # the bounds rf.heartbeat_run enforces
            raise ValueError("lease_ttl must be between 5 and 600 seconds")
        self._transactions = transactions
        self._ttl = ttl

    async def acquire(self, tx: Tx, conversation_id: UUID) -> UUID:
        """CAS on active_run_id. Under READ COMMITTED a concurrent acquirer blocks on the row lock
        and then re-evaluates the WHERE against the committed winner, so exactly one wins.

        Raises NotFound when the conversation is not visible in this tenant (rather than a
        misleading 409), TurnInProgress when another live run holds the lease.
        """
        conn = connection(tx)
        run_id = uuid4()
        won = await conn.execute(
            _ACQUIRE, {"run_id": run_id, "ttl": self._ttl, "conversation_id": conversation_id}
        )
        if won.first() is not None:
            return run_id
        exists = await conn.execute(_EXISTS, {"conversation_id": conversation_id})
        if exists.first() is None:
            raise NotFound("conversation not found")
        raise TurnInProgress()

    async def heartbeat(self, tenant_id: UUID, conversation_id: UUID, run_id: UUID) -> bool:
        """Delegates to rf.heartbeat_run so the API and the LangGraph path share one lease rule;
        with the tenant set, the function only ever touches this tenant's row."""
        async with self._transactions.tenant(tenant_id) as tx:
            result = await connection(tx).execute(
                _HEARTBEAT,
                {"conversation_id": conversation_id, "run_id": run_id, "ttl": self._ttl},
            )
            return bool(result.scalar_one())

    async def release(self, tenant_id: UUID, conversation_id: UUID, run_id: UUID) -> None:
        """Idempotent; a run that already lost the lease never clears its successor's."""
        async with self._transactions.tenant(tenant_id) as tx:
            await connection(tx).execute(
                _RELEASE, {"conversation_id": conversation_id, "run_id": run_id}
            )

    async def lock_pending(self, tx: Tx, run_id: UUID) -> Sequence[PendingApproval]:
        """All of the run's approvals (any status), row-locked until the caller commits. The
        caller decides: resume proceeds only if they are all still `pending`."""
        result = await connection(tx).execute(_LOCK_PENDING, {"run_id": run_id})
        return [_to_pending(row) for row in result.mappings()]

    async def expire_pending(self, tx: Tx, conversation_id: UUID) -> int:
        """Expire every pending approval of the conversation and audit each one, in the caller's
        transaction (after `acquire`), so an approval can never execute on a moved-on thread."""
        conn = connection(tx)
        expired = (
            await conn.execute(_EXPIRE_PENDING, {"conversation_id": conversation_id})
        ).mappings()
        rows = list(expired)
        for row in rows:
            await conn.execute(_AUDIT, _audit_params(conversation_id, row))
        return len(rows)


def _to_pending(row: RowMapping) -> PendingApproval:
    args = row["args"]
    return PendingApproval(
        id=row["id"],
        conversation_id=row["conversation_id"],
        run_id=row["run_id"],
        tool_call=ToolCall(
            id=row["tool_call_id"],
            name=row["tool"],
            arguments=dict(args) if isinstance(args, Mapping) else {},
        ),
        status=row["status"],
    )


def _audit_params(conversation_id: UUID, row: RowMapping) -> dict[str, object]:
    """Identifiers only: tool arguments may hold user content and stay out of the audit log."""
    details = {
        "conversation_id": str(conversation_id),
        "run_id": str(row["run_id"]),
        "tool": row["tool"],
        "tool_call_id": row["tool_call_id"],
        "reason": "new_turn",
    }
    return {
        "tenant_id": row["tenant_id"],
        "target": f"pending_approval:{row['id']}",
        "details": json.dumps(details),
    }


async def chat_active(tx: Tx) -> bool:
    """Worker throttle (DESIGN section 3 Flow B): true while any conversation, in any tenant,
    holds a live run lease. Runs in a system transaction (rf.chat_active is granted to rf_worker
    only) and reveals one boolean, never which tenant is chatting."""
    result = await connection(tx).execute(_CHAT_ACTIVE)
    return bool(result.scalar_one())
