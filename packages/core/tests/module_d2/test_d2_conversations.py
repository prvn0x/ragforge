"""SqlConversationRuns: one turn per conversation, expiry takeover, approvals serialization and
tenant isolation (DESIGN section 3 Flow B, section 8 HITL)."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from datetime import timedelta
from uuid import UUID, uuid4

import psycopg
import pytest
from psycopg import errors

from ragforge_core.adapters.db.conversations import SqlConversationRuns
from ragforge_core.db.engine import SqlTransactionManager
from ragforge_core.domain.errors import NotFound, TurnInProgress
from ragforge_core.domain.ports import ConversationRuns

pytestmark = pytest.mark.integration
Conn = psycopg.Connection[tuple[object, ...]]
MakeConversation = Callable[[], tuple[UUID, UUID]]
ExpireLease = Callable[[UUID], None]


def _lease(superuser: Conn, conversation: UUID) -> tuple[object, ...] | None:
    return superuser.execute(
        "SELECT active_run_id, run_expires_at IS NOT NULL FROM rf.conversations WHERE id = %s",
        (conversation,),
    ).fetchone()


def _add_approval(
    superuser: Conn, tenant: UUID, conversation: UUID, run: UUID, status: str = "pending"
) -> UUID:
    approval = uuid4()
    superuser.execute(
        "INSERT INTO rf.pending_approvals "
        "(id, tenant_id, conversation_id, run_id, tool_call_id, tool, args, status) "
        "VALUES (%s, %s, %s, %s, %s, 'forget_fact', %s, %s)",
        (approval, tenant, conversation, run, uuid4().hex, json.dumps({"q": "x"}), status),
    )
    return approval


async def _acquire(
    transactions: SqlTransactionManager, runs: SqlConversationRuns, tenant: UUID, conv: UUID
) -> UUID:
    async with transactions.tenant(tenant) as tx:
        return await runs.acquire(tx, conv)


def test_adapter_satisfies_the_port(runs: SqlConversationRuns) -> None:
    assert isinstance(runs, ConversationRuns)


def test_lease_ttl_is_bounded(transactions: SqlTransactionManager) -> None:
    with pytest.raises(ValueError, match="lease_ttl"):
        SqlConversationRuns(transactions, lease_ttl=timedelta(hours=1))


# ---------------------------------------------------------------------- one turn at a time
async def test_acquire_then_second_acquire_is_turn_in_progress(
    transactions: SqlTransactionManager,
    runs: SqlConversationRuns,
    superuser: Conn,
    make_conversation: MakeConversation,
) -> None:
    tenant, conversation = make_conversation()
    run = await _acquire(transactions, runs, tenant, conversation)
    assert _lease(superuser, conversation) == (run, True)
    with pytest.raises(TurnInProgress):
        await _acquire(transactions, runs, tenant, conversation)
    assert _lease(superuser, conversation) == (run, True)


async def test_concurrent_acquires_have_exactly_one_winner(
    transactions: SqlTransactionManager,
    runs: SqlConversationRuns,
    superuser: Conn,
    make_conversation: MakeConversation,
) -> None:
    """Racers whose transactions all overlap (each holds its tx open after the CAS): the row
    lock + re-check under READ COMMITTED must let exactly one through."""
    tenant, conversation = make_conversation()
    racers = 4  # below the engine pool size, so every racer really holds a connection
    start = asyncio.Barrier(racers)

    async def racer() -> UUID | None:
        async with transactions.tenant(tenant) as tx:
            await start.wait()
            try:
                run = await runs.acquire(tx, conversation)
            except TurnInProgress:
                return None
            await asyncio.sleep(0.2)  # keep the winner's tx open while the others wait
            return run

    results = await asyncio.gather(*(racer() for _ in range(racers)))
    winners = [r for r in results if r is not None]
    assert len(winners) == 1
    assert _lease(superuser, conversation) == (winners[0], True)


async def test_acquire_unknown_conversation_is_not_found(
    transactions: SqlTransactionManager,
    runs: SqlConversationRuns,
    make_conversation: MakeConversation,
) -> None:
    tenant, _ = make_conversation()
    with pytest.raises(NotFound):
        await _acquire(transactions, runs, tenant, uuid4())


async def test_acquire_rolls_back_with_the_callers_transaction(
    transactions: SqlTransactionManager,
    runs: SqlConversationRuns,
    superuser: Conn,
    make_conversation: MakeConversation,
) -> None:
    tenant, conversation = make_conversation()

    async def fail_after_acquire() -> None:
        async with transactions.tenant(tenant) as tx:
            await runs.acquire(tx, conversation)
            raise RuntimeError("caller failed after acquiring")

    with pytest.raises(RuntimeError):
        await fail_after_acquire()
    assert _lease(superuser, conversation) == (None, False)


# ---------------------------------------------------------------------- expiry + takeover
async def test_expired_lease_is_taken_over_and_the_old_run_is_fenced_out(
    transactions: SqlTransactionManager,
    runs: SqlConversationRuns,
    superuser: Conn,
    make_conversation: MakeConversation,
    expire_lease: ExpireLease,
) -> None:
    tenant, conversation = make_conversation()
    old = await _acquire(transactions, runs, tenant, conversation)
    assert await runs.heartbeat(tenant, conversation, old) is True
    expire_lease(conversation)
    assert await runs.heartbeat(tenant, conversation, old) is False, "expired: never revived"

    new = await _acquire(transactions, runs, tenant, conversation)
    assert new != old
    assert await runs.heartbeat(tenant, conversation, old) is False
    await runs.release(tenant, conversation, old)  # stale release must not clear the successor
    assert _lease(superuser, conversation) == (new, True)
    assert await runs.heartbeat(tenant, conversation, new) is True


async def test_release_frees_the_conversation_and_is_idempotent(
    transactions: SqlTransactionManager,
    runs: SqlConversationRuns,
    superuser: Conn,
    make_conversation: MakeConversation,
) -> None:
    tenant, conversation = make_conversation()
    run = await _acquire(transactions, runs, tenant, conversation)
    await runs.release(tenant, conversation, run)
    await runs.release(tenant, conversation, run)
    assert _lease(superuser, conversation) == (None, False)
    assert await runs.heartbeat(tenant, conversation, run) is False
    await _acquire(transactions, runs, tenant, conversation)


# ---------------------------------------------------------------------- approvals
async def test_lock_pending_locks_the_runs_rows_until_commit(
    transactions: SqlTransactionManager,
    runs: SqlConversationRuns,
    superuser: Conn,
    make_conversation: MakeConversation,
) -> None:
    tenant, conversation = make_conversation()
    run = uuid4()
    first = _add_approval(superuser, tenant, conversation, run)
    second = _add_approval(superuser, tenant, conversation, run, status="approved")
    other = _add_approval(superuser, tenant, conversation, uuid4())
    async with transactions.tenant(tenant) as tx:
        locked = await runs.lock_pending(tx, run)
        assert [p.id for p in locked] == [first, second]
        assert [p.status for p in locked] == ["pending", "approved"]
        assert locked[0].tool_call.name == "forget_fact"
        assert locked[0].tool_call.arguments == {"q": "x"}
        with pytest.raises(errors.LockNotAvailable):
            superuser.execute(
                "SELECT 1 FROM rf.pending_approvals WHERE id = %s FOR UPDATE NOWAIT", (first,)
            )
        superuser.execute(  # another run's rows are not locked
            "SELECT 1 FROM rf.pending_approvals WHERE id = %s FOR UPDATE NOWAIT", (other,)
        )
    superuser.execute(
        "SELECT 1 FROM rf.pending_approvals WHERE id = %s FOR UPDATE NOWAIT", (first,)
    )


async def test_new_turn_expires_pending_approvals_and_audits_them(
    transactions: SqlTransactionManager,
    runs: SqlConversationRuns,
    superuser: Conn,
    make_conversation: MakeConversation,
) -> None:
    tenant, conversation = make_conversation()
    old_run = uuid4()
    pending = _add_approval(superuser, tenant, conversation, old_run)
    approved = _add_approval(superuser, tenant, conversation, old_run, status="approved")
    async with transactions.tenant(tenant) as tx:
        await runs.acquire(tx, conversation)
        assert await runs.expire_pending(tx, conversation) == 1
        assert await runs.expire_pending(tx, conversation) == 0
    rows: dict[object, object] = dict(
        superuser.execute(  # type: ignore[arg-type]
            "SELECT id, status FROM rf.pending_approvals WHERE conversation_id = %s",
            (conversation,),
        ).fetchall()
    )
    assert rows == {pending: "expired", approved: "approved"}
    audit = superuser.execute(
        "SELECT actor, action, target, details FROM rf.audit_log WHERE tenant_id = %s",
        (tenant,),
    ).fetchall()
    assert len(audit) == 1
    actor, action, target, details = audit[0]
    assert (actor, action, target) == ("agent", "approval.expired", f"pending_approval:{pending}")
    assert isinstance(details, dict)
    assert details["run_id"] == str(old_run)
    assert "args" not in details, "tool arguments may hold user content"


async def test_resume_and_new_turn_are_serialized(
    transactions: SqlTransactionManager,
    runs: SqlConversationRuns,
    superuser: Conn,
    make_conversation: MakeConversation,
) -> None:
    """A new turn that won the lease and expired the approvals blocks a concurrent resume's
    lock_pending until it commits; the resume then sees `expired` and must not proceed."""
    tenant, conversation = make_conversation()
    old_run = uuid4()
    _add_approval(superuser, tenant, conversation, old_run)
    expired = asyncio.Event()

    async def new_turn() -> None:
        async with transactions.tenant(tenant) as tx:
            await runs.acquire(tx, conversation)
            await runs.expire_pending(tx, conversation)
            expired.set()
            await asyncio.sleep(0.2)

    async def resume() -> list[str]:
        await expired.wait()
        async with transactions.tenant(tenant) as tx:
            return [p.status for p in await runs.lock_pending(tx, old_run)]

    _, statuses = await asyncio.gather(new_turn(), resume())
    assert statuses == ["expired"]


# ---------------------------------------------------------------------- tenant isolation
async def test_other_tenant_cannot_touch_the_lease_or_approvals(
    transactions: SqlTransactionManager,
    runs: SqlConversationRuns,
    superuser: Conn,
    make_conversation: MakeConversation,
) -> None:
    tenant_a, conversation = make_conversation()
    tenant_b, _ = make_conversation()
    run = await _acquire(transactions, runs, tenant_a, conversation)
    _add_approval(superuser, tenant_a, conversation, run)

    with pytest.raises(NotFound):  # invisible, not "in progress": no existence oracle
        await _acquire(transactions, runs, tenant_b, conversation)
    assert await runs.heartbeat(tenant_b, conversation, run) is False
    await runs.release(tenant_b, conversation, run)
    async with transactions.tenant(tenant_b) as tx:
        assert await runs.lock_pending(tx, run) == []
        assert await runs.expire_pending(tx, conversation) == 0

    assert _lease(superuser, conversation) == (run, True)
    assert superuser.execute(
        "SELECT status FROM rf.pending_approvals WHERE run_id = %s", (run,)
    ).fetchall() == [("pending",)]
