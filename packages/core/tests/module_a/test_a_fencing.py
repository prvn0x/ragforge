"""Zombie fencing (ADR-0010): a worker whose lease was reclaimed can never commit a result.

The side-effect transaction starts with assert_lease, which row-locks the job; reclaim takes the
same lock and SKIPs locked rows. So either the holder commits while it still holds the lease, or
its transaction fails and commits nothing. There is no third interleaving.
"""

from __future__ import annotations

import asyncio
from uuid import UUID

import psycopg
import pytest
from sqlalchemy import text

from module_a.a_support import enqueue, job_row, spec
from ragforge_core.adapters.db.jobs import PgJobQueue
from ragforge_core.db.engine import SqlTransactionManager
from ragforge_core.domain.errors import LeaseLost
from ragforge_core.domain.models import JobLease
from ragforge_core.domain.ports import Tx

pytestmark = pytest.mark.integration
Conn = psycopg.Connection[tuple[object, ...]]

_AUDIT = text(
    "INSERT INTO rf.audit_log (tenant_id, actor, action, target) "
    "VALUES (:tenant, 'test', 'side-effect', :target)"
)


async def _side_effect(tx: Tx, lease: JobLease, marker: str) -> None:
    from ragforge_core.adapters.db.tx import connection

    await connection(tx).execute(_AUDIT, {"tenant": lease.tenant_id, "target": marker})


async def _result_tx(
    txm: SqlTransactionManager,
    queue: PgJobQueue,
    lease: JobLease,
    marker: str,
    *,
    complete: bool = True,
) -> None:
    """A handler's fenced result transaction: assert_lease -> side effect -> complete."""
    async with txm.tenant(lease.tenant_id) as tx:
        await queue.assert_lease(tx, lease)
        await _side_effect(tx, lease, marker)
        if complete:
            await queue.complete(tx, lease)


def _side_effects(superuser: Conn, marker: str) -> int:
    row = superuser.execute(
        "SELECT count(*) FROM rf.audit_log WHERE target = %s", (marker,)
    ).fetchone()
    assert row is not None
    return int(str(row[0]))


def _expire(superuser: Conn, job_id: UUID) -> None:
    superuser.execute(
        "UPDATE rf.jobs SET lease_expires_at = now() - interval '1 second' WHERE id = %s",
        (job_id,),
    )


async def test_reclaimed_zombie_commits_nothing(
    queue: PgJobQueue, txm: SqlTransactionManager, tenant: UUID, superuser: Conn
) -> None:
    job_id = await enqueue(txm, queue, tenant, spec("zombie"))
    (zombie,) = await queue.claim("worker-a", 1)
    _expire(superuser, job_id)  # worker-a paused (GC, network partition...) past its lease
    assert await queue.reclaim_expired() == 1
    # From slot 0008 on, a reclaimed job is backed off (a worker-killing job must not hit the
    # next replica at once); let that backoff elapse so worker-b can claim it now.
    superuser.execute("UPDATE rf.jobs SET run_after = now() WHERE id = %s", (job_id,))
    (fresh,) = await queue.claim("worker-b", 1)
    # +1 at slot 0002; +2 from slot 0008 on (reclaim bumps the version too). Either way: newer.
    assert fresh.job_id == job_id
    assert fresh.lease_version > zombie.lease_version

    # worker-a wakes up and runs its result transaction: side effect + complete.
    with pytest.raises(LeaseLost):
        await _result_tx(txm, queue, zombie, "zombie-write")
    assert _side_effects(superuser, "zombie-write") == 0
    assert await queue.heartbeat(zombie) is False
    assert job_row(superuser, job_id, "status, lease_version, leased_by") == (
        "running",
        fresh.lease_version,
        "worker-b",
    )

    # The legitimate holder still completes normally.
    await _result_tx(txm, queue, fresh, "fresh-write")
    assert _side_effects(superuser, "fresh-write") == 1
    assert job_row(superuser, job_id) == ("done",)


async def test_zombie_fenced_even_before_another_worker_claims(
    queue: PgJobQueue, txm: SqlTransactionManager, tenant: UUID, superuser: Conn
) -> None:
    job_id = await enqueue(txm, queue, tenant, spec("zombie-queued"))
    (zombie,) = await queue.claim("worker-a", 1)
    _expire(superuser, job_id)
    assert await queue.reclaim_expired() == 1  # back to queued, nobody holds it yet
    with pytest.raises(LeaseLost):
        await _result_tx(txm, queue, zombie, "zombie-queued-write", complete=False)
    assert _side_effects(superuser, "zombie-queued-write") == 0
    assert job_row(superuser, job_id, "status, last_error") == (
        "queued",
        "lease expired (worker lost)",
    )


async def test_reclaim_cannot_interleave_with_a_fenced_transaction(
    queue: PgJobQueue, txm: SqlTransactionManager, tenant: UUID, superuser: Conn
) -> None:
    """The lease expired but was not reclaimed yet, and the holder enters its fenced transaction
    first: reclaim skips the locked row, so the holder (still the only owner) commits and nobody
    else ever ran the job."""
    job_id = await enqueue(txm, queue, tenant, spec("in-flight"))
    (lease,) = await queue.claim("worker-a", 1)
    _expire(superuser, job_id)
    async with txm.tenant(tenant) as tx:
        await queue.assert_lease(tx, lease)  # holds the row lock from here to commit
        reclaimed = await asyncio.wait_for(queue.reclaim_expired(), timeout=10)
        assert reclaimed == 0  # row locked by the holder -> skipped, not blocked
        await _side_effect(tx, lease, "holder-write")
        await queue.complete(tx, lease)
    assert _side_effects(superuser, "holder-write") == 1
    assert await queue.reclaim_expired() == 0
    assert job_row(superuser, job_id, "status, lease_version") == ("done", 1)
