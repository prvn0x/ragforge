"""Concurrency guarantees of the queue (ADR-0010), exercised with real parallel transactions.

- N concurrent claimers: every job is claimed exactly once.
- Requeue merge: release/fail racing concurrent enqueues of the same key never surfaces a
  unique violation and never leaves two queued rows (or zero live rows) for the key.
- The "duplicate vanished" branch of requeue's merge loop, forced deterministically.
- Per-connector in-flight cap: never exceeded under racing claimers; the counter matches the
  number of running jobs after every exit path.
"""

from __future__ import annotations

import asyncio
import random
import threading
import time
from collections import Counter
from collections.abc import Callable
from typing import cast
from uuid import UUID

import psycopg
import pytest

from module_a.a_support import enqueue, job_row, spec
from pgtest import connect
from ragforge_core.adapters.db.jobs import PgJobQueue
from ragforge_core.config import Settings
from ragforge_core.db.engine import SqlTransactionManager
from ragforge_core.domain.models import JobLease, JobPriority

pytestmark = pytest.mark.integration
Conn = psycopg.Connection[tuple[object, ...]]
MakeTenant = Callable[..., tuple[UUID, UUID, UUID]]


async def _complete(txm: SqlTransactionManager, queue: PgJobQueue, lease: JobLease) -> None:
    async with txm.tenant(lease.tenant_id) as tx:
        await queue.assert_lease(tx, lease)
        await queue.complete(tx, lease)


# ------------------------------------------------------------------ exactly-once claim
async def test_concurrent_workers_claim_each_job_exactly_once(
    queue: PgJobQueue, txm: SqlTransactionManager, make_tenant: object, superuser: Conn
) -> None:
    tenants = [make_tenant() for _ in range(3)]  # type: ignore[operator]
    priorities = list(JobPriority)
    expected: set[UUID] = set()
    for i in range(240):
        tenant = tenants[i % 3][0]
        expected.add(
            await enqueue(txm, queue, tenant, spec(f"job-{i}", priorities[i % len(priorities)]))
        )

    claimed: list[UUID] = []

    async def worker(name: str) -> None:
        rng = random.Random(name)  # noqa: S311 - test jitter, not crypto
        while True:
            leases = await queue.claim(name, rng.randint(1, 6), prefer_low=rng.random() < 0.25)
            if not leases:
                return
            claimed.extend(lease.job_id for lease in leases)
            await asyncio.sleep(0)

    await asyncio.gather(*(worker(f"w{i}") for i in range(12)))
    duplicates = [job for job, n in Counter(claimed).items() if n > 1]
    assert duplicates == []
    assert set(claimed) == expected
    assert superuser.execute(
        "SELECT count(*) FROM rf.jobs WHERE status = 'running' AND lease_version = 1"
    ).fetchone() == (240,)


# ------------------------------------------------------------------ requeue merge
async def test_requeue_merge_under_concurrent_enqueue(
    queue: PgJobQueue, txm: SqlTransactionManager, tenant: UUID, superuser: Conn
) -> None:
    for round_ in range(25):
        key = f"page-{round_}"
        await enqueue(txm, queue, tenant, spec(key, JobPriority.backfill))
        (lease,) = await queue.claim("w0", 1)

        async def give_back(lease: JobLease = lease, round_: int = round_) -> None:
            if round_ % 2:
                await queue.release(lease)
            else:
                await queue.fail(lease, "boom", 0.0)

        async def racer(key: str = key) -> None:
            await enqueue(txm, queue, tenant, spec(key, JobPriority.webhook))

        async def claimer() -> None:
            await queue.claim("w1", 1)

        racers = [racer() for _ in range(4)] + [give_back(), claimer()]
        random.shuffle(racers)
        await asyncio.gather(*racers)  # any unique_violation would surface here

        rows = superuser.execute(
            "SELECT status, count(*) FROM rf.jobs WHERE tenant_id = %s AND dedupe_key = %s "
            "GROUP BY status",
            (tenant, key),
        ).fetchall()
        by_status = {str(s): int(str(n)) for s, n in rows}
        assert by_status.get("queued", 0) <= 1, by_status
        # No lost work: the key is still queued or being worked on.
        assert by_status.get("queued", 0) + by_status.get("running", 0) >= 1, by_status
        superuser.execute("DELETE FROM rf.jobs")


def test_requeue_retries_when_the_duplicate_is_claimed_mid_merge(
    settings: Settings, superuser: Conn, make_tenant: object
) -> None:
    """Force the third branch of requeue's loop: unique_violation -> the duplicate is claimed
    before the merge UPDATE -> loop -> plain requeue succeeds. Two real sessions, ordered with
    a row lock, so the interleaving is deterministic."""
    tenant = make_tenant()[0]  # type: ignore[operator]
    with connect(settings, "worker") as w:
        w.autocommit = True
        w.execute("SELECT set_config('app.tenant_id', %s, false)", (str(tenant),))
        w.execute("SELECT rf.enqueue_job('k', 20, '{}', 'key', NULL, NULL)")
        ((mine, version),) = w.execute(
            "SELECT job_id, lease_version FROM rf.claim_jobs('w', 1, false)"
        ).fetchall()
        w.execute("SELECT rf.enqueue_job('k', 20, '{}', 'key', NULL, NULL)")
    (dup,) = superuser.execute(
        "SELECT id FROM rf.jobs WHERE dedupe_key = 'key' AND status = 'queued'"
    ).fetchone() or (None,)

    with connect(settings, "superuser") as claimer:
        claimer.execute("SELECT 1 FROM rf.jobs WHERE id = %s FOR UPDATE", (dup,))  # mid-claim
        outcome: list[object] = []

        def release() -> None:
            with connect(settings, "worker") as conn:
                outcome.append(
                    conn.execute("SELECT rf.release_job(%s, %s)", (mine, version)).fetchone()
                )
                conn.commit()

        thread = threading.Thread(target=release)
        thread.start()
        _wait_until_blocked_on_lock(superuser)  # release hit the violation, now waits on dup
        claimer.execute(
            "UPDATE rf.jobs SET status = 'running', lease_expires_at = now() + interval '1 min' "
            "WHERE id = %s",
            (dup,),
        )
        claimer.commit()
        thread.join(timeout=10)
    assert outcome == [("queued",)]  # retried and requeued itself, nothing superseded
    assert job_row(superuser, cast(UUID, mine)) == ("queued",)
    assert job_row(superuser, cast(UUID, dup)) == ("running",)


def _wait_until_blocked_on_lock(superuser: Conn) -> None:
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        row = superuser.execute(
            "SELECT count(*) FROM pg_stat_activity WHERE usename = 'rf_worker' "
            "AND wait_event_type = 'Lock' AND query LIKE '%%release_job%%'"
        ).fetchone()
        if row and int(str(row[0])) >= 1:
            return
        time.sleep(0.05)
    raise AssertionError("release_job never blocked on the duplicate's row lock")


# ------------------------------------------------------------------ connector in-flight cap
def _inflight(superuser: Conn, connector: UUID) -> tuple[int, int]:
    """(counter value, actual running jobs) for the connector."""
    row = superuser.execute(
        "SELECT (SELECT running FROM rf.connector_inflight WHERE connector_id = %s), "
        "(SELECT count(*) FROM rf.jobs WHERE connector_id = %s AND status = 'running')",
        (connector, connector),
    ).fetchone()
    assert row is not None
    return int(str(row[0])), int(str(row[1]))


async def test_connector_inflight_cap_is_race_free(
    queue: PgJobQueue,
    txm: SqlTransactionManager,
    make_connector: Callable[[], tuple[UUID, UUID]],
    superuser: Conn,
) -> None:
    tenant, connector = make_connector()
    for i in range(12):
        await enqueue(txm, queue, tenant, spec(f"page-{i}", connector_id=connector))
    batches = await asyncio.gather(*(queue.claim(f"w{i}", 4) for i in range(10)))
    claimed = [lease for batch in batches for lease in batch]
    assert len(claimed) == 2
    assert _inflight(superuser, connector) == (2, 2)

    # Capped connector does not block the head of the queue for anyone else.
    other = await enqueue(txm, queue, tenant, spec("unrelated"))
    (next_lease,) = await queue.claim("w-other", 5)
    assert next_lease.job_id == other


async def test_inflight_counter_tracks_every_exit_path(
    queue: PgJobQueue,
    txm: SqlTransactionManager,
    make_connector: Callable[[], tuple[UUID, UUID]],
    superuser: Conn,
) -> None:
    tenant, connector = make_connector()
    for i in range(8):
        await enqueue(txm, queue, tenant, spec(f"p-{i}", connector_id=connector))

    async def claim_two() -> list[JobLease]:
        leases = list(await queue.claim("w", 5))
        assert len(leases) == 2
        assert _inflight(superuser, connector) == (2, 2)
        return leases

    a, b = await claim_two()
    await _complete(txm, queue, a)  # complete
    assert _inflight(superuser, connector) == (1, 1)
    await queue.fail(b, "boom", 3600.0)  # retry (parked for an hour)
    assert _inflight(superuser, connector) == (0, 0)

    a, b = await claim_two()
    await queue.release(a)  # release
    await queue.fail(b, "gone", None, permanent=True)  # dead
    assert _inflight(superuser, connector) == (0, 0)

    await claim_two()
    superuser.execute(
        "UPDATE rf.jobs SET lease_expires_at = now() - interval '1 s' "
        "WHERE connector_id = %s AND status = 'running'",
        (connector,),
    )
    assert await queue.reclaim_expired() == 2  # reclaim
    assert _inflight(superuser, connector) == (0, 0)

    await claim_two()
    superuser.execute("DELETE FROM rf.jobs WHERE connector_id = %s", (connector,))  # cascade
    assert _inflight(superuser, connector) == (0, 0)
