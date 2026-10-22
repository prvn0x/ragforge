"""Connector scheduling (slot 0008) against real Postgres: no scheduled twins, no backoff reset,
no starvation, failure backoff on the real fail/reclaim path, nightly reconcile, enqueue in the
connector's tenant, concurrent claimers, cross-tenant isolation of the failure trigger.

Job state changes go through the real definer functions (`enqueue_job`, `fail_job`,
`reclaim_expired_leases` -> `_bury_job` -> the 0008 trigger). Only the claim of a job (making it
`running` under a known lease) is simulated, because `claim_jobs` would also claim other tests'
queued jobs in this shared database. For the same reason every assertion is about this test's
own connector ids, and connector claims use a large limit.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
from datetime import UTC, datetime, timedelta
from typing import Any, cast
from uuid import UUID, uuid4

import psycopg
import pytest
import structlog.testing
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from ragforge_core.adapters.connectors.store import SqlSyncStore
from ragforge_core.adapters.db.jobs import PgJobQueue
from ragforge_core.apps.worker.upkeep import (
    reconcile_spec,
    schedule_due_connectors,
    sync_connector_spec,
)
from ragforge_core.config import Settings
from ragforge_core.db.engine import SqlTransactionManager, make_engine
from ragforge_core.domain.models import JobLease, JobPriority, JobSpec
from ragforge_core.domain.ports import JobQueue
from ragforge_core.services.jobs import FairJobQueue, JobHandlers, WorkerConfig, WorkerRuntime
from ragforge_core.services.sync.deps import SyncDeps
from ragforge_core.services.sync.handlers import run_sync_job
from ragforge_core.services.sync.policy import KIND_SYNC_CONNECTOR

pytestmark = pytest.mark.integration
Conn = psycopg.Connection[tuple[object, ...]]
MakeTenant = Callable[..., tuple[UUID, UUID, UUID]]
LEASE_S = 300
ALL = 1000
BACKOFF_BASE_S = 300  # slot 0008 trigger: 300 s * 2^poll_failures, capped at one day, +/-10%
RECONCILE_LEASE_S = 240


def _within_backoff(seconds: float, backoff: float) -> bool:
    return backoff * 0.9 - 5 < seconds <= backoff * 1.1


@pytest.fixture
async def worker_engine(settings: Settings) -> AsyncIterator[AsyncEngine]:
    engine = make_engine(settings, "worker")
    yield engine
    await engine.dispose()


@pytest.fixture
def queue(worker_engine: AsyncEngine) -> PgJobQueue:
    return PgJobQueue(worker_engine)


@pytest.fixture
def txm(worker_engine: AsyncEngine) -> SqlTransactionManager:
    return SqlTransactionManager(worker_engine)


@pytest.fixture
def connector(superuser: Conn, make_tenant: object) -> Callable[..., tuple[UUID, UUID]]:
    """Factory: (tenant_id, connector_id), active, due since `due_at` (default: an hour ago)."""

    def _make(due_at: str = "now() - interval '1 hour'") -> tuple[UUID, UUID]:
        tenant_id, _, coll = cast(MakeTenant, make_tenant)()
        connector_id = uuid4()
        superuser.execute(
            "INSERT INTO rf.connectors (id, tenant_id, kind, target_collection_id, display_name, "
            f"next_poll_at) VALUES (%s, %s, 'notion', %s, 'c', {due_at})",
            (connector_id, tenant_id, coll),
        )
        return tenant_id, connector_id

    return _make


def _make_due(superuser: Conn, connector_id: UUID) -> None:
    superuser.execute(
        "UPDATE rf.connectors SET next_poll_at = now() - interval '1 second' WHERE id = %s",
        (connector_id,),
    )


def _poll_in_s(superuser: Conn, connector_id: UUID) -> float:
    row = superuser.execute(
        "SELECT extract(epoch FROM next_poll_at - now()) FROM rf.connectors WHERE id = %s",
        (connector_id,),
    ).fetchone()
    assert row is not None
    return float(cast(float, row[0]))


async def _claimed(queue: PgJobQueue) -> set[UUID]:
    return {c for c, _ in await queue.claim_due_connectors(ALL, LEASE_S)}


async def _enqueue_listing(
    queue: PgJobQueue, txm: SqlTransactionManager, tenant: UUID, connector_id: UUID
) -> UUID:
    async with txm.tenant(tenant) as tx:
        return await queue.enqueue(tx, sync_connector_spec(connector_id))


def _set_status(superuser: Conn, job_id: UUID, status: str) -> None:
    """For tests about the claim's view of job states (the twin guard)."""
    lease = "now() + interval '60 seconds'" if status == "running" else "NULL"
    finished = "now()" if status in ("done", "dead") else "NULL"
    superuser.execute(
        f"UPDATE rf.jobs SET status = %s, lease_expires_at = {lease}, finished_at = {finished} "
        "WHERE id = %s",
        (status, job_id),
    )


def _start(
    superuser: Conn, job_id: UUID, *, attempts: int = 1, expires: str = "now() + interval '60s'"
) -> JobLease:
    """Simulate `claim_jobs` for one job (see module doc) and return its lease."""
    row = superuser.execute(
        f"UPDATE rf.jobs SET status = 'running', lease_version = lease_version + 1, "
        f"attempts = %s, leased_by = 'test', lease_expires_at = {expires} WHERE id = %s "
        "RETURNING tenant_id, kind, payload, lease_version, attempts",
        (attempts, job_id),
    ).fetchone()
    assert row is not None
    tenant_id, kind, payload, version, used = row
    return JobLease(
        job_id=job_id,
        tenant_id=cast(UUID, tenant_id),
        kind=str(kind),
        payload=cast(dict[str, object], payload),
        lease_version=int(cast(int, version)),
        attempts=int(cast(int, used)),
    )


def _failures(superuser: Conn, connector_id: UUID) -> int:
    row = superuser.execute(
        "SELECT poll_failures FROM rf.connectors WHERE id = %s", (connector_id,)
    ).fetchone()
    assert row is not None
    return int(cast(int, row[0]))


def _job_status(superuser: Conn, job_id: UUID) -> str:
    row = superuser.execute(
        "SELECT status FROM rf.jobs WHERE id = %s UNION ALL "
        "SELECT status FROM rf.jobs_history WHERE id = %s",
        (job_id, job_id),
    ).fetchone()
    assert row is not None
    return str(row[0])


async def test_a_queued_retrys_backoff_is_never_lowered_by_the_scheduler(
    queue: PgJobQueue,
    txm: SqlTransactionManager,
    superuser: Conn,
    connector: Callable[..., tuple[UUID, UUID]],
) -> None:
    """Real path: a listing fails retryably with Retry-After (fail_job requeues it with
    run_after in the future); a full scheduler tick must not merge into it and reset run_after
    (the pre-0008 bug)."""
    tenant, c = connector()
    job = await _enqueue_listing(queue, txm, tenant, c)
    await queue.fail(_start(superuser, job), "upstream 503", 3600.0)
    row = superuser.execute("SELECT status, run_after FROM rf.jobs WHERE id = %s", (job,))
    status, run_after = cast(tuple[str, datetime], row.fetchone())
    assert status == "queued"
    assert run_after > datetime.now(UTC) + timedelta(minutes=50)

    _make_due(superuser, c)
    await schedule_due_connectors(queue, queue, txm, lease_s=LEASE_S, limit=ALL)

    row = superuser.execute("SELECT run_after FROM rf.jobs WHERE id = %s", (job,))
    assert cast(tuple[datetime], row.fetchone())[0] == run_after


async def test_claimed_connectors_move_to_the_back(
    queue: PgJobQueue, superuser: Conn, connector: Callable[..., tuple[UUID, UUID]]
) -> None:
    """Failing connectors can no longer pin the head of the order (LIMIT starvation)."""
    _, first = connector("timestamptz '2000-01-01'")
    _, second = connector("timestamptz '2000-01-02'")

    assert [c for c, _ in await queue.claim_due_connectors(1, LEASE_S)] == [first]
    assert [c for c, _ in await queue.claim_due_connectors(1, LEASE_S)] == [second]


async def test_the_claim_lease_is_short_and_fixed_even_after_failures(
    queue: PgJobQueue, superuser: Conn, connector: Callable[..., tuple[UUID, UUID]]
) -> None:
    """The twin guard and the backoff are separate timers: if the worker dies between the claim
    and its enqueue, a connector with many failures is due again after ONE short lease, not
    after its (up to one day) backoff."""
    _, c = connector()
    superuser.execute("UPDATE rf.connectors SET poll_failures = 9 WHERE id = %s", (c,))

    assert c in await _claimed(queue)  # ...and the enqueue never happens

    assert LEASE_S - 5 < _poll_in_s(superuser, c) <= LEASE_S


async def test_a_dead_listing_backs_off_exponentially_and_success_resets(
    queue: PgJobQueue,
    txm: SqlTransactionManager,
    superuser: Conn,
    connector: Callable[..., tuple[UUID, UUID]],
) -> None:
    """Real path: fail_job(permanent) -> _bury_job -> dead -> trigger."""
    tenant, c = connector()
    for failures in (1, 2):
        job = await _enqueue_listing(queue, txm, tenant, c)
        await queue.fail(_start(superuser, job), "NotionApiError: 401", None, permanent=True)
        assert _job_status(superuser, job) == "dead"
        assert _failures(superuser, c) == failures
        assert _within_backoff(_poll_in_s(superuser, c), BACKOFF_BASE_S * 2**failures)
    assert c not in await _claimed(queue)  # backed off: not due

    job = await _enqueue_listing(queue, txm, tenant, c)
    lease = _start(superuser, job)
    async with txm.tenant(tenant) as tx:
        await queue.assert_lease(tx, lease)
        await queue.complete(tx, lease)
    assert _failures(superuser, c) == 0


async def test_retries_exhausted_on_the_retryable_path_also_back_off(
    queue: PgJobQueue,
    txm: SqlTransactionManager,
    superuser: Conn,
    connector: Callable[..., tuple[UUID, UUID]],
) -> None:
    tenant, c = connector()
    job = await _enqueue_listing(queue, txm, tenant, c)
    await queue.fail(_start(superuser, job, attempts=8), "timeout", None)  # max_attempts = 8
    assert _job_status(superuser, job) == "dead"
    assert _failures(superuser, c) == 1


async def test_reclaim_burying_a_listing_backs_off(
    queue: PgJobQueue,
    txm: SqlTransactionManager,
    superuser: Conn,
    connector: Callable[..., tuple[UUID, UUID]],
) -> None:
    """A worker that died on its last attempt: reclaim_expired_leases buries the job, and the
    trigger still sees the dead transition (no exit path skips the backoff)."""
    tenant, c = connector()
    job = await _enqueue_listing(queue, txm, tenant, c)
    _start(superuser, job, attempts=8, expires="now() - interval '1 second'")

    await queue.reclaim_expired()

    assert _job_status(superuser, job) == "dead"
    assert _failures(superuser, c) == 1
    assert _within_backoff(_poll_in_s(superuser, c), 2 * BACKOFF_BASE_S)


async def test_a_job_naming_another_tenants_connector_never_touches_it(
    queue: PgJobQueue,
    txm: SqlTransactionManager,
    superuser: Conn,
    connector: Callable[..., tuple[UUID, UUID]],
) -> None:
    """ADR-0013 cross-tenant test for the definer trigger: it matches on (connector, tenant)."""
    tenant_a, _ = connector()
    _, foreign = connector()
    before = _poll_in_s(superuser, foreign)
    async with txm.tenant(tenant_a) as tx:
        job = await queue.enqueue(
            tx,
            JobSpec(
                kind="sync_connector",
                priority=JobPriority.incremental,
                payload={"connector_id": str(foreign)},
                dedupe_key=f"sync_connector:{foreign}",
            ),
        )
    await queue.fail(_start(superuser, job), "x", None, permanent=True)

    assert _job_status(superuser, job) == "dead"
    assert _failures(superuser, foreign) == 0
    assert abs(_poll_in_s(superuser, foreign) - before) < 5


@pytest.mark.parametrize("payload", [{"connector_id": "nope"}, {}, {"connector_id": 7}])
async def test_a_malformed_listing_payload_never_aborts_fail_job(
    queue: PgJobQueue,
    txm: SqlTransactionManager,
    superuser: Conn,
    connector: Callable[..., tuple[UUID, UUID]],
    payload: dict[str, object],
) -> None:
    tenant, _ = connector()
    async with txm.tenant(tenant) as tx:
        job = await queue.enqueue(
            tx,
            JobSpec(
                kind="sync_connector",
                priority=JobPriority.incremental,
                payload=payload,
                dedupe_key=f"sync_connector:malformed:{uuid4()}",
            ),
        )
    await queue.fail(_start(superuser, job), "x", None, permanent=True)
    assert _job_status(superuser, job) == "dead"


async def test_concurrent_claimers_get_disjoint_connectors(
    settings: Settings, connector: Callable[..., tuple[UUID, UUID]]
) -> None:
    """Two schedulers claiming at the same moment (A's transaction still open) never claim the
    same connector: SKIP LOCKED hands B the next ones in order."""
    ours = [connector(f"timestamptz '1990-01-0{i}'")[1] for i in range(1, 5)]
    dsn = settings.dsn("worker", driver="postgresql")
    with psycopg.connect(dsn) as a, psycopg.connect(dsn) as b:
        claim = "SELECT connector_id FROM rf.claim_due_connectors(2, 300)"
        got_a = {cast(UUID, r[0]) for r in a.execute(claim).fetchall()}  # A: not committed yet
        got_b = {cast(UUID, r[0]) for r in b.execute(claim).fetchall()}
        a.commit()
        b.commit()
    assert got_a == set(ours[:2])
    assert got_b == set(ours[2:])


def _reconcile_in_s(superuser: Conn, connector_id: UUID) -> float:
    row = superuser.execute(
        "SELECT extract(epoch FROM next_reconcile_at - now()) FROM rf.connectors WHERE id = %s",
        (connector_id,),
    ).fetchone()
    assert row is not None
    return float(cast(float, row[0]))


async def test_reconcile_claim_is_a_short_lease_and_its_end_schedules_the_next_night(
    queue: PgJobQueue,
    txm: SqlTransactionManager,
    superuser: Conn,
    connector: Callable[..., tuple[UUID, UUID]],
) -> None:
    """Ops review r4 P3-3: the claim moves next_reconcile_at by a SHORT lease only (a lost
    enqueue retries soon); the reconcile job's end schedules the next run at interval +/-5%
    (symmetric: the nightly time does not drift later every day)."""
    tenant, c = connector()
    superuser.execute(
        "UPDATE rf.connectors SET next_reconcile_at = now() - interval '1 second' WHERE id = %s",
        (c,),
    )

    first = {x for x, _ in await queue.claim_reconcile_connectors(ALL, 86_400, RECONCILE_LEASE_S)}
    second = {x for x, _ in await queue.claim_reconcile_connectors(ALL, 86_400, RECONCILE_LEASE_S)}
    assert c in first
    assert c not in second  # leased
    assert RECONCILE_LEASE_S - 5 < _reconcile_in_s(superuser, c) <= RECONCILE_LEASE_S

    async with txm.tenant(tenant) as tx:
        job = await queue.enqueue(tx, reconcile_spec(c))
    lease = _start(superuser, job)
    async with txm.tenant(tenant) as tx:
        await queue.assert_lease(tx, lease)
        await queue.complete(tx, lease)
    assert 86_400 * 0.95 - 5 < _reconcile_in_s(superuser, c) <= 86_400 * 1.05


async def test_a_lost_reconcile_enqueue_is_retried_after_one_lease(
    queue: PgJobQueue, superuser: Conn, connector: Callable[..., tuple[UUID, UUID]]
) -> None:
    _, c = connector()
    superuser.execute(
        "UPDATE rf.connectors SET next_reconcile_at = now() - interval '1 second' WHERE id = %s",
        (c,),
    )
    assert c in {x for x, _ in await queue.claim_reconcile_connectors(ALL, 86_400, 240)}
    # ...the worker dies before enqueueing. One lease later the connector is due again:
    superuser.execute(
        "UPDATE rf.connectors SET next_reconcile_at = next_reconcile_at - interval '241 seconds' "
        "WHERE id = %s",
        (c,),
    )
    assert c in {x for x, _ in await queue.claim_reconcile_connectors(ALL, 86_400, 240)}


async def test_scheduler_enqueues_each_listing_in_its_own_tenant(
    queue: PgJobQueue,
    txm: SqlTransactionManager,
    superuser: Conn,
    connector: Callable[..., tuple[UUID, UUID]],
) -> None:
    (t1, c1), (t2, c2) = connector(), connector()

    await schedule_due_connectors(queue, queue, txm, lease_s=LEASE_S, limit=ALL)

    rows = superuser.execute(
        "SELECT tenant_id, dedupe_key, kind, connector_id FROM rf.jobs "
        "WHERE dedupe_key = ANY(%s) ORDER BY dedupe_key",
        ([f"sync_connector:{c1}", f"sync_connector:{c2}"],),
    ).fetchall()
    assert sorted(rows, key=lambda r: str(r[1])) == sorted(
        [
            (t1, f"sync_connector:{c1}", "sync_connector", None),
            (t2, f"sync_connector:{c2}", "sync_connector", None),
        ],
        key=lambda r: str(r[1]),
    )


async def test_claims_are_worker_only(settings: Settings) -> None:
    with psycopg.connect(settings.dsn("api", driver="postgresql")) as conn:
        with pytest.raises(psycopg.errors.InsufficientPrivilege):
            conn.execute("SELECT * FROM rf.claim_due_connectors(1, 300)")
        conn.rollback()
        with pytest.raises(psycopg.errors.InsufficientPrivilege):
            conn.execute("SELECT * FROM rf.claim_reconcile_connectors(1, 86400, 300)")


# ---------------------------------------------------------------- round 3 (ops review)
async def test_a_running_listing_blocks_a_second_scheduled_one(
    queue: PgJobQueue,
    txm: SqlTransactionManager,
    superuser: Conn,
    connector: Callable[..., tuple[UUID, UUID]],
) -> None:
    """The headline twin guard: a listing that is RUNNING (not just queued) is never joined by
    a scheduled second one, however long it takes."""
    tenant, cid = connector()
    assert cid in await _claimed(queue)
    job = await _enqueue_listing(queue, txm, tenant, cid)
    _start(superuser, job)  # the listing is now running under a lease
    _make_due(superuser, cid)  # ...and has outlived the claim lease
    assert cid not in await _claimed(queue)
    _set_status(superuser, job, "done")  # finished: the next poll may be scheduled
    _make_due(superuser, cid)
    assert cid in await _claimed(queue)


async def test_a_dead_listing_shows_its_error_on_the_connector(
    queue: PgJobQueue,
    txm: SqlTransactionManager,
    superuser: Conn,
    connector: Callable[..., tuple[UUID, UUID]],
) -> None:
    """Backoff must be visible: the connector carries the failure, not a silent `active`."""
    tenant, cid = connector()
    lease = _start(superuser, await _enqueue_listing(queue, txm, tenant, cid))
    await queue.fail(lease, "NotionApiError: 503 upstream", None, permanent=True)
    row = superuser.execute(
        "SELECT status, poll_failures, last_error FROM rf.connectors WHERE id = %s", (cid,)
    ).fetchone()
    assert row == ("active", 1, "listing failed (1 in a row): NotionApiError: 503 upstream")


async def test_heartbeat_never_blocks_on_the_holders_own_result_transaction(
    queue: PgJobQueue,
    txm: SqlTransactionManager,
    superuser: Conn,
    connector: Callable[..., tuple[UUID, UUID]],
) -> None:
    """0002's heartbeat UPDATE waited for the fenced result transaction and, after it committed
    `done`, reported the lease as lost. False must mean only "another holder took it"."""
    tenant, cid = connector()
    lease = _start(superuser, await _enqueue_listing(queue, txm, tenant, cid))
    async with txm.tenant(tenant) as tx:
        await queue.assert_lease(tx, lease)  # holds the job row lock, like a fenced commit
        async with asyncio.timeout(5):  # would hang until the commit with 0002's version
            assert await queue.heartbeat(lease) is True
        await queue.complete(tx, lease)
    assert _job_status(superuser, lease.job_id) == "done"
    assert await queue.heartbeat(lease) is True  # finished by this holder: not a loss
    _set_status(superuser, lease.job_id, "queued")  # reclaimed...
    _start(superuser, lease.job_id)  # ...and claimed by another worker (version + 1)
    assert await queue.heartbeat(lease) is False


async def test_reclaim_moves_the_version_so_a_zombie_holder_learns_it_lost(
    queue: PgJobQueue,
    txm: SqlTransactionManager,
    superuser: Conn,
    connector: Callable[..., tuple[UUID, UUID]],
) -> None:
    """Ops review r4 P2-1: 0002's reclaim superseded (merged into a queued duplicate) or buried
    an expired job at the SAME lease_version, so the zombie holder's heartbeat kept returning
    true and it burned capacity beside the job's new run. Reclaim now bumps the version."""
    tenant, cid = connector()
    job = await _enqueue_listing(queue, txm, tenant, cid)
    zombie = _start(superuser, job, expires="now() - interval '1 second'")  # missed its beats
    duplicate = await _enqueue_listing(queue, txm, tenant, cid)  # e.g. a manual "sync now"
    assert duplicate != job

    await queue.reclaim_expired()

    assert _job_status(superuser, job) == "superseded"  # merged into the queued duplicate
    assert await queue.heartbeat(zombie) is False


async def test_reclaim_burying_an_expired_job_is_a_loss_for_its_holder(
    queue: PgJobQueue,
    txm: SqlTransactionManager,
    superuser: Conn,
    connector: Callable[..., tuple[UUID, UUID]],
) -> None:
    tenant, cid = connector()
    job = await _enqueue_listing(queue, txm, tenant, cid)
    zombie = _start(superuser, job, attempts=8, expires="now() - interval '1 second'")

    await queue.reclaim_expired()

    assert _job_status(superuser, job) == "dead"
    assert await queue.heartbeat(zombie) is False


async def test_the_holders_own_retryable_fail_is_not_a_loss(
    queue: PgJobQueue,
    txm: SqlTransactionManager,
    superuser: Conn,
    connector: Callable[..., tuple[UUID, UUID]],
) -> None:
    """Ops review r4 P3-1: after its own retryable fail (queued, same version) a beat that
    races the handler's exit must not log a spurious job_lease_lost."""
    tenant, cid = connector()
    lease = _start(superuser, await _enqueue_listing(queue, txm, tenant, cid))
    await queue.fail(lease, "upstream 503", None)
    assert _job_status(superuser, lease.job_id) == "queued"
    assert await queue.heartbeat(lease) is True


@pytest.mark.parametrize(("age", "expected"), [("1 hour", "queued"), ("25 hours", "dead")])
def test_defer_through_the_real_claim_refunds_the_attempt_and_is_bounded(
    superuser: Conn,
    connector: Callable[..., tuple[UUID, UUID]],
    age: str,
    expected: str,
) -> None:
    """Ops review r4 P2-2, on the real claim path: `claim_jobs` charges an attempt, `defer_job`
    refunds it; a job enqueued over 24 h ago that still has no handler is buried (audited)
    instead of ping-ponging forever. Runs in ONE transaction that is rolled back: every other
    queued job of the shared database is pushed out of reach first, so the claim can only
    take this test's job."""
    tenant, _ = connector()
    with superuser.transaction(force_rollback=True):
        superuser.execute(
            "UPDATE rf.jobs SET run_after = now() + interval '1 day' WHERE status = 'queued'"
        )
        job = superuser.execute(
            "INSERT INTO rf.jobs (tenant_id, kind, priority, payload, dedupe_key, enqueued_at) "
            "VALUES (%s, 'kind_from_the_future', 0, '{}', %s, now() - %s::interval) RETURNING id",
            (tenant, f"future:{uuid4()}", age),
        ).fetchone()
        assert job is not None
        claimed = superuser.execute(
            "SELECT job_id, lease_version, attempts FROM rf.claim_jobs('test', 1, false)"
        ).fetchall()
        assert [(r[0], r[2]) for r in claimed] == [(job[0], 1)]  # the claim charged one
        (outcome,) = cast(
            tuple[str],
            superuser.execute(
                "SELECT rf.defer_job(%s, %s, 60)", (job[0], claimed[0][1])
            ).fetchone(),
        )
        row = superuser.execute(
            "SELECT status, attempts, last_error FROM rf.jobs WHERE id = %s", (job[0],)
        ).fetchone()
        assert row is not None
        assert (outcome, row[0]) == (expected, expected)
        if expected == "queued":
            assert row[1] == 0  # refunded
        else:
            assert row[2] == "no handler for job kind kind_from_the_future"
            audit = superuser.execute(
                "SELECT details->>'permanent' FROM rf.audit_log "
                "WHERE action = 'job.dead' AND target = %s",
                (f"job:{job[0]}",),
            ).fetchone()
            assert audit == ("true",)


async def test_defer_requeues_without_an_attempt_and_not_before_the_delay(
    queue: PgJobQueue,
    txm: SqlTransactionManager,
    superuser: Conn,
    connector: Callable[..., tuple[UUID, UUID]],
) -> None:
    tenant, cid = connector()
    lease = _start(superuser, await _enqueue_listing(queue, txm, tenant, cid), attempts=3)
    await queue.defer(lease, 120)
    row = superuser.execute(
        "SELECT status, attempts, extract(epoch FROM run_after - now()) FROM rf.jobs WHERE id = %s",
        (lease.job_id,),
    ).fetchone()
    assert row is not None
    status, attempts, delay = row
    assert (status, attempts) == ("queued", 2)  # the claim's attempt is refunded
    assert 100 < float(cast(float, delay)) <= 120


async def test_a_permanent_sync_failure_through_the_real_runtime_is_reported_once(
    queue: PgJobQueue,
    txm: SqlTransactionManager,
    superuser: Conn,
    connector: Callable[..., tuple[UUID, UUID]],
) -> None:
    """WorkerRuntime -> run_sync_job (self-reporting) -> PgJobQueue -> 0008 trigger, end to
    end: a connector whose secret is gone fails permanently exactly once, the job is dead, the
    connector is backed off, and the runtime does not complete or fail it a second time."""
    tenant, cid = connector()  # created without a secret: _open_token fails permanently
    job = await _enqueue_listing(queue, txm, tenant, cid)
    lease = _start(superuser, job)

    class _OneLease:
        """The real queue, but `claim` hands out only this test's job (see module doc)."""

        def __init__(self) -> None:
            self.fails: list[bool] = []
            self.given = False

        def __getattr__(self, name: str) -> object:
            return getattr(queue, name)

        async def claim(
            self, worker_id: str, n: int, *, prefer_low: bool = False
        ) -> list[JobLease]:
            if self.given:
                return []
            self.given = True
            return [lease]

        async def fail(
            self,
            lease: JobLease,
            error: str,
            retry_after_s: float | None,
            *,
            permanent: bool = False,
        ) -> None:
            self.fails.append(permanent)
            await queue.fail(lease, error, retry_after_s, permanent=permanent)

    one = _OneLease()
    deps = SyncDeps(
        tx=txm,
        queue=cast(JobQueue, one),
        store=SqlSyncStore(),
        index=cast(Any, None),
        chunker=cast(Any, None),
        embedders=cast(Any, None),
        secrets=cast(Any, None),
        connectors=cast(Any, None),
        blobs=cast(Any, None),
        parser=cast(Any, None),
    )
    handlers = JobHandlers()

    async def handle(job_lease: JobLease) -> None:
        await run_sync_job(job_lease, deps)

    handlers.register(KIND_SYNC_CONNECTOR, handle, reports_outcome=True)
    runtime = WorkerRuntime(
        worker_id="test",
        queue=cast(FairJobQueue, one),
        transactions=txm,
        handlers=handlers,
        config=WorkerConfig(concurrency=1, poll_interval_s=0.05),
    )
    stop = asyncio.Event()
    run = asyncio.create_task(runtime.run(stop))
    async with asyncio.timeout(10):
        while _job_status(superuser, job) == "running":  # noqa: ASYNC110 - DB state
            await asyncio.sleep(0.02)
    await asyncio.sleep(0.1)  # give a (wrong) second report the chance to happen
    stop.set()
    await asyncio.wait_for(run, 10)

    assert one.fails == [True]
    assert _job_status(superuser, job) == "dead"
    assert _failures(superuser, cid) == 1


async def test_a_long_result_transaction_is_not_reported_as_a_lost_lease(
    queue: PgJobQueue,
    txm: SqlTransactionManager,
    superuser: Conn,
    connector: Callable[..., tuple[UUID, UUID]],
) -> None:
    """The keepalive beats while the handler's fenced transaction holds the job row. With 0002's
    blocking heartbeat that beat returned `false` right after the commit; if the handler still
    had post-commit work (closing an HTTP client, as sync_connector's `async with` does), the
    runtime cancelled it and logged `job_lease_lost` for a job that had succeeded."""
    tenant, cid = connector()
    job = await _enqueue_listing(queue, txm, tenant, cid)
    lease = _start(superuser, job)
    given: list[bool] = []

    class _OneLease:
        def __getattr__(self, name: str) -> object:
            return getattr(queue, name)

        async def claim(
            self, worker_id: str, n: int, *, prefer_low: bool = False
        ) -> list[JobLease]:
            if given:
                return []
            given.append(True)
            return [lease]

    async def handle(job_lease: JobLease) -> None:
        async with txm.tenant(job_lease.tenant_id) as tx:
            await queue.assert_lease(tx, job_lease)
            await asyncio.sleep(0.5)  # several heartbeat intervals inside the fenced tx
            await queue.complete(tx, job_lease)
        await asyncio.sleep(0.3)  # post-commit cleanup, still inside the handler

    handlers = JobHandlers()
    handlers.register(KIND_SYNC_CONNECTOR, handle, reports_outcome=True)
    runtime = WorkerRuntime(
        worker_id="test",
        queue=cast(FairJobQueue, _OneLease()),
        transactions=txm,
        handlers=handlers,
        config=WorkerConfig(concurrency=1, poll_interval_s=0.05, heartbeat_interval_s=0.1),
    )
    stop = asyncio.Event()
    with structlog.testing.capture_logs() as logs:
        run = asyncio.create_task(runtime.run(stop))
        async with asyncio.timeout(10):
            while _job_status(superuser, job) == "running":  # noqa: ASYNC110 - DB state
                await asyncio.sleep(0.02)
        await asyncio.sleep(0.3)
        stop.set()
        await asyncio.wait_for(run, 10)

    assert _job_status(superuser, job) == "done"
    assert [e for e in logs if e["event"] == "job_lease_lost"] == []


# ------------------------------------------------------------------ ops review round 6
async def _old_scheduler(worker_engine: AsyncEngine) -> set[UUID]:
    """What a pre-0008 worker runs (its PgJobQueue.list_due_connectors), as rf_worker."""
    async with worker_engine.begin() as conn:
        rows = (
            await conn.execute(
                text("SELECT connector_id FROM rf.list_due_connectors(:n)"), {"n": ALL}
            )
        ).all()
    return {row.connector_id for row in rows}


async def test_the_pre_0008_scheduler_function_is_a_safe_claim(
    worker_engine: AsyncEngine,
    queue: PgJobQueue,
    txm: SqlTransactionManager,
    superuser: Conn,
    connector: Callable[..., tuple[UUID, UUID]],
) -> None:
    """Ops review r6 P1-1: after a rollback to the previous image, its scheduler calls
    rf.list_due_connectors. Returning no rows silently stopped all polling; it now claims with
    claim_due_connectors' semantics, so old schedulers keep working without twins."""
    _, due = connector()
    busy_tenant, busy = connector()
    await _enqueue_listing(queue, txm, busy_tenant, busy)  # a listing already queued

    first = await _old_scheduler(worker_engine)
    assert due in first
    assert busy not in first  # twin guard holds for old schedulers too
    assert 290 < _poll_in_s(superuser, due) <= 300  # claimed: next poll pushed by the lease
    assert due not in await _old_scheduler(worker_engine)  # not handed out twice


async def test_reclaim_requeues_with_backoff_not_at_once(
    queue: PgJobQueue,
    txm: SqlTransactionManager,
    superuser: Conn,
    connector: Callable[..., tuple[UUID, UUID]],
) -> None:
    """Ops review r6 P2-3: a job that kills its worker must not be runnable again at once (it
    would kill the next replica within seconds). attempts=1 -> 10 s +/-20%."""
    tenant, cid = connector()
    job = await _enqueue_listing(queue, txm, tenant, cid)
    _start(superuser, job, attempts=1, expires="now() - interval '1 second'")

    await queue.reclaim_expired()

    row = superuser.execute(
        "SELECT status, extract(epoch FROM run_after - now()) FROM rf.jobs WHERE id = %s",
        (job,),
    ).fetchone()
    assert row is not None
    assert row[0] == "queued"
    assert 7.0 < float(cast(float, row[1])) <= 12.0


async def test_assert_lease_extends_a_committed_long_result_transaction(
    queue: PgJobQueue,
    txm: SqlTransactionManager,
    superuser: Conn,
    connector: Callable[..., tuple[UUID, UUID]],
) -> None:
    """Ops review r6 P3-1: the heartbeat skips a row locked by the holder's own fenced
    transaction, so the fence itself extends the lease from the CURRENT time."""
    tenant, cid = connector()
    lease = _start(
        superuser,
        await _enqueue_listing(queue, txm, tenant, cid),
        expires="now() + interval '2 seconds'",
    )
    async with txm.tenant(tenant) as tx:
        await queue.assert_lease(tx, lease)
    row = superuser.execute(
        "SELECT extract(epoch FROM lease_expires_at - now()) FROM rf.jobs WHERE id = %s",
        (lease.job_id,),
    ).fetchone()
    assert row is not None
    assert float(cast(float, row[0])) > 50
