"""Slot 0002 queue functions: tenancy, dedupe, grants, NOTIFY, retry/dead, upkeep.

ADR-0010 (scheduling) and ADR-0013 (definer functions, grants, cross-tenant isolation).
"""

from __future__ import annotations

import json
from collections.abc import Callable
from contextlib import AbstractAsyncContextManager
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from typing import cast
from uuid import UUID

import psycopg
import pytest
from sqlalchemy import text
from sqlalchemy.exc import DBAPIError, IntegrityError
from sqlalchemy.ext.asyncio import AsyncEngine

from module_a.a_support import enqueue, job_row, spec
from pgtest import connect
from ragforge_core.adapters.db.jobs import PgJobQueue
from ragforge_core.config import Settings
from ragforge_core.db.engine import SqlTransactionManager
from ragforge_core.domain.errors import LeaseLost
from ragforge_core.domain.models import JobPriority, JobSpec
from ragforge_core.domain.ports import Tx

pytestmark = pytest.mark.integration
Conn = psycopg.Connection[tuple[object, ...]]
MakeTenant = Callable[..., tuple[UUID, UUID, UUID]]

WORKER_FUNCTIONS = (
    "rf.claim_jobs(text, integer, boolean)",
    "rf.assert_lease(uuid, integer)",
    "rf.complete_job(uuid, integer)",
    "rf.report_job_progress(uuid, integer, jsonb)",
    "rf.heartbeat(uuid, integer)",
    "rf.fail_job(uuid, integer, text, double precision, boolean)",
    "rf.release_job(uuid, integer)",
    "rf.reclaim_expired_leases(integer)",
    "rf.move_finished_jobs(interval, integer, interval)",
    "rf.list_due_connectors(integer)",
)
INTERNAL_FUNCTIONS = (
    "rf.requeue(uuid)",
    "rf._claim_one(text, smallint)",
    "rf._bury_job(uuid, text, boolean)",
    "rf._ensure_jobs_history_partition(timestamptz)",
    "rf._oldest_low_class()",
)
ENQUEUE = "rf.enqueue_job(text, integer, jsonb, text, timestamptz, uuid)"


def _can_execute(superuser: Conn, role: str, fn: str) -> bool:
    row = superuser.execute("SELECT has_function_privilege(%s, %s, 'EXECUTE')", (role, fn))
    return bool(cast(tuple[object], row.fetchone())[0])


# ------------------------------------------------------------------ grants
def test_function_grants_are_least_privilege(superuser: Conn) -> None:
    assert _can_execute(superuser, "rf_api", ENQUEUE)
    assert _can_execute(superuser, "rf_worker", ENQUEUE)
    for fn in WORKER_FUNCTIONS:
        assert _can_execute(superuser, "rf_worker", fn), fn
        assert not _can_execute(superuser, "rf_api", fn), fn
        assert not _can_execute(superuser, "rf_metrics", fn), fn
    for fn in INTERNAL_FUNCTIONS:
        for role in ("rf_api", "rf_worker", "rf_metrics", "rf_backup"):
            assert not _can_execute(superuser, role, fn), (role, fn)


def test_queue_stats_is_metrics_only_and_has_no_tenant_columns(
    settings: Settings, superuser: Conn
) -> None:
    cols = {
        str(r[0])
        for r in superuser.execute(
            "SELECT attname FROM pg_attribute WHERE attrelid = 'rf.queue_stats'::regclass "
            "AND attnum > 0"
        ).fetchall()
    }
    assert not {"tenant_id", "id", "payload", "dedupe_key", "last_error"} & cols
    with connect(settings, "metrics") as conn:
        conn.execute("SELECT * FROM rf.queue_stats").fetchall()
    with connect(settings, "api") as conn, pytest.raises(psycopg.errors.InsufficientPrivilege):
        conn.execute("SELECT * FROM rf.queue_stats")


# ------------------------------------------------------------------ enqueue + dedupe
async def test_enqueue_requires_a_tenant(queue: PgJobQueue, txm: SqlTransactionManager) -> None:
    with pytest.raises(DBAPIError, match="requires a tenant"):
        async with txm.system() as tx:
            await queue.enqueue(tx, spec("k"))


async def test_enqueue_dedupes_within_tenant_keeping_most_urgent_schedule(
    queue: PgJobQueue, txm: SqlTransactionManager, tenant: UUID, superuser: Conn
) -> None:
    later = datetime.now(UTC) + timedelta(hours=1)
    original = JobSpec(
        kind="test_job",
        priority=JobPriority.backfill,
        payload={"page": 1},
        dedupe_key="page:1",
        run_after=later,
    )
    first = await enqueue(txm, queue, tenant, original)
    second = await enqueue(txm, queue, tenant, spec("page:1", JobPriority.webhook))
    assert first == second
    priority, run_after, payload = job_row(superuser, first, "priority, run_after, payload")
    assert priority == JobPriority.webhook
    assert cast(datetime, run_after) < later
    assert payload == {"page": 1}  # payload is a pure function of the key: the original stays


async def test_same_dedupe_key_in_two_tenants_never_merges(
    queue: PgJobQueue, txm: SqlTransactionManager, make_tenant: object, settings: Settings
) -> None:
    tenant_a = cast(MakeTenant, make_tenant)()[0]
    tenant_b = cast(MakeTenant, make_tenant)()[0]
    job_a = await enqueue(txm, queue, tenant_a, spec("notion-page-123"))
    job_b = await enqueue(txm, queue, tenant_b, spec("notion-page-123"))
    assert job_a != job_b
    with connect(settings, "api") as conn:  # and each tenant sees only its own job
        conn.execute("SELECT set_config('app.tenant_id', %s, true)", (str(tenant_a),))
        assert [r[0] for r in conn.execute("SELECT id FROM rf.jobs").fetchall()] == [job_a]


async def test_enqueue_rejects_another_tenants_connector(
    queue: PgJobQueue,
    txm: SqlTransactionManager,
    make_connector: Callable[[], tuple[UUID, UUID]],
    tenant: UUID,
) -> None:
    _, foreign_connector = make_connector()
    with pytest.raises(IntegrityError):
        await enqueue(txm, queue, tenant, spec("k", connector_id=foreign_connector))


async def test_enqueue_validates_inputs(
    queue: PgJobQueue, txm: SqlTransactionManager, tenant: UUID
) -> None:
    with pytest.raises(DBAPIError, match="dedupe_key"):
        await enqueue(txm, queue, tenant, spec(""))
    with pytest.raises(ValueError, match="timezone-aware"):
        await enqueue(
            txm,
            queue,
            tenant,
            replace(spec("k"), run_after=datetime(2030, 1, 1)),  # naive on purpose
        )


async def test_enqueue_notifies_rf_jobs_with_the_id_only(
    queue: PgJobQueue, txm: SqlTransactionManager, tenant: UUID, settings: Settings
) -> None:
    with connect(settings, "worker") as listener:
        listener.autocommit = True
        listener.execute("LISTEN rf_jobs")
        job_id = await enqueue(txm, queue, tenant, spec("notify-me"))
        payloads = [n.payload for n in listener.notifies(timeout=5, stop_after=1)]
    assert payloads == [str(job_id)]


# ------------------------------------------------------------------ claim/heartbeat basics
async def test_claim_returns_a_lease_and_heartbeat_extends_only_the_current_version(
    queue: PgJobQueue, txm: SqlTransactionManager, tenant: UUID, superuser: Conn
) -> None:
    job_id = await enqueue(txm, queue, tenant, spec("hb"))
    (lease,) = await queue.claim("w1", 5)
    assert (lease.job_id, lease.tenant_id, lease.lease_version, lease.attempts) == (
        job_id,
        tenant,
        1,
        1,
    )
    assert lease.payload == {"key": "hb"}
    superuser.execute(
        "UPDATE rf.jobs SET lease_expires_at = now() + interval '1 second' WHERE id = %s",
        (job_id,),
    )
    assert await queue.heartbeat(lease) is True
    (remaining,) = job_row(superuser, job_id, "lease_expires_at - now()")
    assert cast(timedelta, remaining) > timedelta(seconds=50)
    assert await queue.heartbeat(replace(lease, lease_version=0)) is False


# ------------------------------------------------------------------ cross-tenant fencing
async def test_tenant_scoped_functions_refuse_another_tenants_job(
    queue: PgJobQueue, txm: SqlTransactionManager, make_tenant: object, superuser: Conn
) -> None:
    owner = cast(MakeTenant, make_tenant)()[0]
    other = cast(MakeTenant, make_tenant)()[0]
    job_id = await enqueue(txm, queue, owner, spec("mine"))
    (lease,) = await queue.claim("w1", 1)
    openers: list[Callable[[], AbstractAsyncContextManager[Tx]]] = [
        lambda: txm.tenant(other),
        txm.system,
    ]
    for opener in openers:
        for call in (queue.assert_lease, queue.complete):
            with pytest.raises(LeaseLost):
                async with opener() as tx:
                    await call(tx, lease)
        with pytest.raises(LeaseLost):
            async with opener() as tx:
                await queue.report_progress(tx, lease, {"pages": 1})
    assert job_row(superuser, job_id, "status, progress") == ("running", {})


# ------------------------------------------------------------------ progress NOTIFY
async def test_progress_is_written_and_notify_carries_only_the_job_id(
    queue: PgJobQueue,
    txm: SqlTransactionManager,
    tenant: UUID,
    settings: Settings,
    superuser: Conn,
) -> None:
    job_id = await enqueue(txm, queue, tenant, spec("progress"))
    (lease,) = await queue.claim("w1", 1)
    with connect(settings, "api") as listener:  # API pods are the listeners (ADR-0003)
        listener.autocommit = True
        listener.execute("LISTEN rf_job_progress")
        async with txm.tenant(tenant) as tx:
            await queue.assert_lease(tx, lease)
            await queue.report_progress(tx, lease, {"pages_done": 3, "secret_title": "x"})
        notes = list(listener.notifies(timeout=5, stop_after=1))
    assert [(n.channel, n.payload) for n in notes] == [("rf_job_progress", str(job_id))]
    assert job_row(superuser, job_id, "progress") == ({"pages_done": 3, "secret_title": "x"},)


# ------------------------------------------------------------------ retry, backoff, dead
@pytest.mark.parametrize("attempts", [0, 1, 2, 5, 7, 8, 30, 1000])
def test_backoff_bounds(superuser: Conn, attempts: int) -> None:
    base = min(5 * 2 ** min(attempts, 30), 900)
    lo, hi = superuser.execute(
        "SELECT rf._job_backoff_s(%s, 0), rf._job_backoff_s(%s, 0.999999)", (attempts, attempts)
    ).fetchone() or (None, None)
    assert lo == pytest.approx(0.8 * base)
    assert hi == pytest.approx(1.2 * base, rel=1e-5)
    assert cast(float, hi) <= 900 * 1.2


async def test_fail_backs_off_within_jitter_bounds_and_retry_after_overrides(
    queue: PgJobQueue, txm: SqlTransactionManager, tenant: UUID, superuser: Conn
) -> None:
    job_id = await enqueue(txm, queue, tenant, spec("flaky"))
    (lease,) = await queue.claim("w1", 1)
    await queue.fail(lease, "boom", None)
    status, attempts, error, delay = job_row(
        superuser, job_id, "status, attempts, last_error, extract(epoch FROM run_after - now())"
    )
    assert (status, attempts, error) == ("queued", 1, "boom")
    # attempts=1 -> 10 s +/- 20 %, measured against a slightly later now().
    assert 8.0 - 2.0 <= float(cast(float, delay)) <= 12.0

    superuser.execute("UPDATE rf.jobs SET run_after = now() WHERE id = %s", (job_id,))
    (lease,) = await queue.claim("w1", 1)
    await queue.fail(lease, "429", 120.0)
    (delay,) = job_row(superuser, job_id, "extract(epoch FROM run_after - now())")
    assert 118.0 <= float(cast(float, delay)) <= 120.5


async def test_permanent_failure_goes_dead_with_an_audit_row(
    queue: PgJobQueue,
    txm: SqlTransactionManager,
    tenant: UUID,
    superuser: Conn,
    settings: Settings,
) -> None:
    job_id = await enqueue(txm, queue, tenant, spec("gone"))
    (lease,) = await queue.claim("w1", 1)
    await queue.fail(lease, "NotFound: page deleted", None, permanent=True)
    assert job_row(
        superuser, job_id, "status, lease_expires_at IS NULL, finished_at IS NOT NULL"
    ) == (
        "dead",
        True,
        True,
    )
    with connect(settings, "api") as conn:  # the tenant sees its own audit row under RLS
        conn.execute("SELECT set_config('app.tenant_id', %s, true)", (str(tenant),))
        rows = conn.execute(
            "SELECT actor, action, details FROM rf.audit_log WHERE target = %s",
            (f"job:{job_id}",),
        ).fetchall()
    assert len(rows) == 1
    actor, action, details = rows[0]
    assert (actor, action) == ("system:job-queue", "job.dead")
    assert cast(dict[str, object], details)["permanent"] is True
    assert cast(dict[str, object], details)["kind"] == "test_job"


async def test_max_attempts_goes_dead_with_an_audit_row(
    queue: PgJobQueue, txm: SqlTransactionManager, tenant: UUID, superuser: Conn
) -> None:
    job_id = await enqueue(txm, queue, tenant, spec("poison"))
    superuser.execute("UPDATE rf.jobs SET max_attempts = 2 WHERE id = %s", (job_id,))
    for expected in ("queued", "dead"):
        superuser.execute("UPDATE rf.jobs SET run_after = now() WHERE id = %s", (job_id,))
        (lease,) = await queue.claim("w1", 1)
        await queue.fail(lease, "boom", None)
        assert job_row(superuser, job_id) == (expected,)
    row = superuser.execute(
        "SELECT details->>'attempts', details->>'permanent' FROM rf.audit_log WHERE target = %s",
        (f"job:{job_id}",),
    ).fetchall()
    assert row == [("2", "false")]


async def test_fail_and_release_are_noops_for_a_lost_lease(
    queue: PgJobQueue, txm: SqlTransactionManager, tenant: UUID, superuser: Conn
) -> None:
    job_id = await enqueue(txm, queue, tenant, spec("lost"))
    (lease,) = await queue.claim("w1", 1)
    async with txm.tenant(tenant) as tx:
        await queue.assert_lease(tx, lease)
        await queue.complete(tx, lease)
    await queue.fail(lease, "late failure", None)
    await queue.release(lease)
    assert job_row(superuser, job_id, "status, last_error") == ("done", None)


async def test_release_gives_the_job_back_without_charging_an_attempt(
    queue: PgJobQueue, txm: SqlTransactionManager, tenant: UUID, superuser: Conn
) -> None:
    job_id = await enqueue(txm, queue, tenant, spec("shutdown"))
    (lease,) = await queue.claim("w1", 1)
    await queue.release(lease)
    assert job_row(superuser, job_id, "status, attempts, lease_expires_at") == ("queued", 0, None)
    (again,) = await queue.claim("w2", 1)
    assert (again.job_id, again.lease_version, again.attempts) == (job_id, 2, 1)


async def test_release_and_retry_merge_into_a_queued_duplicate(
    queue: PgJobQueue, txm: SqlTransactionManager, tenant: UUID, superuser: Conn
) -> None:
    for give_back in ("release", "fail"):
        superuser.execute("DELETE FROM rf.jobs")
        running = await enqueue(
            txm, queue, tenant, spec(f"merge-{give_back}", JobPriority.backfill)
        )
        (lease,) = await queue.claim("w1", 1)
        duplicate = await enqueue(
            txm, queue, tenant, spec(f"merge-{give_back}", JobPriority.maintenance)
        )
        assert duplicate != running  # the running job does not absorb new enqueues
        superuser.execute(
            "UPDATE rf.jobs SET priority = 40, run_after = now() + interval '1 hour' WHERE id = %s",
            (duplicate,),
        )
        superuser.execute("UPDATE rf.jobs SET priority = 10 WHERE id = %s", (running,))
        if give_back == "release":
            await queue.release(lease)
        else:
            await queue.fail(lease, "boom", 0.0)
        assert job_row(superuser, running, "status, finished_at IS NOT NULL") == (
            "superseded",
            True,
        )
        priority, soon = job_row(
            superuser, duplicate, "priority, run_after <= now() + interval '1 second'"
        )
        assert (priority, soon) == (10, True)  # LEAST of both schedules


# ------------------------------------------------------------------ upkeep
async def test_move_finished_jobs_fills_monthly_partitions_and_keeps_default_empty(
    queue: PgJobQueue, txm: SqlTransactionManager, tenant: UUID, superuser: Conn
) -> None:
    ids = [await enqueue(txm, queue, tenant, spec(f"done-{i}")) for i in range(3)]
    for lease in await queue.claim("w1", 3):
        async with txm.tenant(tenant) as tx:
            await queue.assert_lease(tx, lease)
            await queue.complete(tx, lease)
    # One finished an hour ago, one last month (still within retention), one just now.
    superuser.execute(
        "UPDATE rf.jobs SET finished_at = now() - interval '1 hour' WHERE id = %s", (ids[0],)
    )
    superuser.execute(
        "UPDATE rf.jobs SET finished_at = date_trunc('month', now()) - interval '1 day' "
        "WHERE id = %s",
        (ids[1],),
    )
    moved = await queue.move_finished()
    assert moved == 2
    assert job_row(superuser, ids[2]) == ("done",)
    in_history = {
        r[0]
        for r in superuser.execute(
            "SELECT id FROM rf.jobs_history WHERE id = ANY(%s)", ([ids[0], ids[1]],)
        ).fetchall()
    }
    assert in_history == {ids[0], ids[1]}
    assert superuser.execute("SELECT count(*) FROM rf.jobs_history_default").fetchone() == (0,)
    parts = {
        str(r[0])
        for r in superuser.execute(
            "SELECT c.relname FROM pg_inherits i JOIN pg_class c ON c.oid = i.inhrelid "
            "WHERE i.inhparent = 'rf.jobs_history'::regclass"
        ).fetchall()
    }
    now = datetime.now(UTC)
    last_month = now.replace(day=1) - timedelta(days=1)
    next_month = (now.replace(day=1) + timedelta(days=32)).replace(day=1)
    for month in (last_month, now, next_month):
        assert f"jobs_history_p{month:%Y%m}" in parts
    # Partitions are classified and owned like every other table (catalog tests rely on it).
    rows = superuser.execute(
        "SELECT obj_description(c.oid, 'pg_class'), pg_get_userbyid(c.relowner) "
        "FROM pg_class c WHERE c.relname = %s",
        (f"jobs_history_p{last_month:%Y%m}",),
    ).fetchall()
    assert rows == [("class: derived", "rf_system")]


async def test_move_finished_jobs_drops_partitions_past_retention(
    queue: PgJobQueue, superuser: Conn
) -> None:
    superuser.execute("SET ROLE rf_system")
    try:
        superuser.execute("SELECT rf._ensure_jobs_history_partition(now() - interval '120 days')")
    finally:
        superuser.execute("RESET ROLE")
    old = f"jobs_history_p{datetime.now(UTC) - timedelta(days=120):%Y%m}"
    assert superuser.execute("SELECT to_regclass(%s)", (f"rf.{old}",)).fetchone() != (None,)
    await queue.move_finished()
    assert superuser.execute("SELECT to_regclass(%s)", (f"rf.{old}",)).fetchone() == (None,)


async def test_list_due_connectors_returns_only_active_due_ones(
    worker_engine: AsyncEngine,
    make_connector: Callable[[], tuple[UUID, UUID]],
    superuser: Conn,
) -> None:
    """The scheduler entry point a pre-0008 worker calls. At 0002 it only reads; slot 0008 makes
    it a claim (moves next_poll_at), so a rolled-back image keeps polling without twins. Both
    return active, due connectors only."""
    due_tenant, due = make_connector()
    _, paused = make_connector()
    _, future = make_connector()
    superuser.execute("UPDATE rf.connectors SET status = 'paused' WHERE id = %s", (paused,))
    superuser.execute(
        "UPDATE rf.connectors SET next_poll_at = now() + interval '1 hour' WHERE id = %s",
        (future,),
    )
    async with worker_engine.begin() as conn:
        rows = (
            await conn.execute(
                text("SELECT connector_id, tenant_id FROM rf.list_due_connectors(10000)")
            )
        ).all()
    listed = {row.connector_id: row.tenant_id for row in rows}
    assert listed[due] == due_tenant
    assert paused not in listed
    assert future not in listed
    revision = superuser.execute("SELECT version_num FROM rf.alembic_version").fetchone()
    assert revision is not None
    moved = superuser.execute(
        "SELECT next_poll_at > now() FROM rf.connectors WHERE id = %s", (due,)
    ).fetchone()
    assert moved == ((str(revision[0]) >= "0008"),)  # 0008+: claimed, pushed by its lease


def test_payloads_are_canonical_json() -> None:
    from ragforge_core.adapters.db.jobs import _dumps

    assert _dumps({"b": 1, "a": [1, 2]}) == json.dumps({"a": [1, 2], "b": 1}, separators=(",", ":"))
    with pytest.raises(ValueError, match="Out of range"):
        _dumps({"x": float("nan")})
