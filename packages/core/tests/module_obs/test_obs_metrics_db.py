"""Migration 0013 (`rf.connector_freshness`), migration 0014 (`rf.dead_jobs_last_day()`), the
scrape-time DB collector and the tenant surface lookup, on the real test database (ADR-0018
R2-2 and the final review's P1-3).

Freshness cases run inside ONE superuser transaction that pauses every other connector and is
rolled back at the end, so they see exactly the rows they create, whatever other modules of
the session inserted, and leave nothing behind."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from datetime import datetime, timedelta
from typing import cast
from uuid import UUID, uuid4

import psycopg
import pytest
from prometheus_client.core import Metric

from pgtest import connect
from ragforge_core.config import Settings
from ragforge_core.db.engine import SqlTransactionManager, make_engine
from ragforge_core.observability.db_collector import (
    _QUEUE_SQL,
    DbGaugeCollector,
    _queue_families,
    metrics_engine,
)
from ragforge_core.observability.metrics import PRIORITIES
from ragforge_core.observability.surface import DEMO_TENANT_NAME, TenantSurfaces

pytestmark = pytest.mark.integration
Conn = psycopg.Connection[tuple[object, ...]]
MakeTenant = Callable[..., tuple[UUID, UUID, UUID]]
DAY = timedelta(days=1)


# ---------------------------------------------------------------------- privileges + shape
def test_the_view_has_aggregates_only(superuser: Conn) -> None:
    cols = {
        str(r[0])
        for r in superuser.execute(
            "SELECT attname FROM pg_attribute WHERE attrelid = 'rf.connector_freshness'::regclass "
            "AND attnum > 0"
        ).fetchall()
    }
    assert cols == {
        "connector_kind",
        "connectors",
        "oldest_success_at",
        "oldest_success_age_seconds",
    }


def test_only_rf_metrics_can_read_it_and_it_can_read_nothing_else(settings: Settings) -> None:
    with connect(settings, "metrics") as conn:
        conn.execute("SELECT * FROM rf.connector_freshness").fetchall()
    for table in ("rf.jobs", "rf.jobs_history", "rf.connectors"):
        with (
            connect(settings, "metrics") as conn,
            pytest.raises(psycopg.errors.InsufficientPrivilege),
        ):
            conn.execute(f"SELECT 1 FROM {table} LIMIT 1")
    for role in ("api", "worker"):
        with connect(settings, role) as conn, pytest.raises(psycopg.errors.InsufficientPrivilege):
            conn.execute("SELECT * FROM rf.connector_freshness")
        with connect(settings, role) as conn, pytest.raises(psycopg.errors.InsufficientPrivilege):
            conn.execute("SELECT * FROM rf.connector_freshness_rows()")


def test_the_function_is_granted_to_rf_metrics_alone(superuser: Conn) -> None:
    fn = "rf.connector_freshness_rows()"
    for role, allowed in (("rf_metrics", True), ("rf_api", False), ("rf_worker", False),
                          ("rf_backup", False)):  # fmt: skip
        row = superuser.execute(
            "SELECT has_function_privilege(%s, %s, 'EXECUTE')", (role, fn)
        ).fetchone()
        assert row == (allowed,), role


# ---------------------------------------------------------------------- semantics
@pytest.fixture
def isolated(settings: Settings) -> Iterator[Conn]:
    """A superuser transaction in which only this test's connectors are enabled; rolled back."""
    conn = connect(settings, "superuser")
    try:
        conn.execute(
            "UPDATE rf.connectors SET status = 'paused' WHERE status IN ('active', 'error')"
        )
        yield conn
    finally:
        conn.rollback()
        conn.close()


Tenant = tuple[UUID, UUID]  # (tenant, its collection)
# What handle_sync_connector reports: a real listing, and the no-op of an inactive connector.
LISTED = '{"listed": 0, "enqueued": 0}'
SKIPPED = '{"skipped": "connector not active"}'


def _tenant(conn: Conn) -> Tenant:
    tenant, coll = uuid4(), uuid4()
    conn.execute(
        "INSERT INTO rf.tenants (id, name, kind) VALUES (%s, %s, 'personal')",
        (tenant, f"t-{tenant.hex[:6]}"),
    )
    conn.execute(
        "INSERT INTO rf.collections (id, tenant_id, name, active_space_id) VALUES (%s, %s, 'c', 1)",
        (coll, tenant),
    )
    return tenant, coll


def _connector(conn: Conn, owner: Tenant, created: datetime, status: str = "active") -> UUID:
    connector = uuid4()
    conn.execute(
        "INSERT INTO rf.connectors (id, tenant_id, kind, target_collection_id, display_name, "
        "status, created_at) VALUES (%s, %s, 'notion', %s, 'c', %s, %s)",
        (connector, owner[0], owner[1], status, created),
    )
    return connector


def _job(
    conn: Conn,
    owner: Tenant,
    connector: UUID,
    finished: datetime,
    *,
    history: bool = False,
    status: str = "done",
    kind: str = "sync_connector",
    progress: str = LISTED,
) -> None:
    """A finished job; `progress` is what the handler reported (LISTED: a real listing)."""
    table = "rf.jobs_history" if history else "rf.jobs"
    conn.execute(
        f"INSERT INTO {table} (id, tenant_id, kind, priority, payload, dedupe_key, status, "
        "finished_at, progress) VALUES (%s, %s, %s, 20, "
        "jsonb_build_object('connector_id', %s::text), %s, %s, %s, %s::jsonb)",
        (uuid4(), owner[0], kind, str(connector), f"{kind}:{uuid4()}", status, finished, progress),
    )


def _freshness(conn: Conn) -> list[tuple[object, ...]]:
    """Read AS rf_metrics, inside the isolating transaction (its grants are exercised)."""
    conn.execute("SET LOCAL ROLE rf_metrics")
    try:
        return conn.execute(
            "SELECT connector_kind, connectors, oldest_success_at, oldest_success_age_seconds "
            "FROM rf.connector_freshness"
        ).fetchall()
    finally:
        conn.execute("RESET ROLE")


def _now(conn: Conn) -> datetime:
    (now,) = cast(tuple[datetime], conn.execute("SELECT now()").fetchone())
    return now


def test_the_stalest_enabled_connector_wins_and_quiet_polls_count(isolated: Conn) -> None:
    now = _now(isolated)
    a, b = _tenant(isolated), _tenant(isolated)
    fresh = _connector(isolated, a, now - 30 * DAY)
    _job(isolated, a, fresh, now - 20 * DAY, history=True)
    _job(isolated, a, fresh, now - 2 * DAY)  # a poll that changed nothing still completes
    stale = _connector(isolated, b, now - 30 * DAY)
    _job(isolated, b, stale, now - 5 * DAY, history=True)
    _job(isolated, b, stale, now, status="dead")  # a failed listing is not a success
    _job(isolated, b, stale, now, kind="sync_page")  # nor is a page job
    _job(isolated, a, stale, now)  # tenant A's job naming B's connector never freshens it
    _connector(isolated, b, now - 400 * DAY, status="paused")  # disabled: excluded

    ((kind, count, oldest, age),) = _freshness(isolated)
    assert (kind, count) == ("notion", 2)
    assert oldest == now - 5 * DAY
    assert age == pytest.approx((5 * DAY).total_seconds(), abs=1.0)


def test_a_connector_never_synced_counts_from_its_creation(isolated: Conn) -> None:
    now = _now(isolated)
    tenant = _tenant(isolated)
    synced = _connector(isolated, tenant, now - 3 * DAY)
    _job(isolated, tenant, synced, now - DAY)
    _connector(isolated, tenant, now - 9 * DAY)  # created, never synced
    ((_, count, oldest, _),) = _freshness(isolated)
    assert (count, oldest) == (2, now - 9 * DAY)


def test_an_errored_connector_is_still_enabled(isolated: Conn) -> None:
    now = _now(isolated)
    tenant = _tenant(isolated)
    broken = _connector(isolated, tenant, now - 10 * DAY, status="error")
    _job(isolated, tenant, broken, now - 7 * DAY, history=True)
    ((_, count, oldest, _),) = _freshness(isolated)
    assert (count, oldest) == (1, now - 7 * DAY)


def test_a_no_op_completion_never_freshens_a_connector(isolated: Conn) -> None:
    """A "sync now" on an errored connector completes its listing job as `done` WITHOUT listing
    (`_complete_noop`); that must not reset the age the stale-sync alert reads."""
    now = _now(isolated)
    tenant = _tenant(isolated)
    broken = _connector(isolated, tenant, now - 10 * DAY, status="error")
    _job(isolated, tenant, broken, now - 7 * DAY, history=True)
    _job(isolated, tenant, broken, now, progress=SKIPPED)
    _job(isolated, tenant, broken, now - DAY, history=True, progress=SKIPPED)
    _job(isolated, tenant, broken, now, progress="{}")  # no marker at all: not a listing either
    ((_, count, oldest, _),) = _freshness(isolated)
    assert (count, oldest) == (1, now - 7 * DAY)


def test_no_enabled_connector_means_no_row(isolated: Conn) -> None:
    assert _freshness(isolated) == []


# ---------------------------------------------------------------------- dead jobs (0014)
DEAD_FN = "rf.dead_jobs_last_day()"


def test_the_dead_jobs_function_returns_one_number_and_no_tenant_columns(superuser: Conn) -> None:
    row = superuser.execute(
        "SELECT p.prorettype::regtype::text, p.proretset, p.pronargs, p.prosecdef, "
        "pg_get_userbyid(p.proowner) FROM pg_proc p WHERE p.oid = %s::regprocedure",
        (DEAD_FN,),
    ).fetchone()
    assert row == ("bigint", False, 0, True, "rf_system")  # a scalar: no rows, no columns


def test_only_rf_metrics_can_execute_the_dead_jobs_function(
    settings: Settings, superuser: Conn
) -> None:
    for role, allowed in (("rf_metrics", True), ("rf_api", False), ("rf_worker", False),
                          ("rf_backup", False)):  # fmt: skip
        row = superuser.execute(
            "SELECT has_function_privilege(%s, %s, 'EXECUTE')", (role, DEAD_FN)
        ).fetchone()
        assert row == (allowed,), role
    with connect(settings, "metrics") as conn:
        (count,) = cast(tuple[int], conn.execute(f"SELECT {DEAD_FN}").fetchone())
        assert count >= 0
        for table in ("rf.jobs", "rf.jobs_history"):  # the function, never the tables
            with pytest.raises(psycopg.errors.InsufficientPrivilege):
                conn.execute(f"SELECT 1 FROM {table} LIMIT 1")
            conn.rollback()
    with connect(settings, "api") as conn, pytest.raises(psycopg.errors.InsufficientPrivilege):
        conn.execute(f"SELECT {DEAD_FN}")


def _dead_count(conn: Conn) -> int:
    """Read AS rf_metrics, inside the isolating transaction (its grant is exercised)."""
    conn.execute("SET LOCAL ROLE rf_metrics")
    try:
        (count,) = cast(tuple[int], conn.execute(f"SELECT {DEAD_FN}").fetchone())
        return count
    finally:
        conn.execute("RESET ROLE")


def test_dead_jobs_count_in_both_tables_over_24_hours(isolated: Conn) -> None:
    now = _now(isolated)
    tenant = _tenant(isolated)
    connector = _connector(isolated, tenant, now - 3 * DAY)
    before = _dead_count(isolated)
    _job(isolated, tenant, connector, now - timedelta(minutes=1), status="dead")
    _job(isolated, tenant, connector, now - timedelta(hours=3), history=True, status="dead")
    _job(isolated, tenant, connector, now - timedelta(hours=23), history=True, status="dead",
         kind="sync_page")  # fmt: skip
    _job(isolated, tenant, connector, now - timedelta(hours=25), history=True, status="dead")
    _job(isolated, tenant, connector, now - timedelta(hours=26), status="dead")
    _job(isolated, tenant, connector, now - timedelta(hours=1), history=True)  # done
    _job(isolated, tenant, connector, now - timedelta(hours=1), status="failed")
    assert _dead_count(isolated) - before == 3  # older than 24 h, done and failed: not counted


def test_a_dead_job_the_upkeep_archives_is_still_counted(isolated: Conn) -> None:
    """The bug: rf.queue_stats reads rf.jobs only, and rf.move_finished_jobs (the worker's
    upkeep) moves a dead job to rf.jobs_history five minutes after it died."""
    now = _now(isolated)
    tenant = _tenant(isolated)
    connector = _connector(isolated, tenant, now - 3 * DAY)
    before = _dead_count(isolated)
    _job(isolated, tenant, connector, now - timedelta(minutes=10), status="dead")
    assert _dead_count(isolated) - before == 1
    (moved,) = cast(
        tuple[int],
        isolated.execute("SELECT rf.move_finished_jobs(interval '5 minutes')").fetchone(),
    )
    assert moved >= 1
    (in_jobs,) = cast(
        tuple[int],
        isolated.execute(
            "SELECT count(*) FROM rf.jobs WHERE tenant_id = %s", (tenant[0],)
        ).fetchone(),
    )
    assert in_jobs == 0  # really archived
    assert _dead_count(isolated) - before == 1


# ---------------------------------------------------------------------- the collector
def _families(collector: DbGaugeCollector) -> dict[str, Metric]:
    return {f.name: f for f in collector.collect()}


def _value(family: Metric, **labels: str) -> float:
    (sample,) = [s for s in family.samples if s.labels == labels]
    return float(sample.value)


def test_the_collector_reads_queue_stats_and_freshness_as_rf_metrics(
    settings: Settings, superuser: Conn, make_tenant: MakeTenant
) -> None:
    tenant, _, coll = make_tenant()
    connector, queued, dead = uuid4(), uuid4(), uuid4()
    superuser.execute(
        "INSERT INTO rf.connectors (id, tenant_id, kind, target_collection_id, display_name) "
        "VALUES (%s, %s, 'notion', %s, 'c')",
        (connector, tenant, coll),
    )
    superuser.execute(
        "INSERT INTO rf.jobs (id, tenant_id, kind, priority, dedupe_key, enqueued_at) "
        "VALUES (%s, %s, 'sync_page', 0, %s, now() - interval '10 minutes')",
        (queued, tenant, f"obs:{queued}"),
    )
    superuser.execute(
        "INSERT INTO rf.jobs (id, tenant_id, kind, priority, dedupe_key, status, finished_at) "
        "VALUES (%s, %s, 'sync_page', 20, %s, 'dead', now())",
        (dead, tenant, f"obs:{dead}"),
    )
    collector = DbGaugeCollector.from_settings(settings)
    try:
        families = _families(collector)
    finally:
        collector.close()
        superuser.execute("DELETE FROM rf.jobs WHERE id = ANY(%s)", ([queued, dead],))
        superuser.execute("DELETE FROM rf.connectors WHERE id = %s", (connector,))
    ready = families["ragforge_jobs_ready"]
    assert {s.labels["priority"] for s in ready.samples} == set(PRIORITIES)  # zeros included
    assert _value(ready, priority="interactive") >= 1
    oldest = families["ragforge_jobs_oldest_queued_age_seconds"]
    assert _value(oldest, priority="interactive") >= 600
    assert _value(families["ragforge_jobs_dead"]) >= 1
    assert _value(families["ragforge_jobs_expired_leases"]) >= 0
    freshness = families["ragforge_sync_oldest_success_age_seconds"]
    assert _value(freshness, connector_kind="notion") >= 0


def test_the_collector_counts_a_dead_job_after_the_upkeep_archived_it(
    settings: Settings, superuser: Conn, make_tenant: MakeTenant
) -> None:
    """ragforge_jobs_dead comes from rf.dead_jobs_last_day(), not from rf.queue_stats (rf.jobs
    only), so the gauge holds a death for a day instead of dropping to 0 after five minutes."""
    tenant, _, _ = make_tenant()
    archived = uuid4()
    collector = DbGaugeCollector.from_settings(settings)
    try:
        before = _value(_families(collector)["ragforge_jobs_dead"])
        superuser.execute(
            "INSERT INTO rf.jobs_history (id, tenant_id, kind, priority, dedupe_key, status, "
            "finished_at) VALUES (%s, %s, 'sync_page', 20, %s, 'dead', now() - interval '1 hour')",
            (archived, tenant, f"obs:{archived}"),
        )
        after = _value(_families(collector)["ragforge_jobs_dead"])
    finally:
        collector.close()
        superuser.execute("DELETE FROM rf.jobs_history WHERE id = %s", (archived,))
    assert after - before == 1


def test_the_collector_reports_dead_jobs_with_an_empty_queue(isolated: Conn) -> None:
    """The empty-queue row (LEFT JOIN ON true) still carries the dead count: read the exact
    statement the collector runs, as rf_metrics, with rf.jobs emptied inside the transaction."""
    isolated.execute("DELETE FROM rf.jobs")
    now = _now(isolated)
    tenant = _tenant(isolated)
    connector = _connector(isolated, tenant, now - DAY)
    _job(isolated, tenant, connector, now - timedelta(hours=2), history=True, status="dead")
    isolated.execute("SET LOCAL ROLE rf_metrics")
    rows = isolated.execute(str(_QUEUE_SQL)).fetchall()
    isolated.execute("RESET ROLE")
    families = {f.name: f for f in _queue_families(rows)}
    assert len(rows) == 1
    assert rows[0][0] is None  # no priority: only the dead count
    assert _value(families["ragforge_jobs_dead"]) >= 1
    assert {s.value for s in families["ragforge_jobs_ready"].samples} == {0.0}


def test_the_collector_engine_bounds_every_wait(settings: Settings) -> None:
    engine = metrics_engine(settings)
    try:
        with engine.connect() as conn:
            (timeout,) = conn.exec_driver_sql("SHOW statement_timeout").one()
            (user,) = conn.exec_driver_sql("SELECT current_user").one()
        assert (timeout, user) == ("5s", "rf_metrics")
        assert engine.pool.size() == 1  # type: ignore[attr-defined]
    finally:
        engine.dispose()


# ---------------------------------------------------------------------- tenant surface
async def test_the_surface_of_a_tenant_comes_from_its_kind_and_name(
    settings: Settings, superuser: Conn, make_tenant: MakeTenant
) -> None:
    owner, _, _ = make_tenant()
    demo, _, _ = make_tenant(DEMO_TENANT_NAME)
    paired = uuid4()
    superuser.execute(
        "INSERT INTO rf.tenants (id, name, kind, eval_of_tenant_id) VALUES (%s, 'e', 'eval', %s)",
        (paired, owner),
    )
    engine = make_engine(settings, "api")
    surfaces = TenantSurfaces()
    try:
        tx = SqlTransactionManager(engine)
        assert await surfaces.resolve(tx, owner) == "chat"
        assert await surfaces.resolve(tx, demo) == "demo"
        assert await surfaces.resolve(tx, paired) == "eval"
        assert await surfaces.resolve(tx, uuid4()) == "other"  # RLS: no such row visible
    finally:
        await engine.dispose()
        superuser.execute("DELETE FROM rf.tenants WHERE id = %s", (paired,))


async def test_a_failed_lookup_is_other_and_not_cached() -> None:
    class Broken:
        calls = 0

        def tenant(self, tenant_id: UUID) -> object:
            Broken.calls += 1
            raise ConnectionError("db down")

        def system(self) -> object:
            raise AssertionError

    surfaces = TenantSurfaces()
    tenant = uuid4()
    assert await surfaces.resolve(Broken(), tenant) == "other"  # type: ignore[arg-type]
    assert await surfaces.resolve(Broken(), tenant) == "other"  # type: ignore[arg-type]
    assert Broken.calls == 2
