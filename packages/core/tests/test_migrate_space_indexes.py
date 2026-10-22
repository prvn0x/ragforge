"""The `ragforge-index` one-shot (handoff B, request 4), its separation from `ragforge-migrate`,
and the Phase 2 integration slot 0007."""

from __future__ import annotations

import contextlib
import threading
import time
from collections.abc import Callable, Iterator
from uuid import UUID

import psycopg
import pytest
from psycopg import errors

from ragforge_contracts.api import Readiness
from ragforge_core.adapters.db.search import (
    POSSIBLE_INDEX_BUILDERS,
    build_space_index,
    space_index_name,
)
from ragforge_core.apps.index import main as index_main
from ragforge_core.apps.index import space_indexes
from ragforge_core.apps.index.space_indexes import (
    ADVISORY_LOCK_KEY,
    APPLICATION_NAME,
    ensure_space_indexes,
)
from ragforge_core.apps.migrate import main as migrate_main
from ragforge_core.config import Settings

pytestmark = pytest.mark.integration
Conn = psycopg.Connection[tuple[object, ...]]
MakeTenant = Callable[..., tuple[UUID, UUID, UUID]]

DEFAULT_SPACE = 1  # seeded by 0001 (nomic-embed-text, 768, cosine)


def _index_valid(conn: Conn, space_id: int) -> bool | None:
    row = conn.execute(
        "SELECT i.indisvalid FROM pg_index i WHERE i.indexrelid = to_regclass(%s)",
        (f"rf.{space_index_name(space_id)}",),
    ).fetchone()
    return None if row is None else bool(row[0])


def _index_oid(conn: Conn, space_id: int) -> object:
    row = conn.execute("SELECT to_regclass(%s)::oid", (f"rf.{space_index_name(space_id)}",))
    value = row.fetchone()
    assert value is not None
    return value[0]


@pytest.fixture
def scratch_space(superuser: Conn) -> Iterator[int]:
    row = superuser.execute(
        "INSERT INTO rf.embedding_spaces (provider, model, dim, distance, status) "
        "VALUES ('ollama', 'scratch-' || gen_random_uuid(), 8, 'l2', 'active') RETURNING id"
    ).fetchone()
    assert row is not None
    space_id = int(str(row[0]))
    yield space_id
    superuser.execute(f"DROP INDEX IF EXISTS rf.{space_index_name(space_id)}")
    superuser.execute("DELETE FROM rf.embedding_spaces WHERE id = %s", (space_id,))


def test_step_builds_the_default_space_index_and_is_idempotent(
    settings: Settings, superuser: Conn
) -> None:
    first = ensure_space_indexes(settings)
    oid = _index_oid(superuser, DEFAULT_SPACE)
    second = ensure_space_indexes(settings)

    assert DEFAULT_SPACE in first.built
    assert first == second
    assert _index_valid(superuser, DEFAULT_SPACE) is True
    assert _index_oid(superuser, DEFAULT_SPACE) == oid, "a valid index is never rebuilt"


def test_an_invalid_leftover_index_is_rebuilt(
    settings: Settings, superuser: Conn, scratch_space: int
) -> None:
    ensure_space_indexes(settings)
    old = _index_oid(superuser, scratch_space)
    superuser.execute("UPDATE pg_index SET indisvalid = false WHERE indexrelid = %s", (old,))

    report = ensure_space_indexes(settings)

    assert scratch_space in report.built
    assert _index_valid(superuser, scratch_space) is True
    assert _index_oid(superuser, scratch_space) != old


def test_an_invalid_index_is_not_dropped_while_a_build_may_be_running(
    settings: Settings, superuser: Conn, scratch_space: int
) -> None:
    """A CONCURRENTLY build holds ShareUpdateExclusiveLock on rf.embeddings throughout; an
    INVALID index seen meanwhile may be that build, so it is left alone (retried next run)."""
    ensure_space_indexes(settings)
    oid = _index_oid(superuser, scratch_space)
    superuser.execute("UPDATE pg_index SET indisvalid = false WHERE indexrelid = %s", (oid,))
    with psycopg.connect(settings.dsn("superuser", driver="postgresql")) as builder:
        builder.execute("LOCK TABLE rf.embeddings IN SHARE UPDATE EXCLUSIVE MODE")
        report = ensure_space_indexes(settings)
        builder.rollback()

    assert scratch_space in report.skipped
    assert scratch_space not in report.failed
    assert _index_oid(superuser, scratch_space) == oid


def test_a_concurrent_run_skips_the_step(settings: Settings, superuser: Conn) -> None:
    with psycopg.connect(settings.dsn("superuser", driver="postgresql"), autocommit=True) as other:
        other.execute("SELECT pg_advisory_lock(%s)", (ADVISORY_LOCK_KEY,))
        report = ensure_space_indexes(settings)
        other.execute("SELECT pg_advisory_unlock(%s)", (ADVISORY_LOCK_KEY,))

    assert report.locked_out is True
    assert report.built == report.dropped == report.failed == []


def test_one_failing_space_does_not_stop_the_others(
    settings: Settings, monkeypatch: pytest.MonkeyPatch, scratch_space: int
) -> None:
    def flaky(conn: psycopg.Connection[tuple[object, ...]], space_id: int) -> bool:
        if space_id == scratch_space:
            raise RuntimeError("simulated")
        return build_space_index(conn, space_id)

    monkeypatch.setattr(space_indexes, "build_space_index", flaky)
    report = ensure_space_indexes(settings)

    assert report.failed == [scratch_space]
    assert DEFAULT_SPACE in report.built


def test_migrate_is_schema_only(
    settings: Settings, superuser: Conn, monkeypatch: pytest.MonkeyPatch, scratch_space: int
) -> None:
    """api and worker wait for migrate: it must never wait on an index build (ops review P1)."""
    ready = Readiness(
        ready=True, schema_version=8, required_schema=8, code_version=1, min_code_version=0
    )
    monkeypatch.setattr(migrate_main, "get_settings", lambda: settings)
    monkeypatch.setattr(migrate_main, "upgrade_locked", lambda settings: None)
    monkeypatch.setattr(migrate_main, "verify_schema", lambda settings: ready)
    monkeypatch.setattr(migrate_main, "configure_logging", lambda *a, **k: None)

    migrate_main.run()

    assert _index_valid(superuser, scratch_space) is None, "migrate built an index"


def test_the_index_entrypoint_builds_and_exits_zero(
    settings: Settings, superuser: Conn, monkeypatch: pytest.MonkeyPatch, scratch_space: int
) -> None:
    monkeypatch.setattr(index_main, "get_settings", lambda: settings)
    monkeypatch.setattr(index_main, "configure_logging", lambda *a, **k: None)

    index_main.run()  # returns normally: exit status 0

    assert _index_valid(superuser, scratch_space) is True


def test_the_index_entrypoint_exits_one_only_when_the_database_is_unreachable(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    unreachable = settings.model_copy(update={"db_port": 1})
    monkeypatch.setattr(index_main, "get_settings", lambda: unreachable)
    monkeypatch.setattr(index_main, "configure_logging", lambda *a, **k: None)

    with pytest.raises(SystemExit) as exit_info:
        index_main.run()
    assert exit_info.value.code == 1


def test_a_build_that_waits_out_the_lock_bound_is_deferred_not_failed(
    settings: Settings, superuser: Conn, scratch_space: int
) -> None:
    """CONCURRENTLY waits for every writer of rf.embeddings. With a finite lock bound the space
    is deferred (exact search serves it) instead of holding the step, and the next run builds."""
    with psycopg.connect(settings.dsn("superuser", driver="postgresql")) as writer:
        writer.execute("LOCK TABLE rf.embeddings IN ROW EXCLUSIVE MODE")  # an open writer
        report = ensure_space_indexes(settings, lock_timeout="300ms")
        writer.rollback()

    assert scratch_space in report.deferred
    assert scratch_space not in report.failed
    assert DEFAULT_SPACE in report.built  # already valid: no DDL, so no wait

    report = ensure_space_indexes(settings)  # the writer is gone: the retry builds it
    assert scratch_space in report.built
    assert _index_valid(superuser, scratch_space) is True


def test_migrate_still_exits_one_when_the_schema_is_not_ready(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    behind = Readiness(
        ready=False, schema_version=7, required_schema=8, code_version=1, min_code_version=0
    )
    monkeypatch.setattr(migrate_main, "get_settings", lambda: settings)
    monkeypatch.setattr(migrate_main, "upgrade_locked", lambda settings: None)
    monkeypatch.setattr(migrate_main, "verify_schema", lambda settings: behind)
    monkeypatch.setattr(migrate_main, "configure_logging", lambda *a, **k: None)

    with pytest.raises(SystemExit) as exit_info:
        migrate_main.run()
    assert exit_info.value.code == 1


def test_step_drops_the_index_of_an_unused_retired_space(
    settings: Settings, superuser: Conn, scratch_space: int
) -> None:
    ensure_space_indexes(settings)
    assert _index_valid(superuser, scratch_space) is True

    superuser.execute(
        "UPDATE rf.embedding_spaces SET status = 'retired', retired_at = now() WHERE id = %s",
        (scratch_space,),
    )
    report = ensure_space_indexes(settings)

    assert scratch_space in report.dropped
    assert _index_valid(superuser, scratch_space) is None


def test_slot_0007_adds_the_execution_claim_column(superuser: Conn) -> None:
    row = superuser.execute(
        "SELECT data_type FROM information_schema.columns "
        "WHERE table_schema = 'rf' AND table_name = 'pending_approvals' "
        "AND column_name = 'executed_at'"
    ).fetchone()
    assert row == ("timestamp with time zone",)


def test_slot_0007_retired_space_requires_retired_at(superuser: Conn) -> None:
    with pytest.raises(errors.CheckViolation):
        superuser.execute(
            "INSERT INTO rf.embedding_spaces (provider, model, dim, distance, status) "
            "VALUES ('ollama', 'bad-retired', 8, 'l2', 'retired')"
        )
    with pytest.raises(errors.CheckViolation):
        superuser.execute(
            "INSERT INTO rf.embedding_spaces (provider, model, dim, distance, status, retired_at) "
            "VALUES ('ollama', 'bad-active', 8, 'l2', 'active', now())"
        )


# ---------------------------------------------------------------- build-in-progress probe
def _builders(settings: Settings) -> set[int]:
    """The probe exactly as build_space_index runs it: as rf_migrator, SET ROLE rf_system."""
    with psycopg.connect(settings.dsn("migrator", driver="postgresql"), autocommit=True) as conn:
        conn.execute("SET ROLE rf_system")
        return {int(str(r[0])) for r in conn.execute(POSSIBLE_INDEX_BUILDERS).fetchall()}


def test_a_lock_holder_that_may_be_building_counts_as_a_builder(settings: Settings) -> None:
    with psycopg.connect(settings.dsn("superuser", driver="postgresql")) as holder:
        holder.execute("LOCK TABLE rf.embeddings IN SHARE UPDATE EXCLUSIVE MODE")
        assert holder.info.backend_pid in _builders(settings)
        holder.rollback()


def test_a_running_vacuum_is_not_mistaken_for_an_index_build(
    settings: Settings, superuser: Conn, make_tenant: MakeTenant, scratch_space: int
) -> None:
    """VACUUM holds the same ShareUpdateExclusiveLock. A throttled VACUUM over dead rows runs
    long enough to observe it holding the lock while the probe still does not count it."""
    tenant, _, coll = make_tenant()
    superuser.execute(
        "INSERT INTO rf.embeddings (tenant_id, collection_id, space_id, embed_sha256, embedding) "
        "SELECT %s, %s, %s, sha256(g::text::bytea), '[1,2,3,4,5,6,7,8]' "
        "FROM generate_series(1, 20000) g",
        (tenant, coll, scratch_space),
    )
    superuser.execute("DELETE FROM rf.embeddings WHERE space_id = %s", (scratch_space,))

    pid: list[int] = []

    def vacuum() -> None:
        with psycopg.connect(
            settings.dsn("superuser", driver="postgresql"), autocommit=True
        ) as conn:
            pid.append(conn.info.backend_pid)
            conn.execute("SET vacuum_cost_delay = '20ms'")
            conn.execute("SET vacuum_cost_limit = 1")
            with contextlib.suppress(errors.QueryCanceled):  # the test cancels it once observed
                conn.execute("VACUUM rf.embeddings")

    worker = threading.Thread(target=vacuum)
    worker.start()
    try:
        deadline = time.monotonic() + 10
        observed = False
        while time.monotonic() < deadline and worker.is_alive():
            if (
                pid
                and superuser.execute(
                    "SELECT 1 FROM pg_locks l JOIN pg_stat_progress_vacuum v ON v.pid = l.pid "
                    "WHERE l.pid = %s AND l.relation = 'rf.embeddings'::regclass "
                    "AND l.mode = 'ShareUpdateExclusiveLock' AND l.granted",
                    (pid[0],),
                ).fetchone()
            ):
                assert pid[0] not in _builders(settings)
                observed = True
                break
            time.sleep(0.02)
        assert observed, "the throttled VACUUM finished before it could be observed"
    finally:
        if pid:
            superuser.execute("SELECT pg_cancel_backend(%s)", (pid[0],))
        worker.join(timeout=30)


def test_migrate_stops_a_running_index_build_before_upgrading(settings: Settings) -> None:
    """Ops review r4 P2-4: a build from the previous deploy holds SHARE UPDATE EXCLUSIVE on
    rf.embeddings for up to its statement bound; a migration touching that table would fail on
    the migrator's 5 s lock_timeout and keep api and worker down. Migrate takes the index
    step's advisory lock first, terminating the session that holds it, and keeps it for the
    whole upgrade so no new build starts mid-migration."""
    dsn = settings.dsn("migrator", driver="postgresql")
    # Stands in for a running build: the index step's sessions carry its application_name.
    with psycopg.connect(dsn, autocommit=True, application_name=APPLICATION_NAME) as build:
        build.execute("SELECT pg_advisory_lock(%s)", (ADVISORY_LOCK_KEY,))
        build_pid = build.info.backend_pid

        with (
            migrate_main.pause_index_builds(settings, deadline_s=10),
            psycopg.connect(dsn, autocommit=True) as probe,
        ):
            held = probe.execute("SELECT pg_try_advisory_lock(%s)", (ADVISORY_LOCK_KEY,)).fetchone()
            assert held == (False,)  # migrate holds it: no index run can start now
            alive = probe.execute(
                "SELECT count(*) FROM pg_stat_activity WHERE pid = %s", (build_pid,)
            ).fetchone()
            assert alive == (0,)  # the build's session was terminated

    with psycopg.connect(dsn, autocommit=True) as after:  # released after the upgrade
        assert after.execute(
            "SELECT pg_try_advisory_lock(%s)", (ADVISORY_LOCK_KEY,)
        ).fetchone() == (True,)
        after.execute("SELECT pg_advisory_unlock(%s)", (ADVISORY_LOCK_KEY,))


def test_migrate_verifies_applied_checksums_before_upgrading(
    settings: Settings, superuser: Conn, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ops review r4 P3-7: an edited applied revision must stop the deploy BEFORE any newer
    revision applies on top of it."""
    from alembic import command

    from ragforge_core.db import migrate as migrate_db

    upgraded: list[bool] = []
    monkeypatch.setattr(command, "upgrade", lambda *a, **k: upgraded.append(True))
    real = migrate_db.migration_checksums()
    monkeypatch.setattr(migrate_db, "migration_checksums", lambda: {**real, "0001": "0" * 64})
    with pytest.raises(migrate_db.MigrationEdited):
        migrate_db.upgrade_to_head(settings)
    assert upgraded == []
