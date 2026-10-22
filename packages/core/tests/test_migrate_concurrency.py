"""`ragforge-migrate` vs itself, vs the index one-shot, and vs a newer schema (ops review r5).

Real Postgres, real entrypoints. Threads stand in for the separate processes of a retried k8s Job
or two deploy replicas: each entrypoint opens its own connections, so the interleavings are the
ones production sees.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Iterator

import psycopg
import pytest
import structlog.testing

from ragforge_core import CODE_VERSION
from ragforge_core.apps.index import main as index_main
from ragforge_core.apps.index import space_indexes
from ragforge_core.apps.migrate import main as migrate_main
from ragforge_core.config import Settings
from ragforge_core.db import migrate as migrate_db

pytestmark = pytest.mark.integration
Conn = psycopg.Connection[tuple[object, ...]]


@pytest.fixture
def entrypoints(settings: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    for module in (migrate_main, index_main):
        monkeypatch.setattr(module, "get_settings", lambda: settings)
        monkeypatch.setattr(module, "configure_logging", lambda *a, **k: None)


def _in_thread(target: Callable[[], None]) -> tuple[threading.Thread, list[BaseException]]:
    errors: list[BaseException] = []

    def run() -> None:
        try:
            target()
        except BaseException as exc:  # SystemExit included: it is the exit status
            errors.append(exc)

    thread = threading.Thread(target=run)
    thread.start()
    return thread, errors


def test_concurrent_migrate_runs_serialize_and_both_succeed(
    entrypoints: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The second run waits on the migration lock BEFORE pausing index builds, so it never
    terminates the first run's pause session (which holds the index key): both exit 0."""
    spans: list[tuple[float, float]] = []
    real = migrate_db.upgrade_locked

    def slow_upgrade(settings: Settings) -> None:
        started = time.monotonic()
        time.sleep(1.0)  # widen the window in which the second run arrives
        real(settings)
        spans.append((started, time.monotonic()))

    monkeypatch.setattr(migrate_main, "upgrade_locked", slow_upgrade)
    with structlog.testing.capture_logs() as logs:
        first, first_errors = _in_thread(migrate_main.run)
        time.sleep(0.2)
        second, second_errors = _in_thread(migrate_main.run)
        first.join(timeout=60)
        second.join(timeout=60)

    terminated = [e for e in logs if e["event"] == "migrate_terminating_index_build"]
    assert terminated == []  # neither run touched the other's pause session

    assert first_errors == []
    assert second_errors == []
    (a_start, a_end), (b_start, _) = sorted(spans)
    assert b_start >= a_end, "the upgrades overlapped: the migration lock did not serialize"
    assert a_start < b_start


def test_migrate_preempts_a_running_index_build_and_the_index_run_exits_zero(
    entrypoints: None, settings: Settings, superuser: Conn, monkeypatch: pytest.MonkeyPatch
) -> None:
    building = threading.Event()

    def long_build(conn: psycopg.Connection[tuple[object, ...]], report: object) -> None:
        building.set()
        conn.execute("SELECT pg_sleep(30)")  # a build that would hold the key for a long time

    monkeypatch.setattr(space_indexes, "_run", long_build)
    monkeypatch.setattr(migrate_main, "upgrade_locked", lambda settings: None)

    index_run, index_errors = _in_thread(index_main.run)
    assert building.wait(timeout=10)
    started = time.monotonic()
    migrate_main.run()  # terminates the build's session, then "upgrades"
    index_run.join(timeout=10)

    assert not index_run.is_alive()
    assert time.monotonic() - started < 10  # did not wait out the 30 s build
    assert index_errors == []  # preempted is exit 0, not "database unreachable" (exit 1)


def test_migrate_never_terminates_a_non_index_session_holding_the_key(
    entrypoints: None, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Another migrate run's pause session holds the same advisory key; it is identified by its
    application_name and left alone. Here: a plain session holding the key makes the pause
    give up after its deadline instead of killing it."""
    dsn = settings.dsn("migrator", driver="postgresql")
    with psycopg.connect(dsn, autocommit=True) as other:
        other.execute("SELECT pg_advisory_lock(%s)", (space_indexes.ADVISORY_LOCK_KEY,))
        with (
            pytest.raises(migrate_main.IndexBuildNotStopped),
            migrate_main.pause_index_builds(settings, deadline_s=0.5),
        ):
            pass
        assert other.execute("SELECT 1").fetchone() == (1,)  # still alive
        other.execute("SELECT pg_advisory_unlock(%s)", (space_indexes.ADVISORY_LOCK_KEY,))


@pytest.fixture
def schema_ahead(superuser: Conn) -> Iterator[None]:
    """Simulate a rollback: the database carries a revision this code does not know."""
    row = superuser.execute("SELECT version_num FROM rf.alembic_version").fetchone()
    meta = superuser.execute("SELECT min_code_version FROM rf.schema_meta").fetchone()
    assert row is not None
    assert meta is not None
    superuser.execute("UPDATE rf.alembic_version SET version_num = '9999'")
    try:
        yield
    finally:
        superuser.execute("UPDATE rf.alembic_version SET version_num = %s", (row[0],))
        superuser.execute("UPDATE rf.schema_meta SET min_code_version = %s", (meta[0],))


def test_an_older_image_on_a_newer_compatible_schema_passes_migrate(
    entrypoints: None, schema_ahead: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    upgraded: list[bool] = []
    monkeypatch.setattr(migrate_main, "upgrade_locked", lambda settings: upgraded.append(True))

    migrate_main.run()  # returns normally: exit 0

    assert upgraded == []  # no upgrade or checksum step against an unknown revision


def test_an_older_image_on_an_incompatible_schema_fails_migrate(
    entrypoints: None, schema_ahead: None, superuser: Conn
) -> None:
    superuser.execute("UPDATE rf.schema_meta SET min_code_version = %s", (CODE_VERSION + 1,))
    with pytest.raises(SystemExit) as exit_info:
        migrate_main.run()
    assert exit_info.value.code == 1
