"""Build or drop each embedding space's HNSW index (the `ragforge-index` one-shot).

`CREATE/DROP INDEX CONCURRENTLY` cannot run inside a transaction or a function, so it cannot
live in a migration slot or in rf.gc_embeddings (handoff B, request 4). It runs on an
autocommit connection as rf_migrator after `SET ROLE rf_system` (the owner of rf.embeddings;
the migrator has INHERIT FALSE). Both adapter calls are idempotent, so every deploy re-runs it:
a new space gets its index, an interrupted build is finished, and a retired space's empty
index is dropped. Until an index is valid, rf.search_vectors serves that space exactly.

The step is best-effort and off the deploy's critical path (nothing depends on it): an HNSW
index is a performance structure, and exact search serves the space meanwhile.

Session bounds for the build connection (finite on purpose):
- `lock_timeout = 60s`: CONCURRENTLY waits for every older writer and snapshot of
  rf.embeddings, so it would otherwise wait indefinitely behind a long pg_dump (rf_backup has no
  statement timeout) or an anti-wraparound autovacuum. Past the bound the space is DEFERRED:
  logged, retried next run, and the INVALID leftover is dropped best-effort so inserts stop
  paying to maintain it.
- `statement_timeout = 20min`: an overall deadline per build; a build that cannot finish in it
  is deferred the same way (raise `maintenance_work_mem` before retrying a large space). Kept
  short on purpose: the index is an expression index with a partial predicate, so its build
  snapshot pins the xmin horizon for the whole build, and vacuum cannot clean the high-churn
  tables (rf.jobs, rf.connector_inflight, the rate-limit buckets) meanwhile; and the build's
  SHARE UPDATE EXCLUSIVE lock on rf.embeddings would make a migration touching that table fail
  on its 5 s lock_timeout (`ragforge-migrate` therefore terminates a running build first; the
  build is resumable on the next run).
- `maintenance_work_mem = 256MB`: the HNSW graph is built in memory; past this pgvector
  falls back to a slower on-disk build (it never fails). Sized for the local 5 GB VM.
- `max_parallel_maintenance_workers = 0`: a parallel HNSW build places its graph in dynamic
  shared memory, and the compose Postgres runs with `shm_size: 256mb`.

Concurrent runs (two deploys, a rolling restart) are serialized by a session advisory lock
taken with try-lock: the run that loses skips the step, because the winner builds.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final

import psycopg
from psycopg import errors, sql

from ragforge_core.adapters.db.search import (
    IndexBuildInProgress,
    build_space_index,
    drop_retired_space_index,
    space_index_name,
)
from ragforge_core.config import Settings
from ragforge_core.db.engine import sync_conninfo
from ragforge_core.observability.logging import get_logger

log = get_logger(__name__)

Conn = psycopg.Connection[tuple[object, ...]]

# Arbitrary fixed bigint: every release must use the same key for runs to serialize.
ADVISORY_LOCK_KEY: Final = 7_241_905_316_582_011_201
# `ragforge-migrate` terminates only sessions with this application_name (an index build), never
# another migrate run that happens to hold the same key while pausing builds.
APPLICATION_NAME: Final = "ragforge-index"
LOCK_TIMEOUT: Final = "60s"
STATEMENT_TIMEOUT: Final = "20min"
MAINTENANCE_WORK_MEM: Final = "256MB"
# Waiting out a bound means "not now", not "broken": the space is retried on the next run.
_DEFERRABLE = (errors.LockNotAvailable, errors.QueryCanceled)


@dataclass(slots=True)
class SpaceIndexReport:
    built: list[int] = field(default_factory=list)  # valid index present after the step
    dropped: list[int] = field(default_factory=list)
    skipped: list[int] = field(default_factory=list)
    deferred: list[int] = field(default_factory=list)  # hit a bound; exact search serves it
    failed: list[int] = field(default_factory=list)
    locked_out: bool = False  # another run holds the step lock; it does the work


def ensure_space_indexes(
    settings: Settings,
    *,
    lock_timeout: str = LOCK_TIMEOUT,
    statement_timeout: str = STATEMENT_TIMEOUT,
) -> SpaceIndexReport:
    report = SpaceIndexReport()
    dsn = settings.dsn("migrator", driver="postgresql")
    with psycopg.connect(
        sync_conninfo(dsn), autocommit=True, application_name=APPLICATION_NAME
    ) as conn:
        conn.execute("SET ROLE rf_system")
        _apply_session_bounds(conn, lock_timeout, statement_timeout)
        got = conn.execute("SELECT pg_try_advisory_lock(%s)", (ADVISORY_LOCK_KEY,)).fetchone()
        if got is None or not got[0]:
            report.locked_out = True
            log.info("space_index_step_skipped", reason="another_run_holds_lock")
            return report
        try:
            _run(conn, report)
        finally:
            _unlock(conn)
    return report


def _unlock(conn: Conn) -> None:
    """Never mask the step's own error: after `ragforge-migrate` terminated this session the
    connection is dead, and the server already released the session lock with it."""
    if conn.closed:
        return
    try:
        conn.execute("SELECT pg_advisory_unlock(%s)", (ADVISORY_LOCK_KEY,))
    except psycopg.Error as exc:
        log.warning("space_index_unlock_failed", error=type(exc).__name__)


def is_preemption(exc: BaseException) -> bool:
    """`ragforge-migrate` terminated this session (pg_terminate_backend -> SQLSTATE 57P01)."""
    return isinstance(exc, errors.AdminShutdown) or (
        isinstance(exc, psycopg.OperationalError)
        and "terminating connection due to administrator command" in str(exc)
    )


def _apply_session_bounds(conn: Conn, lock_timeout: str, statement_timeout: str) -> None:
    for name, value in (
        ("lock_timeout", lock_timeout),
        ("statement_timeout", statement_timeout),
        ("maintenance_work_mem", MAINTENANCE_WORK_MEM),
        ("max_parallel_maintenance_workers", "0"),
    ):
        conn.execute(sql.SQL("SET {} = {}").format(sql.Identifier(name), sql.Literal(value)))


def _run(conn: Conn, report: SpaceIndexReport) -> None:
    spaces = conn.execute(
        "SELECT id, status = 'retired' FROM rf.embedding_spaces ORDER BY id"
    ).fetchall()
    for space_id, retired in spaces:
        sid = int(str(space_id))
        try:
            _one(conn, sid, bool(retired), report)
        except IndexBuildInProgress:
            report.skipped.append(sid)
            log.info("space_index_skipped", space_id=sid, reason="build_in_progress")
        except _DEFERRABLE as exc:
            report.deferred.append(sid)
            log.warning("space_index_deferred", space_id=sid, error=type(exc).__name__)
            _drop_invalid_leftover(conn, sid)
        except Exception as exc:  # best effort per space; exact search serves it meanwhile
            if conn.closed or is_preemption(exc):
                raise  # the session is gone (terminated by migrate, or dropped): stop the step
            report.failed.append(sid)
            log.error("space_index_failed", space_id=sid, error=type(exc).__name__)


def _drop_invalid_leftover(conn: Conn, sid: int) -> None:
    """A cancelled CONCURRENTLY build leaves an INVALID index that every insert still maintains
    (an HNSW graph insertion each). Drop it under the same bounds; if that waits out the lock
    bound too, the next run's build drops and rebuilds it."""
    name = space_index_name(sid)
    row = conn.execute(
        "SELECT NOT i.indisvalid FROM pg_index i WHERE i.indexrelid = to_regclass(%s)",
        (f"rf.{name}",),
    ).fetchone()
    if row is None or not row[0]:
        return
    try:
        conn.execute(
            sql.SQL("DROP INDEX CONCURRENTLY IF EXISTS rf.{}").format(sql.Identifier(name))
        )
    except _DEFERRABLE as exc:
        log.warning("space_index_leftover_kept", space_id=sid, error=type(exc).__name__)


def _one(conn: Conn, sid: int, retired: bool, report: SpaceIndexReport) -> None:
    if retired:
        if drop_retired_space_index(conn, sid):
            report.dropped.append(sid)
            log.info("space_index_dropped", space_id=sid)
        else:  # still referenced by a collection: keep it until it is not
            report.skipped.append(sid)
            log.info("space_index_skipped", space_id=sid, reason="retired_in_use")
    elif build_space_index(conn, sid):
        report.built.append(sid)
        log.info("space_index_ready", space_id=sid)
    else:  # dim above the HNSW limit: exact search only
        report.skipped.append(sid)
        log.info("space_index_skipped", space_id=sid, reason="dim_over_hnsw_limit")
