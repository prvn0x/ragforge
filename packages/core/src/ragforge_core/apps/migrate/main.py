"""`ragforge-migrate`: apply migrations as rf_migrator (SET ROLE rf_system), then verify.

Schema only, and fatal on failure (exit 1): api and worker start only after this completes, so
it must never wait on anything but the schema. The per-space HNSW index step is a separate,
non-gating one-shot (`ragforge-index`, apps/index): an index build can wait a long time on
older snapshots, and exact search serves a space until its index is valid.

The two can collide across deploys: a build still running from the previous deploy holds SHARE
UPDATE EXCLUSIVE on rf.embeddings for up to its statement bound, and a migration touching that
table would fail on the migrator's 5 s lock_timeout, keeping api and worker down. So migrate
first takes the index step's advisory lock (`pause_index_builds`): if a build holds it, that
session is TERMINATED (not cancelled: a cancel only aborts the current statement, and the step
would move on to its next space while still holding the lock). A terminated CONCURRENTLY build
leaves an INVALID index that the next `ragforge-index` run drops and rebuilds. Holding the lock
for the whole upgrade also keeps a new index run from starting mid-migration.

Ordering between concurrent migrate runs (a retried k8s Job, two deploy replicas): each run
takes the migration lock (db.migrate.MIGRATE_LOCK_KEY) BEFORE pausing index builds, so a second
run waits there. Only sessions whose application_name is the index step's are terminated, never
another migrate run's pause connection, which holds the same advisory key.

Rollback to an older image: when the database's revision is unknown to this code, the schema is
AHEAD (a newer release migrated). Upgrading or verifying checksums would fail by design, so the
run only checks compatibility (`code_version >= min_code_version`, ADR-0002) and exits 0 when
this code may run on that schema, 1 otherwise.
"""

from __future__ import annotations

import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import psycopg

from ragforge_core.apps.index.space_indexes import ADVISORY_LOCK_KEY, APPLICATION_NAME
from ragforge_core.config import Settings, get_settings
from ragforge_core.db.engine import sync_conninfo
from ragforge_core.db.migrate import (
    migration_lock,
    schema_ahead_of_code,
    upgrade_locked,
    verify_schema,
)
from ragforge_core.observability.logging import configure_logging, get_logger

log = get_logger(__name__)

PAUSE_DEADLINE_S = 60.0
PAUSE_POLL_S = 0.2

# pg_locks shows a bigint advisory key as (classid = high 32 bits, objid = low 32 bits,
# objsubid = 1). Only sessions of the caller's own role can be signalled, and the index step
# runs as rf_migrator too (SET ROLE does not change the session's role for this check). Only
# index-step sessions qualify: another migrate run pausing builds holds the same key.
_HOLDERS = """
    SELECT l.pid FROM pg_locks l
      JOIN pg_stat_activity a ON a.pid = l.pid
     WHERE l.locktype = 'advisory' AND l.granted AND l.objsubid = 1
       AND l.classid::bigint = %(hi)s AND l.objid::bigint = %(lo)s
       AND l.pid <> pg_backend_pid()
       AND a.application_name = %(app)s
"""


class IndexBuildNotStopped(RuntimeError):
    """The index step kept its lock past the deadline: migrating now could fail mid-way."""


def _key_parts(key: int) -> dict[str, int]:
    unsigned = key & 0xFFFF_FFFF_FFFF_FFFF
    return {"hi": unsigned >> 32, "lo": unsigned & 0xFFFF_FFFF}


@contextmanager
def pause_index_builds(
    settings: Settings, *, deadline_s: float = PAUSE_DEADLINE_S
) -> Iterator[None]:
    dsn = settings.dsn("migrator", driver="postgresql")
    conn: psycopg.Connection[Any]
    with psycopg.connect(sync_conninfo(dsn), autocommit=True) as conn:
        give_up = time.monotonic() + deadline_s
        while True:
            got = conn.execute("SELECT pg_try_advisory_lock(%s)", (ADVISORY_LOCK_KEY,)).fetchone()
            if got is not None and got[0]:
                break
            if time.monotonic() > give_up:
                raise IndexBuildNotStopped("the index step still holds its lock")
            holders = conn.execute(
                _HOLDERS, {**_key_parts(ADVISORY_LOCK_KEY), "app": APPLICATION_NAME}
            ).fetchall()
            for (pid,) in holders:
                log.warning("migrate_terminating_index_build", pid=int(str(pid)))
                conn.execute("SELECT pg_terminate_backend(%s)", (pid,))
            time.sleep(PAUSE_POLL_S)
        try:
            yield
        finally:
            _release(conn)


def _release(conn: psycopg.Connection[Any]) -> None:
    """Never turn a finished upgrade into a failed deploy over the unlock: if the session is
    gone, the server already released its lock."""
    if conn.closed:
        return
    try:
        conn.execute("SELECT pg_advisory_unlock(%s)", (ADVISORY_LOCK_KEY,))
    except psycopg.Error as exc:
        log.warning("migrate_index_unlock_failed", error=type(exc).__name__)


def run() -> None:
    settings = get_settings()
    configure_logging(settings.log_level, json=settings.log_json)
    with migration_lock(settings):
        ahead = schema_ahead_of_code(settings)
        if ahead is None:
            with pause_index_builds(settings):
                upgrade_locked(settings)
    if ahead is not None:
        readiness = verify_schema(settings)
        log.warning(
            "migrate_schema_ahead",
            db_revision=ahead,
            compatible=readiness.ready,
            reason=readiness.reason,
        )
        if not readiness.ready:
            sys.exit(1)
        return
    readiness = verify_schema(settings)
    if not readiness.ready:
        log.error("migrate_incomplete", reason=readiness.reason)
        sys.exit(1)
    log.info("migrate_done", schema_version=readiness.schema_version)
