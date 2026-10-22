"""Programmatic Alembic runner (no alembic.ini needed; migrations ship inside the package)."""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from importlib.resources import files
from typing import Any

import psycopg
from alembic import command
from alembic.config import Config
from alembic.script import ScriptDirectory

from ragforge_contracts.api import Readiness
from ragforge_core.config import Settings
from ragforge_core.db.engine import sync_conninfo
from ragforge_core.db.readiness import evaluate_readiness


def alembic_config(settings: Settings) -> Config:
    cfg = Config()
    cfg.set_main_option("script_location", str(files("ragforge_core.db").joinpath("migrations")))
    # ConfigParser interpolation: escape % in URL-encoded passwords.
    cfg.set_main_option("sqlalchemy.url", settings.dsn("migrator").replace("%", "%%"))
    return cfg


class MigrationEdited(RuntimeError):
    """An already-applied migration file changed. Alembic would silently skip it."""


def migration_checksums() -> dict[str, str]:
    """revision id (filename prefix, e.g. '0002') -> sha256 of the migration source."""
    versions = files("ragforge_core.db").joinpath("migrations", "versions")
    out: dict[str, str] = {}
    for entry in versions.iterdir():
        if entry.name.endswith(".py") and entry.name[:4].isdigit():
            out[entry.name[:4]] = hashlib.sha256(entry.read_bytes()).hexdigest()
    return out


# Session advisory lock serializing concurrent `ragforge-migrate` runs (two replicas of a deploy
# job, a retried k8s Job): the second waits, then finds nothing to apply, instead of failing on
# "column already exists" or the migrator's 5 s lock_timeout halfway through a revision.
MIGRATE_LOCK_KEY = 0x7266_6D69_6772  # "rfmigr"
MIGRATE_LOCK_WAIT = "10min"


def _connect(settings: Settings, *, autocommit: bool = False) -> psycopg.Connection[Any]:
    """Connect timeout + TCP keepalives (db.engine.CONNECT_ARGS): a network partition must fail
    the one-shot, not hang it, and must not silently drop the serializing advisory lock."""
    return psycopg.connect(
        sync_conninfo(settings.dsn("migrator", driver="postgresql")), autocommit=autocommit
    )


@contextmanager
def migration_lock(settings: Settings) -> Iterator[None]:
    """Serialize `ragforge-migrate` runs (waits up to MIGRATE_LOCK_WAIT). Taken FIRST, before
    anything else the run does (pausing index builds included), so a second run waits here
    instead of acting on the first run's sessions."""
    with _connect(settings, autocommit=True) as lock:
        lock.execute(f"SET lock_timeout = '{MIGRATE_LOCK_WAIT}'")
        lock.execute("SELECT pg_advisory_lock(%s)", (MIGRATE_LOCK_KEY,))
        try:
            yield
        finally:
            # A dead session already had its lock released by the server.
            if not lock.closed:
                with suppress(psycopg.Error):
                    lock.execute("SELECT pg_advisory_unlock(%s)", (MIGRATE_LOCK_KEY,))


def upgrade_to_head(settings: Settings) -> None:
    """Serialized upgrade (see `upgrade_locked`)."""
    with migration_lock(settings):
        upgrade_locked(settings)


def upgrade_locked(settings: Settings) -> None:
    """Call with `migration_lock` held. Verify the applied migrations' checksums BEFORE upgrading
    (an edited applied revision must stop the deploy before any newer revision applies on top of
    it), then upgrade, then record the new checksums (env.py also records each one inside its
    own revision's tx)."""
    _record_and_verify_checksums(settings, record=False)
    command.upgrade(alembic_config(settings), "head")
    _record_and_verify_checksums(settings)


def schema_ahead_of_code(settings: Settings) -> str | None:
    """The database's Alembic revision when THIS code does not know it (a rollback to an older
    image after a newer release migrated), else None. Expand/contract (ADR-0002) lets old code
    run on a newer schema as long as `code_version >= min_code_version`; such a run must not try
    to upgrade (Alembic cannot resolve an unknown revision) or verify checksums (newer applied
    revisions are missing from this code by design)."""
    known = {
        rev.revision
        for rev in ScriptDirectory.from_config(alembic_config(settings)).walk_revisions()
    }
    with _connect(settings) as conn:
        conn.execute("SET ROLE rf_system")
        exists = conn.execute("SELECT to_regclass('rf.alembic_version')").fetchone()
        if exists is None or exists[0] is None:
            return None  # fresh database
        rows = conn.execute("SELECT version_num FROM rf.alembic_version").fetchall()
    unknown = sorted(str(r[0]) for r in rows if str(r[0]) not in known)
    return unknown[0] if unknown else None


def _record_and_verify_checksums(settings: Settings, *, record: bool = True) -> None:
    current = migration_checksums()
    with _connect(settings) as conn:
        conn.execute("SET ROLE rf_system")
        exists = conn.execute("SELECT to_regclass('rf.migration_checksums')").fetchone()
        if exists is None or exists[0] is None:
            return  # fresh database: nothing applied yet, nothing to verify
        rows = conn.execute("SELECT revision, sha256 FROM rf.migration_checksums").fetchall()
        stored: dict[str, str] = {str(r): str(sha) for r, sha in rows}
        changed = sorted(r for r, sha in current.items() if r in stored and stored[r] != sha)
        missing = sorted(set(stored) - set(current))
        if missing:
            raise MigrationEdited(f"applied migration(s) {missing} no longer exist in the code")
        if changed:
            raise MigrationEdited(
                f"applied migration(s) {changed} were edited after being applied, so Alembic will "
                "not re-run them. Dev: `make nuke && make up`. Shared/prod: never edit an applied "
                "migration; add a new one."
            )
        if not record:
            return
        for revision, sha in current.items():
            if revision not in stored:
                conn.execute(
                    "INSERT INTO rf.migration_checksums (revision, sha256) VALUES (%s, %s)",
                    (revision, sha),
                )


def verify_schema(settings: Settings) -> Readiness:
    """Read schema_meta as the migrator. The migrator does not inherit owner privileges
    (GRANT ... WITH INHERIT FALSE), so it must SET ROLE explicitly, exactly like env.py."""
    with _connect(settings) as conn:
        conn.execute("SET ROLE rf_system")
        row = conn.execute("SELECT schema_version, min_code_version FROM rf.schema_meta").fetchone()
    if row is None:
        return evaluate_readiness(None, None)
    return evaluate_readiness(int(row[0]), int(row[1]))
