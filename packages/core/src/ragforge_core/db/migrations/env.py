"""Alembic environment. Runs as rf_migrator and `SET ROLE rf_system`, so every object is
owned by the NOLOGIN owner role (ADR-0013). Each migration runs in its own transaction;
migrations needing CONCURRENTLY must use `op.get_context().autocommit_block()`.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping
from typing import Any

from alembic import context
from alembic.runtime.migration import MigrationContext, MigrationInfo
from sqlalchemy import create_engine, pool

from ragforge_core.config import get_settings
from ragforge_core.db.engine import CONNECT_ARGS
from ragforge_core.db.migrate import migration_checksums

url = context.config.get_main_option("sqlalchemy.url") or get_settings().dsn("migrator")


def _record_checksum(
    ctx: MigrationContext,
    step: MigrationInfo,
    heads: Collection[Any],
    run_args: Mapping[str, Any],
) -> None:
    """Runs inside the migration's own transaction, so a crash can never leave an applied
    revision without its checksum."""
    revision = step.up_revision_id
    if not revision or not step.is_upgrade:
        return
    sha = migration_checksums()[str(revision)]
    connection = context.get_bind()
    connection.exec_driver_sql(
        "INSERT INTO rf.migration_checksums (revision, sha256) VALUES (%(r)s, %(s)s) "
        "ON CONFLICT (revision) DO NOTHING",
        {"r": str(revision), "s": sha},
    )


def run_migrations_online() -> None:
    # Connect timeout + TCP keepalives: a network partition must fail the one-shot, not hang it.
    engine = create_engine(url, poolclass=pool.NullPool, connect_args=dict(CONNECT_ARGS))
    with engine.connect() as connection:
        connection.exec_driver_sql("SET ROLE rf_system")
        connection.exec_driver_sql("SET search_path = rf, ext")
        connection.commit()
        context.configure(
            connection=connection,
            version_table="alembic_version",
            version_table_schema="rf",
            transaction_per_migration=True,
            on_version_apply=_record_checksum,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    raise SystemExit("Offline migrations are not supported; run against a live database.")
run_migrations_online()
