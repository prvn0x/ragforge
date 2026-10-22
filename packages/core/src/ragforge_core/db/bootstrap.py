"""Run bootstrap.sql as a superuser/admin (ADR-0013). Idempotent."""

from __future__ import annotations

import time
from importlib.resources import files

import psycopg
from psycopg import sql

from ragforge_core.config import Settings

_ROLE_PASSWORDS = {
    "migrator_password": "db_migrator_password",
    "api_password": "db_api_password",
    "worker_password": "db_worker_password",
    "metrics_password": "db_metrics_password",
    "backup_password": "db_backup_password",
}


_ROLE_NAMES = {
    "migrator_password": "rf_migrator",
    "api_password": "rf_api",
    "worker_password": "rf_worker",
    "metrics_password": "rf_metrics",
    "backup_password": "rf_backup",
}


def render_bootstrap_sql(
    settings: Settings, conn: psycopg.Connection[tuple[object, ...]]
) -> sql.Composed:
    """Passwords are SCRAM-hashed client-side (libpq), so plaintext never reaches the server's
    statement logs (log_statement=ddl, pgaudit)."""
    template = files("ragforge_core.db").joinpath("bootstrap.sql").read_text(encoding="utf-8")
    params: dict[str, sql.Composable] = {"dbname": sql.Identifier(settings.db_name)}
    for placeholder, attr in _ROLE_PASSWORDS.items():
        secret = getattr(settings, attr).get_secret_value()
        if len(secret) < 16:
            raise ValueError(f"{attr} must be at least 16 characters (run scripts/bootstrap.sh)")
        hashed = conn.pgconn.encrypt_password(
            secret.encode(), _ROLE_NAMES[placeholder].encode(), b"scram-sha-256"
        )
        params[placeholder] = sql.Literal(hashed.decode())
    return sql.SQL(template).format(**params)


def ensure_database(settings: Settings) -> None:
    """Create the application database if missing (connects to the maintenance DB)."""
    maint = settings.model_copy(update={"db_name": "postgres"})
    with psycopg.connect(maint.dsn("superuser", driver="postgresql"), autocommit=True) as conn:
        exists = conn.execute(
            "SELECT 1 FROM pg_database WHERE datname = %s", (settings.db_name,)
        ).fetchone()
        if not exists:
            conn.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(settings.db_name)))


def wait_until_accepting(settings: Settings, timeout_s: float = 60.0) -> None:
    """Wait for a real query over TCP. The postgres image restarts once after initdb, so a
    first successful connection can be followed by a refusal; retry until stable."""
    deadline = time.monotonic() + timeout_s
    maint = settings.model_copy(update={"db_name": "postgres"})
    while True:
        try:
            with psycopg.connect(
                maint.dsn("superuser", driver="postgresql"), connect_timeout=2
            ) as c:
                c.execute("SELECT 1")
                return
        except psycopg.OperationalError:
            if time.monotonic() > deadline:
                raise
            time.sleep(0.5)


def bootstrap_database(settings: Settings) -> None:
    wait_until_accepting(settings)
    ensure_database(settings)
    with psycopg.connect(settings.dsn("superuser", driver="postgresql"), autocommit=True) as conn:
        conn.execute(render_bootstrap_sql(settings, conn))
