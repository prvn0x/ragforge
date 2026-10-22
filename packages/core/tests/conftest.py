"""Shared fixtures: one real Postgres (pgvector) per test session, bootstrapped + migrated exactly
as production does it. Phase 1 modules reuse these fixtures; do not fork them."""

from __future__ import annotations

import base64
import os
import secrets
from collections.abc import AsyncIterator, Iterator
from uuid import UUID, uuid4

import psycopg
import pytest
from pydantic import SecretStr
from sqlalchemy.ext.asyncio import AsyncEngine
from testcontainers.community.postgres import PostgresContainer

from pgtest import connect
from ragforge_core.config import Settings
from ragforge_core.db.bootstrap import bootstrap_database, wait_until_accepting
from ragforge_core.db.engine import make_engine
from ragforge_core.db.migrate import upgrade_to_head

# Rancher Desktop / Colima expose the socket at a host path the Ryuk reaper cannot mount; the
# path *inside* the VM is the standard one (also correct on Linux CI).
os.environ.setdefault("TESTCONTAINERS_DOCKER_SOCKET_OVERRIDE", "/var/run/docker.sock")

PGVECTOR_IMAGE = (
    "pgvector/pgvector:pg17@sha256:ac08538c6f8b9904c33c8224c5e5706dbe760aca29db1d096972b4052c22a75d"
)


def _pw() -> SecretStr:
    return SecretStr(secrets.token_urlsafe(24))


@pytest.fixture(scope="session")
def postgres() -> Iterator[PostgresContainer]:
    with PostgresContainer(
        PGVECTOR_IMAGE, username="postgres", password="postgres-test-only", dbname="ragforge"
    ) as pg:
        yield pg


@pytest.fixture(scope="session")
def settings(postgres: PostgresContainer) -> Settings:
    s = Settings(
        _env_file=None,
        db_host=postgres.get_container_host_ip(),
        db_port=int(postgres.get_exposed_port(5432)),
        db_name="ragforge",
        db_superuser="postgres",
        db_superuser_password=SecretStr("postgres-test-only"),
        db_migrator_password=_pw(),
        db_api_password=_pw(),
        db_worker_password=_pw(),
        db_metrics_password=_pw(),
        db_backup_password=_pw(),
        db_pool_size=5,
        kek=SecretStr(base64.b64encode(os.urandom(32)).decode()),
        log_json=False,
    )
    wait_until_accepting(s)
    bootstrap_database(s)
    target = os.environ.get("RAGFORGE_TEST_TARGET_REVISION")
    if target:  # builder convenience: stop at your own slot while iterating (docs/PHASE1.md)
        from alembic import command

        from ragforge_core.db.migrate import alembic_config

        command.upgrade(alembic_config(s), target)
    else:
        upgrade_to_head(s)
    return s


@pytest.fixture
def superuser(settings: Settings) -> Iterator[psycopg.Connection[tuple[object, ...]]]:
    with connect(settings, "superuser") as conn:
        conn.autocommit = True
        yield conn


@pytest.fixture
async def api_engine(settings: Settings) -> AsyncIterator[AsyncEngine]:
    engine = make_engine(settings, "api")
    yield engine
    await engine.dispose()


@pytest.fixture
def make_tenant(superuser: psycopg.Connection[tuple[object, ...]]) -> object:
    """Factory: create a tenant + user + collection (as superuser). Returns (tenant, user, coll)."""

    def _make(name: str | None = None) -> tuple[UUID, UUID, UUID]:
        tenant, user, coll = uuid4(), uuid4(), uuid4()
        superuser.execute(
            "INSERT INTO rf.tenants (id, name, kind) VALUES (%s, %s, 'personal')",
            (tenant, name or f"t-{tenant.hex[:6]}"),
        )
        superuser.execute(
            "INSERT INTO rf.users (id, tenant_id, display_name) VALUES (%s, %s, 'u')",
            (user, tenant),
        )
        superuser.execute(
            "INSERT INTO rf.collections (id, tenant_id, name, active_space_id) "
            "VALUES (%s, %s, %s, 1)",
            (coll, tenant, f"c-{coll.hex[:6]}"),
        )
        return tenant, user, coll

    return _make
