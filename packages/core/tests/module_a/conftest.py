"""Module A fixtures. Builds on the shared fixtures in packages/core/tests/conftest.py."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from typing import cast
from uuid import UUID, uuid4

import psycopg
import pytest
from sqlalchemy.ext.asyncio import AsyncEngine

from ragforge_core.adapters.db.jobs import PgJobQueue
from ragforge_core.config import Settings
from ragforge_core.db.engine import SqlTransactionManager, make_engine

Conn = psycopg.Connection[tuple[object, ...]]
MakeTenant = Callable[..., tuple[UUID, UUID, UUID]]


@pytest.fixture(autouse=True)
def _empty_queue(superuser: Conn) -> None:
    """claim_jobs is cross-tenant: leftovers from earlier tests would be claimed. Start empty."""
    superuser.execute("DELETE FROM rf.jobs")
    superuser.execute("DELETE FROM rf.connector_inflight")


@pytest.fixture
async def worker_engine(settings: Settings) -> AsyncIterator[AsyncEngine]:
    engine = make_engine(settings.model_copy(update={"worker_db_pool_size": 20}), "worker")
    yield engine
    await engine.dispose()


@pytest.fixture
def queue(worker_engine: AsyncEngine) -> PgJobQueue:
    return PgJobQueue(worker_engine)


@pytest.fixture
def txm(worker_engine: AsyncEngine) -> SqlTransactionManager:
    return SqlTransactionManager(worker_engine)


@pytest.fixture
def tenant(make_tenant: object) -> UUID:
    return cast(MakeTenant, make_tenant)()[0]


@pytest.fixture
def make_connector(superuser: Conn, make_tenant: object) -> Callable[[], tuple[UUID, UUID]]:
    """Factory: (tenant_id, connector_id) for a fresh tenant with one active Notion connector."""

    def _make() -> tuple[UUID, UUID]:
        tenant_id, _, coll = cast(MakeTenant, make_tenant)()
        connector = uuid4()
        superuser.execute(
            "INSERT INTO rf.connectors (id, tenant_id, kind, target_collection_id, display_name) "
            "VALUES (%s, %s, 'notion', %s, 'c')",
            (connector, tenant_id, coll),
        )
        return tenant_id, connector

    return _make
