"""Async engines and tenant-scoped transactions (ADR-0008, ADR-0013).

Every request/job that touches tenant data MUST use `tenant_transaction`. The tenant is set
with `set_config('app.tenant_id', $1, true)`: transaction-local and bind-parameter safe, so
a pooled connection can never carry a tenant into the next transaction.
"""

from __future__ import annotations

import weakref
from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from uuid import UUID

import psycopg.conninfo
from sqlalchemy import QueuePool, text
from sqlalchemy.ext.asyncio import (
    AsyncConnection,
    AsyncEngine,
    create_async_engine,
)

from ragforge_core.config import Settings
from ragforge_core.observability import metrics

_SET_TENANT = text("SELECT set_config('app.tenant_id', :tenant_id, true)")


# Fail fast on pool exhaustion: a caller waiting 30 s (the default) for a connection outlives the
# job lease heartbeat interval and the API's own request budget.
POOL_TIMEOUT_S = 10

# libpq connection parameters for every app connection. Without TCP keepalives a half-open
# connection (a NAT/LB dropping state, a VM pause) blocks a read forever: the server-side
# statement_timeout never fires because the server no longer sees the session. Dead peers are
# detected in ~60 s (30 s idle + 3 probes * 10 s); connect attempts give up after 10 s.
CONNECT_ARGS: dict[str, int] = {
    "connect_timeout": 10,
    "keepalives": 1,
    "keepalives_idle": 30,
    "keepalives_interval": 10,
    "keepalives_count": 3,
}


def sync_conninfo(dsn: str) -> str:
    """A libpq conninfo for plain psycopg connections (one-shots, LISTEN) with CONNECT_ARGS."""
    return psycopg.conninfo.make_conninfo(dsn, **CONNECT_ARGS)


def make_engine(settings: Settings, role: str) -> AsyncEngine:
    pool_size = settings.worker_pool_size if role == "worker" else settings.db_pool_size
    engine = create_async_engine(
        settings.dsn(role),
        pool_size=pool_size,
        max_overflow=0,
        pool_timeout=POOL_TIMEOUT_S,
        connect_args=dict(CONNECT_ARGS),
        pool_pre_ping=True,
        pool_recycle=1800,
        # A DBAPIError message otherwise carries its bound parameters (API key digests, sealed
        # connector tokens) into logs and tracebacks (handoff E2, 1.4).
        hide_parameters=True,
    )
    track_pool(engine, pool=role)
    return engine


def _pool_reading(
    pool: weakref.ref[QueuePool], read: Callable[[QueuePool], int]
) -> Callable[[], float]:
    """A gauge callback that holds the pool WEAKLY: an engine disposed and dropped (tests build
    many) reads 0 instead of being kept alive by its metrics."""

    def value() -> float:
        current = pool()
        return float(read(current)) if current is not None else 0.0

    return value


def track_pool(engine: AsyncEngine, *, pool: str) -> None:
    """`ragforge_db_pool_checked_out` / `ragforge_db_pool_size` {process, pool}, read at scrape
    time from the pool itself (nothing to keep in step). The newest engine of a (process, pool)
    pair wins the series, which in production is the only one."""
    target = engine.pool
    if not isinstance(target, QueuePool):  # NullPool/StaticPool have nothing to gauge
        return
    ref = weakref.ref(target)
    labels = (metrics.process(), metrics.ident_label(pool))
    metrics.DB_POOL_CHECKED_OUT.labels(*labels).set_function(
        _pool_reading(ref, QueuePool.checkedout)
    )
    metrics.DB_POOL_SIZE.labels(*labels).set_function(_pool_reading(ref, QueuePool.size))


@asynccontextmanager
async def tenant_transaction(
    engine: AsyncEngine, tenant_id: UUID
) -> AsyncIterator[AsyncConnection]:
    """Open a transaction scoped to `tenant_id`. RLS filters every tenant table inside it."""
    async with engine.begin() as conn:
        await conn.execute(_SET_TENANT, {"tenant_id": str(tenant_id)})
        yield conn


class SqlTransactionManager:
    """Adapter for the domain `TransactionManager` port. The yielded `Tx` is an AsyncConnection."""

    def __init__(self, engine: AsyncEngine) -> None:
        self._engine = engine

    def tenant(self, tenant_id: UUID) -> AbstractAsyncContextManager[AsyncConnection]:
        return tenant_transaction(self._engine, tenant_id)

    def system(self) -> AbstractAsyncContextManager[AsyncConnection]:
        return system_transaction(self._engine)


@asynccontextmanager
async def system_transaction(engine: AsyncEngine) -> AsyncIterator[AsyncConnection]:
    """A transaction with NO tenant set: tenant tables return zero rows (fail closed).

    Use only to call the named cross-tenant SECURITY DEFINER functions (ADR-0013) or to read
    global tables (schema_meta, embedding_spaces).
    """
    async with engine.begin() as conn:
        yield conn
