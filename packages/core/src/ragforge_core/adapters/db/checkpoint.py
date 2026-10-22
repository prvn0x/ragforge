"""FencedPostgresSaver: the LangGraph async Postgres saver with run-lease fencing (ADR-0008,
DESIGN section 8 "Run lifecycle").

PINNED: langgraph-checkpoint-postgres==3.2.0 (langgraph 1.2.15, langgraph-checkpoint 4.3.0).
This class relies on the library's `aput`/`aput_writes`/`aget_tuple` running every statement on
the connection the saver was constructed with. Re-run tests/module_d2 (the saver contract tests)
before changing the pin; `PINNED_LIBRARY_VERSION` is asserted there.

Guarantees:
- Fencing at the write itself. `aput` and `aput_writes` run in ONE transaction on a connection
  from the LangGraph pool: `rf.assert_run_lease` (row lock FOR UPDATE) -> library write ->
  advance `conversations.head_checkpoint_id` (root namespace only). A stalled pod that resumes
  after its lease was reclaimed fails at the first step, so nothing it wrote is ever committed.
  Holding the row lock until commit also linearizes the write before any takeover.
- Deterministic reads. A read without an explicit checkpoint id resolves the root namespace's
  `head_checkpoint_id`, never "latest by id" (uuid6 ordering depends on each pod's clock).
- Tenant isolation (application-level, ADR-0013: LangGraph tables are outside RLS). Every
  operation sets `app.tenant_id` transaction-locally from the config and first resolves the
  thread through `rf.conversations` under RLS; a thread of another tenant reads as empty and
  rejects writes. Thread-wide deletes/copies are disabled: delete the conversation row instead
  (a trigger in slot 0005 removes its checkpoints).

Config contract (built by `fenced_config`): `configurable.thread_id` = conversation id,
`configurable.rf_tenant_id` (always), `configurable.rf_run_id` (writes). They are UUID objects
on purpose: LangGraph copies str/int configurable values into checkpoint metadata, but not UUIDs.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, Final
from uuid import UUID

import psycopg
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    DeltaChannelHistory,
    get_checkpoint_id,
)
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.checkpoint.serde.base import SerializerProtocol
from langgraph.store.postgres.base import MIGRATIONS as STORE_MIGRATIONS
from psycopg import AsyncConnection
from psycopg.rows import DictRow
from psycopg_pool import AsyncConnectionPool

from ragforge_core.domain.errors import LeaseLost

PINNED_LIBRARY_VERSION: Final = "3.2.0"

CONFIG_TENANT_ID: Final = "rf_tenant_id"
CONFIG_RUN_ID: Final = "rf_run_id"
ROOT_NS: Final = ""

LEASE_LOST_SQLSTATE: Final = "RFL01"  # raised by rf.assert_run_lease (slot 0005)

_SET_TENANT = "SELECT set_config('app.tenant_id', %s, true)"
_ASSERT_LEASE = "SELECT rf.assert_run_lease(%s, %s)"
_READ_HEAD = "SELECT head_checkpoint_id FROM rf.conversations WHERE id = %s"
_ADVANCE_HEAD = (
    "UPDATE rf.conversations SET head_checkpoint_id = %s WHERE id = %s AND active_run_id = %s"
)
_CHECKPOINT_SCHEMA_VERSION = "SELECT max(v) AS v FROM rf.checkpoint_migrations"
_STORE_SCHEMA_VERSION = "SELECT max(v) AS v FROM rf.store_migrations"

LangGraphPool = AsyncConnectionPool[AsyncConnection[DictRow]]


class CheckpointSchemaMismatch(RuntimeError):
    """The database's LangGraph schema is not the one this library version expects."""


def fenced_config(
    conversation_id: UUID, tenant_id: UUID, run_id: UUID | None = None
) -> RunnableConfig:
    """The RunnableConfig the agent runner passes to the graph. `run_id` is required to write."""
    configurable: dict[str, Any] = {
        "thread_id": str(conversation_id),
        CONFIG_TENANT_ID: tenant_id,
    }
    if run_id is not None:
        configurable[CONFIG_RUN_ID] = run_id
    return {"configurable": configurable}


@dataclass(frozen=True, slots=True)
class _Scope:
    """Who is asking: the tenant (always) and, for writes, the run that must hold the lease."""

    tenant_id: UUID
    conversation_id: UUID
    run_id: UUID | None

    def require_run(self) -> UUID:
        if self.run_id is None:
            raise ValueError(f"checkpoint writes need configurable.{CONFIG_RUN_ID}")
        return self.run_id


def _as_uuid(value: object, key: str) -> UUID:
    if isinstance(value, UUID):
        return value
    if isinstance(value, str):
        try:
            return UUID(value)
        except ValueError:
            pass
    raise ValueError(f"configurable.{key} must be a UUID")


def _scope(config: RunnableConfig) -> _Scope:
    """Fail closed: a config without a tenant can never reach checkpoint rows."""
    configurable = config.get("configurable") or {}
    run = configurable.get(CONFIG_RUN_ID)
    return _Scope(
        tenant_id=_as_uuid(configurable.get(CONFIG_TENANT_ID), CONFIG_TENANT_ID),
        conversation_id=_as_uuid(configurable.get("thread_id"), "thread_id"),
        run_id=None if run is None else _as_uuid(run, CONFIG_RUN_ID),
    )


def _with_checkpoint_id(config: RunnableConfig, checkpoint_id: str) -> RunnableConfig:
    configurable = {**(config.get("configurable") or {}), "checkpoint_id": checkpoint_id}
    return {**config, "configurable": configurable}


class _Invisible(Exception):
    """The thread is not a conversation of the scoped tenant (or does not exist)."""


class FencedPostgresSaver(AsyncPostgresSaver):
    """AsyncPostgresSaver whose writes are fenced by the conversation run lease.

    Construct with the LangGraph pool (autocommit, dict_row, prepare_threshold=0; ADR-0008).
    Each fenced operation borrows one pooled connection and runs a plain AsyncPostgresSaver
    bound to it inside an explicit transaction, so the library's own SQL is reused unchanged
    and the parent instance never serializes callers on its single `lock`.
    """

    def __init__(self, pool: LangGraphPool, *, serde: SerializerProtocol | None = None) -> None:
        super().__init__(pool, serde=serde)
        self._pool = pool

    # ------------------------------------------------------------------ lifecycle
    async def setup(self) -> None:
        """Never run DDL at runtime (app roles cannot); the schema comes from slot 0005."""
        await self.assert_schema_ready()

    async def assert_schema_ready(self) -> None:
        """Startup check: the applied checkpoint AND store migrations match this library version
        exactly. `AsyncPostgresStore` is used without `setup()` too, so a library bump that adds
        a store migration (e.g. TTL columns) must stop startup, not fail at query time."""
        async with self._pool.connection() as conn:
            await _check_version(
                conn, _CHECKPOINT_SCHEMA_VERSION, len(self.MIGRATIONS) - 1, "checkpoint"
            )
            await _check_version(conn, _STORE_SCHEMA_VERSION, len(STORE_MIGRATIONS) - 1, "store")

    # ------------------------------------------------------------------ writes (fenced)
    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        scope = _scope(config)
        run_id = scope.require_run()
        async with self._fenced(scope, run_id) as conn:
            next_config = await self._bound(conn).aput(config, checkpoint, metadata, new_versions)
            if config["configurable"].get("checkpoint_ns", ROOT_NS) == ROOT_NS:
                await _advance_head(conn, scope.conversation_id, run_id, checkpoint["id"])
        return next_config

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        scope = _scope(config)
        async with self._fenced(scope, scope.require_run()) as conn:
            await self._bound(conn).aput_writes(config, writes, task_id, task_path)

    # ------------------------------------------------------------------ reads (head + tenant)
    async def aget_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        scope = _scope(config)
        try:
            async with self._visible(scope) as (conn, head):
                resolved = self._resolve(config, head)
                if resolved is None:
                    return None
                return await self._bound(conn).aget_tuple(resolved)
        except _Invisible:
            return None

    async def alist(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> AsyncIterator[CheckpointTuple]:
        """History of ONE visible thread. Listing without a thread would cross tenants.

        Streams from inside the read transaction instead of buffering: an unbounded history
        (limit=None on a long conversation) must not be held in memory. The read transaction
        takes no row locks, so a consumer that writes while iterating is not blocked; closing
        the iterator early (aclose / break in `async for`) ends the transaction and returns the
        connection to the pool."""
        if config is None:
            raise ValueError("alist requires a thread config (cross-thread listing is disabled)")
        scope = _scope(config)
        try:
            async with self._visible(scope) as (conn, _head):
                bound = self._bound(conn)
                async for item in bound.alist(config, filter=filter, before=before, limit=limit):
                    yield item
        except _Invisible:
            return

    async def aget_delta_channel_history(
        self, *, config: RunnableConfig, channels: Sequence[str]
    ) -> Mapping[str, DeltaChannelHistory]:
        scope = _scope(config)
        empty: dict[str, DeltaChannelHistory] = {ch: {"writes": []} for ch in channels}
        try:
            async with self._visible(scope) as (conn, head):
                resolved = self._resolve(config, head)
                if resolved is None:
                    return empty
                return await self._bound(conn).aget_delta_channel_history(
                    config=resolved, channels=channels
                )
        except _Invisible:
            return empty

    # ------------------------------------------------------------------ disabled thread-wide ops
    async def adelete_thread(self, thread_id: str) -> None:
        raise NotImplementedError("delete the conversation row; a trigger removes its checkpoints")

    async def adelete_for_runs(self, run_ids: Sequence[str]) -> None:
        raise NotImplementedError("run-scoped deletes are not tenant-scoped; not supported")

    async def acopy_thread(self, source_thread_id: str, target_thread_id: str) -> None:
        raise NotImplementedError("thread copies are not tenant-scoped; not supported")

    async def aprune(self, thread_ids: Sequence[str], *, strategy: str = "keep_latest") -> None:
        raise NotImplementedError("pruning would bypass head_checkpoint_id; not supported")

    # ------------------------------------------------------------------ internals
    def _bound(self, conn: AsyncConnection[DictRow]) -> AsyncPostgresSaver:
        """A library saver on this one connection (inside our transaction), same serde."""
        return AsyncPostgresSaver(conn, serde=self.serde)

    @staticmethod
    def _resolve(config: RunnableConfig, head: str | None) -> RunnableConfig | None:
        """Root-namespace reads without an id go to the head; None when there is no head yet.
        Subgraph namespaces keep the library's lookup (their rows are fenced like the root's)."""
        if get_checkpoint_id(config):
            return config
        if config["configurable"].get("checkpoint_ns", ROOT_NS) != ROOT_NS:
            return config
        return None if head is None else _with_checkpoint_id(config, head)

    @asynccontextmanager
    async def _tenant_tx(self, tenant_id: UUID) -> AsyncIterator[AsyncConnection[DictRow]]:
        """One explicit transaction on a pooled autocommit connection, tenant set LOCALLY so it
        can never leak to the connection's next borrower."""
        async with self._pool.connection() as conn, conn.transaction():
            await conn.execute(_SET_TENANT, (str(tenant_id),))
            yield conn

    @asynccontextmanager
    async def _fenced(self, scope: _Scope, run_id: UUID) -> AsyncIterator[AsyncConnection[DictRow]]:
        async with self._tenant_tx(scope.tenant_id) as conn:
            try:
                await conn.execute(_ASSERT_LEASE, (scope.conversation_id, run_id))
            except psycopg.Error as exc:
                if exc.sqlstate == LEASE_LOST_SQLSTATE:
                    raise LeaseLost("conversation run lease lost") from None
                raise
            yield conn

    @asynccontextmanager
    async def _visible(
        self, scope: _Scope
    ) -> AsyncIterator[tuple[AsyncConnection[DictRow], str | None]]:
        """Yields (connection, head) for a conversation visible under RLS; else _Invisible."""
        async with self._tenant_tx(scope.tenant_id) as conn:
            row = await (await conn.execute(_READ_HEAD, (scope.conversation_id,))).fetchone()
            if row is None:
                raise _Invisible
            yield conn, row["head_checkpoint_id"]


async def _check_version(
    conn: AsyncConnection[DictRow], query: str, expected: int, name: str
) -> None:
    row = await (await conn.execute(query)).fetchone()
    applied = None if row is None else row["v"]
    if applied != expected:
        raise CheckpointSchemaMismatch(f"{name} schema at v{applied}, library expects v{expected}")


async def _advance_head(
    conn: AsyncConnection[DictRow], conversation_id: UUID, run_id: UUID, checkpoint_id: str
) -> None:
    """Under RLS and the row lock taken by assert_run_lease; zero rows is impossible unless the
    fence is broken, so treat it as a lost lease (the transaction rolls back)."""
    cur = await conn.execute(_ADVANCE_HEAD, (checkpoint_id, conversation_id, run_id))
    if cur.rowcount != 1:
        raise LeaseLost("conversation run lease lost")
