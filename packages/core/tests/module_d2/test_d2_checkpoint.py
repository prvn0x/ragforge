"""FencedPostgresSaver contract tests (ADR-0008): round trip, interrupt/resume, writes ordering,
fencing (stalled pod), head-based reads, tenant isolation, and the slot-0005 schema."""

from __future__ import annotations

import asyncio
import operator
from collections.abc import AsyncGenerator, Callable
from contextlib import aclosing
from importlib.metadata import version
from typing import Annotated, Any, TypedDict, cast
from uuid import UUID, uuid4

import psycopg
import pytest
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import Checkpoint, CheckpointTuple, empty_checkpoint
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.checkpoint.serde.types import ERROR
from langgraph.graph import END, START, StateGraph
from langgraph.store.postgres.aio import AsyncPostgresStore
from langgraph.store.postgres.base import MIGRATIONS as STORE_MIGRATIONS
from langgraph.types import Command, interrupt
from psycopg import AsyncConnection
from psycopg.rows import DictRow
from psycopg_pool import AsyncConnectionPool

from ragforge_core.adapters.db import checkpoint as checkpoint_module
from ragforge_core.adapters.db.checkpoint import (
    PINNED_LIBRARY_VERSION,
    CheckpointSchemaMismatch,
    FencedPostgresSaver,
    fenced_config,
)
from ragforge_core.adapters.db.conversations import SqlConversationRuns
from ragforge_core.db.engine import SqlTransactionManager
from ragforge_core.domain.errors import LeaseLost

pytestmark = pytest.mark.integration
Conn = psycopg.Connection[tuple[object, ...]]
Pool = AsyncConnectionPool[AsyncConnection[DictRow]]
MakeConversation = Callable[[], tuple[UUID, UUID]]
ExpireLease = Callable[[UUID], None]


class State(TypedDict):
    steps: Annotated[list[str], operator.add]


# ---------------------------------------------------------------------- helpers
def _with_ns(config: RunnableConfig, checkpoint_id: str | None = None) -> RunnableConfig:
    configurable = {**config["configurable"], "checkpoint_ns": ""}
    if checkpoint_id is not None:
        configurable["checkpoint_id"] = checkpoint_id
    return {"configurable": configurable}


def _checkpoint(saver: AsyncPostgresSaver, values: dict[str, Any]) -> Checkpoint:
    cp = empty_checkpoint()
    cp["channel_values"] = dict(values)
    cp["channel_versions"] = {k: saver.get_next_version(None, None) for k in values}
    return cp


async def _put(
    saver: FencedPostgresSaver,
    config: RunnableConfig,
    values: dict[str, Any],
    parent: str | None = None,
) -> str:
    cp = _checkpoint(saver, values)
    await saver.aput(
        _with_ns(config, parent), cp, {"source": "loop", "step": 0}, cp["channel_versions"]
    )
    return cp["id"]


def _head(superuser: Conn, conversation: UUID) -> object:
    row = superuser.execute(
        "SELECT head_checkpoint_id FROM rf.conversations WHERE id = %s", (conversation,)
    ).fetchone()
    assert row is not None
    return row[0]


def _count(superuser: Conn, table: str, conversation: UUID) -> int:
    row = superuser.execute(
        f"SELECT count(*) FROM rf.{table} WHERE thread_id = %s",
        (str(conversation),),
    ).fetchone()
    assert row is not None
    return int(str(row[0]))


async def _acquire(
    transactions: SqlTransactionManager, runs: SqlConversationRuns, tenant: UUID, conv: UUID
) -> UUID:
    async with transactions.tenant(tenant) as tx:
        return await runs.acquire(tx, conv)


def _approval_graph(saver: FencedPostgresSaver, on_work: Callable[[], None] = lambda: None) -> Any:
    def work(state: State) -> dict[str, list[str]]:
        on_work()
        return {"steps": ["work"]}

    def ask(state: State) -> dict[str, list[str]]:
        answer = interrupt("approve?")
        return {"steps": [f"ask:{answer}"]}

    builder = StateGraph(State)
    builder.add_node("work", work)
    builder.add_node("ask", ask)
    builder.add_edge(START, "work")
    builder.add_edge("work", "ask")
    builder.add_edge("ask", END)
    return builder.compile(checkpointer=saver)


# ---------------------------------------------------------------------- schema
def test_library_version_is_the_pinned_one() -> None:
    """The saver couples to library internals: an upgrade must re-run these contract tests and
    move the pin deliberately (ADR-0008)."""
    assert version("langgraph-checkpoint-postgres") == PINNED_LIBRARY_VERSION


async def test_schema_is_ready_without_library_setup(saver: FencedPostgresSaver) -> None:
    await saver.assert_schema_ready()
    await saver.setup()  # a check only: rf_api cannot (and must not) run DDL


@pytest.mark.parametrize("schema", ["checkpoint", "store"])
async def test_schema_check_rejects_a_library_with_newer_migrations(
    saver: FencedPostgresSaver, monkeypatch: pytest.MonkeyPatch, schema: str
) -> None:
    """A library bump that adds a migration to EITHER list must stop startup, not fail (or
    silently ignore new columns) at query time."""
    if schema == "checkpoint":
        monkeypatch.setattr(saver, "MIGRATIONS", [*saver.MIGRATIONS, "SELECT 1"])
    else:
        monkeypatch.setattr(checkpoint_module, "STORE_MIGRATIONS", [*STORE_MIGRATIONS, "SELECT 1"])
    with pytest.raises(CheckpointSchemaMismatch, match=schema):
        await saver.assert_schema_ready()


def test_langgraph_tables_live_in_rf_owned_by_rf_system(superuser: Conn) -> None:
    rows = superuser.execute(
        "SELECT c.relname, pg_get_userbyid(c.relowner), obj_description(c.oid, 'pg_class') "
        "FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace "
        "WHERE n.nspname = 'rf' AND c.relname = ANY(%s)",
        (["checkpoints", "checkpoint_blobs", "checkpoint_writes", "store"],),
    ).fetchall()
    assert len(rows) == 4
    assert all(
        owner == "rf_system" and str(cls) == "class: authoritative" for _, owner, cls in rows
    )
    (elsewhere,) = superuser.execute(  # type: ignore[misc]
        "SELECT count(*) FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace "
        "WHERE n.nspname <> 'rf' AND c.relname IN ('checkpoints', 'store')"
    ).fetchone()
    assert elsewhere == 0


@pytest.mark.parametrize(
    "table",
    [
        "checkpoints",
        "checkpoint_blobs",
        "checkpoint_writes",
        "checkpoint_migrations",
        "store",
        "store_migrations",
    ],
)
def test_worker_has_no_access_to_langgraph_tables(superuser: Conn, table: str) -> None:
    """These rows have no tenant_id and no RLS: any worker grant would be a cross-tenant read
    path (ADR-0013). A future worker consumer must read through a tenant-scoped path instead."""
    (has,) = superuser.execute(  # type: ignore[misc]
        "SELECT has_table_privilege('rf_worker', %s, 'SELECT,INSERT,UPDATE,DELETE')",
        (f"rf.{table}",),
    ).fetchone()
    assert has is False


# ---------------------------------------------------------------------- round trip
async def test_round_trip_reads_the_head(
    saver: FencedPostgresSaver,
    transactions: SqlTransactionManager,
    runs: SqlConversationRuns,
    superuser: Conn,
    make_conversation: MakeConversation,
) -> None:
    tenant, conversation = make_conversation()
    run = await _acquire(transactions, runs, tenant, conversation)
    config = fenced_config(conversation, tenant, run)
    first = await _put(saver, config, {"steps": ["a"], "count": 1})
    second = await _put(saver, config, {"steps": ["a", "b"], "count": 2}, parent=first)
    assert _head(superuser, conversation) == second

    latest = await saver.aget_tuple(fenced_config(conversation, tenant))  # reader: no run id
    assert latest is not None
    assert latest.checkpoint["id"] == second
    assert latest.checkpoint["channel_values"] == {"steps": ["a", "b"], "count": 2}
    assert latest.parent_config is not None
    assert latest.parent_config["configurable"]["checkpoint_id"] == first
    assert latest.metadata["step"] == 0
    assert "rf_tenant_id" not in latest.metadata
    assert "rf_run_id" not in latest.metadata

    explicit = await saver.aget_tuple(_with_ns(fenced_config(conversation, tenant), first))
    assert explicit is not None
    assert explicit.checkpoint["channel_values"]["count"] == 1
    history = [t.checkpoint["id"] async for t in saver.alist(fenced_config(conversation, tenant))]
    assert history == [second, first]


async def test_reads_use_head_not_latest_checkpoint_id(
    saver: FencedPostgresSaver,
    transactions: SqlTransactionManager,
    runs: SqlConversationRuns,
    superuser: Conn,
    make_conversation: MakeConversation,
) -> None:
    """uuid6 ordering depends on each pod's clock: a row with a larger id must not win."""
    tenant, conversation = make_conversation()
    reader = fenced_config(conversation, tenant)
    superuser.execute(
        "INSERT INTO rf.checkpoints (thread_id, checkpoint_ns, checkpoint_id, checkpoint) "
        "VALUES (%s, '', 'ffffffff-ffff-6fff-bfff-ffffffffffff', %s)",
        (str(conversation), psycopg.types.json.Jsonb({"v": 4, "id": "x"})),
    )
    assert await saver.aget_tuple(reader) is None, "no head yet: nothing is committed history"
    run = await _acquire(transactions, runs, tenant, conversation)
    head = await _put(saver, fenced_config(conversation, tenant, run), {"count": 1})
    found = await saver.aget_tuple(reader)
    assert found is not None
    assert found.checkpoint["id"] == head


async def test_alist_streams_and_an_abandoned_listing_returns_its_connection(
    saver: FencedPostgresSaver,
    lg_pool: Pool,
    transactions: SqlTransactionManager,
    runs: SqlConversationRuns,
    make_conversation: MakeConversation,
) -> None:
    """alist yields from inside its read transaction (no buffering of an unbounded history).
    Closing the iterator early must end that transaction and free the pooled connection: more
    abandoned listings than the pool has connections would otherwise hang."""
    tenant, conversation = make_conversation()
    run = await _acquire(transactions, runs, tenant, conversation)
    config = fenced_config(conversation, tenant, run)
    parent: str | None = None
    for i in range(3):
        parent = await _put(saver, config, {"count": i}, parent=parent)
    reader = fenced_config(conversation, tenant)

    async def first_then_abandon() -> str:
        # alist is an async generator; the port types it as a plain AsyncIterator.
        listing_gen = cast("AsyncGenerator[CheckpointTuple]", saver.alist(reader))
        async with aclosing(listing_gen) as listing:
            async for item in listing:
                return str(item.checkpoint["id"])
        raise AssertionError("empty history")

    max_size = lg_pool.max_size
    for _ in range(max_size + 2):
        assert await asyncio.wait_for(first_then_abandon(), timeout=5) == parent
    full = [t.checkpoint["id"] async for t in saver.alist(reader, limit=None)]
    assert len(full) == 3


# ---------------------------------------------------------------------- writes ordering
async def test_pending_writes_round_trip_in_library_order(
    saver: FencedPostgresSaver,
    lg_pool: Pool,
    transactions: SqlTransactionManager,
    runs: SqlConversationRuns,
    make_conversation: MakeConversation,
) -> None:
    tenant, conversation = make_conversation()
    run = await _acquire(transactions, runs, tenant, conversation)
    config = fenced_config(conversation, tenant, run)
    cp = await _put(saver, config, {"count": 1})
    target = _with_ns(config, cp)
    await saver.aput_writes(target, [("b", 1), ("a", 2)], task_id="task-2")
    await saver.aput_writes(target, [("c", 3)], task_id="task-1")
    await saver.aput_writes(target, [(ERROR, "boom")], task_id="task-3")
    await saver.aput_writes(target, [(ERROR, "boom")], task_id="task-3")  # idempotent upsert

    found = await saver.aget_tuple(_with_ns(fenced_config(conversation, tenant), cp))
    assert found is not None
    assert found.pending_writes is not None
    within_task_2 = [(c, v) for t, c, v in found.pending_writes if t == "task-2"]
    assert within_task_2 == [("b", 1), ("a", 2)], "per-task write order is preserved"
    assert sum(1 for _, c, _ in found.pending_writes if c == ERROR) == 1

    async with lg_pool.connection() as conn:  # identical to what the plain library returns
        plain = await AsyncPostgresSaver(conn).aget_tuple(_with_ns(config, cp))
    assert plain is not None
    assert found.pending_writes == plain.pending_writes


# ---------------------------------------------------------------------- interrupt / resume
async def test_interrupt_then_resume_under_a_new_run(
    saver: FencedPostgresSaver,
    transactions: SqlTransactionManager,
    runs: SqlConversationRuns,
    superuser: Conn,
    make_conversation: MakeConversation,
) -> None:
    tenant, conversation = make_conversation()
    graph = _approval_graph(saver)
    first = await _acquire(transactions, runs, tenant, conversation)
    out = await graph.ainvoke({"steps": []}, fenced_config(conversation, tenant, first))
    assert "__interrupt__" in out
    state = await graph.aget_state(fenced_config(conversation, tenant))
    assert state.next == ("ask",)
    assert _head(superuser, conversation) == state.config["configurable"]["checkpoint_id"]
    await runs.release(tenant, conversation, first)  # released at the interrupt

    second = await _acquire(transactions, runs, tenant, conversation)
    out = await graph.ainvoke(Command(resume="yes"), fenced_config(conversation, tenant, second))
    assert out["steps"] == ["work", "ask:yes"]
    final = await graph.aget_state(fenced_config(conversation, tenant))
    assert final.next == ()
    assert _head(superuser, conversation) == final.config["configurable"]["checkpoint_id"]


async def test_graph_without_the_lease_writes_nothing(
    saver: FencedPostgresSaver,
    transactions: SqlTransactionManager,
    runs: SqlConversationRuns,
    superuser: Conn,
    make_conversation: MakeConversation,
) -> None:
    tenant, conversation = make_conversation()
    graph = _approval_graph(saver)
    with pytest.raises(LeaseLost):
        await graph.ainvoke({"steps": []}, fenced_config(conversation, tenant, uuid4()))
    assert _count(superuser, "checkpoints", conversation) == 0
    assert _head(superuser, conversation) is None


# ---------------------------------------------------------------------- fencing: stalled pod
async def test_stalled_pod_checkpoint_is_rejected_and_invisible(
    saver: FencedPostgresSaver,
    transactions: SqlTransactionManager,
    runs: SqlConversationRuns,
    superuser: Conn,
    make_conversation: MakeConversation,
    expire_lease: ExpireLease,
) -> None:
    """Pod A stalls past its lease, pod B takes over and writes; A wakes up and writes. A's write
    must fail at the fence, leave no rows, and never become what readers see."""
    tenant, conversation = make_conversation()
    run_a = await _acquire(transactions, runs, tenant, conversation)
    config_a = fenced_config(conversation, tenant, run_a)
    base = await _put(saver, config_a, {"count": 1})

    expire_lease(conversation)  # A stalls
    run_b = await _acquire(transactions, runs, tenant, conversation)
    b_head = await _put(saver, fenced_config(conversation, tenant, run_b), {"count": 2}, base)

    rows_before = {
        t: _count(superuser, t, conversation)
        for t in ("checkpoints", "checkpoint_blobs", "checkpoint_writes")
    }
    stale = _checkpoint(saver, {"count": 99, "steps": ["stale"]})
    with pytest.raises(LeaseLost):
        await saver.aput(_with_ns(config_a, base), stale, {"step": 1}, stale["channel_versions"])
    with pytest.raises(LeaseLost):
        await saver.aput_writes(_with_ns(config_a, base), [("steps", ["stale"])], "task-a")

    assert {t: _count(superuser, t, conversation) for t in rows_before} == rows_before
    assert _head(superuser, conversation) == b_head
    seen = await saver.aget_tuple(fenced_config(conversation, tenant))
    assert seen is not None
    assert seen.checkpoint["channel_values"] == {"count": 2}
    expire_lease(conversation)  # an expired lease that nobody took over is also dead
    with pytest.raises(LeaseLost):
        await _put(saver, fenced_config(conversation, tenant, run_b), {"count": 3}, b_head)


async def test_lease_stolen_mid_node_rejects_the_nodes_writes(
    saver: FencedPostgresSaver,
    superuser: Conn,
    transactions: SqlTransactionManager,
    runs: SqlConversationRuns,
    make_conversation: MakeConversation,
) -> None:
    """The real graph path: a node finishes after its run lost the lease mid-execution."""
    tenant, conversation = make_conversation()
    run = await _acquire(transactions, runs, tenant, conversation)

    def steal() -> None:
        superuser.execute(
            "UPDATE rf.conversations SET active_run_id = %s, "
            "run_expires_at = clock_timestamp() + interval '1 minute' WHERE id = %s",
            (uuid4(), conversation),
        )

    graph = _approval_graph(saver, on_work=steal)
    # durability="sync": every checkpoint before the node is committed before it runs, so the
    # steal deterministically happens between a committed checkpoint and the node's writes.
    with pytest.raises(LeaseLost):
        await graph.ainvoke(
            {"steps": []}, fenced_config(conversation, tenant, run), durability="sync"
        )
    head = _head(superuser, conversation)
    assert head is not None
    seen = await saver.aget_tuple(fenced_config(conversation, tenant))
    assert seen is not None
    assert seen.checkpoint["id"] == head
    assert "work" not in seen.checkpoint["channel_values"].get("steps", [])
    every_write = [
        value
        async for t in saver.alist(fenced_config(conversation, tenant))
        for _task, _channel, value in (t.pending_writes or [])
    ]
    assert ["work"] not in every_write, "the stale node's output was never persisted"


# ---------------------------------------------------------------------- tenant isolation
async def test_other_tenant_cannot_read_or_write_the_thread(
    saver: FencedPostgresSaver,
    transactions: SqlTransactionManager,
    runs: SqlConversationRuns,
    superuser: Conn,
    make_conversation: MakeConversation,
) -> None:
    tenant_a, conversation = make_conversation()
    tenant_b, _ = make_conversation()
    run = await _acquire(transactions, runs, tenant_a, conversation)
    head = await _put(saver, fenced_config(conversation, tenant_a, run), {"secret": "a-only"})

    foreign = fenced_config(conversation, tenant_b)
    assert await saver.aget_tuple(foreign) is None
    assert await saver.aget_tuple(_with_ns(foreign, head)) is None, "even with the exact id"
    assert [t async for t in saver.alist(foreign)] == []
    history = await saver.aget_delta_channel_history(config=foreign, channels=["secret"])
    assert history == {"secret": {"writes": []}}
    # Even holding A's live run id, a writer bound to tenant B is fenced out.
    with pytest.raises(LeaseLost):
        await _put(saver, fenced_config(conversation, tenant_b, run), {"secret": "b"}, head)
    with pytest.raises(LeaseLost):
        await saver.aput_writes(
            _with_ns(fenced_config(conversation, tenant_b, run), head), [("x", 1)], "t"
        )
    assert _head(superuser, conversation) == head


async def test_config_without_tenant_or_run_fails_closed(
    saver: FencedPostgresSaver, make_conversation: MakeConversation
) -> None:
    tenant, conversation = make_conversation()
    with pytest.raises(ValueError, match="rf_tenant_id"):
        await saver.aget_tuple({"configurable": {"thread_id": str(conversation)}})
    with pytest.raises(ValueError, match="rf_run_id"):
        await _put(saver, fenced_config(conversation, tenant), {"count": 1})
    with pytest.raises(ValueError, match="thread config"):
        _ = [t async for t in saver.alist(None)]


async def test_tenant_setting_never_leaks_to_the_next_borrower(
    saver: FencedPostgresSaver,
    lg_pool: Pool,
    transactions: SqlTransactionManager,
    runs: SqlConversationRuns,
    make_conversation: MakeConversation,
) -> None:
    tenant, conversation = make_conversation()
    run = await _acquire(transactions, runs, tenant, conversation)
    await _put(saver, fenced_config(conversation, tenant, run), {"count": 1})
    for _ in range(lg_pool.max_size):
        async with lg_pool.connection() as conn:
            row = await (
                await conn.execute("SELECT current_setting('app.tenant_id', true) AS t")
            ).fetchone()
            assert row is not None
            assert not row["t"]


@pytest.mark.parametrize(
    "call",
    [
        lambda s: s.adelete_thread("t"),
        lambda s: s.adelete_for_runs(["r"]),
        lambda s: s.acopy_thread("a", "b"),
        lambda s: s.aprune(["t"]),
    ],
)
async def test_thread_wide_operations_are_disabled(
    saver: FencedPostgresSaver, call: Callable[[FencedPostgresSaver], Any]
) -> None:
    with pytest.raises(NotImplementedError):
        await call(saver)


# ---------------------------------------------------------------------- deletes + store
async def test_deleting_a_conversation_removes_its_checkpoints(
    saver: FencedPostgresSaver,
    transactions: SqlTransactionManager,
    runs: SqlConversationRuns,
    superuser: Conn,
    make_conversation: MakeConversation,
) -> None:
    tenant, conversation = make_conversation()
    _other_tenant, kept = make_conversation()
    run = await _acquire(transactions, runs, tenant, conversation)
    cp = await _put(saver, fenced_config(conversation, tenant, run), {"steps": ["x"]})
    await saver.aput_writes(_with_ns(fenced_config(conversation, tenant, run), cp), [("a", 1)], "t")
    async with transactions.tenant(tenant) as tx:  # as rf_api, under RLS
        from sqlalchemy import text

        await tx.execute(text("DELETE FROM rf.conversations WHERE id = :c"), {"c": conversation})
    for table in ("checkpoints", "checkpoint_blobs", "checkpoint_writes"):
        assert _count(superuser, table, conversation) == 0
    assert superuser.execute("SELECT 1 FROM rf.conversations WHERE id = %s", (kept,)).fetchone()


async def test_store_works_without_setup_and_follows_user_deletion(
    lg_pool: Pool, superuser: Conn, make_tenant: Callable[..., tuple[UUID, UUID, UUID]]
) -> None:
    tenant, user, _ = make_tenant()
    other_tenant, other_user, _ = make_tenant()
    store = AsyncPostgresStore(lg_pool)
    mine = (str(tenant), str(user), "facts")
    theirs = (str(other_tenant), str(other_user), "facts")
    await store.aput(mine, "f1", {"quote": "my favourite colour is green"})
    await store.aput(theirs, "f1", {"quote": "another user's fact"})
    item = await store.aget(mine, "f1")
    assert item is not None
    assert item.value["quote"].endswith("green")

    superuser.execute("DELETE FROM rf.tenants WHERE id = %s", (tenant,))  # cascades to users
    assert await store.aget(mine, "f1") is None
    assert await store.aget(theirs, "f1") is not None
