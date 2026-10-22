"""Module D2 fixtures: conversations, the ConversationRuns adapter and the LangGraph pool."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from uuid import UUID, uuid4

import psycopg
import pytest
from psycopg import AsyncConnection
from psycopg.rows import DictRow, dict_row
from psycopg_pool import AsyncConnectionPool
from sqlalchemy.ext.asyncio import AsyncEngine

from ragforge_core.adapters.db.checkpoint import FencedPostgresSaver
from ragforge_core.adapters.db.conversations import SqlConversationRuns
from ragforge_core.config import Settings
from ragforge_core.db.engine import SqlTransactionManager

Conn = psycopg.Connection[tuple[object, ...]]
MakeTenant = Callable[..., tuple[UUID, UUID, UUID]]
MakeConversation = Callable[[], tuple[UUID, UUID]]


@pytest.fixture
def make_conversation(superuser: Conn, make_tenant: MakeTenant) -> MakeConversation:
    """Factory: a fresh tenant + conversation. Returns (tenant_id, conversation_id)."""

    def _make() -> tuple[UUID, UUID]:
        tenant, user, coll = make_tenant()
        conversation = uuid4()
        superuser.execute(
            "INSERT INTO rf.conversations (id, tenant_id, user_id, collection_ids) "
            "VALUES (%s, %s, %s, %s)",
            (conversation, tenant, user, [coll]),
        )
        return tenant, conversation

    return _make


@pytest.fixture
def transactions(api_engine: AsyncEngine) -> SqlTransactionManager:
    return SqlTransactionManager(api_engine)


@pytest.fixture
def runs(transactions: SqlTransactionManager) -> SqlConversationRuns:
    return SqlConversationRuns(transactions)


@pytest.fixture
async def lg_pool(
    settings: Settings,
) -> AsyncIterator[AsyncConnectionPool[AsyncConnection[DictRow]]]:
    """The LangGraph pool exactly as ADR-0008 configures it (rf_api, autocommit, dict_row)."""
    pool: AsyncConnectionPool[AsyncConnection[DictRow]] = AsyncConnectionPool(
        settings.dsn("api", driver="postgresql"),
        min_size=1,
        max_size=4,
        kwargs={"autocommit": True, "row_factory": dict_row, "prepare_threshold": 0},
        open=False,
    )
    await pool.open()
    yield pool
    await pool.close()


@pytest.fixture
async def saver(lg_pool: AsyncConnectionPool[AsyncConnection[DictRow]]) -> FencedPostgresSaver:
    """Async: the library saver binds to the running event loop at construction."""
    return FencedPostgresSaver(lg_pool)


@pytest.fixture
def expire_lease(superuser: Conn) -> Callable[[UUID], None]:
    """Simulate a stalled holder whose lease ran out, without waiting on wall-clock time."""

    def _expire(conversation: UUID) -> None:
        superuser.execute(
            "UPDATE rf.conversations SET run_expires_at = clock_timestamp() - interval '1 second' "
            "WHERE id = %s AND active_run_id IS NOT NULL",
            (conversation,),
        )

    return _expire
