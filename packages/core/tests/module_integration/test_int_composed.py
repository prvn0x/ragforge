"""Phase 2 integration: the REAL composition root (`bootstrap.build_api`) on the migrated schema.

Only the models are scripted (chat, embedder, reranker). Everything else is production wiring:
LangGraphAgentRunner + FencedPostgresSaver + AsyncPostgresStore + SqlConversationRuns +
SqlConversationStore/SqlKnowledgeBase + HybridRetriever/PgSearchBackend, as rf_api under RLS.
This exercises the seams the per-module tests fake: chat -> approval interrupt -> resume ->
execute (claim + remove_from_index in one tx) -> exclusion that sync honours.
"""

from __future__ import annotations

import dataclasses
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import httpx
import psycopg
import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine
from sqlalchemy.pool import QueuePool

from module_integration.int_support import (
    FixedEmbedders,
    ScoringReranker,
    ScriptedChat,
    say,
    seed_document,
)
from ragforge_core import bootstrap
from ragforge_core.adapters.blob.local import LocalBlobStore
from ragforge_core.adapters.connectors.files.parser import SandboxedDocumentParser
from ragforge_core.adapters.connectors.notion.factory import NotionConnectorFactory
from ragforge_core.adapters.connectors.store import SqlSyncStore
from ragforge_core.adapters.db.checkpoint import FencedPostgresSaver
from ragforge_core.adapters.db.index import PgDocumentIndex
from ragforge_core.adapters.db.tx import connection
from ragforge_core.adapters.llm.embedder_registry import OllamaEmbedderRegistry
from ragforge_core.adapters.rerank.flashrank import RerankerUnavailable
from ragforge_core.bootstrap import build_api
from ragforge_core.config import Settings
from ragforge_core.db.engine import make_engine
from ragforge_core.domain.errors import Conflict, NotFound
from ragforge_core.domain.models import (
    ApprovalChoice,
    Principal,
    ToolCall,
    TurnDone,
    TurnEvent,
    TurnInterrupt,
)

pytestmark = pytest.mark.integration
Conn = psycopg.Connection[tuple[object, ...]]
MakeTenant = Callable[..., tuple[UUID, UUID, UUID]]
SCOPES = frozenset({"read", "write", "chat", "approvals:write"})


async def _collect(events: AsyncIterator[TurnEvent]) -> list[TurnEvent]:
    return [e async for e in events]


Composed = tuple[bootstrap.ApiContainer, ScriptedChat]


@pytest.fixture
async def composed(settings: Settings) -> AsyncIterator[Composed]:
    chat = ScriptedChat()
    container = await build_api(
        settings, chat=chat, reranker=ScoringReranker(), embedders=FixedEmbedders()
    )
    try:
        yield container, chat
    finally:
        await container.aclose()


# ---------------------------------------------------------------- tests
async def test_approval_flow_end_to_end_executes_once_and_excludes_the_page(
    composed: Composed,
    superuser: Conn,
    make_tenant: MakeTenant,
) -> None:
    container, chat = composed
    tenant, user, coll = make_tenant()
    doc = seed_document(superuser, tenant, coll)
    who = Principal(tenant_id=tenant, user_id=user, scopes=SCOPES)
    remove = ToolCall(id=uuid4().hex, name="remove_from_index", arguments={"document_id": str(doc)})

    chat.script = [say("", remove)]
    first = await _collect(container.agent.run_turn(who, None, [coll], "/remove the handbook"))
    interrupt = first[-1]
    assert isinstance(interrupt, TurnInterrupt), [type(e).__name__ for e in first]
    assert [p.id for p in interrupt.pending] == [remove.id]
    (pending,) = superuser.execute(
        "SELECT status, executed_at FROM rf.pending_approvals WHERE run_id = %s",
        (interrupt.run_id,),
    ).fetchall()
    assert pending == ("pending", None)
    assert _live(superuser, doc), "nothing executes before approval"

    supported: dict[str, object] = {"verdict": "supported", "unsupported_claims": []}
    chat.script = [say("Removed the handbook [T1]."), supported]
    second = await _collect(
        container.agent.resume(who, interrupt.run_id, {remove.id: ApprovalChoice.approve})
    )
    assert isinstance(second[-1], TurnDone), [type(e).__name__ for e in second]

    rows = superuser.execute(
        "SELECT status, executed_at IS NOT NULL FROM rf.pending_approvals WHERE tool_call_id = %s",
        (remove.id,),
    ).fetchall()
    assert rows == [("approved", True)], "claimed exactly once"
    assert not _live(superuser, doc), "tombstoned"
    excluded = superuser.execute(
        "SELECT count(*) FROM rf.exclusions WHERE tenant_id = %s AND source_kind = 'file'",
        (tenant,),
    ).fetchone()
    assert excluded == (1,)
    audit = superuser.execute(
        "SELECT actor FROM rf.audit_log WHERE tenant_id = %s AND action = 'document.removed' "
        "AND target = %s",
        (tenant, str(doc)),
    ).fetchall()
    assert audit == [(f"user:{user}",)]
    conversation = superuser.execute(
        "SELECT active_run_id FROM rf.conversations WHERE tenant_id = %s", (tenant,)
    ).fetchone()
    assert conversation == (None,), "lease released"

    with pytest.raises(Conflict):
        await _collect(
            container.agent.resume(who, interrupt.run_id, {remove.id: ApprovalChoice.approve})
        )
    assert superuser.execute(
        "SELECT count(*) FROM rf.audit_log WHERE tenant_id = %s AND action = 'document.removed'",
        (tenant,),
    ).fetchone() == (1,), "a second resume never repeats the side effect"


async def test_another_tenant_cannot_resume_the_run(
    composed: Composed,
    superuser: Conn,
    make_tenant: MakeTenant,
) -> None:
    container, chat = composed
    tenant, user, coll = make_tenant()
    intruder, intruder_user, _ = make_tenant()
    doc = seed_document(superuser, tenant, coll)
    who = Principal(tenant_id=tenant, user_id=user, scopes=SCOPES)
    remove = ToolCall(id=uuid4().hex, name="remove_from_index", arguments={"document_id": str(doc)})
    chat.script = [say("", remove)]
    interrupt = (await _collect(container.agent.run_turn(who, None, [coll], "/remove it")))[-1]
    assert isinstance(interrupt, TurnInterrupt)

    stranger = Principal(tenant_id=intruder, user_id=intruder_user, scopes=SCOPES)
    with pytest.raises(NotFound):
        await _collect(
            container.agent.resume(stranger, interrupt.run_id, {remove.id: ApprovalChoice.approve})
        )
    assert _live(superuser, doc)


def _live(su: Conn, doc: UUID) -> bool:
    row = su.execute("SELECT deleted_at IS NULL FROM rf.documents WHERE id = %s", (doc,)).fetchone()
    return bool(row and row[0])


# ---------------------------------------------------------------- startup failure cleanup
@dataclasses.dataclass
class Opened:
    """Every resource a builder opened, captured through spies (not global pg_stat_activity
    counts, which other tests' late-closing pools make racy)."""

    engines: list[AsyncEngine] = dataclasses.field(default_factory=list)
    clients: list[httpx.AsyncClient] = dataclasses.field(default_factory=list)
    pools: list[Any] = dataclasses.field(default_factory=list)

    def assert_all_released(self) -> None:
        assert self.engines, "the builder created an engine"
        for engine in self.engines:
            pool = engine.sync_engine.pool
            assert isinstance(pool, QueuePool)
            # dispose() swaps in a fresh, empty pool: nothing checked out, nothing idle.
            assert (pool.checkedout(), pool.checkedin()) == (0, 0)
        assert all(client.is_closed for client in self.clients)
        assert all(pool.closed for pool in self.pools)


@pytest.fixture
def opened(monkeypatch: pytest.MonkeyPatch) -> Opened:
    seen = Opened()
    real_pool = bootstrap._langgraph_pool

    def engine_spy(settings: Settings, role: str) -> AsyncEngine:
        engine = make_engine(settings, role)
        seen.engines.append(engine)
        return engine

    async def pool_spy(settings: Settings, stack: Any) -> Any:
        pool = await real_pool(settings, stack)
        seen.pools.append(pool)
        return pool

    class ClientSpy(httpx.AsyncClient):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            seen.clients.append(self)

    monkeypatch.setattr("ragforge_core.bootstrap.make_engine", engine_spy)
    monkeypatch.setattr(bootstrap, "_langgraph_pool", pool_spy)
    monkeypatch.setattr(httpx, "AsyncClient", ClientSpy)  # the module bootstrap builds from
    return seen


async def test_missing_reranker_model_fails_startup_and_leaks_nothing(
    settings: Settings, opened: Opened, tmp_path: Path
) -> None:
    broken = settings.model_copy(update={"rerank_cache_dir": str(tmp_path)})
    with pytest.raises(RerankerUnavailable, match=str(tmp_path)):
        await build_api(broken, chat=ScriptedChat())
    assert opened.clients, "the embedder http client was opened before the reranker check"
    opened.assert_all_released()


async def test_failure_after_the_langgraph_pool_opened_closes_it(
    settings: Settings, opened: Opened, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def boom(self: object) -> None:
        raise RuntimeError("schema check failed")

    monkeypatch.setattr(FencedPostgresSaver, "assert_schema_ready", boom)
    with pytest.raises(RuntimeError, match="schema check failed"):
        await build_api(
            settings, chat=ScriptedChat(), reranker=ScoringReranker(), embedders=FixedEmbedders()
        )
    assert len(opened.pools) == 1
    opened.assert_all_released()


async def test_build_worker_wires_real_adapters_and_aclose_releases_everything(
    settings: Settings, opened: Opened
) -> None:
    container = await bootstrap.build_worker(settings)
    try:
        sync = container.sync
        assert isinstance(sync.store, SqlSyncStore)
        assert isinstance(sync.index, PgDocumentIndex)
        assert isinstance(sync.embedders, OllamaEmbedderRegistry)
        assert isinstance(sync.connectors, NotionConnectorFactory)
        assert isinstance(sync.parser, SandboxedDocumentParser)
        assert isinstance(sync.blobs, LocalBlobStore)
        assert sync.policy.cursor_overlap.total_seconds() == 0, "the connector owns the overlap"
        assert sync.policy.poll_interval_s == settings.sync_poll_interval_s
        assert sync.embedders.for_space(1) is not None, "the default space was loaded"
        async with container.tx.tenant(uuid4()) as tx:  # the engine works as rf_worker
            await connection(tx).execute(text("SELECT 1"))
    finally:
        await container.aclose()
    opened.assert_all_released()
