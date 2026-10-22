"""Composition root: the ONLY place that wires adapters to ports (Phase 2 integration).

Both builders are async because wiring reads the database: the embedding spaces (one embedder
object per usable space; spaces added later are reloaded on a miss) and, for the API, the
LangGraph schema check and the reranker model. Building an embedder makes no network call, so
an Ollama problem (missing model, wrong dim) surfaces on the first embed, not here; only the
reranker model, which lives in the image, is checked at startup.

Every resource a builder opens is registered on the container's exit stack, so `aclose()` (or a
failure halfway through building) releases all of them.

The API refuses to start without its reranker model: the agent grades on the rerank score only
(ADR-0006), so running without it would make every chat turn abstain while health checks stayed
green. The worker does not rerank.

Model work never runs inside a transaction (PHASE1 rule): the containers only hand out ports;
handlers and routes open their own short transactions around them.
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import AsyncExitStack
from dataclasses import dataclass, field

import httpx
from langgraph.store.postgres import AsyncPostgresStore
from psycopg import AsyncConnection
from psycopg.rows import DictRow, dict_row
from psycopg_pool import AsyncConnectionPool
from sqlalchemy.ext.asyncio import AsyncEngine

from ragforge_core.adapters.blob.local import LocalBlobStore
from ragforge_core.adapters.connectors.files.parser import ParseLimits, SandboxedDocumentParser
from ragforge_core.adapters.connectors.notion.factory import NotionConnectorFactory
from ragforge_core.adapters.connectors.ratelimit import PostgresRateLimiter
from ragforge_core.adapters.connectors.store import SqlSyncStore
from ragforge_core.adapters.crypto.envelope import EnvelopeSecretBox, load_kek
from ragforge_core.adapters.db.agent_data import SqlConversationStore, SqlKnowledgeBase
from ragforge_core.adapters.db.checkpoint import FencedPostgresSaver
from ragforge_core.adapters.db.conversations import SqlConversationRuns
from ragforge_core.adapters.db.index import PgDocumentIndex
from ragforge_core.adapters.db.jobs import PgJobQueue
from ragforge_core.adapters.db.search import PgSearchBackend
from ragforge_core.adapters.llm.embedder_registry import (
    EmbeddingSpace,
    OllamaEmbedderRegistry,
    load_embedding_spaces,
)
from ragforge_core.adapters.llm.provider import build_chat_model
from ragforge_core.adapters.rerank.flashrank import PASSAGE_BUDGET, FlashRankReranker
from ragforge_core.agent import AgentDeps, AgentSettings, LangGraphAgentRunner
from ragforge_core.agent.interview.tokens import QuestionTokens
from ragforge_core.agent.sanitize import MarkdownStreamSanitizer
from ragforge_core.agent.settings import ToolMode
from ragforge_core.config import Settings
from ragforge_core.db.engine import SqlTransactionManager, make_engine
from ragforge_core.domain.ports import ChatModel, EmbedderRegistry, OutputSanitizer, Reranker
from ragforge_core.observability.llm import InstrumentedChatModel
from ragforge_core.rag.chunker import HeadingChunker
from ragforge_core.rag.retriever import HybridRetriever
from ragforge_core.rag.tokens import count_tokens
from ragforge_core.services.sync.deps import SyncDeps
from ragforge_core.services.sync.policy import SyncPolicy

LANGGRAPH_POOL_MAX = 5  # ADR-0008: the checkpointer/store pool, separate from SQLAlchemy's
EMBED_TIMEOUT_S = 60.0


@dataclass(slots=True)
class ApiContainer:
    """Ports the API reads by attribute name (apps/api/deps.py, handoff E2 1.2)."""

    settings: Settings
    engine: AsyncEngine
    tx: SqlTransactionManager
    secrets: EnvelopeSecretBox
    jobs: PgJobQueue
    retriever: HybridRetriever
    reranker: Reranker
    embedders: EmbedderRegistry
    agent: LangGraphAgentRunner
    sanitizer_factory: Callable[[], OutputSanitizer]
    blobs: LocalBlobStore
    sync_store: SqlSyncStore
    # Interview mode (Phase 3b): the agent's own chat model, and the question-token signer.
    chat: ChatModel | None = None
    interview_tokens: QuestionTokens | None = None
    _resources: AsyncExitStack = field(default_factory=AsyncExitStack, repr=False)

    async def aclose(self) -> None:
        try:
            await self._resources.aclose()
        finally:
            await self.engine.dispose()


@dataclass(slots=True)
class WorkerContainer:
    settings: Settings
    engine: AsyncEngine
    tx: SqlTransactionManager
    secrets: EnvelopeSecretBox
    jobs: PgJobQueue
    sync: SyncDeps
    _resources: AsyncExitStack = field(default_factory=AsyncExitStack, repr=False)

    async def aclose(self) -> None:
        try:
            await self._resources.aclose()
        finally:
            await self.engine.dispose()


# ------------------------------------------------------------------ shared pieces
def _secrets(settings: Settings) -> EnvelopeSecretBox:
    """Fails fast at startup if the KEK is missing or malformed."""
    return EnvelopeSecretBox(load_kek(settings.kek.get_secret_value()))


def _question_tokens(settings: Settings) -> QuestionTokens:
    """Interview question tokens, keyed by an HKDF subkey of the KEK (agent/interview/tokens)."""
    kek = load_kek(settings.kek.get_secret_value())
    return QuestionTokens.from_secret(kek, ttl_s=settings.interview_token_ttl_s)


async def _embedders(
    settings: Settings, tx: SqlTransactionManager, stack: AsyncExitStack
) -> OllamaEmbedderRegistry:
    http = await stack.enter_async_context(
        httpx.AsyncClient(base_url=settings.ollama_base_url, timeout=EMBED_TIMEOUT_S)
    )

    async def load() -> list[EmbeddingSpace]:  # a short system read; never a model call
        async with tx.system() as conn:
            return await load_embedding_spaces(conn)

    registry = OllamaEmbedderRegistry(
        await load(),
        http,
        num_ctx=settings.embed_num_ctx,
        keep_alive=settings.ollama_keep_alive,
        batch_size=settings.embed_batch_size,
        loader=load,
    )
    stack.push_async_callback(registry.aclose)  # LIFO: before the http client and the engine
    return registry


def _retriever(settings: Settings) -> HybridRetriever:
    return HybridRetriever(PgSearchBackend(exact_threshold=settings.vector_exact_threshold))


async def _reranker(settings: Settings, stack: AsyncExitStack) -> FlashRankReranker:
    """The model is baked into the image and loaded now (never downloaded). A missing model
    raises `RerankerUnavailable`, which fails the API startup with the model and cache dir.
    Its inference thread is stopped on close (registered before the load, so a failed load
    releases it too)."""
    reranker = FlashRankReranker(
        cache_dir=settings.rerank_cache_dir,
        model_name=settings.rerank_model,
        top_n=settings.rerank_top_n,
        token_counter=count_tokens,
    )
    stack.callback(reranker.close)
    await reranker.load()
    return reranker


# Chunks must fit the cross-encoder whole next to any query (adapters/rerank/flashrank.py):
# max = PASSAGE_BUDGET (445); the target keeps the chunker's default target/max ratio (400/512).
CHUNK_MAX_TOKENS = PASSAGE_BUDGET
CHUNK_TARGET_TOKENS = PASSAGE_BUDGET * 400 // 512


def _chunker() -> HeadingChunker:
    return HeadingChunker(target_tokens=CHUNK_TARGET_TOKENS, max_tokens=CHUNK_MAX_TOKENS)


def _agent_settings(settings: Settings) -> AgentSettings:
    return AgentSettings(
        default_tau=settings.agent_default_tau,
        retrieval_k=settings.agent_retrieval_k,
        rerank_top=settings.rerank_top_n,
        max_tool_rounds=settings.agent_max_tool_rounds,
        tool_mode=ToolMode(settings.agent_tool_mode),
        strict_citations=settings.agent_strict_citations,
        verifier_trusted=settings.agent_verifier_trusted,
        extras=settings.agent_extras,
        summarize_after_messages=settings.agent_summarize_after_messages,
    )


async def _langgraph_pool(
    settings: Settings, stack: AsyncExitStack
) -> AsyncConnectionPool[AsyncConnection[DictRow]]:
    """ADR-0008: autocommit, dict rows, no server-side prepared statements (pgbouncer-safe).
    `check` validates a connection on checkout, so after a Postgres restart a stale pooled
    connection is replaced instead of failing a checkpoint write or a lease heartbeat."""
    pool: AsyncConnectionPool[AsyncConnection[DictRow]] = AsyncConnectionPool(
        settings.dsn("api", driver="postgresql"),
        min_size=1,
        max_size=LANGGRAPH_POOL_MAX,
        kwargs={"autocommit": True, "row_factory": dict_row, "prepare_threshold": 0},
        check=AsyncConnectionPool.check_connection,
        open=False,
    )
    await pool.open(wait=True)
    stack.push_async_callback(pool.close)
    return pool


# ------------------------------------------------------------------ API
async def build_api(
    settings: Settings,
    *,
    chat: ChatModel | None = None,
    reranker: Reranker | None = None,
    embedders: EmbedderRegistry | None = None,
) -> ApiContainer:
    """`chat` / `reranker` / `embedders` replace the model adapters (tests drive the real
    composition with scripted models; production passes none)."""
    engine = make_engine(settings, "api")
    stack = AsyncExitStack()
    try:
        tx = SqlTransactionManager(engine)
        jobs = PgJobQueue(engine)
        if embedders is None:
            embedders = await _embedders(settings, tx, stack)
        retriever = _retriever(settings)
        if reranker is None:
            reranker = await _reranker(settings, stack)

        pool = await _langgraph_pool(settings, stack)
        saver = FencedPostgresSaver(pool)
        await saver.assert_schema_ready()
        if chat is None:
            configured_chat = build_chat_model(settings)  # RAGFORGE_LLM_PROVIDER picks it
            stack.push_async_callback(configured_chat.aclose)
            chat = configured_chat
        # ADR-0018 R2-3: every model call of the API goes through this ONE measured instance
        # (agent, verifier, interview, eval verify), injected models included, so tests see
        # the same metrics production records.
        chat = InstrumentedChatModel(chat)

        agent = LangGraphAgentRunner(
            AgentDeps(
                chat=chat,
                retriever=retriever,
                reranker=reranker,
                embedders=embedders,
                runs=SqlConversationRuns(tx),
                tx=tx,
                jobs=jobs,
                conversations=SqlConversationStore(),
                knowledge=SqlKnowledgeBase(),
                store=AsyncPostgresStore(pool),  # no setup(): rf_api cannot run DDL (slot 0005)
                settings=_agent_settings(settings),
            ),
            checkpointer=saver,
        )
        return ApiContainer(
            settings=settings,
            engine=engine,
            tx=tx,
            secrets=_secrets(settings),
            jobs=jobs,
            retriever=retriever,
            reranker=reranker,
            embedders=embedders,
            agent=agent,
            sanitizer_factory=MarkdownStreamSanitizer,
            blobs=LocalBlobStore(settings.upload_dir),
            sync_store=SqlSyncStore(),
            chat=chat,
            interview_tokens=_question_tokens(settings),
            _resources=stack,
        )
    except BaseException:
        try:
            await stack.aclose()
        finally:
            await engine.dispose()
        raise


# ------------------------------------------------------------------ worker
async def build_worker(settings: Settings, engine: AsyncEngine | None = None) -> WorkerContainer:
    """`engine` lets the entrypoint wait for the schema before wiring (embedding spaces are read
    here); the container takes ownership of it either way."""
    engine = engine or make_engine(settings, "worker")
    stack = AsyncExitStack()
    try:
        tx = SqlTransactionManager(engine)
        jobs = PgJobQueue(engine)
        secrets = _secrets(settings)
        sync = SyncDeps(
            tx=tx,
            queue=jobs,
            store=SqlSyncStore(),
            index=PgDocumentIndex(),
            chunker=_chunker(),
            embedders=await _embedders(settings, tx, stack),
            secrets=secrets,
            # The connector owns the listing overlap (ADR-0007, handoff C1 request 2); the sync
            # policy passes its cursor through unchanged, so the overlap is applied once.
            connectors=NotionConnectorFactory(PostgresRateLimiter(engine)),
            blobs=LocalBlobStore(settings.upload_dir),
            parser=SandboxedDocumentParser(
                ParseLimits(
                    max_bytes=settings.max_upload_mb * 1024 * 1024,
                    timeout_s=settings.parse_timeout_s,
                    memory_bytes=settings.parse_memory_mb * 1024 * 1024,
                )
            ),
            policy=SyncPolicy(poll_interval_s=settings.sync_poll_interval_s),
        )
        return WorkerContainer(
            settings=settings,
            engine=engine,
            tx=tx,
            secrets=secrets,
            jobs=jobs,
            sync=sync,
            _resources=stack,
        )
    except BaseException:
        try:
            await stack.aclose()
        finally:
            await engine.dispose()
        raise
