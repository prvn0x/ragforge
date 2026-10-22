"""Runtime context of one graph run: the ports, the authenticated Principal and the run lease.

Passed as LangGraph's `context=` (never checkpointed). WHY the Principal lives here and not in
the state: a resume is performed by whoever calls it now, and every tenant transaction, store
namespace and audit actor must be derived from THAT authenticated caller, not from a value that
was persisted earlier or that a model could have influenced.
"""

from __future__ import annotations

import asyncio
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from uuid import UUID

from langgraph.store.base import BaseStore

from ragforge_core.agent.ports import ConversationStore, KnowledgeBase
from ragforge_core.agent.settings import AgentSettings
from ragforge_core.domain.errors import LeaseLost
from ragforge_core.domain.models import Principal
from ragforge_core.domain.ports import (
    ChatModel,
    ConversationRuns,
    EmbedderRegistry,
    JobQueue,
    Reranker,
    Retriever,
    TransactionManager,
    Tx,
)
from ragforge_core.observability.logging import get_logger

log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class AgentDeps:
    chat: ChatModel
    retriever: Retriever
    reranker: Reranker  # runs after the retrieval tx closes (CPU model work, ADR-0006)
    embedders: EmbedderRegistry
    runs: ConversationRuns
    tx: TransactionManager
    jobs: JobQueue
    conversations: ConversationStore
    knowledge: KnowledgeBase
    store: BaseStore  # long-term memory (AsyncPostgresStore in prod, InMemoryStore in tests)
    settings: AgentSettings


class RunLease:
    """The conversation run lease held by this process for one run (DESIGN section 8).

    `check()` is called before every node (heartbeat_run in the node wrapper) so a run whose lease
    was stolen or expired stops at the next node boundary; `keepalive()` heartbeats during long
    nodes (a local model can stream for longer than the lease TTL). Once lost, it stays lost.
    """

    def __init__(
        self, runs: ConversationRuns, tenant_id: UUID, conversation_id: UUID, run_id: UUID
    ) -> None:
        self._runs = runs
        self.tenant_id = tenant_id
        self.conversation_id = conversation_id
        self.run_id = run_id
        self.lost = False

    async def _beat(self) -> bool:
        if not self.lost:
            self.lost = not await self._runs.heartbeat(
                self.tenant_id, self.conversation_id, self.run_id
            )
        return not self.lost

    async def check(self) -> None:
        if not await self._beat():
            raise LeaseLost(f"run {self.run_id} lost its conversation lease")

    async def keepalive(self, interval_s: float) -> None:
        """Background heartbeat; the node wrapper already beats at each impure node start.

        WHY a failed beat is logged and retried instead of ending the task: one transient DB
        error (connection reset) is not proof that the lease is gone, and a task that died with
        an exception would leave the lease unbeaten for the rest of a long generation. If the
        database stays unreachable, the next node-boundary `check()` raises on the critical path
        (where the caller can see it), and the fenced checkpointer rejects a stale writer anyway.
        """
        while True:
            await asyncio.sleep(interval_s)
            try:
                if not await self._beat():
                    return
            except Exception as exc:  # background task: never die on a blip
                log.warning(
                    "agent.lease_heartbeat_failed",
                    run_id=str(self.run_id),
                    error=type(exc).__name__,
                )

    async def release(self) -> None:
        await self._runs.release(self.tenant_id, self.conversation_id, self.run_id)


@dataclass(frozen=True, slots=True)
class AgentContext:
    principal: Principal
    conversation_id: UUID
    run_id: UUID
    deps: AgentDeps
    lease: RunLease

    @property
    def settings(self) -> AgentSettings:
        return self.deps.settings

    def tenant_tx(self) -> AbstractAsyncContextManager[Tx]:
        """Every tenant transaction of the run: tenant from the Principal, nothing else."""
        return self.deps.tx.tenant(self.principal.tenant_id)

    @property
    def facts_namespace(self) -> tuple[str, str, str]:
        """Store namespace for long-term facts: always the authenticated (tenant, user)."""
        return (str(self.principal.tenant_id), str(self.principal.user_id), "facts")

    @property
    def actor(self) -> str:
        return f"user:{self.principal.user_id}"
