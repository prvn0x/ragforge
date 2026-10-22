"""Module-local persistence ports of the agent (PHASE1: "a module that needs its own tables
accessed defines a small Protocol"). The frozen `ConversationRuns` port covers the lease and
locking; these cover the remaining rows the agent reads or writes.

Signatures use only domain types and primitives so the adapters (`adapters/db/conversations.py`,
owned by D2/integrator) can implement them structurally without importing `agent` (forbidden by
import-linter). Every method runs inside the caller's tenant `tx`: the tenant is ambient (RLS),
never a parameter, so nothing a model emits can widen it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol, runtime_checkable
from uuid import UUID

from ragforge_core.domain.models import ApprovalChoice, RankedChunk, ToolCall
from ragforge_core.domain.ports import Tx


@runtime_checkable
class ConversationStore(Protocol):
    async def create(self, tx: Tx, user_id: UUID, collection_ids: Sequence[UUID]) -> UUID:
        """Insert a conversation for the caller (tenant from tx); returns its id (= thread_id)."""
        ...

    async def get(self, tx: Tx, conversation_id: UUID) -> tuple[UUID, tuple[UUID, ...]] | None:
        """(owner user_id, collection_ids), or None when invisible under RLS / missing."""
        ...

    async def conversation_for_run(self, tx: Tx, run_id: UUID) -> UUID | None:
        """Conversation of a run's pending approvals. A plain read WITHOUT row locks: resume must
        take the lease (conversation row) BEFORE locking approvals, in the same order as a new
        turn, or the two paths could deadlock."""
        ...

    async def record_pending(
        self, tx: Tx, conversation_id: UUID, run_id: UUID, calls: Sequence[ToolCall]
    ) -> None:
        """INSERT pending_approvals ... ON CONFLICT (conversation_id, tool_call_id) DO NOTHING.
        Idempotent because LangGraph re-runs the approval node on resume."""
        ...

    async def decide_pending(
        self, tx: Tx, run_id: UUID, decisions: Mapping[str, ApprovalChoice]
    ) -> None:
        """Set status approved|rejected + decided_at on rows the caller locked (lock_pending)."""
        ...

    async def claim_execution(self, tx: Tx, conversation_id: UUID, tool_call_id: str) -> bool:
        """True exactly once per approved tool call; False if already executed or not approved.
        Runs in the SAME tx as the tool's side effect, so a rollback releases the claim."""
        ...

    async def summary(self, tx: Tx, conversation_id: UUID) -> str | None:
        """conversation_summaries.summary (written by the summarize_conversation job)."""
        ...


@runtime_checkable
class KnowledgeBase(Protocol):
    async def collection_taus(
        self, tx: Tx, collection_ids: Sequence[UUID]
    ) -> Mapping[UUID, float | None]:
        """collections.tau per visible collection (NULL = not tuned yet -> default)."""
        ...

    async def get_page(
        self, tx: Tx, document_id: UUID, collection_ids: Sequence[UUID], limit: int
    ) -> Sequence[RankedChunk]:
        """The document's live chunks in ordinal order, only if it is in `collection_ids`."""
        ...

    async def list_sources(
        self, tx: Tx, collection_ids: Sequence[UUID]
    ) -> Sequence[Mapping[str, object]]:
        """Connectors/uploads feeding the collections: JSON-safe rows, no secrets."""
        ...

    async def sync_status(
        self, tx: Tx, collection_ids: Sequence[UUID]
    ) -> Sequence[Mapping[str, object]]:
        """Per-connector status, last error and next poll: JSON-safe rows, no secrets."""
        ...

    async def remove_from_index(
        self, tx: Tx, document_id: UUID, collection_ids: Sequence[UUID], actor: str
    ) -> bool:
        """Insert the exclusion, tombstone the document and audit it. False if not visible."""
        ...
