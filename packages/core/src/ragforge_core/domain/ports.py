"""Ports: the interfaces Phase 1 modules implement and consume. FROZEN for Phase 1.

Ownership (docs/PHASE1.md):
  shared (Phase 0) -> TransactionManager, SecretBox
  A data plane     -> JobQueue
  B retrieval      -> Embedder, EmbedderRegistry, Reranker, Chunker, Retriever, DocumentIndex
  C connectors     -> SourceConnector, RateLimiter, BlobStore, DocumentParser
  D agent          -> ChatModel, AgentRunner, ConversationRuns
  E api/safety     -> OutputSanitizer

Transactions: every method that reads or writes tenant data takes the caller's `tx: Tx`, so work
composes atomically (e.g. state change + enqueue in ONE transaction, ADR-0003; assert_lease +
side effects + complete in ONE transaction, ADR-0010). The tenant is ambient in `tx` (set by
TransactionManager.tenant); it is never a parameter an LLM could influence. `Tx` is opaque to
the domain; adapters receive the concrete connection (SQLAlchemy AsyncConnection).
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping, Sequence
from contextlib import AbstractAsyncContextManager
from datetime import datetime
from typing import Protocol, runtime_checkable
from uuid import UUID

from ragforge_core.domain.models import (
    ApprovalChoice,
    ChatMessage,
    ChatStreamEvent,
    ChatTurnUsage,
    Chunk,
    DocumentRecord,
    JobLease,
    JobSpec,
    PendingApproval,
    Principal,
    RankedChunk,
    SealedSecret,
    SourceDocument,
    SourceItemRef,
    ToolSpec,
    TurnEvent,
    UpsertOutcome,
)


class Tx(Protocol):
    """Opaque handle to an open database transaction (tenant-scoped or system)."""


# ----------------------------------------------------------------- shared: transactions
@runtime_checkable
class TransactionManager(Protocol):
    def tenant(self, tenant_id: UUID) -> AbstractAsyncContextManager[Tx]:
        """Open a transaction in which RLS scopes every tenant table to `tenant_id`."""
        ...

    def system(self) -> AbstractAsyncContextManager[Tx]:
        """Open a transaction with NO tenant (tenant tables return zero rows). Only for the named
        cross-tenant SECURITY DEFINER functions (ADR-0013) and global tables."""
        ...


# ----------------------------------------------------------------- A: job queue
@runtime_checkable
class JobQueue(Protocol):
    async def enqueue(self, tx: Tx, spec: JobSpec) -> UUID:
        """Insert or merge (tenant-scoped dedupe on queued jobs) inside the caller's tenant tx."""
        ...

    async def claim(self, worker_id: str, n: int) -> Sequence[JobLease]:
        """Cross-tenant claim (definer function); opens its own system transaction."""
        ...

    async def assert_lease(self, tx: Tx, lease: JobLease) -> None:
        """First statement of every result transaction: row-locks the job and raises
        domain.errors.LeaseLost unless the lease version/status still match (ADR-0010)."""
        ...

    async def report_progress(self, tx: Tx, lease: JobLease, progress: dict[str, object]) -> None:
        """Write job progress (feeds GET /v1/jobs/{id}/events) and NOTIFY rf_job_progress with
        the job id only. Runs in the caller's fenced transaction (after assert_lease)."""
        ...

    async def complete(self, tx: Tx, lease: JobLease) -> None:
        """Mark done inside the same transaction as the side effects (after assert_lease)."""
        ...

    async def heartbeat(self, lease: JobLease) -> bool:
        """Extend the lease (own transaction). False = lease lost: stop and discard work."""
        ...

    async def release(self, lease: JobLease) -> None:
        """Give the job back (shutdown); merges into an existing queued duplicate."""
        ...

    async def fail(
        self,
        lease: JobLease,
        error: str,
        retry_after_s: float | None,
        *,
        permanent: bool = False,
    ) -> None:
        """Retry with backoff (dead after max_attempts); `permanent=True` goes straight to dead
        (e.g. a definitive 404). Own transaction."""
        ...


# ----------------------------------------------------------------- B: retrieval
@runtime_checkable
class Embedder(Protocol):
    model: str
    dim: int

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Applies the embedding space's document prefix."""
        ...

    async def embed_query(self, text: str) -> list[float]:
        """Applies the embedding space's query prefix."""
        ...


@runtime_checkable
class Reranker(Protocol):
    async def rerank(self, query: str, candidates: Sequence[RankedChunk]) -> list[RankedChunk]: ...


@runtime_checkable
class Chunker(Protocol):
    def chunk(self, doc: SourceDocument) -> list[Chunk]: ...


@runtime_checkable
class EmbedderRegistry(Protocol):
    """One Embedder per embedding space (a space = provider + model + dim + prefixes).

    Async callers `await ensure(space_id)` before `for_space(space_id)`: a space created or
    flipped after startup (blue/green, ADR-0009) is then loaded on the FIRST use instead of
    failing it. `ensure` reads the space table only (never a model) and must run with no
    transaction open, like every other step around a model call.
    """

    async def ensure(self, space_id: int) -> None:
        """Make `for_space(space_id)` answer from current state, reloading once on a miss.
        Raises the same errors `for_space` would if the space is still unusable."""
        ...

    def for_space(self, space_id: int) -> Embedder: ...


@runtime_checkable
class Retriever(Protocol):
    async def retrieve(
        self,
        tx: Tx,
        query: str,
        query_vectors: Mapping[int, Sequence[float]],
        collection_ids: Sequence[UUID],
        k: int,
    ) -> list[RankedChunk]:
        """`query_vectors` maps space_id -> query embedding, computed by the caller BEFORE opening
        `tx` (never call a model while a transaction is open). Keep `tx` short."""
        ...

    async def spaces_for(self, tx: Tx, collection_ids: Sequence[UUID]) -> Sequence[int]:
        """Distinct ACTIVE space ids of the collections: the keys `query_vectors` must contain."""
        ...


@runtime_checkable
class DocumentIndex(Protocol):
    """Index persistence (owner B; called by C's sync service and by D's tools).

    Every method runs inside the caller's tenant `tx`. NEVER call an Embedder while a transaction
    is open. The sync flow (DESIGN section 3, Flow A) is:
      1. short tx:  spaces = target_spaces(); missing = touch_and_find_missing() per space
      2. no tx:     vectors = EmbedderRegistry.for_space(s).embed_documents(texts of missing)
      3. fenced tx: JobQueue.assert_lease -> insert_embeddings -> upsert_document ->
                    replace_chunks -> JobQueue.complete
    Touches commit in step 1 and protect the hashes from GC (24 h grace) until step 3 commits.
    """

    async def target_spaces(self, tx: Tx, collection_id: UUID) -> Sequence[int]:
        """active_space_id plus building_space_id (if any): every space a sync must write."""
        ...

    async def touch_and_find_missing(
        self, tx: Tx, collection_id: UUID, space_id: int, embed_sha256s: Sequence[bytes]
    ) -> set[bytes]:
        """Record touches for ALL given hashes; return those with no embedding in the space."""
        ...

    async def insert_embeddings(
        self,
        tx: Tx,
        collection_id: UUID,
        space_id: int,
        vectors: Mapping[bytes, Sequence[float]],
    ) -> int:
        """ON CONFLICT DO NOTHING. Returns rows inserted."""
        ...

    async def upsert_document(self, tx: Tx, record: DocumentRecord) -> UpsertOutcome:
        """Monotonic: applies only if record.fetched_at > stored fetched_at (else `stale`)."""
        ...

    async def replace_chunks(
        self, tx: Tx, document_id: UUID, collection_id: UUID, chunks: Sequence[Chunk]
    ) -> None:
        """Replace the document's chunks (ids from domain.ids.chunk_id)."""
        ...

    async def tombstone(self, tx: Tx, document_id: UUID) -> None: ...

    async def is_excluded(self, tx: Tx, source_kind: str, external_id: str) -> bool: ...


# ----------------------------------------------------------------- C: sources
@runtime_checkable
class SourceConnector(Protocol):
    def list_changed(self, since: datetime | None) -> AsyncIterator[SourceItemRef]: ...

    async def fetch(self, ref: SourceItemRef) -> SourceDocument: ...

    def list_all(self) -> AsyncIterator[SourceItemRef]: ...

    async def confirm_deleted(self, external_id: str) -> bool:
        """True only on definitive evidence (404, in_trash, unshared). Errors -> False."""
        ...


@runtime_checkable
class RateLimiter(Protocol):
    async def acquire(self, key: str, tokens: int = 1) -> None:
        """Block until `tokens` are available for `key` (global across processes)."""
        ...


@runtime_checkable
class BlobStore(Protocol):
    async def put(self, key: str, data: bytes) -> None: ...

    async def get(self, key: str) -> bytes: ...

    async def delete(self, key: str) -> None: ...


@runtime_checkable
class DocumentParser(Protocol):
    async def parse(self, filename: str, data: bytes) -> SourceDocument:
        """Runs in a sandbox (subprocess, time/memory/size limits)."""
        ...


# ----------------------------------------------------------------- D: models
@runtime_checkable
class ChatModel(Protocol):
    provider: str
    model: str

    def stream(
        self,
        system: str,
        messages: Sequence[ChatMessage],
        tools: Sequence[ToolSpec] = (),
    ) -> AsyncIterator[ChatStreamEvent]:
        """Yields TextDelta / ToolCall (complete, one per call) / Usage (exactly once, last).
        Cancellation: the consumer calls `aclose()` on the iterator; implementations must stop
        the upstream request promptly."""
        ...

    async def complete_json(
        self,
        system: str,
        messages: Sequence[ChatMessage],
        schema: dict[str, object],
    ) -> tuple[dict[str, object], ChatTurnUsage]:
        """Structured output via JSON-schema mode (one repair retry is the caller's policy)."""
        ...


@runtime_checkable
class ConversationRuns(Protocol):
    """Run lease + approvals bookkeeping (owner D). Resume and new-turn are serialized: both
    acquire the lease first, then lock the run's pending approvals FOR UPDATE (DESIGN section 8)."""

    async def acquire(self, tx: Tx, conversation_id: UUID) -> UUID:
        """CAS on conversations.active_run_id; returns run_id or raises errors.TurnInProgress."""
        ...

    async def heartbeat(self, tenant_id: UUID, conversation_id: UUID, run_id: UUID) -> bool:
        """Own short tenant transaction (RLS). False = lease lost: stop the run."""
        ...

    async def release(self, tenant_id: UUID, conversation_id: UUID, run_id: UUID) -> None:
        """Own short tenant transaction; only clears the lease if run_id still holds it."""
        ...

    async def lock_pending(self, tx: Tx, run_id: UUID) -> Sequence[PendingApproval]: ...

    async def expire_pending(self, tx: Tx, conversation_id: UUID) -> int: ...


@runtime_checkable
class AgentRunner(Protocol):
    """The turn-level seam between the agent (D) and the API (E). E authenticates, applies the
    OutputSanitizer to TurnText/TurnReplace, and maps TurnEvents to SSE ChatEvents."""

    def run_turn(
        self,
        principal: Principal,
        conversation_id: UUID | None,
        collection_ids: Sequence[UUID],
        message: str,
    ) -> AsyncIterator[TurnEvent]: ...

    def resume(
        self, principal: Principal, run_id: UUID, decisions: dict[str, ApprovalChoice]
    ) -> AsyncIterator[TurnEvent]: ...


# ----------------------------------------------------------------- shared: secrets
@runtime_checkable
class SecretBox(Protocol):
    """Envelope encryption for connector tokens (implemented in Phase 0: adapters/crypto).
    E seals on connect; C opens when syncing. Never log either side."""

    def seal(self, plaintext: bytes, *, associated_data: bytes) -> SealedSecret: ...

    def open(self, sealed: SealedSecret, *, associated_data: bytes) -> bytes: ...


# ----------------------------------------------------------------- E: output safety
@runtime_checkable
class OutputSanitizer(Protocol):
    """Incremental, stateful. One instance per streamed answer (ADR-0011).

    Called ONLY by module E (the API), never by D: markdown escaping is not idempotent, so
    sanitizing twice corrupts answers. E calls `allow()` for every TurnSources event before
    feeding later text.
    """

    def allow(self, urls: Sequence[str]) -> None:
        """Add canonical external_urls to the link allow-list (exact normalized match)."""
        ...

    def feed(self, text: str) -> str:
        """Consume model output; return the safe text that may be emitted now."""
        ...

    def finish(self) -> str:
        """Flush held text (escaped/inert if unresolved)."""
        ...
