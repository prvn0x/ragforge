"""Module E2 test support: fakes for other modules' ports and data builders.

Imported as `module_e2.e2_support` (packages/core/tests is on sys.path via the root conftest and on
mypy_path), so the module name cannot clash with other modules' helpers.
"""

from __future__ import annotations

import dataclasses
import hashlib
import secrets
from collections.abc import AsyncIterator, Mapping, Sequence
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import dataclass, field
from uuid import UUID, uuid4

import psycopg

from ragforge_core.adapters.blob.local import LocalBlobStore
from ragforge_core.domain.models import (
    ApprovalChoice,
    JobLease,
    JobSpec,
    Principal,
    RankedChunk,
    TurnDone,
    TurnEvent,
)
from ragforge_core.domain.ports import Embedder, TransactionManager, Tx
from ragforge_core.services.auth.keys import generate_key

Conn = psycopg.Connection[tuple[object, ...]]
ALL = ("read", "write", "chat", "approvals:write", "admin")


# ----------------------------------------------------------------- data builders
def issue_key(
    superuser: Conn,
    tenant: UUID,
    user: UUID,
    scopes: Sequence[str] = ALL,
    *,
    name: str = "test",
    client_kind: str | None = None,
    revoked: bool = False,
) -> str:
    """Insert an API key row directly (as superuser) and return its plaintext."""
    key = generate_key()
    superuser.execute(
        "INSERT INTO rf.api_keys (tenant_id, user_id, name, prefix, key_sha256, scopes, "
        "client_kind, revoked_at) VALUES (%s, %s, %s, %s, %s, %s, %s, "
        "CASE WHEN %s THEN now() END)",
        (tenant, user, name, key.prefix, key.sha256, list(scopes), client_kind, revoked),
    )
    return key.plaintext


def auth(key: str) -> dict[str, str]:
    return {"authorization": f"Bearer {key}"}


def make_hit(
    collection: UUID, *, url: str | None = "https://notion.so/page-1", n: int = 0
) -> RankedChunk:
    return RankedChunk(
        chunk_id=uuid4(),
        document_id=uuid4(),
        collection_id=collection,
        anchor=f"block-{n}",
        title=f"Doc {n}",
        heading_path=("H1", "H2"),
        display_text="chunk text " * 5,
        display_sha256=hashlib.sha256(str(n).encode()).digest(),
        external_url=url,
        fused_score=0.5 - n * 0.01,
        rerank_score=0.9,
    )


@dataclass
class TenantWorld:
    """One tenant with one of everything the API exposes by id."""

    tenant: UUID
    user: UUID
    collection: UUID
    key: str
    connector: UUID
    document: UUID
    exclusion: UUID
    job: UUID
    history_job: UUID
    conversation: UUID
    run: UUID


def build_world(superuser: Conn, tenant: UUID, user: UUID, collection: UUID) -> TenantWorld:
    key = issue_key(superuser, tenant, user)
    connector, document, exclusion, conversation = uuid4(), uuid4(), uuid4(), uuid4()
    job, history_job, run = uuid4(), uuid4(), uuid4()
    ex = superuser.execute
    ex(
        "INSERT INTO rf.connectors (id, tenant_id, kind, target_collection_id, display_name) "
        "VALUES (%s, %s, 'notion', %s, 'c')",
        (connector, tenant, collection),
    )
    ex(
        "INSERT INTO rf.documents (id, tenant_id, collection_id, connector_id, source_kind, "
        "external_id, title) VALUES (%s, %s, %s, %s, 'notion', %s, 'Doc')",
        (document, tenant, collection, connector, f"page-{document.hex[:8]}"),
    )
    ex(
        "INSERT INTO rf.exclusions (id, tenant_id, source_kind, external_id) "
        "VALUES (%s, %s, 'notion', %s)",
        (exclusion, tenant, f"excluded-{exclusion.hex[:8]}"),
    )
    ex(
        "INSERT INTO rf.jobs (id, tenant_id, kind, priority, dedupe_key) "
        "VALUES (%s, %s, 'sync_connector', 20, %s)",
        (job, tenant, f"k-{job}"),
    )
    ex(
        "INSERT INTO rf.jobs_history (id, tenant_id, kind, priority, dedupe_key, status, "
        "finished_at) VALUES (%s, %s, 'sync_page', 20, %s, 'done', now())",
        (history_job, tenant, f"k-{history_job}"),
    )
    ex(
        "INSERT INTO rf.conversations (id, tenant_id, user_id, collection_ids) "
        "VALUES (%s, %s, %s, %s)",
        (conversation, tenant, user, [collection]),
    )
    ex(
        "INSERT INTO rf.pending_approvals (tenant_id, conversation_id, run_id, tool_call_id, "
        "tool, args) VALUES (%s, %s, %s, 'call-1', 'remove_from_index', '{}')",
        (tenant, conversation, run),
    )
    return TenantWorld(
        tenant=tenant,
        user=user,
        collection=collection,
        key=key,
        connector=connector,
        document=document,
        exclusion=exclusion,
        job=job,
        history_job=history_job,
        conversation=conversation,
        run=run,
    )


# ----------------------------------------------------------------- fakes for other modules' ports
class TrackingTxManager:
    """Wraps the real TransactionManager and counts open transactions, so tests can prove that
    no model call happens while a transaction is open. `completed` counts finished
    transactions, so a test can wait for a route's read to have happened."""

    def __init__(self, inner: TransactionManager) -> None:
        self._inner = inner
        self.open = 0
        self.completed = 0

    def tenant(self, tenant_id: UUID) -> AbstractAsyncContextManager[Tx]:
        return self._track(self._inner.tenant(tenant_id))

    def system(self) -> AbstractAsyncContextManager[Tx]:
        return self._track(self._inner.system())

    @asynccontextmanager
    async def _track(self, cm: AbstractAsyncContextManager[Tx]) -> AsyncIterator[Tx]:
        async with cm as tx:
            self.open += 1
            try:
                yield tx
            finally:
                self.open -= 1
                self.completed += 1


@dataclass
class FakeJobQueue:
    enqueued: list[JobSpec] = field(default_factory=list)

    async def enqueue(self, tx: Tx, spec: JobSpec) -> UUID:
        self.enqueued.append(spec)
        return uuid4()

    async def claim(self, worker_id: str, n: int) -> Sequence[JobLease]:
        raise NotImplementedError

    async def assert_lease(self, tx: Tx, lease: JobLease) -> None:
        raise NotImplementedError

    async def report_progress(self, tx: Tx, lease: JobLease, progress: dict[str, object]) -> None:
        raise NotImplementedError

    async def complete(self, tx: Tx, lease: JobLease) -> None:
        raise NotImplementedError

    async def heartbeat(self, lease: JobLease) -> bool:
        raise NotImplementedError

    async def release(self, lease: JobLease) -> None:
        raise NotImplementedError

    async def fail(
        self, lease: JobLease, error: str, retry_after_s: float | None, *, permanent: bool = False
    ) -> None:
        raise NotImplementedError


class FakeEmbedder:
    model = "fake"
    dim = 3

    def __init__(self, txm: TrackingTxManager, space: int) -> None:
        self._txm = txm
        self._space = space
        self.open_tx_during_calls: list[int] = []

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        raise NotImplementedError

    async def embed_query(self, text: str) -> list[float]:
        self.open_tx_during_calls.append(self._txm.open)
        return [float(self._space), 0.0, 1.0]


class FakeEmbedders:
    def __init__(self, txm: TrackingTxManager) -> None:
        self.by_space: dict[int, FakeEmbedder] = {}
        self._txm = txm

    async def ensure(self, space_id: int) -> None:
        return None

    def for_space(self, space_id: int) -> Embedder:
        return self.by_space.setdefault(space_id, FakeEmbedder(self._txm, space_id))


class FakeReranker:
    """Identity reranker that records how many transactions were open during each call."""

    def __init__(self, txm: TrackingTxManager) -> None:
        self._txm = txm
        self.open_tx_during_calls: list[int] = []
        self.pools: list[int] = []

    async def rerank(self, query: str, candidates: Sequence[RankedChunk]) -> list[RankedChunk]:
        self.open_tx_during_calls.append(self._txm.open)
        self.pools.append(len(candidates))
        return list(candidates)


@dataclass
class FakeRetriever:
    hits: list[RankedChunk] = field(default_factory=list)
    spaces: tuple[int, ...] = (1, 2)
    calls: list[tuple[str, dict[int, list[float]], list[UUID], int]] = field(default_factory=list)

    async def retrieve(
        self,
        tx: Tx,
        query: str,
        query_vectors: Mapping[int, Sequence[float]],
        collection_ids: Sequence[UUID],
        k: int,
    ) -> list[RankedChunk]:
        vectors = {s: list(v) for s, v in query_vectors.items()}
        self.calls.append((query, vectors, list(collection_ids), k))
        return self.hits[:k]

    async def spaces_for(self, tx: Tx, collection_ids: Sequence[UUID]) -> Sequence[int]:
        return self.spaces


@dataclass(frozen=True)
class TurnCall:
    principal: Principal
    conversation_id: UUID | None
    collection_ids: list[UUID]
    message: str


@dataclass(frozen=True)
class ResumeCall:
    principal: Principal
    run_id: UUID
    decisions: dict[str, ApprovalChoice]


@dataclass
class ScriptedAgent:
    """AgentRunner fake: yields `script`, optionally raising `fail` afterwards."""

    script: list[TurnEvent] = field(default_factory=list)
    fail: BaseException | None = None
    fail_before_first: BaseException | None = None
    # Mimic a correct agent (TurnDone echoes the id it was given); False replays the script as is.
    echo_conversation_id: bool = True
    turns: list[TurnCall] = field(default_factory=list)
    resumes: list[ResumeCall] = field(default_factory=list)
    closed: int = 0

    @property
    def called(self) -> bool:
        return bool(self.turns or self.resumes)

    def run_turn(
        self,
        principal: Principal,
        conversation_id: UUID | None,
        collection_ids: Sequence[UUID],
        message: str,
    ) -> AsyncIterator[TurnEvent]:
        self.turns.append(TurnCall(principal, conversation_id, list(collection_ids), message))
        return self._events(conversation_id)

    def resume(
        self, principal: Principal, run_id: UUID, decisions: dict[str, ApprovalChoice]
    ) -> AsyncIterator[TurnEvent]:
        self.resumes.append(ResumeCall(principal, run_id, decisions))
        return self._events()

    async def _events(self, conversation_id: UUID | None = None) -> AsyncIterator[TurnEvent]:
        try:
            if self.fail_before_first is not None:
                raise self.fail_before_first
            for event in self.script:
                if (
                    self.echo_conversation_id
                    and isinstance(event, TurnDone)
                    and conversation_id is not None
                ):
                    event = dataclasses.replace(event, conversation_id=conversation_id)
                yield event
            if self.fail is not None:
                raise self.fail
        finally:
            self.closed += 1


class MarkingSanitizer:
    """Marks everything it emits, so a test can tell sanitized text from raw text.

    feed(t)  -> "<s>t</s>" except it HOLDS the text after a trailing "[" (like a pending link
    construct), which only finish() releases, as "<f>held</f>".
    """

    def __init__(self, log: list[tuple[int, str, object]], ident: int) -> None:
        self._log = log
        self._id = ident
        self._held = ""
        self.allowed: list[str] = []

    def allow(self, urls: Sequence[str]) -> None:
        self.allowed.extend(urls)
        self._log.append((self._id, "allow", tuple(urls)))

    def feed(self, text: str) -> str:
        self._log.append((self._id, "feed", text))
        text = self._held + text
        self._held = ""
        if "[" in text:
            text, _, rest = text.partition("[")
            self._held = "[" + rest
        return f"<s>{text}</s>" if text else ""

    def finish(self) -> str:
        self._log.append((self._id, "finish", None))
        held, self._held = self._held, ""
        return f"<f>{held}</f>" if held else ""


class SanitizerFactory:
    def __init__(self) -> None:
        self.log: list[tuple[int, str, object]] = []
        self.instances: list[MarkingSanitizer] = []

    def __call__(self) -> MarkingSanitizer:
        sanitizer = MarkingSanitizer(self.log, len(self.instances))
        self.instances.append(sanitizer)
        return sanitizer


@dataclass
class Fakes:
    txm: TrackingTxManager
    jobs: FakeJobQueue
    retriever: FakeRetriever
    embedders: FakeEmbedders
    reranker: FakeReranker
    agent: ScriptedAgent
    sanitizers: SanitizerFactory
    blobs: LocalBlobStore


def random_token() -> str:
    return "secret_" + secrets.token_urlsafe(24)
