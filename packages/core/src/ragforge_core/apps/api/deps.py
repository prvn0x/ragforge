"""FastAPI dependencies: ports resolved from app state, authentication and scope checks.

Ports are read from `app.state.container` by attribute name, so Phase 2 only has to add fields to
`bootstrap.ApiContainer` (see docs/handoff/E2.md for the names). Tests replace any of these with
`app.dependency_overrides[...]`, FastAPI's own seam. A port that is not wired yet answers 503
instead of crashing with an AttributeError.
"""

from __future__ import annotations

import math
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from typing import Annotated, cast
from uuid import UUID

import structlog
from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from ragforge_core.adapters.db.tx import connection
from ragforge_core.agent.interview.model import InterviewSettings, Level
from ragforge_core.agent.interview.tokens import QuestionTokens
from ragforge_core.apps.api.jobfeed import JobFeed, PollingJobFeed
from ragforge_core.apps.api.sql import fetch_one
from ragforge_core.apps.api.stores import SqlKeyStore
from ragforge_core.domain.errors import Forbidden, NotFound, RagforgeError
from ragforge_core.domain.models import Principal, Scope
from ragforge_core.domain.ports import (
    AgentRunner,
    BlobStore,
    ChatModel,
    EmbedderRegistry,
    JobQueue,
    OutputSanitizer,
    Reranker,
    Retriever,
    SecretBox,
    TransactionManager,
)
from ragforge_core.rag.retriever import RERANK_TOP
from ragforge_core.services.auth.ports import KeyStore
from ragforge_core.services.auth.principal import AuthenticatedKey, Authenticator
from ragforge_core.services.auth.scopes import require_key_manager, require_scope
from ragforge_core.services.sync.ports import SyncStore
from ragforge_core.services.sync.uploads import MAX_UPLOAD_BYTES

SanitizerFactory = Callable[[], OutputSanitizer]

_bearer = HTTPBearer(auto_error=False, description="API key `rf_<prefix>_<secret>`")


class PortNotConfigured(RagforgeError):
    status = 503
    title = "Service not configured"


def _port(request: Request, name: str) -> object:
    port = getattr(request.app.state.container, name, None)
    if port is None:
        raise PortNotConfigured(f"`{name}` is not wired in this deployment")
    return port


def get_tx_manager(request: Request) -> TransactionManager:
    return cast(TransactionManager, _port(request, "tx"))


def get_secret_box(request: Request) -> SecretBox:
    return cast(SecretBox, _port(request, "secrets"))


def get_job_queue(request: Request) -> JobQueue:
    return cast(JobQueue, _port(request, "jobs"))


def get_retriever(request: Request) -> Retriever:
    return cast(Retriever, _port(request, "retriever"))


def get_embedders(request: Request) -> EmbedderRegistry:
    return cast(EmbedderRegistry, _port(request, "embedders"))


def get_reranker(request: Request) -> Reranker:
    return cast(Reranker, _port(request, "reranker"))


def get_blob_store(request: Request) -> BlobStore:
    return cast(BlobStore, _port(request, "blobs"))


def get_sync_store(request: Request) -> SyncStore:
    return cast(SyncStore, _port(request, "sync_store"))


def _setting(request: Request, name: str, default: int) -> int:
    settings = getattr(request.app.state.container, "settings", None)
    value = getattr(settings, name, None)
    return value if isinstance(value, int) else default


def get_rerank_top(request: Request) -> int:
    """Fused candidates fed to the reranker (settings.rerank_top_n, ADR-0006)."""
    return _setting(request, "rerank_top_n", RERANK_TOP)


def get_max_upload_bytes(request: Request) -> int:
    """settings.max_upload_mb, never above the parser's own 50 MB cap."""
    mb = _setting(request, "max_upload_mb", MAX_UPLOAD_BYTES // (1024 * 1024))
    return min(mb * 1024 * 1024, MAX_UPLOAD_BYTES)


def get_agent_runner(request: Request) -> AgentRunner:
    return cast(AgentRunner, _port(request, "agent"))


def get_sanitizer_factory(request: Request) -> SanitizerFactory:
    """A factory, not an instance: the sanitizer is stateful and one is needed per answer."""
    return cast(SanitizerFactory, _port(request, "sanitizer_factory"))


def get_job_feed(request: Request) -> JobFeed:
    feed = getattr(request.app.state, "job_feed", None)
    return cast(JobFeed, feed) if feed is not None else PollingJobFeed()


@dataclass(frozen=True, slots=True)
class JobStreamConfig:
    """Job SSE timing. The poll is the correctness path; LISTEN only lowers latency.

    `reauth_s`: how often a long-lived stream re-checks that its API key is still unrevoked, so
    revocation also ends streams that were opened before it (bounded staleness, not per poll).
    """

    poll_interval_s: float = 2.0
    keepalive_s: float = 15.0
    reauth_s: float = 30.0


def get_job_stream_config() -> JobStreamConfig:
    return JobStreamConfig()


TxManager = Annotated[TransactionManager, Depends(get_tx_manager)]


def get_key_store(txm: TxManager) -> KeyStore:
    return SqlKeyStore(txm)


async def authenticate(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)],
    store: Annotated[KeyStore, Depends(get_key_store)],
) -> AuthenticatedKey:
    key = await Authenticator(store).authenticate(credentials.credentials if credentials else None)
    structlog.contextvars.bind_contextvars(tenant_id=str(key.principal.tenant_id))
    return key


def require(scope: Scope) -> Callable[[AuthenticatedKey], Awaitable[AuthenticatedKey]]:
    async def dependency(
        key: Annotated[AuthenticatedKey, Depends(authenticate)],
    ) -> AuthenticatedKey:
        require_scope(key.principal, scope)
        return key

    return dependency


async def require_key_manager_dep(
    key: Annotated[AuthenticatedKey, Depends(authenticate)],
) -> AuthenticatedKey:
    """Tenant self-administration (keys, audit): `approvals:write` or `admin` (scopes.py)."""
    require_key_manager(key.principal)
    return key


AnyKey = Annotated[AuthenticatedKey, Depends(authenticate)]
KeyManagerKey = Annotated[AuthenticatedKey, Depends(require_key_manager_dep)]
ReadKey = Annotated[AuthenticatedKey, Depends(require(Scope.read))]
WriteKey = Annotated[AuthenticatedKey, Depends(require(Scope.write))]
ChatKey = Annotated[AuthenticatedKey, Depends(require(Scope.chat))]
ApprovalsKey = Annotated[AuthenticatedKey, Depends(require(Scope.approvals_write))]
AdminKey = Annotated[AuthenticatedKey, Depends(require(Scope.admin))]

Secrets = Annotated[SecretBox, Depends(get_secret_box)]
Jobs = Annotated[JobQueue, Depends(get_job_queue)]
Retrieval = Annotated[Retriever, Depends(get_retriever)]
Embedders = Annotated[EmbedderRegistry, Depends(get_embedders)]
Rerank = Annotated[Reranker, Depends(get_reranker)]
RerankTop = Annotated[int, Depends(get_rerank_top)]
Blobs = Annotated[BlobStore, Depends(get_blob_store)]
SyncStores = Annotated[SyncStore, Depends(get_sync_store)]
MaxUploadBytes = Annotated[int, Depends(get_max_upload_bytes)]
Agent = Annotated[AgentRunner, Depends(get_agent_runner)]
Sanitizers = Annotated[SanitizerFactory, Depends(get_sanitizer_factory)]
Feed = Annotated[JobFeed, Depends(get_job_feed)]
StreamConfig = Annotated[JobStreamConfig, Depends(get_job_stream_config)]


# ----------------------------------------------------------------- interview mode (Phase 3b)
def get_chat_model(request: Request) -> ChatModel:
    """The chat model the agent uses (bootstrap exposes the same instance)."""
    return cast(ChatModel, _port(request, "chat"))


def get_question_tokens(request: Request) -> QuestionTokens:
    return cast(QuestionTokens, _port(request, "interview_tokens"))


def get_interview_settings(request: Request) -> InterviewSettings:
    """Interview tunables from settings; the dataclass defaults where a test container has none."""
    settings = getattr(request.app.state.container, "settings", None)
    if settings is None:
        return InterviewSettings()
    return InterviewSettings(
        tau=settings.agent_default_tau,
        rerank_top=settings.rerank_top_n,
        verifier_trusted=settings.agent_verifier_trusted,
        target_level=Level(settings.interview_target_level),
    )


class TooManyRequests(HTTPException):
    """429 with Retry-After. An HTTPException (not a RagforgeError) because the problem handler
    preserves an HTTPException's headers (apps/api/errors.py) and Retry-After is the point."""

    def __init__(self, retry_after_s: float) -> None:
        seconds = max(1, math.ceil(retry_after_s))
        super().__init__(429, "Too many requests", headers={"retry-after": str(seconds)})


@dataclass(slots=True)
class _Bucket:
    tokens: float
    updated: float
    running: int = 0


class PrincipalLimiter:
    """Per-principal token bucket plus a concurrency cap, in process (one per API replica).

    WHY in process: it bounds what one principal can spend through one replica, which on this
    single-replica deployment is the whole budget, with no extra round trip per call. Several
    replicas multiply the effective limit by the replica count; a shared limiter (Postgres, like
    the Notion rate limiter) is the follow-up if that matters. Single event loop: every method
    runs without awaiting, so no lock is needed. Idle, full buckets are evicted past `max_keys`.
    """

    def __init__(
        self,
        *,
        rate_per_s: float,
        burst: int,
        concurrency: int,
        clock: Callable[[], float] = time.monotonic,
        max_keys: int = 10_000,
    ) -> None:
        if rate_per_s <= 0 or burst < 1 or concurrency < 1:
            raise ValueError("rate_per_s > 0, burst >= 1 and concurrency >= 1 are required")
        self._rate = rate_per_s
        self._burst = float(burst)
        self._concurrency = concurrency
        self._clock = clock
        self._max_keys = max_keys
        self._buckets: dict[tuple[UUID, UUID], _Bucket] = {}

    def acquire(self, key: tuple[UUID, UUID]) -> None:
        """Take one token and one in-flight slot, or raise TooManyRequests (taking nothing)."""
        bucket = self._refilled(key)
        if bucket.running >= self._concurrency:
            raise TooManyRequests(1.0)
        if bucket.tokens < 1.0:
            raise TooManyRequests((1.0 - bucket.tokens) / self._rate)
        bucket.tokens -= 1.0
        bucket.running += 1

    def release(self, key: tuple[UUID, UUID]) -> None:
        bucket = self._buckets.get(key)
        if bucket is not None and bucket.running > 0:
            bucket.running -= 1

    def _refilled(self, key: tuple[UUID, UUID]) -> _Bucket:
        now = self._clock()
        bucket = self._buckets.get(key)
        if bucket is None:
            self._evict(now)
            bucket = self._buckets[key] = _Bucket(self._burst, now)
        bucket.tokens = min(self._burst, bucket.tokens + (now - bucket.updated) * self._rate)
        bucket.updated = now
        return bucket

    def _evict(self, now: float) -> None:
        if len(self._buckets) < self._max_keys:
            return
        full = self._burst / self._rate
        for key, b in list(self._buckets.items()):
            if b.running == 0 and now - b.updated >= full:
                del self._buckets[key]


def _app_limiter(
    request: Request, name: str, *, rate_per_minute: float, burst: int, concurrency: int
) -> PrincipalLimiter:
    """One limiter per app and `name`, built on first use (app.state, like the job feed)."""
    limiter = getattr(request.app.state, name, None)
    if isinstance(limiter, PrincipalLimiter):
        return limiter
    limiter = PrincipalLimiter(
        rate_per_s=rate_per_minute / 60.0, burst=burst, concurrency=concurrency
    )
    setattr(request.app.state, name, limiter)
    return limiter


def get_interview_limiter(request: Request) -> PrincipalLimiter:
    """The interview quota, from settings (the defaults where a test container has none)."""
    settings = getattr(request.app.state.container, "settings", None)
    return _app_limiter(
        request,
        "interview_limiter",
        rate_per_minute=getattr(settings, "interview_rate_per_minute", 10.0),
        burst=getattr(settings, "interview_burst", 5),
        concurrency=getattr(settings, "interview_concurrency", 2),
    )


async def interview_quota(
    key: ChatKey,
    limiter: Annotated[PrincipalLimiter, Depends(get_interview_limiter)],
) -> AsyncIterator[AuthenticatedKey]:
    """Scope `chat` first (403 before any quota is spent), then the principal's quota. The
    slot is held until the route returns (yield dependency), so the concurrency cap covers the
    model calls."""
    principal = (key.principal.tenant_id, key.principal.user_id)
    limiter.acquire(principal)
    try:
        yield key
    finally:
        limiter.release(principal)


InterviewKey = Annotated[AuthenticatedKey, Depends(interview_quota)]
ChatModels = Annotated[ChatModel, Depends(get_chat_model)]
QuestionTokenSigner = Annotated[QuestionTokens, Depends(get_question_tokens)]
InterviewConfig = Annotated[InterviewSettings, Depends(get_interview_settings)]


# ----------------------------------------------------------------- eval surface (ADR-0017 §1)
def get_eval_api_enabled(request: Request) -> bool:
    """RAGFORGE_EVAL_API. Fails CLOSED: a container without settings (or a non-bool value) is
    off. Tests flip it with `dependency_overrides`, FastAPI's own seam."""
    settings = getattr(request.app.state.container, "settings", None)
    return getattr(settings, "eval_api", False) is True


EvalApiEnabled = Annotated[bool, Depends(get_eval_api_enabled)]


async def eval_surface(enabled: EvalApiEnabled) -> None:
    """/v1/eval/* does not exist while the switch is off: 404, before authentication, so an
    unauthenticated probe cannot tell a disabled surface from a missing one."""
    if not enabled:
        raise NotFound("Not Found")


_TENANT_KIND_SQL = "SELECT kind FROM rf.tenants WHERE id = rf.current_tenant()"


async def require_eval_key(
    _surface: Annotated[None, Depends(eval_surface)],
    key: Annotated[AuthenticatedKey, Depends(authenticate)],
    tx: Annotated[TransactionManager, Depends(get_tx_manager)],
) -> AuthenticatedKey:
    """The switch first (parameters resolve in order), then the key, then the `eval` scope,
    then the TENANT: the /v1/eval surface serves only an eval tenant (ADR-0017 §1), so an eval
    key minted into an owner tenant by any future path still cannot read live note text through
    /v1/eval/chunks (defence in depth; EV-API review P2)."""
    require_scope(key.principal, Scope.eval)
    async with tx.tenant(key.principal.tenant_id) as t:
        row = await fetch_one(connection(t), _TENANT_KIND_SQL)
    if row is None or row["kind"] != "eval":
        raise Forbidden("the eval surface serves only an eval tenant")
    return key


EvalKey = Annotated[AuthenticatedKey, Depends(require_eval_key)]


def require_eval_seam(principal: Principal, enabled: bool) -> None:
    """For the ADDITIVE eval seams on shared routes (/v1/query stages, chat ?trace=shown,
    interview chunk_ids): 403, never 404, because the route itself exists for everyone. WHY the
    switch too: ADR-0017 puts the whole surface behind RAGFORGE_EVAL_API, not just /v1/eval."""
    if not enabled:
        raise Forbidden("the eval surface is disabled on this server (RAGFORGE_EVAL_API)")
    require_scope(principal, Scope.eval)


def eval_seam_allowed(principal: Principal, enabled: bool) -> bool:
    """`require_eval_seam` as a predicate, for policy that exempts instead of refusing."""
    return enabled and principal.has(Scope.eval.value)


def get_eval_limiter(request: Request) -> PrincipalLimiter:
    """The /v1/eval/verify quota (settings.eval_*), separate from the interview budget so a
    harness run cannot starve the owner's interview session, and vice versa."""
    settings = getattr(request.app.state.container, "settings", None)
    return _app_limiter(
        request,
        "eval_limiter",
        rate_per_minute=getattr(settings, "eval_rate_per_minute", 30.0),
        burst=getattr(settings, "eval_burst", 10),
        concurrency=getattr(settings, "eval_concurrency", 2),
    )


async def eval_quota(
    key: EvalKey,
    limiter: Annotated[PrincipalLimiter, Depends(get_eval_limiter)],
) -> AsyncIterator[AuthenticatedKey]:
    """Switch, scope, then quota (no quota is spent on a refused call); the slot is held until
    the route returns, so the concurrency cap covers the verifier call."""
    principal = (key.principal.tenant_id, key.principal.user_id)
    limiter.acquire(principal)
    try:
        yield key
    finally:
        limiter.release(principal)


EvalQuotaKey = Annotated[AuthenticatedKey, Depends(eval_quota)]
