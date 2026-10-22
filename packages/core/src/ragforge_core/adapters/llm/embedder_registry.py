"""EmbedderRegistry: space_id -> Embedder, built from rf.embedding_spaces rows (ADR-0009).

A space pins provider + model + dim + prefixes, so the registry is the only place that turns a
space id into a concrete client; nothing else may pick a model for stored vectors. Spaces are
global (not tenant data) and change only via migrations/runbooks, so rows are loaded at startup
(`load_embedding_spaces`).

A space created after startup (blue/green re-embed, ADR-0009) is picked up without a restart:
`for_space` is synchronous (frozen port), so a miss raises the retryable
`EmbeddingSpaceUnavailable` (503) and starts ONE background reload through the injected
`loader`; the next call (job retry, next request) finds the space. A miss is never a permanent
job failure, so sync jobs for the new space retry instead of going dead. Every production caller
(agent retrieval, /v1/query, sync indexing) awaits `ensure(space_id)` first (port contract), so
the first use after a flip waits for one reload instead of failing. A space that is known but
unusable (retired, or a provider this deployment has no adapter for) also triggers a reload on
lookup, at most once per `UNUSABLE_RECHECK_S`, so an un-retired space recovers without a
restart.

Constructing an embedder makes no network call: a missing model or a wrong dim surfaces on the
first embed (as `EmbeddingFailed`), not at startup.

Error messages are fixed and never carry the space id or provider: they reach API clients and
jobs.last_error.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass
from typing import NoReturn

import httpx
from sqlalchemy import text

from ragforge_core.adapters.db.tx import connection
from ragforge_core.adapters.llm.embedder_ollama import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_KEEP_ALIVE,
    DEFAULT_NUM_CTX,
    OllamaEmbedder,
)
from ragforge_core.domain.errors import RagforgeError, ValidationFailed
from ragforge_core.domain.ports import Embedder, Tx
from ragforge_core.observability.logging import get_logger

log = get_logger(__name__)

_SPACES = text(
    "SELECT id, provider, model, dim, doc_prefix, query_prefix, distance, status "
    "FROM rf.embedding_spaces ORDER BY id"
)


@dataclass(frozen=True, slots=True)
class EmbeddingSpace:
    id: int
    provider: str
    model: str
    dim: int
    doc_prefix: str
    query_prefix: str
    distance: str
    status: str  # building | active | retired


async def load_embedding_spaces(tx: Tx) -> list[EmbeddingSpace]:
    """Read every space (global table; any transaction works, including a system one)."""
    result = await connection(tx).execute(_SPACES)
    return [
        EmbeddingSpace(
            id=int(r[0]),
            provider=str(r[1]),
            model=str(r[2]),
            dim=int(r[3]),
            doc_prefix=str(r[4]),
            query_prefix=str(r[5]),
            distance=str(r[6]),
            status=str(r[7]),
        )
        for r in result
    ]


SpaceLoader = Callable[[], Awaitable[list[EmbeddingSpace]]]

_now = time.monotonic  # module-level so tests can drive the recheck window
UNUSABLE_RECHECK_S = 30.0  # at most one reload per window triggered by unusable-space lookups
_RETIRED = "retired"
_FOREIGN = "foreign_provider"


class EmbeddingSpaceUnavailable(RagforgeError):
    """The space is not (yet) known to this process. Retryable: a reload is under way. The
    message is fixed and carries no space id (it reaches API clients and jobs.last_error)."""

    status = 503
    title = "Embedding space not available"

    def __init__(self) -> None:
        super().__init__("embedding space not available yet; retry shortly")


class EmbeddingSpaceRetired(ValidationFailed):
    """Definitive: the collection points at a retired space (a runbook must flip it)."""

    def __init__(self) -> None:
        super().__init__("embedding space is retired")


class EmbeddingSpaceUnsupported(RagforgeError):
    """The space's provider has no adapter in this deployment (configuration, retryable once
    the deployment is fixed)."""

    status = 503
    title = "Embedding provider not available"

    def __init__(self) -> None:
        super().__init__("embedding provider not available in this deployment")


class OllamaEmbedderRegistry:
    """Implements the `EmbedderRegistry` port for provider 'ollama'. One embedder object per
    usable space (no network call; see the module docstring)."""

    def __init__(
        self,
        spaces: Iterable[EmbeddingSpace],
        client: httpx.AsyncClient,
        *,
        num_ctx: int = DEFAULT_NUM_CTX,
        keep_alive: str = DEFAULT_KEEP_ALIVE,
        batch_size: int = DEFAULT_BATCH_SIZE,
        loader: SpaceLoader | None = None,
    ) -> None:
        self._client = client
        self._num_ctx = num_ctx
        self._keep_alive = keep_alive
        self._batch_size = batch_size
        self._loader = loader
        self._embedders: dict[int, OllamaEmbedder] = {}
        self._unusable: dict[int, str] = {}
        self._refresh_lock = asyncio.Lock()
        self._pending: asyncio.Task[bool] | None = None
        self._last_unusable_refresh = float("-inf")
        self._generation = 0  # bumped by every successful reload (single-flight in refresh)
        self._closed = False
        self._apply(spaces)

    def _apply(self, spaces: Iterable[EmbeddingSpace]) -> None:
        """Add/update spaces. An existing embedder is replaced only if its space changed state
        (e.g. retired); unchanged spaces keep their embedder object."""
        for space in spaces:
            if space.status == "retired":
                self._embedders.pop(space.id, None)
                self._unusable[space.id] = _RETIRED
            elif space.provider != "ollama":
                self._embedders.pop(space.id, None)
                self._unusable[space.id] = _FOREIGN
            elif space.id not in self._embedders:
                self._unusable.pop(space.id, None)
                self._embedders[space.id] = OllamaEmbedder(
                    self._client,
                    model=space.model,
                    dim=space.dim,
                    doc_prefix=space.doc_prefix,
                    query_prefix=space.query_prefix,
                    num_ctx=self._num_ctx,
                    keep_alive=self._keep_alive,
                    batch_size=self._batch_size,
                )

    def for_space(self, space_id: int) -> Embedder:
        embedder = self._embedders.get(space_id)
        if embedder is not None:
            return embedder
        reason = self._unusable.get(space_id)
        if reason is not None:
            now = _now()
            if now - self._last_unusable_refresh >= UNUSABLE_RECHECK_S:
                self._last_unusable_refresh = now
                self._schedule_refresh()
            self._raise_unusable(space_id)
        self._schedule_refresh()
        raise EmbeddingSpaceUnavailable()

    async def ensure(self, space_id: int) -> None:
        """Port: answer from current state. Unknown space -> one awaited reload, then decide
        (still unknown -> EmbeddingSpaceUnavailable, no further reload).

        A space cached as unusable (retired / unsupported provider) is re-read too, awaited and
        rate-limited to one reload per UNUSABLE_RECHECK_S: a blue/green rollback may have
        un-retired it. Only a FRESH read may produce the definitive (permanent) retired error;
        inside the rate-limit window the cache may be stale, so the caller gets the retryable
        EmbeddingSpaceUnavailable instead and a background reload is scheduled. A stale cache
        therefore never dead-letters a sync job."""
        if space_id in self._embedders:
            return
        if space_id in self._unusable:
            now = _now()
            if now - self._last_unusable_refresh < UNUSABLE_RECHECK_S:
                self._schedule_refresh()
                raise EmbeddingSpaceUnavailable()
            self._last_unusable_refresh = now
        fresh = await self.refresh()
        if space_id in self._embedders:
            return
        if not fresh or space_id not in self._unusable:
            # A failed reload leaves the cache possibly stale: never answer "retired" from it.
            raise EmbeddingSpaceUnavailable()
        self._raise_unusable(space_id)

    def _raise_unusable(self, space_id: int) -> NoReturn:
        if self._unusable.get(space_id) == _RETIRED:
            raise EmbeddingSpaceRetired()
        raise EmbeddingSpaceUnsupported()

    async def refresh(self) -> bool:
        """Re-read embedding_spaces (one short system read through the loader; no model call).
        Returns True when the registry now reflects a successful read (this one, or one that
        finished while this caller waited: single-flight), False when the read failed and the
        registry kept its current, possibly stale, spaces. Without a loader the constructor's
        spaces are the only truth there is, so that counts as fresh."""
        if self._loader is None:
            return True
        generation = self._generation
        async with self._refresh_lock:
            if self._generation != generation:
                return True  # only a successful reload advances the generation
            try:
                spaces = await self._loader()
            except Exception as exc:
                log.warning("embedding_spaces_reload_failed", error=type(exc).__name__)
                return False
            self._apply(spaces)
            self._generation += 1
            log.info("embedding_spaces_reloaded", spaces=sorted(self._embedders))
            return True

    async def aclose(self) -> None:
        """Cancel a reload still in flight (it would otherwise run against a disposed engine)."""
        self._closed = True
        task, self._pending = self._pending, None
        if task is not None and not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    def _schedule_refresh(self) -> None:
        if self._closed or self._loader is None:
            return
        if self._pending is not None and not self._pending.done():
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:  # called outside an event loop: nothing to schedule on
            return
        self._pending = loop.create_task(self.refresh())
