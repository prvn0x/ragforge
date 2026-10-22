"""Global per-integration rate limiter (RateLimiter port) backed by `rf.take_tokens` (slot 0004).

Why Postgres and not an in-process bucket: Notion limits per integration token, and every worker
process shares that one budget. A per-process bucket would multiply the allowed rate by the
number of workers (ADR-0007). `take_tokens` atomically refills and decrements one row per
integration, so the budget is shared by all processes and all coroutines.

Keys: callers pass `limiter_key(token)` ("sha256:<hex digest>"), never the token itself, so the
plaintext secret never crosses the generic RateLimiter port (where a test double, a metrics or
logging wrapper could record it). The bucket row is keyed by the raw digest, which is exactly
`sha256(token)` (ADR-0007). Any other key string is hashed, so a caller that still passes a raw
value never stores it either.

Max wait: when the shared bucket is in debt for longer than `max_wait_s` (a long upstream
Retry-After applied by `penalize`), `acquire` raises `UpstreamRateLimited(wait)` instead of
sleeping. Every worker then gives its job back to the queue the same way the worker that saw the
429 did, instead of holding a job slot and lease asleep for up to 10 minutes.
"""

from __future__ import annotations

import asyncio
import hashlib
import math
from collections.abc import Awaitable, Callable
from typing import Protocol, runtime_checkable

from sqlalchemy import TextClause, text
from sqlalchemy.ext.asyncio import AsyncEngine

from ragforge_core.db.engine import system_transaction
from ragforge_core.domain.errors import UpstreamRateLimited

# Must match the constants inside rf.take_tokens (0004_connectors.py).
NOTION_RATE_PER_S = 3.0
MAX_TOKENS_PER_CALL = 1800
# Above this, waiting in-process pins a worker slot; re-queueing is cheaper (matches the Notion
# client's `max_retry_after_s`).
DEFAULT_MAX_WAIT_S = 60.0
_KEY_PREFIX = "sha256:"

_TAKE = text("SELECT rf.take_tokens(:key, :n)")
_PENALIZE = text("SELECT rf.penalize_tokens(:key, :n)")

Sleep = Callable[[float], Awaitable[None]]


def limiter_key(token: str) -> str:
    """The non-secret RateLimiter key for an integration token: "sha256:<64 hex>".

    Safe to log or record: SHA-256 of a high-entropy token reveals nothing usable."""
    return _KEY_PREFIX + hashlib.sha256(token.encode("utf-8")).hexdigest()


def integration_key(key: str) -> bytes:
    """The `notion_rate.integration_key` (raw 32-byte SHA-256, never hex) for a limiter key.

    `limiter_key(token)` maps to `sha256(token)`, the same row as hashing the token directly, so
    the bucket identity does not depend on which form a caller used."""
    if key.startswith(_KEY_PREFIX):
        hex_digest = key[len(_KEY_PREFIX) :]
        if len(hex_digest) == 64:
            try:
                return bytes.fromhex(hex_digest)
            except ValueError:
                pass
    return hashlib.sha256(key.encode("utf-8")).digest()


def tokens_for_delay(seconds: float) -> int:
    """How many tokens drain the bucket for `seconds` (used to apply a Retry-After globally)."""
    return max(1, min(MAX_TOKENS_PER_CALL, math.ceil(seconds * NOTION_RATE_PER_S)))


@runtime_checkable
class PenalizingLimiter(Protocol):
    """Optional extension of the (frozen) RateLimiter port: apply an upstream Retry-After to the
    shared budget WITHOUT blocking the caller.

    `acquire(key, n)` cannot express this: it sleeps for the debt, so a caller that wants to give
    its job back to the queue on a long Retry-After could not also make every other worker back
    off. Limiters that do not implement it fall back to `acquire` for short delays only."""

    async def penalize(self, key: str, seconds: float) -> float:
        """Make `key`'s global bucket wait at least `seconds` from now; return the wait in s."""
        ...


class PostgresRateLimiter:
    """RateLimiter port. Each `acquire` is one short system transaction (no tenant needed: the
    bucket is keyed by integration, not tenant), so the row lock is held for microseconds and
    never inside a caller's long transaction."""

    def __init__(
        self,
        engine: AsyncEngine,
        *,
        sleep: Sleep = asyncio.sleep,
        max_wait_s: float | None = DEFAULT_MAX_WAIT_S,
    ) -> None:
        if max_wait_s is not None and max_wait_s < 0:
            raise ValueError("max_wait_s must be >= 0")
        self._engine = engine
        self._sleep = sleep
        self._max_wait_s = max_wait_s

    def __repr__(self) -> str:
        return "PostgresRateLimiter()"

    async def acquire(self, key: str, tokens: int = 1) -> None:
        """Take `tokens` from the key's global bucket, then sleep for the debt, if any.

        Requests above the per-call cap are split, so a long Retry-After still drains fully.
        A wait above `max_wait_s` raises `UpstreamRateLimited(wait)` instead of sleeping. The
        tokens stay taken (one wasted request slot, ~333 ms of extra debt per give-up): the
        alternative, a separate non-sleeping peek, would cost a second round trip on EVERY
        request to save a third of a second on rare ones, and a peek-then-take is racy anyway.
        """
        if tokens < 1:
            raise ValueError("tokens must be >= 1")
        digest = integration_key(key)
        remaining = tokens
        wait_ms = 0
        while remaining > 0:
            n = min(remaining, MAX_TOKENS_PER_CALL)
            wait_ms = await self._take(digest, n)
            remaining -= n
        wait_s = wait_ms / 1000.0
        if self._max_wait_s is not None and wait_s > self._max_wait_s:
            raise UpstreamRateLimited(wait_s)
        if wait_ms > 0:
            await self._sleep(wait_s)

    async def penalize(self, key: str, seconds: float) -> float:
        """Apply a Retry-After to the global bucket without sleeping (`rf.penalize_tokens`).

        Idempotent per window: N workers reporting the same 429 leave ONE wait of `seconds`,
        not N of them. Capped at MAX_TOKENS_PER_CALL (10 minutes) like `acquire`."""
        n = tokens_for_delay(seconds)
        wait_ms = await self._call(_PENALIZE, integration_key(key), n)
        return wait_ms / 1000.0

    async def _take(self, digest: bytes, n: int) -> int:
        return await self._call(_TAKE, digest, n)

    async def _call(self, statement: TextClause, digest: bytes, n: int) -> int:
        async with system_transaction(self._engine) as conn:
            result = await conn.execute(statement, {"key": digest, "n": n})
            return int(result.scalar_one())
