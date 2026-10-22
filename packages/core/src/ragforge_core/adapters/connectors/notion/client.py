"""Thin async Notion HTTP client: version pinning, global rate limiting, retries, pagination.

Secret handling: the integration token lives in a `SecretStr` and is materialized only into the
Authorization header of each request, built inline in the httpx call so no frame local holds it.
The rate limiter gets `limiter_key(token)` (a digest), never the token. It never appears in
`repr`, log events or exceptions, nor anywhere reachable from them:
* an httpx error references the request (and its headers), so `_send` records only its class
  name and raises our own error AFTER leaving the `except` block: `__cause__` AND `__context__`
  are both None (`from None` alone still leaves the httpx error in `__context__`, where error
  reporters that walk the chain would find it);
* the Authorization header is deleted from `response.request` as soon as a response arrives, so a
  `response` local captured with a traceback (Sentry `include_local_variables`, rich
  `show_locals`) holds no token either.

Retry policy:
* 429 -> the Retry-After is applied to the GLOBAL bucket (`PenalizingLimiter.penalize`), so
  every worker sharing the integration backs off, not just this coroutine (ADR-0007
  "Retry-After drains the bucket"). That happens even when the Retry-After is longer than
  `max_retry_after_s` and this call then raises `UpstreamRateLimited` at once (the job goes back
  to the queue instead of pinning a worker slot). Other workers do the same: their next
  `PostgresRateLimiter.acquire` sees a wait above its `max_wait_s` and raises
  `UpstreamRateLimited` instead of sleeping through it.
* 500/502/503/504 and transport errors -> capped exponential backoff with jitter (Retry-After
  wins when present).
* everything else -> raised immediately (`NotionNotFound` only for a 404 whose body carries
  Notion's `object_not_found` code).
"""

from __future__ import annotations

import asyncio
import email.utils
import math
import secrets
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import httpx
from pydantic import SecretStr

from ragforge_core.adapters.connectors.notion.errors import (
    NotionApiError,
    NotionNotFound,
    NotionProtocolError,
    NotionTransportError,
)
from ragforge_core.adapters.connectors.ratelimit import (
    PenalizingLimiter,
    limiter_key,
    tokens_for_delay,
)
from ragforge_core.domain.errors import UpstreamRateLimited
from ragforge_core.domain.ports import RateLimiter
from ragforge_core.observability.logging import get_logger
from ragforge_core.observability.metrics import count_notion

NOTION_VERSION = "2025-09-03"  # data-source model (ADR-0007); never float this
NOTION_API_BASE = "https://api.notion.com/v1"
PAGE_SIZE = 100  # Notion's maximum
NOT_FOUND_CODE = "object_not_found"

_RETRYABLE_STATUS = frozenset({500, 502, 503, 504})
_REDACTED = "[REDACTED]"
_jitter = secrets.SystemRandom()
log = get_logger(__name__)

Sleep = Callable[[float], Awaitable[None]]
Json = dict[str, Any]


@dataclass(frozen=True, slots=True)
class NotionResponse:
    data: Json
    date: datetime | None  # the HTTP Date header: Notion's clock, used for fetched_at


class NotionClient:
    """One client per integration token. Not tied to a tenant: callers decide what to fetch."""

    def __init__(
        self,
        token: str | SecretStr,
        limiter: RateLimiter,
        *,
        http: httpx.AsyncClient | None = None,
        base_url: str = NOTION_API_BASE,
        max_attempts: int = 5,
        backoff_base_s: float = 0.5,
        backoff_cap_s: float = 30.0,
        max_retry_after_s: float = 60.0,
        sleep: Sleep = asyncio.sleep,
    ) -> None:
        secret = token if isinstance(token, SecretStr) else SecretStr(token)
        if not secret.get_secret_value().strip():
            raise ValueError("a Notion integration token is required")
        if max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        self._token = secret
        self._limiter_key = limiter_key(secret.get_secret_value())
        self._limiter = limiter
        self._owns_http = http is None
        self._http = http or httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=10.0))
        self._base_url = base_url.rstrip("/")
        self._max_attempts = max_attempts
        self._backoff_base_s = backoff_base_s
        self._backoff_cap_s = backoff_cap_s
        self._max_retry_after_s = max_retry_after_s
        self._sleep = sleep

    def __repr__(self) -> str:
        return f"NotionClient(base_url={self._base_url!r})"

    async def aclose(self) -> None:
        if self._owns_http:
            await self._http.aclose()

    async def __aenter__(self) -> NotionClient:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    # ------------------------------------------------------------------ requests
    async def request(
        self,
        method: str,
        path: str,
        *,
        params: Mapping[str, str | int] | None = None,
        json: Json | None = None,
    ) -> NotionResponse:
        """One logical API call: rate-limited before EVERY attempt, retried per module policy."""
        last_error: Exception | None = None
        for attempt in range(1, self._max_attempts + 1):
            await self._limiter.acquire(self._limiter_key)
            try:
                response = await self._send(method, path, params, json)
            except NotionTransportError as exc:
                last_error = exc
                await self._backoff(attempt, None, method, path, "transport")
                continue
            if response.is_success:
                return NotionResponse(_json_body(response), _parse_http_date(response))
            retry_after = _parse_retry_after(response)
            if response.status_code == 429:
                last_error = UpstreamRateLimited(retry_after)
                await self._on_rate_limited(retry_after, attempt, method, path)
                continue
            if response.status_code in _RETRYABLE_STATUS:
                last_error = self._api_error(response)
                await self._backoff(attempt, retry_after, method, path, response.status_code)
                continue
            raise self._api_error(response)
        if last_error is None:  # unreachable (max_attempts >= 1); explicit so -O cannot drop it
            raise NotionProtocolError(f"no attempt was made calling Notion {path}")
        raise last_error

    async def paginate(
        self,
        method: str,
        path: str,
        *,
        body: Json | None = None,
        params: Mapping[str, str | int] | None = None,
    ) -> AsyncIterator[Json]:
        """Yield every `results` item across pages. GET endpoints take `start_cursor` as a query
        parameter, POST endpoints in the JSON body (Notion's convention)."""
        cursor: str | None = None
        seen: set[str] = set()
        while True:
            req_params: dict[str, str | int] = dict(params or {})
            req_body: Json | None = None
            if method.upper() == "GET":
                req_params.setdefault("page_size", PAGE_SIZE)
                if cursor:
                    req_params["start_cursor"] = cursor
            else:
                req_body = {**(body or {}), "page_size": PAGE_SIZE}
                if cursor:
                    req_body["start_cursor"] = cursor
            page = await self.request(method, path, params=req_params or None, json=req_body)
            for item in page.data.get("results") or []:
                if isinstance(item, dict):
                    yield item
            cursor = page.data.get("next_cursor") if page.data.get("has_more") else None
            if not cursor:
                return
            if cursor in seen:  # a server bug must not become an infinite (rate-limited) loop
                raise NotionProtocolError(f"pagination cursor repeated for {path}")
            seen.add(cursor)

    # ------------------------------------------------------------------ internals
    async def _send(
        self,
        method: str,
        path: str,
        params: Mapping[str, str | int] | None,
        json: Json | None,
    ) -> httpx.Response:
        failure: str
        try:
            response = await self._http.request(
                method,
                f"{self._base_url}{path}",
                params=params,
                json=json,
                headers={
                    "Authorization": f"Bearer {self._token.get_secret_value()}",
                    "Notion-Version": NOTION_VERSION,
                    "Accept": "application/json",
                },
            )
        except httpx.HTTPError as exc:
            failure = type(exc).__name__  # keep nothing else: exc.request carries the header
        else:
            response.request.headers.pop("Authorization", None)
            count_notion(response.status_code)  # per attempt, retries included
            return response
        count_notion(None)
        # Raised outside the except block, so no exception chain leads back to the httpx error.
        raise NotionTransportError(f"{failure} calling Notion {path}")

    async def _on_rate_limited(
        self, retry_after: float | None, attempt: int, method: str, path: str
    ) -> None:
        delay = retry_after if retry_after is not None else self._backoff_delay(attempt)
        log.warning(
            "notion.rate_limited", method=method, path=path, attempt=attempt, retry_after_s=delay
        )
        give_up = delay > self._max_retry_after_s or attempt >= self._max_attempts
        key = self._limiter_key
        if isinstance(self._limiter, PenalizingLimiter):
            # Applied BEFORE giving up: a long Retry-After is exactly when every other worker
            # of this integration must stop too, not only this coroutine. Non-blocking: when we
            # retry, the next `acquire` pays the wait; when we give up, the job is re-queued.
            await self._limiter.penalize(key, delay)
            if give_up:
                raise UpstreamRateLimited(delay)
            return
        # Plain RateLimiter port: draining means sleeping, so only do it when we will retry.
        if give_up:
            raise UpstreamRateLimited(delay)
        await self._limiter.acquire(key, tokens_for_delay(delay))

    async def _backoff(
        self,
        attempt: int,
        retry_after: float | None,
        method: str,
        path: str,
        reason: int | str,
    ) -> None:
        if attempt >= self._max_attempts:
            return
        delay = retry_after if retry_after is not None else self._backoff_delay(attempt)
        delay = min(delay, self._backoff_cap_s)
        log.warning(
            "notion.retry", method=method, path=path, attempt=attempt, reason=reason, delay_s=delay
        )
        await self._sleep(delay)

    def _backoff_delay(self, attempt: int) -> float:
        ceiling = min(self._backoff_cap_s, self._backoff_base_s * 2.0 ** (attempt - 1))
        return float(ceiling * (0.5 + _jitter.random() / 2))  # jitter de-synchronizes workers

    def _api_error(self, response: httpx.Response) -> NotionApiError:
        code, message = "unknown", ""
        try:
            body = response.json()
        except ValueError:
            body = None
        if isinstance(body, dict):
            code = str(body.get("code") or code)
            message = str(body.get("message") or "")
        message = self._scrub(message)[:500]
        # Only Notion's own `object_not_found` is evidence that the object is gone (or unshared).
        # A 404 from a proxy/edge (HTML body), a wrong base URL or a retired route must stay
        # inconclusive, or confirm_deleted would tombstone pages that still exist.
        found_gone = response.status_code == 404 and code == NOT_FOUND_CODE
        cls = NotionNotFound if found_gone else NotionApiError
        return cls(response.status_code, code, message)

    def _scrub(self, text: str) -> str:
        """Defense in depth: Notion never echoes the token, but a proxy might."""
        token = self._token.get_secret_value()
        return text.replace(token, _REDACTED) if token else text


def _json_body(response: httpx.Response) -> Json:
    try:
        body = response.json()
    except ValueError:
        raise NotionProtocolError("Notion returned a non-JSON success body") from None
    if not isinstance(body, dict):
        raise NotionProtocolError("Notion returned a non-object JSON body")
    return body


def _parse_http_date(response: httpx.Response) -> datetime | None:
    raw = response.headers.get("date")
    if not raw:
        return None
    try:
        parsed = email.utils.parsedate_to_datetime(raw)
    except (TypeError, ValueError):
        return None
    return parsed.astimezone(UTC) if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def _parse_retry_after(response: httpx.Response) -> float | None:
    """Retry-After is either delta-seconds or an HTTP-date (RFC 9110)."""
    raw = response.headers.get("retry-after")
    if not raw:
        return None
    try:
        seconds = float(raw)
    except ValueError:
        try:
            when = email.utils.parsedate_to_datetime(raw)
        except (TypeError, ValueError):
            return None
        now = _parse_http_date(response) or datetime.now(UTC)
        if when.tzinfo is None:
            when = when.replace(tzinfo=UTC)
        seconds = (when - now).total_seconds()
    if math.isnan(seconds) or math.isinf(seconds):
        return None
    return max(0.0, seconds)
