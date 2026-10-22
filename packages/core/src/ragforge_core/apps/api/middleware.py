"""Pure ASGI middleware (no BaseHTTPMiddleware: it buffers/wraps streaming responses and
mishandles client disconnects, which matters for SSE)."""

from __future__ import annotations

import asyncio
import json
import re
import time
import uuid
from dataclasses import dataclass
from typing import Final

import structlog
from starlette.datastructures import MutableHeaders
from starlette.requests import ClientDisconnect
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from ragforge_contracts.api import Problem
from ragforge_core.observability import metrics
from ragforge_core.observability.prompt_context import surface
from ragforge_core.observability.surface import surface_of_path

# scope["state"] key: when the outermost middleware saw the request (chat TTFT starts here).
STARTED_AT: Final = "started_at"
UNMEASURED_PATHS: Final = frozenset({"/healthz", "/readyz"})  # probes would drown real traffic


@dataclass(slots=True)
class _Exchange:
    """What the metrics middleware observed of one request/response."""

    status: int | None = None
    streaming: bool = False  # text/event-stream: excluded from the duration histogram
    completed: bool = False  # the final body message was sent
    disconnected: bool = False  # the client went away before the response completed
    failed: bool = False  # the app raised (anything but a disconnect)


class MetricsMiddleware:
    """`ragforge_http_requests_total{route,status_class}` and, for non-streaming responses,
    `ragforge_http_request_duration_seconds{route}` (ADR-0018 R2-5). Added LAST in create_app,
    so it is the outermost app middleware: it times the whole request, sees the 413s that
    BodyLimitMiddleware answers before routing, and sees the exception behind every 500.

    - `route` is the matched route TEMPLATE, read from the scope after the app ran; a request
      that matched no route (404, or refused before routing) is `unmatched`: never the raw
      path, which would be a label per URL.
    - `status_class` is "disconnected" when the client went away before the response
      completed, so client aborts never read as server errors (or as successes).
    - The probes (/healthz, /readyz) are not measured.
    It also sets the `surface` ContextVar for the request from the route (interview, eval or
    chat); the chat stream refines it from the caller's tenant (eval, demo)."""

    def __init__(self, app: ASGIApp, unmeasured: frozenset[str] = UNMEASURED_PATHS) -> None:
        self.app = app
        self.unmeasured = unmeasured

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        path = scope.get("path", "")
        if scope["type"] != "http" or path in self.unmeasured:
            await self.app(scope, receive, send)
            return
        started = time.perf_counter()
        scope.setdefault("state", {})[STARTED_AT] = started
        seen = _Exchange()

        async def watched_receive() -> Message:
            message = await receive()
            if message["type"] == "http.disconnect":
                seen.disconnected = True
            return message

        async def watched_send(message: Message) -> None:
            if message["type"] == "http.response.start":
                seen.status = int(message["status"])
                seen.streaming = _is_event_stream(message)
            elif message["type"] == "http.response.body" and not message.get("more_body", False):
                seen.completed = True
            await send(message)

        token = surface.set(surface_of_path(path))
        try:
            await self.app(scope, watched_receive, watched_send)
        except BaseException as exc:
            if isinstance(exc, ClientDisconnect | OSError | asyncio.CancelledError):
                seen.disconnected = True
            else:
                seen.failed = True
            raise
        finally:
            surface.reset(token)
            _record(scope, seen, time.perf_counter() - started)


def _is_event_stream(message: Message) -> bool:
    for name, value in message.get("headers", []):
        if name.lower() == b"content-type":
            return bytes(value).lower().startswith(b"text/event-stream")
    return False


def _route(scope: Scope) -> str:
    template = getattr(scope.get("route"), "path", None)
    return metrics.route_label(template) if isinstance(template, str) else metrics.UNMATCHED_ROUTE


def _status_class(seen: _Exchange) -> str:
    if seen.completed and seen.status is not None:
        return metrics.status_class(seen.status)
    if seen.disconnected:
        return "disconnected"
    if seen.failed or seen.status is None:
        return "5xx"  # an exception, or an app that never answered: the server's fault
    return metrics.status_class(seen.status)


def _record(scope: Scope, seen: _Exchange, seconds: float) -> None:
    route = _route(scope)
    metrics.HTTP_REQUESTS.labels(route, _status_class(seen)).inc()
    if not seen.streaming:
        metrics.HTTP_DURATION.labels(route).observe(seconds)


_REQUEST_ID_RE = re.compile(r"^[A-Za-z0-9._-]{1,64}$")


class RequestIdMiddleware:
    """Accept a safe incoming X-Request-ID or mint one; expose it on scope state, in logs, and on
    every response, including error responses built by the exception handlers."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        incoming = ""
        for name, value in scope.get("headers", []):
            if name == b"x-request-id":
                incoming = value.decode("latin-1")
                break
        rid = incoming if _REQUEST_ID_RE.match(incoming) else uuid.uuid4().hex
        scope.setdefault("state", {})["request_id"] = rid
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(request_id=rid)

        async def send_with_id(message: Message) -> None:
            if message["type"] == "http.response.start":
                MutableHeaders(scope=message)["x-request-id"] = rid
            await send(message)

        await self.app(scope, receive, send_with_id)


MAX_BODY_BYTES: Final = 256 * 1024
"""Every JSON request body (chat, query, approvals, connectors) is tiny; uploads have their own
route-level cap and are exempt."""

_UPLOAD_PATH_RE = re.compile(r"^/v1/collections/[^/]+/files$")


class BodyLimitMiddleware:
    """Cap request bodies BEFORE the app runs (DoS guard).

    FastAPI reads and parses a declared JSON body before it resolves dependencies, so without
    this an unauthenticated client could make the server buffer an arbitrarily large body. A
    declared Content-Length over the cap is refused without reading a byte; a streamed or
    chunked body is counted and cut off at the cap. In both cases the client gets a 413
    problem+json (unless the app had already started its response, which then simply ends).
    """

    def __init__(
        self,
        app: ASGIApp,
        max_bytes: int = MAX_BODY_BYTES,
        exempt: re.Pattern[str] = _UPLOAD_PATH_RE,
    ) -> None:
        self.app = app
        self.max_bytes = max_bytes
        self.exempt = exempt

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or self.exempt.match(scope.get("path", "")):
            await self.app(scope, receive, send)
            return
        declared = _content_length(scope)
        if declared is not None and not declared.isdigit():
            await _refuse(scope, send, 400, "Bad request", "invalid Content-Length header")
            return
        if declared is not None and int(declared) > self.max_bytes:
            await _too_large(scope, send, self.max_bytes)
            return

        received = 0
        exceeded = False
        started = False

        async def limited_receive() -> Message:
            nonlocal received, exceeded
            if exceeded:
                return {"type": "http.disconnect"}
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_bytes:
                    exceeded = True  # the app sees a disconnect and stops reading
                    return {"type": "http.disconnect"}
            return message

        async def guarded_send(message: Message) -> None:
            nonlocal started
            if exceeded and not started:
                return  # drop the app's reaction to the disconnect; we answer 413 below
            if message["type"] == "http.response.start":
                started = True
            await send(message)

        try:
            await self.app(scope, limited_receive, guarded_send)
        except Exception:
            if not exceeded:
                raise
        if exceeded and not started:
            await _too_large(scope, send, self.max_bytes)


def _content_length(scope: Scope) -> str | None:
    for name, value in scope.get("headers", []):
        if name == b"content-length":
            return str(value.decode("latin-1")).strip()
    return None


async def _too_large(scope: Scope, send: Send, max_bytes: int) -> None:
    await _refuse(scope, send, 413, "Payload too large", f"request body exceeds {max_bytes} bytes")


async def _refuse(scope: Scope, send: Send, status: int, title: str, detail: str) -> None:
    rid = scope.get("state", {}).get("request_id")
    problem = Problem(
        title=title, status=status, detail=detail, instance=scope.get("path"), request_id=rid
    )
    body = json.dumps(problem.model_dump(mode="json", exclude_none=True)).encode()
    headers = [
        (b"content-type", b"application/problem+json"),
        (b"content-length", str(len(body)).encode()),
        (b"connection", b"close"),
    ]
    if rid:
        headers.append((b"x-request-id", str(rid).encode()))
    await send({"type": "http.response.start", "status": status, "headers": headers})
    await send({"type": "http.response.body", "body": body})
