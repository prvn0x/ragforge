"""RFC 9457 problem+json for every error path. Internal details never leak to clients.

Headers are preserved (Allow on 405, WWW-Authenticate on 401, Retry-After on rate limits), and
X-Request-ID is set here too because the catch-all 500 handler runs outside the middleware stack.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from ragforge_contracts.api import Problem
from ragforge_core.domain.errors import RagforgeError, Unauthorized, UpstreamRateLimited
from ragforge_core.observability.logging import get_logger

PROBLEM_JSON = "application/problem+json"
log = get_logger(__name__)


# HTTP-only conditions of the upload route (no domain meaning, so they live here, not in domain).
class LengthRequired(RagforgeError):
    status = 411
    title = "Length required"


class RequestTimeout(RagforgeError):
    """The client did not send its declared body within the upload read deadline."""

    status = 408
    title = "Request timeout"


class PayloadTooLarge(RagforgeError):
    status = 413
    title = "Payload too large"


class UnsupportedMediaType(RagforgeError):
    status = 415
    title = "Unsupported media type"


class ClientDisconnected(RagforgeError):
    """The client went away mid-body. Nobody reads the response; it exists so the disconnect is
    an ordinary 4xx problem instead of an unhandled error with a traceback in the logs."""

    status = 400
    title = "Client disconnected"


class ServiceBusy(RagforgeError):
    """A bounded local resource (upload slots, the multipart spool) is full: retry shortly."""

    status = 503
    title = "Service busy"
    retry_after_s = 5


def _problem(
    request: Request,
    status: int,
    title: str,
    detail: str | None = None,
    headers: Mapping[str, str] | None = None,
) -> JSONResponse:
    rid = request.scope.get("state", {}).get("request_id")
    body = Problem(
        title=title, status=status, detail=detail, instance=request.url.path, request_id=rid
    )
    out = dict(headers or {})
    if rid:
        out["x-request-id"] = rid
    return JSONResponse(
        body.model_dump(exclude_none=True), status_code=status, media_type=PROBLEM_JSON, headers=out
    )


def _domain_headers(exc: RagforgeError) -> dict[str, str]:
    if isinstance(exc, UpstreamRateLimited) and exc.retry_after_s is not None:
        return {"retry-after": str(max(1, math.ceil(exc.retry_after_s)))}
    if isinstance(exc, ServiceBusy):
        return {"retry-after": str(exc.retry_after_s)}
    if isinstance(exc, Unauthorized):
        return {"www-authenticate": 'Bearer realm="ragforge"'}
    return {}


def install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(RagforgeError)
    async def _domain(request: Request, exc: RagforgeError) -> JSONResponse:
        return _problem(request, exc.status, exc.title, str(exc) or None, _domain_headers(exc))

    @app.exception_handler(RequestValidationError)
    async def _validation(request: Request, exc: RequestValidationError) -> JSONResponse:
        fields = ", ".join(".".join(str(p) for p in e["loc"]) for e in exc.errors())
        return _problem(request, 422, "Validation failed", f"invalid fields: {fields}")

    @app.exception_handler(StarletteHTTPException)
    async def _http(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        return _problem(request, exc.status_code, str(exc.detail), headers=exc.headers)

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
        log.exception("unhandled_error", path=request.url.path)
        return _problem(request, 500, "Internal error")
