"""Map SDK/transport failures to `ToolError`s an AI agent can act on, without leaking the key.

A `ToolError` reaches the agent as an `is_error` tool result with exactly our message, and the
SDK logs that message. So every message built here passes through `Redactor` first: the API never
echoes the key today, but a server bug or a proxy that reflects the Authorization header must not
turn into a leak through this process (defence in depth, cheap).
"""

from __future__ import annotations

import json

import httpx
from mcp.server.mcpserver.exceptions import ToolError
from pydantic import BaseModel

from ragforge_client.ask import clean
from ragforge_client.client import ApiError

REDACTED = "[redacted]"


class Redactor:
    """Replaces every occurrence of the secret, also in its JSON-escaped spelling (a message may
    quote an API response body verbatim). `repr` never shows it."""

    def __init__(self, secret: str) -> None:
        spellings = {secret, json.dumps(secret)[1:-1]} if secret else set()
        self._spellings = sorted(spellings, key=len, reverse=True)

    def __call__(self, text: str) -> str:
        for spelling in self._spellings:
            text = text.replace(spelling, REDACTED)
        return text

    def __repr__(self) -> str:
        return "Redactor(<secret>)"


def redact_model[M: BaseModel](model: M, redact: Redactor) -> M:
    """`model` with the secret removed from every string in it. Applied to every tool output, so
    "the key never reaches the agent" does not depend on each field being redacted where it is
    built (review r1: a post-verdict error warning was missed that way).

    Redacts string VALUES, never the serialized JSON (review r2): replacing inside the JSON text
    could also hit field names or syntax and leave JSON that no longer validates."""
    return type(model).model_validate(_redact_values(model.model_dump(mode="json"), redact))


def _redact_values(value: object, redact: Redactor) -> object:
    if isinstance(value, str):
        return redact(value)
    if isinstance(value, dict):
        return {key: _redact_values(item, redact) for key, item in value.items()}
    if isinstance(value, list):
        return [_redact_values(item, redact) for item in value]
    return value


def display_url(url: str) -> str:
    """The API URL without user:password, query or fragment, for error messages (any of them
    may carry credentials: `?token=`)."""
    try:
        return str(httpx.URL(url).copy_with(userinfo=b"", query=None, fragment=None))
    except httpx.InvalidURL:
        return "<invalid url>"


def _retry_hint(exc: ApiError) -> str:
    # The SDK's ApiError does not carry the Retry-After header yet (handoff MCP.md, SDK gap 1).
    # Read it when a newer SDK exposes it, and say "shortly" until then.
    seconds = getattr(exc, "retry_after", None)
    return f"retry after {seconds} s" if isinstance(seconds, int | float) else "retry shortly"


def api_error_message(exc: ApiError) -> str:
    """One sentence per status, written for the agent: what happened and what to do next."""
    problem = exc.problem
    detail = clean(problem.detail or problem.title)
    status = problem.status
    if status == 401:
        return (
            "RAGForge rejected the API key (401): it is missing, wrong or revoked. The user must "
            "fix the key file (or RAGFORGE_API_KEY) and restart this MCP server."
        )
    if status == 403:
        return (
            f"The API key lacks the scope for this (403): {detail}. MCP keys never hold "
            "approvals:write; sync_now needs the `write` scope."
        )
    if status == 404:
        return f"Not found (404): {detail}"
    if status == 409:
        return f"Conflict (409): {detail}. Wait for the running turn or job to finish, then retry."
    if status == 422:
        return f"Invalid request (422): {detail}"
    if status == 429:
        return f"RAGForge is rate limiting this key (429): {_retry_hint(exc)}."
    if status == 503:
        return f"RAGForge is busy or not ready (503): {detail}; {_retry_hint(exc)}."
    rid = f" (request id {clean(problem.request_id)})" if problem.request_id else ""
    return f"RAGForge API error {status}: {clean(problem.title)}{rid}"


def transport_error_message(exc: httpx.HTTPError, api_url: str) -> str:
    where = display_url(api_url)
    if isinstance(exc, httpx.TimeoutException):
        return f"The RAGForge API at {where} timed out; it may be overloaded. Retry shortly."
    if isinstance(exc, httpx.ConnectError):
        return (
            f"The RAGForge API is unreachable at {where}. Is the stack running (`make up`)? "
            "Check with: curl localhost:8000/readyz"
        )
    return f"The connection to the RAGForge API at {where} failed ({type(exc).__name__})."


def to_tool_error(exc: Exception, api_url: str, redact: Redactor) -> ToolError:
    """The single exit for expected failures. Unknown exceptions are NOT converted: the SDK then
    shows the agent only "Error executing tool <name>" and logs the traceback, which is the right
    behaviour for a bug (no internal text reaches the agent)."""
    if isinstance(exc, ApiError):
        message = api_error_message(exc)
    elif isinstance(exc, httpx.HTTPError):
        message = transport_error_message(exc, api_url)
    else:
        raise TypeError(f"not an expected API failure: {type(exc).__name__}") from exc
    return ToolError(redact(message))


EXPECTED = (ApiError, httpx.HTTPError)
