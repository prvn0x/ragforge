"""Turn every failure the UI can meet into a short, friendly, SAFE message.

Two sources of failure reach the page: an exception from the SDK call (the API is down, a 401
before the stream starts, a dropped connection) and an `error` event inside a stream (the turn
failed after it started). Both end up as a `UiError`, so the page has one way to show a failure
and the conversation above it is never lost.

Safety: the message is built from the HTTP status and the server's RFC 9457 Problem fields only.
An unknown exception is reported by its TYPE name, never `str(exc)`: an exception message is the
one place a secret (a header, a URL with credentials) could leak onto the screen.
"""

from __future__ import annotations

from dataclasses import dataclass

import httpx

from ragforge_client.ask import clean
from ragforge_client.client import ApiError
from ragforge_contracts.api import Problem


@dataclass(frozen=True, slots=True)
class UiError:
    """What the page shows for a failure. `detail` may hold server text: render it escaped."""

    title: str
    detail: str


_BY_STATUS: dict[int, tuple[str, str]] = {
    401: (
        "The API rejected this key",
        "Check the key file (--key-file, RAGFORGE_KEY_FILE or ~/.config/ragforge/<name>.key). "
        "A revoked or mistyped key gives this.",
    ),
    403: ("This key is not allowed to do that", "Use a key with the needed scope."),
    404: ("Not found", "It may have been deleted. Refresh the page."),
    409: (
        "The previous answer is still finishing",
        "Wait a moment, then send it again.",
    ),
    429: (
        "Too many requests right now",
        "The API is rate limiting this key. Wait a few seconds, then send it again.",
    ),
    503: ("The API is not ready", "It may be starting up or migrating. Try again shortly."),
}


def from_problem(problem: Problem) -> UiError:
    """Friendly text for a Problem (HTTP error or a stream `error` event)."""
    title, hint = _BY_STATUS.get(problem.status, (clean(problem.title), ""))
    parts = [clean(problem.detail)] if problem.detail else []
    if hint:
        parts.append(hint)
    if problem.request_id:
        parts.append(f"(request id {clean(problem.request_id)})")
    return UiError(title=title, detail=" ".join(parts))


class TurnTimeout(Exception):
    """A whole turn took longer than its deadline; the stream was closed (the server cancels)."""

    def __init__(self, seconds: float) -> None:
        super().__init__(f"the answer took longer than {seconds:.0f} s")


def describe_failure(exc: BaseException, api_url: str) -> UiError:
    """Friendly text for an exception raised while calling the API.

    `api_url` must already be display-safe (see `config.display_url`): it is shown verbatim.
    """
    if isinstance(exc, ApiError):
        return from_problem(exc.problem)
    if isinstance(exc, TurnTimeout):
        return UiError(
            "The answer took too long",
            "It was stopped after 3 minutes and nothing new was saved. Send it again.",
        )
    if isinstance(exc, httpx.ConnectError | httpx.ConnectTimeout):
        return UiError(
            "Cannot reach the RAGForge API",
            f"Nothing answered at {api_url}. Is the stack running (make up)?",
        )
    if isinstance(exc, httpx.TimeoutException):
        return UiError("The API took too long to answer", "Send it again in a moment.")
    if isinstance(exc, httpx.TransportError):
        return UiError(
            "The connection dropped",
            "The answer stopped part way. Anything shown above is incomplete; send it again.",
        )
    if isinstance(exc, ValueError):  # includes pydantic's ValidationError and JSON errors
        return UiError(
            "The API sent something this UI could not read",
            "The server may be newer than this UI. Update the UI, then try again.",
        )
    return UiError("Something went wrong", f"Unexpected {type(exc).__name__}. Try again.")
