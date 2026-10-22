"""Domain errors. The API maps each to an RFC 9457 problem (apps/api/errors.py)."""

from __future__ import annotations


class RagforgeError(Exception):
    """Base class. `status` and `title` drive the problem+json mapping."""

    status: int = 500
    title: str = "Internal error"


class NotFound(RagforgeError):
    status = 404
    title = "Not found"


class Conflict(RagforgeError):
    status = 409
    title = "Conflict"


class TurnInProgress(Conflict):
    title = "A turn is already running for this conversation"


class Unauthorized(RagforgeError):
    status = 401
    title = "Unauthorized"


class Forbidden(RagforgeError):
    status = 403
    title = "Forbidden"


class ValidationFailed(RagforgeError):
    status = 422
    title = "Validation failed"


class LeaseLost(RagforgeError):
    """A job or run lease was lost; the holder must discard its work.

    See ADR-0010 and DESIGN section 8.
    """

    status = 409
    title = "Lease lost"


class UpstreamRateLimited(RagforgeError):
    status = 503
    title = "Upstream rate limited"

    def __init__(self, retry_after_s: float | None = None) -> None:
        super().__init__(f"retry after {retry_after_s}s" if retry_after_s else "rate limited")
        self.retry_after_s = retry_after_s
