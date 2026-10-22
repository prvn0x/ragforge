"""Notion adapter errors. None of them ever carries the integration token or a request object
(whose headers hold it): they are built from status, Notion error code and a scrubbed message."""

from __future__ import annotations

from ragforge_core.domain.errors import RagforgeError


class NotionError(RagforgeError):
    """Base class for every failure talking to Notion."""

    status = 502
    title = "Notion request failed"


class NotionTransportError(NotionError):
    """Network-level failure (timeout, connection reset) after all retries."""


class NotionProtocolError(NotionError):
    """Notion answered with something we cannot trust (bad JSON, missing Date header, cursor
    loop). Retrying later is the right response; guessing is not."""


class NotionApiError(NotionError):
    """Notion returned a non-success HTTP status that is not retried (or retries ran out)."""

    def __init__(self, http_status: int, code: str, message: str) -> None:
        super().__init__(f"Notion API {http_status} {code}: {message}")
        self.http_status = http_status
        self.code = code
        self.message = message


class NotionNotFound(NotionApiError):
    """404 with Notion's `object_not_found` code: deleted, or not shared with the integration
    (Notion does not distinguish the two). Any other 404 is a plain `NotionApiError`."""
