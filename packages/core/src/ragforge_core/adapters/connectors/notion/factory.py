"""NotionConnectorFactory: opens a live Notion SourceConnector from a decrypted connector token.

It matches the sync service's module-local `ConnectorFactory` port structurally
(`open(*, connector_id, kind, token, config)` returning an async context manager). Adapters may
not import services (import-linter), so the Protocol is not imported here; a test pins the
signature instead.

Why a context manager: the connector owns an httpx client (connection pool). One client per
connector run, closed when the run ends, even when it fails.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from datetime import timedelta
from typing import Any
from uuid import UUID

from pydantic import SecretStr

from ragforge_core.adapters.connectors.notion.client import NOTION_API_BASE, NotionClient
from ragforge_core.adapters.connectors.notion.connector import (
    LIST_OVERLAP,
    MAX_BLOCK_DEPTH,
    NotionConnector,
)
from ragforge_core.domain.errors import ValidationFailed
from ragforge_core.domain.models import SourceKind
from ragforge_core.domain.ports import RateLimiter, SourceConnector


class NotionConnectorFactory:
    """Builds `NotionConnector`s sharing one (global, DB-backed) `RateLimiter`.

    `list_overlap` defaults to the connector's 5 minutes. Pass `timedelta(0)` if the caller
    already subtracts its own overlap from the cursor, so the overlap is applied exactly once.
    """

    def __init__(
        self,
        limiter: RateLimiter,
        *,
        base_url: str = NOTION_API_BASE,
        max_depth: int = MAX_BLOCK_DEPTH,
        list_overlap: timedelta = LIST_OVERLAP,
    ) -> None:
        self._limiter = limiter
        self._base_url = base_url
        self._max_depth = max_depth
        self._list_overlap = list_overlap

    def __repr__(self) -> str:
        return f"NotionConnectorFactory(base_url={self._base_url!r})"

    def open(
        self, *, connector_id: UUID, kind: str, token: bytes, config: Mapping[str, Any]
    ) -> AbstractAsyncContextManager[SourceConnector]:
        """Validate eagerly (a wrong kind or an undecodable token is permanent, and should fail
        before any HTTP client exists), then hand back the session context manager.

        `connector_id` and `config` are accepted for the port's sake; the Notion connector needs
        neither (the token identifies the integration and its workspace)."""
        if kind != SourceKind.notion.value:
            raise ValidationFailed(f"NotionConnectorFactory cannot open a {kind!r} connector")
        secret = _decode_token(token)
        if secret is None:
            raise ValidationFailed("connector token is not valid UTF-8")
        if not secret.get_secret_value().strip():
            raise ValidationFailed("connector token is empty")
        return self._session(secret)

    @asynccontextmanager
    async def _session(self, token: SecretStr) -> AsyncIterator[SourceConnector]:
        async with NotionClient(token, self._limiter, base_url=self._base_url) as client:
            yield NotionConnector(
                client, max_depth=self._max_depth, list_overlap=self._list_overlap
            )


def _decode_token(token: bytes) -> SecretStr | None:
    """None when undecodable. The error is NOT raised from inside the `except`: a
    UnicodeDecodeError's `.object` is the token itself, and raising there (even `from None`)
    would keep it reachable through `__context__`."""
    try:
        return SecretStr(token.decode("utf-8"))
    except UnicodeDecodeError:
        return None
