"""NotionConnectorFactory: the sync service's ConnectorFactory seam (C2 request 3).

Proves: the signature matches what C2 calls, the session yields a working SourceConnector and
closes its HTTP client, a wrong kind or undecodable token fails permanently before any request,
and the token never reaches the limiter or an error message."""

from __future__ import annotations

import inspect
from collections.abc import Iterator
from contextlib import AbstractAsyncContextManager
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import httpx
import pytest
import respx

from module_c1.c1_support import (
    BASE,
    HANDBOOK_ID,
    ROW_ID,
    TOKEN,
    FakeLimiter,
    load,
    serve_search,
)
from ragforge_core.adapters.connectors.notion import (
    NotionApiError,
    NotionConnector,
    NotionConnectorFactory,
)
from ragforge_core.adapters.connectors.ratelimit import limiter_key
from ragforge_core.domain.errors import ValidationFailed
from ragforge_core.domain.ports import SourceConnector

SINCE = datetime(2026, 10, 16, 11, 40, tzinfo=UTC)


@pytest.fixture
def router() -> Iterator[respx.MockRouter]:
    with respx.mock(base_url=BASE, assert_all_mocked=True) as mock:
        yield mock


def _open(
    factory: NotionConnectorFactory, *, kind: str = "notion", token: bytes = TOKEN.encode()
) -> AbstractAsyncContextManager[SourceConnector]:
    return factory.open(connector_id=uuid4(), kind=kind, token=token, config={})


def test_signature_matches_the_sync_services_connector_factory_port() -> None:
    """Adapters may not import services/sync/ports.py (import-linter), so pin the shape here:
    keyword-only (connector_id, kind, token, config)."""
    params = inspect.signature(NotionConnectorFactory.open).parameters
    assert list(params) == ["self", "connector_id", "kind", "token", "config"]
    assert all(
        p.kind is inspect.Parameter.KEYWORD_ONLY for name, p in params.items() if name != "self"
    )


async def test_session_yields_a_working_connector_and_closes_its_http_client(
    router: respx.MockRouter,
) -> None:
    serve_search(router, load("search_changed.json"))
    limiter = FakeLimiter()
    async with _open(NotionConnectorFactory(limiter)) as source:
        assert isinstance(source, SourceConnector)
        assert isinstance(source, NotionConnector)
        refs = [r.external_id async for r in source.list_changed(SINCE)]
        http = source._client._http
    assert refs[:2] == [HANDBOOK_ID, ROW_ID]
    assert http.is_closed
    assert limiter.calls
    assert all(key == limiter_key(TOKEN) for key, _ in limiter.calls)


async def test_session_closes_the_client_when_the_run_fails(router: respx.MockRouter) -> None:
    router.post("/search").mock(return_value=httpx.Response(400, json={"code": "validation_error"}))
    opened: list[httpx.AsyncClient] = []

    async def run() -> None:
        async with _open(NotionConnectorFactory(FakeLimiter())) as source:
            assert isinstance(source, NotionConnector)
            opened.append(source._client._http)
            [r async for r in source.list_all()]

    with pytest.raises(NotionApiError, match="400"):
        await run()
    assert opened[0].is_closed


async def test_factory_overlap_is_passed_to_the_connector(router: respx.MockRouter) -> None:
    serve_search(router, load("search_changed.json"))
    factory = NotionConnectorFactory(FakeLimiter(), list_overlap=timedelta(0))
    async with _open(factory) as source:
        refs = [r.external_id async for r in source.list_changed(SINCE)]
    assert refs == [HANDBOOK_ID, ROW_ID]


@pytest.mark.parametrize(
    ("kind", "token", "message"),
    [
        ("file", TOKEN.encode(), "cannot open"),
        ("notion", b"\xff\xfe" + TOKEN.encode(), "UTF-8"),
        ("notion", b"   ", "empty"),
    ],
)
async def test_bad_kind_or_token_fails_permanently_without_leaking(
    router: respx.MockRouter, kind: str, token: bytes, message: str
) -> None:
    with pytest.raises(ValidationFailed, match=message) as info:
        _open(NotionConnectorFactory(FakeLimiter()), kind=kind, token=token)
    assert not router.calls
    assert TOKEN not in str(info.value)
    assert TOKEN not in repr(info.value)
    # Neither chain link may lead to the UnicodeDecodeError, whose `.object` is the token.
    assert info.value.__cause__ is None
    assert info.value.__context__ is None
    assert TOKEN not in repr(NotionConnectorFactory(FakeLimiter()))
