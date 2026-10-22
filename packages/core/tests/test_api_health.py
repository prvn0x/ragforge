"""API skeleton: health, readiness, request ids, problem+json, and the SDK against it."""

from __future__ import annotations

from collections.abc import AsyncIterator

import httpx
import pytest
from fastapi import FastAPI

from pgtest import IdentityReranker
from ragforge_client import RagforgeClient
from ragforge_core import CODE_REQUIRED_SCHEMA
from ragforge_core.apps.api.app import create_app
from ragforge_core.bootstrap import build_api
from ragforge_core.config import Settings

pytestmark = pytest.mark.integration


@pytest.fixture
async def app(settings: Settings) -> AsyncIterator[FastAPI]:
    container = await build_api(settings, reranker=IdentityReranker())
    application = create_app(settings, container)
    async with application.router.lifespan_context(application):
        yield application


@pytest.fixture
async def http(app: FastAPI) -> AsyncIterator[httpx.AsyncClient]:
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        yield client


async def test_healthz(http: httpx.AsyncClient) -> None:
    r = await http.get("/healthz")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


async def test_readyz_ready_at_head(http: httpx.AsyncClient) -> None:
    r = await http.get("/readyz")
    assert r.status_code == 200
    body = r.json()
    assert body["ready"] is True
    assert body["schema_version"] == CODE_REQUIRED_SCHEMA


async def test_request_id_echoed_and_sanitized(http: httpx.AsyncClient) -> None:
    r = await http.get("/healthz", headers={"x-request-id": "abc-123"})
    assert r.headers["x-request-id"] == "abc-123"
    r = await http.get("/healthz", headers={"x-request-id": "bad id\nwith newline"})
    assert r.headers["x-request-id"] != "bad id\nwith newline"
    assert len(r.headers["x-request-id"]) == 32


async def test_unknown_route_is_problem_json(http: httpx.AsyncClient) -> None:
    r = await http.get("/nope", headers={"x-request-id": "rid-1"})
    assert r.status_code == 404
    assert r.headers["content-type"].startswith("application/problem+json")
    assert r.json()["request_id"] == "rid-1"


async def test_sdk_readiness(app: FastAPI) -> None:
    async with RagforgeClient("http://test", transport=httpx.ASGITransport(app=app)) as client:
        readiness = await client.readiness()
    assert readiness.ready is True
