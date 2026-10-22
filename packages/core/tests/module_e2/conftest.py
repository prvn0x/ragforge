"""Module E2 fixtures: the real API app on the real test database, with other modules' ports
replaced by fakes through FastAPI's dependency_overrides."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Sequence
from pathlib import Path
from uuid import UUID

import httpx
import psycopg
import pytest
from fastapi import FastAPI

from module_e2.e2_support import (
    FakeEmbedders,
    FakeJobQueue,
    FakeReranker,
    FakeRetriever,
    Fakes,
    SanitizerFactory,
    ScriptedAgent,
    TenantWorld,
    TrackingTxManager,
    build_world,
)
from ragforge_core.adapters.blob.local import LocalBlobStore
from ragforge_core.apps.api import deps
from ragforge_core.apps.api.app import create_app
from ragforge_core.bootstrap import build_api
from ragforge_core.config import Settings
from ragforge_core.domain.models import RankedChunk

MakeTenant = Callable[..., tuple[UUID, UUID, UUID]]
Conn = psycopg.Connection[tuple[object, ...]]


class _UnusedReranker:
    async def rerank(self, query: str, candidates: Sequence[RankedChunk]) -> list[RankedChunk]:
        raise AssertionError("routes must use the overridden reranker")


@pytest.fixture
async def e2_app(settings: Settings, tmp_path: Path) -> AsyncIterator[tuple[FastAPI, Fakes]]:
    # The container's own reranker is never reached (routes use fakes.reranker), but build_api
    # refuses to start without a loadable model, which the test host does not have.
    container = await build_api(settings, reranker=_UnusedReranker())  # closed by the lifespan
    txm = TrackingTxManager(container.tx)
    fakes = Fakes(
        txm=txm,
        jobs=FakeJobQueue(),
        retriever=FakeRetriever(),
        embedders=FakeEmbedders(txm),
        reranker=FakeReranker(txm),
        agent=ScriptedAgent(),
        sanitizers=SanitizerFactory(),
        blobs=LocalBlobStore(tmp_path / "blobs"),
    )
    app = create_app(settings, container)
    app.dependency_overrides.update(
        {
            deps.get_tx_manager: lambda: fakes.txm,
            deps.get_job_queue: lambda: fakes.jobs,
            deps.get_retriever: lambda: fakes.retriever,
            deps.get_embedders: lambda: fakes.embedders,
            deps.get_reranker: lambda: fakes.reranker,
            deps.get_blob_store: lambda: fakes.blobs,
            deps.get_agent_runner: lambda: fakes.agent,
            deps.get_sanitizer_factory: lambda: fakes.sanitizers,
            deps.get_job_stream_config: lambda: deps.JobStreamConfig(
                poll_interval_s=0.05, keepalive_s=0.2
            ),
        }
    )
    async with app.router.lifespan_context(app):
        yield app, fakes


@pytest.fixture
def app(e2_app: tuple[FastAPI, Fakes]) -> FastAPI:
    return e2_app[0]


@pytest.fixture
def fakes(e2_app: tuple[FastAPI, Fakes]) -> Fakes:
    return e2_app[1]


@pytest.fixture
async def http(app: FastAPI) -> AsyncIterator[httpx.AsyncClient]:
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        yield client


@pytest.fixture
def world(superuser: Conn, make_tenant: MakeTenant) -> Callable[[], TenantWorld]:
    def _make() -> TenantWorld:
        return build_world(superuser, *make_tenant())

    return _make
