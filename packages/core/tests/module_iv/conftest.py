"""Module IV fixtures: the real API app on the real test database, with the model ports replaced
by fakes through FastAPI's dependency_overrides (the module E2 pattern, reusing its fakes)."""

from __future__ import annotations

import random
from collections.abc import AsyncIterator, Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

import httpx
import pytest
from fastapi import FastAPI

from module_e2.e2_support import (
    FakeEmbedders,
    FakeJobQueue,
    FakeReranker,
    FakeRetriever,
    SanitizerFactory,
    ScriptedAgent,
    TrackingTxManager,
    issue_key,
)
from module_iv.iv_support import Conn, FakeChat, connector
from ragforge_core.adapters.blob.local import LocalBlobStore
from ragforge_core.apps.api import deps
from ragforge_core.apps.api.app import create_app
from ragforge_core.apps.api.routes import interview
from ragforge_core.bootstrap import build_api
from ragforge_core.config import Settings
from ragforge_core.domain.models import RankedChunk

MakeTenant = Callable[..., tuple[UUID, UUID, UUID]]


class _UnusedReranker:
    async def rerank(self, query: str, candidates: Sequence[RankedChunk]) -> list[RankedChunk]:
        raise AssertionError("routes must use the overridden reranker")


@dataclass
class IvFakes:
    txm: TrackingTxManager
    retriever: FakeRetriever
    embedders: FakeEmbedders
    reranker: FakeReranker
    chat: FakeChat
    sanitizers: SanitizerFactory


@pytest.fixture
async def iv_app(settings: Settings, tmp_path: Path) -> AsyncIterator[tuple[FastAPI, IvFakes]]:
    container = await build_api(settings, reranker=_UnusedReranker())  # closed by the lifespan
    txm = TrackingTxManager(container.tx)
    fakes = IvFakes(
        txm=txm,
        retriever=FakeRetriever(hits=[], spaces=(1,)),
        embedders=FakeEmbedders(txm),
        reranker=FakeReranker(txm),
        chat=FakeChat(open_tx=lambda: txm.open),
        sanitizers=SanitizerFactory(),
    )
    app = create_app(settings, container)
    limiter = deps.PrincipalLimiter(rate_per_s=100.0, burst=100, concurrency=4)
    app.dependency_overrides.update(
        {
            deps.get_tx_manager: lambda: fakes.txm,
            deps.get_job_queue: FakeJobQueue,
            deps.get_retriever: lambda: fakes.retriever,
            deps.get_embedders: lambda: fakes.embedders,
            deps.get_reranker: lambda: fakes.reranker,
            deps.get_blob_store: lambda: LocalBlobStore(tmp_path / "blobs"),
            deps.get_agent_runner: ScriptedAgent,
            deps.get_sanitizer_factory: lambda: fakes.sanitizers,
            deps.get_chat_model: lambda: fakes.chat,
            interview.get_interview_rng: lambda: random.Random(7),  # noqa: S311
            # Generous, so sessions are not throttled; the quota test removes this override.
            deps.get_interview_limiter: lambda: limiter,
        }
    )
    async with app.router.lifespan_context(app):
        yield app, fakes


@pytest.fixture
def app(iv_app: tuple[FastAPI, IvFakes]) -> FastAPI:
    return iv_app[0]


@pytest.fixture
def fakes(iv_app: tuple[FastAPI, IvFakes]) -> IvFakes:
    return iv_app[1]


@pytest.fixture
async def http(app: FastAPI) -> AsyncIterator[httpx.AsyncClient]:
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        yield client


@dataclass
class Tenant:
    tenant: UUID
    user: UUID
    collection: UUID
    connector: UUID
    key: str


@pytest.fixture
def new_tenant(superuser: Conn, make_tenant: MakeTenant) -> Callable[..., Tenant]:
    def _make(scopes: Sequence[str] = ("read", "chat")) -> Tenant:
        tenant, user, collection = make_tenant()
        key = issue_key(superuser, tenant, user, scopes)
        return Tenant(tenant, user, collection, connector(superuser, tenant, collection), key)

    return _make
