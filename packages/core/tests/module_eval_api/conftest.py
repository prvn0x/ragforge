"""Module EV-API fixtures: the real API app on the real test database, model ports replaced by
fakes through FastAPI's dependency_overrides (the module E2/IV pattern, reusing their fakes).

RAGFORGE_EVAL_API is ON here (`flags.eval_api`); a test turns it off by setting it False."""

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
from module_eval_api.ev_api_support import Conn
from module_iv.iv_support import FakeChat, connector
from ragforge_core.adapters.blob.local import LocalBlobStore
from ragforge_core.apps.api import deps
from ragforge_core.apps.api.app import create_app
from ragforge_core.apps.api.routes import interview
from ragforge_core.bootstrap import build_api
from ragforge_core.config import Settings
from ragforge_core.domain.models import RankedChunk
from ragforge_core.rag.retriever import HybridRetriever

MakeTenant = Callable[..., tuple[UUID, UUID, UUID]]
EVAL_KEY_SCOPES = ("eval", "chat", "read")


class _UnusedReranker:
    async def rerank(self, query: str, candidates: Sequence[RankedChunk]) -> list[RankedChunk]:
        raise AssertionError("routes must use the overridden reranker")


@dataclass
class Flags:
    eval_api: bool = True


@dataclass
class EvFakes:
    txm: TrackingTxManager
    retriever: FakeRetriever
    embedders: FakeEmbedders
    reranker: FakeReranker
    chat: FakeChat
    agent: ScriptedAgent
    sanitizers: SanitizerFactory
    flags: Flags
    # set by a stages test: the wired retriever becomes a real HybridRetriever over a fake backend
    hybrid: HybridRetriever | None = None


@pytest.fixture
async def ev_app(settings: Settings, tmp_path: Path) -> AsyncIterator[tuple[FastAPI, EvFakes]]:
    container = await build_api(settings, reranker=_UnusedReranker())  # closed by the lifespan
    txm = TrackingTxManager(container.tx)
    fakes = EvFakes(
        txm=txm,
        retriever=FakeRetriever(hits=[], spaces=(1,)),
        embedders=FakeEmbedders(txm),
        reranker=FakeReranker(txm),
        chat=FakeChat(open_tx=lambda: txm.open),
        agent=ScriptedAgent(),
        sanitizers=SanitizerFactory(),
        flags=Flags(),
    )
    app = create_app(settings, container)
    limiter = deps.PrincipalLimiter(rate_per_s=100.0, burst=100, concurrency=4)

    app.dependency_overrides.update(
        {
            deps.get_tx_manager: lambda: fakes.txm,
            deps.get_job_queue: FakeJobQueue,
            deps.get_retriever: lambda: fakes.hybrid or fakes.retriever,
            deps.get_embedders: lambda: fakes.embedders,
            deps.get_reranker: lambda: fakes.reranker,
            deps.get_blob_store: lambda: LocalBlobStore(tmp_path / "blobs"),
            deps.get_agent_runner: lambda: fakes.agent,
            deps.get_sanitizer_factory: lambda: fakes.sanitizers,
            deps.get_chat_model: lambda: fakes.chat,
            deps.get_eval_api_enabled: lambda: fakes.flags.eval_api,
            interview.get_interview_rng: lambda: random.Random(7),  # noqa: S311
            deps.get_interview_limiter: lambda: limiter,
        }
    )
    async with app.router.lifespan_context(app):
        yield app, fakes


@pytest.fixture
def app(ev_app: tuple[FastAPI, EvFakes]) -> FastAPI:
    return ev_app[0]


@pytest.fixture
def fakes(ev_app: tuple[FastAPI, EvFakes]) -> EvFakes:
    return ev_app[1]


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
    key: str  # scopes as requested
    eval_key: str  # eval + chat + read


@pytest.fixture
def new_tenant(superuser: Conn, make_tenant: MakeTenant) -> Callable[..., Tenant]:
    def _make(scopes: Sequence[str] = ("read", "chat")) -> Tenant:
        tenant, user, collection = make_tenant()
        owner, _, _ = make_tenant()
        # the /v1/eval surface serves only an eval tenant: make this one the owner's pair
        superuser.execute(
            "UPDATE rf.tenants SET kind = 'eval', eval_of_tenant_id = %s WHERE id = %s",
            (owner, tenant),
        )
        key = issue_key(superuser, tenant, user, scopes)
        eval_key = issue_key(superuser, tenant, user, EVAL_KEY_SCOPES, name="eval")
        return Tenant(
            tenant, user, collection, connector(superuser, tenant, collection), key, eval_key
        )

    return _make
