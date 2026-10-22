"""EV-SNAP fixtures: the real API app and the real worker handlers on the shared test Postgres.
Only the embedder is fake (no model); the flag dependency is switched on per app."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Iterator
from uuid import UUID, uuid4

import httpx
import psycopg
import pytest
from fastapi import FastAPI
from sqlalchemy.ext.asyncio import AsyncEngine

from module_eval_snap.ev_snap_support import (
    SOURCE_DIM,
    SOURCE_SPACE,
    DemoPage,
    FakeEmbedders,
    Source,
    load_demo_pages,
    seed_source,
)
from pgtest import IdentityReranker
from ragforge_core.adapters.db.index import PgDocumentIndex
from ragforge_core.adapters.db.jobs import PgJobQueue
from ragforge_core.apps.api.app import create_app
from ragforge_core.apps.api.routes.eval_snapshots import require_eval_api
from ragforge_core.apps.worker.main import SqlEvalSnapshotStore
from ragforge_core.bootstrap import build_api
from ragforge_core.config import Settings
from ragforge_core.db.engine import SqlTransactionManager, make_engine
from ragforge_core.services.eval_snapshot import EvalSnapshotDeps

Conn = psycopg.Connection[tuple[object, ...]]
MakeTenant = Callable[..., tuple[UUID, UUID, UUID]]
NEW_SPACE_DIM = 4


@pytest.fixture(scope="session")
def demo_pages() -> list[DemoPage]:
    return load_demo_pages()


@pytest.fixture
def make_source(
    superuser: Conn, make_tenant: MakeTenant, demo_pages: list[DemoPage]
) -> Callable[..., Source]:
    def _make(pages: list[DemoPage] | None = None) -> Source:
        tenant, user, collection = make_tenant()
        return seed_source(
            superuser, tenant, user, collection, demo_pages if pages is None else pages
        )

    return _make


@pytest.fixture
def new_space(superuser: Conn) -> Iterator[int]:
    """A fresh embedding space (a different 'model') to re-embed a snapshot into. Retired
    afterwards, so later tests in the session never see a live foreign-provider space."""
    row = superuser.execute(
        "INSERT INTO rf.embedding_spaces (provider, model, dim, distance, status) "
        "VALUES ('fake', %s, %s, 'cosine', 'building') RETURNING id",
        (f"fake-{uuid4().hex[:12]}", NEW_SPACE_DIM),
    ).fetchone()
    assert row is not None
    space = int(str(row[0]))
    yield space
    superuser.execute(
        "UPDATE rf.embedding_spaces SET status = 'retired', retired_at = now() WHERE id = %s",
        (space,),
    )


@pytest.fixture
async def worker_engine(settings: Settings) -> AsyncIterator[AsyncEngine]:
    engine = make_engine(settings, "worker")
    yield engine
    await engine.dispose()


@pytest.fixture
def queue(worker_engine: AsyncEngine) -> PgJobQueue:
    return PgJobQueue(worker_engine)


@pytest.fixture
def embedders(new_space: int) -> FakeEmbedders:
    return FakeEmbedders({SOURCE_SPACE: SOURCE_DIM, new_space: NEW_SPACE_DIM})


@pytest.fixture
def snap_deps(
    worker_engine: AsyncEngine, queue: PgJobQueue, embedders: FakeEmbedders
) -> EvalSnapshotDeps:
    return EvalSnapshotDeps(
        tx=SqlTransactionManager(worker_engine),
        queue=queue,
        store=SqlEvalSnapshotStore(),
        index=PgDocumentIndex(),
        embedders=embedders,
        embed_batch=8,
    )


async def _app(settings: Settings, *, eval_api: bool) -> AsyncIterator[FastAPI]:
    container = await build_api(settings, reranker=IdentityReranker())  # closed by the lifespan
    app = create_app(settings, container)
    if eval_api:
        app.dependency_overrides[require_eval_api] = lambda: None
    async with app.router.lifespan_context(app):
        yield app


@pytest.fixture
async def http(settings: Settings) -> AsyncIterator[httpx.AsyncClient]:
    async for app in _app(settings, eval_api=True):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            yield client


@pytest.fixture
async def http_flag_off(settings: Settings) -> AsyncIterator[httpx.AsyncClient]:
    async for app in _app(settings, eval_api=False):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            yield client


@pytest.fixture(autouse=True)
def _no_leftover_snapshot_jobs(superuser: Conn) -> Iterator[None]:
    """Remove this module's queued copy jobs after each test. They are enqueued at priority 0,
    and other modules' tests claim the next 20 jobs: leftovers starved them (integration tests
    failed with "job ... was never claimed" when run after this module)."""
    yield
    superuser.execute("DELETE FROM rf.jobs WHERE kind = 'eval_snapshot_copy'")
