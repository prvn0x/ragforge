"""The upload seam end to end on REAL containers: HTTP upload -> real PgJobQueue row -> the real
worker handler (`run_sync_job` with `build_worker`'s SyncDeps) -> searchable via /v1/query.

Plus the two guarantees DELETE /v1/documents makes to sync, checked against the real code:
- lock order: the route waits on the source-item advisory lock (held here exactly as sync's
  fenced commit holds it) WITHOUT holding a row lock on the document, so the two cannot deadlock;
- the exclusion is honoured: an ingest that runs after the delete commits nothing.

Only the models are scripted (embedder, reranker, chat). The parser, blob store, chunker, index
and queue are production adapters.
"""

from __future__ import annotations

import asyncio
import dataclasses
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from uuid import UUID

import httpx
import psycopg
import pytest
from fastapi import FastAPI

from module_e2.e2_support import auth, issue_key
from module_integration.int_support import FixedEmbedders, ScoringReranker, ScriptedChat
from pgtest import connect
from ragforge_core.adapters.db.jobs import PgJobQueue
from ragforge_core.apps.api.app import create_app
from ragforge_core.bootstrap import WorkerContainer, build_api, build_worker
from ragforge_core.config import Settings
from ragforge_core.domain.models import JobLease
from ragforge_core.services.sync.deps import SyncDeps
from ragforge_core.services.sync.handlers import run_sync_job

pytestmark = pytest.mark.integration
Conn = psycopg.Connection[tuple[object, ...]]
MakeTenant = Callable[..., tuple[UUID, UUID, UUID]]
OWNER_SCOPES = ["read", "write", "chat", "approvals:write"]
HANDBOOK = b"# Handbook\n\n## On-call\n\nThe on-call rotation lasts one week.\n"


@pytest.fixture
def upload_settings(settings: Settings, tmp_path: Path) -> Settings:
    """API and worker share one upload directory, as the compose volume does."""
    return settings.model_copy(update={"upload_dir": str(tmp_path / "uploads")})


@pytest.fixture
async def http(upload_settings: Settings) -> AsyncIterator[httpx.AsyncClient]:
    container = await build_api(
        upload_settings,
        chat=ScriptedChat(),
        reranker=ScoringReranker(),
        embedders=FixedEmbedders(),
    )
    app: FastAPI = create_app(upload_settings, container)
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as client,
    ):
        yield client


@pytest.fixture
async def worker(upload_settings: Settings) -> AsyncIterator[tuple[WorkerContainer, SyncDeps]]:
    container = await build_worker(upload_settings)
    try:
        # Production SyncDeps; only the embedder is scripted (no Ollama in tests).
        yield container, dataclasses.replace(container.sync, embedders=FixedEmbedders())
    finally:
        await container.aclose()


async def _claim(queue: PgJobQueue, job_id: UUID) -> JobLease:
    """Claim through the real claim_jobs; leftover jobs of other tests are released untouched."""
    for _ in range(10):
        mine: JobLease | None = None
        for lease in await queue.claim("int-uploads", 20):
            if lease.job_id == job_id:
                mine = lease
            else:
                await queue.release(lease)
        if mine is not None:
            return mine
    raise AssertionError(f"job {job_id} was never claimed")


def _job(su: Conn, job_id: UUID) -> tuple[str, str, object]:
    row = su.execute(
        "SELECT status, kind, progress FROM rf.jobs WHERE id = %s "
        "UNION ALL SELECT status, kind, progress FROM rf.jobs_history WHERE id = %s",
        (job_id, job_id),
    ).fetchone()
    assert row is not None
    return str(row[0]), str(row[1]), row[2]


async def _upload(http: httpx.AsyncClient, key: str, coll: UUID) -> UUID:
    r = await http.post(
        f"/v1/collections/{coll}/files",
        files={"file": ("handbook.md", HANDBOOK, "text/markdown")},
        headers=auth(key),
    )
    assert r.status_code == 202, r.text
    return UUID(r.json()["job_id"])


async def test_upload_is_ingested_by_the_real_worker_handler_and_searchable(
    http: httpx.AsyncClient,
    worker: tuple[WorkerContainer, SyncDeps],
    superuser: Conn,
    make_tenant: MakeTenant,
) -> None:
    container, deps = worker
    tenant, user, coll = make_tenant()
    key = issue_key(superuser, tenant, user, OWNER_SCOPES)

    job_id = await _upload(http, key, coll)
    assert _job(superuser, job_id)[:2] == ("queued", "ingest_file")  # the real queue row

    await run_sync_job(await _claim(container.jobs, job_id), deps)

    status, _, progress = _job(superuser, job_id)
    assert status == "done"
    assert isinstance(progress, dict)
    assert progress["outcome"] == "inserted"
    query = {"query": "on-call rotation", "collection_ids": [str(coll)], "k": 3}
    hits = (await http.post("/v1/query", json=query, headers=auth(key))).json()["hits"]
    assert hits, "the ingested chunks are searchable"
    assert "one week" in hits[0]["citation"]["snippet"]


async def test_an_ingest_after_the_delete_honours_the_exclusion(
    http: httpx.AsyncClient,
    worker: tuple[WorkerContainer, SyncDeps],
    superuser: Conn,
    make_tenant: MakeTenant,
) -> None:
    container, deps = worker
    tenant, user, coll = make_tenant()
    key = issue_key(superuser, tenant, user, OWNER_SCOPES)
    first = await _upload(http, key, coll)
    await run_sync_job(await _claim(container.jobs, first), deps)
    doc = superuser.execute(
        "SELECT id, external_id FROM rf.documents WHERE collection_id = %s", (coll,)
    ).fetchone()
    assert doc is not None
    assert (await http.delete(f"/v1/documents/{doc[0]}", headers=auth(key))).status_code == 204

    # a re-ingest of the same upload (e.g. a retry that was already queued) runs afterwards
    again = superuser.execute(
        "INSERT INTO rf.jobs (tenant_id, kind, priority, payload, dedupe_key) "
        "VALUES (%s, 'ingest_file', 0, jsonb_build_object('upload_id', %s::text), %s) "
        "RETURNING id",
        (tenant, doc[1], f"ingest_file:{doc[1]}"),
    ).fetchone()
    assert again is not None
    await run_sync_job(await _claim(container.jobs, UUID(str(again[0]))), deps)

    status, _, progress = _job(superuser, UUID(str(again[0])))
    assert status == "done"
    # skipped by the handler's up-front check; the fenced commit re-checks too (indexing.py)
    assert progress == {"skipped": "excluded"}
    live = superuser.execute(
        "SELECT deleted_at IS NULL FROM rf.documents WHERE id = %s", (doc[0],)
    ).fetchone()
    assert live == (False,), "the excluded document stays deleted"


async def test_delete_waits_on_the_source_lock_without_holding_the_document_row(
    http: httpx.AsyncClient, settings: Settings, superuser: Conn, make_tenant: MakeTenant
) -> None:
    """Sync's fenced commit takes the source-item advisory lock, then the document row. The API
    delete must take them in the same order, or the two deadlock (40P01)."""
    tenant, user, coll = make_tenant()
    key = issue_key(superuser, tenant, user, OWNER_SCOPES)
    page, doc, connector = "lock-page", UUID(int=tenant.int ^ 1), UUID(int=tenant.int ^ 2)
    superuser.execute(
        "INSERT INTO rf.connectors (id, tenant_id, kind, target_collection_id, display_name) "
        "VALUES (%s, %s, 'notion', %s, 'c')",
        (connector, tenant, coll),
    )
    superuser.execute(
        "INSERT INTO rf.documents (id, tenant_id, collection_id, connector_id, source_kind, "
        "external_id, title) VALUES (%s, %s, %s, %s, 'notion', %s, 'Doc')",
        (doc, tenant, coll, connector, page),
    )
    with connect(settings, "worker") as sync_conn:
        # Hold the lock exactly as SqlSyncStore.lock_source_item does inside sync's commit.
        sync_conn.execute("SELECT set_config('app.tenant_id', %s, true)", (str(tenant),))
        sync_conn.execute(
            "SELECT pg_advisory_xact_lock(hashtextextended("
            "'rf.source:' || rf.current_tenant()::text || ':' || 'notion' || ':' || %s, 0))",
            (page,),
        )
        delete = asyncio.create_task(http.delete(f"/v1/documents/{doc}", headers=auth(key)))
        async with asyncio.timeout(10):
            while True:
                waiting = superuser.execute(
                    "SELECT pid FROM pg_locks WHERE locktype = 'advisory' AND NOT granted"
                ).fetchall()
                if waiting:
                    break
                await asyncio.sleep(0.02)
        pids = [int(str(r[0])) for r in waiting]
        row_locks = superuser.execute(
            "SELECT count(*) FROM pg_locks WHERE pid = ANY(%s) "
            "AND relation = 'rf.documents'::regclass AND mode = 'RowShareLock'",
            (pids,),
        ).fetchone()
        assert row_locks == (0,), "the delete must not lock the document before the source lock"
        sync_conn.commit()  # sync's commit finishes and releases the advisory lock
    async with asyncio.timeout(10):
        response = await delete
    assert response.status_code == 204
