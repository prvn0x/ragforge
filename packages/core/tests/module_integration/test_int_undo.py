"""Undo delete on the REAL container: forgetting an exclusion must bring the item back.

Before Phase 2a review round 3, `?forget_exclusion=true` and `DELETE /v1/exclusions/{id}` only
removed the exclusion: an upload stayed tombstoned for ever (uploads are never re-listed) and a
Notion page stayed deleted until someone edited it (incremental sync only lists changed pages).
These tests drive the real routes, the real PgJobQueue and Postgres; only the models are
scripted.
"""

from __future__ import annotations

import asyncio
import dataclasses
from collections.abc import AsyncIterator, Callable, Mapping
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from datetime import datetime
from typing import Any
from uuid import UUID, uuid4

import httpx
import psycopg
import pytest
from fastapi import FastAPI

from module_e2.e2_support import auth, issue_key
from module_integration.int_support import (
    FixedEmbedders,
    ScoringReranker,
    ScriptedChat,
    seed_document,
)
from ragforge_core.adapters.db.agent_data import SqlKnowledgeBase
from ragforge_core.adapters.db.jobs import PgJobQueue
from ragforge_core.apps.api.app import create_app
from ragforge_core.bootstrap import WorkerContainer, build_api, build_worker
from ragforge_core.config import Settings
from ragforge_core.db.engine import SqlTransactionManager, make_engine
from ragforge_core.domain.errors import NotFound
from ragforge_core.domain.ids import document_id
from ragforge_core.domain.models import (
    JobLease,
    JobPriority,
    JobSpec,
    NormalizedBlock,
    SourceDocument,
    SourceItemRef,
    SourceKind,
)
from ragforge_core.services.sync.deps import SyncDeps
from ragforge_core.services.sync.handlers import run_sync_job
from ragforge_core.services.sync.policy import sync_page_key, sync_page_payload

pytestmark = pytest.mark.integration
Conn = psycopg.Connection[tuple[object, ...]]
MakeTenant = Callable[..., tuple[UUID, UUID, UUID]]
OWNER_SCOPES = ["read", "write", "chat", "approvals:write"]


@pytest.fixture
async def http(settings: Settings) -> AsyncIterator[httpx.AsyncClient]:
    container = await build_api(
        settings, chat=ScriptedChat(), reranker=ScoringReranker(), embedders=FixedEmbedders()
    )
    app: FastAPI = create_app(settings, container)
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as client,
    ):
        yield client


def _doc(su: Conn, doc: UUID) -> tuple[bool, str, str]:
    row = su.execute(
        "SELECT deleted_at IS NULL, source_kind, external_id FROM rf.documents WHERE id = %s",
        (doc,),
    ).fetchone()
    assert row is not None
    return bool(row[0]), str(row[1]), str(row[2])


def _exclusions(su: Conn, tenant: UUID, external_id: str) -> int:
    row = su.execute(
        "SELECT count(*) FROM rf.exclusions WHERE tenant_id = %s AND external_id = %s",
        (tenant, external_id),
    ).fetchone()
    assert row is not None
    return int(str(row[0]))


def _queued(su: Conn, tenant: UUID, dedupe_key: str) -> tuple[str, UUID | None] | None:
    row = su.execute(
        "SELECT kind, connector_id FROM rf.jobs "
        "WHERE tenant_id = %s AND dedupe_key = %s AND status = 'queued'",
        (tenant, dedupe_key),
    ).fetchone()
    return None if row is None else (str(row[0]), row[1] if row[1] is None else UUID(str(row[1])))


async def test_forgetting_a_deleted_upload_restores_it_and_reembeds(
    http: httpx.AsyncClient, superuser: Conn, make_tenant: MakeTenant
) -> None:
    tenant, user, coll = make_tenant()
    key = issue_key(superuser, tenant, user, OWNER_SCOPES)
    doc = seed_document(superuser, tenant, coll)
    _, _, upload = _doc(superuser, doc)

    assert (await http.delete(f"/v1/documents/{doc}", headers=auth(key))).status_code == 204
    assert _doc(superuser, doc)[0] is False
    assert _exclusions(superuser, tenant, upload) == 1
    query = {"query": "on-call", "collection_ids": [str(coll)], "k": 3}
    hits = (await http.post("/v1/query", json=query, headers=auth(key))).json()["hits"]
    assert hits == []

    r = await http.delete(
        f"/v1/documents/{doc}", params={"forget_exclusion": "true"}, headers=auth(key)
    )
    assert r.status_code == 204
    assert _doc(superuser, doc)[0] is True, "the tombstone is lifted at once"
    assert _exclusions(superuser, tenant, upload) == 0
    # re-ingest re-embeds any vector GC removed while the document was tombstoned
    assert _queued(superuser, tenant, f"ingest_file:{upload}") == ("ingest_file", None)
    hits = (await http.post("/v1/query", json=query, headers=auth(key))).json()["hits"]
    assert [h["citation"]["document_id"] for h in hits] == [str(doc)]


async def test_forgetting_a_deleted_notion_page_enqueues_its_page_job(
    http: httpx.AsyncClient, superuser: Conn, make_tenant: MakeTenant
) -> None:
    tenant, user, coll = make_tenant()
    key = issue_key(superuser, tenant, user, OWNER_SCOPES)
    connector, doc, page = uuid4(), uuid4(), "1f2e3d4c-5b6a-4978-8695-a4b3c2d1e0f9"
    superuser.execute(
        "INSERT INTO rf.connectors (id, tenant_id, kind, target_collection_id, display_name) "
        "VALUES (%s, %s, 'notion', %s, 'wiki')",
        (connector, tenant, coll),
    )
    superuser.execute(
        "INSERT INTO rf.documents (id, tenant_id, collection_id, connector_id, source_kind, "
        "external_id, title) VALUES (%s, %s, %s, %s, 'notion', %s, 'Page')",
        (doc, tenant, coll, connector, page),
    )
    assert (await http.delete(f"/v1/documents/{doc}", headers=auth(key))).status_code == 204
    exclusion = superuser.execute(
        "SELECT id FROM rf.exclusions WHERE tenant_id = %s AND external_id = %s", (tenant, page)
    ).fetchone()
    assert exclusion is not None

    r = await http.delete(f"/v1/exclusions/{exclusion[0]}", headers=auth(key))
    assert r.status_code == 204
    assert _exclusions(superuser, tenant, page) == 0
    # a fresh fetch (fetched_at after the tombstone) is what lifts a Notion tombstone; the page
    # job counts against its connector's in-flight cap, so connector_id is set
    assert _queued(superuser, tenant, sync_page_key(connector, page)) == ("sync_page", connector)
    audit = superuser.execute(
        "SELECT action, details->>'jobs_enqueued' FROM rf.audit_log "
        "WHERE tenant_id = %s ORDER BY id DESC LIMIT 1",
        (tenant,),
    ).fetchone()
    assert audit == ("exclusion.delete", "1")


async def test_a_purged_upload_is_reingested_from_its_upload_row(
    http: httpx.AsyncClient, superuser: Conn, make_tenant: MakeTenant
) -> None:
    tenant, user, coll = make_tenant()
    key = issue_key(superuser, tenant, user, OWNER_SCOPES)
    doc = seed_document(superuser, tenant, coll)
    _, _, upload = _doc(superuser, doc)
    assert (await http.delete(f"/v1/documents/{doc}", headers=auth(key))).status_code == 204
    superuser.execute("DELETE FROM rf.documents WHERE id = %s", (doc,))  # purge_tombstones ran
    exclusion = superuser.execute(
        "SELECT id FROM rf.exclusions WHERE tenant_id = %s AND external_id = %s",
        (tenant, upload),
    ).fetchone()
    assert exclusion is not None

    r = await http.delete(f"/v1/exclusions/{exclusion[0]}", headers=auth(key))
    assert r.status_code == 204
    assert _queued(superuser, tenant, f"ingest_file:{upload}") == ("ingest_file", None)


async def test_nothing_left_to_restore_is_409_and_keeps_the_exclusion(
    http: httpx.AsyncClient, superuser: Conn, make_tenant: MakeTenant
) -> None:
    tenant, user, _ = make_tenant()
    key = issue_key(superuser, tenant, user, OWNER_SCOPES)
    gone = str(uuid4())  # an upload whose document AND upload row were purged
    exclusion = uuid4()
    superuser.execute(
        "INSERT INTO rf.exclusions (id, tenant_id, source_kind, external_id) "
        "VALUES (%s, %s, 'file', %s)",
        (exclusion, tenant, gone),
    )
    r = await http.delete(f"/v1/exclusions/{exclusion}", headers=auth(key))
    assert r.status_code == 409
    assert _exclusions(superuser, tenant, gone) == 1, "all or nothing"


# ================================================================ round 4: why it was deleted
# Slot 0009 records why a document is tombstoned. A user/agent exclusion ('excluded') is undone at
# once; a deletion confirmed at the source ('source_deleted') keeps the sticky fetched_at rule.
PAGE = "0a1b2c3d-4e5f-4a6b-8c7d-9e0f1a2b3c4d"


class FakeNotion:
    """A SourceConnector + ConnectorFactory pair. `fetched_at` comes from the DATABASE clock minus
    60 s, exactly what NotionConnector reports (Date header - 60 s): an immediate re-fetch after a
    delete is therefore OLDER than the tombstone, which is the case review round 4 found."""

    def __init__(self, su: Conn) -> None:
        self.su = su
        self.title = "Page v1"
        self.gone = False  # True: the owner deleted/unshared the page (definitive NotFound)

    def open(
        self, *, connector_id: UUID, kind: str, token: bytes, config: Mapping[str, Any]
    ) -> AbstractAsyncContextManager[FakeNotion]:
        @asynccontextmanager
        async def session() -> AsyncIterator[FakeNotion]:
            yield self

        return session()

    async def fetch(self, ref: SourceItemRef) -> SourceDocument:
        if self.gone:
            raise NotFound("page")  # what NotionPageGone is (trashed / object_not_found)
        row = self.su.execute("SELECT now() - interval '60 seconds'").fetchone()
        assert row is not None
        fetched_at = row[0]
        assert isinstance(fetched_at, datetime)
        return SourceDocument(
            ref=SourceItemRef(SourceKind.notion, ref.external_id, fetched_at),
            title=self.title,
            external_url=f"https://notion.so/{ref.external_id.replace('-', '')}",
            blocks=(NormalizedBlock(anchor="b1", kind="paragraph", text="On-call is weekly."),),
            fetched_at=fetched_at,
        )

    async def list_changed(self, since: datetime | None) -> AsyncIterator[SourceItemRef]:
        return
        yield  # pragma: no cover

    async def list_all(self) -> AsyncIterator[SourceItemRef]:
        return
        yield  # pragma: no cover

    async def confirm_deleted(self, external_id: str) -> bool:
        return False


@pytest.fixture
async def worker(
    settings: Settings, superuser: Conn
) -> AsyncIterator[tuple[WorkerContainer, SyncDeps]]:
    container = await build_worker(settings)
    try:
        deps = dataclasses.replace(
            container.sync, embedders=FixedEmbedders(), connectors=FakeNotion(superuser)
        )
        yield container, deps
    finally:
        await container.aclose()


async def _claim(queue: PgJobQueue, job_id: UUID) -> JobLease:
    for _ in range(10):
        mine: JobLease | None = None
        for lease in await queue.claim("int-undo", 20):
            if lease.job_id == job_id:
                mine = lease
            else:
                await queue.release(lease)
        if mine is not None:
            return mine
    raise AssertionError(f"job {job_id} was never claimed")


async def _connector(http: httpx.AsyncClient, key: str, coll: UUID) -> UUID:
    """Created through the API so its token is sealed exactly as production does."""
    body = {"kind": "notion", "target_collection_id": str(coll), "display_name": "wiki",
            "token": "secret_test"}  # fmt: skip
    r = await http.post("/v1/connectors", json=body, headers=auth(key))
    assert r.status_code == 201, r.text
    return UUID(r.json()["id"])


async def _sync_page(
    worker: tuple[WorkerContainer, SyncDeps], tenant: UUID, connector: UUID
) -> UUID:
    """Enqueue (or reuse the queued) sync_page and run it through the real handler."""
    container, deps = worker
    async with container.tx.tenant(tenant) as tx:
        job = await container.jobs.enqueue(
            tx,
            JobSpec(
                kind="sync_page",
                priority=JobPriority.interactive,
                payload=sync_page_payload(connector, PAGE),
                dedupe_key=sync_page_key(connector, PAGE),
                connector_id=connector,
            ),
        )
    await run_sync_job(await _claim(container.jobs, job), deps)
    return job


def _state(su: Conn, doc: UUID) -> tuple[bool, str | None, str]:
    row = su.execute(
        "SELECT deleted_at IS NULL, deleted_reason, title FROM rf.documents WHERE id = %s", (doc,)
    ).fetchone()
    assert row is not None
    return bool(row[0]), None if row[1] is None else str(row[1]), str(row[2])


async def test_immediate_undo_of_a_notion_delete_survives_the_refresh_sync(
    http: httpx.AsyncClient,
    worker: tuple[WorkerContainer, SyncDeps],
    superuser: Conn,
    make_tenant: MakeTenant,
) -> None:
    """Review round 4 P1: delete, undo seconds later, and the refresh sync's fetched_at (Notion
    Date - 60 s) is OLDER than the tombstone. The page must end up live, refreshed content."""
    tenant, user, coll = make_tenant()
    key = issue_key(superuser, tenant, user, OWNER_SCOPES)
    connector = await _connector(http, key, coll)
    await _sync_page(worker, tenant, connector)
    doc = document_id(tenant, coll, connector, PAGE)
    assert _state(superuser, doc) == (True, None, "Page v1")

    assert (await http.delete(f"/v1/documents/{doc}", headers=auth(key))).status_code == 204
    assert _state(superuser, doc)[:2] == (False, "excluded")
    r = await http.delete(
        f"/v1/documents/{doc}", params={"forget_exclusion": "true"}, headers=auth(key)
    )
    assert r.status_code == 204
    # Notion is the source of truth: the API only removes the exclusion and asks for a fetch.
    assert _state(superuser, doc)[:2] == (False, "excluded")
    assert _queued(superuser, tenant, sync_page_key(connector, PAGE)) == ("sync_page", connector)

    _, deps = worker
    assert isinstance(deps.connectors, FakeNotion)
    deps.connectors.title = "Page v2"
    await _sync_page(worker, tenant, connector)  # the queued refresh job from the undo
    assert _state(superuser, doc) == (True, None, "Page v2")


@pytest.mark.parametrize(("reason", "live_after"), [("excluded", True), ("source_deleted", False)])
async def test_a_sync_revives_only_tombstones_whose_cause_is_gone(
    http: httpx.AsyncClient,
    worker: tuple[WorkerContainer, SyncDeps],
    superuser: Conn,
    make_tenant: MakeTenant,
    reason: str,
    live_after: bool,
) -> None:
    """With no exclusion left, an older fetch clears an 'excluded' tombstone (the user wants it
    back) but never a 'source_deleted' one (handoff B's sticky rule: the owner deleted it)."""
    tenant, user, coll = make_tenant()
    key = issue_key(superuser, tenant, user, OWNER_SCOPES)
    connector = await _connector(http, key, coll)
    await _sync_page(worker, tenant, connector)
    doc = document_id(tenant, coll, connector, PAGE)
    superuser.execute(
        "UPDATE rf.documents SET deleted_at = now(), deleted_reason = %s WHERE id = %s",
        (reason, doc),
    )
    _, deps = worker
    assert isinstance(deps.connectors, FakeNotion)
    deps.connectors.title = "Page v2"
    await _sync_page(worker, tenant, connector)
    assert _state(superuser, doc)[0] is live_after


async def test_undo_never_revives_a_page_deleted_at_the_source(
    http: httpx.AsyncClient, superuser: Conn, make_tenant: MakeTenant
) -> None:
    tenant, user, coll = make_tenant()
    key = issue_key(superuser, tenant, user, OWNER_SCOPES)
    connector, doc = uuid4(), uuid4()
    superuser.execute(
        "INSERT INTO rf.connectors (id, tenant_id, kind, target_collection_id, display_name) "
        "VALUES (%s, %s, 'notion', %s, 'wiki')",
        (connector, tenant, coll),
    )
    superuser.execute(  # reconcile confirmed it gone; the trigger defaults the reason
        "INSERT INTO rf.documents (id, tenant_id, collection_id, connector_id, source_kind, "
        "external_id, title, deleted_at) VALUES (%s, %s, %s, %s, 'notion', %s, 'P', now())",
        (doc, tenant, coll, connector, PAGE),
    )
    exclusion = uuid4()
    superuser.execute(
        "INSERT INTO rf.exclusions (id, tenant_id, source_kind, external_id) "
        "VALUES (%s, %s, 'notion', %s)",
        (exclusion, tenant, PAGE),
    )
    assert _state(superuser, doc)[:2] == (False, "source_deleted")

    assert (await http.delete(f"/v1/exclusions/{exclusion}", headers=auth(key))).status_code == 204
    assert _state(superuser, doc)[:2] == (False, "source_deleted")
    assert _queued(superuser, tenant, sync_page_key(connector, PAGE)) is None
    audit = superuser.execute(
        "SELECT details->>'kept_source_deleted' FROM rf.audit_log "
        "WHERE tenant_id = %s ORDER BY id DESC LIMIT 1",
        (tenant,),
    ).fetchone()
    assert audit == ("1",)


def _two_copies(su: Conn, tenant: UUID, coll_a: UUID) -> tuple[UUID, UUID]:
    """One Notion page synced into two collections through two connectors."""
    coll_b = uuid4()
    su.execute(
        "INSERT INTO rf.collections (id, tenant_id, name, active_space_id) VALUES (%s, %s, %s, 1)",
        (coll_b, tenant, f"b-{coll_b}"),
    )
    docs = []
    for coll in (coll_a, coll_b):
        connector, doc = uuid4(), uuid4()
        su.execute(
            "INSERT INTO rf.connectors (id, tenant_id, kind, target_collection_id, display_name) "
            "VALUES (%s, %s, 'notion', %s, 'wiki')",
            (connector, tenant, coll),
        )
        su.execute(
            "INSERT INTO rf.documents (id, tenant_id, collection_id, connector_id, source_kind, "
            "external_id, title) VALUES (%s, %s, %s, %s, 'notion', %s, 'P')",
            (doc, tenant, coll, connector, PAGE),
        )
        docs.append(doc)
    return docs[0], docs[1]


async def test_deleting_one_copy_excludes_every_copy_and_undo_refetches_each(
    http: httpx.AsyncClient, superuser: Conn, make_tenant: MakeTenant
) -> None:
    """The exclusion is tenant-wide; a copy left live would be frozen (every sync skips it)."""
    tenant, user, coll = make_tenant()
    key = issue_key(superuser, tenant, user, OWNER_SCOPES)
    in_a, in_b = _two_copies(superuser, tenant, coll)

    assert (await http.delete(f"/v1/documents/{in_a}", headers=auth(key))).status_code == 204
    assert _state(superuser, in_a)[:2] == (False, "excluded")
    assert _state(superuser, in_b)[:2] == (False, "excluded")
    audit = superuser.execute(
        "SELECT details->>'copies' FROM rf.audit_log WHERE tenant_id = %s AND action = %s",
        (tenant, "document.delete"),
    ).fetchone()
    assert audit == ("2",)

    r = await http.delete(
        f"/v1/documents/{in_a}", params={"forget_exclusion": "true"}, headers=auth(key)
    )
    assert r.status_code == 204
    assert _exclusions(superuser, tenant, PAGE) == 0
    rows = superuser.execute(
        "SELECT connector_id FROM rf.documents WHERE id = ANY(%s)", ([in_a, in_b],)
    ).fetchall()
    for (raw,) in rows:  # each copy is refreshed by its own connector's page job
        connector = UUID(str(raw))
        assert _queued(superuser, tenant, sync_page_key(connector, PAGE)) == (
            "sync_page",
            connector,
        )


async def test_agent_removal_also_excludes_every_copy(
    http: httpx.AsyncClient, superuser: Conn, make_tenant: MakeTenant, settings: Settings
) -> None:
    tenant, _, coll = make_tenant()
    in_a, in_b = _two_copies(superuser, tenant, coll)
    engine = make_engine(settings, "api")  # the agent runs as rf_api
    try:
        async with SqlTransactionManager(engine).tenant(tenant) as tx:
            assert await SqlKnowledgeBase().remove_from_index(tx, in_a, [coll], "user:test")
    finally:
        await engine.dispose()
    assert _state(superuser, in_a)[:2] == (False, "excluded")
    assert _state(superuser, in_b)[:2] == (False, "excluded")


async def test_undo_after_the_owner_deleted_the_page_keeps_it_deleted(
    http: httpx.AsyncClient,
    worker: tuple[WorkerContainer, SyncDeps],
    superuser: Conn,
    make_tenant: MakeTenant,
) -> None:
    """Review round 5 P1: exclude, the owner deletes (or unshares) the page in Notion, then undo.
    The refresh fetch gets a definitive "gone": the copy becomes 'source_deleted' and the job
    completes (not dead), so neither this undo nor a later one can serve the deleted content."""
    tenant, user, coll = make_tenant()
    key = issue_key(superuser, tenant, user, OWNER_SCOPES)
    connector = await _connector(http, key, coll)
    await _sync_page(worker, tenant, connector)
    doc = document_id(tenant, coll, connector, PAGE)
    assert (await http.delete(f"/v1/documents/{doc}", headers=auth(key))).status_code == 204

    _, deps = worker
    assert isinstance(deps.connectors, FakeNotion)
    deps.connectors.gone = True
    r = await http.delete(
        f"/v1/documents/{doc}", params={"forget_exclusion": "true"}, headers=auth(key)
    )
    assert r.status_code == 204
    assert _state(superuser, doc)[:2] == (False, "excluded"), "never revived before the fetch"

    job = await _sync_page(worker, tenant, connector)
    assert _state(superuser, doc)[:2] == (False, "source_deleted")
    status = superuser.execute(
        "SELECT status FROM rf.jobs WHERE id = %s UNION ALL "
        "SELECT status FROM rf.jobs_history WHERE id = %s",
        (job, job),
    ).fetchone()
    assert status == ("done",)


async def test_undo_with_every_copy_purged_asks_each_active_connector(
    http: httpx.AsyncClient, superuser: Conn, make_tenant: MakeTenant
) -> None:
    """No document row left: the page id is global, so every active notion connector of the
    tenant gets one page job (a connector that cannot see the page records nothing)."""
    tenant, user, coll = make_tenant()
    key = issue_key(superuser, tenant, user, OWNER_SCOPES)
    active, paused = uuid4(), uuid4()
    for connector, status in ((active, "active"), (paused, "paused")):
        superuser.execute(
            "INSERT INTO rf.connectors (id, tenant_id, kind, target_collection_id, "
            "display_name, status) VALUES (%s, %s, 'notion', %s, 'wiki', %s)",
            (connector, tenant, coll, status),
        )
    exclusion = uuid4()
    superuser.execute(
        "INSERT INTO rf.exclusions (id, tenant_id, source_kind, external_id) "
        "VALUES (%s, %s, 'notion', %s)",
        (exclusion, tenant, PAGE),
    )
    assert (await http.delete(f"/v1/exclusions/{exclusion}", headers=auth(key))).status_code == 204
    assert _queued(superuser, tenant, sync_page_key(active, PAGE)) == ("sync_page", active)
    assert _queued(superuser, tenant, sync_page_key(paused, PAGE)) is None


class GatedNotion(FakeNotion):
    """The FIRST fetch is computed (fetched_at taken from the DB clock), then held until `gate`
    opens: a slow refresh job that fetched before the owner deleted the page but commits after.
    Later fetches behave like FakeNotion (definitive NotFound once `gone`)."""

    def __init__(self, su: Conn) -> None:
        super().__init__(su)
        self.fetched = asyncio.Event()
        self.gate = asyncio.Event()
        self.calls = 0

    async def fetch(self, ref: SourceItemRef) -> SourceDocument:
        self.calls += 1
        if self.calls > 1:
            return await super().fetch(ref)
        doc = await super().fetch(ref)
        self.fetched.set()
        await self.gate.wait()
        return doc


async def test_a_slow_refresh_cannot_revive_a_page_the_owner_deleted_meanwhile(
    http: httpx.AsyncClient,
    worker: tuple[WorkerContainer, SyncDeps],
    superuser: Conn,
    make_tenant: MakeTenant,
) -> None:
    """Review round 6 P2: undo (J1 fetches) -> the owner deletes the page -> J2 confirms "gone"
    and commits 'source_deleted' -> J1 commits LAST. The upgrade must stamp the confirmation
    time; keeping the old exclusion time let J1's later fetch clear the sticky tombstone."""
    tenant, user, coll = make_tenant()
    key = issue_key(superuser, tenant, user, OWNER_SCOPES)
    connector = await _connector(http, key, coll)
    await _sync_page(worker, tenant, connector)
    doc = document_id(tenant, coll, connector, PAGE)
    assert (await http.delete(f"/v1/documents/{doc}", headers=auth(key))).status_code == 204
    # The exclusion is an hour old, so J1's fetch (DB now - 60 s) is NEWER than it: only the
    # confirmation time can keep J1 from reviving the page.
    superuser.execute(
        "UPDATE rf.documents SET deleted_at = now() - interval '1 hour' WHERE id = %s", (doc,)
    )
    r = await http.delete(
        f"/v1/documents/{doc}", params={"forget_exclusion": "true"}, headers=auth(key)
    )
    assert r.status_code == 204

    container, base = worker
    notion = GatedNotion(superuser)
    deps = dataclasses.replace(base, connectors=notion)
    j1 = await _claim(container.jobs, _job_id(superuser, tenant, sync_page_key(connector, PAGE)))
    first = asyncio.create_task(run_sync_job(j1, deps))
    async with asyncio.timeout(10):
        await notion.fetched.wait()  # J1 holds a fetch from before the owner's deletion

    notion.gone = True  # the owner deletes the page in Notion
    async with container.tx.tenant(tenant) as tx:
        j2 = await container.jobs.enqueue(
            tx,
            JobSpec(
                kind="sync_page",
                priority=JobPriority.interactive,
                payload=sync_page_payload(connector, PAGE),
                dedupe_key=sync_page_key(connector, PAGE),
                connector_id=connector,
            ),
        )
    assert j2 != j1.job_id, "J1 is running, so J2 is a new job"
    await run_sync_job(await _claim(container.jobs, j2), deps)
    assert _state(superuser, doc)[:2] == (False, "source_deleted")

    notion.gate.set()
    async with asyncio.timeout(10):
        await first  # J1 commits last
    assert _state(superuser, doc)[:2] == (False, "source_deleted"), "owner-deleted stays deleted"


def _job_id(su: Conn, tenant: UUID, dedupe_key: str) -> UUID:
    row = su.execute(
        "SELECT id FROM rf.jobs WHERE tenant_id = %s AND dedupe_key = %s AND status = 'queued'",
        (tenant, dedupe_key),
    ).fetchone()
    assert row is not None
    return UUID(str(row[0]))


class FrozenClockNotion(FakeNotion):
    """Every fetch reports the same fetched_at: Notion's Date header has 1 s resolution, so an
    undo refreshed within the second of the previous sync fetches an EQUAL timestamp."""

    def __init__(self, su: Conn, fetched_at: datetime) -> None:
        super().__init__(su)
        self.at = fetched_at

    async def fetch(self, ref: SourceItemRef) -> SourceDocument:
        doc = await super().fetch(ref)
        return dataclasses.replace(
            doc, ref=dataclasses.replace(doc.ref, last_edited_time=self.at), fetched_at=self.at
        )


async def test_undo_refreshed_within_the_same_second_still_revives(
    http: httpx.AsyncClient,
    worker: tuple[WorkerContainer, SyncDeps],
    superuser: Conn,
    make_tenant: MakeTenant,
) -> None:
    """Review round 6 P3: an equal fetched_at used to classify the refresh as 'stale', leaving
    the copy 'excluded' with no exclusion left. The stored content is current: lift it."""
    tenant, user, coll = make_tenant()
    key = issue_key(superuser, tenant, user, OWNER_SCOPES)
    connector = await _connector(http, key, coll)
    row = superuser.execute("SELECT now() - interval '60 seconds'").fetchone()
    assert row is not None
    assert isinstance(row[0], datetime)
    container, base = worker
    deps = dataclasses.replace(base, connectors=FrozenClockNotion(superuser, row[0]))
    worker_frozen = (container, deps)
    await _sync_page(worker_frozen, tenant, connector)
    doc = document_id(tenant, coll, connector, PAGE)
    assert (await http.delete(f"/v1/documents/{doc}", headers=auth(key))).status_code == 204
    r = await http.delete(
        f"/v1/documents/{doc}", params={"forget_exclusion": "true"}, headers=auth(key)
    )
    assert r.status_code == 204

    await _sync_page(worker_frozen, tenant, connector)
    assert _state(superuser, doc) == (True, None, "Page v1")
