"""SqlSyncStore against real Postgres: the rf_api/rf_worker grants, RLS tenant isolation,
monotonic cursors and the exclusion lock."""

from __future__ import annotations

import asyncio
import os
from collections.abc import Callable
from datetime import UTC, datetime
from uuid import UUID, uuid4

import psycopg
import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from ragforge_core.adapters.connectors.store import SqlSyncStore
from ragforge_core.adapters.crypto.envelope import EnvelopeSecretBox
from ragforge_core.db.engine import SqlTransactionManager
from ragforge_core.domain.errors import NotFound
from ragforge_core.domain.models import SourceKind
from ragforge_core.services.sync.handlers import associated_data
from ragforge_core.services.sync.ports import SyncStore

pytestmark = pytest.mark.integration
Conn = psycopg.Connection[tuple[object, ...]]
MakeTenant = Callable[..., tuple[UUID, UUID, UUID]]
STORE: SyncStore = SqlSyncStore()  # structural conformance, checked by mypy
BOX = EnvelopeSecretBox(os.urandom(32))


def at(minute: int) -> datetime:
    return datetime(2026, 10, 16, 8, minute, tzinfo=UTC)


async def make_connector(api_engine: AsyncEngine, tenant: UUID, coll: UUID) -> UUID:
    cid = uuid4()
    sealed = BOX.seal(b"tok", associated_data=associated_data(tenant, cid))
    async with SqlTransactionManager(api_engine).tenant(tenant) as tx:
        await tx.execute(
            text(
                "INSERT INTO rf.connectors (id, tenant_id, kind, target_collection_id, "
                "display_name, config, secret_ciphertext, wrapped_dek) VALUES "
                "(:id, :t, 'notion', :c, 'n', CAST('{\"root\": \"x\"}' AS jsonb), :s, :d)"
            ),
            {"id": cid, "t": tenant, "c": coll, "s": sealed.ciphertext, "d": sealed.wrapped_dek},
        )
    return cid


def seed_doc(
    superuser: Conn, tenant: UUID, coll: UUID, cid: UUID, ext: str, *, deleted: bool = False,
    synced: list[str] | None = None,
) -> UUID:  # fmt: skip
    did = uuid4()
    superuser.execute(
        "INSERT INTO rf.documents (id, tenant_id, collection_id, connector_id, source_kind, "
        "external_id, title, fetched_at, synced_source_ids, deleted_at) "
        "VALUES (%s, %s, %s, %s, 'notion', %s, 't', %s, %s, %s)",
        (did, tenant, coll, cid, ext, at(10), synced or [], at(11) if deleted else None),
    )
    return did


async def test_connector_roundtrip_and_secret_lifecycle(
    api_engine: AsyncEngine, worker_engine: AsyncEngine, make_tenant: MakeTenant
) -> None:
    tenant, _, coll = make_tenant()
    cid = await make_connector(api_engine, tenant, coll)
    txm = SqlTransactionManager(worker_engine)
    async with txm.tenant(tenant) as tx:
        conn = await STORE.load_connector(tx, cid)
    assert conn is not None
    assert conn.status == "active"
    assert conn.config == {"root": "x"}
    assert conn.sealed is not None
    assert "sealed" not in repr(conn)
    assert BOX.open(conn.sealed, associated_data=associated_data(tenant, cid)) == b"tok"

    async with txm.tenant(tenant) as tx:
        await STORE.mark_connector_error(tx, cid, "boom" * 500)
        await STORE.destroy_secret(tx, cid)
        conn = await STORE.load_connector(tx, cid)
    assert conn is not None
    assert conn.sealed is None
    assert conn.status == "deleting"


async def test_cursor_is_monotonic_even_under_concurrent_advances(
    api_engine: AsyncEngine, worker_engine: AsyncEngine, make_tenant: MakeTenant
) -> None:
    tenant, _, coll = make_tenant()
    cid = await make_connector(api_engine, tenant, coll)
    txm = SqlTransactionManager(worker_engine)

    async def advance(value: datetime) -> datetime:
        async with txm.tenant(tenant) as tx:
            return await STORE.advance_cursor(tx, cid, "changed", value)

    assert await advance(at(30)) == at(30)
    assert await advance(at(10)) == at(30)  # never backwards
    await asyncio.gather(*(advance(at(m)) for m in (31, 45, 33, 40, 44)))
    async with txm.tenant(tenant) as tx:
        assert await STORE.read_cursor(tx, cid, "changed") == at(45)
        assert await STORE.read_cursor(tx, cid, "other") is None


async def test_documents_facts_exclude_tombstones_and_delete_in_batches(
    api_engine: AsyncEngine, worker_engine: AsyncEngine, make_tenant: MakeTenant,
    superuser: Conn,
) -> None:  # fmt: skip
    tenant, _, coll = make_tenant()
    cid = await make_connector(api_engine, tenant, coll)
    for i in range(3):
        seed_doc(superuser, tenant, coll, cid, f"p{i}", synced=["src"] if i == 0 else None)
    seed_doc(superuser, tenant, coll, cid, "dead", deleted=True)
    txm = SqlTransactionManager(worker_engine)
    async with txm.tenant(tenant) as tx:
        facts = await STORE.fetched_at_by_external_id(tx, cid, ["p0", "dead", "zzz"])
        live = await STORE.live_documents(tx, cid)
    assert facts == {"p0": at(10)}
    assert [d.external_id for d in live] == ["p0", "p1", "p2"]
    assert list(live[0].synced_source_ids) == ["src"]

    async with txm.tenant(tenant) as tx:
        await STORE.advance_cursor(tx, cid, "changed", at(1))
    counts = []
    for _ in range(4):
        async with txm.tenant(tenant) as tx:
            counts.append(await STORE.delete_connector_documents(tx, cid, 2))
    assert counts == [2, 2, 0, 0]
    async with txm.tenant(tenant) as tx:
        assert await STORE.delete_connector(tx, cid) is True
        assert await STORE.read_cursor(tx, cid, "changed") is None  # cascaded
        assert await STORE.delete_connector(tx, cid) is False


async def test_uploads_and_audit(
    api_engine: AsyncEngine, worker_engine: AsyncEngine, make_tenant: MakeTenant
) -> None:
    tenant, _, coll = make_tenant()
    other_tenant, _, other_coll = make_tenant()
    uid = uuid4()
    api = SqlTransactionManager(api_engine)
    async with api.tenant(tenant) as tx:
        await STORE.insert_upload(
            tx, upload_id=uid, collection_id=coll, blob_key=f"uploads/{uid}", filename="a.md",
            sha256=b"\x01" * 32, size_bytes=3,
        )  # fmt: skip
        await STORE.audit(tx, actor="user", action="upload.accepted", target=str(uid), details={})
    async with SqlTransactionManager(worker_engine).tenant(tenant) as tx:
        up = await STORE.load_upload(tx, uid)
    assert up is not None
    assert up.filename == "a.md"
    assert up.sha256 == b"\x01" * 32
    assert up.created_at.tzinfo is not None

    # Another tenant's collection id is indistinguishable from a missing one.
    with pytest.raises(NotFound):
        async with api.tenant(tenant) as tx:
            await STORE.insert_upload(
                tx, upload_id=uuid4(), collection_id=other_coll, blob_key=f"k/{uuid4()}",
                filename="b", sha256=b"\x02" * 32, size_bytes=1,
            )  # fmt: skip
    assert other_tenant != tenant


async def test_add_exclusion_is_idempotent(
    api_engine: AsyncEngine, worker_engine: AsyncEngine, make_tenant: MakeTenant
) -> None:
    tenant, _, _ = make_tenant()
    async with SqlTransactionManager(api_engine).tenant(tenant) as tx:
        await STORE.add_exclusion(tx, SourceKind.notion, "p1", "removed")
        await STORE.add_exclusion(tx, SourceKind.notion, "p1", "again")
    async with SqlTransactionManager(worker_engine).tenant(tenant) as tx:
        assert await STORE.excluded_ids(tx, SourceKind.notion, ["p1", "p2"]) == {"p1"}
        assert await STORE.excluded_ids(tx, SourceKind.file, ["p1"]) == set()
        assert await STORE.excluded_ids(tx, SourceKind.notion, []) == set()


async def test_exclusion_lock_serializes_against_a_sync_commit(
    api_engine: AsyncEngine, worker_engine: AsyncEngine, make_tenant: MakeTenant
) -> None:
    """The worker's fenced tx blocks on the lock while an exclusion is uncommitted, then sees
    it: a sync can never resurrect a page the user is excluding at that moment."""
    tenant, _, _ = make_tenant()
    api, worker = SqlTransactionManager(api_engine), SqlTransactionManager(worker_engine)
    locked = asyncio.Event()
    release = asyncio.Event()

    async def exclude() -> None:
        async with api.tenant(tenant) as tx:
            await STORE.add_exclusion(tx, SourceKind.notion, "race", None)
            locked.set()
            await release.wait()

    async def sync_commit() -> set[str]:
        await locked.wait()
        async with worker.tenant(tenant) as tx:
            await STORE.lock_source_item(tx, SourceKind.notion, "race")
            return await STORE.excluded_ids(tx, SourceKind.notion, ["race"])

    excluder = asyncio.create_task(exclude())
    checker = asyncio.create_task(sync_commit())
    await locked.wait()
    await asyncio.sleep(0.3)
    assert not checker.done()  # blocked on the advisory lock
    release.set()
    await excluder
    assert await checker == {"race"}


async def test_tenant_isolation(
    api_engine: AsyncEngine, worker_engine: AsyncEngine, make_tenant: MakeTenant,
    superuser: Conn,
) -> None:  # fmt: skip
    a, _, coll_a = make_tenant()
    b, _, _ = make_tenant()
    cid = await make_connector(api_engine, a, coll_a)
    seed_doc(superuser, a, coll_a, cid, "p1")
    api, worker = SqlTransactionManager(api_engine), SqlTransactionManager(worker_engine)
    uid = uuid4()
    async with api.tenant(a) as tx:
        await STORE.add_exclusion(tx, SourceKind.notion, "p1", None)
        await STORE.insert_upload(
            tx, upload_id=uid, collection_id=coll_a, blob_key=f"u/{uid}", filename="f",
            sha256=b"\x03" * 32, size_bytes=1,
        )  # fmt: skip
    async with worker.tenant(a) as tx:
        await STORE.advance_cursor(tx, cid, "changed", at(5))

    async with worker.tenant(b) as tx:
        assert await STORE.load_connector(tx, cid) is None
        assert await STORE.read_cursor(tx, cid, "changed") is None
        assert await STORE.excluded_ids(tx, SourceKind.notion, ["p1"]) == set()
        assert await STORE.load_upload(tx, uid) is None
        assert await STORE.fetched_at_by_external_id(tx, cid, ["p1"]) == {}
        assert await STORE.live_documents(tx, cid) == []
        assert await STORE.delete_connector_documents(tx, cid, 100) == 0
        await STORE.destroy_secret(tx, cid)
        await STORE.mark_connector_error(tx, cid, "x")
        assert await STORE.delete_connector(tx, cid) is False
    with pytest.raises(Exception, match="row-level security"):
        async with worker.tenant(b) as tx:
            await STORE.advance_cursor(tx, cid, "changed", at(50))

    async with worker.tenant(a) as tx:  # tenant A is untouched
        conn = await STORE.load_connector(tx, cid)
        assert conn is not None
        assert conn.status == "active"
        assert conn.sealed is not None
        assert await STORE.read_cursor(tx, cid, "changed") == at(5)
        assert len(await STORE.live_documents(tx, cid)) == 1


async def test_no_tenant_context_fails_closed(worker_engine: AsyncEngine) -> None:
    async with SqlTransactionManager(worker_engine).system() as tx:
        assert await STORE.excluded_ids(tx, SourceKind.notion, ["p1"]) == set()
        assert await STORE.load_connector(tx, uuid4()) is None
