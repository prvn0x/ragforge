"""DocumentIndex + gc_embeddings + the SQL search backend, as rf_worker / rf_api under RLS.

Proves: the monotonic upsert in BOTH commit orders, sequential and truly concurrent (row lock
waits observed in pg_stat_activity); touches + ON CONFLICT DO NOTHING + stats upkeep; chunk
replacement with domain ids; tombstones; GC never deletes a touched or referenced hash, including
when it races a sync in either order; retired-space purge; cross-tenant isolation; the full
retriever over real SQL.
"""

from __future__ import annotations

import asyncio
import hashlib
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

import psycopg
import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from module_b.b_support import (
    DEFAULT_DIM,
    add_chunk,
    add_document,
    add_embedding,
    add_space,
    build_index_as_owner,
    near,
    set_stats,
    sha,
    unit_vector,
    use_space,
)
from ragforge_core.adapters.db.index import PgDocumentIndex
from ragforge_core.adapters.db.search import PgSearchBackend, drop_retired_space_index
from ragforge_core.db.engine import system_transaction, tenant_transaction
from ragforge_core.domain import ids
from ragforge_core.domain.errors import NotFound
from ragforge_core.domain.models import Chunk, DocumentRecord, SourceKind, UpsertOutcome
from ragforge_core.domain.ports import DocumentIndex
from ragforge_core.rag.retriever import HybridRetriever

pytestmark = pytest.mark.integration
Conn = psycopg.Connection[tuple[object, ...]]
MakeTenant = Callable[..., tuple[UUID, UUID, UUID]]
INDEX = PgDocumentIndex()
T0 = datetime(2026, 10, 1, 12, 0, tzinfo=UTC)


def connector(superuser: Conn, tenant: UUID, coll: UUID) -> UUID:
    row = superuser.execute(
        "INSERT INTO rf.connectors (tenant_id, kind, target_collection_id, display_name) "
        "VALUES (%s, 'notion', %s, 'n') RETURNING id",
        (tenant, coll),
    ).fetchone()
    assert row is not None
    return UUID(str(row[0]))


def record(
    tenant: UUID, coll: UUID, conn_id: UUID, *, fetched: datetime, content: str = "v1",
    external_id: str = "page-1", title: str = "Title", links: tuple[str, ...] = (),
) -> DocumentRecord:  # fmt: skip
    return DocumentRecord(
        id=ids.document_id(tenant, coll, conn_id, external_id),
        collection_id=coll,
        connector_id=conn_id,
        upload_id=None,
        source_kind=SourceKind.notion,
        external_id=external_id,
        external_url=f"https://notion.so/{external_id}",
        title=title,
        external_updated_at=fetched,
        fetched_at=fetched,
        page_sha256=sha(content),
        links_to=links,
    )


def chunk(ordinal: int, body: str, title: str = "Title") -> Chunk:
    embedded = f"{title} › H\n\n{body}"
    return Chunk(
        anchor=f"blk-{ordinal}", ordinal=ordinal, heading_path=("H",), display_text=body,
        embedded_text=embedded, display_sha256=sha(body), embed_sha256=sha(embedded),
        token_count=len(body.split()),
    )  # fmt: skip


def stored(superuser: Conn, doc: UUID) -> tuple[object, ...]:
    row = superuser.execute(
        "SELECT fetched_at, page_sha256, title, deleted_at FROM rf.documents WHERE id = %s", (doc,)
    ).fetchone()
    assert row is not None
    return row


async def wait_for_lock_waiter(superuser: Conn, timeout_s: float = 10.0) -> None:
    """Block until some rf_worker backend is waiting on a lock (proves real contention)."""
    deadline = asyncio.get_running_loop().time() + timeout_s
    while asyncio.get_running_loop().time() < deadline:
        row = superuser.execute(
            "SELECT count(*) FROM pg_stat_activity "
            "WHERE usename = 'rf_worker' AND wait_event_type = 'Lock'"
        ).fetchone()
        if row and int(str(row[0])) > 0:
            return
        await asyncio.sleep(0.02)
    raise AssertionError("expected a transaction to block on a row lock")


def test_implements_the_port() -> None:
    assert isinstance(INDEX, DocumentIndex)


# ---------------------------------------------------------------------- upsert_document
async def test_upsert_outcomes(
    b_worker_engine: AsyncEngine, superuser: Conn, make_tenant: MakeTenant
) -> None:
    tenant, _, coll = make_tenant()
    cid = connector(superuser, tenant, coll)
    first = record(tenant, coll, cid, fetched=T0, links=("b", "a", "a"))

    async def upsert(rec: DocumentRecord) -> UpsertOutcome:
        async with tenant_transaction(b_worker_engine, tenant) as tx:
            return await INDEX.upsert_document(tx, rec)

    assert await upsert(first) is UpsertOutcome.inserted
    assert await upsert(first) is UpsertOutcome.stale  # equal fetched_at is stale
    newer_same = record(tenant, coll, cid, fetched=T0 + timedelta(minutes=1), title="Renamed")
    assert await upsert(newer_same) is UpsertOutcome.unchanged
    fetched_at, _, title, _ = stored(superuser, first.id)
    assert fetched_at == T0 + timedelta(minutes=1)
    assert title == "Title"
    newer_changed = record(tenant, coll, cid, fetched=T0 + timedelta(minutes=2), content="v2",
                           title="New", links=("c",))  # fmt: skip
    assert await upsert(newer_changed) is UpsertOutcome.updated
    _, page_sha, title, _ = stored(superuser, first.id)
    assert page_sha == sha("v2")
    assert title == "New"
    links = superuser.execute(
        "SELECT to_external_id FROM rf.document_links WHERE from_document_id = %s", (first.id,)
    ).fetchall()
    assert links == [("c",)]
    assert await upsert(record(tenant, coll, cid, fetched=T0)) is UpsertOutcome.stale


async def test_null_fetched_at_is_treated_as_minus_infinity(
    b_worker_engine: AsyncEngine, superuser: Conn, make_tenant: MakeTenant
) -> None:
    tenant, _, coll = make_tenant()
    cid = connector(superuser, tenant, coll)
    rec = record(tenant, coll, cid, fetched=T0)
    superuser.execute(
        "INSERT INTO rf.documents (id, tenant_id, collection_id, connector_id, source_kind, "
        "external_id, title) VALUES (%s, %s, %s, %s, 'notion', 'page-1', 'old')",
        (rec.id, tenant, coll, cid),
    )
    async with tenant_transaction(b_worker_engine, tenant) as tx:
        assert await INDEX.upsert_document(tx, rec) is UpsertOutcome.updated


async def test_record_id_must_come_from_domain_ids(
    b_worker_engine: AsyncEngine, superuser: Conn, make_tenant: MakeTenant
) -> None:
    tenant, _, coll = make_tenant()
    cid = connector(superuser, tenant, coll)
    forged = DocumentRecord(
        id=uuid4(), collection_id=coll, connector_id=cid, upload_id=None,
        source_kind=SourceKind.notion, external_id="x", external_url=None, title="t",
        external_updated_at=None, fetched_at=T0, page_sha256=sha("x"),
    )  # fmt: skip
    async with tenant_transaction(b_worker_engine, tenant) as tx:
        with pytest.raises(ValueError, match=r"domain\.ids"):
            await INDEX.upsert_document(tx, forged)


@pytest.mark.parametrize("newer_commits_first", [True, False])
async def test_monotonic_upsert_in_both_commit_orders(
    newer_commits_first: bool,
    b_worker_engine: AsyncEngine,
    superuser: Conn,
    make_tenant: MakeTenant,
) -> None:
    tenant, _, coll = make_tenant()
    cid = connector(superuser, tenant, coll)
    older = record(tenant, coll, cid, fetched=T0, content="old")
    newer = record(tenant, coll, cid, fetched=T0 + timedelta(seconds=30), content="new")
    order = [newer, older] if newer_commits_first else [older, newer]
    outcomes = []
    for rec in order:
        async with tenant_transaction(b_worker_engine, tenant) as tx:
            outcomes.append(await INDEX.upsert_document(tx, rec))
    expected = (
        [UpsertOutcome.inserted, UpsertOutcome.stale]
        if newer_commits_first
        else [UpsertOutcome.inserted, UpsertOutcome.updated]
    )
    assert outcomes == expected
    assert stored(superuser, newer.id)[:2] == (newer.fetched_at, sha("new"))


@pytest.mark.parametrize("existing", [False, True], ids=["first-insert", "existing-row"])
@pytest.mark.parametrize(
    "newer_first", [True, False], ids=["newer-locks-first", "older-locks-first"]
)
async def test_concurrent_upserts_serialise_and_newest_wins(
    existing: bool,
    newer_first: bool,
    superuser: Conn,
    make_tenant: MakeTenant,
    b_worker_engine: AsyncEngine,
) -> None:
    """Two sync_page runs for one page, truly concurrent: the second blocks on the first's row
    lock (or its uncommitted insert), then the guard decides after the first commits."""
    tenant, _, coll = make_tenant()
    cid = connector(superuser, tenant, coll)
    older = record(tenant, coll, cid, fetched=T0 + timedelta(seconds=10), content="old")
    newer = record(tenant, coll, cid, fetched=T0 + timedelta(seconds=20), content="new")
    if existing:
        async with tenant_transaction(b_worker_engine, tenant) as tx:
            await INDEX.upsert_document(tx, record(tenant, coll, cid, fetched=T0, content="seed"))
    first, second = (newer, older) if newer_first else (older, newer)

    first_done = asyncio.Event()
    release_first = asyncio.Event()

    async def run_first() -> UpsertOutcome:
        async with tenant_transaction(b_worker_engine, tenant) as tx:
            outcome = await INDEX.upsert_document(tx, first)
            first_done.set()
            await release_first.wait()  # hold the lock until the second is provably blocked
            return outcome

    async def run_second() -> UpsertOutcome:
        await first_done.wait()
        async with tenant_transaction(b_worker_engine, tenant) as tx:
            return await INDEX.upsert_document(tx, second)

    task_first = asyncio.create_task(run_first())
    task_second = asyncio.create_task(run_second())
    await first_done.wait()
    await wait_for_lock_waiter(superuser)
    assert not task_second.done()
    release_first.set()
    out_first, out_second = await asyncio.gather(task_first, task_second)

    assert out_first is (UpsertOutcome.updated if existing else UpsertOutcome.inserted)
    assert out_second is (UpsertOutcome.stale if newer_first else UpsertOutcome.updated)
    assert stored(superuser, newer.id)[:2] == (newer.fetched_at, sha("new"))


# ---------------------------------------------------------------------- embeddings + chunks
async def test_touch_find_missing_insert_and_stats(
    b_worker_engine: AsyncEngine, superuser: Conn, make_tenant: MakeTenant
) -> None:
    tenant, _, coll = make_tenant()
    building = add_space(superuser, 4, status="building")
    superuser.execute(
        "UPDATE rf.collections SET building_space_id = %s WHERE id = %s", (building, coll)
    )
    present, missing_a, missing_b = sha("present"), sha("a"), sha("b")
    add_embedding(superuser, tenant, coll, present, unit_vector(1), touched_hours_ago=None)

    async with tenant_transaction(b_worker_engine, tenant) as tx:
        assert list(await INDEX.target_spaces(tx, coll)) == [1, building]
        missing = await INDEX.touch_and_find_missing(
            tx, coll, 1, [present, missing_a, missing_b, missing_a]
        )
    assert missing == {missing_a, missing_b}
    touches = superuser.execute(
        "SELECT count(*) FROM rf.embedding_touches WHERE collection_id = %s "
        "AND touched_at > now() - interval '1 minute'",
        (coll,),
    ).fetchone()
    assert touches == (3,), "every referenced hash is touched, present or not"

    vectors = {missing_a: unit_vector("a"), missing_b: unit_vector("b"), present: unit_vector(9)}
    async with tenant_transaction(b_worker_engine, tenant) as tx:
        assert await INDEX.insert_embeddings(tx, coll, 1, vectors) == 2  # present: DO NOTHING
    async with tenant_transaction(b_worker_engine, tenant) as tx:
        assert await INDEX.insert_embeddings(tx, coll, 1, vectors) == 0
        assert await INDEX.touch_and_find_missing(tx, coll, 1, [missing_a]) == set()
    stats = superuser.execute(
        "SELECT vector_count FROM rf.collection_space_stats "
        "WHERE collection_id = %s AND space_id = 1",
        (coll,),
    ).fetchone()
    assert stats == (2,), "stats count only rows actually inserted"

    async with tenant_transaction(b_worker_engine, tenant) as tx:
        with pytest.raises(ValueError, match="wrong number of dimensions") as wrong_dim:
            await INDEX.insert_embeddings(tx, coll, building, {sha("x"): [0.1, 0.2]})
        # The message reaches jobs.last_error: no space id or sizes in it (logged instead).
        assert str(building) not in str(wrong_dim.value)
        assert "4" not in str(wrong_dim.value)
        with pytest.raises(ValueError, match="32-byte"):
            await INDEX.touch_and_find_missing(tx, coll, 1, [b"short"])


async def test_replace_chunks_upserts_trims_and_skips_identical_rows(
    b_worker_engine: AsyncEngine, superuser: Conn, make_tenant: MakeTenant
) -> None:
    tenant, _, coll = make_tenant()
    cid = connector(superuser, tenant, coll)
    rec = record(tenant, coll, cid, fetched=T0)
    async with tenant_transaction(b_worker_engine, tenant) as tx:
        await INDEX.upsert_document(tx, rec)
        await INDEX.replace_chunks(
            tx, rec.id, coll, [chunk(0, "alpha"), chunk(1, "beta"), chunk(2, "gamma")]
        )
    rows = superuser.execute(
        "SELECT id, ordinal, xmin::text, tsv IS NOT NULL FROM rf.chunks "
        "WHERE document_id = %s ORDER BY ordinal",
        (rec.id,),
    ).fetchall()
    assert [r[0] for r in rows] == [ids.chunk_id(rec.id, i) for i in range(3)]
    assert all(r[3] for r in rows)
    xmin_first = rows[0][2]

    async with tenant_transaction(b_worker_engine, tenant) as tx:
        await INDEX.replace_chunks(tx, rec.id, coll, [chunk(0, "alpha"), chunk(1, "BETA v2")])
    rows2 = superuser.execute(
        "SELECT ordinal, display_text, xmin::text FROM rf.chunks "
        "WHERE document_id = %s ORDER BY ordinal",
        (rec.id,),
    ).fetchall()
    assert [(r[0], r[1]) for r in rows2] == [(0, "alpha"), (1, "BETA v2")]
    assert rows2[0][2] == xmin_first, "identical chunk must not be rewritten (no churn)"

    async with tenant_transaction(b_worker_engine, tenant) as tx:
        with pytest.raises(ValueError, match=r"0\.\.n-1"):
            await INDEX.replace_chunks(tx, rec.id, coll, [chunk(1, "gap")])
    other_coll = uuid4()
    async with tenant_transaction(b_worker_engine, tenant) as tx:
        with pytest.raises(ValueError, match="different collection"):
            await INDEX.replace_chunks(tx, rec.id, other_coll, [])
        with pytest.raises(NotFound):
            await INDEX.replace_chunks(tx, uuid4(), coll, [])
    async with tenant_transaction(b_worker_engine, tenant) as tx:
        await INDEX.replace_chunks(tx, rec.id, coll, [])
    assert superuser.execute(
        "SELECT count(*) FROM rf.chunks WHERE document_id = %s", (rec.id,)
    ).fetchone() == (0,)


async def test_tombstone_exclusions_and_cross_tenant_isolation(
    b_worker_engine: AsyncEngine, superuser: Conn, make_tenant: MakeTenant
) -> None:
    tenant, _, coll = make_tenant()
    intruder, _, _ = make_tenant()
    cid = connector(superuser, tenant, coll)
    rec = record(tenant, coll, cid, fetched=T0)
    superuser.execute(
        "INSERT INTO rf.exclusions (tenant_id, source_kind, external_id) "
        "VALUES (%s, 'notion', 'secret')",
        (tenant,),
    )
    async with tenant_transaction(b_worker_engine, tenant) as tx:
        await INDEX.upsert_document(tx, rec)
        await INDEX.replace_chunks(tx, rec.id, coll, [chunk(0, "alpha")])
        assert await INDEX.is_excluded(tx, "notion", "secret") is True
        assert await INDEX.is_excluded(tx, "file", "secret") is False

    async with tenant_transaction(b_worker_engine, intruder) as tx:
        assert await INDEX.is_excluded(tx, "notion", "secret") is False
        await INDEX.tombstone(tx, rec.id)  # invisible under RLS: no effect
        with pytest.raises(NotFound):
            await INDEX.replace_chunks(tx, rec.id, coll, [])
        with pytest.raises(NotFound):
            await INDEX.target_spaces(tx, coll)
    assert stored(superuser, rec.id)[3] is None

    async with tenant_transaction(b_worker_engine, tenant) as tx:
        await INDEX.tombstone(tx, rec.id)
    assert stored(superuser, rec.id)[3] is not None
    async with tenant_transaction(b_worker_engine, tenant) as tx:  # a later fetch revives it
        newer = record(tenant, coll, cid, fetched=datetime.now(UTC) + timedelta(minutes=1))
        assert await INDEX.upsert_document(tx, newer) is UpsertOutcome.unchanged
    assert stored(superuser, rec.id)[3] is None


@pytest.mark.parametrize("content", ["v1", "v2"], ids=["unchanged", "updated"])
async def test_sync_fetched_before_a_tombstone_never_revives_it(
    content: str, b_worker_engine: AsyncEngine, superuser: Conn, make_tenant: MakeTenant
) -> None:
    """Review finding: a sync_page that fetched the page (T1) before reconcile tombstoned it
    (T2), but commits after (T3), used to clear deleted_at because its fetched_at was still
    newer than the stored one. The page must stay deleted and keep its tombstone time (the
    7-day purge clock); only a fetch that started after the tombstone may revive it."""
    tenant, _, coll = make_tenant()
    cid = connector(superuser, tenant, coll)
    async with tenant_transaction(b_worker_engine, tenant) as tx:
        await INDEX.upsert_document(tx, record(tenant, coll, cid, fetched=T0))
    in_flight = record(
        tenant, coll, cid, fetched=datetime.now(UTC) - timedelta(minutes=1), content=content
    )  # T1: fetched, still embedding
    async with tenant_transaction(b_worker_engine, tenant) as tx:  # T2: reconcile confirms
        await INDEX.tombstone(tx, in_flight.id)
    tombstoned_at = stored(superuser, in_flight.id)[3]
    assert tombstoned_at is not None
    async with tenant_transaction(b_worker_engine, tenant) as tx:  # T3: the sync commits
        outcome = await INDEX.upsert_document(tx, in_flight)
    assert outcome is (UpsertOutcome.unchanged if content == "v1" else UpsertOutcome.updated)
    fetched_at, page_sha, _, deleted_at = stored(superuser, in_flight.id)
    assert (fetched_at, page_sha) == (in_flight.fetched_at, sha(content))  # still monotonic
    assert deleted_at == tombstoned_at, "an older fetch must not revive a tombstone"

    later = record(
        tenant, coll, cid, fetched=datetime.now(UTC) + timedelta(minutes=1), content=content
    )
    async with tenant_transaction(b_worker_engine, tenant) as tx:
        assert await INDEX.upsert_document(tx, later) is UpsertOutcome.unchanged
    assert stored(superuser, later.id)[3] is None, "a fetch after the tombstone revives it"


# ---------------------------------------------------------------------- gc_embeddings
async def gc(engine: AsyncEngine, batch: int = 5000) -> int:
    async with system_transaction(engine) as tx:
        return int(
            (await tx.execute(text("SELECT rf.gc_embeddings(:b)"), {"b": batch})).scalar_one()
        )


def exists(superuser: Conn, tenant: UUID, embed: bytes, space: int = 1) -> bool:
    row = superuser.execute(
        "SELECT count(*) FROM rf.embeddings "
        "WHERE tenant_id = %s AND embed_sha256 = %s AND space_id = %s",
        (tenant, embed, space),
    ).fetchone()
    return row == (1,)


async def test_gc_deletes_only_stale_unreferenced_embeddings_across_tenants(
    b_worker_engine: AsyncEngine, superuser: Conn, make_tenant: MakeTenant
) -> None:
    a, _, coll_a = make_tenant()
    b, _, coll_b = make_tenant()
    doc = add_document(superuser, a, coll_a)
    _, referenced = add_chunk(superuser, a, coll_a, doc, 0, "kept by a chunk")
    stale, recent, untouched = sha(f"stale{a}"), sha(f"recent{a}"), sha(f"untouched{a}")
    vec = unit_vector(3)
    add_embedding(superuser, a, coll_a, stale, vec, touched_hours_ago=48)
    add_embedding(superuser, a, coll_a, recent, vec, touched_hours_ago=23)
    add_embedding(superuser, a, coll_a, referenced, vec, touched_hours_ago=48)
    add_embedding(
        superuser, a, coll_a, untouched, vec, touched_hours_ago=None, created_hours_ago=72
    )
    # Tenant B holds the SAME hash as A's referenced chunk, unreferenced in B: only B's goes.
    add_embedding(superuser, b, coll_b, referenced, vec, touched_hours_ago=48)
    set_stats(superuser, a, coll_a, 1, 4)
    set_stats(superuser, b, coll_b, 1, 1)

    processed = await gc(b_worker_engine)

    assert processed >= 3  # 2 deleted + 1 re-armed (other tests' rows may add to it)
    assert not exists(superuser, a, stale)
    assert not exists(superuser, b, referenced)
    assert exists(superuser, a, recent)
    assert exists(superuser, a, referenced)
    assert exists(superuser, a, untouched), "no touch row: never deleted (see gc_embeddings)"
    counts: dict[object, object] = {
        r[0]: r[1]
        for r in superuser.execute(
            "SELECT collection_id, vector_count FROM rf.collection_space_stats "
            "WHERE collection_id = ANY(%s) AND space_id = 1",
            ([coll_a, coll_b],),
        ).fetchall()
    }
    assert counts == {coll_a: 3, coll_b: 0}
    rearmed = superuser.execute(
        "SELECT touched_at > now() - interval '1 minute' FROM rf.embedding_touches "
        "WHERE tenant_id = %s AND embed_sha256 = %s",
        (a, referenced),
    ).fetchone()
    assert rearmed == (True,), "a still-referenced stale touch is re-armed, not re-walked"


async def test_gc_rearms_live_touches_so_batches_never_rewalk_them(
    b_worker_engine: AsyncEngine, superuser: Conn, make_tenant: MakeTenant
) -> None:
    """Review finding: every call used to anti-join ALL old-but-referenced touches before it
    found p_batch victims, so a worker looping on full batches did O(live touches) work per
    batch. Now each stale touch is visited once: deleted, or re-armed out of the stale window,
    so the loop-until-0 contract terminates and later batches only see new candidates."""
    tenant, _, coll = make_tenant()
    doc = add_document(superuser, tenant, coll)
    for i in range(30):
        _, embed = add_chunk(superuser, tenant, coll, doc, i, f"live {i}")
        add_embedding(superuser, tenant, coll, embed, unit_vector(i), touched_hours_ago=72)
    victim = sha(f"victim{tenant}")
    add_embedding(superuser, tenant, coll, victim, unit_vector(99), touched_hours_ago=48)

    for _ in range(200):  # the worker's loop, one committed transaction per call
        if await gc(b_worker_engine, batch=7) == 0:
            break
    else:
        raise AssertionError("gc_embeddings never reported an empty batch")

    stale = superuser.execute(
        "SELECT count(*) FROM rf.embedding_touches "
        "WHERE tenant_id = %s AND touched_at < now() - interval '24 hours'",
        (tenant,),
    ).fetchone()
    assert stale == (0,), "every stale touch was resolved; none is left to re-scan"
    assert not exists(superuser, tenant, victim)
    live = superuser.execute(
        "SELECT count(*) FROM rf.embeddings WHERE tenant_id = %s", (tenant,)
    ).fetchone()
    assert live == (30,)


async def test_gc_never_deletes_a_hash_a_sync_is_touching(
    b_worker_engine: AsyncEngine, superuser: Conn, make_tenant: MakeTenant
) -> None:
    """Sync touches first (uncommitted): GC must skip the locked touch and keep the vector."""
    tenant, _, coll = make_tenant()
    embed = sha(f"in-flight{tenant}")
    add_embedding(superuser, tenant, coll, embed, unit_vector(4), touched_hours_ago=48)
    async with tenant_transaction(b_worker_engine, tenant) as sync_tx:
        assert await INDEX.touch_and_find_missing(sync_tx, coll, 1, [embed]) == set()
        await gc(b_worker_engine)  # runs while the sync's touch is uncommitted
        assert exists(superuser, tenant, embed)
    await gc(b_worker_engine)  # after the commit the touch is fresh
    assert exists(superuser, tenant, embed)


async def test_gc_first_then_sync_sees_the_hash_missing_and_reembeds(
    b_worker_engine: AsyncEngine, superuser: Conn, make_tenant: MakeTenant
) -> None:
    """GC deletes first (uncommitted): the sync's touch blocks on GC's row lock, and once GC
    commits the sync reports the hash missing, so it will embed it again (no orphan chunk)."""
    tenant, _, coll = make_tenant()
    embed = sha(f"racing{tenant}")
    add_embedding(superuser, tenant, coll, embed, unit_vector(5), touched_hours_ago=48)
    gc_done = asyncio.Event()
    release_gc = asyncio.Event()

    async def run_gc() -> None:
        async with system_transaction(b_worker_engine) as tx:
            await tx.execute(text("SELECT rf.gc_embeddings()"))
            gc_done.set()
            await release_gc.wait()

    async def run_sync() -> set[bytes]:
        await gc_done.wait()
        async with tenant_transaction(b_worker_engine, tenant) as tx:
            return await INDEX.touch_and_find_missing(tx, coll, 1, [embed])

    gc_task = asyncio.create_task(run_gc())
    sync_task = asyncio.create_task(run_sync())
    await gc_done.wait()
    await wait_for_lock_waiter(superuser)
    release_gc.set()
    await gc_task
    assert await sync_task == {embed}
    assert not exists(superuser, tenant, embed)
    touched = superuser.execute(
        "SELECT touched_at > now() - interval '1 minute' FROM rf.embedding_touches "
        "WHERE tenant_id = %s AND embed_sha256 = %s",
        (tenant, embed),
    ).fetchone()
    assert touched == (True,), "the fresh touch protects the re-embedded vector"


async def test_gc_purges_retired_spaces_after_seven_days_only(
    b_worker_engine: AsyncEngine, superuser: Conn, make_tenant: MakeTenant, b_migrator: Conn
) -> None:
    tenant, _, coll = make_tenant()
    old, recent, in_use = (add_space(superuser, 4) for _ in range(3))
    build_index_as_owner(b_migrator, old)
    for space in (old, recent, in_use):
        for i in range(3):
            add_embedding(
                superuser, tenant, coll, sha(f"r{space}-{i}"), [0.1, 0.2, 0.3, 0.4],
                space_id=space,
            )  # fmt: skip
        set_stats(superuser, tenant, coll, space, 3)
    retire = "UPDATE rf.embedding_spaces SET status = 'retired', retired_at = now() - %s::interval "
    superuser.execute(retire + "WHERE id = %s", ("8 days", old))
    superuser.execute(retire + "WHERE id = %s", ("1 day", recent))
    superuser.execute(retire + "WHERE id = %s", ("30 days", in_use))
    superuser.execute(
        "UPDATE rf.collections SET building_space_id = %s WHERE id = %s", (in_use, coll)
    )

    for _ in range(200):  # batch=2: embeddings AND touches are purged in bounded steps
        if await gc(b_worker_engine, batch=2) == 0:
            break
    else:
        raise AssertionError("gc_embeddings never reported an empty batch")

    def left(table: str, space: int) -> object:
        row = superuser.execute(
            f"SELECT count(*) FROM rf.{table} WHERE space_id = %s",
            (space,),
        ).fetchone()
        assert row is not None
        return row[0]

    tables = ("embeddings", "embedding_touches", "collection_space_stats")
    assert [left(t, old) for t in tables] == [0, 0, 0]
    assert left("embeddings", recent) == 3
    assert left("embeddings", in_use) == 3
    index = f"rf.embeddings_hnsw_space_{old}"
    # GC never takes ACCESS EXCLUSIVE on rf.embeddings: the index outlives the purge ...
    assert superuser.execute("SELECT to_regclass(%s)", (index,)).fetchone() != (None,)
    # ... and the migrator step drops it CONCURRENTLY, only for retired spaces nobody uses.
    assert drop_retired_space_index(b_migrator, in_use) is False
    assert drop_retired_space_index(b_migrator, 1) is False  # the default, active space
    assert drop_retired_space_index(b_migrator, old) is True
    assert superuser.execute("SELECT to_regclass(%s)", (index,)).fetchone() == (None,)
    b_migrator.autocommit = False
    with pytest.raises(ValueError, match="autocommit"):
        drop_retired_space_index(b_migrator, old)


async def test_a_retired_space_keeps_its_rollback_window_then_is_purged(
    b_worker_engine: AsyncEngine, superuser: Conn, make_tenant: MakeTenant
) -> None:
    """Review finding: status = 'retired' with retired_at NULL was never purged. Slot 0007 makes
    that state impossible (CHECK), so a retired space always has the start of its 7-day window:
    GC keeps its vectors inside the window and purges them after it."""
    tenant, _, coll = make_tenant()
    space = add_space(superuser, 4)
    add_embedding(superuser, tenant, coll, sha(f"retired-{space}"), [0.1, 0.2, 0.3, 0.4],
                  space_id=space)  # fmt: skip
    with pytest.raises(psycopg.errors.CheckViolation):
        superuser.execute(
            "UPDATE rf.embedding_spaces SET status = 'retired', retired_at = NULL WHERE id = %s",
            (space,),
        )
    superuser.execute(
        "UPDATE rf.embedding_spaces SET status = 'retired', retired_at = now() WHERE id = %s",
        (space,),
    )

    def vectors() -> object:
        row = superuser.execute(
            "SELECT count(*) FROM rf.embeddings WHERE space_id = %s", (space,)
        ).fetchone()
        assert row is not None
        return row[0]

    while await gc(b_worker_engine) > 0:
        pass
    assert vectors() == 1, "inside the rollback window: kept"
    superuser.execute(
        "UPDATE rf.embedding_spaces SET retired_at = now() - interval '8 days' WHERE id = %s",
        (space,),
    )
    while await gc(b_worker_engine) > 0:
        pass
    assert vectors() == 0


# ---------------------------------------------------------------------- retriever end to end
async def test_collections_on_different_spaces_never_serve_leftover_vectors(
    api_engine: AsyncEngine, superuser: Conn, make_tenant: MakeTenant
) -> None:
    """Review finding (P1): `flipped` moved from space `old` to `new` and still holds its
    leftover (very near) vectors in `old`, which `stays` is served from. Searching every
    collection in every space served those leftovers and gave `flipped` a second RRF list.
    Now each space is searched over the collections it serves only: one list per collection."""
    tenant, _, stays = make_tenant()
    flipped = uuid4()
    old, new = add_space(superuser, 4), add_space(superuser, 4)
    use_space(superuser, stays, old)
    superuser.execute(
        "INSERT INTO rf.collections (id, tenant_id, name, active_space_id) "
        "VALUES (%s, %s, 'flipped', %s)",
        (flipped, tenant, new),
    )
    query = [1.0, 0.0, 0.0, 0.0]
    doc_s = add_document(superuser, tenant, stays, title="Stays")
    doc_f = add_document(superuser, tenant, flipped, title="Flipped")
    stays_chunk, e_s = add_chunk(superuser, tenant, stays, doc_s, 0, "stays body")
    flipped_chunk, e_f = add_chunk(superuser, tenant, flipped, doc_f, 0, "flipped body")
    add_embedding(superuser, tenant, stays, e_s, [0.6, 0.8, 0.0, 0.0], space_id=old)
    add_embedding(superuser, tenant, flipped, e_f, query, space_id=old)  # leftover, nearest
    add_embedding(superuser, tenant, flipped, e_f, [0.0, 1.0, 0.0, 0.0], space_id=new)
    set_stats(superuser, tenant, flipped, old, 1_000_000)  # stale stats must not count either

    retriever = HybridRetriever(PgSearchBackend())
    async with tenant_transaction(api_engine, tenant) as tx:
        old_hits = await PgSearchBackend().vector_search(tx, old, query, [stays, flipped], 10)
        hits = await retriever.retrieve(tx, "", {old: query, new: query}, [stays, flipped], k=10)
    assert [h.chunk_id for h in old_hits] == [stays_chunk], "leftover vector never served"
    assert {h.chunk_id for h in hits} == {stays_chunk, flipped_chunk}
    assert all(h.fused_score == pytest.approx(1 / 61) for h in hits), "one list each"


async def test_hybrid_retriever_over_sql_is_tenant_scoped_and_hydrated(
    api_engine: AsyncEngine, superuser: Conn, make_tenant: MakeTenant
) -> None:
    tenant, _, coll = make_tenant()
    intruder, _, intruder_coll = make_tenant()
    other_space = add_space(superuser, 4)
    building_only = add_space(superuser, 4, status="building")
    second_coll, third_coll = uuid4(), uuid4()
    for c, space in ((second_coll, other_space), (third_coll, building_only)):
        superuser.execute(
            "INSERT INTO rf.collections (id, tenant_id, name, active_space_id) "
            "VALUES (%s, %s, %s, %s)",
            (c, tenant, f"x-{c.hex[:6]}", space),
        )
    doc = add_document(superuser, tenant, coll, title="Runbook", url="https://notion.so/runbook")
    base = unit_vector("topic")
    texts = ["restart the api pods", "rotate the database password", "escalate to oncall"]
    chunk_ids = []
    for i, body in enumerate(texts):
        cid_, embed = add_chunk(
            superuser, tenant, coll, doc, i, body, heading_path=["Ops", f"S{i}"]
        )
        add_embedding(superuser, tenant, coll, embed, near(base, i, noise=0.2 * (i + 1)))
        chunk_ids.append(cid_)
    dup_doc = add_document(superuser, tenant, coll, title="Copy")
    _, dup_embed = add_chunk(superuser, tenant, coll, dup_doc, 0, texts[0])  # same display text
    add_embedding(superuser, tenant, coll, dup_embed, near(base, 99, noise=0.2))
    intruder_doc = add_document(superuser, intruder, intruder_coll)
    _, ie = add_chunk(
        superuser, intruder, intruder_coll, intruder_doc, 0, "rotate intruder secrets"
    )
    add_embedding(superuser, intruder, intruder_coll, ie, base)

    retriever = HybridRetriever(PgSearchBackend())
    async with tenant_transaction(api_engine, tenant) as tx:
        spaces = await retriever.spaces_for(tx, [coll, second_coll, third_coll, intruder_coll])
        assert list(spaces) == sorted([1, other_space])  # building-status space excluded
        hits = await retriever.retrieve(
            tx, "rotate password", {1: base, other_space: [0.5, 0.5, 0.5, 0.5]},
            [coll, second_coll, intruder_coll], k=10,
        )  # fmt: skip
    assert {h.chunk_id for h in hits} <= set(chunk_ids) | {ids.chunk_id(dup_doc, 0)}
    assert hits[0].chunk_id == chunk_ids[1], "in both the vector and the keyword list"
    assert len([h for h in hits if h.display_text == texts[0]]) == 1, "collapsed by display_sha"
    top = hits[0]
    assert top.title == "Runbook"
    assert top.external_url == "https://notion.so/runbook"
    assert top.heading_path == ("Ops", "S1")
    assert top.anchor == "a1"
    assert top.display_sha256 == hashlib.sha256(texts[1].encode()).digest()
    assert top.collection_id == coll
    assert top.document_id == doc
    assert all(h.collection_id != intruder_coll for h in hits)

    async with tenant_transaction(api_engine, intruder) as tx:
        assert await retriever.retrieve(tx, "restart", {1: base}, [coll], k=5) == []


async def test_vector_search_uses_dim_of_the_space(api_engine: AsyncEngine, superuser: Conn,
                                                   make_tenant: MakeTenant) -> None:  # fmt: skip
    tenant, _, coll = make_tenant()
    space = add_space(superuser, 4)
    use_space(superuser, coll, space)
    doc = add_document(superuser, tenant, coll)
    add_chunk(superuser, tenant, coll, doc, 0, "something to join against")
    async with tenant_transaction(api_engine, tenant) as tx:
        with pytest.raises(Exception, match="4 dimensions"):
            await PgSearchBackend().vector_search(tx, space, [0.1] * DEFAULT_DIM, [coll], 5)
