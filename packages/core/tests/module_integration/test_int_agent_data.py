"""Phase 2 integration: the SQL adapters behind the agent's module-local ports
(adapters/db/agent_data.py), on the real migrated schema, as rf_api under RLS."""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import Callable
from uuid import UUID, uuid4

import psycopg
import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from ragforge_core import CODE_REQUIRED_SCHEMA
from ragforge_core.adapters.connectors.store import SqlSyncStore
from ragforge_core.adapters.db.agent_data import SqlConversationStore, SqlKnowledgeBase
from ragforge_core.adapters.db.tx import connection
from ragforge_core.agent.ports import ConversationStore, KnowledgeBase
from ragforge_core.db.engine import SqlTransactionManager
from ragforge_core.domain.models import ApprovalChoice, SourceKind, ToolCall

pytestmark = pytest.mark.integration
Conn = psycopg.Connection[tuple[object, ...]]
MakeTenant = Callable[..., tuple[UUID, UUID, UUID]]

STORE = SqlConversationStore()
KB = SqlKnowledgeBase()


@pytest.fixture
def txm(api_engine: AsyncEngine) -> SqlTransactionManager:
    return SqlTransactionManager(api_engine)


def _sha(s: str) -> bytes:
    return hashlib.sha256(s.encode()).digest()


def _upload_doc(su: Conn, tenant: UUID, coll: UUID, title: str, n_chunks: int) -> UUID:
    upload, doc = uuid4(), uuid4()
    su.execute(
        "INSERT INTO rf.uploads (id, tenant_id, collection_id, blob_key, filename, sha256, "
        "size_bytes) VALUES (%s, %s, %s, %s, %s, %s, 1)",
        (upload, tenant, coll, f"k/{upload}", f"{title}.md", _sha(title)),
    )
    su.execute(
        "INSERT INTO rf.documents (id, tenant_id, collection_id, upload_id, source_kind, "
        "external_id, title) VALUES (%s, %s, %s, %s, 'file', %s, %s)",
        (doc, tenant, coll, upload, str(upload), title),
    )
    for i in reversed(range(n_chunks)):  # inserted out of order on purpose
        su.execute(
            "INSERT INTO rf.chunks (id, tenant_id, collection_id, document_id, anchor, ordinal, "
            "display_text, display_sha256, embed_sha256, token_count) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, 1)",
            (uuid4(), tenant, coll, doc, f"a{i}", i, f"chunk {i}", _sha(f"{doc}{i}"), _sha(str(i))),
        )
    return doc


def _connector(su: Conn, tenant: UUID, coll: UUID) -> UUID:
    cid = uuid4()
    su.execute(
        "INSERT INTO rf.connectors (id, tenant_id, kind, target_collection_id, display_name, "
        "config, secret_ciphertext, wrapped_dek) VALUES (%s, %s, 'notion', %s, 'N', "
        "'{\"integration_id\": \"secret-ish\"}', '\\x01', '\\x02')",
        (cid, tenant, coll),
    )
    return cid


# ---------------------------------------------------------------- protocols
def test_adapters_satisfy_the_agent_ports() -> None:
    assert isinstance(STORE, ConversationStore)
    assert isinstance(KB, KnowledgeBase)


def test_schema_is_at_head_with_slot_0007(superuser: Conn) -> None:
    (version,) = superuser.execute("SELECT schema_version FROM rf.schema_meta").fetchone() or (0,)
    assert version == CODE_REQUIRED_SCHEMA
    cols = superuser.execute(
        "SELECT 1 FROM information_schema.columns WHERE table_schema = 'rf' "
        "AND table_name = 'pending_approvals' AND column_name = 'executed_at'"
    ).fetchall()
    assert cols


# ---------------------------------------------------------------- conversations
async def test_create_get_round_trip_and_tenant_isolation(
    txm: SqlTransactionManager, make_tenant: MakeTenant
) -> None:
    tenant, user, coll = make_tenant()
    other, _, _ = make_tenant()
    async with txm.tenant(tenant) as tx:
        cid = await STORE.create(tx, user, [coll])
        assert await STORE.get(tx, cid) == (user, (coll,))
    async with txm.tenant(other) as tx:
        assert await STORE.get(tx, cid) is None
        assert await STORE.get(tx, uuid4()) is None


async def _pending(
    txm: SqlTransactionManager, make_tenant: MakeTenant, n: int = 2
) -> tuple[UUID, UUID, UUID, list[ToolCall]]:
    tenant, user, coll = make_tenant()
    run = uuid4()
    calls = [ToolCall(id=uuid4().hex, name="remove_from_index", arguments={"q": f"x{i}"})
             for i in range(n)]  # fmt: skip
    async with txm.tenant(tenant) as tx:
        cid = await STORE.create(tx, user, [coll])
        await STORE.record_pending(tx, cid, run, calls)
        await STORE.record_pending(tx, cid, uuid4(), calls)  # re-run of the node: no-op
    return tenant, cid, run, calls


async def test_record_pending_is_idempotent_and_found_by_run(
    txm: SqlTransactionManager, make_tenant: MakeTenant, superuser: Conn
) -> None:
    tenant, cid, run, _ = await _pending(txm, make_tenant)
    rows = superuser.execute(
        "SELECT run_id, status FROM rf.pending_approvals WHERE conversation_id = %s", (cid,)
    ).fetchall()
    assert rows == [(run, "pending"), (run, "pending")]
    async with txm.tenant(tenant) as tx:
        assert await STORE.conversation_for_run(tx, run) == cid
        assert await STORE.conversation_for_run(tx, uuid4()) is None
    other, _, _ = make_tenant()
    async with txm.tenant(other) as tx:
        assert await STORE.conversation_for_run(tx, run) is None


async def test_decide_pending_audits_identifiers_only_and_redeciding_is_a_noop(
    txm: SqlTransactionManager, make_tenant: MakeTenant, superuser: Conn
) -> None:
    tenant, cid, run, calls = await _pending(txm, make_tenant)
    decisions = {calls[0].id: ApprovalChoice.approve, calls[1].id: ApprovalChoice.reject}
    async with txm.tenant(tenant) as tx:
        await STORE.decide_pending(tx, run, decisions)
    async with txm.tenant(tenant) as tx:
        await STORE.decide_pending(tx, run, {calls[0].id: ApprovalChoice.reject})
    rows = dict(
        (r[0], r[1])
        for r in superuser.execute(
            "SELECT tool_call_id, status FROM rf.pending_approvals "
            "WHERE conversation_id = %s AND decided_at IS NOT NULL",
            (cid,),
        ).fetchall()
    )
    assert rows == {calls[0].id: "approved", calls[1].id: "rejected"}
    audits = superuser.execute(
        "SELECT action, details FROM rf.audit_log WHERE tenant_id = %s AND action LIKE "
        "'approval.%%' ORDER BY id",
        (tenant,),
    ).fetchall()
    assert [a for a, _ in audits] == ["approval.approved", "approval.rejected"]
    for _, details in audits:
        dumped = json.dumps(details)
        assert '"args"' not in dumped
        assert "x0" not in dumped


async def test_claim_execution_is_exactly_once_and_rolls_back_with_the_tx(
    txm: SqlTransactionManager, make_tenant: MakeTenant
) -> None:
    tenant, cid, run, calls = await _pending(txm, make_tenant, n=3)
    approved, rejected, pending = (c.id for c in calls)
    async with txm.tenant(tenant) as tx:
        await STORE.decide_pending(
            tx, run, {approved: ApprovalChoice.approve, rejected: ApprovalChoice.reject}
        )
    async with txm.tenant(tenant) as tx:
        assert not await STORE.claim_execution(tx, cid, pending)
        assert not await STORE.claim_execution(tx, cid, rejected)

    class Boom(Exception):
        pass

    async def claim_then_fail() -> None:
        async with txm.tenant(tenant) as tx:
            assert await STORE.claim_execution(tx, cid, approved)
            raise Boom

    with pytest.raises(Boom):
        await claim_then_fail()
    async with txm.tenant(tenant) as tx:
        assert await STORE.claim_execution(tx, cid, approved)  # the rollback released it
    async with txm.tenant(tenant) as tx:
        assert not await STORE.claim_execution(tx, cid, approved)


async def test_summary(
    txm: SqlTransactionManager, make_tenant: MakeTenant, superuser: Conn
) -> None:
    tenant, user, coll = make_tenant()
    async with txm.tenant(tenant) as tx:
        cid = await STORE.create(tx, user, [coll])
        assert await STORE.summary(tx, cid) is None
    superuser.execute(
        "INSERT INTO rf.conversation_summaries (conversation_id, tenant_id, upto_message_id, "
        "summary) VALUES (%s, %s, 'm1', 'so far')",
        (cid, tenant),
    )
    async with txm.tenant(tenant) as tx:
        assert await STORE.summary(tx, cid) == "so far"


# ---------------------------------------------------------------- knowledge base
async def test_collection_taus_including_null(
    txm: SqlTransactionManager, make_tenant: MakeTenant, superuser: Conn
) -> None:
    tenant, _, coll = make_tenant()
    tuned = uuid4()
    superuser.execute(
        "INSERT INTO rf.collections (id, tenant_id, name, active_space_id, tau) "
        "VALUES (%s, %s, 'tuned', 1, 0.5)",
        (tuned, tenant),
    )
    _, _, foreign = make_tenant()
    async with txm.tenant(tenant) as tx:
        taus = await KB.collection_taus(tx, [coll, tuned, foreign])
    assert taus == {coll: None, tuned: pytest.approx(0.5)}


async def test_get_page_order_scope_tombstones_and_limit(
    txm: SqlTransactionManager, make_tenant: MakeTenant, superuser: Conn
) -> None:
    tenant, _, coll = make_tenant()
    doc = _upload_doc(superuser, tenant, coll, "page", 4)
    gone = _upload_doc(superuser, tenant, coll, "gone", 1)
    superuser.execute("UPDATE rf.documents SET deleted_at = now() WHERE id = %s", (gone,))
    async with txm.tenant(tenant) as tx:
        chunks = await KB.get_page(tx, doc, [coll], 3)
        assert [c.anchor for c in chunks] == ["a0", "a1", "a2"]
        assert all(c.rerank_score is None and c.fused_score == 0 for c in chunks)
        assert await KB.get_page(tx, gone, [coll], 10) == []
        assert await KB.get_page(tx, doc, [uuid4()], 10) == []
    other, _, _ = make_tenant()
    async with txm.tenant(other) as tx:
        assert await KB.get_page(tx, doc, [coll], 10) == []


async def test_sources_and_status_are_json_safe_secret_free_and_isolated(
    txm: SqlTransactionManager, make_tenant: MakeTenant, superuser: Conn
) -> None:
    tenant, _, coll = make_tenant()
    connector = _connector(superuser, tenant, coll)
    _upload_doc(superuser, tenant, coll, "notes", 1)
    async with txm.tenant(tenant) as tx:
        sources = await KB.list_sources(tx, [coll])
        status = await KB.sync_status(tx, [coll])
    dumped = json.dumps([sources, status])
    for forbidden in ("config", "secret", "wrapped_dek", "integration_id"):
        assert forbidden not in dumped
    assert {s["type"] for s in sources} == {"connector", "upload"}
    assert [s["id"] for s in status] == [str(connector)]
    assert set(status[0]) == {
        "id", "display_name", "status", "last_error", "poll_failures", "next_poll_at"
    }  # fmt: skip
    assert status[0]["poll_failures"] == 0
    other, _, _ = make_tenant()
    async with txm.tenant(other) as tx:
        assert await KB.list_sources(tx, [coll]) == []
        assert await KB.sync_status(tx, [coll]) == []


async def test_remove_from_index_excludes_tombstones_and_audits(
    txm: SqlTransactionManager, make_tenant: MakeTenant, superuser: Conn
) -> None:
    tenant, _, coll = make_tenant()
    doc = _upload_doc(superuser, tenant, coll, "victim", 1)
    other, _, _ = make_tenant()
    async with txm.tenant(other) as tx:
        assert not await KB.remove_from_index(tx, doc, [coll], "user:x")
    async with txm.tenant(tenant) as tx:
        assert not await KB.remove_from_index(tx, uuid4(), [coll], "user:x")
        assert not await KB.remove_from_index(tx, doc, [uuid4()], "user:x")
        assert await KB.remove_from_index(tx, doc, [coll], "user:x")
        locks = await connection(tx).execute(
            text(
                "SELECT count(*) FROM pg_locks WHERE locktype = 'advisory' "
                "AND pid = pg_backend_pid()"
            )
        )
        assert locks.scalar_one() >= 1  # the source-item lock, held to commit
    async with txm.tenant(tenant) as tx:
        assert not await KB.remove_from_index(tx, doc, [coll], "user:x")  # already deleted
    (deleted,) = superuser.execute(
        "SELECT deleted_at IS NOT NULL FROM rf.documents WHERE id = %s", (doc,)
    ).fetchone() or (False,)
    assert deleted
    (excl,) = superuser.execute(
        "SELECT count(*) FROM rf.exclusions WHERE tenant_id = %s AND source_kind = 'file'",
        (tenant,),
    ).fetchone() or (0,)
    assert excl == 1
    audits = superuser.execute(
        "SELECT actor, target FROM rf.audit_log WHERE tenant_id = %s AND "
        "action = 'document.removed'",
        (tenant,),
    ).fetchall()
    assert audits == [("user:x", str(doc))]


async def test_removal_racing_a_sync_commit_waits_on_the_source_lock_instead_of_deadlocking(
    txm: SqlTransactionManager, make_tenant: MakeTenant, superuser: Conn
) -> None:
    """Sync's fenced commit takes the source-item advisory lock, THEN the document row
    (services/sync/indexing.py -> adapters/db/index.py). A removal that row-locked first and
    then asked for the advisory lock deadlocked against it (Phase 2a review, P1-2)."""
    tenant, _, coll = make_tenant()
    doc = _upload_doc(superuser, tenant, coll, "raced", 1)
    (external_id,) = superuser.execute(
        "SELECT external_id FROM rf.documents WHERE id = %s", (doc,)
    ).fetchone() or ("",)
    pids: list[int] = []

    async def remove() -> bool:
        async with txm.tenant(tenant) as tx:
            pid = await connection(tx).execute(text("SELECT pg_backend_pid()"))
            pids.append(int(pid.scalar_one()))
            return await KB.remove_from_index(tx, doc, [coll], "user:x")

    async with txm.tenant(tenant) as sync_tx:  # plays the sync page commit
        await SqlSyncStore().lock_source_item(sync_tx, SourceKind.file, str(external_id))
        removal = asyncio.create_task(remove())
        for _ in range(250):
            await asyncio.sleep(0.02)
            if pids and _count(
                superuser,
                "SELECT count(*) FROM pg_locks WHERE pid = %s AND locktype = 'advisory' "
                "AND NOT granted",
                pids[0],
            ):
                break
        else:
            pytest.fail("the removal never waited on the source-item lock")
        assert not removal.done()
        assert not _count(  # waiting on the advisory lock while holding NO document row lock
            superuser,
            "SELECT count(*) FROM pg_locks WHERE pid = %s "
            "AND relation = 'rf.documents'::regclass AND mode = 'RowShareLock'",
            pids[0],
        )
        await connection(sync_tx).execute(  # upsert_document's row lock: deadlocked before
            text("SELECT 1 FROM rf.documents WHERE id = :id FOR UPDATE"), {"id": doc}
        )
    assert await asyncio.wait_for(removal, timeout=10) is True


def _count(su: Conn, sql: str, pid: int) -> int:
    row = su.execute(sql, (pid,)).fetchone()
    return int(str(row[0])) if row else 0


async def test_list_sources_hides_uploads_whose_document_was_deleted(
    txm: SqlTransactionManager, make_tenant: MakeTenant, superuser: Conn
) -> None:
    tenant, _, coll = make_tenant()
    kept = _upload_doc(superuser, tenant, coll, "kept", 1)
    gone = _upload_doc(superuser, tenant, coll, "gone", 1)
    del kept
    async with txm.tenant(tenant) as tx:
        assert await KB.remove_from_index(tx, gone, [coll], "user:x")
    async with txm.tenant(tenant) as tx:
        names = [row.get("filename") for row in await KB.list_sources(tx, [coll])]
    assert names == ["kept.md"]

    # 7 days later purge_tombstones hard-deletes the document but not the upload row: the
    # exclusion still keeps the deleted file out of the model's sources (review round 5).
    superuser.execute("DELETE FROM rf.documents WHERE id = %s", (gone,))
    async with txm.tenant(tenant) as tx:
        names = [row.get("filename") for row in await KB.list_sources(tx, [coll])]
    assert names == ["kept.md"]
