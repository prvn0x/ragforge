"""Sync handler guarantees, proven against fakes of the other modules' ports (no DB needed).

Each test names the design rule it proves (ADR-0007 / ADR-0010 / DESIGN section 3 Flow A).
"""

from __future__ import annotations

import hashlib
from uuid import UUID, uuid4

import pytest

from module_c2.c2_fakes import TOKEN, FakePage, Harness, UploadRec, make_harness, ts
from ragforge_core.domain.errors import LeaseLost, NotFound, UpstreamRateLimited
from ragforge_core.domain.ids import document_id
from ragforge_core.domain.models import (
    JobPriority,
    SealedSecret,
    SourceItemRef,
    SourceKind,
    UpsertOutcome,
)
from ragforge_core.services.sync.handlers import associated_data, run_sync_job
from ragforge_core.services.sync.policy import (
    CURSOR_SCOPE,
    KIND_DELETE_CONNECTOR,
    KIND_INGEST_FILE,
    KIND_RECONCILE,
    KIND_SYNC_CONNECTOR,
    KIND_SYNC_PAGE,
)


async def _sync_page(h: Harness, ext: str = "p1") -> None:
    lease = h.lease(KIND_SYNC_PAGE, {"connector_id": str(h.connector), "external_id": ext})
    await run_sync_job(lease, h.deps)


def _doc_id(h: Harness, ext: str) -> UUID:
    return document_id(h.tenant, h.collection, h.connector, ext)


# ================================================================= sync_page
async def test_sync_page_indexes_document_with_domain_ids_and_completes() -> None:
    h = make_harness({"p1": FakePage(ts(10), blocks=("alpha", "beta"))})
    await _sync_page(h)

    w = h.world
    doc = w.documents[document_id(h.tenant, h.collection, h.connector, "p1")]
    assert doc.source_kind is SourceKind.notion
    assert doc.fetched_at == ts(30)
    assert [c.display_text for c in w.chunks[doc.id]] == ["alpha", "beta"]
    assert len(w.completed) == 1
    assert w.failed == []
    assert [a[1] for a in w.audit] == ["document.synced"]
    assert h.factory.tokens == [TOKEN]  # token opened with the tenant/connector-bound AAD
    assert ("notion", "p1") in w.locks  # exclusion lock taken in the fenced tx


async def test_no_model_call_while_a_transaction_is_open_for_every_target_space() -> None:
    h = make_harness({"p1": FakePage(ts(10), blocks=("alpha", "beta"))})
    h.world.spaces = [1, 2]  # active + building (blue/green dual write)
    await _sync_page(h)

    for space in (1, 2):
        embedder = h.registry.by_space[space]
        assert embedder.calls, f"space {space} was not embedded"
    # FakeEmbedder asserts open_tx == 0 on every call; reaching here means it never fired.
    assert h.world.failed == []
    assert {s for (_, s, _) in h.world.embeddings} == {1, 2}


async def test_only_missing_hashes_are_embedded_and_all_hashes_are_touched() -> None:
    h = make_harness({"p1": FakePage(ts(10), blocks=("kept", "new"))})
    kept = hashlib.sha256("Page › \n\nkept".encode()).digest()
    h.world.embeddings.add((h.collection, 1, kept))
    await _sync_page(h)

    assert h.registry.by_space[1].calls == [["Page › \n\nnew"]]
    assert (h.collection, 1, kept) in h.world.touches  # protected from GC even if not embedded


async def test_stale_outcome_skips_chunk_replacement_and_audit() -> None:
    h = make_harness({"p1": FakePage(ts(10), blocks=("older content",), fetched_at=ts(20))})
    newer = h.seed_document("p1", ts(40))  # an equal-or-newer fetch already committed
    h.world.chunks[newer.id] = []
    await _sync_page(h)

    assert h.world.replace_calls == 0
    assert h.world.documents[newer.id].fetched_at == ts(40)
    assert h.world.audit == []
    assert len(h.world.completed) == 1  # completes as a no-op
    assert h.world.progress[-1]["outcome"] == UpsertOutcome.stale.value


async def test_unchanged_content_bumps_fetched_at_without_replacing_chunks() -> None:
    h = make_harness({"p1": FakePage(ts(10), blocks=("same",), fetched_at=ts(20))})
    await _sync_page(h)
    h.source.pages["p1"].fetched_at = ts(50)
    await _sync_page(h)

    assert h.world.replace_calls == 1
    assert h.world.documents[_doc_id(h, "p1")].fetched_at == ts(50)


async def test_lease_lost_during_embedding_commits_nothing_and_does_not_fail_job() -> None:
    h = make_harness({"p1": FakePage(ts(10), blocks=("alpha",))})

    def lose() -> None:
        h.world.lease_valid = False

    h.registry.for_space(1).on_embed = lose
    with pytest.raises(LeaseLost):  # re-raised: the runtime logs job_lease_lost
        await _sync_page(h)

    w = h.world
    assert w.documents == {}
    assert w.chunks == {}
    assert w.embeddings == set()
    assert w.completed == []
    assert w.audit == []
    assert w.progress == []
    assert w.failed == []  # another worker owns the job now: never call fail()


async def test_lease_lost_at_the_fence_rolls_back_every_side_effect() -> None:
    h = make_harness({"p1": FakePage(ts(10), blocks=("alpha",))})
    original = h.queue.heartbeat

    async def heartbeat_then_lose(lease: object) -> bool:
        ok = await original(lease)  # type: ignore[arg-type]
        h.world.lease_valid = False  # reclaimed after our last heartbeat
        return ok

    h.queue.heartbeat = heartbeat_then_lose  # type: ignore[method-assign]
    with pytest.raises(LeaseLost):  # re-raised: the runtime logs job_lease_lost
        await _sync_page(h)

    w = h.world
    assert w.documents == {}
    assert w.embeddings == set()
    assert w.completed == []
    assert w.touches  # step 1 touches DO commit (they only protect hashes from GC)


async def test_excluded_page_is_never_fetched() -> None:
    h = make_harness({"p1": FakePage(ts(10))})
    h.world.exclusions.add((h.tenant, "notion", "p1"))
    await _sync_page(h)

    assert h.source.fetched == []
    assert h.world.documents == {}
    assert len(h.world.completed) == 1


async def test_exclusion_committed_while_embedding_wins_over_the_sync() -> None:
    h = make_harness({"p1": FakePage(ts(10), blocks=("alpha",))})

    def exclude() -> None:  # the user removes the page while we are mid-sync
        h.world.exclusions.add((h.tenant, "notion", "p1"))

    h.registry.for_space(1).on_embed = exclude
    await _sync_page(h)

    assert h.world.documents == {}
    assert h.world.chunks == {}
    assert h.world.progress[-1] == {"outcome": "excluded"}
    assert len(h.world.completed) == 1


async def test_inactive_or_foreign_connector_is_a_noop() -> None:
    h = make_harness({"p1": FakePage(ts(10))})
    paused = h.add_connector(status="paused")
    other_tenant = uuid4()
    foreign = h.add_connector(tenant=other_tenant)
    for cid in (paused, foreign):
        lease = h.lease(KIND_SYNC_PAGE, {"connector_id": str(cid), "external_id": "p1"})
        await run_sync_job(lease, h.deps)  # foreign: our tenant cannot see it (lease tenant)
    assert h.source.fetched == []
    assert len(h.world.completed) == 2


# ================================================================= secrets + failure policy
async def test_secret_bound_to_another_connector_fails_permanently_without_leaking() -> None:
    h = make_harness({"p1": FakePage(ts(10))})
    rec = h.world.connectors[(h.tenant, h.connector)]
    other = uuid4()
    rec.sealed = h.box.seal(TOKEN, associated_data=associated_data(h.tenant, other))
    await _sync_page(h)

    [(_, error, _, permanent)] = h.world.failed
    assert permanent
    assert "secret" in error
    assert TOKEN.decode() not in error
    assert rec.status == "error"
    assert h.factory.tokens == []
    assert h.world.completed == []


async def test_tampered_or_destroyed_secret_is_permanent() -> None:
    h = make_harness({"p1": FakePage(ts(10))})
    rec = h.world.connectors[(h.tenant, h.connector)]
    assert rec.sealed is not None
    # flip the last byte (setting it to 0 was a no-op 1 time in 256: a flaky test)
    tampered = rec.sealed.ciphertext[:-1] + bytes([rec.sealed.ciphertext[-1] ^ 0xFF])
    rec.sealed = SealedSecret(tampered, rec.sealed.wrapped_dek)
    await _sync_page(h)
    rec.status, rec.sealed = "active", None
    await _sync_page(h)
    assert [f[3] for f in h.world.failed] == [True, True]


@pytest.mark.parametrize(
    ("exc", "permanent", "retry_after"),
    [
        (UpstreamRateLimited(7.5), False, 7.5),
        (ConnectionError("reset"), False, None),
    ],
)
async def test_failure_classification(
    exc: Exception, permanent: bool, retry_after: float | None
) -> None:
    h = make_harness({"p1": FakePage(ts(10))})

    async def boom(ref: object) -> object:
        raise exc

    h.source.fetch = boom  # type: ignore[assignment,method-assign]
    await _sync_page(h)
    [(_, _, after, perm)] = h.world.failed
    assert (perm, after) == (permanent, retry_after)


async def test_definitive_404_from_fetch_tombstones_and_completes() -> None:
    """A definitive "gone" is an answer: the copy is tombstoned (source_deleted) in a fenced
    commit and the job completes. Dead-lettering it would leave an undone exclusion's copy for
    reconcile to find, and an undo could revive what the owner deleted."""
    h = make_harness({"p1": FakePage(ts(10))})

    async def gone(ref: object) -> object:
        raise NotFound("page")

    h.source.fetch = gone  # type: ignore[assignment,method-assign]
    await _sync_page(h)
    assert h.world.failed == []
    assert h.world.completed
    assert _doc_id(h, "p1") in h.world.tombstoned


async def test_malformed_payload_and_unknown_kind_are_permanent() -> None:
    h = make_harness()
    await run_sync_job(h.lease(KIND_SYNC_PAGE, {"connector_id": "nope"}), h.deps)
    await run_sync_job(h.lease("mystery", {}), h.deps)
    assert [f[3] for f in h.world.failed] == [True, True]


async def test_connector_returning_a_different_page_is_rejected() -> None:
    h = make_harness({"p1": FakePage(ts(10)), "p2": FakePage(ts(10))})
    real_fetch = h.source.fetch

    async def wrong(ref: object) -> object:
        return await real_fetch(SourceItemRef(SourceKind.notion, "p2", None))

    h.source.fetch = wrong  # type: ignore[assignment,method-assign]
    await _sync_page(h, "p1")
    assert h.world.documents == {}
    assert h.world.failed[0][3] is True


# ================================================================= sync_connector
async def _sync_connector(h: Harness) -> None:
    await run_sync_job(h.lease(KIND_SYNC_CONNECTOR, {"connector_id": str(h.connector)}), h.deps)


async def test_sync_connector_enqueues_only_stale_unexcluded_pages() -> None:
    h = make_harness(
        {
            "fresh": FakePage(ts(10)),  # edited before the stored fetch's minute
            "same_minute": FakePage(ts(20, 30)),  # edit in the fetch's own minute: stale
            "edited": FakePage(ts(25)),
            "new": FakePage(ts(26)),
            "excluded": FakePage(ts(27)),
        }
    )
    h.seed_document("fresh", ts(20, 45))
    h.seed_document("same_minute", ts(20, 45))
    h.seed_document("edited", ts(20, 45))
    h.world.exclusions.add((h.tenant, "notion", "excluded"))
    await _sync_connector(h)

    specs = [s for _, s in h.world.enqueued]
    assert sorted(s.payload["external_id"] for s in specs) == ["edited", "new", "same_minute"]
    for spec in specs:
        ext = spec.payload["external_id"]
        assert spec.kind == KIND_SYNC_PAGE
        assert spec.priority is JobPriority.incremental
        assert spec.dedupe_key == f"sync_page:{h.connector}:{ext}"
        assert spec.payload == {"connector_id": str(h.connector), "external_id": ext}
        assert spec.connector_id == h.connector
    assert all(t == h.tenant for t, _ in h.world.enqueued)
    assert h.world.cursors[(h.tenant, h.connector, CURSOR_SCOPE)] == ts(27)
    assert len(h.world.completed) == 1


async def test_cursor_passes_through_and_never_moves_backwards() -> None:
    # The connector owns the listing overlap (handoff C1, request 2): sync passes the stored
    # cursor unchanged. "overlap" is older than the cursor, as a re-listed page would be.
    h = make_harness({"old": FakePage(ts(5)), "overlap": FakePage(ts(37))})
    key = (h.tenant, h.connector, CURSOR_SCOPE)
    h.world.cursors[key] = ts(40)
    await _sync_connector(h)

    assert h.source.since_seen == [ts(40)]
    assert h.world.cursors[key] == ts(40)
    assert h.world.cursor_requests
    assert all(v >= ts(40) for v in h.world.cursor_requests)

    h.source.pages["newer"] = FakePage(ts(45))
    await _sync_connector(h)
    assert h.world.cursors[key] == ts(45)
    assert all(v >= ts(40) for v in h.world.cursor_requests)


async def test_failed_listing_enqueues_nothing_and_keeps_cursor() -> None:
    h = make_harness({"a": FakePage(ts(10)), "b": FakePage(ts(11)), "c": FakePage(ts(12))})
    h.source.fail_listing_after = 2
    await _sync_connector(h)

    assert h.world.enqueued == []
    assert h.world.cursors == {}
    assert h.world.failed
    assert h.world.failed[0][3] is False


async def test_lease_lost_mid_listing_commits_no_enqueue_or_cursor() -> None:
    pages = {f"p{i:03}": FakePage(ts(10)) for i in range(10)}
    h = make_harness(pages, heartbeat_every=3)
    calls = 0

    def lose_on_second() -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            h.world.lease_valid = False

    h.queue.on_heartbeat = lose_on_second
    with pytest.raises(LeaseLost):  # re-raised: the runtime logs job_lease_lost
        await _sync_connector(h)
    assert h.world.enqueued == []
    assert h.world.cursors == {}
    assert h.world.completed == []
    assert h.world.failed == []


# ================================================================= reconcile_deletes
async def _reconcile(h: Harness) -> None:
    await run_sync_job(h.lease(KIND_RECONCILE, {"connector_id": str(h.connector)}), h.deps)


async def test_reconcile_never_tombstones_without_confirmation() -> None:
    h = make_harness({"listed": FakePage(ts(10))})
    h.seed_document("listed", ts(20))
    unconfirmed = h.seed_document("maybe_gone", ts(20))
    erroring = h.seed_document("api_error", ts(20))
    confirmed = h.seed_document("gone", ts(20))
    h.source.confirm = {"maybe_gone": False, "api_error": RuntimeError("500"), "gone": True}
    await _reconcile(h)

    assert h.world.tombstoned == {confirmed.id}
    # Sync's lock order: each confirmed tombstone takes the source-item lock first, and only
    # confirmed items are locked.
    assert [e for _k, e in h.world.locks] == ["gone"]
    assert unconfirmed.id not in h.world.tombstoned
    assert erroring.id not in h.world.tombstoned
    assert sorted(h.source.confirm_calls) == ["api_error", "gone", "maybe_gone"]
    assert [a[1] for a in h.world.audit] == ["document.tombstoned"]
    assert h.world.progress[-1]["unconfirmed"] == 2
    assert len(h.world.completed) == 1


async def test_reconcile_requeues_non_fresh_and_synced_source_dependents() -> None:
    h = make_harness(
        {
            "fresh": FakePage(ts(10)),
            "stale": FakePage(ts(25)),
            "never_synced": FakePage(ts(5)),
            "embeds_synced": FakePage(ts(10)),
            "embeds_unknown": FakePage(ts(10)),
            "embeds_old": FakePage(ts(10)),
            "source_page": FakePage(ts(25)),  # edited after the dependents were fetched
            "old_source": FakePage(ts(3)),
            "excluded": FakePage(ts(29)),
        }
    )
    for ext in ("fresh", "stale", "source_page", "old_source"):
        h.seed_document(ext, ts(20))
    h.seed_document("embeds_synced", ts(20), synced=("source_page",))
    h.seed_document("embeds_unknown", ts(20), synced=("not_listed_block",))
    h.seed_document("embeds_old", ts(20), synced=("old_source",))
    h.world.exclusions.add((h.tenant, "notion", "excluded"))
    await _reconcile(h)

    requeued = {s.payload["external_id"] for _, s in h.world.enqueued}
    assert requeued == {"stale", "never_synced", "embeds_synced", "embeds_unknown", "source_page"}
    assert all(s.priority is JobPriority.backfill for _, s in h.world.enqueued)
    assert h.world.tombstoned == set()


async def test_reconcile_lease_loss_tombstones_nothing() -> None:
    h = make_harness({})
    doc = h.seed_document("gone", ts(20))
    h.source.confirm = {"gone": True}
    original = h.source.confirm_deleted

    async def confirm_then_lose(ext: str) -> bool:
        h.world.lease_valid = False
        return await original(ext)

    h.source.confirm_deleted = confirm_then_lose  # type: ignore[method-assign,assignment]
    with pytest.raises(LeaseLost):  # re-raised: the runtime logs job_lease_lost
        await _reconcile(h)
    assert doc.id not in h.world.tombstoned
    assert h.world.completed == []


# ================================================================= delete_connector
async def test_delete_connector_destroys_secret_then_documents_then_row() -> None:
    h = make_harness(delete_batch=2)
    for i in range(5):
        h.seed_document(f"p{i}", ts(20))
    other_tenant_doc = h.seed_document("keep", ts(20))
    h.world.doc_tenant[other_tenant_doc.id] = uuid4()
    await run_sync_job(h.lease(KIND_DELETE_CONNECTOR, {"connector_id": str(h.connector)}), h.deps)

    assert (h.tenant, h.connector) not in h.world.connectors
    assert list(h.world.documents) == [other_tenant_doc.id]
    assert h.world.progress[-1] == {"documents_deleted": 5}
    assert [a[1] for a in h.world.audit] == ["connector.deleted"]
    assert len(h.world.completed) == 1


async def test_delete_connector_lease_loss_keeps_connector_and_resumes_idempotently() -> None:
    h = make_harness(delete_batch=2)
    for i in range(5):
        h.seed_document(f"p{i}", ts(20))
    rec = h.world.connectors[(h.tenant, h.connector)]
    h.world.heartbeat_ok = False  # lease lost after the first batch
    lease = h.lease(KIND_DELETE_CONNECTOR, {"connector_id": str(h.connector)})
    with pytest.raises(LeaseLost):  # re-raised: the runtime logs job_lease_lost
        await run_sync_job(lease, h.deps)

    assert rec.sealed is None
    assert rec.status == "deleting"
    assert (h.tenant, h.connector) in h.world.connectors
    assert h.world.completed == []
    assert len(h.world.documents) == 3

    h.world.heartbeat_ok = True
    await run_sync_job(lease, h.deps)  # the retry continues where it stopped
    assert h.world.documents == {}
    assert (h.tenant, h.connector) not in h.world.connectors


# ================================================================= ingest_file
async def _upload(h: Harness, data: bytes, *, sha: bytes | None = None) -> UploadRec:
    up = UploadRec(
        uuid4(),
        h.collection,
        f"uploads/{h.tenant}/x",
        "notes.md",
        sha or hashlib.sha256(data).digest(),
        len(data),
        ts(30),
    )
    h.world.uploads[(h.tenant, up.id)] = up
    h.blobs.data[up.blob_key] = data
    return up


async def test_ingest_file_uses_upload_identity_and_same_indexing_flow() -> None:
    h = make_harness()
    up = await _upload(h, b"first para\n\nsecond para")
    await run_sync_job(h.lease(KIND_INGEST_FILE, {"upload_id": str(up.id)}), h.deps)

    doc_id = document_id(h.tenant, h.collection, None, str(up.id))
    doc = h.world.documents[doc_id]
    assert doc.source_kind is SourceKind.file
    assert doc.connector_id is None
    assert doc.upload_id == up.id
    assert doc.fetched_at == ts(30)
    assert len(h.world.chunks[doc_id]) == 2
    assert h.registry.by_space[1].calls
    assert len(h.world.completed) == 1


async def test_ingest_file_rejects_corrupt_blob_permanently() -> None:
    h = make_harness()
    up = await _upload(h, b"content", sha=b"\x01" * 32)
    await run_sync_job(h.lease(KIND_INGEST_FILE, {"upload_id": str(up.id)}), h.deps)
    assert h.world.documents == {}
    assert h.world.failed[0][3] is True


async def test_ingest_file_respects_exclusions() -> None:
    h = make_harness()
    up = await _upload(h, b"content")
    h.world.exclusions.add((h.tenant, "file", str(up.id)))
    await run_sync_job(h.lease(KIND_INGEST_FILE, {"upload_id": str(up.id)}), h.deps)
    assert h.world.documents == {}
    assert len(h.world.completed) == 1
