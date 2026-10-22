"""Migration 0012's definer functions, called as the real login roles (ADR-0013: a cross-tenant
test per function). The r4 P1 guard, the copy's contract and the exclusion purge."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from uuid import UUID, uuid4

import psycopg
import pytest
from psycopg import errors

from module_eval_snap.ev_snap_support import DemoPage, Source, make_eval_tenant
from pgtest import connect
from ragforge_core.config import Settings
from ragforge_core.domain import ids
from ragforge_core.services.eval_snapshot.digests import (
    SnapshotChunk,
    SnapshotPage,
    content_digest,
    index_digest,
)

pytestmark = pytest.mark.integration
Conn = psycopg.Connection[tuple[object, ...]]
MakeTenant = Callable[..., tuple[UUID, UUID, UUID]]
MakeSource = Callable[..., Source]
SYSTEM_TENANT = UUID(int=0)


@contextmanager
def session(settings: Settings, role: str, tenant: UUID | None) -> Iterator[Conn]:
    """One committed transaction as `role` with app.tenant_id = `tenant` (None: unset)."""
    with connect(settings, role) as conn:
        if tenant is not None:
            conn.execute("SELECT set_config('app.tenant_id', %s, true)", (str(tenant),))
        yield conn
        conn.commit()


def begin(settings: Settings, source: UUID, collection: UUID, target: UUID) -> UUID:
    """What the route does: enqueue the copy job, then begin_eval_snapshot, one transaction."""
    snapshot = uuid4()
    with session(settings, "api", source) as conn:
        row = conn.execute(
            "SELECT rf.enqueue_job('eval_snapshot_copy', 0, '{}'::jsonb, %s)", (str(snapshot),)
        ).fetchone()
        assert row is not None
        conn.execute(
            "SELECT rf.begin_eval_snapshot(%s, %s, %s, %s)", (collection, target, snapshot, row[0])
        )
    return snapshot


def copy(
    settings: Settings, source: UUID | None, collection: UUID, target: UUID, snapshot: UUID
) -> tuple[object, ...]:
    with session(settings, "worker", source) as conn:
        row = conn.execute(
            "SELECT * FROM rf.copy_collection_to_eval(%s, %s, %s)", (collection, target, snapshot)
        ).fetchone()
    assert row is not None
    return row


def snapshot_of(
    settings: Settings, superuser: Conn, src: Source
) -> tuple[UUID, UUID, tuple[object, ...]]:
    """A fresh eval tenant for `src`, and one copied snapshot in it."""
    target, _ = make_eval_tenant(superuser, src.tenant)
    snap = begin(settings, src.tenant, src.collection, target)
    return target, snap, copy(settings, src.tenant, src.collection, target, snap)


def stored(superuser: Conn, snapshot: UUID) -> dict[str, object]:
    row = superuser.execute(
        "SELECT pages, chunks, vectors, links, excluded, purged_pages, content_digest, "
        "index_digest, space_id, ready_at IS NOT NULL FROM rf.eval_snapshots "
        "WHERE collection_id = %s",
        (snapshot,),
    ).fetchone()
    assert row is not None
    keys = ("pages", "chunks", "vectors", "links", "excluded", "purged", "content", "index",
            "space", "ready")  # fmt: skip
    return dict(zip(keys, row, strict=True))


def recomputed(superuser: Conn, tenant: UUID, collection: UUID) -> tuple[object, ...]:
    row = superuser.execute(
        "SELECT * FROM rf._eval_snapshot_digests(%s, %s)", (tenant, collection)
    ).fetchone()
    assert row is not None
    return tuple(row)


# ----------------------------------------------------------------- the r4 P1 guard
def test_copy_into_a_non_eval_tenant_is_refused(
    settings: Settings, make_source: MakeSource, make_tenant: MakeTenant
) -> None:
    src = make_source([])
    personal, _, _ = make_tenant()
    with pytest.raises(errors.InsufficientPrivilege, match="not this tenant's eval tenant"):
        begin(settings, src.tenant, src.collection, personal)
    with pytest.raises(errors.InsufficientPrivilege, match="not this tenant's eval tenant"):
        copy(settings, src.tenant, src.collection, personal, uuid4())


def test_copy_into_another_owners_eval_tenant_is_refused(
    settings: Settings, superuser: Conn, make_source: MakeSource, demo_pages: list[DemoPage]
) -> None:
    mine, theirs = make_source(demo_pages[:1]), make_source([])
    their_eval, _ = make_eval_tenant(superuser, theirs.tenant)
    with pytest.raises(errors.InsufficientPrivilege):
        begin(settings, mine.tenant, mine.collection, their_eval)
    with pytest.raises(errors.InsufficientPrivilege):
        copy(settings, mine.tenant, mine.collection, their_eval, uuid4())
    assert superuser.execute(
        "SELECT count(*) FROM rf.documents WHERE tenant_id = %s", (their_eval,)
    ).fetchone() == (0,)


def test_a_collection_the_caller_cannot_see_is_refused(
    settings: Settings, superuser: Conn, make_source: MakeSource, demo_pages: list[DemoPage]
) -> None:
    mine, theirs = make_source([]), make_source(demo_pages[:1])
    my_eval, _ = make_eval_tenant(superuser, mine.tenant)
    with pytest.raises(errors.NoDataFound, match="source collection not found"):
        begin(settings, mine.tenant, theirs.collection, my_eval)
    with pytest.raises(errors.NoDataFound, match="source collection not found"):
        copy(settings, mine.tenant, theirs.collection, my_eval, uuid4())


@pytest.mark.parametrize("who", ["no tenant", "system", "the eval tenant itself"])
def test_only_a_personal_or_team_session_can_copy(
    settings: Settings, superuser: Conn, make_source: MakeSource, who: str
) -> None:
    src = make_source([])
    target, _ = make_eval_tenant(superuser, src.tenant)
    caller = {"no tenant": None, "system": SYSTEM_TENANT, "the eval tenant itself": target}[who]
    with pytest.raises(errors.InsufficientPrivilege):
        copy(settings, caller, src.collection, target, uuid4())


def test_begin_binds_a_copy_job_of_the_callers_own_tenant(
    settings: Settings, superuser: Conn, make_source: MakeSource
) -> None:
    src = make_source([])
    target, _ = make_eval_tenant(superuser, src.tenant)
    with (
        pytest.raises(errors.InvalidParameterValue, match="eval_snapshot_copy job"),
        session(settings, "api", src.tenant) as conn,
    ):
        conn.execute(
            "SELECT rf.begin_eval_snapshot(%s, %s, %s, %s)",
            (src.collection, target, uuid4(), uuid4()),
        )


def test_the_copy_and_reembed_are_executable_by_the_job_role_only(superuser: Conn) -> None:
    def can(role: str, signature: str) -> bool:
        row = superuser.execute(
            "SELECT has_function_privilege(%s, %s, 'EXECUTE')", (role, signature)
        ).fetchone()
        assert row is not None
        return bool(row[0])

    worker_only = ("rf.copy_collection_to_eval(uuid, uuid, uuid)",
                   "rf.finish_eval_reembed(uuid, integer)")  # fmt: skip
    api_only = ("rf.begin_eval_snapshot(uuid, uuid, uuid, uuid)",
                "rf.provision_eval_tenant(text)")  # fmt: skip
    internal = ("rf._eval_copy_guard(uuid, uuid)", "rf._eval_snapshot_digests(uuid, uuid)",
                "rf._uuid5(uuid, text)", "rf._purge_eval_snapshot_pages()")  # fmt: skip
    for fn in worker_only:
        assert can("rf_worker", fn), fn
        assert not can("rf_api", fn), fn
    for fn in api_only:
        assert can("rf_api", fn), fn
        assert not can("rf_worker", fn), fn
    for fn in (*worker_only, *api_only, *internal):
        for role in ("rf_metrics", "rf_backup"):
            assert not can(role, fn), (role, fn)
    for fn in internal:
        assert not can("rf_api", fn), fn
        assert not can("rf_worker", fn), fn


# ----------------------------------------------------------------- the copy
def test_the_copy_preserves_page_keys_under_upload_ownership_and_enqueues_nothing(
    settings: Settings, superuser: Conn, make_source: MakeSource, demo_pages: list[DemoPage]
) -> None:
    src = make_source()
    other_jobs = "SELECT count(*) FROM rf.jobs WHERE kind <> 'eval_snapshot_copy'"
    jobs_before = superuser.execute(other_jobs).fetchone()
    target, snap, result = snapshot_of(settings, superuser, src)

    assert result[:3] == (
        len(demo_pages),
        sum(len(p.chunks) for p in demo_pages),
        len({c.embed_sha256 for p in demo_pages for c in p.chunks}),
    )
    assert result[3] == len(demo_pages) - 1  # links
    # no ingestion/sync job anywhere; in these tenants only the copy job the route enqueued
    assert superuser.execute(other_jobs).fetchone() == jobs_before
    assert superuser.execute(
        "SELECT kind FROM rf.jobs WHERE tenant_id IN (%s, %s)", (src.tenant, target)
    ).fetchall() == [("eval_snapshot_copy",)]
    upload = superuser.execute(
        "SELECT id, size_bytes, blob_key FROM rf.uploads WHERE tenant_id = %s", (target,)
    ).fetchall()
    assert len(upload) == 1
    assert upload[0][1] == 0
    assert str(upload[0][2]).startswith("eval-snap")
    docs = superuser.execute(
        "SELECT id, connector_id, upload_id, source_kind, external_id, metadata FROM rf.documents "
        "WHERE tenant_id = %s ORDER BY external_id",
        (target,),
    ).fetchall()
    assert {d[4] for d in docs} == {p.page_id for p in demo_pages}
    for doc_id, connector, upload_id, kind, external_id, metadata in docs:
        assert connector is None
        assert upload_id == upload[0][0]
        assert kind == "notion"
        assert doc_id == ids.document_id(target, snap, None, str(external_id))
        assert isinstance(metadata, dict)
        assert "filename" in metadata
    chunk_ids = superuser.execute(
        "SELECT id, document_id, ordinal FROM rf.chunks WHERE tenant_id = %s", (target,)
    ).fetchall()
    assert all(c[0] == ids.chunk_id(UUID(str(c[1])), int(str(c[2]))) for c in chunk_ids)


def test_vectors_are_copied_not_re_embedded(
    settings: Settings, superuser: Conn, make_source: MakeSource
) -> None:
    src = make_source()
    target, snap, _ = snapshot_of(settings, superuser, src)
    pairs = superuser.execute(
        """
        SELECT s.embed_sha256, s.embedding::text = e.embedding::text, s.space_id = e.space_id
          FROM rf.embeddings s
          JOIN rf.embeddings e ON e.tenant_id = %s AND e.collection_id = %s
                              AND e.embed_sha256 = s.embed_sha256
         WHERE s.tenant_id = %s AND s.collection_id = %s
        """,
        (target, snap, src.tenant, src.collection),
    ).fetchall()
    assert pairs
    assert all(same and space for _, same, space in pairs)
    assert superuser.execute(
        "SELECT vector_count FROM rf.collection_space_stats WHERE collection_id = %s", (snap,)
    ).fetchone() == (len(pairs),)


def test_digests_are_stable_across_copies_and_tenants_and_track_content(
    settings: Settings, superuser: Conn, make_source: MakeSource, demo_pages: list[DemoPage]
) -> None:
    first, second = make_source(), make_source()  # same notes, different tenants/collections
    _, snap_a, a = snapshot_of(settings, superuser, first)
    target_b, _, b = snapshot_of(settings, superuser, second)
    assert a[5:] == b[5:]
    # a second snapshot of the same source into the same eval tenant: same digests again
    snap_c = begin(settings, second.tenant, second.collection, target_b)
    assert copy(settings, second.tenant, second.collection, target_b, snap_c)[5:] == a[5:]

    changed = make_source()
    superuser.execute(
        "UPDATE rf.chunks SET display_text = display_text || ' (edited)' "
        "WHERE collection_id = %s AND ordinal = 0 AND document_id = %s",
        (changed.collection, changed.document_ids[demo_pages[0].page_id]),
    )
    _, _, c = snapshot_of(settings, superuser, changed)
    assert c[5] != a[5]  # one edited chunk text moves the content digest
    assert stored(superuser, snap_a)["content"] == a[5]


def test_sql_digests_equal_the_reference_implementation(
    settings: Settings, superuser: Conn, make_source: MakeSource, demo_pages: list[DemoPage]
) -> None:
    _, _, result = snapshot_of(settings, superuser, make_source())
    pages = [
        SnapshotPage(
            "notion",
            p.page_id,
            tuple(
                SnapshotChunk(
                    c.ordinal, c.heading_path, c.display_text, c.display_sha256, c.embed_sha256
                )
                for c in p.chunks
            ),
        )
        for p in demo_pages
    ]
    assert result[5] == content_digest(pages)
    assert result[6] == index_digest(1, pages)


def test_deleting_or_reconnecting_the_live_connector_leaves_the_snapshot(
    settings: Settings, superuser: Conn, make_source: MakeSource, demo_pages: list[DemoPage]
) -> None:
    src = make_source()
    target, snap, result = snapshot_of(settings, superuser, src)
    superuser.execute("DELETE FROM rf.connectors WHERE id = %s", (src.connector,))  # cascades
    assert superuser.execute(
        "SELECT count(*) FROM rf.documents WHERE collection_id = %s", (src.collection,)
    ).fetchone() == (0,)
    # reconnect: a new connector re-syncs the same pages into the live collection
    superuser.execute(
        "INSERT INTO rf.connectors (tenant_id, kind, target_collection_id, display_name) "
        "VALUES (%s, 'notion', %s, 'again')",
        (src.tenant, src.collection),
    )
    counts = superuser.execute(
        "SELECT (SELECT count(*) FROM rf.documents WHERE collection_id = %s), "
        "(SELECT count(*) FROM rf.chunks WHERE collection_id = %s), "
        "(SELECT count(*) FROM rf.embeddings WHERE collection_id = %s)",
        (snap, snap, snap),
    ).fetchone()
    assert counts == result[:3]
    assert recomputed(superuser, target, snap) == result[5:]


def test_the_backup_role_reads_the_whole_snapshot(
    settings: Settings, superuser: Conn, make_source: MakeSource
) -> None:
    """r4 P2 (ADR-0014): the eval tenant holds a full copy of the notes, so it is backed up
    like them. Backups read through rf_backup; prove nothing hides the eval tenant from it."""
    target, snap, result = snapshot_of(settings, superuser, make_source())
    with connect(settings, "backup") as conn:
        row = conn.execute(
            "SELECT (SELECT count(*) FROM rf.documents WHERE tenant_id = %s), "
            "(SELECT count(*) FROM rf.chunks WHERE tenant_id = %s), "
            "(SELECT count(*) FROM rf.embeddings WHERE tenant_id = %s), "
            "(SELECT count(*) FROM rf.eval_snapshots WHERE collection_id = %s), "
            "(SELECT kind FROM rf.tenants WHERE id = %s)",
            (target, target, target, snap, target),
        ).fetchone()
    assert row == (*result[:3], 1, "eval")


# ----------------------------------------------------------------- exclusions (r4 P2)
def _exclude(settings: Settings, tenant: UUID, external_id: str) -> None:
    """As the API does it (rf_api under the tenant's RLS)."""
    with session(settings, "api", tenant) as conn:
        conn.execute(
            "INSERT INTO rf.exclusions (tenant_id, source_kind, external_id) "
            "VALUES (%s, 'notion', %s)",
            (tenant, external_id),
        )


def test_exclusions_are_honoured_at_copy_time(
    settings: Settings, superuser: Conn, make_source: MakeSource, demo_pages: list[DemoPage]
) -> None:
    src = make_source()
    excluded = demo_pages[1].page_id  # still live in the source: an exclusion racing its sync
    _exclude(settings, src.tenant, excluded)
    target, snap, result = snapshot_of(settings, superuser, src)
    assert result[0] == len(demo_pages) - 1
    assert result[4] == 1
    assert superuser.execute(
        "SELECT count(*) FROM rf.documents WHERE tenant_id = %s AND external_id = %s",
        (target, excluded),
    ).fetchone() == (0,)
    # the link from page 0 pointed at the excluded page: dropped with it
    assert superuser.execute(
        "SELECT count(*) FROM rf.document_links WHERE tenant_id = %s AND to_external_id = %s",
        (target, excluded),
    ).fetchone() == (0,)
    assert stored(superuser, snap)["excluded"] == 1


def test_a_later_exclusion_purges_the_page_from_every_snapshot_of_that_owner_only(
    settings: Settings, superuser: Conn, make_source: MakeSource, demo_pages: list[DemoPage]
) -> None:
    mine, theirs = make_source(), make_source()  # the same page ids in another owner's notes
    target, snap, before = snapshot_of(settings, superuser, mine)
    their_target, their_snap, theirs_before = snapshot_of(settings, superuser, theirs)
    page = demo_pages[2]
    page_hashes = {c.embed_sha256 for c in page.chunks}
    other_hashes = {c.embed_sha256 for p in demo_pages if p is not page for c in p.chunks}

    _exclude(settings, mine.tenant, page.page_id)

    assert superuser.execute(
        "SELECT count(*) FROM rf.documents WHERE collection_id = %s AND external_id = %s",
        (snap, page.page_id),
    ).fetchone() == (0,)
    left = {
        r[0]
        for r in superuser.execute(
            "SELECT embed_sha256 FROM rf.embeddings WHERE collection_id = %s", (snap,)
        ).fetchall()
    }
    assert left == other_hashes  # the purged page's own vectors are gone at once
    assert not (page_hashes - other_hashes) & left
    after = stored(superuser, snap)
    assert after["purged"] == 1
    assert after["pages"] == len(demo_pages) - 1
    assert after["content"] != before[5]
    assert (after["content"], after["index"]) == recomputed(superuser, target, snap)
    audit = superuser.execute(
        "SELECT details FROM rf.audit_log WHERE tenant_id = %s AND action = 'eval_snapshot.purged'",
        (target,),
    ).fetchall()
    assert [a[0] for a in audit] == [{"source_kind": "notion", "pages": 1}]
    # the other owner's snapshot of the same page ids is untouched
    assert stored(superuser, their_snap)["content"] == theirs_before[5]
    assert stored(superuser, their_snap)["purged"] == 0
    assert their_target != target


def test_the_copy_is_idempotent_on_retry(
    settings: Settings, superuser: Conn, make_source: MakeSource
) -> None:
    src = make_source()
    target, snap, first = snapshot_of(settings, superuser, src)
    assert copy(settings, src.tenant, src.collection, target, snap) == first
    assert superuser.execute(
        "SELECT count(*) FROM rf.documents WHERE collection_id = %s", (snap,)
    ).fetchone() == (first[0],)


def test_a_source_still_indexing_is_refused_and_nothing_is_written(
    settings: Settings, superuser: Conn, make_source: MakeSource, demo_pages: list[DemoPage]
) -> None:
    src = make_source()
    superuser.execute(
        "DELETE FROM rf.embeddings WHERE collection_id = %s AND embed_sha256 = %s",
        (src.collection, demo_pages[0].chunks[0].embed_sha256),
    )
    target, _ = make_eval_tenant(superuser, src.tenant)
    snap = begin(settings, src.tenant, src.collection, target)
    with pytest.raises(errors.ObjectNotInPrerequisiteState, match="still indexing"):
        copy(settings, src.tenant, src.collection, target, snap)
    assert stored(superuser, snap)["ready"] is False
    assert superuser.execute(
        "SELECT count(*) FROM rf.documents WHERE collection_id = %s", (snap,)
    ).fetchone() == (0,)


# ----------------------------------------------------------------- the paired eval tenant
def _provision(settings: Settings, tenant: UUID | None) -> Sequence[object]:
    with session(settings, "api", tenant) as conn:
        row = conn.execute("SELECT * FROM rf.provision_eval_tenant('eval harness')").fetchone()
    assert row is not None
    return row


def test_one_eval_tenant_per_personal_tenant_and_it_dies_with_its_owner(
    settings: Settings, superuser: Conn, make_tenant: MakeTenant
) -> None:
    owner, _, _ = make_tenant()
    eval_tenant, _ = _provision(settings, owner)
    with session(settings, "api", owner) as conn:
        assert conn.execute("SELECT rf.paired_eval_tenant()").fetchone() == (eval_tenant,)
    assert superuser.execute(
        "SELECT kind, eval_of_tenant_id FROM rf.tenants WHERE id = %s", (eval_tenant,)
    ).fetchone() == ("eval", owner)
    with pytest.raises(errors.UniqueViolation):
        _provision(settings, owner)
    for caller in (UUID(str(eval_tenant)), SYSTEM_TENANT, None):
        with pytest.raises(errors.InsufficientPrivilege):
            _provision(settings, caller)
    other, _, _ = make_tenant()
    with session(settings, "api", other) as conn:  # no pair: NULL, never someone else's
        assert conn.execute("SELECT rf.paired_eval_tenant()").fetchone() == (None,)
    superuser.execute("DELETE FROM rf.tenants WHERE id = %s", (owner,))
    assert superuser.execute(
        "SELECT count(*) FROM rf.tenants WHERE id = %s", (eval_tenant,)
    ).fetchone() == (0,)


def test_the_pairing_is_checked_by_the_schema(superuser: Conn, make_tenant: MakeTenant) -> None:
    owner, _, _ = make_tenant()
    with pytest.raises(errors.CheckViolation):  # an eval tenant needs its owner
        superuser.execute("INSERT INTO rf.tenants (name, kind) VALUES ('x', 'eval')")
    with pytest.raises(errors.CheckViolation):  # only an eval tenant may be paired
        superuser.execute(
            "INSERT INTO rf.tenants (name, kind, eval_of_tenant_id) VALUES ('x', 'personal', %s)",
            (owner,),
        )
