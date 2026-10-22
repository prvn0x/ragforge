"""The HTTP surface end to end on the real app and database: bootstrap the eval tenant, take a
snapshot (the copy job runs through its real worker handler), read its status, re-embed it."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any
from uuid import UUID

import httpx
import psycopg
import pytest

from module_eval_snap.ev_snap_support import (
    SOURCE_SPACE,
    DemoPage,
    FakeEmbedders,
    Source,
    issue_key,
    run_job,
)
from ragforge_core.adapters.db.jobs import PgJobQueue
from ragforge_core.domain.models import JobLease
from ragforge_core.services.eval_snapshot import (
    EvalSnapshotDeps,
    run_copy_job,
    run_reembed_job,
)
from ragforge_core.services.jobs import PermanentJobError

pytestmark = pytest.mark.integration
Conn = psycopg.Connection[tuple[object, ...]]
MakeSource = Callable[..., Source]


def auth(key: str) -> dict[str, str]:
    return {"authorization": f"Bearer {key}"}


async def bootstrap(http: httpx.AsyncClient, src: Source) -> dict[str, Any]:
    response = await http.post("/v1/eval/tenant", headers=auth(src.admin_key))
    assert response.status_code == 201, response.text
    body: dict[str, Any] = response.json()
    return body


async def snapshot(
    http: httpx.AsyncClient, src: Source, queue: PgJobQueue, deps: EvalSnapshotDeps
) -> tuple[dict[str, Any], str]:
    """Bootstrap + POST a snapshot + run its copy job. Returns (final status, eval key)."""
    eval_key = (await bootstrap(http, src))["api_key"]["secret"]
    response = await http.post(
        "/v1/eval/snapshots",
        json={"source_collection_id": str(src.collection)},
        headers=auth(src.admin_key),
    )
    assert response.status_code == 202, response.text
    created = response.json()
    assert created["content_digest"] is None
    assert created["index_digest"] is None

    async def handler(lease: JobLease) -> None:
        await run_copy_job(lease, deps)

    await run_job(queue, UUID(created["job_id"]), handler)
    status = await http.get(
        f"/v1/eval/snapshots/{created['collection_id']}", headers=auth(eval_key)
    )
    assert status.status_code == 200, status.text
    body: dict[str, Any] = status.json()
    return body, eval_key


async def test_the_surface_is_invisible_when_the_flag_is_off(
    http_flag_off: httpx.AsyncClient, make_source: MakeSource
) -> None:
    src = make_source([])
    for method, url in (
        ("POST", "/v1/eval/tenant"),
        ("POST", "/v1/eval/snapshots"),
        ("GET", f"/v1/eval/snapshots/{src.collection}"),
        ("POST", f"/v1/eval/snapshots/{src.collection}/reembed"),
    ):
        anonymous = await http_flag_off.request(method, url)
        authorised = await http_flag_off.request(method, url, headers=auth(src.admin_key))
        assert anonymous.status_code == authorised.status_code == 404, (method, url)


async def test_bootstrap_creates_one_paired_tenant_with_a_dedicated_eval_key(
    http: httpx.AsyncClient, superuser: Conn, make_source: MakeSource
) -> None:
    src = make_source([])
    body = await bootstrap(http, src)
    assert body["tenant"]["kind"] == "eval"
    assert sorted(body["api_key"]["scopes"]) == ["chat", "eval", "read"]
    assert superuser.execute(
        "SELECT eval_of_tenant_id FROM rf.tenants WHERE id = %s", (body["tenant"]["id"],)
    ).fetchone() == (src.tenant,)
    again = await http.post("/v1/eval/tenant", headers=auth(src.admin_key))
    assert again.status_code == 409
    # the eval key itself works, and is confined to the eval tenant
    me = await http.get("/v1/collections", headers=auth(body["api_key"]["secret"]))
    assert me.status_code == 200
    assert me.json()["items"] == []


async def test_the_owners_own_key_can_bootstrap_and_snapshot_but_a_plain_key_cannot(
    http: httpx.AsyncClient, superuser: Conn, make_source: MakeSource
) -> None:
    """Tenant self-administration (integrator fix, found live: owners never hold `admin`): the
    owner's key-manager key (approvals:write) provisions the pair and snapshots its own notes; a
    key without key-manager rights (read/write/chat) is refused."""
    src = make_source([])
    plain = issue_key(superuser, src.tenant, src.user, ["read", "write", "chat"])
    assert (await http.post("/v1/eval/tenant", headers=auth(plain))).status_code == 403
    refused = await http.post(
        "/v1/eval/snapshots",
        json={"source_collection_id": str(src.collection)},
        headers=auth(plain),
    )
    assert refused.status_code == 403
    owner = issue_key(superuser, src.tenant, src.user, ["read", "write", "chat", "approvals:write"])
    assert (await http.post("/v1/eval/tenant", headers=auth(owner))).status_code == 201
    snap = await http.post(
        "/v1/eval/snapshots",
        json={"source_collection_id": str(src.collection)},
        headers=auth(owner),
    )
    assert snap.status_code == 202
    status = await http.get(
        f"/v1/eval/snapshots/{snap.json()['collection_id']}", headers=auth(owner)
    )
    assert status.status_code == 200  # the owner watches its own snapshot (found live)
    assert (
        await http.get(f"/v1/eval/snapshots/{snap.json()['collection_id']}", headers=auth(plain))
    ).status_code == 403


async def test_a_snapshot_needs_the_eval_tenant_first(
    http: httpx.AsyncClient, make_source: MakeSource
) -> None:
    src = make_source([])
    response = await http.post(
        "/v1/eval/snapshots",
        json={"source_collection_id": str(src.collection)},
        headers=auth(src.admin_key),
    )
    assert response.status_code == 409


async def test_another_owners_collection_is_not_found_and_nothing_is_enqueued(
    http: httpx.AsyncClient, superuser: Conn, make_source: MakeSource
) -> None:
    mine, theirs = make_source([]), make_source([])
    await bootstrap(http, mine)
    response = await http.post(
        "/v1/eval/snapshots",
        json={"source_collection_id": str(theirs.collection)},
        headers=auth(mine.admin_key),
    )
    assert response.status_code == 404
    assert superuser.execute(
        "SELECT count(*) FROM rf.jobs WHERE tenant_id = %s", (mine.tenant,)
    ).fetchone() == (0,)  # the enqueue rolled back with the refusal


async def test_snapshot_end_to_end_and_the_owner_never_sees_the_copy(
    http: httpx.AsyncClient,
    make_source: MakeSource,
    queue: PgJobQueue,
    snap_deps: EvalSnapshotDeps,
    demo_pages: list[DemoPage],
) -> None:
    src = make_source()
    status, eval_key = await snapshot(http, src, queue, snap_deps)
    assert len(status["content_digest"]) == len(status["index_digest"]) == 64
    collection = status["collection_id"]

    # the source's admin key sees the same status through its pair
    as_admin = await http.get(f"/v1/eval/snapshots/{collection}", headers=auth(src.admin_key))
    assert as_admin.json() == status
    # the copy's progress is on the existing job feed of the source tenant
    job = await http.get(f"/v1/jobs/{status['job_id']}", headers=auth(src.admin_key))
    assert job.json()["status"] == "done"
    assert job.json()["progress"]["pages"] == len(demo_pages)

    owner_collections = await http.get("/v1/collections", headers=auth(src.admin_key))
    owner_ids = {c["id"] for c in owner_collections.json()["items"]}
    assert collection not in owner_ids
    assert str(src.collection) in owner_ids
    owner_docs = await http.get("/v1/documents?limit=100", headers=auth(src.admin_key))
    assert len(owner_docs.json()["items"]) == len(demo_pages)  # no duplicate pages
    eval_collections = await http.get("/v1/collections", headers=auth(eval_key))
    assert {c["id"] for c in eval_collections.json()["items"]} == {collection}


async def test_status_is_not_found_for_anyone_else(
    http: httpx.AsyncClient,
    make_source: MakeSource,
    queue: PgJobQueue,
    snap_deps: EvalSnapshotDeps,
    superuser: Conn,
) -> None:
    mine, theirs = make_source([]), make_source([])
    status, _ = await snapshot(http, mine, queue, snap_deps)
    their_eval = (await bootstrap(http, theirs))["api_key"]["secret"]
    url = f"/v1/eval/snapshots/{status['collection_id']}"
    assert (await http.get(url, headers=auth(theirs.admin_key))).status_code == 404
    assert (await http.get(url, headers=auth(their_eval))).status_code == 404
    reader = issue_key(superuser, mine.tenant, mine.user, ["read", "chat"])
    assert (await http.get(url, headers=auth(reader))).status_code == 403


async def test_reembed_rebuilds_the_ingest_input_and_flips_the_space(
    http: httpx.AsyncClient,
    superuser: Conn,
    make_source: MakeSource,
    queue: PgJobQueue,
    snap_deps: EvalSnapshotDeps,
    embedders: FakeEmbedders,
    new_space: int,
    demo_pages: list[DemoPage],
) -> None:
    src = make_source()
    before, eval_key = await snapshot(http, src, queue, snap_deps)
    collection = before["collection_id"]
    response = await http.post(
        f"/v1/eval/snapshots/{collection}/reembed",
        json={"space_id": new_space},
        headers=auth(eval_key),
    )
    assert response.status_code == 202, response.text
    pending = response.json()
    assert pending["index_digest"] is None
    assert pending["job_id"] != before["job_id"]

    async def handler(lease: JobLease) -> None:
        await run_reembed_job(lease, snap_deps)

    await run_job(queue, UUID(pending["job_id"]), handler)
    after = (await http.get(f"/v1/eval/snapshots/{collection}", headers=auth(eval_key))).json()
    assert after["content_digest"] == before["content_digest"]
    assert after["index_digest"] != before["index_digest"]
    assert after["job_id"] == pending["job_id"]

    # exactly the texts the ingest path embedded, once per distinct hash
    expected = {c.embedded_text for p in demo_pages for c in p.chunks}
    assert sorted(embedders.for_space(new_space).inputs) == sorted(expected)
    assert superuser.execute(
        "SELECT active_space_id FROM rf.collections WHERE id = %s", (collection,)
    ).fetchone() == (new_space,)
    assert superuser.execute(
        "SELECT count(*) FROM rf.embeddings WHERE collection_id = %s AND space_id = %s",
        (collection, new_space),
    ).fetchone() == (len(expected),)

    # back to the original space: nothing to embed (its vectors were kept), digest restored
    back = await http.post(
        f"/v1/eval/snapshots/{collection}/reembed",
        json={"space_id": SOURCE_SPACE},
        headers=auth(src.admin_key),  # the source's admin may do it too
    )
    assert back.status_code == 202
    await run_job(queue, UUID(back.json()["job_id"]), handler)
    restored = (await http.get(f"/v1/eval/snapshots/{collection}", headers=auth(eval_key))).json()
    assert restored["index_digest"] == before["index_digest"]
    assert embedders.for_space(SOURCE_SPACE).inputs == []


async def test_reembed_refuses_an_input_that_no_longer_matches_its_hash(
    http: httpx.AsyncClient,
    superuser: Conn,
    make_source: MakeSource,
    queue: PgJobQueue,
    snap_deps: EvalSnapshotDeps,
    embedders: FakeEmbedders,
    new_space: int,
    demo_pages: list[DemoPage],
) -> None:
    src = make_source(demo_pages[:2])
    before, eval_key = await snapshot(http, src, queue, snap_deps)
    collection = before["collection_id"]
    superuser.execute(  # a stored text whose embed_sha256 no longer describes it
        "UPDATE rf.chunks SET display_text = display_text || ' drift' "
        "WHERE collection_id = %s AND ordinal = 0",
        (collection,),
    )
    response = await http.post(
        f"/v1/eval/snapshots/{collection}/reembed",
        json={"space_id": new_space},
        headers=auth(eval_key),
    )

    async def handler(lease: JobLease) -> None:
        await run_reembed_job(lease, snap_deps)

    with pytest.raises(PermanentJobError, match="does not match its embed_sha256"):
        await run_job(queue, UUID(response.json()["job_id"]), handler)
    assert embedders.for_space(new_space).inputs == []  # refused before any embed call
    assert superuser.execute(
        "SELECT count(*) FROM rf.embeddings WHERE collection_id = %s AND space_id = %s",
        (collection, new_space),
    ).fetchone() == (0,)
    assert superuser.execute(
        "SELECT active_space_id FROM rf.collections WHERE id = %s", (collection,)
    ).fetchone() == (SOURCE_SPACE,)


async def test_reembed_validates_the_space_and_the_snapshot_state(
    http: httpx.AsyncClient,
    superuser: Conn,
    make_source: MakeSource,
    queue: PgJobQueue,
    snap_deps: EvalSnapshotDeps,
) -> None:
    src = make_source([])
    status, eval_key = await snapshot(http, src, queue, snap_deps)
    url = f"/v1/eval/snapshots/{status['collection_id']}/reembed"
    unknown = await http.post(url, json={"space_id": 999_999}, headers=auth(eval_key))
    assert unknown.status_code == 404
    retired = superuser.execute(
        "INSERT INTO rf.embedding_spaces (provider, model, dim, distance, status, retired_at) "
        "VALUES ('fake', 'retired-for-test-' || gen_random_uuid(), 4, 'cosine', 'retired', now()) "
        "RETURNING id"
    ).fetchone()
    assert retired is not None
    gone = await http.post(url, json={"space_id": retired[0]}, headers=auth(eval_key))
    assert gone.status_code == 422

    pending = await http.post(
        "/v1/eval/snapshots",
        json={"source_collection_id": str(src.collection)},
        headers=auth(src.admin_key),
    )
    copying = await http.post(
        f"/v1/eval/snapshots/{pending.json()['collection_id']}/reembed",
        json={"space_id": SOURCE_SPACE},
        headers=auth(eval_key),
    )
    assert copying.status_code == 409
