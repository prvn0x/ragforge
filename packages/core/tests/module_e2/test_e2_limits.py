"""Request limits and upload robustness (Phase 2a review round 2).

- Every non-upload body is capped before the app runs (FastAPI parses a JSON body BEFORE auth).
- Uploads: bounded concurrency, a full spool is a 503, and a commit-phase failure never deletes
  a blob that a committed row still points to.
- Route-level validation that must run before anything is written or audited.
"""

from __future__ import annotations

import asyncio
import errno
from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from pathlib import Path
from uuid import UUID

import httpx
import psycopg
import pytest
from fastapi import FastAPI
from starlette.formparsers import MultiPartParser

from module_e2.e2_support import Fakes, TenantWorld, auth, make_hit
from ragforge_core.adapters.blob.local import LocalBlobStore
from ragforge_core.adapters.connectors.store import SqlSyncStore
from ragforge_core.apps.api import deps
from ragforge_core.apps.api.errors import ServiceBusy
from ragforge_core.apps.api.middleware import MAX_BODY_BYTES
from ragforge_core.apps.api.routes import sources
from ragforge_core.domain.ports import TransactionManager, Tx
from ragforge_core.services.sync.uploads import accept_upload, upload_blob_key

pytestmark = pytest.mark.integration
Conn = psycopg.Connection[tuple[object, ...]]
World = Callable[[], TenantWorld]


class _Counted:
    """An async request body that records how many bytes the server pulled."""

    def __init__(self, chunk: bytes = b"x" * 65536, chunks: int = 64) -> None:
        self.chunk, self.chunks, self.read = chunk, chunks, 0

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for _ in range(self.chunks):
            self.read += len(self.chunk)
            yield self.chunk


# ---------------------------------------------------------------- body limit (all JSON routes)
async def test_declared_oversized_chat_body_is_413_before_auth_and_unread(
    http: httpx.AsyncClient,
) -> None:
    body = _Counted()
    r = await http.post(
        "/v1/chat",
        content=body,
        headers={"content-type": "application/json", "content-length": str(1024**3)},
    )
    assert r.status_code == 413
    assert r.headers["content-type"].startswith("application/problem+json")
    assert r.headers["x-request-id"]
    assert body.read == 0


async def test_streamed_body_without_length_is_cut_off_at_the_cap(
    http: httpx.AsyncClient,
) -> None:
    body = _Counted()  # 4 MiB, streamed: httpx sends no Content-Length
    r = await http.post("/v1/chat", content=body, headers={"content-type": "application/json"})
    assert r.status_code == 413
    assert body.read <= MAX_BODY_BYTES + len(body.chunk)


async def test_normal_json_requests_pass_the_limit(http: httpx.AsyncClient, world: World) -> None:
    w = world()
    r = await http.post("/v1/collections", json={"name": "small"}, headers=auth(w.key))
    assert r.status_code == 201, r.text


async def test_uploads_are_exempt_from_the_json_cap(
    http: httpx.AsyncClient, fakes: Fakes, world: World
) -> None:
    w = world()
    data = b"# Big\n\n" + b"plain words here. " * 20_000  # ~360 KB, above the JSON cap
    assert len(data) > MAX_BODY_BYTES
    r = await http.post(
        f"/v1/collections/{w.collection}/files",
        files={"file": ("big.md", data, "text/markdown")},
        headers=auth(w.key),
    )
    assert r.status_code == 202, r.text
    assert len(fakes.jobs.enqueued) == 1


# ---------------------------------------------------------------- upload robustness
async def test_uploads_beyond_the_concurrency_bound_get_503_with_retry_after(
    http: httpx.AsyncClient, fakes: Fakes, world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    w = world()
    monkeypatch.setattr(sources, "_upload_slots", sources.UploadSlots(0, 0))  # all slots taken
    r = await http.post(
        f"/v1/collections/{w.collection}/files",
        files={"file": ("a.md", b"# A\n\ntext\n", "text/markdown")},
        headers=auth(w.key),
    )
    assert r.status_code == 503
    assert r.headers["retry-after"] == "5"
    assert fakes.jobs.enqueued == []


def test_one_tenant_cannot_hold_every_upload_slot() -> None:
    slots = sources.UploadSlots(total=2, per_tenant=1)
    a, b = UUID(int=1), UUID(int=2)
    with slots.take(a):
        with pytest.raises(ServiceBusy), slots.take(a):  # same tenant, second upload
            pass
        # another tenant still gets the remaining slot, and then the process is full
        with slots.take(b), pytest.raises(ServiceBusy), slots.take(UUID(int=3)):
            pass
    with slots.take(a), slots.take(b):  # every slot was released, including on errors
        pass


async def test_a_trickling_upload_times_out_with_408_and_frees_its_slot(
    http: httpx.AsyncClient, fakes: Fakes, world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    w = world()
    monkeypatch.setattr(sources, "upload_read_deadline_s", lambda limit: 0.2)

    async def trickle() -> AsyncIterator[bytes]:
        yield b'--b\r\nContent-Disposition: form-data; name="file"; filename="a.md"\r\n\r\n'
        await asyncio.sleep(5)  # the rest never arrives in time
        yield b"# A\r\n--b--\r\n"

    r = await http.post(
        f"/v1/collections/{w.collection}/files",
        content=trickle(),
        headers={
            **auth(w.key),
            "content-type": "multipart/form-data; boundary=b",
            "content-length": "1000",
        },
    )
    assert r.status_code == 408
    assert fakes.jobs.enqueued == []
    with sources._upload_slots.take(w.tenant):  # the slot was released
        pass


async def test_a_malformed_content_length_is_400(http: httpx.AsyncClient) -> None:
    r = await http.post(
        "/v1/chat",
        content=b"{}",
        headers={"content-type": "application/json", "content-length": "12abc"},
    )
    assert r.status_code == 400


async def test_a_full_spool_is_503_not_a_500(
    http: httpx.AsyncClient, fakes: Fakes, world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    w = world()

    async def no_space(self: object) -> object:
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(MultiPartParser, "parse", no_space)
    r = await http.post(
        f"/v1/collections/{w.collection}/files",
        files={"file": ("a.md", b"# A\n\ntext\n", "text/markdown")},
        headers=auth(w.key),
    )
    assert r.status_code == 503
    assert "No space" not in r.text
    assert fakes.jobs.enqueued == []


class _CommitThenFail:
    """The first tenant transaction COMMITS and then reports a failure, like a connection that
    drops after the commit reached Postgres; later transactions behave normally."""

    def __init__(self, inner: TransactionManager) -> None:
        self._inner = inner
        self.failed = False

    def tenant(self, tenant_id: UUID) -> AbstractAsyncContextManager[Tx]:
        return self._tenant(tenant_id)

    def system(self) -> AbstractAsyncContextManager[Tx]:
        return self._inner.system()

    @asynccontextmanager
    async def _tenant(self, tenant_id: UUID) -> AsyncIterator[Tx]:
        async with self._inner.tenant(tenant_id) as tx:
            yield tx
        if not self.failed:
            self.failed = True
            raise ConnectionError("connection lost after COMMIT")


async def test_a_commit_phase_failure_keeps_the_blob_of_a_committed_row(
    fakes: Fakes, world: World, superuser: Conn, tmp_path: Path
) -> None:
    w = world()
    blobs = LocalBlobStore(tmp_path / "commit-blobs")
    with pytest.raises(ConnectionError):
        await accept_upload(
            tx=_CommitThenFail(fakes.txm),
            store=SqlSyncStore(),
            blobs=blobs,
            queue=fakes.jobs,
            tenant_id=w.tenant,
            collection_id=w.collection,
            filename="notes.md",
            data=b"# Notes\n",
            actor="test",
        )
    row = superuser.execute(
        "SELECT id FROM rf.uploads WHERE collection_id = %s", (w.collection,)
    ).fetchone()
    assert row is not None, "the row committed despite the error"
    upload_id = UUID(str(row[0]))
    assert await blobs.get(upload_blob_key(w.tenant, upload_id)) == b"# Notes\n"


# ---------------------------------------------------------------- route validation
async def test_blank_chat_message_is_422_and_creates_no_conversation(
    http: httpx.AsyncClient, world: World, superuser: Conn
) -> None:
    w = world()
    before = superuser.execute(
        "SELECT count(*) FROM rf.conversations WHERE tenant_id = %s", (w.tenant,)
    ).fetchone()
    r = await http.post(
        "/v1/chat",
        json={"message": "   \n\t", "collection_ids": [str(w.collection)]},
        headers=auth(w.key),
    )
    assert r.status_code == 422
    after = superuser.execute(
        "SELECT count(*) FROM rf.conversations WHERE tenant_id = %s", (w.tenant,)
    ).fetchone()
    assert after == before


@pytest.mark.parametrize(
    "decisions",
    [
        {f"call-{i}": "approve" for i in range(51)},  # too many
        {"x" * 129: "approve"},  # id too long
    ],
)
async def test_unbounded_approval_decisions_are_422_and_not_audited(
    http: httpx.AsyncClient, world: World, superuser: Conn, decisions: dict[str, str]
) -> None:
    w = world()
    r = await http.post(
        f"/v1/approvals/{w.run}", json={"decisions": decisions}, headers=auth(w.key)
    )
    assert r.status_code == 422
    audited = superuser.execute(
        "SELECT count(*) FROM rf.audit_log WHERE tenant_id = %s AND action = 'approvals.submit'",
        (w.tenant,),
    ).fetchone()
    assert audited == (0,)


async def test_query_k_above_the_rerank_pool_returns_only_reranked_hits(
    app: FastAPI, http: httpx.AsyncClient, fakes: Fakes, world: World
) -> None:
    w = world()
    app.dependency_overrides[deps.get_rerank_top] = lambda: 2
    fakes.retriever.hits = [make_hit(w.collection, n=i) for i in range(5)]
    r = await http.post(
        "/v1/query",
        json={"query": "q", "collection_ids": [str(w.collection)], "k": 5},
        headers=auth(w.key),
    )
    assert r.status_code == 200, r.text
    assert len(r.json()["hits"]) == 2
    [(_, _, _, k)] = fakes.retriever.calls
    assert k == 2
    assert fakes.reranker.pools == [2]


def test_the_read_deadline_follows_the_declared_length_not_the_maximum() -> None:
    """Review round 4: a client declaring 10 KB must not get 50 MB worth of time to trickle it,
    or two tenants could hold both upload slots for ~a minute per cycle."""
    assert sources.upload_read_deadline_s(10_000) == sources.MIN_UPLOAD_DEADLINE_S
    fifty_mb = 50 * 1024 * 1024
    assert sources.upload_read_deadline_s(fifty_mb) == fifty_mb / sources.MIN_UPLOAD_RATE


async def test_the_route_uses_the_declared_length_for_its_deadline(
    http: httpx.AsyncClient, world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    w = world()
    seen: list[int] = []

    def deadline(declared: int) -> float:
        seen.append(declared)
        return 5.0

    monkeypatch.setattr(sources, "upload_read_deadline_s", deadline)
    r = await http.post(
        f"/v1/collections/{w.collection}/files",
        files={"file": ("a.md", b"# A\n\nsmall\n", "text/markdown")},
        headers=auth(w.key),
    )
    assert r.status_code == 202, r.text
    assert seen == [int(r.request.headers["content-length"])]


async def test_a_full_upload_volume_is_503_not_a_500(
    http: httpx.AsyncClient, fakes: Fakes, world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    w = world()

    async def full(key: str, data: bytes) -> None:
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(fakes.blobs, "put", full)
    r = await http.post(
        f"/v1/collections/{w.collection}/files",
        files={"file": ("a.md", b"# A\n\nsmall\n", "text/markdown")},
        headers=auth(w.key),
    )
    assert r.status_code == 503
    assert r.headers["retry-after"]
    assert "No space" not in r.text
    assert fakes.jobs.enqueued == []
