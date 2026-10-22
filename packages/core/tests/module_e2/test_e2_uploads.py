"""POST /v1/collections/{id}/files: blob + uploads row + ingest_file job (handoff C2)."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from pathlib import Path
from uuid import UUID

import httpx
import psycopg
import pytest
from fastapi import FastAPI

from module_e2.e2_support import Fakes, TenantWorld, auth, issue_key
from ragforge_core.apps.api import deps
from ragforge_core.apps.api.routes.sources import MULTIPART_OVERHEAD
from ragforge_core.services.sync.policy import KIND_INGEST_FILE
from ragforge_core.services.sync.uploads import upload_blob_key

pytestmark = pytest.mark.integration
Conn = psycopg.Connection[tuple[object, ...]]
World = Callable[[], TenantWorld]

MD = b"# Title\n\nSome markdown body.\n"


def _files(name: str = "notes.md", data: bytes = MD) -> dict[str, tuple[str, bytes, str]]:
    return {"file": (name, data, "text/markdown")}


async def test_upload_stores_blob_row_and_enqueues_ingest(
    http: httpx.AsyncClient, fakes: Fakes, world: World, superuser: Conn
) -> None:
    w = world()
    r = await http.post(
        f"/v1/collections/{w.collection}/files", files=_files("../../etc/notes.md"),
        headers=auth(w.key),
    )  # fmt: skip
    assert r.status_code == 202, r.text
    assert set(r.json()) == {"job_id"}

    [spec] = fakes.jobs.enqueued
    assert spec.kind == KIND_INGEST_FILE
    upload_id = spec.payload["upload_id"]
    row = superuser.execute(
        "SELECT tenant_id, collection_id, filename, size_bytes, blob_key FROM rf.uploads "
        "WHERE id = %s",
        (upload_id,),
    ).fetchone()
    assert row is not None
    tenant, collection, filename, size, blob_key = row
    assert (tenant, collection, filename, size) == (w.tenant, w.collection, "notes.md", len(MD))
    assert blob_key == upload_blob_key(w.tenant, upload_id)
    assert await fakes.blobs.get(str(blob_key)) == MD
    audit = superuser.execute(
        "SELECT actor FROM rf.audit_log WHERE tenant_id = %s AND action = 'upload.accepted'",
        (w.tenant,),
    ).fetchall()
    assert len(audit) == 1
    assert str(audit[0][0]).startswith("key:")


async def test_upload_to_a_foreign_collection_is_404_and_writes_nothing(
    http: httpx.AsyncClient, fakes: Fakes, world: World, superuser: Conn
) -> None:
    mine, theirs = world(), world()
    r = await http.post(
        f"/v1/collections/{theirs.collection}/files", files=_files(), headers=auth(mine.key)
    )
    assert r.status_code == 404
    assert fakes.jobs.enqueued == []
    count = superuser.execute(
        "SELECT count(*) FROM rf.uploads WHERE collection_id = %s", (theirs.collection,)
    ).fetchone()
    assert count == (0,)


def _blob_files(tmp_path: Path) -> list[Path]:
    root = tmp_path / "blobs"
    return [f for f in root.rglob("*") if f.is_file()] if root.exists() else []


async def test_oversized_upload_is_413_and_writes_nothing(
    http: httpx.AsyncClient, app: FastAPI, fakes: Fakes, world: World, tmp_path: Path
) -> None:
    w = world()
    app.dependency_overrides[deps.get_max_upload_bytes] = lambda: 10
    r = await http.post(
        f"/v1/collections/{w.collection}/files", files=_files(data=b"x" * 11), headers=auth(w.key)
    )
    assert r.status_code == 413
    assert r.headers["content-type"].startswith("application/problem+json")
    assert fakes.jobs.enqueued == []
    assert _blob_files(tmp_path) == []


async def test_declared_length_over_the_cap_is_413_before_the_body_is_read(
    http: httpx.AsyncClient, fakes: Fakes, world: World
) -> None:
    w = world()
    body = _Tripwire(allowed=0)
    r = await http.post(
        f"/v1/collections/{w.collection}/files",
        content=body,
        headers={**auth(w.key), "content-type": "multipart/form-data; boundary=x",
                 "content-length": str(10 * 1024**3)},
    )  # fmt: skip
    assert r.status_code == 413
    assert body.read == 0
    assert fakes.jobs.enqueued == []


class _Tripwire:
    """An async request body that fails the test if more than `allowed` bytes are pulled."""

    def __init__(
        self, allowed: int, chunk: bytes = b"x" * 65536, chunks: int = 1000, prefix: bytes = b""
    ) -> None:
        self.allowed, self.chunk, self.chunks, self.read = allowed, chunk, chunks, 0
        self.prefix = prefix

    async def __aiter__(self) -> AsyncIterator[bytes]:
        if self.prefix:
            self.read += len(self.prefix)
            yield self.prefix
        for _ in range(self.chunks):
            self.read += len(self.chunk)
            assert self.read <= self.allowed + len(self.chunk), "body read before auth"
            yield self.chunk


async def test_unauthenticated_upload_is_401_without_reading_the_body(
    http: httpx.AsyncClient, fakes: Fakes, world: World
) -> None:
    w = world()
    body = _Tripwire(allowed=0)
    r = await http.post(
        f"/v1/collections/{w.collection}/files",
        content=body,
        headers={"content-type": "multipart/form-data; boundary=x",
                 "content-length": str(len(body.chunk) * body.chunks)},
    )  # fmt: skip
    assert r.status_code == 401
    assert body.read == 0
    assert fakes.jobs.enqueued == []


async def test_body_longer_than_declared_is_cut_off_at_the_cap(
    http: httpx.AsyncClient, app: FastAPI, fakes: Fakes, world: World, tmp_path: Path
) -> None:
    """A client that lies about Content-Length is stopped by the byte counter."""
    w = world()
    app.dependency_overrides[deps.get_max_upload_bytes] = lambda: 1024
    part = (
        b'--x\r\nContent-Disposition: form-data; name="file"; filename="a.md"\r\n'
        b"Content-Type: text/markdown\r\n\r\n"
    )  # a valid part whose file bytes never end
    body = _Tripwire(1024 + MULTIPART_OVERHEAD, chunk=b"y" * 8192, chunks=100, prefix=part)
    r = await http.post(
        f"/v1/collections/{w.collection}/files",
        content=body,
        headers={**auth(w.key), "content-type": "multipart/form-data; boundary=x",
                 "content-length": "100"},
    )  # fmt: skip
    assert r.status_code == 413
    assert body.read < len(body.chunk) * body.chunks
    assert fakes.jobs.enqueued == []
    assert _blob_files(tmp_path) == []


async def test_missing_content_length_is_411(
    http: httpx.AsyncClient, fakes: Fakes, world: World
) -> None:
    w = world()
    r = await http.post(
        f"/v1/collections/{w.collection}/files",
        content=_Tripwire(allowed=0),  # streamed: httpx sends no Content-Length
        headers={**auth(w.key), "content-type": "multipart/form-data; boundary=x"},
    )
    assert r.status_code == 411
    assert fakes.jobs.enqueued == []


@pytest.mark.parametrize(
    ("content_type", "files", "data"),
    [
        ("application/json", None, b"{}"),  # not multipart at all
        (None, {"file": ("report.pdf", b"%PDF-1.7\n...", "application/pdf")}, None),
        (None, {"file": ("blob.bin", b"\x00\x01\x02" * 100, "application/octet-stream")}, None),
    ],
)
async def test_unsupported_types_are_415(
    http: httpx.AsyncClient,
    fakes: Fakes,
    world: World,
    tmp_path: Path,
    content_type: str | None,
    files: dict[str, tuple[str, bytes, str]] | None,
    data: bytes | None,
) -> None:
    w = world()
    headers = auth(w.key) | ({"content-type": content_type} if content_type else {})
    r = await http.post(
        f"/v1/collections/{w.collection}/files", files=files, content=data, headers=headers
    )
    assert r.status_code == 415
    assert fakes.jobs.enqueued == []
    assert _blob_files(tmp_path) == []


async def test_missing_file_part_is_422(
    http: httpx.AsyncClient, fakes: Fakes, world: World
) -> None:
    w = world()
    r = await http.post(
        f"/v1/collections/{w.collection}/files",
        files={"other": ("notes.md", MD, "text/markdown")},
        headers=auth(w.key),
    )
    assert r.status_code == 422
    assert fakes.jobs.enqueued == []


async def test_filename_control_characters_are_stripped_and_empty_names_rejected(
    http: httpx.AsyncClient, fakes: Fakes, world: World, superuser: Conn
) -> None:
    w = world()
    r = await http.post(
        f"/v1/collections/{w.collection}/files",
        files=_files("no\x1b[31mtes\x07.md"),
        headers=auth(w.key),
    )
    assert r.status_code == 202, r.text
    [spec] = fakes.jobs.enqueued
    row = superuser.execute(
        "SELECT filename FROM rf.uploads WHERE id = %s", (spec.payload["upload_id"],)
    ).fetchone()
    assert row is not None
    assert not any(ord(ch) < 32 for ch in str(row[0])), row  # ESC and BEL never stored

    r = await http.post(
        f"/v1/collections/{w.collection}/files", files=_files("/"), headers=auth(w.key)
    )
    assert r.status_code == 422
    assert len(fakes.jobs.enqueued) == 1


async def test_blob_is_deleted_when_the_upload_transaction_fails(
    http: httpx.AsyncClient, fakes: Fakes, world: World, superuser: Conn, tmp_path: Path
) -> None:
    w = world()

    async def boom(tx: object, spec: object) -> UUID:
        raise RuntimeError("queue down")

    fakes.jobs.enqueue = boom  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="queue down"):  # ASGITransport re-raises the 500
        await http.post(
            f"/v1/collections/{w.collection}/files", files=_files(), headers=auth(w.key)
        )
    assert _blob_files(tmp_path) == []
    count = superuser.execute(
        "SELECT count(*) FROM rf.uploads WHERE collection_id = %s", (w.collection,)
    ).fetchone()
    assert count == (0,)


async def test_upload_needs_the_write_scope(
    http: httpx.AsyncClient, fakes: Fakes, world: World, superuser: Conn
) -> None:
    w = world()
    read_only = issue_key(superuser, w.tenant, w.user, ("read",))
    r = await http.post(
        f"/v1/collections/{w.collection}/files", files=_files(), headers=auth(read_only)
    )
    assert r.status_code == 403
    assert fakes.jobs.enqueued == []
