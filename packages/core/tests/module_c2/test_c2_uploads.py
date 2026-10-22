"""accept_upload: blob first, then row + job + audit in one transaction; cleanup on failure."""

from __future__ import annotations

import hashlib
from uuid import uuid4

import pytest

from module_c2.c2_fakes import FakeBlobs, FakeQueue, FakeStore, FakeTxManager, World
from ragforge_core.domain.errors import ValidationFailed
from ragforge_core.domain.models import JobPriority
from ragforge_core.services.sync.uploads import accept_upload, clean_filename


async def test_accept_upload_writes_blob_row_job_and_audit() -> None:
    w, blobs = World(), FakeBlobs()
    tenant, coll = uuid4(), uuid4()
    res = await accept_upload(
        tx=FakeTxManager(w), store=FakeStore(w), blobs=blobs, queue=FakeQueue(w),
        tenant_id=tenant, collection_id=coll, filename="../../etc/notes.md", data=b"hi",
        actor="user:x",
    )  # fmt: skip
    up = w.uploads[(tenant, res.upload_id)]
    assert up.filename == "notes.md"
    assert up.sha256 == hashlib.sha256(b"hi").digest()
    assert blobs.data[up.blob_key] == b"hi"
    assert "notes" not in up.blob_key
    [(t, spec)] = w.enqueued
    assert t == tenant
    assert spec.payload == {"upload_id": str(res.upload_id)}
    assert spec.priority is JobPriority.interactive
    assert [a[1] for a in w.audit] == ["upload.accepted"]


async def test_failed_transaction_deletes_the_blob() -> None:
    w, blobs = World(), FakeBlobs()
    queue = FakeQueue(w)

    async def broken(*a: object, **k: object) -> object:
        raise RuntimeError("db down")

    queue.enqueue = broken  # type: ignore[method-assign,assignment]
    with pytest.raises(RuntimeError):
        await accept_upload(
            tx=FakeTxManager(w), store=FakeStore(w), blobs=blobs, queue=queue,
            tenant_id=uuid4(), collection_id=uuid4(), filename="a.txt", data=b"x", actor="u",
        )  # fmt: skip
    assert blobs.data == {}
    assert len(blobs.deleted) == 1
    assert w.uploads == {}


async def test_oversize_rejected_before_any_write() -> None:
    w, blobs = World(), FakeBlobs()
    with pytest.raises(ValidationFailed):
        await accept_upload(
            tx=FakeTxManager(w), store=FakeStore(w), blobs=blobs, queue=FakeQueue(w),
            tenant_id=uuid4(), collection_id=uuid4(), filename="a.txt", data=b"xx",
            actor="u", max_bytes=1,
        )  # fmt: skip
    assert blobs.data == {}
    assert blobs.deleted == []


@pytest.mark.parametrize("name", ["", "..", "dir/", "\x00\x1b"])
def test_clean_filename_rejects_empty_names(name: str) -> None:
    with pytest.raises(ValidationFailed):
        clean_filename(name)


def test_clean_filename_strips_controls_and_paths() -> None:
    assert clean_filename("C:\\Users\\x\\re\u202eport\x1b.md") == "report.md"
