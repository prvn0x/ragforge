"""LocalBlobStore: atomic writes and path-traversal-safe keys."""

from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

from ragforge_core.adapters.blob.local import InvalidBlobKey, LocalBlobStore
from ragforge_core.domain.errors import NotFound


async def test_roundtrip_overwrite_and_no_temp_files(tmp_path: Path) -> None:
    store = LocalBlobStore(tmp_path)
    await store.put("uploads/t1/u1", b"one")
    await store.put("uploads/t1/u1", b"two")
    assert await store.get("uploads/t1/u1") == b"two"
    files = [p.name for p in (tmp_path / "uploads" / "t1").iterdir()]
    assert files == ["u1"]  # temp file renamed over the target, nothing left behind
    mode = stat.S_IMODE((tmp_path / "uploads" / "t1" / "u1").stat().st_mode)
    assert mode == 0o600


async def test_missing_blob_is_not_found_and_delete_is_idempotent(tmp_path: Path) -> None:
    store = LocalBlobStore(tmp_path)
    with pytest.raises(NotFound):
        await store.get("nope")
    await store.delete("nope")
    await store.put("a", b"x")
    await store.delete("a")
    with pytest.raises(NotFound):
        await store.get("a")


@pytest.mark.parametrize(
    "key",
    [
        "../escape",
        "a/../../escape",
        "/etc/passwd",
        "a/./b",
        "a//b",
        ".hidden",
        "a\\..\\b",
        "a\x00b",
        "",
        "a/",
        "x" * 600,
        "%2e%2e/escape",
    ],
)
async def test_traversal_keys_are_rejected(tmp_path: Path, key: str) -> None:
    store = LocalBlobStore(tmp_path / "root")
    with pytest.raises(InvalidBlobKey):
        await store.put(key, b"x")
    with pytest.raises(InvalidBlobKey):
        await store.get(key)
    assert not (tmp_path / "escape").exists()


async def test_symlink_inside_root_cannot_escape(tmp_path: Path) -> None:
    root, outside = tmp_path / "root", tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    (outside / "secret").write_bytes(b"s3cret")
    os.symlink(outside, root / "link")
    store = LocalBlobStore(root)
    with pytest.raises(InvalidBlobKey):
        await store.get("link/secret")
    with pytest.raises(InvalidBlobKey):
        await store.put("link/new", b"x")
    assert not (outside / "new").exists()


async def test_failed_write_leaves_previous_content(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = LocalBlobStore(tmp_path)
    await store.put("k", b"old")

    def broken_replace(src: str, dst: object) -> None:
        raise OSError("disk gone")

    monkeypatch.setattr(os, "replace", broken_replace)
    with pytest.raises(OSError, match="disk gone"):
        await store.put("k", b"new")
    monkeypatch.undo()
    assert await store.get("k") == b"old"
    assert os.listdir(tmp_path) == ["k"]
