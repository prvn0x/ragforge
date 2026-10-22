"""BlobStore on a local directory (settings.upload_dir: a shared volume between api and worker).

Guarantees:
- **Atomic, durable writes:** data goes to a temp file in the same directory, is fsynced, then
  renamed over the target with os.replace (atomic on POSIX), and the directory is fsynced. Every
  directory created on the way (e.g. `uploads/<tenant>`) is fsynced together with its parent, so
  after a power loss the blob's path still exists once `put` has returned (the uploads row that
  references it commits only after that). A reader never sees a partial blob.
- **Traversal-safe keys:** keys are validated against a strict grammar (no absolute paths, no
  '.'/'..' segments, no backslashes or NULs) AND the resolved path must stay under the root,
  which also defeats symlinks planted inside the root.
- Blocking file I/O runs in a thread so the event loop is never stalled.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import re
import tempfile
from pathlib import Path

from ragforge_core.domain.errors import NotFound, ValidationFailed

_SEGMENT = r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}"
_KEY = re.compile(rf"{_SEGMENT}(?:/{_SEGMENT}){{0,15}}")
_MAX_KEY = 512
_FILE_MODE = 0o600
_DIR_MODE = 0o700


class InvalidBlobKey(ValidationFailed):
    title = "Invalid blob key"


class LocalBlobStore:
    def __init__(self, root: str | os.PathLike[str]) -> None:
        self._root = Path(root).resolve()

    # ------------------------------------------------------------------ port
    async def put(self, key: str, data: bytes) -> None:
        path = self._path(key)
        await asyncio.to_thread(self._write_atomic, path, data)

    async def get(self, key: str) -> bytes:
        path = self._path(key)
        try:
            return await asyncio.to_thread(self._read, path)
        except FileNotFoundError:
            raise NotFound("blob not found") from None

    async def delete(self, key: str) -> None:
        """Idempotent: deleting a missing blob is not an error."""
        path = self._path(key)
        with contextlib.suppress(FileNotFoundError):
            await asyncio.to_thread(path.unlink)

    # ------------------------------------------------------------------ internals
    def _path(self, key: str) -> Path:
        if len(key) > _MAX_KEY or not _KEY.fullmatch(key) or ".." in key.split("/"):
            raise InvalidBlobKey("invalid blob key")
        path = (self._root / key).resolve()
        if not path.is_relative_to(self._root) or path == self._root:
            raise InvalidBlobKey("invalid blob key")
        return path

    def _ensure_dir(self, directory: Path) -> None:
        missing = [d for d in (directory, *directory.parents) if not d.exists()]
        directory.mkdir(mode=_DIR_MODE, parents=True, exist_ok=True)
        # Re-check after creation: a symlinked intermediate directory must not escape the root.
        if not directory.resolve().is_relative_to(self._root):
            raise InvalidBlobKey("invalid blob key")
        # A new directory entry is durable only once its PARENT is fsynced; outermost first.
        for created in reversed(missing):
            self._fsync_dir(created)
            self._fsync_dir(created.parent)

    def _write_atomic(self, path: Path, data: bytes) -> None:
        self._ensure_dir(path.parent)
        fd, tmp = tempfile.mkstemp(prefix=".tmp-", dir=path.parent)
        try:
            with os.fdopen(fd, "wb") as fh:
                fh.write(data)
                fh.flush()
                os.fsync(fh.fileno())
            os.chmod(tmp, _FILE_MODE)
            os.replace(tmp, path)
        except BaseException:
            with contextlib.suppress(FileNotFoundError):
                os.unlink(tmp)
            raise
        self._fsync_dir(path.parent)

    @staticmethod
    def _fsync_dir(directory: Path) -> None:
        fd = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

    @staticmethod
    def _read(path: Path) -> bytes:
        # O_NOFOLLOW: the final component must not be a symlink swapped in after validation.
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        with os.fdopen(fd, "rb") as fh:
            return fh.read()
