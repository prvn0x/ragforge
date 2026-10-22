"""Accepting a file upload (called by the API, module E, for POST /v1/collections/{id}/files).

Order matters for crash safety: the blob is written first, then ONE tenant transaction inserts
the authoritative `uploads` row, enqueues `ingest_file` and writes the audit row. A crash
between the two leaves only an unreferenced blob (harmless; the key is random), never a row
pointing at a missing blob.

If the transaction fails, the blob is deleted ONLY when the failure happened inside the
transaction body, before COMMIT was sent: the rollback is then certain. A failure at or after
COMMIT (in the context manager's exit: connection drop, cancellation) is ambiguous, because the
commit may have reached Postgres, and even a fresh read could run before that commit becomes
visible. So the blob is kept: an orphan blob costs storage, a committed row and `ingest_file`
job pointing at nothing lose the user's file.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import unicodedata
from dataclasses import dataclass
from uuid import UUID, uuid4

from ragforge_core.domain.errors import ValidationFailed
from ragforge_core.domain.models import JobPriority, JobSpec
from ragforge_core.domain.ports import BlobStore, JobQueue, TransactionManager
from ragforge_core.services.sync.policy import KIND_INGEST_FILE
from ragforge_core.services.sync.ports import SyncStore

log = logging.getLogger(__name__)

MAX_UPLOAD_BYTES = 50 * 1024 * 1024
MAX_FILENAME_CHARS = 255


@dataclass(frozen=True, slots=True)
class AcceptedUpload:
    upload_id: UUID
    job_id: UUID


def clean_filename(filename: str) -> str:
    """Keep only the final path component, NFC-normalized, without control characters.

    The filename is display metadata only (blob keys never contain it), but it is shown in UIs
    and fed to the parser's type hints, so it must not smuggle paths or terminal escapes.
    """
    name = unicodedata.normalize("NFC", filename).replace("\\", "/").rsplit("/", 1)[-1]
    name = "".join(ch for ch in name if unicodedata.category(ch)[0] != "C").strip()
    if not name or name in {".", ".."}:
        raise ValidationFailed("filename is empty")
    return name[:MAX_FILENAME_CHARS]


def upload_blob_key(tenant_id: UUID, upload_id: UUID) -> str:
    return f"uploads/{tenant_id}/{upload_id}"


async def accept_upload(
    *,
    tx: TransactionManager,
    store: SyncStore,
    blobs: BlobStore,
    queue: JobQueue,
    tenant_id: UUID,
    collection_id: UUID,
    filename: str,
    data: bytes,
    actor: str,
    max_bytes: int = MAX_UPLOAD_BYTES,
) -> AcceptedUpload:
    if len(data) > max_bytes:
        raise ValidationFailed(f"file exceeds {max_bytes} bytes")
    name = clean_filename(filename)
    upload_id = uuid4()
    key = upload_blob_key(tenant_id, upload_id)
    # Up to 50 MB of hashing would stall every stream on the event loop for ~100 ms.
    digest = await asyncio.to_thread(lambda: hashlib.sha256(data).digest())

    await blobs.put(key, data)
    body_done = False  # True once every statement ran: from here on only COMMIT can fail
    try:
        async with tx.tenant(tenant_id) as t:
            await store.insert_upload(
                t,
                upload_id=upload_id,
                collection_id=collection_id,
                blob_key=key,
                filename=name,
                sha256=digest,
                size_bytes=len(data),
            )
            job_id = await queue.enqueue(
                t,
                JobSpec(
                    kind=KIND_INGEST_FILE,
                    priority=JobPriority.interactive,
                    payload={"upload_id": str(upload_id)},
                    dedupe_key=f"{KIND_INGEST_FILE}:{upload_id}",
                ),
            )
            await store.audit(
                t,
                actor=actor,
                action="upload.accepted",
                target=str(upload_id),
                details={"collection_id": str(collection_id), "size_bytes": len(data)},
            )
            body_done = True
    except BaseException:
        if body_done:
            log.warning("upload %s: commit outcome unknown, keeping blob", upload_id)
        else:
            await _discard_blob(blobs, key)
        raise
    return AcceptedUpload(upload_id=upload_id, job_id=job_id)


async def _discard_blob(blobs: BlobStore, key: str) -> None:
    """Best effort: an orphan blob is harmless, and a cleanup error must never mask the
    original one."""
    try:
        await blobs.delete(key)
    except Exception:
        log.warning("could not delete orphan upload blob %s", key)
