"""Collections and connectors.

Connector tokens are sealed with the SecretBox before they touch the database, bound to the row by
`associated_data = b"connector:<tenant_id>:<connector_id>"` (so a ciphertext copied onto another
row or tenant fails to open), and are never echoed, logged or audited.
"""

from __future__ import annotations

import asyncio
import contextlib
import errno
from collections.abc import AsyncGenerator, Iterator
from typing import Final
from uuid import UUID, uuid4

from fastapi import APIRouter, Request
from sqlalchemy import RowMapping
from sqlalchemy.ext.asyncio import AsyncConnection
from starlette.datastructures import UploadFile as StarletteUploadFile
from starlette.formparsers import MultiPartException, MultiPartParser
from starlette.requests import ClientDisconnect

from ragforge_contracts.api import (
    Accepted,
    Collection,
    CollectionCreate,
    Connector,
    ConnectorCreate,
    Page,
)
from ragforge_core.adapters.connectors.files.parser import SUPPORTED, sniff_mime
from ragforge_core.adapters.db.tx import connection
from ragforge_core.apps.api.deps import (
    Blobs,
    Jobs,
    MaxUploadBytes,
    PortNotConfigured,
    ReadKey,
    Secrets,
    SyncStores,
    TxManager,
    WriteKey,
)
from ragforge_core.apps.api.errors import (
    ClientDisconnected,
    LengthRequired,
    PayloadTooLarge,
    RequestTimeout,
    ServiceBusy,
    UnsupportedMediaType,
)
from ragforge_core.apps.api.pagination import DEFAULT_LIMIT, CursorParam, Limit, page, time_id_after
from ragforge_core.apps.api.sql import audit, fetch_all, fetch_one
from ragforge_core.domain.errors import Conflict, NotFound, ValidationFailed
from ragforge_core.domain.models import JobPriority, JobSpec
from ragforge_core.services.sync.uploads import accept_upload

router = APIRouter(prefix="/v1")

COLLECTION_COLUMNS = "id, name, active_space_id, created_at"
CONNECTOR_COLUMNS = (  # never the secret columns
    "id, kind, target_collection_id, display_name, status, last_error, created_at"
)

SYNC_CONNECTOR = "sync_connector"
DELETE_CONNECTOR = "delete_connector"


def connector_associated_data(tenant_id: UUID, connector_id: UUID) -> bytes:
    return f"connector:{tenant_id}:{connector_id}".encode()


async def require_collections(conn: AsyncConnection, ids: list[UUID]) -> None:
    """404 unless every id is a collection visible to this tenant (RLS hides foreign ones)."""
    wanted = set(ids)
    rows = await fetch_all(
        conn, "SELECT id FROM rf.collections WHERE id = ANY(:ids)", {"ids": list(wanted)}
    )
    if {r["id"] for r in rows} != wanted:
        raise NotFound("collection not found")


# ----------------------------------------------------------------- collections
@router.post("/collections", status_code=201, tags=["collections"])
async def create_collection(body: CollectionCreate, key: WriteKey, txm: TxManager) -> Collection:
    async with txm.tenant(key.principal.tenant_id) as tx:
        conn = connection(tx)
        # The NEWEST active space. Spaces get increasing ids and a blue/green rollout creates the
        # replacement after the space it replaces, so while both are active (between the flip
        # and the retirement of the old one) new collections land on the rollout target, not on
        # the outgoing space they would otherwise keep alive.
        space = await fetch_one(
            conn,
            "SELECT id FROM rf.embedding_spaces WHERE status = 'active' ORDER BY id DESC LIMIT 1",
        )
        if space is None:
            raise PortNotConfigured("no active embedding space")
        row = await fetch_one(
            conn,
            "INSERT INTO rf.collections (tenant_id, name, active_space_id) "  # noqa: S608
            "VALUES (:tenant, :name, :space) ON CONFLICT (tenant_id, name) DO NOTHING "
            f"RETURNING {COLLECTION_COLUMNS}",
            {"tenant": key.principal.tenant_id, "name": body.name, "space": space["id"]},
        )
        if row is None:
            raise Conflict(f"a collection named {body.name!r} already exists")
        await audit(conn, key, "collection.create", str(row["id"]), {"name": body.name})
    return Collection.model_validate(dict(row))


@router.get("/collections", tags=["collections"])
async def list_collections(
    key: ReadKey, txm: TxManager, cursor: CursorParam = None, limit: Limit = DEFAULT_LIMIT
) -> Page[Collection]:
    after, params = time_id_after(cursor)
    async with txm.tenant(key.principal.tenant_id) as tx:
        rows = await fetch_all(
            connection(tx),
            f"SELECT {COLLECTION_COLUMNS} FROM rf.collections WHERE true{after} "  # noqa: S608
            "ORDER BY created_at DESC, id DESC LIMIT :n",
            {**params, "n": limit + 1},
        )
    return page(rows, limit, lambda r: Collection.model_validate(dict(r)))


@router.get("/collections/{collection_id}", tags=["collections"])
async def get_collection(collection_id: UUID, key: ReadKey, txm: TxManager) -> Collection:
    async with txm.tenant(key.principal.tenant_id) as tx:
        row = await fetch_one(
            connection(tx),
            f"SELECT {COLLECTION_COLUMNS} FROM rf.collections WHERE id = :id",  # noqa: S608
            {"id": collection_id},
        )
    if row is None:
        raise NotFound("collection not found")
    return Collection.model_validate(dict(row))


# ----------------------------------------------------------------- uploads
MULTIPART_OVERHEAD = 64 * 1024  # boundary, part headers and the filename around the file bytes
MAX_CONCURRENT_UPLOADS: Final = 2
MAX_UPLOADS_PER_TENANT: Final = 1
# The read deadline is derived from the DECLARED Content-Length, not the maximum allowed size: a
# client that declares 10 KB and trickles it must not hold a slot for the 50 MB worth of time.
MIN_UPLOAD_RATE: Final = 256 * 1024  # bytes/s a client must sustain
MIN_UPLOAD_DEADLINE_S: Final = 10.0


class UploadSlots:
    """Per-process upload slots, never queued (a waiting upload would still hold its socket).

    Each upload in flight holds its spool on the RAM-backed /tmp (256 MB tmpfs in compose) plus
    a copy in process memory, so bounding concurrency bounds both. The per-tenant bound stops one
    tenant from holding every slot; the read deadline stops a slow (or trickling) client from
    holding one forever. Plain counters are safe: the event loop is single-threaded and nothing
    awaits between the check and the increment.
    """

    def __init__(self, total: int, per_tenant: int) -> None:
        self._total, self._per_tenant = total, per_tenant
        self._in_use = 0
        self._by_tenant: dict[UUID, int] = {}

    @contextlib.contextmanager
    def take(self, tenant: UUID) -> Iterator[None]:
        held = self._by_tenant.get(tenant, 0)
        if self._in_use >= self._total or held >= self._per_tenant:
            raise ServiceBusy("too many uploads in progress")
        self._in_use += 1
        self._by_tenant[tenant] = held + 1
        try:
            yield
        finally:
            self._in_use -= 1
            remaining = self._by_tenant[tenant] - 1
            if remaining:
                self._by_tenant[tenant] = remaining
            else:
                del self._by_tenant[tenant]


_upload_slots = UploadSlots(MAX_CONCURRENT_UPLOADS, MAX_UPLOADS_PER_TENANT)


def upload_read_deadline_s(declared: int) -> float:
    """Seconds allowed to receive a body of `declared` bytes."""
    return max(MIN_UPLOAD_DEADLINE_S, declared / MIN_UPLOAD_RATE)


@router.post("/collections/{collection_id}/files", status_code=202, tags=["collections"])
async def upload_file(
    collection_id: UUID,
    request: Request,
    key: WriteKey,
    txm: TxManager,
    jobs: Jobs,
    blobs: Blobs,
    store: SyncStores,
    max_bytes: MaxUploadBytes,
) -> Accepted:
    """Store the file and enqueue `ingest_file` (handoff C2).

    The route declares no File/Form parameter on purpose: FastAPI parses a declared form body
    BEFORE it resolves dependencies, so an unauthenticated client could make the server spool an
    unbounded body. Here authentication, the collection check and the Content-Length check all
    run first, and the body is then read through a byte counter that aborts past the cap.
    """
    limit = max_bytes + MULTIPART_OVERHEAD
    declared = request.headers.get("content-length")
    if declared is None:
        raise LengthRequired("Content-Length is required")
    if not declared.isdigit():
        raise ValidationFailed("invalid Content-Length")
    length = int(declared)
    if length > limit:
        raise PayloadTooLarge(f"file exceeds {max_bytes} bytes")
    if not request.headers.get("content-type", "").startswith("multipart/form-data"):
        raise UnsupportedMediaType("expected multipart/form-data with one `file` part")

    tenant = key.principal.tenant_id
    async with txm.tenant(tenant) as tx:
        await require_collections(connection(tx), [collection_id])

    with _upload_slots.take(tenant):
        try:
            async with asyncio.timeout(upload_read_deadline_s(length)):
                # The client is held to what it declared: more bytes than that are cut off.
                filename, data = await _read_single_file(request, length)
        except TimeoutError:
            raise RequestTimeout("the upload body was not received in time") from None
        if len(data) > max_bytes:
            raise PayloadTooLarge(f"file exceeds {max_bytes} bytes")
        if sniff_mime(data, filename) not in SUPPORTED:
            raise UnsupportedMediaType("supported types: markdown, plain text, docx")
        try:
            accepted = await accept_upload(
                tx=txm,
                store=store,
                blobs=blobs,
                queue=jobs,
                tenant_id=tenant,
                collection_id=collection_id,
                filename=filename,
                data=data,
                actor=key.actor,
                max_bytes=max_bytes,
            )
        except OSError as exc:
            if exc.errno == errno.ENOSPC:  # the uploads volume is full: transient, not a 500
                raise ServiceBusy("upload storage is full") from None
            raise
    return Accepted(job_id=accepted.job_id)


async def _read_single_file(request: Request, limit: int) -> tuple[str, bytes]:
    """Parse the multipart body through a counter: a client that sends more bytes than its
    Content-Length promised is cut off at `limit`, not after the spool has filled. Starlette
    closes any spool file it opened when the stream raises."""

    async def bounded() -> AsyncGenerator[bytes, None]:
        total = 0
        async for chunk in request.stream():
            total += len(chunk)
            if total > limit:
                raise PayloadTooLarge("request body exceeds the upload limit")
            yield chunk

    parser = MultiPartParser(request.headers, bounded(), max_files=1, max_fields=0)
    try:
        form = await parser.parse()
    except MultiPartException as exc:
        raise ValidationFailed(f"malformed multipart body: {exc.message}") from None
    except ClientDisconnect:
        raise ClientDisconnected("the client disconnected during the upload") from None
    except OSError as exc:
        if exc.errno == errno.ENOSPC:  # the spool (tmpfs /tmp) is full: transient, not a 500
            raise ServiceBusy("upload spool is full") from None
        raise
    try:
        part = form.get("file")
        if not isinstance(part, StarletteUploadFile):
            raise ValidationFailed("expected one file part named `file`")
        return part.filename or "", await part.read()
    finally:
        await form.close()


# ----------------------------------------------------------------- connectors
@router.post("/connectors", status_code=201, tags=["connectors"])
async def create_connector(
    body: ConnectorCreate, key: WriteKey, txm: TxManager, box: Secrets
) -> Connector:
    tenant_id = key.principal.tenant_id
    connector_id = uuid4()  # minted here so the ciphertext can be bound to it before INSERT
    sealed = box.seal(
        body.token.get_secret_value().encode(),
        associated_data=connector_associated_data(tenant_id, connector_id),
    )
    async with txm.tenant(tenant_id) as tx:
        conn = connection(tx)
        await require_collections(conn, [body.target_collection_id])
        row = await fetch_one(
            conn,
            "INSERT INTO rf.connectors (id, tenant_id, kind, target_collection_id, display_name, "  # noqa: S608
            "secret_ciphertext, wrapped_dek) VALUES (:id, :tenant, :kind, :coll, :name, :ct, :dek) "
            f"RETURNING {CONNECTOR_COLUMNS}",
            {
                "id": connector_id,
                "tenant": tenant_id,
                "kind": body.kind.value,
                "coll": body.target_collection_id,
                "name": body.display_name,
                "ct": sealed.ciphertext,
                "dek": sealed.wrapped_dek,
            },
        )
        assert row is not None
        await audit(
            conn, key, "connector.create", str(connector_id),
            {"kind": body.kind.value, "target_collection_id": str(body.target_collection_id)},
        )  # fmt: skip
    return Connector.model_validate(dict(row))


@router.get("/connectors", tags=["connectors"])
async def list_connectors(
    key: ReadKey, txm: TxManager, cursor: CursorParam = None, limit: Limit = DEFAULT_LIMIT
) -> Page[Connector]:
    after, params = time_id_after(cursor)
    async with txm.tenant(key.principal.tenant_id) as tx:
        rows = await fetch_all(
            connection(tx),
            f"SELECT {CONNECTOR_COLUMNS} FROM rf.connectors WHERE true{after} "  # noqa: S608
            "ORDER BY created_at DESC, id DESC LIMIT :n",
            {**params, "n": limit + 1},
        )
    return page(rows, limit, lambda r: Connector.model_validate(dict(r)))


@router.get("/connectors/{connector_id}", tags=["connectors"])
async def get_connector(connector_id: UUID, key: ReadKey, txm: TxManager) -> Connector:
    async with txm.tenant(key.principal.tenant_id) as tx:
        row = await _connector(connection(tx), connector_id)
    return Connector.model_validate(dict(row))


@router.post("/connectors/{connector_id}/sync", status_code=202, tags=["connectors"])
async def sync_connector(connector_id: UUID, key: WriteKey, txm: TxManager, jobs: Jobs) -> Accepted:
    async with txm.tenant(key.principal.tenant_id) as tx:
        conn = connection(tx)
        row = await _connector(conn, connector_id, lock=True)
        if row["status"] == "deleting":
            raise Conflict("connector is being deleted")
        job_id = await jobs.enqueue(
            tx, _connector_job(SYNC_CONNECTOR, connector_id, JobPriority.interactive)
        )
        await audit(conn, key, "connector.sync", str(connector_id), {"job_id": str(job_id)})
    return Accepted(job_id=job_id)


@router.delete("/connectors/{connector_id}", status_code=202, tags=["connectors"])
async def delete_connector(
    connector_id: UUID, key: WriteKey, txm: TxManager, jobs: Jobs
) -> Accepted:
    """DESIGN section 9: status `deleting` and the token destroyed NOW (same transaction as the
    job enqueue); the delete_connector job removes documents/chunks later. Idempotent."""
    async with txm.tenant(key.principal.tenant_id) as tx:
        conn = connection(tx)
        row = await fetch_one(
            conn,
            "UPDATE rf.connectors SET status = 'deleting', secret_ciphertext = NULL, "
            "wrapped_dek = NULL WHERE id = :id RETURNING id",
            {"id": connector_id},
        )
        if row is None:
            raise NotFound("connector not found")
        job_id = await jobs.enqueue(
            tx, _connector_job(DELETE_CONNECTOR, connector_id, JobPriority.maintenance)
        )
        await audit(conn, key, "connector.delete", str(connector_id), {"job_id": str(job_id)})
    return Accepted(job_id=job_id)


async def _connector(
    conn: AsyncConnection, connector_id: UUID, *, lock: bool = False
) -> RowMapping:
    row = await fetch_one(
        conn,
        f"SELECT {CONNECTOR_COLUMNS} FROM rf.connectors WHERE id = :id"  # noqa: S608
        + (" FOR SHARE" if lock else ""),
        {"id": connector_id},
    )
    if row is None:
        raise NotFound("connector not found")
    return row


def _connector_job(kind: str, connector_id: UUID, priority: JobPriority) -> JobSpec:
    # payload is a pure function of the dedupe key (JobSpec contract), so duplicates merge.
    # No JobSpec.connector_id: it counts a job against the connector's in-flight cap of 2,
    # which is meant for sync_page only; a listing job holding a slot halves page throughput.
    return JobSpec(
        kind=kind,
        priority=priority,
        payload={"connector_id": str(connector_id)},
        dedupe_key=f"{kind}:{connector_id}",
    )
