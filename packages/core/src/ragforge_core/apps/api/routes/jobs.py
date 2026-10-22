"""GET /v1/jobs/{id} and its SSE progress stream.

Jobs are read from `jobs` and `jobs_history` under RLS (rf_api has SELECT on both; finished jobs
move to history). The stream is woken by `NOTIFY rf_job_progress` (payload = job id only) and
falls back to polling; either way it re-reads the row under RLS and emits only on change.
Authentication happens once per request, so the stream also re-checks every `reauth_s` that its
key is still unrevoked and ends when it is not.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from uuid import UUID

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from sqlalchemy import RowMapping
from sqlalchemy.ext.asyncio import AsyncConnection

from ragforge_contracts.api import Job, JobEvent, JobStatus
from ragforge_contracts.sse import encode_job_event
from ragforge_core.adapters.db.tx import connection
from ragforge_core.apps.api.chat_stream import SSE_HEADERS
from ragforge_core.apps.api.deps import Feed, JobStreamConfig, ReadKey, StreamConfig, TxManager
from ragforge_core.apps.api.jobfeed import JobFeed
from ragforge_core.apps.api.sql import fetch_one
from ragforge_core.domain.errors import NotFound
from ragforge_core.domain.ports import TransactionManager
from ragforge_core.services.auth.principal import AuthenticatedKey

router = APIRouter(prefix="/v1")

JobLoader = Callable[[], Awaitable[RowMapping | None]]
TERMINAL = frozenset({JobStatus.done.value, JobStatus.dead.value, JobStatus.superseded.value})
KEEPALIVE = b": keepalive\n\n"

# Prefer the live row: during the move to history a job is in exactly one table per snapshot,
# but ordering makes the choice explicit rather than plan-dependent.
_JOB_SQL = """
SELECT id, kind, status, attempts, last_error, enqueued_at, progress, finished_at
FROM (
    SELECT 0 AS src, id, kind, status, attempts, last_error, enqueued_at, progress, finished_at
    FROM rf.jobs WHERE id = :id
    UNION ALL
    SELECT 1, id, kind, status, attempts, last_error, enqueued_at, progress, finished_at
    FROM rf.jobs_history WHERE id = :id
) j
ORDER BY src, finished_at DESC NULLS FIRST
LIMIT 1
"""


async def _read_job(conn: AsyncConnection, job_id: UUID) -> RowMapping | None:
    return await fetch_one(conn, _JOB_SQL, {"id": job_id})


async def _load(txm: TransactionManager, tenant_id: UUID, job_id: UUID) -> RowMapping | None:
    async with txm.tenant(tenant_id) as tx:
        return await _read_job(connection(tx), job_id)


def _is_terminal(row: RowMapping) -> bool:
    """Finished rows (finished_at set, e.g. moved to history) end the stream whatever their
    status string, so a status added later can never make a stream hang forever."""
    return row["status"] in TERMINAL or row["finished_at"] is not None


def _event(row: RowMapping) -> JobEvent:
    return JobEvent(
        job_id=row["id"],
        status=row["status"],
        progress=row["progress"] or {},
        last_error=row["last_error"],
    )


@router.get("/jobs/{job_id}", tags=["jobs"])
async def get_job(job_id: UUID, key: ReadKey, txm: TxManager) -> Job:
    row = await _load(txm, key.principal.tenant_id, job_id)
    if row is None:
        raise NotFound("job not found")
    return Job.model_validate(dict(row))


@router.get(
    "/jobs/{job_id}/events",
    tags=["jobs"],
    response_class=StreamingResponse,
    responses={200: {"content": {"text/event-stream": {}}}},
)
async def job_events(
    job_id: UUID, key: ReadKey, txm: TxManager, feed: Feed, config: StreamConfig
) -> StreamingResponse:
    if await _load(txm, key.principal.tenant_id, job_id) is None:
        raise NotFound("job not found")  # before streaming, so a foreign id is a real 404
    body = _stream(job_id, _stream_loader(txm, key, job_id, config.reauth_s), feed, config)
    return StreamingResponse(body, media_type="text/event-stream", headers=SSE_HEADERS)


def _stream_loader(
    txm: TransactionManager, key: AuthenticatedKey, job_id: UUID, reauth_s: float
) -> JobLoader:
    """Re-read the job under RLS; every `reauth_s` the same short transaction first confirms the
    key is still unrevoked (None ends the stream, exactly like a vanished job)."""
    next_check = time.monotonic() + reauth_s

    async def load() -> RowMapping | None:
        nonlocal next_check
        async with txm.tenant(key.principal.tenant_id) as tx:
            conn = connection(tx)
            if time.monotonic() >= next_check:
                if not await _key_active(conn, key.key_id):
                    return None
                next_check = time.monotonic() + reauth_s
            return await _read_job(conn, job_id)

    return load


async def _key_active(conn: AsyncConnection, key_id: UUID) -> bool:
    row = await fetch_one(
        conn, "SELECT 1 FROM rf.api_keys WHERE id = :id AND revoked_at IS NULL", {"id": key_id}
    )
    return row is not None


async def _stream(
    job_id: UUID, load: JobLoader, feed: JobFeed, config: JobStreamConfig
) -> AsyncIterator[bytes]:
    """Subscribe BEFORE the first read and clear the signal BEFORE every read: a notification
    that lands between a read and the wait leaves the signal set, so no update is missed.

    Keepalive pacing counts quiet poll timeouts (not wall-clock time), so it is deterministic."""
    async with feed.subscribe(job_id) as signal:
        last: JobEvent | None = None
        quiet_s = 0.0
        while True:
            signal.clear()
            row = await load()
            if row is None:  # deleted (e.g. tenant removal) or key revoked: stop reporting
                return
            event = _event(row)
            if event != last:
                yield encode_job_event(event)
                last, quiet_s = event, 0.0
            if _is_terminal(row):
                return
            if not await _wait(signal, config.poll_interval_s):
                quiet_s += config.poll_interval_s
                if quiet_s >= config.keepalive_s:
                    yield KEEPALIVE
                    quiet_s = 0.0


async def _wait(signal: asyncio.Event, timeout_s: float) -> bool:
    """True if notified, False on poll timeout."""
    with contextlib.suppress(TimeoutError):
        async with asyncio.timeout(timeout_s):
            await signal.wait()
            return True
    return False
