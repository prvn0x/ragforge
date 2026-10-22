"""Module A test helpers (importable because pytest puts this directory on sys.path)."""

from __future__ import annotations

from uuid import UUID

import psycopg

from ragforge_core.adapters.db.jobs import PgJobQueue
from ragforge_core.db.engine import SqlTransactionManager
from ragforge_core.domain.models import JobPriority, JobSpec

Conn = psycopg.Connection[tuple[object, ...]]


def spec(
    key: str,
    priority: JobPriority = JobPriority.incremental,
    *,
    kind: str = "test_job",
    connector_id: UUID | None = None,
) -> JobSpec:
    return JobSpec(
        kind=kind,
        priority=priority,
        payload={"key": key},
        dedupe_key=key,
        connector_id=connector_id,
    )


async def enqueue(txm: SqlTransactionManager, queue: PgJobQueue, tenant: UUID, s: JobSpec) -> UUID:
    async with txm.tenant(tenant) as tx:
        return await queue.enqueue(tx, s)


def job_row(superuser: Conn, job_id: UUID, columns: str = "status") -> tuple[object, ...]:
    row = superuser.execute(
        f"SELECT {columns} FROM rf.jobs WHERE id = %s",
        (job_id,),
    ).fetchone()
    assert row is not None, f"job {job_id} not found"
    return row
