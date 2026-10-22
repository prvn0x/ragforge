"""PostgreSQL implementation of the `JobQueue` port (ADR-0010).

All logic that must be atomic across workers lives in the slot-0002 SECURITY DEFINER functions;
this adapter only marshals arguments and maps lease loss (SQLSTATE `RFL01`) to
`domain.errors.LeaseLost`.

Transaction rules (PHASE1 rule 8):
- `enqueue`, `assert_lease`, `report_progress`, `complete` run in the CALLER's tenant transaction,
  so they commit or roll back with the caller's side effects.
- `claim`, `heartbeat`, `release`, `fail` and the upkeep methods open their own short system
  transaction (no tenant): they are cross-tenant by design and only reachable via definer
  functions.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from datetime import timedelta
from typing import Any
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.exc import DBAPIError
from sqlalchemy.ext.asyncio import AsyncEngine

from ragforge_core.adapters.db.tx import connection
from ragforge_core.db.engine import system_transaction
from ragforge_core.domain.errors import LeaseLost
from ragforge_core.domain.models import JobLease, JobSpec
from ragforge_core.domain.ports import Tx
from ragforge_core.observability.logging import get_logger

log = get_logger(__name__)

LEASE_LOST_SQLSTATE = "RFL01"
MAX_ERROR_CHARS = 2000  # mirrors the truncation in fail_job; keeps the parameter small too

_ENQUEUE = text(
    "SELECT rf.enqueue_job(:kind, :priority, CAST(:payload AS jsonb), :dedupe_key, "
    ":run_after, :connector_id)"
)
_CLAIM = text(
    "SELECT job_id, tenant_id, kind, payload, lease_version, attempts "
    "FROM rf.claim_jobs(:worker_id, :n, :prefer_low)"
)
_ASSERT = text("SELECT rf.assert_lease(:job_id, :lease_version)")
_PROGRESS = text("SELECT rf.report_job_progress(:job_id, :lease_version, CAST(:progress AS jsonb))")
_COMPLETE = text("SELECT rf.complete_job(:job_id, :lease_version)")
_HEARTBEAT = text("SELECT rf.heartbeat(:job_id, :lease_version)")
_RELEASE = text("SELECT rf.release_job(:job_id, :lease_version)")
_DEFER = text("SELECT rf.defer_job(:job_id, :lease_version, :delay_s)")
_FAIL = text("SELECT rf.fail_job(:job_id, :lease_version, :error, :retry_after_s, :permanent)")
_RECLAIM = text("SELECT rf.reclaim_expired_leases(:limit)")
_MOVE = text("SELECT rf.move_finished_jobs(:min_age, :limit, :retention)")
_CLAIM_DUE = text("SELECT connector_id, tenant_id FROM rf.claim_due_connectors(:limit, :lease_s)")
_CLAIM_RECONCILE = text(
    "SELECT connector_id, tenant_id "
    "FROM rf.claim_reconcile_connectors(:limit, :interval_s, :lease_s)"
)


def _dumps(value: dict[str, Any]) -> str:
    """Canonical JSON: the payload must be a pure function of the dedupe key, so equal payloads
    should serialize identically. Non-JSON values fail here, before any SQL runs."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _lease_params(lease: JobLease) -> dict[str, object]:
    return {"job_id": lease.job_id, "lease_version": lease.lease_version}


@contextmanager
def _lease_errors(lease: JobLease) -> Iterator[None]:
    """Translate the definer functions' lease-lost SQLSTATE into the domain error. The caller's
    transaction is already aborted at that point; propagating rolls it back."""
    try:
        yield
    except DBAPIError as exc:
        if getattr(exc.orig, "sqlstate", None) == LEASE_LOST_SQLSTATE:
            raise LeaseLost(f"lease lost for job {lease.job_id}") from exc
        raise


class PgJobQueue:
    """`JobQueue` port adapter. `engine` must be the worker engine (role rf_worker) for the
    own-transaction methods; tenant-scoped methods use whatever `tx` the caller passes."""

    def __init__(self, engine: AsyncEngine) -> None:
        self._engine = engine

    # ------------------------------------------------------------ caller's tenant transaction
    async def enqueue(self, tx: Tx, spec: JobSpec) -> UUID:
        if spec.run_after is not None and spec.run_after.tzinfo is None:
            raise ValueError("JobSpec.run_after must be timezone-aware")
        result = await connection(tx).execute(
            _ENQUEUE,
            {
                "kind": spec.kind,
                "priority": int(spec.priority),
                "payload": _dumps(spec.payload),
                "dedupe_key": spec.dedupe_key,
                "run_after": spec.run_after,
                "connector_id": spec.connector_id,
            },
        )
        job_id = result.scalar_one()
        assert isinstance(job_id, UUID)
        return job_id

    async def assert_lease(self, tx: Tx, lease: JobLease) -> None:
        with _lease_errors(lease):
            await connection(tx).execute(_ASSERT, _lease_params(lease))

    async def report_progress(self, tx: Tx, lease: JobLease, progress: dict[str, object]) -> None:
        with _lease_errors(lease):
            await connection(tx).execute(
                _PROGRESS, {**_lease_params(lease), "progress": _dumps(dict(progress))}
            )

    async def complete(self, tx: Tx, lease: JobLease) -> None:
        with _lease_errors(lease):
            await connection(tx).execute(_COMPLETE, _lease_params(lease))

    # ------------------------------------------------------------ own system transaction
    async def claim(
        self, worker_id: str, n: int, *, prefer_low: bool = False
    ) -> Sequence[JobLease]:
        """`prefer_low` is the starvation guard (ADR-0010): the worker sets it on every 4th
        claim. It is keyword-only with a default, so this still satisfies the frozen port."""
        async with system_transaction(self._engine) as conn:
            rows = (
                await conn.execute(
                    _CLAIM, {"worker_id": worker_id, "n": n, "prefer_low": prefer_low}
                )
            ).all()
        return [
            JobLease(
                job_id=row.job_id,
                tenant_id=row.tenant_id,
                kind=row.kind,
                payload=dict(row.payload),
                lease_version=row.lease_version,
                attempts=row.attempts,
            )
            for row in rows
        ]

    async def heartbeat(self, lease: JobLease) -> bool:
        """Never blocks on the job row (slot 0008). False only when another holder took the job:
        it is gone, or its lease_version moved on (a new claim, or a reclaim, which bumps it). A
        row locked by this holder's own fenced result transaction, or in any state at this
        lease version (finished, or requeued by this holder's own retryable fail), is alive."""
        async with system_transaction(self._engine) as conn:
            alive = (await conn.execute(_HEARTBEAT, _lease_params(lease))).scalar_one()
        return bool(alive)

    async def release(self, lease: JobLease) -> None:
        async with system_transaction(self._engine) as conn:
            outcome = (await conn.execute(_RELEASE, _lease_params(lease))).scalar_one()
        log.info("job_released", job_id=str(lease.job_id), outcome=outcome)

    async def defer(self, lease: JobLease, delay_s: int) -> None:
        """Requeue without charging an attempt, not runnable before `delay_s` (slot 0008). A job
        enqueued over 24 h ago is buried instead (`dead`, audited): nothing can run it."""
        async with system_transaction(self._engine) as conn:
            outcome = (
                await conn.execute(_DEFER, {**_lease_params(lease), "delay_s": int(delay_s)})
            ).scalar_one()
        if outcome == "dead":
            log.error("job_dead_no_handler", job_id=str(lease.job_id), kind=lease.kind)
        else:
            log.info("job_deferred", job_id=str(lease.job_id), kind=lease.kind, outcome=outcome)

    async def fail(
        self,
        lease: JobLease,
        error: str,
        retry_after_s: float | None,
        *,
        permanent: bool = False,
    ) -> None:
        """A lost lease makes this a no-op ('lost'): the job belongs to someone else now."""
        async with system_transaction(self._engine) as conn:
            outcome = (
                await conn.execute(
                    _FAIL,
                    {
                        **_lease_params(lease),
                        "error": error[:MAX_ERROR_CHARS],
                        "retry_after_s": retry_after_s,
                        "permanent": permanent,
                    },
                )
            ).scalar_one()
        log.info("job_failed", job_id=str(lease.job_id), kind=lease.kind, outcome=outcome)

    # ------------------------------------------------------------ upkeep (worker only)
    async def reclaim_expired(self, limit: int = 500) -> int:
        async with system_transaction(self._engine) as conn:
            count = (await conn.execute(_RECLAIM, {"limit": limit})).scalar_one()
        return int(count)

    async def move_finished(
        self,
        min_age: timedelta = timedelta(minutes=5),
        limit: int = 10_000,
        retention: timedelta = timedelta(days=30),
    ) -> int:
        async with system_transaction(self._engine) as conn:
            moved = (
                await conn.execute(
                    _MOVE, {"min_age": min_age, "limit": limit, "retention": retention}
                )
            ).scalar_one()
        return int(moved)

    async def claim_due_connectors(
        self, limit: int = 100, lease_s: int = 300
    ) -> Sequence[tuple[UUID, UUID]]:
        """Due connectors with no listing queued or running; each one's next_poll_at moves
        forward by the short, fixed claim lease in the same statement, so concurrent schedulers
        and slow listings never get a second scheduled listing (slot 0008). Failure backoff is
        not applied here but by the dead-listing trigger."""
        async with system_transaction(self._engine) as conn:
            rows = (await conn.execute(_CLAIM_DUE, {"limit": limit, "lease_s": lease_s})).all()
        return [(row.connector_id, row.tenant_id) for row in rows]

    async def claim_reconcile_connectors(
        self, limit: int = 100, interval_s: int = 86_400, lease_s: int = 300
    ) -> Sequence[tuple[UUID, UUID]]:
        """Connectors due for the nightly reconcile. The claim takes only the short `lease_s`;
        slot 0008's trigger schedules the next run (`interval_s` +/-5%) when the reconcile job
        ends, so a lost enqueue costs one lease, not one interval."""
        params = {"limit": limit, "interval_s": interval_s, "lease_s": lease_s}
        async with system_transaction(self._engine) as conn:
            rows = (await conn.execute(_CLAIM_RECONCILE, params)).all()
        return [(row.connector_id, row.tenant_id) for row in rows]
