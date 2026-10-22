"""Job outcome metrics (ADR-0018 R2-2: the worker exports only what it alone knows):
`ragforge_jobs_total{kind, outcome}` and `ragforge_job_duration_seconds{kind}`.

Why a decorator on the queue port rather than code in the runtime or the handlers: a job ends
in exactly one queue call (complete, fail, release or defer), made either by the runtime or,
for kinds that report their own outcome (sync, eval snapshots), by the handler inside its
fenced transaction. Wrapping the ONE queue instance both of them are given sees every ending
without touching either, and the clock starts at the claim that leased the job.

Outcome mapping (catalog.JOB_OUTCOMES):
- complete -> done; release / defer -> requeued (given back without charging an attempt);
- fail -> dead when permanent or on the last attempt, else failed (retried with backoff).
  `PgJobQueue.fail` does not return rf.fail_job's verdict, so "last attempt" is predicted from
  the lease (`attempts >= MAX_ATTEMPTS`, the rf.jobs default no code path overrides). A fail
  that lost its lease is still counted: rare, and the DB `ragforge_jobs_dead` gauge is the
  authoritative count of dead jobs.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Final
from uuid import UUID

from ragforge_core.domain.models import JobLease, JobSpec
from ragforge_core.domain.ports import Tx
from ragforge_core.observability import metrics
from ragforge_core.observability.catalog import JOB_OUTCOMES
from ragforge_core.services.jobs.ports import FairJobQueue

MAX_ATTEMPTS: Final = 8  # rf.jobs.max_attempts DEFAULT (migration 0001); never set per job
_MAX_TRACKED: Final = 10_000  # leases lost without an outcome must not grow this forever

_Key = tuple[UUID, int]


class ObservedJobQueue:
    """A `FairJobQueue` that delegates every call to `inner` and records job endings."""

    def __init__(self, inner: FairJobQueue) -> None:
        self._inner = inner
        self._claimed_at: dict[_Key, float] = {}

    # ------------------------------------------------------------ pass-through
    async def enqueue(self, tx: Tx, spec: JobSpec) -> UUID:
        return await self._inner.enqueue(tx, spec)

    async def assert_lease(self, tx: Tx, lease: JobLease) -> None:
        await self._inner.assert_lease(tx, lease)

    async def report_progress(self, tx: Tx, lease: JobLease, progress: dict[str, object]) -> None:
        await self._inner.report_progress(tx, lease, progress)

    async def heartbeat(self, lease: JobLease) -> bool:
        return await self._inner.heartbeat(lease)

    # ------------------------------------------------------------ measured
    async def claim(
        self, worker_id: str, n: int, *, prefer_low: bool = False
    ) -> Sequence[JobLease]:
        leases = await self._inner.claim(worker_id, n, prefer_low=prefer_low)
        now = time.perf_counter()
        if len(self._claimed_at) + len(leases) > _MAX_TRACKED:
            self._claimed_at.clear()
        for lease in leases:
            self._claimed_at[_key(lease)] = now
        return leases

    async def complete(self, tx: Tx, lease: JobLease) -> None:
        await self._inner.complete(tx, lease)
        self._ended(lease, "done")

    async def fail(
        self,
        lease: JobLease,
        error: str,
        retry_after_s: float | None,
        *,
        permanent: bool = False,
    ) -> None:
        await self._inner.fail(lease, error, retry_after_s, permanent=permanent)
        buried = permanent or lease.attempts >= MAX_ATTEMPTS
        self._ended(lease, "dead" if buried else "failed")

    async def release(self, lease: JobLease) -> None:
        await self._inner.release(lease)
        self._ended(lease, "requeued")

    async def defer(self, lease: JobLease, delay_s: int) -> None:
        await self._inner.defer(lease, delay_s)
        self._ended(lease, "requeued")

    def _ended(self, lease: JobLease, outcome: str) -> None:
        kind = metrics.ident_label(lease.kind)
        metrics.JOBS.labels(kind, metrics.bounded(outcome, JOB_OUTCOMES)).inc()
        started = self._claimed_at.pop(_key(lease), None)
        if started is not None:
            metrics.JOB_DURATION.labels(kind).observe(time.perf_counter() - started)


def _key(lease: JobLease) -> _Key:
    return (lease.job_id, lease.lease_version)
