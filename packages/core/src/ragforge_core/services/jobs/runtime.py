"""Worker runtime: claim loop, per-job heartbeat, fenced completion, graceful shutdown (ADR-0010).

Pure orchestration over ports; the app (apps/worker) wires the Postgres queue, the LISTEN
connection that calls `wake()`, and the liveness file.

Guarantees and why:
- Correctness never depends on this process behaving: every result transaction is fenced by
  `assert_lease` in SQL. The runtime's job is liveness and promptness: stop work whose lease is
  gone (cancel the handler), keep live leases alive (heartbeat at a third of the lease), and give
  jobs back on shutdown instead of waiting for lease expiry.
- Concurrency is bounded by `concurrency` running handlers; claims ask only for free slots, so a
  claimed job never waits in-process (its lease would tick while it is not running).
- Wake-ups: NOTIFY (via `wake()`), a finished job (a slot freed), or the poll interval (2 s)
  as the fallback for missed notifications and reconnects.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from ragforge_core.domain.errors import LeaseLost, UpstreamRateLimited
from ragforge_core.domain.models import JobLease
from ragforge_core.domain.ports import TransactionManager
from ragforge_core.observability.logging import get_logger
from ragforge_core.services.jobs.fairness import StarvationGuard
from ragforge_core.services.jobs.ports import FairJobQueue, QueueMaintenance
from ragforge_core.services.jobs.registry import JobHandlers, PermanentJobError

log = get_logger(__name__)

MAX_ERROR_CHARS = 500  # last_error/audit text: type + short message, never full tracebacks
UNKNOWN_KIND_DELAY_S = 60  # a newer worker of a rolling deploy usually picks it up meanwhile
MAX_RUNTIME_ERROR = "job exceeded its max runtime"  # fixed text: shown to users as last_error


@dataclass(frozen=True, slots=True)
class WorkerConfig:
    concurrency: int = 4
    poll_interval_s: float = 2.0
    lease_s: float = 60.0  # must match rf._job_lease()
    heartbeat_interval_s: float = 20.0  # a third of the lease: two missed beats still hold it
    shutdown_grace_s: float = 25.0
    release_timeout_s: float = 5.0
    reclaim_interval_s: float = 15.0
    move_finished_interval_s: float = 300.0

    def __post_init__(self) -> None:
        if self.concurrency < 1:
            raise ValueError("concurrency must be >= 1")
        if not 0 < self.heartbeat_interval_s < self.lease_s:
            raise ValueError("heartbeat_interval_s must be positive and below lease_s")


@dataclass(slots=True)
class _Slot:
    lease: JobLease
    work: asyncio.Task[None] | None = None
    lost: bool = False  # heartbeat said the lease is gone: discard, do not report
    abandoned: bool = False  # shutdown grace expired: release the job
    timed_out: bool = False  # ran past its kind's max runtime: cancelled, failed (retryable)


def describe_error(exc: BaseException) -> str:
    """What goes into last_error and the audit row. Deliberately short: exception messages can
    echo upstream responses, so we cap them rather than persist arbitrary text."""
    message = str(exc).strip().splitlines()[0] if str(exc).strip() else ""
    return f"{type(exc).__name__}: {message}"[:MAX_ERROR_CHARS]


class WorkerRuntime:
    def __init__(
        self,
        *,
        worker_id: str,
        queue: FairJobQueue,
        transactions: TransactionManager,
        handlers: JobHandlers,
        maintenance: QueueMaintenance | None = None,
        config: WorkerConfig | None = None,
        on_tick: Callable[[], Awaitable[None]] | None = None,
    ) -> None:
        self._worker_id = worker_id
        self._queue = queue
        self._tx = transactions
        self._handlers = handlers
        self._maintenance = maintenance
        self._cfg = config or WorkerConfig()
        self._on_tick = on_tick
        self._guard = StarvationGuard()
        self._wake = asyncio.Event()
        self._slots: dict[asyncio.Task[None], _Slot] = {}

    # ------------------------------------------------------------ public API
    def wake(self) -> None:
        """Called by the LISTEN rf_jobs listener (and internally when a slot frees)."""
        self._wake.set()

    @property
    def running(self) -> int:
        return len(self._slots)

    async def run(self, stop: asyncio.Event) -> None:
        """Claim and run jobs until `stop` is set, then drain (finish within grace or release)."""
        maintenance = asyncio.create_task(self._maintenance_loop(stop))
        try:
            while not stop.is_set():
                if self._on_tick is not None:
                    await self._on_tick()
                self._wake.clear()  # before claiming: a NOTIFY during the claim re-wakes us
                if await self._claim_once():
                    continue  # got a full batch: more work is likely queued
                await self._sleep(stop)
        finally:
            maintenance.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await maintenance
            await self._drain()

    # ------------------------------------------------------------ claim loop
    async def _claim_once(self) -> bool:
        """Claim up to the free slots; True when the batch came back full."""
        free = self._cfg.concurrency - len(self._slots)
        if free <= 0:
            return False
        plan = self._guard.plan(free)
        try:
            leases = await self._queue.claim(self._worker_id, plan.n, prefer_low=plan.prefer_low)
        except Exception as exc:  # DB blip: keep the loop alive, retry after the poll interval
            log.warning("job_claim_failed", error=describe_error(exc))
            return False
        self._guard.record(plan, len(leases))
        for lease in leases:
            self._start(lease)
        return len(leases) == plan.n and len(self._slots) < self._cfg.concurrency

    async def _sleep(self, stop: asyncio.Event) -> None:
        waiters = [asyncio.ensure_future(self._wake.wait()), asyncio.ensure_future(stop.wait())]
        try:
            await asyncio.wait(
                waiters, timeout=self._cfg.poll_interval_s, return_when=asyncio.FIRST_COMPLETED
            )
        finally:
            for waiter in waiters:
                waiter.cancel()

    def _start(self, lease: JobLease) -> None:
        slot = _Slot(lease=lease)
        task = asyncio.create_task(self._supervise(slot), name=f"job:{lease.job_id}")
        self._slots[task] = slot
        task.add_done_callback(self._on_job_done)

    def _on_job_done(self, task: asyncio.Task[None]) -> None:
        self._slots.pop(task, None)
        self._wake.set()  # a slot is free: claim again without waiting for the poll
        exc = None if task.cancelled() else task.exception()
        if exc is not None:
            log.error("job_supervisor_crashed", error=describe_error(exc))

    # ------------------------------------------------------------ one job
    async def _supervise(self, slot: _Slot) -> None:
        lease = slot.lease
        handler = self._handlers.get(lease.kind)
        if handler is None:
            # During a rolling deploy a newer worker may know this kind. Failing would charge an
            # attempt each time (a new kind could go dead while old workers linger); a plain
            # release would let this worker re-claim it at once. Defer: no charge, no tight loop.
            # Bounded in SQL: a job enqueued over 24 h ago is buried instead (rf.defer_job).
            log.warning("job_kind_unknown_deferred", job_id=str(lease.job_id), kind=lease.kind)
            await self._safe_defer(lease, UNKNOWN_KIND_DELAY_S)
            return
        slot.work = asyncio.create_task(handler(lease))
        keepalive = asyncio.create_task(self._keepalive(slot))
        try:
            # asyncio.wait (not await): a cancelled handler must not cancel the supervisor.
            done, _ = await asyncio.wait(
                {slot.work}, timeout=self._handlers.max_runtime_s(lease.kind)
            )
            if not done:
                slot.timed_out = True
                slot.work.cancel()
                await asyncio.wait({slot.work}, timeout=self._cfg.release_timeout_s)
        except asyncio.CancelledError:
            slot.work.cancel()  # never leave an orphaned handler running
            raise
        finally:
            keepalive.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await keepalive
        if not slot.work.done():
            await self._hold_zombie(slot)
            return
        await self._settle(slot)

    async def _hold_zombie(self, slot: _Slot) -> None:
        """The handler ignored cancellation past `release_timeout_s`. Report NOTHING: failing
        the job now would requeue it for another worker while this handler may still be
        calling upstream APIs. The keepalive has stopped, so the lease expires and reclaim
        decides (every result transaction is fenced). Keep the SLOT until the handler really
        ends, so `concurrency` stays a true bound on running handlers; a handler that never
        ends leaks one slot, logged loudly, instead of silently oversubscribing."""
        lease = slot.lease
        log.error(
            "job_handler_ignored_cancellation",
            job_id=str(lease.job_id),
            kind=lease.kind,
            timed_out=slot.timed_out,
        )
        assert slot.work is not None
        await asyncio.wait({slot.work})
        log.warning("job_handler_finished_after_cancellation", job_id=str(lease.job_id))

    async def _settle(self, slot: _Slot) -> None:
        """Report the handler's outcome. Lease loss always wins: never report for a lost job."""
        lease, work = slot.lease, slot.work
        assert work is not None
        if not work.done():  # ignored cancellation: never report, reclaim decides
            return
        if slot.lost:
            log.warning("job_lease_lost", job_id=str(lease.job_id), kind=lease.kind)
            return
        if slot.timed_out and not slot.abandoned:
            log.error(
                "job_max_runtime_exceeded",
                job_id=str(lease.job_id),
                kind=lease.kind,
                max_runtime_s=self._handlers.max_runtime_s(lease.kind),
            )
            await self._safe_fail(lease, MAX_RUNTIME_ERROR)  # retryable: may be transient
            return
        if work.cancelled():
            if slot.abandoned:
                await self._safe_release(lease)
            return
        exc = work.exception()
        self_reporting = self._handlers.reports_outcome(lease.kind)
        if exc is None:
            if not self_reporting:
                await self._finish(lease)
        elif isinstance(exc, LeaseLost):
            log.warning("job_lease_lost", job_id=str(lease.job_id), kind=lease.kind)
        elif self_reporting:
            # The handler's own complete/fail call failed (e.g. the DB dropped). Reporting a
            # failure from here would mislabel it (a permanent failure would become transient
            # with the DB error as last_error); the lease expires and reclaim re-runs the job.
            log.error(
                "job_outcome_report_failed",
                job_id=str(lease.job_id),
                kind=lease.kind,
                error=describe_error(exc),
            )
        elif isinstance(exc, PermanentJobError):
            await self._safe_fail(lease, describe_error(exc), permanent=True)
        elif isinstance(exc, UpstreamRateLimited):
            await self._safe_fail(lease, describe_error(exc), retry_after_s=exc.retry_after_s)
        else:
            log.warning("job_failed", job_id=str(lease.job_id), kind=lease.kind, exc_info=exc)
            await self._safe_fail(lease, describe_error(exc))

    async def _finish(self, lease: JobLease) -> None:
        """Complete a job whose handler returned without completing it. Fenced like any result
        transaction: if the handler already completed (or the lease is gone), assert_lease
        raises LeaseLost and nothing happens."""
        try:
            async with self._tx.tenant(lease.tenant_id) as tx:
                await self._queue.assert_lease(tx, lease)
                await self._queue.complete(tx, lease)
        except LeaseLost:
            return
        except Exception as exc:
            log.warning("job_complete_failed", job_id=str(lease.job_id), error=describe_error(exc))

    async def _keepalive(self, slot: _Slot) -> None:
        """Extend the lease; on loss, or once the lease can no longer be trusted, cancel the work.

        The server's lease runs from the server time of the last good beat, which is EARLIER
        than our `last_ok`, and reclaim may hand the job to another worker as soon as it ends.
        So the deadline is `last_ok + lease_s - heartbeat_interval_s` (a beat's worth of
        margin), and every sleep and beat is bounded by the time left: a slow or hanging beat
        can never push cancellation past the point where someone else may already run the job.
        """
        cfg = self._cfg
        last_ok = time.monotonic()
        while True:
            deadline = last_ok + cfg.lease_s - cfg.heartbeat_interval_s
            remaining = deadline - time.monotonic()
            if remaining > 0:
                await asyncio.sleep(min(cfg.heartbeat_interval_s, remaining))
                remaining = deadline - time.monotonic()
            if remaining <= 0:
                log.warning(
                    "job_heartbeat_deadline_passed",
                    job_id=str(slot.lease.job_id),
                    since_last_ok_s=round(time.monotonic() - last_ok, 1),
                )
                self._lose(slot)
                return
            try:
                alive = await asyncio.wait_for(
                    self._queue.heartbeat(slot.lease),
                    timeout=min(cfg.heartbeat_interval_s, remaining),
                )
            except Exception as exc:  # incl. TimeoutError: retry until the deadline
                log.warning("job_heartbeat_failed", error=describe_error(exc))
                continue
            if not alive:
                self._lose(slot)
                return
            last_ok = time.monotonic()

    @staticmethod
    def _lose(slot: _Slot) -> None:
        slot.lost = True
        if slot.work is not None:
            slot.work.cancel()

    async def _safe_fail(
        self,
        lease: JobLease,
        error: str,
        *,
        retry_after_s: float | None = None,
        permanent: bool = False,
    ) -> None:
        try:
            await self._queue.fail(lease, error, retry_after_s, permanent=permanent)
        except Exception as exc:  # the lease will expire and reclaim will requeue it
            log.error("job_fail_report_failed", job_id=str(lease.job_id), error=describe_error(exc))

    async def _safe_defer(self, lease: JobLease, delay_s: int) -> None:
        try:
            await self._queue.defer(lease, delay_s)
        except Exception as exc:  # the lease expires and reclaim requeues it (one attempt)
            log.error("job_defer_failed", job_id=str(lease.job_id), error=describe_error(exc))

    async def _safe_release(self, lease: JobLease) -> None:
        try:
            await self._queue.release(lease)
        except Exception as exc:
            log.error("job_release_failed", job_id=str(lease.job_id), error=describe_error(exc))

    # ------------------------------------------------------------ shutdown
    async def _drain(self) -> None:
        """SIGTERM path: no new claims (the loop has exited); give running jobs the grace period,
        then cancel their handlers and release their leases so another worker can start them
        immediately instead of after lease expiry."""
        supervisors = set(self._slots)
        if not supervisors:
            return
        log.info("worker_draining", running=len(supervisors))
        _, pending = await asyncio.wait(supervisors, timeout=self._cfg.shutdown_grace_s)
        if not pending:
            return
        for task in pending:
            slot = self._slots.get(task)
            if slot is not None and slot.work is not None:
                slot.abandoned = True
                slot.work.cancel()
        _, stuck = await asyncio.wait(pending, timeout=self._cfg.release_timeout_s)
        for task in stuck:  # a handler ignoring cancellation: its lease expires, reclaim wins
            task.cancel()
        log.info("worker_drained", released=len(pending) - len(stuck), abandoned=len(stuck))

    # ------------------------------------------------------------ upkeep
    async def _maintenance_loop(self, stop: asyncio.Event) -> None:
        """Reclaim and move-finished in separate tasks: `move_finished_jobs` waits on an
        advisory lock every replica contends for, and reclaim (which un-sticks jobs of crashed
        workers) must never queue behind it."""
        maintenance = self._maintenance
        if maintenance is None:
            return
        await asyncio.gather(
            self._every(stop, self._cfg.reclaim_interval_s, "reclaim_expired",
                        maintenance.reclaim_expired),
            self._every(stop, self._cfg.move_finished_interval_s, "move_finished",
                        maintenance.move_finished),
        )  # fmt: skip

    async def _every(
        self,
        stop: asyncio.Event,
        interval_s: float,
        name: str,
        step: Callable[[], Awaitable[int]],
    ) -> None:
        while not stop.is_set():
            await self._upkeep(name, step)
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(stop.wait(), timeout=interval_s)

    @staticmethod
    async def _upkeep(name: str, step: Callable[[], Awaitable[int]]) -> None:
        try:
            count = await step()
        except Exception as exc:
            log.warning("queue_upkeep_failed", step=name, error=describe_error(exc))
            return
        if count:
            log.info("queue_upkeep", step=name, count=count)
