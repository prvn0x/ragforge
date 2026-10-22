"""Periodic worker upkeep beside the job runtime: connector scheduling and storage GC.

Each periodic step runs in its OWN task, so a long GC drain never delays the scheduler. A failing
step is logged (exception type only: messages can carry bound SQL values) and retried at its next
interval; it never stops the loop or the worker. Every run is bounded by the step's own timeout
(and every GC round by its own), so a hung connection cannot stop a step forever. Each
successful run is recorded in `UpkeepHealth`; a scheduler that has not succeeded for a while is
reported as a rate-limited error log, NOT through liveness (a scheduler failure is usually the
database, which a restart cannot fix, and restarting every replica would kill in-flight jobs).

Several workers may run these at once without leader election:
- scheduling goes through `rf.claim_due_connectors` / `rf.claim_reconcile_connectors` (slot
  0008), which skip connectors whose job is already queued or running and move each claimed
  connector's next time forward by a short claim lease under row locks (SKIP LOCKED): replicas
  never double-schedule, the scheduler never starts a second listing beside a running one, and a
  queued retry's backoff is never reset. (A manual "sync now" through the API can still run
  beside a scheduled listing; see slot 0008.) Failure backoff lives in 0008's dead-listing
  trigger, so a claim whose enqueue is lost costs one short lease, never the backoff;
- `rf.gc_embeddings` and `rf.purge_tombstones` are bounded per call and idempotent. GC's victim
  selection uses SKIP LOCKED; its retired-space purge does not, so two replicas can collide on it
  (one fails with lock_timeout and retries at its next interval).
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Protocol
from uuid import UUID

from ragforge_core.domain.models import JobPriority, JobSpec
from ragforge_core.domain.ports import JobQueue, TransactionManager
from ragforge_core.observability.logging import get_logger

log = get_logger(__name__)

SYNC_CONNECTOR = "sync_connector"  # services.sync.policy.KIND_SYNC_CONNECTOR
RECONCILE = "reconcile_deletes"  # services.sync.policy.KIND_RECONCILE
GC_BATCH = 5000
PURGE_BATCH = 1000
MAX_DRAIN_ROUNDS = 200  # per run: at most 1M GC rows every interval, then yield until the next
DRAIN_PAUSE_S = 0.05  # between batches, so a drain never monopolises the DB or the event loop
DRAIN_ROUND_TIMEOUT_S = 150.0  # one GC/purge batch; rf_worker's statement_timeout is 120 s


class ConnectorClaims(Protocol):
    async def claim_due_connectors(
        self, limit: int = 100, lease_s: int = 300
    ) -> Sequence[tuple[UUID, UUID]]: ...

    async def claim_reconcile_connectors(
        self, limit: int = 100, interval_s: int = 86_400, lease_s: int = 300
    ) -> Sequence[tuple[UUID, UUID]]: ...


Step = Callable[[], Awaitable[int]]


@dataclass(frozen=True, slots=True)
class Periodic:
    name: str
    every_s: float
    step: Step
    timeout_s: float  # a run that takes longer is abandoned (cancelled) and retried next time
    first_after_s: float = 0.0


class UpkeepHealth:
    """When each step last completed (success only). A step that never completed counts from
    process start, so a fresh worker is not reported dead before its first run."""

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._started = clock()
        self._last_ok: dict[str, float] = {}

    def mark(self, name: str) -> None:
        self._last_ok[name] = self._clock()

    def age(self, name: str) -> float:
        return self._clock() - self._last_ok.get(name, self._started)


def _connector_spec(kind: str, priority: JobPriority, connector_id: UUID) -> JobSpec:
    """`connector_id` stays unset: only page-level jobs count against the per-connector in-flight
    cap (handoff A), and a listing must not take one of its own pages' two slots."""
    return JobSpec(
        kind=kind,
        priority=priority,
        payload={"connector_id": str(connector_id)},
        dedupe_key=f"{kind}:{connector_id}",
    )


def sync_connector_spec(connector_id: UUID) -> JobSpec:
    return _connector_spec(SYNC_CONNECTOR, JobPriority.incremental, connector_id)


def reconcile_spec(connector_id: UUID) -> JobSpec:
    return _connector_spec(RECONCILE, JobPriority.maintenance, connector_id)


async def _enqueue_each(
    claimed: Sequence[tuple[UUID, UUID]],
    spec: Callable[[UUID], JobSpec],
    queue: JobQueue,
    tx: TransactionManager,
) -> int:
    """Enqueue in each connector's own tenant transaction (the tenant comes from the claimed
    row). A failed enqueue only delays that connector by one lease; the others still run."""
    count = 0
    for connector_id, tenant_id in claimed:
        try:
            async with tx.tenant(tenant_id) as t:
                await queue.enqueue(t, spec(connector_id))
        except Exception as exc:
            log.warning(
                "worker_schedule_enqueue_failed",
                connector_id=str(connector_id),
                error=type(exc).__name__,
            )
            continue
        count += 1
    return count


MAX_CLAIM_BATCHES = 10  # per scheduler run: a backlog drains in one run (within its timeout)


async def _claim_and_enqueue(
    claim: Callable[[], Awaitable[Sequence[tuple[UUID, UUID]]]],
    spec: Callable[[UUID], JobSpec],
    queue: JobQueue,
    tx: TransactionManager,
    *,
    limit: int,
    max_batches: int,
) -> int:
    """Claim and enqueue batch after batch while batches come back FULL (a backlog: e.g.
    thousands of connectors after an outage), up to `max_batches`; the step's own timeout
    bounds the whole run. One batch per interval would cap scheduling at `limit` connectors
    per interval per replica."""
    total = 0
    for _ in range(max_batches):
        claimed = await claim()
        total += await _enqueue_each(claimed, spec, queue, tx)
        if len(claimed) < limit:
            break
    return total


async def schedule_due_connectors(
    claims: ConnectorClaims,
    queue: JobQueue,
    tx: TransactionManager,
    *,
    lease_s: int,
    limit: int = 100,
    max_batches: int = MAX_CLAIM_BATCHES,
) -> int:
    """One sync_connector per claimed connector. Success sets next_poll_at = now() + poll
    interval (C2); a dead listing is backed off by slot 0008's trigger (doubling, max 1 day)."""
    return await _claim_and_enqueue(
        lambda: claims.claim_due_connectors(limit, lease_s),
        sync_connector_spec,
        queue,
        tx,
        limit=limit,
        max_batches=max_batches,
    )


async def schedule_reconciles(
    claims: ConnectorClaims,
    queue: JobQueue,
    tx: TransactionManager,
    *,
    interval_s: int,
    lease_s: int,
    limit: int = 100,
    max_batches: int = MAX_CLAIM_BATCHES,
) -> int:
    """Nightly reconcile_deletes per active connector (handoff C2, request 5). The claim takes a
    short `lease_s`; slot 0008 schedules the next run when this one ends."""
    return await _claim_and_enqueue(
        lambda: claims.claim_reconcile_connectors(limit, interval_s, lease_s),
        reconcile_spec,
        queue,
        tx,
        limit=limit,
        max_batches=max_batches,
    )


async def sleep_until_stopped(stop: asyncio.Event, seconds: float) -> bool:
    """Sleep up to `seconds`; True as soon as `stop` is set."""
    if seconds > 0:
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(stop.wait(), timeout=seconds)
    return stop.is_set()


async def drain(
    step: Step,
    *,
    done: Callable[[int], bool],
    stop: asyncio.Event,
    max_rounds: int = MAX_DRAIN_ROUNDS,
    pause_s: float = DRAIN_PAUSE_S,
    round_timeout_s: float = DRAIN_ROUND_TIMEOUT_S,
) -> int:
    """Call `step` (each call its own committed transaction) until `done(result)`, `stop`, or
    `max_rounds`; whatever is left is picked up at the next interval. A round that exceeds
    `round_timeout_s` raises TimeoutError (the step's committed rounds stay committed)."""
    total = 0
    for _ in range(max_rounds):
        if stop.is_set():
            break
        async with asyncio.timeout(round_timeout_s):
            n = await step()
        total += n
        if done(n):
            break
        if await sleep_until_stopped(stop, pause_s):
            break
    else:
        log.info("worker_upkeep_drain_capped", rounds=max_rounds)
    return total


async def run_periodic(
    stop: asyncio.Event,
    task: Periodic,
    *,
    health: UpkeepHealth | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> None:
    if await sleep_until_stopped(stop, task.first_after_s):
        return
    while not stop.is_set():
        started = clock()
        timed_out = False
        try:
            async with asyncio.timeout(task.timeout_s):
                n = await task.step()
        except TimeoutError:
            timed_out = True
            log.error("worker_upkeep_timeout", step=task.name, timeout_s=task.timeout_s)
        except Exception as exc:  # never kill the worker over upkeep
            log.warning("worker_upkeep_failed", step=task.name, error=type(exc).__name__)
        else:
            if health is not None:
                health.mark(task.name)
            if n:
                log.info("worker_upkeep", step=task.name, rows=n)
        # After a timeout wait a FULL interval: a timeout means the database is degraded, and
        # re-running at once would claim another batch (pushing those connectors out by a
        # lease) only to be cancelled mid-enqueue again, starving polling until it recovers.
        delay = task.every_s if timed_out else task.every_s - (clock() - started)
        if await sleep_until_stopped(stop, delay):
            return


async def run_upkeep(
    stop: asyncio.Event, tasks: Sequence[Periodic], health: UpkeepHealth | None = None
) -> None:
    """Every step in its own task: a slow step never delays another step's schedule."""
    async with asyncio.TaskGroup() as group:
        for task in tasks:
            group.create_task(run_periodic(stop, task, health=health), name=f"upkeep:{task.name}")
