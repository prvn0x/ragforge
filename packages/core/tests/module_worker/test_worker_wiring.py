"""Worker wiring on fakes: handler registration, scheduling fan-out, upkeep loop behaviour and
the schema watch. The SQL semantics of the claims live in tests/test_scheduling.py."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from typing import Any, cast
from uuid import UUID, uuid4

import pytest
import structlog.testing

from ragforge_contracts.api import Readiness
from ragforge_core.agent.nodes import SUMMARIZE_JOB
from ragforge_core.apps.worker import main as worker_main
from ragforge_core.apps.worker.main import (
    MAX_RUNTIME_S,
    SCHEDULER_MAX_AGE_S,
    SCHEDULER_STALE_LOG_EVERY_S,
    SCHEDULER_STEP,
    _supervised,
    background_db_holders,
    liveness_tick,
    register_handlers,
    upkeep_tasks,
    watch_schema,
)
from ragforge_core.apps.worker.upkeep import (
    Periodic,
    UpkeepHealth,
    drain,
    run_upkeep,
    schedule_due_connectors,
    schedule_reconciles,
)
from ragforge_core.bootstrap import WorkerContainer
from ragforge_core.config import Settings
from ragforge_core.domain.models import JobPriority, JobSpec
from ragforge_core.domain.ports import JobQueue, TransactionManager
from ragforge_core.services.jobs import JobHandlers
from ragforge_core.services.sync.policy import (
    KIND_RECONCILE,
    KIND_SYNC_CONNECTOR,
    KIND_SYNC_PAGE,
    SYNC_KINDS,
)


class FakeTx:
    def __init__(self) -> None:
        self.tenants: list[UUID | None] = []

    @asynccontextmanager
    async def _open(self, tenant: UUID | None) -> AsyncIterator[UUID | None]:
        self.tenants.append(tenant)
        yield tenant

    def tenant(self, tenant_id: UUID) -> Any:
        return self._open(tenant_id)

    def system(self) -> Any:
        return self._open(None)


class FakeQueue:
    def __init__(self, due: list[tuple[UUID, UUID]], fail_for: UUID | None = None) -> None:
        self.due = due
        self.fail_for = fail_for
        self.enqueued: list[tuple[UUID | None, JobSpec]] = []
        self.claims: list[tuple[object, ...]] = []

    async def claim_due_connectors(
        self, limit: int = 100, lease_s: int = 300
    ) -> Sequence[tuple[UUID, UUID]]:
        self.claims.append(("poll", limit, lease_s))
        return self.due[:limit]

    async def claim_reconcile_connectors(
        self, limit: int = 100, interval_s: int = 86_400, lease_s: int = 300
    ) -> Sequence[tuple[UUID, UUID]]:
        self.claims.append(("reconcile", limit, interval_s, lease_s))
        return self.due[:limit]

    async def enqueue(self, tx: Any, spec: JobSpec) -> UUID:
        if self.fail_for is not None and spec.payload["connector_id"] == str(self.fail_for):
            raise RuntimeError("boom")
        self.enqueued.append((tx, spec))
        return uuid4()


class FakeContainer:
    sync = object()


def _queue(q: FakeQueue) -> JobQueue:
    return cast(JobQueue, q)


def _tx(t: FakeTx) -> TransactionManager:
    return cast(TransactionManager, t)


def test_register_handlers_covers_sync_kinds_and_summarize() -> None:
    handlers = JobHandlers()
    register_handlers(handlers, cast(WorkerContainer, FakeContainer()))
    assert handlers.kinds() == SYNC_KINDS | {SUMMARIZE_JOB}
    # run_sync_job ends every sync job itself: the runtime must not complete it again.
    assert all(handlers.reports_outcome(kind) for kind in SYNC_KINDS)
    assert not handlers.reports_outcome(SUMMARIZE_JOB)  # a no-op: the runtime completes it


async def test_a_crashing_background_task_is_logged_and_never_breaks_shutdown() -> None:
    async def crash() -> None:
        raise RuntimeError("listener bug")

    task = _supervised("listener", crash())
    await asyncio.wait({task})
    assert task.exception() is None  # logged as worker_background_task_crashed, not re-raised


async def test_poll_scheduling_enqueues_in_each_connectors_tenant() -> None:
    (c1, t1), (c2, t2) = (uuid4(), uuid4()), (uuid4(), uuid4())
    queue, tx = FakeQueue([(c1, t1), (c2, t2)]), FakeTx()

    n = await schedule_due_connectors(queue, _queue(queue), _tx(tx), lease_s=300)

    assert n == 2
    assert queue.claims == [("poll", 100, 300)]
    assert [tenant for tenant, _ in queue.enqueued] == [t1, t2]
    spec = queue.enqueued[0][1]
    assert spec.kind == KIND_SYNC_CONNECTOR
    assert spec.priority is JobPriority.incremental
    assert spec.dedupe_key == f"{KIND_SYNC_CONNECTOR}:{c1}"
    assert spec.payload == {"connector_id": str(c1)}
    assert spec.connector_id is None  # listings never take a page in-flight slot


async def test_reconcile_scheduling_uses_maintenance_priority() -> None:
    c, t = uuid4(), uuid4()
    queue, tx = FakeQueue([(c, t)]), FakeTx()

    n = await schedule_reconciles(queue, _queue(queue), _tx(tx), interval_s=86_400, lease_s=240)
    assert n == 1

    # the claim takes only a short lease; slot 0008 schedules the next run when this one ends
    assert queue.claims == [("reconcile", 100, 86_400, 240)]
    ((tenant, spec),) = queue.enqueued
    assert tenant == t
    assert spec.kind == KIND_RECONCILE
    assert spec.priority is JobPriority.maintenance
    assert spec.dedupe_key == f"{KIND_RECONCILE}:{c}"


async def test_one_failed_enqueue_does_not_stop_the_others() -> None:
    bad, good = uuid4(), uuid4()
    queue = FakeQueue([(bad, uuid4()), (good, uuid4())], fail_for=bad)

    assert await schedule_due_connectors(queue, _queue(queue), _tx(FakeTx()), lease_s=300) == 1
    assert [s.payload["connector_id"] for _, s in queue.enqueued] == [str(good)]


async def test_a_failing_step_keeps_running_and_a_slow_step_never_delays_another() -> None:
    stop = asyncio.Event()
    calls = {"fail": 0, "fast": 0}
    slow_started = asyncio.Event()

    async def failing() -> int:
        calls["fail"] += 1
        raise RuntimeError("db down")

    async def fast() -> int:
        calls["fast"] += 1
        return 0

    async def slow() -> int:
        slow_started.set()
        await stop.wait()  # e.g. a long GC drain
        return 0

    tasks = [
        Periodic("slow", 60.0, slow, timeout_s=60.0),
        Periodic("fail", 0.01, failing, timeout_s=1.0),
        Periodic("fast", 0.01, fast, timeout_s=1.0),
    ]
    runner = asyncio.create_task(run_upkeep(stop, tasks))
    await slow_started.wait()
    await asyncio.sleep(0.1)
    stop.set()
    await asyncio.wait_for(runner, timeout=2)

    assert calls["fail"] >= 3
    assert calls["fast"] >= 3


async def test_first_after_delays_the_first_run() -> None:
    stop = asyncio.Event()
    ran: list[str] = []

    async def step() -> int:
        ran.append("x")
        return 0

    runner = asyncio.create_task(
        run_upkeep(stop, [Periodic("late", 60.0, step, timeout_s=1.0, first_after_s=60.0)])
    )
    await asyncio.sleep(0.05)
    stop.set()
    await asyncio.wait_for(runner, timeout=2)
    assert ran == []


async def test_drain_stops_on_done_on_stop_and_at_the_round_cap() -> None:
    stop = asyncio.Event()
    results = iter([5, 3, 0, 9])

    async def step() -> int:
        return next(results)

    assert await drain(step, done=lambda n: n == 0, stop=stop, pause_s=0) == 8

    async def forever() -> int:
        return 1

    assert await drain(forever, done=lambda n: n == 0, stop=stop, max_rounds=4, pause_s=0) == 4

    stop.set()
    assert await drain(forever, done=lambda n: n == 0, stop=stop, pause_s=0) == 0


def _readiness(*, ready: bool, schema: int | None) -> Readiness:
    return Readiness(
        ready=ready,
        schema_version=schema,
        required_schema=8,
        code_version=1,
        min_code_version=0 if schema is not None else None,
        reason=None if ready else "x",
    )


async def test_watch_schema_stops_on_incompatibility_but_not_on_an_unreachable_db(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sequence = iter(
        [
            _readiness(ready=True, schema=8),
            _readiness(ready=False, schema=None),  # DB blip: keep running
            _readiness(ready=False, schema=9),  # contract migration: stop
        ]
    )

    async def fake_check(engine: object) -> Readiness:
        return next(sequence)

    monkeypatch.setattr(worker_main, "check_readiness", fake_check)
    stop, incompatible = asyncio.Event(), asyncio.Event()

    await asyncio.wait_for(
        watch_schema(cast(Any, None), stop, incompatible, every_s=0.001), timeout=2
    )

    assert stop.is_set()
    assert incompatible.is_set()


def test_worker_pool_floor_counts_every_concurrent_holder() -> None:
    """The floor's background count is derived from the tasks the worker really starts, so a new
    upkeep step without a bigger pool fails here instead of exhausting the pool in production."""
    from pydantic import ValidationError

    from ragforge_core.config import WORKER_BACKGROUND_CONNECTIONS, Settings

    settings = Settings(_env_file=None, worker_concurrency=4, db_pool_size=10)
    container = cast(WorkerContainer, _UpkeepContainer())
    tasks = upkeep_tasks(container, asyncio.Event(), settings)
    assert len(background_db_holders(tasks)) == WORKER_BACKGROUND_CONNECTIONS
    floor = 2 * 4 + WORKER_BACKGROUND_CONNECTIONS
    assert settings.worker_pool_floor == floor
    assert settings.worker_pool_size == floor + 1  # unset: floor + one spare
    explicit = Settings(_env_file=None, worker_concurrency=4, worker_db_pool_size=floor)
    assert explicit.worker_pool_size == floor
    with pytest.raises(ValidationError, match="worker_db_pool_size"):
        Settings(_env_file=None, worker_concurrency=4, worker_db_pool_size=floor - 1)


class _UpkeepContainer:
    tx = FakeTx()
    jobs = FakeQueue([])


async def test_a_hung_step_is_abandoned_at_its_timeout_and_runs_again() -> None:
    stop = asyncio.Event()
    calls = 0

    async def hangs() -> int:
        nonlocal calls
        calls += 1
        await asyncio.Event().wait()  # a half-open connection: never returns by itself
        return 0

    runner = asyncio.create_task(run_upkeep(stop, [Periodic("hangs", 0.01, hangs, timeout_s=0.05)]))
    await asyncio.sleep(0.3)
    stop.set()
    await asyncio.wait_for(runner, timeout=2)
    assert calls >= 3  # each run was cut off and the step kept being scheduled


async def test_a_hung_gc_round_is_bounded() -> None:
    async def hangs() -> int:
        await asyncio.Event().wait()
        return 0

    with pytest.raises(TimeoutError):
        await drain(hangs, done=lambda n: n == 0, stop=asyncio.Event(), round_timeout_s=0.05)


async def test_liveness_is_the_claim_loop_and_a_stale_scheduler_only_alerts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ops review r4 P2-3: a scheduler that keeps failing (usually the database) must not make
    every replica's liveness fail, which would restart them all in a loop and kill in-flight
    jobs. Liveness = the claim loop ticked; a stale scheduler is a rate-limited error log."""
    now = [1000.0]
    health = UpkeepHealth(clock=lambda: now[0])
    touched: list[bool] = []

    async def touch() -> None:
        touched.append(True)

    monkeypatch.setattr(worker_main, "_touch_heartbeat", touch)
    tick = liveness_tick(health, clock=lambda: now[0])

    with structlog.testing.capture_logs() as logs:
        await tick()  # fresh process: no alert
        now[0] += SCHEDULER_MAX_AGE_S + 1
        await tick()  # stale scheduler: still alive, one alert
        await tick()  # ...and no second alert within the rate limit
        now[0] += SCHEDULER_STALE_LOG_EVERY_S
        await tick()  # the rate limit passed: alert again
        health.mark(SCHEDULER_STEP)
        await tick()  # fresh again: no alert
    assert touched == [True] * 5
    assert [e["event"] for e in logs].count("worker_scheduler_stale") == 2


def test_claim_leases_outlive_the_step_that_enqueues() -> None:
    """Ops review r4 P3-4: a lease shorter than the scheduler step's own timeout would let a
    second replica claim the same connector while the first is still enqueueing it."""
    fast = Settings(sync_poll_interval_s=30)
    slow = Settings(sync_poll_interval_s=900)
    assert worker_main.poll_claim_lease_s(fast) >= 2 * worker_main.SCHEDULE_TIMEOUT_S
    assert worker_main.poll_claim_lease_s(slow) == 900
    assert worker_main.RECONCILE_CLAIM_LEASE_S >= 2 * worker_main.RECONCILE_TIMEOUT_S


async def test_successful_runs_are_recorded_failed_ones_are_not() -> None:
    stop = asyncio.Event()
    health = UpkeepHealth()

    async def ok() -> int:
        return 0

    async def bad() -> int:
        raise RuntimeError("db down")

    runner = asyncio.create_task(
        run_upkeep(
            stop,
            [Periodic("ok", 0.01, ok, timeout_s=1.0), Periodic("bad", 0.01, bad, timeout_s=1.0)],
            health,
        )
    )
    await asyncio.sleep(0.1)
    stop.set()
    await asyncio.wait_for(runner, timeout=2)
    assert health.age("ok") < 0.1
    assert health.age("bad") >= 0.1  # never marked: counts from process start


# ------------------------------------------------------------------ round 5 (ops)
def test_every_sync_kind_has_a_max_runtime_bound() -> None:
    handlers = JobHandlers()
    register_handlers(handlers, cast(WorkerContainer, FakeContainer()))
    assert handlers.max_runtime_s(KIND_SYNC_CONNECTOR) == MAX_RUNTIME_S[KIND_SYNC_CONNECTOR]
    assert handlers.max_runtime_s(KIND_SYNC_PAGE) == 15 * 60.0
    assert set(MAX_RUNTIME_S) == SYNC_KINDS


async def test_a_timed_out_step_waits_a_full_interval_before_running_again() -> None:
    """A timeout means a degraded database: re-running at once would claim another batch only
    to be cancelled mid-enqueue again (a cancel storm that starves polling)."""
    stop = asyncio.Event()
    calls = 0

    async def hangs() -> int:
        nonlocal calls
        calls += 1
        await asyncio.Event().wait()
        return 0

    runner = asyncio.create_task(run_upkeep(stop, [Periodic("hangs", 0.3, hangs, timeout_s=0.05)]))
    await asyncio.sleep(0.5)  # back to back would be ~10 runs; a full interval allows 2
    stop.set()
    await asyncio.wait_for(runner, timeout=2)
    assert calls == 2


async def test_an_upkeep_crash_stops_the_worker_so_it_restarts() -> None:
    """A failed step is isolated by upkeep; a crash of the upkeep task itself would leave the
    process without a scheduler for its whole life. The hook stops the worker (exit 1)."""
    crashed: list[bool] = []

    async def crash() -> None:
        raise RuntimeError("upkeep bug")

    task = _supervised("upkeep", crash(), on_crash=lambda: crashed.append(True))
    await asyncio.wait({task})
    assert task.exception() is None
    assert crashed == [True]


async def test_the_scheduler_keeps_claiming_while_batches_come_back_full() -> None:
    """Ops review r6 P3-4: one batch per interval capped scheduling at `limit` connectors per
    interval per replica, so a backlog (after an outage) took many intervals to drain."""
    backlog = [(uuid4(), uuid4()) for _ in range(5)]

    class Draining(FakeQueue):
        async def claim_due_connectors(
            self, limit: int = 100, lease_s: int = 300
        ) -> Sequence[tuple[UUID, UUID]]:
            self.claims.append(("poll", limit, lease_s))
            taken, self.due = self.due[:limit], self.due[limit:]
            return taken

    queue = Draining(list(backlog))
    n = await schedule_due_connectors(queue, _queue(queue), _tx(FakeTx()), lease_s=300, limit=2)
    assert n == 5
    assert len(queue.claims) == 3  # 2 + 2 + 1: the short batch ends the run

    capped = Draining([(uuid4(), uuid4()) for _ in range(10)])
    n = await schedule_due_connectors(
        capped, _queue(capped), _tx(FakeTx()), lease_s=300, limit=2, max_batches=2
    )
    assert n == 4  # bounded per run; the rest is due again next interval


def test_the_api_pool_knob_never_inflates_the_worker_pool() -> None:
    from ragforge_core.config import Settings

    small = Settings(_env_file=None, worker_concurrency=4, db_pool_size=10)
    big_api = Settings(_env_file=None, worker_concurrency=4, db_pool_size=100)
    assert big_api.worker_pool_size == small.worker_pool_size == small.worker_pool_floor + 1
