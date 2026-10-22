"""Worker runtime: concurrency limit, heartbeat loss stops work, outcome mapping, graceful
shutdown, starvation-guard cadence, and NOTIFY wake-up end to end."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Sequence
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import dataclass, field
from uuid import UUID, uuid4

import psycopg
import pytest
from sqlalchemy import text

from module_a.a_support import enqueue, job_row, spec
from ragforge_core.adapters.db.jobs import PgJobQueue
from ragforge_core.adapters.db.tx import connection
from ragforge_core.apps.worker.listener import NotificationListener
from ragforge_core.config import Settings
from ragforge_core.db.engine import SqlTransactionManager
from ragforge_core.domain.errors import LeaseLost, UpstreamRateLimited
from ragforge_core.domain.models import JobLease, JobSpec
from ragforge_core.domain.ports import Tx
from ragforge_core.services.jobs import (
    DuplicateHandler,
    JobHandlers,
    PermanentJobError,
    WorkerConfig,
    WorkerRuntime,
    describe_error,
)
from ragforge_core.services.jobs.registry import DEFAULT_MAX_RUNTIME_S
from ragforge_core.services.jobs.runtime import MAX_RUNTIME_ERROR, UNKNOWN_KIND_DELAY_S

Conn = psycopg.Connection[tuple[object, ...]]

FAST = WorkerConfig(
    concurrency=3,
    poll_interval_s=0.02,
    lease_s=1.0,
    heartbeat_interval_s=0.05,
    shutdown_grace_s=0.3,
    release_timeout_s=1.0,
)


# ------------------------------------------------------------------ fakes
@dataclass
class FakeQueue:
    """In-memory FairJobQueue + QueueMaintenance recording every call."""

    queued: list[JobLease] = field(default_factory=list)
    calls: list[tuple[object, ...]] = field(default_factory=list)
    claims: list[tuple[int, bool]] = field(default_factory=list)
    heartbeat_ok: bool = True
    finished: set[UUID] = field(default_factory=set)
    all_finished: asyncio.Event = field(default_factory=asyncio.Event)
    expected: int = -1  # set all_finished once this many jobs completed

    def add(self, kind: str = "k", n: int = 1) -> list[JobLease]:
        leases = [JobLease(uuid4(), uuid4(), kind, {}, 1, 1) for _ in range(n)]
        self.queued.extend(leases)
        return leases

    def outcome(self, lease: JobLease) -> list[tuple[object, ...]]:
        return [c for c in self.calls if c[1] == lease.job_id]

    async def enqueue(self, tx: Tx, spec: JobSpec) -> UUID:
        raise NotImplementedError

    async def claim(
        self, worker_id: str, n: int, *, prefer_low: bool = False
    ) -> Sequence[JobLease]:
        self.claims.append((n, prefer_low))
        taken, self.queued = self.queued[:n], self.queued[n:]
        return taken

    async def assert_lease(self, tx: Tx, lease: JobLease) -> None:
        if lease.job_id in self.finished or not self.heartbeat_ok:
            raise LeaseLost("gone")

    async def report_progress(self, tx: Tx, lease: JobLease, progress: dict[str, object]) -> None:
        self.calls.append(("progress", lease.job_id))

    async def complete(self, tx: Tx, lease: JobLease) -> None:
        self.finished.add(lease.job_id)
        self.calls.append(("complete", lease.job_id))
        if len(self.finished) == self.expected:
            self.all_finished.set()

    async def heartbeat(self, lease: JobLease) -> bool:
        return self.heartbeat_ok

    async def release(self, lease: JobLease) -> None:
        self.calls.append(("release", lease.job_id))

    async def defer(self, lease: JobLease, delay_s: int) -> None:
        self.calls.append(("defer", lease.job_id, delay_s))

    async def fail(
        self, lease: JobLease, error: str, retry_after_s: float | None, *, permanent: bool = False
    ) -> None:
        self.calls.append(("fail", lease.job_id, error, retry_after_s, permanent))

    async def reclaim_expired(self, limit: int = 500) -> int:
        self.calls.append(("reclaim", None))
        return 0

    async def move_finished(self) -> int:
        self.calls.append(("move", None))
        return 0


class FakeTx:
    """TransactionManager whose transactions are inert tokens (FakeQueue ignores them)."""

    def tenant(self, tenant_id: UUID) -> AbstractAsyncContextManager[Tx]:
        return self._tx()

    def system(self) -> AbstractAsyncContextManager[Tx]:
        return self._tx()

    @asynccontextmanager
    async def _tx(self) -> AsyncIterator[Tx]:
        yield object()


def _runtime(queue: FakeQueue, handlers: JobHandlers, config: WorkerConfig = FAST) -> WorkerRuntime:
    return WorkerRuntime(
        worker_id="w",
        queue=queue,
        transactions=FakeTx(),
        handlers=handlers,
        maintenance=queue,
        config=config,
    )


async def _run_until(runtime: WorkerRuntime, done: asyncio.Event) -> None:
    timeout = 5.0
    stop = asyncio.Event()
    task = asyncio.create_task(runtime.run(stop))
    try:
        await asyncio.wait_for(done.wait(), timeout)
    finally:
        stop.set()
        await asyncio.wait_for(task, timeout)


# ------------------------------------------------------------------ registry
def test_registry_rejects_duplicates_and_empty_kinds() -> None:
    handlers = JobHandlers()

    async def handler(lease: JobLease) -> None:
        return None

    handlers.register("sync_page", handler)
    with pytest.raises(DuplicateHandler):
        handlers.register("sync_page", handler)
    with pytest.raises(ValueError, match="non-empty"):
        handlers.register("", handler)
    assert handlers.kinds() == {"sync_page"}


def test_error_descriptions_are_short_single_line() -> None:
    err = describe_error(RuntimeError("first line\nsecret second line " + "x" * 1000))
    assert err.startswith("RuntimeError: first line")
    assert "secret" not in err
    assert len(err) <= 500


# ------------------------------------------------------------------ concurrency + outcomes
async def test_concurrency_limit_is_never_exceeded() -> None:
    queue = FakeQueue(expected=10)
    jobs = queue.add(n=10)
    running = 0
    peak = 0

    async def handler(lease: JobLease) -> None:
        nonlocal running, peak
        running += 1
        peak = max(peak, running)
        await asyncio.sleep(0.02)
        running -= 1

    handlers = JobHandlers()
    handlers.register("k", handler)
    runtime = _runtime(queue, handlers)

    await _run_until(runtime, queue.all_finished)
    assert peak == FAST.concurrency
    assert all(n <= FAST.concurrency for n, _ in queue.claims)
    # Handlers that return without completing are completed by the runtime (fenced).
    assert {c[1] for c in queue.calls if c[0] == "complete"} == {j.job_id for j in jobs}


async def test_lost_lease_cancels_the_handler_and_reports_nothing() -> None:
    queue = FakeQueue()
    (lease,) = queue.add()
    cancelled = asyncio.Event()

    async def handler(lease: JobLease) -> None:
        queue.heartbeat_ok = False  # another worker reclaimed it while we were busy
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            cancelled.set()
            raise

    handlers = JobHandlers()
    handlers.register("k", handler)
    await _run_until(_runtime(queue, handlers), cancelled)
    assert queue.outcome(lease) == []  # no complete, no fail, no release


@pytest.mark.parametrize(
    ("exc", "expected"),
    [
        (PermanentJobError("404 page"), ("PermanentJobError: 404 page", None, True)),
        (UpstreamRateLimited(30.0), ("UpstreamRateLimited: retry after 30.0s", 30.0, False)),
        (KeyError("boom"), ("KeyError: 'boom'", None, False)),
    ],
)
async def test_handler_exceptions_map_to_fail(
    exc: Exception, expected: tuple[str, float | None, bool]
) -> None:
    queue = FakeQueue()
    (lease,) = queue.add()
    seen = asyncio.Event()

    async def handler(lease: JobLease) -> None:
        seen.set()
        raise exc

    handlers = JobHandlers()
    handlers.register("k", handler)
    await _run_until(_runtime(queue, handlers), seen)
    await asyncio.sleep(0.05)
    assert queue.outcome(lease) == [("fail", lease.job_id, *expected)]


async def test_handler_raising_lease_lost_is_discarded() -> None:
    queue = FakeQueue()
    (lease,) = queue.add()
    seen = asyncio.Event()

    async def handler(lease: JobLease) -> None:
        seen.set()
        raise LeaseLost("fenced")

    handlers = JobHandlers()
    handlers.register("k", handler)
    await _run_until(_runtime(queue, handlers), seen)
    assert queue.outcome(lease) == []


@pytest.mark.parametrize(
    "outcome",
    [None, RuntimeError("fail() itself failed: connection reset"), LeaseLost("fenced")],
)
async def test_self_reporting_handlers_are_never_completed_or_failed_again(
    outcome: Exception | None,
) -> None:
    """`reports_outcome=True` (the sync kinds): the handler already completed or failed the job
    in its own fenced transaction. The runtime must not complete it again (that only raised
    lease-lost in Postgres for every job), and an error from the handler's own reporting must
    not be re-reported as a transient failure (that turned permanent failures into retries)."""
    queue = FakeQueue()
    (lease,) = queue.add()
    seen = asyncio.Event()

    async def handler(lease: JobLease) -> None:
        seen.set()
        if outcome is not None:
            raise outcome

    handlers = JobHandlers()
    handlers.register("k", handler, reports_outcome=True)
    await _run_until(_runtime(queue, handlers), seen)
    await asyncio.sleep(0.05)
    assert queue.outcome(lease) == []


def test_one_permanent_job_error_class_for_runtime_and_sync() -> None:
    from ragforge_core.services.sync.deps import PermanentJobError as SyncPermanent

    assert SyncPermanent is PermanentJobError


async def test_unknown_kind_is_deferred_without_charging_an_attempt() -> None:
    """Rolling deploy: an old worker claims a kind only the new release knows. Failing it charged
    an attempt per claim (a new kind could go dead); it is deferred instead (no charge, and a
    delay so this worker does not re-claim it in a loop)."""
    queue = FakeQueue()
    handlers = JobHandlers()

    async def other(lease: JobLease) -> None:
        return None

    handlers.register("known", other)
    (lease,) = queue.add(kind="from-a-newer-release")
    runtime = _runtime(queue, handlers)
    stop = asyncio.Event()
    task = asyncio.create_task(runtime.run(stop))
    await asyncio.sleep(0.1)
    stop.set()
    await task
    assert queue.outcome(lease) == [("defer", lease.job_id, UNKNOWN_KIND_DELAY_S)]


# ------------------------------------------------------------------ shutdown
async def test_shutdown_finishes_quick_jobs_and_releases_slow_ones() -> None:
    queue = FakeQueue()
    quick, slow = queue.add(kind="quick")[0], queue.add(kind="slow")[0]
    started = asyncio.Event()

    async def quick_handler(lease: JobLease) -> None:
        await asyncio.sleep(0.1)  # finishes inside the 0.3 s grace

    async def slow_handler(lease: JobLease) -> None:
        started.set()
        await asyncio.sleep(30)  # never finishes: must be cancelled and released

    handlers = JobHandlers()
    handlers.register("quick", quick_handler)
    handlers.register("slow", slow_handler)
    runtime = _runtime(queue, handlers)
    stop = asyncio.Event()
    task = asyncio.create_task(runtime.run(stop))
    await asyncio.wait_for(started.wait(), 5)
    claims_before = len(queue.claims)
    stop.set()  # SIGTERM
    await asyncio.wait_for(task, 5)
    assert queue.outcome(quick) == [("complete", quick.job_id)]
    assert queue.outcome(slow) == [("release", slow.job_id)]
    assert len(queue.claims) == claims_before  # stopped claiming immediately
    assert runtime.running == 0


# ------------------------------------------------------------------ starvation guard cadence
async def test_runtime_sets_prefer_low_on_every_fourth_claimed_job() -> None:
    queue = FakeQueue()
    queue.add(n=12)
    done = asyncio.Event()

    async def handler(lease: JobLease) -> None:
        if len(queue.finished) >= 11:
            done.set()

    handlers = JobHandlers()
    handlers.register("k", handler)
    await _run_until(
        _runtime(queue, handlers, WorkerConfig(concurrency=1, poll_interval_s=0.01)), done
    )
    flags = [low for n, low in queue.claims if n][:12]
    assert flags == [False, False, False, True] * 3


async def test_maintenance_runs_reclaim_and_move() -> None:
    queue = FakeQueue()
    runtime = _runtime(queue, JobHandlers())
    stop = asyncio.Event()
    task = asyncio.create_task(runtime.run(stop))
    await asyncio.sleep(0.05)
    stop.set()
    await task
    names = [c[0] for c in queue.calls]
    assert "reclaim" in names
    assert "move" in names


# ------------------------------------------------------------------ end to end (real Postgres)
@pytest.mark.integration
async def test_notify_wakes_the_worker_and_the_fenced_handler_commits(
    settings: Settings,
    queue: PgJobQueue,
    txm: SqlTransactionManager,
    tenant: UUID,
    superuser: Conn,
) -> None:
    handled = asyncio.Event()

    async def handler(lease: JobLease) -> None:
        async with txm.tenant(lease.tenant_id) as tx:
            await queue.assert_lease(tx, lease)
            await connection(tx).execute(
                text(
                    "INSERT INTO rf.audit_log (tenant_id, actor, action, target) "
                    "VALUES (:t, 'test', 'e2e', :target)"
                ),
                {"t": lease.tenant_id, "target": f"job:{lease.job_id}"},
            )
            await queue.report_progress(tx, lease, {"step": "done"})
            await queue.complete(tx, lease)
        handled.set()

    handlers = JobHandlers()
    handlers.register("test_job", handler)
    # A 60 s poll: only the NOTIFY can wake the loop within the test's timeout.
    runtime = WorkerRuntime(
        worker_id="e2e",
        queue=queue,
        transactions=txm,
        handlers=handlers,
        maintenance=queue,
        config=WorkerConfig(concurrency=2, poll_interval_s=60.0),
    )
    listener = NotificationListener(settings.dsn("worker", driver="postgresql"), runtime.wake)
    stop = asyncio.Event()
    listen_task = asyncio.create_task(listener.run(stop))
    await asyncio.wait_for(listener.connected.wait(), 10)
    run_task = asyncio.create_task(runtime.run(stop))
    await asyncio.sleep(0.2)  # let the loop go idle on its first (empty) claim

    job_id = await enqueue(txm, queue, tenant, spec("e2e"))
    try:
        await asyncio.wait_for(handled.wait(), 10)
    finally:
        stop.set()
        await asyncio.wait_for(asyncio.gather(run_task, listen_task), 10)
    assert job_row(superuser, job_id, "status, progress") == ("done", {"step": "done"})
    assert superuser.execute(
        "SELECT count(*) FROM rf.audit_log WHERE target = %s", (f"job:{job_id}",)
    ).fetchone() == (1,)


# ------------------------------------------------------------------ max runtime (ops r5 P2-3)
async def test_a_hung_handler_is_cancelled_at_its_max_runtime_and_failed_retryably() -> None:
    """The keepalive renews the lease while the DB answers, so only a wall-clock bound stops a
    handler that never returns from holding its slot (and its connector) forever."""
    queue = FakeQueue()
    (hung,) = queue.add("hang")
    (quick,) = queue.add("quick")  # needs the slot the hung job frees
    cancelled = asyncio.Event()
    quick_ran = asyncio.Event()

    async def hang(lease: JobLease) -> None:
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.set()
            raise

    async def run_quick(lease: JobLease) -> None:
        quick_ran.set()

    handlers = JobHandlers()
    handlers.register("hang", hang, max_runtime_s=0.2)
    handlers.register("quick", run_quick)
    one_slot = WorkerConfig(
        concurrency=1,
        poll_interval_s=0.02,
        lease_s=1.0,
        heartbeat_interval_s=0.05,
        shutdown_grace_s=0.3,
        release_timeout_s=1.0,
    )
    await _run_until(_runtime(queue, handlers, one_slot), quick_ran)

    assert cancelled.is_set()
    assert queue.outcome(hung) == [("fail", hung.job_id, MAX_RUNTIME_ERROR, None, False)]
    assert queue.outcome(quick) == [("complete", quick.job_id)]


def test_max_runtime_defaults_and_must_be_positive() -> None:
    async def handler(lease: JobLease) -> None:
        return None

    handlers = JobHandlers()
    handlers.register("k", handler)
    assert handlers.max_runtime_s("k") == DEFAULT_MAX_RUNTIME_S
    with pytest.raises(ValueError, match="positive"):
        handlers.register("bad", handler, max_runtime_s=0)


async def test_a_stalled_heartbeat_call_is_bounded() -> None:
    """A heartbeat call stuck on a half-open connection must not outlive the lease: it times
    out at the beat interval, and after lease_s without a successful beat the work is cancelled
    and nothing is reported (the job is treated as lost)."""
    queue = FakeQueue()
    (lease,) = queue.add()
    cancelled = asyncio.Event()

    async def stalled(lease: JobLease) -> bool:
        await asyncio.Event().wait()
        return True

    queue.heartbeat = stalled  # type: ignore[method-assign]

    async def handler(lease: JobLease) -> None:
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.set()
            raise

    handlers = JobHandlers()
    handlers.register("k", handler)
    await _run_until(_runtime(queue, handlers), cancelled)
    assert queue.outcome(lease) == []


def test_worker_lease_matches_the_sql_lease(superuser: Conn) -> None:
    """WorkerConfig.lease_s is the heartbeat's notion of the lease; rf._job_lease() is the one
    claims and heartbeats actually set. They are coupled only by value: pin it."""
    row = superuser.execute("SELECT extract(epoch FROM rf._job_lease())").fetchone()
    assert row is not None
    assert float(str(row[0])) == WorkerConfig().lease_s


# ------------------------------------------------------------------ ops review round 6
async def test_a_handler_ignoring_cancellation_is_not_failed_and_keeps_its_slot() -> None:
    """Failing a timed-out job whose handler is still running would requeue it for another
    worker beside the zombie, and freeing the slot would oversubscribe `concurrency`. Neither
    happens: nothing is reported (lease expiry + reclaim decide) and the slot stays taken until
    the handler really ends."""
    queue = FakeQueue()
    (stubborn,) = queue.add("stubborn")
    queue.add("quick")
    let_go = asyncio.Event()
    zombie = asyncio.Event()
    quick_ran = asyncio.Event()

    async def ignores_cancel(lease: JobLease) -> None:
        while not let_go.is_set():
            try:
                await asyncio.sleep(0.01)
            except asyncio.CancelledError:
                zombie.set()  # swallow it, as a buggy __aexit__ might

    async def run_quick(lease: JobLease) -> None:
        quick_ran.set()

    handlers = JobHandlers()
    handlers.register("stubborn", ignores_cancel, max_runtime_s=0.1)
    handlers.register("quick", run_quick)
    one_slot = WorkerConfig(
        concurrency=1,
        poll_interval_s=0.02,
        lease_s=1.0,
        heartbeat_interval_s=0.05,
        shutdown_grace_s=0.3,
        release_timeout_s=0.1,
    )
    runtime = _runtime(queue, handlers, one_slot)
    stop = asyncio.Event()
    task = asyncio.create_task(runtime.run(stop))
    try:
        await asyncio.wait_for(zombie.wait(), 5)
        await asyncio.sleep(0.4)  # well past max_runtime + release_timeout
        assert queue.outcome(stubborn) == []  # no fail: reclaim decides
        assert not quick_ran.is_set()  # the zombie still holds the only slot
        assert runtime.running == 1
        let_go.set()
        await asyncio.wait_for(quick_ran.wait(), 5)
    finally:
        let_go.set()
        stop.set()
        await asyncio.wait_for(task, 5)
    assert queue.outcome(stubborn) == []


async def test_keepalive_gives_up_before_the_lease_can_be_reclaimed() -> None:
    """The server lease runs from the server time of the last good beat; reclaim may hand the
    job on as soon as it ends. With beats hanging, the work must be cancelled a beat's margin
    BEFORE lease_s, not up to a full beat interval after it."""
    queue = FakeQueue()
    queue.add()
    cancelled_after: list[float] = []
    loop = asyncio.get_running_loop()

    async def hangs(lease: JobLease) -> bool:
        await asyncio.Event().wait()
        return True

    queue.heartbeat = hangs  # type: ignore[method-assign]
    done = asyncio.Event()

    async def handler(lease: JobLease) -> None:
        started = loop.time()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled_after.append(loop.time() - started)
            done.set()
            raise

    handlers = JobHandlers()
    handlers.register("k", handler)
    config = WorkerConfig(
        concurrency=1,
        poll_interval_s=0.02,
        lease_s=1.0,
        heartbeat_interval_s=0.3,
        shutdown_grace_s=0.3,
        release_timeout_s=0.5,
    )
    await _run_until(_runtime(queue, handlers, config), done)
    (elapsed,) = cancelled_after
    assert elapsed < config.lease_s - config.heartbeat_interval_s + 0.15  # ~0.7 s, never ~1.3 s
