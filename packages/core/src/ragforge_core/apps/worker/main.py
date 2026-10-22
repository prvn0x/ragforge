"""`ragforge-worker` entrypoint: schema-readiness gate, LISTEN wake-ups, claim loop, SIGTERM drain.

Startup order: wait until the schema is compatible, THEN wire the container (wiring reads the
embedding spaces). `register_handlers` maps every job kind to its handler. Beside the runtime,
`upkeep` schedules connectors (poll + nightly reconcile) and runs storage GC.

Schema compatibility is re-checked every 30 s. When a contract migration makes this code
incompatible (ADR-0002 `min_code_version`), the worker stops claiming, drains like on SIGTERM and
exits with status 1, so the orchestrator restarts it into the startup wait (or the new image).
A database that is merely unreachable is NOT treated as incompatible: the runtime already
retries, and stopping every worker on a blip would turn it into an outage.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import os
import signal
import socket
import sys
import time
from collections.abc import Awaitable, Callable, Coroutine, Iterator, Sequence
from pathlib import Path
from typing import Any
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.exc import DBAPIError
from sqlalchemy.ext.asyncio import AsyncEngine

from ragforge_core.adapters.connectors.notion.db import purge_tombstones
from ragforge_core.adapters.db.tx import connection
from ragforge_core.agent.nodes import SUMMARIZE_JOB
from ragforge_core.apps.worker.job_metrics import ObservedJobQueue
from ragforge_core.apps.worker.listener import NotificationListener
from ragforge_core.apps.worker.upkeep import (
    GC_BATCH,
    PURGE_BATCH,
    Periodic,
    UpkeepHealth,
    drain,
    run_upkeep,
    schedule_due_connectors,
    schedule_reconciles,
    sleep_until_stopped,
)
from ragforge_core.bootstrap import WorkerContainer, build_worker
from ragforge_core.config import WORKER_BACKGROUND_CONNECTIONS, Settings, get_settings
from ragforge_core.db.engine import make_engine
from ragforge_core.db.readiness import check_readiness
from ragforge_core.domain.errors import NotFound
from ragforge_core.domain.models import JobLease
from ragforge_core.domain.ports import Tx
from ragforge_core.observability import metrics
from ragforge_core.observability.logging import configure_logging, get_logger
from ragforge_core.observability.server import MetricsServer, serve_metrics
from ragforge_core.services.eval_snapshot import (
    KIND_SNAPSHOT_COPY,
    KIND_SNAPSHOT_REEMBED,
    CopyOutcome,
    EmbedSource,
    EvalSnapshotDeps,
    ReembedOutcome,
    SnapshotNotReady,
    refusal_for,
    run_copy_job,
    run_reembed_job,
)
from ragforge_core.services.eval_snapshot import MAX_RUNTIME_S as EVAL_SNAPSHOT_MAX_RUNTIME_S
from ragforge_core.services.jobs import JobHandler, JobHandlers, WorkerConfig, WorkerRuntime
from ragforge_core.services.sync.deps import SyncDeps
from ragforge_core.services.sync.handlers import run_sync_job
from ragforge_core.services.sync.policy import (
    KIND_DELETE_CONNECTOR,
    KIND_INGEST_FILE,
    KIND_RECONCILE,
    KIND_SYNC_CONNECTOR,
    KIND_SYNC_PAGE,
    SYNC_KINDS,
)

log = get_logger(__name__)

# Liveness = the claim loop is alive: it touches this file every iteration, and the container
# healthcheck (the k8s liveness probe in Phase 4) fails when it goes stale. Upkeep is NOT part
# of liveness: every step is already cut off at its own timeout (so none can wedge), and a
# scheduler that keeps failing almost always means the database, which a restart cannot fix;
# tying liveness to it would restart every replica in a loop and kill all in-flight jobs. A
# stale scheduler is an alert instead: one error log per SCHEDULER_STALE_LOG_EVERY_S.
HEARTBEAT_FILE = Path("/tmp/worker-heartbeat")  # noqa: S108 - tmpfs in the container
SCHEDULER_STEP = "schedule_connectors"
SCHEDULER_MAX_AGE_S = 300.0  # 10 scheduler intervals without a success
SCHEDULER_STALE_LOG_EVERY_S = 300.0


async def _touch_heartbeat() -> None:
    await asyncio.to_thread(HEARTBEAT_FILE.touch)


def liveness_tick(
    health: UpkeepHealth, clock: Callable[[], float] = time.monotonic
) -> Callable[[], Awaitable[None]]:
    last_logged = -SCHEDULER_STALE_LOG_EVERY_S

    async def tick() -> None:
        nonlocal last_logged
        await _touch_heartbeat()
        age = health.age(SCHEDULER_STEP)
        now = clock()
        if age >= SCHEDULER_MAX_AGE_S and now - last_logged >= SCHEDULER_STALE_LOG_EVERY_S:
            last_logged = now
            log.error("worker_scheduler_stale", age_s=round(age))

    return tick


def _sync_handler(deps: SyncDeps) -> JobHandler:
    """`run_sync_job` ends every job itself (complete or fail, fenced), so the sync kinds are
    registered with `reports_outcome=True` and the runtime does not complete them a second
    time. A lost lease is re-raised and logged by the runtime as `job_lease_lost`."""

    async def handle(lease: JobLease) -> None:
        await run_sync_job(lease, deps)

    return handle


async def _summarize_deferred(lease: JobLease) -> None:
    """Phase 2b. Completed as a no-op so `finalize`'s enqueues don't retry into `dead`: the
    worker has no checkpoint-table privilege yet (handoff D2, request 7)."""
    log.info("summarize_conversation_deferred", job_id=str(lease.job_id))


# Wall-clock bound per kind (the runtime cancels and fails a job past it, retryable). Listings,
# reconciles and connector deletion walk a whole workspace; a page or an upload is one document.
MAX_RUNTIME_S: dict[str, float] = {
    KIND_SYNC_CONNECTOR: 2 * 3600.0,
    KIND_RECONCILE: 2 * 3600.0,
    KIND_DELETE_CONNECTOR: 2 * 3600.0,
    KIND_SYNC_PAGE: 15 * 60.0,
    KIND_INGEST_FILE: 15 * 60.0,
}


def register_handlers(handlers: JobHandlers, container: WorkerContainer) -> None:
    sync = _sync_handler(container.sync)
    for kind in sorted(SYNC_KINDS):
        handlers.register(kind, sync, reports_outcome=True, max_runtime_s=MAX_RUNTIME_S[kind])
    handlers.register(SUMMARIZE_JOB, _summarize_deferred)  # default bound


# ------------------------------------------------------------------ eval snapshot (EV-SNAP)
# ADR-0017 §1. Registered by `register_eval_snapshot_handlers`, called right after
# `register_handlers` in `_main`; kept separate so the sync wiring (and its test) stay as they
# are. `SqlEvalSnapshotStore` is the Postgres side of services.eval_snapshot.ports; it lives
# here only because EV-SNAP owns no adapters path (handoff EV-SNAP: move it, unchanged, to
# adapters/db/eval_snapshot.py).
_COPY_SNAPSHOT = text(
    "SELECT copied_pages, copied_chunks, copied_vectors, copied_links, skipped_excluded, "
    "content_digest_hex, index_digest_hex FROM rf.copy_collection_to_eval(:src, :target, :snap)"
)
_SNAPSHOT_READY = text("SELECT ready_at FROM rf.eval_snapshots WHERE collection_id = :c")
_EMBED_SOURCES = text(
    "SELECT DISTINCT ON (c.embed_sha256) c.embed_sha256, d.title, c.heading_path, "
    "c.display_text FROM rf.chunks c JOIN rf.documents d ON d.id = c.document_id "
    "WHERE c.collection_id = :c AND d.deleted_at IS NULL ORDER BY c.embed_sha256, c.id"
)
_FINISH_REEMBED = text(
    "SELECT content_digest_hex, index_digest_hex, vector_count "
    "FROM rf.finish_eval_reembed(:c, :space)"
)


@contextlib.contextmanager
def _snapshot_refusals() -> Iterator[None]:
    try:
        yield
    except DBAPIError as exc:
        diag = getattr(exc.orig, "diag", None)
        refusal = refusal_for(
            getattr(exc.orig, "sqlstate", None),
            getattr(diag, "message_primary", None),
            constraint=getattr(diag, "constraint_name", None),
        )
        if refusal is None:
            raise
        raise refusal from None


class SqlEvalSnapshotStore:
    """Implements services.eval_snapshot.ports.SnapshotStore. Stateless; share one instance."""

    async def copy(
        self, tx: Tx, *, source_collection_id: UUID, target_tenant_id: UUID, snapshot_id: UUID
    ) -> CopyOutcome:
        params = {"src": source_collection_id, "target": target_tenant_id, "snap": snapshot_id}
        with _snapshot_refusals():
            row = (await connection(tx).execute(_COPY_SNAPSHOT, params)).one()
        return CopyOutcome(*row)

    async def embed_sources(self, tx: Tx, collection_id: UUID) -> Sequence[EmbedSource]:
        conn = connection(tx)
        ready = (await conn.execute(_SNAPSHOT_READY, {"c": collection_id})).first()
        if ready is None:
            raise NotFound("eval snapshot not found")
        if ready[0] is None:
            raise SnapshotNotReady("the snapshot is still being copied")
        rows = await conn.execute(_EMBED_SOURCES, {"c": collection_id})
        return [EmbedSource(bytes(r[0]), r[1], tuple(r[2]), r[3]) for r in rows]

    async def finish_reembed(self, tx: Tx, collection_id: UUID, space_id: int) -> ReembedOutcome:
        with _snapshot_refusals():
            row = (
                await connection(tx).execute(
                    _FINISH_REEMBED, {"c": collection_id, "space": int(space_id)}
                )
            ).one()
        return ReembedOutcome(*row)


def eval_snapshot_deps(container: WorkerContainer) -> EvalSnapshotDeps:
    sync = container.sync
    return EvalSnapshotDeps(
        tx=container.tx,
        queue=container.jobs,
        store=SqlEvalSnapshotStore(),
        index=sync.index,
        embedders=sync.embedders,
        embed_batch=sync.policy.embed_batch,
    )


def register_eval_snapshot_handlers(handlers: JobHandlers, deps: EvalSnapshotDeps) -> None:
    """The handlers complete their own jobs inside their fenced transactions; the runtime's
    own completion then finds the lease gone and does nothing (registry contract)."""

    async def copy(lease: JobLease) -> None:
        await run_copy_job(lease, deps)

    async def reembed(lease: JobLease) -> None:
        await run_reembed_job(lease, deps)

    handlers.register(
        KIND_SNAPSHOT_COPY, copy, max_runtime_s=EVAL_SNAPSHOT_MAX_RUNTIME_S[KIND_SNAPSHOT_COPY]
    )
    handlers.register(
        KIND_SNAPSHOT_REEMBED,
        reembed,
        max_runtime_s=EVAL_SNAPSHOT_MAX_RUNTIME_S[KIND_SNAPSHOT_REEMBED],
    )


_GC = text("SELECT rf.gc_embeddings(:n)")


SCHEDULE_EVERY_S = 30.0
SCHEDULE_TIMEOUT_S = 60.0  # up to MAX_CLAIM_BATCHES x (100 claims + 100 enqueues)
RECONCILE_TIMEOUT_S = 120.0
# Claim leases cover the gap between a claim and its enqueue. They must outlive the step that
# enqueues (else another replica could claim the same connector mid-step), and a lost enqueue
# costs one lease. Polls: at least the poll interval, so a poll is never more frequent.
RECONCILE_CLAIM_LEASE_S = int(2 * RECONCILE_TIMEOUT_S)


def poll_claim_lease_s(settings: Settings) -> int:
    return max(settings.sync_poll_interval_s, int(2 * SCHEDULE_TIMEOUT_S))


GC_EVERY_S = 6 * 3600.0
GC_FIRST_AFTER_S = 60.0  # not 10 min: a worker restarted more often would never collect
GC_TIMEOUT_S = 2 * 3600.0  # the whole drain; each round has its own bound (upkeep.drain)
SCHEMA_CHECK_EVERY_S = 30.0


def upkeep_tasks(
    container: WorkerContainer, stop: asyncio.Event, settings: Settings
) -> list[Periodic]:
    tx, jobs = container.tx, container.jobs

    async def gc_once() -> int:
        async with tx.system() as conn:  # one committed transaction per batch (handoff B)
            return int((await connection(conn).execute(_GC, {"n": GC_BATCH})).scalar_one())

    async def purge_once() -> int:
        async with tx.system() as conn:
            return await purge_tombstones(conn, PURGE_BATCH)

    async def storage_gc() -> int:
        gc = await drain(gc_once, done=lambda n: n == 0, stop=stop)
        return gc + await drain(purge_once, done=lambda n: n < PURGE_BATCH, stop=stop)

    async def schedule_polls() -> int:
        return await schedule_due_connectors(jobs, jobs, tx, lease_s=poll_claim_lease_s(settings))

    async def schedule_nightly() -> int:
        return await schedule_reconciles(
            jobs,
            jobs,
            tx,
            interval_s=settings.sync_reconcile_interval_s,
            lease_s=RECONCILE_CLAIM_LEASE_S,
        )

    return [
        Periodic(SCHEDULER_STEP, SCHEDULE_EVERY_S, schedule_polls, SCHEDULE_TIMEOUT_S),
        Periodic(
            "schedule_reconciles", SCHEDULE_EVERY_S * 10, schedule_nightly, RECONCILE_TIMEOUT_S
        ),
        Periodic("storage_gc", GC_EVERY_S, storage_gc, GC_TIMEOUT_S, GC_FIRST_AFTER_S),
    ]


def background_db_holders(tasks: Sequence[Periodic]) -> list[str]:
    """Every task beside the job slots that can hold a pooled connection at the same moment.
    config.WORKER_BACKGROUND_CONNECTIONS must equal its length (a test enforces it), so adding
    a background task without growing the pool floor fails CI instead of exhausting the pool."""
    runtime = ["claim_loop", "reclaim_expired", "move_finished"]  # services.jobs.runtime
    return [*runtime, "schema_watch", "embedder_registry_reload", *(t.name for t in tasks)]


_MAX_CONNECTIONS = text("SELECT current_setting('max_connections')::int")


async def warn_if_pool_too_large(engine: AsyncEngine, pool_size: int) -> None:
    """Pools open lazily, so an oversized deployment only fails under load ("too many
    clients" on heartbeats and claims). Say so at startup; config.py has the budget."""
    try:
        async with engine.connect() as conn:
            max_connections = int((await conn.execute(_MAX_CONNECTIONS)).scalar_one())
    except Exception as exc:
        log.warning("worker_max_connections_unknown", error=type(exc).__name__)
        return
    if pool_size > max_connections // 3:
        log.warning(
            "worker_pool_large_for_server",
            pool_size=pool_size,
            max_connections=max_connections,
        )


async def watch_schema(
    engine: AsyncEngine,
    stop: asyncio.Event,
    incompatible: asyncio.Event,
    *,
    every_s: float = SCHEMA_CHECK_EVERY_S,
) -> None:
    """Stop the worker when the schema says this code may no longer run (see module doc)."""
    while not await sleep_until_stopped(stop, every_s):
        readiness = await check_readiness(engine)
        if readiness.ready or readiness.schema_version is None:
            continue  # compatible, or the DB is unreachable (not an incompatibility)
        log.error("worker_schema_incompatible_stopping", reason=readiness.reason)
        incompatible.set()
        stop.set()
        return


def _supervised(
    name: str,
    work: Coroutine[Any, Any, None],
    *,
    on_crash: Callable[[], None] | None = None,
) -> asyncio.Task[None]:
    """A background task whose crash is logged when it happens (exception type only), instead
    of surfacing at shutdown and skipping the remaining cleanup. A listener crash only means
    2 s polling, so jobs keep running. An upkeep crash (not a failed step, which upkeep already
    isolates) would leave this process without a scheduler for its whole life, and a restart
    fixes it: its `on_crash` stops the worker, which drains and exits 1."""

    async def run() -> None:
        try:
            await work
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log.error("worker_background_task_crashed", task=name, error=type(exc).__name__)
            if on_crash is not None:
                on_crash()

    return asyncio.create_task(run(), name=f"worker:{name}")


async def _wait_for_schema(engine: AsyncEngine, stop: asyncio.Event) -> bool:
    """Never process jobs against an incompatible schema: wait (liveness stays green)."""
    readiness = await check_readiness(engine)
    while not readiness.ready and not stop.is_set():
        log.warning("worker_waiting_for_schema", reason=readiness.reason)
        await _touch_heartbeat()
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(stop.wait(), timeout=5.0)
        readiness = await check_readiness(engine)
    return readiness.ready


def start_metrics(settings: Settings) -> MetricsServer | None:
    """ADR-0018 R2-6: this process's `/metrics` on its internal port (WORKER_METRICS_PORT).
    Called before the engine exists, so its pool gauges carry `process="worker"`."""
    metrics.set_process("worker")
    metrics.set_build_info()
    return serve_metrics(settings.worker_metrics_port, settings.metrics_addr)


def observe_jobs(container: WorkerContainer) -> ObservedJobQueue:
    """The ONE measured queue every job ending goes through (job_metrics.py): the runtime's
    and the self-reporting sync handlers' (`container.sync.queue`). Claims for connectors and
    queue upkeep keep the plain adapter (`container.jobs`): they end no job."""
    observed = ObservedJobQueue(container.jobs)
    container.sync = dataclasses.replace(container.sync, queue=observed)
    return observed


async def _main() -> int:
    settings = get_settings()
    configure_logging(settings.log_level, json=settings.log_json)
    metrics_server = start_metrics(settings)
    try:
        return await _serve(settings)
    finally:
        if metrics_server is not None:
            metrics_server.close()


async def _serve(settings: Settings) -> int:
    engine = make_engine(settings, "worker")
    worker_id = f"{socket.gethostname()}:{os.getpid()}"
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, stop.set)

    try:
        if not await _wait_for_schema(engine, stop):
            await engine.dispose()
            return 0
    except BaseException:
        await engine.dispose()
        raise
    container = await build_worker(settings, engine=engine)  # owns the engine from here on
    incompatible = asyncio.Event()
    upkeep_crashed = asyncio.Event()
    health = UpkeepHealth()

    def restart_after_upkeep_crash() -> None:
        upkeep_crashed.set()
        stop.set()

    try:
        await warn_if_pool_too_large(container.engine, settings.worker_pool_size)
        tasks = upkeep_tasks(container, stop, settings)
        if len(background_db_holders(tasks)) > WORKER_BACKGROUND_CONNECTIONS:
            log.error("worker_pool_floor_undercounts_background_tasks")
        queue = observe_jobs(container)
        handlers = JobHandlers()
        register_handlers(handlers, container)
        register_eval_snapshot_handlers(
            handlers, dataclasses.replace(eval_snapshot_deps(container), queue=queue)
        )
        runtime = WorkerRuntime(
            worker_id=worker_id,
            queue=queue,
            transactions=container.tx,
            handlers=handlers,
            maintenance=container.jobs,
            config=WorkerConfig(concurrency=settings.worker_concurrency),
            on_tick=liveness_tick(health),
        )
        listener = NotificationListener(settings.dsn("worker", driver="postgresql"), runtime.wake)
        background = [
            _supervised("listener", listener.run(stop)),
            _supervised(
                "upkeep", run_upkeep(stop, tasks, health), on_crash=restart_after_upkeep_crash
            ),
            _supervised("schema_watch", watch_schema(container.engine, stop, incompatible)),
        ]
        log.info("worker_started", worker_id=worker_id, kinds=sorted(handlers.kinds()))
        try:
            await runtime.run(stop)
        finally:
            stop.set()
            for task in background:
                task.cancel()
            await asyncio.gather(*background, return_exceptions=True)  # never skip cleanup
    finally:
        log.info("worker_stopping", worker_id=worker_id)
        await container.aclose()
    return 1 if incompatible.is_set() or upkeep_crashed.is_set() else 0


def run() -> None:
    sys.exit(asyncio.run(_main()))
