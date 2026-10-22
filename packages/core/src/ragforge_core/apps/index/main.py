"""`ragforge-index`: build or drop each embedding space's HNSW index (best-effort one-shot).

Runs after `ragforge-migrate`, and nothing waits for it: api and worker depend on migrate only.
A space without a valid index is served by exact search, so a slow, deferred or failed build
costs latency, never availability, and never blocks a deploy.

Exit status: 0 whenever the step ran, including deferrals, per-space failures (each logged at
warning/error level and retried on the next run) and preemption by `ragforge-migrate` (which
terminates a running build so a migration never waits on it; the next run resumes); 1 only
when the database cannot be reached or the connection drops otherwise.
"""

from __future__ import annotations

import sys

import psycopg

from ragforge_core.apps.index.space_indexes import ensure_space_indexes, is_preemption
from ragforge_core.config import get_settings
from ragforge_core.observability.logging import configure_logging, get_logger

log = get_logger(__name__)


def run() -> None:
    settings = get_settings()
    configure_logging(settings.log_level, json=settings.log_json)
    try:
        report = ensure_space_indexes(settings)
    except psycopg.OperationalError as exc:
        if is_preemption(exc):
            log.warning("space_index_step_preempted", reason="terminated_by_migrate")
            return
        log.error("space_index_step_failed", error=type(exc).__name__)  # unreachable or dropped
        sys.exit(1)
    if report.failed:
        log.error("space_index_step_incomplete", failed=report.failed)
    if report.deferred:
        log.warning("space_index_step_deferred", deferred=report.deferred)
    log.info(
        "space_index_step_done",
        built=report.built,
        dropped=report.dropped,
        skipped=report.skipped,
        locked_out=report.locked_out,
    )
