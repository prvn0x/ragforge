"""Scrape-time DB gauges (ADR-0018 R2-2): queue depth/age, expired leases, dead jobs and sync
freshness, read as `rf_metrics` from `rf.queue_stats`, `rf.dead_jobs_last_day()` (0014) and
`rf.connector_freshness` (0013).

Why ONE scrape-time collector in the API, not gauges sampled by the worker loop:
- the values are cluster-wide facts of the database, so one reader avoids every worker replica
  exporting (and Prometheus summing) the same number;
- reading at scrape time means a value is never older than the scrape, and needs no DB
  connection held in the worker's claim loop;
- `rf_metrics` can read only those aggregates (no tenant ids exist in them) and has a 5 s
  statement_timeout set on the role (db/bootstrap.sql), repeated here per connection.

Dead jobs are NOT read from `rf.queue_stats`: that view covers `rf.jobs` only, and the worker's
upkeep archives a dead row to `rf.jobs_history` five minutes after it died, so the count fell
back to 0 by itself. `rf.dead_jobs_last_day()` counts both tables over 24 h. It is read in the
SAME statement as the queue view (one source, still two statements per scrape, which the
Prometheus scrape timeout is sized for): either both succeed or every queue gauge is absent,
which is what RagforgeDbGaugesMissing (`absent(ragforge_jobs_dead)`) relies on.

Failure handling: Prometheus calls `collect()` in its HTTP thread. A failed read yields NO
sample for that source (absent, never a stale or zero value that looks healthy) and counts
`ragforge_metrics_collect_errors_total{source}`; it never raises, so `/metrics` stays 200 and
the in-process metrics are still served. It is synchronous on purpose: that thread has no
event loop, and a tiny dedicated pool (one connection) keeps it off the app's async pool.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from typing import Final

from prometheus_client.core import GaugeMetricFamily
from prometheus_client.registry import Collector
from sqlalchemy import Engine, TextClause, create_engine, text

from ragforge_core.config import Settings
from ragforge_core.db.engine import CONNECT_ARGS
from ragforge_core.observability import metrics
from ragforge_core.observability.catalog import by_name
from ragforge_core.observability.logging import get_logger

log = get_logger(__name__)

TIMEOUT_S: Final = 5
DB_GAUGES: Final = (
    "ragforge_jobs_ready",
    "ragforge_jobs_oldest_queued_age_seconds",
    "ragforge_jobs_expired_leases",
    "ragforge_jobs_dead",
    "ragforge_sync_oldest_success_age_seconds",
)
# One row per priority with rows in rf.jobs, each carrying the dead count; with an empty queue,
# ONE row of NULL queue columns that still carries it (the LEFT JOIN ON true).
_QUEUE_SQL: Final = text(
    """
    WITH queue AS (
        SELECT priority,
               coalesce(sum(ready), 0)::bigint                            AS ready,
               extract(epoch FROM now() - min(oldest_queued_at))::float8  AS oldest_age_s,
               coalesce(sum(expired_leases), 0)::bigint                   AS expired
          FROM rf.queue_stats
         GROUP BY priority
    )
    SELECT q.priority, q.ready, q.oldest_age_s, q.expired, d.dead
      FROM (SELECT rf.dead_jobs_last_day() AS dead) d
      LEFT JOIN queue q ON true
    """
)
_FRESHNESS_SQL: Final = text(
    "SELECT connector_kind, oldest_success_age_seconds FROM rf.connector_freshness"
)


def metrics_engine(settings: Settings) -> Engine:
    """One connection as rf_metrics, every wait bounded by TIMEOUT_S (connect, pool checkout,
    statement): a scrape must answer within Prometheus' own scrape timeout (10 s default)."""
    return create_engine(
        settings.dsn("metrics"),
        pool_size=1,
        max_overflow=0,
        pool_timeout=TIMEOUT_S,
        pool_pre_ping=True,
        pool_recycle=1800,
        hide_parameters=True,
        connect_args={
            **CONNECT_ARGS,
            "connect_timeout": TIMEOUT_S,
            "options": f"-c statement_timeout={TIMEOUT_S * 1000}",
        },
    )


def _family(name: str) -> GaugeMetricFamily:
    spec = by_name()[name]
    return GaugeMetricFamily(spec.name, spec.help, labels=spec.labels)


class DbGaugeCollector(Collector):
    """A prometheus_client Collector; register it on `metrics.REGISTRY` in the API process."""

    def __init__(self, engine: Engine) -> None:
        self._engine = engine
        self._failing: set[str] = set()

    @classmethod
    def from_settings(cls, settings: Settings) -> DbGaugeCollector:
        return cls(metrics_engine(settings))

    def close(self) -> None:
        self._engine.dispose()

    def describe(self) -> Iterable[GaugeMetricFamily]:
        """Names only, without touching the database: registration must not need it."""
        return [_family(name) for name in DB_GAUGES]

    def collect(self) -> Iterator[GaugeMetricFamily]:
        queue = self._read("queue_stats", _QUEUE_SQL)
        if queue is not None:
            yield from _queue_families(queue)
        freshness = self._read("connector_freshness", _FRESHNESS_SQL)
        if freshness is not None:
            yield _freshness_family(freshness)

    def _read(self, source: str, sql: TextClause) -> Sequence[Sequence[object]] | None:
        try:
            with self._engine.connect() as conn:
                rows = [tuple(r) for r in conn.execute(sql)]
        except Exception as exc:  # absent gauges + a counter; /metrics itself never fails
            metrics.COLLECT_ERRORS.labels(source).inc()
            if source not in self._failing:  # log transitions, not every 15 s scrape
                log.warning("metrics_collect_failed", source=source, error=type(exc).__name__)
            self._failing.add(source)
            return None
        if source in self._failing:
            self._failing.discard(source)
            log.info("metrics_collect_recovered", source=source)
        return rows


def _number(value: object) -> float:
    return float(value) if isinstance(value, int | float) else 0.0


def _queue_families(rows: Sequence[Sequence[object]]) -> list[GaugeMetricFamily]:
    """Every known priority is always present (0 when it has no rows), so "no queue" reads 0,
    not "no data", and a `> 120` alert has a series to evaluate. The dead count is the same on
    every row (it is not per priority), so it is taken once."""
    ready = dict.fromkeys(metrics.PRIORITIES, 0.0)
    oldest = dict.fromkeys(metrics.PRIORITIES, 0.0)
    expired = dead = 0.0
    for priority, n_ready, oldest_age, n_expired, n_dead in rows:
        dead = _number(n_dead)
        if priority is None:  # the empty-queue row: only the dead count
            continue
        label = metrics.priority_label(int(_number(priority)))
        ready[label] = ready.get(label, 0.0) + _number(n_ready)
        oldest[label] = max(oldest.get(label, 0.0), _number(oldest_age))
        expired += _number(n_expired)
    ready_f, oldest_f = _family(DB_GAUGES[0]), _family(DB_GAUGES[1])
    for label in ready:
        ready_f.add_metric([label], ready[label])
        oldest_f.add_metric([label], oldest[label])
    expired_f, dead_f = _family(DB_GAUGES[2]), _family(DB_GAUGES[3])
    expired_f.add_metric([], expired)
    dead_f.add_metric([], dead)
    return [ready_f, oldest_f, expired_f, dead_f]


def _freshness_family(rows: Sequence[Sequence[object]]) -> GaugeMetricFamily:
    family = _family(DB_GAUGES[4])
    for kind, age in rows:
        family.add_metric([metrics.ident_label(str(kind))], max(0.0, _number(age)))
    return family
