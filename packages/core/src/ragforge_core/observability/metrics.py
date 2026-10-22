"""The in-process Prometheus metrics of ADR-0018 revision 2, built ONCE at import from the
LOCKED catalogue (`catalog.py`), plus the label guards every call site goes through.

Why at import, on a dedicated registry:
- `create_app` (and `build_api`) run many times per test session; a metric constructed inside
  them would raise "Duplicated timeseries" on the second call. Module scope runs once.
- `REGISTRY` holds exactly the catalogue (and the collector's own error counter), not the
  library's default process/GC collectors, so "the exported names equal the catalogue" is a
  plain set comparison and nothing unplanned reaches the dashboard.

Why label guards: a label value is a time series. Every value is checked against a bounded set
(or a strict shape for config-derived values such as the model name) and anything else becomes
`OTHER`, so no code path can turn a tenant id, a question or a typo into a new series.

The DB-derived gauges (`ragforge_jobs_ready`, ...) are NOT defined here: they exist only while
a scrape-time read succeeds (`db_collector.py`), so a broken read shows as absent, never as a
stale or zero value that looks healthy.
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable, Iterable, Iterator
from contextlib import AbstractContextManager, contextmanager
from importlib.metadata import PackageNotFoundError, version
from typing import Final

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

from ragforge_core import CODE_REQUIRED_SCHEMA
from ragforge_core.config import LLMProvider
from ragforge_core.domain.models import JobPriority
from ragforge_core.observability import catalog
from ragforge_core.observability.catalog import Metric
from ragforge_core.prompts import PROMPT_VERSION

OTHER: Final = "other"

# Bounded label sets the catalogue leaves to the instrumentation (each ends in OTHER).
PROCESSES: Final = ("api", "worker", OTHER)
PROVIDERS: Final = (*(p.value for p in LLMProvider), OTHER)
PRIORITIES: Final = tuple(p.name for p in JobPriority)
INTERVIEW_ENDPOINTS: Final = ("question", "grade")
INTERVIEW_SOURCES: Final = ("notes", "general")
INTERVIEW_OUTCOMES: Final = ("ok", "no_notes", "exhausted", "rejected", "error")
DIRECTIONS: Final = ("prompt", "completion")
# Every prompt this build can send, plus the ContextVar default (a test proves it never shows).
PROMPT_TAGS: Final = frozenset((*PROMPT_VERSION.split(","), catalog.UNKNOWN_PROMPT))
UNMATCHED_ROUTE: Final = "unmatched"

# Config- or code-derived values without a closed list: a strict shape, else OTHER.
_MODEL_SHAPE: Final = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/-]{0,79}")
_IDENT_SHAPE: Final = re.compile(r"[a-z][a-z0-9_]{0,47}")
_ROUTE_SHAPE: Final = re.compile(r"/[A-Za-z0-9_{}./-]{0,119}")

REGISTRY: Final = CollectorRegistry(auto_describe=True)
_BY_NAME: Final = catalog.by_name()


def bounded(value: str, allowed: Iterable[str]) -> str:
    """`value` when it is one of `allowed`, else OTHER."""
    return value if value in allowed else OTHER


def shaped(value: str, shape: re.Pattern[str]) -> str:
    return value if shape.fullmatch(value) else OTHER


def model_label(model: str) -> str:
    return shaped(model, _MODEL_SHAPE)


def ident_label(value: str) -> str:
    """Job kinds, connector kinds: code constants, never user text."""
    return shaped(value, _IDENT_SHAPE)


def route_label(template: str) -> str:
    """A matched route TEMPLATE (`/v1/conversations/{conversation_id}`), never a raw path."""
    return shaped(template, _ROUTE_SHAPE)


def prompt_label(tag: str) -> str:
    return bounded(tag, PROMPT_TAGS)


def priority_label(priority: int) -> str:
    try:
        return JobPriority(priority).name
    except ValueError:
        return OTHER


# ------------------------------------------------------------------ construction
def _spec(name: str, kind: str) -> Metric:
    """The catalogue entry, checked: a kind mismatch is a programming error caught at import."""
    m = _BY_NAME[name]
    if m.kind != kind:
        raise TypeError(f"{name} is a {m.kind} in the catalogue, not a {kind}")
    return m


def _counter(name: str) -> Counter:
    m = _spec(name, "counter")
    return Counter(m.name, m.help, m.labels, registry=REGISTRY)


def _gauge(name: str) -> Gauge:
    m = _spec(name, "gauge")
    return Gauge(m.name, m.help, m.labels, registry=REGISTRY)


def _histogram(name: str) -> Histogram:
    m = _spec(name, "histogram")
    if m.buckets is None:
        raise TypeError(f"{name} has no buckets in the catalogue")
    return Histogram(m.name, m.help, m.labels, registry=REGISTRY, buckets=m.buckets)


# API process
HTTP_REQUESTS: Final = _counter("ragforge_http_requests_total")
HTTP_DURATION: Final = _histogram("ragforge_http_request_duration_seconds")
CHAT_TTFT: Final = _histogram("ragforge_chat_time_to_first_token_seconds")
CHAT_TURN_DURATION: Final = _histogram("ragforge_chat_turn_duration_seconds")
CHAT_TURNS: Final = _counter("ragforge_chat_turns_total")
CHAT_STREAMS: Final = _gauge("ragforge_chat_streams_in_flight")
VERIFICATIONS: Final = _counter("ragforge_verification_total")
FLAGGED_CLAIMS: Final = _counter("ragforge_flagged_claims_total")
LLM_REQUESTS: Final = _counter("ragforge_llm_requests_total")
LLM_DURATION: Final = _histogram("ragforge_llm_request_duration_seconds")
LLM_TOKENS: Final = _counter("ragforge_llm_tokens_total")
EMBED_REQUESTS: Final = _counter("ragforge_embed_requests_total")
EMBED_DURATION: Final = _histogram("ragforge_embed_request_duration_seconds")
RETRIEVAL_DURATION: Final = _histogram("ragforge_retrieval_duration_seconds")
RETRIEVAL_ORPHANS: Final = _counter("ragforge_retrieval_orphan_hits_total")
INTERVIEW_REQUESTS: Final = _counter("ragforge_interview_requests_total")
DB_POOL_CHECKED_OUT: Final = _gauge("ragforge_db_pool_checked_out")
DB_POOL_SIZE: Final = _gauge("ragforge_db_pool_size")
# worker process
JOBS: Final = _counter("ragforge_jobs_total")
JOB_DURATION: Final = _histogram("ragforge_job_duration_seconds")
NOTION_REQUESTS: Final = _counter("ragforge_notion_requests_total")
# both
BUILD_INFO: Final = _gauge("ragforge_build_info")

# NOT in the catalogue (the locked seam has no slot for it; docs/handoff/OBS-METRICS.md): how
# often the scrape-time DB read failed, so "the DB gauges are absent" has a cause on the board.
COLLECT_ERRORS_NAME: Final = "ragforge_metrics_collect_errors_total"
COLLECT_SOURCES: Final = ("queue_stats", "connector_freshness")
COLLECT_ERRORS: Final = Counter(
    COLLECT_ERRORS_NAME,
    "Scrape-time DB reads that failed (their gauges are absent from that scrape)",
    ("source",),
    registry=REGISTRY,
)


def _preinitialise() -> None:
    """Create every child whose labels are ALL bounded and known at import, at 0, so the first
    event after a restart is an increase() Prometheus can see and a quiet process reads 0
    instead of "no data" (the ratio alerts then have a defined denominator). Series whose
    labels depend on traffic or configuration (route, model, job kind) appear on first use."""
    for source in COLLECT_SOURCES:
        COLLECT_ERRORS.labels(source)
    for surface in catalog.SURFACES:
        for outcome in catalog.TURN_OUTCOMES:
            CHAT_TURNS.labels(surface, outcome)
        for status in catalog.VERIFICATION_STATUSES:
            VERIFICATIONS.labels(surface, status)
        for reason in catalog.CLAIM_REASONS:
            FLAGGED_CLAIMS.labels(surface, reason)
    for status in catalog.NOTION_STATUS:
        NOTION_REQUESTS.labels(status)


_preinitialise()


# ------------------------------------------------------------------ process identity
_process: str = OTHER


def set_process(name: str) -> None:
    """Called once by each entrypoint (api, worker) before anything is measured, so the
    `process` label of embeddings, pools and build_info says which container produced it."""
    global _process
    _process = bounded(name, PROCESSES)


def process() -> str:
    return _process


def package_version() -> str:
    try:
        return version("ragforge-core")
    except PackageNotFoundError:  # running from a source tree without an installed dist
        return "unknown"


def set_build_info() -> None:
    """`ragforge_build_info{version, schema_version, process} 1` for THIS process."""
    BUILD_INFO.labels(package_version(), str(CODE_REQUIRED_SCHEMA), _process).set(1)


# ------------------------------------------------------------------ small recorders
@contextmanager
def timed(observe: Callable[[float], None]) -> Iterator[None]:
    """Observe the block's wall time (perf_counter), also when it raises."""
    started = time.perf_counter()
    try:
        yield
    finally:
        observe(time.perf_counter() - started)


def observe_embed(outcome: str, seconds: float) -> None:
    proc = _process
    EMBED_REQUESTS.labels(proc, bounded(outcome, catalog.EMBED_OUTCOMES)).inc()
    EMBED_DURATION.labels(proc).observe(seconds)


def retrieval_stage(stage: str) -> AbstractContextManager[None]:
    """Time one retrieval stage (catalog.RETRIEVAL_STAGES) into its histogram."""
    child = RETRIEVAL_DURATION.labels(bounded(stage, catalog.RETRIEVAL_STAGES))
    return timed(child.observe)


def count_notion(code: int | None) -> None:
    """One Notion HTTP attempt: its response status, or None when no response arrived."""
    NOTION_REQUESTS.labels(notion_status(code)).inc()


def notion_status(code: int | None) -> str:
    """catalog.NOTION_STATUS for an HTTP status, or "network" when no response arrived."""
    if code is None:
        return "network"
    if code == 429:
        return "429"
    if 200 <= code < 300:
        return "2xx"
    if code >= 500:
        return "5xx"
    return "4xx"  # 4xx, and the 1xx/3xx the API never sends: a request that did not succeed


def status_class(code: int) -> str:
    return bounded(f"{code // 100}xx", catalog.STATUS_CLASSES)


def sample(name: str, labels: dict[str, str] | None = None) -> float:
    """The current value of one sample (0.0 when absent). For tests and debugging; histogram
    parts are addressed by their sample name (`<name>_count`, `<name>_sum`, `<name>_bucket`)."""
    value = REGISTRY.get_sample_value(name, labels or {})
    return 0.0 if value is None else value
