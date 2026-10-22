"""LOCKED metric catalogue (ADR-0018 revision 2). The ONE source of metric names, types, label
names, allowed label values and histogram buckets. OBS-METRICS instruments from it; OBS-STACK's
dashboard and alert-rule tests parse it, so a name can never drift between code and dashboard.

Rules (ADR-0018 §4): labels are bounded enums only. Never a tenant id, user id, chunk id, key
prefix or any text as a label. `surface` keeps eval/demo traffic out of the owner's signals.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal

Kind = Literal["counter", "gauge", "histogram"]

# Buckets (seconds). SLO thresholds sit exactly on an edge (hosted 5 s, local 12 s: DESIGN §2).
TTFT_BUCKETS: Final = (0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0, 30.0)
TURN_BUCKETS: Final = (0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0, 30.0, 60.0, 120.0)
RETRIEVAL_BUCKETS: Final = (0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5)
HTTP_BUCKETS: Final = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)

# Bounded label values (anything else is mapped to "other" by the instrumentation).
SURFACES: Final = ("chat", "interview", "eval", "demo", "other")
STATUS_CLASSES: Final = ("2xx", "3xx", "4xx", "5xx", "disconnected")
TURN_OUTCOMES: Final = ("answered", "abstained", "refused", "error", "interrupted")
VERIFICATION_STATUSES: Final = ("verified", "partial", "unsupported", "unverified")
CLAIM_REASONS: Final = ("not_found", "inexact_code")
LLM_OUTCOMES: Final = (
    "ok", "rate_limited", "quota_exhausted", "timeout", "server", "blocked", "interrupted",
    "output_limit", "config", "other",
)  # fmt: skip
EMBED_OUTCOMES: Final = ("ok", "timeout", "server", "other")
JOB_OUTCOMES: Final = ("done", "failed", "dead", "requeued")
RETRIEVAL_STAGES: Final = ("embed", "search", "rerank")
NOTION_STATUS: Final = ("2xx", "429", "4xx", "5xx", "network")
UNKNOWN_PROMPT: Final = "unknown"  # the ContextVar default: a test asserts it never appears


@dataclass(frozen=True, slots=True)
class Metric:
    name: str
    kind: Kind
    help: str
    labels: tuple[str, ...] = ()
    buckets: tuple[float, ...] | None = None
    source: str = ""  # where it is produced (the builder's map)


CATALOG: Final[tuple[Metric, ...]] = (
    # ---- API process
    Metric("ragforge_http_requests_total", "counter", "HTTP requests by matched route",
           ("route", "status_class"), source="api middleware (outermost)"),
    Metric("ragforge_http_request_duration_seconds", "histogram",
           "HTTP request duration, non-streaming responses only", ("route",), HTTP_BUCKETS,
           source="api middleware"),
    Metric("ragforge_chat_time_to_first_token_seconds", "histogram", "Chat time to first token",
           ("surface", "provider"), TTFT_BUCKETS, source="chat SSE"),
    Metric("ragforge_chat_turn_duration_seconds", "histogram", "Chat turn duration to done",
           ("surface", "provider"), TURN_BUCKETS, source="chat SSE"),
    Metric("ragforge_chat_turns_total", "counter", "Chat turns by outcome",
           ("surface", "outcome"), source="chat SSE"),
    Metric("ragforge_chat_streams_in_flight", "gauge", "Open chat streams", (),
           source="chat SSE"),
    Metric("ragforge_verification_total", "counter",
           "Verification verdicts of ANSWERED turns (abstain/refusal/error excluded)",
           ("surface", "status"), source="chat SSE verification event"),
    Metric("ragforge_flagged_claims_total", "counter", "Flagged claims by reason",
           ("surface", "reason"), source="chat SSE"),
    Metric("ragforge_llm_requests_total", "counter", "Model calls by outcome",
           ("provider", "model", "prompt", "outcome"), source="InstrumentedChatModel"),
    Metric("ragforge_llm_request_duration_seconds", "histogram", "Model call duration",
           ("provider", "model", "prompt", "outcome"), TURN_BUCKETS,
           source="InstrumentedChatModel"),
    Metric("ragforge_llm_tokens_total", "counter", "Model tokens",
           ("provider", "model", "prompt", "direction"), source="InstrumentedChatModel"),
    Metric("ragforge_embed_requests_total", "counter", "Embedding calls by outcome",
           ("process", "outcome"), source="embedder adapter (api + worker)"),
    Metric("ragforge_embed_request_duration_seconds", "histogram", "Embedding call duration",
           ("process",), RETRIEVAL_BUCKETS, source="embedder adapter"),
    Metric("ragforge_retrieval_duration_seconds", "histogram", "Retrieval stage duration",
           ("stage",), RETRIEVAL_BUCKETS, source="agent/retrieval"),
    Metric("ragforge_retrieval_orphan_hits_total", "counter", "Orphan hits (DESIGN §7.3)", (),
           source="agent/retrieval"),
    Metric("ragforge_interview_requests_total", "counter", "Interview calls",
           ("endpoint", "source", "outcome"), source="interview routes"),
    Metric("ragforge_db_pool_checked_out", "gauge", "DB pool connections checked out",
           ("process", "pool"), source="engine pool (api + worker)"),
    Metric("ragforge_db_pool_size", "gauge", "DB pool size", ("process", "pool"),
           source="engine pool"),
    # ---- DB-derived gauges: ONE place, scrape-time collector in the API as rf_metrics
    Metric("ragforge_jobs_ready", "gauge", "Ready queued jobs", ("priority",),
           source="rf.queue_stats via rf_metrics"),
    Metric("ragforge_jobs_oldest_queued_age_seconds", "gauge",
           "Age of the oldest queued job, including jobs waiting out a retry backoff "
           "(pair it with ragforge_jobs_ready)", ("priority",), source="rf.queue_stats"),
    Metric("ragforge_jobs_expired_leases", "gauge", "Running jobs whose lease expired", (),
           source="rf.queue_stats"),
    Metric("ragforge_jobs_dead", "gauge",
           "Jobs that went dead in the last 24 h (rf.jobs + rf.jobs_history)", (),
           source="rf.dead_jobs_last_day() via rf_metrics (migration 0014)"),
    Metric("ragforge_sync_oldest_success_age_seconds", "gauge",
           "Age of the STALEST enabled connector's last successful sync", ("connector_kind",),
           source="rf.connector_freshness via rf_metrics (migration 0013)"),
    # ---- worker process (what only it knows)
    Metric("ragforge_jobs_total", "counter", "Finished jobs", ("kind", "outcome"),
           source="worker"),
    Metric("ragforge_job_duration_seconds", "histogram", "Job duration", ("kind",),
           TURN_BUCKETS, source="worker"),
    Metric("ragforge_notion_requests_total", "counter", "Notion API requests",
           ("status",), source="notion client (worker)"),
    # ---- both
    Metric("ragforge_build_info", "gauge", "Build info (always 1)",
           ("version", "schema_version", "process"), source="api + worker"),
)  # fmt: skip

# Ports (internal only: no `ports:` entry in compose; Prometheus scrapes on the compose network)
API_METRICS_PORT: Final = 9101
WORKER_METRICS_PORT: Final = 9102


def by_name() -> dict[str, Metric]:
    return {m.name: m for m in CATALOG}
