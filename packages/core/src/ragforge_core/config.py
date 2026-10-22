"""Single source of runtime configuration (env vars, `.env` in dev). No other module reads env."""

from __future__ import annotations

import re
from enum import StrEnum
from functools import lru_cache
from typing import Final, Literal, Self

from pydantic import Field, SecretStr, ValidationInfo, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from ragforge_core.observability.catalog import API_METRICS_PORT, WORKER_METRICS_PORT

# The contract's InterviewLevel values (ragforge_contracts.api; a module_iv test keeps them equal).
InterviewLevelName = Literal["sde_1", "sde_2", "senior_sde_1", "senior_sde_2", "staff"]
# Phase 3b's display spellings, after `_interview_level` folds case and turns spaces/hyphens
# into "_": "Senior SDE-1" -> "senior_sde_1" already; only "senior" alone needs an entry.
_LEVEL_ALIASES: Final[dict[str, str]] = {"senior": "senior_sde_1", "staff_engineer": "staff"}


class LLMProvider(StrEnum):
    """Which chat-model adapter serves the agent (adapters/llm/provider.py). Embeddings stay on
    Ollama either way: they are pinned per embedding space (ADR-0009), not by this switch."""

    ollama = "ollama"
    gemini = "gemini"


# Worker tasks that each hold at most one pooled connection beside the job slots: the claim loop,
# the runtime's two queue-maintenance tasks (reclaim, move_finished), the three upkeep steps
# (connector polls, reconciles, storage GC), the schema watch, and an embedding-space registry
# reload. `apps.worker.main.background_db_holders` lists them; a test keeps the two in step. The
# LISTEN connection is outside the pool.
WORKER_BACKGROUND_CONNECTIONS = 8

# Connection budget per deployment (compare with Postgres max_connections; compose sets 200):
#   API replica    : db_pool_size (10) + LangGraph pool (5) + 1 job-progress LISTEN       = 16
#   worker replica : worker_pool_size (2 * worker_concurrency + 8 + 1 spare = 17 at c=4)
#                    + 1 LISTEN                                                          = 18
#   one-shots      : migrate / index / db-bootstrap, 1 each, short-lived
# e.g. 2 API + 3 worker replicas at the defaults = 2*16 + 3*18 = 86, plus superuser/backup
# sessions. The worker logs a warning at startup when its own pool exceeds a third of
# max_connections (it does not refuse to start).


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="RAGFORGE_", env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # --- database (one DSN per role; ADR-0013). Host/port/db are shared.
    db_host: str = "localhost"
    db_port: int = 5432
    db_name: str = "ragforge"
    db_superuser: str = "postgres"
    db_superuser_password: SecretStr = SecretStr("")
    db_migrator_password: SecretStr = SecretStr("")
    db_api_password: SecretStr = SecretStr("")
    db_worker_password: SecretStr = SecretStr("")
    db_metrics_password: SecretStr = SecretStr("")
    db_backup_password: SecretStr = SecretStr("")
    db_pool_size: int = Field(default=10, ge=1, le=100)

    # --- storage
    upload_dir: str = "/data/uploads"  # shared volume (api + worker) locally; S3 in cloud

    # --- secrets
    kek: SecretStr = SecretStr("")  # key-encryption key for connector tokens (base64, 32 bytes)

    # --- models (ADR-0004)
    llm_provider: LLMProvider = LLMProvider.ollama
    ollama_base_url: str = "http://localhost:11434"
    chat_model: str = "llama3.2:3b"
    # No `embed_model` setting: embedding models are pinned per space in rf.embedding_spaces
    # (seeded by 0001, changed by runbook, ADR-0009). RAGFORGE_EMBED_MODEL only tells
    # scripts/models.sh which model to pull.
    chat_num_ctx: int = Field(default=8192, ge=512)  # ADR-0004
    embed_num_ctx: int = 2048
    embed_batch_size: int = Field(default=32, ge=1, le=512)
    ollama_keep_alive: str = "30m"
    # Hosted chat model (used only when llm_provider == gemini). The key is a SecretStr so it
    # never appears in reprs or logs; it is sent only in the x-goog-api-key header.
    # validate_default: the check below must also run when the key is absent from the env.
    gemini_api_key: SecretStr = Field(default=SecretStr(""), validate_default=True)
    gemini_model: str = Field(
        default="gemini-3.5-flash-lite", pattern=r"^(models/)?[A-Za-z0-9][A-Za-z0-9._-]{0,127}$"
    )
    # Tokens the model may think before answering: 0 = off (latency), -1 = model decides.
    gemini_thinking_budget: int = Field(default=0, ge=-1, le=32_768)

    # --- retrieval (module B)
    rerank_model: str = "ms-marco-TinyBERT-L-2-v2"  # baked into the image (docker/Dockerfile)
    rerank_cache_dir: str = "/opt/models/flashrank"
    rerank_top_n: int = Field(default=20, ge=1, le=100)  # candidates fed to the reranker
    vector_exact_threshold: int = Field(default=20_000, ge=0, le=100_000)  # SQL clamps too

    # --- agent (module D, ADR-0006 / ADR-0012)
    agent_default_tau: float = Field(default=0.35, ge=0.0, le=1.0)
    agent_retrieval_k: int = Field(default=5, ge=1, le=20)  # 5 local, 8 hosted
    agent_max_tool_rounds: int = Field(default=3, ge=0, le=10)
    # ADR-0015: "actions" = retrieve -> answer, tools only for explicit actions; "always" = the
    # pre-ADR behaviour (every tool bound on every turn), kept as an override for evals.
    agent_tool_mode: Literal["actions", "always"] = "actions"
    agent_strict_citations: bool = False
    agent_verifier_trusted: bool = True  # False for models below the verifier SLO
    agent_extras: bool = True  # ADR-0016: labelled general-knowledge incident/code extras
    agent_summarize_after_messages: int = Field(default=20, ge=2)

    # --- interview mode (Phases 3b/3c; agent/interview, routes/interview.py)
    # The bar a request WITHOUT a level is asked and graded at: one of the contract's
    # InterviewLevel values. The Phase 3b display names ("Senior SDE-1") are still accepted and
    # mapped (`_interview_level`); anything else fails at startup, never on the first question.
    interview_target_level: InterviewLevelName = "senior_sde_1"
    # Per-principal cost guard on /v1/interview/* (deps.InterviewKey): a token bucket refilled
    # at `rate_per_minute` holding at most `burst`, and at most `concurrency` calls in flight.
    # A grade costs three model calls (grade, two concurrent verifier checks), a question one.
    interview_rate_per_minute: float = Field(default=10.0, gt=0, le=600)
    interview_burst: int = Field(default=5, ge=1, le=100)
    interview_concurrency: int = Field(default=2, ge=1, le=10)
    # How long a question_token stays valid for /grade (a question is answered within minutes).
    interview_token_ttl_s: int = Field(default=86_400, ge=60, le=7 * 86_400)

    # --- eval surface (ADR-0017 §1; apps/api/routes/eval.py and the additive eval seams)
    # The whole surface: /v1/eval/*, /v1/query stages, chat ?trace=shown, interview chunk_ids
    # and the tokenless frozen grade. OFF by default; dev turns it on (compose sets
    # RAGFORGE_EVAL_API=true). Off, /v1/eval/* answers 404 and the additive seams 403.
    eval_api: bool = False
    # Per-principal cost guard on /v1/eval/verify (one ground check per call, up to 2 model
    # calls with the repair retry): the interview limiter's token bucket, with its own budget.
    eval_rate_per_minute: float = Field(default=30.0, gt=0, le=600)
    eval_burst: int = Field(default=10, ge=1, le=100)
    eval_concurrency: int = Field(default=2, ge=1, le=10)

    # --- worker (module A) and sync (module C)
    worker_concurrency: int = Field(default=4, ge=1, le=32)
    # The worker's own pool. Unset: worker_pool_floor + 1, independent of db_pool_size (the API's
    # knob: raising it must not silently inflate every worker replica). See worker_pool_floor
    # for the enumerated worst case.
    worker_db_pool_size: int | None = Field(default=None, ge=1, le=200)
    sync_reconcile_interval_s: int = Field(default=86_400, ge=3600, le=604_800)  # SQL caps at 7 d
    sync_poll_interval_s: int = Field(default=300, ge=30)
    max_upload_mb: int = Field(default=50, ge=1, le=50)  # the parser's own cap is 50 MB
    parse_timeout_s: float = Field(default=60.0, gt=0, le=600)
    parse_memory_mb: int = Field(default=1024, ge=128)

    # --- observability
    log_level: str = "INFO"
    log_json: bool = True
    trace_content: bool = False  # never record prompt/chunk text in spans by default
    # ADR-0018 R2-6: Prometheus `/metrics` on an INTERNAL port per process (never on :8000).
    # 0 disables the listener. The address defaults to loopback; the containers set 0.0.0.0
    # (compose has no `ports:` entry for it, so it stays on the compose network).
    api_metrics_port: int = Field(default=API_METRICS_PORT, ge=0, le=65_535)
    worker_metrics_port: int = Field(default=WORKER_METRICS_PORT, ge=0, le=65_535)
    metrics_addr: str = "127.0.0.1"

    @field_validator("interview_target_level", mode="before")
    @classmethod
    def _interview_level(cls, value: object) -> object:
        """Map a Phase 3b display name ("Senior SDE-1", "staff", "SDE 2") to its level value;
        the Literal then rejects whatever is left, so an unknown level stops the process."""
        if not isinstance(value, str):
            return value
        key = re.sub(r"[\s-]+", "_", value.strip().casefold())
        return _LEVEL_ALIASES.get(key, key)

    @field_validator("gemini_api_key")
    @classmethod
    def _gemini_key_present(cls, key: SecretStr, info: ValidationInfo) -> SecretStr:
        """Selecting Gemini without a key must fail at startup, not on the first chat turn.
        A field validator (not a model validator) on purpose: pydantic echoes a validator's
        input in the error, and here that input is only the (empty) key, never the other
        settings such as database passwords."""
        selected = info.data.get("llm_provider") is LLMProvider.gemini
        if selected and not key.get_secret_value().strip():
            raise ValueError(
                "RAGFORGE_GEMINI_API_KEY must be set when RAGFORGE_LLM_PROVIDER=gemini"
            )
        return key

    @model_validator(mode="after")
    def _worker_pool_fits(self) -> Self:
        """A pool smaller than the worker's worst case shows up as pool timeouts that burn job
        attempts and skip lease heartbeats, never as a clean error. Fail at startup instead."""
        need = self.worker_pool_floor
        if self.worker_db_pool_size is not None and self.worker_db_pool_size < need:
            raise ValueError(
                f"worker_db_pool_size ({self.worker_db_pool_size}) must be >= "
                f"2 * worker_concurrency + {WORKER_BACKGROUND_CONNECTIONS} ({need})"
            )
        return self

    @property
    def worker_pool_floor(self) -> int:
        """Connections the worker can hold at the same moment (apps/worker/main.py):
        per running job, the handler's transaction (rate-limiter and embedding transactions are
        short and sequential within it) plus the keepalive heartbeat: 2 * worker_concurrency;
        plus WORKER_BACKGROUND_CONNECTIONS for tasks that run beside the jobs."""
        return 2 * self.worker_concurrency + WORKER_BACKGROUND_CONNECTIONS

    @property
    def worker_pool_size(self) -> int:
        if self.worker_db_pool_size is not None:
            return self.worker_db_pool_size
        return self.worker_pool_floor + 1  # one spare

    def dsn(self, role: str, *, driver: str = "postgresql+psycopg") -> str:
        """Build a DSN for one of: superuser, migrator, api, worker, metrics, backup."""
        users = {
            "superuser": (self.db_superuser, self.db_superuser_password),
            "migrator": ("rf_migrator", self.db_migrator_password),
            "api": ("rf_api", self.db_api_password),
            "worker": ("rf_worker", self.db_worker_password),
            "metrics": ("rf_metrics", self.db_metrics_password),
            "backup": ("rf_backup", self.db_backup_password),
        }
        user, password = users[role]
        from urllib.parse import quote

        pw = quote(password.get_secret_value(), safe="")
        return f"{driver}://{user}:{pw}@{self.db_host}:{self.db_port}/{self.db_name}"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
