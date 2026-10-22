"""FlashRank cross-encoder reranker (ADR-0006).

* `max_length=512` explicitly, and `PASSAGE_BUDGET` (512 - 3 special tokens - the 64-token query
  share = 445) is the largest chunk the cross-encoder can score WHOLE next to any query.
  Bootstrap configures the chunker with that max, so prose and tables always fit; only a
  code block longer than the budget (the chunker keeps code blocks whole) is cut by FlashRank's
  tokenizer, which then scores just its first ~445 tokens.
* Only the top-20 candidates are scored (latency budget ~250 ms on CPU, measured in M3); the
  remainder keeps its fused order after them, with rerank_score=None.
* ONNX inference is CPU-bound and synchronous, so it runs on the adapter's OWN small executor
  (`RERANK_THREADS`), not the default `to_thread` pool: concurrent turns queue for the model
  instead of oversubscribing the CPU, and blob writes / hashing / multipart spooling on the
  default pool are never stuck behind inference. `close()` shuts it down (bootstrap registers it).
* Per-passage token budget: the cross-encoder sees "[CLS] query [SEP] passage [SEP]" cut at
  512 tokens. The "title › heading path" header helps it, but it is prepended ONLY when query +
  header + chunk still fit; otherwise the header is dropped and the chunk is scored alone. That
  keeps every chunk within `PASSAGE_BUDGET` whole (the failure ADR-0006 is written to avoid);
  see the first bullet for the one exception (over-long code blocks). Token counting is injected
  (`token_counter`, bootstrap passes rag.tokens.count_tokens) because adapters may not import
  rag; it is required so no wiring can silently fall back to "always prepend".
* The query is cut to MAX_QUERY_TOKENS before the passage budget is computed, so the budget
  never goes negative for long (pasted) questions.
* Startup: bootstrap calls `load()` and fails the API startup if the model is missing. The agent
  grades on the rerank score only (ADR-0006), so a "degraded" reranker would make every chat
  turn abstain while health checks stayed green; a crash names the problem instead.
* Graceful degradation is kept for adapters used without `load()` (tests, tools): a missing
  model returns fused order with rerank_score=None and one warning. An inference failure (ONNX /
  tokenizer error) degrades the same way for that call only, so a reranker bug costs ranking
  quality, never the chat turn.
  By default the adapter NEVER downloads: containers run with read-only filesystems and no
  egress, and a first-request download would add seconds to someone's chat turn.
* onnxruntime telemetry is OFF (SECURITY.md §3.5). The onnxruntime 1.31 wheels (macOS and the
  Linux aarch64 one in the image) link Microsoft's 1DS client with a built-in collector URL
  (mobile.events.data.microsoft.com) and initialise it when onnxruntime is IMPORTED (inside the
  container it logs "Failed to persist telemetry device ID" at import). Two switches, both
  verified against the installed library, not assumed: `ORT_DISABLE_TELEMETRY=1` in the
  environment stops that initialisation altogether (set at this module's import, before any
  flashrank/onnxruntime import, and in the image's ENV for any other entrypoint), and
  `onnxruntime.disable_telemetry_events()` turns event collection off before the first
  InferenceSession (called just before flashrank builds its session; skipped only if a future
  onnxruntime drops the function, which a test pins against the locked version).
"""

from __future__ import annotations

import asyncio
import dataclasses
import importlib
import os
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Protocol

from ragforge_core.domain.models import RankedChunk
from ragforge_core.observability.logging import get_logger

DEFAULT_MODEL = "ms-marco-TinyBERT-L-2-v2"
MAX_LENGTH = 512
TOP_N = 20
SPECIAL_TOKENS = 3  # [CLS] query [SEP] passage [SEP]
# The query gets a fixed share of the window so a chunk at the chunker's max (512) minus this
# share always fits: an unbounded query would make the passage budget negative and the
# cross-encoder would silently cut the chunk's tail (the failure ADR-0006 is written to avoid).
MAX_QUERY_TOKENS = 64
# The largest passage scored whole next to any (truncated) query; bootstrap sets the chunker's
# max_tokens to this so only over-long code blocks can exceed it.
PASSAGE_BUDGET = MAX_LENGTH - SPECIAL_TOKENS - MAX_QUERY_TOKENS
# One inference at a time per process: a single ONNX session already uses several intra-op
# threads, so more concurrent calls only oversubscribe the CPU and raise every turn's latency.
RERANK_THREADS = 1
_HEADER_SEPARATOR = " › "

# Read by libonnxruntime when it initialises its telemetry client (at `import onnxruntime`). Forced
# to "1", never defaulted: telemetry must stay off whatever the environment says.
ORT_TELEMETRY_ENV: Final = "ORT_DISABLE_TELEMETRY"

log = get_logger(__name__)


def disable_onnxruntime_telemetry() -> None:
    """Turn onnxruntime's telemetry off before any session exists (see the module docstring).
    Idempotent; a missing onnxruntime (the ingest extra) or function is not an error."""
    os.environ[ORT_TELEMETRY_ENV] = "1"
    try:
        onnxruntime = importlib.import_module("onnxruntime")
    except ImportError:
        return
    disable = getattr(onnxruntime, "disable_telemetry_events", None)
    if callable(disable):
        disable()


# At import, before anything in this process can import onnxruntime through flashrank.
os.environ[ORT_TELEMETRY_ENV] = "1"


class RankerLike(Protocol):
    """The slice of flashrank.Ranker this adapter uses (duck-typed request: .query/.passages)."""

    def rerank(self, request: Any) -> list[dict[str, Any]]: ...


@dataclass(frozen=True, slots=True)
class _Request:
    query: str
    passages: list[dict[str, Any]]


class RerankerUnavailable(RuntimeError):
    """The model could not be loaded (not baked in, or flashrank not installed)."""


class FlashRankReranker:
    """Implements the `Reranker` port."""

    def __init__(
        self,
        *,
        cache_dir: str | Path,
        token_counter: Callable[[str], int],
        model_name: str = DEFAULT_MODEL,
        top_n: int = TOP_N,
        allow_download: bool = False,
        ranker_factory: Callable[[], RankerLike] | None = None,
    ) -> None:
        if top_n <= 0:
            raise ValueError("top_n must be positive")
        self._cache_dir = Path(cache_dir)
        self._model_name = model_name
        self._top_n = top_n
        self._count = token_counter
        self._allow_download = allow_download
        self._factory = ranker_factory or self._load_flashrank
        self._ranker: RankerLike | None = None
        self._unavailable = False
        self._load_lock = asyncio.Lock()
        self._executor = ThreadPoolExecutor(max_workers=RERANK_THREADS, thread_name_prefix="rerank")

    def close(self) -> None:
        """Stop the inference thread (queued calls are cancelled). Idempotent."""
        self._executor.shutdown(wait=False, cancel_futures=True)

    async def _run[T](self, fn: Callable[..., T], *args: Any) -> T:
        return await asyncio.get_running_loop().run_in_executor(self._executor, fn, *args)

    @property
    def available(self) -> bool | None:
        """None until the first rerank attempted a load; then whether the model is usable."""
        if self._ranker is not None:
            return True
        return False if self._unavailable else None

    async def load(self) -> None:
        """Load the model now (on the inference thread) or raise `RerankerUnavailable` naming the
        model and cache dir. Idempotent; concurrent callers share one load."""
        async with self._load_lock:
            if self._ranker is not None:
                return
            try:
                self._ranker = await self._run(self._factory)
            except Exception as exc:
                self._unavailable = True
                raise RerankerUnavailable(
                    f"reranker model {self._model_name!r} could not be loaded from "
                    f"{str(self._cache_dir)!r} ({type(exc).__name__}); bake it into the image "
                    "(docker/Dockerfile models stage) or fix RAGFORGE_RERANK_MODEL / "
                    "RAGFORGE_RERANK_CACHE_DIR"
                ) from exc
            self._unavailable = False

    async def rerank(self, query: str, candidates: Sequence[RankedChunk]) -> list[RankedChunk]:
        head, tail = list(candidates[: self._top_n]), list(candidates[self._top_n :])
        if not head:
            return []
        ranker = await self._get_ranker()
        if ranker is None:
            return head + tail
        query = truncate_to_tokens(query, MAX_QUERY_TOKENS, self._count)
        budget = MAX_LENGTH - SPECIAL_TOKENS - self._count(query)
        passages = [{"id": i, "text": self._passage(c, budget)} for i, c in enumerate(head)]
        try:
            results = await self._run(ranker.rerank, _Request(query, passages))
        except Exception as exc:  # degrade, never fail the turn; the message may quote content
            log.warning("reranker_failed", model=self._model_name, error=type(exc).__name__)
            return head + tail
        scored = [
            dataclasses.replace(head[int(r["id"])], rerank_score=float(r["score"])) for r in results
        ]
        scored.sort(key=lambda c: -(c.rerank_score or 0.0))  # stable: ties keep fused order
        return scored + tail

    async def _get_ranker(self) -> RankerLike | None:
        if self._ranker is not None or self._unavailable:
            return self._ranker
        async with self._load_lock:  # one load even when many turns start at once
            if self._ranker is None and not self._unavailable:
                try:
                    self._ranker = await self._run(self._factory)
                except Exception as exc:
                    self._unavailable = True
                    log.warning(
                        "reranker_unavailable", model=self._model_name, error=type(exc).__name__
                    )
        return self._ranker

    def _load_flashrank(self) -> RankerLike:
        if not self._allow_download and not (self._cache_dir / self._model_name).is_dir():
            raise RerankerUnavailable(f"{self._model_name} not found under {self._cache_dir}")
        # Before flashrank imports onnxruntime and its Ranker creates the first InferenceSession.
        disable_onnxruntime_telemetry()
        try:
            flashrank = importlib.import_module("flashrank")
        except ImportError as exc:
            raise RerankerUnavailable("flashrank is not installed (ingest extra)") from exc
        ranker: RankerLike = flashrank.Ranker(
            model_name=self._model_name, cache_dir=str(self._cache_dir), max_length=MAX_LENGTH
        )
        return ranker

    def _passage(self, hit: RankedChunk, budget: int) -> str:
        """Title and heading path give the cross-encoder the same context the embedder had, but
        only while they fit `budget` together with the chunk: the chunk text always wins."""
        location = _HEADER_SEPARATOR.join(p for p in (hit.title, *hit.heading_path) if p)
        if location and self._count(location) + self._count(hit.display_text) <= budget:
            return f"{location}\n\n{hit.display_text}"
        return hit.display_text


def truncate_to_tokens(text: str, limit: int, count: Callable[[str], int]) -> str:
    """Longest prefix of `text` that `count` puts at or under `limit` tokens.

    Binary search over the character length: only a counter is injected (adapters may not
    import rag's tokenizer), and O(log n) counts of the query are negligible next to inference.
    The counter is monotonic in prefix length up to word-boundary effects; the final check makes
    the result safe even where it is not.
    """
    if count(text) <= limit:
        return text
    low, high = 0, len(text)
    while low < high:
        mid = (low + high + 1) // 2
        if count(text[:mid]) <= limit:
            low = mid
        else:
            high = mid - 1
    prefix = text[:low]
    while prefix and count(prefix) > limit:
        prefix = prefix[:-1]
    return prefix
