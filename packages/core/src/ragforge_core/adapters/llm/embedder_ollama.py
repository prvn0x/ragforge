"""Ollama embedder (`/api/embed`) for one embedding space (ADR-0004, ADR-0009).

Request discipline (DESIGN section 2): every request sends the SAME `num_ctx` (changing it makes
Ollama reload the model) and `keep_alive` (avoid cold reloads between syncs). The space's task
prefixes ("search_document: " / "search_query: " for nomic-embed-text) are applied here and only
here, so callers can never embed a query with the document prefix.

Never logs or raises with input text: chunk text is tenant content.
"""

from __future__ import annotations

import math
import time
from collections.abc import Sequence
from typing import Any

import httpx

from ragforge_core.domain.errors import RagforgeError, UpstreamRateLimited, ValidationFailed
from ragforge_core.observability.logging import get_logger
from ragforge_core.observability.metrics import observe_embed

log = get_logger(__name__)

DEFAULT_NUM_CTX = 2048
DEFAULT_KEEP_ALIVE = "30m"
DEFAULT_BATCH_SIZE = 32
# Ollama's `error` text is free-form and some messages (tokenizer / context errors) quote the
# input, so it is mapped to a fixed category and never copied (see _error_category).
_ERROR_CATEGORIES: tuple[tuple[str, str], ...] = (
    ("not found", "model not found"),
    ("context", "input exceeds the model context"),
    ("too long", "input exceeds the model context"),
    ("out of memory", "out of memory"),
    ("unable to load", "model failed to load"),
    ("invalid", "invalid request"),
)


class EmbeddingFailed(RagforgeError):
    """The embedding provider failed or returned something unusable. Retryable by the job."""

    status = 502
    title = "Embedding provider failed"


class EmbeddingInputRejected(ValidationFailed):
    """The provider rejected this INPUT (400/413/422, e.g. it exceeds the model context).
    Retrying the same input can never succeed, so this is a ValidationFailed: the job runtime
    treats it as permanent and the API maps it to 422 instead of a retryable 502."""

    title = "Embedding input rejected"


class OllamaEmbedder:
    """Implements the `Embedder` port for one space. `client` is a shared httpx.AsyncClient
    whose base_url points at Ollama (host or container, ADR-0004)."""

    def __init__(
        self,
        client: httpx.AsyncClient,
        *,
        model: str,
        dim: int,
        doc_prefix: str = "",
        query_prefix: str = "",
        num_ctx: int = DEFAULT_NUM_CTX,
        keep_alive: str = DEFAULT_KEEP_ALIVE,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> None:
        if dim <= 0 or batch_size <= 0 or num_ctx <= 0:
            raise ValueError("dim, batch_size and num_ctx must be positive")
        self.model = model
        self.dim = dim
        self._client = client
        self._doc_prefix = doc_prefix
        self._query_prefix = query_prefix
        self._num_ctx = num_ctx
        self._keep_alive = keep_alive
        self._batch_size = batch_size

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            batch = list(texts[start : start + self._batch_size])
            try:
                out.extend(await self._embed([self._doc_prefix + t for t in batch]))
            except EmbeddingInputRejected:
                # One input in the batch is too long for the model. Isolate it instead of failing
                # every chunk of the page: the chunker's token count is an ESTIMATE, so the model's
                # own limit is the only authority (first real sync: hex-heavy text overflowed).
                for text in batch:
                    out.append(await self._embed_fitting(text, depth=0))
        return out

    async def _embed_fitting(self, text: str, *, depth: int) -> list[float]:
        """Embed one document; if the model rejects it as too long, embed its two halves (split at a
        line, else word, boundary near the middle) and return their L2-normalised mean. Bounded
        depth: 2**6 = 64 pieces covers any chunk the chunker can emit by orders of magnitude."""
        try:
            return (await self._embed([self._doc_prefix + text]))[0]
        except EmbeddingInputRejected:
            left, right = _halves(text)
            if depth >= _MAX_SPLIT_DEPTH or not left or not right:
                raise
            log.warning("embedder_input_split", depth=depth + 1, chars=len(text))
            a = await self._embed_fitting(left, depth=depth + 1)
            b = await self._embed_fitting(right, depth=depth + 1)
            return _normalised_mean(a, b)

    async def embed_query(self, text: str) -> list[float]:
        return (await self._embed([self._query_prefix + text]))[0]

    async def _embed(self, inputs: list[str]) -> list[list[float]]:
        """One provider request, measured (ADR-0018): `ragforge_embed_requests_total{process,
        outcome}` and its duration. `outcome` is set where each failure is known; it stays
        "other" for parse errors, cancellation and anything unforeseen."""
        started = time.perf_counter()
        outcome = "other"
        try:
            payload = {
                "model": self.model,
                "input": inputs,
                "keep_alive": self._keep_alive,
                "options": {"num_ctx": self._num_ctx},
            }
            try:
                response = await self._client.post("/api/embed", json=payload)
            except httpx.TransportError as exc:
                outcome = "timeout" if isinstance(exc, httpx.TimeoutException) else "server"
                log.warning("embedder_unreachable", error=type(exc).__name__)
                raise EmbeddingFailed("embedding service unreachable") from exc
            if response.status_code in (429, 503):
                outcome = "server"
                raise UpstreamRateLimited(_retry_after(response))
            if response.is_error:
                outcome = "server" if response.status_code >= 500 else "other"
                # The message reaches API clients and jobs.last_error: a fixed category only
                # (never the provider's text, which can quote input); status goes to the log.
                category = _error_category(response)
                log.warning("embedder_http_error", status=response.status_code, category=category)
                # Only statuses that blame the INPUT are permanent; 404 (model not pulled yet)
                # and other config errors stay retryable because an operator can fix them.
                if response.status_code in (400, 413, 422):
                    raise EmbeddingInputRejected(f"embedding input rejected: {category}")
                raise EmbeddingFailed(f"embedding service error: {category}")
            vectors = self._parse(response, expected=len(inputs))
            outcome = "ok"
            return vectors
        finally:
            observe_embed(outcome, time.perf_counter() - started)

    def _parse(self, response: httpx.Response, *, expected: int) -> list[list[float]]:
        try:
            body: Any = response.json()
        except ValueError as exc:
            raise EmbeddingFailed("embedding service returned invalid JSON") from exc
        vectors = body.get("embeddings") if isinstance(body, dict) else None
        if not isinstance(vectors, list) or len(vectors) != expected:
            raise EmbeddingFailed(
                f"embedding service returned {_count(vectors)} embeddings, wanted {expected}"
            )
        out: list[list[float]] = []
        for vector in vectors:
            if not isinstance(vector, list) or len(vector) != self.dim:
                # The model name and dims go to the log only: the message reaches API clients
                # and jobs.last_error, where internal model names do not belong.
                log.warning(
                    "embedding_dim_mismatch", model=self.model, got=_count(vector), want=self.dim
                )
                raise EmbeddingFailed("embedding has the wrong number of dimensions for its space")
            floats = [float(x) for x in vector]
            if not all(math.isfinite(x) for x in floats):
                raise EmbeddingFailed("ollama returned a non-finite embedding")
            out.append(floats)
        return out


_MAX_SPLIT_DEPTH = 6


def _halves(text: str) -> tuple[str, str]:
    """Split near the middle at the closest newline, else space; else a hard cut."""
    mid = len(text) // 2
    for sep in ("\n", " "):
        left_cut, right_cut = text.rfind(sep, 0, mid), text.find(sep, mid)
        cuts = [c for c in (left_cut, right_cut) if c > 0]
        if cuts:
            cut = min(cuts, key=lambda c: abs(c - mid))
            return text[:cut].strip(), text[cut + 1 :].strip()
    return text[:mid], text[mid:]


def _normalised_mean(a: list[float], b: list[float]) -> list[float]:
    mean = [(x + y) / 2 for x, y in zip(a, b, strict=True)]
    norm = math.sqrt(sum(x * x for x in mean)) or 1.0
    return [x / norm for x in mean]


def _retry_after(response: httpx.Response) -> float | None:
    try:
        return float(response.headers.get("retry-after", ""))
    except ValueError:
        return None


def _error_category(response: httpx.Response) -> str:
    """A fixed category for Ollama's `error` field. The text itself is never returned: errors
    such as tokenization failures can echo fragments of the (tenant) input, and this message
    ends up in job error logs."""
    try:
        body = response.json()
    except ValueError:
        return "no detail"
    detail = body.get("error") if isinstance(body, dict) else None
    if not isinstance(detail, str) or not detail:
        return "no detail"
    lowered = detail.lower()
    for marker, category in _ERROR_CATEGORIES:
        if marker in lowered:
            return category
    return "unrecognised error"


def _count(value: object) -> int | str:
    return len(value) if isinstance(value, list) else "no"
