"""`InstrumentedChatModel`: the ChatModel port, measured (ADR-0018 R2-3).

Why a decorator on the port, wired in bootstrap, and not code in each adapter: every model call
of the API (chat nodes, the verifier, interview, extras, the eval verify route) goes through the
ONE ChatModel instance bootstrap builds, so wrapping it there measures all of them with no call
site able to forget, and both adapters (Ollama, Gemini) are measured identically.

The adapters know provider and model but not which prompt they are serving; the node or route
that built the prompt says so through `prompt_context.for_prompt`, read here at call time.

Outcomes come from the adapters' OWN error classes (they already name every failure category;
re-deriving them from HTTP codes here would drift). The one gap: a timeout, a 5xx and a broken
record are all the base `ChatModelError`, told apart only by the adapters' fixed messages, so
those prefixes are matched here and pinned by a test that drives the real adapters.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncGenerator, AsyncIterator, Sequence
from typing import Final

from ragforge_core.adapters.llm.chat_gemini import (
    ModelConfigurationError,
    QuotaExhausted,
    ResponseBlocked,
    StreamInterrupted,
)
from ragforge_core.adapters.llm.chat_ollama import ChatModelError, OutputTruncated
from ragforge_core.domain.errors import UpstreamRateLimited
from ragforge_core.domain.models import (
    ChatMessage,
    ChatStreamEvent,
    ChatTurnUsage,
    ToolSpec,
    UsageReport,
)
from ragforge_core.domain.ports import ChatModel
from ragforge_core.observability import metrics
from ragforge_core.observability.catalog import LLM_OUTCOMES
from ragforge_core.observability.prompt_context import prompt_name

# The base ChatModelError's fixed messages (adapters/llm/chat_ollama.py, chat_gemini.py).
_TIMEOUT_MESSAGES: Final = ("model provider timed out",)
_SERVER_MESSAGES: Final = ("model provider unreachable", "model provider returned HTTP 5")


def llm_outcome(exc: BaseException) -> str:
    """The catalog.LLM_OUTCOMES category of a failed model call. Order matters: the most
    specific class first (QuotaExhausted is a ModelConfigurationError, OutputLimitReached an
    OutputTruncated, every adapter error a ChatModelError)."""
    if isinstance(exc, UpstreamRateLimited):
        return "rate_limited"
    if isinstance(exc, QuotaExhausted):
        return "quota_exhausted"
    if isinstance(exc, ModelConfigurationError):
        return "config"
    if isinstance(exc, ResponseBlocked):
        return "blocked"
    if isinstance(exc, StreamInterrupted | asyncio.CancelledError | GeneratorExit):
        return "interrupted"
    if isinstance(exc, OutputTruncated):
        return "output_limit"
    if type(exc) is ChatModelError:
        message = str(exc)
        if message.startswith(_TIMEOUT_MESSAGES):
            return "timeout"
        if message.startswith(_SERVER_MESSAGES):
            return "server"
    return "other"


def _billed(exc: BaseException) -> ChatTurnUsage | None:
    """What a failed call still billed: truncated and blocked answers carry their usage."""
    usage = getattr(exc, "usage", None)
    return usage if isinstance(usage, ChatTurnUsage) else None


class InstrumentedChatModel:
    """Implements `domain.ports.ChatModel` by delegating to `inner` and recording, per call,
    `ragforge_llm_requests_total`, `ragforge_llm_request_duration_seconds` and
    `ragforge_llm_tokens_total`, labelled {provider, model, prompt[, outcome|direction]}."""

    def __init__(self, inner: ChatModel) -> None:
        self.inner = inner
        self.provider = inner.provider
        self.model = inner.model
        self._provider = metrics.bounded(inner.provider, metrics.PROVIDERS)
        self._model = metrics.model_label(inner.model)

    def stream(
        self,
        system: str,
        messages: Sequence[ChatMessage],
        tools: Sequence[ToolSpec] = (),
    ) -> AsyncIterator[ChatStreamEvent]:
        # The prompt is read NOW (inside the caller's for_prompt block), not on first iteration.
        return self._measured_stream(prompt_name.get(), system, messages, tools)

    async def _measured_stream(
        self,
        prompt: str,
        system: str,
        messages: Sequence[ChatMessage],
        tools: Sequence[ToolSpec],
    ) -> AsyncGenerator[ChatStreamEvent, None]:
        started = time.perf_counter()
        usage: ChatTurnUsage | None = None
        outcome = "interrupted"
        inner = self.inner.stream(system, messages, tools)
        try:
            async for event in inner:
                if isinstance(event, UsageReport):
                    usage = event.usage
                yield event
            outcome = "ok"
        except GeneratorExit:
            # The consumer closed the stream. After the usage report the call had finished.
            outcome = "ok" if usage is not None else "interrupted"
            raise
        except BaseException as exc:
            outcome, usage = llm_outcome(exc), usage or _billed(exc)
            raise
        finally:
            try:
                await _close(inner)  # the port contract: stop the upstream request promptly
            finally:
                self._record(prompt, outcome, time.perf_counter() - started, usage)

    async def complete_json(
        self,
        system: str,
        messages: Sequence[ChatMessage],
        schema: dict[str, object],
    ) -> tuple[dict[str, object], ChatTurnUsage]:
        prompt = prompt_name.get()
        started = time.perf_counter()
        try:
            payload, usage = await self.inner.complete_json(system, messages, schema)
        except BaseException as exc:
            self._record(prompt, llm_outcome(exc), time.perf_counter() - started, _billed(exc))
            raise
        self._record(prompt, "ok", time.perf_counter() - started, usage)
        return payload, usage

    def _record(
        self, prompt: str, outcome: str, seconds: float, usage: ChatTurnUsage | None
    ) -> None:
        labels = (self._provider, self._model, metrics.prompt_label(prompt))
        result = metrics.bounded(outcome, LLM_OUTCOMES)
        metrics.LLM_REQUESTS.labels(*labels, result).inc()
        metrics.LLM_DURATION.labels(*labels, result).observe(seconds)
        if usage is not None:
            metrics.LLM_TOKENS.labels(*labels, "prompt").inc(max(0, usage.prompt_tokens))
            metrics.LLM_TOKENS.labels(*labels, "completion").inc(max(0, usage.completion_tokens))


async def _close(stream: AsyncIterator[ChatStreamEvent]) -> None:
    aclose = getattr(stream, "aclose", None)
    if aclose is not None:
        await aclose()
