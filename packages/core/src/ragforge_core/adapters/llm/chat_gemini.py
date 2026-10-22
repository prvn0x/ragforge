"""Google Gemini ChatModel adapter: REST `streamGenerateContent` (v1beta, SSE) over httpx.

Why REST and not the SDK: no new dependency, and the adapter controls exactly what goes on the
wire and into exceptions (the SDK logs and re-raises upstream text that may echo prompts).

Wire shapes (generativelanguage.googleapis.com/v1beta):
- `POST /v1beta/models/{model}:streamGenerateContent?alt=sse`; each SSE `data:` event is one
  `GenerateContentResponse` (`candidates[0].content.parts`, `finishReason`, `promptFeedback`,
  `usageMetadata`). There is no "done" record: the stream is complete only once a candidate
  carries a `finishReason`;
- `systemInstruction` holds the system prompt; `contents` alternate roles `user` / `model`;
- tools go out as `tools[0].functionDeclarations`; the model answers with `functionCall` parts;
  results go back as `functionResponse` parts in a `user` turn.

Security:
- the API key travels ONLY in the `x-goog-api-key` header, never in the URL: URLs end up in
  access logs, proxies and httpx's own INFO log line. Redirects are never followed, so the header
  cannot be replayed to another host;
- exceptions carry fixed categories (HTTP status, Google's `error.status` enum, block reason),
  never `error.message` or any body text: Google echoes request fragments (prompt text) there;
- two log lines only, both without request content: `gemini_thinking_config_dropped` (model and
  configured budget, once per instance: the flip silently changes cost and latency) and
  `gemini_bare_400_not_thinking` (model, when the bare-400 guess is retired).

Stream contract (domain.ports.ChatModel), same as the Ollama adapter:
- yields TextDelta / ToolCall (complete, one per call) / exactly one UsageReport, LAST;
- tool-call ids are minted here (uuid4 hex), never provider ids. Google's own `functionCall.id`
  (when it sends one) is kept in-process next to the thought signature and echoed back as
  `functionCall.id` / `functionResponse.id` on replay, so parallel calls to the same tool pair by
  id, not only by order;
- the stream must end with a known `finishReason`, else `ChatModelError` (never a silent partial
  answer; `FINISH_REASON_UNSPECIFIED` is not a reason, so it fails closed too);
- a transient failure (rate limit, 5xx error event, timeout, dropped connection, cut-off or broken
  stream) AFTER any text or tool call was yielded raises `StreamInterrupted` instead of the
  retryable class: replaying the generation would duplicate output the consumer already showed
  or acted on;
- a `STOP` with no text and no tool call raises a plain (transient) `ChatModelError`: an empty
  answer is never passed on as a successful turn;
- every request carries `maxOutputTokens` (`ANSWER_MAX_OUTPUT_TOKENS` / `STRUCTURED_...`, plus
  the configured thinking); `MAX_TOKENS` raises `OutputLimitReached` (an `OutputTruncated`, with
  the spent usage) instead of the UsageReport;
- a blocked prompt or answer (`promptFeedback.blockReason`, or finishReason SAFETY, RECITATION,
  BLOCKLIST, PROHIBITED_CONTENT, SPII, LANGUAGE, OTHER, ...) raises `ResponseBlocked`, a typed
  error the agent turns into an abstain. Text already streamed before the block is NOT an answer;
- `aclose()` on the iterator exits the `client.stream(...)` block, closing the HTTP response.

Thinking: `generationConfig.thinkingConfig.thinkingBudget` comes from settings (default 0 = off
for latency). A 400 that is recognisably about the thinking config (a `BadRequest` field
violation on `thinking_config`, one of a few anchored phrases, or Google's bare detail-less
"Request contains an invalid argument.", which is all gemini-3.5-flash-lite says about a budget
of 0) gets ONE retry without it. The
instance stops sending the budget only once that retry is accepted (HTTP 200): an end user cannot
flip it by getting their own words echoed in an unrelated 400, because the retry carries the same
words and fails the same way. Thought parts are never surfaced.

Thought signatures: thinking models attach an opaque `thoughtSignature` to the first
`functionCall` part of a model turn and validate it when that turn is replayed. Domain `ToolCall`
cannot carry it, so the adapter keeps a bounded in-memory LRU map from our minted id to the
signature (a replay refreshes an entry's recency).
When a replayed turn's first call has no cached signature (eviction, restart, another replica,
or a resumed tool approval), Google's documented placeholder for injected/transferred calls is
sent instead, so validation is skipped rather than the conversation failing with a 400.

Read timeout: thought parts are not streamed, so the connection is silent while the model
thinks. The per-request read timeout therefore grows with the CONFIGURED thinking budget (bounded
by `MAX_THINKING_READ_S`), else a large or dynamic budget would time out before the first byte and
every retry would time out the same way. The allowance follows what the operator configured, not
whether `thinkingConfig` is still sent: after the fallback from budget 0 the model thinks at its
own (minimal, on the default model) level, so it gets a short allowance, not the dynamic one, and
a hung connection still fails within about 90 s rather than 300 s for the life of the process.

Errors carry no `retryable` flag: the class says it. `UpstreamRateLimited` is retryable after
its delay; plain `ChatModelError` (5xx, timeout, transport, broken stream) is transient, but only
before any output (see `StreamInterrupted`); `ModelConfigurationError`, `ModelRequestRejected`,
`ResponseBlocked` and `StreamInterrupted` are permanent for a blind retry. A 429 that names a
per-day quota is `QuotaExhausted` (a `ModelConfigurationError`), not a rate limit: retrying every
few minutes cannot succeed before the daily reset.
"""

from __future__ import annotations

import codecs
import json
import math
import re
from collections import OrderedDict
from collections.abc import AsyncGenerator, AsyncIterator, Mapping, Sequence
from contextlib import aclosing
from dataclasses import dataclass, field
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Any, Final, Self
from uuid import uuid4

import httpx
from pydantic import SecretStr

from ragforge_core.adapters.llm.chat_ollama import ChatModelError, OutputTruncated
from ragforge_core.config import Settings
from ragforge_core.domain.errors import RagforgeError, UpstreamRateLimited, ValidationFailed
from ragforge_core.domain.models import (
    ChatMessage,
    ChatStreamEvent,
    ChatTurnUsage,
    Role,
    TextDelta,
    ToolCall,
    ToolSpec,
    UsageReport,
)
from ragforge_core.observability.logging import get_logger

PROVIDER: Final = "gemini"
DEFAULT_BASE_URL: Final = "https://generativelanguage.googleapis.com"
API_KEY_HEADER: Final = "x-goog-api-key"
_DEFAULT_TIMEOUT: Final = httpx.Timeout(connect=5.0, read=60.0, write=30.0, pool=5.0)
CONNECT_RETRIES: Final = 2
BARE_MISSES_TO_DISTRUST: Final = 2
_MODEL_NAME: Final = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_STATUS_ENUM: Final = re.compile(r"^[A-Z][A-Z_]{0,63}$")
_MAX_ERROR_BYTES: Final = 16_384  # error bodies are small; never buffer an unbounded one
_log = get_logger(__name__)

# One SSE event, and one line while it is still arriving: enforced as bytes come in, so a
# runaway line without a newline is a broken stream long before it can exhaust memory.
_MAX_EVENT_CHARS: Final = 4 * 1024 * 1024
# SSE line terminators (WHATWG HTML 9.2.5): CRLF, LF or CR. Deliberately not str.splitlines(),
# which also splits on U+2028 and friends that are legal, unescaped, inside JSON strings.
_LINE_BREAK: Final = re.compile(r"\r\n|\r|\n")
_SIGNATURE_CACHE_SIZE: Final = 1024
# A real thought signature is a few hundred bytes to a few KiB. Larger ones are dropped (the
# replay then sends the placeholder), so 1024 cache entries stay bounded to about 64 MiB at worst
# instead of 1024 x the 4 MiB event cap.
_MAX_SIGNATURE_CHARS: Final = 64 * 1024
# Ceilings on generated tokens per request. Gemini's default output limit is tens of thousands of
# tokens, and the read timeout only bounds the gap BETWEEN chunks, so a looping generation would
# otherwise run (and bill) for minutes. A grounded answer over a few note snippets fits easily in
# 2048 tokens; a structured verdict quotes claims from such an answer, so it gets twice that.
# Hitting the ceiling is a MAX_TOKENS stop, surfaced as `OutputLimitReached`, never a silent cut.
# Constants, not settings: the frozen setting names do not include one (a future
# RAGFORGE_GEMINI_MAX_OUTPUT_TOKENS would replace them).
ANSWER_MAX_OUTPUT_TOKENS: Final = 2048
STRUCTURED_MAX_OUTPUT_TOKENS: Final = 4096
# Thinking tokens are output tokens on Gemini, so the ceiling is raised by what the configured
# thinking may spend: else a thinking model would hit MAX_TOKENS before writing its answer.
# Dynamic thinking (-1 / model default) has no stated size: the largest budget Flash models take.
_DYNAMIC_THINKING_TOKENS: Final = 24_576
# Budget 0 refused (the model must think): it thinks at its own default level, assumed small.
_REFUSED_ZERO_THINKING_TOKENS: Final = 2048
# Google's FunctionCall.id goes back on the wire only to Google; bounded so a hostile stream
# cannot grow the cache entry or the next request without limit.
_PROVIDER_CALL_ID: Final = re.compile(r"[\x21-\x7e]{1,256}")
# Thinking is silent on the wire. Generation runs well above this rate on every hosted Gemini
# model, so budget / rate is a generous upper bound on the silent gap a budget can cause.
_THINKING_TOKENS_PER_S: Final = 100.0
# A dynamic budget (-1) or the model default (no thinkingConfig) has no stated size.
_DYNAMIC_THINKING_ALLOWANCE_S: Final = 240.0
# Budget 0 was configured but the model refused it (it must think): the operator asked for no
# thinking, so the model's own default level applies ("minimal" on the default model). Short, but
# not zero; an operator who wants a heavily thinking model sets an explicit budget instead.
_REFUSED_ZERO_ALLOWANCE_S: Final = 30.0
# Never wait longer than this for the next byte, whatever the budget: a hung connection must
# still fail within minutes.
MAX_THINKING_READ_S: Final = 300.0
# Google's documented value for a functionCall whose real signature is unavailable (a call that
# was injected or transferred from another session): the API skips validation for that part.
SKIP_SIGNATURE: Final = "skip_thought_signature_validator"
# Delays beyond this are clamped: a hostile or buggy header must not park a job for days.
MAX_RETRY_AFTER_S: Final = 300.0
# Printable ASCII, no whitespace: anything else cannot be an HTTP header value, and httpx's
# UnicodeEncodeError would carry the whole key in its args.
_API_KEY: Final = re.compile(r"[\x21-\x7e]{1,512}")
# Phrases Google uses when it rejects the thinking config. Anchored on field names or exact
# wording, never a bare "think", because the message may echo end-user text.
_THINKING_REJECTION: Final = re.compile(
    r"thinking[_ ]?(?:config|budget|level)|thinkingconfig|thinkingbudget|"
    r"only works in thinking mode|thinking is not supported",
    re.IGNORECASE,
)
# Some models (gemini-3.5-flash-lite) reject `thinkingBudget: 0` with only this bare message and
# no details. It is a fixed server string, never echoed request content, so an exact match is safe.
_BARE_INVALID_ARGUMENT: Final = "Request contains an invalid argument."
_RATE_LIMITED: Final = frozenset({429, 503})
# google.rpc.QuotaFailure.violations[].quotaId is an identifier such as
# "GenerateRequestsPerDayPerProjectPerModel-FreeTier": safe to read, never echoed.
_QUOTA_ID: Final = re.compile(r"[A-Za-z0-9_.-]{1,128}")
_DAILY_QUOTA: Final = re.compile(r"PerDay")
_AUTH_FAILED: Final = frozenset({401, 403})


# ---------------------------------------------------------------------- errors
class ModelConfigurationError(ChatModelError):
    """The provider refused us for a reason no retry can fix: bad/revoked API key, permission
    denied, unknown model, unsupported region or billing precondition. Operator action needed."""

    title = "Model provider misconfigured"


class ModelRequestRejected(ChatModelError, ValidationFailed):
    """HTTP 400 INVALID_ARGUMENT: the request itself is invalid (schema, history shape, size).
    Permanent: resending the same request fails the same way. Status stays 502 (it is our request
    to the provider that is wrong, not the end user's input)."""

    status = 502
    title = "Model provider rejected the request"


class QuotaExhausted(ModelConfigurationError):
    """HTTP 429 naming a per-DAY quota (google.rpc.QuotaFailure): the key's daily allowance is
    spent, typically on the free tier. Not `UpstreamRateLimited`: a retry every few minutes
    would fail for hours while the user is told "rate limited". Operator action needed (wait for
    the daily reset, or move the key to a paid tier)."""

    title = "Model provider quota exhausted"


class OutputLimitReached(OutputTruncated):
    """`MAX_TOKENS`: the generation hit the output token ceiling (see
    `ANSWER_MAX_OUTPUT_TOKENS`). Still an `OutputTruncated` for callers that handle both
    providers; only the message differs, since on Gemini it is not the context window."""

    def __init__(self, usage: ChatTurnUsage) -> None:
        super().__init__(usage)
        self.args = ("model output truncated: output token limit reached",)


class ResponseBlocked(ChatModelError):
    """The provider refused to answer (prompt blocked, or the answer was stopped for safety,
    recitation, ...). The agent turns this into an abstain; any text streamed before it must be
    discarded. `reason` is a fixed lowercase category; `usage` is what was billed, if reported."""

    status = 422
    title = "Model response blocked"

    def __init__(self, stage: str, reason: str, usage: ChatTurnUsage | None = None) -> None:
        super().__init__(f"model {stage} blocked by provider policy: {reason}")
        self.stage = stage
        self.reason = reason
        self.usage = usage


class StreamInterrupted(ChatModelError):
    """A transient failure (rate limit, 5xx, timeout, dropped or cut-off stream) struck AFTER
    text or tool calls were already yielded. Not retryable as-is: a replay would emit duplicate
    text, or duplicate tool calls the agent may already have executed. The caller either fails
    the turn or restarts it from scratch after discarding what it showed. `cause` is the fixed
    category of the underlying failure: "rate_limited" or "transient"."""

    title = "Model stream interrupted"

    def __init__(self, cause: str) -> None:
        super().__init__(f"model stream interrupted after output: {cause}")
        self.cause = cause


class UnsupportedSchema(RagforgeError):
    """A JSON schema uses a construct Gemini's schema subset cannot express: a programming
    error in the caller's schema, raised before any request is sent.

    Deliberately NOT a `ValueError` (callers catch that for "the model sent unparseable JSON"
    and would turn a bug into a retry or a malformed-verdict path) and not a `ChatModelError`
    (that reads as a provider failure). It surfaces as an internal error (500)."""

    title = "Unsupported model output schema"


class _ThinkingUnsupported(Exception):
    """The model rejected `thinkingConfig` (e.g. a budget of 0 on a model that must think).
    `bare`: recognised only from Google's generic, detail-less 400, which also covers other
    invalid requests, so it is a guess (see `_accepted_chunks`)."""

    def __init__(self, *, bare: bool = False) -> None:
        super().__init__()
        self.bare = bare


# ---------------------------------------------------------------------- adapter
@dataclass(frozen=True, slots=True)
class _ProviderCall:
    """What Google attached to one of its function calls, kept in-process (keyed by our minted
    id) because domain `ToolCall` cannot carry it: the opaque thought signature and Google's own
    call id. Both go back only to Google, on replay of the turn that produced them."""

    signature: str | None = None
    provider_id: str | None = None


@dataclass(slots=True)
class _TurnState:
    """What one streamed response told us so far; judged once the stream ends."""

    finish_reason: str | None = None
    usage: Mapping[str, Any] = field(default_factory=dict)


class GeminiChatModel:
    """Implements `domain.ports.ChatModel` against the Gemini API (generativelanguage v1beta)."""

    provider: str = PROVIDER

    def __init__(
        self,
        client: httpx.AsyncClient,
        *,
        api_key: SecretStr,
        model: str,
        thinking_budget: int | None = 0,
        owns_client: bool = False,
    ) -> None:
        """`client` carries the base URL and timeouts. `thinking_budget=None` never sends
        `thinkingConfig` (the model's default)."""
        self.model = _model_name(model)
        self._client = client
        self._api_key = _api_key(api_key)
        # The configured budget never changes: it sizes the read timeout. Whether it still goes
        # on the wire is separate state, flipped once by the thinking-config fallback.
        self._thinking_budget = thinking_budget
        self._send_thinking = thinking_budget is not None
        # Whether Google's bare, detail-less 400 is still taken as a thinking rejection. Cleared
        # the first time a retry without thinkingConfig fails too.
        self._trust_bare = True
        self._bare_misses = 0  # consecutive definitive rejections of a bare-400 retry
        self._owns_client = owns_client
        self._calls: OrderedDict[str, _ProviderCall] = OrderedDict()

    @classmethod
    def from_settings(cls, settings: Settings, client: httpx.AsyncClient | None = None) -> Self:
        # Validate before opening a client, so a bad configuration cannot leak an open pool.
        model = _model_name(settings.gemini_model)
        api_key = _api_key(settings.gemini_api_key)
        owned = client is None
        http = client or httpx.AsyncClient(
            base_url=DEFAULT_BASE_URL,
            timeout=_DEFAULT_TIMEOUT,
            follow_redirects=False,
            # httpx retries ONLY a failure to establish the connection (connect error/timeout):
            # nothing was sent, so it is always safe and never duplicates a generation. Phase 2b
            # eval B6b lost 3 of 15 turns to connect timeouts/errors to the API host.
            transport=httpx.AsyncHTTPTransport(retries=CONNECT_RETRIES),
        )
        return cls(
            http,
            api_key=api_key,
            model=model,
            thinking_budget=settings.gemini_thinking_budget,
            owns_client=owned,
        )

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    def __repr__(self) -> str:  # the key must never appear, even in a debugger or a log line
        return f"GeminiChatModel(model={self.model!r})"

    # ------------------------------------------------------------------ ChatModel
    async def stream(
        self,
        system: str,
        messages: Sequence[ChatMessage],
        tools: Sequence[ToolSpec] = (),
    ) -> AsyncIterator[ChatStreamEvent]:
        body = self._request_body(system, messages, ANSWER_MAX_OUTPUT_TOKENS)
        if tools:
            body["tools"] = [{"functionDeclarations": [_function_declaration(t) for t in tools]}]
        state = _TurnState()
        emitted = False
        try:
            # aclosing: the consumer's aclose() must close the HTTP response now, not at GC.
            async with aclosing(self._accepted_chunks(body)) as chunks:
                async for chunk in chunks:
                    for event in self._events(chunk, state):
                        emitted = True
                        yield event
            usage = self._finished_usage(state)
            if not emitted:
                # Gemini sometimes ends a turn with STOP and no parts (seen after a
                # functionResponse, or when thinking consumed the output). Passing it on would
                # commit an empty draft; nothing was shown, so the base class keeps it retryable.
                raise ChatModelError("model returned an empty answer")
        except (UpstreamRateLimited, ChatModelError) as exc:
            cause = _retry_category(exc)
            if emitted and cause is not None:
                raise StreamInterrupted(cause) from None
            raise
        yield UsageReport(usage)

    async def complete_json(
        self,
        system: str,
        messages: Sequence[ChatMessage],
        schema: dict[str, object],
    ) -> tuple[dict[str, object], ChatTurnUsage]:
        """JSON mode (`responseMimeType` + `responseSchema`, converted by `to_gemini_schema`).
        Temperature 0: these calls grade and verify, so they should be repeatable. Streams, so
        the read timeout bounds the gap between chunks rather than the whole generation. Nothing
        is emitted before the result, so its errors keep their retry class."""
        body = self._request_body(system, messages, STRUCTURED_MAX_OUTPUT_TOKENS)
        body["generationConfig"] = {
            **body["generationConfig"],
            "temperature": 0,
            "responseMimeType": "application/json",
            "responseSchema": to_gemini_schema(schema),
        }
        state = _TurnState()
        parts: list[str] = []
        async with aclosing(self._accepted_chunks(body)) as chunks:
            async for chunk in chunks:
                for event in self._events(chunk, state):
                    if not isinstance(event, TextDelta):  # no tools were offered
                        raise ChatModelError("unexpected tool call in structured output")
                    parts.append(event.text)
        usage = self._finished_usage(state)
        return _parse_json_content("".join(parts)), usage

    async def _accepted_chunks(self, body: dict[str, Any]) -> AsyncGenerator[dict[str, Any]]:
        """Chunks of one generation, with the thinking-config fallback.

        `_ThinkingUnsupported` can only come from the status line, before any chunk, so the
        retry never duplicates output. The budget is dropped for good only when the retry is
        accepted: if the 400 was really about something else (say, echoed user text that
        happens to match), the retry fails too, the error surfaces, and the setting survives.
        Concurrent turns may each pay one retry; the end state is the same (no lock needed,
        the assignment happens between awaits)."""
        budget = self._thinking_budget if self._send_thinking else None
        try:
            async with aclosing(self._chunks(body, budget)) as chunks:
                async for chunk in chunks:
                    self._bare_misses = 0  # any accepted response breaks a run of misses
                    yield chunk
            return
        except _ThinkingUnsupported as exc:
            if exc.bare and not self._trust_bare:
                # A bare-400 retry already failed once: this generic 400 is about something else.
                # Surface it as the plain rejection it is, without paying a second request
                # (review round 6, P2: on a 20-requests/day free tier each one counts).
                raise _status_error(
                    httpx.codes.BAD_REQUEST, _BARE_INFO, None, thinking_sent=False
                ) from None
            bare = exc.bare
        accepted = False
        try:
            async with aclosing(self._chunks(body, None)) as chunks:
                async for chunk in chunks:
                    if not accepted:
                        accepted = True
                        self._bare_misses = 0
                        if self._send_thinking:
                            self._send_thinking = False
                            _log.warning(
                                "gemini_thinking_config_dropped",
                                model=self.model,
                                configured_budget=self._thinking_budget,
                            )
                    yield chunk
        except ModelRequestRejected:
            # Only a DEFINITIVE 400 on the retry (sent without thinkingConfig) shows the guess was
            # wrong. A timeout, a 5xx or a quota error says nothing about it, and clearing trust
            # on one would disable the fallback for good on a model that needs it (review round
            # 7, P1: every later turn on flash-lite with budget 0 would fail until restart).
            if bare and not accepted:
                # Two DEFINITIVE rejections in a row, not one: a single request that is invalid
                # for another reason must not latch the fallback off for every tenant until
                # restart (review round 8, P2). Any accepted retry resets the count.
                self._bare_misses += 1
                if self._bare_misses >= BARE_MISSES_TO_DISTRUST:
                    self._trust_bare = False
                    _log.warning("gemini_bare_400_not_thinking", model=self.model)
            raise

    # ------------------------------------------------------------------ request

    def _request_body(
        self, system: str, messages: Sequence[ChatMessage], max_output: int
    ) -> dict[str, Any]:
        """`max_output` is the ceiling for the visible output; `_chunks` adds the thinking
        headroom once it knows whether `thinkingConfig` goes on this request."""
        contents, extra_system = self._contents(messages)
        config = {"candidateCount": 1, "maxOutputTokens": max_output}
        body: dict[str, Any] = {"contents": contents, "generationConfig": config}
        instructions = [text for text in (system, *extra_system) if text]
        if instructions:
            body["systemInstruction"] = {"parts": [{"text": text} for text in instructions]}
        return body

    def _contents(self, messages: Sequence[ChatMessage]) -> tuple[list[dict[str, Any]], list[str]]:
        """Domain messages -> Gemini `contents`. Gemini has no system or tool role: system
        messages join `systemInstruction`, tool results become `functionResponse` parts of a
        `user` turn. Consecutive turns of the same role are merged, so the parallel results of
        one model turn arrive together, as Gemini expects. Results pair with calls by Google's
        call id when it sent one, else by name and order (callers keep call order); our ids
        never leave the process."""
        contents: list[dict[str, Any]] = []
        extra_system: list[str] = []
        names: dict[str, tuple[str, str | None]] = {}
        for message in messages:
            if message.role is Role.system:
                extra_system.append(message.content)
                continue
            role, parts = self._turn(message, names)
            if not parts:  # Gemini rejects a turn with no parts
                continue
            if contents and contents[-1]["role"] == role:
                contents[-1]["parts"].extend(parts)
            else:
                contents.append({"role": role, "parts": parts})
        for content in contents:
            if content["role"] == "model":
                _sign_first_call(content["parts"])
        return contents, extra_system

    def _turn(
        self, message: ChatMessage, names: dict[str, tuple[str, str | None]]
    ) -> tuple[str, list[dict[str, Any]]]:
        if message.role is Role.tool:
            known = names.get(message.tool_call_id or "")
            if known is None:  # same fixed category as Google's 400, but before any request
                raise ModelRequestRejected(
                    "tool result does not answer a tool call in this history"
                )
            name, provider_id = known
            # "output" is the documented key for a function's result (vs "error").
            response: dict[str, Any] = {"name": name, "response": {"output": message.content}}
            if provider_id is not None:
                response["id"] = provider_id
            return "user", [{"functionResponse": response}]
        parts: list[dict[str, Any]] = [{"text": message.content}] if message.content else []
        if message.role is Role.user:
            return "user", parts
        for call in message.tool_calls:
            meta = self._recall(call.id)
            names[call.id] = (call.name, meta.provider_id)
            function_call: dict[str, Any] = {"name": call.name, "args": call.arguments}
            if meta.provider_id is not None:
                function_call["id"] = meta.provider_id
            part: dict[str, Any] = {"functionCall": function_call}
            if meta.signature is not None:
                part["thoughtSignature"] = meta.signature
            parts.append(part)
        return "model", parts

    # ------------------------------------------------------------------ transport
    async def _chunks(
        self, body: dict[str, Any], thinking: int | None
    ) -> AsyncGenerator[dict[str, Any]]:
        """Decoded response chunks. Exactly one place maps transport failures, before or
        mid-stream, to fixed messages (no URL, no exception text)."""
        request = _with_thinking(body, thinking, self._thinking_budget)
        failure: str
        try:
            async with self._client.stream(
                "POST",
                f"/v1beta/models/{self.model}:streamGenerateContent",
                params={"alt": "sse"},
                json=request,
                headers={API_KEY_HEADER: self._api_key.get_secret_value()},
                timeout=_thinking_timeout(
                    self._client.timeout, self._thinking_budget, sent=thinking is not None
                ),
                follow_redirects=False,  # a redirect would replay the key header elsewhere
            ) as resp:
                if resp.status_code != httpx.codes.OK:
                    await _raise_for_status(resp, thinking_sent=thinking is not None)
                async for data in _sse_data(resp):
                    yield _decode_chunk(data)
            return
        except httpx.TimeoutException:
            failure = "model provider timed out"
        except httpx.HTTPError:  # ConnectError, ReadError, RemoteProtocolError, DecodingError...
            failure = "model provider unreachable"
        # Raised AFTER the except block, so the new error has no `__context__`: `from None` only
        # hides the httpx exception from tracebacks, it stays reachable, and its `.request`
        # carries the key header for any reporter that walks the chain or captures locals.
        raise ChatModelError(failure)

    # ------------------------------------------------------------------ response
    def _events(self, chunk: Mapping[str, Any], state: _TurnState) -> list[ChatStreamEvent]:
        """Events of one chunk; records finish reason and the latest (cumulative) usage."""
        usage = chunk.get("usageMetadata")
        if isinstance(usage, dict):
            state.usage = usage
        _check_prompt_feedback(chunk.get("promptFeedback"), self._usage(state))
        candidate = _first_candidate(chunk.get("candidates"))
        if candidate is None:
            return []
        reason = candidate.get("finishReason")
        if reason is not None:
            if not isinstance(reason, str):
                raise ChatModelError("malformed finish reason from model provider")
            state.finish_reason = reason
        events: list[ChatStreamEvent] = []
        for part in _parts(candidate.get("content")):
            if part.get("thought") is True:  # thinking text is never surfaced
                continue
            text = part.get("text")
            if isinstance(text, str) and text:
                events.append(TextDelta(text))
            if "functionCall" in part:
                events.append(self._tool_call(part))
        return events

    def _tool_call(self, part: Mapping[str, Any]) -> ToolCall:
        raw = part.get("functionCall")
        name, arguments = _function_call(raw)
        call = ToolCall(id=uuid4().hex, name=name, arguments=arguments)
        signature = part.get("thoughtSignature")
        provider_id = raw.get("id") if isinstance(raw, dict) else None
        meta = _ProviderCall(
            signature=(
                signature
                if isinstance(signature, str) and 0 < len(signature) <= _MAX_SIGNATURE_CHARS
                else None
            ),
            provider_id=(
                provider_id
                if isinstance(provider_id, str) and _PROVIDER_CALL_ID.fullmatch(provider_id)
                else None
            ),
        )
        if meta != _ProviderCall():
            self._remember(call.id, meta)
        return call

    def _recall(self, call_id: str) -> _ProviderCall:
        """What Google attached to one of our calls, refreshing its recency: a conversation that
        is still being replayed (a long tool approval, many concurrent chats) must not lose its
        signature to insertion age alone."""
        meta = self._calls.get(call_id)
        if meta is None:
            return _ProviderCall()
        self._calls.move_to_end(call_id)
        return meta

    def _remember(self, call_id: str, meta: _ProviderCall) -> None:
        """Bounded LRU (reads refresh recency in `_recall`): an evicted entry degrades to the
        placeholder signature and pairing by order, never to a failed request."""
        self._calls[call_id] = meta
        self._calls.move_to_end(call_id)
        while len(self._calls) > _SIGNATURE_CACHE_SIZE:
            self._calls.popitem(last=False)

    def _finished_usage(self, state: _TurnState) -> ChatTurnUsage:
        """Usage of a completed generation. A stream without a finish reason was cut off; a
        token-limit stop is a truncation and a policy stop is a block, never a normal end."""
        usage = self._usage(state)
        reason = state.finish_reason
        if reason is None:
            raise ChatModelError("model stream ended before completion")
        if reason in _NORMAL_FINISH:
            return usage
        if reason == "MAX_TOKENS":
            raise OutputLimitReached(usage)
        if reason in _BLOCKED_FINISH:
            raise ResponseBlocked("response", reason.lower(), usage)
        if reason in _BAD_TOOL_FINISH:
            raise ChatModelError(f"model provider stopped: {reason.lower()}")
        raise ChatModelError("unexpected finish reason from model provider")

    def _usage(self, state: _TurnState) -> ChatTurnUsage:
        """Thinking tokens are billed as output, so they count as completion tokens."""
        meta = state.usage
        return ChatTurnUsage(
            provider=PROVIDER,
            model=self.model,
            prompt_tokens=_count(meta.get("promptTokenCount"))
            + _count(meta.get("toolUsePromptTokenCount")),
            completion_tokens=_count(meta.get("candidatesTokenCount"))
            + _count(meta.get("thoughtsTokenCount")),
        )


# ---------------------------------------------------------------------- finish reasons
# Only STOP is a normal end. FINISH_REASON_UNSPECIFIED is the proto default ("no reason given"):
# accepting it would let a stream that ended for no stated reason pass as a complete answer.
_NORMAL_FINISH: Final = frozenset({"STOP"})
_BLOCKED_FINISH: Final = frozenset(
    {
        "SAFETY",
        "RECITATION",
        "LANGUAGE",
        "OTHER",
        "BLOCKLIST",
        "PROHIBITED_CONTENT",
        "SPII",
        "IMAGE_SAFETY",
        "IMAGE_PROHIBITED_CONTENT",
        "IMAGE_OTHER",
        "IMAGE_RECITATION",
    }
)
_BAD_TOOL_FINISH: Final = frozenset(
    {"MALFORMED_FUNCTION_CALL", "UNEXPECTED_TOOL_CALL", "TOO_MANY_TOOL_CALLS"}
)
_PROMPT_BLOCK_REASONS: Final = frozenset(
    {"SAFETY", "OTHER", "BLOCKLIST", "PROHIBITED_CONTENT", "IMAGE_SAFETY"}
)


# ---------------------------------------------------------------------- request mapping
def _model_name(model: str) -> str:
    """The model name is interpolated into the URL path: allow only a plain model id."""
    name = model.removeprefix("models/")
    if not _MODEL_NAME.fullmatch(name):
        raise ModelConfigurationError("invalid model name for provider")
    return name


def _api_key(api_key: SecretStr) -> SecretStr:
    """Stripped (a key pasted with a newline would be an invalid header), non-empty and
    printable ASCII. Checked here (run at bootstrap, so still at startup) rather than in
    Settings: pydantic puts the rejected input value into its ValidationError text."""
    key = api_key.get_secret_value().strip()
    if not key:
        raise ModelConfigurationError("model provider API key is not configured")
    if not _API_KEY.fullmatch(key):
        raise ModelConfigurationError("model provider API key has invalid characters")
    return SecretStr(key)


def _sign_first_call(parts: list[dict[str, Any]]) -> None:
    """Gemini validates the signature on the FIRST functionCall part of each model turn (parallel
    calls carry one signature, on the first). If ours is gone, send the documented placeholder:
    a missing signature is a hard 400 on models that enforce it, the placeholder is not."""
    for part in parts:
        if "functionCall" in part:
            part.setdefault("thoughtSignature", SKIP_SIGNATURE)
            return


def _thinking_timeout(base: httpx.Timeout, budget: int | None, *, sent: bool) -> httpx.Timeout:
    """The client's timeouts with the read timeout stretched for silent thinking time.

    `budget` is the CONFIGURED budget; `sent` says whether `thinkingConfig` is on this request.
    Never shorter than the client's own read timeout, never longer than `MAX_THINKING_READ_S`
    unless the client's own timeout already is; no read timeout stays no read timeout."""
    allowance = _thinking_allowance(budget, sent=sent)
    if base.read is None or allowance == 0:
        return base
    read = max(base.read, min(base.read + allowance, MAX_THINKING_READ_S))
    return httpx.Timeout(connect=base.connect, read=read, write=base.write, pool=base.pool)


def _thinking_allowance(budget: int | None, *, sent: bool) -> float:
    """Seconds of silent thinking a request may cause.

    - 0 sent: no thinking, no allowance;
    - 0 refused (fallback, not sent): the model's default level, a short fixed allowance;
    - positive: the time to produce that many thought tokens (kept after a fallback: the
      operator's stated expectation is the best bound there is);
    - -1 (dynamic) or None (model default by choice): no stated size, the full allowance."""
    if budget == 0:
        return 0.0 if sent else _REFUSED_ZERO_ALLOWANCE_S
    if budget is None or budget < 0:
        return _DYNAMIC_THINKING_ALLOWANCE_S
    return budget / _THINKING_TOKENS_PER_S


def _retry_category(exc: Exception) -> str | None:
    """The retry class of an error, or None when it is permanent. Only the exact base
    `ChatModelError` is transient: every subclass names a permanent category by contract."""
    if isinstance(exc, UpstreamRateLimited):
        return "rate_limited"
    if type(exc) is ChatModelError:
        return "transient"
    return None


def _with_thinking(
    body: dict[str, Any], sent: int | None, configured: int | None
) -> dict[str, Any]:
    """The request as sent: `thinkingConfig` when a budget is `sent`, and the output ceiling
    raised by the thinking the CONFIGURED budget may spend (thought tokens count as output)."""
    config = dict(body["generationConfig"])
    if sent is not None:
        config["thinkingConfig"] = {"thinkingBudget": sent}
    config["maxOutputTokens"] = config["maxOutputTokens"] + _thinking_tokens(
        configured, sent=sent is not None
    )
    return {**body, "generationConfig": config}


def _thinking_tokens(budget: int | None, *, sent: bool) -> int:
    """Thought tokens a request may spend, mirroring `_thinking_allowance`."""
    if budget == 0:
        return 0 if sent else _REFUSED_ZERO_THINKING_TOKENS
    if budget is None or budget < 0:
        return _DYNAMIC_THINKING_TOKENS
    return budget


def _function_declaration(tool: ToolSpec) -> dict[str, Any]:
    """Gemini rejects an OBJECT schema with no properties, so a tool without arguments is
    declared without `parameters` at all (the one place an empty object is meaningful)."""
    declaration: dict[str, Any] = {"name": tool.name, "description": tool.description}
    if not _is_argumentless(tool.parameters):
        declaration["parameters"] = to_gemini_schema(tool.parameters)
    return declaration


def _is_argumentless(schema: Mapping[str, object]) -> bool:
    if "$ref" in schema or "anyOf" in schema or schema.get("properties"):
        return False
    return schema.get("type") in (None, "object") and "items" not in schema


# Gemini's `Schema` is an OpenAPI 3.0 subset. Kept: the keywords below. Dropped (they only
# loosen what the model is told; callers validate the result anyway): `$schema`, `$id`, `title`,
# `additionalProperties`, `default`, `examples`, unsupported `format`s, non-string `enum`s,
# `exclusiveMinimum/Maximum`, `uniqueItems`, `$comment`. Rejected (cannot be expressed safely):
# `oneOf`, `allOf`, `not`, conditionals, tuple `items`, recursive `$ref`, several non-null types,
# and an OBJECT without properties (a map / dict-like field): Gemini rejects it at request time
# with a permanent 400, so it fails here, at conversion, where the bug is.
_TYPES: Final = {
    "string": "STRING",
    "number": "NUMBER",
    "integer": "INTEGER",
    "boolean": "BOOLEAN",
    "array": "ARRAY",
    "object": "OBJECT",
}
_COPIED: Final = (
    "description",
    "minItems",
    "maxItems",
    "minLength",
    "maxLength",
    "pattern",
    "minimum",
    "maximum",
    "minProperties",
    "maxProperties",
)
_FORMATS: Final = {
    "STRING": frozenset({"date-time", "enum"}),
    "NUMBER": frozenset({"float", "double"}),
    "INTEGER": frozenset({"int32", "int64"}),
}
_REJECTED: Final = ("oneOf", "allOf", "not", "if", "then", "else", "patternProperties")


def to_gemini_schema(schema: Mapping[str, object]) -> dict[str, Any]:
    """Convert a JSON schema (as pydantic and our prompts write them) to Gemini's subset.
    Local `$ref`s into `$defs`/`definitions` are inlined. Raises `UnsupportedSchema`."""
    defs: dict[str, object] = {}
    for key in ("$defs", "definitions"):
        found = schema.get(key)
        if isinstance(found, dict):
            defs.update({f"#/{key}/{name}": value for name, value in found.items()})
    return _convert(schema, defs, ())


def _convert(node: object, defs: Mapping[str, object], refs: tuple[str, ...]) -> dict[str, Any]:
    if not isinstance(node, Mapping):
        raise UnsupportedSchema("schema node must be an object")
    ref = node.get("$ref")
    if ref is not None:
        if not isinstance(ref, str) or ref not in defs or ref in refs:
            raise UnsupportedSchema("only non-recursive local $ref is supported")
        return _convert(defs[ref], defs, (*refs, ref))
    for keyword in _REJECTED:
        if keyword in node:
            raise UnsupportedSchema(f"'{keyword}' is not supported by Gemini schemas")
    out: dict[str, Any] = {}
    if "anyOf" in node:
        out.update(_any_of(node["anyOf"], defs, refs))
    type_name, nullable = _type_of(node)
    if type_name is not None:
        out["type"] = type_name
    if nullable:
        out["nullable"] = True
    if "type" not in out and "anyOf" not in out:
        raise UnsupportedSchema("every schema node needs a type")
    out.update({key: node[key] for key in _COPIED if key in node})
    _copy_format(node, out)
    _copy_enum(node, out)
    _copy_children(node, out, defs, refs)
    if out.get("type") == "OBJECT" and not out.get("properties"):
        raise UnsupportedSchema("an object needs properties (maps are not supported by Gemini)")
    return out


def _any_of(branches: object, defs: Mapping[str, object], refs: tuple[str, ...]) -> dict[str, Any]:
    """`anyOf: [X, {type: null}]` (pydantic's Optional) becomes X with `nullable: true`."""
    if not isinstance(branches, list) or not branches:
        raise UnsupportedSchema("anyOf must be a non-empty list")
    is_null = [isinstance(b, Mapping) and b.get("type") == "null" for b in branches]
    rest = [_convert(b, defs, refs) for b, null in zip(branches, is_null, strict=True) if not null]
    if not rest:
        raise UnsupportedSchema("anyOf needs a non-null branch")
    merged: dict[str, Any] = rest[0] if len(rest) == 1 else {"anyOf": rest}
    return {**merged, "nullable": True} if any(is_null) else merged


def _type_of(node: Mapping[str, object]) -> tuple[str | None, bool]:
    raw = node.get("type")
    names = [raw] if isinstance(raw, str) else raw if isinstance(raw, list) else []
    nullable = "null" in names
    concrete = [n for n in names if n != "null"]
    if len(concrete) > 1:
        raise UnsupportedSchema("several non-null types are not supported")
    if concrete:
        mapped = _TYPES.get(str(concrete[0]))
        if mapped is None:
            raise UnsupportedSchema("unknown JSON schema type")
        return mapped, nullable
    if raw is not None and not names:
        raise UnsupportedSchema("type must be a string or a list")
    if "properties" in node:
        return "OBJECT", nullable
    if "items" in node:
        return "ARRAY", nullable
    enum = node.get("enum")
    if isinstance(enum, list) and enum:
        # Gemini enums are string-only. Inferring STRING for `[1, 2]` would then drop the enum
        # and tell the model "string" for what is an integer field.
        if not all(isinstance(value, str) for value in enum):
            raise UnsupportedSchema("an enum without a type must list only strings")
        return "STRING", nullable
    if isinstance(node.get("const"), str):
        return "STRING", nullable
    return None, nullable


def _copy_format(node: Mapping[str, object], out: dict[str, Any]) -> None:
    fmt = node.get("format")
    if isinstance(fmt, str) and fmt in _FORMATS.get(str(out.get("type")), frozenset()):
        out["format"] = fmt


def _copy_enum(node: Mapping[str, object], out: dict[str, Any]) -> None:
    """Gemini enums are string-only; a `const` string is a one-value enum."""
    values = node.get("enum")
    const = node.get("const")
    if isinstance(const, str):
        values = [const]
    strings = isinstance(values, list) and values and all(isinstance(v, str) for v in values)
    if strings and isinstance(values, list) and out.get("type") == "STRING":
        out["enum"] = list(values)


def _copy_children(
    node: Mapping[str, object],
    out: dict[str, Any],
    defs: Mapping[str, object],
    refs: tuple[str, ...],
) -> None:
    properties = node.get("properties")
    if isinstance(properties, Mapping):
        out["properties"] = {
            str(name): _convert(child, defs, refs) for name, child in properties.items()
        }
        out["propertyOrdering"] = list(out["properties"])  # deterministic output order
        required = node.get("required")
        if isinstance(required, list):
            out["required"] = [r for r in required if r in out["properties"]]
    items = node.get("items")
    if items is not None:
        if isinstance(items, list):
            raise UnsupportedSchema("tuple-style items are not supported")
        out["items"] = _convert(items, defs, refs)


# ---------------------------------------------------------------------- response mapping
async def _sse_data(resp: httpx.Response) -> AsyncGenerator[str]:
    """Server-sent events -> the `data` payload of each event (multi-line data joined with
    newlines, comments and other fields ignored, a final event without a blank line kept)."""
    lines: list[str] = []
    size = 0
    async for line in _sse_lines(resp):
        if not line:
            if lines:
                yield "\n".join(lines)
                lines, size = [], 0
            continue
        if line.startswith(":"):
            continue
        name, _, value = line.partition(":")
        if name != "data":
            continue
        value = value.removeprefix(" ")
        size += len(value)
        if size > _MAX_EVENT_CHARS:
            raise ChatModelError("oversized event from model provider")
        lines.append(value)
    if lines:
        yield "\n".join(lines)


async def _sse_lines(resp: httpx.Response) -> AsyncGenerator[str]:
    """Lines of the body, split here rather than by `aiter_lines()`: httpx buffers until it
    sees a line break, so an endless line would be held in memory in full before any size check
    could run. Here the unterminated tail is bounded on every network read. SSE is always UTF-8;
    invalid bytes become U+FFFD and then fail JSON decoding as a malformed event."""
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    # The unterminated tail as pieces, joined only when a line break arrives: rebuilding and
    # re-splitting the whole tail on every read was quadratic (a 4 MiB line in 1 KiB reads took
    # 36 s of event-loop CPU; final review, P2). Each character is now joined and split once.
    parts: list[str] = []
    size = 0
    async for raw in resp.aiter_bytes():
        chunk = decoder.decode(raw)
        held_cr = bool(parts) and parts[-1].endswith("\r")
        if not held_cr and "\n" not in chunk and "\r" not in chunk:
            parts.append(chunk)
            size += len(chunk)
        else:
            lines, tail = _split_lines("".join(parts) + chunk)
            for line in lines:
                yield line
            parts, size = [tail], len(tail)
        if size > _MAX_EVENT_CHARS:
            raise ChatModelError("oversized event from model provider")
    lines, pending = _split_lines("".join(parts) + decoder.decode(b"", final=True))
    for line in lines:
        yield line
    if pending:  # a last line without a terminator still counts
        yield pending.removesuffix("\r")


def _split_lines(text: str) -> tuple[list[str], str]:
    """Complete lines and the unterminated tail. A trailing CR is held back with the tail: the
    LF of its CRLF may arrive in the next read, and splitting there would invent a blank line,
    which in SSE ends the event early."""
    held = text.endswith("\r")
    *lines, tail = _LINE_BREAK.split(text[:-1] if held else text)
    return lines, (tail + "\r" if held else tail)


def _decode_chunk(data: str) -> dict[str, Any]:
    chunk = _json_or_none(data)
    if not isinstance(chunk, dict):
        raise ChatModelError("malformed event from model provider")
    error = chunk.get("error")
    if error is not None:  # an error reported mid-stream, after the 200 status line
        info = _ErrorInfo.parse(error)
        code = info.code if info.code is not None else 500
        raise _status_error(code, info, info.retry_delay, thinking_sent=False)
    return chunk


def _check_prompt_feedback(feedback: object, usage: ChatTurnUsage) -> None:
    if feedback is None:
        return
    if not isinstance(feedback, dict):
        raise ChatModelError("malformed prompt feedback from model provider")
    reason = feedback.get("blockReason")
    if reason is None or reason == "BLOCK_REASON_UNSPECIFIED":
        return
    # Any other value blocks (fail closed); only known enum values are echoed as the category.
    known = isinstance(reason, str) and reason in _PROMPT_BLOCK_REASONS
    category = str(reason).lower() if known else "other"
    raise ResponseBlocked("prompt", category, usage)


def _first_candidate(candidates: object) -> Mapping[str, Any] | None:
    if candidates is None:
        return None
    if not isinstance(candidates, list):
        raise ChatModelError("malformed candidates from model provider")
    if not candidates:
        return None
    first = candidates[0]
    if not isinstance(first, dict):
        raise ChatModelError("malformed candidate from model provider")
    return first


def _parts(content: object) -> list[Mapping[str, Any]]:
    if content is None:
        return []
    parts = content.get("parts", []) if isinstance(content, dict) else None
    if not isinstance(parts, list) or not all(isinstance(p, dict) for p in parts):
        raise ChatModelError("malformed content from model provider")
    return parts


def _function_call(raw: object) -> tuple[str, dict[str, Any]]:
    if not isinstance(raw, dict) or not isinstance(raw.get("name"), str) or not raw["name"]:
        raise ChatModelError("malformed tool call from model provider")
    arguments = raw.get("args", {})
    if arguments is None:
        arguments = {}
    if not isinstance(arguments, dict):
        raise ChatModelError("tool call arguments must be a JSON object")
    return raw["name"], arguments


def _parse_json_content(content: str) -> dict[str, object]:
    if not content.strip():
        raise ChatModelError("structured output missing")
    value = _json_or_none(content)
    if value is None:
        raise ChatModelError("structured output is not valid JSON")
    if not isinstance(value, dict):
        raise ChatModelError("structured output must be a JSON object")
    return value


def _json_or_none(text: str) -> object:
    """Parsed JSON, or None when it is not JSON (a literal `null` is never valid here either).
    Callers raise outside any except block: a JSONDecodeError keeps the whole document in `.doc`,
    which is model output, and must not ride along as `__context__`."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _count(value: object) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else 0


# ---------------------------------------------------------------------- errors from HTTP
@dataclass(frozen=True, slots=True)
class _ErrorInfo:
    """The SAFE parts of a google.rpc.Status body: enums and numbers only. `message` and
    `BadRequest.fieldViolations[].field` are read solely to recognise the thinking-config
    rejection and never leave this module."""

    code: int | None = None
    status: str | None = None
    reasons: frozenset[str] = frozenset()
    retry_delay: float | None = None
    thinking_rejected: bool = False
    bare: bool = False  # only Google's generic, detail-less 400 matched
    daily_quota: bool = False

    @classmethod
    def parse(cls, error: object) -> _ErrorInfo:
        if not isinstance(error, dict):
            return cls()
        code = error.get("code")
        status = error.get("status")
        message = error.get("message")
        details = error.get("details")
        reasons: set[str] = set()
        delay: float | None = None
        thinking_field = False
        daily = False
        bare = message == _BARE_INVALID_ARGUMENT and not details
        for detail in details if isinstance(details, list) else []:
            if not isinstance(detail, dict):
                continue
            reason = detail.get("reason")
            if isinstance(reason, str) and _STATUS_ENUM.fullmatch(reason):
                reasons.add(reason)
            delay = delay if delay is not None else _duration(detail.get("retryDelay"))
            thinking_field = thinking_field or _violates_thinking(detail.get("fieldViolations"))
            daily = daily or _daily_quota(detail.get("violations"))
        return cls(
            code=code if isinstance(code, int) and not isinstance(code, bool) else None,
            status=status if isinstance(status, str) and _STATUS_ENUM.fullmatch(status) else None,
            reasons=frozenset(reasons),
            retry_delay=delay,
            bare=bare and not thinking_field,
            thinking_rejected=thinking_field
            or bare
            or (isinstance(message, str) and _THINKING_REJECTION.search(message) is not None),
            daily_quota=daily,
        )


_BARE_INFO: Final = _ErrorInfo(code=400, status="INVALID_ARGUMENT")


def _daily_quota(violations: object) -> bool:
    """google.rpc.QuotaFailure: does any violated quota reset only daily?"""
    return isinstance(violations, list) and any(
        isinstance(v, dict)
        and isinstance(v.get("quotaId"), str)
        and _QUOTA_ID.fullmatch(v["quotaId"]) is not None
        and _DAILY_QUOTA.search(v["quotaId"]) is not None
        for v in violations
    )


def _violates_thinking(violations: object) -> bool:
    """google.rpc.BadRequest: field paths are API field names, never request content."""
    return isinstance(violations, list) and any(
        isinstance(v, dict)
        and isinstance(v.get("field"), str)
        and "thinking" in v["field"].lower().replace("_", "")
        for v in violations
    )


async def _raise_for_status(resp: httpx.Response, *, thinking_sent: bool) -> None:
    info = _ErrorInfo.parse(_json_error(await _read_bounded(resp)))
    retry_after = _retry_after_header(resp.headers.get("retry-after"))
    raise _status_error(
        resp.status_code,
        info,
        retry_after if retry_after is not None else info.retry_delay,
        thinking_sent=thinking_sent,
    )


def _status_error(
    code: int, info: _ErrorInfo, retry_after: float | None, *, thinking_sent: bool
) -> Exception:
    """Fixed categories only: the HTTP code and Google's status enum, never the message."""
    category = info.status or f"HTTP {code}"
    if code == httpx.codes.TOO_MANY_REQUESTS and info.daily_quota:
        return QuotaExhausted("model provider daily quota exhausted")
    if code in _RATE_LIMITED:
        return UpstreamRateLimited(retry_after)
    if code in _AUTH_FAILED or "API_KEY_INVALID" in info.reasons:
        return ModelConfigurationError(f"model provider rejected the credentials: {category}")
    if code == httpx.codes.NOT_FOUND:
        return ModelConfigurationError(f"model not found at provider: {category}")
    if code == httpx.codes.BAD_REQUEST and info.status == "FAILED_PRECONDITION":
        return ModelConfigurationError(f"model provider precondition failed: {category}")
    if code == httpx.codes.BAD_REQUEST and thinking_sent and info.thinking_rejected:
        return _ThinkingUnsupported(bare=info.bare)
    if code in (httpx.codes.BAD_REQUEST, httpx.codes.REQUEST_ENTITY_TOO_LARGE):
        return ModelRequestRejected(f"model provider rejected the request: {category}")
    return ChatModelError(f"model provider returned HTTP {code}: {category}")


async def _read_bounded(resp: httpx.Response) -> bytes:
    buffer = bytearray()
    async for piece in resp.aiter_bytes():
        buffer.extend(piece)
        if len(buffer) >= _MAX_ERROR_BYTES:
            break
    return bytes(buffer[:_MAX_ERROR_BYTES])


def _json_error(raw: bytes) -> object:
    try:
        body = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None
    if isinstance(body, list) and body:  # some Google frontends wrap the status in a list
        body = body[0]
    return body.get("error") if isinstance(body, dict) else None


def _retry_after_header(value: str | None) -> float | None:
    """Retry-After is delta-seconds or an HTTP date (RFC 9110 section 10.2.3)."""
    if value is None:
        return None
    try:
        return _bounded_delay(float(value))
    except ValueError:
        pass
    try:
        when = parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=UTC)
    return _bounded_delay((when - datetime.now(UTC)).total_seconds())


def _duration(value: object) -> float | None:
    """google.protobuf.Duration in JSON form: "17s", "0.5s"."""
    if not isinstance(value, str) or not value.endswith("s"):
        return None
    try:
        return _bounded_delay(float(value[:-1]))
    except ValueError:
        return None


def _bounded_delay(seconds: float) -> float | None:
    """A delay callers can sleep on: NaN is no hint at all, and inf or a huge value is clamped
    to MAX_RETRY_AFTER_S (it flows into API Retry-After headers and job backoff)."""
    if math.isnan(seconds):
        return None
    return min(max(0.0, seconds), MAX_RETRY_AFTER_S)
