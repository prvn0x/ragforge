"""TurnEvent (agent, module D) -> ChatEvent (SSE contract) with output sanitization (ADR-0011).

Invariants this module owns (docs/PHASE1.md, D <-> E seam):
  - Model text is NEVER emitted raw. Every text-bearing event (`token`, `answer.replace`,
    verification claims) passes through an OutputSanitizer first.
  - Exactly ONE sanitizer per answer, fed incrementally: markdown escaping is not idempotent, so
    text is sanitized once, by one stateful instance, in order. A replacement answer and each
    rejected claim are separate texts and get their own instance.
  - `allow(urls)` is called on every TurnSources BEFORE any later text is fed, and the allow-list
    is replayed into every later sanitizer instance, so a link to a hit retrieved mid-turn works.
  - Held text is flushed (`finish`) before `interrupt`, `verification` and `done`, so a client
    sees the complete answer before the events that conclude it.
  - When the request created the conversation, its id is set on `interrupt`, `error` and `done`.
    The API's id wins on `done` too: it is the row the API created, whatever the agent reports.
  - Tool-call arguments (`tool_call`, `interrupt.pending`) are model output as well: every string
    leaf is sanitized (one fresh instance per leaf, allow-list replayed), so the ADR-0011
    invariant holds for EVERY emitted event even if a client renders args as markdown. Keys are
    left as is: they are tool-schema parameter names, which the agent validates.
  - `done.prompt_versions` (ADR-0017 §1) is set for EVERY caller: the per-surface prompt sets
    of the turn (the runner's `TurnTrace`). `done.shown` (ids and digests, never text) only when
    the route asked for it (`trace_shown`, granted to the eval scope alone); else null.

Metrics (ADR-0018 R2-1, `TurnMeter`): time to first token and to `done`, streams in flight,
turns by outcome, and for ANSWERED turns the verification mix and flagged-claim reasons, all
labelled with the request's `surface` (the caller's tenant: owner chat, eval or demo), so eval
and demo traffic never moves the owner's signals. Never a tenant id, never text.
"""

from __future__ import annotations

import contextlib
import time
from collections.abc import AsyncGenerator, AsyncIterator, Callable, Sequence
from contextlib import aclosing
from dataclasses import dataclass, field
from typing import Any, Final, assert_never
from uuid import UUID

import structlog
from fastapi import Request
from fastapi.responses import StreamingResponse
from pydantic import ValidationError

from ragforge_contracts.api import Citation, ClaimReason, Problem, ShownSource, UnsupportedClaim
from ragforge_contracts.sse import (
    AnswerReplaceEvent,
    ChatEvent,
    CitationEvent,
    DoneEvent,
    ErrorEvent,
    ExtraEvent,
    InterruptEvent,
    TokenEvent,
    ToolCallEvent,
    Usage,
    VerificationEvent,
    encode_sse,
)
from ragforge_core.agent.coverage import refusal_only
from ragforge_core.agent.runner import TurnTrace
from ragforge_core.agent.sanitize import fenced, literal
from ragforge_core.apps.api.middleware import STARTED_AT
from ragforge_core.domain.errors import RagforgeError
from ragforge_core.domain.models import (
    RankedChunk,
    ToolCall,
    TurnCitation,
    TurnDone,
    TurnEvent,
    TurnExtra,
    TurnInterrupt,
    TurnReplace,
    TurnSources,
    TurnText,
    TurnToolCall,
    TurnVerification,
)
from ragforge_core.domain.ports import OutputSanitizer
from ragforge_core.observability import metrics
from ragforge_core.observability.catalog import CLAIM_REASONS, SURFACES, VERIFICATION_STATUSES
from ragforge_core.observability.logging import get_logger
from ragforge_core.observability.prompt_context import surface as surface_var
from ragforge_core.observability.surface import TenantSurfaces
from ragforge_core.prompts import render, surface_versions

SNIPPET_CHARS = 300
SSE_HEADERS = {"cache-control": "no-store", "x-accel-buffering": "no"}
log = get_logger(__name__)


def citation_from_hit(
    hit: RankedChunk, *, inferred: bool = False, label: int | None = None
) -> Citation:
    """`url` is dropped (None) when the stored external_url is not a valid http(s) URL."""
    fields = {
        "inferred": inferred,
        "label": label,
        "anchor": hit.anchor,
        "document_id": hit.document_id,
        "title": hit.title,
        "heading_path": list(hit.heading_path),
        "snippet": hit.display_text[:SNIPPET_CHARS],
    }
    try:
        return Citation.model_validate({**fields, "url": hit.external_url})
    except ValidationError:
        return Citation.model_validate({**fields, "url": None})


class TurnEventMapper:
    """Stateful mapper for ONE streamed turn (or one resume)."""

    def __init__(
        self,
        new_sanitizer: Callable[[], OutputSanitizer],
        *,
        new_conversation_id: UUID | None,
        instance: str | None = None,
        request_id: str | None = None,
        trace: TurnTrace | None = None,
        trace_shown: bool = False,
    ) -> None:
        self._new_sanitizer = new_sanitizer
        self._trace = trace
        self._trace_shown = trace_shown
        self._new_conversation_id = new_conversation_id
        self._instance = instance
        self._request_id = request_id
        self._allowed: list[str] = []
        self._answer = new_sanitizer()
        self.finished = False  # a turn ends with `done` or (paused) with `interrupt`

    # ------------------------------------------------------------- public
    def map(self, event: TurnEvent) -> list[ChatEvent]:
        match event:
            case TurnSources():
                self._allow(event.hits)
                return []
            case TurnText():
                return self._tokens(self._answer.feed(event.text))
            case TurnCitation():
                return [
                    CitationEvent(
                        citation=citation_from_hit(
                            event.hit, inferred=event.inferred, label=event.label
                        )
                    )
                ]
            case TurnToolCall():
                return [self._tool_call(event.call)]
            case TurnInterrupt():
                self.finished = True
                return [
                    *self._flush(),
                    InterruptEvent(
                        run_id=event.run_id,
                        conversation_id=self._new_conversation_id,
                        pending=[self._tool_call(c) for c in event.pending],
                    ),
                ]
            case TurnVerification():
                return [*self._flush(), self._verification(event)]
            case TurnExtra():
                # General knowledge, not the user's notes (ADR-0016): never formatted. Prose is
                # LITERAL (all Markdown escaped, URLs defanged); code is one safely fenced block.
                prose = literal(event.text)
                text = f"{fenced(event.code, event.language)}\n{prose}" if event.code else prose
                return [*self._flush(), ExtraEvent(kind=event.kind, text=text.strip())]
            case TurnReplace():
                self._answer = self._fresh()  # the replaced answer's held text is discarded
                return [AnswerReplaceEvent(text=self._sanitize_whole(event.text))]
            case TurnDone():
                self.finished = True
                return [*self._flush(), self._done_event(event)]
            case _:
                assert_never(event)

    def _verification(self, event: TurnVerification) -> VerificationEvent:
        """Each claim is sanitized ONCE and the same text goes into both lists, so a client
        matching `claims[i].text` against `unsupported_claims` (or the answer) finds it.

        `claim_reasons` is index-aligned with `unsupported_claims`; a producer that predates
        reasons sends none (or a list of another length), and every claim is then "not_found",
        the prominent flag: a missing reason must never mute a warning."""
        texts = [self._sanitize_whole(c) for c in event.unsupported_claims]
        reasons: Sequence[str] = event.claim_reasons
        if len(reasons) != len(texts):
            reasons = [ClaimReason.not_found] * len(texts)
        return VerificationEvent(
            status=event.status.value,
            unsupported_claims=texts,
            claims=[
                UnsupportedClaim(text=text, reason=_claim_reason(reason))
                for text, reason in zip(texts, reasons, strict=True)
            ],
        )

    def error(self, exc: BaseException) -> ErrorEvent:
        if isinstance(exc, RagforgeError):
            status, title, detail = exc.status, exc.title, str(exc) or None
        else:  # never leak internals into the stream
            status, title, detail = 500, "Internal error", None
        # Class, status and title only (request_id and tenant_id come from the bound context).
        # Not the detail: some adapters carry upstream error text in it, and logs have another
        # audience and retention than the client's own stream. Without this a 200 OK whose stream
        # later failed left no trace of why (Phase 2b eval: four turns, nothing in the api log).
        log.warning("chat_turn_failed", error=type(exc).__name__, status=status, title=title)
        problem = Problem(
            title=title,
            status=status,
            detail=detail,
            instance=self._instance,
            request_id=self._request_id,
        )
        return ErrorEvent(problem=problem, conversation_id=self._new_conversation_id)

    # ------------------------------------------------------------- internals
    def _fresh(self) -> OutputSanitizer:
        sanitizer = self._new_sanitizer()
        if self._allowed:
            sanitizer.allow(tuple(self._allowed))
        return sanitizer

    def _allow(self, hits: Sequence[RankedChunk]) -> None:
        urls = [h.external_url for h in hits if h.external_url]
        self._allowed.extend(urls)
        self._answer.allow(urls)

    def _flush(self) -> list[ChatEvent]:
        out = self._answer.finish()
        self._answer = self._fresh()  # never feed a finished sanitizer
        return self._tokens(out)

    def _sanitize_whole(self, text: str) -> str:
        sanitizer = self._fresh()
        return sanitizer.feed(text) + sanitizer.finish()

    def _tool_call(self, call: ToolCall) -> ToolCallEvent:
        args = {k: self._sanitize_value(v) for k, v in call.arguments.items()}
        return ToolCallEvent(tool_call_id=call.id, tool=call.name, args=args)

    def _sanitize_value(self, value: Any) -> Any:
        if isinstance(value, str):
            return self._sanitize_whole(value)
        if isinstance(value, dict):
            return {k: self._sanitize_value(v) for k, v in value.items()}
        if isinstance(value, list | tuple):
            return [self._sanitize_value(v) for v in value]
        return value  # numbers, booleans, None render no markup

    @staticmethod
    def _tokens(text: str) -> list[ChatEvent]:
        return [TokenEvent(text=text)] if text else []

    def _done_event(self, done: TurnDone) -> DoneEvent:
        usage = [
            Usage(
                provider=u.provider,
                model=u.model,
                prompt_tokens=u.prompt_tokens,
                completion_tokens=u.completion_tokens,
            )
            for u in done.usage
        ]
        conversation_id = self._new_conversation_id or done.conversation_id
        if self._new_conversation_id is not None and done.conversation_id != conversation_id:
            # Seam violation (docs/handoff/E2.md: TurnDone echoes the id it was given). The
            # client must keep the id of the row the API created, so ours wins.
            log.warning("agent_reported_other_conversation_id")
        return DoneEvent(
            conversation_id=conversation_id,
            usage=usage,
            prompt_version=done.prompt_version,
            trace_id=self._request_id,
            prompt_versions=_prompt_versions(done, self._trace),
            shown=_shown(self._trace) if self._trace_shown else None,
        )


def _prompt_versions(done: TurnDone, trace: TurnTrace | None) -> dict[str, str]:
    """The runner's per-surface sets; for a producer that fills no trace (any other
    AgentRunner), derived from `prompt_version`, whose " + "-joined sets carry the same tags."""
    if trace is not None and trace.done:
        return dict(trace.prompt_versions)
    return surface_versions(*done.prompt_version.split(" + "))


def _shown(trace: TurnTrace | None) -> list[ShownSource]:
    """What the model was shown; empty when the producer recorded nothing (never invented)."""
    if trace is None or not trace.done:
        return []
    return [
        ShownSource(
            label=s.label,
            chunk_id=s.chunk_id,
            document_id=s.document_id,
            display_sha256=s.display_sha256.hex(),
        )
        for s in trace.shown
    ]


# ----------------------------------------------------------------- metrics
# The server-authored abstain text (agent `abstain` node), without its closest-pages list.
ABSTAIN_TEXT: Final = render("abstain", closest="")
_SURFACES: Final = TenantSurfaces()  # tenant kind/name never change: one read per tenant


@dataclass(slots=True)
class TurnMeter:
    """Watches ONE streamed turn and records its chat metrics. Fed the raw TurnEvents (the
    outcome is decided on the agent's own text, before sanitization) and the emitted
    ChatEvents (time to first token is the first token the CLIENT received)."""

    surface: str
    provider: str
    started: float
    answer: list[str] = field(default_factory=list)  # the raw answer text so far
    status: str | None = None
    reasons: tuple[str, ...] = ()
    outcome: str | None = None
    first_token_seen: bool = False

    def __post_init__(self) -> None:
        self.surface = metrics.bounded(self.surface, SURFACES)
        self.provider = metrics.bounded(self.provider, metrics.PROVIDERS)

    def opened(self) -> None:
        metrics.CHAT_STREAMS.inc()

    def turn_event(self, event: TurnEvent) -> None:
        match event:
            case TurnText():
                self.answer.append(event.text)
            case TurnReplace():
                self.answer = [event.text]
            case TurnVerification():
                self.status = event.status.value
                claims, reasons = event.unsupported_claims, event.claim_reasons
                # the mapper's rule: reasons that do not line up mean "not_found" for each claim
                aligned = len(reasons) == len(claims)
                self.reasons = tuple(reasons) if aligned else ("not_found",) * len(claims)
            case TurnInterrupt():
                self.outcome = "interrupted"  # paused for an approval: this stream ends here
            case TurnDone():
                self.outcome = self._settled()
                metrics.CHAT_TURN_DURATION.labels(self.surface, self.provider).observe(
                    time.perf_counter() - self.started
                )
            case _:
                pass

    def emitted(self, events: Sequence[ChatEvent]) -> None:
        if self.first_token_seen:
            return
        if any(isinstance(e, TokenEvent) and e.text for e in events):
            self.first_token_seen = True
            metrics.CHAT_TTFT.labels(self.surface, self.provider).observe(
                time.perf_counter() - self.started
            )

    def failed(self) -> None:
        self.outcome = "error"

    def closed(self) -> None:
        """Always called once: a stream that ended without done/interrupt/error was cut off
        (client disconnect, shutdown), which is `interrupted` too."""
        metrics.CHAT_STREAMS.dec()
        outcome = self.outcome or "interrupted"
        metrics.CHAT_TURNS.labels(self.surface, outcome).inc()
        if outcome == "answered" and self.status is not None:
            status = metrics.bounded(self.status, VERIFICATION_STATUSES)
            metrics.VERIFICATIONS.labels(self.surface, status).inc()
            for reason in self.reasons:
                reason_label = metrics.bounded(reason, CLAIM_REASONS)
                metrics.FLAGGED_CLAIMS.labels(self.surface, reason_label).inc()

    def _settled(self) -> str:
        """answered | abstained | refused, from what the agent actually said: the abstain node's
        server text, or an answer that is nothing but an allowlisted refusal (ADR-0015)."""
        text = "".join(self.answer).strip()
        if text.startswith(ABSTAIN_TEXT):
            return "abstained"
        if text and refusal_only(text):
            return "refused"
        return "answered"


async def turn_meter(request: Request) -> TurnMeter:
    """The meter of this request's turn: its surface from the caller's tenant (bound to the
    request's log context by `deps.authenticate`), its provider from the wired chat model, its
    clock from when the outermost middleware saw the request."""
    container = request.app.state.container
    chat = getattr(container, "chat", None)
    tenant = structlog.contextvars.get_contextvars().get("tenant_id")
    found = surface_var.get()
    if isinstance(tenant, str):
        with contextlib.suppress(ValueError):  # not a UUID: keep the route's surface
            found = await _SURFACES.resolve(container.tx, UUID(tenant))
    surface_var.set(found)
    started = request.scope.get("state", {}).get(STARTED_AT)
    return TurnMeter(
        surface=found,
        provider=str(getattr(chat, "provider", metrics.OTHER)),
        started=started if isinstance(started, float) else time.perf_counter(),
    )


async def stream_turn(
    request: Request,
    events: AsyncIterator[TurnEvent],
    new_sanitizer: Callable[[], OutputSanitizer],
    *,
    new_conversation_id: UUID | None,
    trace: TurnTrace | None = None,
    trace_shown: bool = False,
) -> StreamingResponse:
    """Stream a turn as SSE.

    For an EXISTING conversation (or a resume) the first event is pulled before the response
    starts, so errors raised up front (TurnInProgress -> 409, NotFound -> 404) become proper
    problem+json responses. For a NEW conversation everything goes in-stream instead, because the
    client must learn the new conversation id even when the turn fails immediately.
    """
    mapper = TurnEventMapper(
        new_sanitizer,
        new_conversation_id=new_conversation_id,
        instance=request.url.path,
        request_id=request.scope.get("state", {}).get("request_id"),
        trace=trace,
        trace_shown=trace_shown,
    )
    meter = await turn_meter(request)
    first: list[TurnEvent] = []
    if new_conversation_id is None:
        try:
            first.append(await anext(events))
        except StopAsyncIteration:
            pass
        except BaseException:
            await _close(events)
            raise
    return StreamingResponse(
        _sse_body(mapper, first, events, meter),
        media_type="text/event-stream",
        headers=SSE_HEADERS,
    )


async def _sse_body(
    mapper: TurnEventMapper,
    first: list[TurnEvent],
    events: AsyncIterator[TurnEvent],
    meter: TurnMeter,
) -> AsyncIterator[bytes]:
    meter.opened()
    try:
        async with aclosing(_with_close(events)) as rest:
            try:
                for event in first:
                    for out in _mapped(mapper, meter, event):
                        yield encode_sse(out)
                async for event in rest:
                    for out in _mapped(mapper, meter, event):
                        yield encode_sse(out)
                if not mapper.finished:
                    raise RuntimeError("agent turn ended without a done or interrupt event")
            except Exception as exc:
                if not isinstance(exc, RagforgeError):
                    log.exception("chat_stream_failed")
                meter.failed()
                yield encode_sse(mapper.error(exc))
    finally:
        meter.closed()


def _mapped(mapper: TurnEventMapper, meter: TurnMeter, event: TurnEvent) -> list[ChatEvent]:
    meter.turn_event(event)
    out = mapper.map(event)
    meter.emitted(out)
    return out


async def _with_close(events: AsyncIterator[TurnEvent]) -> AsyncGenerator[TurnEvent, None]:
    """Re-yield `events` and always close the agent's iterator (client disconnect included), so
    the agent can stop promptly and release its run lease."""
    try:
        async for event in events:
            yield event
    finally:
        await _close(events)


async def _close(events: AsyncIterator[TurnEvent]) -> None:
    aclose = getattr(events, "aclose", None)
    if aclose is not None:
        await aclose()


def _claim_reason(reason: str) -> ClaimReason:
    """The contract reason for a domain reason; anything unknown is "not_found" (never mute a
    warning because a value was not recognised, and never fail the stream over it)."""
    try:
        return ClaimReason(reason)
    except ValueError:
        return ClaimReason.not_found
