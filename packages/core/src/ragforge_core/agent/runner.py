"""AgentRunner: the turn-level seam between the agent (D) and the API (E).

Lifecycle of one call (DESIGN section 8, "Run lifecycle"):
1. ONE short tenant transaction takes the conversation run lease (CAS on active_run_id) and then
   locks the pending approvals: a new turn expires them; a resume validates and decides them.
   Both paths lock in the same order (conversation row, then approvals), so they serialize
   instead of deadlocking, and an approved destructive call can never run on a thread that has
   moved on.
2. The graph runs with the lease heartbeated before every node and in the background.
3. The lease is released BEFORE the terminal TurnDone/TurnInterrupt is yielded (so a client that
   resumes or sends a follow-up the instant it sees it never hits TurnInProgress), and as a
   fallback when the run fails or the consumer closes the stream (cancellation: LangGraph
   cancels the running node; the partial draft is discarded because history is committed only
   by `finalize`). Any failure between taking the lease and starting the graph releases it too.

Events are yielded RAW (never sanitized here; module E owns the OutputSanitizer).

Every stream is self-contained for E's link allow-list: E builds a NEW sanitizer per stream, so
a stream that continues a checkpoint (`resume`) first replays TurnSources for every hit the turn
already holds, before any graph event. Text produced after the resume (the continued draft, a
strict regeneration's TurnReplace, the abstain page list, TurnCitations) may link to hits that
were retrieved in an EARLIER stream; without the replay E would defang those legitimate links.
A new turn resets `hits`, so it replays nothing.

Every nested async generator is closed explicitly (`closing`): an abandoned inner generator is
otherwise finalized later by the event loop's GC hook, so a client disconnect would keep the
model streaming and the lease held until then.

Eval trace (ADR-0017 §1): a caller may pass a `TurnTrace`, which the runner fills right before
it yields TurnDone with the per-surface prompt sets and what the model was SHOWN (labels, chunk
ids, digests; never text). A side object, not a new event field, because `domain/**` is frozen
and TurnDone's shape is part of the D -> E seam; the API alone decides who may see `shown`
(eval scope, ?trace=shown). `run_traced` / `resume_traced` hand it to this runner and simply
omit it for any other AgentRunner (test fakes), whose turns then carry no trace.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Final, cast
from uuid import UUID, uuid4

from langchain_core.messages import RemoveMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.types import Command, Interrupt, StreamMode

from ragforge_core.agent import actions
from ragforge_core.agent.context import AgentContext, AgentDeps, RunLease
from ragforge_core.agent.graph import RECURSION_LIMIT, AgentGraph, build_graph
from ragforge_core.agent.llm import closing
from ragforge_core.agent.nodes import RESUME_KEY
from ragforge_core.agent.state import (
    AgentState,
    CallDict,
    HitDict,
    call_from_state,
    call_to_state,
    hit_from_state,
    usage_from_state,
)
from ragforge_core.domain.errors import Conflict, Forbidden, NotFound, ValidationFailed
from ragforge_core.domain.models import (
    ApprovalChoice,
    PendingApproval,
    Principal,
    RankedChunk,
    Scope,
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
    Verification,
)
from ragforge_core.domain.ports import AgentRunner, Tx
from ragforge_core.observability.logging import get_logger
from ragforge_core.prompts import PROMPT_VERSION, surface_versions

log = get_logger(__name__)

# Only these may leave the runner: a node that writes anything else to the custom stream is a
# bug, and dropping it is safer than handing E an object it would not sanitize.
_STREAMABLE: Final = (
    TurnSources,
    TurnText,
    TurnCitation,
    TurnDone,
    TurnToolCall,
    TurnVerification,
    TurnReplace,
    TurnExtra,  # ADR-0016; E renders it literally (sanitize.literal / sanitize.fenced)
)
_STREAM_MODES: Final[tuple[StreamMode, ...]] = ("custom", "updates")
MAX_MESSAGE_CHARS: Final = 8000
CONFIG_TENANT_ID: Final = "rf_tenant_id"  # = adapters.db.checkpoint (D2)
CONFIG_RUN_ID: Final = "rf_run_id"
_DECIDED: Final = {"approved": ApprovalChoice.approve, "rejected": ApprovalChoice.reject}


@dataclass(frozen=True, slots=True)
class ShownHit:
    """One source the model was shown, as `S<label>` (ADR-0017 §1 `shown`)."""

    label: int
    chunk_id: UUID
    document_id: UUID
    display_sha256: bytes


@dataclass(slots=True)
class TurnTrace:
    """Filled when a turn completes (module docstring); empty for a turn that paused at an
    approval or failed. `prompt_versions` is {surface: set} for every build that ran the turn
    (`prompts.surface_versions`); `shown` is in label order. `done` says it was filled."""

    prompt_versions: dict[str, str] = field(default_factory=dict)
    shown: tuple[ShownHit, ...] = ()
    done: bool = False


def shown_hits(values: Mapping[str, Any]) -> tuple[ShownHit, ...]:
    """The sources of a finished turn the model was shown: the retrieval packed into the
    question plus tool-driven retrievals (the rule of `nodes._shown_labels`, which also decides
    which labels a citation may use), resolved against the turn's hits."""
    hits: Sequence[HitDict] = values.get("hits", [])
    labels = sorted({*values.get("context_labels", []), *values.get("tool_hit_labels", [])})
    return tuple(
        ShownHit(
            label=n,
            chunk_id=UUID(hits[n - 1]["chunk_id"]),
            document_id=UUID(hits[n - 1]["document_id"]),
            display_sha256=bytes(hits[n - 1]["display_sha256"]),
        )
        for n in labels
        if 0 < n <= len(hits)
    )


def _require(principal: Principal, scope: Scope) -> None:
    if not principal.has(scope):
        raise Forbidden(f"missing scope {scope.value}")


def _turn_input(message: str, collection_ids: Sequence[UUID]) -> AgentState:
    """The input of a new turn. The raw message is classified ONCE here (ADR-0015): the action,
    if any, is fixed for the whole turn and checkpointed, so a resume binds exactly what the turn
    started with, and nothing a model or a retrieved page says can turn a question into one."""
    intent = actions.detect(message)
    server = actions.server_call(intent, uuid4().hex)  # same id rule as adapter-minted calls
    return _reset(message, collection_ids, intent, server)


def _reset(
    message: str,
    collection_ids: Sequence[UUID],
    intent: actions.Intent,
    server: ToolCall | None = None,
) -> AgentState:
    """A new turn resets EVERY per-turn field. This also abandons an interrupted turn's scratch:
    new input on an interrupted thread restarts at START (the stale approval node is never
    resumed; its rows were expired under the lease in the same step).

    Built as a dict and cast: the REMOVE_ALL marker is a RemoveMessage, which the add_messages
    reducer understands but the AnyMessage annotation does not include."""
    return cast(
        AgentState,
        {
            "turn_messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES)],
            "collection_ids": [str(c) for c in collection_ids],
            "user_message": message,
            "action": intent.action.value if intent.action is not None else "",
            "command_calls": [] if server is None else [call_to_state(server)],
            "question": intent.query,
            "query": intent.query,
            "hits": [],
            "context_labels": [],
            "tool_hit_labels": [],
            "taus": {},
            "grade": "",
            "rewrites": 0,
            "tool_rounds": 0,
            "pending_calls": [],
            "approval_calls": [],
            "decisions": {},
            "draft": "",
            "streamed": False,
            "cited_labels": [],
            "inferred": [],
            "normalized": False,
            "verification": Verification.pending.value,
            "unsupported_claims": [],
            "claim_reasons": [],
            "regenerated": False,
            "abstained": False,
            "usage": [],
            "prompt_versions": [],
        },
    )


def _pending_from(interrupts: Sequence[Interrupt]) -> tuple[ToolCall, ...]:
    calls: list[CallDict] = []
    for item in interrupts:
        if isinstance(item.value, Mapping):
            calls.extend(item.value.get("tool_calls", []))
    return tuple(call_from_state(c) for c in calls)


def _choice(value: object) -> ApprovalChoice:
    try:
        return ApprovalChoice(str(value))
    except ValueError:
        raise ValidationFailed(f"invalid approval decision {value!r}") from None


def _new_choices(pending: set[str], decisions: Mapping[str, object]) -> dict[str, ApprovalChoice]:
    """Anything the caller did not explicitly approve is rejected (fail closed)."""
    return {
        tid: _choice(decisions[tid]) if tid in decisions else ApprovalChoice.reject
        for tid in sorted(pending)
    }


def _recorded_choices(
    rows: Sequence[PendingApproval], decisions: Mapping[str, object]
) -> dict[str, ApprovalChoice]:
    """Rows decided by an earlier resume that never got past the approval/execute step (the
    process died). The recorded decision stands; a caller that now asks for something else is
    told so instead of silently getting the old outcome."""
    recorded = {r.tool_call.id: _DECIDED[r.status] for r in rows}
    for tid, value in decisions.items():
        if _choice(value) is not recorded[tid]:
            raise Conflict("these approvals were already decided differently")
    return recorded


def _validated_choices(
    rows: Sequence[PendingApproval],
    decisions: Mapping[str, object],
    batch: set[str],
    *,
    continuing: bool,
) -> tuple[dict[str, ApprovalChoice], bool]:
    """(choices, still_pending). `batch` is the tool-call ids the checkpoint is waiting on; the
    locked rows must be exactly that batch, so a decision can never be applied to a turn other
    than the one that asked for it. When `continuing` (the graph is already past the interrupt)
    the rows must already be decided: the graph would never read a decision made now."""
    ids = {r.tool_call.id for r in rows}
    if not rows or ids != batch:
        raise Conflict("these approvals are no longer pending")
    unknown = set(decisions) - ids
    if unknown:
        raise ValidationFailed(f"unknown tool_call_id(s): {sorted(unknown)}")
    if all(r.status == "pending" for r in rows) and not continuing:
        return _new_choices(ids, decisions), True
    if all(r.status in _DECIDED for r in rows):
        return _recorded_choices(rows, decisions), False
    raise Conflict("these approvals are no longer pending")


def _scoped(requested: Sequence[UUID], stored: Sequence[UUID]) -> list[UUID]:
    """A follow-up searches the conversation's collections, narrowed by the request if it names
    any; it can never widen or switch them.

    WHY: the stored list is what the conversation was opened over (and what its history and
    memory were built from). Letting each request replace it would let a follow-up silently
    change the scope of `search_notes`/`get_page` mid-thread; a different scope is a new
    conversation. Requested ids outside the stored list are rejected, not dropped, so a client
    bug is visible instead of answering from a narrower scope than the user picked.
    """
    if not requested:
        return list(stored)
    allowed = set(stored)
    outside = [c for c in requested if c not in allowed]
    if outside:
        raise ValidationFailed("collection_ids must be a subset of the conversation's collections")
    return list(dict.fromkeys(requested))


class _Release:
    """Stops the keepalive and releases the lease exactly once, from whichever comes first: the
    terminal event (normal path) or `_drive`'s finally (error/cancel path)."""

    def __init__(self, lease: RunLease, keepalive: asyncio.Task[None]) -> None:
        self._lease = lease
        self._keepalive = keepalive
        self._done = False

    async def __call__(self) -> None:
        """Never raises an Exception (only cancellation of the caller propagates).

        WHY: this runs right before the terminal event. A keepalive that died with an error, or a
        release that hit a DB blip, must not replace a TurnDone/TurnInterrupt whose checkpoint is
        already durable: the client would see an error for a turn that succeeded. A failed
        release is logged; the lease then expires on its TTL, which is the designed fallback.
        """
        if self._done:
            return
        self._done = True
        try:
            self._keepalive.cancel()
            # asyncio.wait never re-raises the task's own exception (unlike `await task`), but
            # still propagates a cancellation of the CURRENT task, which must not be swallowed.
            await asyncio.wait([self._keepalive])
            if not self._keepalive.cancelled() and self._keepalive.exception() is not None:
                log.warning("agent.keepalive_failed", run_id=str(self._lease.run_id))
        finally:
            try:
                await self._lease.release()
            except Exception as exc:  # the lease expires on its TTL; the turn itself succeeded
                log.warning(
                    "agent.lease_release_failed",
                    run_id=str(self._lease.run_id),
                    error=type(exc).__name__,
                )


class LangGraphAgentRunner:
    """Implements domain.ports.AgentRunner on LangGraph 1.x with an injected checkpointer
    (in-memory in tests; D2's fenced Postgres saver in production)."""

    def __init__(self, deps: AgentDeps, checkpointer: BaseCheckpointSaver[str]) -> None:
        self._deps = deps
        self._graph: AgentGraph = build_graph(deps.settings, checkpointer)

    # ------------------------------------------------------------- AgentRunner
    async def run_turn(
        self,
        principal: Principal,
        conversation_id: UUID | None,
        collection_ids: Sequence[UUID],
        message: str,
        *,
        trace: TurnTrace | None = None,
    ) -> AsyncIterator[TurnEvent]:
        _require(principal, Scope.chat)
        if not message.strip() or len(message) > MAX_MESSAGE_CHARS:
            raise ValidationFailed(f"message must be 1..{MAX_MESSAGE_CHARS} characters")
        lease, collections = await self._begin_turn(principal, conversation_id, collection_ids)
        # A new turn resets `hits` (`_turn_input`): nothing earlier may be linked, so no replay.
        payload = _turn_input(message, collections)
        run_events = self._drive(principal, lease, payload, known_hits=(), trace=trace)
        async with closing(run_events) as run:
            async for event in run:
                yield event

    async def resume(
        self,
        principal: Principal,
        run_id: UUID,
        decisions: dict[str, ApprovalChoice],
        *,
        trace: TurnTrace | None = None,
    ) -> AsyncIterator[TurnEvent]:
        # Defense in depth: E checks it too. MCP keys never hold approvals:write, so an agent
        # cannot approve its own destructive actions.
        _require(principal, Scope.approvals_write)
        lease = await self._lease_for_run(principal, run_id)
        try:
            payload, known_hits = await self._resume_payload(principal, lease, run_id, decisions)
        except BaseException:
            await lease.release()  # nothing ran yet: never leave the thread locked until TTL
            raise
        run_events = self._drive(principal, lease, payload, known_hits=known_hits, trace=trace)
        async with closing(run_events) as run:
            async for event in run:
                yield event

    async def _resume_payload(
        self, principal: Principal, lease: RunLease, run_id: UUID, decisions: Mapping[str, object]
    ) -> tuple[Command[Any] | None, tuple[RankedChunk, ...]]:
        """(payload, hits the turn already holds). Read the checkpoint FIRST, then decide the
        rows: approvals are never marked decided for a run the graph cannot actually continue.

        The hits come from the same snapshot: the lease is held, so no other call can change the
        checkpoint before the graph continues from it.

        - Interrupted at the approval node: resume it with the decisions (new, or the ones an
          earlier resume recorded before it died).
        - Already past the interrupt but the approved batch not yet executed (a resumed run died
          in `execute`): continue from the checkpoint (`None`); the execution claim keeps every
          side effect exactly-once.
        """
        snapshot = await self._graph.aget_state(self._config(lease))
        if snapshot.interrupts:
            batch = {c.id for c in _pending_from(snapshot.interrupts)}
        elif "execute" in snapshot.next:
            batch = {c["id"] for c in snapshot.values.get("approval_calls", [])}
        else:
            raise Conflict("this run has no pending approval to resume")
        continuing = not snapshot.interrupts
        choices = await self._decide(principal, run_id, decisions, batch, continuing=continuing)
        hits: list[HitDict] = snapshot.values.get("hits", [])
        known_hits = tuple(hit_from_state(h) for h in hits)
        if continuing:
            return None, known_hits
        # Wrapped on purpose: LangGraph reads a resume dict whose keys all look like xxh3-128
        # hex digests as an {interrupt_id: value} map, and tool_call ids are uuid4 hex (32 hex
        # chars), so a bare decision map would be silently ignored and re-interrupt.
        values = {tid: choice.value for tid, choice in choices.items()}
        return Command(resume={RESUME_KEY: values}), known_hits

    # ------------------------------------------------------------- lease + approvals
    async def _begin_turn(
        self, principal: Principal, conversation_id: UUID | None, collection_ids: Sequence[UUID]
    ) -> tuple[RunLease, list[UUID]]:
        deps = self._deps
        async with deps.tx.tenant(principal.tenant_id) as tx:
            if conversation_id is None:
                if not collection_ids:
                    raise ValidationFailed("collection_ids is required for a new conversation")
                conversation = await deps.conversations.create(
                    tx, principal.user_id, collection_ids
                )
                collections = list(collection_ids)
            else:
                stored = await self._owned(tx, principal, conversation_id)
                conversation, collections = conversation_id, _scoped(collection_ids, stored)
            run_id = await deps.runs.acquire(tx, conversation)
            await deps.runs.expire_pending(tx, conversation)
        return RunLease(deps.runs, principal.tenant_id, conversation, run_id), collections

    async def _lease_for_run(self, principal: Principal, run_id: UUID) -> RunLease:
        """Take the conversation lease for a resume (ownership checked first). Holding the lease
        is what makes the later read-checkpoint-then-decide sequence race-free: a new turn or a
        second resume needs the same lease, so neither can change the approvals in between."""
        deps = self._deps
        async with deps.tx.tenant(principal.tenant_id) as tx:
            conversation = await deps.conversations.conversation_for_run(tx, run_id)
            if conversation is None:
                raise NotFound("no such run")
            await self._owned(tx, principal, conversation)
            new_run = await deps.runs.acquire(tx, conversation)
        return RunLease(deps.runs, principal.tenant_id, conversation, new_run)

    async def _decide(
        self,
        principal: Principal,
        run_id: UUID,
        decisions: Mapping[str, object],
        batch: set[str],
        *,
        continuing: bool,
    ) -> dict[str, ApprovalChoice]:
        """Lock the run's approvals (after the lease: same order as a new turn) and decide them,
        or return the decisions an interrupted earlier resume already recorded."""
        deps = self._deps
        async with deps.tx.tenant(principal.tenant_id) as tx:
            rows = await deps.runs.lock_pending(tx, run_id)
            choices, fresh = _validated_choices(rows, decisions, batch, continuing=continuing)
            if fresh:
                await deps.conversations.decide_pending(tx, run_id, choices)
        return choices

    async def _owned(self, tx: Tx, principal: Principal, conversation_id: UUID) -> list[UUID]:
        """The conversation's collections, if the caller owns it. RLS already hides other
        tenants; the user check stops another member of a team tenant. Absent and not-yours are
        the same NotFound, so ids cannot be probed."""
        row = await self._deps.conversations.get(tx, conversation_id)
        if row is None or row[0] != principal.user_id:
            raise NotFound("no such conversation")
        return list(row[1])

    # ------------------------------------------------------------- graph driving
    def _config(self, lease: RunLease) -> RunnableConfig:
        """Graph config in the shape D2's fenced saver expects (docs/handoff/D2.md, its
        `fenced_config()`): rf_tenant_id/rf_run_id are UUID OBJECTS, because LangGraph copies
        str/int configurable values into checkpoint metadata. The key names are duplicated here
        on purpose: agent may not import adapters (import-linter)."""
        return {
            "configurable": {
                "thread_id": str(lease.conversation_id),
                CONFIG_TENANT_ID: lease.tenant_id,
                CONFIG_RUN_ID: lease.run_id,
            },
            "recursion_limit": RECURSION_LIMIT,
        }

    async def _drive(
        self,
        principal: Principal,
        lease: RunLease,
        payload: AgentState | Command[Any] | None,
        *,
        known_hits: Sequence[RankedChunk],
        trace: TurnTrace | None = None,
    ) -> AsyncIterator[TurnEvent]:
        """`payload` None continues a checkpoint that stopped mid-turn (see `_resume_payload`).

        `known_hits` (required, so no caller can forget it) are the hits the checkpoint already
        holds; they are yielded as ONE TurnSources before any graph event (module docstring)."""
        ctx = AgentContext(principal, lease.conversation_id, lease.run_id, self._deps, lease)
        config = self._config(lease)
        keepalive = asyncio.create_task(lease.keepalive(self._deps.settings.heartbeat_interval_s))
        release = _Release(lease, keepalive)
        interrupts: Sequence[Interrupt] = ()
        try:
            if known_hits:
                yield TurnSources(tuple(known_hits))
            stream = self._graph.astream(
                payload,
                config,
                context=ctx,
                stream_mode=list(
                    _STREAM_MODES
                ),  # must be a list: LangGraph then yields (mode, chunk)
                # sync: each checkpoint is durable (and fenced) before the next node starts, so
                # TurnDone is only sent for a turn whose final state was actually persisted.
                durability="sync",
            )
            async with closing(stream) as chunks:
                async for mode, chunk in chunks:
                    if mode == "custom" and isinstance(chunk, _STREAMABLE):
                        yield chunk
                    elif mode == "updates" and isinstance(chunk, dict):
                        interrupts = chunk.get("__interrupt__", interrupts)
            if interrupts:
                await release()
                yield TurnInterrupt(lease.run_id, _pending_from(interrupts))
                return
            values = (await self._graph.aget_state(config)).values
            prompt_sets: list[str] = values.get("prompt_versions") or [PROMPT_VERSION]
            done = TurnDone(
                conversation_id=lease.conversation_id,
                usage=tuple(usage_from_state(u) for u in values.get("usage", [])),
                prompt_version=" + ".join(prompt_sets),
            )
            versions = surface_versions(*prompt_sets)
            shown = shown_hits(values)
            if trace is not None:
                trace.prompt_versions, trace.shown, trace.done = versions, shown, True
            # Prompt tags and counts only (never text): joins a turn to its per-surface baseline.
            log.info("agent.turn_done", prompt_versions=versions, shown=len(shown))
            await release()
            yield done
        finally:
            await release()


def run_traced(
    agent: AgentRunner,
    principal: Principal,
    conversation_id: UUID | None,
    collection_ids: Sequence[UUID],
    message: str,
    trace: TurnTrace,
) -> AsyncIterator[TurnEvent]:
    """`agent.run_turn`, filling `trace` when `agent` is this runner (module docstring)."""
    if isinstance(agent, LangGraphAgentRunner):
        return agent.run_turn(principal, conversation_id, collection_ids, message, trace=trace)
    return agent.run_turn(principal, conversation_id, collection_ids, message)


def resume_traced(
    agent: AgentRunner,
    principal: Principal,
    run_id: UUID,
    decisions: dict[str, ApprovalChoice],
    trace: TurnTrace,
) -> AsyncIterator[TurnEvent]:
    """`agent.resume`, filling `trace` when `agent` is this runner."""
    if isinstance(agent, LangGraphAgentRunner):
        return agent.resume(principal, run_id, decisions, trace=trace)
    return agent.resume(principal, run_id, decisions)
