"""Graph nodes (DESIGN section 8). Each node is a small async function of (state, runtime) that
returns a partial state update and emits TurnEvents through LangGraph's custom stream.

Emission rules (the D -> E seam, domain/models.py):
- TurnSources is emitted right after EVERY retrieval (graph or tool driven), before the model can
  produce text that links to those hits. A resumed stream starts with ONE TurnSources replaying
  every hit the checkpoint holds (runner `_drive`), because E's allow-list is per stream.
- TurnText carries RAW model text as it streams; nothing here sanitizes (module E does, once).
- A strict regeneration is NOT streamed: it is emitted once as TurnReplace after it has been
  verified, so the client never shows an unverified replacement.
- A streamed draft whose citation markers were normalized (`[1]` -> `[S1]`) is re-sent once as
  TurnReplace, so the text the client holds is the text unsupported claims are matched against.
- TurnCitation(inferred=False) for labels the model cited, then TurnCitation(inferred=True) for
  sources the SERVER attributed and the ground check did not reject (ADR-0015).
"""

from __future__ import annotations

import asyncio
import re
import time
import unicodedata
from collections.abc import Awaitable, Callable, Mapping
from typing import Any, Final, Protocol
from urllib.parse import quote
from uuid import UUID

from langchain_core.messages import HumanMessage, RemoveMessage, ToolMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.runtime import Runtime
from langgraph.types import interrupt

from ragforge_core.agent import actions, retrieval, tools
from ragforge_core.agent.attribution import (
    INEXACT_CODE,
    NOT_FOUND,
    Attribution,
    Inferred,
    all_not_found,
    annotate,
    attribute,
    inferred_labels,
    no_provenance,
    settle,
    uncited_reasons,
)
from ragforge_core.agent.citations import check_citations, normalize_citations
from ragforge_core.agent.context import AgentContext
from ragforge_core.agent.coverage import dotted_names, refusal_only, valid_labels
from ragforge_core.agent.fencing import fence_hits, fence_tool_result, neutralize
from ragforge_core.agent.grounding import ground_check as run_ground_check
from ragforge_core.agent.llm import (
    ai_message,
    consume_stream,
    flatten_tool_exchanges,
    to_chat_messages,
)
from ragforge_core.agent.state import (
    AgentState,
    FactDict,
    InferredDict,
    call_from_state,
    call_to_state,
    claim_reasons_from_state,
    hit_from_state,
    merge_hits,
    usage_to_state,
)
from ragforge_core.agent.structure import markdown_tokens
from ragforge_core.domain.errors import LeaseLost
from ragforge_core.domain.models import (
    ApprovalChoice,
    ChatMessage,
    ChatTurnUsage,
    ClaimReasonCode,
    JobPriority,
    JobSpec,
    Role,
    ToolCall,
    ToolSpec,
    TurnCitation,
    TurnEvent,
    TurnExtra,
    TurnReplace,
    TurnSources,
    TurnText,
    TurnToolCall,
    TurnVerification,
    Verification,
)
from ragforge_core.observability.logging import get_logger
from ragforge_core.observability.prompt_context import for_prompt
from ragforge_core.prompts import PROMPT_VERSION, get_prompt, render

Rt = Runtime[AgentContext]
Update = dict[str, Any]
Node = Callable[[AgentState, Rt], Awaitable[Update]]

log = get_logger(__name__)


class GuardedNode(Protocol):
    """LangGraph's node-with-runtime shape (runtime is keyword-only)."""

    __name__: str

    def __call__(self, state: AgentState, *, runtime: Rt) -> Awaitable[Update]: ...


REJECTED: Final = "The user rejected this action; it was not performed."
ALREADY_DONE: Final = "This action was already performed."
SUMMARIZE_JOB: Final = "summarize_conversation"
RESUME_KEY: Final = "decisions"  # Command(resume={RESUME_KEY: {tool_call_id: choice}})


# ----------------------------------------------------------------- helpers
def _tag(name: str) -> str:
    """The `name@version` a model call is labelled with (ADR-0018 R2-3): every node that sends
    a template to the model wraps the call in `for_prompt(_tag(<template>))`."""
    return get_prompt(name).tag


def _emit(rt: Rt, event: TurnEvent) -> None:
    rt.stream_writer(event)


def _collections(state: AgentState) -> list[UUID]:
    return [UUID(c) for c in state.get("collection_ids", [])]


def _prompt_versions(state: AgentState) -> list[str]:
    """Record the prompt set of THIS build. A turn resumed by a newer deploy renders the newer
    templates, so TurnDone must report every version that actually produced the turn."""
    versions = state.get("prompt_versions", [])
    return versions if PROMPT_VERSION in versions else [*versions, PROMPT_VERSION]


def _after_model(state: AgentState, *used: ChatTurnUsage) -> Update:
    """State update every model-calling node makes: its usage and the prompt version used."""
    return {
        "usage": [*state.get("usage", []), *(usage_to_state(u) for u in used)],
        "prompt_versions": _prompt_versions(state),
    }


def _memory_block(state: AgentState) -> str:
    """Long-term facts and the rolling summary, fenced as DATA for the user-side message.

    WHY not the system prompt: the summary is LLM-written over answers that were themselves
    derived from untrusted sources, so it must not carry system-level trust; the facts are user
    quotes but still data. Each fact carries its short id so forget_fact is usable."""
    parts: list[str] = []
    facts = state.get("memory_facts", [])
    if facts:
        listed = "\n".join(
            f"- [fact:{tools.fact_ref(f['key'])}] {neutralize(f['fact'])}" for f in facts
        )
        parts.append(
            f"Things the user asked you to remember (their words):\n<facts>\n{listed}\n</facts>"
        )
    summary = state.get("summary")
    if summary:
        parts.append(
            f"Summary of the earlier conversation:\n<summary>\n{neutralize(summary)}\n</summary>"
        )
    return "\n\n".join(parts)


def _packed_question(state: AgentState) -> ChatMessage:
    """The turn's user message as the model sees it: fenced memory and sources, then the
    question. Only the RAW message is committed to history (sources would bloat every turn)."""
    sources = fence_hits(state.get("hits", []), state.get("context_labels", []))
    question = state["user_message"]
    standalone = state.get("question", question)
    if standalone != question:
        question = f"{question}\n(Standalone question: {standalone})"
    memory = _memory_block(state)
    data = f"{memory}\n\n{sources}" if memory else sources
    return ChatMessage(role=Role.user, content=f"{data}\n\nQuestion: {question}")


def _generation_messages(state: AgentState, rt: Rt, *, tools_bound: bool) -> list[ChatMessage]:
    """History, the packed question and this turn's tool exchanges.

    With no tools bound, the exchanges are flattened to plain text: hosted providers reject
    tool_use/tool_result blocks in a request that defines no tools (ChatModel adapters may
    assume structured tool messages only arrive together with tool specs)."""
    history = state.get("messages", [])[-rt.context.settings.history_messages :]
    exchanges = to_chat_messages(state.get("turn_messages", []))
    return [
        *to_chat_messages(history),
        _packed_question(state),
        *(exchanges if tools_bound else flatten_tool_exchanges(exchanges)),
    ]


def _tool_results(state: AgentState) -> list[str]:
    """This turn's fenced <tool_result> blocks, in order: result i is cited as [T<i+1>]."""
    return [
        m.content
        for m in state.get("turn_messages", [])
        if isinstance(m, ToolMessage) and isinstance(m.content, str)
    ]


def _tool_message(name: str, call_id: str, body: str, label: int, sources: str = "") -> ToolMessage:
    return ToolMessage(
        content=fence_tool_result(name, body, label=label, sources=sources), tool_call_id=call_id
    )


def _shown_labels(state: AgentState) -> list[int]:
    """Hit labels the model was actually shown: the retrieval packed into the question (the
    LATEST one: a weak pre-rewrite pass never reaches the prompt) plus tool-driven retrievals."""
    return sorted({*state.get("context_labels", []), *state.get("tool_hit_labels", [])})


def _names(state: AgentState) -> frozenset[str]:
    """Dotted names of the SHOWN `S` sources only (never tool results: those are text the model
    steers). Every reader of one draft gets the same set: a mismatch fails closed but flags
    falsely (DOT handoff; ADR-0015 "Dotted names")."""
    hits = state.get("hits", [])
    return dotted_names(
        hits[n - 1]["display_text"] for n in _shown_labels(state) if 0 < n <= len(hits)
    )


def guarded(node: Node, *, check_lease: bool = True) -> GuardedNode:
    """Node wrapper: heartbeat the run lease BEFORE each node (DESIGN section 8), so a run that
    lost its lease (stalled pod, stolen lease) stops at the next node boundary instead of
    spending model calls whose results the fenced checkpointer would reject anyway.

    `check_lease=False` is for PURE nodes (no I/O, no model call, no event): a heartbeat round
    trip there only adds latency before the first token. Safety is unchanged: the next impure
    node checks, the background keepalive keeps beating, and the fenced checkpointer rejects a
    stale writer's checkpoint regardless."""

    async def run(state: AgentState, *, runtime: Runtime[AgentContext]) -> Update:
        if check_lease:
            await runtime.context.lease.check()
        return await node(state, runtime)

    run.__name__ = node.__name__
    run.__qualname__ = node.__qualname__
    return run


# ----------------------------------------------------------------- memory + question
async def load_memory(state: AgentState, rt: Rt) -> Update:
    ctx = rt.context
    items = await ctx.deps.store.asearch(ctx.facts_namespace, limit=ctx.settings.memory_facts)
    facts = [
        FactDict(key=item.key, fact=str(item.value["fact"]))
        for item in items
        if item.value.get("fact")
    ]
    async with ctx.tenant_tx() as tx:
        summary = await ctx.deps.conversations.summary(tx, ctx.conversation_id)
    return {"memory_facts": facts, "summary": summary}


async def contextualize(state: AgentState, rt: Rt) -> Update:
    """Follow-ups only: turn "and the second one?" into a standalone retrieval question."""
    ctx = rt.context
    history = to_chat_messages(state.get("messages", [])[-ctx.settings.history_messages :])
    ask = ChatMessage(role=Role.user, content=f"Latest message:\n{state['user_message']}")
    with for_prompt(_tag("contextualize")):
        result = await consume_stream(ctx.deps.chat, render("contextualize"), [*history, ask])
    question = result.text.strip()[:1000] or state["user_message"]
    return {"question": question, "query": question, **_after_model(state, *_u(result.usage))}


def _u(usage: ChatTurnUsage | None) -> tuple[ChatTurnUsage, ...]:
    return () if usage is None else (usage,)


# ----------------------------------------------------------------- retrieval + grading
async def retrieve(state: AgentState, rt: Rt) -> Update:
    found = await retrieval.retrieve(rt.context, state["query"], _collections(state))
    hits, labels = merge_hits(state.get("hits", []), found.hits)
    if found.hits:
        _emit(rt, TurnSources(tuple(found.hits)))
    return {"hits": hits, "context_labels": labels, "taus": found.taus}


async def grade(state: AgentState, rt: Rt) -> Update:
    hits = state.get("hits", [])
    latest = [hit_from_state(hits[label - 1]) for label in state.get("context_labels", [])]
    ok = retrieval.passes_grade(latest, state.get("taus", {}), rt.context.settings.default_tau)
    return {"grade": "ok" if ok else "weak"}


async def rewrite(state: AgentState, rt: Rt) -> Update:
    ctx = rt.context
    ask = ChatMessage(role=Role.user, content=f"Question: {state['question']}")
    with for_prompt(_tag("rewrite")):
        result = await consume_stream(ctx.deps.chat, render("rewrite"), [ask])
    query = result.text.strip().splitlines()[0][:500] if result.text.strip() else state["question"]
    return {
        "query": query,
        "rewrites": state.get("rewrites", 0) + 1,
        **_after_model(state, *_u(result.usage)),
    }


async def abstain(state: AgentState, rt: Rt) -> Update:
    """'Not in your notes' plus the closest pages. Reached from a weak grade (nothing generated:
    the text is server-authored, so it is `verified`) or after a regenerated answer failed
    verification again (`unsupported`: the model's answers were rejected)."""
    closest = _closest_pages(state, rt.context.settings.closest_pages)
    failed = state.get("verification") == Verification.unsupported.value
    return {
        "draft": render("abstain", closest=closest),
        "abstained": True,
        "cited_labels": [],
        "inferred": [],
        "unsupported_claims": [],
        "claim_reasons": [],
        "verification": (Verification.unsupported if failed else Verification.verified).value,
        "prompt_versions": _prompt_versions(state),
    }


def _closest_pages(state: AgentState, limit: int) -> str:
    seen: dict[str, str] = {}
    ranked = sorted(
        state.get("hits", []),
        key=lambda h: h["rerank_score"] if h["rerank_score"] is not None else float("-inf"),
        reverse=True,
    )
    for hit in ranked:
        if hit["document_id"] in seen or len(seen) >= limit:
            continue
        title = _md_text(hit["title"])
        url = hit["external_url"]
        seen[hit["document_id"]] = f"- [{title}]({_md_url(url)})" if url else f"- {title}"
    if not seen:
        return ""
    return "\n\nThe closest pages I found:\n" + "\n".join(seen.values())


_MD_SPECIAL: Final = re.compile(r"([\\`*_\[\]()<>!#|~])")
_WHITESPACE: Final = re.compile(r"\s+")


def _md_text(text: str) -> str:
    """One-line Markdown text: whitespace collapsed, every inline-syntax character escaped, so a
    title can neither break out of the link text nor start a new block."""
    return _MD_SPECIAL.sub(r"\\\1", _WHITESPACE.sub(" ", text).strip())


def _md_url(url: str) -> str:
    """Link destination with whitespace, parentheses, brackets and angle brackets
    percent-encoded (they end or break a Markdown destination). Existing %XX escapes are kept;
    E's sanitizer still enforces the scheme allow-list."""
    return quote(url.strip(), safe=":/?#@!$&'*+,;=%-._~")


# ----------------------------------------------------------------- generation + tools
def _answer_system(state: AgentState, bound: tuple[ToolSpec, ...]) -> tuple[str, str]:
    """The (prompt tag, system prompt) of one generate step (ADR-0015):

    - tools bound: the answer prompt plus the rules of exactly the bound tools;
    - an action turn's REPLY step (its tool ran, tools are now unbound): `action_reply`, which
      tells the model to report the <tool_result> and cite [T<n>]. The plain answer prompt
      would tell it to answer only from <source> blocks or say "Your notes don't cover this.",
      which is exactly wrong after "/sync" or an approved removal;
    - otherwise (a question): the answer prompt with no tool text at all.

    The tag names the template that makes the call distinct (its metrics label, ADR-0018):
    `actions` when tools are bound (the model may call one), else `action_reply` or `answer`."""
    if bound:
        rules = render("actions", tools=", ".join(t.name for t in bound))
        return _tag("actions"), render("answer", actions=f"\n\n{rules}")
    if _is_action_reply(state):
        return _tag("action_reply"), render("action_reply")
    return _tag("answer"), render("answer", actions="")


def _is_action_reply(state: AgentState) -> bool:
    """An explicit action whose tool has produced a result this turn: the text that follows
    reports that result, it does not answer from pages."""
    return bool(state.get("action")) and bool(_tool_results(state))


def _callable(calls: list[ToolCall], bound: tuple[ToolSpec, ...]) -> list[ToolCall]:
    """A REGISTERED tool that was not bound on this step is dropped, never executed: with only
    remember_fact bound, a model must not reach remove_from_index (ADR-0015). With no tools
    bound every call is dropped. An unknown name is kept so run_tools answers it with an error
    (it can never execute: dispatch is by the registry), as before ADR-0015."""
    if not bound:
        return []
    names = {t.name for t in bound}
    return [c for c in calls if c.name in names or c.name not in tools.REGISTRY]


def _server_calls(state: AgentState, bound: tuple[ToolSpec, ...]) -> list[ToolCall]:
    """The command's server-built call (actions.server_call), if this step may make it: first
    step of the turn and the call's tool is bound (so `max_tool_rounds=0` still disables it)."""
    names = {t.name for t in bound}
    calls = [call_from_state(c) for c in state.get("command_calls", [])]
    if state.get("tool_rounds", 0) or not calls or any(c.name not in names for c in calls):
        return []
    return calls


async def generate(state: AgentState, rt: Rt) -> Update:
    ctx = rt.context
    rounds = state.get("tool_rounds", 0)
    bound = actions.tools_for_turn(ctx.settings, state.get("action", ""), rounds)
    server = _server_calls(state, bound)
    if server:  # `/sync`, `/remember <words>`...: the call is fixed; no model is asked for it
        for call in server:
            _emit(rt, TurnToolCall(call))
        return {
            "pending_calls": [call_to_state(c) for c in server],
            "turn_messages": [ai_message("", server)],
        }
    draft = state.get("draft", "")
    separator = "\n\n" if draft and not draft.endswith("\n") else ""
    first = True

    def on_text(text: str) -> None:
        nonlocal first
        _emit(rt, TurnText(separator + text if first else text))
        first = False

    prompt, system = _answer_system(state, bound)
    with for_prompt(prompt):
        result = await consume_stream(
            ctx.deps.chat,
            system,
            _generation_messages(state, rt, tools_bound=bool(bound)),
            bound,
            on_text,
        )
    calls = _callable(result.tool_calls, bound)
    update: Update = {
        "draft": draft + (separator + result.text if result.text else ""),
        "streamed": state.get("streamed", False) or bool(result.text),
        **_after_model(state, *_u(result.usage)),
        "pending_calls": [call_to_state(c) for c in calls],
    }
    if calls:
        for call in calls:
            _emit(rt, TurnToolCall(call))
        update["turn_messages"] = [ai_message(result.text, calls)]
    return update


async def run_tools(state: AgentState, rt: Rt) -> Update:
    """Run safe tools now; batch every destructive call of this step for ONE approval node."""
    ctx = rt.context
    hits = state.get("hits", [])
    shown = list(state.get("tool_hit_labels", []))
    base = len(_tool_results(state))
    messages: list[ToolMessage] = []
    approvals = []
    for call in map(call_from_state, state.get("pending_calls", [])):
        label = base + len(messages) + 1
        args = tools.parse_args(call)
        if isinstance(args, str):
            messages.append(_tool_message(call.name, call.id, args, label))
        elif tools.is_destructive(call.name):
            approvals.append(call_to_state(call))
        else:
            outcome = await tools.run_safe(
                ctx, args, collection_ids=_collections(state), user_message=state["user_message"]
            )
            sources = ""
            if outcome.hits:
                hits, labels = merge_hits(hits, outcome.hits)
                shown.extend(n for n in labels if n not in shown)
                _emit(rt, TurnSources(tuple(outcome.hits)))
                sources = fence_hits(hits, labels)
            messages.append(_tool_message(call.name, call.id, outcome.text, label, sources))
    return {
        "hits": hits,
        "tool_hit_labels": shown,
        "turn_messages": messages,
        "approval_calls": approvals,
        "pending_calls": [],
        "tool_rounds": state.get("tool_rounds", 0) + 1,
    }


async def approval(state: AgentState, rt: Rt) -> Update:
    """Persist the batch (idempotent: LangGraph re-runs this node on resume, possibly under a
    newer run id, and ON CONFLICT DO NOTHING keeps the original rows), then interrupt ONCE.

    The resume value is the runner's validated decision map, built from rows it locked while
    holding the lease; anything that is not an explicit approval is a rejection."""
    ctx = rt.context
    batch = state.get("approval_calls", [])
    async with ctx.tenant_tx() as tx:
        await ctx.deps.conversations.record_pending(
            tx, ctx.conversation_id, ctx.run_id, [call_from_state(c) for c in batch]
        )
    raw = interrupt({"tool_calls": batch})
    decided = raw.get(RESUME_KEY) if isinstance(raw, Mapping) else None
    decided = decided if isinstance(decided, Mapping) else {}
    decisions = {
        c["id"]: (
            ApprovalChoice.approve
            if decided.get(c["id"]) == ApprovalChoice.approve.value
            else ApprovalChoice.reject
        ).value
        for c in batch
    }
    return {"decisions": decisions}


async def execute(state: AgentState, rt: Rt) -> Update:
    """Execute approved tools exactly once: the claim (keyed by tool_call_id) and the side effect
    share one transaction, so a re-run after a crash finds the claim and does nothing."""
    ctx = rt.context
    decisions = state.get("decisions", {})
    shown_facts = [f["key"] for f in state.get("memory_facts", [])]
    base = len(_tool_results(state))
    messages: list[ToolMessage] = []
    for call in map(call_from_state, state.get("approval_calls", [])):
        label = base + len(messages) + 1
        args = tools.parse_args(call)
        if decisions.get(call.id) != ApprovalChoice.approve.value or isinstance(args, str):
            messages.append(_tool_message(call.name, call.id, REJECTED, label))
            continue
        async with ctx.tenant_tx() as tx:
            if await ctx.deps.conversations.claim_execution(tx, ctx.conversation_id, call.id):
                text = await tools.run_destructive(ctx, tx, args, _collections(state), shown_facts)
            else:
                text = ALREADY_DONE
        messages.append(_tool_message(call.name, call.id, text, label))
    return {"turn_messages": messages, "approval_calls": [], "decisions": {}}


# ----------------------------------------------------------------- verification
def _valid(state: AgentState) -> frozenset[str]:
    return valid_labels(_shown_labels(state), len(_tool_results(state)))


def _attribute(state: AgentState, draft: str) -> Attribution:
    """Attribution of the draft (pure and cheap, so ground_check recomputes it rather than
    checkpointing it). An action's reply reports a TOOL result ("The page was removed."):
    attributing its sentences to whichever page shares words with them would invent
    provenance. They need a [T<n>] from the model; uncovered ones are residue."""
    attributable = [] if _is_action_reply(state) else _shown_labels(state)
    return attribute(draft, state.get("hits", []), attributable, _valid(state), names=_names(state))


def unverified_flags(found: Attribution) -> tuple[tuple[str, ...], tuple[ClaimReasonCode, ...]]:
    """Flags for an answer rejected before any verifier ran: every one `not_found`. A muted
    `inexact_code` presumes the verifier passed the rest; here none did (final review P2; the
    interview path follows the same rule)."""
    claims, _ = uncited_reasons(found)
    red: ClaimReasonCode = "not_found"
    return claims, (red,) * len(claims)


async def verify_citations(state: AgentState, rt: Rt) -> Update:
    """Deterministic, CPU-only: normalize lenient markers, check every label against what the
    model was shown, then propose inferred citations for would-be residue (ADR-0015).
    - invalid label (rule 1) or nothing covered and nothing attributable (rule 2):
      `unsupported` here, before any verifier call is spent on it;
    - a pure refusal makes no claim: `unverified`, and the ground check is skipped (graph
      `after_citations`): there is nothing for it to check."""
    raw = state.get("draft", "")
    if not raw.strip():
        return {
            "verification": Verification.unsupported.value,
            "unsupported_claims": [],
            "claim_reasons": [],
            "inferred": [],
        }
    draft = normalize_citations(raw)
    tools_ran = bool(_tool_results(state))
    strict = rt.context.settings.strict_citations or state.get("regenerated", False)
    report = check_citations(
        draft,
        _shown_labels(state),
        n_tool_results=len(_tool_results(state)),
        strict=strict,
        names=_names(state),
    )
    found = _attribute(state, draft)
    update: Update = {
        "draft": draft,
        "normalized": state.get("normalized", False) or draft != raw,
        "cited_labels": list(report.cited),
        "inferred": [InferredDict(label=i.label, sentence=i.sentence) for i in found.inferred],
    }
    if not report.ok:
        return {
            **update,
            "unsupported_claims": list(report.offending),
            "claim_reasons": list(all_not_found(report.offending)),
            "inferred": [],
            "verification": Verification.unsupported.value,
        }
    if found.refusal_only:
        # A refusal makes no claim, so it has no provenance to show: the strict prompt's
        # "Your notes don't cover this [S1]." must not surface S1 as a citation (review round 2).
        return {
            **update,
            "cited_labels": [],
            "unsupported_claims": [],
            "claim_reasons": [],
            "inferred": [],
            "verification": Verification.unverified.value,
        }
    if no_provenance(found, tools_ran):
        claims, reasons = unverified_flags(found)
        return {
            **update,
            "unsupported_claims": list(claims),
            "claim_reasons": list(reasons),
            "inferred": [],
            "verification": Verification.unsupported.value,
        }
    return {
        **update,
        "unsupported_claims": [],
        "claim_reasons": [],
        "verification": Verification.pending.value,
    }


async def ground_check(state: AgentState, rt: Rt) -> Update:
    """The verifier sees exactly the evidence the answer could rest on: the packed retrieval
    and this turn's tool results (which embed their own tool-driven sources). Not every hit of
    the turn: a weak pre-rewrite hit the model never saw would only make the verifier lenient,
    and without the tool results it would reject correct answers about sync status or about an
    action that was actually performed.

    The verifier sees the answer with the inferred citations written in as `[S<n>]`, so it checks
    that each inferred source supports ITS unit; its verbatim claims are mapped back onto the
    client's text. The verdict is then settled against the residue (attribution.settle)."""
    sources = fence_hits(state.get("hits", []), state.get("context_labels", []))
    evidence = "\n\n".join([sources, *_tool_results(state)])
    draft = state.get("draft", "")
    attribution = _attribute(state, draft)
    # The verifier checks every inferred PAIRING, not just the claim (attribution.Annotated).
    view = annotate(draft, attribution.inferred)
    started = time.monotonic()
    result = await run_ground_check(rt.context.deps.chat, evidence, view.text, also=draft)
    latency_ms = round((time.monotonic() - started) * 1000)
    claims = [c if c in draft else view.to_answer(c) for c in result.unsupported_claims]
    settled = settle(
        draft,
        verdict=result.status,
        conclusive=result.conclusive,
        # a claim that was nothing but an inserted marker names no sentence of the client's text
        located=result.located and all(claims),
        verifier_claims=tuple(dict.fromkeys(c for c in claims if c)),
        valid=_valid(state),
        attribution=attribution,
        tools_ran=bool(_tool_results(state)),
        names=_names(state),
    )
    # Counts and ids only, never text: why an answer is not `verified` (Phase 2b review P1-3).
    # One line per verifier call, with what an offline eval needs to join it to its inputs:
    # the prompt tag, the model, its usage and latency, the chunks it was shown, the outcome.
    hits = state.get("hits", [])
    log.info(
        "answer_settled",
        prompt=get_prompt("ground_check").tag,
        model=rt.context.deps.chat.model,
        prompt_tokens=sum(u.prompt_tokens for u in result.usage),
        completion_tokens=sum(u.completion_tokens for u in result.usage),
        latency_ms=latency_ms,
        chunk_ids=[hits[n - 1]["chunk_id"] for n in state.get("context_labels", [])],
        outcome="parsed" if result.conclusive else "malformed",
        located=result.located,
        status=settled.status.value,
        verdict=result.status.value,
        verifier_claims=len(claims),
        residue=len(settled.residue),
        code_unbacked=len(settled.code_unbacked),
        inexact_code=settled.reasons.count(INEXACT_CODE),
        inferred_kept=len(settled.inferred),
    )
    return {
        "verification": settled.status.value,
        "unsupported_claims": list(settled.unsupported_claims),
        "claim_reasons": list(settled.reasons),
        "inferred": [InferredDict(label=i.label, sentence=i.sentence) for i in settled.inferred],
        **_after_model(state, *result.usage),
    }


async def decide(state: AgentState, rt: Rt) -> Update:
    return {}


async def generate_strict(state: AgentState, rt: Rt) -> Update:
    """The single strict regeneration: no tools (so tool exchanges are flattened to text), one
    citation per sentence, not streamed."""
    ctx = rt.context
    claims = "\n".join(f"- {neutralize(c)}" for c in state.get("unsupported_claims", []))
    note = "Your previous answer was rejected because some claims are not supported."
    if claims:
        note += f" Rejected claims:\n{claims}"
    messages = [
        *_generation_messages(state, rt, tools_bound=False),
        ChatMessage(role=Role.user, content=f"{note}\nAnswer again following the strict rules."),
    ]
    with for_prompt(_tag("answer_strict")):
        result = await consume_stream(ctx.deps.chat, render("answer_strict"), messages)
    return {"draft": result.text, "regenerated": True, **_after_model(state, *_u(result.usage))}


# ----------------------------------------------------------------- turn end
def _flagged(state: AgentState) -> dict[str, ClaimReasonCode]:
    """The state's unsupported claims with their reasons, in order, one entry per text (a dict
    keeps the first position; for a repeated text "not_found" wins). A checkpoint from before
    reasons existed reads every claim as "not_found" (`claim_reasons_from_state`)."""
    flagged: dict[str, ClaimReasonCode] = {}
    claims = state.get("unsupported_claims", [])
    for claim, reason in zip(claims, claim_reasons_from_state(state), strict=True):
        flagged[claim] = NOT_FOUND if flagged.get(claim) == NOT_FOUND else reason
    return flagged


async def finalize(state: AgentState, rt: Rt) -> Update:
    """Emit the final answer events and commit the turn to `messages` (the only place history is
    written; an abandoned or cancelled turn never gets here)."""
    ctx = rt.context
    draft = state.get("draft", "")
    abstained = state.get("abstained", False)
    status = Verification(state.get("verification", Verification.partial.value))
    if not ctx.settings.verifier_trusted and status in (
        Verification.verified,
        Verification.partial,
    ):
        status = Verification.unverified
    if not state.get("streamed", False):
        _emit(rt, TurnText(draft))
    elif state.get("regenerated", False) or abstained or state.get("normalized", False):
        _emit(rt, TurnReplace(draft))
    hits = state.get("hits", [])
    inferred = [Inferred(i["label"], i["sentence"]) for i in state.get("inferred", [])]
    flagged = _flagged(state) if status is not Verification.verified else {}
    if not ctx.settings.verifier_trusted and not abstained:
        # ADR-0015 rule 7 (review round 4): an inferred citation's only gate beyond lexical
        # overlap is the verifier. A verifier declared below its SLO cannot certify server-made
        # provenance, so none is shown and those sentences are highlighted like rule 4a's; the
        # model's own valid labels are still shown (they are the model's claim, not ours).
        for sentence in (i.sentence for i in inferred):
            flagged[sentence] = NOT_FOUND  # nothing checked it: "not found" wins
        inferred = []
    if not abstained:
        cited = state.get("cited_labels", [])
        for label in cited:
            _emit(rt, TurnCitation(hit_from_state(hits[label - 1]), label=label))
        for label in inferred_labels(inferred, cited):
            _emit(rt, TurnCitation(hit_from_state(hits[label - 1]), inferred=True, label=label))
    _emit(rt, TurnVerification(status, tuple(flagged), tuple(flagged.values())))
    await _maybe_enqueue_summary(state, rt)
    return {
        "messages": [HumanMessage(content=state["user_message"]), ai_message(draft)],
        "turn_messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES)],
        "verification": status.value,
    }


# ADR-0016: the user asked for CODE. Within one clause (no sentence break between the verb and
# the noun: "Show me the page. Also what is a code smell?" is not a request), and the noun must
# be code or snippet, or the request "implement ... in <language>" / "with code".
_CODE_NOUN: Final = (
    r"(?<!status )(?<!error )(?<!exit )(?<!zip )(?<!qr )(?<!morse )(?<!postal )(?<!access )"
    r"(?<!dead )(?<!hamming )(?<!huffman )(?<!correcting )"
    r"(?:\b(?:code|snippet)\b(?!-)(?!\s+(?:review|smells?|table|page|freeze|of\s+conduct"
    r"|elimination|path|tree)\b)|\b(?:codes|snippets)\b(?!\s+of\b))"
)
_LANGUAGES: Final = (
    r"(?:(?i:java|python|kotlin|rust|c\+\+|javascript|typescript|scala|ruby)\b|Go\b(?!-))"
)
_CODE_REQUEST: Final = re.compile(
    rf"\b(?i:show|give|write|share|see)\b[^.?!\n]{{0,40}}{_CODE_NOUN}"
    r"|\b(?i:code|snippet)\s+(?i:example|snippet|sample)\b|\b(?i:example|sample)\s+code\b"
    r"|\b(?i:with\s+code|code\s+it(?:\s+up)?)\b"
    r"|\b(?i:write|provide)\b[^.?!\n]{0,40}\b(?i:implementation)\b(?!\s+details)"
    r"|\A\s*(?i:code)\s+(?i:for)\b"
    # an imperative or "how do I" request, not "how does Kafka implement X in Java clients?"
    r"|(?:\A\s*|\b(?i:how\s+(?:do|would|can|should)\s+(?:i|you|we)|how\s+to|can\s+you|please)\s+)"
    rf"(?i:implement\w*|write|code)\b[^.?!\n]{{0,60}}\b(?i:in|using)\s+{_LANGUAGES}"
    r"|\b(?i:write)\s+(?i:a|an)\s+"
    r"(?!(?i:summary|essay|overview|explanation|note|paragraph|description|report|blog|post"
    r"|haiku|poem|story|article|email|letter)\b)"
    r"[^.?!\n]{0,40}\b(?i:using|with)\s+(?!(?i:simple|plain|examples?)\b)\w+"
    rf"|\b(?i:give|show|write|see)\b[^.?!\n]{{0,30}}{_LANGUAGES}\s*(?i:example|snippet|program)"
    r"|\A\s*(?i:show\s+)?(?i:me\s+)?(?i:the\s+)?(?i:code)\s*\??\s*\Z",
)
# answer@10's notes-backed incident: the bold part name, not "used in production:" mid-sentence.
_PRODUCTION_PART: Final = re.compile(
    r"(?:\A|[\n.!?\]]\s*)[*_]{0,2}\s*in\s+production\s*[*_]{0,2}\s*:", re.IGNORECASE
)
# Anything shaped like an answer label: [S1], [S 1], [S1-S3], [S1; S2], (S1), 【S1】, [Source 1].
_EXTRA_LABEL: Final = re.compile(
    r"\\?[\[(【]\s*(?:S|T|source)\s*\d+(?:\s*(?:[-,;&\u2013]|and)\s*(?:S|T|source)?\s*\d+)*\s*\\?[\])】]",
    re.IGNORECASE,
)
# A code extra's language tag: a short, plain token (the rest of a fence's info string is not code).
_INFO_STRING: Final = re.compile(r"[A-Za-z0-9_+.#-]{0,20}")
# Control characters and bidi controls never reach any client in an extra (a web client does not
# run `make ask`'s `clean`).
_UNSAFE_CHARS: Final = re.compile(
    "[\x00-\x08\x0b-\x1f\x7f-\x9f\u061c\u200e\u200f\u202a-\u202e\u2066-\u2069]"
)
MAX_INCIDENT_CHARS: Final = 700
MAX_CODE_LINES: Final = 40
EXTRAS_TIMEOUT_S: Final = 8.0
_OPENER: Final = re.compile(r"[*_]{0,3}\s*common failure pattern\s*:\s*[*_]{0,3}\s*", re.IGNORECASE)


def _cut(text: str, limit: int) -> str:
    """At most `limit` characters, cut at a sentence end, else at a word, with an ellipsis."""
    if len(text) <= limit:
        return text
    head = text[:limit]
    end = max(head.rfind(". "), head.rfind("! "), head.rfind("? "))
    return head[: end + 1] if end > limit // 2 else head.rsplit(" ", 1)[0] + "\u2026"


def _lines(raw: str) -> str:
    """No control or bidi characters. A form feed, vertical tab or line separator becomes a
    SPACE, never nothing (final review r6: deleting "\f" turned `x = 1\fy = 2` into `x = 1y
    = 2`); "\r\n" becomes "\n"."""
    unified = re.sub(r"\r\n?", "\n", raw)
    unified = re.sub(r"[\f\v\u2028\u2029\x85]", " ", unified)
    return _UNSAFE_CHARS.sub("", unified)


# Labels the reader would take for a citation even when they are not ASCII: matched on an NFKC,
# format-free reading (full-width brackets, superscript digits, a zero-width space inside).
_BARE_NUMBER_LABEL: Final = re.compile(r"\[\s*\d{1,3}\s*\]")


def _without_labels(text: str) -> str:
    """`text` with every citation-like label removed, decided on the folded reading. NFKC and
    dropping format characters only change look-alike characters, so prose keeps its meaning;
    identifiers are ASCII and pass through unchanged."""
    folded = "".join(
        c for c in unicodedata.normalize("NFKC", text) if unicodedata.category(c) != "Cf"
    )
    return _BARE_NUMBER_LABEL.sub("", _EXTRA_LABEL.sub("", folded))


def _incident_line(raw: str) -> str:
    """The incident extra: ONE line (the first paragraph, line breaks collapsed), no control or
    bidi characters, no answer labels, opening with "Common failure pattern:" (emphasis around
    that opener tolerated) and saying something after it; else "" (off-format, NONE, empty).
    Nothing else is rewritten: `max_poll_interval_ms` must survive (final review, P1: stripping
    every `_`/`*` corrupted identifiers). E renders the line as literal text, so emphasis,
    headings or links in it never render."""
    first = re.split(r"\n\s*\n", _lines(raw).strip(), maxsplit=1)[0]
    line = " ".join(_without_labels(first).split())
    opener = _OPENER.match(line)
    rest = line[opener.end() :].strip() if opener else ""
    if not rest or rest.rstrip(".").upper() == "NONE":
        return ""
    shown = _cut(rest, MAX_INCIDENT_CHARS - 25)
    return "" if shown in ("", "\u2026") else "Common failure pattern: " + shown


def _code_parts(raw: str) -> tuple[str, str, str] | None:
    """The code extra as (code, language, prose): the FIRST fenced block as CommonMark parses it
    (markdown-it, the verifier's own parser: indented closers, `~~~`, longer fences are all
    read as a renderer reads them; final review, P2), capped at MAX_CODE_LINES; the prose is the
    first non-empty line after the block, labels stripped, cut. E fences the code and renders
    the prose as literal text. None when there is no fenced block."""
    text = _lines(raw)
    fence = next((t for t in markdown_tokens(text) if t.type == "fence"), None)
    if fence is None or fence.map is None or not fence.content.strip():
        return None  # no fenced block, or an empty one
    code = "\n".join(fence.content.rstrip("\n").splitlines()[:MAX_CODE_LINES])
    info = _INFO_STRING.match(fence.info.strip())
    after = next((ln for ln in text.split("\n")[fence.map[1] :] if ln.strip()), "")
    prose = _cut(" ".join(_without_labels(after).split()), 300)
    return code, info.group(0) if info else "", prose


def extras_kind(state: AgentState) -> str | None:
    """Which labelled extra (ADR-0016) this turn gets, if any. Only question turns whose answer
    cites the notes: never an abstain, never a refusal ("Your notes don't cover this." stays an
    honest "not in your notes": review P1), never an action. Code when the user asked for code
    and the answer holds none from the notes; otherwise an incident when the answer has no
    notes-backed "In production" part."""
    draft = state.get("draft", "")
    if state.get("abstained", False) or state.get("action") or not draft.strip():
        return None
    if _CODE_REQUEST.search(state.get("user_message", "")):
        # "show the code" when the notes have none: answer@10 refuses rather than write code,
        # and the code extra exists exactly for that case (review r2, P1). Retrieval was good
        # enough not to abstain, so the topic is in the notes. Any fenced block (``` or ~~~)
        # in the answer is code from the notes already.
        has_code = any(t.type == "fence" for t in markdown_tokens(draft))
        return None if has_code else "code"
    cited = state.get("cited_labels") or state.get("inferred")
    if refusal_only(draft, names=_names(state)) or not cited:
        return None  # an honest "not in your notes" stays one
    return None if _PRODUCTION_PART.search(draft) else "incident"


async def extras(state: AgentState, rt: Rt) -> Update:
    """ADR-0016: one extra model call, AFTER verification and outside it. Its text is emitted as
    TurnExtra (shown apart, labelled "Not from your notes") and is NOT committed to `messages`,
    so later turns never treat it as the user's notes. Any failure of the call itself is
    logged and never fails the turn (the answer was committed in `finalize`)."""
    ctx = rt.context
    kind = extras_kind(state) if ctx.settings.extras else None
    if kind is None:
        return {}
    question = state.get("question") or state.get("user_message", "")
    # The question and the verified answer go in the USER message, fenced as data: never into
    # the system prompt (the question may be a rewrite over retrieved text). The answer is shown
    # so the extra stays consistent with it and does not repeat it.
    data = (
        f"<question>\n{neutralize(question[:500])}\n</question>\n"
        f"<answer_already_shown>\n{neutralize(state.get('draft', '')[:4000])}\n"
        "</answer_already_shown>\nWrite it now."
    )
    try:
        # A budget: the answer is already shown and `done` (and the conversation lease) wait for
        # this call, so a slow provider must not hold the user's next message (review r6).
        async with asyncio.timeout(EXTRAS_TIMEOUT_S):
            with for_prompt(_tag(f"extras_{kind}")):
                result = await consume_stream(
                    ctx.deps.chat,
                    render(f"extras_{kind}"),
                    [ChatMessage(role=Role.user, content=data)],
                )
    except LeaseLost:
        raise  # the run is no longer ours: stop, as every other node does
    except Exception:  # the verified answer is already committed and shown
        log.warning("extras_failed", kind=kind, exc_info=True)
        return {}
    if kind == "incident":
        line = _incident_line(result.text)
        extra = TurnExtra(kind, line) if line else None
    else:
        parts = _code_parts(result.text)
        extra = TurnExtra(kind, parts[2], code=parts[0], language=parts[1]) if parts else None
    if extra is not None:
        _emit(rt, extra)
    else:
        log.info("extras_skipped", kind=kind, reason="off-format, NONE or empty")
    return _after_model(state, *_u(result.usage))


async def _maybe_enqueue_summary(state: AgentState, rt: Rt) -> None:
    """Summarization is an off-path job (DESIGN section 2); dedupe keeps one queued per thread."""
    ctx = rt.context
    if len(state.get("messages", [])) + 2 <= ctx.settings.summarize_after_messages:
        return
    conversation = str(ctx.conversation_id)
    async with ctx.tenant_tx() as tx:
        await ctx.deps.jobs.enqueue(
            tx,
            JobSpec(
                kind=SUMMARIZE_JOB,
                priority=JobPriority.maintenance,
                payload={"conversation_id": conversation},
                dedupe_key=f"{SUMMARIZE_JOB}:{conversation}",
            ),
        )


# Nodes with no I/O, model call or event emission: `guarded` skips their heartbeat (see there).
PURE_NODES: Final[frozenset[Node]] = frozenset({grade, abstain, verify_citations, decide})
