"""Graph state (DESIGN section 8) and its (de)serialization helpers.

WHY plain dicts for hits/tool calls/usage instead of the domain dataclasses: the state is
checkpointed by LangGraph's msgpack serializer. Arbitrary classes deserialize only with a warning
today and not at all under `LANGGRAPH_STRICT_MSGPACK`, so the checkpoint holds primitives and the
domain objects are rebuilt at the node edges.

WHY tenant/user are NOT in the state: they come from the runtime context (the authenticated
Principal of the current call), which is never checkpointed and never produced by a model.
"""

from __future__ import annotations

from typing import Annotated, Any, TypedDict
from uuid import UUID

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

from ragforge_core.domain.models import ChatTurnUsage, ClaimReasonCode, RankedChunk, ToolCall


class HitDict(TypedDict):
    chunk_id: str
    document_id: str
    collection_id: str
    anchor: str
    title: str
    heading_path: list[str]
    display_text: str
    display_sha256: bytes
    external_url: str | None
    fused_score: float
    rerank_score: float | None


class CallDict(TypedDict):
    id: str
    name: str
    arguments: dict[str, Any]


class UsageDict(TypedDict):
    provider: str
    model: str
    prompt_tokens: int
    completion_tokens: int


class FactDict(TypedDict):
    key: str  # store key in the Principal's facts namespace (sha256 of the fact)
    fact: str  # the user's verbatim quote


class InferredDict(TypedDict):
    label: int  # shown hit label the server attributed the sentence to (ADR-0015)
    sentence: str  # verbatim coverage unit of the draft (coverage.segment)


class AgentState(TypedDict, total=False):
    # committed history: appended ONLY by `finalize` (abandoned turns never reach it)
    messages: Annotated[list[AnyMessage], add_messages]
    # scratch for generate -> tools -> generate exchanges inside one turn
    turn_messages: Annotated[list[AnyMessage], add_messages]
    collection_ids: list[str]  # from the API request, never from the model
    user_message: str  # raw latest user message (the remember_fact quote gate checks this)
    action: str  # explicit action detected in user_message (actions.Action value) or ""
    # the tool call a slash command implies, built by the server (actions.server_call): 0 or 1
    command_calls: list[CallDict]
    question: str  # standalone question (contextualized for follow-ups)
    query: str  # current retrieval query (question, or its one rewrite)
    hits: list[HitDict]  # every hit seen this turn; citation label [S<i>] = hits[i - 1]
    context_labels: list[int]  # labels of the latest retrieval (what `generate` is packed with)
    tool_hit_labels: list[int]  # labels of hits shown to the model inside tool results
    taus: dict[str, float | None]
    grade: str  # ok | weak
    rewrites: int
    tool_rounds: int
    pending_calls: list[CallDict]  # tool calls of the last generate step
    approval_calls: list[CallDict]  # destructive calls awaiting the approval node
    decisions: dict[str, str]  # tool_call_id -> approve|reject (validated resume value)
    draft: str
    streamed: bool  # TurnText already sent for this turn (a later answer must be TurnReplace)
    cited_labels: list[int]  # labels the MODEL cited (valid, shown)
    # server attribution (attribution.py): candidates after verify_citations, survivors after
    # ground_check; finalize emits them as TurnCitation(inferred=True)
    inferred: list[InferredDict]
    normalized: bool  # lenient citation markers were rewritten: the client needs TurnReplace
    verification: str  # Verification value
    unsupported_claims: list[str]
    # Why each unsupported claim is flagged, index-aligned ("not_found" | "inexact_code").
    # A checkpoint written before this field existed has no key: `claim_reasons_from_state`
    # then reads every claim as "not_found" (the prominent flag), so an old turn still resumes.
    claim_reasons: list[str]
    regenerated: bool
    abstained: bool
    memory_facts: list[FactDict]  # what the model was shown, with the keys forget_fact resolves
    summary: str | None
    usage: list[UsageDict]
    # distinct PROMPT_VERSIONs of the builds that rendered a template in this turn (a resume can
    # be served by a newer deploy than the one that started the turn)
    prompt_versions: list[str]


def hit_to_state(hit: RankedChunk) -> HitDict:
    return HitDict(
        chunk_id=str(hit.chunk_id),
        document_id=str(hit.document_id),
        collection_id=str(hit.collection_id),
        anchor=hit.anchor,
        title=hit.title,
        heading_path=list(hit.heading_path),
        display_text=hit.display_text,
        display_sha256=hit.display_sha256,
        external_url=hit.external_url,
        fused_score=hit.fused_score,
        rerank_score=hit.rerank_score,
    )


def hit_from_state(hit: HitDict) -> RankedChunk:
    return RankedChunk(
        chunk_id=UUID(hit["chunk_id"]),
        document_id=UUID(hit["document_id"]),
        collection_id=UUID(hit["collection_id"]),
        anchor=hit["anchor"],
        title=hit["title"],
        heading_path=tuple(hit["heading_path"]),
        display_text=hit["display_text"],
        display_sha256=hit["display_sha256"],
        external_url=hit["external_url"],
        fused_score=hit["fused_score"],
        rerank_score=hit["rerank_score"],
    )


def call_to_state(call: ToolCall) -> CallDict:
    return CallDict(id=call.id, name=call.name, arguments=dict(call.arguments))


def call_from_state(call: CallDict) -> ToolCall:
    return ToolCall(id=call["id"], name=call["name"], arguments=dict(call["arguments"]))


def usage_to_state(usage: ChatTurnUsage) -> UsageDict:
    return UsageDict(
        provider=usage.provider,
        model=usage.model,
        prompt_tokens=usage.prompt_tokens,
        completion_tokens=usage.completion_tokens,
    )


def usage_from_state(usage: UsageDict) -> ChatTurnUsage:
    return ChatTurnUsage(
        provider=usage["provider"],
        model=usage["model"],
        prompt_tokens=usage["prompt_tokens"],
        completion_tokens=usage["completion_tokens"],
    )


def merge_hits(existing: list[HitDict], new: list[RankedChunk]) -> tuple[list[HitDict], list[int]]:
    """Append unseen hits (by chunk id) keeping earlier labels stable; return the merged list and
    the labels of `new` in their ranked order. Stable labels matter: a citation emitted while
    streaming must still point at the same chunk after a later tool-driven retrieval."""
    merged = list(existing)
    index = {h["chunk_id"]: i for i, h in enumerate(merged)}
    labels: list[int] = []
    for hit in new:
        key = str(hit.chunk_id)
        if key not in index:
            index[key] = len(merged)
            merged.append(hit_to_state(hit))
        labels.append(index[key] + 1)
    return merged, labels


def claim_reasons_from_state(state: AgentState) -> list[ClaimReasonCode]:
    """The reason of each of the state's `unsupported_claims`, index-aligned.

    Fail toward the prominent flag: when the checkpoint predates reasons (no key) or the two
    lists are not aligned (a producer that set one and not the other), every claim is
    "not_found", and so is an unknown stored value. Never "inexact_code" by default: that muted
    flag must be earned by the attribution that proved only the code was inexact."""
    claims = state.get("unsupported_claims", [])
    stored = state.get("claim_reasons")
    if stored is None or len(stored) != len(claims):
        return ["not_found"] * len(claims)
    return ["inexact_code" if r == "inexact_code" else "not_found" for r in stored]
