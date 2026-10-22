"""POST /v1/query, POST /v1/chat (SSE) and POST /v1/approvals/{run_id} (SSE)."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Annotated, Any, Literal
from uuid import UUID

from fastapi import APIRouter, Query, Request
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncConnection

from ragforge_contracts.api import (
    ApprovalResume,
    ChatRequest,
    QueryHit,
    QueryRequest,
    QueryResponse,
    RetrievalStages,
    StageHit,
)
from ragforge_core.adapters.db.tx import connection
from ragforge_core.agent.retrieval import StageRecorder, Stages, recording, stage_score
from ragforge_core.agent.runner import TurnTrace, resume_traced, run_traced
from ragforge_core.apps.api.chat_stream import citation_from_hit, stream_turn
from ragforge_core.apps.api.deps import (
    Agent,
    ApprovalsKey,
    ChatKey,
    Embedders,
    EvalApiEnabled,
    ReadKey,
    Rerank,
    RerankTop,
    Retrieval,
    Sanitizers,
    TxManager,
    require_eval_seam,
)
from ragforge_core.apps.api.routes.sources import require_collections
from ragforge_core.apps.api.sql import audit, fetch_one
from ragforge_core.domain.errors import NotFound, ValidationFailed
from ragforge_core.domain.models import ApprovalChoice, Principal, RankedChunk
from ragforge_core.domain.ports import EmbedderRegistry
from ragforge_core.rag.retriever import rerank_and_select

router = APIRouter(prefix="/v1")

SSE_DOC: dict[int | str, dict[str, Any]] = {200: {"content": {"text/event-stream": {}}}}
MAX_DECISIONS = 50  # far above one turn's tool calls (max_tool_rounds x calls per round)
MAX_TOOL_CALL_ID = 128  # ids are uuid4 hex (32 chars), minted by the ChatModel adapter


# ----------------------------------------------------------------- query
@router.post("/query", tags=["query"])
async def query(
    body: QueryRequest,
    request: Request,
    key: ReadKey,
    txm: TxManager,
    retriever: Retrieval,
    embedders: Embedders,
    reranker: Rerank,
    rerank_top: RerankTop,
    eval_enabled: EvalApiEnabled,
) -> QueryResponse:
    """Four steps so no transaction is open while a model runs (docs/PHASE1.md):
    short tx (visibility + spaces) -> embed with no tx -> short tx (fused candidates) ->
    rerank + MMR down to `k` with no tx (`Retriever.retrieve` never reranks, handoff B).

    Every returned hit carries a cross-encoder score: the reranker scores at most `rerank_top`
    candidates (RAGFORGE_RERANK_TOP_N, default 20), so `k` above that returns `rerank_top` hits
    rather than padding the answer with unscored, rank-fused tail hits.

    `stages=true` (ADR-0017 §1, eval scope + RAGFORGE_EVAL_API, else 403) runs the SAME four
    steps over the injected retriever's own backend and configuration and the injected reranker,
    recorded (`agent.retrieval.recording`), so `hits` are what the plain call returns and
    `stages` are the lists that produced them. Without the flag the
    code path is exactly the old one."""
    tenant = key.principal.tenant_id
    collection_ids = list(dict.fromkeys(body.collection_ids))
    recorder: StageRecorder | None = None
    if body.stages:
        require_eval_seam(key.principal, eval_enabled)  # before any read or model call
        retriever, recorder = recording(retriever, reranker)
        reranker = recorder.reranker
    async with txm.tenant(tenant) as tx:
        await require_collections(connection(tx), collection_ids)
        spaces = await retriever.spaces_for(tx, collection_ids)
    vectors = await _embed_query(embedders, spaces, body.query)
    async with txm.tenant(tenant) as tx:
        candidates = await retriever.retrieve(tx, body.query, vectors, collection_ids, rerank_top)
    hits = await rerank_and_select(
        body.query, candidates, reranker, rerank_top=rerank_top, final_n=min(body.k, rerank_top)
    )
    return QueryResponse(
        hits=[
            QueryHit(
                citation=citation_from_hit(h), score=h.fused_score, rerank_score=h.rerank_score
            )
            for h in hits
        ],
        trace_id=request.scope.get("state", {}).get("request_id"),
        stages=None if recorder is None else _stages_out(recorder.stages(candidates)),
    )


def _stage_hits(stage: str, hits: Sequence[RankedChunk]) -> list[StageHit]:
    return [
        StageHit(
            chunk_id=h.chunk_id,
            document_id=h.document_id,
            rank=rank,
            score=stage_score(stage, h),
        )
        for rank, h in enumerate(hits, start=1)
    ]


def _stages_out(stages: Stages) -> RetrievalStages:
    return RetrievalStages(
        keyword=_stage_hits("keyword", stages.keyword),
        vector=_stage_hits("vector", stages.vector),
        fused=_stage_hits("fused", stages.fused),
        reranked=_stage_hits("reranked", stages.reranked),
    )


async def _embed_query(
    embedders: EmbedderRegistry, spaces: Sequence[int], text: str
) -> dict[int, list[float]]:
    unique = list(dict.fromkeys(spaces))
    for s in unique:  # no tx open here; a space flipped after startup loads instead of 503s
        await embedders.ensure(s)
    vectors = await asyncio.gather(*(embedders.for_space(s).embed_query(text) for s in unique))
    return dict(zip(unique, vectors, strict=True))


# ----------------------------------------------------------------- chat
@router.post("/chat", tags=["chat"], response_class=StreamingResponse, responses=SSE_DOC)
async def chat(
    body: ChatRequest,
    request: Request,
    key: ChatKey,
    txm: TxManager,
    agent: Agent,
    sanitizers: Sanitizers,
    eval_enabled: EvalApiEnabled,
    trace: Annotated[Literal["shown"] | None, Query()] = None,
) -> StreamingResponse:
    """`?trace=shown` (ADR-0017 §1, eval scope + RAGFORGE_EVAL_API, else 403) adds `shown` to
    the `done` event: the chunk ids and digests the model was shown, never their text."""
    if trace is not None:
        require_eval_seam(key.principal, eval_enabled)  # before a conversation is created
    if not body.message.strip():  # before creating a conversation the runner would reject
        raise ValidationFailed("message must not be blank")
    principal = key.principal
    collection_ids = list(dict.fromkeys(body.collection_ids))
    async with txm.tenant(principal.tenant_id) as tx:
        conn = connection(tx)
        await require_collections(conn, collection_ids)
        if body.conversation_id is not None:
            await _require_conversation(conn, principal, body.conversation_id)
            conversation_id, created = body.conversation_id, False
        else:
            conversation_id, created = (
                await _new_conversation(conn, principal, collection_ids),
                True,
            )
    turn = TurnTrace()
    events = run_traced(agent, principal, conversation_id, collection_ids, body.message, turn)
    return await stream_turn(
        request,
        events,
        sanitizers,
        new_conversation_id=conversation_id if created else None,
        trace=turn,
        trace_shown=trace == "shown",
    )


async def _require_conversation(
    conn: AsyncConnection, principal: Principal, conversation_id: UUID
) -> None:
    """Own conversations only: RLS scopes to the tenant, the user filter scopes to the caller
    (LangGraph tables are outside RLS, so this check is the isolation boundary, DESIGN 8)."""
    row = await fetch_one(
        conn,
        "SELECT 1 FROM rf.conversations WHERE id = :id AND user_id = :user",
        {"id": conversation_id, "user": principal.user_id},
    )
    if row is None:
        raise NotFound("conversation not found")


async def _new_conversation(
    conn: AsyncConnection, principal: Principal, collection_ids: list[UUID]
) -> UUID:
    """The API creates the row up front, so the new id is known even if the turn fails before
    the agent reports it (docs/handoff/E2.md: run_turn always receives a non-null id)."""
    row = await fetch_one(
        conn,
        "INSERT INTO rf.conversations (tenant_id, user_id, collection_ids) "
        "VALUES (:tenant, :user, CAST(:colls AS uuid[])) RETURNING id",
        {"tenant": principal.tenant_id, "user": principal.user_id, "colls": collection_ids},
    )
    assert row is not None
    conversation_id: UUID = row["id"]
    return conversation_id


# ----------------------------------------------------------------- approvals
@router.post(
    "/approvals/{run_id}", tags=["approvals"], response_class=StreamingResponse, responses=SSE_DOC
)
async def resume_approvals(
    run_id: UUID,
    body: ApprovalResume,
    request: Request,
    key: ApprovalsKey,
    txm: TxManager,
    agent: Agent,
    sanitizers: Sanitizers,
) -> StreamingResponse:
    """Needs `approvals:write`, which MCP keys can never hold (an agent cannot approve its own
    destructive tool calls). The agent re-checks pending status under the run lease."""
    if not body.decisions:
        raise ValidationFailed("decisions must not be empty")
    # Bounded before anything is audited: the keys are client text written to audit_log.
    if len(body.decisions) > MAX_DECISIONS or any(
        len(call_id) > MAX_TOOL_CALL_ID for call_id in body.decisions
    ):
        raise ValidationFailed(
            f"at most {MAX_DECISIONS} decisions, tool call ids up to {MAX_TOOL_CALL_ID} chars"
        )
    principal = key.principal
    async with txm.tenant(principal.tenant_id) as tx:
        conn = connection(tx)
        row = await fetch_one(
            conn,
            "SELECT 1 FROM rf.pending_approvals p "
            "JOIN rf.conversations c ON c.id = p.conversation_id "
            "WHERE p.run_id = :run AND c.user_id = :user LIMIT 1",
            {"run": run_id, "user": principal.user_id},
        )
        if row is None:
            raise NotFound("approval run not found")
        await audit(
            conn, key, "approvals.submit", str(run_id),
            {"decisions": {k: v.value for k, v in body.decisions.items()}},
        )  # fmt: skip
    decisions = {call_id: ApprovalChoice(d.value) for call_id, d in body.decisions.items()}
    turn = TurnTrace()
    events = resume_traced(agent, principal, run_id, decisions, turn)
    return await stream_turn(request, events, sanitizers, new_conversation_id=None, trace=turn)
