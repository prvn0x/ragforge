"""Retrieval as the agent calls it: embed OUTSIDE any transaction, then a short tenant tx.

PHASE1 convention: never call a model (embedder or reranker) while a transaction is open, so
this is four steps: (1) short tx for the active spaces and the collections' tau, (2) embed the
query once per space with no tx, (3) short tx for the fused candidates, (4) rerank + MMR down to
`retrieval_k` with no tx (handoff B: `Retriever.retrieve` never reranks). Without step 4 no hit
carries a `rerank_score` and the grade node abstains on every turn.

Per-stage retrieval (ADR-0017 §1, /v1/query `stages=true`): `StageRecorder` wraps the SAME
SearchBackend and Reranker the pipeline uses and records what each returned, so the stages are
the lists the real pipeline produced, not a re-implementation that could drift from it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from uuid import UUID

from ragforge_core.agent.context import AgentContext
from ragforge_core.domain.models import RankedChunk
from ragforge_core.domain.ports import Reranker, Retriever, Tx
from ragforge_core.observability import metrics
from ragforge_core.rag.retriever import HybridRetriever, SearchBackend, rerank_and_select


@dataclass(frozen=True, slots=True)
class Retrieval:
    hits: list[RankedChunk]
    taus: dict[str, float | None]


async def retrieve(ctx: AgentContext, query: str, collection_ids: Sequence[UUID]) -> Retrieval:
    """Each stage is timed (`ragforge_retrieval_duration_seconds{stage}`): embed = the query
    embeddings, search = the candidates transaction, rerank = rerank + MMR (ADR-0018)."""
    deps = ctx.deps
    async with ctx.tenant_tx() as tx:
        served = await _served_collections(deps.retriever, tx, collection_ids)
        spaces = (
            sorted(served)
            if served is not None
            else list(await deps.retriever.spaces_for(tx, collection_ids))
        )
        taus = await deps.knowledge.collection_taus(tx, collection_ids)
    vectors: dict[int, list[float]] = {}
    with metrics.retrieval_stage("embed"):
        for s in spaces:
            await deps.embedders.ensure(s)  # a space flipped after startup loads here, not 503s
            vectors[s] = await deps.embedders.for_space(s).embed_query(query)
    settings = deps.settings
    with metrics.retrieval_stage("search"):
        async with ctx.tenant_tx() as tx:
            candidates = await deps.retriever.retrieve(
                tx, query, vectors, collection_ids, settings.rerank_top
            )
    with metrics.retrieval_stage("rerank"):
        hits = await rerank_and_select(
            query,
            candidates,
            deps.reranker,
            rerank_top=settings.rerank_top,
            final_n=settings.retrieval_k,
        )
    if served is not None:
        _count_orphans(hits, served)
    return Retrieval(hits=hits, taus={str(k): v for k, v in taus.items()})


async def _served_collections(
    retriever: Retriever, tx: Tx, collection_ids: Sequence[UUID]
) -> Mapping[int, Sequence[UUID]] | None:
    """ACTIVE space -> the collections it serves, read once in place of `spaces_for` (which is
    exactly `sorted` of these keys), so the orphan check costs no extra query. None for a
    retriever that is not the wired HybridRetriever (test doubles): no orphan accounting."""
    if not isinstance(retriever, HybridRetriever):
        return None
    if not collection_ids:
        return {}
    return await retriever.backend.active_spaces(tx, list(dict.fromkeys(collection_ids)))


def _count_orphans(hits: Sequence[RankedChunk], served: Mapping[int, Sequence[UUID]]) -> None:
    """`ragforge_retrieval_orphan_hits_total` (DESIGN §7.3, "should be zero"): a returned hit of
    a collection that NO active embedding space serves. Only keyword search can find such a
    chunk, so its vectors are unsearchable (an unflipped/retired space, or a flip racing this
    turn): a repair signal, never a normal state."""
    covered = {c for collections in served.values() for c in collections}
    orphans = sum(1 for h in hits if h.collection_id not in covered)
    if orphans:
        metrics.RETRIEVAL_ORPHANS.inc(orphans)


def top_score(hits: Sequence[RankedChunk]) -> RankedChunk | None:
    """The best reranked hit. Hits without a rerank score cannot pass the grade (ADR-0006 grades
    on the cross-encoder score only; an RRF score is rank-based and not comparable to tau)."""
    scored = [h for h in hits if h.rerank_score is not None]
    return max(scored, key=lambda h: h.rerank_score or 0.0, default=None)


def passes_grade(
    hits: Sequence[RankedChunk], taus: dict[str, float | None], default: float
) -> bool:
    """Top rerank score >= tau of the collection that hit belongs to (default when untuned)."""
    best = top_score(hits)
    if best is None or best.rerank_score is None:
        return False
    tau = taus.get(str(best.collection_id))
    return best.rerank_score >= (default if tau is None else tau)


# ----------------------------------------------------------------- per-stage trace (ADR-0017)
def stage_score(stage: str, hit: RankedChunk) -> float:
    """A stage's own score: ts_rank (keyword), -distance (vector), RRF (fused), the
    cross-encoder's (reranked). Incomparable ACROSS stages: only ranks compare, which is all
    E-R reads."""
    if stage == "reranked" and hit.rerank_score is not None:
        return hit.rerank_score
    return hit.fused_score


@dataclass(frozen=True, slots=True)
class Stages:
    """Each stage's candidates, best first (see `StageRecorder.stages` for the order rules)."""

    keyword: tuple[RankedChunk, ...]
    vector: tuple[RankedChunk, ...]
    fused: tuple[RankedChunk, ...]
    reranked: tuple[RankedChunk, ...]


@dataclass(slots=True)
class StageRecorder:
    """A SearchBackend and a Reranker that delegate to the real ones and keep what they returned.

    Use one per query: build a HybridRetriever over `recorder` (as SearchBackend) and pass
    `recorder.reranker` to `rerank_and_select`. Recording copies the returned lists; nothing is
    reordered on the way through, so the pipeline's own result is unchanged."""

    backend: SearchBackend
    inner_reranker: Reranker
    vector_lists: list[tuple[int, tuple[RankedChunk, ...]]] = field(default_factory=list)
    keyword_list: tuple[RankedChunk, ...] = ()
    reranked_list: tuple[RankedChunk, ...] = ()

    async def active_spaces(
        self, tx: Tx, collection_ids: Sequence[UUID]
    ) -> Mapping[int, Sequence[UUID]]:
        return await self.backend.active_spaces(tx, collection_ids)

    async def vector_search(
        self,
        tx: Tx,
        space_id: int,
        query_vector: Sequence[float],
        collection_ids: Sequence[UUID],
        k: int,
    ) -> Sequence[RankedChunk]:
        hits = tuple(
            await self.backend.vector_search(tx, space_id, query_vector, collection_ids, k)
        )
        self.vector_lists.append((space_id, hits))
        return hits

    async def keyword_search(
        self, tx: Tx, query: str, collection_ids: Sequence[UUID], k: int
    ) -> Sequence[RankedChunk]:
        hits = tuple(await self.backend.keyword_search(tx, query, collection_ids, k))
        self.keyword_list = hits
        return hits

    @property
    def reranker(self) -> Reranker:
        return _RecordingReranker(self)

    def stages(self, fused: Sequence[RankedChunk]) -> Stages:
        """The four stages, each deterministic: equal scores are ordered by chunk_id.

        - keyword: the backend's list, re-sorted by (-score, chunk_id) (the SQL already orders
          that way; re-sorting makes the tie rule this function's, not a backend's promise);
        - vector: one list per active space, interleaved by rank (rank r of every space, spaces
          ascending, before rank r+1), first occurrence of a chunk kept. WHY by rank: distances
          of different embedding spaces are incomparable; ranks are what RRF fuses too;
        - fused: the retriever's own output (RRF, collapsed by display digest, cut to the
          reranker's pool), whose ties already end on chunk_id (`rag.fusion.rrf_fuse`);
        - reranked: the candidates the cross-encoder SCORED, by (-score, chunk_id). Candidates
          it left unscored (reranker unavailable, beyond its top_n) are absent: their RRF score
          is not on the cross-encoder's scale."""
        scored = [h for h in self.reranked_list if h.rerank_score is not None]
        return Stages(
            keyword=_by_score("keyword", self.keyword_list),
            vector=_interleave(self.vector_lists),
            fused=tuple(fused),
            reranked=_by_score("reranked", scored),
        )


class StagesUnsupported(TypeError):
    """The wired retriever is not a HybridRetriever, so there is no backend to record. A server
    wiring error (bootstrap always builds one), raised instead of recording a DIFFERENT pipeline
    than plain /v1/query runs."""


def recording(retriever: Retriever, reranker: Reranker) -> tuple[HybridRetriever, StageRecorder]:
    """The wired retriever, recording: a HybridRetriever over the SAME backend with the SAME
    per-list k and RRF k, so `stages=true` hits equal the plain call's by construction (the A/A
    precondition, ADR-0017 §1), whatever bootstrap configured (review round 1, P2: a rebuilt
    default retriever would silently diverge once a k became a setting).

    Reads HybridRetriever's private configuration: rag/retriever.py is not this module's to
    change; docs/handoff/EV-API.md asks for public accessors."""
    if not isinstance(retriever, HybridRetriever):
        raise StagesUnsupported(f"stages need a HybridRetriever, not {type(retriever).__name__}")
    recorder = StageRecorder(retriever.backend, reranker)
    wrapped = HybridRetriever(recorder, per_list_k=retriever.per_list_k, rrf_k=retriever.rrf_k)
    return wrapped, recorder


@dataclass(frozen=True, slots=True)
class _RecordingReranker:
    recorder: StageRecorder

    async def rerank(self, query: str, candidates: Sequence[RankedChunk]) -> list[RankedChunk]:
        out = await self.recorder.inner_reranker.rerank(query, candidates)
        self.recorder.reranked_list = tuple(out)
        return out


def _by_score(stage: str, hits: Sequence[RankedChunk]) -> tuple[RankedChunk, ...]:
    # str(uuid) orders like Postgres orders uuid (bytes), so this matches the SQL tie-break
    return tuple(sorted(hits, key=lambda h: (-stage_score(stage, h), str(h.chunk_id))))


def _interleave(lists: Sequence[tuple[int, Sequence[RankedChunk]]]) -> tuple[RankedChunk, ...]:
    ordered = [_by_score("vector", hits) for _, hits in sorted(lists, key=lambda item: item[0])]
    seen: set[UUID] = set()
    out: list[RankedChunk] = []
    for rank in range(max((len(hits) for hits in ordered), default=0)):
        for hits in ordered:
            if rank < len(hits) and hits[rank].chunk_id not in seen:
                seen.add(hits[rank].chunk_id)
                out.append(hits[rank])
    return tuple(out)
