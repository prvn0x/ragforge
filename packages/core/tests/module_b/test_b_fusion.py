"""Pure retrieval functions: RRF (k=60), collapse by display_sha256, MMR (lambda 0.7), packing,
and the HybridRetriever orchestration over a fake SearchBackend."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from uuid import UUID, uuid4

import pytest

from ragforge_core.domain.models import RankedChunk
from ragforge_core.domain.ports import Retriever, Tx
from ragforge_core.rag.fusion import collapse_by_display_sha, rrf_fuse
from ragforge_core.rag.mmr import jaccard_similarity, mmr_select
from ragforge_core.rag.packing import pack
from ragforge_core.rag.retriever import HybridRetriever, SearchBackend, rerank_and_select
from ragforge_core.rag.tokens import count_tokens


def hit(
    name: str, text: str | None = None, *, score: float = 0.0, rerank: float | None = None
) -> RankedChunk:
    body = text if text is not None else f"text of {name}"
    return RankedChunk(
        chunk_id=UUID(hashlib.md5(name.encode()).hexdigest()),  # noqa: S324 - stable test ids
        document_id=uuid4(),
        collection_id=uuid4(),
        anchor=name,
        title=f"T-{name}",
        heading_path=("H",),
        display_text=body,
        display_sha256=hashlib.sha256(body.encode()).digest(),
        external_url=None,
        fused_score=score,
        rerank_score=rerank,
    )


A, B, C, D = hit("a"), hit("b"), hit("c"), hit("d")


# ---------------------------------------------------------------------- RRF
def test_rrf_scores_are_sum_of_reciprocal_ranks_with_k60() -> None:
    fused = rrf_fuse([[A, B, C], [B, D]])
    scores = {h.anchor: h.fused_score for h in fused}
    assert scores["b"] == pytest.approx(1 / 62 + 1 / 61)
    assert scores["a"] == pytest.approx(1 / 61)
    assert scores["c"] == pytest.approx(1 / 63)
    assert scores["d"] == pytest.approx(1 / 62)
    assert [h.anchor for h in fused] == ["b", "a", "d", "c"]


def test_rrf_ignores_raw_scores_so_incomparable_spaces_fuse_safely() -> None:
    huge = [hit("a", score=1e9), hit("b", score=-1e9)]
    tiny = [hit("b", score=1e-9), hit("a", score=1e-12)]
    fused = rrf_fuse([huge, tiny])
    assert fused[0].fused_score == pytest.approx(fused[1].fused_score)


def test_rrf_counts_a_duplicate_within_one_list_once() -> None:
    fused = rrf_fuse([[A, A, A]])
    assert fused[0].fused_score == pytest.approx(1 / 61)


def test_rrf_tie_break_is_deterministic() -> None:
    first = rrf_fuse([[A, B], [B, A]])
    second = rrf_fuse([[B, A], [A, B]])
    assert [h.chunk_id for h in first] == [h.chunk_id for h in second]


def test_rrf_empty_and_invalid_k() -> None:
    assert rrf_fuse([]) == []
    assert rrf_fuse([[], []]) == []
    with pytest.raises(ValueError, match="positive"):
        rrf_fuse([[A]], k=0)


# ---------------------------------------------------------------------- collapse
def test_collapse_keeps_best_ranked_copy_per_display_sha() -> None:
    dup1, dup2 = hit("x1", "same paragraph"), hit("x2", "same paragraph")
    out = collapse_by_display_sha([dup1, A, dup2, B])
    assert [h.anchor for h in out] == ["x1", "a", "b"]


# ---------------------------------------------------------------------- MMR
def test_mmr_lambda_one_is_pure_relevance_order() -> None:
    cands = [hit("a", score=3), hit("b", score=2), hit("c", score=1)]
    assert [h.anchor for h in mmr_select(cands, 3, lambda_=1.0)] == ["a", "b", "c"]


def test_mmr_demotes_near_duplicates_at_default_lambda() -> None:
    cands = [
        hit("a", "the cat sat on the mat today", rerank=0.95),
        hit("a2", "the cat sat on the mat today indeed", rerank=0.90),
        hit("b", "quarterly revenue grew in europe", rerank=0.80),
    ]
    picked = [h.anchor for h in mmr_select(cands, 2)]
    assert picked == ["a", "b"]


def test_mmr_prefers_rerank_score_over_fused_score() -> None:
    cands = [
        hit("a", "alpha words", score=9, rerank=0.1),
        hit("b", "beta words", score=1, rerank=0.9),
    ]
    assert mmr_select(cands, 1)[0].anchor == "b"


def test_mmr_edge_cases() -> None:
    assert mmr_select([], 3) == []
    assert mmr_select([A], 0) == []
    assert len(mmr_select([A, B], 5)) == 2
    with pytest.raises(ValueError, match="lambda_"):
        mmr_select([A], 1, lambda_=1.5)
    assert jaccard_similarity(hit("e", ""), A) == 0.0


# ---------------------------------------------------------------------- packing
def test_pack_numbers_sources_and_respects_budget() -> None:
    small = [hit(f"s{i}", "word " * 20) for i in range(10)]
    packed = pack(small, budget_tokens=100)
    assert packed.tokens <= 100
    assert packed.text.startswith("[1] T-s0 › H\n")
    assert len(packed.hits) == packed.text.count("\n[") + 1
    assert count_tokens(packed.text) <= 100


def test_pack_skips_an_oversized_hit_but_keeps_later_ones() -> None:
    big, small = hit("big", "word " * 500), hit("small", "short text")
    packed = pack([big, small], budget_tokens=50)
    assert [h.anchor for h in packed.hits] == ["small"]
    assert packed.text.startswith("[1] ")


# ---------------------------------------------------------------------- retriever
class FakeBackend:
    def __init__(self, spaces: Sequence[int], vector_hits: dict[int, list[RankedChunk]],
                 keyword_hits: list[RankedChunk]) -> None:  # fmt: skip
        self.spaces = list(spaces)
        self.by_space: dict[int, list[UUID]] | None = None
        self.vector_hits = vector_hits
        self.keyword_hits = keyword_hits
        self.calls: list[tuple[str, object]] = []

    async def active_spaces(
        self, tx: Tx, collection_ids: Sequence[UUID]
    ) -> Mapping[int, Sequence[UUID]]:
        self.calls.append(("spaces", list(collection_ids)))
        if self.by_space is not None:
            return self.by_space
        return {space: list(collection_ids) for space in self.spaces}

    async def vector_search(
        self,
        tx: Tx,
        space_id: int,
        query_vector: Sequence[float],
        collection_ids: Sequence[UUID],
        k: int,
    ) -> Sequence[RankedChunk]:
        self.calls.append(("vector", (space_id, list(query_vector), k, list(collection_ids))))
        return self.vector_hits.get(space_id, [])

    async def keyword_search(self, tx: Tx, query: str, collection_ids: Sequence[UUID],
                             k: int) -> Sequence[RankedChunk]:  # fmt: skip
        self.calls.append(("keyword", (query, k)))
        return self.keyword_hits


def test_retriever_satisfies_ports() -> None:
    backend: SearchBackend = FakeBackend([], {}, [])
    assert isinstance(HybridRetriever(backend), Retriever)


async def test_retrieve_fuses_every_space_and_keywords_then_collapses() -> None:
    dup = hit("dup", "text of a")  # same display text as A -> collapsed
    backend = FakeBackend([2, 1], {1: [A, B], 2: [dup, C]}, [B, D])
    retriever = HybridRetriever(backend)
    out = await retriever.retrieve(object(), "q", {1: [0.1], 2: [0.2]}, [uuid4()], k=10)
    assert next(h.anchor for h in out) == "b"  # in two lists
    anchors = {h.anchor for h in out}
    assert {"b", "c", "d"} <= anchors
    assert len(anchors & {"a", "dup"}) == 1
    assert len({h.display_sha256 for h in out}) == len(out)
    vector_calls = [c for c in backend.calls if c[0] == "vector"]
    assert [c[1][0] for c in vector_calls] == [1, 2]  # type: ignore[index]
    assert all(c[1][2] == 40 for c in vector_calls)  # type: ignore[index]


async def test_retrieve_survives_a_space_flip_between_spaces_for_and_retrieve() -> None:
    """Review finding: a blue/green flip landing between spaces_for (tx 1) and retrieve (tx 2)
    raised mid-turn. Now: search active spaces that have vectors, skip (and log) the rest,
    ignore vectors for spaces no longer active, and still run keyword search."""
    backend = FakeBackend([1, 2], {1: [A], 2: [B], 3: [C]}, [D])
    retriever = HybridRetriever(backend)
    out = await retriever.retrieve(object(), "q", {1: [0.1], 3: [0.3]}, [uuid4()], k=5)
    assert {h.anchor for h in out} == {"a", "d"}
    searched = [c[1][0] for c in backend.calls if c[0] == "vector"]  # type: ignore[index]
    assert searched == [1]


async def test_each_space_is_searched_only_over_the_collections_it_serves() -> None:
    """Review finding: every space was searched over every requested collection, so a
    collection's leftover vectors in another collection's space were served and doubled its
    RRF weight. Each space now gets only the collections whose ACTIVE space it is."""
    on_one, on_two = uuid4(), uuid4()
    backend = FakeBackend([], {1: [A], 2: [B]}, [])
    backend.by_space = {2: [on_two], 1: [on_one]}
    retriever = HybridRetriever(backend)
    assert list(await retriever.spaces_for(object(), [on_one, on_two])) == [1, 2]
    await retriever.retrieve(object(), "", {1: [0.1], 2: [0.2]}, [on_one, on_two], k=5)
    searched = {c[1][0]: c[1][3] for c in backend.calls if c[0] == "vector"}  # type: ignore[index]
    assert searched == {1: [on_one], 2: [on_two]}


async def test_retrieve_skips_keyword_search_for_blank_query_and_handles_empty_input() -> None:
    backend = FakeBackend([1], {1: [A]}, [B])
    retriever = HybridRetriever(backend)
    assert await retriever.retrieve(object(), "  ", {1: [0.0]}, [uuid4()], k=5) == [
        rrf_fuse([[A]])[0]
    ]
    assert not [c for c in backend.calls if c[0] == "keyword"]
    assert await retriever.retrieve(object(), "q", {}, [], k=5) == []
    assert await retriever.retrieve(object(), "q", {1: [0.0]}, [uuid4()], k=0) == []


async def test_retrieve_truncates_to_k_and_dedupes_collection_ids() -> None:
    backend = FakeBackend([1], {1: [A, B, C, D]}, [])
    coll = uuid4()
    out = await HybridRetriever(backend).retrieve(object(), "", {1: [0.0]}, [coll, coll], k=2)
    assert [h.anchor for h in out] == ["a", "b"]
    assert backend.calls[0] == ("spaces", [coll])


class FakeReranker:
    def __init__(self, scores: dict[str, float]) -> None:
        self.scores = scores
        self.seen: list[int] = []

    async def rerank(self, query: str, candidates: Sequence[RankedChunk]) -> list[RankedChunk]:
        import dataclasses

        self.seen.append(len(candidates))
        out = [
            dataclasses.replace(c, rerank_score=self.scores.get(c.anchor, 0.0)) for c in candidates
        ]
        return sorted(out, key=lambda c: -(c.rerank_score or 0))


async def test_rerank_and_select_reranks_top20_then_mmr_to_final_n() -> None:
    cands = [hit(f"h{i}", f"distinct topic {i} " + f"w{i} " * 5) for i in range(30)]
    reranker = FakeReranker({"h25": 0.99, "h3": 0.9, "h7": 0.8})
    out = await rerank_and_select("q", cands, reranker, final_n=2)
    assert reranker.seen == [20]  # h25 is outside the top-20 and never reaches the reranker
    assert [h.anchor for h in out] == ["h3", "h7"]
    assert out[0].rerank_score == 0.9
    assert await rerank_and_select("q", [], reranker) == []
