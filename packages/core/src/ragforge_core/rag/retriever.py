"""Hybrid retriever (DESIGN section 7.3) and the post-transaction selection step.

Split in two on purpose:
  * `HybridRetriever.retrieve` (the `Retriever` port) runs INSIDE the caller's short tenant
    transaction: one vector search per active space + one keyword search, RRF, collapse.
    It does no model work, so the transaction stays a few tens of milliseconds.
  * `rerank_and_select` runs AFTER the transaction closes: FlashRank (CPU, ~250 ms) -> MMR ->
    top-n. Holding a pooled connection while the reranker runs would serialise chat turns on
    the pool, and PHASE1 forbids model calls inside a transaction.

The SQL lives in adapters/db/search.py, which satisfies `SearchBackend` structurally (adapters
may not import rag, so the Protocol is defined here and uses domain types only).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol
from uuid import UUID

from ragforge_core.domain.models import RankedChunk
from ragforge_core.domain.ports import Reranker, Tx
from ragforge_core.observability.logging import get_logger
from ragforge_core.rag.fusion import RRF_K, collapse_by_display_sha, rrf_fuse
from ragforge_core.rag.mmr import MMR_LAMBDA, mmr_select

PER_LIST_K = 40  # DESIGN 7.3: k=40 from each of vector and keyword search
RERANK_TOP = 20  # ADR-0006
FINAL_TOP_LOCAL = 5
FINAL_TOP_HOSTED = 8

log = get_logger(__name__)


class SearchBackend(Protocol):
    """What the retriever needs from storage. Implemented by adapters.db.search.PgSearchBackend.

    Every method runs in the caller's tenant `tx`; the tenant is ambient, never a parameter.
    Returned hits are ordered best-first; their `fused_score` holds the backend's raw score
    (only the order matters to RRF).
    """

    async def active_spaces(
        self, tx: Tx, collection_ids: Sequence[UUID]
    ) -> Mapping[int, Sequence[UUID]]:
        """ACTIVE space id -> the requested collections served from it. Spaces whose status is
        not 'active' and collections of other tenants are absent."""
        ...

    async def vector_search(
        self,
        tx: Tx,
        space_id: int,
        query_vector: Sequence[float],
        collection_ids: Sequence[UUID],
        k: int,
    ) -> Sequence[RankedChunk]: ...

    async def keyword_search(
        self, tx: Tx, query: str, collection_ids: Sequence[UUID], k: int
    ) -> Sequence[RankedChunk]: ...


class HybridRetriever:
    """Implements the `Retriever` port."""

    def __init__(
        self, backend: SearchBackend, *, per_list_k: int = PER_LIST_K, rrf_k: int = RRF_K
    ) -> None:
        if per_list_k <= 0:
            raise ValueError("per_list_k must be positive")
        self._backend = backend
        self._per_list_k = per_list_k
        self._rrf_k = rrf_k

    # Read-only accessors for the eval stage recorder (agent.retrieval.recording), which wraps
    # the SAME backend with the SAME k values so stages and hits cannot diverge (ADR-0017 E-R).
    @property
    def backend(self) -> SearchBackend:
        return self._backend

    @property
    def per_list_k(self) -> int:
        return self._per_list_k

    @property
    def rrf_k(self) -> int:
        return self._rrf_k

    async def spaces_for(self, tx: Tx, collection_ids: Sequence[UUID]) -> Sequence[int]:
        if not collection_ids:
            return []
        return sorted(await self._backend.active_spaces(tx, _unique(collection_ids)))

    async def retrieve(
        self,
        tx: Tx,
        query: str,
        query_vectors: Mapping[int, Sequence[float]],
        collection_ids: Sequence[UUID],
        k: int,
    ) -> list[RankedChunk]:
        """Fused, collapsed candidates (best first), at most `k`. Pass k=RERANK_TOP when the
        result feeds `rerank_and_select`.

        Searches every ACTIVE space that has a query vector. A space with no vector happens
        when a blue/green flip lands between `spaces_for` (which chose what to embed) and this
        call, in a different transaction. Raising would turn a rare, benign race into a
        user-visible error, so the space is skipped and logged (`retrieve_space_without_vector`)
        instead: its collections still get keyword hits, and the next turn embeds for the new
        space. Vectors for spaces that are no longer active are ignored the same way.

        Each space is searched over ONLY the collections whose active space it is. A collection
        also keeps vectors in its previous space (instant rollback) and its building space
        (backfill); searching every requested collection in every space would serve those
        stale or unverified vectors and give the collection a second RRF list.
        """
        if k <= 0 or not collection_ids:
            return []
        collections = _unique(collection_ids)
        active = await self._backend.active_spaces(tx, collections)
        missing = sorted(set(active) - set(query_vectors))
        if missing:
            log.warning("retrieve_space_without_vector", spaces=missing)
        lists: list[Sequence[RankedChunk]] = []
        for space in sorted(set(active) & set(query_vectors)):  # one connection: sequential
            lists.append(
                await self._backend.vector_search(
                    tx, space, query_vectors[space], active[space], self._per_list_k
                )
            )
        if query.strip():
            lists.append(
                await self._backend.keyword_search(tx, query, collections, self._per_list_k)
            )
        return collapse_by_display_sha(rrf_fuse(lists, k=self._rrf_k))[:k]


async def rerank_and_select(
    query: str,
    candidates: Sequence[RankedChunk],
    reranker: Reranker,
    *,
    rerank_top: int = RERANK_TOP,
    final_n: int = FINAL_TOP_LOCAL,
    mmr_lambda: float = MMR_LAMBDA,
) -> list[RankedChunk]:
    """Rerank the top `rerank_top` candidates, then MMR down to `final_n`. Call with NO
    transaction open. The grade node compares the first hit's `rerank_score` with tau
    (ADR-0006); it is None when the reranker is unavailable."""
    pool = list(candidates[:rerank_top])
    if not pool:
        return []
    reranked = await reranker.rerank(query, pool)
    return mmr_select(reranked, final_n, lambda_=mmr_lambda)


def _unique(ids: Sequence[UUID]) -> list[UUID]:
    return list(dict.fromkeys(ids))
