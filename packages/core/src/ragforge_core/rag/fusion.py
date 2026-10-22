"""Reciprocal Rank Fusion and duplicate collapse (DESIGN section 7.3).

RRF is rank-based on purpose: cosine distances from different embedding spaces and ts_rank_cd
scores from keyword search live on incomparable scales, but ranks always compare. That is what
makes multi-space queries (blue/green, mixed collections) safe to fuse (ADR-0009).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from uuid import UUID

from ragforge_core.domain.models import RankedChunk

RRF_K = 60


def rrf_fuse(ranked_lists: Sequence[Sequence[RankedChunk]], *, k: int = RRF_K) -> list[RankedChunk]:
    """Fuse ranked lists: score(d) = sum over lists of 1 / (k + rank_d), rank starting at 1.

    A chunk repeated inside one list counts once (at its best rank), so a backend returning
    duplicates cannot inflate a score. Ties break on the best rank seen, then on chunk id, so
    the output is deterministic for identical inputs.
    """
    if k <= 0:
        raise ValueError("k must be positive")
    scores: dict[UUID, float] = {}
    best_rank: dict[UUID, int] = {}
    first_seen: dict[UUID, RankedChunk] = {}
    for hits in ranked_lists:
        seen_here: set[UUID] = set()
        for rank, hit in enumerate(hits, start=1):
            if hit.chunk_id in seen_here:
                continue
            seen_here.add(hit.chunk_id)
            scores[hit.chunk_id] = scores.get(hit.chunk_id, 0.0) + 1.0 / (k + rank)
            best_rank[hit.chunk_id] = min(rank, best_rank.get(hit.chunk_id, rank))
            first_seen.setdefault(hit.chunk_id, hit)
    order = sorted(scores, key=lambda cid: (-scores[cid], best_rank[cid], str(cid)))
    return [dataclasses.replace(first_seen[cid], fused_score=scores[cid]) for cid in order]


def collapse_by_display_sha(hits: Sequence[RankedChunk]) -> list[RankedChunk]:
    """Keep only the first (best-ranked) hit per display_sha256.

    The same paragraph synced into several pages has different embedded text (different
    contextual headers) and therefore different embeddings, so it can surface several times;
    showing it once frees reranker and context budget for distinct evidence (DESIGN 7.2).
    """
    seen: set[bytes] = set()
    out: list[RankedChunk] = []
    for hit in hits:
        if hit.display_sha256 in seen:
            continue
        seen.add(hit.display_sha256)
        out.append(hit)
    return out
