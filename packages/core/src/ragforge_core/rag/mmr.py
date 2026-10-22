"""Maximal Marginal Relevance selection (DESIGN section 7.3: rerank -> MMR -> top-n).

MMR(d) = lambda * relevance(d) - (1 - lambda) * max_{s in selected} similarity(d, s)

Similarity is lexical (Jaccard over word sets) rather than embedding cosine: `RankedChunk`
carries no vector, and fetching vectors for 20 candidates would cost a second round trip and
~60 KB of detoasting per query just to de-duplicate near-identical passages, which word overlap
already catches well. The similarity function is injectable if evals say otherwise.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence

from ragforge_core.domain.models import RankedChunk

MMR_LAMBDA = 0.7
_WORDS = re.compile(r"\w+", re.UNICODE)

Similarity = Callable[[RankedChunk, RankedChunk], float]


def relevance_of(hit: RankedChunk) -> float:
    """Reranker score when the reranker ran, else the RRF score."""
    return hit.rerank_score if hit.rerank_score is not None else hit.fused_score


def jaccard_similarity(a: RankedChunk, b: RankedChunk) -> float:
    words_a = set(_WORDS.findall(a.display_text.lower()))
    words_b = set(_WORDS.findall(b.display_text.lower()))
    if not words_a or not words_b:
        return 0.0
    return len(words_a & words_b) / len(words_a | words_b)


def mmr_select(
    candidates: Sequence[RankedChunk],
    n: int,
    *,
    lambda_: float = MMR_LAMBDA,
    similarity: Similarity = jaccard_similarity,
) -> list[RankedChunk]:
    """Greedily pick `n` hits balancing relevance and novelty.

    Relevance is scaled to [0, 1] first so that `lambda_` means the same thing for reranker
    probabilities and for tiny RRF scores (~0.016). Non-negative scores (the normal case) are
    divided by the maximum, which keeps their ratios; min-max scaling would pin the weakest
    candidate to 0 and, in a pool of a few, make novelty unable to ever beat a near-duplicate.
    Ties keep the input order, so with lambda_ = 1 the output equals the input prefix.
    """
    if not 0.0 <= lambda_ <= 1.0:
        raise ValueError("lambda_ must be in [0, 1]")
    if n <= 0 or not candidates:
        return []
    rel = _scaled([relevance_of(c) for c in candidates])

    remaining = list(range(len(candidates)))
    chosen: list[int] = []
    max_sim = [0.0] * len(candidates)
    while remaining and len(chosen) < n:
        best = max(
            remaining,
            key=lambda i: (lambda_ * rel[i] - (1.0 - lambda_) * max_sim[i], -i),
        )
        chosen.append(best)
        remaining.remove(best)
        for i in remaining:
            max_sim[i] = max(max_sim[i], similarity(candidates[i], candidates[best]))
    return [candidates[i] for i in chosen]


def _scaled(raw: list[float]) -> list[float]:
    low, high = min(raw), max(raw)
    if low >= 0.0:
        return [r / high if high > 0 else 1.0 for r in raw]
    span = high - low
    return [(r - low) / span if span > 0 else 1.0 for r in raw]
