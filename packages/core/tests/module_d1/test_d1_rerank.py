"""The agent reranks its own retrieval (Phase 2 integration finding).

`Retriever.retrieve` returns fused candidates with `rerank_score=None` by contract (handoff B:
reranking is CPU model work and must not run inside a transaction). The grade node only passes
on a rerank score, so an agent that skipped the reranker would abstain on every real turn while
every fake-backed test (whose hits carry scores already) stayed green.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence

from module_d1.d1_fakes import (
    COLLECTION,
    FakeTxManager,
    collect,
    make_harness,
    make_hit,
    of_type,
    principal,
    say,
    verdict,
)
from ragforge_core.agent import AgentSettings
from ragforge_core.domain.models import RankedChunk, TurnSources, TurnVerification, Verification


class ScoringReranker:
    """Gives the fused candidates descending cross-encoder scores, like FlashRank would."""

    def __init__(self, txm: FakeTxManager) -> None:
        self.txm = txm
        self.seen: list[int] = []

    async def rerank(self, query: str, candidates: Sequence[RankedChunk]) -> list[RankedChunk]:
        assert self.txm.depth == 0, "reranker called inside a transaction"
        self.seen.append(len(candidates))
        return [
            dataclasses.replace(c, rerank_score=0.9 - 0.01 * i) for i, c in enumerate(candidates)
        ]


async def test_unscored_candidates_are_reranked_and_the_turn_answers() -> None:
    fused = [make_hit(n, score=None) for n in range(1, 31)]
    rerankers: list[ScoringReranker] = []

    def factory(txm: FakeTxManager) -> ScoringReranker:
        rerankers.append(ScoringReranker(txm))
        return rerankers[-1]

    h = make_harness(
        [say("Fact one holds [S1]."), verdict()],
        results=[fused],
        settings=AgentSettings(heartbeat_interval_s=3600, retrieval_k=5, rerank_top=20),
        reranker=factory,
    )
    events = await collect(h.runner.run_turn(principal(), None, [COLLECTION], "what is fact one?"))

    assert rerankers[0].seen == [20]  # only the top rerank_top candidates reach the model
    (sources,) = of_type(events, TurnSources)
    assert len(sources.hits) == 5  # MMR down to retrieval_k
    assert all(hit.rerank_score is not None for hit in sources.hits)
    (ver,) = of_type(events, TurnVerification)
    assert ver.status is Verification.verified
    assert h.chat.kinds() == ["stream", "json"]  # answered: no abstain shortcut
