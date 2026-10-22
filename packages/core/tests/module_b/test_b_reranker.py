"""FlashRank adapter with a fake ranker: top-20 only, thread offload, graceful degradation."""

from __future__ import annotations

import asyncio
import hashlib
import threading
from dataclasses import replace
from pathlib import Path
from typing import Any
from uuid import uuid4

from ragforge_core.adapters.rerank.flashrank import (
    MAX_LENGTH,
    MAX_QUERY_TOKENS,
    FlashRankReranker,
    truncate_to_tokens,
)
from ragforge_core.domain.models import RankedChunk
from ragforge_core.domain.ports import Reranker
from ragforge_core.rag.tokens import count_tokens


def hit(i: int) -> RankedChunk:
    text = f"passage {i}"
    return RankedChunk(
        chunk_id=uuid4(), document_id=uuid4(), collection_id=uuid4(), anchor=f"a{i}",
        title="Guide", heading_path=("Install",), display_text=text,
        display_sha256=hashlib.sha256(text.encode()).digest(), external_url=None,
        fused_score=1.0 / (60 + i),
    )  # fmt: skip


class FakeRanker:
    """Scores passage i as i/100 so the reranked order reverses the fused order."""

    def __init__(self) -> None:
        self.requests: list[Any] = []
        self.threads: list[str] = []

    def rerank(self, request: Any) -> list[dict[str, Any]]:
        self.requests.append(request)
        self.threads.append(threading.current_thread().name)
        out = [{**p, "score": int(p["text"].rsplit(" ", 1)[-1]) / 100} for p in request.passages]
        return sorted(out, key=lambda r: -r["score"])


async def test_reranks_only_top_20_in_a_worker_thread(tmp_path: Path) -> None:
    fake = FakeRanker()
    reranker = FlashRankReranker(
        cache_dir=tmp_path, token_counter=count_tokens, ranker_factory=lambda: fake
    )
    assert isinstance(reranker, Reranker)
    candidates = [hit(i) for i in range(30)]
    out = await reranker.rerank("how to install", candidates)
    assert len(fake.requests[0].passages) == 20
    assert fake.requests[0].query == "how to install"
    assert fake.requests[0].passages[0]["text"] == "Guide › Install\n\npassage 0"
    assert [h.anchor for h in out[:3]] == ["a19", "a18", "a17"]
    assert out[0].rerank_score == 0.19
    assert [h.anchor for h in out[20:]] == [f"a{i}" for i in range(20, 30)]
    assert all(h.rerank_score is None for h in out[20:])
    assert fake.threads[0] != threading.main_thread().name
    assert MAX_LENGTH == 512
    assert reranker.available is True


async def test_missing_model_degrades_to_fused_order_without_download(tmp_path: Path) -> None:
    reranker = FlashRankReranker(
        cache_dir=tmp_path, token_counter=count_tokens
    )  # real loader, model dir absent
    candidates = [hit(i) for i in range(5)]
    out = await reranker.rerank("q", candidates)
    assert out == candidates
    assert reranker.available is False
    assert await asyncio.to_thread(lambda: list(tmp_path.iterdir())) == []  # no download


async def test_loader_failure_is_attempted_once_even_under_concurrency(tmp_path: Path) -> None:
    calls = 0

    def broken() -> FakeRanker:
        nonlocal calls
        calls += 1
        raise RuntimeError("onnx exploded")

    reranker = FlashRankReranker(
        cache_dir=tmp_path, token_counter=count_tokens, ranker_factory=broken
    )
    results = await asyncio.gather(*(reranker.rerank("q", [hit(1)]) for _ in range(5)))
    assert calls == 1
    assert all(r[0].rerank_score is None for r in results)


async def test_empty_candidates(tmp_path: Path) -> None:
    reranker = FlashRankReranker(
        cache_dir=tmp_path, token_counter=count_tokens, ranker_factory=FakeRanker
    )
    assert await reranker.rerank("q", []) == []
    assert reranker.available is None  # never loaded for nothing


async def test_inference_failure_degrades_to_fused_order(tmp_path: Path) -> None:
    """Review finding: only LOAD failures degraded; an ONNX/tokenizer error inside rerank()
    failed the whole chat turn. Now that call degrades, and the next one tries again."""

    class Flaky(FakeRanker):
        def rerank(self, request: Any) -> list[dict[str, Any]]:
            if not self.requests:
                self.requests.append(request)
                raise RuntimeError("ONNXRuntimeError: secret passage text")
            return super().rerank(request)

    fake = Flaky()
    reranker = FlashRankReranker(
        cache_dir=tmp_path, token_counter=count_tokens, ranker_factory=lambda: fake
    )
    candidates = [hit(i) for i in range(25)]
    out = await reranker.rerank("q", candidates)
    assert out == candidates
    assert all(h.rerank_score is None for h in out)
    assert reranker.available is True  # the model is fine; only that call failed
    again = await reranker.rerank("q", candidates)
    assert again[0].rerank_score == 0.19


async def test_header_is_dropped_when_it_would_push_the_chunk_tail_past_512(
    tmp_path: Path,
) -> None:
    """Review finding: query + "title › heading" + a chunk near the 512 max overflowed the
    cross-encoder window, so the chunk's TAIL was silently cut. The chunk text wins."""
    fake = FakeRanker()
    reranker = FlashRankReranker(
        cache_dir=tmp_path, token_counter=count_tokens, ranker_factory=lambda: fake
    )
    query = "how do I rotate the signing keys"  # 7 tokens
    budget = MAX_LENGTH - 3 - count_tokens(query)
    header_cost = count_tokens("Guide › Install")
    fits = "word " * (budget - header_cost - 1) + "1"  # exactly fills the budget with the header
    too_long = "word " * (budget - header_cost) + "2"  # one token more
    short, long_ = replace(hit(1), display_text=fits), replace(hit(2), display_text=too_long)
    await reranker.rerank(query, [short, long_])
    texts = [p["text"] for p in fake.requests[0].passages]
    assert texts[0] == f"Guide › Install\n\n{fits}"
    assert count_tokens(fits) + header_cost == budget
    assert texts[1] == too_long
    for text in texts:
        assert count_tokens(query) + count_tokens(text) + 3 <= MAX_LENGTH


async def test_long_query_is_cut_so_a_max_length_chunk_is_never_truncated(
    tmp_path: Path,
) -> None:
    """Review finding: budget = 512 - 3 - count(query) went negative for long queries, so the
    cross-encoder cut the chunk. The query now gets at most MAX_QUERY_TOKENS of the window."""
    fake = FakeRanker()
    reranker = FlashRankReranker(
        cache_dir=tmp_path, token_counter=count_tokens, ranker_factory=lambda: fake
    )
    query = "explain " * 600  # ~600 tokens: longer than the whole window
    chunk = "word " * (MAX_LENGTH - 3 - MAX_QUERY_TOKENS - 1) + "1"
    await reranker.rerank(query, [replace(hit(1), display_text=chunk)])
    request = fake.requests[0]
    assert count_tokens(request.query) <= MAX_QUERY_TOKENS
    assert request.query
    assert query.startswith(request.query)
    assert request.passages[0]["text"].endswith(chunk)
    assert count_tokens(request.query) + count_tokens(request.passages[0]["text"]) + 3 <= MAX_LENGTH


def test_truncate_to_tokens_keeps_short_text_and_bounds_long_text() -> None:
    assert truncate_to_tokens("short query", 64, count_tokens) == "short query"
    cjk = "检索增强生成" * 50
    cut = truncate_to_tokens(cjk, 10, count_tokens)
    assert count_tokens(cut) <= 10
    assert cjk.startswith(cut)
    assert len(cut) >= 9
    assert truncate_to_tokens("x", 0, count_tokens) == ""
