"""Phase 2 seams owned by retrieval wiring.

1. A space flipped or created after startup is served on its FIRST use (agent retrieval and
   /v1/query both `await EmbedderRegistry.ensure` before `for_space`), instead of a 503 that
   only the next request would have avoided.
2. The bootstrap-configured chunker never emits a prose chunk the cross-encoder cannot score
   whole next to a maximal query (flashrank PASSAGE_BUDGET).
3. The reranker's inference runs on its own thread, which `close()` stops.
"""

from __future__ import annotations

import threading
from collections.abc import AsyncIterator
from typing import Any
from uuid import uuid4

import httpx
import pytest
import respx

from module_d1.d1_fakes import (
    COLLECTION,
    FakeReranker,
    make_harness,
    make_hit,
    principal,
)
from ragforge_core.adapters.llm.embedder_registry import EmbeddingSpace, OllamaEmbedderRegistry
from ragforge_core.adapters.rerank.flashrank import (
    MAX_QUERY_TOKENS,
    PASSAGE_BUDGET,
    FlashRankReranker,
)
from ragforge_core.agent import AgentDeps, AgentSettings
from ragforge_core.agent.context import AgentContext, RunLease
from ragforge_core.agent.retrieval import retrieve
from ragforge_core.apps.api.routes.ask import _embed_query
from ragforge_core.bootstrap import _chunker
from ragforge_core.domain.models import (
    NormalizedBlock,
    RankedChunk,
    SourceDocument,
    SourceItemRef,
    SourceKind,
)
from ragforge_core.rag.tokens import count_tokens

BASE = "http://ollama.test"
VECTOR = [0.1, 0.2, 0.3]


@pytest.fixture
async def client() -> AsyncIterator[httpx.AsyncClient]:
    async with httpx.AsyncClient(base_url=BASE) as c:
        yield c


def _space(id_: int) -> EmbeddingSpace:
    return EmbeddingSpace(
        id=id_, provider="ollama", model="m", dim=3, doc_prefix="", query_prefix="",
        distance="cosine", status="active",
    )  # fmt: skip


def _late_registry(client: httpx.AsyncClient) -> tuple[OllamaEmbedderRegistry, list[int]]:
    """Starts with NO spaces; the loader knows space 1 (as after a blue/green flip)."""
    loads: list[int] = []

    async def loader() -> list[EmbeddingSpace]:
        loads.append(1)
        return [_space(1)]

    return OllamaEmbedderRegistry([], client, loader=loader), loads


# ---------------------------------------------------------------- 1. first use after a flip
@respx.mock
async def test_query_path_serves_a_space_loaded_after_startup_on_first_use(
    client: httpx.AsyncClient,
) -> None:
    respx.post(f"{BASE}/api/embed").mock(
        return_value=httpx.Response(200, json={"embeddings": [VECTOR]})
    )
    registry, loads = _late_registry(client)

    vectors = await _embed_query(registry, [1, 1], "q")  # duplicate space: embedded once

    assert vectors == {1: VECTOR}
    assert loads == [1], "exactly one awaited reload, then served"


@respx.mock
async def test_agent_path_serves_a_space_loaded_after_startup_on_first_use(
    client: httpx.AsyncClient,
) -> None:
    respx.post(f"{BASE}/api/embed").mock(
        return_value=httpx.Response(200, json={"embeddings": [VECTOR]})
    )
    registry, loads = _late_registry(client)
    h = make_harness(results=[[make_hit(1)]])
    deps = AgentDeps(
        chat=h.chat,
        retriever=h.retriever,
        reranker=FakeReranker(h.txm),
        embedders=registry,
        runs=h.runs,
        tx=h.txm,
        jobs=h.jobs,
        conversations=h.conversations,
        knowledge=h.knowledge,
        store=h.store,
        settings=AgentSettings(heartbeat_interval_s=3600),
    )
    who = principal()
    conversation, run = uuid4(), uuid4()
    ctx = AgentContext(
        principal=who,
        conversation_id=conversation,
        run_id=run,
        deps=deps,
        lease=RunLease(h.runs, who.tenant_id, conversation, run),
    )

    found = await retrieve(ctx, "what is fact one?", [COLLECTION])

    assert [hit.anchor for hit in found.hits] == ["block-1"]
    assert loads == [1]


# ---------------------------------------------------------------- 2. chunk fits the reranker
def test_bootstrap_chunker_keeps_prose_within_the_reranker_budget() -> None:
    assert PASSAGE_BUDGET == 512 - 3 - MAX_QUERY_TOKENS == 445
    sentence = "The rotation policy covers every engineer on the platform team equally. "
    long_section = sentence * 60  # well over 500 tokens in one paragraph
    assert count_tokens(long_section) > 500
    blocks = (
        NormalizedBlock("h", "heading_1", "# Policy", 1),
        NormalizedBlock("p1", "paragraph", long_section),
        NormalizedBlock("p2", "paragraph", sentence * 12),
    )
    ref = SourceItemRef(SourceKind.file, "doc", None)
    chunks = _chunker().chunk(SourceDocument(ref=ref, title="T", external_url=None, blocks=blocks))

    assert len(chunks) > 1
    assert max(count_tokens(c.display_text) for c in chunks) <= PASSAGE_BUDGET
    assert all(c.token_count <= PASSAGE_BUDGET for c in chunks)


# ---------------------------------------------------------------- 3. own inference thread
class _ThreadRecordingRanker:
    def __init__(self) -> None:
        self.threads: list[str] = []

    def rerank(self, request: Any) -> list[dict[str, Any]]:
        self.threads.append(threading.current_thread().name)
        return [{"id": p["id"], "score": 0.5} for p in request.passages]


async def test_reranker_runs_on_its_own_thread_and_close_stops_it() -> None:
    ranker = _ThreadRecordingRanker()
    reranker = FlashRankReranker(
        cache_dir="/nonexistent", token_counter=count_tokens, ranker_factory=lambda: ranker
    )
    await reranker.load()
    hits: list[RankedChunk] = [make_hit(1, score=None), make_hit(2, score=None)]

    scored = await reranker.rerank("q", hits)

    assert all(h.rerank_score == 0.5 for h in scored)
    assert ranker.threads
    assert all(t.startswith("rerank") for t in ranker.threads)
    calls = len(ranker.threads)
    reranker.close()
    reranker.close()  # idempotent
    # After close no inference runs; like any inference failure, the call degrades to fused
    # order (never fails the turn) - only possible during shutdown.
    after = await reranker.rerank("q", hits)
    assert len(ranker.threads) == calls
    assert [h.rerank_score for h in after] == [None, None]
