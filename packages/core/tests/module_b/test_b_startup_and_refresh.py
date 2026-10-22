"""Phase 2a review fixes: eager reranker load (fail startup, never a silent all-abstain mode)
and embedder-registry refresh on a miss (a space added after startup is picked up; a miss is a
retryable 503, never a permanent job failure)."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import httpx
import pytest

from ragforge_core.adapters.llm import embedder_registry
from ragforge_core.adapters.llm.embedder_registry import (
    EmbeddingSpace,
    EmbeddingSpaceRetired,
    EmbeddingSpaceUnavailable,
    EmbeddingSpaceUnsupported,
    OllamaEmbedderRegistry,
)
from ragforge_core.adapters.rerank.flashrank import FlashRankReranker, RerankerUnavailable
from ragforge_core.domain.errors import NotFound, ValidationFailed
from ragforge_core.rag.tokens import count_tokens
from ragforge_core.services.sync.handlers import _PERMANENT


@pytest.fixture
async def client() -> AsyncIterator[httpx.AsyncClient]:
    async with httpx.AsyncClient(base_url="http://ollama.test") as c:
        yield c


def space(id_: int, status: str = "active", provider: str = "ollama") -> EmbeddingSpace:
    return EmbeddingSpace(
        id=id_, provider=provider, model=f"m{id_}", dim=3, doc_prefix="", query_prefix="",
        distance="cosine", status=status,
    )  # fmt: skip


class OkRanker:
    def rerank(self, request: Any) -> list[dict[str, Any]]:
        return [{"id": p["id"], "score": 0.5} for p in request.passages]


# ---------------------------------------------------------------- reranker
async def test_load_raises_naming_model_and_cache_dir_when_the_model_is_missing(
    tmp_path: Path,
) -> None:
    reranker = FlashRankReranker(
        cache_dir=tmp_path, token_counter=count_tokens, model_name="no-such-model"
    )
    with pytest.raises(RerankerUnavailable) as err:
        await reranker.load()
    assert "no-such-model" in str(err.value)
    assert str(tmp_path) in str(err.value)
    assert reranker.available is False


async def test_load_is_eager_idempotent_and_single_flight(tmp_path: Path) -> None:
    loads = 0

    def factory() -> OkRanker:
        nonlocal loads
        loads += 1
        return OkRanker()

    reranker = FlashRankReranker(
        cache_dir=tmp_path, token_counter=count_tokens, ranker_factory=factory
    )
    await asyncio.gather(reranker.load(), reranker.load(), reranker.load())
    await reranker.load()
    assert loads == 1
    assert reranker.available is True


# ---------------------------------------------------------------- registry refresh
def test_a_miss_is_retryable_503_without_the_space_id(client: httpx.AsyncClient) -> None:
    registry = OllamaEmbedderRegistry([space(1)], client)
    with pytest.raises(EmbeddingSpaceUnavailable) as err:
        registry.for_space(77)
    assert err.value.status == 503
    assert "77" not in str(err.value)
    # run_sync_job treats these as permanent; the miss must not be one of them
    assert not isinstance(err.value, _PERMANENT)
    assert not isinstance(err.value, NotFound)


async def test_a_miss_starts_one_reload_and_the_next_call_finds_the_new_space(
    client: httpx.AsyncClient,
) -> None:
    calls = 0
    rows = [space(1)]

    async def loader() -> list[EmbeddingSpace]:
        nonlocal calls
        calls += 1
        await asyncio.sleep(0)
        return list(rows)

    registry = OllamaEmbedderRegistry(list(rows), client, loader=loader)
    first = registry.for_space(1)
    rows.append(space(2))  # a runbook adds a space after startup
    for _ in range(3):
        with pytest.raises(EmbeddingSpaceUnavailable):
            registry.for_space(2)
    await asyncio.sleep(0.05)  # let the background reload run
    assert calls == 1, "concurrent misses share one reload"
    assert registry.for_space(2).model == "m2"
    assert registry.for_space(1) is first, "unchanged spaces keep their embedder"


async def test_ensure_awaits_the_reload_and_retirement_is_picked_up(
    client: httpx.AsyncClient,
) -> None:
    rows = [space(1)]

    async def loader() -> list[EmbeddingSpace]:
        return list(rows)

    registry = OllamaEmbedderRegistry(list(rows), client, loader=loader)
    rows.append(space(3))
    await registry.ensure(3)
    assert registry.for_space(3).model == "m3"
    rows[0] = space(1, status="retired")
    await registry.refresh()
    with pytest.raises(ValidationFailed, match="retired"):
        registry.for_space(1)


async def test_a_failing_loader_keeps_the_current_spaces(client: httpx.AsyncClient) -> None:
    async def loader() -> list[EmbeddingSpace]:
        raise ConnectionError("db down")

    registry = OllamaEmbedderRegistry([space(1)], client, loader=loader)
    await registry.refresh()
    assert registry.for_space(1).model == "m1"
    with pytest.raises(EmbeddingSpaceUnavailable):
        await registry.ensure(9)


def test_a_miss_outside_an_event_loop_still_raises_cleanly(client: httpx.AsyncClient) -> None:
    async def loader() -> list[EmbeddingSpace]:
        return []

    registry = OllamaEmbedderRegistry([], client, loader=loader)
    with pytest.raises(EmbeddingSpaceUnavailable):
        registry.for_space(1)


def test_unusable_space_errors_are_fixed_and_carry_no_id_or_provider(
    client: httpx.AsyncClient,
) -> None:
    registry = OllamaEmbedderRegistry(
        [space(41, status="retired"), space(42, provider="openai")], client
    )
    with pytest.raises(EmbeddingSpaceRetired) as retired:
        registry.for_space(41)
    assert isinstance(retired.value, ValidationFailed)  # definitive for a sync job
    with pytest.raises(EmbeddingSpaceUnsupported) as foreign:
        registry.for_space(42)
    assert foreign.value.status == 503
    for err in (retired.value, foreign.value):
        for leaked in ("41", "42", "openai"):
            assert leaked not in str(err)


async def test_an_unretired_space_recovers_without_a_restart_and_rechecks_are_rate_limited(
    client: httpx.AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows = [space(5, status="retired")]
    calls = 0

    async def loader() -> list[EmbeddingSpace]:
        nonlocal calls
        calls += 1
        return list(rows)

    clock = [1000.0]
    monkeypatch.setattr(embedder_registry, "_now", lambda: clock[0])
    registry = OllamaEmbedderRegistry(list(rows), client, loader=loader)
    rows[0] = space(5)  # a runbook flips the space back to active
    for _ in range(5):
        with pytest.raises(EmbeddingSpaceRetired):
            registry.for_space(5)
    await asyncio.sleep(0.05)
    assert calls == 1, "lookups of an unusable space reload at most once per window"
    assert registry.for_space(5).model == "m5"

    rows[0] = space(5, status="retired")
    await registry.refresh()
    calls = 0
    with pytest.raises(EmbeddingSpaceRetired):
        registry.for_space(5)
    await asyncio.sleep(0.05)
    assert calls == 0, "still inside the window"
    clock[0] += embedder_registry.UNUSABLE_RECHECK_S
    with pytest.raises(EmbeddingSpaceRetired):
        registry.for_space(5)
    await asyncio.sleep(0.05)
    assert calls == 1


async def test_aclose_cancels_a_reload_in_flight_and_schedules_no_more(
    client: httpx.AsyncClient,
) -> None:
    started = asyncio.Event()
    calls = 0

    async def loader() -> list[EmbeddingSpace]:
        nonlocal calls
        calls += 1
        started.set()
        await asyncio.sleep(3600)
        return []

    registry = OllamaEmbedderRegistry([], client, loader=loader)
    with pytest.raises(EmbeddingSpaceUnavailable):
        registry.for_space(1)
    await asyncio.wait_for(started.wait(), 1)
    await asyncio.wait_for(registry.aclose(), 1)  # returns promptly: the reload was cancelled
    with pytest.raises(EmbeddingSpaceUnavailable):
        registry.for_space(1)
    await asyncio.sleep(0.05)
    assert calls == 1, "a closed registry never starts another reload"


async def test_ensure_rereads_a_space_cached_as_retired_before_failing_a_job(
    client: httpx.AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Review round 4: a blue/green rollback un-retires a space. A worker that cached it as
    retired must not dead-letter the job on the stale cache: `ensure` re-reads (rate-limited),
    and inside the rate-limit window it answers with the RETRYABLE error, never the permanent
    retired one. Only a fresh read that still says retired is definitive."""
    rows = [space(5, status="retired")]
    calls = 0

    async def loader() -> list[EmbeddingSpace]:
        nonlocal calls
        calls += 1
        return list(rows)

    clock = [1000.0]
    monkeypatch.setattr(embedder_registry, "_now", lambda: clock[0])
    registry = OllamaEmbedderRegistry(list(rows), client, loader=loader)

    rows[0] = space(5)  # un-retired in the database; the cache still says retired
    await registry.ensure(5)  # awaited re-read: served, no error
    assert calls == 1
    assert registry.for_space(5).model == "m5"

    rows[0] = space(5, status="retired")
    await registry.refresh()  # retired again; the cache now says retired
    clock[0] += embedder_registry.UNUSABLE_RECHECK_S
    with pytest.raises(EmbeddingSpaceRetired):  # a fresh read confirms it: definitive
        await registry.ensure(5)  # ...and starts the rate-limit window

    rows[0] = space(5)  # flipped back again, inside the window this time
    with pytest.raises(EmbeddingSpaceUnavailable) as stale:
        await registry.ensure(5)
    assert not isinstance(stale.value, _PERMANENT), "a stale cache must only cause a retry"
    clock[0] += embedder_registry.UNUSABLE_RECHECK_S
    await registry.ensure(5)  # the window passed: re-read, served


async def test_concurrent_refreshes_share_one_reload(client: httpx.AsyncClient) -> None:
    gate = asyncio.Event()
    calls = 0

    async def loader() -> list[EmbeddingSpace]:
        nonlocal calls
        calls += 1
        await gate.wait()
        return [space(1)]

    registry = OllamaEmbedderRegistry([], client, loader=loader)
    waiters = [asyncio.create_task(registry.refresh()) for _ in range(5)]
    await asyncio.sleep(0.01)
    gate.set()
    await asyncio.gather(*waiters)
    assert calls == 1


async def test_a_failed_reread_never_answers_retired_from_the_stale_cache(
    client: httpx.AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Review round 5: a space cached as retired, a DB blip during the re-read. `refresh` reports
    the read failed, and `ensure` answers with the retryable error, not the permanent retired one
    (which would dead-letter a sync job on a cache that a rollback may already have outdated)."""
    healthy = [True]

    async def loader() -> list[EmbeddingSpace]:
        if not healthy[0]:
            raise ConnectionError("db down")
        return [space(5)]

    clock = [1000.0]
    monkeypatch.setattr(embedder_registry, "_now", lambda: clock[0])
    registry = OllamaEmbedderRegistry([space(5, status="retired")], client, loader=loader)

    healthy[0] = False
    assert await registry.refresh() is False
    with pytest.raises(EmbeddingSpaceUnavailable) as stale:
        await registry.ensure(5)
    assert not isinstance(stale.value, _PERMANENT)

    healthy[0] = True
    clock[0] += embedder_registry.UNUSABLE_RECHECK_S
    await registry.ensure(5)  # the next fresh read serves it
    assert registry.for_space(5).model == "m5"
