"""EV-JUDGE: the judge cache (hit, miss, invalidation by any JudgeKey field or item change), the
cache-key formula against the golden fixtures, and the D0 acceptance helper."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path

import pytest
import respx
from ev_judge_support import FIXTURES, OLLAMA, FakeOllama, good_reply
from hypothesis import given
from hypothesis import strategies as st

from ragforge_client.evals.interfaces import JudgeItem
from ragforge_client.evals.judge import (
    CachingJudge,
    JudgeCache,
    JudgeCall,
    JudgeConfig,
    acceptance,
    cache_key,
    make_item,
    make_judge,
)
from ragforge_client.evals.schema import E_G_CRITERIA, Criterion, JudgeKey, JudgeResult

KEY = JudgeKey(
    provider="ollama",
    model_version="qwen3:8b@abc",
    decoding={"temperature": 0, "num_ctx": 8192},
    prompt="judge_qa@1",
)
ITEM = make_item(E_G_CRITERIA, "q?", "a [S1].", ("note",), facts=("f",))


@dataclass
class CountingJudge:
    """A fake inner judge: scores every criterion 1 and counts what it was asked."""

    key: JudgeKey
    asked: list[tuple[Criterion, ...]] = field(default_factory=list)

    async def score(self, items: Sequence[JudgeItem]) -> list[JudgeResult]:
        out = []
        for item in items:
            self.asked.append(item.criteria)
            out += [
                JudgeResult(
                    cache_key=cache_key(self.key, item.item_digest, c),
                    item_digest=item.item_digest,
                    criterion=c,
                    score=1,
                    reason="r",
                    judge=self.key,
                    scored_at=datetime.now(UTC),
                )
                for c in item.criteria
            ]
        return out


def test_the_cache_key_is_the_one_the_golden_fixtures_use() -> None:
    lines = (FIXTURES / "judge-results.jsonl").read_text().splitlines()
    results = [JudgeResult.model_validate_json(ln) for ln in lines if ln.strip()]
    assert results
    for r in results:
        assert cache_key(r.judge, r.item_digest, r.criterion) == r.cache_key


async def test_hit_miss_and_partial_miss(tmp_path: Path) -> None:
    inner = CountingJudge(KEY)
    judge = CachingJudge(inner, JudgeCache(tmp_path / "judge-cache.jsonl"))
    first = await judge.score([ITEM])
    again = await judge.score([ITEM])
    assert inner.asked == [E_G_CRITERIA]  # the second call never reached the model
    assert [r.model_dump() for r in again] == [r.model_dump() for r in first]
    assert (judge.hits, judge.misses) == (3, 3)
    wider = make_item((*E_G_CRITERIA,), "q2?", "a2", ("note",), facts=("f",))
    await judge.score([wider])
    assert inner.asked[-1] == E_G_CRITERIA


async def test_a_partial_miss_re_asks_the_whole_item(tmp_path: Path) -> None:
    """Asking only the missing criteria would score them under a different prompt and reply
    schema than the cached ones while sharing their keys: the whole item is re-asked."""
    inner = CountingJudge(KEY)
    cache = JudgeCache(tmp_path / "c.jsonl")
    judge = CachingJudge(inner, cache)
    await judge.score([replace(ITEM, criteria=(Criterion.relevance,))])
    results = await judge.score([ITEM])
    assert inner.asked == [(Criterion.relevance,), E_G_CRITERIA]
    assert [r.criterion for r in results] == list(E_G_CRITERIA)
    # the relevance result now comes from the full-item call (newest line wins in the cache)
    relevance = cache.get(cache_key(KEY, ITEM.item_digest, Criterion.relevance))
    assert relevance == results[0]
    await judge.score([ITEM])
    assert len(inner.asked) == 2
    assert (judge.hits, judge.misses) == (3, 4)


def test_the_cache_reads_only_new_lines_and_waits_for_an_unfinished_one(tmp_path: Path) -> None:
    path = tmp_path / "c.jsonl"
    cache = JudgeCache(path)
    inner = CountingJudge(KEY)
    results = asyncio.run(inner.score([ITEM]))
    cache.put(results[:1])
    assert len(cache) == 1
    line = results[1].model_dump_json().encode()
    with path.open("ab") as handle:  # another writer, mid-write
        handle.write(line[:40])
    assert len(cache) == 1
    with path.open("ab") as handle:
        handle.write(line[40:] + b"\n")
    assert len(cache) == 2
    assert cache.get(results[1].cache_key) == results[1]
    path.write_text("")  # replaced (e.g. restored from a backup): re-read from the start
    assert len(cache) == 0


@pytest.mark.parametrize(
    "changed",
    [
        {"model_version": "qwen3:8b@def"},  # an alias that moved
        {"decoding": {"temperature": 0, "num_ctx": 4096}},
        {"prompt": "judge_qa@2"},
        {"provider": "gemini"},
    ],
)
async def test_a_changed_judge_key_misses_the_cache(
    tmp_path: Path, changed: dict[str, object]
) -> None:
    cache = JudgeCache(tmp_path / "c.jsonl")
    await CachingJudge(CountingJudge(KEY), cache).score([ITEM])
    moved = CountingJudge(KEY.model_copy(update=changed))
    await CachingJudge(moved, cache).score([ITEM])
    assert moved.asked == [E_G_CRITERIA]
    assert len(cache) == 6  # both judges' results kept; nothing overwritten


async def test_a_changed_item_misses_the_cache(tmp_path: Path) -> None:
    cache = JudgeCache(tmp_path / "c.jsonl")
    inner = CountingJudge(KEY)
    await CachingJudge(inner, cache).score([ITEM])
    for edited in (
        make_item(E_G_CRITERIA, "q?", "a [S1]. more", ("note",), facts=("f",)),
        make_item(E_G_CRITERIA, "q?", "a [S1].", ("note", "second note"), facts=("f",)),
        make_item(E_G_CRITERIA, "q?", "a [S1].", ("note",), facts=("f", "g")),
    ):
        before = len(inner.asked)
        await CachingJudge(inner, cache).score([edited])
        assert len(inner.asked) == before + 1


async def test_the_cache_survives_a_restart_and_sees_another_writer(tmp_path: Path) -> None:
    path = tmp_path / "judge-cache.jsonl"
    reader = JudgeCache(path)
    assert reader.get("missing") is None
    await CachingJudge(CountingJudge(KEY), JudgeCache(path)).score([ITEM])
    assert reader.get(cache_key(KEY, ITEM.item_digest, Criterion.relevance)) is not None
    assert path.stat().st_mode & 0o777 == 0o600


@respx.mock
async def test_invalid_replies_are_cached_and_counted_not_retried(tmp_path: Path) -> None:
    fake = FakeOllama(lambda user: "not json")
    respx.mock.get(f"{OLLAMA}/api/tags").mock(side_effect=fake.tags)
    respx.mock.post(f"{OLLAMA}/api/chat").mock(side_effect=fake.chat)
    async with await make_judge(JudgeConfig()) as inner:
        judge = CachingJudge(inner, JudgeCache(tmp_path / "c.jsonl"))
        first = await judge.score([ITEM])
        second = await judge.score([ITEM])
    assert len(fake.bodies) == 1
    assert {r.score for r in first + second} == {None}


@respx.mock
async def test_a_real_adapter_result_round_trips_through_the_cache(tmp_path: Path) -> None:
    fake = FakeOllama(good_reply(2))
    respx.mock.get(f"{OLLAMA}/api/tags").mock(side_effect=fake.tags)
    respx.mock.post(f"{OLLAMA}/api/chat").mock(side_effect=fake.chat)
    path = tmp_path / "judge-cache.jsonl"
    async with await make_judge(JudgeConfig()) as inner:
        await CachingJudge(inner, JudgeCache(path)).score([ITEM])
        again = await CachingJudge(inner, JudgeCache(path)).score([ITEM])
    assert len(fake.bodies) == 1
    assert [r.score for r in again] == [2, 2, 2]
    stored = [json.loads(ln) for ln in path.read_text().splitlines()]
    assert {s["cache_key"] for s in stored} == {r.cache_key for r in again}


# ------------------------------------------------------------------ acceptance
def _call(latency: float, outcome: str = "ok") -> JudgeCall:
    return JudgeCall("ollama", "m@1", "judge_qa@1", latency, "d", outcome)  # type: ignore[arg-type]


def test_acceptance_passes_a_fast_valid_judge_and_names_each_failure() -> None:
    good = acceptance([_call(5.0)] * 100)
    assert good.passed
    assert good.valid_rate == 1.0
    assert good.latency_max_s == 5.0
    slow = acceptance([_call(5.0)] * 99 + [_call(61.0)])
    assert not slow.passed
    assert "slowest item 61.0s" in slow.reasons[0]
    sloppy = acceptance([_call(1.0)] * 98 + [_call(1.0, "invalid_json")] * 2)
    assert not sloppy.passed
    assert sloppy.valid_rate == pytest.approx(0.98)
    assert acceptance([_call(1.0)] * 99 + [_call(1.0, "wrong_nonce")]).passed  # exactly 99%
    assert not acceptance([]).passed


@given(
    st.lists(
        st.tuples(
            st.floats(0, 120, allow_nan=False),
            st.sampled_from(["ok", "partial", "invalid_json", "wrong_nonce", "invalid_schema"]),
        ),
        min_size=1,
        max_size=200,
    )
)
def test_acceptance_is_exactly_its_two_rules(calls: list[tuple[float, str]]) -> None:
    result = acceptance([_call(lat, out) for lat, out in calls])
    valid = sum(out in ("ok", "partial") for _, out in calls) / len(calls)
    expected = max(lat for lat, _ in calls) <= 60 and valid >= 0.99
    assert result.passed is expected
    assert result.valid_rate == pytest.approx(valid)


def test_a_torn_cache_tail_does_not_brick_the_cache(tmp_path: Path) -> None:
    path = tmp_path / "judge-cache.jsonl"
    results = asyncio.run(CountingJudge(KEY).score([ITEM]))
    cache = JudgeCache(path)
    cache.put(results[:1])
    with path.open("ab") as handle:
        handle.write(results[1].model_dump_json().encode()[:40])  # a crashed writer
    cache.put(results[2:])
    fresh = JudgeCache(path)  # a restart reads the whole file
    assert len(fresh) == 2
    assert fresh.get(results[2].cache_key) == results[2]
    assert len(cache) == 2
