"""Run lifecycle: cancellation on consumer aclose, lease loss, and history commit discipline."""

from __future__ import annotations

import asyncio
from uuid import UUID

import pytest

from module_d1.d1_fakes import (
    COLLECTION,
    Harness,
    Stream,
    collect,
    make_harness,
    principal,
    say,
    verdict,
)
from ragforge_core.agent import AgentSettings
from ragforge_core.domain.errors import LeaseLost
from ragforge_core.domain.models import TextDelta, TurnDone, TurnEvent, TurnText


async def _history(h: Harness) -> list[object]:
    (cid,) = h.db.conversations
    snapshot = await h.runner._graph.aget_state({"configurable": {"thread_id": str(cid)}})
    return list(snapshot.values.get("messages", []))


async def test_consumer_aclose_cancels_the_model_stream_releases_lease_and_discards_draft() -> None:
    h = make_harness([Stream([TextDelta("Partial ans")], hang_after=True)])
    turn = h.runner.run_turn(principal(), None, [COLLECTION], "q?")
    async for event in turn:
        if isinstance(event, TurnText):
            break
    await asyncio.wait_for(h.chat.started.wait(), 1)
    await asyncio.wait_for(turn.aclose(), 2)  # type: ignore[attr-defined]

    assert h.chat.closed == 1  # ChatModel contract: aclose() reached the port's iterator
    assert h.db.active == {}  # lease released
    assert await _history(h) == []  # the partial draft never reached committed history


async def test_lost_lease_stops_the_run_at_the_next_node_boundary() -> None:
    h = make_harness([say("Fact one [S1]."), verdict()])
    seen: list[TurnEvent] = []

    async def consume() -> None:
        async for event in h.runner.run_turn(principal(), None, [COLLECTION], "q?"):
            seen.append(event)
            if isinstance(event, TurnText):
                h.db.lose_lease = True  # e.g. another pod took over after expiry

    with pytest.raises(LeaseLost):
        await consume()
    assert not any(isinstance(e, TurnDone) for e in seen)
    assert h.chat.kinds() == ["stream"]  # the ground check never ran
    assert await _history(h) == []


async def test_background_keepalive_heartbeats_during_a_long_node() -> None:
    h = make_harness(
        [Stream([TextDelta("slow")], hang_after=True)],
        settings=AgentSettings(heartbeat_interval_s=0.01),
    )
    turn = h.runner.run_turn(principal(), None, [COLLECTION], "q?")
    async for event in turn:
        if isinstance(event, TurnText):
            break
    before = h.runs.heartbeats
    await asyncio.sleep(0.1)
    assert h.runs.heartbeats > before + 2  # beating while the model streams
    await turn.aclose()  # type: ignore[attr-defined]
    after = h.runs.heartbeats
    await asyncio.sleep(0.05)
    assert h.runs.heartbeats == after  # and stopped with the run


async def test_heartbeat_is_checked_before_every_impure_node_only() -> None:
    h = make_harness([say("Fact one [S1]."), verdict()])
    await collect(h.runner.run_turn(principal(), None, [COLLECTION], "q?"))
    # load_memory, retrieve, generate, ground_check, finalize; the pure nodes grade,
    # verify_citations and decide skip the round trip (nothing they do needs the lease)
    assert h.runs.heartbeats == 5


async def test_a_failed_background_heartbeat_neither_loses_turn_done_nor_the_release() -> None:
    """One transient heartbeat error inside the keepalive task must not surface as the turn's
    error, nor leave the conversation locked until the lease TTL."""
    h = make_harness(
        [say("Fact one [S1]."), verdict()],
        settings=AgentSettings(heartbeat_interval_s=0.001),
    )
    beat = h.runs.heartbeat
    failures = 0

    def in_keepalive() -> bool:
        task = asyncio.current_task()
        return task is not None and task.get_coro().__qualname__ == "RunLease.keepalive"  # type: ignore[union-attr]

    async def flaky(tenant_id: UUID, conversation_id: UUID, run_id: UUID) -> bool:
        nonlocal failures
        if in_keepalive() and failures == 0:
            failures += 1
            raise ConnectionError("connection reset")
        return await beat(tenant_id, conversation_id, run_id)

    h.runs.heartbeat = flaky  # type: ignore[method-assign]
    turn = h.runner.run_turn(principal(), None, [COLLECTION], "q?")
    events: list[TurnEvent] = []
    async for event in turn:
        events.append(event)
        if isinstance(event, TurnText):
            await asyncio.sleep(0.02)  # let the keepalive fire (and fail) mid-turn
    assert failures == 1
    assert isinstance(events[-1], TurnDone)
    assert h.db.active == {}  # released


async def test_a_failed_release_does_not_replace_turn_done() -> None:
    h = make_harness([say("Fact one [S1]."), verdict()])

    async def broken(tenant_id: UUID, conversation_id: UUID, run_id: UUID) -> None:
        raise ConnectionError("connection reset")

    h.runs.release = broken  # type: ignore[method-assign]
    events = await collect(h.runner.run_turn(principal(), None, [COLLECTION], "q?"))
    assert isinstance(events[-1], TurnDone)  # the lease will expire on its TTL instead


async def test_history_holds_exactly_user_and_final_answer_per_turn() -> None:
    h = make_harness([say("Fact one is false [S9]."), say("Fact one is true [S1]."), verdict()])
    await collect(h.runner.run_turn(principal(), None, [COLLECTION], "fact one?"))
    history = await _history(h)
    assert [m.content for m in history] == ["fact one?", "Fact one is true [S1]."]  # type: ignore[attr-defined]
