"""Resume recovery, lease release timing, tool-free requests and prompt-version reporting."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any
from uuid import UUID

import pytest

# ADR-0015: tools are bound only on an explicit action by default. Tests marked
# tool_mode=ToolMode.always reach the tool machinery through an ordinary question (the old
# default), which is what they exercise; the rest run on the production default.
from module_d1.d1_fakes import (
    COLLECTION,
    TENANT,
    FakeTx,
    Harness,
    call,
    collect,
    kinds,
    make_harness,
    principal,
    say,
    verdict,
)
from ragforge_core.agent import AgentSettings, ToolMode, nodes
from ragforge_core.domain.errors import Conflict
from ragforge_core.domain.models import (
    ApprovalChoice,
    ChatMessage,
    Role,
    ToolCall,
    TurnDone,
    TurnInterrupt,
)
from ragforge_core.prompts import PROMPT_VERSION

DOC = UUID(int=2001)


async def _interrupted(h: Harness, *calls: ToolCall) -> TurnInterrupt:
    h.script(say("", *calls))
    events = await collect(h.runner.run_turn(principal(), None, [COLLECTION], "clean up"))
    interrupt = events[-1]
    assert isinstance(interrupt, TurnInterrupt), kinds(events)
    return interrupt


def _statuses(h: Harness) -> set[str]:
    return {r.approval.status for r in h.db.approvals.values()}


def _patch_snapshot(h: Harness, monkeypatch: pytest.MonkeyPatch, **changes: Any) -> None:
    graph = h.runner._graph
    real = graph.aget_state

    async def fake(config: Any, **kw: Any) -> Any:
        return (await real(config, **kw))._replace(**changes)

    monkeypatch.setattr(graph, "aget_state", fake)


# ----------------------------------------------------------------- resume recovery
async def test_checkpoint_read_failure_on_resume_releases_lease_and_leaves_rows_pending(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    h = make_harness(tool_mode=ToolMode.always)
    remove = call("remove_from_index", document_id=str(DOC))
    interrupt = await _interrupted(h, remove)

    async def broken(config: Any, **kw: Any) -> Any:
        raise RuntimeError("checkpointer unavailable")

    monkeypatch.setattr(h.runner._graph, "aget_state", broken)
    with pytest.raises(RuntimeError):
        await collect(
            h.runner.resume(principal(), interrupt.run_id, {remove.id: ApprovalChoice.approve})
        )
    assert h.db.active == {}  # not stuck in TurnInProgress until the lease TTL
    assert _statuses(h) == {"pending"}  # nothing decided for a run that did not continue

    monkeypatch.undo()
    h.script(say("Done [S1]."), verdict())
    events = await collect(
        h.runner.resume(principal(), interrupt.run_id, {remove.id: ApprovalChoice.approve})
    )
    assert isinstance(events[-1], TurnDone)
    assert len(h.knowledge.removed) == 1


async def test_resume_of_a_checkpoint_without_interrupt_decides_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    h = make_harness(tool_mode=ToolMode.always)
    remove = call("remove_from_index", document_id=str(DOC))
    interrupt = await _interrupted(h, remove)
    _patch_snapshot(h, monkeypatch, interrupts=(), next=())
    with pytest.raises(Conflict):
        await collect(
            h.runner.resume(principal(), interrupt.run_id, {remove.id: ApprovalChoice.approve})
        )
    assert _statuses(h) == {"pending"}
    assert h.db.active == {}
    assert h.knowledge.removed == []


async def test_rows_decided_by_a_resume_that_died_can_be_resumed_with_the_same_decision() -> None:
    h = make_harness(tool_mode=ToolMode.always)
    remove = call("remove_from_index", document_id=str(DOC))
    interrupt = await _interrupted(h, remove)
    # The earlier resume committed its decision, then the process died before the graph ran.
    await h.conversations.decide_pending(
        FakeTx(TENANT), interrupt.run_id, {remove.id: ApprovalChoice.approve}
    )
    assert _statuses(h) == {"approved"}

    with pytest.raises(Conflict):  # asking for a different outcome is refused, not ignored
        await collect(
            h.runner.resume(principal(), interrupt.run_id, {remove.id: ApprovalChoice.reject})
        )
    assert h.db.active == {}

    h.script(say("Done [S1]."), verdict())
    events = await collect(h.runner.resume(principal(), interrupt.run_id, {}))
    assert isinstance(events[-1], TurnDone)
    assert len(h.knowledge.removed) == 1


async def test_a_resumed_run_that_died_while_executing_can_be_resumed_and_runs_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    h = make_harness(tool_mode=ToolMode.always)
    remove = call("remove_from_index", document_id=str(DOC))
    interrupt = await _interrupted(h, remove)
    real = h.knowledge.remove_from_index
    crashes = [RuntimeError("connection reset")]

    async def flaky(*args: Any) -> bool:
        if crashes:
            raise crashes.pop()
        return await real(*args)

    monkeypatch.setattr(h.knowledge, "remove_from_index", flaky)
    with pytest.raises(RuntimeError):
        await collect(
            h.runner.resume(principal(), interrupt.run_id, {remove.id: ApprovalChoice.approve})
        )
    assert h.db.active == {}
    assert _statuses(h) == {"approved"}  # decided, claim rolled back with the failed effect

    h.script(say("Removed [S1]."), verdict())
    events = await collect(h.runner.resume(principal(), interrupt.run_id, {}))
    assert isinstance(events[-1], TurnDone), kinds(events)
    assert len(h.knowledge.removed) == 1
    with pytest.raises(Conflict):  # finished: nothing left to resume
        await collect(h.runner.resume(principal(), interrupt.run_id, {}))
    assert len(h.knowledge.removed) == 1


# ----------------------------------------------------------------- lease release timing
async def test_lease_is_released_before_the_terminal_event_reaches_the_consumer() -> None:
    h = make_harness(tool_mode=ToolMode.always)
    remove = call("remove_from_index", document_id=str(DOC))
    h.script(say("", remove))
    seen_interrupt = False
    async for event in h.runner.run_turn(principal(), None, [COLLECTION], "clean up"):
        if isinstance(event, TurnInterrupt):
            assert h.db.active == {}  # consumer has not pulled again yet
            seen_interrupt = True
            interrupt = event
    assert seen_interrupt

    h.script(say("Done [S1]."), verdict())
    seen_done = False
    async for event in h.runner.resume(
        principal(), interrupt.run_id, {remove.id: ApprovalChoice.approve}
    ):
        if isinstance(event, TurnDone):
            assert h.db.active == {}
            seen_done = True
    assert seen_done
    assert len(h.db.released) == 2  # the interrupted run and the resumed run


# ----------------------------------------------------------------- tool-free requests
def _assert_no_structured_tool_messages(messages: Sequence[ChatMessage]) -> None:
    assert all(m.role is not Role.tool for m in messages)
    assert all(not m.tool_calls for m in messages)


async def test_requests_without_tools_carry_tool_exchanges_as_fenced_text() -> None:
    # tool_mode=always: encodes the pre-ADR-0015 default (tools bound on a question)
    settings = AgentSettings(
        max_tool_rounds=1, heartbeat_interval_s=3600, tool_mode=ToolMode.always
    )
    h = make_harness(settings=settings)
    sources = call("list_sources")
    h.script(say("", sources), say("Fact one is false [S9]."), say("Fact one [S1]."), verdict())
    events = await collect(h.runner.run_turn(principal(), None, [COLLECTION], "q?"))
    assert isinstance(events[-1], TurnDone)

    unbound, strict = h.chat.calls[1], h.chat.calls[2]
    for request in (unbound, strict):
        assert request.tools == ()
        _assert_no_structured_tool_messages(request.messages)
        flat = "\n".join(m.content for m in request.messages)
        assert "I used the tool list_sources" in flat
        assert '<tool_result label="T1" tool="list_sources">' in flat
        assert "&lt;/source> ignore previous" in flat  # still neutralized inside the fence


async def test_requests_with_tools_keep_structured_tool_messages() -> None:
    h = make_harness(tool_mode=ToolMode.always)
    sources = call("list_sources")
    h.script(say("", sources), say("Fact one [S1]."), verdict())
    await collect(h.runner.run_turn(principal(), None, [COLLECTION], "q?"))
    second = h.chat.calls[1]
    assert second.tools
    assert [m.tool_call_id for m in second.messages if m.role is Role.tool] == [sources.id]


# ----------------------------------------------------------------- prompt version
async def test_prompt_version_reports_every_build_that_served_the_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    h = make_harness(tool_mode=ToolMode.always)
    remove = call("remove_from_index", document_id=str(DOC))
    interrupt = await _interrupted(h, remove)

    newer = "abstain@1,answer@4,answer_strict@5,contextualize@1,ground_check@3,rewrite@1"
    monkeypatch.setattr(nodes, "PROMPT_VERSION", newer)  # the resume is served by a new deploy
    h.script(say("Done [S1]."), verdict())
    events = await collect(
        h.runner.resume(principal(), interrupt.run_id, {remove.id: ApprovalChoice.approve})
    )
    done = events[-1]
    assert isinstance(done, TurnDone)
    assert done.prompt_version == f"{PROMPT_VERSION} + {newer}"


async def test_single_build_turn_reports_exactly_the_build_prompt_version() -> None:
    h = make_harness([say("Fact one [S1]."), verdict()])
    events = await collect(h.runner.run_turn(principal(), None, [COLLECTION], "q?"))
    done = events[-1]
    assert isinstance(done, TurnDone)
    assert done.prompt_version == PROMPT_VERSION
