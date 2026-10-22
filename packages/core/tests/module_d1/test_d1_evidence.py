"""What the verifiers are given and accept: tool results are evidence (cited as [T<n>]), only
hits the model was actually shown are citable, and a follow-up cannot change a conversation's
scope."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from uuid import UUID

import pytest

from module_d1.d1_fakes import (
    COLLECTION,
    COLLECTION_B,
    Judge,
    call,
    collect,
    kinds,
    make_harness,
    make_hit,
    of_type,
    principal,
    say,
    text_of,
    verdict,
)
from ragforge_core.agent import AgentSettings, ToolMode
from ragforge_core.domain.errors import ValidationFailed
from ragforge_core.domain.models import (
    ApprovalChoice,
    TurnCitation,
    TurnDone,
    TurnEvent,
    TurnInterrupt,
    TurnReplace,
    TurnVerification,
    Verification,
)

# tool_mode=always: these tests reach tools through ordinary questions, the pre-ADR-0015
# default (ADR-0015 binds tools only on an explicit action).
STRICT = AgentSettings(strict_citations=True, heartbeat_interval_s=3600, tool_mode=ToolMode.always)


def _judge_requires(*needles: str, absent: str = "") -> Judge:
    """A verifier that says `supported` only if its input holds every needle (and not `absent`):
    a real verifier rejects a claim whose evidence it was never shown."""

    def build(evidence: str) -> dict[str, object]:
        ok = all(n in evidence for n in needles) and not (absent and absent in evidence)
        return {"verdict": "supported" if ok else "unsupported", "unsupported_claims": []}

    return Judge(build)


def _status(events: Sequence[TurnEvent]) -> Verification:
    (ver,) = of_type(events, TurnVerification)
    return ver.status


# ----------------------------------------------------------------- tool results as evidence
async def test_answer_from_a_tool_result_is_verified_in_strict_mode() -> None:
    h = make_harness(settings=STRICT)
    status = call("sync_status")
    h.script(
        say("", status),
        say("Your source is active [T1]."),
        _judge_requires('<tool_result label="T1" tool="sync_status">', '"status": "active"'),
    )
    events = await collect(h.runner.run_turn(principal(), None, [COLLECTION], "is it syncing?"))

    assert h.chat.kinds() == ["stream", "stream", "json"]  # no strict regeneration
    assert _status(events) is Verification.verified
    assert not of_type(events, TurnReplace)
    assert not of_type(events, TurnCitation)  # [T1] is evidence, not a page link
    assert isinstance(events[-1], TurnDone)


async def test_answer_reporting_an_approved_destructive_action_is_verified() -> None:
    """The reviewer's failure: 'I removed that page' must not be judged unsupported (and then
    abstained) after the removal really ran."""
    h = make_harness(settings=STRICT)
    remove = call("remove_from_index", document_id=str(UUID(int=2001)))
    h.script(say("", remove))
    first = await collect(h.runner.run_turn(principal(), None, [COLLECTION], "drop page 1"))
    interrupt = first[-1]
    assert isinstance(interrupt, TurnInterrupt)

    h.script(
        say("I removed that page from the index [T1]."),
        _judge_requires('<tool_result label="T1" tool="remove_from_index">', "Page removed"),
    )
    events = await collect(
        h.runner.resume(principal(), interrupt.run_id, {remove.id: ApprovalChoice.approve})
    )
    assert len(h.knowledge.removed) == 1
    assert _status(events) is Verification.verified
    assert text_of(events) == "I removed that page from the index [T1]."
    assert isinstance(events[-1], TurnDone)


async def test_citing_a_tool_result_that_does_not_exist_is_rejected() -> None:
    h = make_harness([say("Your source is active [T1]."), say("Fact one [S1]."), verdict()])
    events = await collect(h.runner.run_turn(principal(), None, [COLLECTION], "q?"))
    assert h.chat.kinds() == ["stream", "stream", "json"]  # invalid [T1] -> strict regeneration
    assert of_type(events, TurnReplace)[0].text == "Fact one [S1]."


# ----------------------------------------------------------------- only what was shown
async def test_weak_pre_rewrite_hits_are_neither_citable_nor_shown_to_the_verifier() -> None:
    h = make_harness(
        [
            say("better query"),
            say("Weak claim [S1]."),  # S1 = the weak first-pass hit, never in the prompt
            say("Fact three [S2]."),
            _judge_requires("Fact number 3", absent="Fact number 1"),
        ],
        results=[[make_hit(1, 0.01)], [make_hit(3, 0.95)]],
    )
    events = await collect(h.runner.run_turn(principal(), None, [COLLECTION], "q?"))

    assert h.chat.kinds() == ["stream", "stream", "stream", "json"]
    assert "Fact number 1" not in h.chat.calls[1].messages[-1].content  # generate never saw it
    assert of_type(events, TurnReplace)[0].text == "Fact three [S2]."
    assert [c.hit.anchor for c in of_type(events, TurnCitation)] == ["block-3"]
    assert _status(events) is Verification.verified


# ----------------------------------------------------------------- conversation scope
async def test_follow_up_may_narrow_but_never_widen_or_switch_the_collections() -> None:
    h = make_harness([say("Fact one [S1]."), verdict()])
    first = await collect(h.runner.run_turn(principal(), None, [COLLECTION], "q?"))
    cid = first[-1].conversation_id  # type: ignore[union-attr]
    turns = h.chat.kinds()

    for scope in ([COLLECTION_B], [COLLECTION, COLLECTION_B]):
        with pytest.raises(ValidationFailed):
            await collect(h.runner.run_turn(principal(), cid, scope, "and more?"))
    assert h.chat.kinds() == turns  # nothing ran
    assert h.db.active == {}  # and no lease was left behind

    h.script(say("Fact one again?"), say("Fact one [S1]."), verdict())
    await collect(h.runner.run_turn(principal(), cid, [], "again?"))
    assert h.retriever.calls[-1][2] == (COLLECTION,)  # empty request = stored scope


async def test_follow_up_narrowing_searches_only_the_requested_subset() -> None:
    h = make_harness([say("Fact one [S1]."), verdict()])
    first = await collect(h.runner.run_turn(principal(), None, [COLLECTION, COLLECTION_B], "q?"))
    cid = first[-1].conversation_id  # type: ignore[union-attr]
    h.script(say("Fact one again?"), say("Fact one [S1]."), verdict())
    events = await collect(h.runner.run_turn(principal(), cid, [COLLECTION_B], "again?"))
    assert isinstance(events[-1], TurnDone), kinds(events)
    assert h.retriever.calls[-1][2] == (COLLECTION_B,)


# ----------------------------------------------------------------- abstain markdown
async def test_closest_pages_escape_titles_and_encode_urls() -> None:
    hit = replace(
        make_hit(1, 0.01, title="Evil](javascript:x) [y\nz"),
        external_url="https://notion.so/a b)c",
    )
    h = make_harness([say("alt")], results=[[hit]])
    events = await collect(h.runner.run_turn(principal(), None, [COLLECTION], "q?"))
    answer = text_of(events)
    assert r"- [Evil\]\(javascript:x\) \[y z](https://notion.so/a%20b%29c)" in answer
