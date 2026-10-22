"""Every stream is self-contained for E's link allow-list.

Module E builds a NEW OutputSanitizer per stream and allows only the URLs of the TurnSources it
has seen on THAT stream. A resumed turn produces text (the continued answer, a strict
regeneration's TurnReplace, the abstain page list) and citations that may point at hits
retrieved in the EARLIER, interrupted stream, so the resumed stream must open with TurnSources
covering every hit the turn holds, before any graph event.

`_allow_list_violations` models E exactly: an event may only reference URLs/hits that some
earlier TurnSources on the same stream announced.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any
from uuid import UUID

import pytest

# ADR-0015: tools are bound only on an explicit action by default. Tests marked
# tool_mode=ToolMode.always reach the tool machinery through an ordinary question (the old
# default), which is what they exercise; the rest run on the production default.
from module_d1.d1_fakes import (
    COLLECTION,
    Harness,
    call,
    collect,
    kinds,
    make_harness,
    make_hit,
    of_type,
    principal,
    say,
    verdict,
)
from ragforge_core.agent import ToolMode
from ragforge_core.domain.models import (
    ApprovalChoice,
    RankedChunk,
    TurnCitation,
    TurnDone,
    TurnEvent,
    TurnInterrupt,
    TurnReplace,
    TurnSources,
    TurnText,
    TurnVerification,
    Verification,
)

DOC = UUID(int=2001)
_URL = re.compile(r"https?://[^\s)\]>]+")
GRAPH_HITS = (make_hit(1), make_hit(2))  # the graph retrieval of the interrupted stream
TOOL_HIT = make_hit(7, text="Fact seven is from a follow-up search.")  # a tool-driven retrieval


def _allow_list_violations(events: Sequence[TurnEvent]) -> list[str]:
    """What a fresh per-stream sanitizer would defang: references to hits or URLs that no
    EARLIER TurnSources on this stream announced. Streamed TurnText is one growing document (a
    URL can span deltas), so a URL is judged by the allow-list in force when its last character
    arrived; a TurnReplace is a whole new document."""
    allowed_urls: set[str] = set()
    allowed_chunks: set[UUID] = set()
    violations: list[str] = []
    streamed = ""
    allowed_at: list[frozenset[str]] = []  # per character of `streamed`
    for event in events:
        if isinstance(event, TurnSources):
            allowed_chunks |= {h.chunk_id for h in event.hits}
            allowed_urls |= {h.external_url for h in event.hits if h.external_url}
        elif isinstance(event, TurnText):
            streamed += event.text
            allowed_at += [frozenset(allowed_urls)] * len(event.text)
        elif isinstance(event, TurnReplace):
            violations += [u for u in _URL.findall(event.text) if u not in allowed_urls]
        elif isinstance(event, TurnCitation):
            if event.hit.chunk_id not in allowed_chunks:
                violations.append(f"citation {event.hit.anchor}")
            if event.hit.external_url and event.hit.external_url not in allowed_urls:
                violations.append(event.hit.external_url)
    for match in _URL.finditer(streamed):
        if match.group() not in allowed_at[match.end() - 1]:
            violations.append(match.group())
    return violations


def _chunks(hits: Sequence[RankedChunk]) -> set[UUID]:
    return {h.chunk_id for h in hits}


async def _interrupted_after_tool_search(h: Harness) -> tuple[Any, TurnInterrupt, list[TurnEvent]]:
    """Turn 1: graph retrieval (S1, S2), then a tool search (S3 = hit 7), then a destructive
    call. All three hits are announced on THIS stream only."""
    h.retriever.by_query["fact seven"] = [TOOL_HIT]
    remove = call("remove_from_index", document_id=str(DOC))
    h.script(say("", call("search_notes", query="fact seven")), say("", remove))
    events = await collect(h.runner.run_turn(principal(), None, [COLLECTION], "clean up"))
    interrupt = events[-1]
    assert isinstance(interrupt, TurnInterrupt), kinds(events)
    announced = {h.chunk_id for e in of_type(events, TurnSources) for h in e.hits}
    assert announced == _chunks((*GRAPH_HITS, TOOL_HIT))
    return remove, interrupt, events


def _assert_opens_with_every_hit(events: Sequence[TurnEvent]) -> None:
    first = events[0]
    assert isinstance(first, TurnSources), kinds(events)
    assert _chunks(first.hits) == _chunks((*GRAPH_HITS, TOOL_HIT))
    assert first.hits == (*GRAPH_HITS, TOOL_HIT)  # full domain objects, label order kept
    assert _allow_list_violations(events) == []


async def test_resumed_stream_opens_with_sources_for_every_hit_before_any_text() -> None:
    h = make_harness(tool_mode=ToolMode.always)
    remove, interrupt, _ = await _interrupted_after_tool_search(h)

    answer = "Removed it; it cited [S1] and [S3], see https://notion.so/page-7 and [S2]."
    h.script(say(answer), verdict())
    events = await collect(
        h.runner.resume(principal(), interrupt.run_id, {remove.id: ApprovalChoice.approve})
    )

    _assert_opens_with_every_hit(events)
    assert isinstance(events[-1], TurnDone), kinds(events)
    # the resumed stream really does reference hits it never retrieved itself
    assert [c.hit.anchor for c in of_type(events, TurnCitation)] == [
        "block-1",
        "block-7",
        "block-2",
    ]
    assert h.retriever.calls[-1][1] == "fact seven"  # no retrieval ran on the resumed stream
    assert len(of_type(events, TurnSources)) == 1


async def test_strict_regeneration_after_resume_replaces_text_only_after_sources() -> None:
    """The regenerate path: TurnReplace on a resumed stream links to pre-interrupt hits."""
    h = make_harness(tool_mode=ToolMode.always)
    remove, interrupt, _ = await _interrupted_after_tool_search(h)

    regenerated = "Fact seven holds (https://notion.so/page-7) [S3]."
    h.script(
        say("Removed it, and fact one holds [S1]."),
        verdict("unsupported", "fact one holds"),
        say(regenerated),
        verdict(),
    )
    events = await collect(
        h.runner.resume(principal(), interrupt.run_id, {remove.id: ApprovalChoice.approve})
    )

    _assert_opens_with_every_hit(events)
    (replace,) = of_type(events, TurnReplace)
    assert replace.text == regenerated
    assert [c.hit.anchor for c in of_type(events, TurnCitation)] == ["block-7"]


async def test_abstain_after_failed_regeneration_on_resume_links_only_announced_pages() -> None:
    """The abstain page list links every hit of the turn (incl. the tool-driven one)."""
    h = make_harness(tool_mode=ToolMode.always)
    remove, interrupt, _ = await _interrupted_after_tool_search(h)

    h.script(
        say("Fact one holds [S1]."),
        verdict("unsupported", "fact one holds"),
        say("Fact two holds [S2]."),
        verdict("unsupported", "fact two holds"),
    )
    events = await collect(
        h.runner.resume(principal(), interrupt.run_id, {remove.id: ApprovalChoice.approve})
    )

    _assert_opens_with_every_hit(events)
    (replace,) = of_type(events, TurnReplace)
    assert set(_URL.findall(replace.text)) == {
        "https://notion.so/page-1",
        "https://notion.so/page-2",
        "https://notion.so/page-7",
    }
    assert of_type(events, TurnVerification)[0].status is Verification.unsupported


async def test_continuing_a_checkpoint_past_the_interrupt_also_replays_sources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The other continuation path: a resumed run that died in `execute` is continued from the
    checkpoint (payload None), on yet another stream with yet another sanitizer."""
    h = make_harness(tool_mode=ToolMode.always)
    remove, interrupt, _ = await _interrupted_after_tool_search(h)
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

    h.script(say("Removed; see https://notion.so/page-1 [S1] and [S3]."), verdict())
    events = await collect(h.runner.resume(principal(), interrupt.run_id, {}))

    _assert_opens_with_every_hit(events)
    assert isinstance(events[-1], TurnDone), kinds(events)
    assert [c.hit.anchor for c in of_type(events, TurnCitation)] == ["block-1", "block-7"]
    assert len(h.knowledge.removed) == 1


async def test_a_new_turn_never_replays_the_previous_turns_hits() -> None:
    """A follow-up resets `hits`: announcing the old turn's URLs would widen E's allow-list to
    pages this answer cannot cite."""
    h = make_harness(results=[list(GRAPH_HITS), [make_hit(5)]])
    h.script(say("Fact one [S1]."), verdict())
    first = await collect(h.runner.run_turn(principal(), None, [COLLECTION], "q1?"))
    assert isinstance(first[-1], TurnDone)
    cid = first[-1].conversation_id

    h.script(say("standalone q2?"), say("Fact five [S1]."), verdict())
    events = await collect(h.runner.run_turn(principal(), cid, [COLLECTION], "q2?"))

    announced = {hit.chunk_id for e in of_type(events, TurnSources) for hit in e.hits}
    assert announced == _chunks([make_hit(5)])
    assert _allow_list_violations(events) == []
