"""Turn-level behavior of the LangGraph AgentRunner (module D1), driven by a scripted ChatModel."""

from __future__ import annotations

from uuid import UUID

import pytest

from module_d1.d1_fakes import (
    COLLECTION,
    COLLECTION_B,
    OTHER_USER,
    TENANT,
    Json,
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
from ragforge_core.agent import AgentSettings
from ragforge_core.agent.settings import DEFAULT_TAU
from ragforge_core.domain.errors import Forbidden, NotFound, TurnInProgress, ValidationFailed
from ragforge_core.domain.models import (
    ChatTurnUsage,
    TurnCitation,
    TurnDone,
    TurnReplace,
    TurnSources,
    TurnText,
    TurnVerification,
    Verification,
)
from ragforge_core.prompts import PROMPT_VERSION


async def test_event_order_sources_before_text_then_citations_verification_done() -> None:
    h = make_harness([say("Fact one holds [S1]."), verdict()])
    events = await collect(h.runner.run_turn(principal(), None, [COLLECTION], "what is fact one?"))

    assert kinds(events)[0] == "TurnSources"
    first_text = next(i for i, e in enumerate(events) if isinstance(e, TurnText))
    assert all(not isinstance(e, TurnText) for e in events[:1])
    assert first_text > 0
    assert kinds(events)[-3:] == ["TurnCitation", "TurnVerification", "TurnDone"]
    assert text_of(events) == "Fact one holds [S1]."  # RAW, streamed in pieces
    assert len(of_type(events, TurnText)) == 2
    (citation,) = of_type(events, TurnCitation)
    assert citation.hit.anchor == "block-1"
    (ver,) = of_type(events, TurnVerification)
    assert ver.status is Verification.verified
    assert ver.unsupported_claims == ()
    assert h.chat.kinds() == ["stream", "json"]


async def test_done_reports_every_model_call_usage_and_prompt_version() -> None:
    h = make_harness([say("Fact one holds [S1].", prompt=11, completion=4), verdict()])
    events = await collect(h.runner.run_turn(principal(), None, [COLLECTION], "fact one?"))
    done = events[-1]
    assert isinstance(done, TurnDone)
    assert done.prompt_version == PROMPT_VERSION
    # bumped by ADR-0015 (small-model citation prompt; round 5: no closing pleasantries;
    # coverage redesign: a label covers only its own sentence, so yes/no goes in that sentence)
    assert "answer@11" in done.prompt_version
    # bumped by ADR-0015 (the verifier checks each [S<n>] pairing, incl. inferred ones)
    assert "ground_check@3" in done.prompt_version
    assert done.usage == (
        ChatTurnUsage("fake", "fake-chat", 11, 4),
        ChatTurnUsage("fake", "fake-judge", 7, 3),
    )
    assert done.conversation_id in h.db.conversations


async def test_text_is_raw_never_sanitized() -> None:
    # Module AG (blind review round 2, P2): a bare label right after HTML no longer cites the
    # tag, so the label now follows words; the test is about raw text, not about coverage.
    raw = "See ![x](https://evil.example/a.png) <b>hi</b> partitions [S1]."
    h = make_harness([say(raw), verdict()])
    events = await collect(h.runner.run_turn(principal(), None, [COLLECTION], "q?"))
    assert text_of(events) == raw


async def test_abstain_after_one_rewrite_lists_closest_pages_without_generating() -> None:
    weak = [make_hit(1, 0.05), make_hit(2, 0.1, document=UUID(int=2001))]
    h = make_harness([say("alternative query")], results=[weak])
    events = await collect(h.runner.run_turn(principal(), None, [COLLECTION], "unknown topic?"))

    assert h.chat.kinds() == ["stream"]  # the rewrite only: no generate, no ground check
    assert [q for _, q, _ in h.retriever.calls] == ["unknown topic?", "alternative query"]
    assert kinds(events) == [
        "TurnSources",
        "TurnSources",
        "TurnText",
        "TurnVerification",
        "TurnDone",
    ]
    answer = text_of(events)
    assert answer.startswith("I couldn't find this in your notes.")
    assert "[Page 2](https://notion.so/page-2)" in answer  # closest page first (higher score)
    (ver,) = of_type(events, TurnVerification)
    assert ver.status is Verification.verified  # server-authored text, no model claims


@pytest.mark.parametrize("max_rewrites", [0, 1, 2])
async def test_rewrite_loop_is_bounded(max_rewrites: int) -> None:
    script = [say(f"rewrite {i}") for i in range(max_rewrites)]
    h = make_harness(
        script,
        results=[[make_hit(1, 0.0)]],
        settings=AgentSettings(max_rewrites=max_rewrites, heartbeat_interval_s=3600),
    )
    events = await collect(h.runner.run_turn(principal(), None, [COLLECTION], "q?"))
    assert len(h.retriever.calls) == max_rewrites + 1
    assert h.chat.kinds() == ["stream"] * max_rewrites
    assert text_of(events).startswith("I couldn't find this")
    assert h.chat.script == []


async def test_rewrite_that_finds_good_hits_goes_on_to_generate() -> None:
    h = make_harness(
        [say("better query"), say("Fact three [S2]."), verdict()],
        results=[[make_hit(1, 0.01)], [make_hit(3, 0.95)]],
    )
    events = await collect(h.runner.run_turn(principal(), None, [COLLECTION], "q?"))
    (citation,) = of_type(events, TurnCitation)
    assert citation.hit.anchor == "block-3"  # label S2 = second hit of the turn
    assert of_type(events, TurnVerification)[0].status is Verification.verified


async def test_grade_uses_the_top_hits_collection_tau_with_config_default() -> None:
    # 0.5 passes the default tau but not collection B's tuned tau of 0.8.
    assert DEFAULT_TAU < 0.5
    h = make_harness(
        [],
        results=[[make_hit(1, 0.5, collection=COLLECTION_B)]],
        settings=AgentSettings(max_rewrites=0, heartbeat_interval_s=3600),
    )
    h.knowledge.taus[COLLECTION_B] = 0.8
    events = await collect(h.runner.run_turn(principal(), None, [COLLECTION_B], "q?"))
    assert text_of(events).startswith("I couldn't find this")

    h2 = make_harness(
        [say("Fact one [S1]."), verdict()], results=[[make_hit(1, 0.5, collection=COLLECTION)]]
    )
    events = await collect(h2.runner.run_turn(principal(), None, [COLLECTION], "q?"))
    assert of_type(events, TurnVerification)[0].status is Verification.verified


async def test_hits_without_rerank_score_never_pass_the_grade() -> None:
    h = make_harness(
        [],
        results=[[make_hit(1, None)]],
        settings=AgentSettings(max_rewrites=0, heartbeat_interval_s=3600),
    )
    events = await collect(h.runner.run_turn(principal(), None, [COLLECTION], "q?"))
    assert text_of(events).startswith("I couldn't find this")


async def test_invalid_citation_triggers_one_strict_regeneration_and_replace() -> None:
    h = make_harness([say("Fact one is false [S9]."), say("Fact one is true [S1]."), verdict()])
    events = await collect(h.runner.run_turn(principal(), None, [COLLECTION], "fact one?"))

    assert h.chat.kinds() == ["stream", "stream", "json"]  # no ground check on the bad draft
    strict = h.chat.calls[1]
    assert strict.tools == ()  # the regeneration is tool-free
    assert "Strict rules" in strict.system
    assert "Fact one is false [S9]." in strict.messages[-1].content  # told what was rejected
    assert text_of(events) == "Fact one is false [S9]."
    tail = kinds(events)[-4:]
    assert tail == ["TurnReplace", "TurnCitation", "TurnVerification", "TurnDone"]
    assert of_type(events, TurnReplace)[0].text == "Fact one is true [S1]."
    assert of_type(events, TurnVerification)[0].status is Verification.verified


async def test_ground_unsupported_regenerates_then_second_failure_abstains() -> None:
    first = "Fact one is purple [S1]."
    h = make_harness(
        [
            say(first),
            verdict("unsupported", first),
            say("Fact one is still purple [S1]."),
            verdict("unsupported", "Fact one is still purple [S1]."),
        ]
    )
    events = await collect(h.runner.run_turn(principal(), None, [COLLECTION], "fact one?"))
    assert h.chat.kinds() == ["stream", "json", "stream", "json"]
    (replace,) = of_type(events, TurnReplace)
    assert replace.text.startswith("I couldn't find this in your notes.")
    assert not of_type(events, TurnCitation)
    (ver,) = of_type(events, TurnVerification)
    assert ver.status is Verification.unsupported
    assert isinstance(events[-1], TurnDone)


async def test_strict_regeneration_requires_a_citation_per_sentence() -> None:
    h = make_harness(
        [
            say("Fact one is purple [S1]."),
            verdict("unsupported", "Fact one is purple [S1]."),
            say("Fact one is true [S1]. It is also great."),  # 2nd sentence uncited
        ]
    )
    events = await collect(h.runner.run_turn(principal(), None, [COLLECTION], "fact one?"))
    assert h.chat.kinds() == ["stream", "json", "stream"]  # strict check fails before grounding
    (ver,) = of_type(events, TurnVerification)
    assert ver.status is Verification.unsupported
    assert of_type(events, TurnReplace)[0].text.startswith("I couldn't find this")


async def test_ground_check_malformed_twice_is_partial() -> None:
    h = make_harness(
        [say("Fact one holds [S1]."), Json({"verdict": "maybe"}), Json(ValueError("not json"))]
    )
    events = await collect(h.runner.run_turn(principal(), None, [COLLECTION], "fact one?"))
    (ver,) = of_type(events, TurnVerification)
    assert ver.status is Verification.partial
    assert h.chat.kinds() == ["stream", "json", "json"]  # exactly one repair retry
    repair = h.chat.calls[2].messages
    assert "did not match the JSON schema" in repair[-1].content
    assert not of_type(events, TurnReplace)  # partial is final: no regeneration


async def test_ground_check_repair_retry_can_succeed() -> None:
    h = make_harness([say("Fact one holds [S1]."), Json(ValueError("garbage")), verdict()])
    events = await collect(h.runner.run_turn(principal(), None, [COLLECTION], "fact one?"))
    assert of_type(events, TurnVerification)[0].status is Verification.verified
    done = events[-1]
    assert isinstance(done, TurnDone)
    assert len(done.usage) == 2


async def test_partial_verdict_reports_only_verbatim_claims() -> None:
    answer = "Fact one holds [S1]. Fact two is blue [S2]."
    h = make_harness(
        [say(answer), verdict("partially_supported", "Fact two is blue [S2].", "paraphrase")]
    )
    events = await collect(h.runner.run_turn(principal(), None, [COLLECTION], "facts?"))
    (ver,) = of_type(events, TurnVerification)
    assert ver.status is Verification.partial
    assert ver.unsupported_claims == ("Fact two is blue [S2].",)


async def test_untrusted_verifier_reports_unverified_never_verified() -> None:
    h = make_harness(
        [say("Fact one holds [S1]."), verdict()],
        settings=AgentSettings(verifier_trusted=False, heartbeat_interval_s=3600),
    )
    events = await collect(h.runner.run_turn(principal(), None, [COLLECTION], "fact one?"))
    assert of_type(events, TurnVerification)[0].status is Verification.unverified


async def test_follow_up_contextualizes_and_history_is_committed_only_at_turn_end() -> None:
    h = make_harness([say("Fact one holds [S1]."), verdict()])
    first = await collect(h.runner.run_turn(principal(), None, [COLLECTION], "what is fact one?"))
    conversation = first[-1]
    assert isinstance(conversation, TurnDone)
    cid = conversation.conversation_id

    h.script(say("What is fact two?"), say("Fact two holds [S2]."), verdict())
    second = await collect(h.runner.run_turn(principal(), cid, [COLLECTION], "and the other?"))
    assert h.chat.kinds() == ["stream", "json", "stream", "stream", "json"]
    contextualize = h.chat.calls[2]
    assert "standalone" in contextualize.system
    assert [m.content for m in contextualize.messages[:2]] == [
        "what is fact one?",
        "Fact one holds [S1].",
    ]
    assert h.retriever.calls[-1][1] == "What is fact two?"
    generate = h.chat.calls[3]
    # history = raw committed messages only (no packed sources of the earlier turn)
    assert [m.content for m in generate.messages[:2]] == [
        "what is fact one?",
        "Fact one holds [S1].",
    ]
    assert "(Standalone question: What is fact two?)" in generate.messages[2].content
    assert isinstance(second[-1], TurnDone)


async def test_lease_is_released_and_a_concurrent_turn_is_rejected() -> None:
    h = make_harness([say("Fact one holds [S1]."), verdict()])
    events = await collect(h.runner.run_turn(principal(), None, [COLLECTION], "q?"))
    cid = events[-1].conversation_id  # type: ignore[union-attr]
    assert h.db.active == {}
    assert len(h.db.released) == 1

    h.db.active[cid] = UUID(int=99)  # another pod holds the lease
    with pytest.raises(TurnInProgress):
        await collect(h.runner.run_turn(principal(), cid, [COLLECTION], "again?"))
    assert h.chat.kinds() == ["stream", "json"]  # nothing ran


async def test_conversation_of_another_user_or_tenant_is_not_found() -> None:
    h = make_harness([say("Fact one holds [S1]."), verdict()])
    events = await collect(h.runner.run_turn(principal(), None, [COLLECTION], "q?"))
    cid = events[-1].conversation_id  # type: ignore[union-attr]
    with pytest.raises(NotFound):
        await collect(h.runner.run_turn(principal(user=OTHER_USER), cid, [COLLECTION], "q?"))
    from module_d1.d1_fakes import OTHER_TENANT

    with pytest.raises(NotFound):
        await collect(h.runner.run_turn(principal(tenant=OTHER_TENANT), cid, [COLLECTION], "q?"))


async def test_input_validation_and_scope() -> None:
    h = make_harness()
    with pytest.raises(Forbidden):
        await collect(
            h.runner.run_turn(principal(scopes=frozenset({"read"})), None, [COLLECTION], "q")
        )
    with pytest.raises(ValidationFailed):
        await collect(h.runner.run_turn(principal(), None, [COLLECTION], "   "))
    with pytest.raises(ValidationFailed):
        await collect(h.runner.run_turn(principal(), None, [], "q?"))


async def test_no_model_call_happens_inside_a_transaction_and_tenant_is_the_principals() -> None:
    # FakeChat/FakeEmbedder assert txm.depth == 0 on every call; reaching the end proves it.
    h = make_harness([say("Fact one holds [S1]."), verdict()])
    await collect(h.runner.run_turn(principal(), None, [COLLECTION], "q?"))
    assert h.txm.tenants
    assert set(h.txm.tenants) == {TENANT}


async def test_sources_event_carries_every_retrieved_hit() -> None:
    hits = [make_hit(1), make_hit(2), make_hit(3)]
    h = make_harness([say("Fact one [S1]."), verdict()], results=[hits])
    events = await collect(h.runner.run_turn(principal(), None, [COLLECTION], "q?"))
    (sources,) = of_type(events, TurnSources)
    assert sources.hits == tuple(hits)


async def test_summarize_job_enqueued_once_history_is_long() -> None:
    h = make_harness(settings=AgentSettings(summarize_after_messages=2, heartbeat_interval_s=3600))
    h.script(say("Fact one [S1]."), verdict())
    events = await collect(h.runner.run_turn(principal(), None, [COLLECTION], "q?"))
    assert h.jobs.enqueued == []
    cid = events[-1].conversation_id  # type: ignore[union-attr]
    h.script(say("Rewritten?"), say("Fact one [S1]."), verdict())
    await collect(h.runner.run_turn(principal(), cid, [COLLECTION], "again?"))
    ((tenant, spec),) = h.jobs.enqueued
    assert tenant == TENANT
    assert spec.kind == "summarize_conversation"
    assert spec.dedupe_key == f"summarize_conversation:{cid}"
