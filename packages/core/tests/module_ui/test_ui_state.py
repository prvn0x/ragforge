"""The turn reducer: every SSE event folded into conversation state, plus approval decisions."""

from __future__ import annotations

from uuid import uuid4

import pytest
from ui_support import citation, done

from ragforge_contracts.api import ApprovalDecision, Problem
from ragforge_contracts.sse import (
    AnswerReplaceEvent,
    CitationEvent,
    ErrorEvent,
    ExtraEvent,
    InterruptEvent,
    TokenEvent,
    ToolCallEvent,
    VerificationEvent,
)
from ragforge_ui.errors import UiError
from ragforge_ui.state import (
    Conversation,
    Extra,
    Turn,
    cited_titles,
    decide,
    document_id_arg,
    foreign_run,
    in_flight,
    missing_titles,
    offers_code,
    open_interrupt,
    reached_server,
    record,
    resume_prompt,
    settle_question,
    settle_resume,
    start_turn,
)


def _interrupt_event(*ids: str) -> InterruptEvent:
    pending = [
        ToolCallEvent(tool_call_id=i, tool="remove_from_index", args={"document_id": i})
        for i in ids
    ]
    return InterruptEvent(run_id=uuid4(), conversation_id=uuid4(), pending=pending)


def test_tokens_accumulate_and_first_token_time_is_the_first_only() -> None:
    conversation = Conversation()
    turn = start_turn(conversation, "q")
    record(conversation, turn, TokenEvent(text="a"), 0.7)
    record(conversation, turn, TokenEvent(text="b"), 2.0)
    assert (turn.answer, turn.first_token_s) == ("ab", 0.7)


def test_replacement_overwrites_the_streamed_answer() -> None:
    conversation = Conversation()
    turn = start_turn(conversation, "q")
    record(conversation, turn, TokenEvent(text="draft"), 0.1)
    record(conversation, turn, AnswerReplaceEvent(text="final"), 0.2)
    assert (turn.answer, turn.revised) == ("final", True)


def test_citations_are_one_per_label() -> None:
    conversation = Conversation()
    turn = start_turn(conversation, "q")
    for label in (1, 2, 1):
        record(conversation, turn, CitationEvent(citation=citation(label)), 0.0)
    assert [c.label for c in turn.citations] == [1, 2]


def test_unlabelled_citations_dedupe_by_anchor() -> None:
    conversation = Conversation()
    turn = start_turn(conversation, "q")
    a = citation(1).model_copy(update={"label": None, "anchor": "x"})
    b = citation(2).model_copy(update={"label": None, "anchor": "y"})
    for c in (a, b, a):
        record(conversation, turn, CitationEvent(citation=c), 0.0)
    assert [c.anchor for c in turn.citations] == ["x", "y"]


def test_verification_records_status_and_claims_including_unknown_status() -> None:
    conversation = Conversation()
    turn = start_turn(conversation, "q")
    record(conversation, turn, VerificationEvent(status="partial", unsupported_claims=["c"]), 0)
    assert (turn.status, turn.unsupported_claims, turn.verdict_seen) == ("partial", ["c"], True)
    record(conversation, turn, VerificationEvent(status="brand_new"), 0)
    assert turn.status == "brand_new"


def test_extras_never_touch_the_answer() -> None:
    conversation = Conversation()
    turn = start_turn(conversation, "q")
    record(conversation, turn, TokenEvent(text="from notes"), 0.1)
    record(conversation, turn, ExtraEvent(kind="incident", text="general"), 0.2)
    assert turn.answer == "from notes"
    assert turn.extras == [Extra("incident", "general")]


def test_done_sets_the_conversation_for_follow_ups() -> None:
    conversation = Conversation()
    turn = start_turn(conversation, "q")
    conversation_id = uuid4()
    record(conversation, turn, done(conversation_id), 1.0)
    assert conversation.id == conversation_id


def test_mid_stream_error_keeps_the_partial_answer_and_the_new_conversation() -> None:
    conversation = Conversation()
    turn = start_turn(conversation, "q")
    record(conversation, turn, TokenEvent(text="half an ans"), 0.1)
    conversation_id = uuid4()
    problem = Problem(title="Too Many Requests", status=429, detail="slow down")
    record(conversation, turn, ErrorEvent(problem=problem, conversation_id=conversation_id), 0.3)
    assert turn.answer == "half an ans"
    assert turn.error is not None
    assert turn.error.title == "Too many requests right now"
    assert conversation.id == conversation_id


def test_error_without_conversation_id_keeps_the_known_one() -> None:
    known = uuid4()
    conversation = Conversation(id=known)
    turn = start_turn(conversation, "q")
    record(conversation, turn, ErrorEvent(problem=Problem(title="x", status=500)), 0.0)
    assert conversation.id == known


def test_interrupt_is_recorded_with_its_conversation() -> None:
    conversation = Conversation()
    turn = start_turn(conversation, "remove the kafka page from my index")
    event = _interrupt_event("t1")
    record(conversation, turn, event, 0.5)
    assert turn.interrupt is not None
    assert turn.interrupt.run_id == event.run_id
    assert [(p.tool_call_id, p.tool) for p in turn.interrupt.pending] == [
        ("t1", "remove_from_index")
    ]
    assert conversation.id == event.conversation_id
    assert open_interrupt(conversation) is turn.interrupt


def _interrupted() -> tuple[Conversation, Turn]:
    conversation = Conversation()
    turn = start_turn(conversation, "q")
    record(conversation, turn, _interrupt_event("t1"), 0.0)
    return conversation, turn


def test_decide_covers_every_pending_call_and_defaults_to_reject() -> None:
    conversation = Conversation()
    turn = start_turn(conversation, "q")
    record(conversation, turn, _interrupt_event("t1", "t2"), 0.0)
    assert turn.interrupt is not None
    assert decide(turn.interrupt, {"t2", "not-pending"}) == {
        "t1": ApprovalDecision.reject,
        "t2": ApprovalDecision.approve,
    }
    assert set(decide(turn.interrupt, set()).values()) == {ApprovalDecision.reject}


def test_a_new_question_the_server_took_expires_an_open_approval() -> None:
    conversation = Conversation()
    first = start_turn(conversation, "q")
    record(conversation, first, _interrupt_event("t1"), 0.0)
    question = start_turn(conversation, "something else")
    assert first.interrupt is not None
    assert first.interrupt.is_open  # nothing sent yet: still pending on the server
    record(conversation, question, TokenEvent(text="Kafka"), 0.1)
    settle_question(conversation, question)
    assert first.interrupt.expired
    assert open_interrupt(conversation) is None


def test_a_clean_empty_question_also_expires_it() -> None:
    conversation, turn = _interrupted()
    question = start_turn(conversation, "q2")
    question.finished = True
    settle_question(conversation, question)
    assert turn.interrupt is not None
    assert turn.interrupt.expired


@pytest.mark.parametrize("finished", [True, False])
def test_a_question_that_never_reached_the_server_keeps_the_approval_open(finished: bool) -> None:
    """A network error / 401 / 503 before any event (finished with an error), or a stop before
    the first event: the server never started a turn, so it never expired the calls."""
    conversation, turn = _interrupted()
    question = start_turn(conversation, "something else")
    question.finished = finished
    question.error = UiError("Cannot reach the RAGForge API", "...") if finished else None
    settle_question(conversation, question)
    assert turn.interrupt is not None
    assert not turn.interrupt.expired
    assert open_interrupt(conversation) is turn.interrupt  # Approve/Reject stay offered


def test_reached_server_is_one_rule() -> None:
    turn = Turn(prompt="q")
    assert not reached_server(turn)  # nothing yet
    turn.finished, turn.error = True, UiError("x", "y")
    assert not reached_server(turn)  # failed before any event
    turn.error = None
    assert reached_server(turn)  # ended cleanly (an empty stream)
    turn.finished, turn.events = False, 1
    assert reached_server(turn)  # any event, even if cut off later


def test_a_decided_interrupt_is_no_longer_open() -> None:
    conversation = Conversation()
    turn = start_turn(conversation, "q")
    record(conversation, turn, _interrupt_event("t1"), 0.0)
    assert turn.interrupt is not None
    turn.interrupt.decisions = decide(turn.interrupt, {"t1"})
    assert open_interrupt(conversation) is None
    assert open_interrupt(Conversation()) is None


def test_a_resume_that_never_reached_the_server_keeps_the_approval_open() -> None:
    conversation, turn = _interrupted()
    assert turn.interrupt is not None
    decisions = decide(turn.interrupt, {"t1"})
    failed = start_turn(conversation, "Approved: remove_from_index", kind="resume")
    failed.error = UiError("Cannot reach the RAGForge API", "...")
    failed.finished = True
    settle_resume(turn.interrupt, decisions, failed)
    assert turn.interrupt.decisions is None
    assert open_interrupt(conversation) is turn.interrupt  # retry possible past the failed turn
    # a resume cut off before its first event is not decided either
    cut = start_turn(conversation, "Approved: remove_from_index", kind="resume")
    settle_resume(turn.interrupt, decisions, cut)
    assert open_interrupt(conversation) is turn.interrupt


@pytest.mark.parametrize("server_error", [False, True])
def test_a_resume_the_server_answered_settles_the_approval(server_error: bool) -> None:
    conversation, turn = _interrupted()
    assert turn.interrupt is not None
    decisions = decide(turn.interrupt, {"t1"})
    resumed = start_turn(conversation, "Approved: remove_from_index", kind="resume")
    event = ErrorEvent(problem=Problem(title="Boom", status=500)) if server_error else done(uuid4())
    record(conversation, resumed, event, 0.1)
    settle_resume(turn.interrupt, decisions, resumed)  # an error EVENT: the request was taken
    assert turn.interrupt.decisions == decisions
    assert open_interrupt(conversation) is None


def test_a_clean_empty_resume_settles_the_approval() -> None:
    conversation, turn = _interrupted()
    assert turn.interrupt is not None
    resumed = start_turn(conversation, "Rejected: remove_from_index", kind="resume")
    resumed.finished = True
    settle_resume(turn.interrupt, decide(turn.interrupt, set()), resumed)
    assert open_interrupt(conversation) is None


def test_open_interrupt_is_only_the_latest_answered_turns() -> None:
    conversation, _ = _interrupted()
    later = start_turn(conversation, "Approved: x", kind="resume")
    record(conversation, later, TokenEvent(text="ok"), 0.1)  # the server took this resume
    assert open_interrupt(conversation) is None


def test_record_counts_events() -> None:
    conversation = Conversation()
    turn = start_turn(conversation, "q")
    for event in (TokenEvent(text="a"), TokenEvent(text="b"), done(uuid4())):
        record(conversation, turn, event, 0.0)
    assert turn.events == 3


def test_resume_prompt_names_the_tools() -> None:
    conversation = Conversation()
    turn = start_turn(conversation, "q")
    record(conversation, turn, _interrupt_event("t1", "t2"), 0.0)
    assert turn.interrupt is not None
    text = resume_prompt(turn.interrupt, decide(turn.interrupt, {"t1"}))
    assert text == "Approved: remove_from_index · Rejected: remove_from_index"


def _finished(**changes: object) -> Turn:
    turn = Turn(prompt="q", finished=True, citations=[citation(1)], answer="a \\[S1\\]")
    for name, value in changes.items():
        setattr(turn, name, value)
    return turn


def test_show_the_code_is_offered_only_after_a_cited_codeless_finished_answer() -> None:
    assert offers_code(_finished())
    assert not offers_code(_finished(citations=[]))
    assert not offers_code(_finished(finished=False))
    assert not offers_code(_finished(answer="x\n```py\nprint()\n```"))
    assert not offers_code(_finished(answer="x\n~~~\nprint()\n~~~"))
    assert not offers_code(_finished(extras=[Extra("code", "```\nx\n```")]))
    assert offers_code(_finished(extras=[Extra("incident", "pattern")]))
    assert not offers_code(_finished(error=UiError("x", "y")))


# ------------------------------------------------------------------ approval links
def test_foreign_run_is_a_run_this_session_does_not_hold() -> None:
    conversation, turn = _interrupted()
    assert turn.interrupt is not None
    other = uuid4()
    assert foreign_run(conversation, str(other)) == other
    assert foreign_run(Conversation(), str(other)) == other  # e.g. after a reload
    assert foreign_run(conversation, str(turn.interrupt.run_id)) is None  # ours: drawn below
    assert foreign_run(conversation, str(turn.interrupt.run_id).upper()) is None
    assert foreign_run(conversation, None) is None
    assert foreign_run(conversation, "") is None
    assert foreign_run(conversation, "not-a-uuid") is None


# ------------------------------------------------------------------ document titles
def test_document_id_arg_is_canonical_or_none() -> None:
    doc = uuid4()
    assert document_id_arg({"document_id": str(doc).upper()}) == str(doc)
    assert document_id_arg({"document_id": "d1"}) is None
    assert document_id_arg({"document_id": 7}) is None
    assert document_id_arg({"fact_id": str(doc)}) is None


def test_cited_titles_and_missing_titles() -> None:
    conversation = Conversation()
    turn = start_turn(conversation, "q")
    cited = citation(1, title="Kafka basics")
    record(conversation, turn, CitationEvent(citation=cited), 0.0)
    titles = cited_titles(conversation)
    assert titles == {str(cited.document_id): "Kafka basics"}
    pending = [
        ToolCallEvent(
            tool_call_id="t1",
            tool="remove_from_index",
            args={"document_id": str(cited.document_id)},
        )
    ]
    record(conversation, turn, InterruptEvent(run_id=uuid4(), pending=pending), 0.1)
    assert turn.interrupt is not None
    assert not missing_titles(turn.interrupt, titles)  # named by a citation: no API call needed
    assert missing_titles(turn.interrupt, {})
    no_docs = [ToolCallEvent(tool_call_id="t2", tool="forget_fact", args={"fact_id": "f1"})]
    record(conversation, turn, InterruptEvent(run_id=uuid4(), pending=no_docs), 0.2)
    assert not missing_titles(turn.interrupt, {})


@pytest.mark.parametrize(
    ("status", "flying"),
    [
        ("queued", True),
        ("running", True),
        ("done", False),
        ("failed", False),
        ("dead", False),
        ("superseded", False),
        ("paused_by_newer_server", False),
    ],
)
def test_in_flight_jobs(status: str, flying: bool) -> None:
    assert in_flight(status) is flying
