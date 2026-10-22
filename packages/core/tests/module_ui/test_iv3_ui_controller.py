"""IV3-UI (Phase 3c) without Streamlit: `advance` for a level restart and the fallback, and the
render helpers (refusal badge both ways, levels, the split summary, escaping)."""

from __future__ import annotations

from uuid import uuid4

from ui_support import FakeBackend, general_grade, general_question, interview_question

from ragforge_client.client import ApiError
from ragforge_client.interview import (
    Command,
    General,
    InterviewSession,
    Run,
    Start,
    WeakGeneral,
)
from ragforge_contracts.api import (
    INTERVIEW_NO_NOTES,
    InterviewLevel,
    InterviewSource,
    Problem,
)
from ragforge_ui.interview import (
    InterviewUi,
    NoticeCard,
    QuestionCard,
    SummaryCard,
    advance,
)
from ragforge_ui.render import (
    NOTHING_BADGE,
    badge_for,
    level_label,
    question_level,
    summary_headline_markdown,
    weak_general_markdown,
)

COLLECTIONS = [uuid4()]
NO_NOTES = ApiError(Problem(type=INTERVIEW_NO_NOTES, title="No notes", status=409))


def test_badge_says_nothing_to_check_only_for_an_uncited_unverified_turn() -> None:
    assert badge_for("unverified", cited=False) is NOTHING_BADGE
    assert (NOTHING_BADGE.label, NOTHING_BADGE.meaning) == (
        "Nothing to check",
        "No answer came from your notes.",
    )
    old = badge_for("unverified", cited=True)
    assert old is not None
    assert old.label == "Not checked"
    assert "quality bar" in old.meaning
    assert badge_for("unverified") == old  # the default keeps every old caller's meaning
    # uncited but flagged (untrusted verifier dropped inferred citations): claims WERE made
    assert badge_for("unverified", cited=False, flagged=True) == old
    partial = badge_for("partial", cited=False)
    assert partial is not None
    assert partial.label == "Partly verified"
    assert badge_for(None, cited=False) is None


def test_a_level_restart_ends_the_session_with_its_summary_then_begins_anew() -> None:
    ui = InterviewUi()
    backend = FakeBackend(questions=[interview_question(), interview_question(level="staff")])
    advance(ui, Start(("Kafka",)), backend, COLLECTIONS, 0)
    first = ui.session
    advance(ui, Start(("Kafka",), InterviewLevel.staff), backend, COLLECTIONS, 1)
    assert ui.session is not None
    assert ui.session is not first
    assert ui.session.level is InterviewLevel.staff
    kinds = [type(c).__name__ for c in ui.cards if c.position == 1]
    assert kinds == ["NoticeCard", "SummaryCard", "NoticeCard", "QuestionCard"]
    summary = next(c for c in ui.cards if isinstance(c, SummaryCard))
    assert summary.summary.level is InterviewLevel.senior_sde_1  # the old session's bar
    assert summary.summary.skipped == 1
    question = ui.cards[-1]
    assert isinstance(question, QuestionCard)
    assert question.number == 1


def test_an_unknown_level_on_start_adds_a_notice() -> None:
    ui = InterviewUi()
    advance(
        ui, Start((), None, "wizard"), FakeBackend(questions=[interview_question()]), COLLECTIONS, 0
    )
    notices = [c.text for c in ui.cards if isinstance(c, NoticeCard)]
    assert any('Unknown level "wizard": using Senior SDE-1' in n for n in notices)


def test_a_general_step_without_a_session_is_ignored() -> None:
    ui = InterviewUi()
    assert advance(ui, General("Redis"), FakeBackend(), COLLECTIONS, 0).called is False
    assert ui.cards == []


def test_no_notes_waits_on_the_offer_and_the_click_fetches_general() -> None:
    ui = InterviewUi()
    backend = FakeBackend(questions=[NO_NOTES, general_question()])
    advance(ui, Start(("Redis",)), backend, COLLECTIONS, 0)
    assert ui.session is not None
    assert ui.session.waiting
    assert [o.topic for o in ui.session.offers] == ["Redis"]
    assert advance(ui, Run(Command.next), backend, COLLECTIONS, 0).called is False
    assert len(backend.question_requests) == 1
    assert advance(ui, General("Redis"), backend, COLLECTIONS, 0).called
    assert backend.question_requests[1].source is InterviewSource.general
    assert isinstance(ui.cards[-1], QuestionCard)


def test_render_helpers_escape_owner_and_server_text() -> None:
    assert level_label(InterviewLevel.senior_sde_2) == "Senior SDE\\-2"
    assert level_label("**x**") == "\\*\\*x\\*\\*"  # unknown level from a newer server
    harder = interview_question(difficulty="harder", level="staff")
    assert question_level(harder) == "Staff · harder"
    session = InterviewSession.begin(Start(("**Redis**",), InterviewLevel.staff))
    summary = session.summary()
    assert summary_headline_markdown(summary) == ("**Staff · \\*\\*Redis\\*\\* · none graded**")


def test_weak_general_topics_suggest_adding_notes_escaped() -> None:
    session = InterviewSession.begin(Start(("[Redis](x)",)))
    session.exhausted(Problem(type=INTERVIEW_NO_NOTES, title="No notes", status=409))
    session.apply(General())
    session.question_request(COLLECTIONS)
    session.asked(general_question())
    session.graded("lru", general_grade(3))
    summary = session.summary()
    assert summary.weak_general == (WeakGeneral("[Redis](x)", "concept", 3),)
    markdown = weak_general_markdown(summary)
    assert "consider adding notes on \\[Redis\\]\\(x\\)" in markdown
    assert "](x)" not in markdown.replace("\\]\\(x\\)", "")
    assert summary.average is None  # no notes grade: never the general score
