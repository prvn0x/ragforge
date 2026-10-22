"""IV3-UI (Phase 3c) on the real Streamlit page (AppTest) against a fake `Backend`. No network.

Level dropdown (start form and running session; a change starts a new session), the opt-in
general-knowledge fallback (offered on both 409 types, switched ONLY by a click), the general
label on every general card and never on notes cards, `exclude_questions`, the summary split,
the new start phrases, the refusal badge text both ways, and an empty answer key."""

from __future__ import annotations

from uuid import uuid4

import pytest
from streamlit.testing.v1 import AppTest
from ui_support import (
    FakeBackend,
    app_with,
    citation,
    done,
    general_grade,
    general_question,
    interview_grade,
    interview_question,
    texts,
)

from ragforge_client.client import ApiError
from ragforge_client.interview import GENERAL_GRADE_LABEL, GENERAL_LABEL, NO_KEY_NOTE
from ragforge_contracts.api import (
    INTERVIEW_EXHAUSTED,
    INTERVIEW_NO_NOTES,
    InterviewLevel,
    InterviewSource,
    Problem,
)
from ragforge_contracts.sse import CitationEvent, TokenEvent, VerificationEvent
from ragforge_ui.interview_views import LEVEL, SESSION_LEVEL

OFFER = "from general knowledge (not from your notes, not verified)"
NO_NOTES = ApiError(Problem(type=INTERVIEW_NO_NOTES, title="No notes", status=409))
EXHAUSTED = ApiError(Problem(type=INTERVIEW_EXHAUSTED, title="Exhausted", status=409))
# texts the verifier badge shows: never on a general card
VERIFIER_TEXTS = ("Verified", "Partly verified", "Not checked", "Nothing to check", "quality bar")


def _type(at: AppTest, message: str) -> AppTest:
    return at.chat_input[0].set_value(message).run()


def _click(at: AppTest, label: str) -> AppTest:
    matches = [b for b in at.button if label in b.label.replace("\\", "")]
    assert matches, f"no button {label!r}: {[b.label for b in at.button]}"
    return matches[0].click().run()


def _labels(at: AppTest) -> list[str]:
    return [b.label for b in at.button]


def _page(at: AppTest) -> str:
    """The page's text with Markdown backslash escapes removed (the UI escapes every server or
    owner string it places itself; what matters here is what the owner reads)."""
    return "\n".join(texts(at.main)).replace("\\", "")


def _offers(at: AppTest) -> list[str]:
    return [label for label in _labels(at) if OFFER in label.replace("\\", "")]


# ------------------------------------------------------------------ level
def test_the_level_dropdown_defaults_to_senior_sde_1_and_its_choice_is_sent_and_shown() -> None:
    backend = FakeBackend(
        questions=[interview_question(level="staff", token="tok")],
        grades=[interview_grade(6)],
    )
    at = app_with(backend)
    at = at.sidebar.toggle[0].set_value(True).run()
    dropdown = at.selectbox(key=LEVEL)
    assert dropdown.value is InterviewLevel.senior_sde_1
    assert dropdown.options == ["SDE-1", "SDE-2", "Senior SDE-1", "Senior SDE-2", "Staff"]
    at = dropdown.set_value(InterviewLevel.staff).run()
    at.text_input(key="rf_interview_topics").input("Kafka")
    at = _click(at, "Start interview")
    assert not at.exception
    assert backend.question_requests[0].level is InterviewLevel.staff
    page = _page(at)
    assert "Level: Staff" in page  # the question card shows the question's level
    assert "level: Staff (fixed for this interview" in page
    at = _type(at, "the leader waits")
    assert backend.grade_requests[0].level is InterviewLevel.staff  # echoed from the question
    assert backend.grade_requests[0].question_token == "tok"


def test_changing_the_level_mid_session_ends_it_and_starts_a_new_one() -> None:
    backend = FakeBackend(
        questions=[interview_question(), interview_question(level="sde_2")],
        grades=[interview_grade(7)],
    )
    at = _type(app_with(backend), "interview me on Kafka")
    at = _type(at, "acks=all")  # one graded answer in the first session
    assert at.selectbox(key=SESSION_LEVEL).value is InterviewLevel.senior_sde_1
    assert "changing it starts a new one" in _page(at)
    at = at.selectbox(key=SESSION_LEVEL).set_value(InterviewLevel.sde_2).run()
    assert not at.exception
    assert [r.level for r in backend.question_requests] == [
        InterviewLevel.senior_sde_1,
        InterviewLevel.sde_2,
    ]
    assert [r.topic for r in backend.question_requests] == ["Kafka", "Kafka"]
    page = _page(at)
    assert "Level changed to SDE-2: that starts a new interview" in page
    assert "Interview summary" in page  # the first session's summary, before the new one
    assert page.index("Interview summary") < page.rindex("Question 1")
    assert at.selectbox(key=SESSION_LEVEL).value is InterviewLevel.sde_2


def test_a_level_phrase_in_the_chat_box_sets_the_level() -> None:
    backend = FakeBackend(questions=[interview_question(level="staff")])
    at = _type(app_with(backend), "interview me on Kafka as a staff engineer")
    assert backend.question_requests[0].level is InterviewLevel.staff
    assert at.selectbox(key=SESSION_LEVEL).value is InterviewLevel.staff


# ------------------------------------------------------------------ the general fallback
@pytest.mark.parametrize(
    ("problem", "notice"),
    [
        (NO_NOTES, "Your notes have nothing on"),
        (EXHAUSTED, "notes are asked"),
    ],
)
def test_the_fallback_is_offered_on_both_409s_and_only_a_click_switches(
    problem: ApiError, notice: str
) -> None:
    backend = FakeBackend(
        questions=[problem, general_question(text="Q one\\?"), general_question()],
        grades=[general_grade(4)],
    )
    at = _type(app_with(backend), "interview me on Redis")
    assert notice in _page(at)
    assert len(_offers(at)) == 1
    assert "Redis" in _offers(at)[0]
    # no automatic switch: one request, from the notes, and nothing more until the click
    assert [r.source for r in backend.question_requests] == [InterviewSource.notes]
    at = _click(at, "Next question")
    assert len(backend.question_requests) == 1  # "next" while waiting asks nothing
    assert "take a general-knowledge offer" in _page(at)

    at = _click(at, OFFER.split(" (")[0])  # the offer button
    second = backend.question_requests[1]
    assert (second.source, second.topic, second.exclude_questions) == (
        InterviewSource.general,
        "Redis",
        [],
    )
    assert _offers(at) == []  # taken
    page = _page(at)
    assert GENERAL_LABEL in page  # on the question card
    assert not any(t in page for t in VERIFIER_TEXTS)

    at = _type(at, "LRU")
    grade = backend.grade_requests[0]
    assert (grade.source, grade.chunk_ids, grade.question_token) == (
        InterviewSource.general,
        [],
        "gtok",
    )
    page = _page(at)
    assert GENERAL_GRADE_LABEL in page
    assert f"**Model answer** · *{GENERAL_LABEL}*" in page  # on the model answer too
    assert "against the model's own key" in page
    assert "* checked against your notes*" not in page.replace("*checked", "* checked")
    assert "Sources from your notes" not in page  # no source chips
    assert "This question came from" not in page
    assert not any(t in page for t in VERIFIER_TEXTS)
    assert not [e for e in at.error if "Not found in your notes" in e.value]

    at = _click(at, "Next question")
    third = backend.question_requests[2]
    assert third.source is InterviewSource.general
    assert third.exclude_questions == ["Q one\\?"]  # the session's earlier general question


def test_a_topic_list_gets_one_offer_per_topic_and_notes_cards_carry_no_label() -> None:
    backend = FakeBackend(
        questions=[interview_question(topic="Kafka ISR"), NO_NOTES, NO_NOTES, interview_question()]
    )
    at = _type(app_with(backend), "interview me on Kafka, Redis, Raft")
    at = _click(at, "Skip")  # Redis, then Raft: no notes -> each offered; Kafka continues
    assert [r.topic for r in backend.question_requests] == ["Kafka", "Redis", "Raft", "Kafka"]
    offers = _offers(at)
    assert len(offers) == 2
    assert any("Redis" in o for o in offers)
    assert any("Raft" in o for o in offers)
    page = _page(at)
    assert "Question 2" in page
    assert GENERAL_LABEL not in page  # notes questions never carry the general label
    assert all(r.source is InterviewSource.notes for r in backend.question_requests)


def test_a_mixed_session_is_never_offered_the_fallback() -> None:
    backend = FakeBackend(questions=[NO_NOTES])
    at = _type(app_with(backend), "quiz me")
    assert _offers(at) == []
    assert "Interview summary" in _page(at)


def test_the_summary_keeps_notes_and_general_scores_apart() -> None:
    backend = FakeBackend(
        questions=[
            interview_question(topic="Kafka ISR"),
            NO_NOTES,
            interview_question(topic="Kafka acks"),
            general_question(),
        ],
        grades=[interview_grade(5), general_grade(4)],
    )
    at = _type(app_with(backend), "interview me on Kafka, Redis")
    at = _type(at, "acks=all")
    at = _click(at, "Next question")  # Redis: no notes -> offered; Kafka continues
    at = _click(at, OFFER.split(" (")[0])  # the pending Kafka question is skipped for it
    at = _type(at, "LRU")
    at = _click(at, "Stop")
    page = _page(at)
    assert "Senior SDE-1 · Kafka, Redis · avg 5.0 from your notes" in page
    assert "general knowledge avg 4.0 (not verified)" in page
    assert "Weak topics from your notes" in page
    assert "consider adding notes on Redis" in page
    assert "Kafka replication" in page  # the notes topic lists its page to revisit


# ------------------------------------------------------------------ start phrases
@pytest.mark.parametrize(
    "phrase", ["INTERVIEW MODE", "start an interview on Kafka", "start interview"]
)
def test_new_start_phrases_enter_the_mode(phrase: str) -> None:
    backend = FakeBackend(questions=[interview_question()])
    at = _type(app_with(backend), phrase)
    assert backend.chats == []  # the live failure: it went to Q&A and got a refusal
    assert len(backend.question_requests) == 1
    assert at.sidebar.toggle[0].value is True


# ------------------------------------------------------------------ refusal badge
def test_an_uncited_unverified_refusal_says_nothing_to_check() -> None:
    backend = FakeBackend(
        streams=[
            [
                TokenEvent(text="Your notes don't cover this."),
                VerificationEvent(status="unverified"),
                done(uuid4()),
            ]
        ]
    )
    page = _page(_type(app_with(backend), "what is a quasar?"))
    assert "Nothing to check" in page
    assert "No answer came from your notes." in page
    assert "quality bar" not in page
    assert "Not checked" not in page


def test_a_cited_unverified_answer_keeps_the_quality_bar_text() -> None:
    backend = FakeBackend(
        streams=[
            [
                TokenEvent(text="Consumers poll \\[S1\\]."),
                CitationEvent(citation=citation()),
                VerificationEvent(status="unverified"),
                done(uuid4()),
            ]
        ]
    )
    page = _page(_type(app_with(backend), "how do consumers work?"))
    assert "Not checked" in page
    assert "below the verifier's quality bar" in page
    assert "Nothing to check" not in page


def test_an_uncited_but_flagged_unverified_answer_keeps_the_quality_bar_text() -> None:
    """Untrusted verifier: the server drops inferred citations and flags every sentence. The
    answer made claims from the notes, so "Nothing to check" would be false (review round 1)."""
    backend = FakeBackend(
        streams=[
            [
                TokenEvent(text="Kafka elects leaders with Raft."),
                VerificationEvent(
                    status="unverified", unsupported_claims=["Kafka elects leaders with Raft."]
                ),
                done(uuid4()),
            ]
        ]
    )
    page = _page(_type(app_with(backend), "how does Kafka elect leaders?"))
    assert "Not checked" in page
    assert "below the verifier's quality bar" in page
    assert "Nothing to check" not in page
    assert "Kafka elects leaders with Raft." in page


def test_a_general_question_without_a_token_explains_itself_and_can_be_skipped() -> None:
    tokenless = general_question().model_copy(update={"question_token": None})
    backend = FakeBackend(questions=[NO_NOTES, tokenless, general_question()])
    at = _type(app_with(backend), "interview me on Redis")
    at = _click(at, OFFER.split(" (")[0])
    at = _type(at, "LRU")
    page = _page(at)
    assert backend.grade_requests == []
    assert "came without its server token" in page
    assert "validation error" not in page
    at = _click(at, "Skip")
    assert len(backend.question_requests) == 3


# ------------------------------------------------------------------ empty answer key
def test_an_empty_answer_key_still_grades_and_says_so_only_where_the_key_goes() -> None:
    backend = FakeBackend(
        questions=[interview_question(points=[]), interview_question()],
        grades=[interview_grade(6), interview_grade(6)],
    )
    at = _type(app_with(backend), "quiz me")
    assert NO_KEY_NOTE not in _page(at)  # not on the question card (the key is never there)
    at = _type(at, "acks=all")
    assert backend.grade_requests[0].expected_points == []
    assert [m.value for m in at.metric] == ["6/10"]
    assert f"{NO_KEY_NOTE}." in _page(at)
    at = _click(at, "Next question")
    at = _type(at, "acks=all")
    page = _page(at)
    assert page.count(f"{NO_KEY_NOTE}.") == 1  # the second question had a key
    assert "**Answer key** · *checked against your notes*" in page


def test_a_grade_for_a_general_question_is_never_drawn_as_notes() -> None:
    """Defence in depth: even if a server labels the grade of a GENERAL question as notes (with
    a verdict and citations), the card is the general one: no badge, flags or sources."""
    mislabelled = interview_grade(5).model_copy(update={"verification": "verified"})
    backend = FakeBackend(questions=[NO_NOTES, general_question()], grades=[mislabelled])
    at = _type(app_with(backend), "interview me on Redis")
    at = _click(at, OFFER.split(" (")[0])
    at = _type(at, "LRU")
    page = _page(at)
    assert GENERAL_GRADE_LABEL in page
    assert not any(t in page for t in VERIFIER_TEXTS)
    assert "Sources from your notes" not in page


def test_typing_y_takes_the_offer_only_when_no_question_is_pending() -> None:
    backend = FakeBackend(
        questions=[
            interview_question(topic="Kafka"),
            NO_NOTES,
            interview_question(),
            general_question(),
        ],
        grades=[interview_grade(5)],
    )
    at = _type(app_with(backend), "interview me on Kafka, Redis")
    at = _click(at, "Skip")  # Redis offered; a Kafka question is pending
    at = _type(at, "y")  # an ANSWER to the pending question, never the offer
    assert [r.answer for r in backend.grade_requests] == ["y"]
    assert all(r.source is InterviewSource.notes for r in backend.question_requests)
    at = _click(at, OFFER.split(" (")[0])  # the offer stays one click away
    assert backend.question_requests[-1].source is InterviewSource.general
    assert backend.chats == []


def test_a_stray_y_after_a_grade_never_takes_an_old_offer() -> None:
    """Review round 1: the UI idles after every grade, so "y" (meaning "yes, go on") typed while
    Kafka is still asked must not switch to general knowledge on the old Redis offer."""
    backend = FakeBackend(
        questions=[interview_question(topic="Kafka"), NO_NOTES, interview_question()],
        grades=[interview_grade(5)],
        streams=[[TokenEvent(text="Hi."), VerificationEvent(status="unverified"), done(uuid4())]],
    )
    at = _type(app_with(backend), "interview me on Kafka, Redis")
    at = _click(at, "Skip")  # Redis offered; a Kafka question is pending
    at = _type(at, "acks=all")  # graded; nothing pending, Kafka still in the rotation
    at = _type(at, "y")
    assert all(r.source is InterviewSource.notes for r in backend.question_requests)
    assert [m for m, _, _ in backend.chats] == ["y"]  # a normal message, not an opt-in


def test_y_takes_the_offer_while_the_session_waits_on_it() -> None:
    backend = FakeBackend(questions=[NO_NOTES, general_question()])
    at = _type(app_with(backend), "interview me on Redis")
    at = _type(at, "y")
    assert backend.question_requests[-1].source is InterviewSource.general
    assert backend.chats == []
