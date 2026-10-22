"""Interview mode on the real Streamlit page (AppTest) against a fake `Backend`. No network.

Flows: toggle -> topic -> question -> answer -> grade card (score, labelled lists, split flags)
-> next / harder / stop summary; a phrase in the chat box enters the mode; a normal question
suspends it and "grade it" recovers a hedged answer; failures keep the session."""

from __future__ import annotations

from uuid import uuid4

import httpx
from streamlit.testing.v1 import AppTest
from ui_support import (
    FakeBackend,
    app_with,
    cited_answer,
    done,
    interview_grade,
    interview_question,
    texts,
)

from ragforge_client.client import ApiError
from ragforge_contracts.api import (
    INTERVIEW_EXHAUSTED,
    InterviewDifficulty,
    Problem,
    UnsupportedClaim,
)
from ragforge_contracts.sse import TokenEvent, VerificationEvent
from ragforge_ui.interview_views import GRADE_IT


def _type(at: AppTest, message: str) -> AppTest:
    return at.chat_input[0].set_value(message).run()


def _click(at: AppTest, label: str) -> AppTest:
    matches = [b for b in at.button if b.label == label]
    assert matches, f"no button {label!r}: {[b.label for b in at.button]}"
    return matches[0].click().run()


def _labels(at: AppTest) -> list[str]:
    return [b.label for b in at.button]


def _page(at: AppTest) -> str:
    return "\n".join(texts(at.main))


def _index(items: list[str], needle: str) -> int:
    return next(i for i, text in enumerate(items) if needle in text)


def _start_with_toggle(backend: FakeBackend, topics: str = "Kafka, Redis") -> AppTest:
    at = app_with(backend)
    at = at.sidebar.toggle[0].set_value(True).run()
    at.text_input(key="rf_interview_topics").input(topics)
    return _click(at, "Start interview")


def test_toggle_question_answer_grade_card_then_next_harder_and_stop_summary() -> None:
    claims = [
        UnsupportedClaim(text="It is always fast.", reason="not_found"),
        UnsupportedClaim(text="`acks=all`", reason="inexact_code"),
    ]
    backend = FakeBackend(
        questions=[
            interview_question(topic="Kafka ISR", token="tok"),
            interview_question(topic="Redis eviction", kind="scenario"),
            interview_question(topic="Kafka acks", difficulty="harder"),
        ],
        grades=[interview_grade(4, claims)],
    )
    at = _start_with_toggle(backend)
    assert not at.exception
    assert [r.topic for r in backend.question_requests] == ["Kafka"]
    page = _page(at)
    assert "Question 1" in page
    assert "Why does the leader wait for the ISR" in page
    assert "Kafka replication" not in page  # no source titles before the answer: a hint
    assert "Concept" in page  # the kind badge
    assert "Level: Senior SDE\\-1" in page
    assert at.sidebar.toggle[0].value is True
    assert "Skip" in _labels(at)

    at = _type(at, "the leader waits for every ISR member")
    assert backend.chats == []  # graded, never sent to Q&A
    request = backend.grade_requests[0]
    assert (request.answer, request.question_token) == (
        "the leader waits for every ISR member",
        "tok",
    )
    assert [m.value for m in at.metric] == ["4/10"]
    items = texts(at.main)
    assert items[_index(items, "**Right**")].count("the grader") == 1
    assert "checked against your notes" in items[_index(items, "**Missed**")]
    assert "checked against your notes" in items[_index(items, "**Misconceptions**")]
    assert _index(items, "Model answer") < _index(items, "Partly verified")
    assert items[_index(items, "Model answer")] == "**Model answer**"  # no provenance claim
    assert [e.value for e in at.error if "Not found in your notes" in e.value]
    assert "It is always fast" in _page(at)
    inexact = [x for x in at.expander if "match exactly" in x.label]
    assert len(inexact) == 1  # muted, collapsed
    assert "acks=all" in "\n".join(texts(inexact[0]))
    assert not [e for e in at.error if "acks=all" in e.value]  # never red
    assert "Kafka replication" in _page(at)  # sources revealed after grading
    assert "Next question" in _labels(at)

    at = _click(at, "Next question")
    assert backend.question_requests[1].topic == "Redis"  # topics alternate
    assert "Question 2" in _page(at)
    at = _click(at, "Harder")
    assert backend.question_requests[2].difficulty is InterviewDifficulty.harder
    assert "Question 3" in _page(at)

    at = _click(at, "Stop")
    page = _page(at)
    assert "Interview summary" in page
    assert "3 asked · 1 answered · 2 skipped" in page
    assert "Kafka ISR" in page  # a weak topic (4/10) with its page to revisit
    assert "https://www.notion.so/Kafka-basics-abc#block1" in page
    assert at.sidebar.toggle[0].value is False
    assert "Skip" not in _labels(at)


def test_a_phrase_in_the_chat_box_enters_the_mode_and_a_normal_question_suspends_it() -> None:
    hedged = "Is it because the ISR shrinks?"
    backend = FakeBackend(
        streams=[cited_answer(uuid4())],
        questions=[interview_question()],
        grades=[interview_grade(8)],
    )
    at = _type(app_with(backend), "interview me on Kafka")
    assert backend.chats == []
    assert backend.question_requests[0].topic == "Kafka"
    assert at.sidebar.toggle[0].value is True

    at = _type(at, hedged)  # a normal question: Q&A answers it, the interview pauses
    assert [c[0] for c in backend.chats] == [hedged]
    assert "Interview paused" in _page(at)
    assert at.sidebar.toggle[0].value is False
    items = texts(at.main)
    assert _index(items, "Question 1") < _index(items, hedged) < _index(items, "Consumers poll")

    at = _click(at, GRADE_IT)  # it was an answer after all
    assert [r.answer for r in backend.grade_requests] == [hedged]
    assert [m.value for m in at.metric] == ["8/10"]
    assert at.sidebar.toggle[0].value is True


def test_ordinary_chat_is_unchanged_outside_the_mode() -> None:
    backend = FakeBackend(streams=[cited_answer(uuid4())])
    at = _type(app_with(backend), "what is an interview?")
    assert [c[0] for c in backend.chats] == ["what is an interview?"]
    assert backend.question_requests == []
    assert "Skip" not in _labels(at)


def test_an_old_server_flags_every_claim_red_in_chat() -> None:
    backend = FakeBackend(
        streams=[
            [
                TokenEvent(text="Answer."),
                VerificationEvent(status="partial", unsupported_claims=["made up", "`x()`"]),
                done(uuid4()),
            ]
        ]
    )
    at = _type(app_with(backend), "q")
    red = [e.value for e in at.error if "Not found in your notes" in e.value]
    assert len(red) == 1
    assert "made up" in red[0]
    assert "x()" in red[0]
    assert not [x for x in at.expander if "match exactly" in x.label]


def test_chat_claims_split_by_reason_and_an_unknown_reason_is_red() -> None:
    claims = [
        UnsupportedClaim(text="made up", reason="not_found"),
        UnsupportedClaim(text="`createMemento()`", reason="inexact_code"),
        UnsupportedClaim(text="newer reason", reason="contradicted"),
    ]
    backend = FakeBackend(
        streams=[
            [
                TokenEvent(text="Answer."),
                VerificationEvent(
                    status="partial",
                    unsupported_claims=[c.text for c in claims],
                    claims=claims,
                ),
                done(uuid4()),
            ]
        ]
    )
    at = _type(app_with(backend), "q")
    red = next(e.value for e in at.error if "Not found in your notes" in e.value)
    assert "made up" in red
    assert "newer reason" in red
    assert "createMemento" not in red
    (muted,) = [x for x in at.expander if "match exactly" in x.label]
    assert "createMemento()" in "\n".join(texts(muted))


def test_failures_are_friendly_and_keep_the_session() -> None:
    backend = FakeBackend(
        questions=[
            httpx.ConnectError("refused"),
            interview_question(),
        ],
        grades=[ApiError(Problem(title="Too Many Requests", status=429)), interview_grade(9)],
    )
    at = _type(app_with(backend), "quiz me")
    assert any("Cannot reach the RAGForge API" in e.value for e in at.error)
    assert "Next question" in _labels(at)  # nothing pending: retry with one click
    at = _click(at, "Next question")
    assert "Question 1" in _page(at)
    at = _type(at, "acks=all")
    assert any("Too many requests" in e.value for e in at.error)
    assert any("Grade again" in e.value for e in at.error)
    at = _click(at, "Grade again")
    assert [r.answer for r in backend.grade_requests] == ["acks=all", "acks=all"]
    assert [m.value for m in at.metric] == ["9/10"]
    assert not at.exception


def test_401_and_a_missing_endpoint_are_friendly() -> None:
    backend = FakeBackend(
        questions=[
            ApiError(Problem(title="Unauthorized", status=401)),
            ApiError(Problem(title="Not Found", status=404)),
        ]
    )
    at = _type(app_with(backend), "/interview")
    assert any("rejected this key" in e.value for e in at.error)
    at = _click(at, "Next question")
    assert any("no interview mode" in e.value for e in at.error)
    assert "Interview summary" not in _page(at)  # a 404 is not exhaustion


def test_exhaustion_drops_a_topic_offers_it_and_stop_ends_with_the_summary() -> None:
    exhausted = ApiError(Problem(type=INTERVIEW_EXHAUSTED, title="Exhausted", status=409))
    backend = FakeBackend(
        questions=[interview_question(topic="Kafka"), exhausted, interview_question(), exhausted]
    )
    at = _type(app_with(backend), "interview me on Kafka and Redis")
    at = _click(at, "Skip")  # Redis is exhausted: dropped (offered), Kafka continues
    assert [r.topic for r in backend.question_requests] == ["Kafka", "Redis", "Kafka"]
    assert "Continuing with Kafka" in _page(at)
    at = _click(at, "Skip")  # Kafka exhausted too: only the offers are left
    assert "Interview summary" not in _page(at)
    at = _click(at, "Stop")
    page = _page(at)
    assert "Interview summary" in page
    assert "Fully covered" in page
    assert at.sidebar.toggle[0].value is False


def test_stop_typed_in_the_box_ends_and_turning_the_toggle_off_ends_too() -> None:
    backend = FakeBackend(questions=[interview_question(), interview_question()])
    at = _type(app_with(backend), "quiz me")
    at = _type(at, "stop")
    assert "Interview summary" in _page(at)
    at = _type(at, "quiz me")
    at = at.sidebar.toggle[0].set_value(False).run()
    assert _page(at).count("Interview summary") == 2
    assert backend.chats == []


def test_an_unknown_kind_and_level_are_shown_as_text() -> None:
    backend = FakeBackend(questions=[interview_question(kind="puzzle", difficulty="expert")])
    at = _type(app_with(backend), "quiz me")
    page = "\n".join(texts(at.main))
    assert "puzzle" in page
    assert "expert" in page
    assert not at.exception


def test_without_a_selected_source_the_interview_controls_are_disabled_with_a_note() -> None:
    """Round-1 review: a click was queued, then dropped silently with nothing to ask from."""
    backend = FakeBackend(questions=[interview_question()])
    at = _start_with_toggle(backend)
    at = at.sidebar.multiselect[0].set_value([]).run()
    assert not at.exception
    assert [i.value for i in at.info if "Pick at least one source" in i.value]
    controls = [b for b in at.button if b.label in ("Skip", "Harder", "Easier", "Stop")]
    assert len(controls) == 4
    assert all(b.disabled for b in controls)
    assert len(backend.question_requests) == 1
