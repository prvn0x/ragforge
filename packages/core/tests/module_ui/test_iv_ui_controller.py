"""Interview mode in the UI without Streamlit: `advance` (cards, exhaustion, failures, suspend),
the render helpers it relies on, and `SdkBackend`'s two interview calls over a fake transport."""

from __future__ import annotations

import json
from uuid import uuid4

import httpx
import pytest
from pydantic import SecretStr
from ui_support import FakeBackend, interview_grade, interview_question

from ragforge_client.client import ApiError
from ragforge_client.interview import (
    Answer,
    Command,
    InterviewSession,
    Leave,
    Run,
    Start,
)
from ragforge_contracts.api import (
    INTERVIEW_EXHAUSTED,
    InterviewQuestionRequest,
    Problem,
)
from ragforge_ui.api import INTERVIEW_TIMEOUT_S, SdkBackend
from ragforge_ui.interview import (
    AnswerCard,
    ErrorCard,
    GradeCard,
    InterviewUi,
    NoticeCard,
    QuestionCard,
    SummaryCard,
    advance,
    interview_failure,
)
from ragforge_ui.render import (
    bullet_markdown,
    kind_label,
    topic_markdown,
    weak_topics_markdown,
)
from ragforge_ui.views import route_message

COLLECTIONS = [uuid4()]
EXHAUSTED = ApiError(Problem(type=INTERVIEW_EXHAUSTED, title="Exhausted", status=409))


def _kinds(ui: InterviewUi) -> list[str]:
    return [type(card).__name__ for card in ui.cards]


def test_start_fetches_the_first_question_at_the_current_position() -> None:
    ui = InterviewUi(form_open=True)
    backend = FakeBackend(questions=[interview_question()])
    outcome = advance(ui, Start(("Kafka",)), backend, COLLECTIONS, position=3)
    assert outcome.called
    assert outcome.chat is None
    assert _kinds(ui) == ["NoticeCard", "QuestionCard"]
    assert all(card.position == 3 for card in ui.cards)
    assert not ui.form_open
    assert ui.active


def test_a_step_without_a_session_other_than_start_does_nothing() -> None:
    ui = InterviewUi()
    assert advance(ui, Run(Command.skip), FakeBackend(), COLLECTIONS, 0).called is False
    assert ui.cards == []


def test_an_answer_is_shown_then_graded_and_the_grade_keeps_its_question() -> None:
    ui = InterviewUi()
    question = interview_question(token="tok")
    backend = FakeBackend(questions=[question], grades=[interview_grade(7)])
    advance(ui, Start(), backend, COLLECTIONS, 0)
    advance(ui, Answer("acks=all"), backend, COLLECTIONS, 0)
    answer, grade = ui.cards[-2:]
    assert isinstance(answer, AnswerCard)
    assert answer.text == "acks=all"
    assert isinstance(grade, GradeCard)
    assert grade.question is question
    assert backend.grade_requests[0].question_token == "tok"
    assert ui.session is not None
    assert ui.session.current is None


def test_a_failed_grade_keeps_question_and_answer_for_grade_again() -> None:
    ui = InterviewUi()
    backend = FakeBackend(
        questions=[interview_question()],
        grades=[httpx.ReadTimeout("slow"), interview_grade(6)],
    )
    advance(ui, Start(), backend, COLLECTIONS, 0)
    advance(ui, Answer("acks=all"), backend, COLLECTIONS, 0)
    error = ui.cards[-1]
    assert isinstance(error, ErrorCard)
    assert "Grade again" in error.error.detail
    assert ui.unsent_answer == "acks=all"
    assert ui.session is not None
    assert ui.session.current is not None
    advance(ui, Answer(ui.unsent_answer), backend, COLLECTIONS, 0)
    assert isinstance(ui.cards[-1], GradeCard)
    assert ui.unsent_answer is None


def test_an_empty_answer_is_not_sent() -> None:
    ui = InterviewUi()
    backend = FakeBackend(questions=[interview_question()])
    advance(ui, Start(), backend, COLLECTIONS, 0)
    advance(ui, Answer("   "), backend, COLLECTIONS, 0)
    assert backend.grade_requests == []
    assert isinstance(ui.cards[-1], ErrorCard)


def test_a_normal_question_returns_it_for_q_and_a_and_suspends() -> None:
    ui = InterviewUi()
    advance(ui, Start(), FakeBackend(questions=[interview_question()]), COLLECTIONS, 0)
    outcome = advance(ui, Leave("What is ISR?"), FakeBackend(), COLLECTIONS, 0)
    assert outcome.chat == "What is ISR?"
    assert ui.paused
    assert not ui.active


def test_exhaustion_drops_topics_offers_each_and_stop_summarises() -> None:
    ui = InterviewUi()
    backend = FakeBackend(questions=[interview_question(), EXHAUSTED, EXHAUSTED])
    advance(ui, Start(("Kafka", "Redis")), backend, COLLECTIONS, 0)
    advance(ui, Run(Command.skip), backend, COLLECTIONS, 1)
    assert [r.topic for r in backend.question_requests] == ["Kafka", "Redis", "Kafka"]
    session = ui.session
    assert session is not None  # offers are open: the session waits, nothing switched
    assert [o.topic for o in session.offers] == ["Redis", "Kafka"]
    assert all(r.source == "notes" for r in backend.question_requests)
    advance(ui, Run(Command.stop), backend, COLLECTIONS, 1)
    assert ui.session is None
    summary = ui.cards[-1]
    assert isinstance(summary, SummaryCard)
    assert summary.summary.exhausted == ("Redis", "Kafka")
    assert summary.position == 1


@pytest.mark.parametrize(
    ("status", "title"),
    [(409, "Conflict"), (404, "Not Found"), (500, "Internal Server Error")],
)
def test_other_errors_keep_the_session(status: int, title: str) -> None:
    ui = InterviewUi()
    failure = ApiError(Problem(title=title, status=status))
    advance(ui, Start(("Kafka",)), FakeBackend(questions=[failure]), COLLECTIONS, 0)
    assert isinstance(ui.cards[-1], ErrorCard)
    assert ui.active  # still in the mode: "Next question" retries


def test_interview_failure_wording() -> None:
    missing = interview_failure(ApiError(Problem(title="Not Found", status=404)), "http://x")
    assert missing.title == "This API has no interview mode"
    busy = interview_failure(ApiError(Problem(title="Conflict", status=409)), "http://x")
    assert "still finishing" not in busy.title
    down = interview_failure(httpx.ConnectError("no"), "http://x")
    assert "Cannot reach" in down.title


def test_stop_ends_with_the_summary_and_repeat_reshows_the_question() -> None:
    ui = InterviewUi()
    backend = FakeBackend(questions=[interview_question()])
    advance(ui, Start(), backend, COLLECTIONS, 0)
    advance(ui, Run(Command.repeat), backend, COLLECTIONS, 0)
    assert _kinds(ui)[-2:] == ["QuestionCard", "QuestionCard"]
    advance(ui, Run(Command.stop), backend, COLLECTIONS, 0)
    assert isinstance(ui.cards[-1], SummaryCard)
    assert ui.session is None


def test_harder_notice_then_question() -> None:
    ui = InterviewUi()
    backend = FakeBackend(questions=[interview_question(), interview_question()])
    advance(ui, Start(), backend, COLLECTIONS, 0)
    advance(ui, Run(Command.harder), backend, COLLECTIONS, 0)
    notice, question = ui.cards[-2:]
    assert isinstance(notice, NoticeCard)
    assert notice.text == "difficulty: harder (within Senior SDE-1)"
    assert isinstance(question, QuestionCard)
    assert question.number == 2  # the first one was skipped for it


# ------------------------------------------------------------------ routing of typed messages
def test_route_message_reads_the_mode_state() -> None:
    assert route_message(InterviewUi(), "what is ISR?") is None
    assert route_message(InterviewUi(), "quiz me on Kafka") == Start(("Kafka",))
    session = InterviewSession()
    session.asked(interview_question())
    active = InterviewUi(session=session)
    assert route_message(active, "acks=all") == Answer("acks=all")
    assert route_message(active, "skip") == Run(Command.skip)
    session.suspend("What is ISR?")
    assert route_message(active, "stop") is None  # suspended: a chat message
    assert route_message(active, "/answer") == Answer(None)


# ------------------------------------------------------------------ render helpers
def test_bullets_are_one_line_each_and_streamlit_safe() -> None:
    markdown = bullet_markdown(["line one\nline two", "costs $5 :smile:"])
    assert markdown.splitlines()[0] == "- line one line two"
    assert "\\$5" in markdown
    assert "\\:\u2060smile" in markdown


def test_kind_labels_known_and_unknown() -> None:
    assert kind_label("design") == "System design"
    assert kind_label("**puzzle**") == "\\*\\*puzzle\\*\\*"  # unknown: literal text


def test_weak_topics_link_pages_and_escape_titles() -> None:
    session = InterviewSession()
    session.asked(interview_question(topic="[Kafka](javascript:x)", source="Kafka *basics*"))
    session.graded("a", interview_grade(3).model_copy(update={"citations": []}))
    markdown = weak_topics_markdown(session.summary())
    assert "\\[Kafka\\]\\(javascript\\:x\\)" in markdown
    assert "[Kafka \\*basics\\*](https://www.notion.so/Kafka-basics-abc#block1)" in markdown


# ------------------------------------------------------------------ SdkBackend
def test_sdk_backend_posts_interview_calls_with_the_key() -> None:
    seen: list[httpx.Request] = []
    question = interview_question()
    grade = interview_grade(5)

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        body = question if request.url.path.endswith("/question") else grade
        return httpx.Response(200, content=body.model_dump_json())

    backend = SdkBackend(
        "http://api.test", SecretStr("rf_k"), transport=httpx.MockTransport(handler)
    )
    request = InterviewQuestionRequest(collection_ids=COLLECTIONS, topic="Kafka")
    assert backend.interview_question(request) == question
    session = InterviewSession()
    session.asked(question)
    assert backend.interview_grade(session.grade_request(COLLECTIONS, "a")) == grade
    assert [r.url.path for r in seen] == ["/v1/interview/question", "/v1/interview/grade"]
    assert all(r.headers["authorization"] == "Bearer rf_k" for r in seen)
    assert json.loads(seen[0].content)["topic"] == "Kafka"
    assert INTERVIEW_TIMEOUT_S > 30


def test_sdk_backend_raises_api_error_with_the_problem_type() -> None:
    problem = Problem(type=INTERVIEW_EXHAUSTED, title="Exhausted", status=409)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(409, content=problem.model_dump_json())

    backend = SdkBackend("http://api.test", SecretStr("k"), transport=httpx.MockTransport(handler))
    with pytest.raises(ApiError) as caught:
        backend.interview_question(InterviewQuestionRequest(collection_ids=COLLECTIONS))
    assert caught.value.problem.type == INTERVIEW_EXHAUSTED


# ------------------------------------------------------------------ round-1 review fixes
def test_a_read_timeout_says_the_server_is_slow_not_unreachable() -> None:
    slow = interview_failure(httpx.ReadTimeout("slow"), "http://x")
    assert slow.title == f"The server took longer than {INTERVIEW_TIMEOUT_S:.0f} s"
    assert "Wait a moment" in slow.detail
    assert "Cannot reach" in interview_failure(httpx.ConnectTimeout("x"), "http://x").title


def test_sdk_backend_sends_the_interview_timeout_and_a_slow_grade_keeps_the_answer() -> None:
    """Fake transport: both calls carry the 120 s read timeout (the SDK default is 30 s); a
    grade that times out is friendly and leaves "Grade again" with the answer."""
    timeouts: list[float | None] = []

    def handler(request: httpx.Request) -> httpx.Response:
        timeouts.append(request.extensions["timeout"]["read"])
        if request.url.path.endswith("/question"):
            return httpx.Response(200, content=interview_question().model_dump_json())
        raise httpx.ReadTimeout("slow grader", request=request)

    backend = SdkBackend("http://api.test", SecretStr("k"), transport=httpx.MockTransport(handler))
    ui = InterviewUi()
    advance(ui, Start(), backend, COLLECTIONS, 0)
    advance(ui, Answer("acks=all"), backend, COLLECTIONS, 0)
    assert timeouts == [INTERVIEW_TIMEOUT_S, INTERVIEW_TIMEOUT_S]
    error = ui.cards[-1]
    assert isinstance(error, ErrorCard)
    assert "took longer" in error.error.title
    assert ui.unsent_answer == "acks=all"


def test_a_server_escaped_topic_is_escaped_once_not_twice() -> None:
    escaped = "Map\\<K,V> \\& \\[ISR\\]"  # how the server sends "Map<K,V> & [ISR]"
    once = "Map\\<K\\,V\\> \\& \\[ISR\\]"  # renders as Map<K,V> & [ISR]
    assert topic_markdown(escaped) == once
    assert "\\\\" not in once  # never a doubled backslash (that shows as "\\")
    session = InterviewSession()
    session.asked(interview_question(topic=escaped))
    session.graded("a", interview_grade(3))
    assert f"**{once}**" in weak_topics_markdown(session.summary())
