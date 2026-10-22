"""Interview mode in the UI: the page's interview state and the one function that advances it.

No Streamlit here (same split as `state`): the page queues a routed message (`Route`, from the
shared `ragforge_client.interview` rules, so the UI and `make ask` read a message the same way),
then calls `advance`, which drives the shared `InterviewSession`, calls the API through the
synchronous `InterviewBackend`, and appends what to show as `Card`s. Tests drive it with a fake.

Where cards go: the chat turns and the interview are one page. A card remembers how many chat
turns existed when it was made (`position`), and the page draws it right after them, so a
question, a Q&A detour (a "normal question" suspends the interview) and the grade read in the
order they happened.

Failures never lose the session: a failed fetch leaves no question pending ("Next question"
retries), a failed grade keeps the question AND the answer (`unsent_answer`, "Grade again").

Phase 3c: a topic the notes have nothing (left) on opens an `Offer` in the shared session; the
page draws a one-click button for it, and only that click (a `General` step) moves the topic to
general knowledge. A start naming another level (`Do.restart`) ends this session with its
summary card and begins a new one: the level is fixed per session.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol
from uuid import UUID

import httpx

from ragforge_client.client import ApiError
from ragforge_client.interview import (
    INTERVIEW_TIMEOUT_S,
    Do,
    InterviewSession,
    NoQuestion,
    Route,
    Start,
    Summary,
    describe_topics,
    fallback_reason,
    level_name,
    unknown_level_notice,
)
from ragforge_contracts.api import (
    InterviewGrade,
    InterviewGradeRequest,
    InterviewQuestion,
    InterviewQuestionRequest,
)
from ragforge_ui.errors import UiError, describe_failure


class InterviewBackend(Protocol):
    """The two calls interview mode needs (`api.SdkBackend` has them; tests fake them)."""

    @property
    def api_url(self) -> str: ...

    def interview_question(self, request: InterviewQuestionRequest) -> InterviewQuestion: ...

    def interview_grade(self, request: InterviewGradeRequest) -> InterviewGrade: ...


# ------------------------------------------------------------------ cards
@dataclass(frozen=True, slots=True)
class QuestionCard:
    position: int
    question: InterviewQuestion
    number: int


@dataclass(frozen=True, slots=True)
class AnswerCard:
    position: int
    text: str  # the owner's own words: shown as plain text


@dataclass(frozen=True, slots=True)
class GradeCard:
    position: int
    grade: InterviewGrade
    question: InterviewQuestion  # its sources are revealed only now, after the answer


@dataclass(frozen=True, slots=True)
class NoticeCard:
    position: int
    text: str  # written by this UI (topic names inside are escaped when drawn)


@dataclass(frozen=True, slots=True)
class ErrorCard:
    position: int
    error: UiError


@dataclass(frozen=True, slots=True)
class SummaryCard:
    position: int
    summary: Summary


type Card = QuestionCard | AnswerCard | GradeCard | NoticeCard | ErrorCard | SummaryCard


@dataclass
class InterviewUi:
    """Everything interview mode keeps in Streamlit's session state.

    `session` None: no interview (never started, or ended: the summary card is the last word).
    `form_open`: the toggle is on but no interview started yet (the topic box is showing).
    `unsent_answer`: an answer whose grading failed, so "Grade again" can resend it unchanged."""

    session: InterviewSession | None = None
    cards: list[Card] = field(default_factory=list)
    form_open: bool = False
    unsent_answer: str | None = None

    @property
    def active(self) -> bool:
        """In the mode: messages are answers and commands, not Q&A."""
        return self.session is not None and not self.session.paused

    @property
    def paused(self) -> bool:
        return self.session is not None and self.session.paused


@dataclass(frozen=True, slots=True)
class Outcome:
    """What the page does after `advance`. `chat`: send this message to normal Q&A (the
    interview was suspended by it). `called`: an API call was made (the page reruns)."""

    chat: str | None = None
    called: bool = False


# ------------------------------------------------------------------ advancing
def advance(
    ui: InterviewUi,
    step: Route,
    backend: InterviewBackend,
    collections: Sequence[UUID],
    position: int,
) -> Outcome:
    """Apply one routed message (a typed message, a button) and make the API calls it needs."""
    if ui.session is None:
        if not isinstance(step, Start):
            return Outcome()  # a stale button from an interview that already ended
        _fetch(ui, _begin(ui, step, position), backend, collections, position)
        return Outcome(called=True)
    session = ui.session
    action = session.apply(step)
    if action.notice:
        ui.cards.append(NoticeCard(position, action.notice))
    match action.do:
        case Do.chat:
            return Outcome(chat=action.text)
        case Do.end:
            _end(ui, position)
        case Do.restart:  # another level: this session ends (summary), a new one begins
            _end(ui, position)
            return advance(ui, step, backend, collections, position)
        case Do.show if session.current is not None:
            ui.cards.append(QuestionCard(position, session.current, session.number))
        case Do.ask:
            ui.unsent_answer = None  # a new question: the failed answer was for the old one
            _fetch(ui, session, backend, collections, position)
            return Outcome(called=True)
        case Do.grade:
            _grade(ui, session, action.text, backend, collections, position)
            return Outcome(called=True)
    return Outcome()


def _begin(ui: InterviewUi, start: Start, position: int) -> InterviewSession:
    session = ui.session = InterviewSession.begin(start)
    ui.form_open = False
    started = (
        f"Interview started · level: {level_name(session.level)} (fixed for this interview; "
        f"changing it starts a new one) · topic: {describe_topics(start.topics)}"
    )
    ui.cards.append(NoticeCard(position, started))
    unknown = unknown_level_notice(start, session.level)
    if unknown:
        ui.cards.append(NoticeCard(position, unknown))
    return session


def _fetch(
    ui: InterviewUi,
    session: InterviewSession,
    backend: InterviewBackend,
    collections: Sequence[UUID],
    position: int,
) -> None:
    """Fetch the next question. A topic the notes have nothing (left) on (by Problem TYPE,
    never status) is dropped, its general-knowledge offer opened (the controls draw it; nothing
    switches by itself) and the next topic tried. With only offers left, the page waits for a
    click; with nothing left at all the interview ends with its summary."""
    for _ in range(len(session.topics) + 1):  # bounded: each pass drops a topic or returns
        try:
            question = backend.interview_question(session.question_request(collections))
        except ApiError as exc:
            if fallback_reason(exc.problem) is None:
                ui.cards.append(ErrorCard(position, interview_failure(exc, backend.api_url)))
                return
            outcome = session.exhausted(exc.problem)
            ui.cards.append(NoticeCard(position, outcome.notice))
            if outcome.ended:
                _end(ui, position)
                return
            if not session.topics:  # only offers are left: they wait for the owner
                return
            continue
        except (httpx.HTTPError, ValueError) as exc:
            ui.cards.append(ErrorCard(position, interview_failure(exc, backend.api_url)))
            return
        session.asked(question)
        ui.cards.append(QuestionCard(position, question, session.number))
        return


def _grade(
    ui: InterviewUi,
    session: InterviewSession,
    answer: str,
    backend: InterviewBackend,
    collections: Sequence[UUID],
    position: int,
) -> None:
    question = session.current
    try:
        if question is None:
            raise NoQuestion
        request = session.grade_request(collections, answer)
    except NoQuestion:
        ui.cards.append(NoticeCard(position, 'No question is pending: press "Next question".'))
        return
    except ValueError as exc:
        ui.cards.append(ErrorCard(position, UiError("Answer not sent", f"Because {exc}.")))
        return
    ui.cards.append(AnswerCard(position, request.answer))
    try:
        grade = backend.interview_grade(request)
    except (ApiError, httpx.HTTPError, ValueError) as exc:
        ui.unsent_answer = request.answer
        failure = interview_failure(exc, backend.api_url)
        kept = f"{failure.detail} Your answer and the question are kept: press Grade again."
        ui.cards.append(ErrorCard(position, UiError(failure.title, kept.strip())))
        return
    ui.unsent_answer = None
    session.graded(request.answer, grade)
    ui.cards.append(GradeCard(position, grade, question))


def _end(ui: InterviewUi, position: int) -> None:
    session = ui.session
    if session is not None:
        ui.cards.append(SummaryCard(position, session.summary()))
    ui.session = None
    ui.unsent_answer = None


def interview_failure(exc: BaseException, api_url: str) -> UiError:
    """`errors.describe_failure`, except where its chat wording would mislead here: a 404 means
    an API without interview mode (exhaustion is decided by Problem type before this), a plain
    409 is not "the previous answer is still finishing", and a read timeout means the server is
    still working (it charges the interview budget when it finishes), so: wait, then retry."""
    if isinstance(exc, ApiError) and exc.status == 404:
        return UiError(
            "This API has no interview mode",
            "The server may be older than this UI. Update and restart the stack (make up).",
        )
    if isinstance(exc, ApiError) and exc.status == 409:
        problem = exc.problem
        return UiError(problem.title, problem.detail or "Try again in a moment.")
    if isinstance(exc, httpx.TimeoutException) and not isinstance(exc, httpx.ConnectTimeout):
        return UiError(
            f"The server took longer than {INTERVIEW_TIMEOUT_S:.0f} s",
            "It may still be working. Wait a moment, then try again.",
        )
    return describe_failure(exc, api_url)
