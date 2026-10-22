"""Interview mode on the chat page: the cards, the controls and the flagged-claims block.

Thin, like `views`: what happens lives in `ragforge_ui.interview` (and the shared session in
`ragforge_client.interview`); this file only draws. Buttons never call the API themselves: they
hand a routed step to `on_step`, which queues it and sets the page busy, exactly like a chat
message, so a click can never race a running call.

Text rules (render module docstring): question, grade lists and model answer are server
CommonMark (`for_streamlit` / `answer_markdown`); a question's topic is server CommonMark too,
shown on one line (`topic_markdown`: unescaped once, then escaped, never escaped twice); titles
and errors are server text placed by this UI (`md_literal`); the owner's own answer is plain
text (`st.text`).

General-knowledge cards (Phase 3c) branch on the SOURCE first: they carry the one label
(`GENERAL_LABEL`) at the top, use the ADR-0016 extras look (its own avatar, a bold heading and a
caption saying it is not from the notes) and never show a verified badge, flags or source chips,
so model knowledge can never pass for the owner's notes.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import streamlit as st

from ragforge_client.ask import FlaggedClaims, split_claims
from ragforge_client.interview import (
    DEFAULT_LEVEL,
    GENERAL_GRADE_LABEL,
    GENERAL_LABEL,
    GENERAL_NOTE,
    NO_KEY_NOTE,
    WEAK_SCORE,
    Answer,
    Command,
    General,
    InterviewSession,
    Route,
    Run,
    Start,
    general_grade,
    is_general,
    level_name,
    offer_label,
    split_topics,
)
from ragforge_contracts.api import Citation, InterviewGrade, InterviewLevel, InterviewQuestion
from ragforge_ui.interview import (
    AnswerCard,
    Card,
    ErrorCard,
    GradeCard,
    InterviewUi,
    NoticeCard,
    QuestionCard,
    SummaryCard,
)
from ragforge_ui.render import (
    INEXACT_NOTE,
    INEXACT_TITLE,
    NOT_FOUND_TITLE,
    answer_markdown,
    badge_for,
    bullet_markdown,
    citation_line,
    for_streamlit,
    kind_label,
    md_literal,
    question_level,
    summary_headline_markdown,
    summary_line,
    topic_markdown,
    weak_general_markdown,
    weak_topics_markdown,
)

INTERVIEW = "rf_interview"
TOGGLE = "rf_interview_toggle"
TOPICS = "rf_interview_topics"
LEVEL = "rf_interview_level"  # the start form's level dropdown
SESSION_LEVEL = "rf_interview_session_level"  # the running session's level dropdown
GRADE_IT = "↩ That was my answer — grade it"
LEVELS: tuple[InterviewLevel, ...] = tuple(InterviewLevel)
_GENERAL_AVATAR = ":material/public:"

type OnStep = Callable[[Route], None]


def interview_ui() -> InterviewUi:
    ui = st.session_state.get(INTERVIEW)
    if not isinstance(ui, InterviewUi):
        ui = st.session_state[INTERVIEW] = InterviewUi()
    return ui


# ------------------------------------------------------------------ flagged claims (chat too)
def draw_flags(flagged: FlaggedClaims) -> None:
    """Owner decision (2026-10-18): what the notes do NOT support is red and in plain view;
    what only failed an exact code/format match is muted and collapsed. Showing both the same
    way trained the owner to ignore every warning."""
    if flagged.not_found:
        st.error(
            f"**{NOT_FOUND_TITLE}**\n\n{bullet_markdown(flagged.not_found)}",
            icon=":material/report:",
        )
    if flagged.inexact:
        with st.expander(f"{INEXACT_TITLE} · {len(flagged.inexact)}", expanded=False):
            st.caption(INEXACT_NOTE)
            st.caption(bullet_markdown(flagged.inexact))


# ------------------------------------------------------------------ cards
def draw_cards(cards: Sequence[Card], position: int) -> None:
    """The cards made when `position` chat turns existed (module docstring of `interview`)."""
    for card in cards:
        if card.position == position:
            _draw_card(card)


def _draw_card(card: Card) -> None:
    match card:
        case QuestionCard():
            _draw_question(card)
        case AnswerCard(text=text):
            with st.chat_message("user"):
                st.caption("Your answer")
                st.text(text)
        case GradeCard():
            _draw_grade(card)
        case NoticeCard(text=text):
            st.info(md_literal(text), icon=":material/school:")
        case ErrorCard(error=error):
            st.error(f"**{md_literal(error.title)}.** {md_literal(error.detail)}")
        case SummaryCard():
            _draw_summary(card)


def _general_heading(caption: str) -> None:
    """The ADR-0016 extras look: the label in bold, then what it means."""
    st.markdown(f"**{GENERAL_LABEL}**")
    st.caption(caption)


def _draw_question(card: QuestionCard) -> None:
    """Its sources are NOT shown here: page titles before answering are a hint. They appear on
    the grade card (a general question has none)."""
    question = card.question
    general = is_general(question)
    avatar = _GENERAL_AVATAR if general else ":material/school:"
    with st.chat_message("assistant", avatar=avatar), st.container(border=True):
        if general:
            _general_heading(GENERAL_NOTE)
        st.markdown(f"**Question {card.number}** · {topic_markdown(question.topic)}")
        st.badge(kind_label(question.kind), color="gray" if general else "blue")
        st.caption(f"Level: {question_level(question)}")
        st.markdown(for_streamlit(question.question))


def _draw_grade(card: GradeCard) -> None:
    """Branches on the grade's SOURCE first (design review): a general grade never shows the
    verifier badge texts, flags or sources."""
    if general_grade(card.grade, card.question):
        _draw_general_grade(card.grade, card.question)
    else:
        _draw_notes_grade(card.grade, card.question)


def _draw_notes_grade(grade: InterviewGrade, question: InterviewQuestion) -> None:
    """Score, then the judgement and the three lists (labelled by who decided them: the summary
    and "Right" are the grading model's judgement; "Missed" and "Misconceptions" are checked
    against the notes by the server), the answer key, then the model answer with its verdict
    and sources."""
    with st.chat_message("assistant", avatar=":material/grading:"), st.container(border=True):
        st.metric("Score", f"{grade.score}/10")
        st.caption("The grader's judgement")
        st.markdown(for_streamlit(grade.summary))
        _draw_list("Right", "the grader's judgement", grade.right)
        _draw_list("Missed", "checked against your notes", grade.missed)
        _draw_list("Misconceptions", "checked against your notes", grade.misconceptions)
        if question.expected_points:
            _draw_list("Answer key", "checked against your notes", question.expected_points)
        else:
            st.caption(f"{NO_KEY_NOTE}.")
        # neutral heading: the verdict badge below says how well the notes back it
        st.markdown("**Model answer**")
        st.markdown(answer_markdown(grade.model_answer))
        badge = badge_for(
            str(grade.verification),
            cited=bool(grade.citations),
            flagged=bool(grade.unsupported_claims),
        )
        if badge is not None:
            st.badge(badge.label, icon=badge.icon, color=badge.color)  # type: ignore[arg-type]
            st.caption(badge.meaning)
        draw_flags(split_claims(grade.unsupported_claims, []))
        _draw_sources("Sources from your notes", grade.citations)
        _draw_sources("This question came from", question.citations)


def _draw_general_grade(grade: InterviewGrade, question: InterviewQuestion) -> None:
    with st.chat_message("assistant", avatar=_GENERAL_AVATAR), st.container(border=True):
        _general_heading(GENERAL_GRADE_LABEL)
        st.metric("Score", f"{grade.score}/10")
        st.caption("The grader's judgement")
        st.markdown(for_streamlit(grade.summary))
        _draw_list("Right", "the grader's judgement", grade.right)
        _draw_list("Missed", "against the model's own key", grade.missed)
        _draw_list("Misconceptions", "against the model's own key", grade.misconceptions)
        _draw_list("The model's answer key", "not verified", question.expected_points)
        st.markdown(f"**Model answer** · *{GENERAL_LABEL}*")
        st.markdown(answer_markdown(grade.model_answer))


def _draw_list(title: str, who: str, items: Sequence[str]) -> None:
    if items:
        st.markdown(f"**{title}** · *{who}*\n\n{bullet_markdown(items)}")


def _draw_sources(title: str, citations: Sequence[Citation]) -> None:
    if citations:
        lines = [f"- {citation_line(c, i)}" for i, c in enumerate(citations, start=1)]
        st.markdown(f"**{title}**\n\n" + "\n".join(lines))


def _draw_summary(card: SummaryCard) -> None:
    """Notes-based and general-knowledge results apart (owner decision): never one average."""
    summary = card.summary
    with st.chat_message("assistant", avatar=":material/summarize:"), st.container(border=True):
        st.markdown("**Interview summary**")
        st.markdown(summary_headline_markdown(summary))
        st.caption(summary_line(summary))
        if summary.weak:
            st.markdown(
                f"**Weak topics from your notes** ({WEAK_SCORE}/10 or less) · pages to revisit"
            )
            st.markdown(weak_topics_markdown(summary))
        elif summary.notes_answered:
            st.caption("No weak topics from your notes: every graded answer scored above 6/10.")
        if summary.general_answered:
            with st.container(border=True):
                _general_heading(
                    f"{summary.general_answered} graded from the model's own knowledge, not "
                    "from your notes; its average is kept apart."
                )
                if summary.weak_general:
                    st.markdown(weak_general_markdown(summary))
        if summary.exhausted:
            covered = ", ".join(md_literal(t) for t in summary.exhausted)
            st.caption(f"Fully covered (no new notes left): {covered}")


# ------------------------------------------------------------------ controls
def draw_controls(
    ui: InterviewUi, *, busy: bool, on_step: OnStep, has_sources: bool = True
) -> None:
    """Under the last card: the start form, the session buttons, or the paused bar. Without a
    selected source every control is disabled with a note: a click would otherwise be queued
    and dropped with nothing to ask from (round-1 review, P2)."""
    if ui.session is None and not ui.form_open:
        return
    if not has_sources:
        st.info("Pick at least one source in the sidebar to use interview mode.")
    busy = busy or not has_sources
    if ui.session is None:
        _draw_start_form(busy=busy, on_step=on_step)
        return
    session = ui.session
    if session.paused:
        _draw_paused(ui, busy=busy, on_step=on_step)
        return
    _draw_offers(session, busy=busy, on_step=on_step)
    pending = session.current is not None
    if ui.unsent_answer is not None and pending:
        st.button(
            "Grade again",
            key="rf_iv_regrade",
            icon=":material/replay:",
            type="primary",
            on_click=on_step,
            args=(Answer(ui.unsent_answer),),
            disabled=busy,
        )
    buttons: list[tuple[str, Command, str]] = [
        ("Skip", Command.skip, ":material/skip_next:")
        if pending
        else ("Next question", Command.next, ":material/arrow_forward:"),
        ("Harder", Command.harder, ":material/trending_up:"),
        ("Easier", Command.easier, ":material/trending_down:"),
        ("Next topic", Command.next_topic, ":material/swap_horiz:"),
        ("Stop", Command.stop, ":material/stop_circle:"),
    ]
    for column, (label, command, icon) in zip(st.columns(len(buttons)), buttons, strict=True):
        column.button(
            label,
            key=f"rf_iv_{command.name}",
            icon=icon,
            on_click=on_step,
            args=(Run(command),),
            disabled=busy,
            width="stretch",
        )
    _draw_session_level(session, busy=busy, on_step=on_step)


def _draw_offers(session: InterviewSession, *, busy: bool, on_step: OnStep) -> None:
    """One button per open offer (per topic). Only this click moves a topic to general
    knowledge (owner decision: never automatic)."""
    for index, offer in enumerate(session.offers):
        st.button(
            md_literal(offer_label(offer.topic)),
            key=f"rf_iv_general_{index}",
            icon=_GENERAL_AVATAR,
            on_click=on_step,
            args=(General(offer.topic),),
            disabled=busy,
            help="Questions and grading from the model's own knowledge. Nothing is checked.",
        )


def _draw_session_level(session: InterviewSession, *, busy: bool, on_step: OnStep) -> None:
    """The running session's level. It is fixed per session (one average, one bar), so a change
    STARTS A NEW interview on the same topics; the caption says so before anyone clicks. The
    widget mirrors the session (set before it is drawn, like the toggle)."""
    st.session_state[SESSION_LEVEL] = session.level
    st.selectbox(
        "Level",
        LEVELS,
        key=SESSION_LEVEL,
        format_func=level_name,
        on_change=_restart_at_level,
        args=(session, on_step),
        disabled=busy,
        help="Fixed for this interview: changing it starts a new one.",
    )
    st.caption(
        f"Level {level_name(session.level)} is fixed for this interview: changing it "
        "starts a new one."
    )


def _restart_at_level(session: InterviewSession, on_step: OnStep) -> None:
    level = st.session_state.get(SESSION_LEVEL)
    if isinstance(level, InterviewLevel) and level is not session.level:
        on_step(Start(tuple(session.named_topics), level=level))


def _draw_start_form(*, busy: bool, on_step: OnStep) -> None:
    with st.container(border=True):
        st.markdown("**Interview mode** · one question at a time, graded against your notes")
        topic, level = st.columns([3, 1])
        topic.text_input(
            "Topic(s), optional",
            key=TOPICS,
            placeholder="e.g. Kafka, Redis (empty: mixed across all your notes)",
            max_chars=400,
            disabled=busy,
        )
        level.selectbox(
            "Level",
            LEVELS,
            index=LEVELS.index(DEFAULT_LEVEL),
            key=LEVEL,
            format_func=level_name,
            disabled=busy,
            help="The bar questions are asked and graded at. Fixed for the interview.",
        )
        st.button(
            "Start interview",
            key="rf_iv_start",
            type="primary",
            icon=":material/play_arrow:",
            on_click=_start_from_form,
            args=(on_step,),
            disabled=busy,
        )


def _start_from_form(on_step: OnStep) -> None:
    raw = st.session_state.get(TOPICS)
    level = st.session_state.get(LEVEL)
    on_step(
        Start(
            split_topics(raw if isinstance(raw, str) else ""),
            level=level if isinstance(level, InterviewLevel) else DEFAULT_LEVEL,
        )
    )


def _draw_paused(ui: InterviewUi, *, busy: bool, on_step: OnStep) -> None:
    """A normal question suspended the interview. It may have been a hedged answer ("Is it
    because …?"), so grading it is one click (design review: leaving is reversible)."""
    session = ui.session
    if session is None:
        return
    with st.container(border=True):
        st.caption("Interview paused: your last message went to your notes as a normal question.")
        columns = st.columns(3)
        if session.left_message is not None and session.current is not None:
            columns[0].button(
                GRADE_IT,
                key="rf_iv_grade_it",
                on_click=on_step,
                args=(Answer(None),),
                disabled=busy,
                type="primary",
            )
        columns[1].button(
            "Back to the question" if session.current is not None else "Back to the interview",
            key="rf_iv_resume",
            icon=":material/undo:",
            on_click=on_step,
            args=(Run(Command.resume),),
            disabled=busy,
        )
        columns[2].button(
            "End interview",
            key="rf_iv_end",
            icon=":material/stop_circle:",
            on_click=on_step,
            args=(Run(Command.stop),),
            disabled=busy,
        )


def input_placeholder(ui: InterviewUi) -> str:
    if ui.active and ui.session is not None and ui.session.current is not None:
        return "Your answer… (or: repeat, skip, harder, easier, next topic, stop)"
    if ui.active and ui.session is not None and ui.session.offers:
        return "/general takes the offer · next, next topic, stop… (or ask your notes)"
    if ui.active:
        return "next, harder, easier, next topic, stop… (or ask your notes)"
    return "Ask your notes… (or: interview me on <topic>)"
