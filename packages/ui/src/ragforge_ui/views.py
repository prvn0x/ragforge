"""The Streamlit pages. Kept thin: decisions live in `state` and `render`, which are unit-tested.

Flow of a chat turn: the user's message (or a queued "show the code", or an approval decision)
is queued by a widget callback, which also sets the busy flag; the script then reruns with every
input disabled, starts a `Turn`, and folds the stream into it event by event while a placeholder
shows progress. When the stream ends the script reruns, and every turn, old or new, is drawn by
`draw_turn`. One drawing path means a fresh answer and a re-drawn one can never look different.

Interview mode shares the page and the busy/queue discipline: a typed message is routed by the
shared rules (`ragforge_client.interview`: a start phrase, a session command, an answer, or a
normal question that suspends the interview and goes to Q&A), buttons queue a routed step, and
`ragforge_ui.interview.advance` runs it on the next script run. Its cards sit between the chat
turns in the order they happened.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterator
from typing import Any
from uuid import UUID

import streamlit as st

from ragforge_client.ask import split_claims
from ragforge_client.client import ApiError
from ragforge_client.interview import Command, Route, Run, detect_start, route, route_suspended
from ragforge_contracts.api import ApprovalDecision, Collection, Connector, Job
from ragforge_contracts.sse import AnswerReplaceEvent, ChatEvent, TokenEvent, VerificationEvent
from ragforge_ui.api import Backend, drain
from ragforge_ui.errors import UiError, describe_failure
from ragforge_ui.interview import InterviewUi, advance
from ragforge_ui.interview_views import (
    INTERVIEW,
    TOGGLE,
    draw_cards,
    draw_controls,
    draw_flags,
    input_placeholder,
    interview_ui,
)
from ragforge_ui.render import (
    EXTRAS_HEADING,
    INFERRED_NOTE,
    answer_markdown,
    badge_for,
    call_summary,
    citation_line,
    collection_labels,
    extra_view,
    md_literal,
    preview_text,
    timings_text,
)
from ragforge_ui.state import (
    SHOW_THE_CODE,
    Conversation,
    Interrupt,
    PendingCall,
    Turn,
    cited_titles,
    decide,
    foreign_run,
    in_flight,
    missing_titles,
    offers_code,
    open_interrupt,
    record,
    resume_prompt,
    settle_question,
    settle_resume,
    start_turn,
)

CONVERSATION = "rf_conversation"
QUEUED_PROMPT = "rf_queued_prompt"
QUEUED_DECISIONS = "rf_queued_decisions"
QUEUED_STEP = "rf_queued_interview_step"  # a routed interview step from a button
COLLECTIONS = "rf_collections"
SELECTED = "rf_selected_collections"
JOBS = "rf_jobs"  # [(job id, connector id)], newest first: syncs started in this session
SYNCING = "rf_syncing"  # connector ids with a sync in flight, as of the last Sources draw
SYNC_NOTICE = "rf_sync_notice"  # what the last "Sync now" click did, shown once
RUN_ID_PARAM = "run_id"  # ?run_id=<uuid>: an approval link (MCP's, or this page's own)
CHAT_INPUT = "rf_chat_input"
# True from the click (or submit) that starts a turn until its stream ends. Every widget that
# could rerun the script is drawn disabled meanwhile: a rerun mid-stream would cut the turn off
# before `done`, i.e. before the client learns the conversation id (SSE contract: no next
# message before `done`; the server holds the conversation lease until then).
BUSY = "rf_busy"
MAX_MESSAGE_CHARS = 8000  # contracts ChatRequest.message max_length
_MAX_COLLECTIONS = 20  # ChatRequest.collection_ids max_length


def is_busy() -> bool:
    return bool(st.session_state.get(BUSY, False))


# ================================================================== chat page
def chat_page(backend: Backend) -> None:
    conversation = _conversation()
    interview = interview_ui()
    busy = is_busy()
    collections = _collections(backend, busy=busy)
    selected = _pick_collections(collections, busy=busy)
    _interview_toggle(interview, busy=busy)
    if st.sidebar.button("New conversation", icon=":material/add_comment:", disabled=busy):
        st.session_state[CONVERSATION] = conversation = Conversation()
        # a clean page: an interview in progress ends without a summary (its cards are gone)
        st.session_state[INTERVIEW] = interview = InterviewUi()
        st.query_params.pop(RUN_ID_PARAM, None)  # the owner left that approval behind

    if not conversation.turns and not interview.cards:
        st.caption(
            "Ask about anything in your Notion notes. Answers are checked against them. "
            'Say "interview me on Kafka" (or use Interview mode) to practise.'
        )
    actionable = open_interrupt(conversation)
    _approval_link(conversation, actionable, busy=busy)
    if actionable is not None:
        _resolve_titles(backend, conversation, actionable)
    for index, turn in enumerate(conversation.turns):
        draw_cards(interview.cards, index)
        draw_turn(
            turn,
            latest=index == len(conversation.turns) - 1 and not _cards_after(interview, index),
            busy=busy,
            actionable=turn.interrupt is not None and turn.interrupt is actionable,
        )
    draw_cards(interview.cards, len(conversation.turns))
    draw_controls(interview, busy=busy, on_step=_queue_step, has_sources=bool(selected))

    # submit_mode="disable": the browser greys the box out the moment the owner presses Enter,
    # before this script even reruns; `disabled=busy` covers turns started by a button.
    st.chat_input(
        input_placeholder(interview),
        key=CHAT_INPUT,
        disabled=busy or not selected,
        submit_mode="disable",
        max_chars=MAX_MESSAGE_CHARS,  # the API's ChatRequest limit: refuse it here, not as a 422
        on_submit=_on_submit,
    )
    streamed = False
    try:
        streamed = _run_queued(backend, conversation, interview, selected)
    finally:
        st.session_state[BUSY] = False  # also when Streamlit stops the script mid-stream
    if streamed or busy:
        st.rerun()  # draw the finished turn the one way every turn is drawn, inputs enabled


def _run_queued(
    backend: Backend, conversation: Conversation, interview: InterviewUi, selected: list[UUID]
) -> bool:
    """Start the queued work (approval decisions win over an interview step, which wins over a
    prompt). True if anything ran (the page then reruns to draw it the usual way)."""
    decisions = st.session_state.pop(QUEUED_DECISIONS, None)
    interrupt = open_interrupt(conversation)
    if decisions is not None and interrupt is not None:
        _stream_resume(backend, conversation, interrupt, decisions)
        return True
    step: Route | None = st.session_state.pop(QUEUED_STEP, None)
    prompt = st.session_state.pop(QUEUED_PROMPT, None)
    if not selected:
        return False
    if step is None and prompt:
        step = route_message(interview, prompt)
        if step is None and interview.paused and interview.session is not None:
            interview.session.chatted()  # Q&A goes on: "grade it" no longer means that message
    if step is not None:
        with st.spinner("Working with your notes…"):
            outcome = advance(interview, step, backend, selected, len(conversation.turns))
        if outcome.chat is None:
            return True
        prompt = outcome.chat  # a normal question: the interview is suspended, Q&A answers it
    if not prompt:
        return False
    _stream_question(backend, conversation, prompt, selected)
    return True


def route_message(interview: InterviewUi, message: str) -> Route | None:
    """What a typed message means to the interview; None = ordinary Q&A. The same rules as
    `make ask` (`ragforge_client.interview`): outside the mode only a start phrase counts."""
    session = interview.session
    if session is None:
        return detect_start(message)
    if session.paused:
        return route_suspended(message)
    return route(message, pending=session.current is not None, offering=session.offering)


def _cards_after(interview: InterviewUi, index: int) -> bool:
    """Interview cards drawn after chat turn `index`: that turn is then no longer the latest."""
    return any(card.position > index for card in interview.cards)


def _stream_question(
    backend: Backend, conversation: Conversation, prompt: str, selected: list[UUID]
) -> None:
    turn = start_turn(conversation, prompt)
    try:
        _stream(
            backend, conversation, turn, lambda: backend.chat(prompt, selected, conversation.id)
        )
    finally:
        settle_question(conversation, turn)  # a question that never reached the server leaves
        # an open approval decidable


def _stream_resume(
    backend: Backend,
    conversation: Conversation,
    interrupt: Interrupt,
    decisions: dict[str, ApprovalDecision],
) -> None:
    turn = start_turn(conversation, resume_prompt(interrupt, decisions), kind="resume")
    try:
        _stream(backend, conversation, turn, lambda: backend.resume(interrupt.run_id, decisions))
    finally:
        settle_resume(interrupt, decisions, turn)  # a resume that never reached the server
        # leaves Approve/Reject in place for a retry


def _stream(
    backend: Backend,
    conversation: Conversation,
    turn: Turn,
    open_stream: Callable[[], Iterator[ChatEvent]],
    clock: Callable[[], float] = time.monotonic,
) -> None:
    """Stream one turn into `turn`, showing progress in a placeholder.

    Failures land on the turn as a `UiError` (the conversation above stays). A Streamlit stop or
    rerun mid-stream raises a BaseException subclass, which is deliberately NOT caught: the turn
    then stays unfinished and is drawn as cut off."""
    _draw_user(turn)
    with st.chat_message("assistant"):
        placeholder = st.empty()
        placeholder.caption("Searching your notes…")
        started = clock()

        def handle(event: ChatEvent) -> None:
            record(conversation, turn, event, clock() - started)
            _show_progress(placeholder, turn, event)

        try:
            drain(open_stream(), handle)
        except Exception as exc:
            # After `verification` the answer and its check are committed (DESIGN section 8):
            # only the extras can be missing, and "send it again" would duplicate the turn.
            turn.error = (
                UiError(
                    "The stream stopped after the answer was checked",
                    "The answer and its check above are complete and saved; general-knowledge "
                    "extras, if any, are missing. No need to send it again.",
                )
                if turn.verdict_seen
                else describe_failure(exc, backend.api_url)
            )
        turn.total_s = clock() - started
        turn.finished = True


def _show_progress(placeholder: Any, turn: Turn, event: ChatEvent) -> None:
    if isinstance(event, TokenEvent | AnswerReplaceEvent | VerificationEvent) and turn.answer:
        if turn.verdict_seen:
            placeholder.markdown(answer_markdown(turn.answer))
        else:  # SSE contract: plain text until `verification`
            placeholder.text(preview_text(turn.answer))


def _on_submit() -> None:
    """Runs before the script reruns, so that very run already draws every input disabled."""
    prompt = st.session_state.get(CHAT_INPUT)
    if isinstance(prompt, str) and prompt.strip():
        _queue_prompt(prompt)


# ------------------------------------------------------------------ one turn
def draw_turn(turn: Turn, *, latest: bool, busy: bool = False, actionable: bool = False) -> None:
    """Order matters (ADR-0016): answer, verdict, unsupported claims, citations, then chrome,
    and the general-knowledge extras LAST, in their own bordered box, so nothing from the notes
    (and no verdict) ever sits beneath text that isn't from the notes.

    `actionable`: this turn's interrupt is the one the owner can still decide (Approve/Reject)."""
    _draw_user(turn)
    with st.chat_message("assistant"):
        _draw_answer(turn)
        _draw_verdict(turn)
        _draw_citations(turn)
        _draw_tools(turn)
        if turn.interrupt is not None:
            _draw_interrupt(turn.interrupt, actionable=actionable, busy=busy)
        _draw_error(turn.error)
        _draw_cut_off(turn)
        if latest and offers_code(turn):
            st.button(
                "Show the code",
                icon=":material/code:",
                on_click=_queue_prompt,
                args=(SHOW_THE_CODE,),
                disabled=busy,
                help="Ask for a code example of this answer as the next turn.",
            )
        if turn.finished:
            st.caption(timings_text(turn.first_token_s, turn.total_s))
        _draw_extras(turn)


def _draw_user(turn: Turn) -> None:
    with st.chat_message("user"):
        if turn.kind == "resume":
            st.caption("Your decision")
        st.text(turn.prompt)  # plain text: the user's words are shown exactly as typed


def _draw_answer(turn: Turn) -> None:
    """Markdown only once the server's verdict arrived (SSE contract: plain text until
    `verification`). A draft cut off or failed BEFORE its verdict was never checked and never
    saved (DESIGN section 8): it is shown as plain, muted text under that label, never as a
    formatted answer (integrated review P1: a 429 mid-stream left a bold "answer" behind)."""
    if not turn.answer:
        return
    if turn.verdict_seen:
        st.markdown(answer_markdown(turn.answer))
        if turn.revised:
            st.caption("This answer was rewritten after verification.")
        return
    # every answer the server saves carries a verdict, so no verdict means an unchecked draft
    st.caption("Unverified draft: it stopped before it was checked, and was not saved.")
    st.text(preview_text(turn.answer))


def _draw_cut_off(turn: Turn) -> None:
    """A turn the page stopped streaming (tab reload, a rerun). After `verification` the answer
    and its check are complete and the server has saved them; only the extras can be missing."""
    if turn.finished:
        return
    if turn.verdict_seen:
        st.caption(
            "The page stopped before the end: the answer and its check above are complete;"
            " general-knowledge extras, if any, were cut off."
        )
    else:
        st.caption("This answer was interrupted before it finished. Send it again.")


def _draw_verdict(turn: Turn) -> None:
    # an unverified turn that cited nothing (a refusal) had nothing to check: `badge_for`
    badge = badge_for(
        turn.status,
        cited=bool(turn.citations),
        flagged=bool(turn.claims or turn.unsupported_claims),
    )
    if badge is None:
        return
    st.badge(badge.label, icon=badge.icon, color=badge.color)  # type: ignore[arg-type]
    st.caption(badge.meaning)
    draw_flags(split_claims(turn.claims, turn.unsupported_claims))


def _draw_citations(turn: Turn) -> None:
    if not turn.citations:
        return
    lines = [f"- {citation_line(c, i)}" for i, c in enumerate(turn.citations, start=1)]
    st.markdown("**Sources from your notes**\n\n" + "\n".join(lines))
    if any(c.inferred for c in turn.citations):
        st.caption(INFERRED_NOTE)


def _draw_tools(turn: Turn) -> None:
    if turn.tools:
        st.caption("Used: " + ", ".join(md_literal(t) for t in turn.tools))


def _draw_interrupt(interrupt: Interrupt, *, actionable: bool, busy: bool) -> None:
    if interrupt.expired:
        st.caption("These actions were not approved before the next question; they expired.")
        return
    if interrupt.decisions is not None or not actionable:
        return  # the resume turn below shows the decision
    with st.container(border=True):
        st.markdown("**Waiting for your approval** — these actions change your index:")
        several = len(interrupt.pending) > 1
        for call in interrupt.pending:
            st.markdown(f"**{md_literal(call.tool)}**")
            summary = call_summary(call, interrupt.titles or {})
            if summary is not None:
                st.markdown(summary)
            st.code(json.dumps(call.args, indent=2, ensure_ascii=False), language="json")
            if several:  # one choice per call: an injected call can be rejected on its own
                st.checkbox("Approve this one", key=_approval_key(interrupt, call), disabled=busy)
        approve, reject = st.columns(2)
        approve.button(
            "Approve checked, reject the rest" if several else "Approve",
            type="primary",
            icon=":material/check:",
            on_click=_queue_checked if several else _queue_decisions,
            args=(interrupt,)
            if several
            else (decide(interrupt, {interrupt.pending[0].tool_call_id}),),
            disabled=busy,
        )
        reject.button(
            "Reject all" if several else "Reject",
            icon=":material/close:",
            on_click=_queue_decisions,
            args=(decide(interrupt, set()),),
            disabled=busy,
        )


def _approval_key(interrupt: Interrupt, call: PendingCall) -> str:
    return f"rf_approve-{interrupt.run_id}-{call.tool_call_id}"


def _queue_checked(interrupt: Interrupt) -> None:
    """Read the checkboxes at click time; anything not explicitly checked is rejected."""
    checked = {
        call.tool_call_id
        for call in interrupt.pending
        if st.session_state.get(_approval_key(interrupt, call)) is True
    }
    _queue_decisions(decide(interrupt, checked))


def _approval_link(conversation: Conversation, actionable: Interrupt | None, *, busy: bool) -> None:
    """Keep `?run_id=` honest.

    The open approval's run goes into the URL, so a reload (which loses session state) lands on
    the notice below instead of a page with no trace of it. A run this session doesn't hold (an
    MCP client's link, or a reloaded page) can't be loaded: the API has no read endpoint for
    pending calls yet, so the page says what to do instead of ignoring the link."""
    raw = st.query_params.get(RUN_ID_PARAM)
    foreign = foreign_run(conversation, raw)
    if foreign is not None:
        st.warning(
            f"**An action is waiting for approval** (run {str(foreign)[:8]}…), but this page "
            "can't load it: it was requested from an MCP client, or this page was reloaded. "
            "Nothing happens unless it is approved. To do it, ask for the action again here "
            "and approve it below.",
            icon=":material/pending_actions:",
        )
        st.button("Dismiss", key="rf_dismiss_run", on_click=_drop_run_link, disabled=busy)
    elif actionable is not None:
        st.query_params[RUN_ID_PARAM] = str(actionable.run_id)
    elif raw is not None:
        _drop_run_link()  # decided or expired here, or malformed


def _drop_run_link() -> None:
    st.query_params.pop(RUN_ID_PARAM, None)


def _resolve_titles(backend: Backend, conversation: Conversation, interrupt: Interrupt) -> None:
    """Name the documents the pending calls touch: citations first (free), then one bounded
    `documents()` walk only if an id is still unnamed. A failed lookup leaves `titles` unset, so
    the next rerun retries; the card falls back to the id meanwhile."""
    if interrupt.titles is not None:
        return
    titles = cited_titles(conversation)
    if missing_titles(interrupt, titles):
        try:
            titles |= {str(d.id): d.title for d in backend.documents()}
        except Exception:  # the raw id is still shown; naming it is a convenience
            return
    interrupt.titles = titles


def _draw_error(error: UiError | None) -> None:
    if error is not None:
        st.error(f"**{md_literal(error.title)}.** {md_literal(error.detail)}")


def _draw_extras(turn: Turn) -> None:
    if not turn.extras:
        return
    with st.container(border=True):
        st.markdown(f"**{EXTRAS_HEADING}**")
        st.caption("Written by the model, not checked against your notes. Verify before relying.")
        for extra in turn.extras:
            view = extra_view(extra)
            st.markdown(f"*{view.title}*")
            if view.before:
                st.markdown(view.before)
            if view.code is not None:
                st.code(view.code, language=view.language)
            if view.after:
                st.markdown(view.after)


def _queue_prompt(prompt: str) -> None:
    st.session_state[QUEUED_PROMPT] = prompt
    st.session_state[BUSY] = True


def _queue_decisions(decisions: dict[str, ApprovalDecision]) -> None:
    st.session_state[QUEUED_DECISIONS] = decisions
    st.session_state[BUSY] = True


def _queue_step(step: Route) -> None:
    st.session_state[QUEUED_STEP] = step
    st.session_state[BUSY] = True


# ------------------------------------------------------------------ interview toggle
def _interview_toggle(interview: InterviewUi, *, busy: bool) -> None:
    """The sidebar switch mirrors the interview (on while in the mode or while the topic box is
    open), so a typed "interview me" or "stop" flips it too. Set before the widget is drawn,
    which Streamlit allows for a keyed widget."""
    st.session_state[TOGGLE] = interview.active or (
        interview.session is None and interview.form_open
    )
    st.sidebar.toggle(
        "Interview mode",
        key=TOGGLE,
        on_change=_on_toggle,
        disabled=busy,
        help="One question at a time from your notes, asked and graded at the level you pick "
        "(default Senior SDE-1).",
    )


def _on_toggle() -> None:
    interview = interview_ui()
    on = bool(st.session_state.get(TOGGLE))
    if on and interview.paused:
        _queue_step(Run(Command.resume))
    elif on and interview.session is None:
        interview.form_open = True
    elif not on:
        interview.form_open = False
        if interview.session is not None:
            _queue_step(Run(Command.stop))  # ends with the summary card


# ------------------------------------------------------------------ collections
def _conversation() -> Conversation:
    conversation = st.session_state.get(CONVERSATION)
    if not isinstance(conversation, Conversation):
        conversation = st.session_state[CONVERSATION] = Conversation()
    return conversation


def _collections(backend: Backend, *, busy: bool) -> list[Collection]:
    """Collections, loaded once per session (refreshable).

    A failed load shows the error and falls back to the last good list (or none, which disables
    the chat box): the conversation is still drawn below, never hidden by a sidebar failure."""
    refresh = st.sidebar.button("Refresh sources", icon=":material/refresh:", disabled=busy)
    cached = st.session_state.get(COLLECTIONS)
    if isinstance(cached, list) and not refresh:
        return cached
    try:
        collections = backend.collections()
    except Exception as exc:
        _draw_error(describe_failure(exc, backend.api_url))
        return cached if isinstance(cached, list) else []
    st.session_state[COLLECTIONS] = collections
    if not collections:
        st.info("This key sees no collections yet. Connect Notion first (make connect-notion).")
    return collections


def _pick_collections(collections: list[Collection], *, busy: bool) -> list[UUID]:
    """Keyed by ID (names aren't unique); the label is only what the picker shows."""
    labels = collection_labels(collections)
    options = list(labels)
    chosen = st.sidebar.multiselect(
        "Search in",
        options,
        default=options[:_MAX_COLLECTIONS],
        format_func=lambda cid: labels.get(cid, "(removed)"),
        max_selections=_MAX_COLLECTIONS,
        key=SELECTED,
        disabled=busy,
    )
    return [cid for cid in chosen if cid in labels]


# ================================================================== sources page
def sources_page(backend: Backend) -> None:
    st.header("Sources")
    try:
        collections = backend.collections()
        connectors = backend.connectors()
    except Exception as exc:
        _draw_error(describe_failure(exc, backend.api_url))
        return
    names = {c.id: c.name for c in collections}
    st.subheader("Collections")
    if collections:
        st.dataframe(
            [{"name": c.name, "created": c.created_at.strftime("%Y-%m-%d")} for c in collections],
            hide_index=True,
        )
    else:
        st.caption("No collections yet.")
    jobs = _job_statuses(backend)
    syncing = {cid for _, cid, job in jobs if isinstance(job, Job) and in_flight(str(job.status))}
    st.session_state[SYNCING] = syncing
    st.subheader("Connectors")
    _draw_sync_notice()
    if not connectors:
        st.caption("No connectors yet. Add Notion with make connect-notion.")
    for connector in connectors:
        name = names.get(connector.target_collection_id)
        _draw_connector(backend, connector, name, syncing=connector.id in syncing)
    _draw_jobs(jobs)


_STATUS_COLORS = {"active": "green", "paused": "gray", "error": "red", "deleting": "orange"}


def _draw_connector(
    backend: Backend, connector: Connector, collection: str | None, *, syncing: bool
) -> None:
    status = str(connector.status)
    with st.container(border=True):
        st.markdown(f"**{md_literal(connector.display_name)}** · {md_literal(str(connector.kind))}")
        st.badge(md_literal(status), color=_STATUS_COLORS.get(status, "gray"))  # type: ignore[arg-type]
        where = md_literal(collection) if collection else "an unknown collection"
        st.caption(f"Into {where} · added {connector.created_at:%Y-%m-%d}")
        if connector.last_error:
            st.error(f"**Last error:** {md_literal(connector.last_error)}")
        st.button(
            "Syncing…" if syncing else "Sync now",
            key=f"sync-{connector.id}",
            icon=":material/sync:",
            disabled=syncing or status == "deleting",
            on_click=_sync,
            args=(backend, connector.id),
        )


def _sync(backend: Backend, connector_id: UUID) -> None:
    """Button callback: runs BEFORE the rerun, so that very run lists the job and draws the
    button disabled. The `SYNCING` guard drops a second click that arrives before then (a
    double click), so one click queues one job."""
    syncing: set[UUID] = st.session_state.setdefault(SYNCING, set())
    if connector_id in syncing:
        return
    try:
        job_id = backend.sync_connector(connector_id)
    except Exception as exc:
        st.session_state[SYNC_NOTICE] = describe_failure(exc, backend.api_url)
        return
    syncing.add(connector_id)
    jobs: list[tuple[UUID, UUID]] = st.session_state.setdefault(JOBS, [])
    jobs.insert(0, (job_id, connector_id))
    del jobs[5:]
    st.session_state[SYNC_NOTICE] = "Sync queued. Its progress is under Recent syncs."


def _draw_sync_notice() -> None:
    notice = st.session_state.pop(SYNC_NOTICE, None)
    if isinstance(notice, UiError):
        _draw_error(notice)
    elif isinstance(notice, str):
        st.success(notice)


def _job_statuses(backend: Backend) -> list[tuple[UUID, UUID, Job | UiError]]:
    """Syncs started from this browser session, with their current status (one cheap GET each;
    the API has no "list jobs" endpoint, so these are the only job ids the UI knows).

    A job the API no longer knows (404: pruned, or another tenant's after a key change) is
    dropped from the session, so it can't show the same error on every rerun."""
    jobs: list[tuple[UUID, UUID]] = st.session_state.get(JOBS, [])
    out: list[tuple[UUID, UUID, Job | UiError]] = []
    for job_id, connector_id in list(jobs):
        try:
            out.append((job_id, connector_id, backend.job(job_id)))
        except ApiError as exc:
            if exc.status == 404:
                jobs.remove((job_id, connector_id))
                continue
            out.append((job_id, connector_id, describe_failure(exc, backend.api_url)))
        except Exception as exc:
            out.append((job_id, connector_id, describe_failure(exc, backend.api_url)))
    return out


def _draw_jobs(jobs: list[tuple[UUID, UUID, Job | UiError]]) -> None:
    if not jobs:
        return
    st.subheader("Recent syncs")
    st.button("Refresh status", icon=":material/refresh:")  # a click reruns, which re-reads
    for _, _, job in jobs:
        if isinstance(job, UiError):
            _draw_error(job)
            continue
        progress = ", ".join(
            f"{md_literal(str(k))}: {md_literal(str(v))}" for k, v in job.progress.items()
        )
        line = f"**{md_literal(str(job.status))}** {md_literal(job.kind)} · attempt {job.attempts}"
        st.markdown(line + (f" · {progress}" if progress else ""))
        if job.last_error:
            st.caption(f"Last error: {md_literal(job.last_error)}")
