"""The blind label queue (ADR-0017 §2 D5, §4): the owner scores E-G answers for calibration.

Blind: the queue draws a `QueueItem`, which holds the question, the shown notes and the answer
and nothing else, so no judge score, judge reason or server verdict can reach this view. It never
shows a held-out row (even after a reveal elsewhere) and writes ONLY through `LabelWriter`.

Sample: the queue is the stratified calibration sample of ADR-0017 §4 (by answer tier and
verdict, `datasets.calibration_sample`), not every answered row, and it lists E-G runs only by a
neutral name (start time and a short file id), never the arm or model being calibrated. E-A
answer labels and adjudications are labelled with the CLI.

Notes: faithfulness is "supported by the notes shown", so the queue fetches the shown notes
itself (digest-checked, never stored) and offers an item only when every one of them is still
the text the model saw. An item whose notes changed or vanished is skipped with its reason, as
stale rows are; when the notes can't be fetched at all, nothing is offered. A row that recorded
no shown notes is never offered (as in the CLI). Which items were skipped for changed notes is
remembered for the session by item digest (no text), so a rerun fetches only the item shown.

Keyboard: the scores are radios (Tab moves between them, arrow keys pick), and Enter in the
note box submits the form, so a label is three arrow presses and Enter.
"""

from __future__ import annotations

from datetime import UTC, date, datetime

import streamlit as st

from ragforge_client.evals.datasets import CALIBRATION_N
from ragforge_client.evals.schema import E_G_CRITERIA, Criterion, EvalKind
from ragforge_ui.eval_view.context import EvalContext
from ragforge_ui.eval_view.model import QueueItem, criteria_label, labelled_digests, queue_items
from ragforge_ui.eval_view.notes import Note, NoteState
from ragforge_ui.eval_view.widgets import draw_notes, fetch_notes, run_picker
from ragforge_ui.render import answer_markdown, md_literal

RUN = "rf_eval_queue_run"
BATCH = "rf_eval_batch"
SAVED = "rf_eval_saved"  # the item id just saved (an id, never text)
CHANGED = "rf_eval_notes_changed"  # item digests whose notes failed the check (ids, never text)
SCORES = (0, 1, 2)
# Short anchors; the full ones are the judge prompt's, so owner and judge score the same thing.
ANCHORS: dict[Criterion, str] = {
    Criterion.relevance: "Does it answer what was asked?",
    Criterion.completeness: "Does it cover what the question needs?",
    Criterion.faithfulness: "Is every claim supported by the notes shown?",
}
_SCORE_NAMES = {0: "0 · no", 1: "1 · partly", 2: "2 · yes"}
KEYBOARD_HINT = "Keyboard: Tab between criteria, arrow keys pick a score, Enter in the note saves."


def label_view(ctx: EvalContext) -> None:
    st.subheader("Label queue (blind)")
    st.caption(
        "Only the question, the notes the model was shown and the answer: no judge score, no "
        "reason, no verdict. Held-out rows never appear here. Length is not quality."
    )
    entry = run_picker(ctx, "Run", RUN, EvalKind.generator, name=lambda e: e.blind_label)
    if entry is None:
        return
    queue = queue_items(entry.run, ctx.datasets.questions, ctx.judge_by_key)
    if ctx.datasets.questions is None:
        st.warning("The question set is not available, so no item can be labelled.")
    st.caption(
        f"The calibration sample: up to {CALIBRATION_N} items stratified by answer tier and "
        "verdict (ADR-0017 §4), the same design as the CLI's `label`. E-A answer labels and "
        "adjudications: use the CLI."
    )
    if queue.outside_sample:
        st.caption(f"Not drawn into the sample: {queue.outside_sample} labelable item(s).")
    for why, count in sorted(queue.skipped.items()):
        st.caption(f"Not in the queue: {count} {why}.")
    read = ctx.latest_labels()
    if read.problem:
        st.warning(
            f"{md_literal(read.problem)}: labelling is off until the file is fixed, so no item "
            "is labelled twice."
        )
    done = labelled_digests(read.labels)
    todo = [item for item in queue.items if item.item_digest not in done]
    total = len(queue.items)
    if total:
        st.progress((total - len(todo)) / total)
        st.caption(f"{total - len(todo)} of {total} labelled")
    saved = st.session_state.pop(SAVED, None)
    if saved:
        st.success(f"Saved the label for {md_literal(str(saved))}.")
    if not todo:
        st.info("Nothing left to label in this run.")
        return
    if ctx.labels.writer is None or read.problem:
        if ctx.labels.why:
            st.warning(md_literal(ctx.labels.why))
        return
    found = _next_with_notes(ctx, todo)
    if found is not None:
        _item(ctx, *found)


NOTES_CHANGED = "whose shown notes changed or are gone since the run"


def _next_with_notes(
    ctx: EvalContext, todo: list[QueueItem]
) -> tuple[QueueItem, list[Note]] | None:
    """The first item whose shown notes are all still the text the model saw, with them.

    Fetching stops at the first usable item; an item found changed earlier this session is
    skipped without a request. A fetch that fails outright offers nothing, since the owner
    could then only label blind to the notes."""
    changed: set[str] = st.session_state.setdefault(CHANGED, set())
    skipped = 0
    found: tuple[QueueItem, list[Note]] | None = None
    for item in todo:
        if item.item_digest in changed:
            skipped += 1
            continue
        notes = fetch_notes(ctx, item.shown)
        if notes is None:
            break  # fetch_notes said why
        if notes and all(n.state is NoteState.ok for n in notes):
            found = (item, notes)
            break
        changed.add(item.item_digest)
        skipped += 1
    if skipped:
        st.caption(f"Skipped: {skipped} {NOTES_CHANGED}.")
    if found is None and skipped == len(todo):
        st.info("No item left whose shown notes are unchanged since the run.")
    return found


def _item(ctx: EvalContext, item: QueueItem, notes: list[Note]) -> None:
    with st.container(border=True):
        st.markdown(f"**Question:** {md_literal(item.question)}")
        st.markdown(f"**Notes the model was shown** ({len(notes)})")
        draw_notes(notes)
        st.markdown("**Answer**")
        st.markdown(answer_markdown(item.answer))
    batch = st.text_input(
        "Label batch (calibration is recomputed per batch)",
        value=f"ui-{date.today():%Y-%m-%d}",
        key=BATCH,
    ).strip()
    _form(ctx, item, batch)


def _form(ctx: EvalContext, item: QueueItem, batch: str) -> None:
    digest = item.item_digest
    with st.form(key=f"rf_eval_label:{digest}", clear_on_submit=True):
        chosen: dict[Criterion, int | None] = {}
        for criterion in E_G_CRITERIA:
            chosen[criterion] = st.radio(
                f"**{criterion.value}**: {ANCHORS[criterion]}",
                SCORES,
                index=None,  # no default: a preselected score anchors the owner
                format_func=_SCORE_NAMES.__getitem__,
                horizontal=True,
                key=f"rf_eval_score:{criterion.value}:{digest}",
            )
        note = st.text_input("Note (optional)", key=f"rf_eval_note:{digest}")
        st.caption(KEYBOARD_HINT)
        submitted = st.form_submit_button("Save label")
    if submitted:
        _save(ctx, item, chosen, note, batch)


def _save(
    ctx: EvalContext,
    item: QueueItem,
    chosen: dict[Criterion, int | None],
    note: str,
    batch: str,
) -> None:
    scores = {c: s for c, s in chosen.items() if s is not None}
    if len(scores) != len(E_G_CRITERIA):
        st.warning("Score every criterion before saving: a partial label can't calibrate.")
        return
    if not batch:
        st.warning("Give the label batch a name.")
        return
    writer = ctx.labels.writer
    assert writer is not None  # the form is only drawn with a writer
    label = criteria_label(item, scores, note=note, batch_id=batch, at=datetime.now(UTC))
    try:
        writer.append(label)
    except (OSError, ValueError) as exc:  # type only: never echo file contents
        st.error(f"**The label was not saved** ({type(exc).__name__}). Try again.")
        return
    st.session_state[SAVED] = item.item_id
    st.rerun()
