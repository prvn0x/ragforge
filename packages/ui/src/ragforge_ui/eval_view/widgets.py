"""Shared drawing pieces of the Eval page.

Text rules (render module docstring): anything read from a run, dataset or label file and placed
in Markdown by this page goes through `md_literal` (ids, model names, reasons, notes by the
owner); answer text is the server's sanitized CommonMark (`answer_markdown`); note text is shown
with `st.text` (exact characters, nothing interpreted). Tables are `st.dataframe`, whose cells are
never Markdown. Nothing uses `unsafe_allow_html`.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

import streamlit as st

from ragforge_client.ask import KeyFileError
from ragforge_client.evals.interfaces import record_reveal, revealed_at
from ragforge_client.evals.schema import EvalKind, MetricValue, ShownRef
from ragforge_ui.errors import describe_failure
from ragforge_ui.eval_view.context import EvalContext
from ragforge_ui.eval_view.data import RunEntry
from ragforge_ui.eval_view.model import ci_cell, header_facts, value_text
from ragforge_ui.eval_view.notes import Note, NoteState, shown_notes
from ragforge_ui.render import md_literal

REVEALED = "rf_eval_revealed"  # run keys (paths) revealed in this session; ids only, no text


def table(rows: Sequence[Mapping[str, str]]) -> None:
    if rows:
        st.dataframe(list(rows), hide_index=True)
    else:
        st.caption("Nothing to show.")


def run_picker(
    ctx: EvalContext,
    label: str,
    key: str,
    *kinds: EvalKind,
    index: int = 0,
    name: Callable[[RunEntry], str] = lambda e: e.label,
) -> RunEntry | None:
    entries = ctx.runs_of(*kinds)
    if not entries:
        what = " or ".join(k.value for k in kinds) if kinds else "eval"
        st.info(f"No {what} runs in {md_literal(str(ctx.paths.runs))}.")
        return None
    return entry_picker(entries, label, key, index=index, name=name)


def entry_picker(
    entries: Sequence[RunEntry],
    label: str,
    key: str,
    *,
    index: int = 0,
    name: Callable[[RunEntry], str] = lambda e: e.label,
) -> RunEntry | None:
    """A run selectbox keyed by run file path (run ids are not promised unique)."""
    by_key = {e.key: e for e in entries}
    chosen = st.selectbox(
        label,
        list(by_key),
        index=min(index, len(by_key) - 1),
        format_func=lambda k: name(by_key[k]),
        key=key,
    )
    return by_key.get(chosen) if chosen else None


def header_strip(entry: RunEntry) -> None:
    """The facts that make a run comparable, on one compact block."""
    facts = header_facts(entry.header)
    st.markdown("  \n".join(f"**{name}:** {md_literal(value)}" for name, value in facts))
    if entry.run.footer is None:
        st.caption("Unfinished run (resumable): numbers may be partial.")
    if entry.header.override:
        st.warning(f"**Gate override recorded:** {md_literal(entry.header.override)}")


def metric_tiles(metrics: Sequence[MetricValue]) -> None:
    """Headline metrics: the value (or its trust state), the CI, n and the method."""
    if not metrics:
        return
    for column, metric in zip(st.columns(len(metrics)), metrics, strict=True):
        with column:
            st.metric(md_literal(metric.name), value_text(metric))
            st.caption(f"95% CI {ci_cell(metric)} · n {metric.n} · {metric.method.value}")


def summary_problem(entry: RunEntry) -> bool:
    """Say why a run has no summary; True when there is none to draw."""
    if entry.summary is None:
        st.info(f"No numbers for this run: {md_literal(entry.summary_problem or 'no summary')}.")
        return True
    s = entry.summary
    if s.stale_items or s.errors_ours or s.errors_transient:
        st.caption(
            f"Excluded: {s.stale_items} stale item(s); errors: {s.errors_ours} ours, "
            f"{s.errors_transient} provider-transient."
        )
    return False


def reveal_control(entry: RunEntry, hidden: int, *, key: str) -> bool:
    """Held-out rows stay hidden unless the owner types the run id to reveal them (a typo-proof
    deliberate act, not a stray click). Revealed for this session only, and said so.

    Only the Drill-down offers it (Compare flips and the label queue never show held-out rows).
    The reveal is RECORDED in `<run>.reveal.json` (interfaces.record_reveal; ADR-0017 D1's one
    reveal): once written, the rows stay visible here and the page says when they were revealed."""
    revealed: set[str] = st.session_state.setdefault(REVEALED, set())
    stamp = revealed_at(entry.run.path)
    if stamp is not None or entry.key in revealed:
        st.warning(
            f"Held-out rows of this run were revealed ({stamp or 'this session'}). A held-out "
            "row seen while tuning is no longer held out: use a fresh held-out set to tune again."
        )
        return True
    if hidden == 0:
        return False
    st.caption(f"{hidden} held-out (heldout2) row(s) hidden.")
    with st.expander("Reveal held-out rows"):
        typed = st.text_input("Type the run id to reveal its held-out rows", key=key)
        if typed and typed == entry.header.run_id:
            record_reveal(entry.run.path, entry.header.run_id)
            revealed.add(entry.key)
            st.rerun()
    return False


def notes_block(ctx: EvalContext, refs: Sequence[ShownRef], *, key: str) -> None:
    """The shown notes, fetched only while the toggle is on and never stored (module notes)."""
    if not refs:
        st.caption("This row recorded no shown notes.")
        return
    if not st.toggle(f"Show the {len(refs)} note(s) the model was shown", key=key):
        return
    notes = fetch_notes(ctx, refs)
    if notes is not None:
        draw_notes(notes)


def fetch_notes(ctx: EvalContext, refs: Sequence[ShownRef]) -> list[Note] | None:
    """The digest-checked notes, or None after saying why they could not be fetched. The text
    lives only in the returned list (a local of this script run), never in state or a cache."""
    try:
        return shown_notes(refs, ctx.chunk_source())
    except KeyFileError as exc:  # its message names the file, never the key
        st.error(f"**No usable eval key.** {md_literal(str(exc))}")
    except Exception as exc:  # shown as a friendly message; never str(exc)
        failure = describe_failure(exc, ctx.api_url)
        st.error(f"**{md_literal(failure.title)}.** {md_literal(failure.detail)}")
    return None


def draw_notes(notes: Sequence[Note]) -> None:
    for note in notes:
        where = " › ".join(md_literal(part) for part in note.where)
        st.markdown(f"**\\[S{note.label}\\]** {where}")
        if note.state is NoteState.ok and note.text is not None:
            st.text(note.text)
        elif note.state is NoteState.changed:
            st.warning("This note changed since the run (digest mismatch): not shown.")
        else:
            st.warning("This note is not visible to the eval key (deleted or out of scope).")
