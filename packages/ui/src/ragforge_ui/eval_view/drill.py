"""Drill-down (ADR-0017 §6): one row of one run, with everything that explains its numbers.

Unlike the label queue this view is NOT blind: it shows the verdict, the catching layer, the
judge's scores and reasons and the owner's label side by side, for diagnosis.
"""

from __future__ import annotations

import streamlit as st

from ragforge_client.evals.schema import (
    AnswerOutput,
    GradeOutput,
    Label,
    RetrievalOutput,
    RunRow,
    VerifyOutput,
)
from ragforge_ui.eval_view.context import EvalContext
from ragforge_ui.eval_view.data import RunEntry
from ragforge_ui.eval_view.model import (
    NONE,
    calibration_state,
    caught_layer,
    fmt,
    owner_labels,
    row_judgements,
    trust_for,
    visible_rows,
)
from ragforge_ui.eval_view.widgets import (
    header_strip,
    notes_block,
    reveal_control,
    run_picker,
    table,
)
from ragforge_ui.render import answer_markdown, badge_for, for_streamlit, md_literal

RUN = "rf_eval_drill_run"
ITEM = "rf_eval_drill_item"
NOTES = "rf_eval_drill_notes"


def drill_view(ctx: EvalContext) -> None:
    st.subheader("Drill-down")
    entry = run_picker(ctx, "Run", RUN)
    if entry is None:
        return
    header_strip(entry)
    hidden = visible_rows(entry.run, ctx.datasets, reveal=False)
    reveal = reveal_control(entry, hidden.hidden, key="rf_eval_reveal_drill")
    rows = list(entry.run.rows) if reveal else hidden.rows
    if not rows:
        st.info("No rows to show.")
        return
    # by position: a resumed run may hold an errored attempt and its retry under one item id
    index = st.selectbox(
        "Item",
        range(len(rows)),
        format_func=lambda i: _row_label(rows[i]),
        key=f"{ITEM}:{entry.key}",
    )
    row = rows[index or 0]
    _row(ctx, entry, row)


def _row_label(row: RunRow) -> str:
    suffix = " (error)" if row.error else " (stale)" if row.stale else ""
    return f"{row.item_id}{suffix}"


def _row(ctx: EvalContext, entry: RunEntry, row: RunRow) -> None:
    st.caption(
        f"attempt {row.attempt} · first token {fmt(row.timings.ttft_s)} s · total "
        f"{fmt(row.timings.total_s)} s · tokens {row.tokens.prompt}+{row.tokens.completion}"
        + (" · STALE: a key no longer resolves or its digest changed" if row.stale else "")
    )
    if row.error is not None:
        status = f" {row.error.status}" if row.error.status else ""
        st.error(f"**Error ({row.error.cls.value}{status}).** {md_literal(row.error.message)}")
    out = row.output
    if isinstance(out, AnswerOutput):
        _answer(ctx, entry, row, out)
    elif isinstance(out, VerifyOutput):
        _verify(ctx, row, out)
    elif isinstance(out, GradeOutput):
        _grade(out)
    elif isinstance(out, RetrievalOutput):
        _retrieval(ctx, row, out)
    notes_block(ctx, row.shown, key=f"{NOTES}:{entry.key}:{row.item_id}:{row.attempt}")


def _question_text(ctx: EvalContext, item_id: str) -> str | None:
    qs = ctx.datasets.questions
    q = next((q for q in qs.questions if q.id == item_id), None) if qs else None
    return q.question if q else None


def _answer(ctx: EvalContext, entry: RunEntry, row: RunRow, out: AnswerOutput) -> None:
    question = _question_text(ctx, row.item_id)
    if question:
        st.markdown(f"**Question:** {md_literal(question)}")
    badge = badge_for(out.status, cited=bool(out.cited_pages), flagged=out.flagged > 0)
    if badge is not None:
        st.badge(badge.label, icon=badge.icon, color=badge.color)  # type: ignore[arg-type]
        st.caption(badge.meaning)
    st.markdown(answer_markdown(out.answer))
    if out.cited_pages:
        st.caption("Cited: " + ", ".join(md_literal(p) for p in out.cited_pages))
    judged = row_judgements(row, ctx.judge_by_key)
    state = calibration_state(ctx.calibrations.items, entry.header.judge)
    st.markdown("**Judge**")
    table(
        [
            {
                "criterion": j.criterion.value,
                "score": NONE if j.score is None else str(j.score),
                "judge": trust_for(j.criterion, state, entry.summary),
                "reason": j.reason,
            }
            for j in judged
        ]
    )
    if row.judge_refs and not judged:
        st.caption("The judge results this row refers to are not in the judge file.")
    read = ctx.latest_labels()
    if read.problem:
        st.warning(f"{md_literal(read.problem)}: showing the readable lines only.")
    _labels(owner_labels({j.item_digest for j in judged}, read.labels))


def _labels(labels: list[Label]) -> None:
    st.markdown("**Your label**")
    if not labels:
        st.caption("Not labelled.")
        return
    for lb in labels:
        parts = [f"{c.value} {s}" for c, s in sorted(lb.scores.items())]
        for name, value in (
            ("correct", lb.correct),
            ("invented", lb.invented),
            ("confirmed", lb.confirmed),
        ):
            if value is not None:
                parts.append(f"{name} {'yes' if value else 'no'}")
        line = f"{lb.kind.value}: {', '.join(parts) or NONE} · batch {lb.batch_id}"
        st.markdown(md_literal(line))
        if lb.note:
            st.caption(md_literal(lb.note))


def _verify(ctx: EvalContext, row: RunRow, out: VerifyOutput) -> None:
    vs = ctx.datasets.verifier
    item = next((i for i in vs.items if i.id == row.item_id), None) if vs else None
    if item is not None:
        st.markdown(f"**Question:** {md_literal(item.question)}")
        st.markdown(for_streamlit(item.answer))
        if item.mutation is not None and item.span is not None:
            planted = item.answer[item.span.start : item.span.end]
            expected = item.expected_layer.value if item.expected_layer else NONE
            st.markdown(
                f"**Planted:** {md_literal(item.mutation.value)} at “{md_literal(planted)}” · "
                f"expected layer {expected}"
            )
    st.markdown(f"**Verdict:** {md_literal(out.status)} · **caught by:** {caught_layer(out, item)}")
    st.caption(
        f"coverage flags {len(out.coverage_spans)} · located verifier claims "
        f"{len(out.verifier_spans)} · unlocated {out.unlocated_verifier_claims} · verifier "
        f"{'ran' if out.verifier_ran else 'did not run'}"
        f"{', objected' if out.verifier_objected else ''} · model "
        f"{md_literal(out.verifier_model or NONE)} · prompt "
        f"{md_literal(out.verifier_prompt or NONE)}"
    )


def _grade(out: GradeOutput) -> None:
    st.markdown(
        md_literal(
            f"{out.seed_id} · {out.tier.value} answer · graded at {out.level} · replicate "
            f"{out.replicate}"
        )
    )
    st.metric("Score", f"{out.score}/10")
    for title, quotes in (
        ("Misconceptions given (before the server's filters)", out.misconceptions_given),
        ("Misconceptions kept (after the quote rule and the verifier)", out.misconceptions_kept),
    ):
        st.markdown(f"**{title}**")
        st.markdown("\n".join(f"- {md_literal(q)}" for q in quotes) or NONE)


def _retrieval(ctx: EvalContext, row: RunRow, out: RetrievalOutput) -> None:
    qs = ctx.datasets.questions
    q = next((q for q in qs.questions if q.id == row.item_id), None) if qs else None
    if q is not None:
        st.markdown(f"**Question:** {md_literal(q.question)} · need {q.need.value}")
    relevance = q.relevance if q else {}
    stages = list(out.stages)
    depth = max((len(v) for v in out.stages.values()), default=0)
    table(
        [
            {"rank": str(i + 1)} | {s: _ranked(out.stages[s], i, relevance) for s in stages}
            for i in range(min(depth, 10))
        ]
    )
    if out.top_rerank_score is not None:
        st.caption(f"Top reranked score {fmt(out.top_rerank_score)}")


def _ranked(pages: list[str], i: int, relevance: dict[str, int]) -> str:
    if i >= len(pages):
        return ""
    page = pages[i]
    grade = relevance.get(page)
    return f"{page} (relevance {grade})" if grade else page
