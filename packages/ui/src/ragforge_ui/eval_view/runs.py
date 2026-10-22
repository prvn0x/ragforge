"""Runs and Compare views (ADR-0017 §6)."""

from __future__ import annotations

from typing import Literal

import streamlit as st

from ragforge_client.evals.schema import GateCheck, GateOutcome
from ragforge_ui.eval_view.context import EvalContext
from ragforge_ui.eval_view.data import RunEntry, load_gate
from ragforge_ui.eval_view.model import (
    EXPLORATORY,
    NONE,
    comparability,
    fmt,
    headline,
    metric_deltas,
    metric_row,
    paired,
    short,
    visible_rows,
)
from ragforge_ui.eval_view.widgets import (
    entry_picker,
    header_strip,
    metric_tiles,
    run_picker,
    summary_problem,
    table,
)
from ragforge_ui.render import md_literal

RUN = "rf_eval_run"
BASE = "rf_eval_base"
CAND = "rf_eval_cand"

Color = Literal["green", "orange", "red", "gray"]
_OUTCOME: dict[GateOutcome, tuple[str, Color]] = {
    GateOutcome.pass_: ("PASS", "green"),
    GateOutcome.warn: ("WARN", "orange"),
    GateOutcome.block: ("BLOCK", "red"),
    GateOutcome.not_evaluable: ("NOT EVALUABLE", "gray"),
}


# ------------------------------------------------------------------ runs
def runs_view(ctx: EvalContext) -> None:
    st.subheader("Runs")
    table([_overview(e) for e in ctx.entries])
    entry = run_picker(ctx, "Run", RUN)
    if entry is None:
        return
    header_strip(entry)
    _gate(ctx, entry)
    if summary_problem(entry):
        return
    assert entry.summary is not None
    metric_tiles(headline(entry.summary))
    with st.expander("All metrics of this run", expanded=True):
        table([metric_row(m) for m in entry.summary.metrics])


def _overview(entry: RunEntry) -> dict[str, str]:
    h = entry.header
    return {
        "run": h.run_id,
        "kind": h.eval_kind.value,
        "surface": h.surface.value,
        "arm": h.arm,
        "started": f"{h.started_at:%Y-%m-%d %H:%M}",
        "finished": "yes" if entry.run.footer else "no (resumable)",
        "snapshot": short(h.snapshot.content_digest) if h.snapshot else "demo",
        "summary": "yes" if entry.summary else "no",
    }


def _gate(ctx: EvalContext, entry: RunEntry) -> None:
    surface = entry.header.surface
    verdict, why = load_gate(ctx.paths.runs, surface)
    if why:
        st.warning(md_literal(why))
        return
    if verdict is None:
        st.caption(f"No gate verdict for the {surface.value} surface yet.")
        return
    text, color = _OUTCOME.get(verdict.outcome, (verdict.outcome.value, "gray"))
    st.markdown(
        f"**Gate ({surface.value}):** baseline {md_literal(verdict.baseline_run)} → candidate "
        f"{md_literal(verdict.candidate_run)}"
    )
    st.badge(text, color=color)
    run_id = entry.header.run_id
    if run_id not in (verdict.baseline_run, verdict.candidate_run):
        st.caption("This run is not part of the latest gate verdict for its surface.")
    if verdict.override:
        st.warning(f"**Gate overridden:** {md_literal(verdict.override)}")
    table([_check_row(c) for c in verdict.checks])


def _check_row(check: GateCheck) -> dict[str, str]:
    return {
        "check": check.name,
        "outcome": check.outcome.value,
        "method": check.method,
        "n": str(check.n),
        "effect": fmt(check.effect),
        "p": fmt(check.p_value),
        "MDE": fmt(check.mde),
        "note": check.note,
    }


# ------------------------------------------------------------------ compare
def compare_view(ctx: EvalContext) -> None:
    st.subheader("Compare two runs")
    cand = run_picker(ctx, "Candidate", CAND)
    if cand is None:
        return
    others = [e for e in ctx.runs_of(cand.header.eval_kind) if e.key != cand.key]
    if not others:
        st.info(f"No other {cand.header.eval_kind.value} run to compare it with.")
        return
    base = entry_picker(others, "Baseline", BASE)
    if base is None:
        return
    why = comparability(base.header, cand.header)
    if why is not None:
        st.error(f"**These runs can't be compared.** {md_literal(why)}")
        return
    st.caption(f"Paired deltas and p-values are {EXPLORATORY}.")
    if base.summary and cand.summary:
        st.markdown("**Metric deltas** (candidate minus baseline)")
        table(metric_deltas(base.summary, cand.summary))
    else:
        st.caption("Metric deltas need both runs' summaries.")
    # never revealed here (see widgets.reveal_control): flips would expose held-out items
    shown = [visible_rows(e.run, ctx.datasets, reveal=False) for e in (base, cand)]
    hidden = sum(v.hidden for v in shown)
    if hidden:
        st.caption(
            f"{hidden} held-out (heldout2) row(s) are left out of the flips; Compare never "
            "reveals them (the Drill-down can, one row at a time)."
        )
    ds = ctx.datasets
    questions = {q.id: q for q in ds.questions.questions} if ds.questions else {}
    items = {i.id: i for i in ds.verifier.items} if ds.verifier else {}
    result = paired(shown[0].rows, shown[1].rows, questions, items)
    st.markdown("**Per-item flips**")
    table(
        [
            {
                "paired outcome": d.measure,
                "pairs": str(d.pairs),
                "pass → fail": str(d.lost),
                "fail → pass": str(d.gained),
                "exact McNemar p (exploratory)": fmt(d.p_value),
            }
            for d in result.discordance
        ]
    )
    if result.only_one:
        st.caption(f"{result.only_one} item(s) are in one run only and are not paired.")
    table(
        [
            {
                "item": f.item_id,
                "measure": f.measure,
                "baseline": f.baseline,
                "candidate": f.candidate,
            }
            for f in result.flips
        ]
        or [{"item": NONE, "measure": "no item changed", "baseline": "", "candidate": ""}]
    )
