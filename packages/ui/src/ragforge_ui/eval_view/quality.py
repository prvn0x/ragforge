"""Verifier, Interview and Calibration views (ADR-0017 §3 E-V / E-I, §4)."""

from __future__ import annotations

from typing import Literal

import streamlit as st

from ragforge_client.evals.schema import (
    E_G_CRITERIA,
    EXPECTED_LAYER,
    EvalKind,
    JudgeKey,
    MetricValue,
    Mutation,
)
from ragforge_ui.eval_view.context import EvalContext
from ragforge_ui.eval_view.data import RunEntry
from ragforge_ui.eval_view.model import (
    CALIBRATION_PREFIX,
    NO_TRUSTED_JUDGE,
    NONE,
    SLO_FALSE_REJECT_HIGH,
    SLO_RECALL_LOW,
    by_name,
    calibration_row,
    calibration_state,
    ci_cell,
    confusion_rows,
    judge_keys,
    judge_text,
    k_of_n,
    level_means,
    metric_row,
    slo_verdict,
    summary_trust,
    value_text,
    with_prefix,
)
from ragforge_ui.eval_view.widgets import (
    header_strip,
    metric_tiles,
    run_picker,
    summary_problem,
    table,
)
from ragforge_ui.render import md_literal

VERIFIER_RUN = "rf_eval_verifier_run"
FALSE_VERIFIED_CURVE = "verifier.false_verified@"  # EV-METRICS names (not in the D0 oracle)
FALSE_VERIFIED_REAL = "generator.false_verified.confirmed"
INTERVIEW_RUN = "rf_eval_interview_run"
JUDGE = "rf_eval_calibration_judge"

_SLO_COLOR: dict[str, Literal["green", "red", "gray"]] = {
    "pass": "green",
    "fail": "red",
    "not evaluable": "gray",
}


def _cells(metric: MetricValue) -> dict[str, str]:
    return {
        "value": value_text(metric),
        "95% CI": ci_cell(metric),
        "n": str(metric.n),
        "method": metric.method.value,
    }


# ------------------------------------------------------------------ verifier
def verifier_view(ctx: EvalContext) -> None:
    st.subheader("Verifier (planted errors)")
    entry = run_picker(ctx, "Run", VERIFIER_RUN, EvalKind.verifier)
    if entry is None:
        return
    header_strip(entry)
    if summary_problem(entry):
        return
    summary = entry.summary
    m = by_name(summary)
    slo = slo_verdict(summary)
    model = entry.header.verifier_model or "the verifier model"
    st.markdown(
        f"**SLO for {md_literal(model)}:** badge-flip recall lower bound ≥ {SLO_RECALL_LOW} and "
        f"verifier-caused false-reject upper bound ≤ {SLO_FALSE_REJECT_HIGH}"
    )
    st.badge(f"SLO {slo.state}", color=_SLO_COLOR[slo.state])
    st.caption(md_literal(slo.text))
    metric_tiles(
        [
            m[n]
            for n in (
                "verifier.badge_flip_recall",
                "verifier.span_recall",
                "verifier.caught_unlocated",
            )
            if n in m
        ]
    )
    st.caption(
        "Badge-flip recall and caught-unlocated count only sources the build verified clean in "
        "the same run; span recall counts a catch only where a flag overlaps the planted span."
    )
    st.markdown("**Span recall by mutation**")
    table(
        [
            {
                "mutation": metric.name.removeprefix("verifier.span_recall."),
                "expected layer": _expected(metric.name),
                **_cells(metric),
            }
            for metric in with_prefix(summary, "verifier.span_recall.")
        ]
    )
    st.markdown("**Caught by layer** (share of planted items)")
    table(
        [
            {"layer": metric.name.removeprefix("verifier.caught_by."), **_cells(metric)}
            for metric in with_prefix(summary, "verifier.caught_by.")
        ]
    )
    st.markdown("**False reject on clean items**")
    table(
        [
            {"cause": _cause(metric.name), **_cells(metric)}
            for metric in with_prefix(summary, "verifier.false_reject")
        ]
    )
    st.caption(
        "Only the verifier-caused part has an SLO; coverage-caused is the cost of the coverage "
        "guarantee (ADR-0017 §3)."
    )
    _false_verified(ctx, entry)


def _false_verified(ctx: EvalContext, entry: RunEntry) -> None:
    """ADR-0017 §3: the synthetic false-verified curve of this verifier run next to the real
    rate, the owner-CONFIRMED k/n of the newest E-G run (a lower bound: the judge screen that
    flags answers for adjudication misses some). Two estimates of one quantity, side by side."""
    st.markdown("**False verified: synthetic (D3) vs real (confirmed)**")
    curve = with_prefix(entry.summary, FALSE_VERIFIED_CURVE)
    table(
        [{"base error rate": m.name.removeprefix(FALSE_VERIFIED_CURVE), **_cells(m)} for m in curve]
    )
    real = next(
        (
            by_name(e.summary)[FALSE_VERIFIED_REAL]
            for e in ctx.runs_of(EvalKind.generator)
            if FALSE_VERIFIED_REAL in by_name(e.summary)
        ),
        None,
    )
    if real is None:
        st.caption("Real rate: no E-G run has adjudicated judge-flagged answers yet.")
        return
    st.caption(
        f"Real rate (owner-confirmed, a lower bound): {k_of_n(real)} · 95% CI {ci_cell(real)} · "
        f"{real.method.value}"
    )


def _expected(name: str) -> str:
    try:
        return EXPECTED_LAYER[Mutation(name.removeprefix("verifier.span_recall."))].value
    except ValueError:
        return NONE


def _cause(name: str) -> str:
    return {
        "verifier.false_reject": "all",
        "verifier.false_reject.verifier_caused": "verifier-caused",
        "verifier.false_reject.coverage_caused": "coverage-caused",
    }.get(name, name)


# ------------------------------------------------------------------ interview
def interview_view(ctx: EvalContext) -> None:
    st.subheader("Interview grading")
    entry = run_picker(ctx, "Run", INTERVIEW_RUN, EvalKind.interview)
    if entry is None:
        return
    header_strip(entry)
    if summary_problem(entry):
        return
    summary = entry.summary
    m = by_name(summary)
    ranking = m.get("interview.ranking_accuracy")
    if ranking is not None:
        st.metric("Ranking accuracy (tier pairs at the seed level)", k_of_n(ranking))
        st.caption(f"95% CI {ci_cell(ranking)} · {ranking.method.value}")
    st.markdown("**Misconception detection**: the planted sentence among the grade's quotes")
    table(
        [
            {"stage": stage, **_cells(m[name])}
            for stage, name in (
                ("given (before the server's filters)", "interview.detection_given"),
                ("kept (after the quote rule and the verifier)", "interview.detection_kept"),
            )
            if name in m
        ]
    )
    st.markdown("**Score bands**")
    table(
        [
            {"band": metric.name.removeprefix("interview.band."), **_cells(metric)}
            for metric in with_prefix(summary, "interview.band.")
        ]
    )
    st.markdown("**Level calibration** (gold and partial, each graded twice)")
    table(
        [
            {"statistic": metric.name.removeprefix("interview.level."), **_cells(metric)}
            for metric in with_prefix(summary, "interview.level.")
        ]
    )
    st.caption("Mean score per level (the trend Page's L tests; a good grader falls toward Staff)")
    table(level_means(entry.run.rows))
    shown = {
        "interview.ranking_accuracy",
        "interview.detection_given",
        "interview.detection_kept",
    }
    other = [
        metric
        for metric in (summary.metrics if summary else [])
        if metric.name not in shown
        and not metric.name.startswith(("interview.band.", "interview.level."))
    ]
    if other:
        st.markdown("**Other metrics**")
        table([metric_row(metric) for metric in other])


# ------------------------------------------------------------------ calibration
def calibration_view(ctx: EvalContext) -> None:
    st.subheader("Judge calibration")
    records = ctx.calibrations.items
    if not records:
        _summary_calibration(ctx)
        return
    keys = judge_keys(records)
    judge = st.selectbox(
        "Judge",
        range(len(keys)),
        index=_default_judge(ctx, keys),
        format_func=lambda i: judge_text(keys[i]),
        key=JUDGE,
    )
    state = calibration_state(records, keys[judge or 0])
    if not state.any_trusted:
        st.error(NO_TRUSTED_JUDGE)
    for criterion in state.untrusted:
        st.warning(f"Judge not trusted for {criterion.value}: its scores don't count.")
    missing = [c for c in E_G_CRITERIA if c not in state.current]
    if missing:
        st.caption("Not calibrated yet: " + ", ".join(c.value for c in missing) + ".")
    table([calibration_row(cal) for cal in state.current.values()])
    st.caption(
        "Trusted = kappa ≥ 0.6 AND its CI lower bound ≥ 0.4 AND no significant leniency "
        "(paired sign test of judge minus owner)."
    )
    for criterion, cal in state.current.items():
        st.markdown(f"**Confusion matrix: {criterion.value}** (rows: you, columns: the judge)")
        table(confusion_rows(cal))
    if state.other_judges:
        st.caption(
            f"{state.other_judges} record(s) belong to other judge keys and are not used: a new "
            "judge version is recalibrated on the same labelled items."
        )


def _default_judge(ctx: EvalContext, keys: list[JudgeKey]) -> int:
    """The judge of the newest judged run, so the panel opens on the judge in use."""
    for entry in ctx.entries:
        if entry.header.judge is not None and entry.header.judge in keys:
            return keys.index(entry.header.judge)
    return 0


def _summary_calibration(ctx: EvalContext) -> None:
    """No calibration records: the newest E-G summary's calibration numbers, if any."""
    st.info(
        "No calibration records yet. The confusion matrix, the kappa CI and leniency appear "
        "once calibration is computed from a label batch."
    )
    entry = next((e for e in ctx.runs_of(EvalKind.generator) if e.summary), None)
    if entry is None:
        st.error(NO_TRUSTED_JUDGE)
        return
    summary = entry.summary
    trust = summary_trust(summary)
    if not any(trust.values()):
        st.error(NO_TRUSTED_JUDGE)
    for name, trusted in trust.items():
        if not trusted:
            st.warning(f"Judge not trusted for {md_literal(name)}: its scores don't count.")
    st.caption(
        f"From run {md_literal(entry.header.run_id)}. Kappa and raw agreement are shown for "
        "every criterion, trusted or not: they are the evidence for the trust state. The note "
        "column carries the leniency p and the label count trust needs."
    )
    table([metric_row(mv) for mv in with_prefix(summary, CALIBRATION_PREFIX)])
