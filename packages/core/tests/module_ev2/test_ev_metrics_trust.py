"""EV-METRICS round 3: judge trust needs evidence, judge means use one judge key, binary is
declared not inferred, duplicate grades are refused, the E-V false-verified curve and the
confirmed false-verified k/n, and the one-sided sign-flip test."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import replace

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from test_ev_metrics_support import datasets, fixture_run, judge_results, labels, with_rows

from ragforge_client.evals import gate, metrics, stats
from ragforge_client.evals.schema import (
    AnswerOutput,
    Criterion,
    GateOutcome,
    GradeOutput,
    JudgeKey,
    JudgeResult,
    Label,
    LabelKind,
    Method,
    MetricValue,
)


def _key() -> JudgeKey:
    key = fixture_run("generator").header.judge
    assert key is not None
    return key


def _calibration_set(
    owner_scores: list[int], judge_scores: list[int], crit: Criterion = Criterion.faithfulness
) -> tuple[list[JudgeResult], list[Label]]:
    template, label = judge_results()[0], labels()[0]
    results, owner = [], []
    for i, (o, j) in enumerate(zip(owner_scores, judge_scores, strict=True)):
        d = f"{i:064x}"
        results.append(
            template.model_copy(
                update={
                    "item_digest": d,
                    "cache_key": d,
                    "criterion": crit,
                    "score": j,
                    "judge": _key(),
                }
            )
        )
        owner.append(
            label.model_copy(update={"item_digest": d, "item_id": f"i{i}", "scores": {crit: o}})
        )
    return results, owner


# ------------------------------------------------------------------ trust at tiny n (r2 P1)
@pytest.mark.parametrize("scores", [[0, 2], [0, 1, 2], [2, 2, 0]])
def test_a_few_perfectly_agreeing_labels_never_make_a_judge_trusted(scores: list[int]) -> None:
    """Kappa is 1.0 and every defined resample is 1.0: the percentile CI would be [1, 1] and
    pass 'CI low >= 0.4'. It must read as no CI, and the judge as untrusted."""
    results, owner = _calibration_set(scores, scores)
    cal = metrics.calibrate(Criterion.faithfulness, _key(), results, owner)
    assert cal.kappa_linear == 1.0
    assert cal.kappa_ci is None
    assert cal.trusted is False


def test_kappa_ci_is_none_when_degenerate_or_mostly_undefined() -> None:
    assert stats.kappa_ci([(0, 0), (2, 2)]) is None
    # the defined-resample floor itself: a statistic undefined on most resamples gives no CI
    units = [0.0, 1.0, 2.0, 3.0]

    def rarely(sample: Sequence[float]) -> float | None:
        return sum(sample) if max(sample) == 3.0 and min(sample) == 0.0 else None

    assert stats.bootstrap_ci(units, rarely, n_resamples=400) is not None
    assert stats.bootstrap_ci(units, rarely, n_resamples=400, min_defined=300) is None


@settings(max_examples=60, deadline=None)
@given(
    st.lists(
        st.tuples(st.integers(0, 2), st.integers(0, 2)),
        min_size=0,
        max_size=15,
    )
)
def test_a_kappa_ci_is_never_zero_width(pairs: list[tuple[int, int]]) -> None:
    ci = stats.kappa_ci(pairs, n_resamples=300)
    assert ci is None or ci[0] < ci[1]


def _fair_30() -> tuple[list[JudgeResult], list[Label]]:
    """30 items, the judge agrees on 28 and is off by one once each way: trusted at n = 30."""
    owner = [0] * 10 + [1] * 10 + [2] * 10
    judge = list(owner)
    judge[0], judge[25] = 1, 1
    return _calibration_set(owner, judge)


def test_the_minimum_label_count_is_part_of_trust(monkeypatch: pytest.MonkeyPatch) -> None:
    results, owner = _fair_30()
    monkeypatch.setattr(metrics, "MIN_CALIBRATION_LABELS", 30)
    assert metrics.calibrate(Criterion.faithfulness, _key(), results, owner).trusted is True
    monkeypatch.setattr(metrics, "MIN_CALIBRATION_LABELS", 31)
    assert metrics.calibrate(Criterion.faithfulness, _key(), results, owner).trusted is False


def test_the_label_floor_is_documented_in_the_metric_note() -> None:
    summary = metrics.summarize(fixture_run("generator"), datasets(), labels(), judge_results())
    kappa = metrics.metric_map(summary)["calibration.relevance.kappa_linear"]
    assert f"n >= {metrics.MIN_CALIBRATION_LABELS}" in kappa.note


# ------------------------------------------------------------------ judge key (r2 P2)
def test_judge_refs_of_another_judge_key_are_excluded_and_counted() -> None:
    run = fixture_run("generator")
    other = _key().model_copy(update={"model_version": "moved-alias"})
    results = judge_results()
    moved = {
        jr.cache_key
        for jr in results
        if jr.criterion is Criterion.relevance and jr.score is not None
    }
    victim = sorted(moved)[0]
    mixed = [
        jr.model_copy(update={"judge": other}) if jr.cache_key == victim else jr for jr in results
    ]
    before = metrics.metric_map(metrics.summarize(run, datasets(), labels(), results))
    after = metrics.metric_map(metrics.summarize(run, datasets(), labels(), mixed))
    name = "generator.judge.relevance"
    assert after[name].n == before[name].n - 1
    assert "1 judge refs from another judge key excluded" in after[name].note


# ------------------------------------------------------------------ binary is declared (r2 P2)
def test_a_judge_mean_of_ones_is_not_treated_as_a_binomial() -> None:
    """All four 0..2 scores happen to be 1: a binomial CP [0.40, 1.0] would cap at 1 on a 0..2
    scale. Declared non-binary, the degenerate bootstrap reports no interval instead."""
    m = metrics.clustered_mean("generator.judge.x", [[1.0], [1.0], [1.0], [1.0]], binary=False)
    assert (m.method, m.ci_low, m.ci_high) == (Method.point, None, None)
    mixed = metrics.clustered_mean("generator.judge.x", [[1.0], [0.0], [1.0], [1.0]], binary=False)
    assert mixed.method is Method.cluster_bootstrap
    assert "Clopper-Pearson" not in mixed.note


def test_a_declared_binary_metric_with_another_value_is_refused() -> None:
    with pytest.raises(metrics.MetricsError, match="declared binary"):
        metrics.clustered_mean("x", [[1.0, 2.0]], binary=True)


# ------------------------------------------------------------------ duplicate grades (r2 P2)
def test_a_duplicated_seed_level_grade_is_refused_and_the_gate_is_not_evaluable() -> None:
    run = fixture_run("interview")
    first = next(r for r in run.rows if isinstance(r.output, GradeOutput))
    dup = first.model_copy(update={"item_id": first.item_id + "-dup"})
    bad = replace(with_rows(run, lambda r: r, "dup"), rows=[*run.rows, dup])
    with pytest.raises(metrics.MetricsError, match="duplicates the grade"):
        metrics.summarize(bad, datasets(), [], [])
    verdict = gate.evaluate(run, bad, [], datasets())
    assert verdict.outcome is GateOutcome.not_evaluable


# ------------------------------------------------------------------ false-verified (r2 P1)
def test_false_verified_rate_known_value() -> None:
    # base rate 10%, recall 0.8 (miss 0.2), false-reject 0.1: 0.02 / (0.02 + 0.81)
    assert metrics.false_verified_rate(0.10, 0.2, 0.9) == pytest.approx(0.02 / 0.83)
    assert metrics.false_verified_rate(0.10, 0.0, 0.9) == 0.0  # a perfect verifier
    assert metrics.false_verified_rate(0.10, 0.0, 0.0) is None  # nothing ever verifies


def _mv(value: float, low: float, high: float, n: int = 20) -> MetricValue:
    return MetricValue(
        name="x", value=value, ci_low=low, ci_high=high, n=n, method=Method.cluster_bootstrap
    )


unit = st.floats(0.0, 1.0, allow_nan=False)


@settings(max_examples=120, deadline=None)
@given(
    st.lists(unit, min_size=3, max_size=3).map(sorted),
    st.lists(st.floats(0.0, 0.95, allow_nan=False), min_size=3, max_size=3).map(sorted),
)
def test_the_curve_bounds_bracket_the_value_and_rise_with_the_base_rate(
    recall: list[float], reject: list[float]
) -> None:
    rec = _mv(recall[1], recall[0], recall[2])
    fr = _mv(reject[1], reject[0], reject[2])
    curve = metrics.false_verified_curve(rec, fr)
    assert [p.base_rate for p in curve] == list(metrics.FALSE_VERIFIED_BASE_RATES)
    for p in curve:
        assert p.value is not None
        assert p.low is not None
        assert p.high is not None
        assert p.low - 1e-12 <= p.value <= p.high + 1e-12
        assert 0.0 <= p.low <= p.high <= 1.0
    values = [p.value for p in curve if p.value is not None]
    assert values == sorted(values)  # more wrong answers, more wrong-but-verified


def test_the_fixture_verifier_publishes_the_curve() -> None:
    m = metrics.metric_map(metrics.summarize(fixture_run("verifier"), datasets(), [], []))
    rec, fr = m["verifier.badge_flip_recall"], m["verifier.false_reject"]
    assert rec.value is not None
    assert fr.value is not None
    for rate in metrics.FALSE_VERIFIED_BASE_RATES:
        fv = m[f"verifier.false_verified@{rate:.2f}"]
        assert fv.method is Method.point
        expected = metrics.false_verified_rate(rate, 1 - rec.value, 1 - fr.value)
        assert fv.value is not None
        assert expected is not None
        assert math.isclose(fv.value, expected)
        assert "conservative plug-in bounds" in fv.note


# ------------------------------------------------------------------ confirmed k/n (r2 P1)
def _verified_digests() -> dict[str, list[str]]:
    run = fixture_run("generator")
    by_key = {jr.cache_key: jr for jr in judge_results()}
    out: dict[str, list[str]] = {}
    for r in run.rows:
        if isinstance(r.output, AnswerOutput) and r.output.status == "verified":
            out[r.item_id] = sorted({by_key[k].item_digest for k in r.judge_refs if k in by_key})
    return out


def _adjudication(digest: str, confirmed: bool) -> Label:
    return labels()[0].model_copy(
        update={
            "item_digest": digest,
            "kind": LabelKind.adjudication,
            "scores": {},
            "confirmed": confirmed,
        }
    )


def test_confirmed_false_verified_is_k_over_the_verified_answers_with_an_exact_ci() -> None:
    run = fixture_run("generator")
    verified = _verified_digests()
    flagged = [d[0] for d in verified.values() if d][:2]
    assert len(flagged) == 2
    adj = [_adjudication(flagged[0], True), _adjudication(flagged[1], False)]
    m = metrics.confirmed_false_verified(run, judge_results(), adj)
    assert m is not None
    assert (m.n, m.method, m.value) == (len(verified), Method.clopper_pearson, 1 / len(verified))
    assert (m.ci_low, m.ci_high) == stats.clopper_pearson(1, len(verified))
    assert "2 judge-flagged answers adjudicated, 1 confirmed" in m.note
    summary = metrics.summarize(run, datasets(), [*labels(), *adj], judge_results())
    assert "generator.false_verified.confirmed" in metrics.metric_map(summary)


def test_no_adjudication_is_not_reported_as_zero_false_verified() -> None:
    assert metrics.confirmed_false_verified(fixture_run("generator"), judge_results(), []) is None
    stranger = [_adjudication("f" * 64, True)]  # an answer of some other run
    assert (
        metrics.confirmed_false_verified(fixture_run("generator"), judge_results(), stranger)
        is None
    )


# ------------------------------------------------------------------ one-sided sign-flip
def test_one_sided_sign_flip_known_values() -> None:
    assert stats.sign_flip_test([1, 1, 1], alternative="greater").p_value == 1 / 8
    assert stats.sign_flip_test([1, 1, 1], alternative="less").p_value == 1.0
    assert stats.sign_flip_test([-1, -1, -1], alternative="less").p_value == 1 / 8
    assert stats.sign_flip_test([1, 1, 1]).p_value == 2 / 8


@settings(max_examples=100, deadline=None)
@given(st.lists(st.integers(-4, 4), min_size=1, max_size=10))
def test_two_sided_is_twice_the_smaller_one_sided_tail(diffs: list[int]) -> None:
    """The sign-flip null is symmetric, so for S != 0 the two tails are disjoint mirrors."""
    floats = [float(d) for d in diffs]
    if math.fsum(floats) == 0:
        return
    two = stats.sign_flip_test(floats).p_value
    less = stats.sign_flip_test(floats, alternative="less").p_value
    greater = stats.sign_flip_test(floats, alternative="greater").p_value
    assert two == pytest.approx(min(1.0, 2 * min(less, greater)))
    assert less + greater >= 1.0  # both tails include the observed value


def test_the_real_floor_is_the_adrs_50_labels_per_criterion() -> None:
    """Owner decision 2026-10-21: 50, as ADR-0017 §4 says (30 good pairs are NOT enough)."""
    assert metrics.MIN_CALIBRATION_LABELS == 50
    results, owner = _fair_30()
    assert metrics.calibrate(Criterion.faithfulness, _key(), results, owner).trusted is False
