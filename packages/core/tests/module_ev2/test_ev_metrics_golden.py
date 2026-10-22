"""EV-METRICS golden oracle: `metrics.summarize` reproduces every value of the independent
`reference_metrics.py` (to 1e-9) on every fixture run, names each interval's method, and applies
the row hygiene rules (latest attempt, stale and errored rows excluded and counted)."""

from __future__ import annotations

import math

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from test_ev_metrics_support import (
    KINDS,
    datasets,
    expected,
    fixture_run,
    judge_results,
    labels,
    renamed,
    with_rows,
)

from ragforge_client.evals import metrics, stats
from ragforge_client.evals.runfile import RunFile
from ragforge_client.evals.schema import (
    Criterion,
    DatasetRef,
    ErrorClass,
    JudgeKey,
    Method,
    MetricValue,
    RetrievalOutput,
    RunError,
    RunRow,
    Span,
    Split,
)

# Metric names this module adds on top of the oracle (documented in docs/handoff/EV-METRICS.md).
EXTRA_NAMES = {
    "generator.abstain_correct",
    "generator.verified",
    "interview.level.page_l.p_value",
    "interview.level.sde1_vs_staff.sign_p",
    *(f"verifier.false_verified@{r:.2f}" for r in (0.05, 0.10, 0.20, 0.30)),
}


def _all_metrics() -> dict[str, MetricValue]:
    out: dict[str, MetricValue] = {}
    for kind in KINDS:
        summary = metrics.summarize(fixture_run(kind), datasets(), labels(), judge_results())
        out.update(metrics.metric_map(summary))
    return out


# The D0 fixture has 12 labelled pairs per criterion, under the real floor of 50: these tests
# exercise the OTHER trust rules (kappa, CI, leniency), so they pin a floor the fixture clears.
FIXTURE_LABEL_FLOOR = 10


@pytest.fixture(scope="module")
def produced() -> dict[str, MetricValue]:
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(metrics, "MIN_CALIBRATION_LABELS", FIXTURE_LABEL_FLOOR)
        return _produced()


def _produced() -> dict[str, MetricValue]:
    return _all_metrics()


def test_every_oracle_value_is_reproduced_to_1e_9(produced: dict[str, MetricValue]) -> None:
    exp = expected()
    assert len(exp) == 60
    for name, value in exp.items():
        assert name in produced, name
        got = produced[name].value
        if value is None:
            assert got is None, name
        else:
            assert got is not None, name
            assert math.isclose(got, value, rel_tol=0, abs_tol=1e-9), (name, got, value)


def test_no_unexpected_metric_names(produced: dict[str, MetricValue]) -> None:
    """A renamed or invented metric would silently break the Eval page and the gate."""
    assert set(produced) == set(expected()) | EXTRA_NAMES


def test_the_oracle_itself_still_yields_the_committed_values() -> None:
    from test_ev_metrics_support import reference

    ref = reference()
    live = {**ref.retrieval(), **ref.verifier(), **ref.interview(), **ref.generator()}
    assert live == expected()


@pytest.mark.parametrize(
    ("name", "method", "n"),
    [
        ("retrieval.reranked.hit@5", Method.wilson, 12),  # need=any answerable questions
        ("retrieval.reranked.mrr", Method.cluster_bootstrap, 12),
        ("retrieval.reranked.ndcg@10", Method.cluster_bootstrap, 13),  # every answerable one
        ("retrieval.unanswerable.below_tau", Method.wilson, 7),
        ("verifier.span_recall", Method.cluster_bootstrap, 12),  # planted items
        ("verifier.badge_flip_recall", Method.cluster_bootstrap, 10),  # on clean-verified sources
        ("verifier.false_reject.verifier_caused", Method.cluster_bootstrap, 8),  # clean items
        ("interview.ranking_accuracy", Method.clopper_pearson, 16),  # 4 seeds x 4 pairs
        ("interview.level.page_l", Method.exact, 8),  # 4 seeds x 2 tiers
        ("calibration.relevance.kappa_linear", Method.bootstrap, 12),
        ("generator.answer_rate", Method.wilson, 20),
        ("generator.judge.relevance", Method.cluster_bootstrap, 12),  # answered rows only
    ],
)
def test_each_metric_names_its_method_and_true_n(
    produced: dict[str, MetricValue], name: str, method: Method, n: int
) -> None:
    m = produced[name]
    assert (m.method, m.n) == (method, n)
    if method is not Method.exact:
        assert m.ci_low is not None
        assert m.ci_high is not None
        assert m.value is not None
        assert 0 <= m.ci_low <= m.value <= m.ci_high or name.endswith("kappa_linear")


def test_judge_metrics_carry_the_calibration_trust_state(
    produced: dict[str, MetricValue],
) -> None:
    """Trusted needs kappa >= 0.6 AND CI low >= 0.4 AND no leniency: relevance (kappa 0.71) has a
    CI low under 0.4 at n = 12, so it is NOT trusted even though its point value clears 0.6."""
    assert produced["generator.judge.relevance"].trusted is False
    assert produced["calibration.relevance.kappa_linear"].trusted is False
    assert produced["generator.judge.completeness"].trusted is True
    assert produced["generator.answer_rate"].trusted is None  # not judge-derived
    cal = metrics.calibrate(Criterion.relevance, _judge_key(), judge_results(), labels(), seed=0)
    assert cal.kappa_ci is not None
    assert cal.kappa_ci[0] < 0.4
    assert cal.n == 12
    assert sum(map(sum, cal.confusion)) == 12


def _judge_key() -> JudgeKey:
    key = fixture_run("generator").header.judge
    assert key is not None
    return key


def test_a_lenient_judge_is_never_trusted() -> None:
    """Every judge score one above the owner: kappa may stay high-ish, the sign test must veto."""
    key = _judge_key()
    lenient = [
        jr.model_copy(update={"score": min(2, (jr.score or 0) + 1)}) for jr in judge_results()
    ]
    owner_low = [lb.model_copy(update={"scores": dict.fromkeys(lb.scores, 0)}) for lb in labels()]
    cal = metrics.calibrate(Criterion.faithfulness, key, lenient, owner_low)
    assert cal.leniency_p is not None
    assert cal.leniency_p < 0.05
    assert cal.trusted is False


def test_invalid_judge_replies_are_counted_not_imputed() -> None:
    results = [
        jr.model_copy(update={"score": None}) if jr.criterion is Criterion.relevance else jr
        for jr in judge_results()
    ]
    summary = metrics.summarize(fixture_run("generator"), datasets(), labels(), results)
    m = metrics.metric_map(summary)["generator.judge.relevance"]
    assert m.value is None
    assert m.n == 0
    assert "12 invalid judge replies excluded" in m.note


def test_summaries_are_deterministic() -> None:
    run = fixture_run("verifier")
    a = metrics.summarize(run, datasets(), [], [])
    b = metrics.summarize(run, datasets(), [], [])
    assert a == b


# ------------------------------------------------------------------ row hygiene
def test_stale_and_errored_rows_are_excluded_and_counted() -> None:
    run = fixture_run("retrieval")
    stale_ids = {"kafka-ordering"}
    ours_ids = {"producer-durability"}
    transient_ids = {"cache-stampede", "unanswerable-graphql"}

    def change(r: RunRow) -> RunRow:
        if r.item_id in stale_ids:
            return r.model_copy(update={"stale": True})
        if r.item_id in ours_ids:
            return r.model_copy(
                update={"output": None, "error": RunError(cls=ErrorClass.ours, message="boom")}
            )
        if r.item_id in transient_ids:
            err = RunError(cls=ErrorClass.provider_transient, status=429, message="rate limited")
            return r.model_copy(update={"output": None, "error": err})
        return r

    summary = metrics.summarize(with_rows(run, change), datasets(), [], [])
    assert (summary.stale_items, summary.errors_ours, summary.errors_transient) == (1, 1, 2)
    m = metrics.metric_map(summary)
    assert m["retrieval.reranked.hit@5"].n == 12 - 3
    assert m["retrieval.unanswerable.below_tau"].n == 7 - 1


def test_the_latest_attempt_wins_so_a_rerun_transient_error_is_not_counted() -> None:
    run = fixture_run("retrieval")
    first = run.rows[0]
    err = RunError(cls=ErrorClass.provider_transient, status=503, message="unavailable")
    failed = first.model_copy(update={"output": None, "error": err, "attempt": 1})
    retried = first.model_copy(update={"attempt": 2})
    resumed = renamed(run, "resumed")
    resumed = type(run)(run.path, resumed.header, [failed, *run.rows[1:], retried], run.footer)
    summary = metrics.summarize(resumed, datasets(), [], [])
    assert summary.errors_transient == 0
    assert metrics.metric_map(summary) == metrics.metric_map(
        metrics.summarize(run, datasets(), [], [])
    )


def test_a_run_from_a_different_dataset_is_refused() -> None:
    run = fixture_run("retrieval")
    other = renamed(run, "x", dataset=DatasetRef(name="demo", digest="0" * 64, splits=[Split.dev]))
    with pytest.raises(metrics.MetricsError, match="digest mismatch"):
        metrics.summarize(other, datasets(), [], [])


def test_a_row_of_the_wrong_output_kind_is_refused() -> None:
    run = fixture_run("verifier")
    bad = with_rows(
        run,
        lambda r: (
            r.model_copy(update={"output": RetrievalOutput(stages={"fused": []})})
            if r.item_id == "c-aof"
            else r
        ),
    )
    with pytest.raises(metrics.MetricsError, match="VerifyOutput"):
        metrics.summarize(bad, datasets(), [], [])


def test_badge_flip_counts_only_sources_verified_clean_in_the_same_run() -> None:
    """If a clean source is NOT verified in this run, its mutations say nothing about flips."""
    run = fixture_run("verifier")
    before = metrics.metric_map(metrics.summarize(run, datasets(), [], []))
    unverify = with_rows(
        run,
        lambda r: (
            r.model_copy(update={"output": r.output.model_copy(update={"status": "partial"})})
            if r.item_id == "c-lag" and r.output is not None
            else r
        ),
    )
    after = metrics.metric_map(metrics.summarize(unverify, datasets(), [], []))
    # c-lag has 2 planted items (p-typo-2 flipped, p-entity-2 flipped-unlocated)
    assert before["verifier.badge_flip_recall"].n == 10
    assert after["verifier.badge_flip_recall"].n == 8
    assert after["verifier.span_recall"] == before["verifier.span_recall"]


def _set_output(run_id: str, item_id: str, **update: object) -> RunFile:
    run = fixture_run("verifier")

    def change(r: RunRow) -> RunRow:
        if r.item_id != item_id or r.output is None:
            return r
        return r.model_copy(update={"output": r.output.model_copy(update=update)})

    return with_rows(run, change, run_id)


def test_span_overlap_is_half_open_so_a_touching_flag_is_not_a_catch() -> None:
    """p-number-1's span is [43, 54): a verifier span [30, 43) touches it but shares no char."""
    touching = _set_output("touch", "p-number-1", verifier_spans=[Span(start=30, end=43)])
    inside = _set_output("inside", "p-number-1", verifier_spans=[Span(start=53, end=60)])
    m_touch = metrics.metric_map(metrics.summarize(touching, datasets(), [], []))
    m_inside = metrics.metric_map(metrics.summarize(inside, datasets(), [], []))
    assert m_touch["verifier.span_recall.number_drift"].value == 0.0
    assert m_inside["verifier.span_recall.number_drift"].value == 0.5


def test_a_verifier_objection_is_verifier_caused_whatever_coverage_did() -> None:
    """r4 P2: c-deploy was rejected by coverage alone; add a verifier objection and it moves to
    verifier-caused (coverage-caused counts coverage-ALONE rejects only)."""
    both = _set_output("both", "c-deploy", verifier_objected=True)
    m = metrics.metric_map(metrics.summarize(both, datasets(), [], []))
    assert m["verifier.false_reject.verifier_caused"].value == 0.25
    assert m["verifier.false_reject.coverage_caused"].value == 0.0
    assert m["verifier.false_reject"].value == 0.25


def test_significant_leniency_alone_vetoes_trust(monkeypatch: pytest.MonkeyPatch) -> None:
    """30 items, the judge agrees on 20 and is one point MORE lenient on 10: kappa and its CI
    clear the bar, so only the sign test (10 up, 0 down: p = 2/1024) can refuse trust."""
    monkeypatch.setattr(metrics, "MIN_CALIBRATION_LABELS", FIXTURE_LABEL_FLOOR)
    key = _judge_key()
    template = judge_results()[0]
    owner_scores = [0] * 10 + [1] * 10 + [2] * 10
    judge_scores = [1] * 5 + [0] * 5 + [2] * 5 + [1] * 5 + [2] * 10
    results, owner = [], []
    for i, (o, j) in enumerate(zip(owner_scores, judge_scores, strict=True)):
        d = f"{i:064x}"
        results.append(
            template.model_copy(
                update={
                    "item_digest": d,
                    "cache_key": d,
                    "criterion": Criterion.faithfulness,
                    "score": j,
                    "judge": key,
                }
            )
        )
        owner.append(
            labels()[0].model_copy(
                update={"item_digest": d, "item_id": f"i{i}", "scores": {Criterion.faithfulness: o}}
            )
        )
    cal = metrics.calibrate(Criterion.faithfulness, key, results, owner)
    assert cal.kappa_linear is not None
    assert cal.kappa_linear >= 0.6
    assert cal.kappa_ci is not None
    assert cal.kappa_ci[0] >= 0.4
    assert cal.leniency_p == pytest.approx(2 / 1024)
    assert cal.trusted is False
    fair = [
        r.model_copy(update={"score": o}) if i < 10 else r
        for i, (r, o) in enumerate(zip(results, owner_scores, strict=True))
    ]
    fair[0] = fair[0].model_copy(update={"score": 1})  # one disagreement up, one down
    fair[25] = fair[25].model_copy(update={"score": 1})
    assert metrics.calibrate(Criterion.faithfulness, key, fair, owner).trusted is True


# ------------------------------------------------------------------ small-n intervals (r1 P1)
def test_a_perfect_clustered_proportion_reports_the_exact_bound_not_a_zero_width_ci() -> None:
    """3/3 in 2 clusters: every bootstrap resample is 1.0, so the percentile CI is [1, 1]."""
    m = metrics.clustered_mean("x", [[1.0, 1.0], [1.0]], binary=True)
    assert m.method is Method.clopper_pearson
    assert m.ci_low is not None
    assert abs(m.ci_low - (0.025 ** (1 / 3))) < 1e-9  # CP low of 3/3 is (alpha/2)^(1/n)
    assert m.ci_high == 1.0
    zero = metrics.clustered_mean("x", [[0.0], [0.0]], binary=True)
    assert zero.ci_high is not None
    assert abs(zero.ci_high - (1 - 0.025 ** (1 / 2))) < 1e-9


def test_a_degenerate_non_binary_bootstrap_reports_no_interval() -> None:
    m = metrics.clustered_mean("judge", [[2.0, 2.0], [2.0]], binary=False)
    assert (m.method, m.ci_low, m.ci_high) == (Method.point, None, None)
    assert "degenerate" in m.note


@settings(max_examples=60, deadline=None)
@given(
    st.lists(st.lists(st.sampled_from([0.0, 1.0]), min_size=1, max_size=4), min_size=1, max_size=12)
)
def test_a_binary_clustered_ci_is_never_narrower_than_clopper_pearson(
    clusters: list[list[float]],
) -> None:
    flat = [v for c in clusters for v in c]
    low, high = stats.clopper_pearson(round(sum(flat)), len(flat))
    m = metrics.clustered_mean("x", clusters, binary=True)
    assert m.ci_low is not None
    assert m.ci_high is not None
    assert m.value is not None
    assert m.ci_low <= low + 1e-12
    assert m.ci_high >= high - 1e-12
    assert m.ci_low <= m.value <= m.ci_high
