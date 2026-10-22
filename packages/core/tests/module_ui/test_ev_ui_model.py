"""EV-UI pure view models: formatting, held-out policy (fail closed), compare rules, flips and
their exploratory p-values, the catching layer against the golden oracle, the SLO rule, the
blind queue and calibration selection. Property tests with hypothesis where a rule is general."""

from __future__ import annotations

import json
import tomllib
from datetime import timedelta
from pathlib import Path
from uuid import UUID

import pytest
from hypothesis import given
from hypothesis import strategies as st
from test_ev_ui_support import (
    DEMO,
    FIXTURES,
    JUDGE,
    T0,
    answer_row,
    calibration,
    header,
    judged,
    summary_of,
)

from ragforge_client.evals.interfaces import Datasets
from ragforge_client.evals.runfile import RunFile, read_run
from ragforge_client.evals.schema import (
    E_G_CRITERIA,
    Criterion,
    EvalKind,
    JudgeKey,
    Label,
    LabelKind,
    Method,
    MetricValue,
    Need,
    Question,
    QuestionSet,
    RunRow,
    RunSummary,
    ShownRef,
    SnapshotRef,
    Span,
    Split,
    VerifierItem,
    VerifierSet,
    VerifyOutput,
)
from ragforge_ui.eval_view.model import (
    NONE,
    NOT_TRUSTED,
    QueueItem,
    binary_outcomes,
    calibration_state,
    caught_layer,
    ci_cell,
    comparability,
    criteria_label,
    fmt,
    is_heldout,
    judge_index,
    k_of_n,
    labelled_digests,
    mcnemar_exact,
    metric_deltas,
    metric_row,
    paired,
    queue_items,
    slo_verdict,
    summary_trust,
    value_text,
    visible_rows,
)


def _q(qid: str, split: Split = Split.dev, answerable: bool = True) -> Question:
    if not answerable:
        return Question(id=qid, split=split, question=f"{qid}?", answerable=False)
    return Question(id=qid, split=split, question=f"{qid}?", relevance={"file:a.md": 2})


def _qs(*questions: Question) -> QuestionSet:
    return QuestionSet(name="synthetic", questions=list(questions))


def _run(rows: list[RunRow], *, splits: tuple[Split, ...] = (Split.dev,)) -> RunFile:
    return RunFile(Path("run.jsonl"), header("r", splits=splits), rows, None)


def _metric(name: str, value: float | None, **kw: object) -> MetricValue:
    return MetricValue.model_validate(
        {"name": name, "value": value, "n": 20, "method": "point"} | kw
    )


# ------------------------------------------------------------------ formatting
@pytest.mark.parametrize(
    ("value", "text"),
    [
        (None, NONE),
        (float("nan"), NONE),
        (0.0, "0"),
        (1.0, "1"),
        (0.9375, "0.938"),
        (0.08333333333333333, "0.0833"),
        (0.611111111111111, "0.611"),
        (397.0, "397.0"),
        (1234.56, "1234.6"),
        (-0.1, "-0.1"),
    ],
)
def test_fmt(value: float | None, text: str) -> None:
    assert fmt(value) == text


@given(st.floats(min_value=-1e7, max_value=1e7, allow_nan=False))
def test_fmt_never_prints_an_exponent_in_the_range_of_eval_numbers(value: float) -> None:
    if abs(value) >= 1e-4 or value == 0:
        assert "e" not in fmt(value)


def test_an_untrusted_judge_metric_hides_its_value_and_interval() -> None:
    row = metric_row(
        _metric("generator.judge.relevance", 1.9, trusted=False, ci_low=1.0, ci_high=2.0)
    )
    assert (row["value"], row["95% CI"], row["judge"]) == (NOT_TRUSTED, NONE, NOT_TRUSTED)
    untrusted = _metric("generator.judge.relevance", 1.9, trusted=False, ci_low=1.0, ci_high=2.0)
    assert ci_cell(untrusted) == NONE
    assert ci_cell(untrusted.model_copy(update={"trusted": True})) == "[1, 2]"
    assert ci_cell(_metric("retrieval.reranked.mrr", 0.5, ci_low=0.4, ci_high=0.6)) == "[0.4, 0.6]"
    trusted = metric_row(_metric("generator.judge.relevance", 1.9, trusted=True))
    assert (trusted["value"], trusted["judge"]) == ("1.9", "trusted")
    plain = metric_row(_metric("retrieval.reranked.mrr", 0.5))
    assert plain["judge"] == ""


def test_k_of_n_only_when_the_value_is_a_whole_count() -> None:
    assert k_of_n(_metric("x", 15 / 16, n=16)) == "15/16"
    assert k_of_n(_metric("x", 0.9375, n=92)) == "0.938 (n 92)"
    assert k_of_n(_metric("x", None, n=0)) == f"{NONE} (n 0)"


# ------------------------------------------------------------------ held-out
def test_heldout_rows_are_hidden_and_revealed_only_on_request() -> None:
    ds = Datasets(questions=_qs(_q("a"), _q("b", Split.heldout2)))
    run = _run([answer_row("a", "x"), answer_row("b", "y")], splits=(Split.dev, Split.heldout2))
    hidden = visible_rows(run, ds, reveal=False)
    assert ([r.item_id for r in hidden.rows], hidden.hidden) == (["a"], 1)
    assert [r.item_id for r in visible_rows(run, ds, reveal=True).rows] == ["a", "b"]


def test_an_unknown_split_is_held_out_only_when_the_run_declares_heldout2() -> None:
    row = answer_row("unknown", "x")
    with_heldout = header("r", splits=(Split.dev, Split.heldout2))
    dev_only = header("r", splits=(Split.dev,))
    assert is_heldout(row, with_heldout, Datasets())
    assert not is_heldout(row, dev_only, Datasets())


def test_a_verifier_item_built_from_a_heldout_answer_is_held_out() -> None:
    vs = VerifierSet.model_validate(json.loads((DEMO / "verifier.json").read_text()))
    item = vs.items[0]
    ds = Datasets(questions=_qs(_q(item.source_id, Split.heldout2)), verifier=vs)
    row = RunRow(item_id=item.id, input_digest="x")
    head = header("v", EvalKind.verifier)
    assert is_heldout(row, head, ds)
    assert not is_heldout(row, head, Datasets(questions=_qs(_q(item.source_id)), verifier=vs))


def _demo_verifier() -> VerifierSet:
    return VerifierSet.model_validate(json.loads((DEMO / "verifier.json").read_text()))


def test_a_verifier_item_linked_to_no_question_fails_closed() -> None:
    """D0's verifier ids match no question id: in a run declaring heldout2 they are hidden."""
    vs = _demo_verifier()
    rows = [RunRow(item_id=i.id, input_digest="x") for i in vs.items]
    declares = header("v", EvalKind.verifier, splits=(Split.dev, Split.heldout2))
    dev_only = header("v", EvalKind.verifier, splits=(Split.dev,))
    with_questions = Datasets(questions=_demo_questions(), verifier=vs)
    no_questions = Datasets(verifier=vs)
    for ds in (with_questions, no_questions):
        assert all(is_heldout(r, declares, ds) for r in rows)
        assert not any(is_heldout(r, dev_only, ds) for r in rows)


def test_a_verifier_item_linked_to_a_dev_question_is_dev() -> None:
    vs = _demo_verifier()
    item = vs.items[0]
    declares = header("v", EvalKind.verifier, splits=(Split.dev, Split.heldout2))
    row = RunRow(item_id=item.id, input_digest="x")
    assert not is_heldout(row, declares, Datasets(questions=_qs(_q(item.source_id)), verifier=vs))
    assert not is_heldout(row, declares, Datasets(questions=_qs(_q(item.id)), verifier=vs))


@given(
    st.lists(
        st.tuples(st.sampled_from([Split.dev, Split.heldout2, None]), st.booleans()),
        max_size=12,
    )
)
def test_hidden_rows_are_exactly_the_heldout_or_unknown_ones(
    spec: list[tuple[Split | None, bool]],
) -> None:
    """Property: with reveal off, no visible row is held out, and every dropped row is."""
    questions = [_q(f"q{i}", split) for i, (split, _) in enumerate(spec) if split is not None]
    ds = Datasets(questions=_qs(*questions))
    run = _run(
        [answer_row(f"q{i}", "x") for i in range(len(spec))], splits=(Split.dev, Split.heldout2)
    )
    shown = visible_rows(run, ds, reveal=False)
    known = {q.id: q.split for q in questions}
    assert all(known.get(r.item_id) is Split.dev for r in shown.rows)
    assert shown.hidden == sum(1 for s, _ in spec if s is not Split.dev)


# ------------------------------------------------------------------ compare
def _snap(content: str) -> SnapshotRef:
    return SnapshotRef(collection_id="c", content_digest=content, index_digest="i")


def test_comparability_rules() -> None:
    base = header("a", snapshot=_snap("x"))
    assert comparability(base, header("b", snapshot=_snap("x"))) is None
    assert comparability(header("a"), header("b")) is None  # both demo, same dataset
    assert "different snapshot content" in (
        comparability(base, header("b", snapshot=_snap("y"))) or ""
    )
    assert "different corpora" in (comparability(base, header("b")) or "")
    assert "different eval kinds" in (comparability(base, header("b", EvalKind.verifier)) or "")
    other = header("b", snapshot=_snap("x")).model_copy(
        update={"dataset": base.dataset.model_copy(update={"digest": "e" * 64})}
    )
    assert "different datasets" in (comparability(base, other) or "")


def test_the_index_digest_may_differ_between_arms() -> None:
    a = header("a", snapshot=_snap("x"))
    b = header("b", snapshot=SnapshotRef(collection_id="c", content_digest="x", index_digest="j"))
    assert comparability(a, b) is None


def test_metric_deltas_never_show_a_delta_for_an_untrusted_judge_metric() -> None:
    base = RunSummary(
        run_id="a",
        eval_kind=EvalKind.generator,
        header_digest="h",
        metrics=[
            _metric("generator.answer_rate", 0.6),
            _metric("generator.judge.relevance", 1.0, trusted=True),
        ],
    )
    cand = base.model_copy(
        update={
            "metrics": [
                _metric("generator.answer_rate", 0.7),
                _metric("generator.judge.relevance", 1.5, trusted=False),
            ]
        }
    )
    rows = {r["metric"]: r for r in metric_deltas(base, cand)}
    assert rows["generator.answer_rate"]["delta"] == "+0.1"
    assert rows["generator.judge.relevance"]["delta"] == NOT_TRUSTED
    assert rows["generator.judge.relevance"]["candidate"] == NOT_TRUSTED


def test_paired_flips_and_discordance() -> None:
    base = [answer_row("a", "x"), answer_row("b", "x"), answer_row("c", "x", status="partial")]
    cand = [
        answer_row("a", "x", status="partial"),
        answer_row("b", "x"),
        answer_row("c", "x"),
        answer_row("d", "x"),
    ]
    result = paired(base, cand, {})
    verified = next(d for d in result.discordance if d.measure == "verified")
    assert (verified.pairs, verified.lost, verified.gained) == (3, 1, 1)
    assert {(f.item_id, f.baseline, f.candidate) for f in result.flips} == {
        ("a", "pass", "fail"),
        ("c", "fail", "pass"),
    }
    assert result.only_one == 1


HIT5 = "reranked: relevant page in top 5"
ALL5 = "reranked: all relevant pages in top 5"


def test_the_retrieval_flip_measure_pairs_exactly_the_questions_of_hit_at_5() -> None:
    run = read_run(FIXTURES / "run-retrieval.jsonl")
    qs = {q.id: q for q in _demo_questions().questions}
    hits = [o[HIT5] for o in (binary_outcomes(r, qs) for r in run.rows) if HIT5 in o]
    metric = next(
        m for m in summary_of("retrieval").metrics if m.name == "retrieval.reranked.hit@5"
    )
    # the same questions as the metric's definition (reference_metrics: need=any, answerable)
    assert len(hits) == sum(1 for q in qs.values() if q.answerable and q.need is Need.any)
    assert not any(
        ALL5 in binary_outcomes(r, qs) for r in run.rows if qs[r.item_id].need is Need.any
    )
    assert sum(hits) / len(hits) == pytest.approx(metric.value)


def test_a_need_all_question_needs_every_relevant_page_in_the_top_5() -> None:
    q = Question.model_validate(
        {
            "id": "q",
            "split": "dev",
            "question": "q?",
            "need": "all",
            "relevance": {"file:a.md": 2, "file:b.md": 1},
        }
    )
    qs = {"q": q}

    def outcome(*pages: str) -> dict[str, bool]:
        row = RunRow.model_validate(
            {
                "item_id": "q",
                "input_digest": "x",
                "output": {"kind": "retrieval", "stages": {"reranked": list(pages)}},
            }
        )
        return binary_outcomes(row, qs)

    assert outcome("file:a.md", "file:b.md") == {ALL5: True}
    assert outcome("file:a.md", "file:x.md") == {ALL5: False}


# ------------------------------------------------------------------ verifier flips
def _planted() -> VerifierItem:
    """A planted D0 item (with its real span), renamed so the test reads simply."""
    item = next(i for i in _demo_verifier().items if i.mutation is not None)
    return item.model_copy(update={"id": "p-1", "source_id": "c-1"})


def _clean() -> VerifierItem:
    item = next(i for i in _demo_verifier().items if i.mutation is None)
    return item.model_copy(update={"id": "c-1", "source_id": "c-1"})


def _at_span() -> Span:
    span = _planted().span
    assert span is not None
    return span


def _away() -> Span:
    span = _at_span()
    return Span(start=span.end, end=span.end + 1)  # half-open: touches, never overlaps


def _vrow(item_id: str, status: str, *, flag: Span | None = None) -> RunRow:
    out = VerifyOutput(
        status=status,
        coverage_spans=[],
        verifier_spans=[flag] if flag else [],
        verifier_ran=True,
        verifier_objected=flag is not None,
        unlocated_verifier_claims=0,
    )
    return RunRow(item_id=item_id, input_digest="x", output=out)


def test_a_planted_item_newly_caught_reads_as_fail_to_pass() -> None:
    items = {"p-1": _planted(), "c-1": _clean()}
    base = [_vrow("p-1", "verified"), _vrow("c-1", "verified")]
    cand = [_vrow("p-1", "unsupported", flag=_at_span()), _vrow("c-1", "verified")]
    result = paired(base, cand, {}, items)
    by = {d.measure: d for d in result.discordance}
    for measure in ("planted: caught at the span", "planted: badge flipped"):
        assert (by[measure].pairs, by[measure].lost, by[measure].gained) == (1, 0, 1)
    assert by["clean: verified clean"].pairs == 1
    assert "verified" not in by  # planted and clean are never pooled
    assert {(f.item_id, f.measure, f.baseline, f.candidate) for f in result.flips} == {
        ("p-1", "planted: caught at the span", "fail", "pass"),
        ("p-1", "planted: badge flipped", "fail", "pass"),
    }


def test_a_flag_away_from_the_planted_span_flips_the_badge_but_is_no_span_catch() -> None:
    row = _vrow("p-1", "unsupported", flag=_away())
    assert binary_outcomes(row, {}, {"p-1": _planted()}, frozenset({"c-1"})) == {
        "planted: caught at the span": False,
        "planted: badge flipped": True,
    }


def test_a_clean_item_rejected_is_a_fail_and_an_unknown_item_has_no_measure() -> None:
    assert binary_outcomes(_vrow("c-1", "partial"), {}, {"c-1": _clean()}) == {
        "clean: verified clean": False
    }
    assert binary_outcomes(_vrow("zz", "verified"), {}, {"c-1": _clean()}) == {}


def _demo_questions() -> QuestionSet:
    return QuestionSet.model_validate(tomllib.loads((DEMO / "questions.toml").read_text()))


def test_mcnemar_known_values() -> None:
    assert mcnemar_exact(0, 0) == 1.0
    assert mcnemar_exact(0, 5) == pytest.approx(2 / 32)
    assert mcnemar_exact(1, 1) == 1.0
    assert mcnemar_exact(2, 8) == pytest.approx(2 * (1 + 10 + 45) / 1024)


@given(st.integers(0, 60), st.integers(0, 60))
def test_mcnemar_is_a_symmetric_probability(lost: int, gained: int) -> None:
    p = mcnemar_exact(lost, gained)
    assert 0 < p <= 1
    assert p == mcnemar_exact(gained, lost)
    if lost == gained:
        assert p == 1.0


@given(st.integers(1, 40))
def test_mcnemar_shrinks_as_one_sided_evidence_grows(n: int) -> None:
    assert mcnemar_exact(0, n + 1) < mcnemar_exact(0, n)
    assert mcnemar_exact(0, n) == pytest.approx(min(1.0, 2 / 2**n))


# ------------------------------------------------------------------ verifier
def test_caught_layer_reproduces_the_golden_caught_by_shares() -> None:
    """Cross-check against the independent oracle: the drill-down's per-item layer, tallied over
    the golden verifier run, gives exactly expected_summaries.json's caught_by shares."""
    vs = VerifierSet.model_validate(json.loads((DEMO / "verifier.json").read_text()))
    items = {i.id: i for i in vs.items}
    run = read_run(FIXTURES / "run-verifier.jsonl")
    planted = [r for r in run.rows if items[r.item_id].mutation is not None]
    layers: dict[str, int] = {}
    for row in planted:
        assert isinstance(row.output, VerifyOutput)
        layer = caught_layer(row.output, items[row.item_id])
        layer = "missed" if layer == "caught, unlocated" else layer
        layers[layer] = layers.get(layer, 0) + 1
    expected = json.loads((FIXTURES / "expected_summaries.json").read_text())
    for layer in ("coverage", "verifier", "both", "missed"):
        share = layers.get(layer, 0) / len(planted)
        assert share == pytest.approx(expected[f"verifier.caught_by.{layer}"]), layer


def test_caught_layer_names_unlocated_and_clean_items() -> None:
    vs = VerifierSet.model_validate(json.loads((DEMO / "verifier.json").read_text()))
    items = {i.id: i for i in vs.items}
    rows = {r.item_id: r for r in read_run(FIXTURES / "run-verifier.jsonl").rows}

    def layer(item: str) -> str:
        out = rows[item].output
        assert isinstance(out, VerifyOutput)
        return caught_layer(out, items[item])

    assert layer("p-entity-2") == "caught, unlocated"
    assert layer("p-number-2") == "missed"  # flagged elsewhere: not a span catch
    assert layer("c-durability") == "clean item"
    assert layer("c-deploy") == "clean item, rejected"
    out = rows["c-durability"].output
    assert isinstance(out, VerifyOutput)
    assert caught_layer(out, None).startswith("unknown item")


def _verifier_summary(recall: tuple[float, float] | None, reject_high: float | None) -> RunSummary:
    lo, hi = recall or (None, None)
    return RunSummary(
        run_id="v",
        eval_kind=EvalKind.verifier,
        header_digest="h",
        metrics=[
            _metric(
                "verifier.badge_flip_recall",
                0.9,
                ci_low=lo,
                ci_high=hi,
                method=Method.cluster_bootstrap,
            ),
            _metric("verifier.false_reject.verifier_caused", 0.1, ci_low=0.0, ci_high=reject_high),
        ],
    )


@pytest.mark.parametrize(
    ("recall", "reject_high", "state"),
    [
        ((0.75, 0.99), 0.20, "pass"),  # both bounds exactly at the line pass
        ((0.7499, 0.99), 0.10, "fail"),
        ((0.9, 0.99), 0.2001, "fail"),
        (None, 0.1, "not evaluable"),
        ((0.9, 0.99), None, "not evaluable"),
    ],
)
def test_slo_verdict_uses_the_bounds(
    recall: tuple[float, float] | None, reject_high: float | None, state: str
) -> None:
    assert slo_verdict(_verifier_summary(recall, reject_high)).state == state


def test_slo_on_a_point_only_summary_is_not_evaluable() -> None:
    assert slo_verdict(summary_of("verifier")).state == "not evaluable"
    assert slo_verdict(None).state == "not evaluable"


# ------------------------------------------------------------------ the blind queue
SHOWN = [ShownRef(label=1, chunk_id=str(UUID(int=1)), display_sha256="a" * 64)]


def test_queue_items_carry_no_judge_data_and_skip_what_cannot_be_labelled() -> None:
    qs = _qs(_q("ok"), _q("held", Split.heldout2), _q("abs"), _q("stale"), _q("nojudge"))
    refs = {d: [f"{d}:{c.value}" for c in E_G_CRITERIA] for d in ("ok", "held", "abs", "stale")}
    rows = [
        answer_row("ok", "A.", refs=refs["ok"], shown=SHOWN),
        answer_row("held", "H.", refs=refs["held"], shown=SHOWN),
        answer_row("abs", "-", abstained=True, refs=refs["abs"], shown=SHOWN),
        answer_row("stale", "S.", refs=refs["stale"], stale=True, shown=SHOWN),
        answer_row("nojudge", "N.", shown=SHOWN),
        answer_row("missing", "M.", refs=["x"], shown=SHOWN),
    ]
    index = judge_index([r for d in refs for r in judged(d)])
    queue = queue_items(_run(rows, splits=(Split.dev, Split.heldout2)), qs, index)
    assert [(i.item_id, i.item_digest) for i in queue.items] == [("ok", "ok")]
    assert queue.skipped == {
        "held out (never labelled)": 1,
        "abstained (the judge scores answered rows only)": 1,
        "stale (its notes changed)": 1,
        "not judged yet": 1,
        "not in the question set (split unknown)": 1,
    }
    assert set(QueueItem.__dataclass_fields__) == {
        "item_id",
        "item_digest",
        "question",
        "answer",
        "shown",
    }


def test_a_row_whose_judge_results_name_two_items_is_not_labelled() -> None:
    row = answer_row("ok", "A.", refs=["a:relevance", "b:relevance"], shown=SHOWN)
    index = judge_index([*judged("a"), *judged("b")])
    queue = queue_items(_run([row]), _qs(_q("ok")), index)
    assert queue.items == []
    assert queue.skipped == {"judge results disagree on the item": 1}


def test_without_a_question_set_nothing_is_labelable() -> None:
    queue = queue_items(
        _run([answer_row("ok", "A.", refs=["ok:relevance"], shown=SHOWN)]),
        None,
        judge_index(judged("ok")),
    )
    assert queue.items == []


@given(st.lists(st.tuples(st.sampled_from(list(Split)), st.booleans(), st.booleans()), max_size=10))
def test_the_queue_never_holds_a_heldout_abstained_or_unjudged_row(
    spec: list[tuple[Split, bool, bool]],
) -> None:
    questions = [_q(f"q{i}", split) for i, (split, _, _) in enumerate(spec)]
    rows, results = [], []
    for i, (_, abstained, is_judged) in enumerate(spec):
        refs = [f"d{i}:{c.value}" for c in E_G_CRITERIA] if is_judged else []
        rows.append(answer_row(f"q{i}", "A.", abstained=abstained, refs=refs, shown=SHOWN))
        results += judged(f"d{i}") if is_judged else []
    queue = queue_items(_run(rows), _qs(*questions), judge_index(results))
    expected = [
        f"q{i}"
        for i, (split, abstained, is_judged) in enumerate(spec)
        if split is Split.dev and not abstained and is_judged
    ]
    assert [i.item_id for i in queue.items] == expected


def test_a_row_that_recorded_no_shown_notes_is_never_queued() -> None:
    row = answer_row("ok", "A.", refs=[f"ok:{c.value}" for c in E_G_CRITERIA])
    queue = queue_items(_run([row]), _qs(_q("ok")), judge_index(judged("ok")))
    assert queue.items == []
    assert queue.skipped == {"with no shown notes recorded (faithfulness needs them)": 1}


def test_only_the_newest_row_of_a_resumed_item_is_queued() -> None:
    old = answer_row("ok", "OLD.", refs=[f"old:{c.value}" for c in E_G_CRITERIA], shown=SHOWN)
    new = answer_row("ok", "NEW.", refs=[f"new:{c.value}" for c in E_G_CRITERIA], shown=SHOWN)
    queue = queue_items(
        _run([old, new]), _qs(_q("ok")), judge_index([*judged("old"), *judged("new")])
    )
    assert [(i.answer, i.item_digest) for i in queue.items] == [("NEW.", "new")]


def test_the_queue_is_the_stratified_calibration_sample() -> None:
    """60 labelable items in two verdict strata (50 verified, 10 partial): a sample of 10 is
    drawn in proportion with both strata present, is the same on every call, and counts the
    rest as outside the sample; a convenience sample (file order) would be all verified."""
    rows, results, questions = [], [], []
    for i in range(60):
        status = "verified" if i < 50 else "partial"
        refs = [f"d{i:02}:{c.value}" for c in E_G_CRITERIA]
        rows.append(answer_row(f"q{i}", "A.", status=status, refs=refs, shown=SHOWN))
        results += judged(f"d{i:02}")
        questions.append(_q(f"q{i}"))
    run, qs, index = _run(rows), _qs(*questions), judge_index(results)
    queue = queue_items(run, qs, index, sample_n=10)
    ids = [i.item_id for i in queue.items]
    assert len(ids) == 10
    assert queue.outside_sample == 50
    partial = [i for i in ids if int(i[1:]) >= 50]
    assert len(partial) == 2  # 10 x 10/60, rounded by largest remainder
    assert ids == [i.item_id for i in queue_items(run, qs, index, sample_n=10).items]
    assert ids != [f"q{i}" for i in range(10)]


def test_labelled_digests_need_every_criterion_of_a_criteria_label() -> None:
    def label(digest: str, kind: LabelKind, scores: dict[Criterion, int]) -> Label:
        return Label(
            item_digest=digest,
            item_id=digest,
            kind=kind,
            scores=scores,
            batch_id="b",
            labelled_at=T0,
        )

    full = {c: 1 for c in E_G_CRITERIA}
    done = labelled_digests(
        [
            label("full", LabelKind.criteria, full),
            label("part", LabelKind.criteria, {Criterion.relevance: 2}),
            label("answer", LabelKind.answer, {}),
        ]
    )
    assert done == {"full"}


def test_criteria_label_is_a_locked_label() -> None:
    item = QueueItem("q1", "d1", "Q?", "A.", ())
    label = criteria_label(item, {c: 2 for c in E_G_CRITERIA}, note="  ok  ", batch_id="b2", at=T0)
    assert (label.item_digest, label.kind, label.note, label.batch_id) == (
        "d1",
        LabelKind.criteria,
        "ok",
        "b2",
    )
    assert Label.model_validate_json(label.model_dump_json()) == label


# ------------------------------------------------------------------ calibration
OTHER = JudgeKey(provider="gemini", model_version="flash-001", prompt="judge_qa@1")


def test_calibration_state_takes_the_newest_record_of_this_judge() -> None:
    records = [
        calibration(Criterion.relevance, trusted=True, at=T0),
        calibration(Criterion.relevance, trusted=False, at=T0 + timedelta(hours=1)),
        calibration(Criterion.completeness, trusted=True, at=T0),
        calibration(Criterion.faithfulness, trusted=True, at=T0, judge=OTHER),
    ]
    state = calibration_state(records, JUDGE)
    assert set(state.current) == {Criterion.relevance, Criterion.completeness}
    assert state.untrusted == [Criterion.relevance]
    assert state.any_trusted
    assert state.other_judges == 1
    assert calibration_state(records[:2], JUDGE).untrusted == [Criterion.relevance]
    assert not calibration_state([], JUDGE).any_trusted


# ------------------------------------------------------------------ calibration metrics
def test_an_untrusted_calibration_metric_still_shows_its_kappa_and_ci() -> None:
    """Kappa is the measure of trust, not a judge result: hiding it would hide the evidence."""
    kappa = _metric(
        "calibration.relevance.kappa_linear", 0.41, ci_low=0.2, ci_high=0.6, trusted=False
    )
    row = metric_row(kappa)
    assert (row["value"], row["95% CI"], row["judge"]) == ("0.41", "[0.2, 0.6]", "not trusted")
    judge = _metric("generator.judge.relevance", 0.9, ci_low=0.8, ci_high=1.0, trusted=False)
    assert (value_text(judge), ci_cell(judge)) == ("not trusted", "-")


def test_summary_trust_reads_the_calibration_flag_first() -> None:
    summary = RunSummary(
        run_id="r",
        eval_kind=EvalKind.generator,
        header_digest="h" * 64,
        metrics=[
            _metric("calibration.relevance.kappa_linear", 0.4, trusted=False),
            _metric("generator.judge.relevance", 0.9, trusted=True),  # a stale copy: ignored
            _metric("generator.judge.completeness", 0.9, trusted=True),
            _metric("generator.answer_rate", 0.6),
        ],
    )
    assert summary_trust(summary) == {"relevance": False, "completeness": True}
    assert summary_trust(None) == {}


def test_badge_flip_needs_its_clean_source_verified_in_the_same_run() -> None:
    """Integrator fix of review P1: the baseline verifies c-1 and misses p-1; the candidate
    rejects everything. That must NOT read as a badge-flip gain; the clean loss is reported."""
    items = {"p-1": _planted(), "c-1": _clean()}
    base = [_vrow("c-1", "verified"), _vrow("p-1", "verified")]
    cand = [_vrow("c-1", "partial"), _vrow("p-1", "partial", flag=_away())]
    result = paired(base, cand, {}, items)
    by = {d.measure: d for d in result.discordance}
    assert "planted: badge flipped" not in by  # c-1 is not verified clean in the candidate
    assert (by["clean: verified clean"].lost, by["clean: verified clean"].gained) == (1, 0)
