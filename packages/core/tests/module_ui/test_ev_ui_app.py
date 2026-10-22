"""EV-UI on the real Streamlit app (AppTest), over COPIES of the golden fixtures and synthetic
runs. No network, no model, never evals/private.

Every view shows the numbers of the committed *.summary.json files; compare refuses runs on
different corpora; held-out rows stay hidden until revealed; the label queue is blind, never
held out and writes only through the LabelWriter; untrusted judge criteria are said so."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from uuid import UUID

import pytest
from streamlit.testing.v1 import AppTest
from test_ev_ui_support import (
    FIXTURES,
    GOLDEN,
    T0,
    FakeLabels,
    RecordingChunks,
    answer_row,
    calibration,
    cells,
    chunk,
    copy_run,
    eval_app,
    everything,
    golden_paths,
    header,
    judged,
    page_text,
    questions_toml,
    read_lines,
    shown_ref,
    summary_of,
    write_jsonl,
    write_lines,
    write_run,
)
from ui_support import APP, FakeBackend

from ragforge_client.evals.datasets import LabelStore, seal_questions
from ragforge_client.evals.schema import (
    E_G_CRITERIA,
    Criterion,
    GateCheck,
    GateOutcome,
    GateVerdict,
    Label,
    LabelKind,
    Method,
    MetricValue,
    ShownRef,
    Split,
    Surface,
)
from ragforge_ui.eval_view import drill, labels, runs
from ragforge_ui.eval_view.config import EvalPaths
from ragforge_ui.eval_view.data import FileLabelReader
from ragforge_ui.eval_view.model import NO_TRUSTED_JUDGE, fmt, k_of_n
from ragforge_ui.eval_view.widgets import REVEALED

FIXTURES_LABELS = FIXTURES / "labels.jsonl"


ORACLE: dict[str, float] = json.loads((FIXTURES / "expected_summaries.json").read_text())
ORACLE_PREFIXES = {
    "generator": ("generator.", "calibration."),
    "interview": ("interview.",),
    "retrieval": ("retrieval.",),
    "verifier": ("verifier.",),
}


def _oracle_for(run: str) -> dict[str, float]:
    """The executable definition's values (reference_metrics.py) of one golden run."""
    return {k: v for k, v in ORACLE.items() if k.startswith(ORACLE_PREFIXES[run])}


@pytest.fixture
def paths(tmp_path: Path) -> EvalPaths:
    return golden_paths(tmp_path)


def _pick(at: AppTest, key: str, path: Path) -> AppTest:
    return at.selectbox(key=key).set_value(str(path)).run()


def _rows(at: AppTest, column: str) -> dict[str, dict[str, str]]:
    return {row[column]: row for row in cells(at) if column in row}


# ------------------------------------------------------------------ navigation
def test_eval_is_a_page_of_the_sidebar_and_chat_stays_the_default(paths: EvalPaths) -> None:
    at = AppTest.from_file(str(APP), default_timeout=30)
    at.session_state["rf_backend"] = FakeBackend()
    at = at.run()
    assert at.sidebar.radio[0].options == ["Chat", "Sources", "Eval"]
    assert at.sidebar.radio[0].value == "Chat"
    assert not at.exception


def test_an_empty_runs_directory_says_so(tmp_path: Path) -> None:
    (tmp_path / "runs").mkdir()
    at = eval_app(EvalPaths.under(tmp_path / "runs", tmp_path), None)
    assert not at.exception
    assert "No eval runs in" in page_text(at)


# ------------------------------------------------------------------ runs view
@pytest.mark.parametrize("name", GOLDEN)
def test_runs_view_shows_every_number_of_each_golden_summary(paths: EvalPaths, name: str) -> None:
    at = _pick(eval_app(paths), runs.RUN, paths.runs / f"run-{name}.jsonl")
    assert not at.exception
    shown = _rows(at, "metric")
    summary = summary_of(name)
    assert summary.metrics
    for metric in summary.metrics:
        row = shown[metric.name]
        assert row["value"] == fmt(metric.value), metric.name
        assert row["n"] == str(metric.n)
        assert row["method"] == metric.method.value
    oracle = _oracle_for(name)
    assert oracle
    for metric_name, value in oracle.items():  # the LOCKED oracle, not only the summary file
        assert shown[metric_name]["value"] == fmt(value), metric_name
    page = page_text(at)
    assert "Snapshot content:** demo (no snapshot)" in page
    assert "chat: answer@11, verifier: ground_check@3" in page
    assert "fake-chat" in page


def test_runs_view_tiles_show_the_headline_values(paths: EvalPaths) -> None:
    at = _pick(eval_app(paths), runs.RUN, paths.runs / "run-retrieval.jsonl")
    tiles = {m.label.replace("\\", ""): m.value for m in at.metric}
    expected = {m.name: m for m in summary_of("retrieval").metrics}
    for name in ("retrieval.reranked.hit@5", "retrieval.reranked.mrr", "retrieval.fused.recall@20"):
        assert tiles[name] == fmt(expected[name].value)


UNTRUSTED_CI = {"ci_low": 0.123, "ci_high": 0.987, "value": 0.456}


def _untrust_relevance(paths: EvalPaths) -> None:
    path = paths.runs / "run-generator.summary.json"
    summary = summary_of("generator")
    metrics = [
        m.model_copy(update={"trusted": False, **UNTRUSTED_CI})
        if m.name == "generator.judge.relevance"
        else m
        for m in summary.metrics
    ]
    path.write_text(summary.model_copy(update={"metrics": metrics}).model_dump_json())


def test_an_untrusted_judge_metric_is_never_shown_as_a_number(paths: EvalPaths) -> None:
    _untrust_relevance(paths)
    at = _pick(eval_app(paths), runs.RUN, paths.runs / "run-generator.jsonl")
    row = _rows(at, "metric")["generator.judge.relevance"]
    assert row["value"] == "not trusted"
    assert row["95% CI"] == "-"
    assert _rows(at, "metric")["generator.judge.faithfulness"]["value"] == "1"
    tile = next(m for m in at.metric if m.label.replace("\\", "") == "generator.judge.relevance")
    assert tile.value == "not trusted"
    text = everything(at)
    for number in ("0.123", "0.987", "0.456"):  # neither in a tile, its caption nor a table
        assert number not in text, number


def test_an_untrusted_judge_metric_shows_no_number_on_any_view(paths: EvalPaths) -> None:
    _untrust_relevance(paths)
    copy_run(paths, "generator", "cand")
    for view in ("Runs", "Compare", "Calibration", "Drill-down"):
        text = everything(eval_app(paths, view))
        for number in ("0.123", "0.987", "0.456"):
            assert number not in text, (view, number)


def test_runs_view_shows_intervals_when_the_summary_has_them(paths: EvalPaths) -> None:
    path = paths.runs / "run-verifier.summary.json"
    summary = summary_of("verifier")
    metrics = [
        m.model_copy(update={"ci_low": 0.61, "ci_high": 0.97, "method": Method.cluster_bootstrap})
        if m.name == "verifier.badge_flip_recall"
        else m
        for m in summary.metrics
    ]
    path.write_text(summary.model_copy(update={"metrics": metrics}).model_dump_json())
    at = _pick(eval_app(paths), runs.RUN, paths.runs / "run-verifier.jsonl")
    row = _rows(at, "metric")["verifier.badge_flip_recall"]
    assert row["95% CI"] == "[0.61, 0.97]"
    assert row["method"] == "cluster_bootstrap"
    assert "95% CI [0.61, 0.97] · n 20 · cluster_bootstrap" in page_text(at)


def test_runs_view_shows_the_gate_state_and_any_override(paths: EvalPaths) -> None:
    copy_run(paths, "generator", "cand", header={"override": "accepted: new chunker"})
    verdict = GateVerdict(
        surface=Surface.chat,
        baseline_run="golden-generator",
        candidate_run="cand",
        checks=[
            GateCheck(
                name="correct%", outcome=GateOutcome.block, method="exact McNemar", n=20, mde=0.25
            )
        ],
        outcome=GateOutcome.block,
        override="owner: shipped anyway",
    )
    (paths.runs / "gate-chat.json").write_text(verdict.model_dump_json())
    at = _pick(eval_app(paths), runs.RUN, paths.runs / "run-cand.jsonl")
    page = page_text(at)
    assert "BLOCK" in page
    assert "Gate overridden:** owner: shipped anyway" in page
    assert "Gate override recorded:** accepted: new chunker" in page
    check = _rows(at, "check")["correct%"]
    assert (check["outcome"], check["n"], check["MDE"]) == ("block", "20", "0.25")


def test_a_broken_summary_is_reported_by_type_without_its_contents(paths: EvalPaths) -> None:
    (paths.runs / "run-retrieval.summary.json").write_text('{"secret-answer-text": 1}')
    at = _pick(eval_app(paths), runs.RUN, paths.runs / "run-retrieval.jsonl")
    page = page_text(at)
    assert "run-retrieval.summary.json: unreadable (ValidationError)" in page
    assert "secret-answer-text" not in page


# ------------------------------------------------------------------ verifier and interview
def test_verifier_view_shows_recall_by_mutation_and_layer_and_the_false_reject_split(
    paths: EvalPaths,
) -> None:
    at = eval_app(paths, "Verifier")
    assert not at.exception
    expected = {m.name: m for m in summary_of("verifier").metrics}
    by_mutation = _rows(at, "mutation")
    assert len(by_mutation) == 7
    for mutation, row in by_mutation.items():
        assert row["value"] == fmt(expected[f"verifier.span_recall.{mutation}"].value)
    assert by_mutation["fabricated_sentence"]["expected layer"] == "coverage"
    for layer, row in _rows(at, "layer").items():
        assert row["value"] == fmt(expected[f"verifier.caught_by.{layer}"].value)
    causes = _rows(at, "cause")
    assert causes["all"]["value"] == fmt(expected["verifier.false_reject"].value)
    assert causes["verifier-caused"]["value"] == "0.125"
    assert causes["coverage-caused"]["value"] == "0.125"
    tiles = {m.label.replace("\\", ""): m.value for m in at.metric}
    assert tiles["verifier.badge_flip_recall"] == "0.9"
    assert tiles["verifier.caught_unlocated"] == "0.1"
    page = page_text(at)
    assert "SLO not evaluable" in page  # the golden summaries carry point values only
    assert "badge-flip recall lower bound ≥ 0.75" in page


@pytest.mark.parametrize(
    ("recall_low", "reject_high", "state"),
    [(0.80, 0.15, "SLO pass"), (0.70, 0.15, "SLO fail"), (0.80, 0.25, "SLO fail")],
)
def test_the_verifier_slo_is_judged_on_the_interval_bounds(
    paths: EvalPaths, recall_low: float, reject_high: float, state: str
) -> None:
    summary = summary_of("verifier")
    bounds = {
        "verifier.badge_flip_recall": (recall_low, 0.99),
        "verifier.false_reject.verifier_caused": (0.0, reject_high),
    }
    metrics = [
        m.model_copy(update={"ci_low": bounds[m.name][0], "ci_high": bounds[m.name][1]})
        if m.name in bounds
        else m
        for m in summary.metrics
    ]
    path = paths.runs / "run-verifier.summary.json"
    path.write_text(summary.model_copy(update={"metrics": metrics}).model_dump_json())
    page = page_text(eval_app(paths, "Verifier"))
    assert state in page
    assert ("verifier_trusted=false" in page) == (state == "SLO fail")


def test_interview_view_shows_ranking_detection_and_the_level_trend(paths: EvalPaths) -> None:
    at = eval_app(paths, "Interview")
    assert not at.exception
    expected = {m.name: m for m in summary_of("interview").metrics}
    ranking = expected["interview.ranking_accuracy"]
    assert fmt(ranking.value) == fmt(ORACLE["interview.ranking_accuracy"])
    tile = next(m for m in at.metric if m.label.startswith("Ranking accuracy"))
    assert tile.value == k_of_n(ranking)  # k/n or value (n): whatever n the summary has
    stages = _rows(at, "stage")
    assert stages["given (before the server's filters)"]["value"] == "1"
    assert stages["kept (after the quote rule and the verifier)"]["value"] == "0.75"
    stats = _rows(at, "statistic")
    assert stats["page_l"]["value"] == fmt(expected["interview.level.page_l"].value) == "397.0"
    assert stats["retest_sd"]["value"] == fmt(expected["interview.level.retest_sd"].value)
    bands = _rows(at, "band")
    assert bands["gold_ge_7"]["value"] == bands["wrong_le_3"]["value"] == "1"
    trend = _rows(at, "tier")
    assert set(trend) == {"gold", "partial"}
    assert all(trend["gold"][lv] != "-" for lv in ("sde_1", "staff"))


def test_ranking_accuracy_is_shown_as_k_of_n_when_it_is_a_count(paths: EvalPaths) -> None:
    summary = summary_of("interview")
    metrics = [
        m.model_copy(update={"n": 16, "ci_low": 0.70, "ci_high": 0.998, "method": Method.exact})
        if m.name == "interview.ranking_accuracy"
        else m
        for m in summary.metrics
    ]
    path = paths.runs / "run-interview.summary.json"
    path.write_text(summary.model_copy(update={"metrics": metrics}).model_dump_json())
    page = page_text(eval_app(paths, "Interview"))
    assert "15/16" in page
    assert "95% CI [0.7, 0.998] · exact" in page


# ------------------------------------------------------------------ compare
def test_compare_refuses_runs_on_different_snapshot_content(paths: EvalPaths) -> None:
    snap = {"collection_id": "c", "index_digest": "i" * 64}
    copy_run(paths, "generator", "a", header={"snapshot": {**snap, "content_digest": "a" * 64}})
    copy_run(paths, "generator", "b", header={"snapshot": {**snap, "content_digest": "b" * 64}})
    at = eval_app(paths, "Compare")
    at = _pick(at, runs.CAND, paths.runs / "run-b.jsonl")
    at = _pick(at, runs.BASE, paths.runs / "run-a.jsonl")
    page = page_text(at)
    assert "These runs can't be compared." in page
    assert "different snapshot content" in page
    assert "aaaaaaaaaaaa vs bbbbbbbbbbbb" in page
    assert not [r for r in cells(at) if "metric" in r]  # no deltas drawn


def test_compare_refuses_a_snapshot_run_against_a_demo_run(paths: EvalPaths) -> None:
    snap = {"collection_id": "c", "index_digest": "i" * 64, "content_digest": "a" * 64}
    copy_run(paths, "generator", "snap", header={"snapshot": snap})
    at = _pick(eval_app(paths, "Compare"), runs.CAND, paths.runs / "run-snap.jsonl")
    at = _pick(at, runs.BASE, paths.runs / "run-generator.jsonl")
    assert "different corpora" in page_text(at)


def test_compare_offers_only_runs_of_the_same_kind_as_baseline(paths: EvalPaths) -> None:
    at = _pick(eval_app(paths, "Compare"), runs.CAND, paths.runs / "run-verifier.jsonl")
    assert "No other verifier run to compare it with." in page_text(at)


def test_compare_shows_exploratory_deltas_and_the_items_that_flipped(paths: EvalPaths) -> None:
    path = copy_run(paths, "generator", "cand")
    lines = read_lines(path)
    flipped = next(x for x in lines[1:-1] if x["output"]["status"] == "verified")
    flipped["output"]["status"] = "partial"
    write_lines(path, lines)
    summary = json.loads((paths.runs / "run-cand.summary.json").read_text())
    for m in summary["metrics"]:
        if m["name"] == "generator.answer_rate":
            m["value"] = 0.5
    (paths.runs / "run-cand.summary.json").write_text(json.dumps(summary))
    at = _pick(eval_app(paths, "Compare"), runs.CAND, path)
    at = _pick(at, runs.BASE, paths.runs / "run-generator.jsonl")
    assert not at.exception
    assert "exploratory" in page_text(at)
    delta = _rows(at, "metric")["generator.answer_rate"]
    assert (delta["baseline"], delta["candidate"], delta["delta"]) == ("0.6", "0.5", "-0.1")
    flips = [r for r in cells(at) if r.get("item") == flipped["item_id"]]
    assert flips == [
        {
            "item": flipped["item_id"],
            "measure": "verified",
            "baseline": "pass",
            "candidate": "fail",
        }
    ]
    verified = _rows(at, "paired outcome")["verified"]
    assert (verified["pass → fail"], verified["fail → pass"]) == ("1", "0")
    assert verified["exact McNemar p (exploratory)"] == "1"


# ------------------------------------------------------------------ held-out rows
def _heldout_set(paths: EvalPaths) -> tuple[Path, Path]:
    questions_toml(
        paths.questions,
        [
            ("q-dev", Split.dev, "What does the dev question ask?"),
            ("q-held", Split.heldout2, "SEALED held-out question?"),
        ],
    )
    rows = [
        answer_row("q-dev", "Dev answer [S1].", refs=_refs("dev")),
        answer_row("q-held", "SEALED held-out answer.", refs=_refs("held")),
    ]
    head = header("held-a", splits=(Split.dev, Split.heldout2))
    a = write_run(paths.runs / "run-held-a.jsonl", head, rows)
    b = write_run(
        paths.runs / "run-held-b.jsonl",
        header("held-b", splits=(Split.dev, Split.heldout2), started=T0.replace(hour=13)),
        rows,
    )
    write_jsonl(paths.judge_results, [*judged("dev"), *judged("held")])
    return a, b


def test_drill_down_hides_held_out_rows_until_the_run_id_is_typed(paths: EvalPaths) -> None:
    a, _ = _heldout_set(paths)
    at = _pick(eval_app(paths, "Drill-down"), drill.RUN, a)
    items = at.selectbox(key=f"{drill.ITEM}:{a}").options
    assert items == ["q-dev"]
    assert "SEALED" not in everything(at)
    assert "1 held-out (heldout2) row(s) hidden." in page_text(at)
    at = at.text_input(key="rf_eval_reveal_drill").input("held-b").run()  # wrong id: no reveal
    assert at.selectbox(key=f"{drill.ITEM}:{a}").options == ["q-dev"]
    at = at.text_input(key="rf_eval_reveal_drill").input("held-a").run()
    assert at.selectbox(key=f"{drill.ITEM}:{a}").options == ["q-dev", "q-held"]
    assert "were revealed" in page_text(at)
    # the reveal is RECORDED next to the run (ADR-0017 D1), and honoured in a fresh session
    from ragforge_client.evals.interfaces import reveal_path, revealed_at

    assert reveal_path(a).exists()
    assert revealed_at(a) is not None
    fresh = _pick(eval_app(paths, "Drill-down"), drill.RUN, a)
    assert fresh.selectbox(key=f"{drill.ITEM}:{a}").options == ["q-dev", "q-held"]


def test_compare_pairs_only_visible_rows_and_hides_held_out_flips(paths: EvalPaths) -> None:
    a, b = _heldout_set(paths)
    lines = read_lines(b)
    for x in lines[1:-1]:
        x["output"]["status"] = "partial"  # both items flip
    write_lines(b, lines)
    at = _pick(eval_app(paths, "Compare"), runs.CAND, b)
    at = _pick(at, runs.BASE, a)
    flipped = {r["item"] for r in cells(at) if "item" in r}
    assert flipped == {"q-dev"}
    assert "SEALED" not in everything(at)
    assert "2 held-out (heldout2) row(s) are left out of the flips" in page_text(at)
    assert not at.text_input  # Compare offers no reveal (ADR-0017 D1: one reveal, not counted)
    at.session_state[REVEALED] = {str(a), str(b)}  # even after a reveal in the Drill-down
    assert "SEALED" not in everything(at.run())


def test_a_run_declaring_heldout2_hides_rows_of_unknown_split(paths: EvalPaths) -> None:
    """Fail closed: no question set, so no row can be proven dev."""
    a, _ = _heldout_set(paths)
    paths.questions.unlink()
    at = _pick(eval_app(paths, "Drill-down"), drill.RUN, a)
    assert "No rows to show." in page_text(at)
    assert "2 held-out (heldout2) row(s) hidden." in page_text(at)


# ------------------------------------------------------------------ the blind label queue
VERDICT_WORDS = ("Verified", "Partly verified", "Not supported", "Not checked", "verified")


def _queue_set(paths: EvalPaths, reason: str = "JUDGE-REASON-SENTINEL") -> Path:
    questions_toml(
        paths.questions,
        [
            ("q1", Split.dev, "First dev question?"),
            ("q2", Split.dev, "Second dev question?"),
            ("q-held", Split.heldout2, "SEALED held-out question?"),
            ("q-abs", Split.dev, "Abstained question?"),
        ],
    )
    rows = [
        answer_row("q1", "First answer.", status="partial", refs=_refs("d1"), shown=_shown("q1")),
        answer_row("q2", "Second answer.", status="verified", refs=_refs("d2"), shown=_shown("q2")),
        answer_row("q-held", "SEALED held-out answer.", refs=_refs("dh"), shown=_shown("q-held")),
        answer_row("q-abs", "Abstained.", abstained=True, refs=_refs("da"), shown=_shown("q-abs")),
    ]
    path = write_run(
        paths.runs / "run-queue.jsonl",
        header("queue", splits=(Split.dev, Split.heldout2), started=T0.replace(hour=23)),
        rows,
    )
    results = [*judged("d1", reason=reason), *judged("d2", reason=reason)]
    results += [*judged("dh", reason=reason), *judged("da", reason=reason)]
    write_jsonl(paths.judge_results, results)
    return path


def _shown(item_id: str) -> list[ShownRef]:
    """One shown note per row (the queue offers only rows that recorded their notes), with the
    text `_chunks_for` serves for that item, so the digest check passes."""
    return [shown_ref(1, chunk("NOTE-ONE-TEXT" if item_id == "q1" else "NOTE-TWO-TEXT"))]


def _refs(digest: str) -> list[str]:
    return [f"{digest}:{c.value}" for c in E_G_CRITERIA]


def test_the_label_queue_shows_no_judge_score_reason_or_verdict(paths: EvalPaths) -> None:
    path = _queue_set(paths)
    at = _pick(eval_app(paths, "Label queue", chunks=_chunks_for(path)), labels.RUN, path)
    assert not at.exception
    text = everything(at)
    assert "First dev question?" in text
    assert "First answer." in text
    assert "JUDGE-REASON-SENTINEL" not in text
    assert not cells(at)  # no table at all: nothing a score could hide in
    assert not at.metric
    assert not at.get("badge")
    for word in VERDICT_WORDS:
        assert word not in text.replace("verified clean", ""), word


def test_the_label_queue_never_offers_held_out_items_even_after_a_reveal(
    paths: EvalPaths,
) -> None:
    path = _queue_set(paths)
    store = FakeLabels()
    at = eval_app(paths, "Label queue", labels=store, chunks=_chunks_for(path))
    at.session_state[REVEALED] = {str(path)}  # revealed elsewhere this session
    at = _pick(at.run(), labels.RUN, path)
    for _ in range(2):
        assert "SEALED" not in everything(at)
        at = _label(at, 2, 1, 0)
    assert [lb.item_id for lb in store.appended] == ["q1", "q2"]
    assert "Nothing left to label in this run." in page_text(at)
    assert "SEALED" not in everything(at)
    assert "1 held out (never labelled)" in page_text(at)


def _label(at: AppTest, *scores: int, note: str = "") -> AppTest:
    radios = [r for r in at.radio if r.key and r.key.startswith("rf_eval_score:")]
    assert len(radios) == 3
    for radio, score in zip(radios, scores, strict=True):
        radio.set_value(score)
    if note:
        next(t for t in at.text_input if t.key and t.key.startswith("rf_eval_note:")).input(note)
    return at.button[-1].click().run()


def test_a_label_is_written_through_the_writer_with_the_judged_item_digest(
    paths: EvalPaths,
) -> None:
    path = _queue_set(paths)
    store = FakeLabels()
    at = _pick(
        eval_app(paths, "Label queue", labels=store, chunks=_chunks_for(path)), labels.RUN, path
    )
    at = at.text_input(key=labels.BATCH).input("b7").run()
    at = _label(at, 2, 1, 0, note="misses the retry")
    assert not at.exception
    (label,) = store.appended
    assert label.item_digest == "d1"  # pairs with its judge results
    assert label.item_id == "q1"
    assert label.kind is LabelKind.criteria
    assert label.scores == {
        Criterion.relevance: 2,
        Criterion.completeness: 1,
        Criterion.faithfulness: 0,
    }
    assert (label.batch_id, label.note) == ("b7", "misses the retry")
    page = page_text(at)
    assert "Saved the label for q1." in page
    assert "Second dev question?" in page  # the queue moved on
    assert "1 of 2 labelled" in page


def test_a_partial_label_is_refused_and_nothing_is_written(paths: EvalPaths) -> None:
    path = _queue_set(paths)
    store = FakeLabels()
    at = _pick(
        eval_app(paths, "Label queue", labels=store, chunks=_chunks_for(path)), labels.RUN, path
    )
    radios = [r for r in at.radio if r.key and r.key.startswith("rf_eval_score:")]
    radios[0].set_value(2)
    at = at.button[-1].click().run()
    assert store.appended == []
    assert "Score every criterion" in page_text(at)


def test_a_failed_write_says_so_without_details(paths: EvalPaths) -> None:
    path = _queue_set(paths)
    store = FakeLabels(fail=PermissionError("/secret/path/labels.jsonl"))
    at = _pick(
        eval_app(paths, "Label queue", labels=store, chunks=_chunks_for(path)), labels.RUN, path
    )
    at = _label(at, 1, 1, 1)
    page = page_text(at)
    assert "The label was not saved** (PermissionError)" in page
    assert "/secret/path" not in page


def _break_labels(paths: EvalPaths) -> LabelStore:
    """The real store over a labels.jsonl with one good line and one it refuses."""
    good = Label(
        item_digest="d1",
        item_id="q1",
        kind=LabelKind.criteria,
        scores=dict.fromkeys(E_G_CRITERIA, 2),
        note="GOOD-LABEL-NOTE",
        batch_id="b1",
        labelled_at=T0,
    )
    bad = {"schema_version": 2, "item_digest": "x", "kind": "criteria"}
    paths.labels.write_text(good.model_dump_json() + "\n" + json.dumps(bad) + "\n")
    return LabelStore(paths.labels)


def test_a_label_file_the_store_refuses_turns_labelling_off_without_crashing(
    paths: EvalPaths,
) -> None:
    path = _queue_set(paths)
    store = _break_labels(paths)
    before = paths.labels.read_bytes()
    at = _pick(
        eval_app(paths, "Label queue", labels=store, chunks=_chunks_for(path)), labels.RUN, path
    )
    assert not at.exception
    page = page_text(at)
    assert "labels.jsonl: unreadable (DatasetError)" in page
    assert "labelling is off" in page
    assert "1 of 2 labelled" in page  # the good line still counts
    assert not [r for r in at.radio if r.key and r.key.startswith("rf_eval_score:")]
    assert paths.labels.read_bytes() == before


def test_a_label_file_the_store_refuses_still_shows_its_good_labels_in_the_drill_down(
    paths: EvalPaths,
) -> None:
    path = _queue_set(paths)
    at = _pick(eval_app(paths, "Drill-down", labels=_break_labels(paths)), drill.RUN, path)
    assert not at.exception
    page = page_text(at)
    assert "labels.jsonl: unreadable (DatasetError)" in page
    assert "GOOD-LABEL-NOTE" in page


def test_the_real_label_store_writes_one_line_per_label(paths: EvalPaths) -> None:
    path = _queue_set(paths)
    store = LabelStore(paths.labels)
    at = _pick(
        eval_app(paths, "Label queue", labels=store, chunks=_chunks_for(path)), labels.RUN, path
    )
    at = _label(at, 2, 2, 1)
    assert not at.exception
    golden = FileLabelReader(FIXTURES_LABELS).latest()
    new = [lb for lb in store.latest() if lb.item_id == "q1"]
    assert [lb.scores[Criterion.faithfulness] for lb in new] == [1]
    assert len(store.all()) == len(golden) + 1


# ------------------------------------------------------------------ the queue's shown notes
def _queue_with_notes(paths: EvalPaths, *, first_changed: bool = False) -> tuple[Path, list[str]]:
    questions_toml(
        paths.questions,
        [("q1", Split.dev, "First dev question?"), ("q2", Split.dev, "Second dev question?")],
    )
    one, two = chunk("NOTE-ONE-TEXT"), chunk("NOTE-TWO-TEXT")
    stale_digest = "0" * 64 if first_changed else None
    rows = [
        answer_row(
            "q1", "First answer.", refs=_refs("d1"), shown=[shown_ref(1, one, digest=stale_digest)]
        ),
        answer_row("q2", "Second answer.", refs=_refs("d2"), shown=[shown_ref(1, two)]),
    ]
    path = write_run(paths.runs / "run-qn.jsonl", header("qn", started=T0.replace(hour=23)), rows)
    write_jsonl(paths.judge_results, [*judged("d1"), *judged("d2")])
    return path, [one.display_text, two.display_text]


def test_the_queue_shows_the_shown_notes_without_a_toggle_and_never_stores_them(
    paths: EvalPaths,
) -> None:
    path, texts = _queue_with_notes(paths)
    source = _chunks_for(path)
    at = _pick(eval_app(paths, "Label queue", chunks=source), labels.RUN, path)
    assert not at.exception
    assert not at.toggle
    assert texts[0] in page_text(at)
    assert [r for r in at.radio if r.key and r.key.startswith("rf_eval_score:")]
    assert texts[0] not in repr(list(at.session_state.values()))


def test_an_item_whose_notes_changed_is_skipped_with_its_reason(paths: EvalPaths) -> None:
    path, texts = _queue_with_notes(paths, first_changed=True)
    at = _pick(eval_app(paths, "Label queue", chunks=_chunks_for(path)), labels.RUN, path)
    page = page_text(at)
    assert "Skipped: 1 whose shown notes changed or are gone since the run." in page
    assert "First dev question?" not in page
    assert "Second dev question?" in page
    assert texts[1] in page


def test_without_an_eval_key_the_queue_offers_nothing_to_label(
    paths: EvalPaths, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("RAGFORGE_EVAL_KEY_FILE", raising=False)
    path, _ = _queue_with_notes(paths)
    store = FakeLabels()
    at = _pick(eval_app(paths, "Label queue", labels=store), labels.RUN, path)
    assert not at.exception
    assert "No usable eval key." in page_text(at)
    assert not [r for r in at.radio if r.key and r.key.startswith("rf_eval_score:")]
    assert "First answer." not in page_text(at)


def _chunks_for(run: Path) -> RecordingChunks:
    """The chunks a run's rows refer to, as the server would serve them now."""
    held = []
    for line in read_lines(run)[1:-1]:
        for ref in line["shown"]:
            text = "NOTE-ONE-TEXT" if line["item_id"] == "q1" else "NOTE-TWO-TEXT"
            held.append(chunk(text, chunk_id=UUID(ref["chunk_id"])))
    return RecordingChunks(held)


# ------------------------------------------------------------------ calibration
def test_calibration_panel_shows_kappa_ci_confusion_and_untrusted_criteria(
    paths: EvalPaths,
) -> None:
    write_jsonl(
        paths.calibrations,
        [
            calibration(Criterion.relevance, trusted=True, kappa=0.75),
            calibration(Criterion.completeness, trusted=False, kappa=0.45),
            calibration(Criterion.faithfulness, trusted=True, kappa=0.8),
        ],
    )
    at = eval_app(paths, "Calibration")
    assert not at.exception
    page = page_text(at)
    assert "Judge not trusted for completeness: its scores don't count." in page
    assert "Judge not trusted for relevance" not in page
    assert NO_TRUSTED_JUDGE not in page
    table = _rows(at, "criterion")
    assert table["completeness"]["trusted"] == "not trusted"
    assert table["relevance"]["kappa (linear)"] == "0.75"
    assert table["relevance"]["95% CI"] == "[0.55, 0.85]"
    assert table["relevance"]["leniency p"] == "0.5"
    owner = [r for r in cells(at) if r.get("owner") == "owner 1"]
    assert owner[0] == {"owner": "owner 1", "judge 0": "1", "judge 1": "15", "judge 2": "2"}


def test_no_trusted_judge_is_a_stated_outcome(paths: EvalPaths) -> None:
    write_jsonl(
        paths.calibrations,
        [calibration(c, trusted=False, kappa=0.3) for c in E_G_CRITERIA],
    )
    page = page_text(eval_app(paths, "Calibration"))
    assert NO_TRUSTED_JUDGE in page
    for c in ("relevance", "completeness", "faithfulness"):
        assert f"Judge not trusted for {c}" in page


def test_a_newer_calibration_of_the_same_criterion_wins(paths: EvalPaths) -> None:
    write_jsonl(
        paths.calibrations,
        [
            calibration(Criterion.relevance, trusted=False, at=T0.replace(hour=13)),
            calibration(Criterion.relevance, trusted=True, at=T0),
        ],
    )
    page = page_text(eval_app(paths, "Calibration"))
    assert "Judge not trusted for relevance" in page


def test_without_calibration_records_the_summary_numbers_are_shown(paths: EvalPaths) -> None:
    at = eval_app(paths, "Calibration")
    shown = _rows(at, "metric")
    for metric in summary_of("generator").metrics:
        if metric.name.startswith("calibration."):
            assert shown[metric.name]["value"] == fmt(metric.value)
    assert "No calibration records yet" in page_text(at)


# ------------------------------------------------------------------ drill-down
def test_drill_down_shows_answer_verdict_judge_reasons_and_owner_label(paths: EvalPaths) -> None:
    store = FakeLabels(initial=FileLabelReader(paths.labels).latest())  # the golden labels
    at = eval_app(paths, "Drill-down", labels=store)
    at = _pick(at, drill.RUN, paths.runs / "run-generator.jsonl")
    assert not at.exception
    page = page_text(at)
    assert "How does Acme keep all events of one shipment in order in Kafka?" in page
    assert "Answer to kafka-ordering [S1]." in page
    assert "Partly verified" in page  # the verdict badge (not blind here)
    judge = _rows(at, "criterion")
    assert judge["relevance"]["score"] == "0"
    assert judge["relevance"]["reason"] == "fixture"
    assert judge["relevance"]["judge"] == "trusted"  # the summary's flag (no records)
    assert "criteria: completeness 1, faithfulness 2, relevance 1 · batch b1" in page


def test_drill_down_verifier_rows_show_the_catching_layer(paths: EvalPaths) -> None:
    run = paths.runs / "run-verifier.jsonl"
    at = _pick(eval_app(paths, "Drill-down"), drill.RUN, run)
    key = f"{drill.ITEM}:{run}"
    expected = {
        "p-moved-1": "both",
        "p-entity-2": "caught, unlocated",
        "p-negation-1": "missed",
        "p-number-2": "missed",  # flagged elsewhere: not a span catch
        "c-breaker": "clean item, rejected",
    }
    for item, layer in expected.items():
        options = at.selectbox(key=key).options
        at = at.selectbox(key=key).set_value(options.index(item)).run()
        assert f"caught by:** {layer}" in page_text(at), item


def test_shown_notes_are_fetched_only_on_demand_digest_checked_and_never_stored(
    paths: EvalPaths,
) -> None:
    good = chunk("Consumers in one group split the partitions.")
    changed = chunk("The text now differs.")
    gone = chunk("Never served.")
    source = RecordingChunks([good, changed])
    rows = [
        answer_row(
            "kafka-ordering",
            "Answer [S1] [S2] [S3].",
            shown=[
                shown_ref(1, good),
                shown_ref(2, changed, digest="0" * 64),  # the run saw other text
                shown_ref(3, gone),
            ],
        )
    ]
    run = write_run(
        paths.runs / "run-notes.jsonl", header("notes", started=T0.replace(hour=23)), rows
    )
    at = _pick(eval_app(paths, "Drill-down", chunks=source), drill.RUN, run)
    assert source.calls == []  # nothing fetched until asked
    assert good.display_text not in everything(at)
    at = at.toggle[0].set_value(True).run()
    assert not at.exception
    assert len(source.calls) == 1
    page = page_text(at)
    assert good.display_text in page
    assert changed.display_text not in page
    assert "changed since the run (digest mismatch)" in page
    assert "not visible to the eval key" in page
    stored = repr(list(at.session_state.values()))
    assert good.display_text not in stored


def test_without_an_eval_key_notes_say_how_to_configure_it(
    paths: EvalPaths, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("RAGFORGE_EVAL_KEY_FILE", raising=False)
    c = chunk("text")
    run = write_run(
        paths.runs / "run-nokey.jsonl",
        header("nokey", started=T0.replace(hour=23)),
        [answer_row("kafka-ordering", "A [S1].", shown=[shown_ref(1, c)])],
    )
    at = _pick(eval_app(paths, "Drill-down"), drill.RUN, run)
    at = at.toggle[0].set_value(True).run()
    assert not at.exception
    assert "No usable eval key." in page_text(at)
    assert "--eval-key-file" in page_text(at)


def test_a_failing_note_fetch_is_a_friendly_message(paths: EvalPaths) -> None:
    class Down(RecordingChunks):
        def chunks(self, chunk_ids: object) -> list[object]:  # type: ignore[override]
            raise RuntimeError("token=sk-should-not-show")

    c = chunk("text")
    run = write_run(
        paths.runs / "run-down.jsonl",
        header("down", started=T0.replace(hour=23)),
        [answer_row("kafka-ordering", "A [S1].", shown=[shown_ref(1, c)])],
    )
    at = _pick(eval_app(paths, "Drill-down", chunks=Down()), drill.RUN, run)
    at = at.toggle[0].set_value(True).run()
    page = page_text(at)
    assert "Unexpected RuntimeError" in page
    assert "sk-should-not-show" not in page


def test_answer_html_is_never_rendered_as_html(paths: EvalPaths) -> None:
    run = write_run(
        paths.runs / "run-html.jsonl",
        header("html", started=T0.replace(hour=23)),
        [answer_row("kafka-ordering", "<script>alert(1)</script> **bold**")],
    )
    at = _pick(eval_app(paths, "Drill-down"), drill.RUN, run)
    assert not at.exception
    assert all(not md.proto.allow_html for md in at.markdown)


def test_every_view_renders_on_the_golden_fixtures_without_error(paths: EvalPaths) -> None:
    for view in (
        "Runs",
        "Compare",
        "Drill-down",
        "Label queue",
        "Calibration",
        "Verifier",
        "Interview",
    ):
        at = eval_app(paths, view)
        assert not at.exception, view
        assert all(not md.proto.allow_html for md in at.markdown), view


# ------------------------------------------------------------------ round 3: private layout
def _sealed_layout(paths: EvalPaths) -> Path:
    """The private layout: a SEALED questions.toml (dev + heldout2) and a separate sealed
    heldout2.toml, both written the official way (`seal_questions`)."""
    questions_toml(
        paths.questions,
        [
            ("q1", Split.dev, "First dev question?"),
            ("q2", Split.dev, "Second dev question?"),
            ("q-held", Split.heldout2, "SEALED held-out question?"),
        ],
    )
    seal_questions(paths.questions)
    questions_toml(paths.heldout2, [("q-held2", Split.heldout2, "SEALED second held-out?")])
    seal_questions(paths.heldout2)
    rows = [
        answer_row(q, f"Answer {q}.", refs=_refs(f"d-{q}"), shown=_shown(q))
        for q in ("q1", "q2", "q-held", "q-held2")
    ]
    rows.append(answer_row("q-bare", "No notes.", refs=_refs("d-bare")))
    questions = paths.questions.read_text()
    paths.questions.write_text(
        questions.replace(
            '[[questions]]\nid = "q1"',
            '[[questions]]\nid = "q-bare"\nsplit = "dev"\nquestion = "Bare?"\n'
            'relevance = { "file:a.md" = 2 }\n\n[[questions]]\nid = "q1"',
        )
    )  # dev rows only: the seal covers heldout2 rows, so adding a dev row keeps it valid
    write_jsonl(
        paths.judge_results,
        [r for q in ("q1", "q2", "q-held", "q-held2", "bare") for r in judged(f"d-{q}")],
    )
    return write_run(
        paths.runs / "run-sealed.jsonl",
        header("sealed", splits=(Split.dev, Split.heldout2), started=T0.replace(hour=23)),
        rows,
    )


def test_a_sealed_question_file_and_a_separate_heldout2_file_are_read(paths: EvalPaths) -> None:
    path = _sealed_layout(paths)
    at = _pick(eval_app(paths, "Drill-down"), drill.RUN, path)
    assert not at.exception
    page = page_text(at)
    assert "unreadable" not in page
    assert at.selectbox(key=f"{drill.ITEM}:{path}").options == ["q1", "q2", "q-bare"]
    assert "2 held-out (heldout2) row(s) hidden." in page
    assert "SEALED" not in everything(at)


def test_dev_items_of_a_sealed_layout_are_labelable_and_held_out_ones_never(
    paths: EvalPaths,
) -> None:
    path = _sealed_layout(paths)
    store = FakeLabels()
    at = _pick(
        eval_app(paths, "Label queue", labels=store, chunks=_chunks_for(path)), labels.RUN, path
    )
    assert "First dev question?" in page_text(at)
    assert "Not in the queue: 1 with no shown notes recorded" in page_text(at)
    for _ in range(2):
        assert "SEALED" not in everything(at)
        at = _label(at, 2, 2, 2)
    assert sorted(lb.item_id for lb in store.appended) == ["q1", "q2"]
    assert "Nothing left to label in this run." in page_text(at)


def test_an_edited_sealed_heldout2_file_is_a_problem_and_fails_closed(paths: EvalPaths) -> None:
    path = _sealed_layout(paths)
    paths.heldout2.write_text(paths.heldout2.read_text().replace("second", "EDITED"))
    at = _pick(eval_app(paths, "Drill-down"), drill.RUN, path)
    assert not at.exception
    page = page_text(at)
    assert "heldout2.toml: unreadable (SealError)" in page
    assert "No rows to show." in page  # no question set: no row can be proven dev
    assert "EDITED" not in everything(at)


# ------------------------------------------------------------------ round 3: queue details
def test_the_queue_picker_names_runs_neutrally_and_lists_only_e_g_runs(paths: EvalPaths) -> None:
    path = _queue_set(paths)
    at = _pick(eval_app(paths, "Label queue", chunks=_chunks_for(path)), labels.RUN, path)
    options = [str(o) for o in at.selectbox(key=labels.RUN).options]
    assert len(options) == 2  # the golden E-G run and the queue run; no other kind
    for option in options:
        for hidden in ("queue", "synthetic", "generator"):  # run id, arm, kind
            assert hidden not in option
        assert " · run " in option


def test_notes_of_a_skipped_item_are_not_refetched_on_a_rerun(paths: EvalPaths) -> None:
    path, _ = _queue_with_notes(paths, first_changed=True)
    source = _chunks_for(path)
    at = _pick(eval_app(paths, "Label queue", chunks=source), labels.RUN, path)
    before = len(source.calls)
    at = at.text_input(key=labels.BATCH).input("b2").run()  # a rerun
    assert len(source.calls) == before + 1  # only the item on screen
    assert "Skipped: 1 whose shown notes changed" in page_text(at)


# ------------------------------------------------------------------ round 3: calibration
def test_an_untrusted_criterion_still_shows_its_kappa_ci_and_raw_agreement(
    paths: EvalPaths,
) -> None:
    summary = summary_of("generator")
    changes: dict[str, dict[str, object]] = {
        "calibration.relevance.kappa_linear": {
            "value": 0.41,
            "ci_low": 0.2,
            "ci_high": 0.6,
            "trusted": False,
            "note": "leniency p=0.03; trust needs n >= 30",
        },
        "calibration.relevance.raw_agreement": {"value": 0.7, "trusted": False},
        "generator.judge.relevance": {"trusted": False},
    }
    metrics = [m.model_copy(update=changes.get(m.name, {})) for m in summary.metrics]
    (paths.runs / "run-generator.summary.json").write_text(
        summary.model_copy(update={"metrics": metrics}).model_dump_json()
    )
    at = eval_app(paths, "Calibration")
    rows = _rows(at, "metric")
    kappa = rows["calibration.relevance.kappa_linear"]
    assert (kappa["value"], kappa["95% CI"], kappa["judge"]) == (
        "0.41",
        "[0.2, 0.6]",
        "not trusted",
    )
    assert rows["calibration.relevance.raw_agreement"]["value"] == "0.7"
    assert kappa["note"] == "leniency p=0.03; trust needs n >= 30"  # EV-METRICS puts them there
    page = page_text(at)
    assert "Judge not trusted for relevance: its scores don't count." in page
    assert "Judge not trusted for completeness" not in page


def test_untrusted_summary_kappa_everywhere_is_the_no_trusted_judge_outcome(
    paths: EvalPaths,
) -> None:
    summary = summary_of("generator")
    metrics = [
        m.model_copy(update={"trusted": False}) if m.trusted is not None else m
        for m in summary.metrics
    ]
    (paths.runs / "run-generator.summary.json").write_text(
        summary.model_copy(update={"metrics": metrics}).model_dump_json()
    )
    page = page_text(eval_app(paths, "Calibration"))
    assert NO_TRUSTED_JUDGE in page
    for c in ("relevance", "completeness", "faithfulness"):
        assert f"Judge not trusted for {c}" in page


# ------------------------------------------------------------------ round 3: false verified
def test_the_verifier_view_sets_the_synthetic_curve_beside_the_confirmed_real_rate(
    paths: EvalPaths,
) -> None:
    for name, extra in (
        (
            "verifier",
            MetricValue(
                name="verifier.false_verified@0.10", value=0.012, n=20, method=Method.point
            ),
        ),
        (
            "generator",
            MetricValue(
                name="generator.false_verified.confirmed",
                value=0.25,
                ci_low=0.05,
                ci_high=0.57,
                n=8,
                method=Method.exact,
            ),
        ),
    ):
        summary = summary_of(name)
        path = paths.runs / f"run-{name}.summary.json"
        updated = summary.model_copy(update={"metrics": [*summary.metrics, extra]})
        path.write_text(updated.model_dump_json())
    at = eval_app(paths, "Verifier")
    assert _rows(at, "base error rate")["0.10"]["value"] == "0.012"
    assert "Real rate (owner-confirmed, a lower bound): 2/8 · 95% CI [0.05, 0.57] · exact" in (
        page_text(at)
    )


def test_without_adjudications_the_real_false_verified_rate_says_so(paths: EvalPaths) -> None:
    page = page_text(eval_app(paths, "Verifier"))
    assert "Real rate: no E-G run has adjudicated judge-flagged answers yet." in page


# ------------------------------------------------------------------ round 3: bad flags
def test_a_malformed_eval_flag_is_an_error_on_the_page_not_a_stop(
    paths: EvalPaths, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sys, "argv", ["app.py", "--eval-runs"])
    at = eval_app(paths, None)
    assert not at.exception
    page = page_text(at)
    assert "Bad command-line flags" in page
    assert "--eval-runs" in page


# ------------------------------------------------------------------ reachable without the daily key
def test_the_eval_page_opens_without_the_daily_key(
    paths: EvalPaths, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Integrator (EV-UI handoff): the Eval page needs no daily key; Chat still does."""
    from test_ev_ui_support import FakeLabels

    from ragforge_ui.eval_page import LABELS, PATHS

    monkeypatch.setenv("RAGFORGE_KEY_FILE", str(paths.runs / "no-such.key"))
    at = AppTest.from_file(str(APP), default_timeout=30)
    at.session_state[PATHS] = paths
    at.session_state[LABELS] = FakeLabels()
    at.run()
    assert at.error  # Chat (the default page) cannot load the missing key
    at = at.sidebar.radio[0].set_value("Eval").run()
    assert not at.exception, at.exception
    assert not at.error
    assert "Evals" in page_text(at)


@pytest.mark.parametrize("check", ["host_refused", "origin_refused"])
def test_the_eval_page_is_still_behind_the_host_and_origin_checks(
    paths: EvalPaths, monkeypatch: pytest.MonkeyPatch, check: str
) -> None:
    from ragforge_ui import config as ui_config

    monkeypatch.setattr(ui_config, check, lambda *args: True)
    at = AppTest.from_file(str(APP), default_timeout=30)
    at.session_state["rf_backend"] = FakeBackend()
    at.run()
    assert any("opened by another web page" in e.value for e in at.error)
    assert not at.sidebar.radio  # no page (Eval included) renders behind a refused guard
