"""Review round 2 P1: errored and timed-out turns can never make an arm meet the bar.

A timeout's latency is only bounded below (it took at least the deadline), so it counts in p95 at
its deadline instead of being dropped; any errored turn is a bar miss; and a run whose own
question timeout was under the p95 bar cannot pass, since that deadline makes p95 < 10 s true by
construction.
"""

from __future__ import annotations

import io
from pathlib import Path

from ev_support import (
    GOOD_EOS,
    GOOD_LRU,
    GOOD_RAFT,
    outcome,
    perfect_run_outcomes,
    write_questions,
    write_run,
)

from ragforge_client.evalrun import (
    ABSTAIN_PREFIX,
    Completeness,
    Label,
    Outcome,
    answer_digest,
    compare_runs,
    judge_bar,
    latency_samples,
    load_questions,
    save_labels,
    summarize,
)

FULL = Completeness(asked=4, expected=4, finished=True)


def _ok_labels(run_path: Path) -> None:
    save_labels(
        run_path,
        {
            qid: Label("ok", answer_digest(text), "t")
            for qid, text in (
                ("dev-eos", GOOD_EOS),
                ("dev-lru", GOOD_LRU),
                ("secret-heldout-raft", GOOD_RAFT),
            )
        },
    )


def _probe_outcomes() -> list[Outcome]:
    """The reviewer's probe: 3 labelled-ok answers in 2 s, the unanswerable one timed out."""
    rows = perfect_run_outcomes(total_s=2.0)[:3]
    stalled = outcome("dev-off", "", total_s=300.0, error="timeout after 300s")
    stalled.timed_out = True
    return [*rows, stalled]


def _row(out: io.StringIO, label: str) -> str:
    return next(line for line in out.getvalue().splitlines() if line.startswith(f"| {label} "))


def test_review_probe_timeout_on_unanswerable_is_not_a_yes(tmp_path: Path) -> None:
    questions = write_questions(tmp_path)
    path = write_run(
        tmp_path / "probe.jsonl", _probe_outcomes(), label="probe", footer={"complete": True}
    )
    path.write_text(
        path.read_text().replace('"notes": ""', '"notes": "", "question_timeout_s": 300.0', 1)
    )
    _ok_labels(path)
    out = io.StringIO()
    compare_runs([str(path)], str(questions), out=out)
    row = _row(out, "probe")
    assert "**yes**" not in row
    assert "1 errored turn(s) (1 timed out)" in row
    assert "p95 ≥300.0≥10s" in row  # the timeout is in the sample, at its deadline
    assert "| ≥2.0 | ≥300.0 |" in row  # p50 / p95 marked as lower bounds


def test_timeout_counts_at_its_deadline_even_if_measured_short() -> None:
    """total_s can undershoot the deadline by clock jitter; the deadline is the floor."""
    stalled = outcome("q", "", total_s=299.9, error="timeout after 300s")
    stalled.timed_out = True
    assert latency_samples([stalled], 300.0) == [300.0]


def test_legacy_timeout_rows_are_recognised_from_the_error_text() -> None:
    """Runs recorded before `timed_out` existed only have the error string."""
    legacy = outcome("q", "", total_s=300.0, error="timeout after 300s")
    fast_5xx = outcome("r", "", total_s=0.1, error="503 Unavailable")
    assert latency_samples([legacy, fast_5xx], None) == [300.0]


def test_fast_error_stays_out_of_latency_but_still_misses_the_bar(tmp_path: Path) -> None:
    questions = load_questions(write_questions(tmp_path))
    rows = perfect_run_outcomes(total_s=2.0)
    rows[1] = outcome("dev-lru", "", total_s=0.1, error="503 Unavailable")
    s = summarize(questions, rows)
    assert (s["p95_s"], s["errors"], s["timeouts"]) == (2.0, 1, 0)
    verdict = judge_bar(s, FULL)
    assert verdict.met is False
    assert "1 errored turn(s)" in verdict.reasons


def test_question_timeout_under_the_bar_can_never_be_a_yes(tmp_path: Path) -> None:
    """Gaming case: `run --question-timeout 9` makes every completed turn < 10 s by construction."""
    questions = write_questions(tmp_path)
    path = write_run(
        tmp_path / "gamed.jsonl",
        perfect_run_outcomes(total_s=2.0),
        label="gamed",
        footer={"complete": True},
    )
    path.write_text(
        path.read_text().replace('"notes": ""', '"notes": "", "question_timeout_s": 9', 1)
    )
    _ok_labels(path)
    out = io.StringIO()
    compare_runs([str(path)], str(questions), out=out)
    row = _row(out, "gamed")
    assert "**yes**" not in row
    assert row.endswith("| no: question-timeout 9s<10s |")


def test_question_timeout_at_or_above_the_bar_and_no_errors_can_pass(tmp_path: Path) -> None:
    questions = load_questions(write_questions(tmp_path))
    labels = {
        o.id: Label("ok", answer_digest(o.final), "t")
        for o in perfect_run_outcomes()
        if not o.final.startswith(ABSTAIN_PREFIX)
    }
    s = summarize(questions, perfect_run_outcomes(total_s=2.0), labels, question_timeout_s=10.0)
    assert judge_bar(s, FULL).met is True
