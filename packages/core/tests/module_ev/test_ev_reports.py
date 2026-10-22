"""`score` and `compare`: held-out rows stay hidden, the arms table carries every metric, and the
success-bar flag is right at its edges (including "unknown" while invented% is unlabelled)."""

from __future__ import annotations

import io
import math
from pathlib import Path
from typing import Any

import pytest
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
    Completeness,
    InventedStats,
    Label,
    answer_digest,
    compare_runs,
    judge_bar,
    load_questions,
    load_run,
    run,
    run_completeness,
    save_labels,
    score_runs,
    summarize,
)


def _all_ok_labels(run_path: Path) -> None:
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


def _summary(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "correct": 90.0,
        "p95_s": 5.0,
        "invented": InventedStats(answered=10, labelled=10, invented=0, unsure=0),
    }
    return base | overrides


FULL = Completeness(asked=4, expected=4, finished=True)


@pytest.mark.parametrize(
    ("overrides", "met", "reason"),
    [
        ({}, True, ""),
        ({"correct": 80.0}, True, ""),  # the bar is >= 80
        ({"correct": 79.9}, False, "correct 80<80"),
        ({"correct": math.nan}, False, "correct -<80"),
        ({"p95_s": 10.0}, False, "p95 10.0≥10s"),  # the bar is strictly < 10 s
        ({"invented": InventedStats(50, 50, 1, 0)}, True, ""),  # exactly 2%
        ({"invented": InventedStats(50, 50, 1, 1)}, False, "invented 4.0>2"),  # unsure counts
        ({"invented": InventedStats(10, 0, 0, 0)}, None, "invented% unlabelled"),
        ({"invented": InventedStats(10, 9, 0, 0)}, None, "invented% partial 9/10"),
        # A measured miss is reported as "no" even when labels are missing.
        ({"correct": 40.0, "invented": InventedStats(10, 0, 0, 0)}, False, "correct 40<80"),
        # Partial labels: invented/answered is a lower bound, so 3/10 already fails the bar...
        ({"invented": InventedStats(10, 3, 3, 0)}, False, "invented ≥30.0>2 (partial 3/10)"),
        # ...but unsure does NOT count toward the lower bound: a relabel may turn it into ok, so
        # counting it would make the "bound" fall (review round 3)...
        ({"invented": InventedStats(10, 3, 0, 1)}, None, "invented% partial 3/10"),
        # ...but a bound still within the bar stays unknown (1/50 = 2% is not over).
        ({"invented": InventedStats(50, 10, 1, 0)}, None, "invented% partial 10/50"),
    ],
)
def test_judge_bar_edges(overrides: dict[str, Any], met: bool | None, reason: str) -> None:
    verdict = judge_bar(_summary(**overrides), FULL)
    assert verdict.met is met
    assert reason in "; ".join(verdict.reasons)


@pytest.mark.parametrize(
    ("done", "met", "reason"),
    [
        (Completeness(asked=1, expected=4, finished=False), False, "incomplete 1/4"),
        # The footer says complete, but the split has questions the run never asked.
        (Completeness(asked=3, expected=4, finished=True), False, "incomplete 3/4"),
        # Killed before the footer, yet every question asked: cannot vouch for it either way.
        (Completeness(asked=4, expected=4, finished=None), None, "no footer"),
        (Completeness(asked=2, expected=4, finished=None), False, "incomplete 2/4"),
    ],
)
def test_judge_bar_never_passes_a_partial_run(
    done: Completeness, met: bool | None, reason: str
) -> None:
    verdict = judge_bar(_summary(), done)  # otherwise perfect and fully labelled
    assert verdict.met is met
    assert reason in "; ".join(verdict.reasons)


def test_compare_rejects_an_interrupted_run_with_perfect_answers(tmp_path: Path) -> None:
    """Regression (review P1): footer complete=false + 1 perfect, labelled answer was '**yes**'."""
    questions = write_questions(tmp_path)
    path = write_run(
        tmp_path / "killed.jsonl",
        perfect_run_outcomes()[:1],
        label="A3-8b",
        footer={"complete": False, "resources": {}},
    )
    _all_ok_labels(path)
    out = io.StringIO()
    compare_runs([str(path)], str(questions), out=out)
    row = next(line for line in out.getvalue().splitlines() if line.startswith("| A3-8b"))
    assert "**yes**" not in row
    assert row.endswith("| no: incomplete 1/4 |")
    assert "| 1/4 incomplete |" in row


def test_completeness_counts_only_the_runs_split(tmp_path: Path) -> None:
    questions = load_questions(write_questions(tmp_path))
    dev_only = [o for o in perfect_run_outcomes() if o.id != "secret-heldout-raft"]
    path = write_run(tmp_path / "dev.jsonl", dev_only, split="dev", footer={"complete": True})
    assert run_completeness(load_run(path), questions) == Completeness(3, 3, True)


def test_compare_table_flags_only_the_arm_meeting_the_bar(tmp_path: Path) -> None:
    questions = write_questions(tmp_path)
    fast = write_run(
        tmp_path / "fast.jsonl",
        perfect_run_outcomes(total_s=2.0),
        label="A2-fast",
        notes="max_tool_rounds=0 | thinking off",
        footer={
            "complete": True,
            "resources": {
                "samples": 3,
                "swap_used_mb": 2048.0,
                "swap_baseline_mb": 1536.0,
                "ollama_rss_mb": 5120.0,
            },
        },
    )
    _all_ok_labels(fast)
    slow = write_run(
        tmp_path / "slow.jsonl",
        perfect_run_outcomes(total_s=12.0),
        label="A3-slow",
        footer={"complete": True},
    )
    _all_ok_labels(slow)
    unlabelled = write_run(
        tmp_path / "u.jsonl",
        perfect_run_outcomes(),
        label="A4-unlabelled",
        footer={"complete": True},
    )
    out = io.StringIO()
    compare_runs([str(fast), str(slow), str(unlabelled)], str(questions), out=out)
    rows = {line.split(" | ")[0].lstrip("| "): line for line in out.getvalue().splitlines()}
    assert rows["A2-fast"].endswith("| **yes** |")
    assert "max_tool_rounds=0 \\| thinking off" in rows["A2-fast"]  # pipe escaped, row intact
    assert "| 2.0 | 0.5 | 5.0 |" in rows["A2-fast"]  # peak swap / swap delta / ollama GB
    assert "| 4/4 |" in rows["A2-fast"]  # every question of the split asked
    assert "| ollama: m:1b |" in rows["A2-fast"]  # provider shown with the model
    assert "| 100/0 |" in rows["A2-fast"]  # cited model/inferred
    assert "| 1/1 |" in rows["A2-fast"]  # abstain ok
    assert "no: p95 12.0≥10s" in rows["A3-slow"]
    assert "| - | - | - |" in rows["A3-slow"]  # memory not observed: dashes, not 0
    assert "unknown: invented% unlabelled" in rows["A4-unlabelled"]
    assert rows["A4-unlabelled"].count("|") == rows["A2-fast"].count("|") - 1  # minus escaped |


def test_errored_turns_do_not_pull_latency_down(tmp_path: Path) -> None:
    """A 0.1 s 5xx is not a fast answer: p95 is over completed turns, errors counted beside it."""
    questions = load_questions(write_questions(tmp_path))
    outcomes = [
        outcome("dev-eos", GOOD_EOS, model=["Kafka Delivery"], total_s=12.0),
        outcome("dev-lru", "", total_s=0.1, error="503 Unavailable"),
    ]
    s = summarize(questions, outcomes)
    assert (s["p50_s"], s["p95_s"], s["errors"]) == (12.0, 12.0, 1)
    assert judge_bar(s, FULL).met is False


def test_compare_never_prints_per_question_ids(tmp_path: Path) -> None:
    questions = write_questions(tmp_path)
    path = write_run(tmp_path / "r.jsonl", perfect_run_outcomes(), split="heldout")
    out = io.StringIO()
    compare_runs([str(path)], str(questions), out=out)
    assert "secret-heldout-raft" not in out.getvalue()
    assert "dev-eos" not in out.getvalue()


def test_score_verbose_hides_heldout_rows_unless_revealed(tmp_path: Path) -> None:
    questions = write_questions(tmp_path)
    path = write_run(tmp_path / "r.jsonl", perfect_run_outcomes())
    hidden = io.StringIO()
    score_runs([str(path)], str(questions), verbose=True, out=hidden)
    assert "dev-eos" in hidden.getvalue()
    assert "secret-heldout-raft" not in hidden.getvalue()
    assert "1 held-out row(s) hidden" in hidden.getvalue()
    shown = io.StringIO()
    score_runs([str(path)], str(questions), verbose=True, reveal=True, out=shown)
    assert "secret-heldout-raft" in shown.getvalue()


def test_score_table_shows_split_and_invented_state(tmp_path: Path) -> None:
    questions = write_questions(tmp_path)
    path = write_run(
        tmp_path / "r.jsonl",
        [
            outcome("dev-eos", GOOD_EOS, model=["Kafka Delivery"]),
            outcome("dev-lru", GOOD_LRU, inferred=["Caching"]),
        ],
    )
    out = io.StringIO()
    score_runs([str(path)], str(questions), verbose=False, out=out)
    row = out.getvalue().splitlines()[2]
    assert "| 50 | 50/50 | unlabelled |" in row  # model-only% then cited model/inferred%
    header = out.getvalue().splitlines()[0]
    assert "| model-only% | cited model/inferred% |" in header
    assert header.count("|") == row.count("|")


def test_cli_compare_and_score_parse(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    questions = write_questions(tmp_path)
    path = write_run(tmp_path / "r.jsonl", perfect_run_outcomes(), label="cli-arm")
    run(["--questions", str(questions), "compare", str(path)])
    assert "cli-arm" in capsys.readouterr().out
    run(["--questions", str(questions), "score", "-v", "--reveal", str(path)])
    assert "secret-heldout-raft" in capsys.readouterr().out
