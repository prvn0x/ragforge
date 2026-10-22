"""EV-METRICS round 3: a run that measured LESS than its baseline never passes the gate.

An empty, truncated (no footer) or item-dropping candidate used to pass because pairing only
sees the intersection of the two arms. Each surface is probed here, including the warn-only
interview surface: an incomplete run is not a measurement, so it is not_evaluable everywhere."""

from __future__ import annotations

from dataclasses import replace

import pytest
from test_ev_metrics_support import datasets, fixture_run, renamed, with_rows

from ragforge_client.evals import gate
from ragforge_client.evals.runfile import RunFile
from ragforge_client.evals.schema import (
    AnswerOutput,
    AnswerTier,
    GateOutcome,
    GateVerdict,
    GradeOutput,
    RunRow,
)

KINDS = ("retrieval", "generator", "interview", "verifier")


def _graded(run_id: str = "gen") -> RunFile:
    def change(r: RunRow) -> RunRow:
        assert isinstance(r.output, AnswerOutput)
        return r.model_copy(update={"grades": {"correct": not r.output.abstained}})

    return with_rows(fixture_run("generator"), change, run_id)


def _base(kind: str) -> RunFile:
    return _graded() if kind == "generator" else fixture_run(kind)


def _coverage(verdict: GateVerdict) -> str:
    return next(c.note for c in verdict.checks if c.name == "coverage")


@pytest.mark.parametrize("kind", KINDS)
def test_an_empty_candidate_is_not_evaluable_and_never_passes(kind: str) -> None:
    base = _base(kind)
    cand = replace(renamed(base, "empty"), rows=[])
    verdict = gate.evaluate(base, cand, [], datasets())
    assert verdict.outcome is GateOutcome.not_evaluable, gate.format_verdict(verdict)
    assert not gate.passes(verdict)
    assert not gate.passes(verdict.model_copy(update={"override": "ship it"}))
    assert f"{len({r.item_id for r in base.rows})} of" in _coverage(verdict)


@pytest.mark.parametrize("kind", KINDS)
def test_a_truncated_unfinished_candidate_is_not_evaluable(kind: str) -> None:
    """The first 3 rows and no footer: a crashed or still-resumable run."""
    base = _base(kind)
    cand = replace(renamed(base, "crashed"), rows=base.rows[:3], footer=None)
    verdict = gate.evaluate(base, cand, [], datasets())
    assert verdict.outcome is GateOutcome.not_evaluable, gate.format_verdict(verdict)
    assert not gate.passes(verdict)
    note = _coverage(verdict)
    assert "no footer" in note
    assert "absent from candidate" in note


def test_a_finished_run_that_dropped_items_is_not_evaluable() -> None:
    """A footer is not enough: a runner that silently skips items measured less."""
    base = fixture_run("retrieval")
    cand = replace(renamed(base, "dropped"), rows=base.rows[1:])
    verdict = gate.evaluate(base, cand, [], datasets())
    assert verdict.outcome is GateOutcome.not_evaluable
    assert "1 of 20 baseline item(s) absent" in _coverage(verdict)


def test_an_unfinished_baseline_is_not_evaluable() -> None:
    base = fixture_run("retrieval")
    verdict = gate.evaluate(replace(base, footer=None), renamed(base, "c"), [], datasets())
    assert verdict.outcome is GateOutcome.not_evaluable
    assert "baseline run has no footer" in _coverage(verdict)


@pytest.mark.parametrize("kind", ["retrieval", "generator", "interview"])
def test_a_complete_aa_candidate_passes_coverage(kind: str) -> None:
    base = _base(kind)
    verdict = gate.evaluate(base, renamed(base, "c"), [], datasets())
    assert next(c for c in verdict.checks if c.name == "coverage").outcome is GateOutcome.pass_


def test_a_baseline_item_the_candidate_has_no_value_for_blocks_the_check() -> None:
    """The row is present (coverage passes) but its deterministic grade is gone: chat.correct
    measured fewer items than the baseline, which blocks rather than pairing on the rest."""
    base = _graded()
    victim = next(r.item_id for r in base.rows if r.grades.get("correct") is True)
    cand = with_rows(
        base, lambda r: r.model_copy(update={"grades": {}}) if r.item_id == victim else r, "c"
    )
    verdict = gate.evaluate(base, cand, [], datasets())
    correct = next(c for c in verdict.checks if c.name == "chat.correct")
    assert correct.outcome is GateOutcome.block
    assert "1 baseline item(s) have no candidate value" in correct.note
    assert next(c for c in verdict.checks if c.name == "coverage").outcome is GateOutcome.pass_


def test_a_metric_neither_arm_records_only_warns() -> None:
    """The ungraded fixture has no `correct` grade in either arm: nothing to compare, and no
    evidence of a regression either, so it warns (never a silent pass, never a false block)."""
    base = fixture_run("generator")
    verdict = gate.evaluate(base, renamed(base, "c"), [], datasets())
    correct = next(c for c in verdict.checks if c.name == "chat.correct")
    assert correct.outcome is GateOutcome.warn
    assert correct.n == 0
    assert "neither arm records this metric" in correct.note


def test_the_level_trend_warn_is_one_sided() -> None:
    """5 of the 8 fixture blocks flattened (L falls in each), 3 untouched: the one-sided p is
    1/32 < 0.05 and warns; a two-sided test would say 2/32 and miss a real fall."""
    base = fixture_run("interview")
    lowered = {
        ("s-ratelimit", AnswerTier.gold),
        ("s-ratelimit", AnswerTier.partial),
        ("s-rebalance", AnswerTier.gold),
        ("s-rebalance", AnswerTier.partial),
        ("s-outbox", AnswerTier.gold),
    }

    def flatten(r: RunRow) -> RunRow:
        out = r.output
        if isinstance(out, GradeOutput) and (out.seed_id, out.tier) in lowered:
            return r.model_copy(update={"output": out.model_copy(update={"score": 5})})
        return r

    verdict = gate.evaluate(base, with_rows(base, flatten, "flat5"), [], datasets())
    level = next(c for c in verdict.checks if c.name == "interview.level.page_l")
    assert level.p_value == 1 / 32
    assert level.outcome is GateOutcome.warn
