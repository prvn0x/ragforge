"""Integrator (EV-UI handoff): generator summaries append Calibration records for the Eval page's
full panel, once per (criterion, judge key, label batch), so the trust state never flaps."""

from __future__ import annotations

from pathlib import Path

from test_ev_metrics_support import fixture_run, judge_results, labels

from ragforge_client.evals.interfaces import CALIBRATION_FILE
from ragforge_client.evals.runners import write_calibrations
from ragforge_client.evals.schema import Calibration, Label


def _records(directory: Path) -> list[Calibration]:
    text = (directory / CALIBRATION_FILE).read_text()
    return [Calibration.model_validate_json(x) for x in text.splitlines() if x.strip()]


def test_calibrations_are_written_once_per_label_batch(tmp_path: Path) -> None:
    run = fixture_run("generator")
    assert run.header.judge is not None
    owner = labels()
    assert write_calibrations(tmp_path, run.header.judge, judge_results(), owner) == 3
    first = _records(tmp_path)
    assert {c.criterion.value for c in first} == {"relevance", "completeness", "faithfulness"}
    assert all(sum(map(sum, c.confusion)) == c.n == 12 for c in first)
    # the same batch again: nothing appended (no flapping)
    assert write_calibrations(tmp_path, run.header.judge, judge_results(), owner) == 0
    # a NEW label batch: a new record per criterion
    newer: list[Label] = [lb.model_copy(update={"batch_id": "b2"}) for lb in owner]
    assert write_calibrations(tmp_path, run.header.judge, judge_results(), newer) == 3
    assert len(_records(tmp_path)) == 6


def test_no_labels_writes_no_record(tmp_path: Path) -> None:
    run = fixture_run("generator")
    assert run.header.judge is not None
    assert write_calibrations(tmp_path, run.header.judge, judge_results(), []) == 0
    assert not (tmp_path / CALIBRATION_FILE).exists()
