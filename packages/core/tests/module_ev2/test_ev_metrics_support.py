"""Shared loaders and run builders for the EV-METRICS tests (no tests here).

Everything is built from the committed D0 demo set and its golden fixtures, or synthesised in
memory from them: no model, no stack, no private data."""

from __future__ import annotations

import importlib.util
import json
import tomllib
from collections.abc import Callable
from dataclasses import replace
from functools import cache
from pathlib import Path
from types import ModuleType

from ragforge_client.evals.interfaces import Datasets
from ragforge_client.evals.runfile import RunFile, read_run
from ragforge_client.evals.schema import (
    InterviewSet,
    JudgeResult,
    Label,
    QuestionSet,
    RunRow,
    VerifierSet,
)

ROOT = Path(__file__).resolve().parents[4]
DEMO = ROOT / "evals" / "demo"
FIXTURES = DEMO / "fixtures"
KINDS = ("retrieval", "verifier", "interview", "generator")


@cache
def datasets() -> Datasets:
    return Datasets(
        questions=QuestionSet.model_validate(tomllib.load((DEMO / "questions.toml").open("rb"))),
        verifier=VerifierSet.model_validate(json.loads((DEMO / "verifier.json").read_text())),
        interview=InterviewSet.model_validate(json.loads((DEMO / "interview.json").read_text())),
    )


def fixture_run(kind: str) -> RunFile:
    return read_run(FIXTURES / f"run-{kind}.jsonl")


def labels() -> list[Label]:
    lines = (FIXTURES / "labels.jsonl").read_text().splitlines()
    return [Label.model_validate_json(line) for line in lines if line.strip()]


def judge_results() -> list[JudgeResult]:
    lines = (FIXTURES / "judge-results.jsonl").read_text().splitlines()
    return [JudgeResult.model_validate_json(line) for line in lines if line.strip()]


def expected() -> dict[str, float | None]:
    data: dict[str, float | None] = json.loads((FIXTURES / "expected_summaries.json").read_text())
    return data


@cache
def reference() -> ModuleType:
    """The independent stdlib oracle, imported by path (it is a script, not a package)."""
    spec = importlib.util.spec_from_file_location("ev_reference", FIXTURES / "reference_metrics.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def with_rows(run: RunFile, change: Callable[[RunRow], RunRow], run_id: str = "") -> RunFile:
    """A copy of `run` with every row passed through `change` (and optionally a new run id)."""
    header = run.header.model_copy(update={"run_id": run_id}) if run_id else run.header
    return replace(run, header=header, rows=[change(r) for r in run.rows])


def renamed(run: RunFile, run_id: str, **header: object) -> RunFile:
    return replace(run, header=run.header.model_copy(update={"run_id": run_id, **header}))
