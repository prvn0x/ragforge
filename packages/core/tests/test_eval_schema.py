"""ADR-0017 lock step: the D0 demo set and its golden fixtures load under the LOCKED schemas, and
the independent reference oracle reproduces the committed expected values (so neither drifts
silently). No model, no stack, no private data."""

from __future__ import annotations

import json
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest
from pydantic import TypeAdapter

from ragforge_client.evals.schema import (
    EXPECTED_LAYER,
    InterviewSet,
    JudgeResult,
    Label,
    Mutation,
    QuestionSet,
    RunFooter,
    RunHeader,
    RunRow,
    VerifierSet,
    digest,
    parse_page_key,
)
from ragforge_contracts.api import ApiScope

ROOT = Path(__file__).resolve().parents[3]
DEMO = ROOT / "evals" / "demo"
FIXTURES = DEMO / "fixtures"


def test_the_demo_datasets_validate() -> None:
    qs = QuestionSet.model_validate(tomllib.load((DEMO / "questions.toml").open("rb")))
    assert len(qs.questions) == 20
    vs = VerifierSet.model_validate(json.loads((DEMO / "verifier.json").read_text()))
    assert {i.mutation for i in vs.items if i.mutation} == set(Mutation)  # every type planted
    iv = InterviewSet.model_validate(json.loads((DEMO / "interview.json").read_text()))
    assert len(iv.seeds) == 4


def test_every_planted_span_marks_text_that_differs_from_its_clean_source() -> None:
    vs = VerifierSet.model_validate(json.loads((DEMO / "verifier.json").read_text()))
    clean = {i.id: i.answer for i in vs.items if i.mutation is None}
    for item in vs.items:
        if item.span is None:
            continue
        assert item.source_id in clean
        assert item.answer != clean[item.source_id]
        assert item.span.end <= len(item.answer)
        assert item.expected_layer is EXPECTED_LAYER[item.mutation]  # type: ignore[index]


@pytest.mark.parametrize(
    "name",
    ["run-retrieval.jsonl", "run-verifier.jsonl", "run-interview.jsonl", "run-generator.jsonl"],
)
def test_the_golden_runs_are_valid_v2_run_files(name: str) -> None:
    lines = [json.loads(x) for x in (FIXTURES / name).read_text().splitlines() if x.strip()]
    RunHeader.model_validate(lines[0])
    RunFooter.model_validate(lines[-1])
    rows = [RunRow.model_validate(x) for x in lines[1:-1]]
    assert rows
    assert RunFooter.model_validate(lines[-1]).rows == len(rows)


def test_labels_and_judge_results_validate() -> None:
    for line in (FIXTURES / "labels.jsonl").read_text().splitlines():
        Label.model_validate_json(line)
    for line in (FIXTURES / "judge-results.jsonl").read_text().splitlines():
        JudgeResult.model_validate_json(line)


def test_the_reference_oracle_reproduces_the_committed_expected_values(tmp_path: Path) -> None:
    committed = json.loads((FIXTURES / "expected_summaries.json").read_text())
    copy = tmp_path / "fixtures"
    copy.mkdir()
    for f in FIXTURES.iterdir():
        if f.is_file():
            (copy / f.name).write_bytes(f.read_bytes())
    for name in ("questions.toml", "verifier.json", "interview.json"):
        (tmp_path / name).write_bytes((DEMO / name).read_bytes())
    oracle = [sys.executable, "-I", str(copy / "reference_metrics.py")]
    subprocess.run(oracle, check=True)  # noqa: S603 - our own committed script, no shell
    assert json.loads((copy / "expected_summaries.json").read_text()) == committed


def test_digest_is_canonical_and_page_keys_round_trip() -> None:
    assert digest({"b": 1, "a": [1, 2]}) == digest({"a": [1, 2], "b": 1})
    key = parse_page_key("file:kafka-basics.md")
    assert (key.source_kind, key.key) == ("file", "kafka-basics.md")
    with pytest.raises(ValueError, match="page key"):
        parse_page_key("no-colon")


def test_the_eval_scope_exists_in_the_contract() -> None:
    assert ApiScope("eval") is ApiScope.eval
    assert TypeAdapter(ApiScope).validate_python("eval") is ApiScope.eval
