"""LOCKED seams between the eval builders (integrator-owned). Each module implements its part;
callers depend only on these signatures, so 6 builders never edit each other's files.

    metrics.summarize(run, datasets, labels, judge_results) -> RunSummary          (B2)
    stats.*  (pure functions; see the list in B2's spec)                            (B2)
    gate.evaluate(baseline, candidate, aa_runs, datasets) -> GateVerdict            (B2)
    judge.Judge protocol + judge.make_judge(config) + judge.JudgeCache              (B3a)
    datasets.Datasets + datasets.load_* + datasets.LabelStore                       (B3a)
    runners.run_*(...) -> Path                                                      (B3b)

`Datasets` is the bundle every metric/gate function receives, so no one reparses files.
"""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from ragforge_client.evals.schema import (
    Criterion,
    InterviewSet,
    JudgeKey,
    JudgeResult,
    Label,
    QuestionSet,
    VerifierSet,
)


@dataclass(frozen=True)
class Datasets:
    """The parsed datasets a run was made from (any may be None for kinds that do not use it)."""

    questions: QuestionSet | None = None
    verifier: VerifierSet | None = None
    interview: InterviewSet | None = None


@dataclass(frozen=True)
class JudgeItem:
    """One thing to judge. `item_digest` is the digest of exactly what the judge is shown."""

    item_digest: str
    criteria: tuple[Criterion, ...]
    question: str
    answer: str
    # the notes shown to the answer model (text fetched via /v1/eval/chunks, digest-checked)
    notes: tuple[str, ...] = ()
    facts: tuple[str, ...] = ()  # E-G completeness reference
    level: str | None = None  # E-I at_level


class Judge(Protocol):
    @property
    def key(self) -> JudgeKey: ...

    async def score(self, items: Sequence[JudgeItem]) -> list[JudgeResult]:
        """One JudgeResult per (item, criterion). A reply that fails the schema gives score
        None (counted, never imputed); never raises for a bad reply, only for transport."""
        ...


class LabelReader(Protocol):
    def latest(self) -> list[Label]:
        """The newest label per (item_digest, kind)."""
        ...


class LabelWriter(Protocol):
    def append(self, label: Label) -> None:
        """Atomic, locked append to labels.jsonl (the CLI and the Eval page may both write)."""
        ...


# File conventions (LOCKED): next to `<run>.jsonl` a runner writes `<run>.summary.json`
# (RunSummary) once the run finishes; labels live in `<dataset dir>/labels.jsonl`; judge results
# in `<dataset dir>/judge-cache.jsonl`; gate verdicts in `<runs dir>/gate-<surface>.json`.
SUMMARY_SUFFIX = ".summary.json"
# `<dataset dir>/calibration.jsonl`: one Calibration per line, appended when a generator run's
# summary is written and ONLY for a new (criterion, judge key, label batch): no flapping.
CALIBRATION_FILE = "calibration.jsonl"
# `<run>.reveal.json`: the RECORD that a run's heldout2 rows were revealed (ADR-0017 D1: one
# reveal). Written by whoever reveals (the Eval page or the CLI), honoured by both; holds the
# run id and the time only, never text.
REVEAL_SUFFIX = ".reveal.json"


def reveal_path(run_path: Path) -> Path:
    name = run_path.name.removesuffix(".jsonl")
    return run_path.with_name(name + REVEAL_SUFFIX)


def revealed_at(run_path: Path) -> str | None:
    """When this run's held-out rows were revealed (ISO time), or None."""
    path = reveal_path(run_path)
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    value = data.get("revealed_at")
    return value if isinstance(value, str) else None


def record_reveal(run_path: Path, run_id: str) -> str:
    """Write the reveal record (0600, exclusive: the FIRST reveal is the one recorded)."""
    stamp = datetime.now(UTC).isoformat()
    payload = json.dumps({"run_id": run_id, "revealed_at": stamp}) + "\n"
    try:
        fd = os.open(reveal_path(run_path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        return revealed_at(run_path) or stamp
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(payload)
    return stamp
