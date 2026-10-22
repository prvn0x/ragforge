"""Everything one script run of the Eval page reads, loaded once and handed to every view."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from functools import cached_property

from ragforge_client.evals.interfaces import Datasets
from ragforge_client.evals.schema import Calibration, EvalKind, JudgeResult
from ragforge_ui.config import display_url
from ragforge_ui.eval_view.config import EvalPaths
from ragforge_ui.eval_view.data import (
    DatasetBundle,
    JsonlRead,
    LabelAccess,
    LabelsRead,
    RunEntry,
    load_datasets,
    load_runs,
    read_jsonl,
    read_labels,
)
from ragforge_ui.eval_view.model import judge_index
from ragforge_ui.eval_view.notes import ChunkSource


@dataclass
class EvalContext:
    paths: EvalPaths
    entries: list[RunEntry]
    bundle: DatasetBundle
    judge: JsonlRead[JudgeResult]
    calibrations: JsonlRead[Calibration]
    labels: LabelAccess
    # Built only when the owner asks for note text (it reads the eval key file then).
    chunk_source: Callable[[], ChunkSource]
    api_url: str  # display-safe (no userinfo): shown verbatim in error boxes
    problems: list[str] = field(default_factory=list)

    @property
    def datasets(self) -> Datasets:
        return self.bundle.datasets

    @cached_property
    def judge_by_key(self) -> dict[str, JudgeResult]:
        return judge_index(self.judge.items)

    def latest_labels(self) -> LabelsRead:
        """The owner's labels, read on demand (the queue re-reads after each save)."""
        return read_labels(self.labels, self.paths.labels)

    def runs_of(self, *kinds: EvalKind) -> list[RunEntry]:
        return [e for e in self.entries if not kinds or e.header.eval_kind in kinds]


def load_context(
    paths: EvalPaths,
    labels: LabelAccess,
    chunk_source: Callable[[], ChunkSource],
    api_url: str,
) -> EvalContext:
    bundle = load_datasets(paths)
    judge = read_jsonl(paths.judge_results, JudgeResult)
    calibrations = read_jsonl(paths.calibrations, Calibration)
    runs = load_runs(paths.runs)
    problems = [*bundle.problems, *runs.problems]
    for read, path in ((judge, paths.judge_results), (calibrations, paths.calibrations)):
        if read.problem:
            problems.append(read.problem)
        if read.skipped:
            problems.append(f"{path.name}: {read.skipped} unreadable line(s) skipped")
    return EvalContext(
        paths=paths,
        entries=runs.entries,
        bundle=bundle,
        judge=judge,
        calibrations=calibrations,
        labels=labels,
        chunk_source=chunk_source,
        api_url=display_url(api_url),
        problems=problems,
    )
