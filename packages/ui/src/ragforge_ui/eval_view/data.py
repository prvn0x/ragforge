"""Local file access for the Eval page: read-only, except labels through `LabelWriter`.

Every reader degrades instead of raising: a missing or broken file becomes a short problem
string the page shows, so one bad file never blanks the whole page. Problem strings name the
file and the exception TYPE only: a validation message quotes the offending input, which here
can be answer text, and the page has no reason to echo file contents into an error box.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import BaseModel

from ragforge_client.evals.datasets import (
    LabelStore,
    load_interview_set,
    load_questions,
    load_verifier_set,
)
from ragforge_client.evals.interfaces import SUMMARY_SUFFIX, Datasets, LabelReader, LabelWriter
from ragforge_client.evals.runfile import RunFile, read_run
from ragforge_client.evals.schema import (
    GateVerdict,
    Label,
    LabelKind,
    QuestionSet,
    RunHeader,
    RunSummary,
    Surface,
)
from ragforge_ui.eval_view.config import EvalPaths


def problem(path: Path, exc: BaseException) -> str:
    return f"{path.name}: unreadable ({type(exc).__name__})"


# ------------------------------------------------------------------ runs and summaries
@dataclass(frozen=True, slots=True)
class RunEntry:
    """One v2 run with its summary (written by the runner when the run finished)."""

    run: RunFile
    summary: RunSummary | None
    summary_problem: str | None = None

    @property
    def key(self) -> str:
        """Stable widget value: the run file's path (run ids are not promised unique)."""
        return str(self.run.path)

    @property
    def header(self) -> RunHeader:
        return self.run.header

    @property
    def label(self) -> str:
        h = self.header
        return f"{h.run_id} · {h.eval_kind.value} · {h.arm} · {h.started_at:%Y-%m-%d %H:%M}"

    @property
    def blind_label(self) -> str:
        """For the label queue: the start time and a short id of the file only. The run id and
        the arm often name the model being calibrated, and the owner's label must not depend on
        it (the CLI hides arm and model for the same reason)."""
        short = hashlib.sha256(self.key.encode()).hexdigest()[:8]
        return f"{self.header.started_at:%Y-%m-%d %H:%M} · run {short}"


def summary_path(run_path: Path) -> Path:
    """`<run>.jsonl` -> `<run>.summary.json` (the LOCKED convention, interfaces.py)."""
    return run_path.with_name(run_path.name.removesuffix(".jsonl") + SUMMARY_SUFFIX)


def load_summary(run_path: Path) -> tuple[RunSummary | None, str | None]:
    path = summary_path(run_path)
    if not path.exists():
        return None, "no summary yet (the run is unfinished or was never summarized)"
    try:
        return RunSummary.model_validate_json(path.read_bytes()), None
    except (OSError, ValueError) as exc:
        return None, problem(path, exc)


@dataclass(frozen=True, slots=True)
class RunsRead:
    entries: list[RunEntry]
    problems: list[str] = field(default_factory=list)


def load_runs(directory: Path) -> RunsRead:
    """Every readable v2 run, newest first by start time.

    The same rule as `runfile.iter_runs` (v1 runs, label and judge files are not v2 runs and are
    skipped silently), applied per file with `read_run` so that one file the OS refuses to read
    is a problem line rather than an empty Runs list (`iter_runs` lets OSError escape the
    whole iteration)."""
    entries: list[RunEntry] = []
    problems: list[str] = []
    try:
        candidates = sorted(directory.glob("*.jsonl"))
    except OSError as exc:
        return RunsRead([], [problem(directory, exc)])
    for path in candidates:
        try:
            run = read_run(path)
        except ValueError:  # RunFileError, JSON and validation errors: not a v2 run
            continue
        except OSError as exc:
            problems.append(problem(path, exc))
            continue
        summary, why = load_summary(run.path)
        entries.append(RunEntry(run, summary, why))
    entries.sort(key=lambda e: e.header.started_at, reverse=True)
    return RunsRead(entries, problems)


def load_gate(directory: Path, surface: Surface) -> tuple[GateVerdict | None, str | None]:
    """The latest verdict for a surface (`<runs dir>/gate-<surface>.json`), if any."""
    path = directory / f"gate-{surface.value}.json"
    if not path.exists():
        return None, None
    try:
        return GateVerdict.model_validate_json(path.read_bytes()), None
    except (OSError, ValueError) as exc:
        return None, problem(path, exc)


# ------------------------------------------------------------------ JSONL side files
@dataclass(frozen=True, slots=True)
class JsonlRead[T: BaseModel]:
    items: list[T]
    skipped: int = 0  # unparseable lines (a crash mid-append, a newer schema)
    problem: str | None = None  # the whole file was unreadable


def read_jsonl[T: BaseModel](path: Path, model: type[T]) -> JsonlRead[T]:
    """Each valid line as `model`; a bad line is counted and skipped, never fatal."""
    if not path.exists():
        return JsonlRead([])
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, ValueError) as exc:
        return JsonlRead([], problem=problem(path, exc))
    items: list[T] = []
    skipped = 0
    for line in lines:
        if not line.strip():
            continue
        try:
            items.append(model.model_validate_json(line))
        except ValueError:
            skipped += 1
    return JsonlRead(items, skipped)


# ------------------------------------------------------------------ datasets
@dataclass(frozen=True, slots=True)
class DatasetBundle:
    datasets: Datasets
    problems: list[str] = field(default_factory=list)


def _load[T](path: Path, parse: Callable[[Path], T], problems: list[str]) -> T | None:
    if not path.exists():
        return None
    try:
        return parse(path)
    except (OSError, ValueError) as exc:  # DatasetError and JSON errors are ValueErrors
        problems.append(problem(path, exc))
        return None


def load_datasets(paths: EvalPaths) -> DatasetBundle:
    """The datasets the runs were made from: question text, splits (held-out filtering), the
    planted spans of D3 items and the D4 seeds. Any may be absent; the page says what it lacks.

    Parsed by EV-JUDGE's loaders, the ones the CLI and the metrics use, so the page reads a
    sealed `questions.toml` (its top-level `seal` key) and the private layout's separate sealed
    `heldout2.toml` exactly as they do. A seal that fails (`SealError`) is a problem line and
    leaves the questions unknown, which the held-out rule treats as held out (fail closed)."""
    problems: list[str] = []
    files = [p for p in (paths.questions, paths.heldout2) if p.exists()]
    questions = _load_questions(files, problems)
    verifier = _load(paths.verifier, load_verifier_set, problems)
    interview = _load(paths.interview, load_interview_set, problems)
    return DatasetBundle(Datasets(questions, verifier, interview), problems)


def _load_questions(files: list[Path], problems: list[str]) -> QuestionSet | None:
    if not files:
        return None
    try:
        return load_questions(*files)
    except (OSError, ValueError) as exc:  # DatasetError / SealError / TOML errors: ValueErrors
        problems.append(problem(_culprit(files, exc), exc))
        return None


def _culprit(files: list[Path], exc: BaseException) -> Path:
    """The file an error names (the loaders put the path first), else the first file."""
    text = str(exc)
    return next((p for p in files if str(p) in text), files[0])


# ------------------------------------------------------------------ labels
class FileLabelReader:
    """Read-only `LabelReader` over labels.jsonl: the newest line per (item_digest, kind) wins
    (append order, as the LOCKED Label contract says). Unlike the store it skips a bad line, so
    the drill-down can still show the good labels of a file the store refuses."""

    def __init__(self, path: Path) -> None:
        self._path = path

    def latest(self) -> list[Label]:
        newest: dict[tuple[str, LabelKind], Label] = {}
        for label in read_jsonl(self._path, Label).items:
            newest[(label.item_digest, label.kind)] = label
        return list(newest.values())


@dataclass(frozen=True, slots=True)
class LabelAccess:
    reader: LabelReader
    writer: LabelWriter | None  # None: read-only; `why` says how to enable labelling
    why: str = ""


def label_store(path: Path) -> LabelAccess:
    """EV-JUDGE's `LabelStore` (atomic, locked append; the CLI may write concurrently). The
    page never writes labels.jsonl itself: a second writer without the store's lock could
    interleave with the CLI's appends."""
    store = LabelStore(path)
    reader: LabelReader = store
    writer: LabelWriter = store
    return LabelAccess(reader, writer)


@dataclass(frozen=True, slots=True)
class LabelsRead:
    labels: list[Label]
    problem: str | None = None  # set: the reader refused the file; do not write this rerun


def read_labels(access: LabelAccess, path: Path) -> LabelsRead:
    """The owner's newest labels, degrading like every other reader here.

    The store raises on ANY invalid line (a hand edit, a newer schema). Then the good lines are
    still shown (read-only, line by line) but the caller must not write: a writer that cannot
    read which items are labelled could label one twice."""
    try:
        return LabelsRead(access.reader.latest())
    except (OSError, ValueError) as exc:
        return LabelsRead(FileLabelReader(path).latest(), problem(path, exc))
