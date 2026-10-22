"""EV-UI test support (no tests here): golden-fixture copies, synthetic runs and small fakes.

Everything is built from the committed D0 demo set (`evals/demo`) and its golden fixtures, or is
synthetic. Nothing here may ever read `evals/private`.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from streamlit.testing.v1 import AppTest
from ui_support import APP, FakeBackend, texts

from ragforge_client.evals.datasets import LabelStore
from ragforge_client.evals.schema import (
    AnswerOutput,
    Calibration,
    Criterion,
    DatasetRef,
    EvalKind,
    JudgeKey,
    JudgeResult,
    Label,
    LabelKind,
    RunFooter,
    RunHeader,
    RunRow,
    RunSummary,
    ShownRef,
    SnapshotRef,
    Split,
    Surface,
)
from ragforge_contracts.api import EvalChunk, PageKey
from ragforge_ui.eval_page import CHUNKS, LABELS, PATHS, VIEW
from ragforge_ui.eval_view.config import EvalPaths

REPO = Path(__file__).resolve().parents[4]
DEMO = REPO / "evals" / "demo"
FIXTURES = DEMO / "fixtures"
GOLDEN = ("generator", "interview", "retrieval", "verifier")
T0 = datetime(2026, 10, 21, 12, 0, tzinfo=UTC)
JUDGE = JudgeKey(
    provider="ollama",
    model_version="qwen3:8b-q4_K_M",
    decoding={"temperature": 0, "num_ctx": 8192},
    prompt="judge_qa@1",
)


# ------------------------------------------------------------------ golden copies
def golden_paths(tmp: Path) -> EvalPaths:
    """A private-style layout in `tmp` holding COPIES of the golden fixtures and D0 datasets."""
    runs, data = tmp / "runs", tmp / "data"
    runs.mkdir(parents=True)
    data.mkdir()
    for name in GOLDEN:
        shutil.copy(FIXTURES / f"run-{name}.jsonl", runs)
        shutil.copy(FIXTURES / f"run-{name}.summary.json", runs)
    for name in ("questions.toml", "verifier.json", "interview.json"):
        shutil.copy(DEMO / name, data)
    shutil.copy(FIXTURES / "labels.jsonl", data / "labels.jsonl")
    shutil.copy(FIXTURES / "judge-results.jsonl", data / "judge-cache.jsonl")
    return EvalPaths.under(runs, data)


def summary_of(name: str) -> RunSummary:
    return RunSummary.model_validate_json((FIXTURES / f"run-{name}.summary.json").read_bytes())


def read_lines(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def write_lines(path: Path, lines: Sequence[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(x) + "\n" for x in lines))


def copy_run(
    paths: EvalPaths, name: str, new: str, *, header: dict[str, Any] | None = None
) -> Path:
    """A copy of golden run `name` saved as run-<new>.jsonl with a new run id (and summary)."""
    lines = read_lines(paths.runs / f"run-{name}.jsonl")
    lines[0] = {**lines[0], "run_id": new, **(header or {})}
    path = paths.runs / f"run-{new}.jsonl"
    write_lines(path, lines)
    summary = json.loads((paths.runs / f"run-{name}.summary.json").read_text())
    (paths.runs / f"run-{new}.summary.json").write_text(json.dumps({**summary, "run_id": new}))
    return path


# ------------------------------------------------------------------ synthetic runs
def header(
    run_id: str,
    kind: EvalKind = EvalKind.generator,
    *,
    splits: Sequence[Split] = (Split.dev,),
    snapshot: SnapshotRef | None = None,
    started: datetime = T0,
) -> RunHeader:
    return RunHeader(
        run_id=run_id,
        eval_kind=kind,
        surface=Surface.chat,
        arm="synthetic",
        started_at=started,
        dataset=DatasetRef(name="synthetic", digest="d" * 64, splits=list(splits)),
        snapshot=snapshot,
        chat_model="fake-chat",
        judge=JUDGE,
    )


def answer_row(
    item_id: str,
    answer: str,
    *,
    status: str = "verified",
    abstained: bool = False,
    refs: Sequence[str] = (),
    shown: Sequence[ShownRef] = (),
    stale: bool = False,
) -> RunRow:
    return RunRow(
        item_id=item_id,
        input_digest="i" * 64,
        output=AnswerOutput(
            answer=answer,
            status=status,
            cited_pages=["file:a.md"],
            abstained=abstained,
        ),
        judge_refs=list(refs),
        shown=list(shown),
        stale=stale,
    )


def write_run(path: Path, head: RunHeader, rows: Sequence[RunRow]) -> Path:
    lines = [head.model_dump_json(), *(r.model_dump_json() for r in rows)]
    lines.append(RunFooter(finished_at=T0, rows=len(rows)).model_dump_json())
    path.write_text("\n".join(lines) + "\n")
    return path


def judged(item_digest: str, *, score: int = 2, reason: str = "fine") -> list[JudgeResult]:
    """One JudgeResult per E-G criterion for one item."""
    return [
        JudgeResult(
            cache_key=f"{item_digest}:{c.value}",
            item_digest=item_digest,
            criterion=c,
            score=score,
            reason=reason,
            judge=JUDGE,
            scored_at=T0,
        )
        for c in (Criterion.relevance, Criterion.completeness, Criterion.faithfulness)
    ]


def write_jsonl(path: Path, models: Sequence[Any]) -> None:
    path.write_text("".join(m.model_dump_json() + "\n" for m in models))


def questions_toml(path: Path, questions: Sequence[tuple[str, Split, str]]) -> None:
    """A minimal v2 question set: (id, split, question text), each answerable."""
    blocks = [
        f'[[questions]]\nid = "{qid}"\nsplit = "{split.value}"\nquestion = "{text}"\n'
        f'relevance = {{ "file:a.md" = 2 }}\n'
        for qid, split, text in questions
    ]
    path.write_text('schema_version = 2\nname = "synthetic"\n\n' + "\n".join(blocks))


def calibration(
    criterion: Criterion,
    *,
    trusted: bool,
    kappa: float = 0.7,
    at: datetime = T0,
    judge: JudgeKey = JUDGE,
    confusion: list[list[int]] | None = None,
) -> Calibration:
    return Calibration(
        criterion=criterion,
        judge=judge,
        label_batch_id="b1",
        n=50,
        kappa_linear=kappa,
        kappa_ci=(kappa - 0.2, min(1.0, kappa + 0.1)),
        raw_agreement=0.8,
        confusion=confusion or [[10, 2, 0], [1, 15, 2], [0, 3, 17]],
        leniency_p=0.5,
        trusted=trusted,
        computed_at=at,
    )


# ------------------------------------------------------------------ fakes
@dataclass
class FakeLabels:
    """LabelReader + LabelWriter in memory: records every append (newest per key wins)."""

    initial: list[Label] = field(default_factory=list)
    appended: list[Label] = field(default_factory=list)
    fail: BaseException | None = None

    def latest(self) -> list[Label]:
        newest: dict[tuple[str, LabelKind], Label] = {}
        for label in [*self.initial, *self.appended]:
            newest[(label.item_digest, label.kind)] = label
        return list(newest.values())

    def append(self, label: Label) -> None:
        if self.fail is not None:
            raise self.fail
        self.appended.append(label)


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


class RecordingChunks:
    """`ChunkSource` fake."""

    def __init__(self, chunks: Sequence[EvalChunk] = ()) -> None:
        self.held = list(chunks)
        self.calls: list[list[UUID]] = []

    def chunks(self, chunk_ids: Sequence[UUID]) -> list[EvalChunk]:
        self.calls.append(list(chunk_ids))
        wanted = set(chunk_ids)
        return [c for c in self.held if c.chunk_id in wanted]


def chunk(text: str, *, chunk_id: UUID | None = None, digest: str | None = None) -> EvalChunk:
    return EvalChunk(
        chunk_id=chunk_id or uuid4(),
        document_id=uuid4(),
        page_key=PageKey(source_kind="file", key="kafka-basics.md"),
        heading_path=["Consumers"],
        ordinal=0,
        display_text=text,
        display_sha256=digest or sha(text),
    )


def shown_ref(label: int, c: EvalChunk, *, digest: str | None = None) -> ShownRef:
    return ShownRef(
        label=label, chunk_id=str(c.chunk_id), display_sha256=digest or c.display_sha256
    )


# ------------------------------------------------------------------ AppTest
def eval_app(
    paths: EvalPaths,
    view: str | None = "Runs",
    *,
    labels: FakeLabels | LabelStore | None = None,
    chunks: RecordingChunks | None = None,
) -> AppTest:
    """The real app, on its Eval page, with injected paths, label store and chunk source."""
    at = AppTest.from_file(str(APP), default_timeout=30)
    at.session_state["rf_backend"] = FakeBackend()
    at.session_state[PATHS] = paths
    at.session_state[LABELS] = labels if labels is not None else FakeLabels()
    if chunks is not None:
        at.session_state[CHUNKS] = chunks
    at.run()
    at = at.sidebar.radio[0].set_value("Eval").run()
    assert not at.exception, at.exception
    return at if view is None else at.radio(key=VIEW).set_value(view).run()


def page_text(at: AppTest) -> str:
    """What the owner reads: the page's text with Markdown backslash escapes removed."""
    return "\n".join(texts(at.main)).replace("\\", "")


def frames(at: AppTest) -> list[Any]:
    """The page's dataframes (pandas objects; typed Any: pandas ships no stubs here)."""
    return [df.value for df in at.dataframe]


def cells(at: AppTest) -> list[dict[str, str]]:
    """Every dataframe row on the page as {column: text}."""
    out: list[dict[str, str]] = []
    for frame in frames(at):
        out += [{str(k): str(v) for k, v in row.items()} for row in frame.to_dict("records")]
    return out


def everything(at: AppTest) -> str:
    """Page text plus every table cell and widget label/option: for "never shown" checks."""
    parts = [page_text(at)]
    parts += [" ".join(row.values()) for row in cells(at)]
    for widget in [*at.radio, *at.selectbox]:
        parts.append(str(widget.label))
        parts += [str(o) for o in widget.options]
    parts += [str(t.label) for t in at.toggle]
    return "\n".join(parts).replace("\\", "")
