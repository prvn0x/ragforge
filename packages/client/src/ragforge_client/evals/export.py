"""`ragforge-eval2 results`: the generated results tables of `docs/RESULTS.md` (Phase 5 R2-3, R3-2).

Nothing in RESULTS.md is typed by hand. A committed manifest (`evals/results.toml`) names every
table, the exact run ids behind it and the provider and model that produced them. This module
renders those runs into marker blocks in the doc:

    <!-- generated:<table> run_ids=<id,...> digests=<header_digest,...> -->
    ...rendered markdown...
    <!-- /generated -->

`--write` replaces only the block contents; `--check` re-renders and exits 1 on any difference,
so a hand edit or a stale number cannot survive `make check`.

Privacy is the reason this module is strict rather than convenient (the private runs hold the
owner's notes). It reads by allow-list, never by deny-list:

- from `<run>.summary.json` only `name, value, ci_low, ci_high, n, method, trusted`, and `note`
  only when every `; `-separated part of it matches a fixed template from metrics.py (a free-form
  note could carry a page title, so an unknown note is dropped, not escaped);
- a metric name outside `METRIC_NAME` is refused (the export fails without echoing it);
- from the run header (line 1 of `<run>.jsonl`) only `chat_model, verifier_model, dataset.name,
  dataset.splits, started_at`. Never `arm` (free text: the manifest names it), `snapshot`
  (collection ids) or any row.

Fresh clones have no `.jsonl` (they stay git-ignored), so `--write` also writes a committed
sidecar `<run>.header.json` with only those header fields; `--check` reads the summary, the
manifest and the sidecar, and cross-checks the header against the sidecar and the manifest only
when the `.jsonl` is present. Private tables are skipped, with a printed note, on a machine
without `evals/private/runs`.
"""

from __future__ import annotations

import difflib
import json
import os
import re
import tempfile
import tomllib
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Literal, TextIO

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from ragforge_client.evals.runfile import RunFileError
from ragforge_client.evals.schema import SCHEMA_VERSION, Method, RunHeader, RunSummary, Split

MANIFEST_VERSION: Literal[1] = 1
DEFAULT_DOC = Path("docs/RESULTS.md")
DEFAULT_MANIFEST = Path("evals/results.toml")
PLACEHOLDER = "not generated yet: run `ragforge-eval2 results --write`"
PENDING = "pending owner labels"

# R2-3: a metric name is dotted lower-case words; `@` for k and base rates ("hit@5", "@0.05").
METRIC_NAME = re.compile(r"^[a-z_]+(\.[a-z0-9_@]+)+$")
# runners.py: f"{kind}-{arm}-{%Y%m%dT%H%M%SZ}", the arm limited to RunHeader.arm's characters.
RUN_ID = re.compile(
    r"^(retrieval|generator|verifier|interview|application|latency)"
    r"-[A-Za-z0-9._:-]{1,64}-\d{8}T\d{6}Z$"
)
# Free text that lands in a markdown table cell or a caption: no cell breaks, no HTML comments.
_SAFE_TEXT = r"^[^|\n\r<>`*\[\]\\]{1,120}$"
_HEX64 = re.compile(r"^[0-9a-f]{64}$")

# ------------------------------------------------------------------ note allow-list
# Every note template metrics.py can emit (one regex per `; `-joined part). Placeholders accept
# numbers only, so no template can carry text from a note or a question.
_NUM = r"-?\d+(?:\.\d+)?(?:e[+-]?\d+)?"
_NUM_OR_NA = rf"(?:{_NUM}|n/a)"
NOTE_TEMPLATES: tuple[re.Pattern[str], ...] = tuple(
    re.compile(rf"^{p}$")
    for p in (
        r"bootstrap over questions",
        r"bootstrap degenerate \(every resample equal\): no interval",
        r"exact CP: bootstrap degenerate",
        r"a bound widened to the exact Clopper-Pearson bound",
        r"no tau in header",
        r"on \d+ clean-verified sources",
        r"P\(wrong \| verified\) at base rate \d+%",
        rf"conservative plug-in bounds \[{_NUM}, {_NUM}\] from the recall and false-reject CI "
        r"bounds \(not a joint CI\)",
        r"no bounds: a component has no interval",
        r"tier pairs at the seed level",
        r"a tie counts as wrong",
        r"cells graded twice",
        r"no complete level block",
        rf"exact permutation p={_NUM} \(one-sided, staff -> sde_1 increasing\)",
        r"P\(L\* >= L\) under within-block exchangeability",
        r"pooled paired sign test: \+\d+ -\d+ =\d+",
        r"no judge key in the run header",
        rf"calibration n=\d+, kappa={_NUM_OR_NA}(?:, uncalibrated \(< \d+ labels\)|, not trusted)?",
        r"\d+ judge refs from another judge key excluded",
        r"\d+ invalid judge replies excluded",
        r"\d+ judge refs without a result",
        r"iid pairs bootstrap over labelled items",
        rf"leniency p={_NUM_OR_NA}",
        r"trust needs n >= \d+",
        r"no CI: bootstrap degenerate or too few defined resamples",
        r"\d+ judge-flagged answers adjudicated, \d+ confirmed",
        r"a lower bound \(the judge screen misses some\)",
        r"first questions of the pass, excluded above",
    )
)


def allowed_note(note: str) -> str:
    """The note unchanged when every part matches a template, else "" (dropped whole: keeping
    the matching parts of a note that also holds free text would still leak its shape)."""
    if not note:
        return ""
    parts = note.split("; ")
    if all(any(t.fullmatch(part) for t in NOTE_TEMPLATES) for part in parts):
        return note
    return ""


class ExportError(ValueError):
    """A clean, user-facing failure (exit 2). Never carries note text or a refused name."""


# ------------------------------------------------------------------ manifest
class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


SafeText = Annotated[str, Field(pattern=_SAFE_TEXT)]


class MetricSpec(_Strict):
    """One row: a metric of the table's runs, or (pending) a row the owner's labels will fill."""

    name: str
    label: SafeText
    format: Literal["ratio", "seconds", "number"] = "ratio"
    pending: bool = False  # no run backs it yet: renders "pending owner labels"

    @field_validator("name")
    @classmethod
    def _metric_name(cls, value: str) -> str:
        if not METRIC_NAME.fullmatch(value):
            raise ValueError("metric name outside ^[a-z_]+(\\.[a-z0-9_@]+)+$ (refused)")
        return value


class TableSpec(_Strict):
    id: Annotated[str, Field(pattern=r"^[a-z0-9-]{1,40}$")]
    dataset: Literal["demo", "private"]
    title: SafeText
    provider: SafeText
    model: SafeText
    verifier_model: SafeText | None = None  # when it differs from `model`
    # Which header fields `model` is checked against: the chat (and verifier) model, or, for a
    # retrieval table, "<embed_model>, <rerank_model>" (a retrieval run records the server's
    # chat model too, but no chat model produced its numbers).
    models_from: Literal["chat", "retrieval"] = "chat"
    run_ids: list[str]
    metrics: Annotated[list[MetricSpec], Field(min_length=1)]

    @field_validator("run_ids")
    @classmethod
    def _run_ids(cls, value: list[str]) -> list[str]:
        for run_id in value:
            if not RUN_ID.fullmatch(run_id):
                raise ValueError(f"not a run id: {run_id!r}")
        if len(set(value)) != len(value):
            raise ValueError("a run id is listed twice")
        return value

    @model_validator(mode="after")
    def _backed(self) -> TableSpec:
        live = [m for m in self.metrics if not m.pending]
        if live and not self.run_ids:
            raise ValueError(f"{self.id}: metrics without run_ids (only pending rows may)")
        if not live and self.run_ids:
            raise ValueError(f"{self.id}: run_ids on a table of pending rows only")
        return self

    @property
    def expected_verifier_model(self) -> str:
        return self.verifier_model or self.model


class Manifest(_Strict):
    schema_version: Literal[1]
    table: Annotated[list[TableSpec], Field(min_length=1)]

    @model_validator(mode="after")
    def _unique(self) -> Manifest:
        ids = [t.id for t in self.table]
        if len(set(ids)) != len(ids):
            raise ValueError("a table id is listed twice")
        return self


def load_manifest(path: Path) -> Manifest:
    try:
        return Manifest.model_validate(tomllib.loads(path.read_text(encoding="utf-8")))
    except FileNotFoundError:
        raise ExportError(f"{path}: manifest not found") from None
    except tomllib.TOMLDecodeError as exc:
        raise ExportError(f"{path}: {exc}") from None
    except ValidationError as exc:
        first = exc.errors()[0]
        where = ".".join(str(p) for p in first["loc"])
        raise ExportError(f"{path}: {where}: {first['msg']}") from None


# ------------------------------------------------------------------ the header sidecar
class DatasetFields(_Strict):
    name: Annotated[str, Field(pattern=r"^[a-z0-9_-]{1,40}$")]
    splits: list[Split]


class HeaderFields(_Strict):
    """Everything this module may know about a run header (R2-3): the sidecar's whole schema."""

    chat_model: str | None
    verifier_model: str | None
    embed_model: str | None = None  # header.retrieval: public model names, nothing else
    rerank_model: str | None = None
    dataset: DatasetFields
    started_at: datetime


def header_fields(header: RunHeader) -> HeaderFields:
    """Project a parsed header onto the allow-list; nothing else is ever copied."""
    return HeaderFields(
        chat_model=header.chat_model,
        verifier_model=header.verifier_model,
        embed_model=header.retrieval.embed_model if header.retrieval else None,
        rerank_model=header.retrieval.rerank_model if header.retrieval else None,
        dataset=DatasetFields(name=header.dataset.name, splits=list(header.dataset.splits)),
        started_at=header.started_at,
    )


def read_header_line(path: Path) -> HeaderFields:
    """Line 1 of a run file only (the rows hold answer text and are never read here). The same
    version and type checks as `runfile.read_run`, without loading the rows."""
    with path.open(encoding="utf-8", newline="") as fh:
        first = fh.readline()
    try:
        data = json.loads(first)
    except json.JSONDecodeError:
        raise ExportError(f"{path.name}: line 1 is not JSON") from None
    if data.get("schema_version") != SCHEMA_VERSION or data.get("type") != "header":
        raise ExportError(f"{path.name}: not a v{SCHEMA_VERSION} run file")
    try:
        return header_fields(RunHeader.model_validate(data))
    except (ValidationError, RunFileError):
        raise ExportError(f"{path.name}: invalid run header") from None


def sidecar_text(fields: HeaderFields) -> str:
    """Sorted keys, 2-space indent, trailing newline: byte-identical for identical fields."""
    return json.dumps(fields.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"


def read_sidecar(path: Path) -> HeaderFields:
    try:
        return HeaderFields.model_validate_json(path.read_text(encoding="utf-8"))
    except ValidationError:
        raise ExportError(
            f"{path.name}: not a header sidecar (unknown or missing fields)"
        ) from None


def write_public(path: Path, text: str) -> None:
    """Atomic replace, mode 0644: these files are committed (unlike run files, which are 0600)."""
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
        os.chmod(tmp, 0o644)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


# ------------------------------------------------------------------ summaries
@dataclass(frozen=True)
class Metric:
    """The allow-listed projection of a `MetricValue`."""

    name: str
    value: float | None
    ci_low: float | None
    ci_high: float | None
    n: int
    method: Method
    trusted: bool | None
    note: str


@dataclass(frozen=True)
class RunData:
    run_id: str
    header_digest: str
    header: HeaderFields
    metrics: Mapping[str, Metric]


def read_summary(path: Path, run_id: str, warn: Callable[[str], None]) -> tuple[str, list[Metric]]:
    """(header_digest, metrics) of one summary; a refused name fails the export, a note outside
    the templates is dropped (and said, by metric name only)."""
    try:
        summary = RunSummary.model_validate_json(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ExportError(f"missing run: {path.name}") from None
    except ValidationError:
        raise ExportError(f"{path.name}: not a v{SCHEMA_VERSION} run summary") from None
    if summary.run_id != run_id:
        raise ExportError(f"{path.name}: holds a different run than its file name")
    if not _HEX64.fullmatch(summary.header_digest):
        raise ExportError(f"{path.name}: header_digest is not a SHA-256 hex digest")
    metrics: list[Metric] = []
    for index, mv in enumerate(summary.metrics):
        if not METRIC_NAME.fullmatch(mv.name):
            raise ExportError(f"{path.name}: metric #{index + 1} has a refused name")
        note = allowed_note(mv.note)
        if mv.note and not note:
            warn(f"{run_id}: note dropped on {mv.name} (not an allowed template)")
        metrics.append(
            Metric(mv.name, mv.value, mv.ci_low, mv.ci_high, mv.n, mv.method, mv.trusted, note)
        )
    return summary.header_digest, metrics


def check_models(table: TableSpec, run_id: str, header: HeaderFields) -> None:
    """R2-3: a header model that disagrees with the manifest fails; a null one defers to it."""
    if table.models_from == "retrieval":
        if header.embed_model is not None and (
            f"{header.embed_model}, {header.rerank_model}" != table.model
        ):
            raise ExportError(f"{run_id}: header retrieval models differ from the manifest's model")
        return
    if header.chat_model is not None and header.chat_model != table.model:
        raise ExportError(f"{run_id}: header chat_model differs from the manifest's model")
    if header.verifier_model is not None and header.verifier_model != table.expected_verifier_model:
        raise ExportError(f"{run_id}: header verifier_model differs from the manifest")


def load_run(
    runs_dir: Path, table: TableSpec, run_id: str, *, write: bool, warn: Callable[[str], None]
) -> RunData:
    digest_, metrics = read_summary(runs_dir / f"{run_id}.summary.json", run_id, warn)
    run_file = runs_dir / f"{run_id}.jsonl"
    sidecar = runs_dir / f"{run_id}.header.json"
    if run_file.exists():
        header = read_header_line(run_file)
        check_models(table, run_id, header)
        if write:
            if not sidecar.exists() or sidecar.read_text(encoding="utf-8") != sidecar_text(header):
                write_public(sidecar, sidecar_text(header))
        elif not sidecar.exists() or read_sidecar(sidecar) != header:
            raise ExportError(f"{sidecar.name}: out of date with its run file; run --write")
    elif sidecar.exists():
        header = read_sidecar(sidecar)
    else:
        raise ExportError(f"missing run: neither {run_file.name} nor {sidecar.name}")
    by_name = {m.name: m for m in metrics}
    if len(by_name) != len(metrics):
        raise ExportError(f"{run_id}: a metric name appears twice in its summary")
    return RunData(run_id, digest_, header, by_name)


# ------------------------------------------------------------------ rendering
METHOD_LABELS: Mapping[Method, str] = {
    Method.point: "point (no CI)",
    Method.wilson: "Wilson",
    Method.clopper_pearson: "exact Clopper-Pearson",
    Method.cluster_bootstrap: "cluster bootstrap",
    Method.bootstrap: "bootstrap",
    Method.exact: "exact",
}


def fmt_value(value: float | None, fmt: str) -> str:
    if value is None:
        return "n/a"
    if fmt == "seconds":
        return f"{value:.2f} s"
    if fmt == "number":
        return f"{value:.4g}"
    return f"{value:.3f}"


def fmt_ci(m: Metric, fmt: str) -> str:
    if m.ci_low is None or m.ci_high is None:
        return "-"
    return f"[{fmt_value(m.ci_low, fmt)}, {fmt_value(m.ci_high, fmt)}]"


def cell(text: str) -> str:
    """Escape the two characters that break a markdown table cell or start emphasis."""
    return text.replace("\\", "\\\\").replace("|", "\\|").replace("*", "\\*")


def find_metric(table: TableSpec, runs: Sequence[RunData], name: str) -> Metric:
    """A row's metric must come from exactly one of the table's runs (no silent first-wins)."""
    found = [r.metrics[name] for r in runs if name in r.metrics]
    if not found:
        raise ExportError(f"{table.id}: no listed run has the metric {name}")
    if len(found) > 1:
        raise ExportError(f"{table.id}: {name} is in several listed runs; split the table")
    return found[0]


def _no_header_model(table: TableSpec, header: HeaderFields) -> bool:
    if table.models_from == "retrieval":
        return header.embed_model is None
    return header.chat_model is None and header.verifier_model is None


def caption(table: TableSpec, runs: Sequence[RunData], rows: Sequence[Metric]) -> str:
    """Provider, model, dataset, n and date (R2-3), all from the manifest, sidecar and rows."""
    model = table.model
    if table.verifier_model:
        model += f", verifier {table.verifier_model}"
    if any(_no_header_model(table, r.header) for r in runs):
        model += " (model from the manifest)"
    names = sorted({r.header.dataset.name for r in runs})
    splits = sorted({s.value for r in runs for s in r.header.dataset.splits})
    dates = sorted({r.header.started_at.astimezone(UTC).date().isoformat() for r in runs})
    ns = [m.n for m in rows]
    n_text = f"n = {ns[0]}" if min(ns) == max(ns) else f"n = {min(ns)} to {max(ns)} (per row)"
    date_text = dates[0] if len(dates) == 1 else f"{dates[0]} to {dates[-1]}"
    return (
        f"**{table.title}.** Provider: {table.provider}. Model: {model}. "
        f"Dataset: {table.dataset} (set `{', '.join(names)}`, split {', '.join(splits)}). "
        f"{n_text}. Run date (UTC): {date_text}."
    )


TABLE_HEAD = "| Metric | Value | 95% CI | n | Method | Note |\n|---|---|---|---|---|---|\n"


def render_table(table: TableSpec, runs: Sequence[RunData]) -> str:
    """The block body. Rows follow the manifest's order; every number is formatted to a fixed
    precision, so identical inputs give byte-identical output."""
    if not runs:
        lines = [f"**{table.title}.** Not run yet: these rows need the owner's labels.", ""]
        lines.append("| Metric | Value |\n|---|---|")
        lines.extend(f"| {cell(spec.label)} | {PENDING} |" for spec in table.metrics)
        return "\n".join(lines) + "\n"
    pairs = [(spec, find_metric(table, runs, spec.name)) for spec in table.metrics]
    out = [caption(table, runs, [m for _, m in pairs]), "", TABLE_HEAD.rstrip("\n")]
    for spec, m in pairs:
        if spec.pending or m.trusted is False:
            value, ci = PENDING, "-"
        else:
            value, ci = fmt_value(m.value, spec.format), fmt_ci(m, spec.format)
        method = METHOD_LABELS[m.method]
        out.append(f"| {cell(spec.label)} | {value} | {ci} | {m.n} | {method} | {cell(m.note)} |")
    return "\n".join(out) + "\n"


def opening_marker(table: TableSpec, runs: Sequence[RunData]) -> str:
    run_ids = ",".join(table.run_ids) or "-"
    digests = ",".join(r.header_digest for r in runs) or "-"
    return f"<!-- generated:{table.id} run_ids={run_ids} digests={digests} -->"


# ------------------------------------------------------------------ the doc
BLOCK = re.compile(
    r"^<!-- generated:(?P<id>[a-z0-9-]+) run_ids=(?P<runs>\S+) digests=(?P<digests>\S+) -->\n"
    r"(?P<body>.*?)^<!-- /generated -->$",
    re.MULTILINE | re.DOTALL,
)
_OPEN_LINE = re.compile(r"^<!-- generated:", re.MULTILINE)
_CLOSE_LINE = re.compile(r"^<!-- /generated -->$", re.MULTILINE)


def find_blocks(doc: str, manifest: Manifest) -> dict[str, re.Match[str]]:
    """Every block, by table id. A malformed marker, a block the manifest does not list, a
    duplicate, or a manifest table with no block all fail: nothing is matched loosely."""
    blocks: dict[str, re.Match[str]] = {}
    matches = list(BLOCK.finditer(doc))
    if len(matches) != len(_OPEN_LINE.findall(doc)) or len(matches) != len(
        _CLOSE_LINE.findall(doc)
    ):
        raise ExportError("a generated marker is malformed or unclosed")
    known = {t.id for t in manifest.table}
    for match in matches:
        table_id = match["id"]
        if table_id not in known:
            raise ExportError(f"block {table_id} is not listed in the manifest")
        if table_id in blocks:
            raise ExportError(f"block {table_id} appears twice")
        blocks[table_id] = match
    missing = sorted(known - set(blocks))
    if missing:
        raise ExportError(f"no block in the doc for: {', '.join(missing)}")
    return blocks


@dataclass(frozen=True)
class Outcome:
    doc: str  # the doc with every rendered block replaced
    skipped: list[str]  # table ids skipped (private data absent)


def render_doc(
    doc: str,
    manifest: Manifest,
    root: Path,
    *,
    write: bool,
    only: Iterable[str] | None,
    warn: Callable[[str], None],
) -> Outcome:
    blocks = find_blocks(doc, manifest)
    selected = set(only) if only is not None else {t.id for t in manifest.table}
    unknown = sorted(selected - set(blocks))
    if unknown:
        raise ExportError(f"unknown table: {', '.join(unknown)}")
    rendered: dict[str, str] = {}
    skipped: list[str] = []
    for table in manifest.table:
        if table.id not in selected:
            continue
        runs_dir = root / table.dataset / "runs"
        if table.run_ids and table.dataset == "private" and not runs_dir.is_dir():
            skipped.append(table.id)
            continue
        runs = [load_run(runs_dir, table, r, write=write, warn=warn) for r in table.run_ids]
        body = render_table(table, runs)
        rendered[table.id] = f"{opening_marker(table, runs)}\n{body}<!-- /generated -->"

    def replace(match: re.Match[str]) -> str:
        return rendered.get(match["id"], match[0])

    return Outcome(BLOCK.sub(replace, doc), skipped)


def results(
    *,
    mode: Literal["check", "write"],
    doc_path: Path = DEFAULT_DOC,
    manifest_path: Path = DEFAULT_MANIFEST,
    root: Path = Path("evals"),
    only: Sequence[str] | None = None,
    out: TextIO,
    err: TextIO,
) -> int:
    """Exit 0 when the doc matches (check) or was written (write); 1 when check finds a diff."""
    manifest = load_manifest(manifest_path)
    try:
        doc = doc_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise ExportError(f"{doc_path}: not found") from None

    def warn(message: str) -> None:
        print(f"note: {message}", file=err)

    outcome = render_doc(doc, manifest, root, write=mode == "write", only=only, warn=warn)
    for table_id in outcome.skipped:
        print(
            f"skipped {table_id}: {root / 'private' / 'runs'} not found "
            "(private tables are generated on the owner's machine)",
            file=out,
        )
    if mode == "write":
        if outcome.doc != doc:
            write_public(doc_path, outcome.doc)
            print(f"wrote {doc_path}", file=out)
        else:
            print(f"{doc_path}: already up to date", file=out)
        return 0
    if outcome.doc == doc:
        print(f"{doc_path}: generated tables are up to date", file=out)
        return 0
    diff = difflib.unified_diff(
        doc.splitlines(keepends=True),
        outcome.doc.splitlines(keepends=True),
        fromfile=f"{doc_path} (on disk)",
        tofile=f"{doc_path} (re-rendered)",
    )
    err.writelines(diff)
    print(f"{doc_path}: generated tables differ; run `ragforge-eval2 results --write`", file=err)
    return 1
