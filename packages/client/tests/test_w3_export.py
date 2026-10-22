"""W3-RESULTS: the results exporter (`ragforge-eval2 results`, Phase 5 R2-3 / R3-2).

The privacy tests plant the kind of text a private run can hold (a page title in a note, an odd
metric name, a header with collection ids and a free-text arm) and prove none of it reaches the
doc, the sidecar or the terminal. Everything here is synthetic; the repo tests at the end run
`--check` on the committed demo tables.
"""

from __future__ import annotations

import io
import json
import stat
from datetime import UTC, datetime
from pathlib import Path

import pytest

from ragforge_client.evals import cli
from ragforge_client.evals import export as ex
from ragforge_client.evals.schema import (
    DatasetRef,
    EvalKind,
    Method,
    MetricValue,
    RetrievalConfig,
    RunHeader,
    RunSummary,
    SnapshotRef,
    Split,
    Surface,
)

RUN = "generator-A-20260101T000000Z"
DIGEST = "ab" * 32
TITLE = "Zebra Quarterly Roadmap Secrets"  # a planted "page title"
COLLECTION = "0b9f6c1e-1111-4222-8333-944455556666"
ARM_TEXT = "owner-free-text-arm"

MANIFEST = """
schema_version = 1

[[table]]
id = "demo-generator"
dataset = "demo"
title = "Answers, demo"
provider = "Google Gemini"
model = "model-x"
run_ids = ["{run}"]
metrics = [
  {{ name = "generator.correct", label = "correct" }},
  {{ name = "generator.judge.relevance", label = "judge relevance", format = "number" }},
]

[[table]]
id = "private-generator"
dataset = "private"
title = "Answers, private"
provider = "Google Gemini"
model = "model-x"
run_ids = ["generator-A-20260102T000000Z"]
metrics = [{{ name = "generator.correct", label = "correct" }}]
"""

DOC = """# Results

Intro text that is not generated.

<!-- generated:demo-generator run_ids=- digests=- -->
{placeholder}
<!-- /generated -->

<!-- generated:private-generator run_ids=generator-A-20260102T000000Z digests=- -->
{placeholder}
<!-- /generated -->
""".replace("{placeholder}", ex.PLACEHOLDER)


def metric(name: str, value: float | None = 0.9, **kw: object) -> MetricValue:
    base: dict[str, object] = {
        "name": name,
        "value": value,
        "ci_low": 0.7,
        "ci_high": 0.97,
        "n": 20,
        "method": Method.wilson,
    }
    base.update(kw)
    return MetricValue.model_validate(base)


def default_metrics() -> list[MetricValue]:
    return [
        metric("generator.correct", note=f"bootstrap over questions; {TITLE}"),
        metric(
            "generator.judge.relevance",
            1.5,
            trusted=False,
            method=Method.point,
            ci_low=None,
            ci_high=None,
            note="calibration n=3, kappa=n/a, uncalibrated (< 50 labels)",
        ),
    ]


def write_summary(runs: Path, run_id: str, metrics: list[MetricValue]) -> None:
    summary = RunSummary(
        run_id=run_id, eval_kind=EvalKind.generator, header_digest=DIGEST, metrics=metrics
    )
    (runs / f"{run_id}.summary.json").write_text(summary.model_dump_json(), encoding="utf-8")


def write_runfile(runs: Path, run_id: str, *, chat_model: str | None = "model-x") -> None:
    header = RunHeader(
        run_id=run_id,
        eval_kind=EvalKind.generator,
        surface=Surface.chat,
        arm=ARM_TEXT,
        started_at=datetime(2026, 1, 1, 0, 0, tzinfo=UTC),
        dataset=DatasetRef(name="demo", digest="cd" * 32, splits=[Split.dev]),
        snapshot=SnapshotRef(collection_id=COLLECTION, content_digest="e" * 64, index_digest="f"),
        chat_model=chat_model,
        prompt_versions={"chat": f"about {TITLE}"},
    )
    row = json.dumps({"type": "row", "item_id": "q1", "answer": TITLE})
    text = header.model_dump_json() + "\n" + row + "\n"
    (runs / f"{run_id}.jsonl").write_text(text, encoding="utf-8")


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    runs = tmp_path / "evals" / "demo" / "runs"
    runs.mkdir(parents=True)
    write_summary(runs, RUN, default_metrics())
    write_runfile(runs, RUN)
    (tmp_path / "evals" / "results.toml").write_text(MANIFEST.format(run=RUN), encoding="utf-8")
    (tmp_path / "RESULTS.md").write_text(DOC, encoding="utf-8")
    return tmp_path


def run_export(repo: Path, mode: str, *tables: str) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    code = ex.results(
        mode="write" if mode == "write" else "check",
        doc_path=repo / "RESULTS.md",
        manifest_path=repo / "evals" / "results.toml",
        root=repo / "evals",
        only=list(tables) or None,
        out=out,
        err=err,
    )
    return code, out.getvalue(), err.getvalue()


def everything_written(repo: Path) -> str:
    """The doc plus every sidecar: all the exporter publishes."""
    parts = [(repo / "RESULTS.md").read_text(encoding="utf-8")]
    parts += [p.read_text(encoding="utf-8") for p in repo.glob("evals/*/runs/*.header.json")]
    return "\n".join(parts)


# ------------------------------------------------------------------ privacy filters
def test_note_with_a_page_title_is_dropped_whole(repo: Path) -> None:
    code, out, err = run_export(repo, "write")
    assert code == 0
    written = everything_written(repo)
    assert TITLE not in written + out + err
    # the allowed half of the note is dropped too: no partial notes
    row = next(ln for ln in written.splitlines() if ln.startswith("| correct |"))
    assert row.endswith("| Wilson |  |")
    assert "note dropped on generator.correct" in err


def test_allowed_compound_notes_are_kept() -> None:
    for note in (
        "bootstrap over questions; bootstrap degenerate (every resample equal): no interval",
        "on 7 clean-verified sources; a bound widened to the exact Clopper-Pearson bound",
        "P(wrong | verified) at base rate 5%; conservative plug-in bounds [0.000, 0.044] from "
        "the recall and false-reject CI bounds (not a joint CI)",
        "exact permutation p=0.06 (one-sided, staff -> sde_1 increasing)",
        "calibration n=12, kappa=0.512, not trusted; 2 invalid judge replies excluded",
        "iid pairs bootstrap over labelled items; leniency p=n/a; trust needs n >= 50",
    ):
        assert ex.allowed_note(note) == note


@pytest.mark.parametrize(
    "note",
    [
        TITLE,
        "on Zebra clean-verified sources",  # a placeholder that is not a number
        "bootstrap over questions;" + TITLE,  # wrong separator: one unknown part
        "bootstrap over questions " + TITLE,  # a template prefix is not enough
        "exact permutation p=0.06 (one-sided, staff -> sde_1 increasing) " + TITLE,
    ],
)
def test_notes_outside_the_templates_are_dropped(note: str) -> None:
    assert ex.allowed_note(note) == ""


def test_every_note_in_the_committed_demo_summaries_is_allowed() -> None:
    """The templates track metrics.py: a demo note that no longer matches means a template
    drifted (the real notes would silently disappear from RESULTS.md)."""
    runs = Path(__file__).resolve().parents[3] / "evals" / "demo" / "runs"
    notes = [
        m.note
        for path in sorted(runs.glob("*.summary.json"))
        for m in RunSummary.model_validate_json(path.read_text(encoding="utf-8")).metrics
        if m.note
    ]
    assert notes
    assert [n for n in notes if ex.allowed_note(n) != n] == []


def test_odd_metric_name_in_a_summary_is_refused_without_echoing_it(repo: Path) -> None:
    runs = repo / "evals" / "demo" / "runs"
    odd = MetricValue(name=f"generator.{TITLE}", value=1.0, n=1, method=Method.point)
    write_summary(runs, RUN, [*default_metrics(), odd])
    with pytest.raises(ex.ExportError, match="refused name") as info:
        run_export(repo, "write")
    assert TITLE not in str(info.value)
    assert ex.PLACEHOLDER in (repo / "RESULTS.md").read_text(encoding="utf-8")


@pytest.mark.parametrize("name", ["Generator.correct", "generator", "generator.co rrect", "a..b"])
def test_odd_metric_name_in_the_manifest_is_refused(repo: Path, name: str) -> None:
    manifest = repo / "evals" / "results.toml"
    text = manifest.read_text(encoding="utf-8").replace('"generator.correct"', f'"{name}"', 1)
    manifest.write_text(text, encoding="utf-8")
    with pytest.raises(ex.ExportError, match="refused"):
        run_export(repo, "check")


def test_header_sidecar_holds_only_the_allow_listed_fields(repo: Path) -> None:
    assert run_export(repo, "write")[0] == 0
    sidecar = repo / "evals" / "demo" / "runs" / f"{RUN}.header.json"
    data = json.loads(sidecar.read_text(encoding="utf-8"))
    assert data == {
        "chat_model": "model-x",
        "verifier_model": None,
        "embed_model": None,
        "rerank_model": None,
        "dataset": {"name": "demo", "splits": ["dev"]},
        "started_at": "2026-01-01T00:00:00Z",
    }
    assert stat.S_IMODE(sidecar.stat().st_mode) == 0o644
    written = everything_written(repo)
    for leaked in (COLLECTION, ARM_TEXT, TITLE, "e" * 64, "cd" * 32, "prompt_versions"):
        assert leaked not in written


def test_a_sidecar_with_an_extra_field_is_refused(repo: Path) -> None:
    run_export(repo, "write")
    runs = repo / "evals" / "demo" / "runs"
    (runs / f"{RUN}.jsonl").unlink()
    sidecar = runs / f"{RUN}.header.json"
    data = json.loads(sidecar.read_text(encoding="utf-8"))
    data["snapshot"] = {"collection_id": COLLECTION}
    sidecar.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ex.ExportError, match="not a header sidecar"):
        run_export(repo, "check")


# ------------------------------------------------------------------ manifest and runs agree
def test_missing_run_fails(repo: Path) -> None:
    (repo / "evals" / "demo" / "runs" / f"{RUN}.summary.json").unlink()
    with pytest.raises(ex.ExportError, match="missing run"):
        run_export(repo, "write")


def test_run_with_neither_jsonl_nor_sidecar_fails(repo: Path) -> None:
    (repo / "evals" / "demo" / "runs" / f"{RUN}.jsonl").unlink()
    with pytest.raises(ex.ExportError, match="missing run"):
        run_export(repo, "write")


def test_summary_of_another_run_fails(repo: Path) -> None:
    runs = repo / "evals" / "demo" / "runs"
    other = "generator-A-20260109T000000Z"
    write_summary(runs, other, default_metrics())
    (runs / f"{other}.summary.json").replace(runs / f"{RUN}.summary.json")
    with pytest.raises(ex.ExportError, match="different run"):
        run_export(repo, "write")


def test_a_newer_run_is_never_picked(repo: Path) -> None:
    runs = repo / "evals" / "demo" / "runs"
    newer = "generator-A-20991231T000000Z"
    write_summary(runs, newer, [metric("generator.correct", 0.123)])
    write_runfile(runs, newer)
    run_export(repo, "write")
    doc = (repo / "RESULTS.md").read_text(encoding="utf-8")
    assert "0.123" not in doc
    assert f"run_ids={RUN} " in doc


def test_doc_block_the_manifest_does_not_list_fails(repo: Path) -> None:
    doc = repo / "RESULTS.md"
    extra = "\n<!-- generated:demo-unlisted run_ids=- digests=- -->\nx\n<!-- /generated -->\n"
    doc.write_text(doc.read_text(encoding="utf-8") + extra, encoding="utf-8")
    with pytest.raises(ex.ExportError, match="not listed in the manifest"):
        run_export(repo, "check")


def test_manifest_table_without_a_block_fails(repo: Path) -> None:
    doc = repo / "RESULTS.md"
    text = doc.read_text(encoding="utf-8")
    start = text.index("<!-- generated:private-generator")
    doc.write_text(text[:start], encoding="utf-8")
    with pytest.raises(ex.ExportError, match="no block in the doc for: private-generator"):
        run_export(repo, "check")


def test_malformed_marker_fails(repo: Path) -> None:
    doc = repo / "RESULTS.md"
    text = doc.read_text(encoding="utf-8").replace("digests=- -->", "digests=-->", 1)
    doc.write_text(text, encoding="utf-8")
    with pytest.raises(ex.ExportError, match="malformed"):
        run_export(repo, "check")


def test_metric_missing_from_the_runs_fails(repo: Path) -> None:
    runs = repo / "evals" / "demo" / "runs"
    write_summary(runs, RUN, [metric("generator.correct")])
    with pytest.raises(ex.ExportError, match=r"no listed run has the metric generator\.judge"):
        run_export(repo, "write")


def test_header_model_that_disagrees_with_the_manifest_fails(repo: Path) -> None:
    write_runfile(repo / "evals" / "demo" / "runs", RUN, chat_model="another-model")
    with pytest.raises(ex.ExportError, match="differs from the manifest"):
        run_export(repo, "write")


def test_null_header_model_uses_the_manifest_and_says_so(repo: Path) -> None:
    write_runfile(repo / "evals" / "demo" / "runs", RUN, chat_model=None)
    run_export(repo, "write")
    doc = (repo / "RESULTS.md").read_text(encoding="utf-8")
    assert "Model: model-x (model from the manifest)." in doc


def test_caption_and_rows(repo: Path) -> None:
    run_export(repo, "write")
    doc = (repo / "RESULTS.md").read_text(encoding="utf-8")
    assert (
        "**Answers, demo.** Provider: Google Gemini. Model: model-x. Dataset: demo "
        "(set `demo`, split dev). n = 20. Run date (UTC): 2026-01-01."
    ) in doc
    assert "| correct | 0.900 | [0.700, 0.970] | 20 | Wilson |  |" in doc
    # an uncalibrated judge never shows its number
    assert (
        "| judge relevance | pending owner labels | - | 20 | point (no CI) | "
        "calibration n=3, kappa=n/a, uncalibrated (< 50 labels) |"
    ) in doc
    assert f"<!-- generated:demo-generator run_ids={RUN} digests={DIGEST} -->" in doc
    assert doc.startswith("# Results\n\nIntro text that is not generated.\n")


# ------------------------------------------------------------------ --check
def test_check_passes_on_fresh_output_and_fails_on_a_one_character_edit(repo: Path) -> None:
    assert run_export(repo, "write")[0] == 0
    code, out, _ = run_export(repo, "check")
    assert code == 0
    assert "up to date" in out
    doc = repo / "RESULTS.md"
    text = doc.read_text(encoding="utf-8")
    doc.write_text(text.replace("0.900", "0.901", 1), encoding="utf-8")
    code, _, err = run_export(repo, "check")
    assert code == 1
    assert "-| correct | 0.901" in err
    assert "+| correct | 0.900" in err
    assert doc.read_text(encoding="utf-8") != text  # --check never writes


def test_check_uses_the_sidecar_when_the_run_file_is_absent(repo: Path) -> None:
    """A fresh clone: summaries and sidecars are committed, run files are not."""
    run_export(repo, "write")
    (repo / "evals" / "demo" / "runs" / f"{RUN}.jsonl").unlink()
    assert run_export(repo, "check")[0] == 0


def test_check_fails_when_the_sidecar_is_stale(repo: Path) -> None:
    run_export(repo, "write")
    write_runfile(repo / "evals" / "demo" / "runs", RUN, chat_model=None)
    with pytest.raises(ex.ExportError, match="out of date"):
        run_export(repo, "check")


def test_private_tables_are_skipped_without_private_runs(repo: Path) -> None:
    code, out, _ = run_export(repo, "write")
    assert code == 0
    assert "skipped private-generator" in out
    doc = (repo / "RESULTS.md").read_text(encoding="utf-8")
    private = doc[doc.index("<!-- generated:private-generator") :]
    assert ex.PLACEHOLDER in private
    code, out, _ = run_export(repo, "check")
    assert code == 0
    assert "skipped private-generator" in out


def test_private_tables_are_checked_when_private_runs_exist(repo: Path) -> None:
    run_export(repo, "write")
    (repo / "evals" / "private" / "runs").mkdir(parents=True)
    with pytest.raises(ex.ExportError, match="missing run"):
        run_export(repo, "check")


def test_output_is_deterministic(repo: Path) -> None:
    run_export(repo, "write")
    first = everything_written(repo)
    code, out, _ = run_export(repo, "write")
    assert code == 0
    assert "already up to date" in out
    assert everything_written(repo) == first
    # a second tree from the same inputs renders the same bytes
    second = repo / "copy"
    (second / "evals" / "demo" / "runs").mkdir(parents=True)
    for p in (repo / "evals" / "demo" / "runs").glob(f"{RUN}.*"):
        if not p.name.endswith(".header.json"):
            (second / "evals" / "demo" / "runs" / p.name).write_bytes(p.read_bytes())
    (second / "evals" / "results.toml").write_bytes((repo / "evals" / "results.toml").read_bytes())
    (second / "RESULTS.md").write_text(DOC, encoding="utf-8")
    run_export(second, "write")
    assert everything_written(second) == first


def test_table_filter(repo: Path) -> None:
    run_export(repo, "write", "private-generator")
    assert ex.PLACEHOLDER in (repo / "RESULTS.md").read_text(encoding="utf-8").split("private")[0]
    with pytest.raises(ex.ExportError, match="unknown table"):
        run_export(repo, "check", "nope")


def test_pending_only_table_needs_no_runs() -> None:
    table = ex.TableSpec.model_validate(
        {
            "id": "private-labels-pending",
            "dataset": "private",
            "title": "Pending",
            "provider": "none yet",
            "model": "none yet",
            "run_ids": [],
            "metrics": [{"name": "verifier.span_recall", "label": "span recall", "pending": True}],
        }
    )
    body = ex.render_table(table, [])
    assert "| span recall | pending owner labels |" in body
    with pytest.raises(ValueError, match="only pending rows"):
        ex.TableSpec.model_validate(
            {**table.model_dump(), "metrics": [{"name": "a.b", "label": "x"}]}
        )


def test_manifest_text_cannot_break_the_markdown(repo: Path) -> None:
    manifest = repo / "evals" / "results.toml"
    text = manifest.read_text(encoding="utf-8").replace('label = "correct"', 'label = "a | b"', 1)
    manifest.write_text(text, encoding="utf-8")
    with pytest.raises(ex.ExportError, match="label"):
        run_export(repo, "check")


# ------------------------------------------------------------------ the CLI wiring
def test_cli_results_exit_codes(repo: Path) -> None:
    out, err = io.StringIO(), io.StringIO()
    env = cli.Env(out=out, err=err)
    args = [
        "--doc",
        str(repo / "RESULTS.md"),
        "--manifest",
        str(repo / "evals" / "results.toml"),
        "--root",
        str(repo / "evals"),
    ]
    assert cli.main(["results", "--write", *args], env) == 0
    assert cli.main(["results", "--check", *args], env) == 0
    doc = repo / "RESULTS.md"
    doc.write_text(doc.read_text(encoding="utf-8").replace("0.900", "0.9", 1), encoding="utf-8")
    assert cli.main(["results", "--check", *args], env) == 1
    (repo / "evals" / "demo" / "runs" / f"{RUN}.summary.json").unlink()
    assert cli.main(["results", "--check", *args], env) == 2
    assert "error: missing run" in err.getvalue()


def test_cli_results_needs_exactly_one_mode() -> None:
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["results"])
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["results", "--check", "--write"])


# ------------------------------------------------------------------ the committed doc
ROOT = Path(__file__).resolve().parents[3]
DEMO_TABLES = [
    t.id
    for t in ex.load_manifest(ROOT / "evals" / "results.toml").table
    if t.dataset == "demo" or not t.run_ids
]


def test_committed_demo_tables_are_up_to_date() -> None:
    """R2-3: the demo blocks of docs/RESULTS.md match the committed summaries and sidecars."""
    out, err = io.StringIO(), io.StringIO()
    code = ex.results(
        mode="check",
        doc_path=ROOT / "docs" / "RESULTS.md",
        manifest_path=ROOT / "evals" / "results.toml",
        root=ROOT / "evals",
        only=DEMO_TABLES,
        out=out,
        err=err,
    )
    assert code == 0, err.getvalue()


def test_committed_readme_headline_is_up_to_date() -> None:
    """The README's headline table is generated the same way (evals/readme-results.toml)."""
    err = io.StringIO()
    code = ex.results(
        mode="check",
        doc_path=ROOT / "README.md",
        manifest_path=ROOT / "evals" / "readme-results.toml",
        root=ROOT / "evals",
        out=io.StringIO(),
        err=err,
    )
    assert code == 0, err.getvalue()


def _private_blocks_generated() -> bool:
    doc = (ROOT / "docs" / "RESULTS.md").read_text(encoding="utf-8")
    return ex.PLACEHOLDER not in doc


@pytest.mark.skipif(
    not (ROOT / "evals" / "private" / "runs").is_dir() or not _private_blocks_generated(),
    reason="private runs absent, or the integrator has not run `results --write` on them yet",
)
def test_committed_private_tables_are_up_to_date() -> None:  # pragma: no cover - owner only
    code = ex.results(
        mode="check",
        doc_path=ROOT / "docs" / "RESULTS.md",
        manifest_path=ROOT / "evals" / "results.toml",
        root=ROOT / "evals",
        out=io.StringIO(),
        err=io.StringIO(),
    )
    assert code == 0


# ------------------------------------------------------------------ retrieval tables (integrator)
def _retrieval_table(model: str = "embed-x, rerank-y") -> ex.TableSpec:
    return ex.TableSpec.model_validate(
        {
            "id": "private-retrieval",
            "dataset": "private",
            "title": "Retrieval",
            "provider": "local",
            "model": model,
            "models_from": "retrieval",
            "run_ids": [RUN],
            "metrics": [{"name": "retrieval.reranked.mrr", "label": "MRR"}],
        }
    )


def _fields(chat: str | None, embed: str | None, rerank: str | None) -> ex.HeaderFields:
    return ex.HeaderFields(
        chat_model=chat,
        verifier_model=chat,
        embed_model=embed,
        rerank_model=rerank,
        dataset=ex.DatasetFields(name="demo", splits=[Split.dev]),
        started_at=datetime(2026, 1, 1, tzinfo=UTC),
    )


def test_a_retrieval_table_ignores_the_servers_chat_model() -> None:
    # A retrieval run records the server's chat model too; no chat model produced its numbers.
    ex.check_models(_retrieval_table(), RUN, _fields("some-chat-model", "embed-x", "rerank-y"))


def test_a_retrieval_table_checks_the_embed_and_rerank_models() -> None:
    with pytest.raises(ex.ExportError, match="retrieval models differ"):
        ex.check_models(_retrieval_table(), RUN, _fields(None, "embed-x", "other-reranker"))


def test_a_retrieval_header_without_models_defers_to_the_manifest() -> None:
    ex.check_models(_retrieval_table(), RUN, _fields(None, None, None))


def test_header_fields_copy_only_the_retrieval_model_names() -> None:
    header = RunHeader(
        run_id=RUN,
        eval_kind=EvalKind.retrieval,
        surface=Surface.chat,
        arm=ARM_TEXT,
        started_at=datetime(2026, 1, 1, tzinfo=UTC),
        dataset=DatasetRef(name="demo", digest="cd" * 32, splits=[Split.dev]),
        retrieval=RetrievalConfig(
            embed_model="embed-x",
            space_id=7,
            chunker_version="chunker-" + TITLE,
            tau=0.5,
            rerank_model="rerank-y",
            rerank_top=20,
        ),
    )
    dumped = ex.header_fields(header).model_dump_json()
    assert '"embed_model":"embed-x"' in dumped
    assert '"rerank_model":"rerank-y"' in dumped
    assert TITLE not in dumped
    assert ARM_TEXT not in dumped


def test_a_trailing_newline_does_not_pass_the_name_and_id_patterns() -> None:
    # `re.match(...$)` accepts "x\n"; fullmatch does not (a newline would break the table).
    assert not ex.METRIC_NAME.fullmatch("retrieval.reranked.mrr\n")
    assert ex.METRIC_NAME.fullmatch("retrieval.reranked.mrr")
