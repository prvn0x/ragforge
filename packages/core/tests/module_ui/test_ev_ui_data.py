"""EV-UI local file access, config and shown notes: degrade instead of failing, name files and
exception TYPES only, digest-check every note, keep the eval key out of reprs. No network: the
SDK path runs against respx."""

from __future__ import annotations

import json
import os
from pathlib import Path
from uuid import uuid4

import httpx
import pytest
import respx
from test_ev_ui_support import (
    FIXTURES,
    GOLDEN,
    T0,
    chunk,
    golden_paths,
    header,
    sha,
    shown_ref,
    write_run,
)

from ragforge_client.ask import KeyFileError
from ragforge_client.evals.datasets import LabelStore
from ragforge_client.evals.schema import Label, LabelKind, ShownRef, Surface
from ragforge_contracts.api import EvalChunksResponse
from ragforge_ui.eval_view.config import DEFAULT_RUNS, EvalConfigError, load_eval_config
from ragforge_ui.eval_view.context import load_context
from ragforge_ui.eval_view.data import (
    FileLabelReader,
    LabelAccess,
    label_store,
    load_datasets,
    load_gate,
    load_runs,
    read_jsonl,
    read_labels,
    summary_path,
)
from ragforge_ui.eval_view.notes import NoteState, SdkChunkSource, sha256_hex, shown_notes


# ------------------------------------------------------------------ config
def test_config_defaults_to_the_private_runs_dir_and_its_parent() -> None:
    config = load_eval_config([], {})
    assert config.paths.runs == DEFAULT_RUNS
    assert config.paths.labels == DEFAULT_RUNS.parent / "labels.jsonl"
    assert config.paths.judge_results == DEFAULT_RUNS.parent / "judge-cache.jsonl"
    assert config.key_file is None


def test_flags_win_over_env_and_unknown_flags_are_ignored(tmp_path: Path) -> None:
    env = {"RAGFORGE_EVAL_RUNS": "/env/runs", "RAGFORGE_EVAL_KEY_FILE": "/env/eval.key"}
    argv = [
        "--key-file",
        "ui.key",
        "--eval-runs",
        str(tmp_path / "r"),
        "--eval-data",
        str(tmp_path / "d"),
    ]
    config = load_eval_config(argv, env)
    assert config.paths.runs == tmp_path / "r"
    assert config.paths.questions == tmp_path / "d" / "questions.toml"
    assert config.key_file == Path("/env/eval.key")
    assert load_eval_config([], env).paths.runs == Path("/env/runs")


# ------------------------------------------------------------------ runs and summaries
@pytest.mark.parametrize("argv", [["--eval-runs"], ["--eval-key-file"], ["--url"]])
def test_a_flag_without_its_value_raises_a_config_error_not_system_exit(argv: list[str]) -> None:
    with pytest.raises(EvalConfigError, match="--"):
        load_eval_config(argv, {})


def test_load_runs_reads_every_golden_run_and_skips_label_and_judge_files() -> None:
    read = load_runs(FIXTURES)  # the directory also holds labels.jsonl, judge-results.jsonl
    assert read.problems == []
    entries = read.entries
    assert sorted(e.header.run_id for e in entries) == sorted(f"golden-{n}" for n in GOLDEN)
    assert all(e.summary is not None and e.summary_problem is None for e in entries)


def test_runs_are_newest_first_and_a_missing_summary_is_explained(tmp_path: Path) -> None:
    old = write_run(tmp_path / "run-old.jsonl", header("old"), [])
    write_run(tmp_path / "run-new.jsonl", header("new", started=T0.replace(hour=20)), [])
    entries = load_runs(tmp_path).entries
    assert [e.header.run_id for e in entries] == ["new", "old"]
    assert entries[0].summary is None
    assert "no summary yet" in (entries[0].summary_problem or "")
    assert summary_path(old) == tmp_path / "run-old.summary.json"


def test_a_missing_runs_directory_is_no_runs(tmp_path: Path) -> None:
    assert load_runs(tmp_path / "absent").entries == []


@pytest.mark.skipif(os.geteuid() == 0, reason="root reads a mode-000 file")
def test_one_unreadable_run_file_is_a_problem_line_not_an_empty_list(tmp_path: Path) -> None:
    write_run(tmp_path / "run-ok.jsonl", header("ok"), [])
    locked = write_run(tmp_path / "run-locked.jsonl", header("locked"), [])
    locked.chmod(0)
    try:
        read = load_runs(tmp_path)
    finally:
        locked.chmod(0o600)
    assert [e.header.run_id for e in read.entries] == ["ok"]
    assert read.problems == ["run-locked.jsonl: unreadable (PermissionError)"]


def test_gate_files_are_read_per_surface_and_broken_ones_named(tmp_path: Path) -> None:
    assert load_gate(tmp_path, Surface.chat) == (None, None)
    (tmp_path / "gate-chat.json").write_text('{"surface": "chat", "answer": "private text"}')
    verdict, why = load_gate(tmp_path, Surface.chat)
    assert verdict is None
    assert why == "gate-chat.json: unreadable (ValidationError)"


def test_read_jsonl_counts_bad_lines_instead_of_failing(tmp_path: Path) -> None:
    good = Label(
        item_digest="d", item_id="q", kind=LabelKind.criteria, batch_id="b", labelled_at=T0
    )
    path = tmp_path / "labels.jsonl"
    path.write_text(good.model_dump_json() + "\n{not json\n\n" + '{"schema_version": 9}\n')
    read = read_jsonl(path, Label)
    assert (read.items, read.skipped, read.problem) == ([good], 2, None)
    assert read_jsonl(tmp_path / "absent.jsonl", Label).items == []


def test_datasets_load_and_a_broken_one_is_a_problem_not_a_crash(tmp_path: Path) -> None:
    paths = golden_paths(tmp_path)
    bundle = load_datasets(paths)
    assert bundle.problems == []
    assert bundle.datasets.questions is not None
    assert len(bundle.datasets.questions.questions) == 20
    assert bundle.datasets.verifier is not None
    assert bundle.datasets.interview is not None
    paths.verifier.write_text("[]")
    broken = load_datasets(paths)
    assert broken.datasets.verifier is None
    assert broken.problems == ["verifier.json: unreadable (DatasetError)"]


# ------------------------------------------------------------------ labels
def _label(digest: str, kind: LabelKind, note: str) -> Label:
    return Label(
        item_digest=digest, item_id=digest, kind=kind, note=note, batch_id="b", labelled_at=T0
    )


def test_the_file_reader_keeps_the_newest_line_per_digest_and_kind(tmp_path: Path) -> None:
    path = tmp_path / "labels.jsonl"
    lines = [
        _label("a", LabelKind.criteria, "first"),
        _label("a", LabelKind.answer, "other kind"),
        _label("a", LabelKind.criteria, "second"),
    ]
    path.write_text("".join(x.model_dump_json() + "\n" for x in lines))
    latest = {(lb.item_digest, lb.kind): lb.note for lb in FileLabelReader(path).latest()}
    assert latest == {("a", LabelKind.criteria): "second", ("a", LabelKind.answer): "other kind"}


def test_label_store_is_the_eval_layers_store(tmp_path: Path) -> None:
    access = label_store(tmp_path / "labels.jsonl")
    assert isinstance(access.reader, LabelStore)
    assert access.writer is access.reader
    assert access.reader.path == tmp_path / "labels.jsonl"


def test_a_label_file_the_store_refuses_is_a_problem_and_its_good_lines_still_show(
    tmp_path: Path,
) -> None:
    path = tmp_path / "labels.jsonl"
    good = _label("a", LabelKind.criteria, "kept")
    path.write_text(
        good.model_dump_json()
        + "\n"
        + json.dumps({"schema_version": 2, "item_digest": "x", "kind": "criteria"})
        + "\n"
    )
    store = LabelStore(path)
    with pytest.raises(ValueError, match="invalid label field"):
        store.latest()  # the store's own contract: any invalid line raises
    read = read_labels(LabelAccess(store, store), path)
    assert read.problem == "labels.jsonl: unreadable (DatasetError)"
    assert [lb.note for lb in read.labels] == ["kept"]


def test_a_readable_label_file_has_no_problem(tmp_path: Path) -> None:
    path = tmp_path / "labels.jsonl"
    store = LabelStore(path)
    store.append(_label("a", LabelKind.criteria, "one"))
    read = read_labels(LabelAccess(store, store), path)
    assert (read.problem, [lb.note for lb in read.labels]) == (None, ["one"])


# ------------------------------------------------------------------ shown notes
class Chunks:
    def __init__(self, *held: object) -> None:
        self.held = list(held)
        self.calls = 0

    def chunks(self, chunk_ids: object) -> list:  # type: ignore[type-arg]
        self.calls += 1
        return self.held


def test_a_note_is_shown_only_when_both_digests_match_the_run() -> None:
    good = chunk("exact text")
    lying = chunk("edited text", digest=sha("original text"))  # server digest says "original"
    moved = chunk("new text")
    refs = [
        shown_ref(2, lying, digest=sha("original text")),
        shown_ref(1, good),
        shown_ref(3, moved, digest=sha("old text")),
        ShownRef(label=4, chunk_id="not-a-uuid", display_sha256="0" * 64),
        ShownRef(label=5, chunk_id=str(uuid4()), display_sha256="0" * 64),
    ]
    notes = shown_notes(refs, Chunks(good, lying, moved))
    assert [(n.label, n.state) for n in notes] == [
        (1, NoteState.ok),
        (2, NoteState.changed),  # our own hash catches a server digest that lies
        (3, NoteState.changed),
        (4, NoteState.missing),
        (5, NoteState.missing),
    ]
    assert notes[0].text == "exact text"
    assert notes[0].where == ("file:kafka-basics.md", "Consumers")
    assert all(n.text is None for n in notes[1:])


def test_no_refs_means_no_fetch() -> None:
    source = Chunks()
    assert shown_notes([], source) == []
    assert source.calls == 0


def test_sha256_hex_is_the_chunkers_display_digest() -> None:
    assert sha256_hex("abc") == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def _key(tmp_path: Path, mode: int = 0o600) -> Path:
    key = tmp_path / "eval.key"
    key.write_text("rf_eval_SECRET\n")
    key.chmod(mode)
    return key


@respx.mock
def test_the_sdk_source_sends_the_eval_key_and_never_shows_it(tmp_path: Path) -> None:
    c = chunk("note text")
    route = respx.get("http://api.test/v1/eval/chunks").mock(
        return_value=httpx.Response(
            200, json=json.loads(EvalChunksResponse(chunks=[c]).model_dump_json())
        )
    )
    source = SdkChunkSource("http://api.test", _key(tmp_path))
    assert source.chunks([c.chunk_id]) == [c]
    request = route.calls.last.request
    assert request.headers["authorization"] == "Bearer rf_eval_SECRET"
    assert str(c.chunk_id) in str(request.url)
    shown = SdkChunkSource("http://user:pw@api.test", _key(tmp_path))
    assert "SECRET" not in repr(shown)
    assert "pw" not in repr(shown)


def test_a_key_file_others_can_read_is_refused(tmp_path: Path) -> None:
    source = SdkChunkSource("http://api.test", _key(tmp_path, 0o644))
    with pytest.raises(KeyFileError, match="readable by others"):
        source.chunks([uuid4()])


def test_the_context_keeps_only_a_display_safe_api_url(tmp_path: Path) -> None:
    """Error boxes print `ctx.api_url` verbatim, so userinfo must never reach it."""
    paths = golden_paths(tmp_path)
    ctx = load_context(
        paths,
        label_store(paths.labels),
        lambda: pytest.fail("no fetch"),
        "http://user:sk-secret@localhost:8000/?token=x",
    )
    assert ctx.api_url == "http://localhost:8000/"
    assert "sk-secret" not in repr(ctx)
