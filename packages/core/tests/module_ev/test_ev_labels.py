"""Human invented-content labels: the sidecar is the only source of invented%, labels are bound to
the exact answer text, and the interactive walk is resumable and leak-free for held-out rows."""

from __future__ import annotations

import io
import json
import os
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest
from ev_support import (
    GOOD_EOS,
    GOOD_LRU,
    GOOD_RAFT,
    outcome,
    perfect_run_outcomes,
    write_questions,
    write_run,
)

from ragforge_client import evalrun
from ragforge_client.evalrun import (
    Label,
    answer_digest,
    heldout_revealed_at,
    invented_stats,
    label_run,
    labels_path,
    load_labels,
    load_questions,
    save_labels,
    summarize,
)


def _script(*answers: str) -> Callable[[str], str]:
    it: Iterator[str] = iter(answers)

    def prompt(_: str) -> str:
        try:
            return next(it)
        except StopIteration:
            raise EOFError from None

    return prompt


def _label(verdict: str, text: str) -> Label:
    return Label(verdict, answer_digest(text), "t")  # type: ignore[arg-type]


def test_sidecar_sits_next_to_the_run(tmp_path: Path) -> None:
    assert labels_path(tmp_path / "20261017-A1.jsonl") == tmp_path / "20261017-A1.labels.json"


def test_without_labels_invented_is_unlabelled_not_zero(tmp_path: Path) -> None:
    s = summarize(load_questions(write_questions(tmp_path)), perfect_run_outcomes())
    assert s["invented"].status == "unlabelled"
    assert s["invented"].answered == 3  # abstentions are not answers


def test_labels_drive_invented_pct(tmp_path: Path) -> None:
    labels = {
        "dev-eos": _label("ok", GOOD_EOS),
        "dev-lru": _label("invented", GOOD_LRU),
        "secret-heldout-raft": _label("unsure", GOOD_RAFT),
    }
    inv = invented_stats(perfect_run_outcomes(), labels)
    assert inv.status == "labelled"
    assert round(inv.pct) == 33
    assert round(inv.upper_pct) == 67  # unsure counts as invented for the bar


def test_stale_label_for_different_answer_text_is_ignored(tmp_path: Path) -> None:
    labels = {"dev-eos": _label("ok", "an older answer"), "dev-lru": _label("ok", GOOD_LRU)}
    inv = invented_stats(perfect_run_outcomes(), labels)
    assert (inv.labelled, inv.status) == (1, "partial 1/3")


def test_labels_on_abstentions_or_errors_do_not_count(tmp_path: Path) -> None:
    outcomes = [outcome("dev-eos", GOOD_EOS, error="504 timeout")]
    inv = invented_stats(outcomes, {"dev-eos": _label("invented", GOOD_EOS)})
    assert (inv.answered, inv.status) == (0, "no answers")


@pytest.mark.parametrize(
    "body",
    [
        json.dumps({"labels": {"dev-eos": {"verdict": "maybe", "answer_sha256": "x"}}}),
        json.dumps(
            {"labels": {"dev-eos": {"verdict": "ok"}}}
        ),  # no digest: cannot detect staleness
        json.dumps({"labels": {"dev-eos": "ok"}}),
        json.dumps({"labels": []}),
        json.dumps([1, 2]),
        "{not json",
    ],
)
def test_malformed_sidecar_fails_loudly(tmp_path: Path, body: str) -> None:
    run = write_run(tmp_path / "r.jsonl", perfect_run_outcomes())
    labels_path(run).write_text(body)
    with pytest.raises(ValueError):  # noqa: PT011 - each malformed shape has its own message
        load_labels(run)


def test_save_load_round_trip_leaves_no_temp_files(tmp_path: Path) -> None:
    run = write_run(tmp_path / "r.jsonl", perfect_run_outcomes())
    save_labels(run, {"dev-eos": _label("ok", GOOD_EOS)})
    assert load_labels(run) == {"dev-eos": _label("ok", GOOD_EOS)}
    assert sorted(p.name for p in tmp_path.iterdir()) == ["r.jsonl", "r.labels.json"]


def test_label_walk_hides_heldout_and_saves_each_answer(tmp_path: Path) -> None:
    questions = load_questions(write_questions(tmp_path))
    run = write_run(tmp_path / "r.jsonl", perfect_run_outcomes())
    out = io.StringIO()
    labels = label_run(
        run, questions, reveal=False, relabel=False, prompt=_script("o", "i"), out=out
    )
    assert {k: v.verdict for k, v in labels.items()} == {"dev-eos": "ok", "dev-lru": "invented"}
    assert load_labels(run) == labels  # persisted, not just returned
    text = out.getvalue()
    assert "secret-heldout-raft" not in text
    assert "SECRET heldout question text" not in text
    assert "1 held-out answer(s) not shown" in text


def test_label_walk_reveal_includes_heldout(tmp_path: Path) -> None:
    questions = load_questions(write_questions(tmp_path))
    run = write_run(tmp_path / "r.jsonl", perfect_run_outcomes())
    labels = label_run(
        run, questions, reveal=True, relabel=False, prompt=_script("o", "o", "u"), out=io.StringIO()
    )
    assert labels["secret-heldout-raft"].verdict == "unsure"


def test_label_walk_resumes_skips_and_quits(tmp_path: Path) -> None:
    questions = load_questions(write_questions(tmp_path))
    run = write_run(tmp_path / "r.jsonl", perfect_run_outcomes())
    # Invalid input is re-asked; skip leaves dev-eos unlabelled; quit stops before dev-lru.
    label_run(run, questions, reveal=False, relabel=False, prompt=_script("x", "s", "q"))
    assert load_labels(run) == {}
    label_run(run, questions, reveal=False, relabel=False, prompt=_script("o"))
    assert set(load_labels(run)) == {"dev-eos"}  # EOF after one answer = quit, progress kept
    # A second session only asks about what is still unlabelled.
    out = io.StringIO()
    label_run(run, questions, reveal=False, relabel=False, prompt=_script("i"), out=out)
    assert "=== dev-eos ===" not in out.getvalue()
    assert load_labels(run)["dev-lru"].verdict == "invented"


def test_label_walk_strips_terminal_escapes_from_model_text(tmp_path: Path) -> None:
    questions = load_questions(write_questions(tmp_path))
    hostile = outcome("dev-eos", "fine\x1b]52;c;cGF3bmVk\x07 answer", model=["Kafka Delivery"])
    run = write_run(tmp_path / "r.jsonl", [hostile])
    out = io.StringIO()
    label_run(run, questions, reveal=False, relabel=False, prompt=_script("o"), out=out)
    assert "\x1b" not in out.getvalue()
    assert "\x07" not in out.getvalue()


def test_save_fsyncs_the_data_before_the_rename(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Durability (review P2): an un-fsynced rename can leave an empty sidecar after power loss."""
    events: list[str] = []
    real_fsync, real_replace = os.fsync, os.replace

    def fsync(fd: int) -> None:
        events.append("fsync")
        real_fsync(fd)

    def replace(src: str, dst: Path) -> None:
        events.append("replace")
        real_replace(src, dst)

    monkeypatch.setattr(os, "fsync", fsync)
    monkeypatch.setattr(os, "replace", replace)
    run = write_run(tmp_path / "r.jsonl", perfect_run_outcomes())
    save_labels(run, {"dev-eos": _label("ok", GOOD_EOS)})
    assert events[:2] == ["fsync", "replace"]  # data is on disk before it becomes visible
    assert load_labels(run) == {"dev-eos": _label("ok", GOOD_EOS)}


def test_first_heldout_reveal_is_stamped_and_announced_later(tmp_path: Path) -> None:
    questions = load_questions(write_questions(tmp_path))
    run = write_run(tmp_path / "r.jsonl", perfect_run_outcomes())
    label_run(run, questions, reveal=False, relabel=False, prompt=_script("o", "o"))
    assert heldout_revealed_at(run) is None  # dev-only labelling never stamps
    label_run(run, questions, reveal=True, relabel=False, prompt=_script("q"))
    stamp = heldout_revealed_at(run)
    assert stamp is not None  # shown then quit: still a reveal
    assert set(load_labels(run)) == {"dev-eos", "dev-lru"}  # earlier labels kept
    out = io.StringIO()
    label_run(run, questions, reveal=True, relabel=False, prompt=_script("o"), out=out)
    assert f"first revealed at {stamp}" in out.getvalue()
    assert heldout_revealed_at(run) == stamp  # the FIRST reveal, never moved forward
    assert load_labels(run)["secret-heldout-raft"].verdict == "ok"


def test_cli_label_hides_heldout_and_persists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The `label` subcommand end to end through run([...]) with a scripted terminal."""
    questions = write_questions(tmp_path)
    run_path = write_run(tmp_path / "r.jsonl", perfect_run_outcomes())
    monkeypatch.setattr("builtins.input", _script("o", "i"))
    evalrun.run(["--questions", str(questions), "label", str(run_path)])
    printed = capsys.readouterr().out
    assert {k: v.verdict for k, v in load_labels(run_path).items()} == {
        "dev-eos": "ok",
        "dev-lru": "invented",
    }
    assert "secret-heldout-raft" not in printed
    assert "SECRET heldout question text" not in printed
    assert "1 held-out answer(s) not shown" in printed
    assert heldout_revealed_at(run_path) is None
