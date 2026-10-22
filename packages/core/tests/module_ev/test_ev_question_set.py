"""Review round 4 regressions: the run records the question set it asked, so deleting (or adding)
a question after the run can never flip the success-bar verdict to **yes**; every kind of edit to
the questions file is proven to fail the bar through `compare`; and Ctrl-C in `label` exits like
`run` does (code 130, one line, labels kept)."""

from __future__ import annotations

import builtins
import io
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import pytest
from ev_support import (
    GOOD_EOS,
    GOOD_LRU,
    GOOD_RAFT,
    QUESTIONS,
    outcome,
    write_run,
)

from ragforge_client import evalrun
from ragforge_client.evalrun import (
    ABSTAIN_PREFIX,
    Label,
    answer_digest,
    compare_runs,
    judge_bar,
    load_labels,
    load_questions,
    load_run,
    question_digest,
    recorded_question_ids,
    removed_questions,
    run_completeness,
    save_labels,
    score_runs,
    summarize_run,
)

WRONG_LRU = "It evicts at random."  # answered, cited, but no expected fact: incorrect


# ------------------------------------------------------------------ helpers
def _blocks() -> list[str]:
    head, *blocks = QUESTIONS.split("[[q]]\n")
    assert head.strip() == ""
    return blocks


def _write(path: Path, blocks: list[str]) -> Path:
    path.write_text("".join(f"[[q]]\n{b}" for b in blocks), encoding="utf-8")
    return path


def _without(blocks: list[str], qid: str) -> list[str]:
    kept = [b for b in blocks if f'id = "{qid}"' not in b]
    assert len(kept) == len(blocks) - 1
    return kept


def _dev_run(tmp_path: Path, qpath: Path, *, lru_answer: str = GOOD_LRU) -> Path:
    """A complete dev-split run as `run` records it now: header asked set, fingerprinted rows,
    a complete footer, and every answer labelled ok by a human."""
    dev = [q for q in load_questions(qpath) if q.split == "dev"]
    by_id = {q.id: q for q in dev}
    rows = [
        outcome("dev-eos", GOOD_EOS, model=["Kafka Delivery"]),
        outcome("dev-lru", lru_answer, model=["Caching"]),
        outcome("dev-off", f"{ABSTAIN_PREFIX}\n- closest: none"),
    ]
    rows = [
        replace(o, question_split=by_id[o.id].split, question_sha256=question_digest(by_id[o.id]))
        for o in rows
    ]
    run = write_run(
        tmp_path / "r.jsonl",
        rows,
        split="dev",
        footer={"complete": True, "resources": {}},
        questions={q.id: question_digest(q) for q in dev},
    )
    labels = {o.id: Label("ok", answer_digest(o.final), "t") for o in rows if o.id != "dev-off"}
    save_labels(run, labels)
    return run


def _verdict_cell(run: Path, qpath: Path) -> str:
    out = io.StringIO()
    compare_runs([str(run)], str(qpath), out=out)
    row = next(line for line in out.getvalue().splitlines() if line.startswith("| arm | dev |"))
    return row.rstrip(" |").rsplit("| ", 1)[1]


# ------------------------------------------------------------------ P1: removed question
def test_deleting_a_failing_question_after_the_run_does_not_flip_the_verdict(
    tmp_path: Path,
) -> None:
    """The reviewer's reproduction: a run with one wrong answer is 'no'; deleting that question's
    [[q]] block used to turn the same run file into '**yes**' with no warning."""
    qpath = _write(tmp_path / "questions.toml", _blocks())
    run = _dev_run(tmp_path, qpath, lru_answer=WRONG_LRU)
    assert _verdict_cell(run, qpath).startswith("no: ")

    _write(qpath, _without(_blocks(), "dev-lru"))
    questions = load_questions(qpath)
    run_file = load_run(run)
    s = summarize_run(run_file, questions)
    assert s["correct"] == 100.0  # what the old code graded: only the survivor
    assert s["removed"] == ["dev-lru"]
    verdict = judge_bar(s, run_completeness(run_file, questions))
    assert verdict.met is False
    assert "1 question(s) removed since the run" in verdict.reasons
    assert "**yes**" not in _verdict_cell(run, qpath)


def test_header_alone_catches_a_removed_question_that_has_no_row(tmp_path: Path) -> None:
    """An errored-out or legacy (unfingerprinted) row is not needed: the header lists the id."""
    qpath = _write(tmp_path / "questions.toml", _blocks())
    rows = [outcome("dev-eos", GOOD_EOS, model=["Kafka Delivery"])]  # no fingerprint
    questions = load_questions(qpath)
    assert removed_questions(questions, rows, {"dev-eos", "gone"}) == ["gone"]
    assert removed_questions(questions, rows) == []  # legacy run: nothing to compare against


def test_fingerprinted_row_alone_catches_a_removed_question(tmp_path: Path) -> None:
    """A run recorded before the header field still flags removal through its row fingerprint."""
    qpath = _write(tmp_path / "questions.toml", _without(_blocks(), "dev-lru"))
    row = replace(outcome("dev-lru", GOOD_LRU), question_split="dev", question_sha256="ab")
    assert removed_questions(load_questions(qpath), [row]) == ["dev-lru"]


def test_recorded_ids_ignore_a_malformed_header() -> None:
    assert recorded_question_ids({"questions": ["not", "a", "dict"]}) == frozenset()
    assert recorded_question_ids({}) == frozenset()
    assert recorded_question_ids({"questions": {"a": "d"}}) == frozenset({"a"})


def test_score_warns_with_a_count_and_never_names_a_removed_heldout_question(
    tmp_path: Path,
) -> None:
    qpath = _write(tmp_path / "questions.toml", _blocks())
    raft = next(q for q in load_questions(qpath) if q.id == "secret-heldout-raft")
    row = replace(
        outcome("secret-heldout-raft", GOOD_RAFT, model=["Consensus"]),
        question_split="heldout",
        question_sha256=question_digest(raft),
    )
    run = write_run(
        tmp_path / "h.jsonl",
        [row],
        split="heldout",
        footer={"complete": True},
        questions={raft.id: question_digest(raft)},
    )
    _write(qpath, _without(_blocks(), "secret-heldout-raft"))
    out = io.StringIO()
    score_runs([str(run)], str(qpath), verbose=True, out=out)
    printed = out.getvalue()
    assert "warning: arm: 1 question(s) removed since run" in printed
    assert "secret-heldout-raft" not in printed
    assert GOOD_RAFT not in printed


# ------------------------------------------------------------------ every edit kind fails
def _edit_text(blocks: list[str]) -> list[str]:
    return [
        b.replace("How does an LRU cache evict?", "How does an LFU cache evict?") for b in blocks
    ]


def _edit_split(blocks: list[str]) -> list[str]:
    return [
        b.replace('split = "dev"', 'split = "heldout"') if 'id = "dev-lru"' in b else b
        for b in blocks
    ]


def _edit_answerable(blocks: list[str]) -> list[str]:
    flipped = 'answerable = true\npages = ["Bread"]\nfacts = [{ fact = "x", match = ["x"] }]'
    return [b.replace("answerable = false", flipped) for b in blocks]


def _edit_delete(blocks: list[str]) -> list[str]:
    return _without(blocks, "dev-lru")


def _edit_add(blocks: list[str]) -> list[str]:
    extra = (
        'id = "dev-new"\nsplit = "dev"\nquestion = "New?"\npages = ["P"]\n'
        'facts = [{ fact = "y", match = ["y"] }]\n\n'
    )
    return [*blocks, extra]


@pytest.mark.parametrize(
    ("edit", "reason"),
    [
        (_edit_text, "1 question(s) changed since the run"),
        (_edit_split, "1 question(s) changed since the run"),
        (_edit_answerable, "1 question(s) changed since the run"),
        (_edit_delete, "1 question(s) removed since the run"),
        (_edit_add, "incomplete 3/4"),
    ],
    ids=["text", "split", "answerable", "deleted", "added"],
)
def test_every_question_set_edit_after_a_passing_run_fails_the_bar(
    tmp_path: Path, edit: Callable[[list[str]], list[str]], reason: str
) -> None:
    qpath = _write(tmp_path / "questions.toml", _blocks())
    run = _dev_run(tmp_path, qpath)
    assert _verdict_cell(run, qpath) == "**yes**"  # the baseline really passes

    _write(qpath, edit(_blocks()))
    cell = _verdict_cell(run, qpath)
    assert cell.startswith("no: ")
    assert reason in cell


def test_fact_and_page_corrections_still_pass(tmp_path: Path) -> None:
    """The intended re-score workflow is untouched by the asked-set check."""
    qpath = _write(tmp_path / "questions.toml", _blocks())
    run = _dev_run(tmp_path, qpath)
    _write(
        qpath, [b.replace('pages = ["Caching"]', 'pages = ["Caching", "LRU"]') for b in _blocks()]
    )
    assert _verdict_cell(run, qpath) == "**yes**"


# ------------------------------------------------------------------ Ctrl-C in label
def test_cli_label_ctrl_c_is_one_line_and_keeps_saved_labels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    qpath = _write(tmp_path / "questions.toml", _blocks())
    run = _dev_run(tmp_path, qpath)
    labels_path = evalrun.labels_path(run)
    labels_path.unlink()  # start unlabelled
    answers = iter(["o"])

    def prompt(_: str) -> str:
        try:
            return next(answers)
        except StopIteration:
            raise KeyboardInterrupt from None

    monkeypatch.setattr(builtins, "input", prompt)
    with pytest.raises(SystemExit) as exc:
        evalrun.run(["--questions", str(qpath), "label", str(run)])
    assert exc.value.code == 130
    err = capsys.readouterr().err.strip().splitlines()
    assert err == [f"ragforge-eval: interrupted; labels saved, resume with: label {run}"]
    assert set(load_labels(run)) == {"dev-eos"}  # the answer given before Ctrl-C was kept
