"""EV-JUDGE: dataset loading (D0 and synthetic private-form files), heldout2 sealing, the label
store (newest wins; two concurrent writer PROCESSES lose nothing) and the lexical overlap flag."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import pytest
from ev_judge_support import DEMO, FIXTURES, T0
from hypothesis import assume, given
from hypothesis import strategies as st

from ragforge_client.evals.datasets import (
    DatasetError,
    LabelStore,
    SealError,
    append_jsonl,
    content_lemmas,
    jaccard,
    load_datasets,
    load_interview_set,
    load_questions,
    load_verifier_set,
    overlap_flags,
    question_page_overlap,
    seal_digest,
    seal_questions,
    verify_seal,
)
from ragforge_client.evals.schema import Criterion, Label, LabelKind, Split

PRIVATE_FORM = """\
# synthetic stand-in for the private file form: name "private", dev + heldout2 in one file
schema_version = 2
name = "private"

[[questions]]
id = "dev-one"
split = "dev"
question = "How does the consumer group rebalance?"
relevance = { "notion:0001" = 2 }
facts = [{ fact = "heartbeats", match = ["heartbeat"] }]

[[questions]]
id = "ho-one"
split = "heldout2"
question = "What bounds AOF loss?"
relevance = { "notion:0002" = 2 }
facts = [{ fact = "everysec", match = ["everysec"] }]

[[questions]]
id = "ho-two"
split = "heldout2"
question = "Why per-key limits?"
answerable = false
near_miss = true
"""


# ------------------------------------------------------------------ loading
def test_the_demo_set_loads_into_one_bundle() -> None:
    ds = load_datasets(DEMO)
    assert ds.questions is not None
    assert len(ds.questions.questions) == 20
    assert ds.verifier is not None
    assert len(ds.verifier.items) == 20
    assert ds.interview is not None
    assert len(ds.interview.seeds) == 4
    assert load_questions(DEMO / "questions.toml") == ds.questions


def test_a_private_form_file_loads_with_both_splits(tmp_path: Path) -> None:
    path = tmp_path / "questions.toml"
    path.write_text(PRIVATE_FORM)
    qs = load_questions(path)
    assert qs.name == "private"
    assert [q.split for q in qs.questions] == [Split.dev, Split.heldout2, Split.heldout2]


def test_a_v1_file_is_refused_with_the_migration_hint(tmp_path: Path) -> None:
    path = tmp_path / "questions.toml"
    path.write_text('[[q]]\nid = "x"\nsplit = "dev"\nquestion = "?"\n')
    with pytest.raises(DatasetError, match=r"v1 question file.*relevance"):
        load_questions(path)


def test_bad_files_fail_loudly_with_the_place(tmp_path: Path) -> None:
    path = tmp_path / "q.toml"
    path.write_text(PRIVATE_FORM.replace('split = "dev"', 'split = "dev"\ntypo = 1'))
    with pytest.raises(DatasetError, match=r"questions\.0\.typo"):
        load_questions(path)
    path.write_text("not = [toml")
    with pytest.raises(DatasetError, match="not TOML"):
        load_questions(path)
    other = tmp_path / "other.toml"
    other.write_text(PRIVATE_FORM)
    path.write_text(PRIVATE_FORM)
    with pytest.raises(DatasetError, match="duplicate question id"):
        load_questions(path, other)


def test_verifier_and_interview_sets_validate_their_ids(tmp_path: Path) -> None:
    text = (DEMO / "verifier.json").read_text()
    assert len(load_verifier_set(DEMO / "verifier.json").items) == 20
    orphan = tmp_path / "v.json"
    orphan.write_text(text.replace('"source_id": "c-aof"', '"source_id": "c-nope"', 1))
    with pytest.raises(DatasetError, match="its own source"):
        load_verifier_set(orphan)
    last = text.rindex('"source_id": "c-aof"')  # a planted item of c-aof
    orphan.write_text(text[:last] + '"source_id": "c-nope"' + text[last + 20 :])
    with pytest.raises(DatasetError, match="without a clean source"):
        load_verifier_set(orphan)
    newer = tmp_path / "i.json"
    newer.write_text(
        (DEMO / "interview.json").read_text().replace('"schema_version": 2', '"schema_version": 3')
    )
    with pytest.raises(DatasetError, match="schema_version"):
        load_interview_set(newer)


# ------------------------------------------------------------------ sealing
def test_seal_then_verify_then_an_edit_fails(tmp_path: Path) -> None:
    path = tmp_path / "heldout2.toml"
    path.write_text(PRIVATE_FORM)
    with pytest.raises(SealError, match="not sealed"):
        load_questions(path, require_seal=True)
    seal = seal_questions(path)
    assert path.read_text().startswith(f'seal = "{seal}"')
    assert verify_seal(path) == seal
    assert load_questions(path, require_seal=True).questions[1].id == "ho-one"
    with pytest.raises(SealError, match="already sealed"):
        seal_questions(path)

    path.write_text(path.read_text().replace("What bounds AOF loss?", "What bounds AOF data loss?"))
    with pytest.raises(SealError, match="changed after it was sealed"):
        load_questions(path)  # a sealed file is verified on EVERY load, required or not


@pytest.mark.parametrize(
    ("edit", "breaks"),
    [
        (lambda t: t.replace('"everysec"]', '"everysec", "every sec"]'), True),  # match edited
        (lambda t: t.replace('split = "heldout2"', 'split = "dev"', 1), True),  # moved out
        (lambda t: t.replace('"notion:0002" = 2', '"notion:0002" = 1'), True),
        (lambda t: t.replace("How does the consumer", "How do the consumers"), False),  # dev row
        (lambda t: t.replace("# synthetic", "# edited comment, synthetic"), False),
    ],
)
def test_the_seal_covers_exactly_the_heldout2_meaning(
    tmp_path: Path, edit: Callable[[str], str], breaks: bool
) -> None:
    path = tmp_path / "q.toml"
    path.write_text(PRIVATE_FORM)
    seal_questions(path)
    path.write_text(edit(path.read_text()))
    if breaks:
        with pytest.raises(SealError):
            verify_seal(path)
    else:
        assert verify_seal(path) is not None


def test_the_seal_is_order_independent_and_ignores_dev_rows(tmp_path: Path) -> None:
    path = tmp_path / "q.toml"
    path.write_text(PRIVATE_FORM)
    qs = load_questions(path)
    assert seal_digest(qs.questions) == seal_digest(list(reversed(qs.questions)))
    assert seal_digest(qs.questions) == seal_digest(
        [q for q in qs.questions if q.split is Split.heldout2]
    )


def test_sealing_keeps_the_file_private(tmp_path: Path) -> None:
    path = tmp_path / "q.toml"
    path.write_text(PRIVATE_FORM)
    path.chmod(0o600)
    seal_questions(path)
    assert path.stat().st_mode & 0o777 == 0o600


# ------------------------------------------------------------------ label store
def _label(
    digest: str, kind: LabelKind = LabelKind.criteria, note: str = "", batch: str = "b1"
) -> Label:
    return Label(
        item_digest=digest,
        item_id="i",
        kind=kind,
        scores={} if kind is not LabelKind.criteria else {Criterion.relevance: 2},
        correct=True if kind is LabelKind.answer else None,
        note=note,
        batch_id=batch,
        labelled_at=T0,
    )


def test_the_newest_line_per_digest_and_kind_wins(tmp_path: Path) -> None:
    store = LabelStore(tmp_path / "labels.jsonl")
    assert store.latest() == []
    store.append(_label("a", note="first"))
    store.append(_label("a", LabelKind.answer, note="answer"))
    store.append(_label("b", note="b"))
    # a relabel with an OLDER timestamp still wins: file order is the real order
    store.append(
        _label("a", note="second").model_copy(
            update={"labelled_at": datetime(2020, 1, 1, tzinfo=UTC)}
        )
    )
    latest = {(lb.item_digest, lb.kind): lb.note for lb in store.latest()}
    assert latest == {
        ("a", LabelKind.criteria): "second",
        ("a", LabelKind.answer): "answer",
        ("b", LabelKind.criteria): "b",
    }
    assert len(store.all()) == 4
    assert (tmp_path / "labels.jsonl").stat().st_mode & 0o777 == 0o600


def test_the_fixture_labels_load_through_the_store() -> None:
    labels = LabelStore(FIXTURES / "labels.jsonl").latest()
    assert labels
    assert all(lb.batch_id for lb in labels)


def test_a_corrupt_label_file_fails_loudly(tmp_path: Path) -> None:
    path = tmp_path / "labels.jsonl"
    path.write_text('{"schema_version": 2, "item_digest": "x"}\n')
    with pytest.raises(DatasetError, match="invalid label"):
        LabelStore(path).latest()
    path.write_text('{"schema_version": 1}\n')
    with pytest.raises(DatasetError, match="schema_version"):
        LabelStore(path).latest()


_WRITER = textwrap.dedent(
    """
    import sys
    from datetime import UTC, datetime
    from pathlib import Path
    from ragforge_client.evals.datasets import LabelStore
    from ragforge_client.evals.schema import Label, LabelKind

    store = LabelStore(Path(sys.argv[1]))
    for i in range(int(sys.argv[3])):
        store.append(Label(item_digest=f"{sys.argv[2]}-{i}", item_id="i", kind=LabelKind.answer,
                           correct=True, batch_id="b", labelled_at=datetime.now(UTC)))
    """
)


def test_two_writer_processes_lose_no_label(tmp_path: Path) -> None:
    """The CLI and the Eval page may append at the same moment: neither loses a line, and no
    line is interleaved with another (every line parses)."""
    path = tmp_path / "labels.jsonl"
    n = 60
    procs = [
        subprocess.Popen([sys.executable, "-c", _WRITER, str(path), name, str(n)])  # noqa: S603
        for name in ("cli", "page")
    ]
    assert [p.wait(timeout=120) for p in procs] == [0, 0]
    labels = LabelStore(path).all()
    assert len(labels) == 2 * n
    assert {lb.item_digest for lb in labels} == {
        f"{w}-{i}" for w in ("cli", "page") for i in range(n)
    }


# ------------------------------------------------------------------ lexical overlap
def test_a_question_lifted_from_a_note_sentence_is_flagged(tmp_path: Path) -> None:
    page = "# Redis\n\nWith appendfsync everysec, at most about one second of writes is lost.\n"
    lifted = "With appendfsync everysec, how much of writes is lost at most?"
    reworded = "AOF: worst-case data loss on crash?"
    assert question_page_overlap(lifted, page) > 0.5
    assert question_page_overlap(reworded, page) < 0.5
    path = tmp_path / "q.toml"
    path.write_text(
        PRIVATE_FORM.replace("What bounds AOF loss?", lifted).replace(
            "Why per-key limits?", "Why per-key limits?"
        )
    )
    flags = overlap_flags(load_questions(path), {"notion:0002": page, "notion:0001": page})
    assert [(f.question_id, f.page_key, f.flagged) for f in flags] == [
        ("ho-one", "notion:0002", True)
    ]


def test_lemmas_drop_stopwords_and_meet_across_inflections() -> None:
    assert content_lemmas("The partitions are rebalancing") == content_lemmas(
        "a partition rebalanced"
    )
    assert "max.poll.interval.ms" in content_lemmas("exceeding max.poll.interval.ms")
    assert content_lemmas("the of and is") == frozenset()


_words = st.lists(
    st.sampled_from(["kafka", "redis", "lag", "poll", "the", "acks", "replica", "is"]), max_size=12
)


@given(_words, _words)
def test_jaccard_properties(a: list[str], b: list[str]) -> None:
    la, lb = content_lemmas(" ".join(a)), content_lemmas(" ".join(b))
    score = jaccard(la, lb)
    assert 0.0 <= score <= 1.0
    assert score == jaccard(lb, la)
    assert jaccard(la, la) == (1.0 if la else 0.0)
    assert question_page_overlap(" ".join(a), " ".join(a) + ".") == (1.0 if la else 0.0)


@given(st.text(alphabet="abcdefghijklmnopqrstuvwxyz", min_size=3, max_size=12))
def test_a_plural_always_meets_its_singular(word: str) -> None:
    """Fixed after review: the old `-es` rule made "caches" -> "cach" but left "cache", so the
    two never met and the flag under-reported."""
    assume(not word.endswith("s"))
    single, plural = content_lemmas(word), content_lemmas(word + "s")
    assume(single and plural)  # neither form is a stopword
    assert single == plural


@pytest.mark.parametrize(
    "forms",
    [
        ("cache", "caches", "cached", "caching"),
        ("box", "boxes"),
        ("policy", "policies"),
        ("class", "classes"),
        ("rebalance", "rebalances", "rebalanced", "rebalancing"),
    ],
)
def test_inflected_forms_share_one_lemma(forms: tuple[str, ...]) -> None:
    assert len({content_lemmas(f) for f in forms}) == 1, [content_lemmas(f) for f in forms]


# ------------------------------------------------------------------ heldout2 always sealed
def test_a_heldout2_file_must_be_sealed_without_any_flag(tmp_path: Path) -> None:
    """The bar does not rest on callers passing require_seal: loading a `heldout2.toml` (by
    path or through the bundle) without its seal fails, and deleting the seal line of a sealed
    file and then editing it fails the same way."""
    (tmp_path / "questions.toml").write_text(DEV_ONLY)
    held = tmp_path / "heldout2.toml"
    held.write_text(HELDOUT_ONLY)
    with pytest.raises(SealError, match="not sealed"):
        load_questions(held)
    with pytest.raises(SealError, match="not sealed"):
        load_datasets(tmp_path)
    seal_questions(held)
    bundle = load_datasets(tmp_path)
    assert bundle.questions is not None
    assert {q.id for q in bundle.questions.questions} == {"dev-one", "ho-one"}
    unsealed = "".join(held.read_text().splitlines(keepends=True)[1:])
    held.write_text(unsealed.replace("What bounds AOF loss?", "What bounds AOF data loss?"))
    with pytest.raises(SealError, match="not sealed"):
        load_datasets(tmp_path)


DEV_ONLY = PRIVATE_FORM.split('[[questions]]\nid = "ho-one"')[0]
HELDOUT_ONLY = (
    'schema_version = 2\nname = "private"\n\n[[questions]]\nid = "ho-one"'
    + PRIVATE_FORM.split('id = "ho-one"')[1].split("[[questions]]")[0]
)


# ------------------------------------------------------------------ append-only JSONL
def test_an_append_writes_in_place_and_a_reader_skips_an_unfinished_line(tmp_path: Path) -> None:
    """One O_APPEND write per append (the file is extended, never rewritten), and a line still
    being written (no newline yet) is not read as a corrupt label."""
    path = tmp_path / "labels.jsonl"
    store = LabelStore(path)
    store.append(_label("a"))
    inode = path.stat().st_ino
    store.append(_label("b"))
    assert path.stat().st_ino == inode
    with path.open("ab") as handle:
        handle.write(_label("c").model_dump_json().encode()[:30])  # a write in progress
    assert [lb.item_digest for lb in store.all()] == ["a", "b"]
    with pytest.raises(ValueError, match="newline"):
        append_jsonl(path, ["x\ny"])


def test_a_torn_tail_is_cut_so_the_next_append_keeps_every_label(tmp_path: Path) -> None:
    """A writer that crashed mid-line leaves a fragment with no newline. The next append must
    CUT it, not terminate it: a terminated fragment is a corrupt line in the middle of the file,
    and every later read would refuse the whole label store."""
    path = tmp_path / "labels.jsonl"
    store = LabelStore(path)
    store.append(_label("a"))
    with path.open("ab") as handle:
        handle.write(_label("torn").model_dump_json().encode()[:30])  # the crash
    store.append(_label("b"))
    assert [lb.item_digest for lb in store.latest()] == ["a", "b"]
    assert path.read_bytes().count(b"\n") == 2


def test_a_file_that_is_only_a_fragment_is_cut_to_nothing(tmp_path: Path) -> None:
    path = tmp_path / "labels.jsonl"
    path.write_bytes(b"x" * 10_000)  # longer than one backwards-read block, and no newline
    LabelStore(path).append(_label("a"))
    assert [lb.item_digest for lb in LabelStore(path).all()] == ["a"]


def test_a_short_write_leaves_no_fragment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A full disk mid-write: the append fails loudly AND the file is back to its last line."""
    path = tmp_path / "labels.jsonl"
    store = LabelStore(path)
    store.append(_label("a"))
    before = path.read_bytes()
    real_write = os.write

    def short(fd: int, data: bytes) -> int:
        return real_write(fd, data[:7])

    monkeypatch.setattr(os, "write", short)
    with pytest.raises(OSError, match="short write"):
        store.append(_label("b"))
    monkeypatch.undo()
    assert path.read_bytes() == before
    store.append(_label("c"))
    assert [lb.item_digest for lb in store.latest()] == ["a", "c"]


@pytest.mark.parametrize("sep", ["\u2028", "\u2029", "\u0085"])
def test_a_pasted_unicode_line_separator_never_breaks_the_label_file(
    tmp_path: Path, sep: str
) -> None:
    """Integrator fix of review P1: JSON writes these raw and str.splitlines() splits on them."""
    store = LabelStore(tmp_path / "labels.jsonl")
    first = Label(
        item_digest="d1",
        item_id="q1",
        kind=LabelKind.criteria,
        scores={Criterion.relevance: 2},
        note=f"pasted from notion{sep}second line",
        batch_id="b1",
        labelled_at=T0,
    )
    store.append(first)
    store.append(first.model_copy(update={"item_digest": "d2", "item_id": "q2", "note": "ok"}))
    notes = sorted(lb.note for lb in store.latest())
    assert notes == ["ok", f"pasted from notion{sep}second line"]


def test_a_run_file_with_a_unicode_line_separator_in_an_answer_reads_back(tmp_path: Path) -> None:
    from ragforge_client.evals.runfile import RunWriter, read_run

    golden = read_run(FIXTURES / "run-generator.jsonl")
    row = golden.rows[0]
    assert row.output is not None
    odd = row.model_copy(update={"output": row.output.model_copy(update={"answer": "a\u2028b"})})
    path = tmp_path / "run.jsonl"
    with RunWriter(path, golden.header) as writer:
        writer.row(odd)
        writer.finish()
    back = read_run(path)
    assert back.rows[0].output == odd.output
