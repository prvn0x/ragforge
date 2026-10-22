"""EV-JUDGE: the private D3 builder on synthetic v2 rows built from the D0 demo's clean answers
and chunk text (no private data), and the stratified calibration sample."""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from collections import Counter
from pathlib import Path
from types import ModuleType

import pytest
from ev_judge_support import (
    DEMO,
    T0,
    FakeChunkSource,
    answer_row,
    chunk_for,
    demo_chunks,
    header,
)
from hypothesis import given, settings
from hypothesis import strategies as st

from ragforge_client.client import RagforgeClient
from ragforge_client.evals.datasets import (
    CalibrationCandidate,
    ChunkSource,
    build_d3,
    calibration_candidates,
    calibration_sample,
    check_validity,
    item_digest_of,
    load_verifier_set,
    planted_span,
)
from ragforge_client.evals.runfile import RunFile
from ragforge_client.evals.schema import (
    EXPECTED_LAYER,
    AnswerOutput,
    ErrorClass,
    Label,
    LabelKind,
    Mutation,
    Question,
    QuestionSet,
    RunError,
    Split,
    VerifierSet,
)

DEMO_VERIFIER = VerifierSet.model_validate(json.loads((DEMO / "verifier.json").read_text()))
CLEAN = [i for i in DEMO_VERIFIER.items if i.mutation is None]


def _build_demo() -> ModuleType:
    spec = importlib.util.spec_from_file_location("build_demo", DEMO / "build_demo.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["build_demo"] = module  # its dataclasses look their module up while defined
    spec.loader.exec_module(module)
    return module


def test_planted_span_reproduces_every_demo_item_exactly() -> None:
    """The private builder's span algorithm IS D0's: for every planted demo spec it yields the
    committed answer and span byte for byte (so a private span means what a demo span means),
    and every demo item also passes the private builder's checks, which are at least as strict
    (D0 leaves the coverage-layer checks structural; here they are mechanical)."""
    demo = _build_demo()
    by_id = {i.id: i for i in DEMO_VERIFIER.items}
    chunks = demo_chunks()
    for pid, source, _mutation, old, new, check in demo.PLANTED:
        mutated, span = planted_span(by_id[source].answer, old, new)
        assert mutated == by_id[pid].answer, pid
        assert span == by_id[pid].span, pid
        texts = [chunk_for(chunks, k).text for k in by_id[source].shown]
        if check in (
            "token_absent_from_shown",
            "number_absent_from_shown",
            "entity_absent_from_cited",
        ):
            assert check_validity(check, new, texts) is demo.validity_ok(check, new, texts), pid
        elif check == "no_label_no_attribution":
            assert check_validity(check, new.removeprefix(old).strip(), texts), pid
        elif check == "new_page_lacks_content":
            sentence = by_id[source].answer.split(". ")[-1]
            assert check_validity(check, sentence, texts, target_text=texts[1]), pid
        elif check == "line_in_source_block":
            assert check_validity(check, old, texts), pid


def _the_sdk_is_a_chunk_source(client: RagforgeClient) -> ChunkSource:
    return client  # mypy proves the real SDK satisfies the builder's port


# ------------------------------------------------------------------ synthetic private runs
def _questions(split: Split = Split.dev) -> QuestionSet:
    return QuestionSet(
        name="private",
        questions=[
            Question(
                id=i.id,
                split=split,
                question=i.question,
                relevance={f"file:{i.shown[0].page.key}": 2},
            )
            for i in CLEAN
        ],
    )


def _rows_and_labels(statuses: list[str] | None = None) -> tuple[RunFile, list[Label]]:
    chunks = demo_chunks()
    rows, labels = [], []
    for n, item in enumerate(CLEAN):
        shown = [chunk_for(chunks, k) for k in item.shown]
        status = statuses[n] if statuses else "verified"
        rows.append(answer_row(item.id, item.answer, shown, status=status))
        labels.append(
            Label(
                item_digest=item_digest_of(item.question, item.answer, [c.text for c in shown]),
                item_id=item.id,
                kind=LabelKind.answer,
                correct=True,
                invented=False,
                batch_id="b1",
                labelled_at=T0,
            )
        )
    return RunFile(Path("r.jsonl"), header(), rows, None), labels


async def test_d3_on_synthetic_rows_gives_demo_style_items_that_pass_their_checks(
    tmp_path: Path,
) -> None:
    run, labels = _rows_and_labels()
    source = FakeChunkSource(demo_chunks())
    built = await build_d3([run], _questions(), labels, source)
    vs = built.verifier
    clean = {i.id: i for i in vs.items if i.mutation is None}
    planted = [i for i in vs.items if i.mutation is not None]
    assert len(clean) == len(CLEAN)
    assert built.skipped == {}
    assert len(planted) >= 2 * len(clean)
    assert len({i.mutation for i in planted}) == len(Mutation)  # every type, like D0
    texts = {c.key.model_dump_json(): c.text for c in demo_chunks()}
    for item in planted:
        src = clean[item.source_id]
        assert item.shown == src.shown
        assert item.question == src.question
        assert item.span is not None
        assert item.validity_check is not None
        assert item.answer != src.answer
        assert item.expected_layer is EXPECTED_LAYER[item.mutation]  # type: ignore[index]
        assert item.owner_reviewed is (item.mutation is not Mutation.negation_flip)
        shown_texts = [texts[k.model_dump_json()] for k in item.shown]
        marked = item.answer[item.span.start : item.span.end]
        if item.validity_check in ("token_absent_from_shown", "number_absent_from_shown"):
            assert check_validity(item.validity_check, marked, shown_texts), item.id
        if item.mutation is Mutation.entity_swap:
            assert not any(marked in t for t in shown_texts), item.id
        if item.mutation is Mutation.fabricated_sentence:
            assert "[S" not in marked
            assert marked in item.answer
            assert marked not in src.answer
    # what a runner writes, a loader reads
    path = tmp_path / "verifier.json"
    path.write_text(vs.model_dump_json())
    assert load_verifier_set(path) == vs
    assert {len(c) for c in source.calls}
    assert all(len(c) <= 100 for c in source.calls)


async def test_d3_selects_on_owner_labels_never_on_the_verdict() -> None:
    statuses = [
        "unverified",
        "partial",
        "verified",
        "unverified",
        "verified",
        "partial",
        "verified",
        "partial",
    ]
    run, labels = _rows_and_labels(statuses)
    wrong = {labels[0].item_digest: labels[0].model_copy(update={"correct": False})}
    invented = {labels[1].item_digest: labels[1].model_copy(update={"invented": True})}
    kept = [wrong.get(lb.item_digest) or invented.get(lb.item_digest) or lb for lb in labels]
    built = await build_d3([run], _questions(), kept, FakeChunkSource(demo_chunks()))
    sources = {i.question for i in built.verifier.items if i.mutation is None}
    assert sources == {i.question for i in CLEAN[2:]}  # unverified rows labelled correct stay
    assert built.skipped == {"not_owner_correct": 2}


async def test_d3_never_uses_heldout2_errors_or_a_changed_note() -> None:
    run, labels = _rows_and_labels()
    chunks = demo_chunks()
    first = chunk_for(chunks, CLEAN[0].shown[0])
    tampered = FakeChunkSource(chunks, tamper={first.chunk_id: first.text + " edited"})
    errored = run.rows[1].model_copy(
        update={"output": None, "error": RunError(cls=ErrorClass.ours, message="boom")}
    )
    run = RunFile(run.path, run.header, [run.rows[0], errored, *run.rows[2:]], None)
    built = await build_d3([run], _questions(), labels, tampered)
    stale = sum(1 for n, i in enumerate(CLEAN) if first.key in i.shown and n != 1)  # 1 errored
    assert built.skipped == {"error_or_stale": 1, "stale_chunk": stale}
    held = await build_d3([run], _questions(Split.heldout2), labels, FakeChunkSource(chunks))
    assert held.verifier.items == []
    assert held.skipped["heldout2"] == len(CLEAN)


@pytest.mark.parametrize("lie", [False, True])
async def test_note_text_that_does_not_match_its_digest_is_never_used(lie: bool) -> None:
    """Whether the server reports the new digest or (lying, or a bug) the old one for edited text,
    the row is stale: a validity check against text the model never saw proves nothing."""
    run, labels = _rows_and_labels()
    chunks = demo_chunks()
    first = chunk_for(chunks, CLEAN[2].shown[0])
    source = FakeChunkSource(chunks, tamper={first.chunk_id: first.text + " edited"}, lie=lie)
    built = await build_d3([run], _questions(), labels, source)
    assert built.skipped == {"stale_chunk": sum(first.key in i.shown for i in CLEAN)}


async def test_a_label_on_a_different_answer_does_not_apply() -> None:
    run, labels = _rows_and_labels()
    edited = []
    for row in run.rows:
        assert isinstance(row.output, AnswerOutput)
        output = row.output.model_copy(update={"answer": row.output.answer + " Also X."})
        edited.append(row.model_copy(update={"output": output}))
    run = RunFile(run.path, run.header, edited, None)
    built = await build_d3([run], _questions(), labels, FakeChunkSource(demo_chunks()))
    assert built.verifier.items == []


def _second_answers(run: RunFile) -> tuple[RunFile, list[Label]]:
    """Another run answering the same questions differently (another arm), all owner-correct."""
    rows, labels = [], []
    chunks = demo_chunks()
    for item, row in zip(CLEAN, run.rows, strict=True):
        assert isinstance(row.output, AnswerOutput)
        answer = row.output.answer + " See the notes."
        output = row.output.model_copy(update={"answer": answer})
        rows.append(row.model_copy(update={"output": output}))
        texts = [chunk_for(chunks, k).text for k in item.shown]
        labels.append(
            Label(
                item_digest=item_digest_of(item.question, answer, texts),
                item_id=item.id,
                kind=LabelKind.answer,
                correct=True,
                invented=False,
                batch_id="b2",
                labelled_at=T0,
            )
        )
    return RunFile(Path("r2.jsonl"), header("r2"), rows, None), labels


async def test_d3_keeps_one_clean_source_per_question_across_runs() -> None:
    """The bootstrap clusters on source_id as the source QUESTION (ADR-0017 §3 E-V): two runs
    answering one question must not become two clusters. The pick is the smallest digest, so it
    does not depend on run order, and a repeat of the same answer (an A/A run) is one item."""
    run, labels = _rows_and_labels()
    other, other_labels = _second_answers(run)
    source = FakeChunkSource(demo_chunks())
    both = await build_d3([run, other, run], _questions(), labels + other_labels, source)
    flipped = await build_d3([other, run], _questions(), labels + other_labels, source)
    assert both.verifier == flipped.verifier
    clean = [i for i in both.verifier.items if i.mutation is None]
    assert len(clean) == len(CLEAN)  # one per question, not one per answer
    assert both.skipped == {"duplicate_question": 2 * len(CLEAN)}
    assert flipped.skipped == {"duplicate_question": len(CLEAN)}
    by_question = {i.question: i for i in clean}
    chunks = demo_chunks()
    for item in CLEAN:
        texts = [chunk_for(chunks, k).text for k in item.shown]
        a = item_digest_of(item.question, item.answer, texts)
        b = item_digest_of(item.question, item.answer + " See the notes.", texts)
        kept = item.answer if a < b else item.answer + " See the notes."
        assert by_question[item.question].answer == kept
    sources = {i.id for i in clean}
    assert {i.source_id for i in both.verifier.items} == sources  # every plant on a kept one


# ------------------------------------------------------------------ calibration sample
_strata = st.tuples(
    st.sampled_from(["correct", "incorrect", "ungraded"]),
    st.sampled_from(["verified", "partial", "unverified"]),
)


@st.composite
def _candidates(draw: st.DrawFn) -> list[CalibrationCandidate]:
    rows = draw(st.lists(st.tuples(_strata, st.sampled_from(list(Split))), max_size=150))
    return [
        CalibrationCandidate(f"q{i}", f"{i:064x}", split, tier, verdict)
        for i, ((tier, verdict), split) in enumerate(rows)
    ]


@settings(max_examples=200)
@given(_candidates(), st.integers(0, 80), st.integers(0, 3))
def test_the_sample_is_stratified_reproducible_and_never_held_out(
    candidates: list[CalibrationCandidate], n: int, seed: int
) -> None:
    eligible = [c for c in candidates if c.split is not Split.heldout2]
    sample = calibration_sample(candidates, n=n, seed=seed)
    assert len(sample) == min(n, len(eligible))
    assert all(c.split is not Split.heldout2 for c in sample)
    assert len({c.item_digest for c in sample}) == len(sample)
    assert sample == calibration_sample(list(reversed(candidates)), n=n, seed=seed)
    sizes = Counter(c.stratum for c in eligible)
    got = Counter(c.stratum for c in sample)
    if len(sample) >= len(sizes):  # every stratum represented when n allows
        assert set(got) == set(sizes)
    # proportional within rounding plus the floor of one: no stratum with room left is starved,
    # and none takes more than its share unless another stratum ran out of items
    capped = any(got[s] == size for s, size in sizes.items())
    for stratum, size in sizes.items():
        assert got[stratum] <= size
        quota = len(sample) * size / len(eligible)
        if got[stratum] < size:
            assert got[stratum] >= math.floor(quota) - 1, (stratum, got, sizes)
        if not capped:
            assert got[stratum] <= math.ceil(quota) + 1, (stratum, got, sizes)


def test_the_sample_takes_50_by_default_and_skips_labelled_items() -> None:
    cands = [
        CalibrationCandidate(
            f"q{i}",
            f"{i:064x}",
            Split.dev,
            "correct" if i % 3 else "incorrect",
            "verified" if i % 2 else "partial",
        )
        for i in range(120)
    ]
    sample = calibration_sample(cands)
    assert len(sample) == 50
    counts = Counter(c.stratum for c in sample)
    pool = Counter(c.stratum for c in cands)
    for stratum, k in counts.items():
        assert abs(k - 50 * pool[stratum] / 120) <= 1  # proportional within rounding
    done = frozenset(c.item_digest for c in sample[:10])
    assert not done & {c.item_digest for c in calibration_sample(cands, exclude_digests=done)}
    assert calibration_sample(cands, seed=1) != sample  # the seed matters


def test_a_tiny_stratum_still_gets_one_item() -> None:
    big = [
        CalibrationCandidate(f"b{i}", f"{i:064x}", Split.dev, "correct", "verified")
        for i in range(100)
    ]
    rare = CalibrationCandidate("r", "f" * 64, Split.dev, "incorrect", "unverified")
    sample = calibration_sample([*big, rare])
    assert len(sample) == 50
    assert rare in sample


def test_candidates_come_from_answered_rows_with_tier_and_verdict() -> None:
    run, _ = _rows_and_labels(
        [
            "verified",
            "partial",
            "unverified",
            "verified",
            "verified",
            "partial",
            "verified",
            "partial",
        ]
    )
    graded = [
        r.model_copy(update={"grades": {"correct": i % 2 == 0}}) for i, r in enumerate(run.rows)
    ]
    run = RunFile(run.path, run.header, graded, None)
    qs = _questions()
    cands = calibration_candidates([run, run], qs, lambda row: f"{row.item_id:0>64}"[-64:])
    assert len(cands) == len(CLEAN)  # duplicates across runs kept once
    assert cands[0].stratum == ("correct", "verified")
    assert cands[1].stratum == (
        "incorrect",
        "partial",
    )
    assert calibration_candidates([run], qs, lambda row: None) == []


async def test_d3_uses_the_newest_label_even_when_given_all_of_them() -> None:
    """Passed `store.all()` instead of `.latest()`, an old "correct" that the owner later
    overruled must not admit the answer as a clean source."""
    run, labels = _rows_and_labels()
    overruled = labels[0].model_copy(update={"correct": False})
    built = await build_d3(
        [run], _questions(), [*labels, overruled], FakeChunkSource(demo_chunks())
    )
    sources = {i.question for i in built.verifier.items if i.mutation is None}
    assert sources == {i.question for i in CLEAN[1:]}
    assert built.skipped == {"not_owner_correct": 1}


def test_candidates_skip_stale_rows_and_runs_of_another_kind() -> None:
    from ragforge_client.evals.schema import EvalKind

    run, _ = _rows_and_labels()
    rows = [r.model_copy(update={"stale": i == 0}) for i, r in enumerate(run.rows)]
    current = RunFile(run.path, run.header, rows, None)
    app = RunFile(
        run.path, run.header.model_copy(update={"eval_kind": EvalKind.application}), rows, None
    )
    qs = _questions()

    def digest_for(row: object) -> str:
        return f"{getattr(row, 'item_id', '')}".rjust(64, "0")[-64:]

    cands = calibration_candidates([current, app], qs, digest_for)
    assert {c.item_id for c in cands} == {i.id for i in CLEAN[1:]}  # no stale row, no E-A run
    both = calibration_candidates([app], qs, digest_for, kinds=frozenset({EvalKind.application}))
    assert len(both) == len(CLEAN) - 1
