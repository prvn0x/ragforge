"""Review round 3 regressions: an arm that answers must-abstain questions can never meet the bar,
questions edited after a run are flagged (and held-out stays hidden when a question moves), a
server that never sent `Citation.inferred` yields "unknown" provenance, server text is cleaned,
the held-out reveal stamp survives later sessions, and the swap baseline is never a mid-run read."""

from __future__ import annotations

import io
from collections.abc import Callable, Iterator
from dataclasses import replace
from pathlib import Path
from uuid import uuid4

from ev_support import (
    GOOD_EOS,
    GOOD_LRU,
    GOOD_RAFT,
    FakeChat,
    citation,
    perfect_run_outcomes,
    write_questions,
    write_run,
)

from ragforge_client.evalrun import (
    RUBRIC,
    Completeness,
    InventedStats,
    Label,
    Outcome,
    ResourcePeaks,
    ResourceSample,
    answer_digest,
    ask_one,
    compare_runs,
    fmt_invented,
    heldout_revealed_at,
    judge_bar,
    label_run,
    load_labels,
    load_questions,
    load_run,
    question_digest,
    run_completeness,
    save_labels,
    score_runs,
    summarize,
)
from ragforge_contracts.api import Citation, Problem
from ragforge_contracts.sse import CitationEvent, ErrorEvent, decode_sse_data, encode_sse

FULL = Completeness(asked=4, expected=4, finished=True)
ANSWERED_OFF = "Use 70% hydration."  # a plausible answer to a question the notes cannot answer


def _script(*answers: str) -> Callable[[str], str]:
    it: Iterator[str] = iter(answers)

    def prompt(_: str) -> str:
        try:
            return next(it)
        except StopIteration:
            raise EOFError from None

    return prompt


def _fingerprinted(path: Path) -> list[Outcome]:
    """The perfect run as the current `run` records it: every row carries its question print."""
    by_id = {q.id: q for q in load_questions(path)}
    return [
        replace(o, question_split=by_id[o.id].split, question_sha256=question_digest(by_id[o.id]))
        for o in perfect_run_outcomes()
    ]


def _answers_must_abstain(outcomes: list[Outcome]) -> list[Outcome]:
    return [
        replace(o, final=ANSWERED_OFF, draft=ANSWERED_OFF) if o.id == "dev-off" else o
        for o in outcomes
    ]


def _every_answer_ok(run_path: Path) -> None:
    texts = {"dev-eos": GOOD_EOS, "dev-lru": GOOD_LRU, "secret-heldout-raft": GOOD_RAFT}
    texts["dev-off"] = ANSWERED_OFF  # even a human "ok" on it must not rescue the arm
    save_labels(run_path, {k: Label("ok", answer_digest(v), "t") for k, v in texts.items()})


# ------------------------------------------------------------------ P1: must-abstain answered
def test_answering_a_must_abstain_question_can_never_meet_the_bar(tmp_path: Path) -> None:
    """Reviewer's reproduction: dev-off answered, every answer labelled ok, was '**yes**'."""
    qpath = write_questions(tmp_path)
    questions = load_questions(qpath)
    run = write_run(
        tmp_path / "r.jsonl",
        _answers_must_abstain(_fingerprinted(qpath)),
        footer={"complete": True, "resources": {}},
    )
    _every_answer_ok(run)
    s = summarize(questions, load_run(run).outcomes, load_labels(run))
    assert s["correct"] == 100.0  # every answerable question is right: only invented% catches it
    inv: InventedStats = s["invented"]
    assert (inv.answered, inv.labelled, inv.invented, inv.forced) == (4, 4, 1, 1)
    verdict = judge_bar(s, run_completeness(load_run(run), questions))
    assert verdict.met is False
    assert "invented 25.0>2 (incl. 1 answered must-abstain)" in verdict.reasons
    out = io.StringIO()
    compare_runs([str(run)], str(qpath), out=out)
    row = next(line for line in out.getvalue().splitlines() if line.startswith("| arm | all |"))
    assert "**yes**" not in row
    assert "25 (incl. 1 answered must-abstain)" in row


def test_must_abstain_answer_needs_no_label_to_count(tmp_path: Path) -> None:
    """Without any human label the forced answer is still invented, so the partial lower bound
    already fails the bar: the owner is not asked to label an arm that cannot win."""
    qpath = write_questions(tmp_path)
    s = summarize(load_questions(qpath), _answers_must_abstain(_fingerprinted(qpath)))
    inv: InventedStats = s["invented"]
    assert (inv.labelled, inv.invented, inv.status) == (1, 1, "partial 1/4")
    assert judge_bar(s, FULL).met is False


def test_abstaining_arm_is_unaffected_by_the_rule(tmp_path: Path) -> None:
    qpath = write_questions(tmp_path)
    run = write_run(tmp_path / "r.jsonl", _fingerprinted(qpath), footer={"complete": True})
    _every_answer_ok(run)
    s = summarize(load_questions(qpath), load_run(run).outcomes, load_labels(run))
    assert s["invented"] == InventedStats(answered=3, labelled=3, invented=0, unsure=0)
    assert judge_bar(s, FULL).met is True


def test_label_walk_shows_a_rubric_and_never_asks_about_must_abstain_answers(
    tmp_path: Path,
) -> None:
    qpath = write_questions(tmp_path)
    run = write_run(tmp_path / "r.jsonl", _answers_must_abstain(_fingerprinted(qpath)))
    out = io.StringIO()
    labels = label_run(
        run, load_questions(qpath), reveal=False, relabel=False, prompt=_script("o", "o"), out=out
    )
    printed = out.getvalue()
    assert RUBRIC in printed
    assert "Sourdough hydration?" not in printed
    assert ANSWERED_OFF not in printed
    assert "1 answered must-abstain question(s) counted as invented" in printed
    assert set(labels) == {"dev-eos", "dev-lru"}  # two prompts were enough: dev-off never asked


def test_fmt_invented_says_why_the_rate_is_not_zero() -> None:
    assert fmt_invented(InventedStats(4, 4, 1, 0, forced=1)) == "25 (incl. 1 answered must-abstain)"
    # review round 6: the headline is the bar's conservative number, unsure counted as invented
    assert fmt_invented(InventedStats(13, 13, 0, 3)) == "23 (0 if 3 unsure are ok)"


# ------------------------------------------------------------------ question edited after the run
def _edit(qpath: Path, old: str, new: str) -> None:
    text = qpath.read_text(encoding="utf-8")
    assert old in text
    qpath.write_text(text.replace(old, new), encoding="utf-8")


def test_question_text_edited_after_the_run_fails_the_bar(tmp_path: Path) -> None:
    qpath = write_questions(tmp_path)
    run = write_run(tmp_path / "r.jsonl", _fingerprinted(qpath), footer={"complete": True})
    _every_answer_ok(run)
    _edit(qpath, "How does an LRU cache evict?", "How does an LFU cache evict?")
    questions = load_questions(qpath)
    s = summarize(questions, load_run(run).outcomes, load_labels(run))
    assert s["changed"] == ["dev-lru"]
    verdict = judge_bar(s, run_completeness(load_run(run), questions))
    assert verdict.met is False
    assert "1 question(s) changed since the run" in verdict.reasons
    out = io.StringIO()
    score_runs([str(run)], str(qpath), verbose=True, out=out)
    assert "warning: arm: 1 question(s) changed since run" in out.getvalue()
    assert "dev-lru" in out.getvalue()
    assert "(question changed since the run)" in out.getvalue()


def test_flipping_answerable_after_the_run_is_flagged(tmp_path: Path) -> None:
    qpath = write_questions(tmp_path)
    outcomes = _fingerprinted(qpath)
    _edit(qpath, 'question = "Sourdough hydration?"\nanswerable = false', "")
    _edit(qpath, '\n[[q]]\nid = "dev-off"\nsplit = "dev"\n', "\n")
    qpath.write_text(
        qpath.read_text(encoding="utf-8")
        + '\n[[q]]\nid = "dev-off"\nsplit = "dev"\nquestion = "Sourdough hydration?"\n'
        'pages = ["Bread"]\nfacts = [{ fact = "x", match = ["x"] }]\n',
        encoding="utf-8",
    )
    assert summarize(load_questions(qpath), outcomes)["changed"] == ["dev-off"]


def test_fact_and_page_corrections_are_not_flagged(tmp_path: Path) -> None:
    """Re-scoring after correcting expected facts/pages is the intended workflow."""
    qpath = write_questions(tmp_path)
    outcomes = _fingerprinted(qpath)
    _edit(qpath, 'pages = ["Caching"]', 'pages = ["Caching", "Cache Eviction"]')
    _edit(qpath, 'match = ["least recently"]', 'match = ["least recently", "lru"]')
    assert summarize(load_questions(qpath), outcomes)["changed"] == []


def test_legacy_rows_without_a_fingerprint_are_not_flagged(tmp_path: Path) -> None:
    qpath = write_questions(tmp_path)
    _edit(qpath, "How does an LRU cache evict?", "Reworded")
    assert summarize(load_questions(qpath), perfect_run_outcomes())["changed"] == []


def test_question_moved_from_heldout_to_dev_stays_hidden(tmp_path: Path) -> None:
    """Moving a question to dev after a run must not reveal its recorded held-out answer."""
    qpath = write_questions(tmp_path)
    run = write_run(tmp_path / "r.jsonl", _fingerprinted(qpath))
    _edit(
        qpath,
        'id = "secret-heldout-raft"\nsplit = "heldout"',
        'id = "secret-heldout-raft"\nsplit = "dev"',
    )
    questions = load_questions(qpath)
    score_out, label_out = io.StringIO(), io.StringIO()
    score_runs([str(run)], str(qpath), verbose=True, out=score_out)
    label_run(run, questions, reveal=False, relabel=False, prompt=_script("o", "o"), out=label_out)
    for printed in (score_out.getvalue(), label_out.getvalue()):
        assert "secret-heldout-raft" not in printed
        assert GOOD_RAFT not in printed
    assert "1 held-out row(s) hidden" in score_out.getvalue()
    assert "1 held-out answer(s) not shown" in label_out.getvalue()
    assert heldout_revealed_at(run) is None


# ------------------------------------------------------------------ provenance must be sent
async def test_citation_without_the_inferred_field_makes_provenance_unknown(
    tmp_path: Path,
) -> None:
    """A server build that predates `Citation.inferred` would make every citation look
    model-cited; the field's absence must read as unknown, never as False."""
    (q,) = [q for q in load_questions(write_questions(tmp_path)) if q.id == "dev-eos"]
    legacy = Citation(
        anchor="S1", document_id=uuid4(), title="Kafka Delivery", heading_path=[], url=None,
        snippet="",
    )  # fmt: skip
    out = await ask_one(FakeChat([CitationEvent(citation=legacy)]), [], q)
    assert (out.cited_titles, out.inferred_titles, out.model_titles()) == (
        ["Kafka Delivery"],
        None,
        None,
    )


async def test_one_legacy_citation_makes_the_whole_turn_unknown(tmp_path: Path) -> None:
    (q,) = [q for q in load_questions(write_questions(tmp_path)) if q.id == "dev-eos"]
    legacy = Citation.model_validate(citation("Caching").model_dump(exclude={"inferred"}))
    events = [CitationEvent(citation=citation("Kafka Delivery", inferred=True))]
    events.append(CitationEvent(citation=legacy))
    out = await ask_one(FakeChat(events), [], q)
    assert out.inferred_titles is None


def test_current_server_wire_format_carries_the_inferred_field() -> None:
    """Proves the rule above cannot misfire today: encode_sse serialises defaults too, so a
    current server's `inferred=False` arrives as an explicitly set field."""
    decoded = decode_sse_data(
        encode_sse(CitationEvent(citation=citation("Caching"))).decode().split("data: ", 1)[1]
    )
    assert isinstance(decoded, CitationEvent)
    assert "inferred" in decoded.citation.model_fields_set


# ------------------------------------------------------------------ server text is cleaned
async def test_error_event_title_is_cleaned(tmp_path: Path) -> None:
    (q,) = [q for q in load_questions(write_questions(tmp_path)) if q.id == "dev-eos"]
    problem = Problem(type="about:blank", title="Boom\x1b[2J", status=500)
    out = await ask_one(FakeChat([ErrorEvent(problem=problem)]), [], q)
    assert out.error == "500 Boom[2J"


def test_score_cleans_server_reported_model_names(tmp_path: Path) -> None:
    qpath = write_questions(tmp_path)
    rows = [replace(o, models=["evil|\x1b]0;x\x07m"]) for o in perfect_run_outcomes()]
    run = write_run(tmp_path / "r.jsonl", rows)
    out = io.StringIO()
    score_runs([str(run)], str(qpath), verbose=False, out=out)
    row = out.getvalue().splitlines()[2]
    assert "\x1b" not in row
    assert "\x07" not in row
    assert "evil\\|" in row  # a pipe cannot add a column


# ------------------------------------------------------------------ held-out reveal stamp
def test_reveal_stamp_survives_a_later_non_reveal_relabel_session(tmp_path: Path) -> None:
    questions = load_questions(write_questions(tmp_path))
    run = write_run(tmp_path / "r.jsonl", perfect_run_outcomes())
    label_run(run, questions, reveal=True, relabel=False, prompt=_script("o", "o", "o"))
    stamp = heldout_revealed_at(run)
    assert stamp is not None
    label_run(run, questions, reveal=False, relabel=True, prompt=_script("i", "i"))
    assert heldout_revealed_at(run) == stamp
    assert load_labels(run)["dev-eos"].verdict == "invented"  # the relabel session did save
    assert load_labels(run)["secret-heldout-raft"].verdict == "ok"


# ------------------------------------------------------------------ swap baseline
def test_failed_first_sample_leaves_no_baseline_rather_than_a_mid_run_one() -> None:
    peaks = ResourcePeaks()
    peaks.failures += 1  # what sample_until records when the first sample raises
    peaks.add(ResourceSample(9000.0, None))
    assert (peaks.swap_baseline_mb, peaks.swap_delta_mb) == (None, None)


def test_first_sample_without_swap_leaves_no_baseline() -> None:
    peaks = ResourcePeaks()
    for sample in (ResourceSample(None, 10.0), ResourceSample(9000.0, None)):
        peaks.add(sample)
    assert peaks.swap_baseline_mb is None


def test_first_good_sample_is_the_baseline() -> None:
    peaks = ResourcePeaks()
    for sample in (ResourceSample(8000.0, None), ResourceSample(9000.0, None)):
        peaks.add(sample)
    assert (peaks.swap_baseline_mb, peaks.swap_delta_mb) == (8000.0, 1000.0)


def test_compare_header_names_the_completeness_column_asked(tmp_path: Path) -> None:
    qpath = write_questions(tmp_path)
    run = write_run(tmp_path / "r.jsonl", perfect_run_outcomes(), footer={"complete": True})
    out = io.StringIO()
    compare_runs([str(run)], str(qpath), out=out)
    assert out.getvalue().startswith("| arm | split | asked | model |")
