"""`ragforge-eval`: question-set validation, fact matching, grading and the summary table."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from ragforge_client.evalrun import (
    ABSTAIN_PREFIX,
    MODEL_REFUSAL,
    Fact,
    Outcome,
    Question,
    fact_matched,
    grade,
    load_questions,
    mcnemar_exact,
    paired,
    percentile,
    read_run,
    summarize,
    wilson,
)

QUESTIONS = """
[[q]]
id = "eos"
split = "dev"
question = "Idempotent producer vs exactly-once?"
pages = ["Kafka Delivery Guarantees"]
facts = [
  { fact = "retries", match = ["Retries"] },
  { fact = "transactions", match = ["transaction"] },
  { fact = "read committed", match = ["read_committed"] },
]

[[q]]
id = "off"
split = "heldout"
question = "Sourdough hydration?"
answerable = false
"""


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "questions.toml"
    path.write_text(text)
    return path


def test_load_questions_parses_and_lowercases_matches(tmp_path: Path) -> None:
    eos, off = load_questions(_write(tmp_path, QUESTIONS))
    assert eos.facts[0].match == ("retries",)
    assert eos.pages == ("Kafka Delivery Guarantees",)
    assert not off.answerable
    assert off.facts == ()


@pytest.mark.parametrize(
    ("text", "error"),
    [
        ('[[q]]\nid="a"\nsplit="train"\nquestion="?"\nanswerable=false', "split must be"),
        ('[[q]]\nid="a"\nsplit="dev"\nquestion="?"\npages=["P"]', "needs pages"),
        (
            '[[q]]\nid="a"\nsplit="dev"\nquestion="?"\nanswerable=false\n'
            '[[q]]\nid="a"\nsplit="dev"\nquestion="?"\nanswerable=false',
            "duplicate",
        ),
    ],
)
def test_load_questions_rejects_malformed_sets(tmp_path: Path, text: str, error: str) -> None:
    with pytest.raises(ValueError, match=error):
        load_questions(_write(tmp_path, text))


def test_bare_numbers_do_not_match_inside_other_numbers() -> None:
    eight = Fact("threshold 8", ("8",))
    assert fact_matched(eight, "a chain longer than 8 nodes")
    assert not fact_matched(eight, "context of 2048 tokens")
    assert fact_matched(Fact("fencing", ("fenc",)), "the producer is FENCED off")


def _question() -> Question:
    return Question(
        id="eos",
        split="dev",
        question="?",
        pages=("Kafka Delivery Guarantees",),
        facts=(
            Fact("retries", ("retries",)),
            Fact("transactions", ("transaction",)),
            Fact("read committed", ("read_committed",)),
        ),
    )


def test_grade_needs_two_thirds_of_facts_and_an_expected_citation() -> None:
    q = _question()
    text = "Idempotence handles retries; EOS adds a transaction."
    good = grade(
        q, Outcome(id="eos", draft=text, final=text, cited_titles=["Kafka Delivery Guarantees"])
    )
    assert (good.facts_hit, good.facts_needed, good.correct) == (2, 2, True)

    uncited = grade(q, Outcome(id="eos", draft=text, final=text, cited_titles=["Other page"]))
    assert uncited.content_correct
    assert not uncited.correct

    thin = grade(
        q,
        Outcome(id="eos", final="It handles retries.", cited_titles=["Kafka Delivery Guarantees"]),
    )
    assert not thin.content_correct


def test_grade_credits_the_draft_when_the_verifier_replaced_it() -> None:
    draft = "retries, transaction, read_committed"
    g = grade(_question(), Outcome(id="eos", draft=draft, final=ABSTAIN_PREFIX, replaced=True))
    assert not g.answered
    assert not g.correct
    assert g.draft_content_correct


def test_unanswerable_is_correct_only_when_abstaining_without_error() -> None:
    q = Question(id="off", split="heldout", question="?", answerable=False)
    assert grade(q, Outcome(id="off", final=f"{ABSTAIN_PREFIX}\n- closest")).correct
    assert not grade(q, Outcome(id="off", final="Use 75% hydration.")).correct
    assert not grade(q, Outcome(id="off", final=ABSTAIN_PREFIX, error="504 timeout")).correct


def test_percentile_is_nearest_rank() -> None:
    values = [float(v) for v in range(1, 16)]  # 15 answers
    assert percentile(values, 50) == 8.0
    assert percentile(values, 95) == 15.0  # with n=15, p95 is the slowest answer
    assert percentile([], 95) != percentile([], 95)  # NaN


def test_summarize_round_trips_a_recorded_run(tmp_path: Path) -> None:
    questions = load_questions(_write(tmp_path, QUESTIONS))
    run = tmp_path / "run.jsonl"
    rows = [
        {"header": {"label": "arm", "split": "all", "started": "t", "url": "u"}},
        {
            "id": "eos",
            "draft": "retries + transaction",
            "final": "retries + transaction",
            "cited_titles": ["Kafka Delivery Guarantees"],
            "unsupported_claims": ["x"],
            "models": ["llama3.2:3b"],
            "first_token_s": 2.0,
            "total_s": 9.0,
        },
        {"id": "off", "final": ABSTAIN_PREFIX, "total_s": 1.0, "tools": ["search_notes"]},
    ]
    run.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    header, outcomes = read_run(run)
    s = summarize(questions, outcomes)
    assert header["label"] == "arm"
    assert (s["correct"], s["abstain_ok"], s["flagged"]) == (100.0, 100.0, 100.0)
    assert (s["p95_s"], s["ttft_p50_s"], s["tool_turns"]) == (9.0, 2.0, 1)
    assert s["models"] == ["llama3.2:3b"]


def test_the_models_own_refusal_is_an_abstain_not_an_invented_answer() -> None:
    """Review P1: answer.md tells the model to reply MODEL_REFUSAL; grading it as an answer
    failed must-abstain questions and counted them as invented."""
    q = Question(id="off", split="dev", question="?", answerable=False)
    assert grade(q, Outcome(id="off", final=f"  {MODEL_REFUSAL}")).correct
    s = summarize([q], [Outcome(id="off", final=MODEL_REFUSAL)])
    assert s["abstain_n"] == (1, 1)
    assert s["invented"].forced == 0


def test_abstain_phrases_match_the_prompt_templates() -> None:
    """Drift guard: the client cannot import the server's templates, so pin them here."""
    templates = Path(__file__).parents[1] / "src/ragforge_core/prompts/templates"
    assert ABSTAIN_PREFIX in (templates / "abstain.md").read_text()
    assert f'"{MODEL_REFUSAL}"' in (templates / "answer.md").read_text()


def test_fact_keywords_match_whole_words_with_inflections() -> None:
    """Review P2: "ack" matched "stack" and "cas" matched "case"."""
    ack = Fact("acks", ("ack",))
    cas = Fact("cas", ("cas",))
    weight = Fact("weighted", ("weight",))
    assert not fact_matched(ack, "a call stack")
    assert fact_matched(ack, "the broker sends an ACK")
    assert fact_matched(ack, "acks=all")
    assert not fact_matched(cas, "in that case")
    assert fact_matched(cas, "insert with CAS.")
    assert fact_matched(weight, "weighted load")
    assert not fact_matched(weight, "weightlifting")
    assert fact_matched(Fact("r", ("60-70",)), "about 60-70% of RAM")
    assert fact_matched(Fact("r", ("read_committed",)), "use `read_committed`")


def test_wilson_interval_shows_how_little_thirteen_questions_prove() -> None:
    low, high = wilson(10, 13)
    assert (round(low), round(high)) == (50, 92)
    assert all(math.isnan(v) for v in wilson(0, 0))  # no questions, no interval
    assert wilson(13, 13)[1] == 100.0


def test_paired_mcnemar_on_shared_answerable_questions() -> None:
    assert mcnemar_exact(0, 0) == 1.0
    assert round(mcnemar_exact(1, 5), 2) == 0.22  # the A0 vs A1 numbers from the review
    q = _question()
    other = Question(id="x", split="dev", question="?", pages=("P",), facts=(Fact("f", ("f",)),))
    hit = Outcome(id="eos", final="retries transaction", cited_titles=["Kafka Delivery Guarantees"])
    miss = Outcome(id="eos", final="nothing")
    base = summarize([q, other], [hit, Outcome(id="x", final="f", cited_titles=["P"])])["grades"]
    new = summarize([q], [miss])["grades"]
    assert paired(base, new) == (1, 1, 0, 1.0)
    assert paired(base, summarize([other], [])["grades"]) is None


@pytest.mark.parametrize(
    ("text", "is_abstain"),
    [
        (MODEL_REFUSAL, True),
        (MODEL_REFUSAL.replace("'", "\u2019"), True),  # curly apostrophe folded
        ("Your notes don't cover this [S1].", True),  # review round 3: as refusal_only reads it
        ("Your notes don't cover this question.", True),
        ("Your notes don't cover this. I hope this helps!", True),
        (f"{MODEL_REFUSAL} However, Raft compacts its log with snapshots.", False),
        (f"{ABSTAIN_PREFIX}\n\nThe closest pages I found:\n- Raft\n- [Kafka](https://x)", True),
        (f"{ABSTAIN_PREFIX}\n\nBut Raft uses snapshots.", False),
    ],
)
def test_abstain_is_the_whole_text_not_a_prefix(text: str, is_abstain: bool) -> None:
    """Review round 2, P2: a refusal followed by an answer is an answer (and can be invented)."""
    from ragforge_client.evalrun import abstained

    assert abstained(text) is is_abstain


def test_closest_header_matches_the_server() -> None:
    from ragforge_client.evalrun import _CLOSEST_HEADER

    nodes = Path(__file__).parents[1] / "src/ragforge_core/agent/nodes.py"
    assert f"\\n\\n{_CLOSEST_HEADER}\\n" in nodes.read_text()


def test_fact_digits_ignore_labels_and_decimals() -> None:
    """Review round 3, P2: "3" matched "[S3]" and "0.3"."""
    three = Fact("rf", ("3",))
    assert not fact_matched(three, "Kafka keeps 2 replicas [S3].")
    assert not fact_matched(three, "Kafka keeps 2 replicas \\[S3\\].")
    assert not fact_matched(three, "It uses 0.3 GB.")
    assert not fact_matched(three, "Version 3.5 is out.")
    assert fact_matched(three, "Replication factor is 3 [S1].")
    assert fact_matched(three, "It needs 3.")


def test_cli_explains_a_missing_question_set(tmp_path: Path) -> None:
    from ragforge_client.evalrun import run

    with pytest.raises(SystemExit, match="no question set at"):
        run(["--questions", str(tmp_path / "missing.toml"), "compare", "x.jsonl"])


def test_relabels_are_kept_in_an_append_only_history(tmp_path: Path) -> None:
    """Review round 7, P1: resolving an "unsure" must leave an auditable trail."""
    import json as _json

    from ragforge_client.evalrun import Label, labels_path, save_labels

    run = tmp_path / "r.jsonl"
    run.write_text("")
    save_labels(run, {"q1": Label("unsure", "aa", "t1"), "q2": Label("ok", "bb", "t1")})
    save_labels(run, {"q1": Label("ok", "aa", "t2"), "q2": Label("ok", "bb", "t2")})
    save_labels(run, {"q1": Label("invented", "aa", "t3"), "q2": Label("ok", "bb", "t3")})
    body = _json.loads(labels_path(run).read_text())
    common = {"from_answer": "aa", "to_answer": "aa"}
    assert body["history"] == [
        {"id": "q1", "from": "unsure", "to": "ok", "from_at": "t1", "to_at": "t2", **common},
        {"id": "q1", "from": "ok", "to": "invented", "from_at": "t2", "to_at": "t3", **common},
    ]


def test_label_history_refuses_a_malformed_record_and_keeps_the_reveal_stamp(
    tmp_path: Path,
) -> None:
    import json as _json

    from ragforge_client.evalrun import EvalSetupError, Label, labels_path, save_labels

    run = tmp_path / "r.jsonl"
    run.write_text("")
    save_labels(run, {"q1": Label("ok", "aa", "t1")}, heldout_revealed_at="2026-10-18T05:52Z")
    save_labels(run, {"q1": Label("ok", "aa", "t2")})  # no stamp passed: kept
    body = _json.loads(labels_path(run).read_text())
    assert body["heldout_revealed_at"] == "2026-10-18T05:52Z"
    labels_path(run).write_text(_json.dumps({**body, "history": "lost"}))
    with pytest.raises(EvalSetupError, match="malformed label history"):
        save_labels(run, {"q1": Label("invented", "aa", "t3")})


def test_relabelling_a_changed_answer_with_the_same_verdict_is_recorded(tmp_path: Path) -> None:
    """Review round 9, P2: the history logged only verdict changes."""
    import json as _json

    from ragforge_client.evalrun import Label, labels_path, save_labels

    run = tmp_path / "r.jsonl"
    run.write_text("")
    save_labels(run, {"q1": Label("ok", "aa", "t1")})
    save_labels(run, {"q1": Label("ok", "bb", "t2")})
    (entry,) = _json.loads(labels_path(run).read_text())["history"]
    assert (entry["from_answer"], entry["to_answer"], entry["from"], entry["to"]) == (
        "aa",
        "bb",
        "ok",
        "ok",
    )


async def test_extras_are_recorded_but_never_graded() -> None:
    """ADR-0016: an extra is kept for sampling; the answer text alone is graded."""
    from uuid import uuid4

    from ragforge_client.evalrun import ask_one
    from ragforge_contracts.sse import DoneEvent, ExtraEvent, TokenEvent

    class _Rf:
        async def chat(self, message: str, collection_ids: object):  # type: ignore[no-untyped-def]
            yield TokenEvent(text="Idempotence handles retries [S1].")
            yield ExtraEvent(kind="incident", text="Common failure pattern\\: max\\_poll\\.")
            yield DoneEvent(conversation_id=uuid4(), usage=[], prompt_version="v")

    q = _question()
    out = await ask_one(_Rf(), [], q)
    assert out.extras == ["incident: Common failure pattern: max_poll."]  # readable for labels
    assert "Common failure pattern" not in out.final
