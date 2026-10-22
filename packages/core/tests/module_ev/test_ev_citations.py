"""Model-cited vs server-inferred citations: recorded separately, credited the same for
"correct", and reported as separate rates so inference never masquerades as model skill."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from ev_support import (
    GOOD_EOS,
    GOOD_LRU,
    FakeChat,
    citation,
    outcome,
    write_questions,
    write_run,
)

from ragforge_client.evalrun import (
    Outcome,
    ask_one,
    grade,
    load_questions,
    load_run,
    summarize,
)
from ragforge_contracts.sse import CitationEvent, DoneEvent, TokenEvent, Usage


async def test_ask_one_records_provenance_and_arm_usage(tmp_path: Path) -> None:
    q = load_questions(write_questions(tmp_path))[0]
    fake = FakeChat(
        [
            TokenEvent(text=GOOD_EOS),
            CitationEvent(citation=citation("Kafka Delivery", inferred=True)),
            CitationEvent(citation=citation("Other", inferred=False)),
            CitationEvent(citation=citation("Other", inferred=True)),  # model-cited already
            DoneEvent(
                conversation_id=uuid4(),
                usage=[
                    Usage(
                        provider="gemini", model="g-flash", prompt_tokens=10, completion_tokens=3
                    ),
                    Usage(provider="gemini", model="g-flash", prompt_tokens=5, completion_tokens=2),
                ],
                prompt_version="p7",
            ),
        ]
    )
    out = await ask_one(fake, [], q)
    assert out.cited_titles == ["Kafka Delivery", "Other"]
    assert out.inferred_titles == ["Kafka Delivery"]
    assert out.model_titles() == ["Other"]
    assert (out.providers, out.models, out.prompt_version) == (["gemini"], ["g-flash"], "p7")
    assert (out.prompt_tokens, out.completion_tokens) == (15, 5)


async def test_a_later_model_citation_upgrades_an_inferred_one(tmp_path: Path) -> None:
    q = load_questions(write_questions(tmp_path))[0]
    fake = FakeChat(
        [
            CitationEvent(citation=citation("Kafka Delivery", inferred=True)),
            CitationEvent(citation=citation("Kafka Delivery", inferred=False)),
        ]
    )
    out = await ask_one(fake, [], q)
    assert out.inferred_titles == []
    assert out.model_titles() == ["Kafka Delivery"]


def test_correct_counts_inferred_citations_but_model_only_rate_does_not(tmp_path: Path) -> None:
    eos, lru, *_ = load_questions(write_questions(tmp_path))
    by_model = grade(eos, outcome("dev-eos", GOOD_EOS, model=["Kafka Delivery"]))
    by_server = grade(lru, outcome("dev-lru", GOOD_LRU, inferred=["Caching"]))
    assert (by_model.correct, by_model.correct_model, by_model.cite_source) == (True, True, "model")
    assert (by_server.correct, by_server.correct_model, by_server.cite_source) == (
        True,
        False,
        "inferred",
    )


def test_model_citation_of_expected_page_wins_over_inferred_one(tmp_path: Path) -> None:
    eos = load_questions(write_questions(tmp_path))[0]
    g = grade(eos, outcome("dev-eos", GOOD_EOS, model=["Kafka Delivery"], inferred=["Other"]))
    assert g.cite_source == "model"
    # An inferred citation of a WRONG page does not make the answer correct.
    wrong = grade(eos, outcome("dev-eos", GOOD_EOS, inferred=["Other"]))
    assert (wrong.correct, wrong.content_correct, wrong.cite_source) == (False, True, "none")


def test_summary_reports_both_rates_for_a_mixed_run(tmp_path: Path) -> None:
    questions = load_questions(write_questions(tmp_path))
    outcomes = [
        outcome("dev-eos", GOOD_EOS, model=["Kafka Delivery"]),
        outcome("dev-lru", GOOD_LRU, inferred=["Caching"]),
        outcome("secret-heldout-raft", "No idea.", inferred=["Consensus"]),
    ]
    s = summarize(questions, outcomes)
    assert round(s["correct"]) == 67  # 2 of 3 answerable, one via inference
    assert round(s["correct_model"]) == 33
    assert round(s["cited"]) == 100
    assert (round(s["cited_model"]), round(s["cited_inferred"])) == (33, 67)


def test_runs_recorded_before_the_split_report_provenance_as_unknown(tmp_path: Path) -> None:
    """Old run files have cited_titles but no inferred_titles: we must not guess "model"."""
    questions = load_questions(write_questions(tmp_path))
    legacy = Outcome(id="dev-eos", final=GOOD_EOS, cited_titles=["Kafka Delivery"])
    g = grade(questions[0], legacy)
    assert (g.correct, g.cite_source, g.correct_model) == (True, "unknown", None)
    s = summarize(questions, [legacy])
    assert s["correct"] == 100.0
    assert s["correct_model"] is None
    assert s["cited_model"] is None


def test_load_run_tolerates_unknown_fields_and_reads_the_footer(tmp_path: Path) -> None:
    path = write_run(
        tmp_path / "r.jsonl",
        [outcome("dev-eos", GOOD_EOS)],
        footer={"complete": True, "resources": {"swap_used_mb": 10.0}},
    )
    text = path.read_text(encoding="utf-8").replace('"id":', '"from_a_newer_writer": 1, "id":', 1)
    path.write_text(text, encoding="utf-8")
    run = load_run(path)
    assert [o.id for o in run.outcomes] == ["dev-eos"]
    assert run.footer == {"complete": True, "resources": {"swap_used_mb": 10.0}}
