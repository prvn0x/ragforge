"""EV-RUN: every runner against a fake SDK that replays the golden D0 fixtures.

Equivalence: a runner fed the fixture's own outputs must write the fixture's rows, row for row
(only fields the fake cannot reproduce, such as wall-clock timings, are left out), and a header
with the fixture header's configuration digest.
"""

from __future__ import annotations

import json
import stat
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid5

import pytest
from test_ev_run_fakes import (
    DEMO,
    FIXTURES,
    GOLDEN_JUDGE,
    GOLDEN_PROMPTS,
    GOLDEN_RETRIEVAL,
    NS,
    SNAPSHOT_ID,
    Crash,
    FakeEvalClient,
    FakeJudge,
    FakeSummarizer,
    api_error,
    comparable,
    golden,
    load_interview,
    load_questions,
    load_verifier,
    page_text,
)

from ragforge_client.evals.datasets import item_digest_of
from ragforge_client.evals.runfile import RunFile, RunFileError, header_digest, read_run
from ragforge_client.evals.runners import (
    COLD_QUESTIONS,
    Pacer,
    RetryPolicy,
    RunAborted,
    RunContext,
    answer_grades,
    cluster_order,
    final_path,
    find_resumable,
    judge_phase,
    key_id,
    load_judge_results,
    run_application,
    run_generator,
    run_interview,
    run_latency,
    run_retrieval,
    run_verifier,
    staging_path,
    summary_path,
)
from ragforge_client.evals.schema import (
    E_G_CRITERIA,
    AnswerOutput,
    ErrorClass,
    EvalKind,
    Fact,
    GradeOutput,
    Question,
    QuestionOutput,
    RunRow,
    Split,
    VerifyOutput,
)
from ragforge_contracts.api import ChunkKey

T0 = datetime(2026, 10, 21, 12, 0, tzinfo=UTC)
COLD = COLD_QUESTIONS


class Sleeps:
    def __init__(self) -> None:
        self.calls: list[float] = []

    async def __call__(self, seconds: float) -> None:
        self.calls.append(seconds)


def make_ctx(tmp_path: Path, client: FakeEvalClient, **changes: object) -> RunContext:
    runs = tmp_path / "runs"
    runs.mkdir(exist_ok=True)
    ctx = RunContext(
        client=client,
        collection_ids=(uuid5(NS, "collection"),),
        arm="golden",
        runs_dir=runs,
        dataset_dir=tmp_path,
        retrieval=GOLDEN_RETRIEVAL,
        chat_model="fake-chat",
        verifier_model="fake-verifier",
        prompt_versions=dict(GOLDEN_PROMPTS),
        policy=RetryPolicy(backoff_s=1.0, item_timeout_s=5.0),
        pacer=Pacer(sleep=Sleeps()),
        now=lambda: T0,
    )
    return replace(ctx, **changes)  # type: ignore[arg-type]


def by_id(rows: list[RunRow]) -> dict[str, RunRow]:
    return {r.item_id: r for r in rows}


def calls(client: FakeEvalClient, method: str) -> list[str]:
    return [c for c in client.log if c.startswith(f"{method}:")]


# ------------------------------------------------------------------ equivalence with the fixtures
async def test_retrieval_rows_and_header_match_the_golden_run(tmp_path: Path) -> None:
    path = await run_retrieval(make_ctx(tmp_path, FakeEvalClient()), load_questions())
    run, gold = read_run(path), golden("retrieval")
    assert [comparable(r) for r in run.rows] == [comparable(r) for r in gold.rows]
    assert run.footer is not None
    assert run.footer.rows == len(gold.rows)
    assert header_digest(run.header) == header_digest(gold.header)


async def test_verifier_rows_and_header_match_the_golden_run(tmp_path: Path) -> None:
    path = await run_verifier(make_ctx(tmp_path, FakeEvalClient()), load_verifier())
    run, gold = read_run(path), golden("verifier")
    mine = {k: comparable(v, "shown") for k, v in by_id(run.rows).items()}
    assert mine == {k: comparable(v, "shown") for k, v in by_id(gold.rows).items()}
    assert header_digest(run.header) == header_digest(gold.header)
    # the run records what was shown, as keys AND ids (drill-down), S1 first
    row = by_id(run.rows)["p-moved-1"]
    assert [s.label for s in row.shown] == [1, 2]
    assert all(s.chunk_key is not None for s in row.shown)


async def test_verifier_verifies_each_clean_source_before_its_mutations(tmp_path: Path) -> None:
    client = FakeEvalClient()
    await run_verifier(make_ctx(tmp_path, client), load_verifier())
    order = [c.split(":", 1)[1] for c in calls(client, "verify")]
    items = {i.id: i for i in load_verifier().items}
    for position, item_id in enumerate(order):
        item = items[item_id]
        if item.mutation is not None:
            assert order.index(item.source_id) < position, item_id


async def test_interview_grades_match_the_golden_run(tmp_path: Path) -> None:
    path = await run_interview(make_ctx(tmp_path, FakeEvalClient()), load_interview())
    run, gold = read_run(path), golden("interview")
    grades = {r.item_id: r for r in run.rows if isinstance(r.output, GradeOutput)}
    gold_rows = by_id(gold.rows)
    # the runner grades every replicate; the fixture dropped replicate 1 of one seed's partial
    # tier off its seed level (sde_1) on purpose
    dropped = {
        f"s-idempotency/partial/{lv}/1" for lv in ("sde_2", "senior_sde_1", "senior_sde_2", "staff")
    }
    assert set(grades) - set(gold_rows) == dropped
    assert set(gold_rows) <= set(grades)
    drop = ("grades", "output")
    for item_id, gold_row in gold_rows.items():
        mine = grades[item_id]
        assert comparable(mine, *drop, "shown") == comparable(gold_row, *drop, "shown")
        assert isinstance(mine.output, GradeOutput)
        assert isinstance(gold_row.output, GradeOutput)
        assert mine.output.score == gold_row.output.score
        assert mine.output.misconceptions_kept == gold_row.output.misconceptions_kept
        # the contract has no pre-filter quotes: `given` is a lower bound and says so
        assert mine.output.misconceptions_given == mine.output.misconceptions_kept
        assert mine.grades == {"given_is_kept": True}
    assert header_digest(run.header) == header_digest(gold.header)


async def test_interview_sweeps_only_gold_and_partial_with_two_replicates(tmp_path: Path) -> None:
    path = await run_interview(make_ctx(tmp_path, FakeEvalClient()), load_interview())
    outs = [r.output for r in read_run(path).rows if isinstance(r.output, GradeOutput)]
    seeds = {s.id: s for s in load_interview().seeds}
    for out in outs:
        if out.tier.value in ("wrong", "misconception"):
            assert out.level == seeds[out.seed_id].seed_level
    cells: dict[tuple[str, str, str], int] = {}
    for out in outs:
        cells[(out.seed_id, out.tier.value, out.level)] = (
            cells.get((out.seed_id, out.tier.value, out.level), 0) + 1
        )
    assert set(cells.values()) == {2}
    assert len(cells) == len(seeds) * (5 + 5 + 1 + 1)


async def test_interview_generates_questions_on_the_seed_notes(tmp_path: Path) -> None:
    client = FakeEvalClient()
    path = await run_interview(make_ctx(tmp_path, client), load_interview(), questions_per_seed=2)
    rows = [r for r in read_run(path).rows if isinstance(r.output, QuestionOutput)]
    assert len(rows) == 2 * len(load_interview().seeds)
    for row in rows:
        assert isinstance(row.output, QuestionOutput)
        assert row.grades["seed_chunk_included"] is True
        assert set(row.output.chunk_ids) == {s.chunk_id for s in row.shown}
    code_rows = [r for r in rows if r.item_id.startswith("s-idempotency/")]
    assert all("code_ref" in r.grades for r in code_rows)
    assert all("code_ref" not in r.grades for r in rows if r not in code_rows)


async def test_generator_rows_match_golden_and_judge_runs_after_answering(tmp_path: Path) -> None:
    client = FakeEvalClient()
    judge = FakeJudge(client.log)
    path = await run_generator(make_ctx(tmp_path, client), load_questions(), judge)
    run, gold = read_run(path), golden("generator")
    drop = ("shown", "judge_refs", "timings", "grades")
    assert [comparable(r, *drop) for r in run.rows] == [comparable(r, *drop) for r in gold.rows]
    assert header_digest(run.header) == header_digest(gold.header)
    assert run.header.judge == GOLDEN_JUDGE
    # ADR-0017 §4: the judge never runs while answering; the staging file is gone
    last_chat = max(i for i, c in enumerate(client.log) if c.startswith("chat:"))
    first_judge = min(i for i, c in enumerate(client.log) if c.startswith("judge:"))
    assert last_chat < first_judge
    assert not staging_path(path).exists()
    # answered rows carry one ref per criterion; abstains are never judged
    for row in run.rows:
        assert isinstance(row.output, AnswerOutput)
        assert len(row.judge_refs) == (0 if row.output.abstained else len(E_G_CRITERIA))
    # the judge saw the shown notes and the facts, under the label queue's own digest
    questions = {q.id: q for q in load_questions().questions}
    item = judge.items[0]
    row = next(r for r in run.rows if r.judge_refs)
    q = questions[row.item_id]
    assert isinstance(row.output, AnswerOutput)
    notes = tuple(page_text(p) for p in row.output.cited_pages)
    assert item.notes == notes
    facts = [f.fact for f in q.facts]
    assert item.item_digest == item_digest_of(q.question, row.output.answer, notes, facts)


async def test_generator_records_the_deterministic_correct_grade(tmp_path: Path) -> None:
    path = await run_generator(make_ctx(tmp_path, FakeEvalClient()), load_questions(), None)
    questions = {q.id: q for q in load_questions().questions}
    rows = read_run(path).rows
    for row in rows:
        assert isinstance(row.output, AnswerOutput)
        if not questions[row.item_id].answerable:
            assert row.grades == {"correct": row.output.abstained}
        else:
            assert set(row.grades) == {
                "correct",
                "cited_expected",
                "cited_by_model",
                "facts_hit",
                "facts_needed",
            }
    # kafka-idle-consumers: "Answer to kafka-idle-consumers [S1]." hits its one fact ("idle")
    # and cites its expected page
    assert by_id(rows)["kafka-idle-consumers"].grades["correct"] is True


QUESTION = Question(
    id="q",
    split=Split.dev,
    question="How?",
    pages=["Kafka Basics"],
    relevance={"file:kafka-basics.md": 2},
    facts=[
        Fact(fact="a", match=["same partition"]),
        Fact(fact="b", match=["shipment_id"]),
        Fact(fact="c", match=["within a partition"]),
    ],
)


def _answer(text: str, *, abstained: bool = False) -> AnswerOutput:
    return AnswerOutput(answer=text, status="verified", cited_pages=[], abstained=abstained)


@pytest.mark.parametrize(
    ("text", "titles", "abstained", "correct"),
    [
        ("Same partition, keyed by shipment_id [S1].", ["Kafka Basics"], False, True),
        ("Same partition, keyed by shipment_id [S1].", ["Other Page"], False, False),
        ("Keyed by shipment_id only [S1].", ["Kafka Basics"], False, False),
        ("Same partition, keyed by shipment_id.", ["Kafka Basics"], True, False),
    ],
)
def test_answer_grades_follow_the_phase_2b_rule(
    text: str, titles: list[str], abstained: bool, correct: bool
) -> None:
    grades = answer_grades(QUESTION, _answer(text, abstained=abstained), titles)
    assert grades["correct"] is correct
    assert grades["facts_needed"] == 2


def test_an_unanswerable_question_is_correct_only_when_it_abstains() -> None:
    q = Question(id="u", split=Split.dev, question="Sourdough?", answerable=False)
    assert answer_grades(q, _answer("x", abstained=True), [])["correct"] is True
    assert answer_grades(q, _answer("x"), [])["correct"] is False


async def test_application_then_latency_pass_with_cold_questions_apart(tmp_path: Path) -> None:
    client = FakeEvalClient()
    ctx = make_ctx(tmp_path, client)
    app = read_run(await run_application(ctx, load_questions()))
    lat = read_run(await run_latency(ctx, load_questions()))
    gold = by_id(golden("generator").rows)
    assert app.header.eval_kind is EvalKind.application
    assert app.header.judge is None
    assert lat.header.eval_kind is EvalKind.latency
    for row in app.rows:
        assert row.tokens == gold[row.item_id].tokens
        assert row.timings.total_s is not None
    assert [r.grades["cold"] for r in lat.rows[:3]] == [True, True, False]


# ------------------------------------------------------------------ header
async def test_header_records_the_snapshot_digests(tmp_path: Path) -> None:
    ctx = make_ctx(tmp_path, FakeEvalClient(), snapshot_collection_id=SNAPSHOT_ID)
    header = read_run(await run_retrieval(ctx, load_questions())).header
    assert header.snapshot is not None
    assert header.snapshot.collection_id == str(SNAPSHOT_ID)
    assert (header.snapshot.content_digest, header.snapshot.index_digest) == (
        "content-d",
        "index-d",
    )
    assert header.transient_retries == 2
    assert header.dataset.splits == [Split.dev]


async def test_a_snapshot_without_digests_refuses_to_start(tmp_path: Path) -> None:
    client = FakeEvalClient(snapshot_digests=(None, None))
    ctx = make_ctx(tmp_path, client, snapshot_collection_id=SNAPSHOT_ID)
    with pytest.raises(RunAborted, match="no digest"):
        await run_retrieval(ctx, load_questions())
    assert not calls(client, "query")
    assert not list(ctx.runs_dir.iterdir())


async def test_prompt_versions_and_verifier_model_come_from_the_server(tmp_path: Path) -> None:
    client = FakeEvalClient(
        prompt_versions={"chat": "answer@12"},
        verifier_prompt="ground_check@4",
        verifier_model="served-verifier",
    )
    ctx = make_ctx(tmp_path, client, prompt_versions={"chat": "configured@1"})
    gen = read_run(await run_generator(ctx, load_questions(), None)).header
    ver = read_run(await run_verifier(ctx, load_verifier())).header
    assert gen.prompt_versions == {"chat": "answer@12"}
    assert ver.prompt_versions["verifier"] == "ground_check@4"
    assert ver.verifier_model == "served-verifier"


async def test_a_prompt_change_mid_run_is_an_error_of_ours(tmp_path: Path) -> None:
    class Redeployed(FakeEvalClient):
        def __init__(self) -> None:
            super().__init__()
            self.asked = 0

        async def chat(self, message, collection_ids, *, conversation_id=None, trace_shown=False):  # type: ignore[no-untyped-def]
            self.asked += 1
            if self.asked == 3:
                self.prompt_versions = {"chat": "answer@12"}
            async for event in super().chat(message, collection_ids, trace_shown=trace_shown):
                yield event

    run = read_run(await run_application(make_ctx(tmp_path, Redeployed()), load_questions()))
    errors = [r for r in run.rows if r.error is not None]
    assert len(errors) == len(run.rows) - 2
    assert all(e.error and e.error.cls is ErrorClass.ours for e in errors)
    assert all(e.error and "prompt_versions[chat]" in e.error.message for e in errors)
    assert run.header.prompt_versions["chat"] == "answer@11"


async def test_heldout2_is_never_run(tmp_path: Path) -> None:
    ctx = make_ctx(tmp_path, FakeEvalClient(), splits=(Split.dev, Split.heldout2))
    with pytest.raises(RunAborted, match="sealed"):
        await run_generator(ctx, load_questions(), None)


# ------------------------------------------------------------------ errors and retries
async def test_transient_errors_are_retried_then_succeed(tmp_path: Path) -> None:
    client = FakeEvalClient(fail={("query", "kafka-ordering"): [api_error(429), api_error(503)]})
    ctx = make_ctx(tmp_path, client)
    run = read_run(await run_retrieval(ctx, load_questions()))
    row = by_id(run.rows)["kafka-ordering"]
    assert row.error is None
    assert row.attempt == 3
    assert row.output is not None
    assert len(calls(client, "query")) == len(run.rows) + 2
    assert isinstance(ctx.pacer.sleep, Sleeps)
    assert ctx.pacer.sleep.calls == [1.0, 2.0]


async def test_transient_errors_are_classified_after_the_retries(tmp_path: Path) -> None:
    client = FakeEvalClient(fail={("query", "kafka-ordering"): [api_error(503)] * 3})
    run = read_run(await run_retrieval(make_ctx(tmp_path, client), load_questions()))
    row = by_id(run.rows)["kafka-ordering"]
    assert row.output is None
    assert row.attempt == 3
    assert row.error is not None
    assert (row.error.cls, row.error.status) == (ErrorClass.provider_transient, 503)


async def test_errors_of_ours_are_not_retried(tmp_path: Path) -> None:
    client = FakeEvalClient(fail={("query", "kafka-ordering"): [api_error(400)]})
    run = read_run(await run_retrieval(make_ctx(tmp_path, client), load_questions()))
    row = by_id(run.rows)["kafka-ordering"]
    assert row.error is not None
    assert row.error.cls is ErrorClass.ours
    assert row.attempt == 1


async def test_a_provider_error_event_mid_stream_is_transient(tmp_path: Path) -> None:
    client = FakeEvalClient(fail={("chat-stream", "kafka-ordering"): [api_error(429)]})
    run = read_run(await run_application(make_ctx(tmp_path, client), load_questions()))
    row = by_id(run.rows)["kafka-ordering"]
    assert row.error is None
    assert row.attempt == 2


async def test_a_done_event_without_shown_is_an_error_of_ours(tmp_path: Path) -> None:
    run = read_run(
        await run_application(
            make_ctx(tmp_path, FakeEvalClient(with_shown=False)), load_questions()
        )
    )
    assert all(r.error is not None and "shown" in r.error.message for r in run.rows)


# ------------------------------------------------------------------ stale keys (principle 6)
async def test_unresolved_chunk_keys_mark_items_stale_without_a_call(tmp_path: Path) -> None:
    items = {i.id: i for i in load_verifier().items}
    breaker_key = key_id(items["c-breaker"].shown[0])
    client = FakeEvalClient(stale_chunks={breaker_key})
    run = read_run(await run_verifier(make_ctx(tmp_path, client), load_verifier()))
    stale = {r.item_id for r in run.rows if r.stale}
    expected = {i.id for i in items.values() if any(key_id(k) == breaker_key for k in i.shown)}
    assert stale == expected
    assert "c-breaker" in stale
    assert all(r.output is None and r.error is None for r in run.rows if r.stale)
    assert not {c.split(":", 1)[1] for c in calls(client, "verify")} & stale


async def test_a_missing_relevant_page_marks_the_question_stale(tmp_path: Path) -> None:
    client = FakeEvalClient(missing_pages={"file:kafka-basics.md"})
    run = read_run(await run_retrieval(make_ctx(tmp_path, client), load_questions()))
    stale = {r.item_id for r in run.rows if r.stale}
    expected = {q.id for q in load_questions().questions if "file:kafka-basics.md" in q.relevance}
    assert stale == expected
    assert stale
    assert not {c.split(":", 1)[1] for c in calls(client, "query")} & stale


async def test_notes_that_changed_since_answering_are_not_judged(tmp_path: Path) -> None:
    from test_ev_run_fakes import page_chunk_id

    client = FakeEvalClient(changed_text={page_chunk_id("file:kafka-basics.md")})
    judge = FakeJudge(client.log)
    run = read_run(await run_generator(make_ctx(tmp_path, client), load_questions(), judge))
    hit = by_id(run.rows)["kafka-idle-consumers"]  # cites kafka-basics.md
    assert hit.stale
    assert hit.judge_refs == []


# ------------------------------------------------------------------ resume
async def test_resume_after_a_crash_keeps_every_finished_row(tmp_path: Path) -> None:
    client = FakeEvalClient(crash_on="rebalance-triggers")
    ctx = make_ctx(tmp_path, client)
    with pytest.raises(Crash):
        await run_retrieval(ctx, load_questions())
    path = find_resumable(ctx.runs_dir, EvalKind.retrieval, "golden")
    assert path is not None
    partial = read_run(path)
    assert partial.footer is None
    assert len(partial.rows) == 4
    assert stat.S_IMODE(path.stat().st_mode) == 0o600

    again = FakeEvalClient()
    done = read_run(await run_retrieval(make_ctx(tmp_path, again, resume=path), load_questions()))
    assert done.footer is not None
    assert [r.item_id for r in done.rows] == [r.item_id for r in golden("retrieval").rows]
    assert len(calls(again, "query")) == len(done.rows) - 4
    assert find_resumable(ctx.runs_dir, EvalKind.retrieval, "golden") is None


async def test_resume_refuses_a_changed_configuration_before_any_call(tmp_path: Path) -> None:
    ctx = make_ctx(tmp_path, FakeEvalClient(crash_on="rebalance-triggers"))
    with pytest.raises(Crash):
        await run_retrieval(ctx, load_questions())
    path = find_resumable(ctx.runs_dir, EvalKind.retrieval, "golden")
    assert path is not None
    again = FakeEvalClient()
    changed = make_ctx(tmp_path, again, resume=path, chat_model="another-model")
    with pytest.raises(RunFileError, match="configuration differs"):
        await run_retrieval(changed, load_questions())
    assert not calls(again, "query")


async def test_a_failed_judge_keeps_the_answers_and_resume_only_judges(tmp_path: Path) -> None:
    client = FakeEvalClient()
    ctx = make_ctx(tmp_path, client)
    broken = FakeJudge(client.log, raise_with=RuntimeError("https://x/?key=SECRET"), fail_after=3)
    with pytest.raises(RunAborted) as caught:
        await run_generator(ctx, load_questions(), broken)
    assert "SECRET" not in str(caught.value)
    resumable = find_resumable(ctx.runs_dir, EvalKind.generator, "golden")
    assert resumable is not None
    judged_before = len(read_run(resumable).rows)
    assert judged_before > 0  # the judge phase had written rows before it failed

    again = FakeEvalClient()
    judge = FakeJudge(again.log)
    path = await run_generator(make_ctx(tmp_path, again, resume=resumable), load_questions(), judge)
    run = read_run(path)
    assert run.footer is not None
    assert len(run.rows) == len(golden("generator").rows)
    assert not calls(again, "chat")
    assert len(judge.items) == len(broken_answered(run)) - 3
    assert not staging_path(path).exists()


def broken_answered(run: RunFile) -> list[RunRow]:
    return [r for r in run.rows if isinstance(r.output, AnswerOutput) and not r.output.abstained]


# ------------------------------------------------------------------ summary
async def test_the_summary_is_written_next_to_the_finished_run(tmp_path: Path) -> None:
    summarizer = FakeSummarizer()
    ctx = make_ctx(tmp_path, FakeEvalClient(), summarize=summarizer)
    path = await run_verifier(ctx, load_verifier())
    out = summary_path(path)
    assert out.exists()
    assert stat.S_IMODE(out.stat().st_mode) == 0o600
    ((run, datasets, labels, results),) = summarizer.calls
    assert run.footer is not None
    assert datasets.verifier is not None
    assert labels == []
    assert results == []


# ------------------------------------------------------------------ added: guards the fakes hid
async def test_an_abstain_is_never_judged_even_when_it_was_shown_notes(tmp_path: Path) -> None:
    # a real server shows notes on an abstain too; the judge scores answered rows only (§3 E-G)
    client = FakeEvalClient(pad_shown="file:kafka-basics.md")
    judge = FakeJudge(client.log)
    run = read_run(await run_generator(make_ctx(tmp_path, client), load_questions(), judge))
    abstains = [r for r in run.rows if isinstance(r.output, AnswerOutput) and r.output.abstained]
    assert abstains
    assert all(r.shown and r.judge_refs == [] for r in abstains)
    assert len(judge.items) == len(broken_answered(run))


async def test_resume_retries_the_rows_that_failed(tmp_path: Path) -> None:
    client = FakeEvalClient(
        fail={("query", "kafka-ordering"): [api_error(400)]}, crash_on="rebalance-triggers"
    )
    ctx = make_ctx(tmp_path, client)
    with pytest.raises(Crash):
        await run_retrieval(ctx, load_questions())
    path = find_resumable(ctx.runs_dir, EvalKind.retrieval, "golden")
    assert path is not None
    failed = [r.item_id for r in read_run(path).rows if r.error is not None]
    assert failed == ["kafka-ordering"]

    again = FakeEvalClient()
    run = read_run(await run_retrieval(make_ctx(tmp_path, again, resume=path), load_questions()))
    assert "query:kafka-ordering" in again.log
    attempts = [r for r in run.rows if r.item_id == "kafka-ordering"]
    assert [r.error is None for r in attempts] == [False, True]  # the newest row wins


async def test_the_judge_refuses_answers_that_have_not_finished(tmp_path: Path) -> None:
    client = FakeEvalClient(crash_on="rebalance-triggers")
    ctx = make_ctx(tmp_path, client)
    judge = FakeJudge(client.log)
    with pytest.raises(Crash):
        await run_generator(ctx, load_questions(), judge)
    staging = find_resumable(ctx.runs_dir, EvalKind.generator, "golden")
    assert staging is not None
    assert staging == staging_path(final_path(staging))
    with pytest.raises(RunAborted, match="answering has not finished"):
        await judge_phase(ctx, judge, staging, final_path(staging), lambda _: None)
    assert judge.items == []
    assert not final_path(staging).exists()


# ------------------------------------------------------------------ review round 1
async def test_resuming_the_judge_phase_with_another_judge_is_refused(tmp_path: Path) -> None:
    client = FakeEvalClient()
    ctx = make_ctx(tmp_path, client)
    broken = FakeJudge(client.log, raise_with=RuntimeError("down"), fail_after=3)
    with pytest.raises(RunAborted):
        await run_generator(ctx, load_questions(), broken)
    resumable = find_resumable(ctx.runs_dir, EvalKind.generator, "golden")
    assert resumable is not None
    before = resumable.read_bytes()

    other = FakeJudge(client.log, key_=GOLDEN_JUDGE.model_copy(update={"provider": "gemini"}))
    with pytest.raises(RunAborted, match="being judged by ollama/qwen3"):
        await run_generator(make_ctx(tmp_path, client, resume=resumable), load_questions(), other)
    assert other.items == []
    assert resumable.read_bytes() == before  # nothing was appended under the wrong judge


async def test_a_judge_switched_before_any_row_was_judged_names_the_new_judge(
    tmp_path: Path,
) -> None:
    # answering crashed; the owner resumes with the Gemini fallback: every row gets that judge
    client = FakeEvalClient(crash_on="rebalance-triggers")
    ctx = make_ctx(tmp_path, client)
    with pytest.raises(Crash):
        await run_generator(ctx, load_questions(), FakeJudge(client.log))
    staging = find_resumable(ctx.runs_dir, EvalKind.generator, "golden")
    assert staging is not None
    assert read_run(staging).header.judge is None  # answering never depends on the judge

    gemini = GOLDEN_JUDGE.model_copy(update={"provider": "gemini", "model_version": "flash"})
    again = FakeEvalClient()
    judge = FakeJudge(again.log, key_=gemini)
    path = await run_generator(make_ctx(tmp_path, again, resume=staging), load_questions(), judge)
    run = read_run(path)
    assert run.header.judge == gemini
    assert judge.items  # it scored the rows the header says it scored


def _verifier_order_first_unran() -> tuple[list[str], str]:
    gold = by_id(golden("verifier").rows)
    order = [i.id for i in cluster_order(load_verifier().items)]
    unran = next(
        i for i in order if isinstance(o := gold[i].output, VerifyOutput) and not o.verifier_ran
    )
    return order[: order.index(unran)], unran


async def test_a_first_answer_without_a_verifier_does_not_open_the_header(tmp_path: Path) -> None:
    # the items before the first coverage-caught one fail transiently, so it answers first
    before, unran = _verifier_order_first_unran()
    client = FakeEvalClient(
        verifier_model="served-verifier",
        verifier_prompt="ground_check@4",
        verifier_only_when_ran=True,
        fail={("verify", i): [api_error(503)] * 3 for i in before},
    )
    ctx = make_ctx(tmp_path, client, verifier_model=None, prompt_versions={})
    run = read_run(await run_verifier(ctx, load_verifier()))
    assert run.header.verifier_model == "served-verifier"
    assert run.header.prompt_versions == {"verifier": "ground_check@4"}
    mid_run = [r for r in run.rows if r.error and "mid-run" in r.error.message]
    assert mid_run == []
    assert by_id(run.rows)[unran].output is not None


async def test_a_verifier_model_change_mid_run_is_still_caught(tmp_path: Path) -> None:
    class Swapped(FakeEvalClient):
        async def eval_verify(self, request):  # type: ignore[no-untyped-def]
            if len(calls(self, "verify")) >= 4:
                self.verifier_model = "other-verifier"
            return await super().eval_verify(request)

    client = Swapped(verifier_model="served-verifier", verifier_prompt="ground_check@4")
    run = read_run(await run_verifier(make_ctx(tmp_path, client), load_verifier()))
    assert run.header.verifier_model == "served-verifier"
    errors = [r for r in run.rows if r.error is not None]
    assert errors
    assert all(r.error and "verifier_model" in r.error.message for r in errors)


async def test_the_header_records_the_chat_model_the_server_ran(tmp_path: Path) -> None:
    client = FakeEvalClient(chat_model="qwen3:8b", verifier_usage="verifier-model")
    ctx = make_ctx(tmp_path, client, chat_model=None)
    run = read_run(await run_application(ctx, load_questions()))
    assert run.header.chat_model == "qwen3:8b"  # the first usage entry, never the verifier's
    assert all(r.error is None for r in run.rows)


async def test_a_chat_model_change_mid_run_is_an_error_of_ours(tmp_path: Path) -> None:
    class FellBack(FakeEvalClient):
        async def chat(self, message, collection_ids, *, conversation_id=None, trace_shown=False):  # type: ignore[no-untyped-def]
            if len(calls(self, "chat")) >= 2:
                self.chat_model = "gemini-fallback"
            async for event in super().chat(message, collection_ids, trace_shown=trace_shown):
                yield event

    run = read_run(await run_application(make_ctx(tmp_path, FellBack()), load_questions()))
    assert run.header.chat_model == "fake-chat"
    errors = [r for r in run.rows if r.error is not None]
    assert len(errors) == len(run.rows) - 2
    assert all(r.error and "chat_model" in r.error.message for r in errors)


async def test_a_resumed_latency_pass_marks_its_own_first_answers_cold(tmp_path: Path) -> None:
    ctx = make_ctx(tmp_path, FakeEvalClient(crash_on="rebalance-triggers"))
    with pytest.raises(Crash):
        await run_latency(ctx, load_questions())
    path = find_resumable(ctx.runs_dir, EvalKind.latency, "golden")
    assert path is not None
    first = len(read_run(path).rows)
    assert first > COLD
    run = read_run(
        await run_latency(make_ctx(tmp_path, FakeEvalClient(), resume=path), load_questions())
    )
    cold = [bool(r.grades.get("cold")) for r in run.rows]
    expected = [i < COLD for i in range(first)] + [i < COLD for i in range(len(run.rows) - first)]
    assert cold == expected


async def test_a_stream_without_a_verification_event_is_an_error_of_ours(tmp_path: Path) -> None:
    client = FakeEvalClient(with_verification=False)
    run = read_run(await run_application(make_ctx(tmp_path, client), load_questions()))
    assert all(r.error is not None and "verification" in r.error.message for r in run.rows)
    assert all(r.error and r.error.cls is ErrorClass.ours for r in run.rows)


async def test_server_inferred_citations_are_not_the_models(tmp_path: Path) -> None:
    run = read_run(
        await run_application(
            make_ctx(tmp_path, FakeEvalClient(inferred_citations=True)), load_questions()
        )
    )
    row = by_id(run.rows)["kafka-idle-consumers"]
    assert row.grades["cited_expected"] is True
    assert row.grades["cited_by_model"] is False
    (tmp_path / "b").mkdir()
    plain = read_run(
        await run_application(make_ctx(tmp_path / "b", FakeEvalClient()), load_questions())
    )
    assert by_id(plain.rows)["kafka-idle-consumers"].grades["cited_by_model"] is True


async def test_the_judge_is_paced_on_its_own_budget(tmp_path: Path) -> None:
    client = FakeEvalClient()
    sleeps = Sleeps()
    clock = iter(float(i) for i in range(10_000))
    judge_pacer = Pacer(rpm=6.0, clock=lambda: next(clock), sleep=sleeps)
    ctx = make_ctx(tmp_path, client, judge_pacer=judge_pacer)
    judge = FakeJudge(client.log)
    await run_generator(ctx, load_questions(), judge)
    assert len(sleeps.calls) == len(judge.items) - 1  # every judge call after the first waited
    assert all(s == pytest.approx(9.0) for s in sleeps.calls)  # a 10 s gap, 1 s already gone


async def test_a_generated_question_is_judged_on_its_own_grounding(tmp_path: Path) -> None:
    seed_ids = {key_id(k) for s in load_interview().seeds for k in s.chunk_keys}
    other = next(
        c
        for c in json.loads((DEMO / "chunks.json").read_text())
        if key_id(ChunkKey.model_validate(c["key"])) not in seed_ids
    )
    extra = uuid5(NS, key_id(ChunkKey.model_validate(other["key"])))
    client = FakeEvalClient(extra_question_chunk=extra)
    judge = FakeJudge(client.log)
    run = read_run(await run_interview(make_ctx(tmp_path, client), load_interview(), judge))
    questions = [r for r in run.rows if isinstance(r.output, QuestionOutput)]
    assert questions
    for row in questions:
        assert row.shown[-1].chunk_id == str(extra)
        assert [s.label for s in row.shown] == list(range(1, len(row.shown) + 1))
    # every question-quality judge item saw the widened notes, the extra one last
    assert len(judge.items) == len(questions)
    assert all(item.notes[-1] == other["text"] for item in judge.items)


def test_reading_the_judge_cache_skips_a_line_being_written(tmp_path: Path) -> None:
    cache = tmp_path / "judge-cache.jsonl"
    good = (FIXTURES / "judge-results.jsonl").read_text().splitlines()[0]
    cache.write_text(good + "\n" + good[: len(good) // 2])  # a second write in progress
    assert len(load_judge_results(cache)) == 1
    cache.write_text(good + "\n" + '{"not": "a result"}\n')
    with pytest.raises(RunAborted, match=r"judge-cache\.jsonl:2"):
        load_judge_results(cache)


async def test_an_interview_header_without_an_interview_prompt_warns(tmp_path: Path) -> None:
    lines: list[str] = []
    await run_interview(make_ctx(tmp_path, FakeEvalClient(), log=lines.append), load_interview())
    assert any("no interview prompt version" in line for line in lines)


async def test_a_value_first_reported_after_the_header_fills_a_gap(tmp_path: Path) -> None:
    # session 1: a server that reports no verifier crashes mid-run; session 2's server reports
    # one. The header cannot change, but "not reported" before is not a mid-run change.
    order = [i.id for i in cluster_order(load_verifier().items)]
    ctx = make_ctx(tmp_path, FakeEvalClient(crash_on=order[6]), verifier_model=None)
    with pytest.raises(Crash):
        await run_verifier(ctx, load_verifier())
    path = find_resumable(ctx.runs_dir, EvalKind.verifier, "golden")
    assert path is not None
    again = FakeEvalClient(verifier_model="served-verifier")
    resumed = make_ctx(tmp_path, again, resume=path, verifier_model=None)
    run = read_run(await run_verifier(resumed, load_verifier()))
    assert run.header.verifier_model is None
    assert all(r.error is None for r in run.rows)
