"""EV-RUN: behaviour that only shows end to end, through the REAL metrics where it matters:
a resumed retry must win over the failure it recovered, an inconclusive verifier is a transient
error (never an objection), a chunk the server cannot look up fails the item, a judge phase
that crashed after finishing does not wedge resume, and the latency cold marks vs metrics.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from uuid import UUID, uuid5

import pytest
from test_ev_run_fakes import (
    NS,
    Crash,
    FakeEvalClient,
    FakeJudge,
    api_error,
    golden,
    load_questions,
    load_verifier,
    page_chunk_id,
)
from test_ev_run_runners import by_id, calls, make_ctx

from ragforge_client.evals import metrics, runners
from ragforge_client.evals.interfaces import Datasets
from ragforge_client.evals.runfile import read_run
from ragforge_client.evals.runners import (
    CHUNKS_BATCH,
    PageLookup,
    RunAborted,
    find_resumable,
    run_generator,
    run_latency,
    run_retrieval,
    run_verifier,
    staging_path,
)
from ragforge_client.evals.schema import ErrorClass, EvalKind, RetrievalOutput, VerifyOutput
from ragforge_contracts.api import EvalChunksResponse, EvalVerifyRequest, EvalVerifyResponse

ITEM, CRASH_AT = "kafka-ordering", "rebalance-triggers"  # ITEM is asked before CRASH_AT


# ------------------------------------------------------------------ attempts across a resume
async def test_a_resumed_retry_of_a_transient_failure_wins_in_metrics(tmp_path: Path) -> None:
    client = FakeEvalClient(fail={("query", ITEM): [api_error(503)] * 3}, crash_on=CRASH_AT)
    with pytest.raises(Crash):
        await run_retrieval(make_ctx(tmp_path, client), load_questions())
    path = find_resumable(tmp_path / "runs", EvalKind.retrieval, "golden")
    assert path is not None
    (failed,) = [r for r in read_run(path).rows if r.item_id == ITEM]
    assert failed.error is not None
    assert failed.error.cls is ErrorClass.provider_transient
    assert failed.attempt == 3

    again = FakeEvalClient()
    run = read_run(await run_retrieval(make_ctx(tmp_path, again, resume=path), load_questions()))
    assert [r.attempt for r in run.rows if r.item_id == ITEM] == [3, 4]  # attempt only grows
    selection = metrics.select_rows(run)
    assert selection.rows[ITEM].error is None
    summary = metrics.summarize(run, Datasets(questions=load_questions()), [], [])
    assert summary.errors_transient == 0
    assert summary.errors_ours == 0


# ------------------------------------------------------------------ an inconclusive verifier
@dataclass
class InconclusiveClient(FakeEvalClient):
    """The verifier of `item` comes back inconclusive (HTTP 200) `times` times."""

    item: str = ""
    times: int = 0
    seen: int = field(default=0, init=False)

    async def eval_verify(self, request: EvalVerifyRequest) -> EvalVerifyResponse:
        response = await super().eval_verify(request)
        if self._vid[request.answer] == self.item and self.seen < self.times:
            self.seen += 1
            return response.model_copy(
                update={"verifier_conclusive": False, "verifier_verdict": "partial"}
            )
        return response


def _ran_item() -> str:
    rows = golden("verifier").rows
    return next(
        r.item_id for r in rows if isinstance(r.output, VerifyOutput) and r.output.verifier_ran
    )


async def test_an_inconclusive_verifier_is_retried_not_counted_as_an_objection(
    tmp_path: Path,
) -> None:
    item = _ran_item()
    client = InconclusiveClient(item=item, times=1)
    run = read_run(await run_verifier(make_ctx(tmp_path, client), load_verifier()))
    row = by_id(run.rows)[item]
    gold = by_id(golden("verifier").rows)[item]
    assert row.error is None
    assert row.attempt == 2
    assert row.output == gold.output  # the retried verdict, not the inconclusive one


async def test_an_always_inconclusive_verifier_is_a_transient_error(tmp_path: Path) -> None:
    item = _ran_item()
    client = InconclusiveClient(item=item, times=99)
    run = read_run(await run_verifier(make_ctx(tmp_path, client), load_verifier()))
    row = by_id(run.rows)[item]
    assert row.output is None
    assert row.error is not None
    assert row.error.cls is ErrorClass.provider_transient
    assert row.attempt == 3
    assert metrics.summarize(run, Datasets(verifier=load_verifier()), [], []).errors_transient == 1


# ------------------------------------------------------------------ chunks the lookup lacks
@dataclass
class DroppingClient(FakeEvalClient):
    """`/v1/eval/chunks` leaves `drop` out (a server bug under RLS); records batch sizes."""

    drop: set[UUID] = field(default_factory=set)
    batches: list[int] = field(default_factory=list)

    async def eval_chunks(self, chunk_ids: Sequence[UUID]) -> EvalChunksResponse:
        self.batches.append(len(chunk_ids))
        response = await super().eval_chunks([c for c in chunk_ids if c not in self.drop])
        return response


async def test_a_stage_chunk_the_lookup_cannot_map_fails_the_item(tmp_path: Path) -> None:
    gold = by_id(golden("retrieval").rows)[ITEM].output
    assert isinstance(gold, RetrievalOutput)
    only_here = page_chunk_id(gold.stages["reranked"][0])
    client = DroppingClient(drop={only_here})
    run = read_run(await run_retrieval(make_ctx(tmp_path, client), load_questions()))
    row = by_id(run.rows)[ITEM]
    assert row.output is None
    assert row.error is not None
    assert row.error.cls is ErrorClass.ours
    assert "absent from /v1/eval/chunks" in row.error.message
    assert not any(
        p.startswith("unresolved:")
        for r in run.rows
        if isinstance(r.output, RetrievalOutput)
        for pages in r.output.stages.values()
        for p in pages
    )


async def test_the_page_lookup_fetches_in_batches(tmp_path: Path) -> None:
    client = DroppingClient()
    lookup = PageLookup(make_ctx(tmp_path, client))
    ids = [uuid5(NS, f"absent-{n}") for n in range(2 * CHUNKS_BATCH + 7)]
    with pytest.raises(runners.ItemFailure):
        await lookup.load(ids)  # none of them exists: the item fails, after asking for all
    assert client.batches == [CHUNKS_BATCH, CHUNKS_BATCH, 7]
    client.batches.clear()
    with pytest.raises(runners.ItemFailure):
        await lookup.load(ids[:3])  # unknown ids are asked again, never cached as absent
    assert client.batches == [3]


# ------------------------------------------------------------------ a crash after the judge
async def test_a_crash_after_the_judge_finished_does_not_wedge_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = FakeEvalClient()
    ctx = make_ctx(tmp_path, client)

    def killed(_: Path) -> None:
        raise Crash("killed between finish() and the staging removal")

    monkeypatch.setattr(runners, "_remove", killed)
    with pytest.raises(Crash):
        await run_generator(ctx, load_questions(), FakeJudge(client.log))
    monkeypatch.undo()
    final = next((tmp_path / "runs").glob("generator-*.jsonl"))
    staging = staging_path(final)
    assert staging.exists()
    assert read_run(final).footer is not None

    # an explicit resume of the staging file: nothing to do, and the leftover is cleaned up
    again = FakeEvalClient()
    with pytest.raises(RunAborted, match="already finished: removed its leftover staging"):
        await run_generator(
            make_ctx(tmp_path, again, resume=staging), load_questions(), FakeJudge(again.log)
        )
    assert not staging.exists()
    assert not calls(again, "chat")
    assert read_run(final).footer is not None


async def test_find_resumable_drops_a_staging_file_whose_run_finished(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = FakeEvalClient()

    def killed(_: Path) -> None:
        raise Crash("killed")

    monkeypatch.setattr(runners, "_remove", killed)
    with pytest.raises(Crash):
        await run_generator(make_ctx(tmp_path, client), load_questions(), FakeJudge(client.log))
    monkeypatch.undo()
    final = next((tmp_path / "runs").glob("generator-*.jsonl"))
    assert find_resumable(tmp_path / "runs", EvalKind.generator, "golden") is None
    assert not staging_path(final).exists()


# ------------------------------------------------------------------ latency cold marks
def _cold_ids(path: Path) -> tuple[list[str], list[str]]:
    run = read_run(path)
    marked = [r.item_id for r in run.rows if r.grades.get("cold")]
    cold, _ = metrics.cold_and_warm_rows(run)
    return marked, [r.item_id for r in cold]


async def test_without_a_resume_the_cold_marks_agree_with_metrics(tmp_path: Path) -> None:
    path = await run_latency(make_ctx(tmp_path, FakeEvalClient()), load_questions())
    marked, by_metrics = _cold_ids(path)
    assert marked == by_metrics
    assert len(marked) == runners.COLD_QUESTIONS


@pytest.mark.xfail(
    strict=True,
    reason="EV-METRICS ignores grades['cold'] (handoff EV-RUN.md): a resumed session's cold "
    "starts land in the warm percentiles. Remove this mark when metrics honours the grade.",
)
async def test_after_a_resume_metrics_reports_every_marked_cold_row_apart(
    tmp_path: Path,
) -> None:
    with pytest.raises(Crash):
        await run_latency(make_ctx(tmp_path, FakeEvalClient(crash_on=CRASH_AT)), load_questions())
    path = find_resumable(tmp_path / "runs", EvalKind.latency, "golden")
    assert path is not None
    done = await run_latency(make_ctx(tmp_path, FakeEvalClient(), resume=path), load_questions())
    marked, by_metrics = _cold_ids(done)
    assert sorted(marked) == sorted(by_metrics)


# ------------------------------------------------------------------ resume vs judge setting
# Integrator fix of review P1: a resume whose judge setting differs from the original run must
# refuse clearly; never silently re-answer everything, never crash mid-way.
async def test_a_judged_run_resumed_without_a_judge_refuses_and_spends_nothing(
    tmp_path: Path,
) -> None:
    from test_ev_run_fakes import Crash, FakeEvalClient, FakeJudge, load_questions
    from test_ev_run_runners import make_ctx

    from ragforge_client.evals.runners import RunAborted, find_resumable, run_generator
    from ragforge_client.evals.schema import EvalKind

    crashing = FakeEvalClient(crash_on="rebalance-triggers")
    ctx = make_ctx(tmp_path, crashing)
    with pytest.raises(Crash):
        await run_generator(ctx, load_questions(), FakeJudge(crashing.log))
    staged = find_resumable(ctx.runs_dir, EvalKind.generator, "golden")
    assert staged is not None
    again = FakeEvalClient()
    with pytest.raises(RunAborted, match="started with a judge"):
        await run_generator(make_ctx(tmp_path, again, resume=staged), load_questions(), None)
    assert not [c for c in again.log if c.startswith("chat:")]  # nothing re-answered
    assert staged.exists()  # the staged answers are kept for a correct resume


async def test_an_unjudged_run_resumed_with_a_judge_refuses_cleanly(tmp_path: Path) -> None:
    from test_ev_run_fakes import Crash, FakeEvalClient, FakeJudge, load_questions
    from test_ev_run_runners import make_ctx

    from ragforge_client.evals.runners import RunAborted, find_resumable, run_generator
    from ragforge_client.evals.schema import EvalKind

    crashing = FakeEvalClient(crash_on="rebalance-triggers")
    ctx = make_ctx(tmp_path, crashing)
    with pytest.raises(Crash):
        await run_generator(ctx, load_questions(), None)
    started = find_resumable(ctx.runs_dir, EvalKind.generator, "golden")
    assert started is not None
    again = FakeEvalClient()
    with pytest.raises(RunAborted, match="started without a judge"):
        await run_generator(
            make_ctx(tmp_path, again, resume=started), load_questions(), FakeJudge(again.log)
        )
