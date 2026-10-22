"""EV-RUN: the `ragforge-eval2` CLI: argument parsing, the run/summarize/label/baseline/gate/
snapshot commands against fakes, the blind label flow, and the demo tenant setup.

CLI tests are synchronous on purpose: `main` runs its own event loop, as the console script does.
"""

from __future__ import annotations

import asyncio
import io
import shutil
import stat
from collections.abc import AsyncIterator, Mapping, Sequence
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID, uuid5

import pytest
from test_ev_run_fakes import (
    DEMO,
    FIXTURES,
    GOLDEN_PROMPTS,
    GOLDEN_RETRIEVAL,
    NS,
    SNAPSHOT_ID,
    FakeEvalClient,
    FakeJudge,
    FakeLabels,
    FakeSummarizer,
    load_questions,
)

from ragforge_client.ask import KeyFileError
from ragforge_client.evals.cli import (
    DEMO_KINDS,
    CliError,
    Components,
    DemoDeps,
    Env,
    Paths,
    Target,
    build_parser,
    eval_key_file,
    labelable,
    main,
    merge_verdicts,
    setup_demo,
)
from ragforge_client.evals.datasets import item_digest_of
from ragforge_client.evals.interfaces import Datasets, Judge, JudgeItem
from ragforge_client.evals.runfile import RunFile, header_digest, read_run
from ragforge_client.evals.runners import (
    Pacer,
    RetryPolicy,
    RunContext,
    run_generator,
    summary_path,
)
from ragforge_client.evals.schema import (
    AnswerOutput,
    Criterion,
    EvalKind,
    GateCheck,
    GateOutcome,
    GateVerdict,
    JudgeResult,
    LabelKind,
    Split,
    Surface,
)
from ragforge_contracts.api import (
    ApiKeyCreated,
    Collection,
    EvalReembedRequest,
    EvalSnapshot,
    EvalSnapshotRequest,
    Job,
    JobEvent,
    Tenant,
    TenantProvisioned,
)

KEY = "rf_live_TOPSECRETKEY"
NOW = datetime(2026, 10, 21, 12, 0, tzinfo=UTC)
COLLECTION = uuid5(NS, "collection")


# ------------------------------------------------------------------ the CLI harness
@dataclass
class Harness:
    root: Path
    key_file: Path
    client: FakeEvalClient
    summarizer: FakeSummarizer = field(default_factory=FakeSummarizer)
    labels: FakeLabels = field(default_factory=FakeLabels)
    answers: list[str] = field(default_factory=list)
    judges: list[tuple[str, Path]] = field(default_factory=list)
    verdict: GateOutcome = GateOutcome.pass_
    kind_verdicts: dict[EvalKind, GateOutcome] = field(default_factory=dict)  # per kind
    connected_with: list[str] = field(default_factory=list)
    out: io.StringIO = field(default_factory=io.StringIO)
    err: io.StringIO = field(default_factory=io.StringIO)
    clock: datetime = NOW

    @property
    def runs(self) -> Path:
        return self.root / "demo" / "runs"

    def env(self) -> Env:
        @asynccontextmanager
        async def connect(url: str, key: str) -> AsyncIterator[Any]:
            self.connected_with.append(key)
            yield self.client

        @asynccontextmanager
        async def open_judge(
            table: Mapping[str, Any], prompt: str, cache: Path
        ) -> AsyncIterator[Judge]:
            self.judges.append((prompt, cache))
            yield FakeJudge(self.client.log)

        def evaluate(
            baseline: RunFile, candidate: RunFile, aa: Sequence[RunFile], datasets: Datasets, /
        ) -> GateVerdict:
            outcome = self.kind_verdicts.get(candidate.header.eval_kind, self.verdict)
            check = GateCheck(
                name=f"{candidate.header.eval_kind.value}.x",
                outcome=outcome,
                method="exact",
                n=20,
            )
            return GateVerdict(
                surface=candidate.header.surface,
                baseline_run=baseline.header.run_id,
                candidate_run=candidate.header.run_id,
                checks=[check],
                outcome=outcome,
            )

        components = Components(
            summarize=self.summarizer,
            gate_evaluate=evaluate,
            open_judge=open_judge,
            label_store=lambda path: self.labels,
        )
        answers = iter(self.answers)
        return Env(
            components=components,
            connect=connect,
            environ={},
            out=self.out,
            err=self.err,
            prompt=lambda _: next(answers),
            now=lambda: self.clock,
        )

    def main(self, *argv: str) -> int:
        common = ["--root", str(self.root), "--key-file", str(self.key_file)]
        args = list(argv)
        # common options go after the (sub)subcommand, as argparse parents define them there
        split = 2 if args[0] in ("baseline", "snapshot") else 1
        return main([*args[:split], *common, *args[split:]], self.env())


@pytest.fixture
def harness(tmp_path: Path) -> Harness:
    demo = tmp_path / "evals" / "demo"
    (demo / "runs").mkdir(parents=True)
    for name in ("questions.toml", "verifier.json", "interview.json"):
        shutil.copy(DEMO / name, demo / name)
    Paths(tmp_path / "evals", "demo").target.write_text(
        Target(collection_ids=[COLLECTION]).model_dump_json()
    )
    key_file = tmp_path / "eval.key"
    key_file.write_text(KEY + "\n")
    key_file.chmod(0o600)
    return Harness(tmp_path / "evals", key_file, FakeEvalClient())


def _secret_free(h: Harness) -> None:
    assert KEY not in h.out.getvalue() + h.err.getvalue()


# ------------------------------------------------------------------ parsing
def test_parser_accepts_the_documented_commands() -> None:
    parser = build_parser()
    args = parser.parse_args(["run", "generator", "--dataset", "demo", "--arm", "A", "--resume"])
    assert (args.kind, args.dataset, args.arm, args.resume) == ("generator", "demo", "A", True)
    args = parser.parse_args(["gate", "--surface", "chat", "--override", "known flake"])
    assert (args.surface, args.override, args.dataset) == ("chat", "known flake", "private")
    args = parser.parse_args(["baseline", "set", "r.jsonl", "--note", "first"])
    assert (args.action, args.run, args.note) == ("set", "r.jsonl", "first")
    args = parser.parse_args(["snapshot", "reembed", "--space", "3"])
    assert (args.action, args.space) == ("reembed", 3)
    assert parser.parse_args(["demo"]).arm == "demo"


@pytest.mark.parametrize(
    "argv",
    [
        ["run", "nonsense", "--dataset", "demo", "--arm", "A"],
        ["run", "retrieval", "--dataset", "elsewhere", "--arm", "A"],
        ["run", "retrieval", "--dataset", "demo", "--arm", "../escape"],
        ["run", "retrieval", "--dataset", "demo", "--arm", "A", "--rpm", "0"],
        ["run", "retrieval", "--dataset", "demo"],
        ["gate", "--surface", "nope"],
        ["baseline", "set", "r.jsonl"],
    ],
)
def test_parser_rejects_bad_arguments(argv: list[str]) -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(argv)


# ------------------------------------------------------------------ run / summarize
def test_run_writes_the_run_and_its_summary(harness: Harness) -> None:
    assert harness.main("run", "retrieval", "--dataset", "demo", "--arm", "A") == 0
    path = Path(harness.out.getvalue().strip())
    run = read_run(path)
    assert run.footer is not None
    assert run.header.arm == "A"
    assert summary_path(path).exists()
    assert len(harness.summarizer.calls) == 1
    assert harness.connected_with == [KEY]
    _secret_free(harness)


def test_run_application_adds_the_latency_pass(harness: Harness) -> None:
    assert harness.main("run", "application", "--dataset", "demo", "--arm", "A") == 0
    kinds = [read_run(Path(p)).header.eval_kind.value for p in harness.out.getvalue().split()]
    assert kinds == ["application", "latency"]


@pytest.mark.parametrize(
    ("kind", "prompt"), [("generator", "judge_qa@1"), ("interview", "judge_question@1")]
)
def test_a_judge_config_opens_the_cached_judge_for_the_kind(
    harness: Harness, tmp_path: Path, kind: str, prompt: str
) -> None:
    config = tmp_path / "judge.toml"
    config.write_text('[judge]\nprovider = "ollama"\n')
    argv = ["run", kind, "--dataset", "demo", "--arm", "A", "--judge-config", str(config)]
    assert harness.main(*argv) == 0
    assert harness.judges == [(prompt, harness.root / "demo" / "judge-cache.jsonl")]


def test_run_without_a_target_fails_cleanly(harness: Harness) -> None:
    Paths(harness.root, "demo").target.unlink()
    assert harness.main("run", "retrieval", "--dataset", "demo", "--arm", "A") == 2
    assert "run target" in harness.err.getvalue()
    assert not harness.connected_with  # nothing touched the network


def test_a_key_file_readable_by_others_is_refused(harness: Harness) -> None:
    harness.key_file.chmod(0o644)
    assert harness.main("run", "retrieval", "--dataset", "demo", "--arm", "A") == 2
    assert "readable by others" in harness.err.getvalue()
    _secret_free(harness)


def test_resume_without_an_unfinished_run_fails_cleanly(harness: Harness) -> None:
    assert harness.main("run", "verifier", "--dataset", "demo", "--arm", "A", "--resume") == 2
    assert "no unfinished verifier run" in harness.err.getvalue()


def test_summarize_checks_the_dataset_digest(harness: Harness) -> None:
    run = harness.runs / "run-retrieval.jsonl"
    shutil.copy(FIXTURES / "run-retrieval.jsonl", run)
    assert harness.main("summarize", str(run)) == 0
    assert summary_path(run).exists()
    questions = harness.root / "demo" / "questions.toml"
    questions.write_text(questions.read_text().replace("in order in Kafka", "in order"))
    assert harness.main("summarize", str(run)) == 2
    assert "dataset changed" in harness.err.getvalue()


# ------------------------------------------------------------------ the blind label queue
def _generator_run(harness: Harness, judge: FakeJudge | None = None) -> Path:
    ctx = RunContext(
        client=harness.client,
        collection_ids=(COLLECTION,),
        arm="A",
        runs_dir=harness.runs,
        dataset_dir=harness.root / "demo",
        retrieval=GOLDEN_RETRIEVAL,
        prompt_versions=dict(GOLDEN_PROMPTS),
        policy=RetryPolicy(backoff_s=0.0),
        pacer=Pacer(),
        now=lambda: NOW,
    )
    return asyncio.run(run_generator(ctx, load_questions(), judge))


def test_the_label_queue_is_blind_and_shares_the_judge_digest(harness: Harness) -> None:
    judge = FakeJudge(harness.client.log)
    run_path = _generator_run(harness, judge)
    harness.answers = ["2", "1", "0", "q"]
    assert harness.main("label", str(run_path), "--batch", "b7") == 0
    (label,) = harness.labels.labels
    assert label.kind is LabelKind.criteria
    assert label.batch_id == "b7"
    assert label.scores == {
        Criterion.relevance: 2,
        Criterion.completeness: 1,
        Criterion.faithfulness: 0,
    }
    # the owner saw the question, the notes and the answer ...
    shown = harness.out.getvalue()
    row = next(r for r in read_run(run_path).rows if r.item_id == label.item_id)
    assert isinstance(row.output, AnswerOutput)
    assert row.output.answer in shown
    assert "Synthetic note for" in shown
    # ... and nothing of what is being calibrated: no verdict, no judge reason, no arm/model
    for hidden in ("verified", "partial", "fake", "fake-chat", "golden", "judge"):
        assert hidden not in shown.lower()
    # the label pairs with the judge's result for the same item
    assert label.item_digest in {i.item_digest for i in judge.items}
    _secret_free(harness)


def test_labelled_items_are_not_asked_again(harness: Harness) -> None:
    run_path = _generator_run(harness)
    harness.answers = ["2", "2", "2", "q"]
    harness.main("label", str(run_path))
    first = harness.labels.labels[0].item_id
    harness.answers = ["1", "1", "1", "q"]
    harness.main("label", str(run_path))
    assert [lb.item_id for lb in harness.labels.labels][1] != first


def test_skip_moves_on_without_writing(harness: Harness) -> None:
    run_path = _generator_run(harness)
    harness.answers = ["2", "s", "q"]
    harness.main("label", str(run_path))
    assert harness.labels.labels == []


def test_the_criteria_queue_is_the_stratified_sample(harness: Harness) -> None:
    run_path = _generator_run(harness)
    harness.answers = ["1"] * 30
    assert harness.main("label", str(run_path), "--sample", "2") == 0
    assert len(harness.labels.labels) == 2
    assert "2 label(s) written, 0 left" in harness.out.getvalue()
    # a second session samples only from what is still unlabelled
    harness.main("label", str(run_path), "--sample", "2")
    ids = [lb.item_id for lb in harness.labels.labels]
    assert len(ids) == 4
    assert len(set(ids)) == 4


def test_heldout2_rows_never_reach_the_label_queue(harness: Harness) -> None:
    run = read_run(_generator_run(harness))
    qs = load_questions()
    answered = [
        r.item_id for r in run.rows if isinstance(r.output, AnswerOutput) and not r.output.abstained
    ]
    sealed = qs.model_copy(
        update={
            "questions": [
                q.model_copy(update={"split": Split.heldout2}) if q.id == answered[0] else q
                for q in qs.questions
            ]
        }
    )
    ids = [row.item_id for row, _ in labelable(run, sealed)]
    assert answered[0] not in ids
    assert set(ids) == set(answered[1:])


def test_label_digest_includes_the_facts(harness: Harness) -> None:
    run_path = _generator_run(harness)
    harness.answers = ["0", "0", "0", "q"]
    harness.main("label", str(run_path))
    (label,) = harness.labels.labels
    q = next(q for q in load_questions().questions if q.id == label.item_id)
    row = next(r for r in read_run(run_path).rows if r.item_id == label.item_id)
    assert isinstance(row.output, AnswerOutput)
    notes = [f"Synthetic note for {p}." for p in row.output.cited_pages]
    facts = [f.fact for f in q.facts]
    assert label.item_digest == item_digest_of(q.question, row.output.answer, notes, facts)


@dataclass
class RecordingJudge(FakeJudge):
    results: list[JudgeResult] = field(default_factory=list)

    async def score(self, items: Sequence[JudgeItem]) -> list[JudgeResult]:
        out = await super().score(items)
        self.results.extend(out)
        return out


def test_the_adjudication_queue_holds_only_flagged_verified_answers(harness: Harness) -> None:
    judge = RecordingJudge(harness.client.log)
    run_path = _generator_run(harness, judge)
    verified = [
        r
        for r in read_run(run_path).rows
        if isinstance(r.output, AnswerOutput) and r.output.status == "verified" and r.judge_refs
    ]
    assert len(verified) >= 2
    flagged = verified[0]
    cache = [  # the judge flags only `flagged` (faithfulness 1); every other score is a 2
        r.model_copy(
            update={
                "score": 1
                if r.criterion is Criterion.faithfulness and r.cache_key in flagged.judge_refs
                else 2,
                "reason": "JUDGE-REASON",
            }
        ).model_dump_json()
        for r in judge.results
    ]
    (harness.root / "demo" / "judge-cache.jsonl").write_text("\n".join(cache) + "\n")
    harness.answers = ["y"]
    assert harness.main("label", str(run_path), "--adjudicate", "--batch", "adj1") == 0
    (label,) = harness.labels.labels
    assert (label.kind, label.item_id, label.confirmed) == (
        LabelKind.adjudication,
        flagged.item_id,
        True,
    )
    # the label joins the judge result it adjudicates (metrics.confirmed_false_verified)
    faithfulness = {r.item_digest for r in judge.results if r.cache_key in flagged.judge_refs}
    assert label.item_digest in faithfulness
    shown = harness.out.getvalue()
    assert "JUDGE-REASON" not in shown
    assert "1 label(s) written, 0 left" in shown
    _secret_free(harness)


def test_adjudication_needs_a_judged_generator_run(harness: Harness) -> None:
    run_path = _generator_run(harness)  # no judge
    assert harness.main("label", str(run_path), "--adjudicate") == 2
    assert "judged generator run" in harness.err.getvalue()


# ------------------------------------------------------------------ baseline and gate
def _two_retrieval_runs(harness: Harness) -> tuple[Path, Path]:
    assert harness.main("run", "retrieval", "--dataset", "demo", "--arm", "base") == 0
    assert harness.main("run", "retrieval", "--dataset", "demo", "--arm", "cand") == 0
    base, cand = harness.out.getvalue().split()
    return Path(base), Path(cand)


def _verifier_floor(harness: Harness, base_arm: str, cand_arm: str) -> Path:
    """A verifier baseline plus a candidate run: the E-V hard floor every chat gate needs."""
    for arm in (base_arm, cand_arm):
        harness.out.truncate(0)
        harness.out.seek(0)
        assert harness.main("run", "verifier", "--dataset", "demo", "--arm", arm) == 0
        path = Path(harness.out.getvalue().split()[-1])
        if arm == base_arm:
            assert harness.main("baseline", "set", str(path), "--note", "verifier base") == 0
    harness.out.truncate(0)
    harness.out.seek(0)
    return path


def test_gate_blocks_and_an_override_is_recorded(harness: Harness) -> None:
    base, cand = _two_retrieval_runs(harness)
    assert harness.main("baseline", "set", str(base), "--note", "first baseline") == 0
    _verifier_floor(harness, "base", "cand")
    assert (harness.runs / "baseline-chat-retrieval.json").exists()
    harness.verdict = GateOutcome.block
    assert harness.main("gate", "--surface", "chat", "--dataset", "demo") == 1
    verdict = GateVerdict.model_validate_json((harness.runs / "gate-chat.json").read_text())
    assert verdict.outcome is GateOutcome.block
    assert verdict.candidate_run.split(",")[0] == read_run(cand).header.run_id

    before = header_digest(read_run(cand).header)
    argv = ["gate", "--surface", "chat", "--dataset", "demo", "--override", "flaky host"]
    assert harness.main(*argv) == 0
    after = read_run(cand)
    assert after.header.override == "flaky host"
    assert after.footer is not None
    assert header_digest(after.header) == before  # the configuration identity is unchanged
    assert stat.S_IMODE(cand.stat().st_mode) == 0o600


def test_the_chat_gate_fails_when_the_verifier_floor_blocks(harness: Harness) -> None:
    base, _ = _two_retrieval_runs(harness)
    assert harness.main("baseline", "set", str(base), "--note", "b") == 0
    v_cand = _verifier_floor(harness, "base", "cand")
    harness.kind_verdicts = {
        EvalKind.retrieval: GateOutcome.pass_,
        EvalKind.verifier: GateOutcome.block,
    }
    assert harness.main("gate", "--surface", "chat", "--dataset", "demo") == 1
    verdict = GateVerdict.model_validate_json((harness.runs / "gate-chat.json").read_text())
    assert verdict.outcome is GateOutcome.block
    assert {c.name for c in verdict.checks} == {"retrieval.x", "verifier.x"}
    assert read_run(v_cand).header.run_id in verdict.candidate_run
    harness.kind_verdicts = {}
    assert harness.main("gate", "--surface", "chat", "--dataset", "demo") == 0


def test_the_chat_gate_refuses_without_the_verifier_floor(harness: Harness) -> None:
    base, _ = _two_retrieval_runs(harness)
    assert harness.main("baseline", "set", str(base), "--note", "b") == 0
    assert harness.main("gate", "--surface", "chat", "--dataset", "demo") == 2
    assert "hard floor needs a verifier baseline" in harness.err.getvalue()
    assert not (harness.runs / "gate-chat.json").exists()


@pytest.mark.parametrize(
    ("outcome", "code"),
    [(GateOutcome.not_evaluable, 1), (GateOutcome.block, 0), (GateOutcome.warn, 0)],
)
def test_an_override_excuses_a_block_never_a_not_evaluable(
    harness: Harness, outcome: GateOutcome, code: int
) -> None:
    base, _ = _two_retrieval_runs(harness)
    assert harness.main("baseline", "set", str(base), "--note", "b") == 0
    _verifier_floor(harness, "base", "cand")
    harness.kind_verdicts = {EvalKind.verifier: outcome}
    argv = ["gate", "--surface", "chat", "--dataset", "demo", "--override", "known flake"]
    assert harness.main(*argv) == code
    verdict = GateVerdict.model_validate_json((harness.runs / "gate-chat.json").read_text())
    assert verdict.override == "known flake"


def test_an_earlier_override_does_not_pass_a_new_gate(harness: Harness) -> None:
    base, _ = _two_retrieval_runs(harness)
    assert harness.main("baseline", "set", str(base), "--note", "b") == 0
    _verifier_floor(harness, "base", "cand")
    harness.verdict = GateOutcome.block
    argv = ["gate", "--surface", "chat", "--dataset", "demo"]
    assert harness.main(*argv, "--override", "once") == 0
    assert harness.main(*argv) == 1


def test_gate_without_a_baseline_fails_cleanly(harness: Harness) -> None:
    assert harness.main("gate", "--surface", "verifier", "--dataset", "demo") == 2
    assert "set a baseline first" in harness.err.getvalue()


def test_a_baseline_must_be_a_finished_run(harness: Harness) -> None:
    unfinished = harness.runs / "partial.jsonl"
    lines = (FIXTURES / "run-retrieval.jsonl").read_text().splitlines()[:-1]
    unfinished.write_text("\n".join(lines) + "\n")
    assert harness.main("baseline", "set", str(unfinished), "--note", "x") == 2


def test_merged_verdicts_take_the_worst_outcome_in_the_gates_order() -> None:
    def verdict(outcome: GateOutcome) -> GateVerdict:
        check = GateCheck(name=outcome.value, outcome=outcome, method="m", n=1)
        return GateVerdict(
            surface=Surface.chat,
            baseline_run="b",
            candidate_run="c",
            checks=[check],
            outcome=outcome,
        )

    merged = merge_verdicts(
        Surface.chat, [verdict(GateOutcome.block), verdict(GateOutcome.not_evaluable)], None
    )
    assert merged.outcome is GateOutcome.not_evaluable
    assert len(merged.checks) == 2
    assert (
        merge_verdicts(
            Surface.chat, [verdict(GateOutcome.pass_), verdict(GateOutcome.warn)], None
        ).outcome
        is GateOutcome.warn
    )


# ------------------------------------------------------------------ snapshot
class SnapshotClient(FakeEvalClient):
    async def eval_snapshot(self, request: EvalSnapshotRequest) -> EvalSnapshot:
        self.log.append(f"snapshot-create:{request.source_collection_id}")
        return EvalSnapshot(
            collection_id=SNAPSHOT_ID, eval_tenant_id=uuid5(NS, "t"), job_id=uuid5(NS, "j")
        )

    async def eval_reembed(self, collection_id: UUID, request: EvalReembedRequest) -> EvalSnapshot:
        self.log.append(f"reembed:{collection_id}:{request.space_id}")
        return await self.eval_snapshot_status(collection_id)


def test_snapshot_create_then_status_and_reembed(harness: Harness) -> None:
    harness.client = SnapshotClient()
    source = uuid5(NS, "live")
    assert harness.main("snapshot", "create", "--source-collection", str(source)) == 0
    target = Target.model_validate_json(Paths(harness.root, "private").target.read_text())
    assert target.snapshot_collection_id == SNAPSHOT_ID
    assert target.collection_ids == [SNAPSHOT_ID]
    assert harness.main("snapshot", "status") == 0
    assert "content content-d, index index-d" in harness.out.getvalue()
    assert harness.main("snapshot", "reembed", "--space", "2") == 0
    assert f"reembed:{SNAPSHOT_ID}:2" in harness.client.log


class TenantClient(FakeEvalClient):
    async def eval_tenant_create(self) -> TenantProvisioned:
        self.log.append("eval-tenant-create")
        key = ApiKeyCreated(
            id=uuid5(NS, "ek"), name="eval", prefix="rf", scopes=["eval", "chat", "read"],
            created_at=NOW, last_used_at=None, revoked_at=None, secret="EVAL-SECRET",
        )  # fmt: skip
        tenant = Tenant(id=uuid5(NS, "eval-tenant"), name="eval", kind="eval", created_at=NOW)
        return TenantProvisioned(tenant=tenant, user_id=uuid5(NS, "eu"), api_key=key)


def test_tenant_create_writes_the_eval_key_0600_and_never_prints_it(
    harness: Harness, tmp_path: Path
) -> None:
    harness.client = TenantClient()
    key_path = tmp_path / "keys" / "eval.key"
    assert harness.main("tenant", "create", "--eval-key-file", str(key_path)) == 0
    assert key_path.read_text().strip() == "EVAL-SECRET"
    assert key_path.stat().st_mode & 0o777 == 0o600
    assert "EVAL-SECRET" not in harness.out.getvalue() + harness.err.getvalue()
    assert str(key_path) in harness.out.getvalue()
    # a second provisioning is refused before any call: the key file already exists
    assert harness.main("tenant", "create", "--eval-key-file", str(key_path)) == 2
    assert harness.client.log.count("eval-tenant-create") == 1


# ------------------------------------------------------------------ demo setup
DEMO_SNAPSHOT = uuid5(NS, "demo-snapshot")


@dataclass
class FakeOwner:
    """The demo tenant's owner key: collection, uploads, the eval pair and the snapshot."""

    status: str = "done"
    log: list[str] = field(default_factory=list)
    status_polls: int = 0

    async def create_collection(self, name: str) -> Collection:
        self.log.append(f"collection:{name}")
        return Collection(id=COLLECTION, name=name, active_space_id=1, created_at=NOW)

    async def eval_tenant_create(self) -> TenantProvisioned:
        self.log.append("eval-tenant-create")
        key = ApiKeyCreated(
            id=uuid5(NS, "k2"), name="eval", prefix="rf", scopes=["eval", "chat", "read"],
            created_at=NOW, last_used_at=None, revoked_at=None, secret="EVAL-SECRET",
        )  # fmt: skip
        tenant = Tenant(id=uuid5(NS, "eval-t"), name="eval", kind="eval", created_at=NOW)
        return TenantProvisioned(tenant=tenant, user_id=uuid5(NS, "u"), api_key=key)

    async def eval_snapshot(self, request: EvalSnapshotRequest) -> EvalSnapshot:
        self.log.append(f"snapshot:{request.source_collection_id}")
        return EvalSnapshot(
            collection_id=DEMO_SNAPSHOT, eval_tenant_id=uuid5(NS, "eval-t"), job_id=uuid5(NS, "j")
        )

    async def eval_snapshot_status(self, collection_id: UUID) -> EvalSnapshot:
        self.status_polls += 1  # the digests land on the second poll
        done = self.status_polls >= 2
        return EvalSnapshot(
            collection_id=collection_id, eval_tenant_id=uuid5(NS, "eval-t"),
            job_id=uuid5(NS, "j"), content_digest="c" if done else None,
            index_digest="i" if done else None,
        )  # fmt: skip

    async def job_events(self, job_id: UUID) -> AsyncIterator[JobEvent]:
        yield JobEvent(job_id=job_id, status="running")

    async def get_job(self, job_id: UUID) -> Job:
        return Job(
            id=job_id, kind="ingest", status=self.status, attempts=1, last_error=None,
            enqueued_at=NOW,
        )  # fmt: skip


def _demo_deps(tmp_path: Path, owner: FakeOwner, uploads: list[tuple[str, str]]) -> DemoDeps:
    def connect_owner(key: str) -> AbstractAsyncContextManager[Any]:
        @asynccontextmanager
        async def cm() -> AsyncIterator[Any]:
            assert key == "OWNER-SECRET"
            yield owner

        return cm()

    def uploader(key: str) -> Any:
        async def upload(collection_id: UUID, name: str, content: bytes) -> UUID:
            uploads.append((key, name))
            return uuid5(NS, name)

        return upload

    return DemoDeps(
        owner_key="OWNER-SECRET", owner=connect_owner, uploader=uploader,
        key_path=tmp_path / "eval" / "demo.key", poll_s=0.0,
    )  # fmt: skip


def _env(out: io.StringIO, err: io.StringIO) -> Env:
    return Env(components=Components(), out=out, err=err, now=lambda: NOW, environ={})


def test_demo_setup_walks_the_owner_path_and_targets_the_snapshot(tmp_path: Path) -> None:
    """Integrator fix: the demo uses the owner's real path (upload -> eval pair -> snapshot), so
    its runs hit an EVAL tenant's snapshot exactly like the private runs."""
    owner, uploads = FakeOwner(), list[tuple[str, str]]()
    deps = _demo_deps(tmp_path, owner, uploads)
    paths = Paths(tmp_path / "evals", "demo")
    pages = [(p.name, p.read_bytes()) for p in sorted((DEMO / "pages").glob("*.md"))]
    out, err = io.StringIO(), io.StringIO()
    target = asyncio.run(setup_demo(deps, pages, paths, _env(out, err)))
    assert target.collection_ids == [DEMO_SNAPSHOT]
    assert target.snapshot_collection_id == DEMO_SNAPSHOT
    assert Target.model_validate_json(paths.target.read_text()) == target
    assert [n for _, n in uploads] == [n for n, _ in pages]
    assert len(pages) == 15
    assert owner.log[-2:] == ["eval-tenant-create", f"snapshot:{COLLECTION}"]
    assert deps.key_path.read_text().strip() == "EVAL-SECRET"
    assert stat.S_IMODE(deps.key_path.stat().st_mode) == 0o600
    assert "SECRET" not in out.getvalue() + err.getvalue()
    # a second demo run reuses the eval pair (no second provisioning) and snapshots again
    again = FakeOwner()
    asyncio.run(setup_demo(_demo_deps(tmp_path, again, []), pages, paths, _env(out, err)))
    assert "eval-tenant-create" not in again.log
    assert f"snapshot:{COLLECTION}" in again.log


def test_demo_setup_stops_on_a_failed_ingest(tmp_path: Path) -> None:
    deps = _demo_deps(tmp_path, FakeOwner(status="failed"), [])
    env = _env(io.StringIO(), io.StringIO())
    with pytest.raises(CliError, match="ended failed"):
        asyncio.run(setup_demo(deps, [("a.md", b"# A")], Paths(tmp_path, "demo"), env))


def test_the_answer_queue_never_asks_a_labelled_item_again(harness: Harness) -> None:
    # application runs label correct/invented (no calibration sample in front of the queue)
    assert harness.main("run", "application", "--dataset", "demo", "--arm", "A") == 0
    app_run = harness.out.getvalue().split()[0]
    harness.answers = ["y", "n", "q"]
    assert harness.main("label", app_run) == 0
    (first,) = harness.labels.labels
    assert first.kind is LabelKind.answer
    assert (first.correct, first.invented) == (True, False)
    harness.answers = ["n", "y", "q"]
    harness.main("label", app_run)
    second = harness.labels.labels[1]
    assert second.item_digest != first.item_digest


# ------------------------------------------------------------------ review round 1
def _retrieval_run(harness: Harness, arm: str, day: int) -> Path:
    harness.clock = datetime(2026, 10, day, 12, 0, tzinfo=UTC)
    harness.out.truncate(0)
    harness.out.seek(0)
    assert harness.main("run", "retrieval", "--dataset", "demo", "--arm", arm) == 0
    return Path(harness.out.getvalue().split()[-1])


def test_gate_picks_the_newest_run_of_the_arm_not_the_name_that_sorts_last(
    harness: Harness,
) -> None:
    base = _retrieval_run(harness, "A", 1)
    _retrieval_run(harness, "zz-exp", 2)  # sorts after every arm-A file by name
    newest = _retrieval_run(harness, "A", 21)
    assert harness.main("baseline", "set", str(base), "--note", "b") == 0
    _verifier_floor(harness, "vbase", "A")

    # two arms and no --arm: refuse rather than gate a stale experiment
    assert harness.main("gate", "--surface", "chat", "--dataset", "demo") == 2
    assert "several arms (A, zz-exp)" in harness.err.getvalue()

    assert harness.main("gate", "--surface", "chat", "--dataset", "demo", "--arm", "A") == 0
    verdict = GateVerdict.model_validate_json((harness.runs / "gate-chat.json").read_text())
    assert verdict.candidate_run.split(",")[0] == read_run(newest).header.run_id


def test_gate_orders_candidates_by_start_time_not_filename(harness: Harness) -> None:
    base = _retrieval_run(harness, "A", 1)
    late = _retrieval_run(harness, "B", 21)
    early = _retrieval_run(harness, "B", 2)
    # give the older run a name that sorts last: the gate must still pick the newer one
    early.rename(early.with_name("retrieval-B-99999999T000000Z.jsonl"))
    assert harness.main("baseline", "set", str(base), "--note", "b") == 0
    _verifier_floor(harness, "vbase", "B")
    assert harness.main("gate", "--surface", "chat", "--dataset", "demo", "--arm", "B") == 0
    verdict = GateVerdict.model_validate_json((harness.runs / "gate-chat.json").read_text())
    assert verdict.candidate_run.split(",")[0] == read_run(late).header.run_id


def test_each_dataset_uses_its_own_eval_key(tmp_path: Path) -> None:
    key_dir = tmp_path / "eval"
    (key_dir / "demo").mkdir(parents=True)
    demo_key = key_dir / "demo" / "demo.key"
    demo_key.write_text("D\n")
    # only the demo key exists: the private run does NOT quietly pick it up
    with pytest.raises(KeyFileError, match="no API key"):
        eval_key_file("private", None, {}, key_dir)
    assert eval_key_file("demo", None, {}, key_dir) == demo_key
    private = key_dir / "notes.key"
    private.write_text("P\n")
    assert eval_key_file("private", None, {}, key_dir) == private
    assert eval_key_file("demo", None, {"RAGFORGE_EVAL_KEY_FILE": str(private)}, key_dir) == (
        demo_key
    )
    with pytest.raises(CliError, match="demo tenant's key"):
        eval_key_file("private", str(demo_key), {}, key_dir)


def test_collection_override_is_refused_for_private_runs(harness: Harness) -> None:
    argv = ["run", "retrieval", "--dataset", "private", "--arm", "A"]
    (harness.root / "private").mkdir()
    shutil.copy(DEMO / "questions.toml", harness.root / "private" / "questions.toml")
    assert harness.main(*argv, "--collection", str(COLLECTION)) == 2
    assert "uses the snapshot" in harness.err.getvalue()
    assert not harness.connected_with


def test_judge_rpm_is_parsed_for_run_and_demo() -> None:
    parser = build_parser()
    args = parser.parse_args(
        ["run", "generator", "--dataset", "demo", "--arm", "A", "--judge-rpm", "2"]
    )
    assert args.judge_rpm == 2.0
    assert parser.parse_args(["demo", "--judge-rpm", "1"]).judge_rpm == 1.0


def test_demo_runs_the_latency_pass_before_any_judged_kind() -> None:
    order = list(DEMO_KINDS)
    assert set(order) == {
        EvalKind.retrieval,
        EvalKind.generator,
        EvalKind.verifier,
        EvalKind.interview,
        EvalKind.application,
    }
    assert order.index(EvalKind.application) < order.index(EvalKind.generator)
    assert order.index(EvalKind.application) < order.index(EvalKind.interview)


def test_seal_stamps_a_heldout_file_and_an_edit_then_fails_the_load(
    harness: Harness, tmp_path: Path
) -> None:
    from ragforge_client.evals.datasets import DatasetError, load_questions

    held = tmp_path / "heldout2.toml"
    held.write_text(
        'schema_version = 2\nname = "private"\n\n[[questions]]\nid = "h-one"\n'
        'split = "heldout2"\nquestion = "Why does ISR shrink?"\n'
        'relevance = { "notion:0001" = 2 }\n'
    )
    assert harness.main("seal", str(held)) == 0
    assert "sealed" in harness.out.getvalue()
    load_questions(held, require_seal=True)  # sealed and unchanged: loads
    held.write_text(held.read_text().replace("Why does ISR shrink?", "Why does ISR grow?"))
    with pytest.raises(DatasetError):
        load_questions(held, require_seal=True)


# ------------------------------------------------------------------ gate evidence freshness (P1-2)
def test_a_hard_floor_run_of_another_verifier_refuses_the_gate() -> None:
    """Final review P1-2: a chat candidate with ground_check@4 must not pass on an E-V run made
    with @3 (evidence from before the change)."""
    from dataclasses import replace

    from ragforge_client.evals.cli import CliError, same_build

    gen = read_run(FIXTURES / "run-generator.jsonl")
    ver = read_run(FIXTURES / "run-verifier.jsonl")
    changed = replace(
        gen,
        header=gen.header.model_copy(
            update={"prompt_versions": {**gen.header.prompt_versions, "verifier": "ground_check@4"}}
        ),
    )
    same_build([gen], [ver])  # the same verifier: accepted
    with pytest.raises(CliError, match="another verifier"):
        same_build([changed], [ver])
    with pytest.raises(CliError, match="different builds"):
        same_build([gen, changed], [])


def test_a_candidate_older_than_its_baseline_refuses_the_gate(harness: Harness) -> None:
    older = _retrieval_run(harness, "A", 1)
    newer = _retrieval_run(harness, "A", 21)
    assert older.exists()
    assert harness.main("baseline", "set", str(newer), "--note", "b") == 0
    _verifier_floor(harness, "vbase", "A")
    code = harness.main("gate", "--surface", "chat", "--dataset", "demo", "--candidate", str(older))
    assert code == 2
    assert "older than the baseline" in harness.err.getvalue()


def test_a_retrieval_only_gate_needs_no_verifier_floor(harness: Harness) -> None:
    """ADR-0017 §5: a retrieval change gets the cheap retrieval gate; the E-V hard floor guards
    chat/verifier changes (pinned by the other gate tests, which gate chat answers)."""
    base = _retrieval_run(harness, "A", 1)
    cand = _retrieval_run(harness, "A", 2)
    assert harness.main("baseline", "set", str(base), "--note", "b") == 0
    code = harness.main(
        "gate", "--surface", "chat", "--dataset", "demo", "--kind", "retrieval",
        "--candidate", str(cand),
    )  # fmt: skip
    assert code == 0


def test_a_dataset_arm_toml_fills_every_run_header(tmp_path: Path) -> None:
    """τ fix: without --arm-config the dataset's own arm.toml is used, so the retrieval config
    (incl. tau) is in every header and unanswerable questions get their below-tau check."""
    from ragforge_client.evals.cli import load_arm_config

    (tmp_path / "arm.toml").write_text(
        'chat_model = "m"\n\n[retrieval]\nembed_model = "e"\nspace_id = 1\n'
        'chunker_version = "c"\ntau = 0.35\nrerank_model = "r"\nrerank_top = 20\n'
    )
    arm = load_arm_config(None, tmp_path)
    assert arm.retrieval is not None
    assert arm.retrieval.tau == 0.35
    assert load_arm_config(None, tmp_path / "missing").retrieval is None
