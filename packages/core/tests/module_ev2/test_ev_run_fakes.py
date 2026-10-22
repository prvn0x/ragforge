"""Fakes for the EV-RUN tests (no tests here): a fake eval SDK client that REPLAYS the golden D0
fixtures as server responses, a fake judge and a fake summarizer.

The fake answers each request from the fixture row of the same item, so a runner driven by it
must reproduce the fixture rows row for row: that is the equivalence the tests check. Note text
comes from `evals/demo/chunks.json` (synthetic, committed); nothing private is ever read.
"""

from __future__ import annotations

import hashlib
import json
import tomllib
from collections import Counter
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import NAMESPACE_URL, UUID, uuid5

from ragforge_client.client import ApiError
from ragforge_client.evals.interfaces import Datasets, JudgeItem
from ragforge_client.evals.runfile import RunFile, read_run
from ragforge_client.evals.runners import key_id
from ragforge_client.evals.schema import (
    AnswerOutput,
    GradeOutput,
    InterviewSet,
    JudgeKey,
    JudgeResult,
    Label,
    Method,
    MetricValue,
    QuestionSet,
    RetrievalConfig,
    RetrievalOutput,
    RunRow,
    RunSummary,
    VerifierSet,
    VerifyOutput,
    digest,
    page_key_str,
    parse_page_key,
)
from ragforge_contracts.api import (
    ChunkKey,
    Citation,
    EvalChunk,
    EvalChunksResponse,
    EvalCoverageUnit,
    EvalResolveRequest,
    EvalResolveResponse,
    EvalSnapshot,
    EvalVerifierClaim,
    EvalVerifyRequest,
    EvalVerifyResponse,
    InterviewGrade,
    InterviewGradeRequest,
    InterviewQuestion,
    InterviewQuestionRequest,
    Problem,
    QueryResponse,
    ResolvedChunk,
    ResolvedPage,
    RetrievalStages,
    ShownSource,
    StageHit,
)
from ragforge_contracts.sse import (
    ChatEvent,
    CitationEvent,
    DoneEvent,
    ErrorEvent,
    TokenEvent,
    Usage,
    VerificationEvent,
)

ROOT = Path(__file__).resolve().parents[4]
DEMO = ROOT / "evals" / "demo"
FIXTURES = DEMO / "fixtures"
NS = uuid5(NAMESPACE_URL, "ragforge-ev-run-tests")

# the golden headers' configuration (make_fixtures.py), so header digests can be compared
GOLDEN_RETRIEVAL = RetrievalConfig(
    embed_model="nomic-embed-text",
    space_id=1,
    chunker_version="heading-v1",
    tau=0.35,
    rerank_model="ms-marco-MiniLM-L-12-v2",
    rerank_top=20,
)
GOLDEN_PROMPTS = {"chat": "answer@11", "verifier": "ground_check@3"}
GOLDEN_JUDGE = JudgeKey(
    provider="ollama",
    model_version="qwen3:8b-q4_K_M",
    decoding={"temperature": 0, "num_ctx": 8192},
    prompt="judge_qa@1",
)
SNAPSHOT_ID = uuid5(NS, "snapshot")


def load_questions() -> QuestionSet:
    return QuestionSet.model_validate(tomllib.loads((DEMO / "questions.toml").read_text()))


def load_verifier() -> VerifierSet:
    return VerifierSet.model_validate_json((DEMO / "verifier.json").read_text())


def load_interview() -> InterviewSet:
    return InterviewSet.model_validate_json((DEMO / "interview.json").read_text())


def demo_datasets() -> Datasets:
    return Datasets(
        questions=load_questions(), verifier=load_verifier(), interview=load_interview()
    )


def golden(kind: str) -> RunFile:
    return read_run(FIXTURES / f"run-{kind}.jsonl")


def document_id(page: str) -> UUID:
    return uuid5(NS, f"doc:{page}")


def page_chunk_id(page: str) -> UUID:
    """Retrieval / chat: one synthetic chunk per page."""
    return uuid5(NS, f"chunk:{page}")


def page_text(page: str) -> str:
    return f"Synthetic note for {page}."


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def comparable(row: RunRow, *drop: str) -> dict[str, Any]:
    """A row as JSON without the fields a test does not compare (wall-clock timings, ...)."""
    data = row.model_dump(mode="json")
    for name in drop:
        if "." in name:
            outer, inner = name.split(".", 1)
            if isinstance(data.get(outer), dict):
                data[outer].pop(inner, None)
        else:
            data.pop(name, None)
    return data


class Crash(Exception):
    """Not an item error: stands for a killed process (the runner must not swallow it)."""


@dataclass
class FakeEvalClient:
    """Replays the golden fixtures. `fail[(method, item)]` is a queue of exceptions raised
    before answering; `crash_on` raises Crash for that item (simulates a killed process)."""

    log: list[str] = field(default_factory=list)
    fail: dict[tuple[str, str], list[Exception]] = field(default_factory=dict)
    crash_on: str | None = None
    missing_pages: set[str] = field(default_factory=set)
    stale_chunks: set[str] = field(default_factory=set)  # key_id of chunk keys
    changed_text: set[UUID] = field(default_factory=set)  # chunk ids whose text moved
    snapshot_digests: tuple[str | None, str | None] = ("content-d", "index-d")
    prompt_versions: dict[str, str] = field(default_factory=lambda: dict(GOLDEN_PROMPTS))
    verifier_prompt: str | None = None
    verifier_model: str | None = None
    with_shown: bool = True
    pad_shown: str | None = None  # a page shown on EVERY turn (an abstain still saw notes)
    chat_model: str = "fake-chat"  # what `done.usage` reports as the answering model
    inferred_citations: bool = False  # every citation server-inferred (the model cited none)
    with_verification: bool = True
    verifier_usage: str | None = None  # a verifier model in `usage` after the chat model's
    verifier_only_when_ran: bool = False  # a real server reports no verifier it did not run
    extra_question_chunk: UUID | None = None  # the server widens a generated question's notes

    def __post_init__(self) -> None:
        self.questions = load_questions()
        self.verifier = load_verifier()
        self.interview = load_interview()
        self.chunks: list[dict[str, Any]] = json.loads((DEMO / "chunks.json").read_text())
        self._chunk_text = {
            uuid5(NS, key_id(ChunkKey.model_validate(c["key"]))): (
                ChunkKey.model_validate(c["key"]),
                c["text"],
            )
            for c in self.chunks
        }
        self._retrieval = {r.item_id: r for r in golden("retrieval").rows}
        self._generator = {r.item_id: r for r in golden("generator").rows}
        self._verify = {r.item_id: r for r in golden("verifier").rows}
        self._interview = {r.item_id: r for r in golden("interview").rows}
        self._qid = {q.question: q.id for q in self.questions.questions}
        self._vid = {i.answer: i.id for i in self.verifier.items}
        self._grades: Counter[tuple[str, str, str]] = Counter()
        self._pages = {p for r in self._retrieval.values() if r.output for p in self._stages(r)} | {
            page_key_str(ChunkKey.model_validate(c["key"]).page) for c in self.chunks
        }

    @staticmethod
    def _stages(row: RunRow) -> set[str]:
        assert isinstance(row.output, RetrievalOutput)
        return {p for pages in row.output.stages.values() for p in pages}

    def _step(self, method: str, item: str) -> None:
        self.log.append(f"{method}:{item}")
        if self.crash_on == item:
            raise Crash(item)
        queue = self.fail.get((method, item))
        if queue:
            raise queue.pop(0)

    # ------------------------------------------------------------- identity
    async def eval_resolve(self, request: EvalResolveRequest) -> EvalResolveResponse:
        self.log.append("resolve")
        pages = [
            ResolvedPage(key=k, status="missing")
            if page_key_str(k) in self.missing_pages or page_key_str(k) not in self._pages
            else ResolvedPage(key=k, status="ok", document_id=document_id(page_key_str(k)))
            for k in request.page_keys
        ]
        chunks = [
            ResolvedChunk(key=k, status="stale", candidates=[uuid5(NS, "candidate")])
            if key_id(k) in self.stale_chunks
            else ResolvedChunk(key=k, status="ok", chunk_id=uuid5(NS, key_id(k)))
            for k in request.chunk_keys
        ]
        return EvalResolveResponse(pages=pages, chunks=chunks)

    async def eval_chunks(self, chunk_ids: Sequence[UUID]) -> EvalChunksResponse:
        self.log.append("chunks")
        out: list[EvalChunk] = []
        for cid in chunk_ids:
            if cid in self._chunk_text:
                key, text = self._chunk_text[cid]
                page = key.page
                heading = key.heading_path
            else:
                name = next((p for p in self._pages if page_chunk_id(p) == cid), None)
                if name is None:
                    continue
                page, heading, text = parse_page_key(name), [name], page_text(name)
            shown = text + " (edited)" if cid in self.changed_text else text
            out.append(
                EvalChunk(
                    chunk_id=cid,
                    document_id=document_id(page_key_str(page)),
                    page_key=page,
                    heading_path=heading,
                    ordinal=0,
                    display_text=shown,
                    display_sha256=sha(shown),
                    # like the server: the stable key of what is served (occurrence 0 here)
                    chunk_key=ChunkKey(
                        page=page, heading_path=list(heading), display_sha256=sha(shown)
                    ),
                )
            )
        return EvalChunksResponse(chunks=out)

    async def eval_snapshot_status(self, collection_id: UUID) -> EvalSnapshot:
        self.log.append("snapshot")
        content, index = self.snapshot_digests
        return EvalSnapshot(
            collection_id=collection_id,
            eval_tenant_id=uuid5(NS, "eval-tenant"),
            job_id=uuid5(NS, "job"),
            content_digest=content,
            index_digest=index,
        )

    # ------------------------------------------------------------- E-R
    async def query(
        self, query: str, collection_ids: Sequence[UUID], *, k: int = 8, stages: bool = False
    ) -> QueryResponse:
        item = self._qid[query]
        self._step("query", item)
        out = self._retrieval[item].output
        assert isinstance(out, RetrievalOutput)
        assert stages

        def hits(pages: list[str], top: float | None) -> list[StageHit]:
            return [
                StageHit(
                    chunk_id=page_chunk_id(p),
                    document_id=document_id(p),
                    rank=i + 1,
                    score=(top if top is not None and i == 0 else 0.9 - i / 100),
                )
                for i, p in enumerate(pages)
            ]

        st = out.stages
        return QueryResponse(
            hits=[],
            stages=RetrievalStages(
                keyword=hits(st["keyword"], None),
                vector=hits(st["vector"], None),
                fused=hits(st["fused"], None),
                reranked=hits(st["reranked"], out.top_rerank_score),
            ),
        )

    # ------------------------------------------------------------- E-G / E-A
    async def chat(
        self,
        message: str,
        collection_ids: Sequence[UUID],
        *,
        conversation_id: UUID | None = None,
        trace_shown: bool = False,
    ) -> AsyncIterator[ChatEvent]:
        item = self._qid[message]
        self._step("chat", item)
        row = self._generator[item]
        out = row.output
        assert isinstance(out, AnswerOutput)
        assert trace_shown
        queue = self.fail.get(("chat-stream", item))
        if queue:
            exc = queue.pop(0)
            assert isinstance(exc, ApiError)
            yield ErrorEvent(problem=exc.problem)
            return
        q = next(q for q in self.questions.questions if q.id == item)
        yield TokenEvent(text=out.answer)
        for n, page in enumerate(out.cited_pages, 1):
            title = q.pages[0] if q.pages else page
            yield CitationEvent(
                citation=Citation(
                    anchor="a",
                    document_id=document_id(page),
                    title=title,
                    heading_path=[],
                    url=None,
                    snippet="",
                    label=n,
                    inferred=self.inferred_citations,
                )
            )
        if self.with_verification:
            yield VerificationEvent(status=out.status)
        shown = [
            ShownSource(
                label=n,
                chunk_id=page_chunk_id(p),
                document_id=document_id(p),
                display_sha256=sha(page_text(p)),
            )
            for n, p in enumerate([*out.cited_pages, *filter(None, [self.pad_shown])], 1)
        ]
        yield DoneEvent(
            conversation_id=uuid5(NS, "conv"),
            usage=[
                Usage(
                    provider="fake",
                    model=self.chat_model,
                    prompt_tokens=row.tokens.prompt,
                    completion_tokens=row.tokens.completion,
                ),
                *(
                    Usage(provider="fake", model=m, prompt_tokens=0, completion_tokens=0)
                    for m in filter(None, [self.verifier_usage])
                    if not out.abstained
                ),
            ],
            prompt_version="answer@11",
            prompt_versions=self.prompt_versions,
            shown=shown if self.with_shown else None,
        )

    # ------------------------------------------------------------- E-V
    async def eval_verify(self, request: EvalVerifyRequest) -> EvalVerifyResponse:
        item = self._vid[request.answer]
        self._step("verify", item)
        out = self._verify[item].output
        assert isinstance(out, VerifyOutput)
        answer = request.answer
        reports = out.verifier_ran or not self.verifier_only_when_ran
        claims = [
            EvalVerifierClaim(text=answer[s.start : s.end], start=s.start, end=s.end)
            for s in out.verifier_spans
        ] + [EvalVerifierClaim(text="?") for _ in range(out.unlocated_verifier_claims)]
        if out.verifier_objected:
            verdict: str | None = "partial"
        else:
            verdict = "verified" if out.verifier_ran else None
        return EvalVerifyResponse(
            status=out.status,
            claims=[],
            coverage_units=[
                EvalCoverageUnit(
                    text=answer[s.start : s.end], start=s.start, end=s.end, kind="uncited"
                )
                for s in out.coverage_spans
            ],
            verifier_claims=claims,
            verifier_ran=out.verifier_ran,
            verifier_verdict=verdict,
            verifier_model=self.verifier_model if reports else None,
            verifier_prompt=self.verifier_prompt if reports else None,
        )

    # ------------------------------------------------------------- E-I
    async def interview_grade(self, request: InterviewGradeRequest) -> InterviewGrade:
        seed = next(s for s in self.interview.seeds if s.question == request.question)
        tier = next(t for t, a in seed.answers.items() if a == request.answer)
        assert request.level is not None
        assert request.question_token is None
        level = request.level.value
        rep = self._grades[(seed.id, tier.value, level)]
        self._grades[(seed.id, tier.value, level)] += 1
        item = f"{seed.id}/{tier.value}/{level}/{rep}"
        self._step("grade", item)
        row = self._interview.get(item) or self._interview[f"{seed.id}/{tier.value}/{level}/0"]
        out = row.output
        assert isinstance(out, GradeOutput)
        return InterviewGrade(
            score=out.score,
            summary="fake",
            right=[],
            missed=[],
            misconceptions=out.misconceptions_kept,
            model_answer="fake",
            citations=[],
            verification="verified",
            level=level,
        )

    async def interview_question(self, request: InterviewQuestionRequest) -> InterviewQuestion:
        self._step("question", ",".join(str(c) for c in request.chunk_ids))
        level = request.level.value if request.level else "senior_sde_1"
        return InterviewQuestion(
            question=f"Explain `{request.kinds[0].value}` trade-offs from these notes.",
            kind=request.kinds[0],
            topic="t",
            difficulty="standard",
            chunk_ids=[*request.chunk_ids, *filter(None, [self.extra_question_chunk])],
            expected_points=["a point"],
            citations=[],
            level=level,
        )


def api_error(status: int) -> ApiError:
    return ApiError(Problem(title="boom", status=status))


@dataclass
class FakeJudge:
    """Scores deterministically; appends to the shared call log to prove the ordering."""

    log: list[str]
    key_: JudgeKey = GOLDEN_JUDGE
    items: list[JudgeItem] = field(default_factory=list)
    raise_with: Exception | None = None
    fail_after: int = 0  # raise `raise_with` once this many items were scored

    @property
    def key(self) -> JudgeKey:
        return self.key_

    async def score(self, items: Sequence[JudgeItem]) -> list[JudgeResult]:
        if self.raise_with is not None and len(self.items) >= self.fail_after:
            raise self.raise_with
        out: list[JudgeResult] = []
        for item in items:
            self.log.append(f"judge:{item.item_digest[:8]}")
            self.items.append(item)
            for c in item.criteria:
                out.append(
                    JudgeResult(
                        cache_key=digest([self.key_.model_dump(mode="json"), item.item_digest, c]),
                        item_digest=item.item_digest,
                        criterion=c,
                        score=len(item.answer) % 3,
                        reason="fake",
                        judge=self.key_,
                        scored_at=datetime(2026, 10, 21, tzinfo=UTC),
                    )
                )
        return out


@dataclass
class FakeSummarizer:
    calls: list[tuple[RunFile, Datasets, list[Label], list[JudgeResult]]] = field(
        default_factory=list
    )

    def __call__(
        self,
        run: RunFile,
        datasets: Datasets,
        labels: Sequence[Label],
        judge_results: Sequence[JudgeResult],
        /,
    ) -> RunSummary:
        self.calls.append((run, datasets, list(labels), list(judge_results)))
        return RunSummary(
            run_id=run.header.run_id,
            eval_kind=run.header.eval_kind,
            header_digest=run.header_digest,
            metrics=[
                MetricValue(name="rows", value=len(run.rows), n=len(run.rows), method=Method.point)
            ],
        )


@dataclass
class FakeLabels:
    labels: list[Label] = field(default_factory=list)

    def latest(self) -> list[Label]:
        newest: dict[tuple[str, str], Label] = {}
        for label in self.labels:
            newest[(label.item_digest, label.kind.value)] = label
        return list(newest.values())

    def append(self, label: Label) -> None:
        self.labels.append(label)
