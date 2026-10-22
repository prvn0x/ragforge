"""The ADR-0017 §3 runners: E-R, E-G, E-V, E-I, E-A and the E-A latency pass.

Each runner turns one dataset into one run file v2 through `RunWriter`, against the SDK only (no
server core, ADR-0005). They share one driver, so every run gets the same guarantees:

- **Stable identity (principle 6).** Page and chunk keys are resolved through `/v1/eval/resolve`
  once, at the start. A key that no longer resolves marks its item `stale`; it is never re-mapped.
- **A header that names the configuration.** Snapshot digests come from `eval_snapshot_status`;
  prompt versions and the verifier model come from what the SERVER reports it ran (the `done`
  event, the verify response), so a silently changed prompt can never hide behind a config file.
  The header is therefore written after the first item answers (a "probe"), and every later item
  is checked against it: a mid-run change is an error of ours, which blocks the gate.
- **Errors are classified.** 429 / 5xx are provider-transient: retried up to `transient_retries`
  times, then recorded as such. Anything else is ours and counts.
- **Resumable.** A crash keeps every finished row; `resume` skips them and refuses a run whose
  configuration differs (`RunWriter`).
- **Paced for the free tier.** One SDK request per `60 / rpm` seconds, sequentially. `rpm`
  counts OUR requests, not the provider calls each one makes on the server (see `Pacer`).
- **The judge never runs during answering or the latency pass (ADR-0017 §4).** E-G answers go to a
  staging file first; only when answering has finished does the judge score them, and only then
  is the final run file written, rows plus judge refs. Answers survive a judge failure.

Run files never hold note text (§7): notes are fetched by id for the judge, digest-checked, and
discarded.
"""

from __future__ import annotations

import asyncio
import hashlib
import itertools
import math
import os
import re
import tempfile
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable, Mapping, Sequence
from contextlib import ExitStack
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import Protocol
from uuid import UUID

import httpx
from pydantic import ValidationError

from ragforge_client.ask import clean
from ragforge_client.client import ApiError
from ragforge_client.evalrun import Fact as MatchFact
from ragforge_client.evalrun import abstained, fact_matched
from ragforge_client.evals import metrics as metrics_mod
from ragforge_client.evals.datasets import append_jsonl, complete_lines, content_lemmas
from ragforge_client.evals.interfaces import (
    CALIBRATION_FILE,
    SUMMARY_SUFFIX,
    Datasets,
    Judge,
    LabelReader,
)
from ragforge_client.evals.judge import make_item
from ragforge_client.evals.runfile import RunFile, RunWriter, read_run
from ragforge_client.evals.schema import (
    E_G_CRITERIA,
    E_I_CRITERIA,
    AnswerOutput,
    AnswerTier,
    Calibration,
    Criterion,
    DatasetRef,
    ErrorClass,
    EvalKind,
    GradeOutput,
    InterviewSeed,
    InterviewSet,
    JudgeKey,
    JudgeResult,
    Label,
    Question,
    QuestionOutput,
    QuestionSet,
    RetrievalConfig,
    RetrievalOutput,
    RunError,
    RunHeader,
    RunRow,
    RunSummary,
    ShownRef,
    SnapshotRef,
    Span,
    Split,
    Surface,
    Timings,
    Tokens,
    VerifierItem,
    VerifierSet,
    VerifyOutput,
    digest,
    page_key_str,
    parse_page_key,
)
from ragforge_contracts.api import (
    ChunkKey,
    EvalChunksResponse,
    EvalResolveRequest,
    EvalResolveResponse,
    EvalSnapshot,
    EvalVerifyRequest,
    EvalVerifyResponse,
    InterviewGrade,
    InterviewGradeRequest,
    InterviewKind,
    InterviewLevel,
    InterviewQuestion,
    InterviewQuestionRequest,
    PageKey,
    Problem,
    QueryResponse,
    RetrievalStages,
    StageHit,
)
from ragforge_contracts.sse import (
    AnswerReplaceEvent,
    ChatEvent,
    CitationEvent,
    DoneEvent,
    ErrorEvent,
    TokenEvent,
    VerificationEvent,
)

LEVELS: tuple[str, ...] = tuple(level.value for level in InterviewLevel)  # sde_1 .. staff
SWEEP_TIERS = (AnswerTier.gold, AnswerTier.partial)  # the level sweep (ADR-0017 §3 E-I)
REPLICATES = (0, 1)
COLD_QUESTIONS = 2  # latency pass: the first two are cold and reported apart
RESOLVE_BATCH = 500  # EvalResolveRequest max_length
CHUNKS_BATCH = (
    100  # ids per GET /v1/eval/chunks (the server takes 200; the query string stays short)
)
_MESSAGE_LIMIT = 300
_TRANSIENT_STATUSES = frozenset({429}) | frozenset(range(500, 600))


# ------------------------------------------------------------------ the SDK surface we use
class EvalApi(Protocol):
    """The part of `RagforgeClient` the runners call (tests pass a fake with these methods)."""

    async def query(
        self, query: str, collection_ids: Sequence[UUID], *, k: int = 8, stages: bool = False
    ) -> QueryResponse: ...

    def chat(
        self,
        message: str,
        collection_ids: Sequence[UUID],
        *,
        conversation_id: UUID | None = None,
        trace_shown: bool = False,
    ) -> AsyncIterator[ChatEvent]: ...

    async def interview_question(self, request: InterviewQuestionRequest) -> InterviewQuestion: ...

    async def interview_grade(self, request: InterviewGradeRequest) -> InterviewGrade: ...

    async def eval_verify(self, request: EvalVerifyRequest) -> EvalVerifyResponse: ...

    async def eval_chunks(self, chunk_ids: Sequence[UUID]) -> EvalChunksResponse: ...

    async def eval_resolve(self, request: EvalResolveRequest) -> EvalResolveResponse: ...

    async def eval_snapshot_status(self, collection_id: UUID) -> EvalSnapshot: ...


class Summarizer(Protocol):
    """`metrics.summarize` (interfaces.py); injected so the runners do not depend on B2's code."""

    def __call__(
        self,
        run: RunFile,
        datasets: Datasets,
        labels: Sequence[Label],
        judge_results: Sequence[JudgeResult],
        /,
    ) -> RunSummary: ...


# ------------------------------------------------------------------ configuration
class RunAborted(RuntimeError):
    """The run cannot start or continue (no snapshot digest, a sealed split, a failed judge
    fetch). Every finished row is already on disk; the message never holds note text or keys."""


@dataclass
class Pacer:
    """At most `rpm` SDK requests per minute, as an even spacing (the free tier counts requests
    per minute; an even gap never bursts past it). `rpm=None`: no pacing.

    `rpm` counts the CLIENT's requests, not the provider calls behind them: one chat turn costs
    the server at least two (the answer, then the ground check), a verify one, a grade or a
    generated question one or two. Against a provider limit of P requests a minute, set `rpm`
    to about P / (provider calls per item), e.g. `--rpm 5` for a 10-RPM tier on E-G / E-A. A
    429 that slips through is still retried and classified transient, it only wastes budget."""

    rpm: float | None = None
    clock: Callable[[], float] = time.monotonic
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep
    _last: float | None = field(default=None, init=False)

    async def wait(self) -> None:
        if self.rpm is None or self.rpm <= 0:
            return
        gap = 60.0 / self.rpm
        now = self.clock()
        if self._last is not None and now - self._last < gap:
            await self.sleep(gap - (now - self._last))
            now = self.clock()
        self._last = now


@dataclass(frozen=True)
class RetryPolicy:
    """`transient_retries` is recorded in the header (the provider-error policy, §7)."""

    transient_retries: int = 2
    backoff_s: float = 2.0  # doubled per retry
    item_timeout_s: float | None = 300.0  # a stalled stream costs one row, not the run


def _no_log(_: str) -> None:
    return None


def _utcnow() -> datetime:
    return datetime.now(UTC)


@dataclass(frozen=True)
class RunContext:
    """Everything a runner needs besides its dataset. `snapshot_collection_id=None` is the D0
    demo tenant (no snapshot, ADR-0017 §2); every private run names its snapshot."""

    client: EvalApi
    collection_ids: tuple[UUID, ...]
    arm: str
    runs_dir: Path
    dataset_dir: Path  # labels.jsonl and judge-cache.jsonl live here (interfaces.py)
    snapshot_collection_id: UUID | None = None
    retrieval: RetrievalConfig | None = None
    chat_model: str | None = None
    verifier_model: str | None = None
    prompt_versions: Mapping[str, str] = field(default_factory=dict)
    splits: tuple[Split, ...] = (Split.dev,)
    policy: RetryPolicy = field(default_factory=RetryPolicy)
    pacer: Pacer = field(default_factory=Pacer)
    # the judge has its own budget (Gemini Flash's free tier is ~20 requests a day, ADR §4) and
    # often another provider than the stack, so it is paced apart from the SDK calls
    judge_pacer: Pacer = field(default_factory=Pacer)
    summarize: Summarizer | None = None
    labels: LabelReader | None = None
    resume: Path | None = None  # an unfinished run file of the same kind to continue
    log: Callable[[str], None] = _no_log
    now: Callable[[], datetime] = _utcnow
    query_k: int = 10


# ------------------------------------------------------------------ per-item results
@dataclass(frozen=True)
class Observed:
    """What the server says it ran for one item. The run keeps one reference (the probe's, with
    gaps filled by later items) and every item must agree with it, so a prompt or model deployed
    mid-run cannot mix two arms in one run.

    A field an item does not report (None, or a surface missing from `prompt_versions`) is "not
    reported", never a conflict: an E-V item whose verifier did not run reports no verifier, and
    a chat turn reports no verifier model. `complete` says whether the item reported everything
    its kind can report; the header waits for a complete item (see `drive`)."""

    prompt_versions: Mapping[str, str] = field(default_factory=dict)
    verifier_model: str | None = None
    chat_model: str | None = None
    complete: bool = True

    @classmethod
    def of_header(cls, header: RunHeader) -> Observed:
        """The reference a resumed run continues against: the header fields the SERVER reports
        for this kind (`REPORTED`). The other fields came from `--arm-config` and must be
        rebuilt from the config, so a changed config is refused on resume, not inherited."""
        fields = REPORTED.get(header.eval_kind, frozenset())
        return cls(
            dict(header.prompt_versions) if "prompt_versions" in fields else {},
            header.verifier_model if "verifier_model" in fields else None,
            header.chat_model if "chat_model" in fields else None,
        )

    def conflicts(self, reference: Observed) -> list[str]:
        out = [
            f"prompt_versions[{surface}]"
            for surface, version in sorted(self.prompt_versions.items())
            if reference.prompt_versions.get(surface, version) != version
        ]
        for name in ("verifier_model", "chat_model"):
            mine, theirs = getattr(self, name), getattr(reference, name)
            if mine is not None and theirs is not None and mine != theirs:
                out.append(name)
        return out

    def merged(self, later: Observed) -> Observed:
        """This reference with the gaps `later` fills (this one's own values always win)."""
        return Observed(
            prompt_versions={**later.prompt_versions, **self.prompt_versions},
            verifier_model=self.verifier_model or later.verifier_model,
            chat_model=self.chat_model or later.chat_model,
            complete=self.complete or later.complete,
        )


_CHAT_REPORTS = frozenset({"prompt_versions", "chat_model"})  # the `done` event
REPORTED: dict[EvalKind, frozenset[str]] = {  # header fields the server reports, per kind
    EvalKind.generator: _CHAT_REPORTS,
    EvalKind.application: _CHAT_REPORTS,
    EvalKind.latency: _CHAT_REPORTS,
    EvalKind.verifier: frozenset({"prompt_versions", "verifier_model"}),  # the verify response
}


Output = RetrievalOutput | AnswerOutput | VerifyOutput | GradeOutput | QuestionOutput


@dataclass(frozen=True)
class Produced:
    output: Output
    shown: list[ShownRef] = field(default_factory=list)
    timings: Timings = field(default_factory=Timings)
    tokens: Tokens = field(default_factory=Tokens)
    grades: dict[str, float | int | bool | None] = field(default_factory=dict)
    observed: Observed = field(default_factory=Observed)


@dataclass(frozen=True)
class Work:
    """One item of a run. `stale` items are recorded without a call (principle 6)."""

    item_id: str
    input_digest: str
    produce: Callable[[], Awaitable[Produced]]
    stale: bool = False
    shown: tuple[ShownRef, ...] = ()


class ItemFailure(Exception):
    """A per-item failure of ours that is not an HTTP error (e.g. no `shown` on a done event)."""


class InconclusiveVerifier(ItemFailure):
    """The server's verifier ran but could not decide (`verifier_conclusive=False`), which the
    server reports with HTTP 200 when ITS provider timed out or returned 429/5xx.

    Decision (ADR-0017 §3 E-V): this is classified provider-transient, retried, then recorded
    as a transient error, never as an objection. Counting it as an objection would turn a flaky
    free tier into "verifier-caused false-reject" (the SLO's upper bound) and could mark a good
    verifier model untrusted. The cost is bounded: transient rows are excluded from E-V metrics
    and from gate pairing, and a model that is ALWAYS inconclusive leaves nothing paired, which
    the gate blocks on."""


class StreamProblem(Exception):
    """An `error` event inside a chat stream: classified by its status like an HTTP error."""

    def __init__(self, problem: Problem) -> None:
        super().__init__(" ".join(filter(None, (str(problem.status), problem.title))))
        self.problem = problem


# ------------------------------------------------------------------ errors and retries
def classify(exc: BaseException) -> RunError:
    """429 / 5xx from the server or its provider: transient (ADR-0017 §3 E-A). Everything else,
    including transport failures to OUR server and timeouts, is ours and counts.

    Why a timeout is ours, even when a free-tier provider stalled: the server bounds every
    provider call itself and turns a stalled provider into a 5xx / `error` event, which IS
    transient here. A client-side `item_timeout_s` expiring means our server did not do that
    (a hung stream, a missing bound), which is exactly what the gate's "no errors of our own"
    floor exists to catch."""
    status: int | None = None
    if isinstance(exc, ApiError):
        status = exc.status
    elif isinstance(exc, StreamProblem):
        status = exc.problem.status
    transient = status in _TRANSIENT_STATUSES or isinstance(exc, InconclusiveVerifier)
    cls = ErrorClass.provider_transient if transient else ErrorClass.ours
    if isinstance(exc, TimeoutError):
        message = "timeout"
    elif isinstance(exc, httpx.HTTPError):
        # the type names the failure; httpx messages hold the URL at most, never the headers
        message = f"transport {type(exc).__name__}: {exc}"
    elif isinstance(exc, ValidationError):
        message = f"malformed response: {exc.error_count()} validation error(s)"
    else:
        message = str(exc)
    return RunError(cls=cls, status=status, message=clean(message)[:_MESSAGE_LIMIT])


_ITEM_ERRORS = (
    ApiError,
    StreamProblem,
    ItemFailure,
    httpx.HTTPError,
    TimeoutError,
    ValidationError,
)


async def with_retries[T](
    call: Callable[[], Awaitable[T]], policy: RetryPolicy, pacer: Pacer
) -> tuple[T | None, RunError | None, int]:
    """(result, None, attempts) or (None, error, attempts). Only transient errors are retried;
    an error of ours is final at once (re-asking would hide a real failure)."""
    attempt = 0
    while True:
        attempt += 1
        await pacer.wait()
        try:
            async with asyncio.timeout(policy.item_timeout_s):
                return await call(), None, attempt
        except _ITEM_ERRORS as exc:
            error = classify(exc)
        if error.cls is ErrorClass.ours or attempt > policy.transient_retries:
            return None, error, attempt
        await pacer.sleep(policy.backoff_s * 2 ** (attempt - 1))


# ------------------------------------------------------------------ identity (principle 6)
def key_id(key: PageKey | ChunkKey) -> str:
    """A hashable identity for a key (ChunkKey holds a list, so the model is not hashable)."""
    return digest(key.model_dump(mode="json"))


@dataclass(frozen=True)
class Resolution:
    pages: Mapping[str, UUID]  # key_id -> document_id, only keys that resolved ok
    chunks: Mapping[str, UUID]  # key_id -> chunk_id, only keys that resolved ok

    def chunk_ids(self, keys: Iterable[ChunkKey]) -> list[UUID] | None:
        """Every key's id in order, or None when any one is stale or missing."""
        ids = [self.chunks.get(key_id(k)) for k in keys]
        return None if any(i is None for i in ids) else [i for i in ids if i is not None]

    def pages_ok(self, keys: Iterable[PageKey]) -> bool:
        return all(key_id(k) in self.pages for k in keys)


async def resolve_keys(
    ctx: RunContext, pages: Iterable[PageKey], chunks: Iterable[ChunkKey]
) -> Resolution:
    """Resolve once, at the start, in batches the contract accepts. Only `ok` counts: a stale
    key is reported through the item's `stale` flag, never silently re-mapped to a candidate."""
    page_list = list({key_id(p): p for p in pages}.values())
    chunk_list = list({key_id(c): c for c in chunks}.values())
    page_ids: dict[str, UUID] = {}
    chunk_ids: dict[str, UUID] = {}
    for start in range(0, max(len(page_list), len(chunk_list)), RESOLVE_BATCH):
        request = EvalResolveRequest(
            collection_ids=list(ctx.collection_ids),
            page_keys=page_list[start : start + RESOLVE_BATCH],
            chunk_keys=chunk_list[start : start + RESOLVE_BATCH],
        )
        response = await _setup_call(ctx, partial(ctx.client.eval_resolve, request))
        for page in response.pages:
            if page.status == "ok" and page.document_id is not None:
                page_ids[key_id(page.key)] = page.document_id
        for chunk in response.chunks:
            if chunk.status == "ok" and chunk.chunk_id is not None:
                chunk_ids[key_id(chunk.key)] = chunk.chunk_id
    return Resolution(page_ids, chunk_ids)


async def _setup_call[T](ctx: RunContext, call: Callable[[], Awaitable[T]]) -> T:
    """A call the run cannot proceed without (resolve, snapshot status, note text for the
    judge): transient errors are retried, then the run stops cleanly (it is resumable)."""
    result, error, _ = await with_retries(call, ctx.policy, Pacer())
    if error is not None or result is None:
        assert error is not None
        raise RunAborted(f"{error.cls.value} error: {error.message}")
    return result


async def snapshot_ref(ctx: RunContext) -> SnapshotRef | None:
    """The snapshot digests for the header; a snapshot whose copy job has not finished has no
    digest yet, and a run on it could not be compared, so it refuses to start."""
    if ctx.snapshot_collection_id is None:
        return None
    collection_id = ctx.snapshot_collection_id
    snap = await _setup_call(ctx, partial(ctx.client.eval_snapshot_status, collection_id))
    if snap.content_digest is None or snap.index_digest is None:
        raise RunAborted("the eval snapshot has no digest yet: its copy job has not finished")
    return SnapshotRef(
        collection_id=str(snap.collection_id),
        content_digest=snap.content_digest,
        index_digest=snap.index_digest,
    )


class PageLookup:
    """chunk id -> page key, through `/v1/eval/chunks` (cached for the run). Stage hits and
    citations carry database ids; metrics need stable page keys (principle 6).

    Every chunk the server returned in a stage or as shown must map: the eval key sees exactly
    the eval tenant (RLS), so an absent chunk is a server bug. It fails the item as an error of
    ours instead of silently becoming an irrelevant page that lowers hit@k, MRR and nDCG."""

    def __init__(self, ctx: RunContext) -> None:
        self._ctx = ctx
        self._pages: dict[UUID, str] = {}
        self._documents: dict[UUID, str] = {}
        self._keys: dict[UUID, ChunkKey] = {}

    async def load(self, chunk_ids: Iterable[UUID]) -> None:
        """Fetch every unknown id (in `CHUNKS_BATCH`es), then require all of them."""
        wanted = set(chunk_ids)
        missing = sorted(wanted - self._pages.keys(), key=str)
        for start in range(0, len(missing), CHUNKS_BATCH):
            response = await self._ctx.client.eval_chunks(missing[start : start + CHUNKS_BATCH])
            for chunk in response.chunks:
                page = page_key_str(chunk.page_key)
                self._pages[chunk.chunk_id] = page
                self._documents[chunk.document_id] = page
                if chunk.chunk_key is not None:
                    self._keys[chunk.chunk_id] = chunk.chunk_key
        absent = len(wanted - self._pages.keys())
        if absent:
            raise ItemFailure(
                f"{absent} chunk(s) the server returned are absent from /v1/eval/chunks"
            )

    def chunk_key(self, chunk_id: UUID) -> ChunkKey | None:
        """The stable key of a loaded chunk (None from a server that predates the field)."""
        return self._keys.get(chunk_id)

    def page(self, chunk_id: UUID) -> str:
        page = self._pages.get(chunk_id)
        if page is None:  # `load` was not called for it: a bug of ours, never a silent miss
            raise ItemFailure("a chunk was ranked before its page was looked up")
        return page

    def document(self, document_id: UUID) -> str | None:
        return self._documents.get(document_id)


def pages_in_rank_order(hits: Sequence[StageHit], lookup: Callable[[UUID], str]) -> list[str]:
    """Chunks de-duplicated to pages (ADR-0017 §3 E-R: page level), each page at its BEST rank.
    Ties are broken by chunk id (the server's rule), so the order is reproducible."""
    pages: list[str] = []
    seen: set[str] = set()
    for hit in sorted(hits, key=lambda h: (h.rank, str(h.chunk_id))):
        page = lookup(hit.chunk_id)
        if page not in seen:
            seen.add(page)
            pages.append(page)
    return pages


# ------------------------------------------------------------------ the driver
@dataclass(frozen=True)
class Plan:
    kind: EvalKind
    surface: Surface
    dataset: DatasetRef
    work: list[Work]
    judge: JudgeKey | None = None


def dataset_ref(
    name: str, dataset: QuestionSet | VerifierSet | InterviewSet, splits: Sequence[Split]
) -> DatasetRef:
    """The digest covers the WHOLE parsed dataset (the fixtures' rule), so any edit shows."""
    return DatasetRef(
        name=name, digest=digest(dataset.model_dump(mode="json")), splits=list(splits)
    )


def new_run_path(ctx: RunContext, kind: EvalKind) -> tuple[str, Path]:
    run_id = f"{kind.value}-{ctx.arm}-{ctx.now():%Y%m%dT%H%M%SZ}"
    return run_id, ctx.runs_dir / f"{run_id}.jsonl"


def build_header(
    ctx: RunContext,
    plan: Plan,
    run_id: str,
    started_at: datetime,
    snapshot: SnapshotRef | None,
    observed: Observed | None,
) -> RunHeader:
    """What the server reported wins over `--arm-config` (a config file can be stale; the server
    says what it ran); the config fills only what the server does not report."""
    prompts = dict(ctx.prompt_versions)
    verifier_model = ctx.verifier_model
    chat_model = ctx.chat_model
    if observed is not None:
        prompts |= dict(observed.prompt_versions)
        verifier_model = observed.verifier_model or verifier_model
        chat_model = observed.chat_model or chat_model
        _warn_config_drift(ctx, observed)
    if plan.kind is EvalKind.interview and Surface.interview.value not in prompts:
        ctx.log(
            "warning: no interview prompt version in the header (the interview responses do not "
            "report one yet): an interview prompt change will not show; set it in --arm-config"
        )
    return RunHeader(
        run_id=run_id,
        eval_kind=plan.kind,
        surface=plan.surface,
        arm=ctx.arm,
        started_at=started_at,
        dataset=plan.dataset,
        snapshot=snapshot,
        retrieval=ctx.retrieval,
        chat_model=chat_model,
        verifier_model=verifier_model,
        prompt_versions=prompts,
        judge=plan.judge,
        transient_retries=ctx.policy.transient_retries,
    )


def _warn_config_drift(ctx: RunContext, observed: Observed) -> None:
    """The config named one model and the server ran another: the header records the server's,
    and the owner should know their config file is stale."""
    pairs = (
        ("chat_model", ctx.chat_model, observed.chat_model),
        ("verifier_model", ctx.verifier_model, observed.verifier_model),
    )
    for name, configured, served in pairs:
        if configured is not None and served is not None and configured != served:
            ctx.log(f"warning: --arm-config {name} is {configured}, the server ran {served}")


async def execute(
    work: Work, ctx: RunContext, prior_attempts: int = 0
) -> tuple[RunRow, Observed | None]:
    """One item. `prior_attempts` (the item's highest attempt in the run being resumed) is
    added to this session's attempts, so `attempt` only grows across sessions: metrics and the
    gate keep the HIGHEST attempt per item, and a retry restarting at 1 after a 3-attempt
    transient failure would otherwise lose to the failure it recovered."""
    if work.stale:
        return RunRow(
            item_id=work.item_id,
            input_digest=work.input_digest,
            shown=list(work.shown),
            stale=True,
            attempt=prior_attempts + 1,
        ), None
    produced, error, attempts = await with_retries(work.produce, ctx.policy, ctx.pacer)
    if produced is None:
        return RunRow(
            item_id=work.item_id,
            input_digest=work.input_digest,
            shown=list(work.shown),
            error=error,
            attempt=prior_attempts + attempts,
        ), None
    return RunRow(
        item_id=work.item_id,
        input_digest=work.input_digest,
        shown=produced.shown or list(work.shown),
        output=produced.output,
        grades=produced.grades,
        timings=produced.timings,
        tokens=produced.tokens,
        attempt=prior_attempts + attempts,
    ), produced.observed


def _consistent(row: RunRow, observed: Observed | None, reference: Observed) -> RunRow:
    conflicts = observed.conflicts(reference) if observed is not None else []
    if not conflicts:
        return row
    message = "configuration changed mid-run: " + ", ".join(conflicts)
    error = RunError(cls=ErrorClass.ours, message=message)
    return row.model_copy(update={"output": None, "error": error})


PROBE_LIMIT = 5  # answered rows buffered at most while waiting for a complete observation


@dataclass
class _Probe:
    """The rows answered before the header exists, and the reference they establish.

    The header is opened on the first COMPLETE observation (for E-V: an item whose verifier
    ran and reported its model and prompt), because an incomplete one (a coverage-caught item)
    would leave the header without a verifier, and every later item would look like a mid-run
    change. Earlier rows only fill gaps. If `PROBE_LIMIT` answers in a row are incomplete (an
    older server that never reports), the header opens with what was reported, so a crash never
    costs more than a few rows."""

    rows: list[tuple[RunRow, Observed | None]] = field(default_factory=list)

    def ready(self, seen: Observed | None) -> bool:
        answered = sum(s is not None for _, s in self.rows)
        return seen is not None and (seen.complete or answered >= PROBE_LIMIT)

    def reference(self) -> Observed | None:
        """The complete observation (else the first one), gaps filled by the others."""
        seen = [s for _, s in self.rows if s is not None]
        if not seen:
            return None
        first = next((s for s in seen if s.complete), seen[0])
        reference = first
        for other in seen:
            reference = reference.merged(other)
        return reference


async def drive(ctx: RunContext, plan: Plan, path: Path, run_id: str, *, resume: bool) -> Path:
    """Run every pending item into `path`. The header records what the server REPORTED it ran,
    so it is opened once a probe item answers (`_Probe`); rows before it are buffered, then
    checked against the reference like every later row."""
    existing = unfinished_run(path) if resume else None
    done = {r.item_id for r in existing.rows if r.error is None} if existing else set()
    prior = prior_attempts(existing) if existing else {}
    started_at = existing.header.started_at if existing else ctx.now()
    snapshot = await snapshot_ref(ctx)
    pending = [w for w in plan.work if w.item_id not in done]
    with ExitStack() as stack:
        writer: RunWriter | None = None
        reference: Observed | None = None
        probe = _Probe()

        def open_writer(seen: Observed | None) -> RunWriter:
            head = build_header(ctx, plan, run_id, started_at, snapshot, seen)
            return stack.enter_context(RunWriter(path, head, resume=existing is not None))

        if existing is not None:  # a changed configuration is refused before any call
            reference = Observed.of_header(existing.header)
            writer = open_writer(reference)
        for index, work in enumerate(pending, 1):
            row, seen = await execute(work, ctx, prior.get(work.item_id, 0))
            ctx.log(_progress(index, len(pending), row))
            if writer is not None and reference is not None:
                writer.row(_consistent(row, seen, reference))
                reference = reference.merged(seen) if seen is not None else reference
                continue
            probe.rows.append((row, seen))
            if probe.ready(seen):
                reference = probe.reference()
                writer = _flush_probe(ctx, probe, open_writer(reference), reference)
        if writer is None:  # never ready: nothing answered, or too few answers to wait for
            reference = probe.reference()
            writer = _flush_probe(ctx, probe, open_writer(reference), reference)
        writer.finish()
    return path


def prior_attempts(run: RunFile) -> dict[str, int]:
    """The highest attempt per item already in a run (what a resumed retry continues from)."""
    highest: dict[str, int] = {}
    for row in run.rows:
        highest[row.item_id] = max(highest.get(row.item_id, 0), row.attempt)
    return highest


def _flush_probe(
    ctx: RunContext, probe: _Probe, writer: RunWriter, reference: Observed | None
) -> RunWriter:
    if reference is not None and not reference.complete:
        ctx.log(
            "warning: the server did not report its full configuration; the header "
            "records what it did report (and --arm-config for the rest)"
        )
    for row, seen in probe.rows:
        writer.row(_consistent(row, seen, reference) if reference is not None else row)
    probe.rows.clear()
    return writer


def unfinished_run(path: Path) -> RunFile | None:
    """The run to continue at `path` (None: nothing there yet); a finished one is refused."""
    if not path.exists():
        return None
    run = read_run(path)
    if run.footer is not None:
        raise RunAborted(f"{path.name} already finished: nothing to resume")
    return run


def _progress(index: int, total: int, row: RunRow) -> str:
    if row.stale:
        state = "stale"
    elif row.error is not None:
        state = f"error {row.error.cls.value}"
    else:
        state = "ok"
    return f"[{index}/{total}] {row.item_id}: {state}"


def resume_target(ctx: RunContext, kind: EvalKind) -> tuple[str, Path, bool]:
    if ctx.resume is not None:
        return read_run(ctx.resume).header.run_id, ctx.resume, True
    run_id, path = new_run_path(ctx, kind)
    return run_id, path, False


# ------------------------------------------------------------------ summaries
def summary_path(run_path: Path) -> Path:
    name = run_path.name.removesuffix(".jsonl")
    return run_path.with_name(name + SUMMARY_SUFFIX)


def write_private(path: Path, text: str) -> None:
    """Atomic 0600 write (temp file, fsync, rename): a reader never sees half a summary."""
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def load_judge_results(path: Path) -> list[JudgeResult]:
    """`<dataset dir>/judge-cache.jsonl` (interfaces.py), newest line per cache key wins.

    Read with the judge cache's own rules (`datasets.complete_lines`): another process (the
    Eval page, a second run) may be appending, so an unterminated last line is a write in
    progress and is left out, while a complete line that is not a judge result is refused."""
    if not path.exists():
        return []
    lines, _ = complete_lines(path.read_bytes())
    by_key: dict[str, JudgeResult] = {}
    for number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            result = JudgeResult.model_validate_json(line)
        except ValidationError:
            raise RunAborted(f"{path.name}:{number}: not a v2 judge result") from None
        by_key[result.cache_key] = result
    return list(by_key.values())


JUDGE_CACHE = "judge-cache.jsonl"  # in the dataset dir (interfaces.py)


def summarize_run(
    run_path: Path,
    datasets: Datasets,
    summarize: Summarizer,
    *,
    dataset_dir: Path,
    labels: LabelReader | None,
    extra_results: Sequence[JudgeResult] = (),
) -> Path:
    """Write `<run>.summary.json` next to the run (interfaces.py), from the run, the owner's
    latest labels and every judge result known for it (the cache plus this session's)."""
    results = {r.cache_key: r for r in load_judge_results(dataset_dir / JUDGE_CACHE)}
    results |= {r.cache_key: r for r in extra_results}
    latest = labels.latest() if labels is not None else []
    run = read_run(run_path)
    summary = summarize(run, datasets, latest, list(results.values()))
    out = summary_path(run_path)
    write_private(out, summary.model_dump_json(indent=2) + "\n")
    if run.header.eval_kind is EvalKind.generator and run.header.judge is not None:
        write_calibrations(dataset_dir, run.header.judge, list(results.values()), latest)
    return out


def write_calibrations(
    dataset_dir: Path,
    judge: JudgeKey,
    results: Sequence[JudgeResult],
    labels: Sequence[Label],
) -> int:
    """Append a Calibration per E-G criterion to `calibration.jsonl` (interfaces.py), for the
    Eval page's full panel (confusion matrix, leniency). Only a NEW (criterion, judge key, label
    batch) is appended, so the trust state never flaps between runs (ADR-0017 §4)."""
    path = dataset_dir / CALIBRATION_FILE
    seen: set[tuple[str, str, str]] = set()
    if path.exists():
        for line in complete_lines(path.read_bytes())[0]:
            if line.strip():
                cal = Calibration.model_validate_json(line)
                seen.add((cal.criterion.value, cal.judge.model_dump_json(), cal.label_batch_id))
    fresh: list[str] = []
    for criterion in E_G_CRITERIA:
        cal = metrics_mod.calibrate(criterion, judge, results, labels)
        if cal.n == 0:
            continue  # nothing labelled yet: no record (the page says "no labels")
        ident = (criterion.value, judge.model_dump_json(), cal.label_batch_id)
        if ident not in seen:
            fresh.append(cal.model_dump_json())
    if fresh:
        append_jsonl(path, fresh)
    return len(fresh)


def write_summary(
    ctx: RunContext,
    run_path: Path,
    datasets: Datasets,
    extra_results: Sequence[JudgeResult] = (),
) -> Path | None:
    """The summary at the end of a run; skipped when no summarizer is wired (`summarize`
    can always be run later on the finished run file)."""
    if ctx.summarize is None:
        return None
    return summarize_run(
        run_path,
        datasets,
        ctx.summarize,
        dataset_dir=ctx.dataset_dir,
        labels=ctx.labels,
        extra_results=extra_results,
    )


# ------------------------------------------------------------------ dataset selection
def selected_questions(qs: QuestionSet, splits: Sequence[Split]) -> list[Question]:
    """heldout2 is sealed (ADR-0017 D1): its one reveal belongs to the datasets module, never to
    a routine run, so a runner refuses it outright."""
    if Split.heldout2 in splits:
        raise RunAborted("heldout2 is sealed: a runner never asks it (use the reveal flow)")
    return [q for q in qs.questions if q.split in splits]


def question_digest(q: Question) -> str:
    return digest(q.model_dump(mode="json"))


# ------------------------------------------------------------------ E-R
async def _retrieve(ctx: RunContext, q: Question, lookup: PageLookup) -> Produced:
    response = await ctx.client.query(q.question, ctx.collection_ids, k=ctx.query_k, stages=True)
    if response.stages is None:
        raise ItemFailure("no `stages` in the query response (eval scope / RAGFORGE_EVAL_API off?)")
    stages: RetrievalStages = response.stages
    hits = [*stages.keyword, *stages.vector, *stages.fused, *stages.reranked]
    await lookup.load(h.chunk_id for h in hits)
    top = None
    if not q.answerable and stages.reranked:  # abstain correctness by retrieval: top vs tau
        top = min(stages.reranked, key=lambda h: (h.rank, str(h.chunk_id))).score
    output = RetrievalOutput(
        stages={
            "keyword": pages_in_rank_order(stages.keyword, lookup.page),
            "vector": pages_in_rank_order(stages.vector, lookup.page),
            "fused": pages_in_rank_order(stages.fused, lookup.page),
            "reranked": pages_in_rank_order(stages.reranked, lookup.page),
        },
        top_rerank_score=top,
    )
    return Produced(output=output)


async def run_retrieval(ctx: RunContext, qs: QuestionSet) -> Path:
    """E-R: per-stage page rankings for every selected question (deterministic, no model)."""
    questions = selected_questions(qs, ctx.splits)
    resolution = await resolve_keys(
        ctx, (parse_page_key(k) for q in questions for k in q.relevance), ()
    )
    lookup = PageLookup(ctx)
    work = [
        Work(
            item_id=q.id,
            input_digest=question_digest(q),
            produce=partial(_retrieve, ctx, q, lookup),
            stale=not resolution.pages_ok(parse_page_key(k) for k in q.relevance),
        )
        for q in questions
    ]
    plan = Plan(EvalKind.retrieval, Surface.chat, dataset_ref(qs.name, qs, ctx.splits), work)
    run_id, path, resume = resume_target(ctx, EvalKind.retrieval)
    await drive(ctx, plan, path, run_id, resume=resume)
    write_summary(ctx, path, Datasets(questions=qs))
    return path


# ------------------------------------------------------------------ chat turns (E-G, E-A)
@dataclass
class ChatTurn:
    draft: str = ""
    final: str | None = None
    status: str | None = None
    flagged: int = 0
    cited_documents: list[UUID] = field(default_factory=list)
    cited_titles: list[str] = field(default_factory=list)
    model_titles: list[str] = field(default_factory=list)  # cited by the model, not inferred
    done: DoneEvent | None = None
    ttft_s: float | None = None
    total_s: float | None = None

    @property
    def answer(self) -> str:
        return self.final if self.final is not None else self.draft


async def chat_turn(ctx: RunContext, message: str) -> ChatTurn:
    """One chat turn with `?trace=shown`. An `error` event raises, so it is classified exactly
    like an HTTP error (a provider 429 mid-stream is transient too)."""
    turn = ChatTurn()
    started = time.monotonic()
    async for event in ctx.client.chat(message, ctx.collection_ids, trace_shown=True):
        match event:
            case TokenEvent(text=text):
                if turn.ttft_s is None:
                    turn.ttft_s = time.monotonic() - started
                turn.draft += text
            case AnswerReplaceEvent(text=text):
                turn.final = text
            case CitationEvent(citation=citation):
                turn.cited_documents.append(citation.document_id)
                turn.cited_titles.append(citation.title)
                if not citation.inferred:
                    turn.model_titles.append(citation.title)
            case VerificationEvent():
                turn.status = str(event.status)
                turn.flagged = len(event.claims) or len(event.unsupported_claims)
            case DoneEvent():
                turn.done = event
            case ErrorEvent(problem=problem):
                raise StreamProblem(problem)
            case _:
                pass
    turn.total_s = time.monotonic() - started
    return turn


def _unique(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(values))


async def answer_question(ctx: RunContext, q: Question, lookup: PageLookup) -> Produced:
    """E-G / E-A: ask, then reduce the turn to `AnswerOutput`. Citations carry document ids;
    they are mapped to stable page keys through the chunks the model was SHOWN."""
    turn = await chat_turn(ctx, q.question)
    done = turn.done
    if done is None:
        raise ItemFailure("the chat stream ended without a done event")
    if done.shown is None:
        raise ItemFailure("no `shown` on the done event (eval scope / RAGFORGE_EVAL_API off?)")
    await lookup.load(s.chunk_id for s in done.shown)
    shown = [
        ShownRef(
            label=s.label,
            chunk_id=str(s.chunk_id),
            display_sha256=s.display_sha256,
            chunk_key=lookup.chunk_key(s.chunk_id),  # D3 builds from these keys (review P0)
        )
        for s in sorted(done.shown, key=lambda s: s.label)
    ]
    if turn.status is None:  # finalize always emits one: a stream without it is broken
        raise ItemFailure("the chat stream ended without a verification event")
    cited = _unique(lookup.document(d) or f"unresolved:{d}" for d in turn.cited_documents)
    answer = turn.answer
    output = AnswerOutput(
        answer=answer,
        status=turn.status,
        cited_pages=cited,
        abstained=abstained(answer),
        flagged=turn.flagged,
    )
    return Produced(
        output=output,
        shown=shown,
        grades=answer_grades(q, output, turn.cited_titles, turn.model_titles),
        timings=Timings(ttft_s=turn.ttft_s, total_s=turn.total_s),
        tokens=Tokens(
            prompt=sum(u.prompt_tokens for u in done.usage),
            completion=sum(u.completion_tokens for u in done.usage),
        ),
        observed=Observed(
            prompt_versions=dict(done.prompt_versions), chat_model=answering_model(done)
        ),
    )


def answering_model(done: DoneEvent) -> str | None:
    """The model that answered: the turn's FIRST model call. Every model call before the
    answer exists is the chat model's; the verifier (which may be another model and also shows
    in `usage`) only runs on a finished draft, so it is never first."""
    return done.usage[0].model if done.usage else None


def answer_grades(
    q: Question,
    output: AnswerOutput,
    cited_titles: Sequence[str],
    model_titles: Sequence[str] | None = None,
) -> dict[str, float | int | bool | None]:
    """The deterministic correct% of Phase 2b, unchanged (ADR-0017 §3 E-G): an answerable
    question is correct when it is answered, an expected page (`pages`, by title) is cited and at
    least 2/3 of its facts match on word boundaries; an unanswerable one when it abstains.

    v1's `cite_source` is kept as `cited_by_model`: whether the MODEL cited an expected page
    (server-inferred citations excluded), so a model that stops citing does not hide behind the
    server's inference. `model_titles=None`: provenance unknown, recorded as None."""
    if not q.answerable:
        return {"correct": output.abstained}
    facts = [MatchFact(f.fact, tuple(m.lower() for m in f.match)) for f in q.facts]
    needed = math.ceil(len(facts) * 2 / 3)
    hit = sum(fact_matched(f, output.answer) for f in facts)
    cited = any(t in q.pages for t in cited_titles)
    by_model = None if model_titles is None else any(t in q.pages for t in model_titles)
    return {
        "correct": not output.abstained and cited and hit >= needed,
        "cited_expected": cited,
        "cited_by_model": by_model,
        "facts_hit": hit,
        "facts_needed": needed,
    }


def _question_work(
    ctx: RunContext,
    questions: Sequence[Question],
    resolution: Resolution,
    lookup: PageLookup,
    grades: Callable[[], dict[str, float | int | bool | None]] | None = None,
) -> list[Work]:
    """`grades` adds grades to each answered item, called in the order this PROCESS answers."""

    def produce(q: Question) -> Callable[[], Awaitable[Produced]]:
        async def run() -> Produced:
            produced = await answer_question(ctx, q, lookup)
            if grades is None:
                return produced
            return replace(produced, grades={**produced.grades, **grades()})

        return run

    return [
        Work(
            item_id=q.id,
            input_digest=question_digest(q),
            produce=produce(q),
            stale=not resolution.pages_ok(parse_page_key(k) for k in q.relevance),
        )
        for q in questions
    ]


async def _resolve_questions(ctx: RunContext, questions: Sequence[Question]) -> Resolution:
    return await resolve_keys(ctx, (parse_page_key(k) for q in questions for k in q.relevance), ())


# ------------------------------------------------------------------ the post-hoc judge
STAGING_SUFFIX = ".answers.staging"  # not *.jsonl: the Eval page never lists a staging file


def staging_path(final: Path) -> Path:
    return final.with_name(final.name.removesuffix(".jsonl") + STAGING_SUFFIX)


def final_path(path: Path) -> Path:
    """The run file for a staging path (or the path itself)."""
    if path.name.endswith(STAGING_SUFFIX):
        return path.with_name(path.name.removesuffix(STAGING_SUFFIX) + ".jsonl")
    return path


@dataclass(frozen=True)
class JudgeTarget:
    question: str
    answer: str
    criteria: tuple[Criterion, ...]
    facts: tuple[str, ...] = ()
    level: str | None = None


async def shown_notes(ctx: RunContext, shown: Sequence[ShownRef]) -> tuple[str, ...] | None:
    """The shown texts, in label order, or None when any chunk is gone or its text changed
    since the run recorded it (the judge must see exactly what the model saw)."""
    ids = [UUID(s.chunk_id) for s in shown]
    response = await _setup_call(ctx, partial(ctx.client.eval_chunks, ids))
    return checked_texts(shown, response)


def checked_texts(
    shown: Sequence[ShownRef], response: EvalChunksResponse
) -> tuple[str, ...] | None:
    """Texts in S-label order, each matching the digest the run recorded AND its own content
    (a server-side digest that drifted from the text would otherwise pass unnoticed)."""
    by_id = {str(c.chunk_id): c for c in response.chunks}
    texts: list[str] = []
    for ref in sorted(shown, key=lambda s: s.label):
        chunk = by_id.get(ref.chunk_id)
        if chunk is None or chunk.display_sha256 != ref.display_sha256:
            return None
        if hashlib.sha256(chunk.display_text.encode()).hexdigest() != ref.display_sha256:
            return None
        texts.append(chunk.display_text)
    return tuple(texts)


async def judge_row(
    ctx: RunContext, judge: Judge, row: RunRow, target: JudgeTarget | None
) -> tuple[RunRow, list[JudgeResult]]:
    if target is None or row.error is not None or row.stale or not row.shown:
        return row, []
    notes = await shown_notes(ctx, row.shown)
    if notes is None:
        return row.model_copy(update={"stale": True}), []
    # make_item computes the digest from exactly what is shown (the label queue uses the same
    # function, so owner labels join these results)
    item = make_item(
        target.criteria, target.question, target.answer, notes, target.facts, target.level
    )
    await ctx.judge_pacer.wait()
    try:
        scored = await judge.score([item])
    except Exception as exc:  # any adapter: the message may hold a provider URL, so not shown
        raise RunAborted(
            f"the judge failed ({type(exc).__name__}); the answers are saved: resume the run"
        ) from None
    results = [r for r in scored if r.item_digest == item.item_digest]
    order = {c: i for i, c in enumerate(target.criteria)}
    results.sort(key=lambda r: order.get(r.criterion, len(order)))
    return row.model_copy(update={"judge_refs": [r.cache_key for r in results]}), results


async def judge_phase(
    ctx: RunContext,
    judge: Judge,
    staging: Path,
    final: Path,
    target: Callable[[RunRow], JudgeTarget | None],
) -> list[JudgeResult]:
    """Copy the finished staging rows into the run file, each with its judge refs. Runs only
    after answering has finished (never concurrently, ADR-0017 §4). Resumable by row count:
    the final rows are the staging rows in order, so the first `len(final.rows)` are done.

    The run file's header names THIS judge (answering never depends on the judge, so the
    staging header names none). A judge phase resumed with another judge is refused: its rows
    would mix two judges' scores under one key. Before any row is judged, switching judges
    (e.g. Ollama failed at startup, Gemini instead) is fine: every row gets the new one."""
    staged = read_run(staging)
    if staged.footer is None:
        raise RunAborted("answering has not finished: resume the run first")
    resumed = unfinished_run(final)
    if resumed is not None and resumed.header.judge != judge.key:
        raise RunAborted(
            f"{final.name} was being judged by {_judge_name(resumed.header.judge)}, not "
            f"{_judge_name(judge.key)}: resume with that judge, or delete {final.name} to "
            "re-judge every answer with this one (the answers are kept in the staging file)"
        )
    already = len(resumed.rows) if resumed is not None else 0
    header = staged.header.model_copy(update={"judge": judge.key})
    results: list[JudgeResult] = []
    with RunWriter(final, header, resume=resumed is not None) as writer:
        for index, row in enumerate(staged.rows):
            if index < already:
                continue
            judged, found = await judge_row(ctx, judge, row, target(row))
            results.extend(found)
            writer.row(judged)
            ctx.log(f"[judge {index + 1}/{len(staged.rows)}] {row.item_id}: {len(found)} scores")
        writer.finish()
    _remove(staging)
    return results


def _judge_name(key: JudgeKey | None) -> str:
    return "no judge" if key is None else f"{key.provider}/{key.model_version} ({key.prompt})"


async def answer_then_judge(
    ctx: RunContext,
    plan: Plan,
    judge: Judge | None,
    target: Callable[[RunRow], JudgeTarget | None],
) -> tuple[Path, list[JudgeResult]]:
    """Answer into a staging file, then judge into the run file. Resume picks up either phase.
    The staging header names no judge: `judge_phase` writes the judge that actually scores."""
    run_id, path, resume = resume_target(ctx, plan.kind)
    final = final_path(path)
    staging = staging_path(final)
    # Branch on what EXISTS, not on whether a judge was passed this time (integrator fix of
    # review P1): a resume with a different judge setting must never silently re-answer every
    # item (spending the free-tier budget resume exists to protect) or crash mid-way.
    if resume and judge is None and staging.exists():
        raise RunAborted(f"{final.name} was started with a judge: resume it with --judge-config")
    if resume and judge is not None and not staging.exists():
        started = unfinished_run(final)
        if started is not None and started.header.judge is None:
            raise RunAborted(
                f"{final.name} was started without a judge: resume it without --judge-config"
            )
    if judge is None:
        await drive(ctx, replace(plan, judge=None), final, run_id, resume=resume)
        return final, []
    drop_finished_staging(final, staging)
    if answering_pending(final, staging):
        await drive(ctx, replace(plan, judge=None), staging, run_id, resume=resume)
    return final, await judge_phase(ctx, judge, staging, final, target)


def answering_pending(final: Path, staging: Path) -> bool:
    """False once the staging file is finished (or its rows already reached the run file). An
    unfinished run file counts as judge-phase output only if its header names a judge."""
    started = unfinished_run(final)
    if started is not None and started.header.judge is not None:
        return False
    return not (staging.exists() and read_run(staging).footer is not None)


def drop_finished_staging(final: Path, staging: Path) -> None:
    """A crash between the judge phase's `finish()` and the staging file's removal leaves a
    finished run next to a stale staging file. Without this, every `--resume` would find that
    staging file and abort on the finished run, forever. Remove it and say there is nothing to
    resume (the run file is complete)."""
    if not (staging.exists() and final.exists()):
        return
    if read_run(final).footer is not None:
        _remove(staging)
        raise RunAborted(f"{final.name} already finished: removed its leftover staging file")


def _remove(path: Path) -> None:
    """The staging file goes once its rows are in the run file (it would otherwise resume)."""
    path.unlink()


def find_resumable(runs_dir: Path, kind: EvalKind, arm: str) -> Path | None:
    """The newest unfinished run of (kind, arm): an unfinished run file, or a staging file
    (whose judge phase has not finished, whether or not answering has)."""
    candidates = [*runs_dir.glob("*.jsonl"), *runs_dir.glob(f"*{STAGING_SUFFIX}")]
    found: list[tuple[datetime, Path]] = []
    for path in candidates:
        try:
            run = read_run(path)
        except (ValueError, OSError):
            continue
        staging = path.name.endswith(STAGING_SUFFIX)
        if staging and _finished(final_path(path)):
            _remove(path)  # a judge phase that finished just before a crash (see above)
            continue
        same = run.header.eval_kind is kind and run.header.arm == arm
        if same and (staging or run.footer is None):
            found.append((run.header.started_at, path))
    if not found:
        return None
    newest = max(found, key=lambda pair: (pair[0], pair[1].name))[1]
    final = final_path(newest)
    return final if final.exists() else newest


def _finished(path: Path) -> bool:
    try:
        return path.exists() and read_run(path).footer is not None
    except (ValueError, OSError):
        return False


# ------------------------------------------------------------------ E-G
async def run_generator(ctx: RunContext, qs: QuestionSet, judge: Judge | None) -> Path:
    """E-G: answer every selected question, THEN judge the answered rows (relevance,
    completeness against the D1 facts, faithfulness to the shown notes)."""
    questions = selected_questions(qs, ctx.splits)
    by_id = {q.id: q for q in questions}
    resolution = await _resolve_questions(ctx, questions)
    work = _question_work(ctx, questions, resolution, PageLookup(ctx))
    plan = Plan(
        EvalKind.generator,
        Surface.chat,
        dataset_ref(qs.name, qs, ctx.splits),
        work,
        judge=judge.key if judge is not None else None,
    )

    def target(row: RunRow) -> JudgeTarget | None:
        q = by_id.get(row.item_id)
        out = row.output
        if q is None or not isinstance(out, AnswerOutput) or out.abstained:
            return None  # judge scores answered rows only (ADR-0017 §3 E-G)
        facts = tuple(f.fact for f in q.facts)
        return JudgeTarget(q.question, out.answer, E_G_CRITERIA, facts=facts)

    path, results = await answer_then_judge(ctx, plan, judge, target)
    write_summary(ctx, path, Datasets(questions=qs), results)
    return path


# ------------------------------------------------------------------ E-A and the latency pass
async def run_application(ctx: RunContext, qs: QuestionSet) -> Path:
    """E-A: the full stack on the snapshot; correctness, tokens and classified errors. No judge
    and no latency here: latency comes from `run_latency`, a separate pass."""
    questions = selected_questions(qs, ctx.splits)
    resolution = await _resolve_questions(ctx, questions)
    work = _question_work(ctx, questions, resolution, PageLookup(ctx))
    plan = Plan(EvalKind.application, Surface.chat, dataset_ref(qs.name, qs, ctx.splits), work)
    run_id, path, resume = resume_target(ctx, EvalKind.application)
    await drive(ctx, plan, path, run_id, resume=resume)
    write_summary(ctx, path, Datasets(questions=qs))
    return path


async def run_latency(ctx: RunContext, qs: QuestionSet) -> Path:
    """The dedicated latency pass (ADR-0017 §3 E-A): every question again, nothing else running
    (no judge). The first `COLD_QUESTIONS` answers warm the models and are marked `cold`, so the
    TTFT / total percentiles exclude them and report them apart.

    "First" means first answered by THIS process, not first in the list: after a crash and a
    resume the models start unloaded again, so the resumed session's first answers are the
    cold ones, and their rows say so (`grades["cold"]`).

    Known gap (handoff, EV-METRICS): `metrics.cold_and_warm_rows` takes the first
    COLD_QUESTIONS by first-asked order and ignores `grades["cold"]`, so after a resume the
    resumed session's cold starts are counted in the warm percentiles. Without a resume the
    two agree (tested). Latency is warn-only, so the gate is not affected."""
    questions = selected_questions(qs, ctx.splits)
    resolution = await _resolve_questions(ctx, questions)
    answered = itertools.count()

    def cold() -> dict[str, float | int | bool | None]:
        return {"cold": next(answered) < COLD_QUESTIONS}

    work = _question_work(ctx, questions, resolution, PageLookup(ctx), grades=cold)
    plan = Plan(EvalKind.latency, Surface.chat, dataset_ref(qs.name, qs, ctx.splits), work)
    run_id, path, resume = resume_target(ctx, EvalKind.latency)
    await drive(ctx, plan, path, run_id, resume=resume)
    write_summary(ctx, path, Datasets(questions=qs))
    return path


# ------------------------------------------------------------------ E-V
def _span(start: int | None, end: int | None) -> Span | None:
    if start is None or end is None or end <= start:
        return None
    return Span(start=start, end=end)


def verify_output(response: EvalVerifyResponse) -> VerifyOutput:
    """Reduce the pre-merge evidence to what E-V needs. `caught_by` is computed later from
    these spans (in metrics, against the mutation span), never taken from the server."""
    located = [s for c in response.verifier_claims if (s := _span(c.start, c.end)) is not None]
    verdict = response.verifier_verdict
    objected = bool(response.verifier_claims) or (verdict is not None and verdict != "verified")
    return VerifyOutput(
        status=str(response.status),
        verifier_ran=response.verifier_ran,
        verifier_objected=objected,
        coverage_spans=[
            s for u in response.coverage_units if (s := _span(u.start, u.end)) is not None
        ],
        verifier_spans=located,
        unlocated_verifier_claims=len(response.verifier_claims) - len(located),
        verifier_model=response.verifier_model,
        verifier_prompt=response.verifier_prompt,
    )


async def _verify(ctx: RunContext, item: VerifierItem, chunk_ids: list[UUID]) -> Produced:
    response = await ctx.client.eval_verify(
        EvalVerifyRequest(
            answer=item.answer,
            shown_chunk_ids=chunk_ids,
            collection_ids=list(ctx.collection_ids),
        )
    )
    if response.verifier_ran and response.verifier_conclusive is False:
        raise InconclusiveVerifier("the verifier ran but was inconclusive (provider trouble)")
    prompts = {"verifier": response.verifier_prompt} if response.verifier_prompt else {}
    # an item the coverage gate caught (or one refused) never reaches the verifier, so it
    # cannot report one: it must not open the header (see `_Probe`)
    complete = response.verifier_ran and None not in (
        response.verifier_model,
        response.verifier_prompt,
    )
    return Produced(
        output=verify_output(response),
        observed=Observed(
            prompt_versions=prompts, verifier_model=response.verifier_model, complete=complete
        ),
    )


def cluster_order(items: Sequence[VerifierItem]) -> list[VerifierItem]:
    """Each clean source right before its mutations: the verifier is stochastic, so a planted
    item is compared with its clean source verified moments apart, in the same run."""
    groups: dict[str, list[VerifierItem]] = {}
    for item in items:
        groups.setdefault(item.source_id, []).append(item)
    return [
        item
        for group in groups.values()
        for item in sorted(group, key=lambda i: i.mutation is not None)
    ]


async def run_verifier(ctx: RunContext, vs: VerifierSet) -> Path:
    """E-V: every D3 item through the full pipeline (`/v1/eval/verify`). Clean items are always
    in the run, so each clean source is re-verified in the same run as its mutations."""
    resolution = await resolve_keys(ctx, (), (k for i in vs.items for k in i.shown))
    clean_sources = {i.source_id for i in vs.items if i.mutation is None}
    work: list[Work] = []
    for item in cluster_order(vs.items):
        if item.mutation is not None and item.source_id not in clean_sources:
            ctx.log(f"warning: {item.id} has no clean source item; it cannot count a badge flip")
        ids = resolution.chunk_ids(item.shown)
        shown = (
            tuple(
                ShownRef(label=n, chunk_id=str(c), display_sha256=k.display_sha256, chunk_key=k)
                for n, (c, k) in enumerate(zip(ids, item.shown, strict=True), 1)
            )
            if ids is not None
            else ()
        )
        work.append(
            Work(
                item_id=item.id,
                input_digest=digest(item.model_dump(mode="json")),
                produce=partial(_verify, ctx, item, ids or []),
                stale=ids is None,
                shown=shown,
            )
        )
    plan = Plan(EvalKind.verifier, Surface.verifier, dataset_ref(vs.name, vs, ctx.splits), work)
    run_id, path, resume = resume_target(ctx, EvalKind.verifier)
    await drive(ctx, plan, path, run_id, resume=resume)
    write_summary(ctx, path, Datasets(verifier=vs))
    return path


# ------------------------------------------------------------------ E-I
def grade_levels(seed: InterviewSeed, tier: AnswerTier) -> tuple[str, ...]:
    """Gold and partial are swept across all 5 levels; wrong and misconception are graded at
    the seed's own level only (ADR-0017 §3 E-I: the sweep would only measure noise there)."""
    return LEVELS if tier in SWEEP_TIERS else (seed.seed_level,)


async def _grade(
    ctx: RunContext,
    seed: InterviewSeed,
    tier: AnswerTier,
    level: str,
    replicate: int,
    chunk_ids: list[UUID],
) -> Produced:
    grade = await ctx.client.interview_grade(
        InterviewGradeRequest(
            collection_ids=list(ctx.collection_ids),
            question=seed.question,
            kind=InterviewKind(seed.kind),
            chunk_ids=chunk_ids,
            expected_points=seed.expected_points,
            answer=seed.answers[tier],
            level=InterviewLevel(level),
        )
    )
    quotes = list(grade.misconceptions)
    output = GradeOutput(
        seed_id=seed.id,
        tier=tier,
        level=level,
        replicate=replicate,
        score=grade.score,
        # The grade contract returns only the quotes that survived the server's quote rule and
        # verifier; the pre-filter list is not exposed, so `given` is a lower bound (= kept),
        # and the row says so (handoff: an additive `misconceptions_given` field).
        misconceptions_given=quotes,
        misconceptions_kept=quotes,
    )
    return Produced(output=output, grades={"given_is_kept": True})


LEAK_THRESHOLD = 0.5  # share of the expected points' content lemmas the question gives away
_FENCE = re.compile(r"```[^\n]*\n(.*?)```", re.DOTALL)
_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*\(?")


def answer_leak(question: str, expected_points: Sequence[str]) -> bool:
    """The question gives its own answer away: more than LEAK_THRESHOLD of the expected points'
    content lemmas already appear in it (ADR-0017 §3 E-I, deterministic)."""
    points = frozenset().union(*(content_lemmas(p) for p in expected_points))
    if not points:
        return False
    return len(points & content_lemmas(question)) / len(points) > LEAK_THRESHOLD


def _code_shaped(name: str, *, called: bool) -> bool:
    camel = any(c.isupper() for c in name[1:]) and any(c.islower() for c in name)
    return len(name) >= 3 and (called or "_" in name or "." in name or camel)


def code_identifiers(notes: Sequence[str]) -> frozenset[str]:
    """Code-shaped names in the notes' fenced blocks (dotted, snake_case, camelCase or called),
    plus the shaped parts of a dotted name (`store.put_if_absent` also yields `put_if_absent`):
    plain words inside a code block would make every question "reference the code"."""
    names: set[str] = set()
    for note in notes:
        for block in _FENCE.findall(note):
            for token in _IDENTIFIER.findall(block):
                called, name = token.endswith("("), token.rstrip("(")
                parts = name.split(".")
                candidates = [(name, called), *((p, False) for p in parts)]
                candidates.append((parts[-1], called))
                names.update(n for n, c in candidates if _code_shaped(n, called=c))
    return frozenset(names)


def references_code(question: str, notes: Sequence[str]) -> bool:
    """kind = code => the question points at the code: a code span or a name from it."""
    if "`" in question:
        return True
    return any(
        re.search(rf"(?<![\w.]){re.escape(n)}(?![\w])", question) for n in code_identifiers(notes)
    )


def question_grades(
    seed: InterviewSeed, question: InterviewQuestion, seed_ids: Sequence[UUID], notes: Sequence[str]
) -> dict[str, float | int | bool | None]:
    """The deterministic question-quality checks (booleans: metrics reports each as a share)."""
    grades: dict[str, float | int | bool | None] = {
        "leak": answer_leak(question.question, question.expected_points),
        "seed_chunk_included": any(c in question.chunk_ids for c in seed_ids),
    }
    if seed.kind == "code":
        grades["code_ref"] = references_code(question.question, notes)
    return grades


async def _generate(
    ctx: RunContext, seed: InterviewSeed, chunk_ids: list[UUID], shown: tuple[ShownRef, ...]
) -> Produced:
    question = await ctx.client.interview_question(
        InterviewQuestionRequest(
            collection_ids=list(ctx.collection_ids),
            kinds=[InterviewKind(seed.kind)],
            chunk_ids=chunk_ids,
            level=InterviewLevel(seed.seed_level),
        )
    )
    grounded, notes = await grounding(ctx, question.chunk_ids, shown)
    output = QuestionOutput(
        seed_id=seed.id,
        question=question.question,
        question_kind=str(question.kind),
        level=str(question.level),
        chunk_ids=[str(c) for c in question.chunk_ids],
        expected_points=list(question.expected_points),
    )
    return Produced(
        output=output,
        shown=list(grounded),
        grades=question_grades(seed, question, chunk_ids, notes),
    )


async def grounding(
    ctx: RunContext, chunk_ids: Sequence[UUID], seed_shown: tuple[ShownRef, ...]
) -> tuple[tuple[ShownRef, ...], tuple[str, ...]]:
    """The notes the GENERATED question is grounded in (its own `chunk_ids`, which the server
    may widen beyond the seed's), as shown refs plus texts. They become the row's `shown`, so
    the question-quality judge and the code-reference check see what the question was built
    from, not only the seed. No chunk ids (a general question): the seed's notes."""
    if not chunk_ids:
        ids = [UUID(s.chunk_id) for s in seed_shown]
        texts = checked_texts(seed_shown, await ctx.client.eval_chunks(ids))
        return seed_shown, texts or ()
    response = await ctx.client.eval_chunks(chunk_ids)
    by_id = {c.chunk_id: c for c in response.chunks}
    if any(c not in by_id for c in chunk_ids):
        raise ItemFailure("the generated question names chunks /v1/eval/chunks does not return")
    keys = {s.chunk_id: s.chunk_key for s in seed_shown}
    refs = tuple(
        ShownRef(
            label=n,
            chunk_id=str(c),
            display_sha256=by_id[c].display_sha256,
            chunk_key=keys.get(str(c)),
        )
        for n, c in enumerate(chunk_ids, 1)
    )
    texts = checked_texts(refs, response)
    if texts is None:  # the server's digest does not match its own text
        raise ItemFailure("a grounding chunk's text does not match its digest")
    return refs, texts


def _seed_shown(seed: InterviewSeed, ids: list[UUID] | None) -> tuple[ShownRef, ...]:
    if ids is None:
        return ()
    return tuple(
        ShownRef(label=n, chunk_id=str(c), display_sha256=k.display_sha256, chunk_key=k)
        for n, (c, k) in enumerate(zip(ids, seed.chunk_keys, strict=True), 1)
    )


def interview_work(
    ctx: RunContext, iv: InterviewSet, resolution: Resolution, questions_per_seed: int
) -> list[Work]:
    """Frozen D4 grades (tokenless, eval scope) for every tier and level, two replicates each,
    then `questions_per_seed` generated questions targeted at the seed's notes."""
    work: list[Work] = []
    for seed in iv.seeds:
        ids = resolution.chunk_ids(seed.chunk_keys)
        shown = _seed_shown(seed, ids)
        for tier in AnswerTier:
            for level in grade_levels(seed, tier):
                for rep in REPLICATES:
                    work.append(
                        Work(
                            item_id=f"{seed.id}/{tier.value}/{level}/{rep}",
                            input_digest=digest([seed.id, tier.value, level, rep]),
                            produce=partial(_grade, ctx, seed, tier, level, rep, ids or []),
                            stale=ids is None,
                            shown=shown,
                        )
                    )
        for n in range(questions_per_seed):
            work.append(
                Work(
                    item_id=f"{seed.id}/question/{seed.seed_level}/{n}",
                    input_digest=digest([seed.id, "question", seed.seed_level, n]),
                    produce=partial(_generate, ctx, seed, ids or [], shown),
                    stale=ids is None,
                    shown=shown,
                )
            )
    return work


async def run_interview(
    ctx: RunContext,
    iv: InterviewSet,
    judge: Judge | None = None,
    *,
    questions_per_seed: int = 2,
) -> Path:
    """E-I: grading quality on the frozen answers, the level sweep, and generated-question
    quality. With a judge, the questions are judged post-hoc (grounded, on topic, at level),
    report-only and uncalibrated unless 40 are labelled (ADR-0017 §3)."""
    resolution = await resolve_keys(ctx, (), (k for s in iv.seeds for k in s.chunk_keys))
    work = interview_work(ctx, iv, resolution, questions_per_seed)
    plan = Plan(
        EvalKind.interview,
        Surface.interview,
        dataset_ref(iv.name, iv, ctx.splits),
        work,
        judge=judge.key if judge is not None else None,
    )

    topics = {s.id: s.topic for s in iv.seeds}

    def target(row: RunRow) -> JudgeTarget | None:
        out = row.output
        if not isinstance(out, QuestionOutput) or out.seed_id not in topics:
            return None
        # judge_question@1 shows TOPIC and GENERATED QUESTION, the seed notes and the level
        return JudgeTarget(topics[out.seed_id], out.question, E_I_CRITERIA, level=out.level)

    path, results = await answer_then_judge(ctx, plan, judge, target)
    write_summary(ctx, path, Datasets(interview=iv), results)
    return path
