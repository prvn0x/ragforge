"""LOCKED schemas of the eval layer (ADR-0017 §2, §4, §7). The builders code against these and
never change them; a needed change goes through the integrator.

Files we write are strict (`extra="forbid"`): a typo in a hand-edited dataset fails loudly. Every
file carries `schema_version`; readers refuse a version they do not know.

Identity (ADR-0017 principle 6): datasets and run files never key items by database ids. Pages
are `PageKey`s and chunks `ChunkKey`s (ragforge_contracts.api), resolved at run time through
`/v1/eval/resolve`. Run rows also record the ids and digests they actually used, for drill-down.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ragforge_contracts.api import ChunkKey, PageKey

SCHEMA_VERSION: Literal[2] = 2


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


# ------------------------------------------------------------------ digests
def canonical_json(value: Any) -> bytes:
    """Sorted keys, no whitespace, UTF-8: the input of every digest in the eval layer."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def digest(value: Any) -> str:
    """Hex SHA-256 of `canonical_json(value)`; an item digest hashes model_dump(mode="json")."""
    return hashlib.sha256(canonical_json(value)).hexdigest()


def page_key_str(key: PageKey) -> str:
    """ "notion:<page id>" / "file:<filename>": the TOML-friendly form used in D1 `relevance`."""
    return f"{key.source_kind}:{key.key}"


def parse_page_key(text: str) -> PageKey:
    kind, sep, key = text.partition(":")
    if not sep or not kind or not key:
        raise ValueError(f"not a page key (expected '<source_kind>:<key>'): {text!r}")
    return PageKey(source_kind=kind, key=key)


# ------------------------------------------------------------------ enums
class EvalKind(StrEnum):
    retrieval = "retrieval"  # E-R
    generator = "generator"  # E-G
    verifier = "verifier"  # E-V
    interview = "interview"  # E-I
    application = "application"  # E-A
    latency = "latency"  # the dedicated latency pass of E-A


class Surface(StrEnum):
    chat = "chat"
    verifier = "verifier"
    interview = "interview"
    judge = "judge"


class Split(StrEnum):
    dev = "dev"
    heldout2 = "heldout2"  # sealed; builders never read it; one reveal only


class Need(StrEnum):
    any = "any"  # any one relevant page answers it -> hit@k, MRR
    all = "all"  # every relevant page is needed -> recall@k


class Mutation(StrEnum):
    identifier_typo = "identifier_typo"
    number_drift = "number_drift"
    entity_swap = "entity_swap"
    negation_flip = "negation_flip"
    fabricated_sentence = "fabricated_sentence"
    citation_moved = "citation_moved"
    guard_dropped = "guard_dropped"


class Layer(StrEnum):
    coverage = "coverage"
    verifier = "verifier"
    either = "either"  # citation_moved: coverage or verifier may catch it


# Each mutation's declared expected catching layer (ADR-0017 §2 D3 table). LOCKED.
EXPECTED_LAYER: dict[Mutation, Layer] = {
    Mutation.identifier_typo: Layer.verifier,
    Mutation.number_drift: Layer.verifier,
    Mutation.entity_swap: Layer.verifier,
    Mutation.negation_flip: Layer.verifier,
    Mutation.fabricated_sentence: Layer.coverage,
    Mutation.citation_moved: Layer.either,
    Mutation.guard_dropped: Layer.coverage,
}


class AnswerTier(StrEnum):
    gold = "gold"
    partial = "partial"
    wrong = "wrong"
    misconception = "misconception"  # gold + one planted false sentence (a minimal pair)


class Criterion(StrEnum):
    """Judge criteria, each scored 0 (no) / 1 (partly) / 2 (yes) against written anchors."""

    relevance = "relevance"  # E-G
    completeness = "completeness"  # E-G
    faithfulness = "faithfulness"  # E-G (and the verified-answer screen)
    grounded = "grounded"  # E-I question quality
    on_topic = "on_topic"  # E-I question quality
    at_level = "at_level"  # E-I question quality


E_G_CRITERIA: tuple[Criterion, ...] = (
    Criterion.relevance,
    Criterion.completeness,
    Criterion.faithfulness,
)
E_I_CRITERIA: tuple[Criterion, ...] = (Criterion.grounded, Criterion.on_topic, Criterion.at_level)

Score = Annotated[int, Field(ge=0, le=2)]


# ------------------------------------------------------------------ D1: Q&A questions
class Fact(_Strict):
    fact: str
    match: Annotated[list[str], Field(min_length=1)]


class Question(_Strict):
    """One D1 / D0 question. `pages` (titles) grades citations as before; `relevance` (keyed by
    `page_key_str`) grades retrieval: 2 = the primary page, 1 = also relevant."""

    id: Annotated[str, Field(pattern=r"^[a-z0-9][a-z0-9-]{0,63}$")]
    split: Split
    question: Annotated[str, Field(min_length=1, max_length=2000)]
    answerable: bool = True
    near_miss: bool = False  # an adjacent topic the notes do not cover (answerable = False)
    pages: list[str] = Field(default_factory=list)
    facts: list[Fact] = Field(default_factory=list)
    relevance: dict[str, Annotated[int, Field(ge=1, le=2)]] = Field(default_factory=dict)
    need: Need = Need.any

    @model_validator(mode="after")
    def _consistent(self) -> Question:
        for key in self.relevance:
            parse_page_key(key)
        if self.answerable and not self.relevance:
            raise ValueError(f"{self.id}: an answerable question needs `relevance`")
        if not self.answerable and (self.relevance or self.facts):
            raise ValueError(f"{self.id}: an unanswerable question has no relevance or facts")
        if self.near_miss and self.answerable:
            raise ValueError(f"{self.id}: a near-miss is unanswerable by definition")
        return self


class QuestionSet(_Strict):
    schema_version: Literal[2] = SCHEMA_VERSION
    name: str  # "demo" | "private"
    questions: list[Question]


# ------------------------------------------------------------------ D3: planted verifier items
class Span(_Strict):
    start: Annotated[int, Field(ge=0)]  # [start, end) in the mutated answer
    end: Annotated[int, Field(ge=0)]

    @model_validator(mode="after")
    def _ordered(self) -> Span:
        if self.end <= self.start:
            raise ValueError("span: end must be after start")
        return self


class VerifierItem(_Strict):
    """A clean item (mutation None) or a planted one. `source_id` is the cluster for the
    clustered bootstrap: every mutation of one answer shares its clean source's id."""

    id: str
    source_id: str
    question: str
    answer: str  # the clean answer, or the mutated one
    shown: Annotated[list[ChunkKey], Field(min_length=1)]  # ordered: shown[i] is S<i+1>
    mutation: Mutation | None = None
    span: Span | None = None  # where the mutation sits in `answer`
    validity_check: str | None = None  # which mechanical check proved it wrong (ADR-0017 D3)
    owner_reviewed: bool = False  # required True for negation_flip

    @model_validator(mode="after")
    def _planted_has_span(self) -> VerifierItem:
        planted = self.mutation is not None
        if planted != (self.span is not None) or planted != (self.validity_check is not None):
            raise ValueError(f"{self.id}: a planted item has a mutation, a span and a check")
        if self.span is not None and self.span.end > len(self.answer):
            raise ValueError(f"{self.id}: span outside the answer")
        return self

    @property
    def expected_layer(self) -> Layer | None:
        return EXPECTED_LAYER[self.mutation] if self.mutation else None


class VerifierSet(_Strict):
    schema_version: Literal[2] = SCHEMA_VERSION
    name: str
    items: list[VerifierItem]


# ------------------------------------------------------------------ D4: interview seeds
class InterviewSeed(_Strict):
    """Everything a frozen grade needs. Levels/kinds are the contract enum values as strings."""

    id: str
    topic: str
    kind: Literal["concept", "scenario", "design", "code"]
    seed_level: Literal["sde_1", "sde_2", "senior_sde_1", "senior_sde_2", "staff"]
    question: str
    expected_points: list[str]
    chunk_keys: Annotated[list[ChunkKey], Field(min_length=1, max_length=10)]
    answers: dict[AnswerTier, str]
    misconception_sentence: str  # the planted false sentence inside answers[misconception]

    @model_validator(mode="after")
    def _all_tiers(self) -> InterviewSeed:
        if set(self.answers) != set(AnswerTier):
            raise ValueError(f"{self.id}: needs exactly the four answer tiers")
        if self.misconception_sentence not in self.answers[AnswerTier.misconception]:
            raise ValueError(f"{self.id}: the misconception sentence must be in that answer")
        return self


class InterviewSet(_Strict):
    schema_version: Literal[2] = SCHEMA_VERSION
    name: str
    seeds: list[InterviewSeed]


# ------------------------------------------------------------------ D5: owner labels
class LabelKind(StrEnum):
    criteria = "criteria"  # judge calibration: every criterion of the item at once
    answer = "answer"  # E-A: correct / invented (the Phase 2b labels, v2 form)
    adjudication = "adjudication"  # a judge-flagged verified answer, confirmed or not
    mutation_review = "mutation_review"  # D3: the planted error really is an error


class Label(_Strict):
    """One append-only line of `labels.jsonl`. The newest line per (item_digest, kind) wins."""

    schema_version: Literal[2] = SCHEMA_VERSION
    item_digest: str  # digest of the item AS SHOWN to the owner (question + notes + answer)
    item_id: str
    kind: LabelKind
    scores: dict[Criterion, Score] = Field(default_factory=dict)  # kind = criteria
    correct: bool | None = None  # kind = answer
    invented: bool | None = None  # kind = answer
    confirmed: bool | None = None  # kind = adjudication / mutation_review
    note: str = ""
    batch_id: str  # calibration is recomputed only on a new batch (no flapping)
    labelled_at: datetime


# ------------------------------------------------------------------ judge
class JudgeKey(_Strict):
    """Everything that can change a judge's output. A changed field invalidates the cache AND
    marks calibration stale."""

    provider: Literal["ollama", "gemini", "other"]
    model_version: str  # what the PROVIDER returned (aliases move silently), not the config
    decoding: dict[str, float | int | str] = Field(default_factory=dict)  # temperature, num_ctx...
    prompt: str  # judge prompt name@version


class JudgeResult(_Strict):
    schema_version: Literal[2] = SCHEMA_VERSION
    cache_key: str  # digest({"judge": JudgeKey, "item": item_digest, "batch": batch_key|None})
    item_digest: str
    criterion: Criterion
    score: Score | None  # None: the reply was invalid (counted, never imputed)
    reason: str
    judge: JudgeKey
    batch_key: str | None = None  # digest of the sorted member digests, when batched
    scored_at: datetime


# ------------------------------------------------------------------ run file v2
class ErrorClass(StrEnum):
    provider_transient = "provider_transient"  # 429 / 5xx: re-run before counting
    ours = "ours"  # anything else: counts, and blocks the gate


class RunError(_Strict):
    cls: ErrorClass
    status: int | None = None
    message: str  # no secrets, no note text


class SnapshotRef(_Strict):
    collection_id: str
    content_digest: str  # the gate requires equality
    index_digest: str  # recorded; may differ between arms (re-embed)


class RetrievalConfig(_Strict):
    embed_model: str
    space_id: int
    chunker_version: str
    tau: float
    rerank_model: str
    rerank_top: int


class DatasetRef(_Strict):
    name: str
    digest: str  # digest of the whole dataset file's parsed model_dump
    splits: list[Split]


class RunHeader(_Strict):
    schema_version: Literal[2] = SCHEMA_VERSION
    type: Literal["header"] = "header"
    run_id: str
    eval_kind: EvalKind
    surface: Surface
    arm: Annotated[str, Field(pattern=r"^[A-Za-z0-9._:-]{1,64}$")]
    started_at: datetime
    dataset: DatasetRef
    snapshot: SnapshotRef | None = None  # None only for the D0 demo tenant
    retrieval: RetrievalConfig | None = None
    chat_model: str | None = None
    verifier_model: str | None = None
    prompt_versions: dict[str, str] = Field(default_factory=dict)  # per surface
    judge: JudgeKey | None = None
    transient_retries: int = 2
    override: str | None = None  # a gate override reason, shown on the Eval page


class ShownRef(_Strict):
    label: int
    chunk_id: str
    display_sha256: str
    chunk_key: ChunkKey | None = None


class Timings(_Strict):
    ttft_s: float | None = None
    total_s: float | None = None


class Tokens(_Strict):
    prompt: int = 0
    completion: int = 0


# Outputs, one shape per eval kind (a discriminated union on `kind`).
class RetrievalOutput(_Strict):
    kind: Literal["retrieval"] = "retrieval"
    # stage name -> page keys (page_key_str) in rank order, chunks de-duplicated to pages
    stages: dict[Literal["keyword", "vector", "fused", "reranked"], list[str]]
    top_rerank_score: float | None = None  # for unanswerable questions vs tau


class AnswerOutput(_Strict):
    """E-G / E-A: one chat turn."""

    kind: Literal["answer"] = "answer"
    answer: str
    status: str  # the verification status string
    cited_pages: list[str]  # page_key_str of each citation, in order
    abstained: bool
    flagged: int = 0


class VerifyOutput(_Strict):
    """E-V: the /v1/eval/verify evidence, reduced to what the metrics need."""

    kind: Literal["verify"] = "verify"
    status: str
    verifier_ran: bool
    verifier_objected: bool  # a non-verified verdict or at least one verifier claim
    coverage_spans: list[Span]
    verifier_spans: list[Span]  # located verifier claims
    unlocated_verifier_claims: int
    verifier_model: str | None = None
    verifier_prompt: str | None = None


class GradeOutput(_Strict):
    """E-I: one grade of one frozen answer at one level (replicate 0 or 1)."""

    kind: Literal["grade"] = "grade"
    seed_id: str
    tier: AnswerTier
    level: str
    replicate: Annotated[int, Field(ge=0, le=1)]
    score: Annotated[int, Field(ge=0, le=10)]
    misconceptions_given: list[str]  # before the server's quote rule and verifier
    misconceptions_kept: list[str]


class QuestionOutput(_Strict):
    """E-I: one generated question (question quality)."""

    kind: Literal["question"] = "question"
    seed_id: str
    question: str
    question_kind: str
    level: str
    chunk_ids: list[str]
    expected_points: list[str]


Output = Annotated[
    RetrievalOutput | AnswerOutput | VerifyOutput | GradeOutput | QuestionOutput,
    Field(discriminator="kind"),
]


class RunRow(_Strict):
    schema_version: Literal[2] = SCHEMA_VERSION
    type: Literal["row"] = "row"
    item_id: str
    input_digest: str
    shown: list[ShownRef] = Field(default_factory=list)
    output: Output | None = None  # None when the item errored
    grades: dict[str, float | int | bool | None] = Field(default_factory=dict)  # deterministic
    judge_refs: list[str] = Field(default_factory=list)  # JudgeResult.cache_key
    timings: Timings = Field(default_factory=Timings)
    tokens: Tokens = Field(default_factory=Tokens)
    error: RunError | None = None
    attempt: int = 1
    stale: bool = False  # a key did not resolve, or its digest changed (principle 6)


class RunFooter(_Strict):
    schema_version: Literal[2] = SCHEMA_VERSION
    type: Literal["footer"] = "footer"
    finished_at: datetime
    rows: int


# ------------------------------------------------------------------ summaries, calibration, gate
class Method(StrEnum):
    point = "point"  # no interval (n too small, or a count)
    wilson = "wilson"
    clopper_pearson = "clopper_pearson"
    cluster_bootstrap = "cluster_bootstrap"
    bootstrap = "bootstrap"  # an item-level (non-clustered) bootstrap, e.g. the kappa CI
    exact = "exact"


class MetricValue(_Strict):
    name: str  # e.g. "retrieval.reranked.hit@5", "verifier.badge_flip_recall"
    value: float | None
    ci_low: float | None = None
    ci_high: float | None = None
    n: int
    method: Method
    trusted: bool | None = None  # judge-derived metrics only; None = not judge-derived
    note: str = ""


class RunSummary(_Strict):
    schema_version: Literal[2] = SCHEMA_VERSION
    run_id: str
    eval_kind: EvalKind
    header_digest: str
    metrics: list[MetricValue]
    stale_items: int = 0
    errors_ours: int = 0
    errors_transient: int = 0


class Calibration(_Strict):
    schema_version: Literal[2] = SCHEMA_VERSION
    criterion: Criterion
    judge: JudgeKey
    label_batch_id: str
    n: int
    kappa_linear: float | None
    kappa_ci: tuple[float, float] | None
    raw_agreement: float | None
    confusion: list[list[int]]  # rows = owner score 0..2, cols = judge score 0..2
    leniency_p: float | None  # paired sign test of judge - owner
    trusted: bool  # kappa >= 0.6 AND ci_low >= 0.4 AND no significant leniency (ADR-0017 §4)
    computed_at: datetime


class GateOutcome(StrEnum):
    pass_ = "pass"  # noqa: S105 - a gate outcome, not a password
    warn = "warn"
    block = "block"
    not_evaluable = "not_evaluable"  # e.g. a chunker change on the snapshot (ADR-0017 §1)


class GateCheck(_Strict):
    name: str
    outcome: GateOutcome
    method: str
    n: int
    effect: float | None = None  # candidate - baseline
    p_value: float | None = None
    mde: float | None = None  # from the A/A discordance rate
    note: str = ""


class GateVerdict(_Strict):
    schema_version: Literal[2] = SCHEMA_VERSION
    surface: Surface
    baseline_run: str
    candidate_run: str
    checks: list[GateCheck]
    outcome: GateOutcome
    override: str | None = None
