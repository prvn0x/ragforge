"""Value objects of interview mode (Phases 3b/3c). Immutable; no persistence or HTTP concerns.

The enums mirror `ragforge_contracts.api.InterviewKind` / `InterviewDifficulty` /
`InterviewLevel` / `InterviewSource` value for value
(a test keeps them identical), the same way `domain.models.Scope` mirrors `ApiScope`: the core
business layers never import the wire contracts, only the API maps between the two.

Every text field here is RAW (model output or note text). The API sanitizes it exactly once,
on the way out (ADR-0011).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Final
from uuid import UUID

from ragforge_core.domain.models import ClaimReasonCode, RankedChunk, Verification


class Kind(StrEnum):
    concept = "concept"
    scenario = "scenario"
    design = "design"
    code = "code"


class Difficulty(StrEnum):
    easier = "easier"
    standard = "standard"
    harder = "harder"


class Level(StrEnum):
    """The bar a session is asked AND graded at (owner decision 2026-10-19). Difficulty is a
    fine adjustment within it (`prompts.LEVELS`, `prompts.DIFFICULTY_RULES`)."""

    sde_1 = "sde_1"
    sde_2 = "sde_2"
    senior_sde_1 = "senior_sde_1"
    senior_sde_2 = "senior_sde_2"
    staff = "staff"


class Source(StrEnum):
    """Where a question comes from: the notes (grounded, verified), or the model's general
    knowledge (opt-in by the client only; never cited, never verified)."""

    notes = "notes"
    general = "general"


ALL_KINDS: Final = tuple(Kind)
DEFAULT_LEVEL: Final = Level.senior_sde_1


@dataclass(frozen=True, slots=True)
class CallTags:
    """What every `interview_model_call` log line carries besides the call itself, so an eval
    can slice cost, latency and failure categories by level, kind, difficulty and source. Enum
    values only: never text."""

    level: Level | None = None
    kind: Kind | None = None
    difficulty: Difficulty | None = None
    source: Source | None = None

    def fields(self) -> dict[str, str | None]:
        return {
            "level": self.level.value if self.level else None,
            "kind": self.kind.value if self.kind else None,
            "difficulty": self.difficulty.value if self.difficulty else None,
            "source": self.source.value if self.source else None,
        }


@dataclass(frozen=True, slots=True)
class InterviewSettings:
    """Tunables, filled from `config.Settings` by the API (these defaults serve evals/tests)."""

    # Retrieval hits below this rerank score are not "about the topic" (ADR-0006's tau).
    tau: float = 0.35
    rerank_top: int = 20
    # False: the chat model is below the verifier SLO, so nothing is ever badged verified.
    verifier_trusted: bool = True
    # The bar a request without a level is asked and graded at (operator config, never user
    # text; a request's own level is a closed enum, so the system prompts hold no user text).
    target_level: Level = DEFAULT_LEVEL
    # Candidate pool sizes for the seed sampler (chunks for a topic, pages for "any topic").
    topic_pool: int = 200
    sample_pool: int = 50
    # The seed plus at most this many neighbours of the same page go to the model.
    context_notes: int = 3
    # Acronym topics ("HLD"): at most this many distinct titles / heading segments are scanned
    # for a phrase whose initials spell the acronym, and at most `max_expansions` phrases are
    # searched for (each costs one FTS query and one embed + rerank).
    phrase_pool: int = 5000
    max_expansions: int = 3


@dataclass(frozen=True, slots=True)
class QuestionRequest:
    collection_ids: tuple[UUID, ...]
    topic: str | None
    kinds: tuple[Kind, ...]
    difficulty: Difficulty
    exclude_chunk_ids: frozenset[UUID]
    level: Level = DEFAULT_LEVEL  # already RESOLVED by the route (None -> the configured one)
    source: Source = Source.notes
    # source=general only: questions already asked (client-sent, untrusted DATA)
    exclude_questions: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class DraftQuestion:
    """A question as the model wrote it: RAW text, sanitized by the API before it leaves."""

    question: str
    kind: Kind
    topic: str
    difficulty: Difficulty
    notes: tuple[RankedChunk, ...]  # labelled S1..Sn in this order; () for source=general
    # notes: only the points that passed `verify.verified_items` (lexical pass + ground check);
    # general: the model's own key, label-free and unverified.
    expected_points: tuple[str, ...]
    level: Level = DEFAULT_LEVEL
    source: Source = Source.notes


@dataclass(frozen=True, slots=True)
class GradeRequest:
    collection_ids: tuple[UUID, ...]
    question: str  # client-sent: untrusted DATA
    kind: Kind
    difficulty: Difficulty
    chunk_ids: tuple[UUID, ...]
    expected_points: tuple[str, ...]  # client-sent hints: untrusted DATA
    answer: str  # the user's answer: untrusted DATA
    level: Level = DEFAULT_LEVEL  # resolved; bound by the question token when one is sent
    source: Source = Source.notes


@dataclass(frozen=True, slots=True)
class FlaggedClaim:
    text: str  # RAW, a verbatim substring of the model answer
    reason: ClaimReasonCode


@dataclass(frozen=True, slots=True)
class CheckedAnswer:
    """The model answer after the chat path's verification (ADR-0015 rules)."""

    text: str  # RAW, citation markers normalized (`[1]` -> `[S1]`): claims match this text
    status: Verification
    claims: tuple[FlaggedClaim, ...]
    cited: tuple[int, ...]  # labels the model cited, first-citation order
    inferred: tuple[int, ...]  # labels the server attributed (passed the ground check)


@dataclass(frozen=True, slots=True)
class Misconception:
    """A statement of the user's answer that the notes contradict. `correction` (what the notes
    say, with its [S<n>] label) is verified; `said` is a quote of the user's answer: kept only
    when it occurs in that answer, with citation markers removed (`grade.quoted`)."""

    said: str
    correction: str


@dataclass(frozen=True, slots=True)
class DraftGrade:
    """A grade as the model wrote it (RAW text) plus the verified model answer.

    `summary` and `right` are the model's JUDGEMENT of the user's answer: there is nothing in
    the notes to verify them against, so clients label them as such. `missed` and
    `misconceptions` state what the notes say: each passed the same lexical and ground
    verification as the model answer, and an item that did not is dropped (never shown)."""

    score: int
    summary: str
    right: tuple[str, ...]
    missed: tuple[str, ...]
    misconceptions: tuple[Misconception, ...]
    answer: CheckedAnswer
    notes: tuple[RankedChunk, ...]  # labelled S1..Sn in this order; () for source=general
    level: Level = DEFAULT_LEVEL
    source: Source = Source.notes
