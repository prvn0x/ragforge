"""Shared fakes for the UI tests: a scripted `Backend` and event builders. No network."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from streamlit.testing.v1 import AppTest
from streamlit.testing.v1.element_tree import Block, Node

from ragforge_contracts.api import (
    ApprovalDecision,
    Citation,
    Collection,
    Connector,
    Document,
    InterviewGrade,
    InterviewGradeRequest,
    InterviewQuestion,
    InterviewQuestionRequest,
    Job,
    UnsupportedClaim,
)
from ragforge_contracts.sse import ChatEvent, CitationEvent, DoneEvent, TokenEvent

APP = Path(__file__).parents[3] / "ui" / "src" / "ragforge_ui" / "app.py"
NOW = datetime(2026, 10, 18, tzinfo=UTC)
# What the exposure tests must see refused; never bound by any test (S104 is about binding).
ALL_INTERFACES = "0.0.0.0"  # noqa: S104
COLLECTION = Collection(id=uuid4(), name="notes", active_space_id=1, created_at=NOW)


def citation(label: int = 1, *, inferred: bool = False, title: str = "Kafka basics") -> Citation:
    return Citation(
        anchor=f"block-{label}",
        document_id=uuid4(),
        title=title,
        heading_path=["Consumers"],
        url="https://www.notion.so/Kafka-basics-abc#block1",
        snippet="...",
        inferred=inferred,
        label=label,
    )


def done(conversation_id: UUID) -> DoneEvent:
    return DoneEvent(conversation_id=conversation_id, usage=[], prompt_version="answer@11")


def cited_answer(conversation_id: UUID, text: str = "Consumers poll \\[S1\\].") -> list[ChatEvent]:
    return [TokenEvent(text=text), CitationEvent(citation=citation()), done(conversation_id)]


@dataclass
class FakeBackend:
    """Plays back one scripted stream per call, in order, and records every request."""

    # one entry per stream: events (an exception in the list is raised there), or an exception
    # raised before the first event
    streams: list[Sequence[ChatEvent | BaseException] | BaseException] = field(default_factory=list)
    collections_list: list[Collection] = field(default_factory=lambda: [COLLECTION])
    connectors_list: list[Connector] = field(default_factory=list)
    chats: list[tuple[str, list[UUID], UUID | None]] = field(default_factory=list)
    resumes: list[tuple[UUID, dict[str, ApprovalDecision]]] = field(default_factory=list)
    synced: list[UUID] = field(default_factory=list)
    jobs: dict[UUID, Job] = field(default_factory=dict)
    job_errors: dict[UUID, BaseException] = field(default_factory=dict)
    documents_list: list[Document] = field(default_factory=list)
    documents_calls: int = 0
    fail_collections: BaseException | None = None
    fail_documents: BaseException | None = None
    # interview mode: one scripted reply per call (an exception is raised), requests recorded
    questions: list[InterviewQuestion | Exception] = field(default_factory=list)
    grades: list[InterviewGrade | Exception] = field(default_factory=list)
    question_requests: list[InterviewQuestionRequest] = field(default_factory=list)
    grade_requests: list[InterviewGradeRequest] = field(default_factory=list)

    @property
    def api_url(self) -> str:
        return "http://api.test"

    def chat(
        self, message: str, collection_ids: Sequence[UUID], conversation_id: UUID | None
    ) -> Iterator[ChatEvent]:
        self.chats.append((message, list(collection_ids), conversation_id))
        return self._next()

    def resume(
        self, run_id: UUID, decisions: Mapping[str, ApprovalDecision]
    ) -> Iterator[ChatEvent]:
        self.resumes.append((run_id, dict(decisions)))
        return self._next()

    def collections(self) -> list[Collection]:
        if self.fail_collections is not None:
            raise self.fail_collections
        return self.collections_list

    def connectors(self) -> list[Connector]:
        return self.connectors_list

    def documents(self) -> list[Document]:
        self.documents_calls += 1
        if self.fail_documents is not None:
            raise self.fail_documents
        return self.documents_list

    def sync_connector(self, connector_id: UUID) -> UUID:
        self.synced.append(connector_id)
        job_id = uuid4()
        self.jobs[job_id] = Job(
            id=job_id,
            kind="sync_connector",
            status="running",
            attempts=1,
            last_error=None,
            enqueued_at=NOW,
            progress={"pages": 3},
        )
        return job_id

    def job(self, job_id: UUID) -> Job:
        if job_id in self.job_errors:
            raise self.job_errors[job_id]
        return self.jobs[job_id]

    def interview_question(self, request: InterviewQuestionRequest) -> InterviewQuestion:
        self.question_requests.append(request)
        reply = self.questions.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply

    def interview_grade(self, request: InterviewGradeRequest) -> InterviewGrade:
        self.grade_requests.append(request)
        reply = self.grades.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply

    def _next(self) -> Iterator[ChatEvent]:
        script = self.streams.pop(0)

        def play() -> Iterator[ChatEvent]:
            if isinstance(script, BaseException):
                raise script
            for event in script:
                if isinstance(event, BaseException):
                    raise event
                yield event

        return play()


def app_with(backend: FakeBackend) -> AppTest:
    at = AppTest.from_file(str(APP), default_timeout=30)
    at.session_state["rf_backend"] = backend
    return at.run()


def texts(node: Node | Block) -> list[str]:
    """Every visible text value under `node`, in page order (depth first)."""
    out: list[str] = []
    children: dict[int, Any] | None = getattr(node, "children", None)
    if children:
        for key in sorted(children):
            out += texts(children[key])
        return out
    for attr in ("value", "label", "body"):
        value = getattr(node, attr, None)
        if isinstance(value, str) and value:
            out.append(value)
            break
    return out


# ------------------------------------------------------------------ interview builders
def interview_question(
    topic: str = "Kafka ISR",
    kind: str = "concept",
    difficulty: str = "standard",
    token: str | None = None,
    source: str = "Kafka replication",
    level: str = "senior_sde_1",
    points: list[str] | None = None,
) -> InterviewQuestion:
    return InterviewQuestion(
        question="Why does the leader wait for the ISR\\?",
        kind=kind,
        topic=topic,
        difficulty=difficulty,
        chunk_ids=[uuid4()],
        expected_points=points if points is not None else ["durability"],
        citations=[citation(1, title=source)],
        question_token=token,
        level=level,
    )


def general_question(
    topic: str = "Redis eviction",
    level: str = "senior_sde_1",
    text: str = "How does Redis evict\\?",
) -> InterviewQuestion:
    """A general-knowledge question (Phase 3c): no chunks, no citations, a token (required)."""
    return InterviewQuestion(
        question=text,
        kind="concept",
        topic=topic,
        difficulty="standard",
        chunk_ids=[],
        expected_points=["maxmemory policies"],
        citations=[],
        question_token="gtok",
        level=level,
        source="general",
    )


def general_grade(score: int = 4) -> InterviewGrade:
    return InterviewGrade(
        score=score,
        summary="Thin\\.",
        right=["names LRU"],
        missed=["LFU"],
        misconceptions=["evicts at random"],
        model_answer="Redis evicts by its maxmemory policy.",
        citations=[],
        verification="unverified",
        source="general",
    )


def interview_grade(score: int = 5, claims: list[UnsupportedClaim] | None = None) -> InterviewGrade:
    return InterviewGrade(
        score=score,
        summary="Partly right\\.",
        right=["names the ISR"],
        missed=["min.insync.replicas"],
        misconceptions=["acks=1 is durable"],
        model_answer="The leader waits for the ISR \\[S1\\].",
        citations=[citation(1, title="Kafka replication")],
        verification="partial",
        unsupported_claims=claims if claims is not None else [],
    )
