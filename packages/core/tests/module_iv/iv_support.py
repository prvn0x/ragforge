"""Module IV test support: a prompt-aware fake chat model and note builders (not a test module).

`FakeChat` answers each prompt by NAME (it recognises the system prompt), so a test scripts
"the question call returns X, the verifier says Y" without depending on call order (the grade's
two ground checks run concurrently). It records every call and fails if a model is called while
a tenant transaction is open (PHASE1 rule), when given the tracking transaction manager.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID, uuid4

import psycopg

from ragforge_core.domain.models import ChatMessage, ChatTurnUsage, RankedChunk
from ragforge_core.prompts import render

Conn = psycopg.Connection[tuple[object, ...]]
Payload = dict[str, object]
Responder = Callable[[str], Payload | Exception]  # user message -> reply (or error to raise)

QUESTION = "interview_question"
GRADE = "interview_grade"
GENERAL_QUESTION = "interview_general_question"
GENERAL_GRADE = "interview_general_grade"
VERIFY = "ground_check"


def prompt_name(system: str) -> str:
    if system == render(VERIFY):
        return VERIFY
    if system.startswith("You write one general-knowledge interview question"):
        return GENERAL_QUESTION
    if system.startswith("You judge a candidate's general-knowledge interview answer"):
        return GENERAL_GRADE
    if system.startswith("You are a principal engineer interviewing"):
        return QUESTION
    if system.startswith("You grade a candidate's interview answer"):
        return GRADE
    raise AssertionError(f"unknown system prompt: {system[:60]!r}")


def supported(_: str) -> Payload:
    return {"verdict": "supported", "unsupported_claims": []}


@dataclass(frozen=True)
class Call:
    name: str
    system: str
    messages: list[ChatMessage]
    schema: dict[str, object]

    @property
    def user(self) -> str:
        return self.messages[0].content


@dataclass
class FakeChat:
    """`responders[name]` is a list consumed in order (the last one repeats)."""

    responders: dict[str, list[Responder]] = field(default_factory=dict)
    open_tx: Callable[[], int] = lambda: 0
    provider: str = "fake"
    model: str = "fake-chat"
    calls: list[Call] = field(default_factory=list)

    def on(self, name: str, *replies: Payload | Exception | Responder) -> FakeChat:
        self.responders[name] = [r if callable(r) else _const(r) for r in replies]
        return self

    def names(self) -> list[str]:
        return [c.name for c in self.calls]

    def of(self, name: str) -> list[Call]:
        return [c for c in self.calls if c.name == name]

    def stream(self, *args: Any, **kwargs: Any) -> Any:
        raise AssertionError("interview mode never streams")

    async def complete_json(
        self, system: str, messages: Sequence[ChatMessage], schema: dict[str, object]
    ) -> tuple[dict[str, object], ChatTurnUsage]:
        assert self.open_tx() == 0, "model called while a transaction is open"
        name = prompt_name(system)
        self.calls.append(Call(name, system, list(messages), schema))
        queue = self.responders.get(name) or [supported if name == VERIFY else _missing(name)]
        responder = queue.pop(0) if len(queue) > 1 else queue[0]
        reply = responder("\n".join(m.content for m in messages))
        if isinstance(reply, Exception):
            raise reply
        return reply, ChatTurnUsage(self.provider, self.model, 11, 5)


def _const(reply: Payload | Exception) -> Responder:
    return lambda _: reply


def _missing(name: str) -> Responder:
    def fail(_: str) -> Payload:
        raise AssertionError(f"unscripted {name} call")

    return fail


def question_reply(
    question: str = "Why can only in-sync replicas become leader, and what does it cost?",
    topic: str = "Kafka ISR",
    points: Sequence[str] = (
        "Only in-sync replicas can become leader [S1]",
        "The leader tracks the in-sync replicas [S1]",
        "A replica that falls behind leaves the in-sync set [S1]",
    ),
) -> Payload:
    return {"question": question, "topic": topic, "expected_points": list(points)}


def grade_reply(**overrides: object) -> Payload:
    reply: Payload = {
        "score": 6,
        "summary": "Correct core, misses the cost.",
        "right": ["Named the ISR"],
        "missed": ["A replica that falls behind leaves the in-sync set [S1]"],
        "misconceptions": [
            {
                "said": "Any replica can become leader",
                "correction": "Only in-sync replicas can become leader [S1]",
            }
        ],
        "model_answer": "Only in-sync replicas can become leader [S1]. "
        "The leader tracks the in-sync replicas [S1].",
    }
    reply.update(overrides)
    return reply


KAFKA_TEXT = (
    "Only in-sync replicas can become leader. The leader tracks the in-sync replicas. "
    "A replica that falls behind leaves the in-sync set."
)


def note(
    n: int,
    *,
    text: str = KAFKA_TEXT,
    title: str = "Kafka replication",
    document: UUID | None = None,
    collection: UUID | None = None,
) -> RankedChunk:
    return RankedChunk(
        chunk_id=UUID(int=5000 + n),
        document_id=document or UUID(int=6000 + n),
        collection_id=collection or UUID(int=7000),
        anchor=f"block-{n}",
        title=title,
        heading_path=("Replication",),
        display_text=text,
        display_sha256=hashlib.sha256(text.encode()).digest(),
        external_url=f"https://www.notion.so/page-{n}",
        fused_score=0.0,
    )


# ----------------------------------------------------------------- database builders
@dataclass
class Page:
    document: UUID
    chunks: list[UUID]


def connector(superuser: Conn, tenant: UUID, collection: UUID) -> UUID:
    cid = uuid4()
    superuser.execute(
        "INSERT INTO rf.connectors (id, tenant_id, kind, target_collection_id, display_name) "
        "VALUES (%s, %s, 'notion', %s, 'c')",
        (cid, tenant, collection),
    )
    return cid


def page(
    superuser: Conn,
    tenant: UUID,
    collection: UUID,
    connector_id: UUID,
    title: str,
    sections: Sequence[tuple[Sequence[str], str]],
    *,
    deleted: bool = False,
) -> Page:
    """One page with one chunk per (heading path, text), in page order."""
    document = uuid4()
    superuser.execute(
        "INSERT INTO rf.documents (id, tenant_id, collection_id, connector_id, source_kind, "
        "external_id, external_url, title, deleted_at) VALUES (%s, %s, %s, %s, 'notion', %s, %s, "
        "%s, CASE WHEN %s THEN now() END)",
        (
            document,
            tenant,
            collection,
            connector_id,
            f"page-{document.hex[:8]}",
            f"https://www.notion.so/{document.hex}",
            title,
            deleted,
        ),
    )
    chunks: list[UUID] = []
    for ordinal, (headings, text) in enumerate(sections):
        chunk = uuid4()
        digest = hashlib.sha256(f"{chunk}{text}".encode()).digest()
        superuser.execute(
            "INSERT INTO rf.chunks (id, tenant_id, collection_id, document_id, anchor, ordinal, "
            "heading_path, display_text, display_sha256, embed_sha256, token_count) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 10)",
            (
                chunk,
                tenant,
                collection,
                document,
                f"block-{ordinal}",
                ordinal,
                list(headings),
                text,
                digest,
                digest,
            ),
        )
        chunks.append(chunk)
    return Page(document, chunks)


LONG = " It matters because the cluster must stay available while replicas fail and recover."
