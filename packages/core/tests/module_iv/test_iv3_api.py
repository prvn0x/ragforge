"""Interview Phase 3c on the real app and the real test database (RLS, the SQL reads): levels
and their token binding, the general-knowledge fallback, acronym topics, NO_NOTES vs EXHAUSTED
and weak question replies. Fake models only (no network)."""

from __future__ import annotations

from collections.abc import Callable, Collection, Sequence
from typing import Any
from uuid import UUID, uuid4

import httpx
from fastapi import FastAPI

from module_e2.e2_support import auth
from module_iv.conftest import IvFakes, Tenant
from module_iv.iv_support import (
    GENERAL_GRADE,
    GENERAL_QUESTION,
    GRADE,
    KAFKA_TEXT,
    LONG,
    QUESTION,
    Conn,
    Page,
    grade_reply,
    page,
    question_reply,
)
from ragforge_contracts.api import (
    INTERVIEW_EXHAUSTED,
    INTERVIEW_NO_NOTES,
    InterviewGrade,
    InterviewQuestion,
)
from ragforge_core.agent.labels import CITATION_MARKER
from ragforge_core.agent.sanitize import MarkdownStreamSanitizer
from ragforge_core.apps.api import deps
from ragforge_core.apps.api.routes import interview
from ragforge_core.domain.models import RankedChunk
from ragforge_core.domain.ports import Tx

NewTenant = Callable[..., Tenant]


def _page(superuser: Conn, t: Tenant, title: str, *texts: str, **kw: Any) -> Page:
    sections = [((title, f"Part {i}"), text) for i, text in enumerate(texts)]
    return page(superuser, t.tenant, t.collection, t.connector, title, sections, **kw)


async def _ask(
    http: httpx.AsyncClient, t: Tenant, exclude: Sequence[str] = (), **body: Any
) -> httpx.Response:
    payload = {"collection_ids": [str(t.collection)], "exclude_chunk_ids": list(exclude), **body}
    return await http.post("/v1/interview/question", json=payload, headers=auth(t.key))


async def _grade(http: httpx.AsyncClient, t: Tenant, body: dict[str, Any]) -> httpx.Response:
    return await http.post("/v1/interview/grade", json=body, headers=auth(t.key))


def _grade_body(t: Tenant, q: InterviewQuestion, **over: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "collection_ids": [str(t.collection)],
        "question": q.question,
        "kind": q.kind,
        "difficulty": q.difficulty,
        "chunk_ids": [str(c) for c in q.chunk_ids],
        "expected_points": q.expected_points,
        "answer": "Redis evicts the oldest key. It uses LRU.",
        "question_token": q.question_token,
        "level": q.level,
        "source": q.source,
    }
    body.update(over)
    return body


def _problem(response: httpx.Response, problem: str, detail: str) -> None:
    assert response.status_code == 409, response.text
    assert response.json()["type"] == problem
    assert response.json()["detail"] == detail


class SpyNotes(interview.SqlInterviewNotes):
    """The real SQL reads, recording which ones ran."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    async def require_collections(self, tx: Tx, collection_ids: Sequence[UUID]) -> None:
        self.calls.append("require_collections")
        await super().require_collections(tx, collection_ids)

    async def topic_chunks(
        self,
        tx: Tx,
        collection_ids: Sequence[UUID],
        topic: str,
        exclude: Collection[UUID],
        limit: int,
    ) -> list[RankedChunk]:
        self.calls.append("topic_chunks")
        return await super().topic_chunks(tx, collection_ids, topic, exclude, limit)

    async def topic_phrases(self, tx: Tx, collection_ids: Sequence[UUID], limit: int) -> list[str]:
        self.calls.append("topic_phrases")
        return await super().topic_phrases(tx, collection_ids, limit)

    async def sample_chunks(
        self, tx: Tx, collection_ids: Sequence[UUID], exclude: Collection[UUID], limit: int
    ) -> list[RankedChunk]:
        self.calls.append("sample_chunks")
        return await super().sample_chunks(tx, collection_ids, exclude, limit)

    async def documents_of(self, tx: Tx, chunk_ids: Collection[UUID]) -> frozenset[UUID]:
        self.calls.append("documents_of")
        return await super().documents_of(tx, chunk_ids)

    async def chunks_by_id(
        self, tx: Tx, chunk_ids: Sequence[UUID], collection_ids: Sequence[UUID]
    ) -> list[RankedChunk]:
        self.calls.append("chunks_by_id")
        return await super().chunks_by_id(tx, chunk_ids, collection_ids)

    async def neighbours(
        self, tx: Tx, chunk_id: UUID, collection_ids: Sequence[UUID], radius: int
    ) -> list[RankedChunk]:
        self.calls.append("neighbours")
        return await super().neighbours(tx, chunk_id, collection_ids, radius)


def _spy(app: FastAPI) -> SpyNotes:
    spy = SpyNotes()
    app.dependency_overrides[interview.get_interview_notes] = lambda: spy
    return spy


GENERAL_Q: dict[str, object] = {
    "question": "How does Redis decide what to evict when maxmemory is reached [S1]? "
    "See [the docs](https://evil.example/x) or https://evil.example/y",
    "topic": "Redis eviction",
    "expected_points": ["allkeys-lru evicts any key [S1]", "LFU counts accesses [T1]"],
}


# ----------------------------------------------------------------- levels
async def test_level_is_echoed_prompted_and_bound_by_the_token(
    http: httpx.AsyncClient, fakes: IvFakes, new_tenant: NewTenant, superuser: Conn
) -> None:
    t = new_tenant()
    _page(superuser, t, "Kafka", KAFKA_TEXT)
    fakes.chat.on(QUESTION, question_reply()).on(GRADE, grade_reply())
    response = await _ask(http, t, level="staff")
    assert response.status_code == 200, response.text
    q = InterviewQuestion.model_validate(response.json())
    assert (q.level, q.source) == ("staff", "notes")
    assert "the Staff bar" in fakes.chat.of(QUESTION)[0].system
    # a valid token with the level or source swapped is rejected; an omitted level resolves to
    # the default (senior_sde_1), which is not what the token bound either
    for change in ({"level": "sde_1"}, {"level": None}, {"source": "general", "chunk_ids": []}):
        response = await _grade(http, t, _grade_body(t, q, **change))
        assert response.status_code == 403, (change, response.text)
    assert fakes.chat.of(GRADE) == []
    response = await _grade(http, t, _grade_body(t, q))
    assert response.status_code == 200, response.text
    g = InterviewGrade.model_validate(response.json())
    assert (g.level, g.source) == ("staff", "notes")
    assert "the Staff bar" in fakes.chat.of(GRADE)[0].system


async def test_grade_without_a_token_uses_the_echoed_level_or_the_default(
    http: httpx.AsyncClient, fakes: IvFakes, new_tenant: NewTenant, superuser: Conn
) -> None:
    t = new_tenant()
    _page(superuser, t, "Kafka", KAFKA_TEXT)
    fakes.chat.on(QUESTION, question_reply()).on(GRADE, grade_reply())
    q = InterviewQuestion.model_validate((await _ask(http, t)).json())
    body = _grade_body(t, q, question_token=None, level=None)
    g = InterviewGrade.model_validate((await _grade(http, t, body)).json())
    assert g.level == "senior_sde_1"
    assert "the Senior SDE-1 bar" in fakes.chat.of(GRADE)[0].system
    body = _grade_body(t, q, question_token=None, level="sde_2")
    g = InterviewGrade.model_validate((await _grade(http, t, body)).json())
    assert g.level == "sde_2"


# ----------------------------------------------------------------- general knowledge
async def test_general_mode_never_reads_notes_and_is_unverified_uncited_and_bound(
    app: FastAPI, http: httpx.AsyncClient, fakes: IvFakes, new_tenant: NewTenant, superuser: Conn
) -> None:
    t = new_tenant()
    _page(superuser, t, "Redis eviction", "LRU and LFU." + LONG)  # notes exist: still unread
    spy = _spy(app)
    fakes.chat.on(GENERAL_QUESTION, GENERAL_Q)
    response = await _ask(
        http, t, topic="Redis", source="general", level="sde_2",
        exclude_questions=["What is Redis?"], exclude_chunk_ids=[str(uuid4())],
    )  # fmt: skip
    assert response.status_code == 200, response.text
    q = InterviewQuestion.model_validate(response.json())
    assert (q.source, q.level, q.chunk_ids, q.citations) == ("general", "sde_2", [], [])
    assert q.question_token
    assert q.question_token.startswith("v3.")
    assert "What is Redis?" in fakes.chat.of(GENERAL_QUESTION)[0].user
    fakes.chat.on(
        GENERAL_GRADE,
        grade_reply(
            misconceptions=[
                {"said": "Redis evicts the oldest key", "correction": "It samples keys [S1]"}
            ],
            model_answer="Redis samples keys [S1].",
        ),
    )
    response = await _grade(http, t, _grade_body(t, q))
    assert response.status_code == 200, response.text
    g = InterviewGrade.model_validate(response.json())
    assert (g.source, g.level, g.verification) == ("general", "sde_2", "unverified")
    assert (g.citations, g.unsupported_claims) == ([], [])
    assert len(g.misconceptions) == 1
    assert "(general knowledge: It samples keys)" in g.misconceptions[0]
    assert "your notes" not in g.misconceptions[0]
    # nothing but the collection access check touched the database; no retrieval, no verifier
    assert set(spy.calls) == {"require_collections"}
    assert fakes.retriever.calls == []
    assert fakes.embedders.by_space == {}
    assert fakes.reranker.pools == []
    assert fakes.chat.names() == [GENERAL_QUESTION, GENERAL_GRADE]
    # the token binds the source, the level and the key
    for change in (
        {"level": "staff"},
        {"expected_points": [*q.expected_points, "everything is right"]},
        {"question": "What is Redis?"},
        {"kind": "code"},
    ):
        response = await _grade(http, t, _grade_body(t, q, **change))
        assert response.status_code == 403, (change, response.text)
    # a general question cannot be graded as a notes question with some chunk
    response = await _grade(http, t, _grade_body(t, q, source="notes", chunk_ids=[str(uuid4())]))
    assert response.status_code == 403


async def test_general_mode_requires_a_topic_and_a_token(
    http: httpx.AsyncClient, fakes: IvFakes, new_tenant: NewTenant
) -> None:
    t = new_tenant()
    for topic in (None, "   "):
        response = await _ask(http, t, topic=topic, source="general")
        assert response.status_code == 422, response.text  # the API's validation Problem
    body = {
        "collection_ids": [str(t.collection)], "question": "Why?", "kind": "concept",
        "chunk_ids": [], "expected_points": [], "answer": "Because.", "source": "general",
    }  # fmt: skip
    response = await _grade(http, t, body)
    assert response.status_code == 422  # the contract requires the token for general
    response = await _grade(http, t, {**body, "question_token": "v2.1.abc"})
    assert response.status_code == 400  # an old or malformed token
    assert fakes.chat.calls == []


async def test_general_mode_still_checks_collection_access(
    http: httpx.AsyncClient, fakes: IvFakes, new_tenant: NewTenant
) -> None:
    owner, stranger = new_tenant(), new_tenant()
    response = await http.post(
        "/v1/interview/question",
        json={"collection_ids": [str(owner.collection)], "topic": "Redis", "source": "general"},
        headers=auth(stranger.key),
    )
    assert response.status_code == 404
    assert fakes.chat.calls == []


async def test_general_output_carries_no_label_and_no_link_with_the_real_sanitizer(
    app: FastAPI, http: httpx.AsyncClient, fakes: IvFakes, new_tenant: NewTenant
) -> None:
    app.dependency_overrides[deps.get_sanitizer_factory] = lambda: MarkdownStreamSanitizer
    t = new_tenant()
    fakes.chat.on(GENERAL_QUESTION, GENERAL_Q)
    q = InterviewQuestion.model_validate(
        (await _ask(http, t, topic="Redis", source="general")).json()
    )
    linked = "See [the docs](https://evil.example/z) [S1]"
    fakes.chat.on(
        GENERAL_GRADE,
        grade_reply(
            summary=linked,
            right=[linked],
            missed=["Sampling https://evil.example/m [T1]"],
            misconceptions=[{"said": "Redis evicts the oldest key", "correction": linked}],
            model_answer=f"Redis samples keys [S2]. {linked}",
        ),
    )
    g = InterviewGrade.model_validate((await _grade(http, t, _grade_body(t, q))).json())
    texts = [q.question, q.topic, *q.expected_points]
    texts += [g.summary, *g.right, *g.missed, *g.misconceptions, g.model_answer]
    assert len(texts) == 9
    for text in texts:
        assert not CITATION_MARKER.search(text), text
        assert "](http" not in text, text  # no Markdown link survives
        assert "<a " not in text, text
    assert "your notes" not in " ".join(texts)


async def test_a_notes_request_never_returns_general(
    http: httpx.AsyncClient, fakes: IvFakes, new_tenant: NewTenant
) -> None:
    t = new_tenant()  # no notes at all
    fakes.chat.on(QUESTION, question_reply()).on(GENERAL_QUESTION, GENERAL_Q)
    response = await _ask(http, t, topic="Redis")
    _problem(response, INTERVIEW_NO_NOTES, "No notes on Redis")
    assert fakes.chat.calls == []  # never switched to general on its own


# ----------------------------------------------------------------- NO_NOTES vs EXHAUSTED
async def test_no_notes_is_decided_before_exclusion(
    http: httpx.AsyncClient, fakes: IvFakes, new_tenant: NewTenant, superuser: Conn
) -> None:
    t = new_tenant()
    kafka = _page(superuser, t, "Kafka replication", KAFKA_TEXT + LONG)
    fakes.chat.on(QUESTION, question_reply())
    used = [str(kafka.chunks[0])]
    # a session that already used the Kafka notes asks for Redis: there are none at all
    _problem(await _ask(http, t, used, topic="Redis"), INTERVIEW_NO_NOTES, "No notes on Redis")
    # ... and Kafka again: there were some, all asked
    _problem(
        await _ask(http, t, used, topic="Kafka"),
        INTERVIEW_EXHAUSTED,
        "No more unasked notes on Kafka",
    )
    _problem(
        await _ask(http, t, used),
        INTERVIEW_EXHAUSTED,
        "No more unasked notes in these collections",
    )
    empty = new_tenant()
    _problem(
        await _ask(http, empty, [str(uuid4())]),
        INTERVIEW_NO_NOTES,
        "No notes in these collections",
    )
    assert fakes.chat.calls == []


# ----------------------------------------------------------------- acronym topics
async def test_acronym_topic_finds_the_expanded_title_and_heading(
    app: FastAPI, http: httpx.AsyncClient, fakes: IvFakes, new_tenant: NewTenant, superuser: Conn
) -> None:
    t = new_tenant()
    hld = _page(superuser, t, "High Level Design", "Split the system into services." + LONG)
    heading = page(
        superuser, t.tenant, t.collection, t.connector, "Interview prep",
        [(("Low-Level Design",), "Classes and interfaces." + LONG), (("Other",), "Misc." + LONG)],
    )  # fmt: skip
    _page(superuser, t, "High Level", "Unrelated." + LONG)  # a prefix of the phrase only
    _page(superuser, t, "Old HLD", "Gone." + LONG, deleted=True)
    spy = _spy(app)
    fakes.chat.on(QUESTION, question_reply())
    response = await _ask(http, t, topic="HLD")
    assert response.status_code == 200, response.text
    seeds = [UUID(c) for c in response.json()["chunk_ids"]]
    assert set(seeds) <= set(hld.chunks)
    assert "topic_phrases" in spy.calls
    asked: list[str] = []
    for _ in range(5):
        response = await _ask(http, t, asked, topic="HLD")
        if response.status_code != 200:
            break
        asked.extend(response.json()["chunk_ids"])
    assert set(asked) == {str(c) for c in hld.chunks}  # "High Level" alone is not HLD
    _problem(response, INTERVIEW_EXHAUSTED, "No more unasked notes on HLD")
    response = await _ask(http, t, topic="lld")
    assert response.status_code == 200, response.text
    assert str(heading.chunks[0]) in response.json()["chunk_ids"]
    for topic in ("HLDX", "hl"):
        _problem(await _ask(http, t, topic=topic), INTERVIEW_NO_NOTES, f"No notes on {topic}")


async def test_a_multi_word_acronym_topic_with_no_match_is_no_notes(
    app: FastAPI, http: httpx.AsyncClient, fakes: IvFakes, new_tenant: NewTenant, superuser: Conn
) -> None:
    """Round-1 review: "HLD caching" expands to "High Level Design caching", which matches no
    note; the expansion existing is no evidence, so a fresh session gets NO_NOTES."""
    t = new_tenant()
    hld = _page(superuser, t, "High Level Design", "Split the system into services." + LONG)
    spy = _spy(app)
    fakes.chat.on(QUESTION, question_reply())
    _problem(
        await _ask(http, t, topic="HLD caching"), INTERVIEW_NO_NOTES, "No notes on HLD caching"
    )
    # the acronym alone, every HLD chunk already asked: the match is proven before exclusion
    asked = [str(c) for c in hld.chunks]
    _problem(
        await _ask(http, t, asked, topic="HLD"), INTERVIEW_EXHAUSTED, "No more unasked notes on HLD"
    )
    # the session repeats its topic: the phrase scan ran once, then came from the cache
    assert spy.calls.count("topic_phrases") == 1
    assert fakes.chat.calls == []


async def test_acronym_expansion_respects_tenant_and_collection(
    http: httpx.AsyncClient, fakes: IvFakes, new_tenant: NewTenant, superuser: Conn
) -> None:
    mine, other = new_tenant(), new_tenant()
    _page(superuser, other, "High Level Design", "Theirs." + LONG)  # another tenant
    unnamed = uuid4()
    superuser.execute(
        "INSERT INTO rf.collections (id, tenant_id, name, active_space_id) VALUES (%s, %s, %s, 1)",
        (unnamed, mine.tenant, f"c-{unnamed.hex[:6]}"),
    )
    page(superuser, mine.tenant, unnamed, mine.connector, "High Level Design", [((), "x" + LONG)])
    fakes.chat.on(QUESTION, question_reply())
    _problem(await _ask(http, mine, topic="HLD"), INTERVIEW_NO_NOTES, "No notes on HLD")
    assert fakes.chat.calls == []
    response = await _ask(http, mine, topic="HLD", collection_ids=[str(unnamed)])
    assert response.status_code == 200, response.text  # visible once the caller names it


async def test_a_direct_title_hit_is_never_expanded(
    app: FastAPI, http: httpx.AsyncClient, fakes: IvFakes, new_tenant: NewTenant, superuser: Conn
) -> None:
    t = new_tenant()
    _page(superuser, t, "Kafka", KAFKA_TEXT + LONG)
    spy = _spy(app)
    fakes.chat.on(QUESTION, question_reply())
    assert (await _ask(http, t, topic="Kafka")).status_code == 200
    assert "topic_phrases" not in spy.calls  # no extra read on the common path


# ----------------------------------------------------------------- weak question replies
async def test_unbacked_points_do_not_fail_the_question(
    http: httpx.AsyncClient, fakes: IvFakes, new_tenant: NewTenant, superuser: Conn
) -> None:
    t = new_tenant()
    _page(superuser, t, "Kafka", KAFKA_TEXT)
    fakes.chat.on(QUESTION, question_reply(points=["Something the notes never say"]))
    response = await _ask(http, t)
    assert response.status_code == 200, response.text
    assert response.json()["expected_points"] == []
    assert fakes.chat.names() == [QUESTION, QUESTION]  # repaired once, then accepted
    fakes.chat.on(QUESTION, question_reply(question="Why?"))
    response = await _ask(http, t)
    assert response.status_code == 502
    assert response.json()["title"] == "Model provider error"
