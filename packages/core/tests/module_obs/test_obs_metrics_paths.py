"""The catalogue metrics whose source is a ROUTE or an agent STAGE move on the real code path
(ADR-0018 R2-1), not only when their helper is called directly:

- `ragforge_interview_requests_total{endpoint, source, outcome}`: the real interview routes on
  the real test database (the module IV app and fakes), one increment per call, per outcome;
- `ragforge_retrieval_duration_seconds{stage}`: one agent turn (the D1 harness: production graph
  and nodes) times embed, search and rerank once each;
- `ragforge_retrieval_orphan_hits_total`: the production `HybridRetriever` over a fake backend
  whose active spaces do not serve the hit's collection counts every such hit, and no other.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any
from uuid import UUID

import httpx
import pytest
from langgraph.checkpoint.memory import InMemorySaver

from module_d1.d1_fakes import (
    COLLECTION,
    COLLECTION_B,
    FakeReranker,
    collect,
    make_harness,
    make_hit,
    principal,
    say,
    verdict,
)
from module_e2.e2_support import auth
from module_iv.conftest import (  # noqa: F401 - pytest fixtures, used by name
    IvFakes,
    Tenant,
    app,
    fakes,
    http,
    iv_app,
    new_tenant,
)
from module_iv.iv_support import (
    GRADE,
    KAFKA_TEXT,
    QUESTION,
    Conn,
    grade_reply,
    page,
    question_reply,
)
from module_obs.obs_metrics_support import Delta
from ragforge_core.agent import AgentDeps, AgentSettings, LangGraphAgentRunner
from ragforge_core.domain.models import RankedChunk
from ragforge_core.rag.retriever import HybridRetriever

INTERVIEW = "ragforge_interview_requests_total"
RETRIEVAL = "ragforge_retrieval_duration_seconds"
ORPHANS = "ragforge_retrieval_orphan_hits_total"
NewTenant = Callable[..., Tenant]


# ---------------------------------------------------------------------- interview routes
def _outcomes(delta: Delta) -> dict[tuple[str, str, str], float]:
    """{(endpoint, source, outcome): increase} of the interview counter."""
    out: dict[tuple[str, str, str], float] = {}
    for (name, labels), increase in delta.changes().items():
        if name == INTERVIEW:
            got = dict(labels)
            out[(got["endpoint"], got["source"], got["outcome"])] = increase
    return out


async def _ask(client: httpx.AsyncClient, t: Tenant, **body: Any) -> httpx.Response:
    payload = {"collection_ids": [str(t.collection)], **body}
    return await client.post("/v1/interview/question", json=payload, headers=auth(t.key))


def _grade_body(t: Tenant, q: dict[str, Any], **over: Any) -> dict[str, Any]:
    body = {
        "collection_ids": [str(t.collection)],
        "question": q["question"],
        "kind": q["kind"],
        "difficulty": q["difficulty"],
        "chunk_ids": q["chunk_ids"],
        "expected_points": q["expected_points"],
        "answer": "ISR replicas lead.",
        "question_token": q["question_token"],
    }
    body.update(over)
    return body


@pytest.mark.integration
async def test_each_interview_question_outcome_is_counted_once(
    http: httpx.AsyncClient,  # noqa: F811 - the imported fixture, by name
    fakes: IvFakes,  # noqa: F811
    new_tenant: NewTenant,  # noqa: F811
    superuser: Conn,
) -> None:
    t = new_tenant()
    fakes.chat.on(QUESTION, question_reply())

    delta = Delta((INTERVIEW,))
    assert (await _ask(http, t, topic="kubernetes")).status_code == 409  # nothing at all
    assert _outcomes(delta) == {("question", "notes", "no_notes"): 1}

    sections = [(("Kafka", "Replication"), KAFKA_TEXT)]
    page(superuser, t.tenant, t.collection, t.connector, "Kafka", sections)
    delta = Delta((INTERVIEW,))
    first = await _ask(http, t)
    assert first.status_code == 200, first.text
    assert _outcomes(delta) == {("question", "notes", "ok"): 1}

    delta = Delta((INTERVIEW,))
    used = first.json()["chunk_ids"]
    assert (await _ask(http, t, exclude_chunk_ids=used)).status_code == 409  # all asked
    assert _outcomes(delta) == {("question", "notes", "exhausted"): 1}


@pytest.mark.integration
async def test_each_interview_grade_outcome_is_counted_once(
    http: httpx.AsyncClient,  # noqa: F811
    fakes: IvFakes,  # noqa: F811
    new_tenant: NewTenant,  # noqa: F811
    superuser: Conn,
) -> None:
    t = new_tenant()
    sections = [(("Kafka", "Replication"), KAFKA_TEXT)]
    page(superuser, t.tenant, t.collection, t.connector, "Kafka", sections)
    fakes.chat.on(QUESTION, question_reply())
    question = (await _ask(http, t)).json()
    grade = "/v1/interview/grade"

    fakes.chat.on(GRADE, grade_reply())
    delta = Delta((INTERVIEW,))
    response = await http.post(grade, json=_grade_body(t, question), headers=auth(t.key))
    assert response.status_code == 200, response.text
    assert _outcomes(delta) == {("grade", "notes", "ok"): 1}

    delta = Delta((INTERVIEW,))
    forged = _grade_body(t, question, question_token="v1.123.not-a-mac")
    assert (await http.post(grade, json=forged, headers=auth(t.key))).status_code == 400
    assert _outcomes(delta) == {("grade", "notes", "rejected"): 1}  # a 4xx problem

    fakes.chat.on(GRADE, grade_reply(score=11))  # out of range: a model failure
    delta = Delta((INTERVIEW,))
    response = await http.post(grade, json=_grade_body(t, question), headers=auth(t.key))
    assert response.status_code == 502
    assert _outcomes(delta) == {("grade", "notes", "error"): 1}


# ---------------------------------------------------------------------- retrieval stages
def _stage_counts(delta: Delta) -> dict[str, float]:
    return {
        dict(labels)["stage"]: increase
        for (name, labels), increase in delta.changes().items()
        if name == f"{RETRIEVAL}_count"
    }


async def test_one_agent_turn_times_each_retrieval_stage_once() -> None:
    h = make_harness([say("Fact one holds [S1]."), verdict()])
    delta = Delta((RETRIEVAL,))
    await collect(h.runner.run_turn(principal(), None, [COLLECTION], "what is fact one?"))
    assert _stage_counts(delta) == {"embed": 1.0, "search": 1.0, "rerank": 1.0}


# ---------------------------------------------------------------------- orphan hits
class _Backend:
    """A SearchBackend whose ACTIVE spaces serve `served`; every hit comes from keyword search
    (the only list that can return a chunk no active space covers)."""

    def __init__(self, served: Sequence[UUID], hits: Sequence[RankedChunk]) -> None:
        self.served = list(served)
        self.hits = list(hits)

    async def active_spaces(
        self, tx: object, collection_ids: Sequence[UUID]
    ) -> Mapping[int, Sequence[UUID]]:
        return {1: self.served}

    async def vector_search(
        self,
        tx: object,
        space_id: int,
        query_vector: Sequence[float],
        collection_ids: Sequence[UUID],
        k: int,
    ) -> Sequence[RankedChunk]:
        return []

    async def keyword_search(
        self, tx: object, query: str, collection_ids: Sequence[UUID], k: int
    ) -> Sequence[RankedChunk]:
        return self.hits


async def _orphans_of_one_turn(served: Sequence[UUID]) -> float:
    hits = [make_hit(1), make_hit(2)]
    h = make_harness([say("Fact one holds [S1]."), verdict()])
    deps = AgentDeps(
        chat=h.chat,
        retriever=HybridRetriever(_Backend(served, hits)),
        reranker=FakeReranker(h.txm),
        embedders=h.embedders,
        runs=h.runs,
        tx=h.txm,
        jobs=h.jobs,
        conversations=h.conversations,
        knowledge=h.knowledge,
        store=h.store,
        settings=AgentSettings(heartbeat_interval_s=3600),
    )
    runner = LangGraphAgentRunner(deps, InMemorySaver())
    delta = Delta((ORPHANS,))
    await collect(runner.run_turn(principal(), None, [COLLECTION], "what is fact one?"))
    return delta.of(ORPHANS)


async def test_hits_of_a_collection_no_active_space_serves_are_counted() -> None:
    assert await _orphans_of_one_turn([COLLECTION_B]) == 2  # both hits are in COLLECTION


async def test_hits_of_a_served_collection_are_not_orphans() -> None:
    assert await _orphans_of_one_turn([COLLECTION]) == 0
