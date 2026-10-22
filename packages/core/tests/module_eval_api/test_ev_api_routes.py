"""The eval seams on the real app and the real test database (RLS, SQL, scopes, the switch), with
fake models: /v1/eval/verify|chunks|resolve, /v1/query stages, chat ?trace=shown, interview
chunk_ids and the eval-scope frozen grade, and the eval key rules at key creation."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Sequence
from typing import Any
from uuid import UUID, uuid4

import httpx
import pytest
from fastapi import FastAPI

from module_e2.e2_support import auth, issue_key, make_hit
from module_eval_api.conftest import EvFakes, Tenant
from module_eval_api.ev_api_support import (
    Conn,
    DbPage,
    d0_item,
    d0_note,
    file_page,
    mutated_text,
    notion_page,
    objects,
    sha_hex,
)
from module_iv.iv_support import (
    GRADE,
    KAFKA_TEXT,
    LONG,
    QUESTION,
    VERIFY,
    grade_reply,
    page,
    question_reply,
)
from ragforge_contracts.api import (
    EvalChunksResponse,
    EvalResolveResponse,
    EvalVerifyResponse,
    InterviewGrade,
    InterviewQuestion,
    QueryResponse,
)
from ragforge_contracts.sse import DoneEvent, decode_sse_data
from ragforge_core.apps.api import deps
from ragforge_core.domain.models import RankedChunk, TurnDone
from ragforge_core.domain.ports import Tx
from ragforge_core.prompts import PROMPT_VERSION, PROMPT_VERSIONS
from ragforge_core.rag.retriever import HybridRetriever

NewTenant = Callable[..., Tenant]
Json = dict[str, Any]


def _d0_page(superuser: Conn, t: Tenant, item_id: str) -> tuple[DbPage, str]:
    """The D0 item's (single) shown chunk as an uploaded file page of tenant `t`."""
    item = d0_item(item_id)
    (key,) = item.shown
    note = d0_note(key)
    db = file_page(
        superuser, t.tenant, t.collection, key.page.key, [(key.heading_path, note.display_text)]
    )
    return db, item.answer


async def _verify(http: httpx.AsyncClient, key: str, body: Json) -> httpx.Response:
    return await http.post("/v1/eval/verify", json=body, headers=auth(key))


def _verify_body(t: Tenant, answer: str, shown: Sequence[UUID]) -> Json:
    return {
        "answer": answer,
        "shown_chunk_ids": [str(c) for c in shown],
        "collection_ids": [str(t.collection)],
    }


# ----------------------------------------------------------------- the switch and the scope
EVAL_CALLS: list[tuple[str, str, Json | None]] = [
    ("GET", f"/v1/eval/chunks?ids={uuid4()}", None),
    (
        "POST",
        "/v1/eval/verify",
        {"answer": "a", "shown_chunk_ids": [str(uuid4())], "collection_ids": [str(uuid4())]},
    ),
    ("POST", "/v1/eval/resolve", {"collection_ids": [str(uuid4())]}),
]


@pytest.mark.parametrize(("method", "url", "body"), EVAL_CALLS)
async def test_the_eval_surface_is_404_while_switched_off_even_unauthenticated(
    http: httpx.AsyncClient, fakes: EvFakes, new_tenant: NewTenant, method: str, url: str,
    body: Json | None,
) -> None:  # fmt: skip
    fakes.flags.eval_api = False
    t = new_tenant()
    for headers in ({}, auth(t.eval_key)):
        response = await http.request(method, url, json=body, headers=headers)
        assert response.status_code == 404, response.text


@pytest.mark.parametrize(("method", "url", "body"), EVAL_CALLS)
async def test_the_eval_surface_needs_the_eval_scope(
    http: httpx.AsyncClient, new_tenant: NewTenant, method: str, url: str, body: Json | None
) -> None:
    t = new_tenant(("read", "write", "chat", "approvals:write", "admin"))
    assert (await http.request(method, url, json=body)).status_code == 401
    response = await http.request(method, url, json=body, headers=auth(t.key))
    assert response.status_code == 403, response.text
    assert "eval" in response.json()["detail"]


async def test_the_additive_seams_are_403_without_scope_or_switch(
    http: httpx.AsyncClient, fakes: EvFakes, new_tenant: NewTenant, superuser: Conn
) -> None:
    t = new_tenant()
    colls = [str(t.collection)]
    calls = [
        ("/v1/query", {"query": "q", "collection_ids": colls, "stages": True}),
        ("/v1/chat?trace=shown", {"collection_ids": colls, "message": "hi"}),
        ("/v1/interview/question", {"collection_ids": colls, "chunk_ids": [str(uuid4())]}),
    ]
    for url, body in calls:
        response = await http.post(url, json=body, headers=auth(t.key))
        assert response.status_code == 403, (url, response.text)
    fakes.flags.eval_api = False
    for url, body in calls:
        response = await http.post(url, json=body, headers=auth(t.eval_key))
        assert response.status_code == 403, (url, response.text)
        assert "RAGFORGE_EVAL_API" in response.json()["detail"]
    assert not fakes.agent.called
    assert not fakes.chat.calls


class _NoQuota:
    """An interview limiter that fails the test if any quota is taken."""

    def acquire(self, key: tuple[UUID, UUID]) -> None:
        raise AssertionError("a refused eval seam spent interview quota")

    def release(self, key: tuple[UUID, UUID]) -> None:
        raise AssertionError("released a slot that was never taken")


async def test_a_refused_interview_seam_spends_no_quota(
    http: httpx.AsyncClient, app: FastAPI, new_tenant: NewTenant
) -> None:
    """Review round 1, P2: `chunk_ids` is refused (403 scope, 422 general mode) BEFORE the
    interview limiter is touched, like `eval_quota`."""
    app.dependency_overrides[deps.get_interview_limiter] = _NoQuota
    t = new_tenant()
    colls = [str(t.collection)]
    targeted = {"collection_ids": colls, "chunk_ids": [str(uuid4())]}
    refused = await http.post("/v1/interview/question", json=targeted, headers=auth(t.key))
    assert refused.status_code == 403, refused.text
    general = {**targeted, "source": "general", "topic": "kafka"}
    invalid = await http.post("/v1/interview/question", json=general, headers=auth(t.eval_key))
    assert invalid.status_code == 422, invalid.text


# ----------------------------------------------------------------- /v1/eval/verify
async def test_verify_runs_the_pipeline_on_rls_read_notes(
    http: httpx.AsyncClient, fakes: EvFakes, new_tenant: NewTenant, superuser: Conn
) -> None:
    t = new_tenant()
    db, answer = _d0_page(superuser, t, "c-durability")
    response = await _verify(http, t.eval_key, _verify_body(t, answer, db.chunks))
    assert response.status_code == 200, response.text
    out = EvalVerifyResponse.model_validate(response.json())
    assert out.status == "verified"
    assert out.verifier_ran
    assert out.verifier_conclusive
    assert out.claims == []
    assert out.coverage_units == []
    assert out.verifier_claims == []
    assert out.verifier_model == "fake-chat"
    assert out.verifier_prompt == "ground_check@3"
    (call,) = fakes.chat.of(VERIFY)
    assert "acks=all" in call.user  # the verifier read the RLS-read note text


async def test_verify_measures_an_untrusted_verifier_by_its_own_verdict(
    http: httpx.AsyncClient, app: FastAPI, new_tenant: NewTenant, superuser: Conn
) -> None:
    """Review round 1, P1: with RAGFORGE_AGENT_VERIFIER_TRUSTED=false the chat badge is
    downgraded, but E-V is what decides that flag, so a clean item must still come back
    `verified` (else badge-flip recall has no clean set and no model could ever qualify)."""
    container = app.state.container
    container.settings = container.settings.model_copy(update={"agent_verifier_trusted": False})
    t = new_tenant()
    db, answer = _d0_page(superuser, t, "c-durability")
    response = await _verify(http, t.eval_key, _verify_body(t, answer, db.chunks))
    assert response.status_code == 200, response.text
    out = EvalVerifyResponse.model_validate(response.json())
    assert (out.status, out.verifier_verdict, out.claims) == ("verified", "verified", [])


async def test_verify_returns_pre_merge_evidence_with_spans(
    http: httpx.AsyncClient, fakes: EvFakes, new_tenant: NewTenant, superuser: Conn
) -> None:
    t = new_tenant()
    item = d0_item("p-fabricated-1")
    assert item.span is not None
    db, answer = _d0_page(superuser, t, "p-fabricated-1")
    quoted = answer.split(" [S1]")[0]  # the verifier objects to the first sentence too
    fakes.chat.on(VERIFY, objects(quoted, "a paraphrase nobody can place"))
    response = await _verify(http, t.eval_key, _verify_body(t, answer, db.chunks))
    out = EvalVerifyResponse.model_validate(response.json())
    assert out.status == "partial"
    unit = next(u for u in out.coverage_units if u.text == mutated_text(item))
    assert (unit.start, unit.end, unit.kind) == (item.span.start, item.span.end, "uncited")
    spans = {c.text: (c.start, c.end) for c in out.verifier_claims}
    assert spans["a paraphrase nobody can place"] == (None, None)
    assert spans[quoted] == (0, len(quoted))
    assert all(answer[c.start : c.end] == c.text for c in out.claims)


async def test_verify_404s_for_notes_outside_the_collections_or_tenant(
    http: httpx.AsyncClient, new_tenant: NewTenant, superuser: Conn, make_tenant: Any
) -> None:
    t, other = new_tenant(), new_tenant()
    mine, answer = _d0_page(superuser, t, "c-durability")
    theirs, _ = _d0_page(superuser, other, "c-durability")
    _, _, second_collection = make_tenant()  # a collection of yet another tenant
    gone = file_page(superuser, t.tenant, t.collection, "old.md", [((), "x")], deleted=True)
    for shown, colls in (
        (theirs.chunks, [t.collection]),  # another tenant's chunk: invisible
        (mine.chunks, [other.collection]),  # another tenant's collection
        (mine.chunks, [second_collection]),
        ([*mine.chunks, *gone.chunks], [t.collection]),  # a tombstoned page is never verified
    ):
        body = {"answer": answer, "shown_chunk_ids": [str(c) for c in shown],
                "collection_ids": [str(c) for c in colls]}  # fmt: skip
        assert (await _verify(http, t.eval_key, body)).status_code == 404


async def test_verify_rejects_a_chunk_shown_twice(
    http: httpx.AsyncClient, new_tenant: NewTenant, superuser: Conn
) -> None:
    t = new_tenant()
    db, answer = _d0_page(superuser, t, "c-durability")
    response = await _verify(http, t.eval_key, _verify_body(t, answer, db.chunks * 2))
    assert response.status_code == 422


# ----------------------------------------------------------------- /v1/eval/chunks
async def test_chunks_return_text_keys_and_digests_in_request_order(
    http: httpx.AsyncClient, new_tenant: NewTenant, superuser: Conn
) -> None:
    t, other = new_tenant(), new_tenant()
    f = file_page(superuser, t.tenant, t.collection, "kafka.md", [(("Kafka",), "a"), ((), "b")])
    n = notion_page(superuser, t.tenant, t.collection, t.connector, "notion-page-1", [((), "c")])
    legacy = file_page(superuser, t.tenant, t.collection, "x.md", [((), "d")], with_filename=False)
    gone = file_page(superuser, t.tenant, t.collection, "gone.md", [((), "e")], deleted=True)
    foreign = file_page(superuser, other.tenant, other.collection, "kafka.md", [((), "f")])
    ids = [n.chunks[0], f.chunks[1], foreign.chunks[0], f.chunks[0], gone.chunks[0],
           legacy.chunks[0], uuid4()]  # fmt: skip
    response = await http.get(
        "/v1/eval/chunks", params={"ids": ",".join(map(str, ids))}, headers=auth(t.eval_key)
    )
    assert response.status_code == 200, response.text
    chunks = EvalChunksResponse.model_validate(response.json()).chunks
    assert [c.chunk_id for c in chunks] == [n.chunks[0], f.chunks[1], f.chunks[0], legacy.chunks[0]]
    by_id = {c.chunk_id: c for c in chunks}
    first = by_id[f.chunks[0]]
    assert (first.page_key.source_kind, first.page_key.key) == ("file", "kafka.md")
    assert first.heading_path == ["Kafka"]
    assert first.ordinal == 0
    assert first.display_text == "a"
    assert first.display_sha256 == sha_hex("a")
    assert by_id[n.chunks[0]].page_key.model_dump() == {
        "source_kind": "notion",
        "key": "notion-page-1",
    }
    legacy_key = by_id[legacy.chunks[0]].page_key
    assert legacy_key.source_kind == "file"
    assert UUID(legacy_key.key)


async def test_chunks_reject_malformed_ids(http: httpx.AsyncClient, new_tenant: NewTenant) -> None:
    t = new_tenant()
    for ids in ("nope", ""):
        response = await http.get("/v1/eval/chunks", params={"ids": ids}, headers=auth(t.eval_key))
        assert response.status_code == 422


# ----------------------------------------------------------------- /v1/eval/resolve
def _chunk_key(
    source: str, key: str, heading: Sequence[str], text: str, occurrence: int = 0
) -> Json:
    return {"page": {"source_kind": source, "key": key}, "heading_path": list(heading),
            "display_sha256": sha_hex(text), "occurrence": occurrence}  # fmt: skip


async def test_resolve_keys_to_ids_stale_and_missing(
    http: httpx.AsyncClient, new_tenant: NewTenant, superuser: Conn
) -> None:
    t, other = new_tenant(), new_tenant()
    f = file_page(
        superuser, t.tenant, t.collection, "kafka.md",
        [(("K",), "same"), (("K",), "edited"), (("K",), "same"), (("Other",), "o")],
    )  # fmt: skip
    n = notion_page(superuser, t.tenant, t.collection, t.connector, "np-1", [((), "n")])
    file_page(superuser, other.tenant, other.collection, "theirs.md", [((), "t")])
    body = {
        "collection_ids": [str(t.collection)],
        "page_keys": [
            {"source_kind": "file", "key": "kafka.md"},
            {"source_kind": "notion", "key": "np-1"},
            {"source_kind": "notion", "key": "kafka.md"},  # right key, wrong kind: missing
            {"source_kind": "file", "key": "theirs.md"},  # another tenant's: missing
        ],
        "chunk_keys": [
            _chunk_key("file", "kafka.md", ["K"], "same", 1),
            _chunk_key("file", "kafka.md", ["K"], "original"),  # edited since: stale
            _chunk_key("notion", "np-1", [], "n"),
            _chunk_key("file", "nope.md", ["K"], "same"),
        ],
    }
    response = await http.post("/v1/eval/resolve", json=body, headers=auth(t.eval_key))
    assert response.status_code == 200, response.text
    out = EvalResolveResponse.model_validate(response.json())
    assert [(p.status, p.document_id) for p in out.pages] == [
        ("ok", f.document),
        ("ok", n.document),
        ("missing", None),
        ("missing", None),
    ]
    ok, stale, notion, missing = out.chunks
    assert (ok.status, ok.chunk_id) == ("ok", f.chunks[2])  # the second "same" under K
    assert (stale.status, stale.chunk_id, stale.candidates) == ("stale", None, f.chunks[:3])
    assert (notion.status, notion.chunk_id) == ("ok", n.chunks[0])
    assert (missing.status, missing.candidates) == ("missing", [])


async def test_resolve_never_picks_one_of_two_live_uploads_of_a_filename(
    http: httpx.AsyncClient, new_tenant: NewTenant, superuser: Conn
) -> None:
    """Principle 6: a page key naming two live documents is `ambiguous` (the client then marks
    the item stale), never silently the newest; a tombstoned upload is not a third candidate."""
    t = new_tenant()
    older = file_page(superuser, t.tenant, t.collection, "a.md", [((), "v1")])
    newest = file_page(superuser, t.tenant, t.collection, "a.md", [((), "v2")])
    file_page(superuser, t.tenant, t.collection, "a.md", [((), "v3")], deleted=True)
    single = file_page(superuser, t.tenant, t.collection, "b.md", [((), "b")])
    file_page(superuser, t.tenant, t.collection, "b.md", [((), "b-old")], deleted=True)
    superuser.execute(
        "UPDATE rf.documents SET created_at = now() + interval '1 minute' WHERE id = %s",
        (newest.document,),
    )
    body = {
        "collection_ids": [str(t.collection)],
        "page_keys": [{"source_kind": "file", "key": k} for k in ("a.md", "b.md")],
        "chunk_keys": [_chunk_key("file", "a.md", [], "v2"), _chunk_key("file", "b.md", [], "b")],
    }
    response = await http.post("/v1/eval/resolve", json=body, headers=auth(t.eval_key))
    assert response.status_code == 200, response.text
    out = EvalResolveResponse.model_validate(response.json())
    assert [(p.status, p.document_id) for p in out.pages] == [
        ("ambiguous", None),
        ("ok", single.document),
    ]
    ambiguous, ok = out.chunks
    assert (ambiguous.status, ambiguous.chunk_id) == ("ambiguous", None)
    assert ambiguous.candidates == [newest.chunks[0], older.chunks[0]]  # newest document first
    assert (ok.status, ok.chunk_id) == ("ok", single.chunks[0])


async def test_resolve_404s_for_a_foreign_collection(
    http: httpx.AsyncClient, new_tenant: NewTenant
) -> None:
    t, other = new_tenant(), new_tenant()
    body = {"collection_ids": [str(other.collection)]}
    response = await http.post("/v1/eval/resolve", json=body, headers=auth(t.eval_key))
    assert response.status_code == 404


# ----------------------------------------------------------------- /v1/query stages
class _Backend:
    def __init__(self, collection: UUID) -> None:
        self.vector = [make_hit(collection, n=i) for i in range(3)]
        # a three-way score tie: the stage's order must come from chunk_id, not from this list
        self.keyword = [_scored(make_hit(collection, n=10 + i), 0.5) for i in range(3)]
        self.ks: list[int] = []  # the per-list k every search was asked for

    async def active_spaces(self, tx: Tx, c: Sequence[UUID]) -> dict[int, list[UUID]]:
        return {1: list(c)}

    async def vector_search(
        self, tx: Tx, s: int, v: Sequence[float], c: Sequence[UUID], k: int
    ) -> list[RankedChunk]:
        self.ks.append(k)
        return self.vector

    async def keyword_search(self, tx: Tx, q: str, c: Sequence[UUID], k: int) -> list[RankedChunk]:
        self.ks.append(k)
        return self.keyword


def _scored(hit: RankedChunk, score: float) -> RankedChunk:
    return dataclasses.replace(hit, fused_score=score)


async def test_query_stages_for_the_eval_scope_only_and_old_clients_unchanged(
    http: httpx.AsyncClient, fakes: EvFakes, new_tenant: NewTenant
) -> None:
    t = new_tenant()
    backend = _Backend(t.collection)
    fakes.hybrid = HybridRetriever(backend, per_list_k=7, rrf_k=11)  # NOT the defaults
    body = {"query": "kafka", "collection_ids": [str(t.collection)], "k": 3}
    plain = await http.post("/v1/query", json=body, headers=auth(t.key))
    assert plain.status_code == 200
    assert plain.json()["stages"] is None
    traced = await http.post("/v1/query", json={**body, "stages": True}, headers=auth(t.eval_key))
    assert traced.status_code == 200, traced.text
    # the A/A precondition: the recorded pipeline IS the wired one (same backend, same k's)
    assert plain.json()["hits"]
    assert traced.json()["hits"] == plain.json()["hits"]
    assert set(backend.ks) == {7}
    stages = QueryResponse.model_validate(traced.json()).stages
    assert stages is not None
    for stage in (stages.keyword, stages.vector, stages.fused, stages.reranked):
        assert [h.rank for h in stage] == list(range(1, len(stage) + 1))
    assert [h.chunk_id for h in stages.keyword] == sorted(
        (h.chunk_id for h in backend.keyword), key=str
    )
    assert {h.chunk_id for h in stages.fused} == {
        h.chunk_id for h in [*backend.vector, *backend.keyword]
    }
    again = await http.post("/v1/query", json={**body, "stages": True}, headers=auth(t.eval_key))
    assert again.json()["stages"] == traced.json()["stages"]  # deterministic


# ----------------------------------------------------------------- chat: done.shown + versions
def _done(response: httpx.Response) -> DoneEvent:
    events = [
        decode_sse_data(line.removeprefix("data: "))
        for line in response.text.splitlines()
        if line.startswith("data: ")
    ]
    (done,) = [e for e in events if isinstance(e, DoneEvent)]
    return done


async def test_done_has_prompt_versions_for_everyone_and_shown_only_when_traced(
    http: httpx.AsyncClient, fakes: EvFakes, new_tenant: NewTenant
) -> None:
    t = new_tenant()
    fakes.agent.script = [
        TurnDone(conversation_id=uuid4(), usage=(), prompt_version=PROMPT_VERSION)
    ]
    body = {"collection_ids": [str(t.collection)], "message": "hi"}
    plain = _done(await http.post("/v1/chat", json=body, headers=auth(t.key)))
    assert plain.shown is None
    assert plain.prompt_versions == PROMPT_VERSIONS
    assert plain.prompt_version == PROMPT_VERSION  # the old field stays
    untraced = _done(await http.post("/v1/chat", json=body, headers=auth(t.eval_key)))
    assert untraced.shown is None  # the eval scope alone does not add it: ?trace=shown does
    traced = _done(await http.post("/v1/chat?trace=shown", json=body, headers=auth(t.eval_key)))
    assert traced.shown == []  # requested; this fake agent records nothing (never invented)


async def test_an_unknown_trace_is_rejected(http: httpx.AsyncClient, new_tenant: NewTenant) -> None:
    t = new_tenant()
    body = {"collection_ids": [str(t.collection)], "message": "hi"}
    response = await http.post("/v1/chat?trace=text", json=body, headers=auth(t.eval_key))
    assert response.status_code == 422


# ----------------------------------------------------------------- interview
def _kafka_page(superuser: Conn, t: Tenant) -> list[UUID]:
    sections = [(("Kafka", f"Part {i}"), f"{KAFKA_TEXT} {i}.{LONG}") for i in range(4)]
    return page(superuser, t.tenant, t.collection, t.connector, "Kafka", sections).chunks


async def test_an_eval_key_targets_the_seed_note(
    http: httpx.AsyncClient, fakes: EvFakes, new_tenant: NewTenant, superuser: Conn
) -> None:
    t = new_tenant()
    chunks = _kafka_page(superuser, t)
    fakes.chat.on(QUESTION, question_reply())
    body = {"collection_ids": [str(t.collection)], "chunk_ids": [str(chunks[3])]}
    response = await http.post("/v1/interview/question", json=body, headers=auth(t.eval_key))
    assert response.status_code == 200, response.text
    question = InterviewQuestion.model_validate(response.json())
    assert question.chunk_ids[0] == chunks[3]  # the seed is S1
    assert set(question.chunk_ids) <= set(chunks)
    assert question.question_token
    (call,) = fakes.chat.of(QUESTION)
    assert f"{KAFKA_TEXT} 3." in call.user  # the question was written from the seed


async def test_targeting_needs_visible_notes_and_notes_mode(
    http: httpx.AsyncClient, fakes: EvFakes, new_tenant: NewTenant, superuser: Conn
) -> None:
    t, other = new_tenant(), new_tenant()
    foreign = _kafka_page(superuser, other)
    body = {"collection_ids": [str(t.collection)], "chunk_ids": [str(foreign[0])]}
    response = await http.post("/v1/interview/question", json=body, headers=auth(t.eval_key))
    assert response.status_code == 404
    general = {**body, "source": "general", "topic": "kafka"}
    response = await http.post("/v1/interview/question", json=general, headers=auth(t.eval_key))
    assert response.status_code == 422
    assert not fakes.chat.calls


def _grade_body(t: Tenant, chunks: Sequence[UUID], **extra: Any) -> Json:
    return {
        "collection_ids": [str(t.collection)],
        "question": "Why can only in-sync replicas become leader?",
        "kind": "concept",
        "chunk_ids": [str(c) for c in chunks],
        "expected_points": ["Only in-sync replicas can become leader [S1]"],
        "answer": "Because the ISR has every acknowledged write.",
        **extra,
    }


async def test_an_eval_scope_notes_grade_needs_no_token_at_any_level(
    http: httpx.AsyncClient, fakes: EvFakes, new_tenant: NewTenant, superuser: Conn
) -> None:
    t = new_tenant()
    chunks = _kafka_page(superuser, t)[:1]
    fakes.chat.on(GRADE, grade_reply())
    for level in ("sde_1", "staff"):
        body = _grade_body(t, chunks, level=level)
        response = await http.post("/v1/interview/grade", json=body, headers=auth(t.eval_key))
        assert response.status_code == 200, response.text
        assert InterviewGrade.model_validate(response.json()).level == level
    # general mode still needs its token, eval scope or not (the contract's validator)
    general = {**_grade_body(t, []), "source": "general"}
    response = await http.post("/v1/interview/grade", json=general, headers=auth(t.eval_key))
    assert response.status_code == 422


async def test_a_sent_token_still_binds_its_level_for_every_key(
    http: httpx.AsyncClient, fakes: EvFakes, new_tenant: NewTenant, superuser: Conn
) -> None:
    """Real keys keep v3 unchanged, and the eval exemption never weakens a token that IS sent."""
    t = new_tenant()
    chunks = _kafka_page(superuser, t)
    fakes.chat.on(QUESTION, question_reply())
    fakes.chat.on(GRADE, grade_reply())
    asked = await http.post(
        "/v1/interview/question",
        json={"collection_ids": [str(t.collection)], "level": "sde_1"},
        headers=auth(t.key),
    )
    q = InterviewQuestion.model_validate(asked.json())
    bound = _grade_body(
        t,
        q.chunk_ids,
        question=q.question,
        kind=q.kind,
        expected_points=q.expected_points,
        question_token=q.question_token,
    )
    ok = await http.post(
        "/v1/interview/grade", json={**bound, "level": "sde_1"}, headers=auth(t.key)
    )
    assert ok.status_code == 200, ok.text
    for key in (t.key, t.eval_key):
        moved = await http.post(
            "/v1/interview/grade", json={**bound, "level": "staff"}, headers=auth(key)
        )
        assert moved.status_code == 403, moved.text
    assert set(q.chunk_ids) <= set(chunks)


# ----------------------------------------------------------------- key creation
async def test_key_creation_refuses_eval_on_mcp_keys_and_beside_other_scopes(
    http: httpx.AsyncClient, new_tenant: NewTenant, superuser: Conn
) -> None:
    t = new_tenant()
    every = ("read", "write", "chat", "approvals:write", "admin", "eval")
    manager = issue_key(superuser, t.tenant, t.user, every, name="ui:owner")
    for name, scopes in (("mcp-eval", ["eval", "read"]), ("eval", ["eval", "write"])):
        response = await http.post(
            "/v1/api-keys", json={"name": name, "scopes": scopes}, headers=auth(manager)
        )
        assert response.status_code == 422, response.text
    created = await http.post(
        "/v1/api-keys",
        json={"name": "eval-harness", "scopes": ["eval", "chat", "read"]},
        headers=auth(manager),
    )
    assert created.status_code == 201, created.text
    assert sorted(created.json()["scopes"]) == ["chat", "eval", "read"]


@pytest.mark.parametrize(("method", "url", "body"), EVAL_CALLS)
async def test_an_eval_key_in_an_owner_tenant_is_refused(
    http: httpx.AsyncClient,
    superuser: Any,
    make_tenant: Callable[[], tuple[UUID, UUID, UUID]],
    method: str,
    url: str,
    body: Json | None,
) -> None:
    """Defence in depth (integrator, EV-API review P2): the /v1/eval surface serves only an eval
    tenant, so an eval-scoped key minted into the owner's tenant never reads live notes."""
    from module_eval_api.conftest import EVAL_KEY_SCOPES

    tenant, user, _ = make_tenant()  # kind = 'personal'
    key = issue_key(superuser, tenant, user, EVAL_KEY_SCOPES, name="eval")
    response = await http.request(method, url, json=body, headers=auth(key))
    assert response.status_code == 403, response.text
    assert "eval tenant" in response.text
