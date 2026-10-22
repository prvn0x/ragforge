"""Pure pieces of the eval seams (no database): key/scope rules, per-surface prompt versions, the
grade-token rule, chunk-key resolution, id parsing, the per-stage recorder and the shown trace."""

from __future__ import annotations

import dataclasses
import hashlib
from collections.abc import Mapping, Sequence
from typing import Any
from uuid import UUID

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from module_d1.d1_fakes import COLLECTION, collect, make_harness, principal, say, verdict
from ragforge_contracts.api import ChunkKey, PageKey, ResolveStatus
from ragforge_contracts.sse import DoneEvent
from ragforge_core.agent.interview.model import Source
from ragforge_core.agent.nodes import _shown_labels
from ragforge_core.agent.retrieval import (
    StageRecorder,
    Stages,
    StagesUnsupported,
    recording,
    stage_score,
)
from ragforge_core.agent.runner import ShownHit, TurnTrace, run_traced, shown_hits
from ragforge_core.agent.state import AgentState, hit_to_state
from ragforge_core.apps.api.chat_stream import TurnEventMapper
from ragforge_core.apps.api.routes.eval import PageChunk, parse_ids, resolve_chunk
from ragforge_core.apps.api.routes.interview import TokenRule, token_rule
from ragforge_core.domain.errors import ValidationFailed
from ragforge_core.domain.models import Principal, RankedChunk, TurnDone
from ragforge_core.domain.ports import Reranker, Tx
from ragforge_core.prompts import PROMPT_VERSION, PROMPT_VERSIONS, surface_of, surface_versions
from ragforge_core.rag.retriever import HybridRetriever, SearchBackend, rerank_and_select
from ragforge_core.services.auth.scopes import (
    ClientKind,
    check_eval_scopes,
    effective_scopes,
    is_key_manager,
    validate_new_key_scopes,
)

EVERY = frozenset({"read", "write", "chat", "approvals:write", "admin", "eval"})


def _principal(scopes: frozenset[str]) -> Principal:
    return Principal(tenant_id=UUID(int=1), user_id=UUID(int=2), scopes=scopes)


# ----------------------------------------------------------------- key and scope rules
@pytest.mark.parametrize("name", ["mcp", "claude-mcp", "MCP:laptop"])
def test_eval_is_never_granted_to_an_mcp_key(name: str) -> None:
    with pytest.raises(ValidationFailed, match="never granted to an MCP key"):
        validate_new_key_scopes(_principal(EVERY), ["eval", "chat", "read"], name)


@pytest.mark.parametrize("extra", ["write", "approvals:write", "admin"])
def test_an_eval_key_holds_only_eval_chat_and_read(extra: str) -> None:
    with pytest.raises(ValidationFailed, match="only eval, chat and read"):
        check_eval_scopes(ClientKind.user, ["eval", "chat", extra])


def test_a_dedicated_eval_key_is_accepted_and_is_no_key_manager() -> None:
    scopes, kind = validate_new_key_scopes(_principal(EVERY), ["eval", "chat", "read"], "eval")
    assert scopes == {"eval", "chat", "read"}
    assert kind is ClientKind.user
    assert not is_key_manager(_principal(scopes))  # eval never administers keys


def test_keys_without_eval_keep_their_old_rules() -> None:
    check_eval_scopes(ClientKind.mcp, ["read", "chat"])  # MCP keys unchanged
    assert validate_new_key_scopes(_principal(EVERY), ["read"], "mcp-cursor") == (
        frozenset({"read"}),
        ClientKind.mcp,
    )


def test_an_mcp_key_row_carrying_eval_loses_it_at_authentication() -> None:
    """Defense in depth: the CLI mints keys without the API's checks."""
    assert effective_scopes(["read", "eval"], ClientKind.mcp) == {"read"}
    assert effective_scopes(["read", "eval"], ClientKind.user) == {"read", "eval"}


# ----------------------------------------------------------------- per-surface prompt versions
def test_every_template_belongs_to_exactly_one_surface() -> None:
    tags = PROMPT_VERSION.split(",")
    split = [t for s in PROMPT_VERSIONS.values() for t in s.split(",")]
    assert sorted(split) == sorted(tags)  # a partition: nothing lost, nothing counted twice
    assert PROMPT_VERSIONS["verifier"] == "ground_check@3"
    assert all(t.startswith("interview_") for t in PROMPT_VERSIONS["interview"].split(","))
    assert "answer@11" in PROMPT_VERSIONS["chat"].split(",")


def test_surface_of_names_the_verifier_and_interview_templates() -> None:
    assert surface_of("ground_check") == "verifier"
    assert surface_of("interview_grade") == "interview"
    assert surface_of("answer") == surface_of("rewrite") == "chat"


def test_a_turn_resumed_by_a_newer_build_reports_both_sets_per_surface() -> None:
    old, new = "answer@10,ground_check@3", "answer@11,ground_check@3"
    assert surface_versions(old, new) == {
        "chat": "answer@10 + answer@11",
        "verifier": "ground_check@3",  # unchanged across the builds: listed once
    }


_NAMES = st.sampled_from(["answer", "ground_check", "interview_grade", "rewrite", "abstain"])
_TAGS = st.builds(lambda n, v: f"{n}@{v}", _NAMES, st.integers(1, 30))


@given(st.lists(st.lists(_TAGS, min_size=1, max_size=6), min_size=1, max_size=3))
def test_surface_versions_keeps_every_tag_of_every_set(sets: list[list[str]]) -> None:
    """Property: each surface reports exactly the tags of its templates, from every set."""
    out = surface_versions(*(",".join(s) for s in sets))
    seen = {t for s in out.values() for part in s.split(" + ") for t in part.split(",")}
    assert seen == {t for s in sets for t in s}
    assert all(surface_of(t.split("@")[0]) == surface for surface, v in out.items()
               for part in v.split(" + ") for t in part.split(","))  # fmt: skip


# ----------------------------------------------------------------- the grade-token rule
@pytest.mark.parametrize(
    ("scopes", "source", "enabled", "enforce", "rule"),
    [
        ({"chat", "eval"}, Source.notes, True, False, TokenRule.exempt),
        ({"chat", "eval"}, Source.notes, True, True, TokenRule.exempt),  # survives enforcement
        ({"chat", "eval"}, Source.general, True, False, TokenRule.required),  # general: always
        ({"chat", "eval"}, Source.notes, False, False, TokenRule.optional),  # switch off
        ({"chat", "eval"}, Source.notes, False, True, TokenRule.required),
        ({"chat"}, Source.notes, True, False, TokenRule.optional),  # today's v3 path
        ({"chat"}, Source.notes, True, True, TokenRule.required),  # enforcement for real keys
        ({"chat"}, Source.general, True, False, TokenRule.required),
    ],
)
def test_token_rule_exempts_only_eval_scope_notes_grades(
    scopes: set[str], source: Source, enabled: bool, enforce: bool, rule: TokenRule
) -> None:
    assert token_rule(_principal(frozenset(scopes)), source, enabled, enforce=enforce) is rule


# ----------------------------------------------------------------- chunk-key resolution
PAGE = PageKey(source_kind="file", key="kafka.md")


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _key(heading: Sequence[str], text: str, occurrence: int = 0) -> ChunkKey:
    return ChunkKey(
        page=PAGE, heading_path=list(heading), display_sha256=_sha(text), occurrence=occurrence
    )


def _page(*chunks: tuple[Sequence[str], str]) -> list[PageChunk]:
    return [PageChunk(UUID(int=100 + i), tuple(h), _sha(t)) for i, (h, t) in enumerate(chunks)]


def test_identical_chunks_under_one_heading_resolve_by_occurrence() -> None:
    page = _page((["K"], "same"), (["K"], "other"), (["K"], "same"))
    assert resolve_chunk(_key(["K"], "same", 0), page).chunk_id == UUID(int=100)
    assert resolve_chunk(_key(["K"], "same", 1), page).chunk_id == UUID(int=102)


def test_a_changed_chunk_is_stale_with_candidates_under_its_heading_only() -> None:
    page = _page((["K"], "new text"), (["Other"], "x"), (["K"], "more"))
    out = resolve_chunk(_key(["K"], "old text"), page)
    assert out.status is ResolveStatus.stale
    assert out.chunk_id is None
    assert out.candidates == [UUID(int=100), UUID(int=102)]


def test_an_occurrence_beyond_the_equal_chunks_is_stale() -> None:
    out = resolve_chunk(_key(["K"], "same", 1), _page((["K"], "same")))
    assert out.status is ResolveStatus.stale
    assert out.candidates == [UUID(int=100)]


def test_a_missing_page_is_missing() -> None:
    out = resolve_chunk(_key(["K"], "x"), None)
    assert out.status is ResolveStatus.missing
    assert out.candidates == []


_CHUNKS = st.lists(
    st.tuples(st.sampled_from([("A",), ("A", "B"), ()]), st.sampled_from(["x", "y", "z"])),
    min_size=1,
    max_size=12,
)


@given(_CHUNKS)
def test_every_chunk_key_round_trips_to_its_own_chunk(
    chunks: list[tuple[tuple[str, ...], str]],
) -> None:
    """Property (principle 6): the key computed for each chunk of a page (occurrence = rank among
    equal (heading, digest) in ordinal order) resolves back to exactly that chunk."""
    page = _page(*chunks)
    seen: dict[tuple[tuple[str, ...], str], int] = {}
    for chunk, (heading, text) in zip(page, chunks, strict=True):
        occurrence = seen.get((heading, text), 0)
        seen[(heading, text)] = occurrence + 1
        resolved = resolve_chunk(_key(heading, text, occurrence), page)
        assert resolved.status is ResolveStatus.ok
        assert resolved.chunk_id == chunk.chunk_id


# ----------------------------------------------------------------- ids
def test_parse_ids_dedupes_in_order_and_rejects_garbage() -> None:
    a, b = UUID(int=1), UUID(int=2)
    assert parse_ids(f"{b}, {a},{b}") == [b, a]
    for bad in ("", " , ", "not-a-uuid", ",".join(str(UUID(int=i)) for i in range(201))):
        with pytest.raises(ValidationFailed):
            parse_ids(bad)


# ----------------------------------------------------------------- per-stage retrieval
def _hit(n: int, score: float, rerank: float | None = None, sha: int | None = None) -> RankedChunk:
    return RankedChunk(
        chunk_id=UUID(int=n),
        document_id=UUID(int=1000 + n),
        collection_id=UUID(int=9),
        anchor="a",
        title="t",
        heading_path=(),
        display_text=f"text {n}",
        display_sha256=hashlib.sha256(str(sha if sha is not None else n).encode()).digest(),
        external_url=None,
        fused_score=score,
        rerank_score=rerank,
    )


class FakeBackend:
    def __init__(
        self, vector: Mapping[int, Sequence[RankedChunk]], keyword: Sequence[RankedChunk]
    ) -> None:
        self._vector, self._keyword = vector, keyword

    async def active_spaces(self, tx: Tx, collection_ids: Sequence[UUID]) -> dict[int, list[UUID]]:
        return {s: list(collection_ids) for s in self._vector}

    async def vector_search(
        self, tx: Tx, space_id: int, v: Sequence[float], c: Sequence[UUID], k: int
    ) -> list[RankedChunk]:
        return list(self._vector[space_id])[:k]

    async def keyword_search(
        self, tx: Tx, query: str, c: Sequence[UUID], k: int
    ) -> list[RankedChunk]:
        return list(self._keyword)[:k]


class ScoreReranker:
    """Scores by a table; ids missing from it stay unscored (like FlashRank past its top_n)."""

    def __init__(self, scores: Mapping[int, float]) -> None:
        self._scores = scores

    async def rerank(self, query: str, candidates: Sequence[RankedChunk]) -> list[RankedChunk]:
        out = [
            dataclasses.replace(c, rerank_score=self._scores.get(c.chunk_id.int))
            for c in candidates
        ]
        return sorted(out, key=lambda c: -(c.rerank_score or -1.0))


async def _pipeline(
    backend: SearchBackend, reranker: Reranker
) -> tuple[list[RankedChunk], list[RankedChunk]]:
    """/v1/query's steps 2-4 on fakes: (final hits, fused candidates)."""
    tx: Any = object()
    collections = [UUID(int=9)]
    retriever = HybridRetriever(backend)
    vectors = {s: [1.0] for s in await retriever.spaces_for(tx, collections)}
    fused = await retriever.retrieve(tx, "q", vectors, collections, 20)
    hits = await rerank_and_select("q", fused, reranker, rerank_top=20, final_n=5)
    return hits, fused


async def _recorded(backend: SearchBackend, reranker: Reranker) -> tuple[list[RankedChunk], Stages]:
    recorder = StageRecorder(backend, reranker)
    hits, fused = await _pipeline(recorder, recorder.reranker)
    return hits, recorder.stages(fused)


async def test_stages_break_score_ties_by_chunk_id() -> None:
    keyword = [_hit(9, 0.5), _hit(3, 0.5), _hit(7, 0.9)]  # backend order is NOT the tie rule
    backend = FakeBackend({1: [_hit(5, -0.1), _hit(2, -0.1)]}, keyword)
    _, stages = await _recorded(backend, ScoreReranker({2: 0.4, 3: 0.4, 5: 0.9, 7: 0.1}))
    assert [h.chunk_id.int for h in stages.keyword] == [7, 3, 9]
    assert [h.chunk_id.int for h in stages.vector] == [2, 5]
    assert [h.chunk_id.int for h in stages.reranked] == [5, 2, 3, 7]  # 9 unscored: absent
    assert [stage_score("reranked", h) for h in stages.reranked] == [0.9, 0.4, 0.4, 0.1]
    fused = [h.fused_score for h in stages.fused]
    assert fused == sorted(fused, reverse=True)  # the retriever's own RRF order


async def test_recording_keeps_the_wired_retrievers_backend_and_configuration() -> None:
    """`stages=true` must run the pipeline old clients get, not a default-built one."""
    keyword = [_hit(n, 1.0 - n / 10) for n in (7, 2, 11, 12, 13)]
    backend = FakeBackend({1: [_hit(5, -0.1), _hit(2, -0.3)]}, keyword)
    wired = HybridRetriever(backend, per_list_k=3, rrf_k=5)  # both NOT the defaults
    wrapped, recorder = recording(wired, ScoreReranker({2: 0.4, 5: 0.9, 7: 0.1}))
    tx: Any = object()
    collections = [UUID(int=9)]
    vectors = {s: [1.0] for s in await wired.spaces_for(tx, collections)}
    plain = await wired.retrieve(tx, "q", vectors, collections, 20)
    recorded = await wrapped.retrieve(tx, "q", vectors, collections, 20)
    assert recorded == plain  # same RRF k and per-list k: identical fused candidates
    assert recorder.backend is backend
    assert len(recorder.keyword_list) == 3  # per_list_k carried over (and really recorded)


def test_recording_refuses_a_retriever_it_cannot_record() -> None:
    with pytest.raises(StagesUnsupported):
        recording(FakeRetrieverStub(), ScoreReranker({}))


class FakeRetrieverStub:
    async def spaces_for(self, tx: Tx, collection_ids: Sequence[UUID]) -> Sequence[int]:
        return []

    async def retrieve(
        self,
        tx: Tx,
        query: str,
        query_vectors: Mapping[int, Sequence[float]],
        collection_ids: Sequence[UUID],
        k: int,
    ) -> list[RankedChunk]:
        return []  # fmt: skip


async def test_vector_lists_of_several_spaces_interleave_by_rank() -> None:
    spaces = {2: [_hit(4, -0.5), _hit(1, -0.9)], 1: [_hit(8, -0.2), _hit(4, -0.3)]}
    _, stages = await _recorded(FakeBackend(spaces, []), ScoreReranker({}))
    # rank 1 of space 1 (8), rank 1 of space 2 (4), rank 2 of space 1 (4: seen), rank 2 of 2 (1)
    assert [h.chunk_id.int for h in stages.vector] == [8, 4, 1]
    assert stages.reranked == ()  # an unavailable reranker scores nothing: no reranked stage


_SCORED = st.dictionaries(st.integers(1, 30), st.sampled_from([0.1, 0.5, 0.9]), max_size=10)


def _backend(vector: dict[int, float], keyword: dict[int, float], *, reverse: bool) -> FakeBackend:
    """Best-first lists as a backend returns them; `reverse` flips the order WITHIN equal
    scores only, i.e. a backend whose tie order differs."""

    def ranked(scores: dict[int, float]) -> list[RankedChunk]:
        hits = sorted((_hit(n, s) for n, s in scores.items()), key=lambda h: h.chunk_id.int,
                      reverse=reverse)  # fmt: skip
        return sorted(hits, key=lambda h: -h.fused_score)  # stable: ties keep the order above

    return FakeBackend({1: ranked(vector)}, ranked(keyword))


@settings(max_examples=50, deadline=None)
@given(vector=_SCORED, keyword=_SCORED, rerank=_SCORED)
async def test_recording_changes_nothing_and_stages_are_deterministic(
    vector: dict[int, float], keyword: dict[int, float], rerank: dict[int, float]
) -> None:
    """Property: the recorded pipeline returns the bare pipeline's hits, and the keyword,
    vector and reranked stages do not depend on the backend's order among equal scores."""
    bare, _ = await _pipeline(_backend(vector, keyword, reverse=False), ScoreReranker(rerank))
    hits, stages = await _recorded(_backend(vector, keyword, reverse=False), ScoreReranker(rerank))
    _, flipped = await _recorded(_backend(vector, keyword, reverse=True), ScoreReranker(rerank))
    assert [h.chunk_id for h in hits] == [h.chunk_id for h in bare]
    for name in ("keyword", "vector", "reranked"):
        mine, theirs = getattr(stages, name), getattr(flipped, name)
        assert [h.chunk_id for h in mine] == [h.chunk_id for h in theirs]
        keys = [(-stage_score(name, h), str(h.chunk_id)) for h in mine]
        assert keys == sorted(keys)


# ----------------------------------------------------------------- shown + prompt versions
def _state(hits: list[RankedChunk], context: list[int], tools: list[int]) -> AgentState:
    return AgentState(
        hits=[hit_to_state(h) for h in hits], context_labels=context, tool_hit_labels=tools
    )


@given(
    context=st.lists(st.integers(1, 5), max_size=5),
    tools=st.lists(st.integers(1, 6), max_size=3),
)
def test_shown_hits_follow_the_nodes_shown_label_rule(context: list[int], tools: list[int]) -> None:
    hits = [_hit(n, 0.1) for n in range(1, 6)]
    state = _state(hits, context, tools)
    shown = shown_hits(dict(state))
    assert [s.label for s in shown] == [n for n in _shown_labels(state) if n <= len(hits)]
    assert all(s.chunk_id == hits[s.label - 1].chunk_id for s in shown)


async def test_the_runner_fills_the_trace_with_what_the_model_was_shown() -> None:
    h = make_harness([say("Fact one holds [S1]."), verdict()])
    trace = TurnTrace()
    events = await collect(run_traced(h.runner, principal(), None, [COLLECTION], "fact?", trace))
    done = events[-1]
    assert isinstance(done, TurnDone)
    assert type(done) is TurnDone
    assert trace.done
    assert [s.label for s in trace.shown] == [1, 2]
    assert [s.chunk_id for s in trace.shown] == [UUID(int=1001), UUID(int=1002)]
    assert trace.prompt_versions == surface_versions(PROMPT_VERSION)


def _done_event(trace: TurnTrace | None, *, shown: bool) -> DoneEvent:
    mapper = TurnEventMapper(lambda: _Identity(), new_conversation_id=None, trace=trace,
                             trace_shown=shown)  # fmt: skip
    (event,) = mapper.map(TurnDone(UUID(int=5), (), PROMPT_VERSION))
    assert isinstance(event, DoneEvent)
    return event


class _Identity:
    def allow(self, urls: Sequence[str]) -> None: ...

    def feed(self, text: str) -> str:
        return text

    def finish(self) -> str:
        return ""


def test_done_carries_shown_only_when_traced_and_prompt_versions_always() -> None:
    trace = TurnTrace()
    trace.prompt_versions, trace.done = {"chat": "answer@11"}, True
    hit = _hit(3, 0.1)
    trace.shown = (ShownHit(1, hit.chunk_id, hit.document_id, hit.display_sha256),)
    traced = _done_event(trace, shown=True)
    assert traced.shown is not None
    assert [s.model_dump() for s in traced.shown] == [
        {
            "label": 1,
            "chunk_id": hit.chunk_id,
            "document_id": hit.document_id,
            "display_sha256": hit.display_sha256.hex(),
        }
    ]
    untraced = _done_event(trace, shown=False)
    assert untraced.shown is None  # never on the wire without the eval grant
    assert untraced.prompt_versions == {"chat": "answer@11"}
    # a producer with no trace: derived from prompt_version, empty shown (never invented)
    plain = _done_event(None, shown=True)
    assert plain.shown == []
    assert plain.prompt_versions == PROMPT_VERSIONS
