"""Interview mode's two operations with their I/O: short tenant transactions around the reads,
every model call (embedder, reranker, chat) with NO transaction open (PHASE1 rule).

The decisions themselves live in pure modules (`selection`, `acronyms`, `prompts`, `replies`,
`verify`, `question`, `grade`); this module only fetches their inputs. The tenant comes only from
the authenticated Principal (the route passes `principal.tenant_id`), never from request text.

Sources (Phase 3c): `notes` (the default) retrieves and verifies; `general` is the client's
explicit opt-in fallback and reads NO note at all (only the collection access check every route
makes): no retrieval, no chunk, no verifier. The server never switches source on its own: a
notes request that finds nothing is a 409 the client may answer by offering general mode.

Targeted questions (ADR-0017 §3 E-I, eval scope only, checked by the route): `targeted_question`
skips seed selection and asks from the given notes, so the eval can generate question-quality
items from known seeds. Everything after selection (context, kind, prompt, verification) is the
normal path's.
"""

from __future__ import annotations

import random
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Final
from uuid import UUID

from ragforge_core.agent.interview.acronyms import acronym_words, expanded_topics
from ragforge_core.agent.interview.grade import grade_answer, grade_general
from ragforge_core.agent.interview.model import (
    DraftGrade,
    DraftQuestion,
    GradeRequest,
    InterviewSettings,
    QuestionRequest,
    Source,
)
from ragforge_core.agent.interview.phrases import PhraseCache
from ragforge_core.agent.interview.ports import InterviewNotes
from ragforge_core.agent.interview.question import write_general_question, write_question
from ragforge_core.agent.interview.selection import (
    Exhausted,
    choose_kind,
    clean_topic,
    context_notes,
    pick_seed,
    unique_chunks,
)
from ragforge_core.domain.errors import Conflict, NotFound, RagforgeError
from ragforge_core.domain.models import RankedChunk
from ragforge_core.domain.ports import (
    ChatModel,
    EmbedderRegistry,
    Reranker,
    Retriever,
    TransactionManager,
    Tx,
)
from ragforge_core.observability.logging import get_logger
from ragforge_core.rag.retriever import rerank_and_select

log = get_logger(__name__)


class InterviewExhausted(Conflict):
    """Every note matching the request was already asked this session. The API maps it to a
    409 Problem whose `type` is the contract's INTERVIEW_EXHAUSTED."""

    title = "Interview exhausted"

    def __init__(self, why: Exhausted) -> None:
        super().__init__(why.message)
        self.why = why


class InterviewNoNotes(InterviewExhausted):
    """Nothing matches the request at all, BEFORE the session's exclusions are applied (so a
    session that already used Kafka notes and asks for Redis, with no Redis notes, gets this,
    not "exhausted"). The API maps it to Problem.type INTERVIEW_NO_NOTES."""

    title = "No matching notes"


class TopicRequired(RagforgeError):
    """source=general without a usable topic (the contract rejects a blank one already; this
    is the domain's own guard for any other caller, e.g. an eval harness)."""

    status = 400
    title = "Bad request"


@dataclass(frozen=True, slots=True)
class InterviewDeps:
    tx: TransactionManager
    notes: InterviewNotes
    retriever: Retriever
    embedders: EmbedderRegistry
    reranker: Reranker
    chat: ChatModel
    settings: InterviewSettings = field(default_factory=InterviewSettings)
    # The API passes one process-wide cache; the default (one per deps) suits evals and tests.
    phrases: PhraseCache = field(default_factory=PhraseCache)


@dataclass(frozen=True, slots=True)
class _Candidates:
    direct: list[RankedChunk]  # the topic's own title/heading and search hits (or the sample)
    expanded: list[RankedChunk]  # hits of acronym expansions only: ranked below direct ones
    covered: frozenset[UUID]  # pages the session already asked about
    # anything at all before exclusion; None: not known yet (no topic, exclusions given, and the
    # sample came back empty: `_any_note` decides, only then)
    matched: bool | None


async def next_question(
    deps: InterviewDeps,
    tenant_id: UUID,
    request: QuestionRequest,
    rng: random.Random | None = None,
) -> DraftQuestion:
    """Seed -> its page context -> kind -> one question (source=notes), or one general-knowledge
    question (source=general). Raises `InterviewNoNotes` / `InterviewExhausted`."""
    if request.source is Source.general:
        return await _general_question(deps, tenant_id, request)
    rng = rng or random.Random()  # noqa: S311 - sampling notes, not security
    collections = list(request.collection_ids)
    exclude = request.exclude_chunk_ids
    topic = clean_topic(request.topic)
    found = await _candidates(deps, tenant_id, collections, topic, exclude)
    seed = pick_seed(
        [*found.direct, *found.expanded],
        exclude=exclude,
        covered=found.covered,
        kinds=request.kinds,
        rng=rng,
        secondary={c.chunk_id for c in found.expanded},
    )
    if seed is None:
        matched = found.matched
        if matched is None:
            matched = await _any_note(deps, tenant_id, collections)
        why = Exhausted(topic, nothing_matched=not matched)
        raise InterviewNoNotes(why) if not matched else InterviewExhausted(why)
    limit = deps.settings.context_notes
    async with deps.tx.tenant(tenant_id) as tx:
        page = await deps.notes.neighbours(tx, seed.chunk_id, collections, radius=limit - 1)
    notes = context_notes(seed, page, exclude, limit)
    return await write_question(
        deps.chat,
        notes,
        kind=choose_kind(notes, request.kinds),
        difficulty=request.difficulty,
        topic=topic,
        settings=deps.settings,
        level=request.level,
    )


async def targeted_question(
    deps: InterviewDeps, tenant_id: UUID, request: QuestionRequest, seed_ids: Sequence[UUID]
) -> DraftQuestion:
    """A notes question from `seed_ids` (seed = the first; S1.. in the given order), never from
    selection. 404 unless EVERY id is a visible chunk of the requested collections (which one
    is never said). The seed's nearest page neighbours complete the context up to
    `context_notes`, exactly as for a selected seed (minus the session's used chunks), so a
    targeted question sees the material a sampled one would."""
    ids = list(dict.fromkeys(seed_ids))
    if not ids:
        raise NotFound("notes not found")
    collections = list(request.collection_ids)
    limit = deps.settings.context_notes
    async with deps.tx.tenant(tenant_id) as tx:
        await deps.notes.require_collections(tx, collections)
        rows = {c.chunk_id: c for c in await deps.notes.chunks_by_id(tx, ids, collections)}
        if set(rows) != set(ids):
            raise NotFound("notes not found")
        seed = rows[ids[0]]
        page = await deps.notes.neighbours(tx, seed.chunk_id, collections, radius=limit - 1)
    context = context_notes(seed, page, request.exclude_chunk_ids, limit)
    notes = unique_chunks((rows[i] for i in ids), context)[: max(limit, len(ids))]
    return await write_question(
        deps.chat,
        notes,
        kind=choose_kind(notes, request.kinds),
        difficulty=request.difficulty,
        topic=clean_topic(request.topic),
        settings=deps.settings,
        level=request.level,
    )


async def _general_question(
    deps: InterviewDeps, tenant_id: UUID, request: QuestionRequest
) -> DraftQuestion:
    topic = clean_topic(request.topic)
    if topic is None:
        raise TopicRequired("topic: required for source=general")
    async with deps.tx.tenant(tenant_id) as tx:  # the access check only: no note is read
        await deps.notes.require_collections(tx, list(request.collection_ids))
    return await write_general_question(
        deps.chat,
        topic=topic,
        kind=request.kinds[0],  # the client's due kind: no material decides it here
        difficulty=request.difficulty,
        level=request.level,
        asked_before=request.exclude_questions,
    )


# Words a topic phrase carries that no note needs to contain for it to be about the topic.
_GATE_FILLER: Final = frozenset({
    # never real topic words ("deep", "basic", "concept" are: deep learning, basic auth,
    # concept drift; re-review P2)
    "vs", "versus", "basics", "fundamentals", "interview", "interviews", "question",
    "questions", "overview", "intro", "introduction",
})  # fmt: skip
_GATE_SPLIT: Final = re.compile(r"[\s/\-_.]+")


def gate_terms(topic: str) -> str:
    """The topic as the lexical gate reads it: `/-_.` are spaces (as the gate's SQL reads the
    note) and filler words are dropped ("Kafka vs RabbitMQ" -> "Kafka RabbitMQ"). Empty when
    only filler is left."""
    words = [w for w in _GATE_SPLIT.split(topic) if w and w.casefold() not in _GATE_FILLER]
    return " ".join(words)


async def _candidates(
    deps: InterviewDeps,
    tenant_id: UUID,
    collections: list[UUID],
    topic: str | None,
    exclude: frozenset[UUID],
) -> _Candidates:
    """With a topic: every chunk of the pages whose title or headings match it, plus the hybrid
    search hits for it that clear tau; when an acronym-shaped word of the topic has no title hit
    of its own, the same two searches for its expansions too (`acronyms.py`), ranked below.
    Without a topic: a broad sample, one chunk per page."""
    settings = deps.settings
    async with deps.tx.tenant(tenant_id) as tx:
        await deps.notes.require_collections(tx, collections)
        covered = await deps.notes.documents_of(tx, exclude) if exclude else frozenset()
        if topic is None:
            sample = await deps.notes.sample_chunks(tx, collections, exclude, settings.sample_pool)
            known = True if sample else (False if not exclude else None)
            return _Candidates(list(sample), [], covered, known)
        titled = list(
            await deps.notes.topic_chunks(tx, collections, topic, exclude, settings.topic_pool)
        )
        titled_any = bool(titled) or (
            bool(exclude) and bool(await deps.notes.topic_chunks(tx, collections, topic, (), 1))
        )
        expansions = (
            [] if titled_any else await _expansions(deps, tx, tenant_id, collections, topic)
        )
        expanded_titled: list[RankedChunk] = []
        expanded_any = False  # some expansion has a title/heading hit, before exclusion
        for expansion in expansions:
            hits = await deps.notes.topic_chunks(
                tx, collections, expansion, exclude, settings.topic_pool
            )
            expanded_titled.extend(hits)
            expanded_any = (
                expanded_any
                or bool(hits)
                or (
                    bool(exclude)
                    and bool(await deps.notes.topic_chunks(tx, collections, expansion, (), 1))
                )
            )
        spaces = list(await deps.retriever.spaces_for(tx, collections))
    searched = await _search(deps, tenant_id, collections, topic, spaces)
    expanded_searched = [
        hit
        for expansion in expansions
        for hit in await _search(deps, tenant_id, collections, expansion, spaces)
    ]
    direct = unique_chunks(titled, (h for h in searched if h.chunk_id not in exclude))
    seen = {c.chunk_id for c in direct}
    expanded = [
        c
        for c in unique_chunks(expanded_titled, expanded_searched)
        if c.chunk_id not in exclude and c.chunk_id not in seen
    ]
    # Evidence only: an expansion merely EXISTING proves nothing once the topic has other words
    # ("HLD caching" -> "High Level Design caching" may match no note at all, round-1 review).
    matched = titled_any or bool(searched) or expanded_any or bool(expanded_searched)
    return _Candidates(direct, expanded, covered, matched)


async def _expansions(
    deps: InterviewDeps, tx: Tx, tenant_id: UUID, collections: list[UUID], topic: str
) -> list[str]:
    """The topic with its acronyms expanded, or [] (no acronym-shaped word: no extra read)."""
    if not acronym_words(topic):
        return []
    key = PhraseCache.key(tenant_id, collections)
    phrases = deps.phrases.get(key)
    cached = phrases is not None
    if phrases is None:
        phrases = await _read_phrases(deps, tx, collections)
        deps.phrases.put(key, phrases)
    expansions = expanded_topics(topic, phrases, deps.settings.max_expansions)
    log.info(
        "interview_topic_expanded",
        phrases=len(phrases),
        expansions=len(expansions),
        cached=cached,
    )
    return expansions


async def _read_phrases(deps: InterviewDeps, tx: Tx, collections: list[UUID]) -> tuple[str, ...]:
    pool = deps.settings.phrase_pool
    phrases = tuple(await deps.notes.topic_phrases(tx, collections, pool + 1))
    if len(phrases) > pool:
        # Bounded on purpose; past the cap some titles are not considered (the read is ordered,
        # so always the same ones). Counts only: titles are note text and never logged.
        log.warning("interview_topic_phrases_truncated", limit=pool)
        phrases = phrases[:pool]
    return phrases


async def _any_note(deps: InterviewDeps, tenant_id: UUID, collections: list[UUID]) -> bool:
    """Whether the collections hold ANY visible note, ignoring the session's exclusions: one
    cheap existence read, run only on the no-topic "nothing left" path (NO_NOTES vs EXHAUSTED;
    with a topic, `_candidates` already knows)."""
    async with deps.tx.tenant(tenant_id) as tx:
        return bool(await deps.notes.sample_chunks(tx, collections, (), 1))


async def _search(
    deps: InterviewDeps,
    tenant_id: UUID,
    collections: list[UUID],
    query: str,
    spaces: Sequence[int],
) -> list[RankedChunk]:
    """The chat path's retrieval (embed -> fused candidates -> rerank), keeping only hits about
    the topic (rerank score >= tau). Embedding and reranking run with no transaction open.
    NOT filtered by the session's exclusions: the caller needs the raw hits for NO_NOTES."""
    settings = deps.settings
    vectors: dict[int, list[float]] = {}
    for space in dict.fromkeys(spaces):
        await deps.embedders.ensure(space)
        vectors[space] = await deps.embedders.for_space(space).embed_query(query)
    async with deps.tx.tenant(tenant_id) as tx:
        fused = await deps.retriever.retrieve(tx, query, vectors, collections, settings.rerank_top)
    ranked = await rerank_and_select(
        query,
        fused,
        deps.reranker,
        rerank_top=settings.rerank_top,
        final_n=settings.rerank_top,
    )
    relevant = [h for h in ranked if h.rerank_score is not None and h.rerank_score >= settings.tau]
    # A search hit must also MENTION the topic: the reranker alone let an unrelated note through
    # for a topic the notes lack, so the client never got NO_NOTES and its general-knowledge
    # offer (ports.InterviewNotes.mentioning).
    async with deps.tx.tenant(tenant_id) as tx:
        terms = gate_terms(query)
        if not terms:  # nothing but filler: the reranker's judgement stands
            return relevant
        named = await deps.notes.mentioning(tx, collections, [h.chunk_id for h in relevant], terms)
    return [h for h in relevant if h.chunk_id in named]


async def grade(deps: InterviewDeps, tenant_id: UUID, request: GradeRequest) -> DraftGrade:
    """source=notes: the notes are re-read through RLS, never trusted from the client; 404
    unless EVERY chunk id resolves (a chunk the caller cannot see is indistinguishable from a
    missing one). source=general: the access check only, then the judge; no note is read."""
    collections = list(request.collection_ids)
    if request.source is Source.general:
        async with deps.tx.tenant(tenant_id) as tx:
            await deps.notes.require_collections(tx, collections)
        return await grade_general(deps.chat, request)
    ids = list(dict.fromkeys(request.chunk_ids))
    if not ids:  # the contract requires 1..10; an eval harness gets the same answer
        raise NotFound("notes not found")
    async with deps.tx.tenant(tenant_id) as tx:
        await deps.notes.require_collections(tx, collections)
        rows = {c.chunk_id: c for c in await deps.notes.chunks_by_id(tx, ids, collections)}
    if set(rows) != set(ids):
        raise NotFound("notes not found")
    notes = [rows[i] for i in ids]  # the question's order: its S1..Sn labels
    return await grade_answer(deps.chat, notes, request, deps.settings)
