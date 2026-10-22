"""Which note the next question is built from, and what kind of question it asks. Pure: no I/O,
no model; the randomness is an injected `random.Random`, so an eval can replay a session.

Owner decisions (docs/NEXT.md, "TO DO NEXT" 1): "on Kafka" means ALL the Kafka pages (title or
heading match, plus retrieval), with broad coverage and no repeats; with no topic, questions mix
topics across all the notes; all four kinds of question (concept, scenario, design, code).

How a seed is chosen, best tier first:
1. never a chunk the session already used (`exclude`): a hard filter, applied in SQL too;
1b. a direct hit of the topic beats a hit of an acronym expansion only (`secondary`: the topic
   "HLD" read as "High Level Design" is a guess, the topic's own words are not);
2. a page the session has not covered yet beats one it has (broad coverage);
3. a chunk that fits the DUE kind (requested[0]) beats one that does not (this is what makes
   the rotation reach code questions: concept fits every note);
3b. then one that fits any requested kind (only matters when the request omits concept);
4. a chunk with enough text to ask about beats a heading-only fragment.
Within the best tier a PAGE is drawn uniformly first, then a chunk of it: drawing chunks directly
would let one long page crowd out every short one.

Kind selection is GENERIC on purpose (design review): nothing here reads page titles or English
keywords, which would only work for one person's notes. The one structural fact the server
checks is the one a wrong guess makes harmful: a code question needs a code block in the notes
it is built from, or the model invents the code it then grades against. Scenario and design
questions are framed by the model from whatever failure modes and trade-offs the notes hold
(the prompt says so), and every note can ground a concept question. A requested kind with no
fitting material falls back to another kind (the response says which); it is never treated as
"nothing left to ask".
"""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Collection, Iterable, Sequence
from dataclasses import dataclass
from typing import Final
from uuid import UUID

from ragforge_core.agent.interview.model import Kind
from ragforge_core.agent.structure import markdown_tokens
from ragforge_core.domain.models import RankedChunk

# Below this many characters a chunk is usually a heading or a one-line fragment: too little to
# build a question on (still used when nothing better exists).
MIN_SEED_CHARS: Final = 80
_CODE_TOKENS: Final = frozenset({"fence", "code_block"})
_ALWAYS: Final = frozenset({Kind.concept, Kind.scenario, Kind.design})


def clean_topic(topic: str | None) -> str | None:
    """None for a missing or blank topic (both mean "mix every topic"); inner whitespace runs
    collapse to one space."""
    if topic is None:
        return None
    flat = " ".join(topic.split())
    return flat or None


def has_code_block(text: str) -> bool:
    """A fenced or indented code block, as CommonMark parses it (the verifier's own parser,
    `structure.markdown_tokens`), at any nesting depth. Inline code spans do not count: a
    backticked name is not enough material for a code/LLD question."""
    return any(token.type in _CODE_TOKENS for token in markdown_tokens(text))


def fitting_kinds(notes: Iterable[RankedChunk]) -> frozenset[Kind]:
    """The kinds of question these notes can ground."""
    if any(has_code_block(note.display_text) for note in notes):
        return frozenset(Kind)
    return _ALWAYS


def choose_kind(notes: Sequence[RankedChunk], requested: Sequence[Kind]) -> Kind:
    """The first requested kind the notes fit (the client rotates `requested`, so the rotation
    is deterministic and client-held); `concept` when none fits (only `code` was requested and
    the notes hold no code)."""
    fits = fitting_kinds(notes)
    return next((kind for kind in requested if kind in fits), Kind.concept)


def unique_chunks(*groups: Iterable[RankedChunk]) -> list[RankedChunk]:
    """Chunks of all groups, each once (first occurrence wins), in order."""
    seen: dict[UUID, RankedChunk] = {}
    for group in groups:
        for chunk in group:
            seen.setdefault(chunk.chunk_id, chunk)
    return list(seen.values())


def _tier(
    chunk: RankedChunk,
    covered: Collection[UUID],
    requested: Sequence[Kind],
    secondary: Collection[UUID] = (),
) -> tuple[bool, bool, bool, bool, bool]:
    """Lower sorts first: (only an expansion's hit, page already covered, does not fit the DUE
    kind, fits no requested kind, too short). The client lists the due kind first; ranking on
    it is what makes the rotation reach code questions (concept fits every note, so "fits any"
    alone never steered: final review P1, code was picked ~1 turn in 20)."""
    fits = fitting_kinds((chunk,))
    return (
        chunk.chunk_id in secondary,
        chunk.document_id in covered,
        bool(requested) and requested[0] not in fits,
        not (fits & set(requested)),
        len(chunk.display_text.strip()) < MIN_SEED_CHARS,
    )


def pick_seed(
    candidates: Sequence[RankedChunk],
    *,
    exclude: Collection[UUID],
    covered: Collection[UUID],
    kinds: Sequence[Kind],
    rng: random.Random,
    secondary: Collection[UUID] = (),
) -> RankedChunk | None:
    """The note to ask about, or None when every candidate was already used. `secondary` are
    candidates found only through an acronym expansion (tier 1b)."""
    pool = [c for c in unique_chunks(candidates) if c.chunk_id not in exclude]
    if not pool:
        return None
    tiers = {c.chunk_id: _tier(c, covered, kinds, secondary) for c in pool}
    best = min(tiers.values())
    by_page: dict[UUID, list[RankedChunk]] = defaultdict(list)
    for chunk in pool:
        if tiers[chunk.chunk_id] == best:
            by_page[chunk.document_id].append(chunk)
    page = rng.choice(sorted(by_page, key=str))  # sorted: a seeded rng is reproducible
    return rng.choice(sorted(by_page[page], key=lambda c: str(c.chunk_id)))


def context_notes(
    seed: RankedChunk, page: Sequence[RankedChunk], exclude: Collection[UUID], limit: int
) -> tuple[RankedChunk, ...]:
    """The seed plus up to `limit` - 1 of its nearest page neighbours, in page order. A
    neighbour the session already used is left out, so nothing is asked about twice through
    the back door. `page` is in page order (`InterviewNotes.neighbours`)."""
    order = {c.chunk_id: i for i, c in enumerate(page)}
    if seed.chunk_id not in order:  # the page changed between the two reads: the seed alone
        return (seed,)
    at = order[seed.chunk_id]
    others = [c for c in page if c.chunk_id != seed.chunk_id and c.chunk_id not in exclude]
    nearest = sorted(others, key=lambda c: (abs(order[c.chunk_id] - at), order[c.chunk_id]))
    chosen = {seed.chunk_id, *(c.chunk_id for c in nearest[: max(limit - 1, 0)])}
    return tuple(c for c in page if c.chunk_id in chosen)


@dataclass(frozen=True, slots=True)
class Exhausted:
    """Why no seed was found, for the 409 detail (plain text; clients branch on the Problem
    type): nothing matched at all before exclusion, or all of it was asked."""

    topic: str | None
    nothing_matched: bool

    @property
    def message(self) -> str:
        scope = f"on {self.topic}" if self.topic else "in these collections"
        if self.nothing_matched:
            return f"No notes {scope}"
        return f"No more unasked notes {scope}"
