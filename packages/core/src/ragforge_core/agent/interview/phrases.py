"""A short-lived, bounded cache of the visible titles and heading segments (the acronym matcher's
input, `acronyms.py`).

WHY: the phrase read (`InterviewNotes.topic_phrases`) scans every visible chunk of the
collections, and a session asks the same topic question after question, so without a cache an
acronym-shaped topic with no title hit of its own repeats that scan on every question. A few
seconds of staleness is harmless: an expansion only FEEDS the normal topic reads, which apply
RLS and every visibility filter again, so a phrase of a page removed in the meantime finds
nothing, and a page added in the meantime is seen after the TTL.

The key is (tenant, collection set): the phrases are read under that tenant's RLS for exactly
those collections, so one tenant's phrases can never answer another tenant's request. No lock:
two concurrent misses both read and the later write wins, which costs one extra read and is
never wrong. Stdlib only (no caching library is a dependency): an `OrderedDict` gives the
oldest-first eviction.
"""

from __future__ import annotations

import time
from collections import OrderedDict
from collections.abc import Callable, Collection, Sequence
from uuid import UUID

type PhraseKey = tuple[UUID, frozenset[UUID]]


class PhraseCache:
    def __init__(
        self,
        *,
        ttl_s: float = 60.0,
        max_entries: int = 256,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._ttl_s = ttl_s
        self._max_entries = max_entries
        self._clock = clock
        self._entries: OrderedDict[PhraseKey, tuple[float, tuple[str, ...]]] = OrderedDict()

    @staticmethod
    def key(tenant_id: UUID, collection_ids: Collection[UUID]) -> PhraseKey:
        return tenant_id, frozenset(collection_ids)

    def get(self, key: PhraseKey) -> tuple[str, ...] | None:
        entry = self._entries.get(key)
        if entry is None:
            return None
        stored_at, phrases = entry
        if self._clock() - stored_at >= self._ttl_s:
            del self._entries[key]
            return None
        return phrases

    def put(self, key: PhraseKey, phrases: Sequence[str]) -> None:
        if self._ttl_s <= 0 or self._max_entries <= 0:
            return
        self._entries.pop(key, None)
        self._entries[key] = (self._clock(), tuple(phrases))
        while len(self._entries) > self._max_entries:
            self._entries.popitem(last=False)
