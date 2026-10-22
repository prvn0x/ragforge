"""Acronym topics: "HLD" finds the page titled "High Level Design" (Phase 3c, live failure).

Pure: the caller fetches the visible titles and heading segments (under RLS and the same filters
as every other read, `ports.InterviewNotes.topic_phrases`) and this module matches in Python, so
no user text ever becomes a SQL pattern. There is no hardcoded acronym list: a topic word spells
an acronym when the INITIALS of a contiguous run of a phrase's words spell it.

Rules (design review, 2026-10-19), each for a reason:
- Only ACRONYM-SHAPED words are expanded: 3 to 6 ASCII letters, or 2 letters typed in CAPITALS
  ("DB" yes, "db" or "of" no). Shorter or lowercase two-letter words are ordinary words far too
  often. Stop words are never acronyms.
- Only a word with no title hit of its own is expanded (the caller decides, from the direct
  FTS match): "Kafka" with Kafka pages is never re-read as an acronym.
- Initials are tried WITH and WITHOUT stop words ("Data Structures and Algorithms" spells DSA
  without "and", "Separation of Concerns" spells SOC with "of"). A run never starts or ends on a
  stop word, so "of Concerns" is never a run of its own.
- Hyphens split words ("High-Level Design" is three words); any other non-ASCII-alphanumeric
  word (accents, other scripts, digits-only) breaks a run: ASCII/Latin initials only, so a
  look-alike letter can never complete an acronym.
- For an acronym of 3 or more letters, ", " and " & " also join words inside a run (a list
  spelled out: "Atomicity, Consistency, Isolation & Durability" spells ACID, "Consistency,
  Availability, Partition tolerance" spells CAP). Two-letter acronyms keep the strict breaks:
  "A, B" is far more often two things than one name. Every other break (":", "(", "/", a comma
  with no space after it) still ends the phrase.
- A run must cover the WHOLE phrase, or spell an acronym of 3 or more letters. "HL" therefore
  matches a page titled "High Level" but not "High Level Design" (a two-letter prefix of a longer
  name is a different thing far more often than not); "HLD" matches "Notes on High Level Design".
- Case-insensitive on both sides; the expansion is the run's words as the page writes them.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from typing import Final

ACRONYM_LETTERS: Final = re.compile(r"[A-Za-z]{2,6}")
# Word separators inside a phrase: whitespace and hyphen-like characters. Anything else that is
# not an ASCII letter or digit (":", "(", "/", "&", ...) ends the PHRASE, so "System Design: High
# Level Design" holds two phrases, and "HLD" can cover the second one whole.
_PHRASE_BREAK: Final = re.compile(r"[^A-Za-z0-9\s\-\u2010\u2011\u2013]+")
_WORD_BREAK: Final = re.compile(r"[\s\-\u2010\u2011\u2013]+")
# A spelled-out list: a comma or ampersand FOLLOWED by whitespace (3+ letter acronyms only).
_LIST_JOIN: Final = re.compile(r"\s*(?:,|&)\s+")
_ASCII_WORD: Final = re.compile(r"[A-Za-z][A-Za-z0-9]*")
STOP_WORDS: Final = frozenset(
    {"a", "an", "and", "at", "by", "for", "in", "of", "on", "or", "the", "to", "vs", "with"}
)
_MIN_PARTIAL: Final = 3  # an acronym this long may match a run inside a longer phrase


def acronym_shaped(word: str) -> bool:
    """3 to 6 ASCII letters, or exactly 2 typed in capitals; never a stop word."""
    if not ACRONYM_LETTERS.fullmatch(word) or word.casefold() in STOP_WORDS:
        return False
    return len(word) >= 3 or word.isupper()


def acronym_words(topic: str) -> list[str]:
    """The topic's acronym-shaped words, in order, each once."""
    return list(dict.fromkeys(w for w in topic.split() if acronym_shaped(w)))


def _phrases(text: str) -> Iterable[list[str]]:
    """The phrases of a title or heading segment, each as its words; a word that is not ASCII
    alphanumeric becomes "", which no run may cross."""
    for part in _PHRASE_BREAK.split(text):
        words = [w for w in _WORD_BREAK.split(part) if w]
        if words:
            yield [w if _ASCII_WORD.fullmatch(w) else "" for w in words]


def _spells(run: Sequence[str], acronym: str, *, skip_stop_words: bool) -> bool:
    if not run or run[0].casefold() in STOP_WORDS or run[-1].casefold() in STOP_WORDS:
        return False
    if skip_stop_words:
        run = [w for w in run if w.casefold() not in STOP_WORDS]
    return "".join(w[0] for w in run).casefold() == acronym


def phrase_matches(acronym: str, text: str) -> list[str]:
    """Every contiguous word run of `text` whose initials spell `acronym` (rules above), as the
    page writes it."""
    target = acronym.casefold()
    found: list[str] = []
    texts = [text]
    if len(target) >= _MIN_PARTIAL and (joined := _LIST_JOIN.sub(" ", text)) != text:
        texts.append(joined)
    for words in (w for t in texts for w in _phrases(t)):
        n = len(words)
        for start in range(n):
            for end in range(start + 1, n + 1):
                run = words[start:end]
                if "" in run:
                    break  # a non-ASCII word breaks every run through it
                if len(run) > 2 * len(target):  # longer runs cannot spell it, even with skips
                    break
                whole = start == 0 and end == n
                if not (whole or len(target) >= _MIN_PARTIAL):
                    continue
                if _spells(run, target, skip_stop_words=False) or _spells(
                    run, target, skip_stop_words=True
                ):
                    found.append(" ".join(run))
    return list(dict.fromkeys(found))


def expand(acronym: str, phrases: Iterable[str], limit: int) -> list[str]:
    """The distinct expansions (case-insensitively) of `acronym` over the visible titles and
    heading segments, at most `limit`, shortest first (the tightest reading of the acronym),
    then alphabetical for a reproducible order."""
    seen: dict[str, str] = {}
    for text in phrases:
        for match in phrase_matches(acronym, text):
            seen.setdefault(match.casefold(), match)
    return sorted(seen.values(), key=lambda m: (len(m), m.casefold()))[:limit]


def has_word(word: str, phrases: Iterable[str]) -> bool:
    """`word` occurs as a whole word (case-insensitively) in some phrase: it has a title or
    heading hit of its own, so it is not read as an acronym."""
    target = word.casefold()
    return any(
        w.casefold() == target
        for text in phrases
        for part in _PHRASE_BREAK.split(text)
        for w in _WORD_BREAK.split(part)
    )


def expanded_topics(topic: str, phrases: Sequence[str], limit: int) -> list[str]:
    """The topic with each acronym-shaped word that has no hit of its own replaced by one of
    its expansions: "HLD" -> ["High Level Design"], "HLD basics" -> ["High Level Design
    basics"]. At most `limit` topics; [] when nothing expands."""
    words = topic.split()
    topics: list[str] = []
    for acronym in acronym_words(topic):
        if has_word(acronym, phrases):
            continue
        for phrase in expand(acronym, phrases, limit):
            replaced = " ".join(phrase if w == acronym else w for w in words)
            topics.append(replaced)
    return list(dict.fromkeys(topics))[:limit]
