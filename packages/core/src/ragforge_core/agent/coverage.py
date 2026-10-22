"""Coverage: the verification guarantee of ADR-0015, stated as an ALLOWLIST (fail closed).

The guarantee: an answer may be `verified` only if its RESIDUE is empty, where the residue is
the WHOLE answer (prose, code at any nesting depth, any script) minus
  (a) the spans covered by a valid model citation: a label covers ONLY the text from the previous
      boundary (sentence end, line break, cell, HTML or previous label) up to and including it;
  (b) the spans covered by an inferred citation that passed the ground check;
  (c) the spans that exactly match a small closed allowlist of non-claim text (`allowed`) or
      that state nothing (`structure.states_nothing`): whole-sentence pleasantries and lead-ins,
      whole-sentence refusals, markdown structure.
Residue that says something, or any code, can never be `verified`.

WHERE the units come from (review of the redesign, round 4): `structure.structure`, a real
CommonMark parse. This module segments ONLY the visible text of each paragraph, heading and
table cell the parser found; code blocks (fenced or indented, at any depth), HTML blocks and text
the parser hides (reference definitions, dropped cells) are units of their own. There is no
second grammar here: no fence, indentation, backtick, list, task-box, strikethrough or bare-tag
rule. What the parser shows but this module cannot classify is an opaque unit (residue unless a
model label covers it; never inferred, never allowlisted). Three small text rules remain, and
each can only ADD a cut or REJECT a label, never make text state nothing (ADR-0015 section 4,
"What is not the parser's"): `_ENUMERATOR`, `_TAG_LIKE` and the unterminated "<!--".

How visible text is cut (fail closed: an extra cut only adds residue):
- a sentence ends at EVERY Unicode Sentence_Terminal character (plus the ellipsis, dot leaders,
  the Greek question mark and a few vertical/Tibetan forms), whatever follows it; closing quotes
  and brackets right after it stay with it. A semicolon (any script) ends a clause the same way.
  The ONLY exemptions are a single "." inside a token (digit.digit; lower.lower in an
  identifier; an interior dot of a dotted name the SHOWN sources hold verbatim, `dotted_names`,
  except a lone dot before a capital, "fast.It") and the final dot of "e.g."/"i.e.";
- every "|" (a cell boundary or a visual separator), every line break (soft or hard, and the
  Unicode line separators), and both sides of HTML (parsed inline HTML, or text shaped like a
  tag, which the sanitizer shows literally) are boundaries; only HTML the PARSER recognised as
  bare break/block tags states nothing (`structure.bare_html`);
- a code span, an autolink, a linkified URL and an image are atomic: a cut never falls inside
  one, so "`a. b` [S1]" is ONE unit and its code must be whole and exact in S1; a URL whose
  text ends in a terminator or closer still ends the sentence right after it;
- within a sentence, every canonical label `[S1]`/`[T2]`/`[S1, S2]` that the reader sees as a
  label ends a unit; the text after the last label is its own unit unless it states nothing;
- a leading enumerator ("(7)", "7)", "a.") that is not a list numeral is cut off as its own unit
  (it may BE the answer), which nothing covers.

Where a label counts (`_labels`): only in plain visible text whose source is literally the label
(`structure.Atom.eligible`: not in code, a link's text or destination, an autolink, a linkified
URL, strikethrough, HTML, an escape or an entity), not inside a URL-shaped word (a renderer may
linkify it). Strikethrough is the parser's GFM reading (one or two tildes); a part with a run of
three or more tildes that markdown-it's default reading strikes has no label at all.

A run of labels with no words before it cites the unit right before it (at most
`_MAX_LABEL_RUNS` runs) ONLY when the reader sees them together: in the same paragraph, heading
or cell with nothing but whitespace, terminators, closers and pipes between, across at most one
soft line break; in the paragraph right after a code block (the next line, same container); or
in the cell right after the cell it cites. It never cites a unit that is HTML, tag-shaped text
or a leading enumerator. Anywhere else it is an orphan and covers nothing.

CPU-cheap and free of model calls: one parse per distinct text (cached), and every scan here is
linear in the text (`test_ag_perf` bounds 20,000-character degenerate drafts).
"""

from __future__ import annotations

import html
import re
import unicodedata
from bisect import bisect_left, bisect_right
from collections.abc import Collection, Iterable, Iterator
from dataclasses import dataclass, replace
from enum import StrEnum
from functools import lru_cache
from itertools import pairwise
from typing import Final

from ragforge_core.agent.labels import CITATION_MARKER, label_names
from ragforge_core.agent.structure import (
    Atom,
    Block,
    Code,
    Inline,
    Kind,
    Part,
    bare_html,
    says_something,
    states_nothing,
    structure,
)

__all__ = [
    "APOSTROPHES",
    "TERMINATORS",
    "Allowed",
    "Run",
    "Unit",
    "allowed",
    "clean",
    "covered_by_model",
    "dotted_names",
    "ends_sentence",
    "mixed_script",
    "opaque",
    "refusal_only",
    "residue",
    "says_something",
    "segment",
    "sentence_ends",
    "source_sentences",
    "states_nothing",
    "strip_end",
    "valid_labels",
    "visible_runs",
]


# ---------------------------------------------------------------- text normalisation
_NON_ASCII: Final = re.compile(r"[^\x00-\x7f]")


def _code_points(spec: str) -> frozenset[str]:
    chars: set[str] = set()
    for item in spec.split():
        lo, _, hi = item.partition("-")
        chars.update(chr(c) for c in range(int(lo, 16), int(hi or lo, 16) + 1))
    return frozenset(chars)


# Unicode 15.0 Default_Ignorable_Code_Point (DerivedCoreProperties.txt): code points a renderer
# shows as nothing (or as a blank) when it has no glyph for them. Review of the parser rewrite,
# round 1 (P2): stripping only Cf left the Hangul fillers (Lo), the combining grapheme joiner and
# variation selectors (Mn) in a word, so "no\u3164t" hid a negation. A test checks every Cf
# character is stripped too (the two sets overlap but neither contains the other).
DEFAULT_IGNORABLE: Final = _code_points(
    """
    00AD 034F 061C 115F-1160 17B4-17B5 180B-180F 200B-200F 202A-202E 2060-206F 3164 FE00-FE0F
    FEFF FFA0 FFF0-FFF8 1BCA0-1BCA3 1D173-1D17A E0000-E0FFF
    """
)
# Ignorable LETTERS (the Hangul fillers): a tokenizer reads them as part of a word, a renderer
# shows a blank. Never legitimate inside an answer's word, so text holding one is opaque.
_FILLERS: Final = frozenset(c for c in DEFAULT_IGNORABLE if unicodedata.category(c)[0] == "L")


def _drop_ignorable(match: re.Match[str]) -> str:
    char = match.group(0)
    return "" if char in DEFAULT_IGNORABLE or unicodedata.category(char) == "Cf" else char


def clean(text: str) -> str:
    """NFKC with every format character (Cf: zero-width space/joiner, soft hyphen, BOM, tags) and
    every other default-ignorable code point (Hangul fillers, combining grapheme joiner,
    variation selectors) removed, before and after normalising. WHY (review of the redesign,
    round 4; parser rewrite, round 1): "not" with a zero-width space or a Hangul filler inside
    renders as "not" (or "no t"), but a word tokenizer sees other words; every word-level reading
    (allowlist, evidence, negation) reads the text as the reader sees it."""
    if text.isascii():
        return text  # NFKC is the identity and there is no ignorable character
    stripped = _NON_ASCII.sub(_drop_ignorable, text)
    return _NON_ASCII.sub(_drop_ignorable, unicodedata.normalize("NFKC", stripped))


_WORD: Final = re.compile(r"\w+(?:'\w+)?")
# Every character read as the apostrophe of a word ("doesn't"), folded BEFORE `clean` (NFKC turns
# the acute accent U+00B4 into a space and a combining mark, which would split the word): the
# real ones (U+2019, U+2018, U+02BC, U+201B, U+FF07) and the look-alikes a reader takes for one
# (blind review round 3, P2: "doesn\u00b4t", "doesn\u2032t" and "doesn`t" read as "doesn" + "t",
# so the negation was missed): acute and grave accents (U+00B4, U+0060, U+FF40, U+02CA, U+02CB),
# primes (U+2032, U+2035, U+02B9), modifier commas and the vertical line (U+02BB, U+02BD,
# U+02C8), the saltillo (U+A78C), Greek tonos/oxia/varia (U+0384, U+1FFD, U+1FEF), the Armenian
# apostrophe (U+055A) and the Hebrew geresh (U+05F3). Exactly this list; any OTHER modifier,
# symbol or punctuation character between two Latin letters makes text opaque (`_joined`).
APOSTROPHES: Final = str.maketrans(
    dict.fromkeys(
        "\u2019\u2018\u02bc\u201b\uff07\u00b4\u0060\uff40\u02ca\u02cb\u2032\u2035\u02b9"
        "\u02bb\u02bd\u02c8\ua78c\u0384\u1ffd\u1fef\u055a\u05f3",
        "'",
    )
)
_JOINER_CATEGORIES: Final = frozenset({"Lm", "Sk", "Po"})
# Alphabets with letters that look like Latin ones (Unicode confusables): a word mixing two of
# them ("not" spelled with a Greek omicron, "Kafka" with a Cyrillic a) reads as one word to the
# reader and as no known word to a tokenizer.
_CONFUSABLE_SCRIPTS: Final = frozenset({"LATIN", "GREEK", "CYRILLIC", "ARMENIAN", "CHEROKEE"})


@lru_cache(maxsize=4096)
def _script(char: str) -> str:
    name = unicodedata.name(char, "")
    return name.split(" ", 1)[0] if name else ""


def _latin_letter(char: str) -> bool:
    return char.isalpha() and _script(char) == "LATIN"


def _joined(text: str) -> bool:
    """A non-ASCII modifier letter, modifier symbol or punctuation character that is not a folded
    apostrophe sits between two Latin letters ("doesn\u02eet"): a reader may take it for an
    apostrophe, a tokenizer splits or joins the word, so the text is opaque (fail closed)."""
    stripped = _NON_ASCII.sub(_drop_ignorable, text)
    return any(
        not char.isascii()
        and unicodedata.category(char) in _JOINER_CATEGORIES
        and char.translate(APOSTROPHES) == char
        and _latin_letter(stripped[i - 1])
        and _latin_letter(stripped[i + 1])
        for i, char in enumerate(stripped[1:-1], start=1)
    )


def mixed_script(word: str) -> bool:
    """`word` mixes letters of two confusable alphabets: it is opaque (it cannot carry lexical
    evidence, and it matches no allowlisted phrase). Latin next to CJK is not confusable."""
    scripts = {_script(c) for c in word if c.isalpha()}
    return len(scripts & _CONFUSABLE_SCRIPTS) >= 2


# Bidirectional formatting controls (embeddings, overrides, isolates, marks). Unlike invisible
# joiners they CHANGE WHAT THE READER SEES: an RLO displays "ton" as "not" and "9092" as "2909".
# Stripping them (as `clean` does for true fillers) would check the logical string while the
# reader sees another, so text holding one is never evidence and never covered (review of the
# parser rewrite, round 3, P1). Fail closed: such a unit is residue even under a model label.
BIDI_CONTROLS: Final = frozenset(
    "\u061c\u200e\u200f\u202a\u202b\u202c\u202d\u202e\u2066\u2067\u2068\u2069"
)


# R and AL letters, and AN (Arabic-Indic digits): "\u0661\u0660 \u0662\u0660" can display with
# its two numbers swapped without any R/AL letter (review round 5, P2).
_RTL: Final = frozenset({"R", "AL", "AN"})


def has_rtl(text: str) -> bool:
    """Any right-to-left character (Hebrew, Arabic, ...: bidi class R or AL). The Unicode bidi
    algorithm may then DISPLAY the line in another order than the logical one the checker reads:
    "ports are \u05d0 9092 9093" can show "9093 9092", and under first-strong paragraph direction
    whole phrases swap ("Postgres ... faster than ... MySQL"), with no control character
    involved (review rounds 2 and 3). Fail closed like a bidi control: the answer is residue.
    The owner's notes are English; an RTL corpus would need a rendered-order check instead."""
    if not text.isascii() and any(unicodedata.bidirectional(c) in _RTL for c in text):
        return True
    # written as a character reference ("&#x5D0;"), as for bidi controls (review round 4, P2)
    if "&" not in text:
        return False
    decoded = html.unescape(text)
    return any(unicodedata.bidirectional(c) in _RTL for c in decoded)


# Characters that render as blank but are not spaces: a renderer shows a gap where the checker
# sees text, so a look-alike terminator before one ("1850\u2e30\u2800Kafka") hid the sentence
# end (review round 9, P2). Fail closed like a bidi control: the answer is residue.
BLANK_GLYPHS: Final = frozenset("\u2800\u115f\u1160\u3164\uffa0")


def has_blank_glyph(text: str) -> bool:
    return not text.isascii() and any(c in BLANK_GLYPHS or c in _FILLERS for c in text)


def has_bidi(text: str) -> bool:
    """A bidi control, raw or written as an HTML character reference ("&#x202E;", "&rlm;"):
    the checker reads entities as the characters they stand for, and a renderer that does not
    escape "&" (unlike sanitize.py) would show the reversal (review of Phase 2b, P2)."""
    if not text.isascii():
        return any(c in BIDI_CONTROLS for c in text) or ("&" in text and _bidi_ref(text))
    return "&" in text and _bidi_ref(text)


def _bidi_ref(text: str) -> bool:
    return any(c in BIDI_CONTROLS for c in html.unescape(text))


def opaque(text: str) -> bool:
    """`text` says something the word tokenizer cannot read faithfully: a substantive character
    left after removing every word ("⠝⠁⠎⠁", "🇳🇦", "€"), a mixed-script word, a Hangul filler
    (`_FILLERS`), or a word joined by an apostrophe look-alike that is not folded (`_joined`).
    Lexical evidence and the allowlist are blind to such text, so neither may account for it
    (fail closed)."""
    if not text.isascii() and (any(c in _FILLERS for c in text) or _joined(text)):
        return True
    if has_bidi(text):
        return True
    cleaned = clean(text)
    if any(mixed_script(w) for w in _WORD.findall(cleaned)):
        return True
    return says_something(_WORD.sub(" ", cleaned))


# ---------------------------------------------------------------- sentence boundaries
# Unicode 15.0 Sentence_Terminal (PropList.txt), united with every punctuation character whose
# name says FULL STOP / QUESTION MARK / EXCLAMATION MARK / DANDA / ELLIPSIS / INTERROBANG, plus
# 2024-2026 (dot leaders, ellipsis), 037E (Greek question mark), FE12/FE15/FE16 (vertical forms)
# and 0F0D-0F12 (Tibetan shads). A superset on purpose: an extra cut only adds residue. A test
# rescans unicodedata and fails if a punctuation character named like a terminator is missing.
TERMINATORS: Final = _code_points(
    """
    0021 002E 003F 00A1 00BF 037E 055C 055E 0589 061D-061F 06D4 0700-0702 07F9 0837 0839
    083D-083E 0964-0965 0F0D-0F12 104A-104B 1362 1367-1368 166E 1735-1736 1801 1803 1809
    1944-1945 1AA8-1AAB 1B5A-1B5B 1B5E-1B5F 1B7D-1B7E 1C3B-1C3C 1C7E-1C7F 2024-2026 203C-203D
    2047-2049 2CF9-2CFB 2CFE 2E18 2E2E 2E3C 2E53-2E54 3002 A4FF A60E-A60F A6F3 A6F7 A876-A877
    A8CE-A8CF A92F A9C8-A9C9 AA5D-AA5F AAF0-AAF1 ABEB FE12 FE15-FE16 FE19 FE52 FE56-FE57 FF01
    FF0E FF1F FF61 10A56-10A57 10F55-10F59 10F86-10F89 11047-11048 110BE-110C1 11141-11143
    111C5-111C6 111CD 111DE-111DF 11238-11239 1123B-1123C 112A9 1144B-1144C 115C2-115C3
    115C9-115D7 11641-11642 1173C-1173E 11944 11946 11A42-11A43 11A9B-11A9C 11C41-11C42
    11EF7-11EF8 11F43-11F44 16A6E-16A6F 16AF5 16B37-16B38 16B44 16E98 1BC9F 1DA88 1E95E-1E95F
    """
)
# A semicolon joins two independent clauses, so it ends the clause a label can cover ("NASA built
# Kafka in 1850;Kafka stores records [S1]." must not let S1 cover 1850). Any script's semicolon.
CLAUSE_ENDS: Final = frozenset(";\uff1b\u061b\u204f\u0387\ufe54\ufe14")
# Characters a reader takes for a full stop although Unicode does not call them one (middle
# dots, dot operators, the Lisu tone letter U+A4F8): "NASA built Kafka in 1850<U+A4F8> Kafka
# stores records [S1]." reads as two sentences, so S1 must not cover the first (Phase 2b, P2).
DOT_LOOKALIKES: Final = frozenset(
    "\u00b7\u0701\u0702\u16eb\u1427\u2027\u2219\u22c5\u2e31\u30fb\ua4f8\ua4f9\uff65"
)
_ENDS: Final = TERMINATORS | CLAUSE_ENDS | DOT_LOOKALIKES
_DOT_WORDS: Final = (
    "DOT",
    "SPOT",
    "POINT",
    "STOP",
    "BULLET",
    "QUESTION MARK",
    "EXCLAMATION MARK",
    "THEREFORE",
    "ELLIPSIS",
)


@lru_cache(maxsize=4096)
def _dot_like(char: str) -> bool:
    """A non-ASCII character that, followed by a space, a reader may take for a sentence end:
    any other-punctuation (Po: raised dot, bullet, sof pasuq, fongman, tsheg, ...) or a mark,
    modifier or symbol whose name says it is a dot ("Z NOTATION SPOT", "DOT ABOVE", "COMBINING
    DOT BELOW") or a question/exclamation mark, ellipsis or "therefore" (the emoji \u2757 and
    \u2753, \u22ef, \u2234: review round 4, P2). A closed list missed eleven of these (round
    2), so this is a rule over Unicode data. An extra cut only adds residue (fail safe);
    letters are exempt so words are never cut, and arrows and operators are not punctuation."""
    if char.isascii():
        return False
    category = unicodedata.category(char)
    if category == "Po":
        return True
    if category[0] == "L" and category != "Lm":
        return False
    return any(w in unicodedata.name(char, "") for w in _DOT_WORDS)


def _foreign_tail(text: str, i: int) -> bool:
    """A non-ASCII character that is not a cased letter and does not continue a non-ASCII word:
    "1850\u01c3", "(1850)\u01c3", "1850 \u0294" (review rounds 5 and 6, P2: U+01C3 reads as
    "!", U+0294 as "?", Arabic-Indic zero as "."; a rule after three closed-list misses, then
    widened beyond "right after an ASCII letter or digit"). In the owner's English notes such a
    character before a space is far likelier a disguised terminator than text. Exempt: cased
    letters ("caf\u00e9") and a character right after a non-ASCII letter (a CJK or Thai word
    keeps its own characters). `_ends_at` adds "followed by a visible space"; an extra cut only
    adds residue."""
    char = text[i]
    category = unicodedata.category(char)
    # Not: cased letters, our own placeholder, invisible format/control characters, combining
    # marks (they attach to the character before; dot-shaped ones are `_dot_like` by name).
    if (
        char.isascii()
        or char == _PLACEHOLDER
        or category in ("Ll", "Lu", "Lt", "Mn", "Me")
        or category[0] == "C"
    ):
        return False
    prev = text[i - 1] if i > 0 else " "
    if prev.isspace():
        # free-standing ("1850 \u01c3 Kafka"): only letters and digits posing as marks; a
        # symbol between spaces ("3 \u00d7 4", an arrow, an emoji) is not a terminator
        return category in ("Lo", "Lm", "Nd", "No")
    return prev.isascii() or not unicodedata.category(prev).startswith("L")


def _ends_inside(code: str) -> bool:
    """The span's text ends a sentence: a terminator, possibly followed by closers, emphasis
    marks or invisible characters ("`2017.`", "`1850.)`", "`x."`": review rounds 8 and 9)."""
    i = len(code.rstrip())
    while i > 0 and (
        _closes(code[i - 1]) or code[i - 1] in "*_~" or unicodedata.category(code[i - 1]) == "Cf"
    ):
        i -= 1
    return i > 0 and ends_sentence(code[i - 1])


def ends_sentence(char: str) -> bool:
    """`char` ends a sentence wherever it stands: a terminator, a clause end or a look-alike.
    (`_dot_like`/`_foreign_tail` also need a following space; at the end of a code span the span
    edge is that boundary.)"""
    return char in _ENDS or _dot_like(char) or (not char.isascii() and _foreign_tail(char, 0))


def _ends_at(text: str, i: int, end: int) -> bool:
    """`text[i]` ends a sentence: a terminator, a clause end, a listed look-alike, or a
    `_dot_like` or `_foreign_tail` character followed by whitespace or the end of the text."""
    char = text[i]
    if char in _ENDS:
        return True
    if not (_dot_like(char) or _foreign_tail(text, i)):
        return False
    # What the reader sees after it: skip invisible format characters (ZWSP, word joiner, soft
    # hyphen: category Cf), emphasis/strike/code delimiters and closers (review round 3, P2:
    # "1850\u02d9\u200b Kafka" and "1850**\u02d9** Kafka" hid the end).
    j = i + 1
    while j < end and (
        unicodedata.category(text[j]) == "Cf" or text[j] in "*_~`" or _closes(text[j])
    ):
        j += 1
    return j >= end or text[j].isspace()


_ABBREVIATIONS: Final = ("e.g.", "i.e.")
# A closed list of the extensions, TLDs and abbreviation letters that follow a lower.lower dot
# inside an identifier ("node.js", "postgresql.conf", "e.g").
_EXTENSIONS: Final = frozenset(
    """
    app cfg com conf cpp css csv db dev env g e go h html ini io java js json jsx kt lock log md
    net org php pl py rb rs scss sh so sql svg ts tsx txt xml yaml yml zip
    """.split()  # noqa: SIM905
)
_CHAIN: Final = "._/(`"


def _closes(char: str) -> bool:
    """A closing quote or bracket that belongs to the sentence a terminator just ended."""
    return char in "\"'" or unicodedata.category(char) in ("Pe", "Pf")


def _identifier_dot(text: str, i: int, end: int) -> bool:
    """The lower.lower dot at `i` belongs to an identifier: the word after it is a known
    extension, or the token goes on after that word ("os.path.join("). Only what FOLLOWS the
    dot counts (a chain before it let "node.js.it runs ..." hide a sentence end)."""
    j = i + 1
    while j < end and text[j].isalnum():
        j += 1
    return text[i + 1 : j].casefold() in _EXTENSIONS or (j < end and text[j] in _CHAIN)


def _dot_in_token(text: str, i: int, start: int, end: int) -> bool:
    """The single "." at `i` is inside a token, not a sentence end: "3.5", an identifier
    (`_identifier_dot`), or the final dot of "e.g."/"i.e.". The only exemptions."""
    prev = text[i - 1] if i > start else ""
    nxt = text[i + 1] if i + 1 < end else ""
    if prev.isdigit() and nxt.isdigit():
        return True
    if prev.islower() and nxt.islower():
        return _identifier_dot(text, i, end)
    word = text[max(start, i - 3) : i + 1]
    before = text[i - 4] if i - 4 >= start else ""
    return word.casefold() in _ABBREVIATIONS and not before.isalnum()


# ---------------------------------------------------------------- source-backed dotted names
_NAME_CHARS: Final = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_")
_NAME_RUN: Final = re.compile(r"[A-Za-z0-9_.]+")
# what may follow an exempt dot: never a capital ("fast.It", "2.8.It", "U.S.The": `_NamedRuns`)
_AFTER_DOT: Final = frozenset("abcdefghijklmnopqrstuvwxyz0123456789_")


def _shown_pieces(text: str) -> Iterator[str]:
    """The text a reader sees in a page, in pieces no name may span: each inline part with its
    emphasis marks dropped and its escapes and entities resolved (what `_view` shows of an
    answer), each code span, URL and code block on its own.

    WHY the parsed view (review round 2, P2): the answer side is matched on its visible text, so
    the source side must be too. On raw Markdown "max.in.**flight**.requests" gave the partial
    names "max.in" and "requests" and "max\\_poll.ms" gave "poll.ms": names the reader never
    sees as a whole run, while the name the reader DOES see was missing. Code is kept: a config
    key is most often written in a code span ("`{logger.name}`"), and the reader sees it."""
    for part in structure(text).parts:
        if isinstance(part, Block):
            if part.code is not None:
                yield part.code.text
            continue  # HTML or opaque: nothing a reader sees as text
        shown: list[str] = []
        for atom in part.atoms:
            if atom.kind in (Kind.text, Kind.escape):
                shown.append(atom.visible)
            elif atom.kind is not Kind.mark:  # a mark shows nothing: the run continues
                shown.append(" ")  # every other atom bounds the run
                if atom.kind is Kind.code and atom.code is not None:
                    yield atom.code.text
                elif atom.kind is Kind.url:
                    yield atom.visible
        yield "".join(shown)


def dotted_names(texts: Iterable[str]) -> frozenset[str]:
    """Every maximal ASCII `[A-Za-z0-9_.]` run of what a reader sees in `texts` (Markdown pages,
    `_shown_pieces`) that, with its leading and trailing dots stripped, still holds a dot:
    "max.in.flight.requests.per.connection", "logger.name".

    The caller passes the SHOWN `S` sources only, never `T` tool results: a tool result is text
    the model can steer, so a name from one would let the model choose which of its own dots
    stop ending sentences. Verbatim on purpose (case-sensitive, no NFKC): the exemption it feeds
    (`_NamedRuns`) must only ever accept a run the reader can find, character for character, in
    a page they were shown."""
    names: set[str] = set()
    for text in texts:
        for piece in _shown_pieces(text):
            for match in _NAME_RUN.finditer(piece):
                name = match.group(0).strip(".")
                if "." in name:
                    names.add(name)
    return frozenset(names)


class _NamedRuns:
    """The source-backed dotted-name exemption of `sentence_ends` (DOT, 2026-10-19).

    WHY: the identifier rule (`_identifier_dot`) only looks AFTER a dot, so the LAST dot of a
    dotted name followed by a space cut: "max.in.flight.requests.per.connection to 1 [S2]." and
    "/actuator/loggers/{logger.name} with ... [S1]." each became an uncited fragment, flagged
    "Not found in your notes" although the notes say exactly that. Widening that rule by shape
    alone cannot be made safe ("Kafka is fast.and.it never loses data" has the shape of a
    dotted name and hides a sentence end), so the evidence is the source: an interior ASCII "."
    followed by `[A-Za-z0-9_]` is in-token when the maximal `[A-Za-z0-9_.]` run around it,
    leading/trailing dots stripped, is one of `names` (`dotted_names` of the shown sources).

    A capital right after the dot is a sentence start a reader DOES see ("Redis is fast.It keeps
    data", "Kafka 2.8.It added KRaft", "the U.S.The company"): a missing space that pasted or
    PDF-extracted notes are full of, and that a shared page could plant. So only a dot followed
    by `[a-z0-9_]` is ever exempt, whatever the run's dot count (review round 1 and round 2, P1:
    exempting a capital in a 2+-dot run let "..., as 2.8.It added KRaft mode [S1]." carry an
    unbacked clause). "java.util.List" therefore keeps its cut before "List", exactly as it did
    before this rule (an extra cut is the safe side). A lowercase typo ("durable.it") is still
    exempt when a source holds it verbatim: the reader sees neither a space nor a capital.

    It only ever REMOVES a cut that the source itself shows as one token, never one followed by
    whitespace or the end of the text ("see a.b. Then" still cuts), and every other rule stays
    ahead of it (a dot run, a homoglyph dot, `_dot_like`, bidi and non-ASCII handling are
    untouched). One run is cached at a time: `sentence_ends` walks left to right and runs are
    disjoint, so every character is scanned O(1) times (`test_ag_perf`'s linear bound)."""

    __slots__ = ("_end", "_names", "_run", "_start", "_text")

    def __init__(self, text: str, start: int, end: int, names: frozenset[str]) -> None:
        self._text, self._start, self._end, self._names = text, start, end, names
        self._run: tuple[int, int, int, int, bool] = (-1, -1, -1, -1, False)

    def exempt(self, i: int) -> bool:
        """The ASCII "." at `i` is inside a source-backed dotted name."""
        text = self._text
        if i + 1 >= self._end or text[i + 1] not in _AFTER_DOT:
            # a space, a closer, an emphasis mark, the end, or a capital (a sentence start a
            # reader sees: class docstring): cut
            return False
        lo, hi, a, b, known = self._run
        if not lo <= i < hi:
            lo, hi = i, i + 1
            while lo > self._start and (text[lo - 1] in _NAME_CHARS or text[lo - 1] == "."):
                lo -= 1
            while hi < self._end and (text[hi] in _NAME_CHARS or text[hi] == "."):
                hi += 1
            a, b = lo, hi
            while a < b and text[a] == ".":
                a += 1
            while b > a and text[b - 1] == ".":
                b -= 1
            known = text[a:b] in self._names
            self._run = (lo, hi, a, b, known)
        return known and a < i < b


def sentence_ends(
    text: str, start: int, end: int, *, names: frozenset[str] = frozenset()
) -> Iterator[int]:
    """The (exclusive) end of every sentence or semicolon clause in text[start:end].

    `names` (`dotted_names` of the shown `S` sources) exempts the interior dots of a dotted
    name the sources hold verbatim (`_NamedRuns`); empty, the cuts are exactly as before."""
    named = _NamedRuns(text, start, end, names) if names else None
    i = start
    while i < end:
        if not _ends_at(text, i, end):
            i += 1
            continue
        j = i + 1
        while j < end and _ends_at(text, j, end):
            j += 1
        if (
            j - i == 1
            and text[i] == "."
            and (_dot_in_token(text, i, start, end) or (named is not None and named.exempt(i)))
        ):
            i = j
            continue
        while j < end and _closes(text[j]):
            j += 1
        yield j
        i = j


def strip_end(text: str) -> str:
    """`text` without its final terminators and closing quotes/brackets."""
    end = len(text)
    while end and (text[end - 1] in _ENDS or _closes(text[end - 1])):
        end -= 1
    return text[:end]


# ---------------------------------------------------------------- the closed allowlist
class Allowed(StrEnum):
    """Why a unit is not residue although nothing cites it."""

    pleasantry = "pleasantry"
    refusal = "refusal"


# Matched against the WHOLE unit's words (cleaned, casefolded, apostrophes normalised, joined by
# single spaces): punctuation and emoji are ignored, every word counts. A unit that adds any
# word ("No, it doesn't cover that.", "Your notes don't cover this; no.") is not allowlisted.
# The list only grows with a test.
_OFFER: Final = (
    r"(?:(?:any|some)\s+)?(?:(?:other|more|further|additional|follow\s+up)\s+)?questions?"
)
_PLEASANTRY: Final = re.compile(
    r"\A(?:"
    r"(?:please\s+)?(?:let\s+me\s+know|feel\s+free\s+to\s+ask)"
    rf"(?:\s+if\s+you\s+(?:have|need)\s+(?:{_OFFER}|anything\s+else|more\s+(?:help|details)))?"
    r"|(?:i\s+)?hope\s+(?:this|that)\s+helps"
    r"|(?:i\s+am|i'm)\s+(?:happy|glad)\s+to\s+help"
    r"|happy\s+to\s+help"
    # lead-ins that announce an answer without stating anything (a line or sentence of its own)
    r"|here\s+is\s+(?:what\s+i\s+found|a\s+summary|what\s+your\s+notes\s+say)"
    r"|here's\s+(?:what\s+i\s+found|a\s+summary|what\s+your\s+notes\s+say)"
    r")\Z"
)
# The refusal the answer prompt asks for ("Your notes don't cover this.") and its close variants.
_REFUSAL: Final = re.compile(
    r"\A(?:"
    r"(?:your|my|the)\s+(?:notes|sources)\s+(?:don't|do\s+not)\s+cover\s+(?:this|that|it)"
    r"(?:\s+question)?"
    r"|i\s+(?:couldn't|could\s+not)\s+find\s+(?:this|that|it)\s+in\s+your\s+notes"
    r")\Z"
)


def fold(text: str) -> str:
    """The unit's words, apostrophes folded (`APOSTROPHES`), cleaned, casefolded, joined by
    single spaces."""
    return " ".join(_WORD.findall(clean(text.translate(APOSTROPHES)).casefold()))


def allowed(text: str) -> Allowed | None:
    """The closed allowlist (c). Whole-unit matches only, never by word; a unit with text the
    word tokenizer cannot read faithfully is not allowlisted (`opaque`)."""
    words = fold(text)
    if not words or opaque(text):
        return None
    if _REFUSAL.match(words):
        return Allowed.refusal
    if _PLEASANTRY.match(words):
        return Allowed.pleasantry
    return None


# ---------------------------------------------------------------- units
@dataclass(frozen=True, slots=True)
class Unit:
    """One coverage unit: a sentence piece ending at a label or a boundary, a code block, or an
    opaque part (HTML block, text the parser hides, a part it could not classify)."""

    start: int  # [start, end) in the answer; `text` is answer[start:end]
    end: int
    text: str
    code: bool = False  # a code block (`codes` holds it)
    labels: tuple[str, ...] = ()  # the labels terminating or citing it ("S1", "T2"), valid or not
    body: str = ""  # `text` with its real label markers and bare HTML tags blanked
    visible: str = ""  # what the reader sees (code spans and URLs included): lexical evidence
    codes: tuple[Code, ...] = ()  # every code element in it (a block, or its inline spans)
    opaque: bool = False  # never inferred, never allowlisted

    @property
    def substantive(self) -> bool:
        """Residue-bearing if uncovered: a character not positively known to state nothing, or
        any code (a fence's info string included). A bare label states nothing."""
        if self.code:
            block = self.codes[0]
            return bool(block.text.strip() or block.info.strip())
        return says_something(self.body) or any(c.text.strip() for c in self.codes)

    @property
    def allowed(self) -> Allowed | None:
        """Labels do not change what the words are: "Your notes don't cover this [S1]." (the
        strict prompt's refusal) is still a refusal, so it makes no claim and is never
        `verified`. Code and opaque text are never allowlisted."""
        if self.code or self.opaque or self.codes:
            return None
        return allowed(self.body)


_LEAD_PUNCT: Final = " \t;,:\u2013\u2014"
# A leading enumerator that is not an accepted list numeral: "(7)", "7)", "(a)", "b.": cut off
# as its own unit (review round 1: "(7) Kafka ... [S1]." made the 7 look covered).
_ENUMERATOR: Final = re.compile(r"(?:\(\w{1,3}\)|\w{1,3}[.)])(?=\s)")
_LINE_SEPARATORS: Final = frozenset("\x0b\x0c\x1c\x1d\x1e\x85\u2028\u2029")
_PLACEHOLDER: Final = "\ufffc"
# Text shaped like a tag that the parser did NOT take as HTML ("<p NASA built Kafka in 1850>"
# has an invalid attribute name; another renderer may still treat it as a tag): a boundary on
# both sides and no label inside it. It only ever ADDS a cut or REJECTS a label (fail closed); it
# never makes text state nothing (only a tag the parser recognised can: `structure.bare_html`).
_TAG_LIKE: Final = re.compile(r"<[ \t]*/?[ \t]*[A-Za-z][^<>\n]*>")
# A unit absorbs at most this many bare label runs ("A works. [S2] [S3]"); later runs are
# orphans (review round 2, P1: merging thousands of runs was quadratic).
_MAX_LABEL_RUNS: Final = 8


def _trim(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and (text[start] in _LEAD_PUNCT or text[start].isspace()):
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def _blank(text: str, spans: Iterable[tuple[int, int]], offset: int) -> str:
    chars = list(text)
    for a, b in spans:
        a, b = max(a - offset, 0), min(b - offset, len(chars))
        if a < b:
            chars[a:b] = " " * (b - a)
    return "".join(chars)


def _url_shaped(word: str) -> bool:
    """A word a renderer may turn into a link (the sanitizer's BARE_URL_PATTERN, as substring
    tests so it stays linear): a scheme, "www.", "mailto:"/"xmpp:", "//", or "@" then "."."""
    lower = word.casefold()
    at = lower.find("@")
    return (
        "://" in lower
        or "www." in lower
        or "mailto:" in lower
        or "xmpp:" in lower
        or "//" in lower
        or (at != -1 and "." in lower[at:])
    )


@dataclass(slots=True)
class _View:
    """The visible text of one inline part, each character mapped to its source span."""

    text: str
    starts: list[int]
    ends: list[int]
    owner: list[int]  # the atom each character comes from
    cuts: set[int]  # forced boundaries (indices into `text`)


def _view(part: Inline, names: frozenset[str] = frozenset()) -> _View:
    chars: list[str] = []
    starts: list[int] = []
    ends: list[int] = []
    owner: list[int] = []
    cuts: set[int] = set()
    for index, atom in enumerate(part.atoms):
        if atom.kind is Kind.mark:
            continue
        if atom.kind is Kind.text:
            chars.extend(atom.visible)
            starts.extend(range(atom.start, atom.end))
            ends.extend(range(atom.start + 1, atom.end + 1))
            owner.extend([index] * len(atom.visible))
            continue
        if atom.kind is Kind.escape:
            pieces = atom.visible or " "
        else:  # atomic: a code span, URL or image is never cut; HTML and breaks are boundaries
            pieces = "\n" if atom.kind in (Kind.soft, Kind.hard) else _PLACEHOLDER
            if atom.kind in (Kind.html, Kind.soft, Kind.hard):
                cuts.update((len(chars), len(chars) + 1))
            elif atom.kind is Kind.code and atom.code is not None and _ends_inside(atom.code.text):
                # review round 8, P1: "in `2017.` Kafka ... [S1]." reads as two sentences; the
                # terminator inside the span must end the first, exactly as for a URL
                cuts.add(len(chars) + 1)
            elif atom.kind is Kind.url and strip_end(atom.visible) != atom.visible:
                # found by the round-4 stress run: linkify swallowed a final "…", and the
                # sentence end it hid let one label cover the sentence before it
                cuts.add(len(chars) + 1)
        chars.extend(pieces)
        starts.extend([atom.start] * len(pieces))
        ends.extend([atom.end] * len(pieces))
        owner.extend([index] * len(pieces))
    text = "".join(chars)
    for match in _TAG_LIKE.finditer(text):
        cuts.update((match.start(), match.end()))
    for i, char in enumerate(text):
        if char == "|":
            cuts.add(i + 1)
        elif char in _LINE_SEPARATORS:
            cuts.update((i, i + 1))
    cuts.update(sentence_ends(text, 0, len(text), names=names))
    return _View(text, starts, ends, owner, cuts)


def _html_spans(part: Inline, view: _View) -> list[tuple[int, int]]:
    """The source spans of the part's parsed HTML and of its tag-shaped text, in order."""
    spans = [(a.start, a.end) for a in part.atoms if a.kind is Kind.html]
    spans += [
        (view.starts[m.start()], view.ends[m.end() - 1]) for m in _TAG_LIKE.finditer(view.text)
    ]
    merged: list[tuple[int, int]] = []
    for a, b in sorted(spans):
        if merged and a < merged[-1][1]:
            merged[-1] = (merged[-1][0], max(b, merged[-1][1]))
        else:
            merged.append((a, b))
    return merged


def _labels(part: Inline, view: _View) -> list[re.Match[str]]:
    """The canonical markers the reader sees as labels (module docstring)."""
    text = view.text
    words = [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]
    word_starts = [a for a, _ in words]
    # an unterminated comment hides the rest from an HTML-aware renderer, and a tag-shaped
    # chunk is HTML to some renderers: no label after the one or inside the other
    comment = text.find("<!--")
    tags = [(m.start(), m.end()) for m in _TAG_LIKE.finditer(text)]
    tag_starts = [a for a, _ in tags]
    urls = [i for i, k in enumerate(view.owner) if part.atoms[k].kind is Kind.url]
    found: list[re.Match[str]] = []
    for match in CITATION_MARKER.finditer(text):
        a, b = match.start(), match.end()
        atom = part.atoms[view.owner[a]]
        if view.owner[b - 1] != view.owner[a] or atom.kind is not Kind.text or not atom.eligible:
            continue
        if comment != -1 and comment < a:
            continue
        tag = bisect_right(tag_starts, a) - 1
        if tag >= 0 and tags[tag][1] > a:
            continue
        w0, w1 = words[bisect_right(word_starts, a) - 1]
        joined = bisect_left(urls, w0) < bisect_left(urls, w1)  # glued to a linkified URL
        if joined or _url_shaped(text[w0:w1]):
            continue  # inside a URL-shaped word: a renderer may linkify it
        found.append(match)
    return found


class _Builder:
    """Builds the units of one document (`segment`)."""

    def __init__(self, answer: str, names: frozenset[str] = frozenset()) -> None:
        self.answer = answer
        self.names = names  # `dotted_names` of the shown sources (`sentence_ends`)
        self.units: list[Unit] = []
        self.runs = 0  # bare label runs merged into units[-1] (`_MAX_LABEL_RUNS`)
        self.previous: Part | None = None
        self.atom_starts: list[int] = []  # of the inline part being built (bisection)
        self.spans: list[tuple[int, int]] = []  # its labels' source spans, in order
        self.span_starts: list[int] = []
        # source spans of the current part's HTML and tag-shaped text, sorted, non-overlapping
        self.html: list[tuple[int, int]] = []
        self.html_starts: list[int] = []
        self.enumerators: set[int] = set()  # indexes in `units` of leading-enumerator units

    def _spans_in(self, s: int, e: int) -> list[tuple[int, int]]:
        """The label spans of the current part inside [s, e), by bisection."""
        return self.spans[bisect_left(self.span_starts, s) : bisect_left(self.span_starts, e)]

    def _atoms(self, part: Inline, s: int, e: int) -> Iterator[Atom]:
        """The atoms of `part` overlapping [s, e), by bisection (linear overall)."""
        index = max(bisect_right(self.atom_starts, s) - 1, 0)
        while index < len(part.atoms) and part.atoms[index].start < e:
            if part.atoms[index].end > s:
                yield part.atoms[index]
            index += 1

    # ------------------------------------------------------------ blocks
    def block(self, part: Block) -> None:
        text = self.answer[part.start : part.end]
        if part.code is not None:
            self.units.append(Unit(part.start, part.end, text, code=True, codes=(part.code,)))
        else:
            body = " " * len(text) if part.kind == "html" and bare_html(text) else text
            self.units.append(
                Unit(part.start, part.end, text, body=body, visible=text, opaque=True)
            )
        self.runs = 0

    # ------------------------------------------------------------ inline parts
    def inline(self, part: Inline) -> None:
        self.atom_starts = [a.start for a in part.atoms]
        view = _view(part, self.names)
        labels = _labels(part, view)
        self.spans = [(view.starts[m.start()], view.ends[m.end() - 1]) for m in labels]
        self.span_starts = [a for a, _ in self.spans]
        self.html = _html_spans(part, view)
        self.html_starts = [a for a, _ in self.html]
        bounds = sorted({0, *(c for c in view.cuts if 0 < c < len(view.text))})
        bounds.append(len(view.text))
        first_unit = len(self.units)
        marks = iter(labels)
        pending = next(marks, None)
        for a, b in pairwise(bounds):
            src_a = part.start if a == 0 else view.starts[a]
            src_b = part.end if b == len(view.text) else view.starts[b]
            pieces: list[tuple[int, int, tuple[str, ...]]] = []
            cursor = src_a
            while pending is not None and pending.end() <= b:
                if pending.start() >= a:
                    end = view.ends[pending.end() - 1]
                    pieces.append((cursor, end, tuple(label_names(pending.group(0)))))
                    cursor = end
                pending = next(marks, None)
            a_tail, b_tail = _trim(self.answer, cursor, src_b)
            if not pieces or says_something(self.answer[a_tail:b_tail]):
                pieces.append((cursor, src_b, ()))
            elif a_tail < b_tail:  # nothing but punctuation or a pictograph: the label's unit
                pieces[-1] = (pieces[-1][0], b_tail, pieces[-1][2])
            for index, (s, e, names) in enumerate(pieces):
                s, e = _trim(self.answer, s, e)
                if index == 0:
                    enum = _ENUMERATOR.match(self.answer, s, e)
                    if enum is not None:
                        count = len(self.units)
                        self._add(part, s, enum.end(), (), first_unit)
                        if len(self.units) > count:
                            self.enumerators.add(count)
                        s, e = _trim(self.answer, enum.end(), e)
                if s < e or names:
                    self._add(part, s, e, names, first_unit)

    def _make(self, part: Inline, s: int, e: int, names: tuple[str, ...]) -> Unit:
        atoms = list(self._atoms(part, s, e))
        text = self.answer[s:e]
        bare = [(a.start, a.end) for a in atoms if a.kind is Kind.html and bare_html(a.visible)]
        body = _blank(text, [*self._spans_in(s, e), *bare], s)
        visible: list[str] = []
        for atom in atoms:
            if atom.kind is Kind.text:
                visible.append(self.answer[max(atom.start, s) : min(atom.end, e)])
            elif atom.kind is not Kind.mark:
                visible.append(atom.visible)
        codes = tuple(a.code for a in atoms if a.code is not None)
        return Unit(
            s,
            e,
            text,
            labels=names,
            body=body,
            visible="".join(visible),
            codes=codes,
            opaque=any(a.opaque for a in atoms),
        )

    def _add(
        self,
        part: Inline,
        s: int,
        e: int,
        names: tuple[str, ...],
        first_unit: int,
    ) -> None:
        unit = self._make(part, s, e, names)
        if unit.substantive:
            self.units.append(unit)
            self.runs = 0
            return
        if not names:
            return  # states nothing (punctuation, a pictograph, a bare tag)
        inside = self._spans_in(s, e)
        label_at = inside[0][0] if inside else s
        target = self._cited(part, label_at, first_unit)
        if target is None or self.runs >= _MAX_LABEL_RUNS:
            self.units.append(unit)  # an orphan label run: covers nothing, but is validated
            self.runs = _MAX_LABEL_RUNS
            return
        self.runs += 1
        prev = self.units[-1]
        labels = (*prev.labels, *names)
        if prev.code or prev.start < part.start:
            self.units[-1] = replace(prev, labels=labels)
        else:
            self.units[-1] = self._make(part, prev.start, unit.end, labels)

    def _has_html(self, unit: Unit) -> bool:
        """`unit` holds HTML or tag-shaped text of the current part (`_html_spans`)."""
        index = bisect_left(self.html_starts, unit.end) - 1
        return index >= 0 and self.html[index][1] > unit.start

    def _cited(self, part: Inline, label_at: int, first_unit: int) -> Unit | None:
        """The unit a bare label run cites, or None (an orphan): see the module docstring.
        `label_at` is where its first label starts: anything between the cited unit and it (an
        emoji, a bare tag, a hard break) makes it an orphan.

        A unit that is a leading enumerator, or that holds HTML (parsed or tag-shaped), is
        never cited by a bare run (blind review round 2, P2): an HTML-aware renderer hides a
        comment or a tag's attributes, and the reader then sees the label as citing the sentence
        before it; "3) [S1]" would let a label cover a bare enumerator that may BE the answer."""
        if not self.units:
            return None
        prev = self.units[-1]
        if len(self.units) - 1 in self.enumerators or self._has_html(prev):
            return None
        if len(self.units) > first_unit:  # same paragraph, heading or cell
            gap = self.answer[prev.end : label_at]
            hard = any(
                a.kind in (Kind.hard, Kind.html) and a.start >= prev.end
                for a in self._atoms(part, prev.end, label_at)
            )
            ok = all(c.isspace() or c in _ENDS or _closes(c) or c == "|" for c in gap)
            return prev if ok and not hard and gap.count("\n") <= 1 else None
        before = self.previous
        if (
            isinstance(before, Block)
            and before.code is not None
            and prev.code
            and part.kind == "paragraph"
            and before.last_line + 1 == part.first_line
            and before.parent == part.parent
            and label_at == part.start
        ):
            return prev
        if (
            isinstance(before, Inline)
            and part.kind == "cell"
            and before.kind == "cell"
            and before.row == part.row
            and prev.start >= before.start
            and label_at == part.start
        ):
            return prev
        return None


@lru_cache(maxsize=64)
def segment(answer: str, *, names: frozenset[str] = frozenset()) -> tuple[Unit, ...]:
    """Every unit of the answer, in order: all of its text that is not structure, plus bare
    label runs (which state nothing, so `substantive` is False, but whose labels the citation
    check validates).

    `names` (`dotted_names` of the SHOWN `S` sources, never tool results) keeps a dotted name
    the sources hold verbatim in one unit (`sentence_ends`). Every caller judging one answer
    must pass the same set, or one sees a unit another cut in two."""
    builder = _Builder(answer, names)
    for part in structure(answer).parts:
        if isinstance(part, Block):
            builder.block(part)
        else:
            builder.inline(part)
        builder.previous = part
    return tuple(builder.units)


@dataclass(frozen=True, slots=True)
class Run:
    """A maximal run of visible text (plain text, escapes, entities) of one inline part, each
    character mapped to its source span; `labels` holds where accepted labels start in it."""

    text: str
    starts: tuple[int, ...]
    ends: tuple[int, ...]
    labels: frozenset[int]


@lru_cache(maxsize=64)
def visible_runs(answer: str, *, names: frozenset[str] = frozenset()) -> tuple[Run, ...]:
    """Every run of text the reader sees as text, with the labels accepted in it. The citation
    check looks for citation-shaped fragments HERE, so a marker the reader sees but that is not
    accepted as a label (escaped "\\[S9\\]", an entity, struck through, in a URL-shaped word, in
    a link's text, after "<!--") is an invalid fragment, never decoration riding on a label.

    The text of every autolink and linkified URL is a run of its own with no accepted label
    (review of the parser rewrite, round 1, P2): "https://x.example/[S1]" shows the marker
    whether or not linkify made it a link, so the check rejects it in both configurations.

    `names` is accepted so every reader of one answer takes the same seam (`segment`); runs and
    accepted labels do not depend on sentence cuts, so it changes nothing here today."""
    runs: list[Run] = []
    for part in structure(answer).parts:
        if isinstance(part, Block):
            continue
        for atom in part.atoms:
            if atom.kind is Kind.url:
                autolink = answer.startswith("<", atom.start) and answer[atom.end - 1] == ">"
                a, b = (atom.start + 1, atom.end - 1) if autolink else (atom.start, atom.end)
                if a < b:
                    runs.append(
                        Run(
                            answer[a:b], tuple(range(a, b)), tuple(range(a + 1, b + 1)), frozenset()
                        )
                    )
        view = _view(part, names)
        accepted = {m.start() for m in _labels(part, view)}
        start = 0
        for i in range(len(view.text) + 1):
            inside = i < len(view.text) and part.atoms[view.owner[i]].kind in (
                Kind.text,
                Kind.escape,
            )
            if inside:
                continue
            if i > start:
                runs.append(
                    Run(
                        view.text[start:i],
                        tuple(view.starts[start:i]),
                        tuple(view.ends[start:i]),
                        frozenset(a - start for a in accepted if start <= a < i),
                    )
                )
            start = i + 1
    return tuple(runs)


def source_sentences(text: str, *, names: frozenset[str] = frozenset()) -> list[str]:
    """The visible text of every prose unit of a source page: its sentences, cut exactly as an
    answer is (attribution's per-sentence number and negation evidence), with the same `names`
    so a dotted name is one token on both sides."""
    return [u.visible for u in segment(text, names=names) if not u.code and u.visible.strip()]


# ---------------------------------------------------------------- residue
def covered_by_model(unit: Unit, valid: Collection[str]) -> bool:
    """(a): the unit ends in labels and every one of them names something the model was shown."""
    return bool(unit.labels) and all(label in valid for label in unit.labels)


def residue(
    answer: str,
    valid: Collection[str],
    inferred: Iterable[str],
    *,
    names: frozenset[str] = frozenset(),
) -> list[str]:
    """THE guarantee's residue (verbatim unit texts), computed over the whole answer.

    `valid` are the label names the model was shown ("S1", "T2"); `inferred` the verbatim unit
    texts whose inferred citation PASSED the ground check (b). Everything else that is
    substantive and not allowlisted (c) is residue. `names`: as for `segment` (the same set
    attribution used, or a unit it kept whole is residue here in two pieces)."""
    kept = set(inferred)
    # Decode character references once and run every whole-answer check on both forms: another
    # renderer may show "&#x2800;" as the blank glyph it names (final review, P2).
    decoded = html.unescape(answer) if "&" in answer else answer
    if any(check(t) for t in {answer, decoded} for check in (has_bidi, has_rtl, has_blank_glyph)):
        # ANSWER-level, not unit-level (review, P1): a control in a unit that states nothing
        # ("\u202e.") is dropped by segment() yet still reverses everything after it in the
        # paragraph. Any bidi control anywhere: the whole answer is residue, never verified.
        return [answer]
    return list(
        dict.fromkeys(
            u.text
            for u in segment(answer, names=names)
            if has_bidi(u.text)  # display differs from what was checked: never covered
            or (
                u.substantive
                and not covered_by_model(u, valid)
                and u.text not in kept
                and u.allowed is None
            )
        )
    )


def refusal_only(answer: str, *, names: frozenset[str] = frozenset()) -> bool:
    """The whole answer is an allowlisted refusal, optionally with pleasantries: it makes no
    claim, so it is never badged `verified` (it is `unverified`, ADR-0015)."""
    kinds = [u.allowed for u in segment(answer, names=names) if u.substantive]
    return Allowed.refusal in kinds and all(k is not None for k in kinds)


def valid_labels(shown: Iterable[int], n_tool_results: int) -> frozenset[str]:
    """Label names a model citation may use this turn."""
    return frozenset([*(f"S{n}" for n in shown), *(f"T{n}" for n in range(1, n_tool_results + 1))])
