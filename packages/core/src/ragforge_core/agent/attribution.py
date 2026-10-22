"""Server-side citation attribution and the verification status of a mixed answer (ADR-0015).

WHY this exists: on llama3.2:3b the dominant failure was a draft with the right facts and ZERO
`[S<n>]` labels (NEXT.md, Phase 2b: arms A0/A1). Such a draft carries no provenance, so it is
discarded and the user gets an abstain. The server fills the gap deterministically:

1. `attribute` (CPU only, no model call): every coverage unit (coverage.segment) that no model
   label covers and the allowlist does not exempt gets the shown source with the strongest
   lexical evidence over its VISIBLE text (content-word coverage, content bigrams as the
   tie-break), or, for a code block, the source holding it as a whole code element, exactly.
   Every inline code span of a prose unit must be a whole code element of that source too.
   Below the bar the unit gets no source; an opaque unit (`coverage.opaque`, `Unit.opaque`)
   never gets one.
2. `settle`, after the ground check: an inferred citation survives only if the verifier reached a
   conclusive verdict that is not "unsupported", every claim it flagged was located in the
   answer, AND no flagged claim overlaps that unit. The status is then decided by the RESIDUE
   (coverage.residue): `verified` needs it empty.

WHY attribution no longer decides what a "claim" is: the guarantee lives in coverage.py as an
allowlist; attribution only proposes citations for text that would otherwise be residue. A
unit it cannot support stays residue, so its weaknesses (paraphrase, short sentences, scripts it
tokenizes poorly) cost recall, never correctness.

Why lexical evidence and not embeddings: deterministic, explainable in a test, microseconds per
sentence; an embedding call per sentence would add model latency to every turn (ADR-0015: no
extra model call on the happy path).

Lexical support (all must hold):
- matched >= MIN_MATCHED content tokens, covering >= MIN_COVERAGE of the unit's distinct ones;
- every numeric token appears in the source, and the word right after it in the unit (its unit
  noun; the word before it when nothing follows) is one of its neighbours there (`_numbers_ok`:
  catches "3 partitions" against "3 brokers" and "every 60 brokers" against "every 60 seconds");
- a unit with a negation ("not", "never", "doesn't"...) is only supported by a source whose
  best-matching sentence also has one (`_negation_ok`): "Kafka does not store records in
  partitions" shares every content word with its source, and lexical overlap must not lend it
  provenance. The opposite direction (source negates, unit does not) is the verifier's job.
  Negation is read on a SKELETON (`skeleton`): format characters removed, accents dropped and
  Greek/Cyrillic/Armenian look-alike letters read as the Latin ones, so "n\\u200bot" and an
  all-Greek spelling of "NOT" (nu, omicron, tau) are "not" (review of the redesign, round 4);
  apostrophe-less contractions ("doesnt") and letter-by-letter spellings ("n-o-t") count too, and
  apostrophe look-alikes are folded first ("doesn\u00b4t" is "doesn't": `coverage.APOSTROPHES`).
  ADR-0015 section 4.1 lists exactly which forms are read; anything else is the verifier's.

Code (review of the redesign, round 4): code elements come from the CommonMark parse of the
answer AND of each source page (`structure.structure`): fenced and indented blocks at any depth,
and inline spans, with the parser's own content. Code is covered only as a WHOLE element of a
source, exact up to trailing whitespace, a uniform dedent (the common leading-whitespace string)
and blank edge lines (`code_matches`);
a span inside a source's fenced block is part of that block, not an inline span of the page.
ONE narrow exception (owner decision, 2026-10-18, NEXT.md "false Not verified flags"): an inline
span of the answer that is a single identifier (`identifier_name`) is backed when that exact name
is a whole token of a code element of the cited page (`_code_names`).

Every flagged claim carries a reason (`Settled.reasons`): "not_found" (the verifier or the
residue) or "inexact_code" (only its code was not matched: a muted flag on the client).
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Collection, Iterable, Iterator, Sequence
from dataclasses import dataclass
from itertools import pairwise
from typing import Final

from ragforge_core.agent.coverage import (
    APOSTROPHES,
    Unit,
    clean,
    covered_by_model,
    has_bidi,
    has_blank_glyph,
    has_rtl,
    opaque,
    refusal_only,
    residue,
    segment,
    source_sentences,
    strip_end,
)
from ragforge_core.agent.state import HitDict
from ragforge_core.agent.structure import Atom, Code, Inline, Kind, structure
from ragforge_core.domain.models import ClaimReasonCode, Verification

MIN_MATCHED: Final = 3
MIN_COVERAGE: Final = 0.6
NOT_FOUND: Final[ClaimReasonCode] = "not_found"
INEXACT_CODE: Final[ClaimReasonCode] = "inexact_code"

# Unicode-aware: "Grosse" with an eszett is one token, and a non-Latin sentence yields tokens.
# A number is ONE token with its sign, decimals and thousands separators (review round 1: "-3"
# matched a source's "3", "3.5" matched "3" and "5"). A sign counts only where it cannot be a
# hyphen ("kafka-3" is a name and a 3).
_NUMBER: Final = r"(?<![\w.,])[-+\u2212]?\d+(?:[.,]\d+)*\w*"
_WORD: Final = re.compile(rf"{_NUMBER}|\w+(?:'\w+)?")
_MINUS: Final = str.maketrans({"\u2212": "-"})
_LABEL_MARK: Final = re.compile(r"\[[ST]\d{1,4}(?:\s*,\s*[ST]\d{1,4})*\]", re.IGNORECASE)
# English function words plus answer boilerplate ("notes", "according"): words that make two
# unrelated sentences look alike without carrying a fact.
_STOPWORDS: Final = frozenset(
    """
    a about above after again against all also am an and any are as at be because been before
    being below between both but by can could did do does doing down during each either else etc
    even ever every few for from further had has have having he her here hers him his how i if in
    into is it its itself just let like may me might more most much must my no nor not now of off
    on once one only or other our ours out over own per same shall she should so some such than that
    the their theirs them then there these they this those though through thus to too under until up
    upon us very via was we were what when where whether which while who whom whose why will with
    within without would yes yet you your yours according note notes source sources based mentioned
    mention mentions stated says said described
    """.split()  # noqa: SIM905 (a word list reads better than 150 quoted strings)
)
_AUXILIARIES: Final = (
    *("do", "does", "did", "is", "are", "was", "were", "has", "have", "had", "can", "could"),
    *("wo", "would", "should", "must", "need", "ai", "dare", "might", "sha"),
)
_NEGATIONS: Final = frozenset(
    {"not", "no", "never", "none", "nor", "neither", "nobody", "nothing", "nowhere", "cannot"}
    | {"without", "unable"}
    # negating verbs (review round 1: "Kafka fails to store records in partitions."). This only
    # withholds candidates, so a missing word costs nothing the guarantee relies on: the unit is
    # then the verifier's to catch, exactly as the ADR states for contradiction.
    | {"fail", "fails", "failed", "lack", "lacks", "lacked", "refuse", "refuses", "refused"}
    # hedges that negate or nearly negate (review round 2, P2: "Kafka hardly stores records")
    | {"hardly", "rarely", "seldom", "barely", "scarcely"}
    | {"incapable"}
    # contractions written without the apostrophe (blind review round 2, P2: "Kafka doesnt
    # store records"): a closed list, since "-nt" alone also ends "client" and "event"
    | {f"{aux}nt" for aux in _AUXILIARIES}
)
# A word spelled out letter by letter with hyphens or dots ("n-o-t", "n.o.t") is read as the
# word for negation only (blind review round 2, P2).
_SPELLED: Final = re.compile(r"(?<!\w)\w(?:[-.]\w)+(?!\w)")
# Number words become numeric tokens (review round 2, P2: "three partitions" escaped the number
# check because "three" was a plain word). On BOTH sides, so "three brokers" is supported by
# "3 brokers" and "3 partitions" is still refused against "3 brokers" (`_numbers_ok`).
_UNITS: Final = (  # noqa: SIM905
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen"
    " fifteen sixteen seventeen eighteen nineteen twenty"
).split()
_TENS: Final = "thirty forty fifty sixty seventy eighty ninety".split()  # noqa: SIM905
_ORDINALS: Final = (  # noqa: SIM905
    "first second third fourth fifth sixth seventh eighth ninth tenth eleventh twelfth"
    " thirteenth fourteenth fifteenth sixteenth seventeenth eighteenth nineteenth twentieth"
).split()
_SUFFIX: Final = {1: "st", 2: "nd", 3: "rd"}
_NUMBER_WORDS: Final = {
    **{word: str(n) for n, word in enumerate(_UNITS)},
    **{word: str(30 + 10 * n) for n, word in enumerate(_TENS)},
    **{word: f"{n}{_SUFFIX.get(n, 'th')}" for n, word in enumerate(_ORDINALS, start=1)},
    "dozen": "12",
    "hundred": "100",
    "thousand": "1000",
    "million": "1000000",
    "billion": "1000000000",
}


def _stem(word: str) -> str:
    """Light plural folding so "partitions" supports "partition"; deliberately no real stemmer
    (a stemmer that merges "organization"/"organ" would manufacture support)."""
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 3 and word.endswith("s") and not word.endswith(("ss", "us", "is")):
        return word[:-1]
    return word


def _keep(word: str) -> bool:
    """Drop stopwords and single ASCII letters ("x", "i"); a single digit or a single non-Latin
    character (one CJK ideograph is a word) is content."""
    if word in _STOPWORDS or not word:
        return False
    return len(word) > 1 or not (word.isascii() and word.isalpha())


def _words(text: str) -> list[str]:
    """Every cleaned (apostrophe look-alikes folded first, `coverage.APOSTROPHES`; NFKC, no
    format characters), casefolded word, stopwords included, with label markers removed."""
    folded = clean(_LABEL_MARK.sub(" ", text).translate(APOSTROPHES)).casefold()
    folded = folded.translate(_MINUS)
    words = (w.removesuffix("'s") for w in _WORD.findall(folded))
    return [_NUMBER_WORDS.get(w, w) for w in words]


# Look-alike letters read as the Latin letter they imitate (Unicode confusables, the common
# Greek, Cyrillic and Armenian ones). Used ONLY to detect negation: evidence never comes from a
# skeleton, and a word mixing alphabets is opaque (coverage.mixed_script).
_LOOKALIKES: Final = str.maketrans(
    "ΑΒΕΖΗΙΚΜΝΟΡΤΥΧαεικνορτυχγАВЕКМНОРСТХЅІЈаеорсухѕіјһԁԛԝօոսհ",
    "ABEZHIKMNOPTYXaeikvoptuxyABEKMHOPCTXSIJaeopcyxsijhdqwonuh",
)


# Latin small capitals ("ɴᴏᴛ"): NFKC keeps them, and they are Latin, so a word of them is not
# mixed-script. Generated from the Unicode names (every "LATIN LETTER SMALL CAPITAL <X>" of the
# IPA, phonetic and Latin extension blocks), read as the plain letter for negation only (review of
# the parser rewrite, round 1, P2).
_SMALL_CAPITAL: Final = re.compile(r"LATIN LETTER SMALL CAPITAL ([A-Z])")
_SMALL_CAPITALS: Final = str.maketrans(
    {
        chr(code): match.group(1).lower()
        for lo, hi in ((0x0250, 0x02AF), (0x1D00, 0x1DBF), (0xA720, 0xA7FF), (0x10780, 0x107BF))
        for code in range(lo, hi + 1)
        if (match := _SMALL_CAPITAL.fullmatch(unicodedata.name(chr(code), ""))) is not None
    }
)


def skeleton(text: str) -> str:
    """`text` as the reader reads it: apostrophe look-alikes folded, cleaned (no ignorable
    characters), accents dropped,
    look-alike letters and Latin small capitals mapped to plain Latin, casefolded."""
    decomposed = unicodedata.normalize("NFD", clean(text.translate(APOSTROPHES)))
    plain = "".join(c for c in decomposed if not unicodedata.combining(c))
    return plain.translate(_LOOKALIKES).translate(_SMALL_CAPITALS).casefold()


def content_tokens(text: str) -> list[str]:
    """Content tokens in order (duplicates kept, so bigrams stay adjacent)."""
    return [_stem(w) for w in _words(text) if _keep(w)]


def negated(text: str) -> bool:
    """The text carries a negation word (`_NEGATIONS`, including apostrophe-less contractions)
    or an n't contraction, read on its `skeleton` with letter-by-letter spellings joined."""
    read = _SPELLED.sub(lambda m: m.group(0).replace("-", "").replace(".", ""), skeleton(text))
    return any(w in _NEGATIONS or w.endswith("n't") for w in _words(read))


def _numeric(token: str) -> bool:
    """A token carrying a number ("9093", "v2", "16gb"): each must appear in the source."""
    return any(c.isdigit() for c in token)


def _bigrams(tokens: Sequence[str]) -> set[tuple[str, str]]:
    return set(pairwise(tokens))


def _neighbours(tokens: Sequence[str]) -> dict[str, frozenset[str]]:
    """For each numeric token, the content tokens directly before and after it (any occurrence).
    Content-token adjacency skips stopwords: "16384 hash slots" and "the port is 9093" bind
    16384 to "hash" and 9093 to "port"."""
    near: dict[str, set[str]] = {}
    for i, token in enumerate(tokens):
        if _numeric(token):
            near.setdefault(token, set()).update(tokens[max(i - 1, 0) : i], tokens[i + 1 : i + 2])
    return {n: frozenset(words) for n, words in near.items()}


def _anchors(tokens: Sequence[str]) -> Iterable[tuple[str, str | None]]:
    """(number, anchor) for every numeric token of a unit: the anchor is the content token right
    AFTER it (the unit noun: "60 seconds"), or the one before it when nothing follows ("the port
    is 9093"), or None when it stands alone ("Yes, 3.")."""
    for i, token in enumerate(tokens):
        if _numeric(token):
            after = tokens[i + 1] if i + 1 < len(tokens) else None
            before = tokens[i - 1] if i else None
            yield token, after if after is not None else before


@dataclass(frozen=True, slots=True)
class _Sentence:
    tokens: frozenset[str]
    negated: bool


@dataclass(frozen=True, slots=True)
class _Source:
    label: int
    tokens: frozenset[str]
    bigrams: frozenset[tuple[str, str]]
    numbers: dict[str, frozenset[str]]  # numeric token -> its neighbours (`_neighbours`)
    sentences: tuple[_Sentence, ...]
    code_blocks: frozenset[str]  # WHOLE code blocks of the page (`_code_text`)
    code_spans: frozenset[str]  # WHOLE inline code spans of the page, stripped
    code_names: frozenset[str]  # identifiers that are whole tokens of its code (`_code_names`)
    code_runs: tuple[tuple[str, ...], ...]  # each code element as `_code_run` tokens


def _source(label: int, hit: HitDict, names: frozenset[str] = frozenset()) -> _Source:
    """A source's evidence includes its title and heading path: "Partitions are ordered logs"
    under a page titled "Kafka" is supported by that page. `names` cuts its sentences exactly as
    the answer's (`coverage.dotted_names`): a dotted name is one token on both sides."""
    parts = [hit["title"], *hit["heading_path"], hit["display_text"]]
    tokens = content_tokens(" ".join(parts))
    near: dict[str, set[str]] = {}
    spans: list[_Sentence] = []
    # Per SOURCE SENTENCE: in "... in partitions. The cluster has 3 brokers." the 3 must not
    # borrow "partitions" from the sentence before it, and a negation belongs to its sentence.
    for part in parts:
        for text in source_sentences(part, names=names) or [part]:
            sentence_tokens = content_tokens(text)
            spans.append(_Sentence(frozenset(sentence_tokens), negated(text)))
            for number, words in _neighbours(sentence_tokens).items():
                near.setdefault(number, set()).update(words)
    return _Source(
        label,
        frozenset(tokens),
        frozenset(_bigrams(tokens)),
        {n: frozenset(w) for n, w in near.items()},
        tuple(spans),
        *_source_code(hit["display_text"]),
    )


def _visible_piece(atom: Atom) -> str:
    """One atom of an ANSWER paragraph as the reader sees it around its code spans (`joined_code`):
    plain eligible text as is, emphasis delimiters as `_EMPHASIS` (stepped over at a span's
    edges), everything else (escapes, entities, breaks, HTML, links, images) as `_HIDDEN`,
    which joins whatever it touches."""
    match atom.kind:
        case Kind.text if atom.eligible:
            return atom.visible
        case Kind.mark if not atom.opaque:
            return _EMPHASIS
        case _:
            return _HIDDEN


_HIDDEN: Final = "\ufffc"  # OBJECT REPLACEMENT CHARACTER: an atom that joins its neighbours
# An emphasis/strikethrough delimiter: invisible, stepped over at a code span's edges.
_EMPHASIS: Final = "\x03"
# Punctuation that may sit between a quote and the next space without continuing its token.
_TRAILING: Final = frozenset(".,;:!?")


def _opens(char: str) -> bool:
    return char in "\"'" or unicodedata.category(char) in ("Ps", "Pi")


def _closer(char: str) -> bool:
    return char in "\"'" or unicodedata.category(char) in ("Pe", "Pf")


def _bounded(prose: str, start: int, end: int) -> bool:
    """prose[start:end] stands alone in what the reader sees, FAIL CLOSED: before it only the
    start, a visible space (`_gap`), or opening brackets/quotes after one; after it only the end,
    a visible space, or `.,;:!?` and closing brackets/quotes followed by one. Anything else (a
    letter, another code span, `_HIDDEN`, an invisible character) joins it."""
    i = start
    while i > 0 and (_opens(prose[i - 1]) or prose[i - 1] == _EMPHASIS):
        i -= 1
    if i > 0 and not _gap(prose[i - 1]):
        return False
    j = end
    while j < len(prose) and (prose[j] in _TRAILING or _closer(prose[j]) or prose[j] == _EMPHASIS):
        j += 1
    return j == len(prose) or _gap(prose[j])


# Zs spaces narrower than about a quarter em: two code boxes around one look touching.
_THIN_SPACES: Final = frozenset("\u2006\u2009\u200a\u202f")


def _gap(char: str) -> bool:
    """A VISIBLE space: ASCII space, tab, line feed, or a Unicode space separator (Zs).
    `str.isspace` also accepts VT, FF, FS-US, NEL and U+2028/2029, which a renderer may show as
    nothing at all: "rm -rf /\x1ftmp" reads "rm -rf /tmp" (review round 4, P2)."""
    return char in " \t\n" or (unicodedata.category(char) == "Zs" and char not in _THIN_SPACES)


def _source_code(
    text: str,
) -> tuple[frozenset[str], frozenset[str], frozenset[str], tuple[tuple[str, ...], ...]]:
    """Every whole code block, every whole inline span, every whole-token identifier and the
    token run of every code element (`_code_run`) of a page (the SHOWN chunk's text), from its
    CommonMark parse: a span inside a fenced block is part of that block, not a span of the page.

    Inline spans of the page count for the identifier rule too: the page's author marked the
    name as code exactly as a block does, and the parser gives the span's exact content."""
    found = structure(text).code
    blocks = frozenset(b for b in (_code_text(c.text) for c in found if c.block) if b)
    spans = frozenset(t for t in (_code_text(c.text) for c in found if not c.block) if t)
    names = frozenset(n for c in found for n in _code_names(c.text))
    runs = tuple(run for c in found if (run := _code_run(c.text)))
    return blocks, spans, names, runs


def _numbers_ok(unit_tokens: Sequence[str], source: _Source) -> bool:
    """Every number of the unit appears in the source AND its anchor (`_anchors`) is one of its
    neighbours there, for EVERY occurrence.

    WHY adjacency and not membership (review round 4): "Kafka stores records in 3 partitions."
    against a source saying "3 brokers" passes membership. WHY the noun after the number and
    not "either neighbour" (review of the redesign, round 3, P2): "Autovacuum runs every 60
    brokers." shared its LEFT neighbour ("run") with "runs every 60 seconds", so the swapped unit
    noun passed. The anchor may sit on either side of the number in the source ("brokers: 3"
    supports "3 brokers"): a swapped noun is not a neighbour of the number on either side, and
    the reorder keeps a paraphrase's recall. A number with no neighbour only needs to appear."""
    for number, anchor in _anchors(unit_tokens):
        theirs = source.numbers.get(number)
        if theirs is None or (anchor is not None and anchor not in theirs):
            return False
    return True


def _negation_ok(distinct: set[str], is_negated: bool, source: _Source) -> bool:
    """A negated unit needs a negated best-matching source sentence (module docstring)."""
    if not is_negated:
        return True
    best = max(source.sentences, key=lambda s: len(distinct & s.tokens), default=None)
    return best is not None and best.negated


@dataclass(frozen=True, slots=True)
class Evidence:
    """One source's lexical support for one unit."""

    label: int
    matched: int
    coverage: float
    bigrams: int
    numbers_ok: bool
    negation_ok: bool = True

    @property
    def supports(self) -> bool:
        return (
            self.numbers_ok
            and self.negation_ok
            and self.matched >= MIN_MATCHED
            and self.coverage >= MIN_COVERAGE
        )


def score(unit_tokens: Sequence[str], source: _Source, *, is_negated: bool = False) -> Evidence:
    distinct = set(unit_tokens)
    matched = distinct & source.tokens
    return Evidence(
        label=source.label,
        matched=len(matched),
        coverage=len(matched) / len(distinct) if distinct else 0.0,
        bigrams=len(_bigrams(unit_tokens) & source.bigrams),
        numbers_ok=_numbers_ok(unit_tokens, source),
        negation_ok=_negation_ok(distinct, is_negated, source),
    )


def best_source(text: str, sources: Iterable[_Source]) -> Evidence | None:
    """The supporting source with the strongest evidence, or None. Ties prefer more shared
    bigrams (word order), then the lower label (the higher-ranked hit)."""
    tokens = content_tokens(text)
    is_negated = negated(text)
    ranked = sorted(
        (e for e in (score(tokens, s, is_negated=is_negated) for s in sources) if e.supports),
        key=lambda e: (-e.coverage, -e.bigrams, -e.matched, e.label),
    )
    return ranked[0] if ranked else None


# Code tokens: a number with its decimals, a word, or one punctuation character.
_CODE_TOKEN: Final = re.compile(r"\d+(?:\.\d+)*\w*|\w+|[^\w\s]")
# A fence's info string other than a language tag may carry prose ("``` NASA built Kafka"); some
# renderers show it. Such a block is never covered (fail closed).
_LANGUAGES: Final = frozenset(
    """
    bash c cpp cs csharp css console diff dockerfile go graphql html ini java javascript js json
    jsx kotlin kt makefile markdown md plaintext powershell proto py python rb ruby rust scala sh
    shell sql swift text toml ts tsx txt typescript xml yaml yml zsh
    """.split()  # noqa: SIM905
)


def _nontrivial(token: str) -> bool:
    """A token that can carry a fact: a number, or a word of two or more characters."""
    return any(c.isdigit() for c in token) or (len(token) > 1 and any(c.isalnum() for c in token))


_BLANK: Final = " \t"


def _code_text(code: str) -> str:
    """Code compared EXACTLY: only trailing spaces and tabs, a uniform dedent (spaces and tabs)
    and blank edge lines are normalised. Internal whitespace, relative indentation and every line
    are kept.

    Lines are split on "\n" ONLY (markdown-it already turned CR LF and CR into it). WHY (review
    of the parser rewrite, round 1, P2): `str.splitlines` also splits on U+2028/U+2029, NEL, VT,
    FF and FS/GS/RS, so "cd /tmp/build<U+2028>rm -rf cache" (one line to the reader, a different
    command when pasted) matched a two-line source block. For the same reason only spaces and
    tabs are "whitespace" here: a trailing form feed or line separator is part of the code.

    The dedent removes the longest leading-whitespace PREFIX STRING every non-blank line shares
    (textwrap.dedent semantics). WHY (blind review round 2, P2): removing the minimum COUNT of
    blank characters took one space from one line and one tab from another, so " if ok:" /
    "\t    deploy()" / "\tcleanup()" matched a source whose cleanup() is not under the `if`:
    relative indentation changed, which is not a uniform dedent."""
    lines = [ln.rstrip(_BLANK) for ln in code.split("\n")]
    while lines and not lines[0]:
        lines.pop(0)
    while lines and not lines[-1]:
        lines.pop()
    prefix = _common_indent([ln for ln in lines if ln])
    return "\n".join(ln.removeprefix(prefix) for ln in lines)


def _common_indent(lines: Sequence[str]) -> str:
    """The longest string of spaces and tabs that starts every line of `lines`."""
    if not lines:
        return ""
    indents = [ln[: len(ln) - len(ln.lstrip(_BLANK))] for ln in lines]
    first, last = min(indents), max(indents)
    width = next((i for i, (a, b) in enumerate(zip(first, last, strict=False)) if a != b), None)
    return first if width is None else first[:width]


# ---------------------------------------------------------------- single identifiers
# The grammar of an answer span the token rule may back (design review, binding): ASCII letters,
# digits and "_", segments joined by single dots, each starting with a letter or "_", optionally
# ending in "()" with nothing inside. ASCII ONLY and NO normalisation: a look-alike letter, a bidi
# control, an invisible character, a look-alike terminator or a non-ASCII digit can never be part
# of an accepted span, so the span falls back to the whole-element rule (fail closed).
_SEGMENT: Final = r"[A-Za-z_][A-Za-z0-9_]*"
_DOTTED: Final = rf"{_SEGMENT}(?:\.{_SEGMENT})*"
_IDENTIFIER_SPAN: Final = re.compile(rf"(?P<name>{_DOTTED})(?:\(\))?")
_SOURCE_NAME: Final = re.compile(_DOTTED)
_CALL: Final = "()"
# A lower-case letter or digit directly followed by an upper-case one: `createMemento`.
_CAMEL_HUMP: Final = re.compile(r"[a-z0-9][A-Z]")
# Characters that CONTINUE a token in code although the grammar excludes them: identifier
# characters, "$" (Java/JS identifiers), "-" (kebab-case keys and CLI flags: `allkeys` is not a
# token of `allkeys-lru`, `verbose` not one of `--verbose`), and path separators ("/", "\\").
_CONTINUES: Final = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_$-/\\"
)
# Code split for the inexact/not-found split (`_excerpt_of`): one token per word, number or
# punctuation character, with `_GAP` wherever whitespace separated two tokens.
_RUN_TOKEN: Final = re.compile(r"\s+|\d+(?:\.\d+)*\w*|\w+|[^\w\s]")
_GAP: Final = " "
# An excerpt ending in one of these leaves an argument open when the source goes on: `rm -rf /`
# cut from `rm -rf /tmp/cache`, `retries=` cut from `retries=3`, `--topic` without its value.
_OPEN_END: Final = frozenset("/\\=:-,|&([{<+*@$%")


def _shaped(name: str) -> bool:
    """`name` LOOKS like an identifier, not a word or a value: it has a "_" or a ".", ends in
    "()", or has a camel hump. WHY (design review): `Memento`, `acks`, `true`, `null` and `id` are
    also ordinary words or literal values found in nearly any code, so a stray occurrence in the
    page must not back a sentence that uses them as words or asserts them as values; they keep the
    whole-element rule. A name of this shape is specific enough that finding it whole in the cited
    page's code is evidence the model copied it rather than invented it."""
    return (
        "_" in name or "." in name or name.endswith(_CALL) or _CAMEL_HUMP.search(name) is not None
    ) and any(c.isalpha() for c in name)


def identifier_name(span: str) -> bool:
    """`span` (an inline code span's parser text, unstripped) is ONE identifier the token rule may
    back: the ASCII grammar above, fully (so no spaces, no "=", no arguments, no operators, no
    hyphen, no quote, no non-ASCII character), and identifier-shaped (`_shaped`).

    `acks=all` states a setting and `foo(x)` an argument: both keep the whole-element rule."""
    return _IDENTIFIER_SPAN.fullmatch(span) is not None and _shaped(span)


def _code_names(code: str) -> Iterator[str]:
    """Every answer span `identifier_name` may be backed by, as a WHOLE token of this code element.

    For each maximal dotted run of identifier characters (`_SOURCE_NAME`) whose neighbours are
    ASCII and do not continue it (`_CONTINUES`):
    - each of its segments: a SIMPLE answer name matches any segment (`createMemento` is a token
      of `this.createMemento()`: a method is named on its own);
    - the whole run, only when it is maximal (no "." continues it on either side): a DOTTED answer
      name must equal a whole source name (`interval.ms` and `max.poll` are NOT tokens of
      `max.poll.interval.ms`: a sub-path names a different key);
    - its last segment and the whole run followed by "()", when "(" comes right after it, whatever
      the arguments: `createMemento()` names the method `createMemento(state)` declares or calls,
      and the parentheses in prose mean "the function", not "called with no argument". What the
      sentence then SAYS about the call is the verifier's to check.
    A run next to a non-ASCII character (a letter, a combining mark, a zero-width joiner) may
    continue visually, so it yields nothing (fail closed). Code holding a bidi control, a
    right-to-left character or a blank glyph may DISPLAY in another order than the one scanned, so
    it yields nothing at all."""
    if has_bidi(code) or has_rtl(code) or has_blank_glyph(code):
        return
    for match in _SOURCE_NAME.finditer(code):
        start, end = match.span()
        before = code[start - 1] if start else " "
        after = code[end] if end < len(code) else " "
        if not (before.isascii() and after.isascii()) or before in _CONTINUES:
            continue
        if after in _CONTINUES:
            continue
        nxt = code[end + 1] if end + 1 < len(code) else " "
        call = after == "("
        segments = match.group(0).split(".")
        yield from segments
        if call:
            yield segments[-1] + _CALL
        # A "." before the run, or one after it that a token character follows, means the run is
        # part of a longer name the regex could not take whole ("1.a.b", "a.b.1c").
        whole = before != "." and not (after == "." and (not nxt.isascii() or nxt in _CONTINUES))
        if whole and len(segments) > 1:
            yield match.group(0)
            if call:
                yield match.group(0) + _CALL


def _token_backed(code: Code, source: _Source) -> bool:
    """The single-identifier rule: an INLINE span that is one identifier-shaped name
    (`identifier_name`) and a whole token of the source's code (`_code_names`).

    WHY this is safe where every earlier partial match leaked (`code_matches`): one identifier has
    no whitespace, order, guard or argument to drop, so the only ways to alter it are a different
    character (not the same token: the match is exact and case-sensitive) or a substring (rejected
    by the whole-token boundaries). What the sentence SAYS about the name is the verifier's, as for
    the prose around it. Only the code of the shown chunk the label cites is read, never its prose:
    the prose-quote exception removed after review round 9 stays removed."""
    return not code.block and identifier_name(code.text) and code.text in source.code_names


def code_matches(code: Code, source: _Source) -> bool:
    """`code` (a block or an inline span of the answer) is a WHOLE code block or a WHOLE inline
    code span of the source, exactly (`_code_text`); a block's info string must be empty or a
    language tag.

    WHY whole elements only (review of the redesign: a code bypass in four consecutive rounds):
    every partial match leaked. Token matching dropped whitespace ("rm -rf . /cache" quoted
    "rm -rf ./cache"), subsequences dropped guard lines, excerpting one line left its enclosing
    `if` behind, and word overlap matched inline commands with inverted meaning. Excerpted or
    edited code is therefore never covered: it stays residue (highlighted, `partial`). The one
    exception, single identifiers (`_token_backed`), applies ONLY to code the model labelled
    (`_model_code_ok`), never here: inferred citations and `verbatim_source` keep this rule."""
    if code.info and code.info.casefold() not in _LANGUAGES:
        return False
    text = _code_text(code.text)
    if not text:
        return False
    if text in source.code_blocks:
        return True
    return "\n" not in text and text in source.code_spans


def verbatim_source(code: Code, sources: Iterable[_Source]) -> int | None:
    """The first shown source holding this code block whole and exactly (`code_matches`). An
    inferred code citation also needs 3 non-trivial tokens ("a / b / c" is on any page)."""
    tokens = _CODE_TOKEN.findall(code.text)
    if sum(_nontrivial(t) for t in tokens) < MIN_MATCHED:
        return None
    for source in sources:
        if code_matches(code, source):
            return source.label
    return None


def _inline_ok(unit: Unit, source: _Source | None) -> bool:
    """Every inline code span of a prose unit is a whole code element of `source`."""
    return not unit.codes or (
        source is not None and all(code_matches(c, source) for c in unit.codes)
    )


def _label_source(label: str, sources: Sequence[_Source]) -> _Source | None:
    """The shown source a model label ("S2") names; tool labels ("T1") have no page text."""
    if not label.startswith("S") or not label[1:].isdigit():
        return None
    return next((src for src in sources if src.label == int(label[1:])), None)


def _model_code_ok(unit: Unit, sources: Sequence[_Source]) -> bool:
    """Code the MODEL labelled (a block cited by a label line, or spans in a labelled sentence)
    must be whole and exact in ONE of the `S` sources it cites, exactly like inferred code; a
    label does not make altered code true (review of the redesign, P1). Here, and only here, an
    inline span may instead be a single identifier that is a whole token of that source's code
    (`_token_backed`). Every fail-closed rule runs first: the caller has already rejected joined
    spans (`joined_code`), and segmentation has already cut a span that ends in a terminator
    (coverage `_ends_inside`), so the token rule only ever sees a span that stands alone."""
    if not unit.codes:
        return True
    for label in unit.labels:
        source = _label_source(label, sources)
        if source is not None and all(
            code_matches(c, source) or _token_backed(c, source) for c in unit.codes
        ):
            return True
    return False


def _code_run(code: str) -> tuple[str, ...]:
    """`code` as tokens (`_RUN_TOKEN`) with every whitespace run, line breaks included, read as one
    `_GAP`, and none at the edges; () for code that may DISPLAY in another order or hide a
    character (a bidi control, a right-to-left character, a blank glyph): never "inexact"."""
    if has_bidi(code) or has_rtl(code) or has_blank_glyph(code):
        return ()
    tokens = [_GAP if t.isspace() else t for t in _RUN_TOKEN.findall(code)]
    if tokens and tokens[0] == _GAP:
        tokens = tokens[1:]
    if tokens and tokens[-1] == _GAP:
        tokens = tokens[:-1]
    return tuple(tokens)


def _open_end(excerpt: Sequence[str]) -> bool:
    """The excerpt stops where an argument is still owed: after an `_OPEN_END` character, or on
    a flag (`-x`, `--topic`) whose value the source then gives."""
    last_word = max((i for i, token in enumerate(excerpt) if token == _GAP), default=-1) + 1
    return excerpt[-1] in _OPEN_END or excerpt[last_word] == "-"


def _excerpt_of(answer: tuple[str, ...], source: tuple[str, ...]) -> bool:
    """`answer` is a contiguous run of `source` (both `_code_run`s) that starts and ends at a
    whitespace boundary of the source and, unless it reaches the source's end, does not stop with
    an argument open (`_open_end`).

    WHY a run and not a set of words (blind review round 1, P1): every word of `acks=all` is on a
    page saying `acks=1` and `retries=all`, and every word of `enable.idempotence=false` on one
    saying `enable.idempotence=true` and `enable.auto.commit=false`; a bag of words muted both.
    Order, punctuation and adjacency must survive, and the gaps too: "rm -rf . /cache" is not
    "rm -rf ./cache". The boundaries reject a cut inside a word (`interval.ms` from
    `max.poll.interval.ms`); the open end rejects `rm -rf /` cut from `rm -rf /tmp/cache`."""
    if not answer:
        return False
    padded = (_GAP, *source, _GAP)
    needle = (_GAP, *answer)
    width = len(needle)
    for at in range(len(padded) - width):
        if padded[at : at + width] != needle or padded[at + width] != _GAP:
            continue
        if at + width == len(padded) - 1 or not _open_end(answer):
            return True
    return False


def _tokens_in_source(unit: Unit, source: _Source) -> bool:
    """Every code element of `unit` is an excerpt (`_excerpt_of`) of ONE code element of
    `source`, and every identifier-shaped name in it (`_shaped`) is a whole token of that code
    (`_code_names`), all case-sensitive. Decides only the REASON of a flag, never coverage
    (`code_reason`): a typo (`exactly_one_v2` for the page's `exactly_once_v2`), a changed value
    (`acks=all` for `acks=1`) or a truncated command is "not found", the prominent flag."""
    for code in unit.codes:
        run = _code_run(code.text)
        if not any(_excerpt_of(run, theirs) for theirs in source.code_runs):
            return False
        names = (m.group(0) for m in _SOURCE_NAME.finditer(code.text))
        if any(_shaped(name) and name not in source.code_names for name in names):
            return False
    return True


def code_reason(unit: Unit, sources: Sequence[_Source], *, joined: bool) -> ClaimReasonCode:
    """Why a model-labelled unit whose code was not backed is flagged: "inexact_code" (a muted
    flag) only when the code is not joined to another span and is an unaltered excerpt of a code
    element of some `S` source the unit cites (`_tokens_in_source`), so it is only shorter than
    that element; "not_found" otherwise. A tool label (`T<n>`) has no page code to compare with:
    "not_found". Fails toward the prominent flag."""
    if joined:
        return NOT_FOUND
    cited = (_label_source(label, sources) for label in unit.labels)
    present = any(s is not None and _tokens_in_source(unit, s) for s in cited)
    return INEXACT_CODE if present else NOT_FOUND


# ---------------------------------------------------------------- attribution
@dataclass(frozen=True, slots=True)
class Inferred:
    label: int  # 1-based hit label, always one of the shown labels
    sentence: str  # verbatim text of the coverage unit it was attributed to


@dataclass(frozen=True, slots=True)
class Attribution:
    inferred: tuple[Inferred, ...]  # candidate inferred citations (before the ground check)
    uncited: tuple[str, ...]  # would-be residue with no candidate or valid provenance (verbatim)
    covered: bool  # some unit is covered by a valid model label
    refusal_only: bool  # the whole answer is an allowlisted refusal (+/- pleasantries)
    # model-labelled units whose code is not a whole, exact code block/span of a cited page
    code_unbacked: tuple[str, ...] = ()
    # the part of code_unbacked whose code only differs in form from a cited page (`code_reason`):
    # flagged "inexact_code" unless something else flags it (`flag_reasons`)
    code_inexact: tuple[str, ...] = ()


def _candidate(unit: Unit, sources: Sequence[_Source]) -> int | None:
    """The proposed source for one would-be residue unit. A prose unit holding text the word
    tokenizer cannot read faithfully ("... partitions ⠝⠁⠎⠁.", a word mixing alphabets) or hiding
    text (any part of a link: its destination and title; an image) gets none: lexical evidence is
    blind to that text, so it must not lend the unit provenance (review round 2, P1; round 4;
    parser rewrite, round 3)."""
    if unit.code:
        return verbatim_source(unit.codes[0], sources)
    if unit.opaque or opaque(unit.visible):
        return None
    evidence = best_source(unit.visible, sources)
    if evidence is None:
        return None
    chosen = next((src for src in sources if src.label == evidence.label), None)
    return evidence.label if _inline_ok(unit, chosen) else None


_CODE_MARK: Final = "\x02"  # one inline code span of the answer, in its visible reading


def joined_code(answer: str) -> frozenset[tuple[int, int]]:
    """(start, end) of every inline code span of the ANSWER that is not a whole run of tokens in
    what the reader sees (`_bounded`, with each span as one `_CODE_MARK`). Two spans touching,
    or joined by an invisible character, emphasis or a hard break (`rm -rf /`**`var/cache`**),
    render as ONE command although each passes `code_matches` alone (review round 4, P1: the
    answer-side mirror of the source-side join). Such a unit's code is never backed."""
    joined: set[tuple[int, int]] = set()
    for part in structure(answer).parts:
        if not isinstance(part, Inline):
            continue
        pieces: list[str] = []
        spans: list[tuple[int, Code]] = []
        for atom in part.atoms:
            if atom.kind is Kind.code and atom.code is not None:
                spans.append((sum(map(len, pieces)), atom.code))
                pieces.append(_CODE_MARK)
            else:
                pieces.append(" " if atom.kind is Kind.soft else _answer_piece(atom))
        visible = "".join(pieces)
        for at, code in spans:
            if not _bounded(visible, at, at + 1):
                joined.add((code.start, code.end))
    return frozenset(joined)


def _answer_piece(atom: Atom) -> str:
    """`_visible_piece`, except that a soft line break is the space a renderer shows."""
    return " " if atom.kind is Kind.soft else _visible_piece(atom)


def _codes_joined(unit: Unit, joined: frozenset[tuple[int, int]]) -> bool:
    return not unit.code and any((c.start, c.end) in joined for c in unit.codes)


def attribute(
    answer: str,
    hits: Sequence[HitDict],
    shown: Iterable[int],
    valid: Collection[str],
    *,
    names: frozenset[str] = frozenset(),
) -> Attribution:
    """Propose an inferred citation for every unit that would otherwise be residue. `shown` are
    the 1-based labels the model was shown: a label it was not shown can never be inferred (the
    evidence a citation points at must be evidence the answer could have been written from).
    `valid` are the label names a model citation may use (coverage.valid_labels).

    `names` is `coverage.dotted_names` of the SHOWN `S` sources' text (never tool results): the
    dotted names whose interior dots are not sentence ends (`coverage.sentence_ends`). Pass the
    same set to `settle` and the citation check of this answer; empty is the old segmentation."""
    sources = [_source(label, hits[label - 1], names) for label in sorted(set(shown))]
    inferred: list[Inferred] = []
    uncited: list[str] = []
    code_unbacked: list[str] = []
    red: set[str] = set()  # code-unbacked texts with at least one "not_found" occurrence
    covered = False
    joined = joined_code(answer)
    for unit in segment(answer, names=names):
        if not unit.substantive or unit.allowed is not None:
            continue
        if unit.labels:
            # the model's own citation; check_citations rejects it when a label is invalid
            is_joined = _codes_joined(unit, joined)
            if covered_by_model(unit, valid) and (is_joined or not _model_code_ok(unit, sources)):
                code_unbacked.append(unit.text)  # labelled, but the code is not on the page
                uncited.append(unit.text)  # residue with no provenance, highlighted
                if code_reason(unit, sources, joined=is_joined) == NOT_FOUND:
                    red.add(unit.text)  # one red occurrence makes every same-text flag red
                continue
            covered = covered or covered_by_model(unit, valid)
            continue
        label = None if _codes_joined(unit, joined) else _candidate(unit, sources)
        if label is None:
            uncited.append(unit.text)
            # Uncited on its own account: a code-unbacked unit with the same text must not mute
            # this occurrence (claims are keyed by text, so both share one reason).
            red.add(unit.text)
        else:
            inferred.append(Inferred(label, unit.text))
    return Attribution(
        tuple(dict.fromkeys(inferred)),
        tuple(dict.fromkeys(uncited)),
        covered,
        refusal_only(answer, names=names),
        tuple(dict.fromkeys(code_unbacked)),
        tuple(u for u in dict.fromkeys(code_unbacked) if u not in red),
    )


def no_provenance(attribution: Attribution, tools_ran: bool) -> bool:
    """Nothing in the answer is covered by a model label and nothing can be inferred: it is
    `unsupported` (strict regeneration, then abstain), decided without a model call. A pure
    refusal makes no claim (`unverified`, not this). Turns where a tool ran are exempt: their
    text rests on tool results ("Fact forgotten."), which the ground check sees."""
    return not (
        attribution.refusal_only or attribution.covered or attribution.inferred or tools_ran
    )


# ---------------------------------------------------------------- verifier view
@dataclass(frozen=True, slots=True)
class Annotated:
    """The answer as the VERIFIER sees it: every inferred citation written in as `[S<n>]`.

    WHY (review round 2): without the markers the verifier checks the claim against ALL shown
    sources, so a lexically best but wrong page could be emitted as provenance on a `verified`
    answer. With them, ground_check@3 checks each claim against the source it is paired with,
    exactly as it does for the model's own labels, at zero extra model calls. The client never
    sees these markers: inferred citations are emitted as TurnCitation(inferred=True), and the
    verifier's verbatim claims are mapped back onto the client's text (`to_answer`)."""

    answer: str  # the text the client holds
    text: str  # the annotated text the verifier sees
    inserts: tuple[tuple[int, int], ...]  # [start, end) of each inserted marker in `text`

    def _to_answer_pos(self, pos: int) -> int:
        shift = 0
        for start, end in self.inserts:
            if pos <= start:
                break
            shift += min(pos, end) - start  # a position inside a marker clamps to its start
        return pos - shift

    def to_answer(self, claim: str) -> str:
        """The verifier's claim (a verbatim substring of `text`) as a verbatim substring of
        `answer`, without the inserted markers; "" when it is nothing but a marker."""
        start = self.text.find(claim)
        if start == -1:
            return ""
        return self.answer[
            self._to_answer_pos(start) : self._to_answer_pos(start + len(claim))
        ].strip()


def annotate(answer: str, inferred: Sequence[Inferred]) -> Annotated:
    """Insert ` [S<n>]` at the end of each inferred sentence, before its final punctuation (the
    position the answer prompt teaches), or on its own line after an inferred code block."""
    points: list[tuple[int, str]] = []
    cursor = 0
    for item in inferred:
        at = answer.find(item.sentence, cursor)
        if at == -1:
            at = answer.find(item.sentence)
            if at == -1:
                continue
        if "\n" in item.sentence:
            points.append((at + len(item.sentence), f"\n[S{item.label}]"))
        else:
            body = strip_end(item.sentence)
            points.append((at + len(body), f" [S{item.label}]"))
        cursor = at + len(item.sentence)
    parts: list[str] = []
    inserts: list[tuple[int, int]] = []
    prev = 0
    length = 0
    for pos, marker in sorted(points):
        parts.append(answer[prev:pos])
        length += pos - prev
        parts.append(marker)
        inserts.append((length, length + len(marker)))
        length += len(marker)
        prev = pos
    parts.append(answer[prev:])
    return Annotated(answer, "".join(parts), tuple(inserts))


# ---------------------------------------------------------------- verification status
@dataclass(frozen=True, slots=True)
class Settled:
    status: Verification
    inferred: tuple[Inferred, ...]  # inferred citations that passed the ground check
    unsupported_claims: tuple[str, ...]  # verbatim; highlighted by the client
    # The part of unsupported_claims that coverage (not the verifier) produced, and of that the
    # code that is not whole and exact in a cited page. Counted in logs, never shown.
    residue: tuple[str, ...] = ()
    code_unbacked: tuple[str, ...] = ()
    # Why each of unsupported_claims is flagged, index-aligned (`claim_reasons`).
    reasons: tuple[ClaimReasonCode, ...] = ()


def merge_reasons(
    pairs: Iterable[tuple[str, ClaimReasonCode]],
) -> tuple[tuple[str, ...], tuple[ClaimReasonCode, ...]]:
    """(claims, reasons) from (text, reason) pairs: one entry per text at its first position,
    "not_found" winning when a text comes with both reasons. The two tuples are split only AFTER
    the merge, so they are index-aligned by construction (`TurnVerification.claim_reasons`)."""
    merged: dict[str, ClaimReasonCode] = {}
    for text, reason in pairs:
        merged[text] = NOT_FOUND if NOT_FOUND in (merged.get(text), reason) else reason
    return tuple(merged), tuple(merged.values())


def flag_reasons(
    answer: str,
    attribution: Attribution,
    verifier_claims: Sequence[str] = (),
    left: Collection[str] = (),
    *,
    placed: bool = True,
) -> tuple[tuple[str, ...], tuple[ClaimReasonCode, ...]]:
    """Every flagged claim with its reason, by the SOURCE of the flag, in the order the claims
    have always had (verifier claims, residue, code-unbacked units, other uncited units):
    - a ground-check claim and a residue unit (`left`): "not_found";
    - a code-unbacked unit: its `code_reason` ("inexact_code" when listed in
      `Attribution.code_inexact`), which takes precedence over its own `uncited` membership
      (attribute adds every code-unbacked unit there too), EXCEPT that "not_found" wins when the
      unit is also residue on its own account or a verifier claim OVERLAPS it by position
      (`_overlaps`: a claim quoting part of the unit objects to the unit, whatever the code);
    - any other uncited unit: "not_found".
    `code_inexact` already excludes a text that also occurs as a plain uncited unit (`attribute`),
    so one rule per text suffices: muted iff it is in `muted`, and the verifier's own claims are
    never muted. This never changes the status: an "inexact_code" claim still makes the answer
    `partial`.

    `placed` is False when the verifier's objections could not all be placed in the answer (see
    `settle`): nothing is muted then. WHY (blind review round 1, P2): the muted flag covers the
    whole sentence, prose included, and an objection nobody can locate may be to that prose
    ("Delete it with `kafka-topics --create ...`"), so it fails toward the prominent flag."""
    muted = {
        unit
        for unit in (attribution.code_inexact if placed else ())
        if unit not in left and not _overlaps(answer, unit, verifier_claims)
    }
    texts = [*left, *attribution.code_unbacked, *attribution.uncited]
    return merge_reasons(
        [
            *((c, NOT_FOUND) for c in verifier_claims),
            *((t, INEXACT_CODE if t in muted else NOT_FOUND) for t in texts),
        ]
    )


def uncited_reasons(
    attribution: Attribution,
) -> tuple[tuple[str, ...], tuple[ClaimReasonCode, ...]]:
    """`attribution.uncited` (answer order, as the claims of a turn rejected before the ground
    check) with their reasons: "inexact_code" for `code_inexact` units, else "not_found". No
    verifier ran, so nothing can overlap a muted unit."""
    code_inexact = set(attribution.code_inexact)
    return merge_reasons(
        (u, INEXACT_CODE if u in code_inexact else NOT_FOUND) for u in attribution.uncited
    )


def all_not_found(claims: Sequence[str]) -> tuple[ClaimReasonCode, ...]:
    """The reasons of claims no attribution classified (an invalid label, an "unsupported"
    verdict): all "not_found", the prominent flag."""
    return tuple(NOT_FOUND for _ in claims)


def _spans(answer: str, text: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start = answer.find(text)
    while text and start != -1:
        spans.append((start, start + len(text)))
        start = answer.find(text, start + 1)
    return spans


def _overlaps(answer: str, sentence: str, claims: Sequence[str]) -> bool:
    """True if a flagged claim shares any characters with the unit in the answer. Claims are
    verbatim substrings (grounding.parse_verdict), so an overlap is a positional fact."""
    mine = _spans(answer, sentence)
    return any(
        s < ce and cs < e for claim in claims for cs, ce in _spans(answer, claim) for s, e in mine
    )


def settle(
    answer: str,
    *,
    verdict: Verification,
    conclusive: bool,
    located: bool,
    verifier_claims: Sequence[str],
    valid: Collection[str],
    attribution: Attribution,
    tools_ran: bool,
    names: frozenset[str] = frozenset(),
) -> Settled:
    """Final status of an answer mixing model and inferred citations (ADR-0015, "Status").

    0. A pure refusal makes no claim: `unverified`, nothing highlighted.
    1. Verifier says unsupported -> unsupported; every inferred citation is dropped.
    2. The claim-level gate is OPEN only for a conclusive verdict whose flagged claims were all
       located in the answer. Through an open gate an inferred citation survives unless a
       flagged claim overlaps its unit; through a closed gate none survives.
    3. The residue (coverage.residue) is computed with the survivors. It is highlighted, with
       the verifier's claims.
    4. Nothing covered (no valid model label, no survivor) and no tool ran: `unverified` when the
       gate was closed over existing candidates (rule 4a: shown, not checked, no citation),
       otherwise `unsupported`.
    5. `verified` only for a verified, located verdict with EMPTY residue and at least one
       covered unit; everything else is `partial`.
    6. Each flagged claim gets a reason (`Settled.reasons`, `flag_reasons`). This never
       changes the status: an "inexact_code" claim still makes the answer `partial`. Through a
       closed gate, or after a "partial" verdict that names no claim, every reason is
       "not_found": an objection that cannot be placed may be to any sentence.

    `names`: the `attribute` call's set (`coverage.dotted_names`), so the residue is cut into
    the same units attribution judged.
    """
    if attribution.refusal_only:
        return Settled(Verification.unverified, (), ())
    if verdict is Verification.unsupported:
        claims, reasons = merge_reasons((c, NOT_FOUND) for c in verifier_claims)
        return Settled(Verification.unsupported, (), claims, reasons=reasons)
    gate_open = conclusive and located
    kept = tuple(
        i
        for i in (attribution.inferred if gate_open else ())
        if not _overlaps(answer, i.sentence, verifier_claims)
    )
    # Model-labelled units whose code is not whole and exact in a cited page are residue too.
    # Everything attribution already rejected stays residue, whatever `residue` decides: it keys
    # survivors by TEXT, so a rejected unit (say, joined code, decided by POSITION) whose text
    # also occurs once in clean form would drop out with the clean twin (review round 6, P1).
    # Uncited units can never become covered later, so this only ever fails closed.
    left = residue(answer, valid, (i.sentence for i in kept), names=names)
    rest = list(dict.fromkeys([*left, *attribution.code_unbacked, *attribution.uncited]))
    # Reasons only change what the client SHOWS; the status below reads `rest`, as before.
    # An objection is placed only through an open gate, and a "partial" verdict naming no claim
    # objected to something it did not name: then nothing is muted (`flag_reasons`).
    placed = gate_open and (verdict is not Verification.partial or bool(verifier_claims))
    unsupported, reasons = flag_reasons(answer, attribution, verifier_claims, left, placed=placed)
    covered = attribution.covered or bool(kept)
    why = (tuple(rest), tuple(attribution.code_unbacked), reasons)
    if not covered and not tools_ran:
        if attribution.inferred and not gate_open:
            return Settled(Verification.unverified, (), unsupported, *why)
        return Settled(Verification.unsupported, (), unsupported, *why)
    if verdict is Verification.verified and located and covered and not rest:
        return Settled(Verification.verified, kept, unsupported, *why)
    return Settled(Verification.partial, kept, unsupported, *why)


def inferred_labels(inferred: Iterable[Inferred], model_cited: Iterable[int]) -> list[int]:
    """Distinct inferred labels in answer order, minus labels the model itself cited (a page
    the model cited is shown once, as a model citation: the stronger claim wins)."""
    cited = set(model_cited)
    return [n for n in dict.fromkeys(i.label for i in inferred) if n not in cited]
