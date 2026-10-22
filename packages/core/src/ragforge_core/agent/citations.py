"""Deterministic citation check (ADR-0011 rule 4): every cited label must name evidence the model
was actually SHOWN this turn; strict mode also requires at least one citation per sentence.

Citation syntax is `[S<n>]` (or `[S1, S3]`), where n is the 1-based position in the turn's hit
list, and `[T<n>]` for the n-th tool result of the turn (sync status, "Remembered.", the outcome
of an approved removal...). WHY tool labels: answers about the user's sources or about an action
the agent performed rest on tool output, not on retrieved pages; without a label they could
never satisfy strict mode, and a correct answer reporting a performed destructive action would
be regenerated and then abstained.

WHY labels instead of raw anchors: anchors are opaque ids (Notion block UUIDs) that small local
models copy unreliably; a short label is copied exactly, and the mapping label -> hit ->
anchor is done here, deterministically, so a model can never cite an anchor that was not
retrieved.

Offending sentences are returned VERBATIM (exact substrings of the answer) because the SSE
contract highlights unsupported claims by text match after sanitization (TurnVerification).

Lenient input, canonical output (ADR-0015): small local models rarely copy `[S1]` exactly; they
write `[1]`, `(S1)`, `[Source 1]`, `[s1; s2]` or "Source: S1". `normalize_citations` rewrites every
such CITATION-SHAPED marker to the canonical `[S<n>]` form BEFORE the check, so a correct answer
is not discarded for its punctuation. It never decides validity: a lenient marker naming a label
that was not shown becomes a canonical invalid label, which `check_citations` then rejects
exactly like a fabricated `[S9]`. Accepting it silently, or leaving it in the text, would show
the reader provenance that does not exist.

One definition of "this sentence carries a citation" (ADR-0015, review round 2): a CITATION-SHAPED
fragment that is neither canonical nor normalizable (`[S2-S9]` with a range too wide to expand,
`[S2, S10000]`, `[1, S2]`, `(S1-S3)` mid-sentence) is an INVALID label, so its sentence is
offending. Before this rule such a sentence was invisible to both checks: `check_citations` saw
no canonical label (so nothing invalid) and attribution saw a `[S<digit>` prefix (so "the model
cited it"), and the reader was shown a reference to a source that was never shown.

One definition of a unit and of where a label can be (coverage review, round 1; parser rewrite,
round 4): the check runs over `coverage.segment`, the units the guarantee is computed on;
normalisation rewrites ONLY `structure.Structure.label_spans` and stray fragments are looked for
ONLY in `coverage.visible_runs`, both the plain visible text of the CommonMark parse (not code at
any depth, link destinations, autolinks, strikethrough or HTML). There is no second sentence
splitter, fence regex or code-span scanner here: strict mode's "one label per sentence" means one
label per coverage unit, and text is code for the check exactly when it is code for the guarantee.
"""

from __future__ import annotations

import re
from bisect import bisect_right
from collections.abc import Collection, Iterator
from dataclasses import dataclass
from typing import Final

from ragforge_core.agent.coverage import segment, visible_runs
from ragforge_core.agent.labels import CITATION_MARKER, all_labels, label_names
from ragforge_core.agent.structure import structure

__all__ = [
    "CITATION_MARKER",
    "CitationReport",
    "check_citations",
    "label_names",
    "labels_in",
    "normalize_citations",
]


# ---------------------------------------------------------------- lenient forms (normalized)
# Each alternative captures the label list in one named group. All require a citation-shaped
# context (brackets, a parenthesis at a clause end, or a line-final "Source:" cue) so prose such
# as "Amazon S3", "see S3", "arr[1]" or "[2024]" is never read as a citation. Accepted risk: a
# bracketed small number at a clause end ("... in Python [3].") reads as a citation, because `[1]`
# is the form small models write most; if that label was not shown it is rejected like any
# fabricated one. Mid-clause ("list [1, 2] in Python") it is left as prose.
_SEP: Final = r"(?:\s*(?:[,;]|\band\b)\s*|\s+)"
_SLABEL: Final = r"[STst]\s?\d{1,3}"
_NLABEL: Final = r"\d{1,2}"
# a list that starts with an S/T label may continue with bare numbers: "[S1, 7]" names S7
_SLIST: Final = rf"{_SLABEL}(?:{_SEP}(?:{_SLABEL}|{_NLABEL}))*"
_LENIENT: Final = re.compile(
    # [S1] [s1] [S1, S2] [S1; S2] [S 1] [Source 1] [Sources 1, 2] [source: S1] [Source S1]
    rf"(?<!\w)\[\s*(?:sources?\s*:?\s*)?(?P<a>{_SLIST})\s*\](?![(:])"
    rf"|(?<!\w)\[\s*sources?\s*:?\s*(?P<b>{_NLABEL}(?:{_SEP}{_NLABEL})*)\s*\](?![(:])"
    # [1] [1, 2] (numbers only, at most two digits: "[2024]" is a year, not a citation), at a
    # clause end only (review round 3): in "Use list [1, 2] in Python." the brackets are prose,
    # and reading them as S1/S2 would show model provenance the model never wrote
    rf"|(?<!\w)\[(?P<c>{_NLABEL}(?:\s*[,;]\s*{_NLABEL})*)\](?![(:\w])"
    rf"(?=\s*(?:[.!?,;:\[(]|$))"
    # (S1) (S1, S2) (Source 1) (source: S2), at a clause end only: in "Amazon Simple Storage
    # Service (S3) is cheap" the parenthesis names a product, not a source
    rf"|\(\s*(?:sources?\s*:?\s*)?(?P<d>{_SLIST})\s*\)(?=\s*(?:[.!?,;:\[(]|$))"
    rf"|\(\s*sources?\s*:?\s*(?P<e>{_NLABEL}(?:{_SEP}{_NLABEL})*)\s*\)"
    # a bare S1 only after a "Source:" cue that starts a line or follows a sentence end, AND only
    # when it ends the line ("... partitions.\nSource: S1"). No "see"/"cf." cue: "see S3",
    # "cf. S3" and "(see S3)" are how people write about AWS S3, which HLD notes are full of.
    rf"|(?P<cue>(?:^|(?<=[.!?])\s+)sources?(?:\s*:\s*|\s+))(?P<f>{_SLIST})(?=\s*[.!?]?\s*$)",
    re.IGNORECASE,
)
_ANY_LABEL: Final = re.compile(r"([STst])?\s?(\d{1,3})")

# A bracketed label range: [S1-S3], [S1-3], [S1 to S3], [Sources 1-3]. Expanded to the labels it
# names when it spans at most _MAX_RANGE labels; a wider or descending range stays as written and
# is then rejected as a stray fragment (`_stray_spans`), never silently trusted.
_RANGE: Final = re.compile(
    r"(?<!\w)\[\s*(?:sources?\s*:?\s*)?(?P<kind>[ST])\s?(?P<lo>\d{1,3})"
    r"\s*(?:-|\u2013|\u2014|\bto\b|\bthrough\b)\s*(?:(?P=kind)\s?)?(?P<hi>\d{1,3})\s*\](?![(:])",
    re.IGNORECASE,
)
_MAX_RANGE: Final = 8

# What a citation-shaped fragment is made of: labels, numbers, separators and range words. A
# bracket holding anything else ("[S3 bucket]", "[S3 docs](url)") is prose, not a citation.
_LABELISH: Final = r"(?:[\sSTst\d,;:/&+\-\u2013\u2014]|\b(?:and|to|through)\b)"
_STRAY_BRACKET: Final = re.compile(rf"\[(?P<body>{_LABELISH}*)\]", re.IGNORECASE)
# A parenthesis is citation-shaped only with TWO labels and a separator: "(S3)" alone is how
# people name Amazon S3 mid-sentence, and "(S3, EC2 and Lambda)" lists AWS services. Found in
# two linear steps (review round 2, P1: the single lazy regex backtracked quadratically, 4.5 s on
# "(" + "S1-" * 7000): a parenthesis without nested ones, whose body is only label-ish text and
# holds a label, a separator and a second number (possessive quantifiers: no backtracking).
_PAREN: Final = re.compile(r"\((?P<body>[^()]*)\)")
_RANGE_WORDS: Final = re.compile(r"\b(?:and|to|through)\b", re.IGNORECASE)
_LABELISH_CHARS: Final = frozenset("STst,;:/&+-\u2013\u2014")  # plus whitespace and digits
_TWO_LABELS: Final = re.compile(
    r"[ST]\s?+\d++\s*+(?:[-,;/&+\u2013\u2014]|\b(?:and|to|through)\b)\s*+(?:[ST]\s?+)?\d",
    re.IGNORECASE,
)


def _paren_fragments(prose: str) -> Iterator[re.Match[str]]:
    for match in _PAREN.finditer(prose):
        body = _RANGE_WORDS.sub(" ", match["body"])
        if all(
            c.isspace() or c.isdigit() or c in _LABELISH_CHARS for c in body
        ) and _TWO_LABELS.search(match["body"]):
            yield match


_HAS_LABEL: Final = re.compile(r"[ST]\s?\d", re.IGNORECASE)


def _expand_range(match: re.Match[str]) -> str:
    kind, lo, hi = match["kind"].upper(), int(match["lo"]), int(match["hi"])
    if not lo <= hi < lo + _MAX_RANGE:
        return match.group(0)
    return "[" + ", ".join(f"{kind}{n}" for n in range(lo, hi + 1)) + "]"


def _rewrite(text: str) -> str:
    return _LENIENT.sub(_canonical, _RANGE.sub(_expand_range, text))


# The words right before a clause-end "(S3)": a run of Capitalised words (or "AWS"/"S3" tokens),
# looked for only in the _NAME_WINDOW characters before it (review round 2, P1: searching the whole
# prefix before every "(S<n>)" was quadratic, 7 s on a 30k-character capitalised run).
_NAME_RUN: Final = re.compile(r"(?:\b[A-Z][\w-]*\s+)*\b[A-Z][\w-]*\s*\Z")
_NAME_WINDOW: Final = 120
_VENDORS: Final = frozenset({"amazon", "aws"})


def _names_a_product(before: str, label: str) -> bool:
    """A single "(S<n>)" that closes a product name or defines its acronym, not a citation.

    Review round 5, P2: "Media is stored in Amazon Simple Storage Service (S3)." was rewritten to
    "... [S3].", a model citation the model never wrote (or, with fewer than 3 sources shown, a
    rejected correct answer). Left as prose when the parenthesis directly follows:
    - the same token ("Logs go to S3 (S3).");
    - a Capitalised run naming a vendor ("Amazon ...", "AWS ...");
    - a Capitalised run with at least n words starting with the label's letter: the acronym
      definition "Simple Storage Service (S3)" (three S-words).
    Residual risk (ADR-0015): "stored in object storage (S3)." still reads as a citation."""
    match = _NAME_RUN.search(before)
    if match is None:
        return False
    words = match.group(0).split()
    letter, count = label[0].casefold(), int(label[1:])
    if words[-1].casefold() == label.casefold():
        return True
    if any(w.casefold() in _VENDORS for w in words):
        return True
    return len(words) >= 2 and sum(w[0].casefold() == letter for w in words) >= count


def _canonical(match: re.Match[str]) -> str:
    """The canonical marker for one lenient match: numbers alone mean sources (S)."""
    group = next(g for g in "abcdef" if match.group(g) is not None)
    if group == "d":
        single = re.fullmatch(r"\s*([STst])\s?(\d{1,3})\s*", match.group("d"))
        before = match.string[max(0, match.start() - _NAME_WINDOW) : match.start()]
        if single is not None and _names_a_product(before, "".join(single.groups()).upper()):
            return match.group(0)
    labels = [
        f"{(kind or 'S').upper()}{int(n)}" for kind, n in _ANY_LABEL.findall(match.group(group))
    ]
    marker = f"[{', '.join(dict.fromkeys(labels))}]"
    cue = match.group("cue")
    # "x, see S1." -> "x, [S1]." keeps one separating space; a cue at line start keeps none
    return f" {marker}" if cue is not None and cue[:1].isspace() else marker


_MAX_PASSES: Final = 3


def normalize_citations(answer: str) -> str:
    """Every citation-shaped marker in plain visible text rewritten to `[S<n>]`/`[T<n>]` (see
    the module docstring). Code at any depth, links, strikethrough and HTML are untouched;
    validity is NOT decided here.

    Run to a fixpoint (bounded): one rewrite can expose another, e.g. `[s1](S2)` is skipped as a
    possible Markdown link until `(S2)` becomes `[S2]`. Every pass only produces canonical
    markers, which are fixed points, so the result is idempotent."""
    for _ in range(_MAX_PASSES):
        normalized = _normalize_once(answer)
        if normalized == answer:
            break
        answer = normalized
    return answer


def _normalize_once(answer: str) -> str:
    """One rewrite pass over the plain visible text of the parse (`label_spans`)."""
    out: list[str] = []
    pos = 0
    for start, end in structure(answer).label_spans:
        out.append(answer[pos:start])
        out.append(_rewrite(answer[start:end]))
        pos = end
    out.append(answer[pos:])
    return "".join(out)


@dataclass(frozen=True, slots=True)
class CitationReport:
    cited: tuple[int, ...]  # valid SOURCE labels in first-citation order (deduplicated)
    # labels ("S9", "T4") that name nothing the model was shown, and stray fragments ("[S2-S9]")
    invalid: tuple[str, ...]
    offending: tuple[str, ...]  # verbatim units that fail the check

    @property
    def ok(self) -> bool:
        return not self.offending


def _strays(text: str) -> list[re.Match[str]]:
    """Citation-shaped fragments of `text` (plain visible text) that are NOT canonical markers
    (offsets are offsets of `text`). Canonical markers are blanked first, so `[S1][S2]` is not a
    fragment."""
    prose = CITATION_MARKER.sub(lambda m: " " * len(m.group(0)), text)
    found = [*_STRAY_BRACKET.finditer(prose), *_paren_fragments(prose)]
    return [m for m in sorted(found, key=re.Match.start) if _HAS_LABEL.search(m["body"])]


def _stray_spans(text: str, names: frozenset[str] = frozenset()) -> list[tuple[int, int]]:
    """[start, end) of every stray fragment of `text`, looked for in the text the reader sees
    (`coverage.visible_runs`): a citation-shaped fragment that is not canonical, AND a canonical
    marker the reader sees but that is not accepted as a label (escaped, struck through, in a
    link's text or a URL-shaped word): the reader would take both for a citation."""
    spans: list[tuple[int, int]] = []
    for run in visible_runs(text, names=names):
        found = [m for m in CITATION_MARKER.finditer(run.text) if m.start() not in run.labels]
        found.extend(_strays(run.text))
        spans.extend((run.starts[m.start()], run.ends[m.end() - 1]) for m in found)
    return sorted(spans)


def labels_in(text: str) -> list[int]:
    """Source labels ([S<n>]) in order of appearance."""
    return [n for kind, n in all_labels(text) if kind == "S"]


def check_citations(
    answer: str,
    source_labels: Collection[int],
    *,
    n_tool_results: int = 0,
    strict: bool,
    names: frozenset[str] = frozenset(),
) -> CitationReport:
    """Every label of every coverage unit must name something the model was shown (rule 1);
    a citation-shaped fragment that is not a canonical marker is invalid; strict mode also needs
    a label on every prose unit that states something and is not allowlisted. Code units are
    not required to carry one (the residue makes uncited code `partial`), but a label citing a
    code block is checked like any other.

    `source_labels` are the hit labels the model was shown (not every hit of the turn: a weak
    first-pass hit dropped by the rewrite was never in its prompt, so citing it proves nothing).
    """
    shown = set(source_labels)
    # `names`: dotted names of the shown sources (coverage.dotted_names), so a backed
    # "logger.name" does not cut the unit and leave an unlabelled fragment (DOT, ADR-0015)
    units = segment(answer, names=names)
    # fragments are looked for per plain-text span, so a unit cut can never split one in two
    strays = _stray_spans(answer, names)
    # each fragment is matched to the units it overlaps by bisection (review round 2, P1: the
    # all-pairs test was quadratic, 4.5 s on "(" + "S1-" * 7000)
    starts = [u.start for u in units]
    held: dict[int, list[tuple[int, int]]] = {}
    for s, e in strays:
        index = max(bisect_right(starts, s) - 1, 0)
        while index < len(units) and units[index].start < e:
            if s < units[index].end:
                held.setdefault(index, []).append((s, e))
            index += 1
    # ordered sets: an answer may hold thousands of labels (linear, not quadratic)
    cited: dict[int, None] = {}
    invalid: dict[str, None] = {}
    offending: dict[str, None] = {}
    claimed: set[tuple[int, int]] = set()
    for index, unit in enumerate(units):
        labels = [(name[0], int(name[1:])) for name in unit.labels]
        bad = [
            f"{kind}{n}"
            for kind, n in labels
            if not (n in shown if kind == "S" else 1 <= n <= n_tool_results)
        ]
        mine = held.get(index, [])
        claimed.update(mine)
        bad.extend(answer[s:e] for s, e in mine)  # names no label we can check: invalid
        invalid.update(dict.fromkeys(bad))
        cited.update(dict.fromkeys(n for kind, n in labels if kind == "S" and n in shown))
        needs_label = strict and not unit.code and unit.substantive and unit.allowed is None
        if bad or (needs_label and not unit.labels):
            # Highlight the unit AND every invalid fragment it holds: a fragment may begin
            # before the unit ("[;S0]" ahead of the label that ends it).
            start = min([unit.start, *(fs for fs, _ in mine)])
            end = max([unit.end, *(fe for _, fe in mine)])
            offending[answer[start:end]] = None
    for s, e in strays:
        if (s, e) not in claimed:  # a fragment no unit holds is still shown to the reader
            invalid[answer[s:e]] = None
            offending[answer[s:e]] = None
    return CitationReport(cited=tuple(cited), invalid=tuple(invalid), offending=tuple(offending))
