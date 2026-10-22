"""The answer's structure, from a real CommonMark parse (ADR-0015 section 4).

WHY a parser (review of the coverage redesign, round 4): the guarantee was computed over a
structure found by regexes (fence lines, indented lines, backtick runs, list markers, table
rows), a second Markdown grammar next to the one the reader's renderer uses. Every place the two
disagreed was a bypass: indented code with a label read as prose, a fence nested in a quote or a
list item read as prose, an inline span cut in two by "." or a line break, a double-backtick
span, a label inside a link destination, an autolink or a strikethrough. Now ONE structural map
comes from markdown-it-py (CommonMark, plus GFM tables and strikethrough, plus linkify when
linkify-it-py is installed), with source positions:

- every code element at any nesting depth: fenced and indented blocks (`Block`, kind "code")
  and every inline span (`Atom`, kind CODE), each with the parser's own content;
- every visible text run (`Atom` kinds TEXT/ESCAPE), and which of them may hold a citation
  label (`Atom.eligible`: plain text outside links, autolinks, strikethrough, HTML and code);
- links (their destination and title are not visible text, so every atom of a link is
  `Atom.opaque`), autolinks and linkified URLs, inline and block HTML, images, hard and soft
  line breaks;
- everything else (reference definitions, extra table cells the parser drops, numerals of a
  list that does not start at 1 or of an item with nothing but labels in it, any token type
  this module does not know) as OPAQUE blocks.

Positions: markdown-it-py gives block tokens line ranges but inline tokens no offsets. Each
inline token's content is aligned back onto its source lines, and each child token is aligned
onto that content IN ORDER, every step checked (a text token must equal the source it spans; a
link's extent is recomputed by markdown-it-py's own link rule). Any step that does not check out
makes the whole block opaque: fail closed, never guessed.

Accounting: every source character is inside a part (an inline block, a code/HTML block, an
opaque block) or is structure that states nothing (container markers, fences of a code part,
table delimiters, blank lines). A leftover that says something becomes an opaque block, so text
the parser hides from its tokens (a reference definition, a dropped cell) is residue, never
nothing.
"""

from __future__ import annotations

import importlib.util
import re
import unicodedata
from bisect import bisect_right
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field, replace
from enum import StrEnum
from functools import lru_cache
from typing import Final

from markdown_it import MarkdownIt
from markdown_it.rules_inline import image as image_rule
from markdown_it.rules_inline import link as link_rule
from markdown_it.rules_inline.state_inline import StateInline
from markdown_it.token import Token

from ragforge_core.agent.labels import CITATION_MARKER

# ---------------------------------------------------------------- "states nothing": an allowlist
# Review of the redesign, round 2 (P1): "states nothing" was `not str.isalnum()`, a denylist one
# layer down. A character is IGNORABLE only when it is positively known to state nothing:
# - whitespace, Unicode punctuation (P*), separators (Z*), combining marks (M*), format
#   characters (Cf: ZWJ, ZWSP, the BOM);
# - the ASCII symbols markdown is made of: "|" (tables), ">" (quotes), "`"/"~" (code), "+", "<",
#   "=", "^";
# - a pictograph: a symbol (So/Sk/Sm) in the arrow, technical, geometric, dingbat and emoji
#   blocks (the regional indicators and the enclosed/squared letters of U+1F100-1F1FF are
#   deliberately outside the ranges).
# AND its NFKC form has no letter or digit, and its Unicode name does not say LETTER/DIGIT/
# NUMBER/NUMERAL/KEYCAP/FRACTION. Unassigned, private-use and control characters are
# substantive. ACCEPTED RESIDUAL (ADR-0015 section 4): a pictograph can carry meaning ("👍" for
# "yes"); it states nothing here, so an answer's pictographs are never residue.
_MARKDOWN_SYMBOLS: Final = frozenset("|>`~+<=^")
_PICTOGRAPH_BLOCKS: Final = (
    (0x2190, 0x21FF),  # arrows
    (0x2300, 0x23FF),  # miscellaneous technical (watch, hourglass...)
    (0x25A0, 0x27BF),  # geometric shapes, miscellaneous symbols, dingbats
    (0x2B00, 0x2BFF),  # miscellaneous symbols and arrows
    (0x1F300, 0x1F6FF),  # pictographs, emoticons, transport
    (0x1F900, 0x1FAFF),  # supplemental pictographs, symbols and pictographs extended-A
)
_TEXT_NAMES: Final = ("LETTER", "DIGIT", "NUMBER", "NUMERAL", "KEYCAP", "FRACTION")


@lru_cache(maxsize=16384)
def states_nothing(char: str) -> bool:
    """`char` is positively known to state nothing (see the allowlist above)."""
    if char.isspace():
        return True
    category = unicodedata.category(char)
    code = ord(char)
    kind = (
        category[0] in "PZM"
        or category == "Cf"
        or char in _MARKDOWN_SYMBOLS
        or (
            category in ("So", "Sk", "Sm")
            and any(lo <= code <= hi for lo, hi in _PICTOGRAPH_BLOCKS)
        )
    )
    if not kind:
        return False
    name = unicodedata.name(char, "")
    if not name or any(word in name for word in _TEXT_NAMES):
        return False
    return not any(c.isalnum() for c in unicodedata.normalize("NFKC", char))


def says_something(text: str) -> bool:
    """`text` holds a character that is not positively known to state nothing."""
    return not all(states_nothing(c) for c in text)


# ---------------------------------------------------------------- the parser
LINKIFY: Final = importlib.util.find_spec("linkify_it") is not None
"""linkify-it-py is optional (handoff P2B-AG request 8). The map does not depend on it for the
guarantee: without it a bare URL is text (a label inside a URL-shaped word never counts and is an
invalid fragment, `coverage._labels`); with it the URL is an atom whose text is scanned for
markers all the same (`coverage.visible_runs`). Only sentence cuts inside a URL differ, and the
module_ag property and attacks run under both configurations."""


def _parser(linkify: bool = LINKIFY) -> MarkdownIt:
    """CommonMark with raw HTML (so HTML is SEEN and classified, not read as prose), GFM tables
    and GFM strikethrough (one or two tildes, opener and closer of the same length, a run of
    three or more is text: markdown-it-py's own `strikethrough_single_tilde` mode, so the parser,
    not a heuristic, decides what is struck through), linkify when available (the client
    configuration the E1 handoff recommends: no fuzzy links). `text_join` is off so escapes and
    entities stay separate tokens: a label is read only where its source is literally the
    label."""
    md = MarkdownIt(
        "commonmark", {"html": True, "linkify": linkify, "strikethrough_single_tilde": True}
    )
    md.enable(["table", "strikethrough"])
    if linkify:
        md.enable("linkify")
        if md.linkify is not None:
            md.linkify.set({"fuzzy_link": False})
    md.disable("text_join")
    return md


_MD: Final = _parser()


def markdown_tokens(text: str) -> list[Token]:
    """The block tokens of `text` as the verifier's own CommonMark configuration reads it."""
    return _MD.parse(text)


# markdown-it's own (non-GFM) strikethrough: "~~" pairs, and a run of three or more tildes split
# into pairs. Only consulted for an inline part holding such a run (`_inline`), where the two
# readings can differ: a label struck through by EITHER reading is not a label (fail closed).
_PAIRS: Final = MarkdownIt("commonmark", {"html": True}).enable("strikethrough")

_BARE_NAMES: Final = frozenset(
    """
    br hr p div li ul ol tr td th table thead tbody h1 h2 h3 h4 h5 h6 blockquote pre details
    summary section article dl dt dd
    """.split()  # noqa: SIM905
)


def _bare_tag(tag: str) -> bool:
    """`tag`, an HTML tag the PARSER recognised, is a bare line-break or block tag: its name and
    nothing else (attribute text is text the reader sees; the sanitizer shows raw HTML
    literally)."""
    inner = tag[1:-1] if tag.startswith("<") and tag.endswith(">") else ""
    inner = inner.removeprefix("/").strip(" \t\n").removesuffix("/").rstrip(" \t\n")
    return inner.casefold() in _BARE_NAMES


@lru_cache(maxsize=1024)
def bare_html(text: str) -> bool:
    """`text` (an inline HTML token or an HTML block) is, by the parser's own reading, nothing but
    bare break/block tags and whitespace: it states nothing (ADR-0015 section 4 (c))."""
    if not text.strip():
        return False
    children = [c for t in _MD.parseInline(text.strip()) for c in t.children or []]
    return bool(children) and all(
        (c.type == "html_inline" and _bare_tag(c.content))
        or c.type in ("softbreak", "hardbreak")
        or (c.type == "text" and not c.content.strip())
        for c in children
    )


# ---------------------------------------------------------------- the map
class Kind(StrEnum):
    """What one inline piece of an answer is, for the reader."""

    text = "text"  # literal text: its source is exactly what the reader sees
    escape = "escape"  # a backslash escape or an entity: visible, never part of a label
    code = "code"  # an inline code span (atomic: never cut, never a label)
    html = "html"  # inline HTML or a comment (atomic, a boundary on both sides)
    soft = "soft"  # a soft line break
    hard = "hard"  # a hard line break
    image = "image"  # an image (atomic; the sanitizer replaces it with a notice)
    url = "url"  # an autolink or a linkified URL (atomic; visible, never a label)
    mark = "mark"  # emphasis/strikethrough delimiters, link brackets and destination: no text


@dataclass(frozen=True, slots=True)
class Code:
    """One code element of a document: a fenced/indented block or an inline span."""

    start: int  # [start, end) in the document (a block: its whole lines, fences included)
    end: int
    text: str  # the parser's content (an inline span: CommonMark-normalised)
    block: bool
    info: str = ""  # a fence's info string


@dataclass(frozen=True, slots=True)
class Atom:
    """One inline piece of an `Inline` part, with its source span."""

    kind: Kind
    start: int
    end: int
    visible: str = ""  # what the reader sees ("" for marks)
    eligible: bool = False  # plain text outside links, strikethrough, HTML and code
    code: Code | None = None
    opaque: bool = False  # in or of a link (its destination/title), or an image


@dataclass(frozen=True, slots=True)
class Inline:
    """A paragraph, heading or table cell: its inline content and where it sits."""

    start: int  # [start, end) of its content in the document
    end: int
    atoms: tuple[Atom, ...]
    kind: str  # "paragraph" | "heading" | "cell"
    first_line: int
    last_line: int
    parent: int  # the innermost container (token index), -1 at top level
    row: int = -1  # a table cell's row (token index)


@dataclass(frozen=True, slots=True)
class Block:
    """A code block, an HTML block, or text the parser hides or this module cannot classify."""

    kind: str  # "code" | "html" | "opaque"
    start: int
    end: int
    first_line: int
    last_line: int
    parent: int = -1
    code: Code | None = None


Part = Inline | Block


@dataclass(frozen=True, slots=True)
class Structure:
    parts: tuple[Part, ...]  # in document order
    code: tuple[Code, ...]  # every code element, at any depth, in document order
    label_spans: tuple[tuple[int, int], ...]  # where a citation label may be written


# ---------------------------------------------------------------- source positions
# Container markers and whitespace before a content line; whitespace (an ATX heading's closing
# hashes) after it. "Whitespace" is Unicode whitespace (`\s`, the set `str.isspace` and so
# `str.strip` use): markdown-it strips a paragraph's, heading's and cell's content with
# `str.strip()`, so a trailing NBSP, U+202F or U+3000 is not in the content (blind review round 3,
# P2: an ASCII-only suffix made such a paragraph fail alignment and become opaque). What these
# skip is unclaimed text, accounted by `_gaps` like any other (whitespace states nothing).
_PREFIX: Final = re.compile(r"[\s>*+\-#\d.)]*")
_SUFFIX: Final = re.compile(r"[\s#]*")
_BREAK: Final = re.compile(r"[ \t]*\\?\n[ \t]*")


class _Misaligned(Exception):
    """The parser's view could not be mapped back onto the source: the part is opaque."""


def _normalized(text: str) -> tuple[str, list[int] | None]:
    """markdown-it's own input normalisation (CR LF and CR to LF, NUL to U+FFFD), with the map
    back to the original offsets (None when nothing changed)."""
    if "\r" not in text and "\0" not in text:
        return text, None
    out: list[str] = []
    back: list[int] = []
    i = 0
    while i < len(text):
        char = text[i]
        back.append(i)
        if char == "\r":
            out.append("\n")
            i += 2 if text[i + 1 : i + 2] == "\n" else 1
            continue
        out.append("�" if char == "\0" else char)
        i += 1
    back.append(len(text))
    return "".join(out), back


class _Doc:
    """The parse of one text, in the parser's (normalised) coordinates."""

    def __init__(self, text: str) -> None:
        self.src = text
        self.env: dict[str, object] = {}
        self.tokens = _MD.parse(text, self.env)
        self.starts = [0]
        for i, char in enumerate(text):
            if char == "\n":
                self.starts.append(i + 1)

    def line(self, n: int) -> tuple[int, int]:
        """[start, end) of line n, without its newline."""
        start = self.starts[n]
        end = self.starts[n + 1] - 1 if n + 1 < len(self.starts) else len(self.src)
        return start, end

    def lines(self, first: int, last: int) -> tuple[int, int]:
        """[start, end) of lines first..last (inclusive)."""
        return self.line(first)[0], self.line(min(last, len(self.starts) - 1))[1]


def _line_offsets(doc: _Doc, first_line: int, content: str) -> tuple[list[int], list[int]]:
    """Source [start, end) of every character of a paragraph's or heading's inline content: each
    content line is its source line minus container markers before it and whitespace (or a
    heading's closing hashes) after it."""
    starts: list[int] = []
    ends: list[int] = []
    rows = content.split("\n")
    for i, row in enumerate(rows):
        if first_line + i >= len(doc.starts):
            raise _Misaligned
        a, b = doc.line(first_line + i)
        line = doc.src[a:b]
        at = line.find(row) if row else -1
        while at != -1 and not (
            _PREFIX.fullmatch(line, 0, at) and _SUFFIX.fullmatch(line, at + len(row))
        ):
            at = line.find(row, at + 1)
        if at == -1:
            raise _Misaligned
        starts.extend(range(a + at, a + at + len(row)))
        ends.extend(range(a + at + 1, a + at + len(row) + 1))
        if i + 1 < len(rows):
            starts.append(b)
            ends.append(b + 1)
    return starts, ends


def _cell_offsets(doc: _Doc, cursor: int, end: int, content: str) -> tuple[list[int], list[int]]:
    """Source offsets of a table cell's content, read from `cursor` on its row: the table rule
    trims the cell (`str.strip`: any Unicode whitespace) and turns an escaped pipe into a pipe,
    nothing else."""
    pos = cursor
    while pos < end and (doc.src[pos].isspace() or doc.src[pos] == "|"):  # str.strip's set
        pos += 1
    starts: list[int] = []
    ends: list[int] = []
    for char in content:
        if pos < end and doc.src[pos] == char:
            starts.append(pos)
            ends.append(pos + 1)
            pos += 1
        elif char == "|" and doc.src[pos : pos + 2] == "\\|":
            starts.append(pos)
            ends.append(pos + 2)
            pos += 2
        else:
            raise _Misaligned
    return starts, ends


def _extent(
    rule: Callable[[StateInline, bool], bool], content: str, pos: int, env: dict[str, object]
) -> int:
    """Where the construct at `pos` ends, by markdown-it-py's own rule run silently: a link's
    destination and title are parsed exactly as the parser parsed them."""
    state = StateInline(content, _MD, env, [])
    state.pos = pos
    if not rule(state, True):
        raise _Misaligned
    return state.pos


def _closing_ticks(content: str, pos: int, length: int) -> int:
    """End of the backtick run of exactly `length` that closes the span opened at `pos`."""
    at = pos + length
    while True:
        at = content.find("`", at)
        if at == -1:
            raise _Misaligned
        run = at
        while run < len(content) and content[run] == "`":
            run += 1
        if run - at == length:
            return run
        at = run


class _Aligner:
    """Aligns one inline token's children onto its content, in order, checking every step."""

    def __init__(self, content: str, starts: list[int], ends: list[int], env: dict[str, object]):
        self.content = content
        self.starts = starts
        self.ends = ends
        self.env = env
        self.pos = 0
        self.atoms: list[Atom] = []
        self.links: list[int] = []
        self.struck = 0

    def _span(self, end: int) -> tuple[int, int]:
        """Source [start, end) of content[pos:end]."""
        if end <= self.pos:
            raise _Misaligned
        return self.starts[self.pos], self.ends[end - 1]

    def _emit(
        self,
        kind: Kind,
        end: int,
        visible: str = "",
        *,
        eligible: bool = False,
        code: Code | None = None,
        opaque: bool = False,
    ) -> None:
        start, stop = self._span(end)
        opaque = opaque or bool(self.links)  # every atom of a link (see `_child`)
        self.atoms.append(Atom(kind, start, stop, visible, eligible, code, opaque))
        self.pos = end

    def _expect(self, text: str) -> int:
        if not text or not self.content.startswith(text, self.pos):
            raise _Misaligned
        return self.pos + len(text)

    def run(self, children: Sequence[Token]) -> tuple[Atom, ...]:
        i = 0
        while i < len(children):
            i = self._child(children, i)
        if self.content[self.pos :].strip():
            raise _Misaligned
        return tuple(self.atoms)

    def _child(self, children: Sequence[Token], i: int) -> int:
        token = children[i]
        kind = token.type
        if kind == "text":
            if token.content:
                end = self._expect(token.content)
                start, stop = self._span(end)
                # a table cell's "\|" reads as "|": visible, but not literally a label's source
                literal = stop - start == len(token.content)
                eligible = literal and not self.links and not self.struck
                kind_of = Kind.text if literal else Kind.escape
                self._emit(kind_of, end, token.content, eligible=eligible)
        elif kind == "text_special":
            self._emit(Kind.escape, self._expect(token.markup), token.content)
        elif kind in ("softbreak", "hardbreak"):
            match = _BREAK.match(self.content, self.pos)
            if match is None:
                raise _Misaligned
            self._emit(Kind.soft if kind == "softbreak" else Kind.hard, match.end(), "\n")
        elif kind == "code_inline":
            self._expect(token.markup)
            end = _closing_ticks(self.content, self.pos, len(token.markup))
            code = Code(*self._span(end), token.content, block=False)
            self._emit(Kind.code, end, token.content, code=code)
        elif kind == "html_inline":
            self._emit(Kind.html, self._expect(token.content), token.content)
        elif kind == "image":
            end = _extent(image_rule, self.content, self.pos, self.env)
            self._emit(Kind.image, end, "\ufffc", opaque=True)
        elif kind == "link_open" and token.markup in ("autolink", "linkify"):
            return self._url(children, i)
        elif kind == "link_open":
            # A link hides text the reader may still see: its destination (on hover, and the
            # sanitizer keeps it) and its title. EVERY atom from "[" to the end of the
            # destination is opaque, so any unit overlapping the link is opaque, wherever a
            # sentence cut falls inside its text (blind review round 3, P2: only "[" was, so
            # '[! Kafka ...](u "NASA built Kafka")' put the text, destination and title in an
            # inferred unit that did not hold the opaque atom).
            end = _extent(link_rule, self.content, self.pos, self.env)
            self._emit(Kind.mark, self._expect("["), opaque=True)
            self.links.append(end)
        elif kind == "link_close":
            if not self.links:
                raise _Misaligned
            self._expect("]")
            self._emit(Kind.mark, self.links[-1], opaque=True)
            self.links.pop()
        elif kind in ("s_open", "s_close", "em_open", "em_close", "strong_open", "strong_close"):
            self._emit(Kind.mark, self._expect(token.markup))
            self.struck += {"s_open": 1, "s_close": -1}.get(kind, 0)
        else:
            raise _Misaligned  # a token this module does not know: fail closed
        return i + 1

    def _url(self, children: Sequence[Token], i: int) -> int:
        """An autolink (`<url>`) or a linkified URL: one atomic visible piece."""
        if i + 2 >= len(children) or children[i + 2].type != "link_close":
            raise _Misaligned
        url = children[i + 1].content
        if children[i].markup == "autolink":
            end = self._expect("<")
            close = self.content.find(">", end)
            if close == -1:
                raise _Misaligned
            self._emit(Kind.url, close + 1, url)
        else:
            self._emit(Kind.url, self._expect(url), url)
        return i + 3


# ---------------------------------------------------------------- the block walk
_CONTAINERS: Final = frozenset({"blockquote_open", "list_item_open"})
_CLOSERS: Final = frozenset({"blockquote_close", "list_item_close"})
_STRUCTURE: Final = frozenset(
    """
    paragraph_open paragraph_close heading_open heading_close hr bullet_list_open
    bullet_list_close ordered_list_open ordered_list_close table_open table_close thead_open
    thead_close tbody_open tbody_close tr_open tr_close th_open th_close td_open td_close
    """.split()  # noqa: SIM905
)


@dataclass(slots=True)
class _List:
    ordered: bool
    start: int = 1
    numbered: bool = False  # its numerals state nothing (it continues the count, or starts at 1)
    items: int = 0


@dataclass(slots=True)
class _Walk:
    """Mutable state of `_walk`."""

    doc: _Doc
    parts: list[Part] = field(default_factory=list)
    claimed: list[tuple[int, int]] = field(default_factory=list)
    stack: list[int] = field(default_factory=list)
    lists: list[_List] = field(default_factory=list)
    counted: int = 0  # the last numeral the most recent ordered list rendered
    markers: dict[int, int] = field(default_factory=dict)  # line -> where to look for a marker
    # per open list item: its numeral's span (None: not an actual numbered list's) and the
    # number of parts before it (`_list_item_close`)
    items: list[tuple[tuple[int, int] | None, int]] = field(default_factory=list)
    row: int = -1
    row_lines: tuple[int, int] = (0, 0)
    row_cursor: int = 0
    cells: list[Inline] = field(default_factory=list)
    row_failed: bool = False

    @property
    def parent(self) -> int:
        return self.stack[-1] if self.stack else -1

    def opaque(self, first: int, last: int) -> None:
        start, end = self.doc.lines(first, last)
        self.parts.append(Block("opaque", start, end, first, last, self.parent))
        self.claimed.append((start, end))


def _inline(walk: _Walk, tokens: Sequence[Token], i: int) -> None:
    token = tokens[i]
    opener = tokens[i - 1].type if i else ""
    first, last = (token.map or [0, 1])[0], (token.map or [0, 1])[1] - 1
    if not token.content:
        return
    cell = opener in ("th_open", "td_open")
    try:
        if cell:
            _, row_end = walk.doc.line(walk.row_lines[0])
            starts, ends = _cell_offsets(walk.doc, walk.row_cursor, row_end, token.content)
            walk.row_cursor = ends[-1]
        else:
            starts, ends = _line_offsets(walk.doc, first, token.content)
        atoms = _Aligner(token.content, starts, ends, walk.doc.env).run(token.children or [])
    except _Misaligned:
        if cell:
            walk.row_failed = True
        else:
            walk.opaque(first, last)
        return
    if "~~~" in token.content and _struck_in_pairs(token.content):
        atoms = tuple(replace(a, eligible=False) for a in atoms)
    kind = "cell" if cell else ("heading" if opener == "heading_open" else "paragraph")
    part = Inline(starts[0], ends[-1], atoms, kind, first, last, walk.parent, walk.row)
    if cell:
        walk.cells.append(part)
    else:
        walk.parts.append(part)
        walk.claimed.append((part.start, part.end))


def _struck_in_pairs(content: str) -> bool:
    """A renderer with markdown-it's default strikethrough strikes part of `content` that GFM
    shows as text ("~~~[S1]~~~"): no label of the part counts (rare; fail closed)."""
    return any(c.type == "s_open" for t in _PAIRS.parseInline(content) for c in t.children or [])


def _list_item(walk: _Walk, token: Token) -> None:
    """Find the numeral of an item of an actual numbered list. It is claimed as structure only
    when the item closes with content (`_list_item_close`); any other numeral is left unclaimed,
    so it becomes an opaque block (it may BE the answer: "3.")."""
    current = walk.lists[-1] if walk.lists else None
    span: tuple[int, int] | None = None
    if current is not None and current.ordered:
        current.items += 1
        if current.numbered and token.map is not None:
            line = token.map[0]
            start, end = walk.doc.line(line)
            marker = f"{token.info}{token.markup}"
            at = walk.doc.src.find(marker, walk.markers.get(line, start), end)
            if at != -1:
                span = (at, at + len(marker))
                walk.markers[line] = span[1]
    walk.items.append((span, len(walk.parts)))


def _list_item_close(walk: _Walk) -> None:
    """Claim the item's numeral only if the item SAYS something besides citation labels.

    WHY (blind review round 3, P2): "1. A [S1].\n\nHere is what I found:\n\n2." showed a lone
    "2." after an allowlisted lead-in, and "1)\n2)\n3)" rendered a count, both `verified`: a
    numeral with nothing after it is exactly the "a numeral may BE the answer" shape. Its
    numeral is left unclaimed, so `_gaps` makes it residue."""
    span, first_part = walk.items.pop()
    if span is not None and any(_says(p) for p in walk.parts[first_part:]):
        walk.claimed.append(span)


def _says(part: Part) -> bool:
    """`part` holds something besides citation labels (any block, code, a URL or an image
    counts; an opaque part says something by definition)."""
    if isinstance(part, Block):
        return True
    shown = "".join(a.visible for a in part.atoms if a.kind is not Kind.mark)
    return says_something(CITATION_MARKER.sub(" ", shown)) or any(a.opaque for a in part.atoms)


def _table_row(walk: _Walk, token: Token, opening: bool) -> None:
    if opening:
        lines = token.map or [0, 1]
        walk.row, walk.row_lines = id(token), (lines[0], lines[1] - 1)
        walk.row_cursor = walk.doc.line(lines[0])[0]
        walk.cells, walk.row_failed = [], False
        return
    if walk.row_failed:
        walk.opaque(*walk.row_lines)
    else:
        walk.parts.extend(walk.cells)
        walk.claimed.extend((c.start, c.end) for c in walk.cells)
    walk.row, walk.cells = -1, []


def _block_start(doc: _Doc, token: Token, line_start: int) -> int:
    """Where a code or HTML block starts on its first line: at its fence, or at its first content
    line, after the container markers. WHY (found by the round-4 stress run): the block used to
    own its whole first line, so in "3. ```" the numeral of a list that does not start at 1 rode
    on a verbatim-attributed block. The markers left before it are accounted like any other
    unclaimed text (`_gaps`)."""
    end = doc.line(token.map[0])[1] if token.map else line_start
    line = doc.src[line_start:end]
    if token.type == "fence":
        at = line.find(token.markup)
    else:  # the first content line ends the first source line; its indentation is the block's
        first = token.content.split("\n", 1)[0].strip()
        at = line.rfind(first) if first else 0
        while at > 0 and line[at - 1] in " \t":
            at -= 1
    return line_start + max(at, 0)


def _walk(doc: _Doc) -> _Walk:
    walk = _Walk(doc)
    tokens = doc.tokens
    for i, token in enumerate(tokens):
        kind = token.type
        if kind in _CONTAINERS:
            if kind == "list_item_open":
                _list_item(walk, token)
            walk.stack.append(i)
        elif kind in _CLOSERS:
            if kind == "list_item_close":
                _list_item_close(walk)
            walk.stack.pop()
        elif kind in ("ordered_list_open", "bullet_list_open"):
            ordered = kind == "ordered_list_open"
            start = int(token.attrGet("start") or 1) if ordered else 1
            walk.lists.append(_List(ordered, start, start in (1, walk.counted + 1)))
        elif kind in ("ordered_list_close", "bullet_list_close"):
            done = walk.lists.pop()
            if done.ordered:
                walk.counted = done.start + done.items - 1
        elif kind == "inline":
            _inline(walk, tokens, i)
        elif kind in ("fence", "code_block", "html_block") and token.map is not None:
            first, last = token.map[0], token.map[1] - 1
            start, end = doc.lines(first, last)
            start = _block_start(doc, token, start)
            code = None
            if kind != "html_block":
                code = Code(start, end, token.content, block=True, info=token.info.strip())
            block = Block(
                "html" if code is None else "code", start, end, first, last, walk.parent, code
            )
            walk.parts.append(block)
            walk.claimed.append((start, end))
        elif kind in ("tr_open", "tr_close"):
            _table_row(walk, token, kind == "tr_open")
        elif kind not in _STRUCTURE:
            lines = token.map or [0, len(doc.starts)]
            walk.opaque(lines[0], lines[1] - 1)  # a token this module does not know
    return walk


def _gaps(doc: _Doc, claimed: list[tuple[int, int]]) -> Iterator[tuple[int, int]]:
    """Every unclaimed run of one line that says something (a reference definition, a dropped
    table cell, a numeral outside a numbered list), trimmed to its substantive characters.
    Unclaimed text is what the parser consumed as structure, so "#" (a heading marker, named
    NUMBER SIGN) states nothing here."""
    spans = sorted(claimed)
    cursor = 0
    for start, end in [*spans, (len(doc.src), len(doc.src))]:
        if start > cursor:
            for piece in re.finditer(r"[^\n]+", doc.src[cursor:start]):
                a, b = cursor + piece.start(), cursor + piece.end()
                while a < b and _structural(doc.src[a]):
                    a += 1
                while b > a and _structural(doc.src[b - 1]):
                    b -= 1
                if a < b:
                    yield a, b
        cursor = max(cursor, end)


def _structural(char: str) -> bool:
    return char == "#" or states_nothing(char)


def _line_of(doc: _Doc, pos: int) -> int:
    return bisect_right(doc.starts, pos) - 1


@lru_cache(maxsize=64)
def structure(text: str) -> Structure:
    """THE structural map of `text` (an answer or a source page), cached per text: the check,
    normalisation, attribution and the residue all read the same parse."""
    norm, back = _normalized(text)
    doc = _Doc(norm)
    walk = _walk(doc)
    parts = list(walk.parts)
    for a, b in _gaps(doc, walk.claimed):
        line = _line_of(doc, a)
        parts.append(Block("opaque", a, b, line, line))
    parts.sort(key=lambda p: (p.start, p.end))
    if back is not None:
        parts = [_moved(p, back) for p in parts]
    code: list[Code] = []
    spans: list[tuple[int, int]] = []
    for part in parts:
        if isinstance(part, Block):
            if part.code is not None:
                code.append(part.code)
            continue
        for atom in part.atoms:
            if atom.code is not None:
                code.append(atom.code)
            if atom.kind is Kind.text and atom.eligible:
                if spans and spans[-1][1] == atom.start:
                    spans[-1] = (spans[-1][0], atom.end)
                else:
                    spans.append((atom.start, atom.end))
    return Structure(tuple(parts), tuple(code), tuple(spans))


def _moved(part: Part, back: list[int]) -> Part:
    """`part` in the original text's offsets (markdown-it normalised CR LF / CR / NUL)."""

    def code(c: Code | None) -> Code | None:
        return None if c is None else Code(back[c.start], back[c.end], c.text, c.block, c.info)

    if isinstance(part, Block):
        return Block(
            part.kind,
            back[part.start],
            back[part.end],
            part.first_line,
            part.last_line,
            part.parent,
            code(part.code),
        )
    atoms = tuple(
        Atom(a.kind, back[a.start], back[a.end], a.visible, a.eligible, code(a.code), a.opaque)
        for a in part.atoms
    )
    return Inline(
        back[part.start],
        back[part.end],
        atoms,
        part.kind,
        part.first_line,
        part.last_line,
        part.parent,
        part.row,
    )
