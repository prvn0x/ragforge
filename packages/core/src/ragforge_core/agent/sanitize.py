"""Incremental streaming output sanitizer (ADR-0011); implements the `OutputSanitizer` port.

Model output is untrusted: retrieved documents can carry prompt injections that try to make the
answer *render* into a fetch (a remote image), a link to an attacker's URL (often with secrets in
the query string), raw HTML, or a GFM bare-URL autolink. The client renders tokens while they
stream, so the checks run here, on the server, before any `token` event, and they must hold for
every prefix of the stream, not just the finished answer.

Why the output is safe (the invariants the tests in `tests/module_e1` prove):

1. **Text is inert by construction.** In Text state every `[`, `]`, `<`, `!` and `&` the model
   wrote is backslash-escaped. Without an unescaped `[` or `<`, a CommonMark/GFM renderer cannot
   form a link, image, reference definition, autolink, raw HTML or entity from our text. The only
   links in the output are those this module writes itself, with the *stored canonical* URL of an
   allow-listed hit as the href, never the model's spelling of it.
2. **GFM extended autolinks** (`scheme://`, `www.`, `mailto:`/`xmpp:`, e-mail) are matched on
   the *decoded* text of each whitespace-delimited run, i.e. what the renderer's text node will
   contain after backslash escapes are resolved, so `www\\.evil.com` cannot sneak through. A match
   is emitted either as an allow-listed link or defanged inside a code span (`evil[.]com`) that
   cannot autolink.
   A run is matched together with the *word it joins*: a construct that emits nothing (a dropped
   `<javascript:x>`) or whose text is re-scanned (a dropped link) would otherwise glue two
   harmless-looking runs into one URL (`ww<javascript:x>w.evil.com`). If the URL starts in text
   already emitted, the new run is defanged into a code span, which ends the word.
3. **Verbatim regions are only emitted where every renderer agrees they are code.** Fenced code
   is entered only for an unambiguous top-level fence (column 0, no open container, CommonMark
   opener rules); a fence-looking line we decline is escaped, so the renderer cannot open a
   fence we do not know about and the two views never drift apart. Inline code is emitted
   verbatim only when it closes on the same line and contains no `|` (a GFM table would split
   the span at the pipe and expose its content as live markdown).
4. **Chunking cannot matter.** Every decision depends only on input inside a bounded window
   (`HOLD_LIMIT` chars from the start of the held construct), never on where the stream was
   split. Feeding any split of the same text yields byte-identical concatenated output, so token
   boundaries cannot be used to smuggle anything (property-tested).
5. **Overflow is inert.** A construct still unresolved after `HOLD_LIMIT` chars is flushed as
   escaped text (its opening `[`/`<`/backtick escaped), so the remainder can never complete it.

6. **Scanning is linear.** A `[` is only scanned when a `](` exists in its window, and scans that
   do reach one draw on a credit that grows with the stream position. A bracket flood therefore
   costs bounded CPU per input char instead of `HOLD_LIMIT` steps per `[` (the API event loop is
   shared by every tenant's stream). An exhausted credit makes `[` plain text: fail safe.
   A dropped link rebuilds the working buffer, so `feed()` moves input into it `_FEED_WINDOW`
   chars at a time: each rebuild copies a bounded buffer, never the rest of a huge feed.
   Invariant 4 makes this internal chunking invisible in the output.
   Bare-URL detection is linear too: `BARE_URL_PATTERN` is the *specification*, but searching
   it unanchored backtracks quadratically (its e-mail and scheme alternatives can start at every
   char and run to the end of the word), and the joined word is re-searched on every run.
   `_find_bare_url` returns the same leftmost match start by locating each alternative's fixed
   anchor (`@`, `://`, `//`, `www.`, `mailto:`/`xmpp:`) with C-speed scans and walking back to its
   start once. It also skips the part of the joined word already searched: the tail is known to
   hold no match, so only matches that reach the new run are looked for. A run held across
   feeds resumes its scan instead of restarting it.

Accepted cosmetic costs (fail safe): code fences inside lists/blockquotes render as text; every
word is held until its delimiter (one word of latency); images are always removed (a relative
`src` is still a fetch from the client's origin); links inside link text lose their formatting;
any `//` in a word is defanged (linkify links protocol-relative `//host`); any word with an `@`
followed later by a `.` is defanged (linkify links Unicode/IDN e-mail domains).
"""

from __future__ import annotations

import re
import string
from collections.abc import Iterable, Sequence
from enum import Enum, auto
from typing import Final
from urllib.parse import urlsplit

__all__ = [
    "HOLD_LIMIT",
    "IMAGE_REMOVED",
    "MarkdownStreamSanitizer",
    "normalize_url",
]

HOLD_LIMIT: Final = 512
"""Max chars a construct (link, autolink, inline code, run, fence opener) is held unresolved."""

IMAGE_REMOVED: Final = "\\[image removed\\]"

_RUN_TERMINATORS: Final = frozenset("`[]!<|")
_ASCII_PUNCT: Final = frozenset("!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~")
_TRAILING_PUNCT: Final = frozenset(".,:;!?*_~'\"")
_DROPPED_SCHEMES: Final = frozenset({"javascript", "data", "vbscript"})
_DEFAULT_PORTS: Final = {"http": 80, "https": 443}

# Anything a GFM renderer (or markdown-it's linkify) could turn into a link from plain text.
# Deliberately broader than GFM: any scheme, `www.` anywhere, `mailto:`/`xmpp:` anywhere (linkify
# links `mailto:a@b` with no dot and `mailto:;@x` with any local part), any `//` (linkify's
# protocol-relative schema links `//host`, also after `http&#58;` or a fullwidth colon that no
# scheme rule sees), and any `@` followed later in the word by a `.`. The e-mail alternative is
# Unicode-aware on purpose: linkify links IDN domains (`a@évil.com`, `a@evil.рф`) and local parts
# with `;:&=+$,"`. Its local part may be empty, so it fires on the `@` itself and never depends on
# chars emitted before the joined word (the run-joining guard in `_emit_run` relies on that; the
# other alternatives use only [A-Za-z0-9+.\-/:], which no emission ends in).
BARE_URL_PATTERN: Final = re.compile(
    r"[A-Za-z][A-Za-z0-9+.\-]*://"
    r"|[Ww][Ww][Ww]\."
    r"|(?i:mailto|xmpp):"
    r"|[^\s@]*@[^\s@]*\."
    r"|//"
)
# Fixed-length anchors of BARE_URL_PATTERN, plus what `_find_bare_url` derives from them. They are
# searched with C-speed scans that do constant work per char, never with the quadratic pattern.
_FIXED_ANCHORS: Final = re.compile(r"[Ww][Ww][Ww]\.|(?i:mailto|xmpp):|//")
_FIXED_ANCHOR_MAX: Final = len("mailto:")
_EMAIL_ANCHOR: Final = re.compile(r"@[^\s@]*\.")  # starts only at `@`: disjoint segments
_EMAIL_SEGMENT_END: Final = re.compile(r"[\s@]")
_SCHEME_LETTERS: Final = frozenset(string.ascii_letters)
_SCHEME_CHARS: Final = _SCHEME_LETTERS | frozenset(string.digits + "+.-")
# The part of an output line that can only be container markers (indent, `>`, list bullets).
_LINE_PREFIX: Final = re.compile(r"(?:[ \t]|>|[-+*](?=[ \t])|\d{1,9}[.)](?=[ \t]))*")
# Lines that open (or may continue) a container; `|` covers GFM tables, which in markdown-it
# swallow a following column-0 fence line as a table row.
_CONTAINER_LINE: Final = re.compile(r"[ \t]*(?:>|[-+*](?:[ \t]|$)|\d{1,9}[.)](?:[ \t]|$))")
_FENCE_OPEN: Final = re.compile(r"(`{3,}|~{3,})(.*)", re.DOTALL)
_AUTOLINK_URI: Final = re.compile(r"[A-Za-z][A-Za-z0-9+.\-]{1,31}:[^\s<>\x00-\x1f\x7f]*")
_AUTOLINK_EMAIL: Final = re.compile(
    r"[A-Za-z0-9.!#$%&'*+/=?^_`{|}~\-]+@[A-Za-z0-9](?:[A-Za-z0-9\-]{0,61}[A-Za-z0-9])?"
    r"(?:\.[A-Za-z0-9](?:[A-Za-z0-9\-]{0,61}[A-Za-z0-9])?)*"
)
_INLINE_IMAGE: Final = re.compile(r"!\[[^\]\n]*\]\([^)\n]*\)")
_ESCAPABLE: Final = re.compile(r"([!-/:-@\[-`{-~])")
_ESCAPE_PAIR: Final = re.compile(r"\\([!-/:-@\[-`{-~])")
_DESTINATION_ESCAPABLE: Final = re.compile(r"([\\<>|])")
_SAFE_FRAGMENT: Final = re.compile(r"[A-Za-z0-9_\-]{1,128}")
_LINE_KEEP: Final = 1024  # output-line chars kept for line-start decisions (bounds the work)
_TAIL_KEEP: Final = HOLD_LIMIT  # decoded chars of the current joined word kept (see _emit_run)
# Link-scan credit: scans that reach a `](` may spend BASE + PER_CHAR * stream position steps in
# total. Normal prose spends a few steps per link; a `[[[...](x)` flood exhausts it and its `[`
# become plain text. Only completed scans are charged, so the credit (and every decision) is a
# function of the input alone, never of how it was chunked.
_SCAN_CREDIT_BASE: Final = 64 * HOLD_LIMIT
_SCAN_CREDIT_PER_CHAR: Final = 8
# Input moved into the working buffer per drain (see invariant 6); > HOLD_LIMIT so constructs fit.
_FEED_WINDOW: Final = 4 * HOLD_LIMIT
_PLACEHOLDER: Final = "\x00"  # input NULs are replaced with U+FFFD, so this cannot collide


class _NeedMore(Exception):
    """The held construct cannot be resolved until more input arrives."""


class _Overflow(Exception):
    """The held construct passed HOLD_LIMIT without resolving: flush it as inert text."""


class _State(Enum):
    TEXT = auto()
    FENCE = auto()
    DROP_LINE = auto()  # inside a removed reference definition, until end of line


# --------------------------------------------------------------------------- pure helpers
def normalize_url(url: str) -> tuple[str, str] | None:
    """Return `(match_key, fragment)` for an http(s) URL, or None if it can never be linked.

    The key is scheme + lower-cased host (+ non-default port) + path (+ query), with credentials
    stripped. The query is part of the key on purpose: ADR-0011 names scheme/host/path, and
    matching the query too only makes the match stricter (we emit the stored URL anyway).
    """
    try:
        parts = urlsplit(url)
        port = parts.port
    except ValueError:
        return None
    scheme = parts.scheme.lower()
    host = parts.hostname
    if scheme not in _DEFAULT_PORTS or not host:
        return None
    if ":" in host:
        host = f"[{host}]"
    netloc = host if port in (None, _DEFAULT_PORTS[scheme]) else f"{host}:{port}"
    key = f"{scheme}://{netloc}{parts.path or '/'}"
    if parts.query:
        key = f"{key}?{parts.query}"
    return key, parts.fragment


def _escape(text: str) -> str:
    """Backslash-escape every ASCII punctuation char: renders identically, parses as nothing."""
    return _ESCAPABLE.sub(r"\\\1", text)


def _inert_label(markdown: str) -> str:
    """Link text we re-emit: images removed, model escapes resolved, then everything escaped."""
    text = _INLINE_IMAGE.sub(_PLACEHOLDER, markdown)
    text = _ESCAPE_PAIR.sub(r"\1", text)
    return _escape(text).replace(_PLACEHOLDER, IMAGE_REMOVED)


def _unescape(text: str) -> str:
    return _ESCAPE_PAIR.sub(r"\1", text)


def _find_bare_url(text: str, searched: int = 0) -> int:
    """Start of the leftmost `BARE_URL_PATTERN` match in `text`, or -1; linear time.

    `text[:searched]` must hold no match on its own (it is the tail of a joined word that was
    already searched), so only matches reaching past it are looked for: the cost is the new
    text plus a bounded look-back, never a re-search of the whole word.

    Equivalence with the pattern, per alternative (property-tested against it):
    * `www.`, `mailto:`/`xmpp:`, `//`: fixed-length, so a C-speed search finds the first one.
      One wholly inside `text[:searched]` would be a match there, so the search starts just
      far enough back to catch one straddling the boundary.
    * `scheme://`: its leftmost start is the leftmost letter of the run of scheme chars ending
      at the *first* `://`. A later `://` cannot start earlier (the first one's `:` ends every
      later run), and a `://` with no letter before it still matches `//` one char later.
    * e-mail: the local part may be empty, so a match exists iff some `@` has a `.` before the
      next `@`/whitespace, and its leftmost start is just after the previous `@`/whitespace.
      Only the first such `@` matters. Inside `text[:searched]` only the last `@` can qualify
      (its segment is the only one that reaches the new text).
    A `://` or a qualifying `@` always means a match, and its start may precede the other
    anchors, so both are resolved independently. The Python walk-backs run only when a match
    exists, and a match ends the joined word, so each char is walked back over at most once.
    """
    best = len(text)
    fixed = _FIXED_ANCHORS.search(text, max(0, searched - _FIXED_ANCHOR_MAX + 1))
    if fixed is not None:
        best = fixed.start()
    sep = text.find("://", max(0, searched - 2))
    if sep != -1:  # its start may precede `best` even when the `://` does not
        index, start = sep, -1
        while index and text[index - 1] in _SCHEME_CHARS:
            index -= 1
            if text[index] in _SCHEME_LETTERS:
                start = index
        if start != -1:
            best = min(best, start)
    at, segment_end = text.rfind("@", 0, searched), 0
    if at != -1:  # the segment after it ends at the next `@` or whitespace
        stop = _EMAIL_SEGMENT_END.search(text, at + 1)
        segment_end = len(text) if stop is None else stop.start()
    if at == -1 or text.find(".", searched, segment_end) == -1:
        email = _EMAIL_ANCHOR.search(text, max(searched, at + 1))
        at = email.start() if email is not None else -1
    if at != -1:  # likewise: the local part may reach back before `best`
        index = at
        while index and not (text[index - 1] == "@" or text[index - 1].isspace()):
            index -= 1
        best = min(best, index)
    return best if best < len(text) else -1


def _split_trailing(region: str) -> tuple[str, str]:
    """Split GFM-style trailing punctuation (and unbalanced `)`) off a bare URL."""
    end = len(region)
    opens, closes = region.count("("), region.count(")")  # counts in region[:end]
    while end:
        ch = region[end - 1]
        if ch in _TRAILING_PUNCT:
            end -= 1
        elif ch == ")" and opens < closes:
            end, closes = end - 1, closes - 1
        else:
            break
    if _find_bare_url(region[:end]) != 0:
        return region, ""  # trimming broke the match (e.g. `www.`): defang all of it
    return region[:end], region[end:]


def _is_fence_closer(line: str, char: str, length: int) -> bool:
    pattern = rf" {{0,3}}{re.escape(char)}{{{length},}}[ \t]*"
    return re.fullmatch(pattern, line) is not None


# --------------------------------------------------------------------------- the machine
class MarkdownStreamSanitizer:
    """One instance per streamed answer. Not thread-safe (one stream = one task)."""

    def __init__(self, allowed_urls: Iterable[str] = ()) -> None:
        self._allowed: dict[str, str] = {}
        self._buf = ""  # unprocessed input (CR-normalized)
        self._pos = 0
        self._final = False
        self._finished = False
        self._pending_cr = False
        self._out: list[str] = []
        self._state = _State.TEXT
        self._fence_char = ""
        self._fence_len = 0
        self._fence_line = ""
        self._fence_line_may_close = True
        # The OUTPUT of the current line (the renderer sees output, not input), capped at
        # _LINE_KEEP chars; the facts needed beyond the cap are kept as flags.
        self._out_line = ""
        self._line_truncated = False
        self._line_pipe = False
        self._line_blank = True
        self._last_out = ""
        self._container_open = False
        self._prev_blank = True
        self._run_overflow = False
        # Decoded text of the word the next run would join: only runs separated by nothing in
        # the output (a dropped or rewritten construct) accumulate here; any emission resets it.
        self._tail = ""
        self._tail_cut = False
        # A run held for more input: (stream position of its start, scan index relative to it,
        # raw items, decoded items), so the next feed resumes the scan instead of restarting it.
        self._held_run: tuple[int, int, list[str], list[str]] | None = None
        # Link-scan accounting: `_offset` is the stream position of `_buf[0]`.
        self._offset = 0
        self._scan_spent = 0
        self._reach = 0
        self.allow(tuple(allowed_urls))

    # ------------------------------------------------------------------ port
    def allow(self, urls: Sequence[str]) -> None:
        """Add canonical URLs. Only http(s) URLs without whitespace/control chars are accepted,
        so a poisoned `external_url` can never make us emit a `javascript:` href."""
        for url in urls:
            if any(ord(ch) <= 0x20 or ord(ch) == 0x7F for ch in url):
                continue
            normalized = normalize_url(url)
            if normalized is not None:
                self._allowed.setdefault(normalized[0], url)

    def feed(self, text: str) -> str:
        if self._finished:
            raise RuntimeError("feed() after finish(): use one sanitizer per answer")
        text = self._normalize(text)
        if len(text) <= _FEED_WINDOW:
            self._buf += text
            return self._drain()
        out: list[str] = []
        for index in range(0, len(text), _FEED_WINDOW):
            self._buf += text[index : index + _FEED_WINDOW]
            out.append(self._drain())
        return "".join(out)

    def finish(self) -> str:
        if self._finished:
            return ""
        self._finished = True
        self._final = True
        if self._pending_cr:
            self._pending_cr = False
            self._buf += "\n"
        return self._drain()

    # ------------------------------------------------------------------ plumbing
    def _normalize(self, text: str) -> str:
        """CommonMark treats CR and CRLF as line endings; normalize so our line view matches.
        A trailing CR is held because the next chunk may start with its LF."""
        if self._pending_cr:
            text = "\r" + text
            self._pending_cr = False
        if text.endswith("\r"):
            self._pending_cr = True
            text = text[:-1]
        return text.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "\ufffd")

    def _drain(self) -> str:
        while self._pos < len(self._buf) and self._step():
            pass
        self._offset += self._pos
        self._buf = self._buf[self._pos :]
        self._pos = 0
        out = "".join(self._out)
        self._out = []
        return out

    def _emit(self, text: str) -> None:
        """Every emission ends the joined word: each one is whitespace, a code span, a link, or
        ends in a char (`[`, `]`, `<`, `!`, `|`, a backtick) that no URL pattern contains.
        `_emit_run` sets the tail again after emitting a run."""
        if text:
            self._out.append(text)
            self._last_out = text[-1]
            self._tail, self._tail_cut = "", False
            if self._state is _State.TEXT:
                self._track_line(text)

    def _track_line(self, text: str) -> None:
        room = _LINE_KEEP - len(self._out_line)
        self._out_line += text[:room]
        self._line_truncated = self._line_truncated or len(text) > room
        self._line_pipe = self._line_pipe or "|" in text
        self._line_blank = self._line_blank and not text.strip(" \t")

    def _reset_line(self) -> None:
        self._out_line, self._line_truncated = "", False
        self._line_pipe, self._line_blank = False, True

    def _at_line_prefix(self) -> bool:
        """True while the output line holds only container markers (indent, `>`, bullets)."""
        return not self._line_truncated and _LINE_PREFIX.fullmatch(self._out_line) is not None

    def _rewrite(self, replacement: str, end: int) -> None:
        """Replace buf[pos:end] with `replacement` and re-scan it as ordinary text. Used when a
        link is dropped but its text kept; always strictly shrinks the buffer (terminates). The
        copy is bounded by the held construct plus `_FEED_WINDOW`, so floods stay linear.
        Nothing is emitted, so the re-scanned text joins the current word (see `_emit_run`)."""
        self._offset += end - len(replacement)
        self._buf = replacement + self._buf[end:]
        self._pos = 0
        self._held_run = None  # stream positions of the rewritten text are reused

    def _at(self, index: int, start: int) -> str:
        """Char at `index` for a construct starting at `start`; '' at end of a finished stream."""
        if index - start >= HOLD_LIMIT:
            raise _Overflow
        if index < len(self._buf):
            if index > self._reach:
                self._reach = index
            return self._buf[index]
        if self._final:
            return ""
        raise _NeedMore

    def _skip_blanks(self, index: int, start: int) -> int:
        while self._at(index, start) in (" ", "\t"):
            index += 1
        return index

    # ------------------------------------------------------------------ dispatch
    def _step(self) -> bool:
        """Process one unit. False means: hold the rest until more input arrives."""
        if self._state is _State.FENCE:
            self._fence_step()
            return True
        if self._state is _State.DROP_LINE:
            self._drop_step()
            return True
        continuing = self._run_overflow
        self._run_overflow = False
        try:
            self._text_step(continuing)
        except _NeedMore:
            self._run_overflow = continuing
            return False
        return True

    def _text_step(self, continuing: bool) -> None:
        char = self._buf[self._pos]
        if char == "\n":
            self._end_line()
            self._emit_newline()
            self._pos += 1
        elif (
            char in "`~"
            and not self._out_line
            and not self._container_open
            and self._try_open_fence()
        ):
            return
        elif char == "`":
            self._inline_code()
        elif char == "[":
            if not (self._at_line_prefix() and self._try_definition()):
                self._bracket()
        elif char == "!":
            self._bang()
        elif char == "<":
            self._angle()
        elif char in "]|":
            self._emit("\\]" if char == "]" else "|")
            self._pos += 1
        elif char == "\\" and self._at(self._pos + 1, self._pos) in ("\n", "|"):
            # Hard line break / escaped table pipe: keep the model's meaning, end the run here.
            pair = self._buf[self._pos : self._pos + 2]
            self._emit("\\" if pair == "\\\n" else pair)
            self._pos += 1 if pair == "\\\n" else 2
        elif char.isspace():
            self._emit(char)
            self._pos += 1
        else:
            self._run(continuing)

    def _emit_newline(self) -> None:
        """A newline is not tracked as line content (the line just ended), but ends the word."""
        self._out.append("\n")
        self._last_out = "\n"
        self._tail, self._tail_cut = "", False

    def _end_line(self) -> None:
        """Track open containers from the OUTPUT line (conservatively: a container stays open
        until a blank line is followed by an unindented non-container line)."""
        line, blank = self._out_line, self._line_blank
        if _CONTAINER_LINE.match(line) or self._line_pipe:
            self._container_open = True
        elif self._container_open and self._prev_blank and not blank and line[0] not in " \t":
            self._container_open = False
        self._prev_blank = blank
        self._reset_line()

    # ------------------------------------------------------------------ fenced code
    def _try_open_fence(self) -> bool:
        """At column 0 with no open container: open a fence for a valid CommonMark opener.
        The whole opener line is held (bounded) because the info-string rule needs all of it."""
        buf, start = self._buf, self._pos
        char = buf[start]
        if self._at(start + 1, start) != char or self._at(start + 2, start) != char:
            return False
        newline = buf.find("\n", start, start + HOLD_LIMIT)
        if newline == -1:
            if len(buf) - start >= HOLD_LIMIT:
                return False
            if not self._final:
                raise _NeedMore
            newline = len(buf)
        match = _FENCE_OPEN.fullmatch(buf[start:newline])
        if match is None or (char == "`" and "`" in match.group(2)):
            return False
        self._emit(match.group(0))
        self._pos = newline
        self._state = _State.FENCE
        self._fence_char, self._fence_len = char, len(match.group(1))
        self._fence_line, self._fence_line_may_close = "", True
        return True

    def _fence_step(self) -> None:
        """Pass code through unheld; only track whether the current line closes the fence."""
        buf, start = self._buf, self._pos
        newline = buf.find("\n", start)
        end = len(buf) if newline == -1 else newline
        chunk = buf[start:end]
        if self._fence_line_may_close:
            if chunk.strip(" \t" + self._fence_char):
                self._fence_line_may_close, self._fence_line = False, ""
            else:
                self._fence_line += chunk
        self._emit(chunk)
        if newline == -1:
            self._pos = end
            return
        self._emit("\n")
        self._pos = newline + 1
        closes = self._fence_line_may_close and _is_fence_closer(
            self._fence_line, self._fence_char, self._fence_len
        )
        self._fence_line, self._fence_line_may_close = "", True
        if closes:
            self._state = _State.TEXT
            self._reset_line()
            self._container_open = False
            self._prev_blank = False

    # ------------------------------------------------------------------ reference definitions
    def _try_definition(self) -> bool:
        """`[label]: dest ["title"]` at line start (after container markers) is removed."""
        start = self._pos
        try:
            index = start + 1
            while (char := self._at(index, start)) != "]":
                if char in ("", "\n", "["):
                    return False
                if char == "\\":
                    if self._at(index + 1, start) in ("", "\n"):
                        return False
                    index += 1
                index += 1
            if index == start + 1 or self._at(index + 1, start) != ":":
                return False
            if not self._definition_tail_ok(index + 2, start):
                return False
        except _Overflow:
            return False
        self._state = _State.DROP_LINE
        return True

    def _definition_tail_ok(self, index: int, start: int) -> bool:
        index = self._skip_blanks(index, start)
        char = self._at(index, start)
        if char in ("", "\n"):
            return True  # destination on the next line (that line is then sanitized as text)
        if char == "<":
            index += 1
            while (char := self._at(index, start)) not in ("", "\n", ">"):
                index += 2 if char == "\\" else 1
            if char != ">":
                return False
            index += 1
        else:
            while (char := self._at(index, start)) and not char.isspace():
                index += 1
        index = self._skip_blanks(index, start)
        return self._at(index, start) in ("", "\n", '"', "'", "(")

    def _drop_step(self) -> None:
        newline = self._buf.find("\n", self._pos)
        if newline == -1:
            self._pos = len(self._buf)
            return
        self._pos = newline
        self._state = _State.TEXT

    # ------------------------------------------------------------------ links and images
    def _scan_inline_link(self, start: int) -> tuple[str, str, int] | None:
        """Parse `[text](dest "title")` at `start`; None if it is not an inline link."""
        index, depth = start + 1, 0
        while True:
            char = self._at(index, start)
            if char in ("", "\n"):
                return None
            if char == "\\":
                if self._at(index + 1, start) in ("", "\n"):
                    return None
                index += 2
                continue
            if char == "]" and depth == 0:
                break
            depth += {"[": 1, "]": -1}.get(char, 0)
            index += 1
        text = self._buf[start + 1 : index]
        if self._at(index + 1, start) != "(":
            return None
        index = self._skip_blanks(index + 2, start)
        parsed = self._scan_destination(index, start)
        if parsed is None:
            return None
        dest, index = parsed
        index = self._skip_blanks(index, start)
        if (opener := self._at(index, start)) in ('"', "'", "("):
            closer = ")" if opener == "(" else opener
            index += 1
            while (char := self._at(index, start)) != closer:
                if char in ("", "\n"):
                    return None
                index += 2 if char == "\\" else 1
            index = self._skip_blanks(index + 1, start)
        if self._at(index, start) != ")":
            return None
        return text, _unescape(dest), index + 1

    def _scan_destination(self, index: int, start: int) -> tuple[str, int] | None:
        if self._at(index, start) == "<":
            end = index + 1
            while (char := self._at(end, start)) != ">":
                if char in ("", "\n", "<"):
                    return None
                end += 2 if char == "\\" else 1
            return self._buf[index + 1 : end], end + 1
        end, parens = index, 0
        while True:
            char = self._at(end, start)
            if char == "" or char.isspace() or ord(char) < 0x20:
                break
            if char == "\\":
                end += 2
                continue
            if char == ")":
                if parens == 0:
                    break
                parens -= 1
            elif char == "(":
                parens += 1
            end += 1
        return self._buf[index:end], end

    def _scan_link(self, start: int) -> tuple[str, str, int] | None:
        """`_scan_inline_link` behind two guards that keep bracket floods linear.

        1. C-speed precheck: an inline link needs a `](` before the line ends inside the window.
           Only a *decided* window short-circuits; otherwise the full scan runs (or waits).
        2. Credit: completed scans whose range contains a `](` are charged their length. A
           decided precheck implies no `](` in range, so it would never have been charged
           either: the spent total is the same however the input was chunked.
        `_NeedMore` propagates uncharged; the scan is repeated identically on more input.
        """
        if self._link_impossible(start):
            return None
        budget = _SCAN_CREDIT_BASE + _SCAN_CREDIT_PER_CHAR * (self._offset + start)
        if self._scan_spent > budget:
            return None
        self._reach = start
        try:
            link = self._scan_inline_link(start)
        except _Overflow:
            link = None
        if self._buf.find("](", start, self._reach + 1) != -1:
            self._scan_spent += self._reach - start
        return link

    def _link_impossible(self, start: int) -> bool:
        buf = self._buf
        limit = min(len(buf), start + HOLD_LIMIT)
        newline = buf.find("\n", start, limit)
        stop = limit if newline == -1 else newline
        if buf.find("](", start, stop) != -1:
            return False
        return newline != -1 or limit == start + HOLD_LIMIT or self._final

    def _bracket(self) -> None:
        link = self._scan_link(self._pos)
        if link is None:  # not an inline link (or reference-style / overflow): plain text
            self._emit("\\[")
            self._pos += 1
            return
        text, dest, end = link
        href = self._allowed_href(dest)
        if href is not None:
            self._emit_link(_inert_label(text), href)
            self._pos = end
        else:  # drop the link (incl. javascript:/data:), keep its text
            self._rewrite(text, end)

    def _bang(self) -> None:
        start = self._pos
        link: tuple[str, str, int] | None = None
        if self._at(start + 1, start) == "[":
            link = self._scan_link(start + 1)
        if link is None:  # `!` alone, or a reference-style image: plain text
            self._emit("\\!")
            self._pos += 1
            return
        self._emit(IMAGE_REMOVED)
        self._pos = link[2]

    def _angle(self) -> None:
        """Autolinks follow the link rule; anything else starting with `<` is escaped HTML."""
        start = index = self._pos
        try:
            while (char := self._at(index + 1, start)) != ">":
                if not char or char == "<" or char.isspace() or ord(char) < 0x20:
                    raise _Overflow
                index += 1
        except _Overflow:
            self._emit("\\<")
            self._pos += 1
            return
        content, end = self._buf[start + 1 : index + 1], index + 2
        if _AUTOLINK_URI.fullmatch(content):
            if content.split(":", 1)[0].lower() in _DROPPED_SCHEMES:
                self._pos = end
                return
            href = self._allowed_href(content)
            if href is not None:
                self._emit_link(_escape(href), href)
                self._pos = end
                return
            self._rewrite(content, end)  # re-scanned as a bare URL: defanged
        elif _AUTOLINK_EMAIL.fullmatch(content):
            self._rewrite(content, end)
        else:
            self._emit("\\<")
            self._pos += 1

    def _allowed_href(self, candidate: str) -> str | None:
        """The stored canonical URL for an exact normalized match, else None. A safe anchor
        fragment may be carried over: fragments never reach a server."""
        if candidate[:4].lower() == "www.":
            candidate = "http://" + candidate
        normalized = normalize_url(candidate.strip())
        if normalized is None:
            return None
        key, fragment = normalized
        canonical = self._allowed.get(key)
        if canonical is None:
            return None
        if fragment and "#" not in canonical and _SAFE_FRAGMENT.fullmatch(fragment):
            return f"{canonical}#{fragment}"
        return canonical

    def _emit_link(self, label: str, href: str) -> None:
        self._emit(f"[{label}](<{_DESTINATION_ESCAPABLE.sub(r'\\\1', href)}>)")

    # ------------------------------------------------------------------ code spans
    def _inline_code(self) -> None:
        """Held to the matching backtick run on the same line; verbatim if closed, else the
        opening run is escaped and the rest is re-scanned as text."""
        start = index = self._pos
        close = -1
        try:
            while self._at(index, start) == "`":
                index += 1
            run = index - start
            scan = index
            while (char := self._at(scan, start)) not in ("", "\n"):
                if char != "`":
                    scan += 1
                    continue
                run_end = scan
                while self._at(run_end, start) == "`":
                    run_end += 1
                if run_end - scan == run:
                    close = run_end
                    break
                scan = run_end
        except _Overflow:
            close = -1
        if close == -1 or "|" in self._buf[index : close - (index - start)]:
            self._emit("\\`" * (index - start))
            self._pos = index
            return
        self._emit_code_span_raw(self._buf[start:close])
        self._pos = close

    def _emit_code_span_raw(self, span: str) -> None:
        """Two code spans must never touch: `a``b` would re-pair the backticks."""
        self._emit((" " if self._last_out == "`" else "") + span)

    def _emit_defanged(self, text: str) -> None:
        """Inert code span: backticks stripped, dots bracketed, delimiter longer than any run.
        A run never holds an unescaped backtick, so stripping loses at most an escaped one."""
        content = text.replace("`", "").replace(".", "[.]")
        if content:
            self._emit_code_span_raw(f"`{content}`")
        elif text:
            self._emit("\\`" * len(text))

    # ------------------------------------------------------------------ word runs / bare URLs
    def _run(self, continuing: bool) -> None:
        """A whitespace/special-delimited word, held whole so a bare URL is seen in one piece.
        Items are raw chars or backslash pairs; `decoded` is what the renderer's text node holds.
        """
        buf, start = self._buf, self._pos
        index, overflow = start, False
        raw: list[str] = []
        decoded: list[str] = []
        held, self._held_run = self._held_run, None
        if held is not None and held[0] == self._offset + start:
            index, raw, decoded = start + held[1], held[2], held[3]
        while True:
            if index - start >= HOLD_LIMIT:
                overflow = True
                break
            # Waiting for the next char (or a backslash's partner): keep the scan to resume.
            if not self._final and (
                index >= len(buf)
                or (buf[index] == "\\" and index + 1 >= len(buf) and index + 1 - start < HOLD_LIMIT)
            ):
                self._held_run = (self._offset + start, index - start, raw, decoded)
                raise _NeedMore
            if index >= len(buf):
                break
            char = buf[index]
            if char == "\\":
                following = self._at(index + 1, start) if index + 1 - start < HOLD_LIMIT else ""
                if following in ("\n", "|"):
                    break
                if following and following in _ASCII_PUNCT:
                    raw.append(char + following)
                    decoded.append(following)
                    index += 2
                else:  # a lone backslash is made literal so it can never escape OUR next char
                    raw.append("\\\\")
                    decoded.append("\\")
                    index += 1
                continue
            if char.isspace() or char in _RUN_TERMINATORS:
                break
            raw.append("\\&" if char == "&" else char)  # no entity can decode into syntax
            decoded.append(char)
            index += 1
        self._pos = index
        if continuing or overflow:
            self._emit_defanged("".join(decoded))  # overflow: inert, can never complete later
            self._run_overflow = overflow
            return
        self._emit_run("".join(raw), "".join(decoded))

    def _emit_run(self, raw: str, decoded: str) -> None:
        """Emit one run, matching bare URLs against the joined word `tail + decoded`.

        The renderer sees the joined word, not our runs: `ww` + (dropped construct) + `w.evil`
        renders as `www.evil`. Emitted text cannot be retracted, so when a URL starts in the
        tail the whole run is defanged; the code span ends the word and the tail stays a
        harmless fragment (`ww`, `https`, `me`). Matches never start before the tail: every
        other emission ends in a char no URL pattern contains (see `_emit`).
        """
        tail, tail_cut = self._tail, self._tail_cut
        if tail_cut:
            self._emit_defanged(decoded)  # word too long to track: fail safe
            return
        found = _find_bare_url(tail + decoded, len(tail))  # the tail alone holds no match
        if found != -1 and found < len(tail):
            self._emit_defanged(decoded)
            return
        if found == -1:
            fence_like = decoded.startswith("~~~") and raw[0] == "~"
            if fence_like and (self._line_truncated or self._at_line_prefix()):
                raw = "\\" + raw  # a fence-looking line we did not open must not open one
            self._emit(raw)
            self._set_tail(tail + decoded)
            return
        start = found - len(tail)
        url, suffix = _split_trailing(decoded[start:])
        self._emit(_escape(decoded[:start]))
        href = self._allowed_href(url) if "@" not in url or "://" in url else None
        if href is not None:
            self._emit_link(_escape(href), href)
        else:
            self._emit_defanged(url)
        self._emit(_escape(suffix))
        self._set_tail(suffix)

    def _set_tail(self, word: str) -> None:
        if len(word) > _TAIL_KEEP:
            self._tail, self._tail_cut = "", True
        else:
            self._tail = word


_URL_SCHEME: Final = re.compile(r"(?i)\b([a-z][a-z0-9+.-]*):(//)")
# Self-sufficient (review r6): these renderers clean their own input instead of trusting the
# producer. Control characters (except "\n"/"\t" in code), bidi controls, other format
# characters, and the line/paragraph separators.
_UNSAFE: Final = re.compile(
    "[\x00-\x08\x0b-\x1f\x7f-\x9f\u061c\u200b-\u200f\u202a-\u202e\u2060-\u206f\u2028\u2029\ufeff]"
)
_LANGUAGE: Final = re.compile(r"[A-Za-z0-9_+.#-]{0,20}")
_WWW: Final = re.compile(r"(?i)\bwww\.")


def literal(text: str) -> str:
    """`text` as LITERAL Markdown: every ASCII punctuation character backslash-escaped (no
    emphasis, heading, list, link, HTML or code can form; identifiers like `max_poll_ms` render
    exactly), line breaks collapsed to spaces, and URLs defanged first ("https[:]//",
    "www[.]") so a renderer's or terminal's link detection finds nothing. For general-knowledge
    extras (ADR-0016), which are never formatted."""
    flat = " ".join(_UNSAFE.sub("", text).split())
    return _escape(_WWW.sub("www[.]", _URL_SCHEME.sub(r"\1[:]\2", flat)))


def fenced(code: str, language: str) -> str:
    """`code` as one fenced block that no backtick run inside it can close: the fence is one
    backtick longer than the longest run in the code (CommonMark), and the language tag is the
    caller's checked `[A-Za-z0-9_+.#-]` string."""
    code = _UNSAFE.sub("", code.replace("\r", ""))
    language = language if _LANGUAGE.fullmatch(language) else ""
    longest = max((len(m.group(0)) for m in re.finditer(r"`+", code)), default=0)
    fence = "`" * max(3, longest + 1)
    return f"{fence}{language}\n{code}\n{fence}"
