"""Pure presentation helpers: what to show, as strings. No Streamlit calls here.

Trust model for text (ADR-0011, ADR-0016):
- Answer text, claims and extras are CommonMark the server already sanitized: no raw HTML, no
  link it didn't write itself. They are rendered as Markdown, never with `unsafe_allow_html`.
- Streamlit's Markdown adds non-CommonMark syntax the server's sanitizer doesn't know about:
  `$...$` (LaTeX) and `:...:` (emoji, `:material/x:` icons, `:red[...]` colours). `for_streamlit`
  escapes those two characters outside code, which CommonMark renders back as the same literal
  character (a backslash escape of ASCII punctuation is always valid, also in link targets).
  The escape stops LaTeX and `:red[...]` (tokenizer syntax) but NOT emoji or icons: checked in
  the frontend bundled with streamlit 1.66, `remark-emoji` and the icon plugin match text nodes
  after escapes are resolved, and `:material/` is rewritten in the raw source before parsing.
  So a shortcode-shaped `:name:` also gets an invisible WORD JOINER (U+2060) after its first
  colon, which no shortcode pattern accepts. It is only inserted there, never in plain `a: b`.
- Everything else from the server that the UI places in Markdown itself (titles, headings, tool
  names, problem text) goes through `md_literal`: every ASCII punctuation escaped, so it can
  never form a link, emphasis, heading or list.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from urllib.parse import quote
from uuid import UUID

from ragforge_client.ask import (
    NOTHING_TO_CHECK,
    clean,
    nothing_to_check,
    plain_line,
    style,
    unescape_outside_fences,
)
from ragforge_client.interview import (
    Summary,
    level_name,
    question_line,
    summary_headline,
)
from ragforge_contracts.api import (
    Citation,
    Collection,
    InterviewKind,
    InterviewLevel,
    InterviewQuestion,
)
from ragforge_ui.state import Extra, PendingCall, document_id_arg

_ASCII_PUNCT = frozenset("!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~")
_STREAMLIT_SPECIAL = frozenset("$:")

# A fenced block (``` or ~~~, 3+) closed by a run of the SAME character at least as long,
# indented 0-3 spaces (CommonMark; integrated review P2: a longer closer was read as "unclosed",
# leaving model text after it unescaped), or left open to the end of the text.
_FENCE = re.compile(
    r"^(?P<fence>(?P<char>[`~])(?P=char){2,})(?P<info>[^\n]*)\n(?P<body>.*?)"
    r"(?:^ {0,3}(?P=fence)(?P=char)*[ \t]*$|\Z)",
    re.DOTALL | re.MULTILINE,
)
# A section starts only where its name is a bold label: `**Technically:**` or `**Technically**:`.
# A bold "Technically" inside a sentence is emphasis, not a section, and stays in place.
_SECTION = re.compile(r"\s*(\*\*(?:In simple terms|Technically|In production)(?::\*\*|\*\*:))")
# What follows the first colon of an emoji (`:smile:`, `:+1:`) or icon (`:material/home:`)
# shortcode, as remark-emoji and Streamlit's icon plugin match them.
_SHORTCODE_TAIL = re.compile(r"(?:material/\w+|[\w+-]+)\\?:")
WORD_JOINER = "\u2060"
_LANGUAGE = re.compile(r"[A-Za-z0-9_+#.-]{1,30}")


# ------------------------------------------------------------------ text safety
def md_literal(text: str) -> str:
    """Server text shown as exactly these characters in Markdown: one line, nothing active."""
    flat = " ".join(clean(text).split())
    return "".join(_escaped(flat, i, c) if c in _ASCII_PUNCT else c for i, c in enumerate(flat))


def _escaped(text: str, i: int, char: str) -> str:
    """`char` (ASCII punctuation at `text[i]`) backslash-escaped, plus a word joiner when it
    opens an emoji/icon shortcode (see module docstring: the escape alone doesn't stop those)."""
    if char == ":" and _SHORTCODE_TAIL.match(text, i + 1):
        return f"\\:{WORD_JOINER}"
    return f"\\{char}"


def for_streamlit(markdown: str) -> str:
    """Server CommonMark made safe for Streamlit's Markdown dialect (see module docstring)."""
    return _outside_fences(clean(markdown), _escape_streamlit_specials)


def answer_markdown(answer: str) -> str:
    """The final answer as Markdown, with each section name ("In simple terms", "Technically",
    "In production", ADR-0016) starting its own paragraph: a model that runs them together on
    one line would otherwise render one dense block."""
    return for_streamlit(_outside_fences(answer, _split_sections).lstrip("\n"))


def preview_text(answer: str) -> str:
    """The streaming answer as PLAIN text. The SSE contract says tokens render as plain text
    until `verification` arrives; showing the raw escapes (`\\[S1\\]`) would be noise, so the
    terminal client's unescaping is reused and its bold markers dropped."""
    return style(clean(unescape_outside_fences(answer)), tty=False)


def _outside_fences(text: str, transform: Callable[[str], str]) -> str:
    out: list[str] = []
    last = 0
    for block in _FENCE.finditer(text):
        out += [transform(text[last : block.start()]), block.group(0)]
        last = block.end()
    out.append(transform(text[last:]))
    return "".join(out)


def _escape_streamlit_specials(text: str) -> str:
    """Backslash `$` and `:` unless already escaped; code spans are left exactly as they are."""
    out: list[str] = []
    i = 0
    while i < len(text):
        char = text[i]
        if char == "\\" and i + 1 < len(text):
            # an existing escape stays one escape (the server escapes `:` itself, so a
            # shortcode usually arrives as `\:smile\:` and still needs its word joiner)
            out.append(
                _escaped(text, i + 1, text[i + 1]) if text[i + 1] == ":" else text[i : i + 2]
            )
            i += 2
            continue
        if char == "`":
            run = len(text[i:]) - len(text[i:].lstrip("`"))
            close = _closing_run(text, i + run, run)
            if close is not None:
                out.append(text[i : close + run])  # a code span: literal, untouched
                i = close + run
                continue
            out.append(text[i : i + run])
            i += run
            continue
        out.append(_escaped(text, i, char) if char in _STREAMLIT_SPECIAL else char)
        i += 1
    return "".join(out)


def _closing_run(text: str, start: int, run: int) -> int | None:
    """Index of the next backtick run of exactly `run` characters (CommonMark code span end)."""
    for match in re.finditer(r"`+", text[start:]):
        if len(match.group(0)) == run:
            return start + match.start()
    return None


def _split_sections(text: str) -> str:
    return _SECTION.sub(lambda m: "\n\n" + m.group(1), text)


# ------------------------------------------------------------------ verification badge
@dataclass(frozen=True, slots=True)
class Badge:
    label: str
    color: str  # a Streamlit badge colour
    icon: str
    meaning: str  # plain language: what the owner should do with the answer


_BADGES: dict[str, Badge] = {
    "verified": Badge(
        "Verified",
        "green",
        ":material/verified:",
        "Every statement is backed by your notes.",
    ),
    "partial": Badge(
        "Partly verified",
        "orange",
        ":material/rule:",
        "Most of it is backed by your notes; the statements listed below are not.",
    ),
    "unsupported": Badge(
        "Not supported",
        "red",
        ":material/report:",
        "Your notes don't back this answer. Don't rely on it.",
    ),
    "unverified": Badge(
        "Not checked",
        "gray",
        ":material/help:",
        "The model in use is below the verifier's quality bar, so nothing was checked.",
    ),
}


_NOTHING_LABEL, _NOTHING_MEANING = NOTHING_TO_CHECK.split(": ", 1)
NOTHING_BADGE = Badge(
    _NOTHING_LABEL,
    "gray",
    ":material/remove_done:",
    _NOTHING_MEANING[:1].upper() + _NOTHING_MEANING[1:],
)


def badge_for(status: str | None, *, cited: bool = True, flagged: bool = False) -> Badge | None:
    """The badge for a verification status. A status a newer server adds is shown neutrally
    (never green): the UI can't vouch for a meaning it doesn't know.

    `cited` False with `unverified` (live failure, 2026-10-19): a refusal or an uncited answer
    had nothing from the notes to check, so "below the verifier's quality bar" would be false;
    it says "Nothing to check" instead (`ask.nothing_to_check`, the same rule in `make ask`).
    Only when nothing was `flagged` either: an uncited unverified answer WITH flagged claims made
    claims the verifier skipped, so the "below the quality bar" text is the true one there."""
    if status is None:
        return None
    if nothing_to_check(status, cited=cited, flagged=flagged):
        return NOTHING_BADGE
    return _BADGES.get(
        status,
        Badge(
            md_literal(status),
            "gray",
            ":material/help:",
            "This UI doesn't know this status; treat the answer as unchecked.",
        ),
    )


# ------------------------------------------------------------------ citations
INFERRED_NOTE = "\\* matched by the server to a sentence the answer left uncited, then checked"


def citation_line(citation: Citation, index: int) -> str:
    """One citation chip as Markdown: `[S1] Title › heading`, linked to the Notion page.

    The tag is the label the ANSWER uses (`[S3]` stays S3, never a renumbering); without a label
    (older server) a plain index. An inferred citation gets `S2*` plus "matched by the server":
    the server, not the model, attached it, so it must never look like the model's own source.
    """
    tag = f"S{citation.label}" if citation.label is not None else str(index)
    if citation.inferred:
        tag += "\\*"
    where = " › ".join(md_literal(part) for part in [citation.title, *citation.heading_path])
    href = safe_href(str(citation.url)) if citation.url is not None else None
    place = f"[{where}]({href})" if href else where
    how = " · *matched by the server*" if citation.inferred else ""
    return f"**\\[{tag}\\]** {place}{how}"


def safe_href(url: str) -> str | None:
    """An http(s) URL safe inside `(...)` of a Markdown link, or None.

    The URL is the server's stored canonical Notion URL (`HttpUrl`), so this is belt and braces:
    characters that would end or bend the link destination are percent-encoded."""
    if not url.lower().startswith(("https://", "http://")):
        return None
    return quote(clean(url), safe=":/?#[]@!$&*+,;=%-._~")


# ------------------------------------------------------------------ approvals
def call_summary(call: PendingCall, titles: Mapping[str, str]) -> str | None:
    """What a pending call would do, in words, as Markdown; None if it names no document.

    Why: per-call Approve boxes exist so an injected call can be rejected on its own, but a bare
    UUID gives the owner nothing to judge. A document the UI can't name is said so plainly: an id
    that matches none of the owner's pages is exactly what an injected call could carry."""
    doc = document_id_arg(call.args)
    if doc is None:
        return None
    short = f"document {doc[:8]}…"
    title = titles.get(doc)
    if title is None:
        return f"An **unknown document** ({short}): this UI found no title for it. Check the id."
    named = f"“{md_literal(title)}”"
    if call.tool == "remove_from_index":
        return f"Remove {named} from the index ({short})"
    return f"Acts on {named} ({short})"


# ------------------------------------------------------------------ extras
_EXTRA_TITLES = {"incident": "A common production failure pattern", "code": "A code example"}
EXTRAS_HEADING = "Not from your notes — general knowledge (not verified)"


@dataclass(frozen=True, slots=True)
class ExtraView:
    title: str
    before: str  # Markdown (may be empty)
    code: str | None
    language: str | None
    after: str  # Markdown (may be empty)


def extra_view(extra: Extra) -> ExtraView:
    """Split an extra for display: prose as Markdown, and the first fenced block as code for
    `st.code` (exact characters, copyable, highlighted by its language when it names one)."""
    title = _EXTRA_TITLES.get(extra.kind, "General knowledge")
    block = _FENCE.search(extra.text) if extra.kind == "code" else None
    if block is None:
        return ExtraView(title, for_streamlit(extra.text), None, None, "")
    info = block.group("info").strip().split(" ", 1)[0]
    language = info if _LANGUAGE.fullmatch(info) else None
    return ExtraView(
        title=title,
        before=for_streamlit(extra.text[: block.start()].strip()),
        code=block.group("body").removesuffix("\n"),
        language=language,
        after=for_streamlit(extra.text[block.end() :].strip()),
    )


# ------------------------------------------------------------------ collections
def collection_labels(collections: Sequence[Collection]) -> dict[UUID, str]:
    """Picker labels keyed by collection ID. Names aren't unique, so a name shared by two
    collections gets a short ID suffix: both stay selectable and tell-apart-able."""
    counts = Counter(c.name for c in collections)
    return {
        c.id: c.name if counts[c.name] == 1 else f"{c.name} · {str(c.id)[:8]}" for c in collections
    }


# ------------------------------------------------------------------ small bits
def timings_text(first_token_s: float | None, total_s: float | None) -> str:
    first = f"{first_token_s:.1f} s" if first_token_s is not None else "-"
    total = f"{total_s:.1f} s" if total_s is not None else "-"
    return f"first token {first} · total {total}"


# ------------------------------------------------------------------ flagged claims
NOT_FOUND_TITLE = "Not found in your notes"
INEXACT_TITLE = "Couldn't match exactly (code/format)"
INEXACT_NOTE = (
    "Code shortened or reformatted from your notes: every name in it is in the cited note, but "
    "the lines around it (a guard, a warning, a trailing option) may be missing. Check it."
)


def bullet_markdown(items: Sequence[str]) -> str:
    """Server CommonMark items as a Markdown list, one line each: a newline inside an item would
    end the list item (or open a block the item never meant)."""
    return "\n".join(f"- {for_streamlit(' '.join(item.split()))}" for item in items)


# ------------------------------------------------------------------ interview
_KIND_LABELS: dict[str, str] = {
    InterviewKind.concept: "Concept",
    InterviewKind.scenario: "Scenario",
    InterviewKind.design: "System design",
    InterviewKind.code: "Code / LLD",
}


def kind_label(kind: InterviewKind | str) -> str:
    """Badge text for a question kind; a kind this UI doesn't know (newer server) is shown as
    its plain text, escaped."""
    return _KIND_LABELS.get(kind, md_literal(str(kind)))


def level_label(level: InterviewLevel | str) -> str:
    """A level's display name, escaped (an unknown level from a newer server is its text)."""
    return md_literal(level_name(level))


def question_level(question: InterviewQuestion) -> str:
    """ "Staff" or "Staff · harder" (the difficulty only when not standard), escaped."""
    return md_literal(question_line(question))


def topic_markdown(topic: str) -> str:
    """An interview topic on one line. The server already escaped it as CommonMark, so it is
    unescaped once and then escaped by `md_literal`: escaping it as is would show the server's
    backslashes ("Map\\<K,V>")."""
    return md_literal(plain_line(topic))


def weak_topics_markdown(summary: Summary) -> str:
    """The summary's weak topics, each with the pages to revisit as links (escaped titles)."""
    lines: list[str] = []
    for weak in summary.weak:
        lines.append(
            f"- **{topic_markdown(weak.topic)}** ({kind_label(weak.kind)}, {weak.score}/10)"
        )
        for page in weak.pages:
            href = safe_href(page.url) if page.url else None
            title = md_literal(page.title)
            lines.append(f"    - [{title}]({href})" if href else f"    - {title}")
    return "\n".join(lines)


def summary_headline_markdown(summary: Summary) -> str:
    """ "Staff · Kafka · avg 6.5" (topics are the owner's own words: escaped), in bold."""
    return f"**{md_literal(summary_headline(summary))}**"


def summary_line(summary: Summary) -> str:
    return f"{summary.asked} asked · {summary.answered} answered · {summary.skipped} skipped"


def weak_general_markdown(summary: Summary) -> str:
    """Weak general-knowledge answers: no pages to revisit, so the advice is to add notes."""
    return "\n".join(
        f"- **{md_literal(w.topic)}** ({kind_label(w.kind)}, {w.score}/10): consider adding "
        f"notes on {md_literal(w.topic)}"
        for w in summary.weak_general
    )
