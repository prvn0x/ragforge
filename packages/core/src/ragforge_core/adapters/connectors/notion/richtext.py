"""Notion rich text -> Markdown, collecting page links from mentions on the way.

Mentions keep their visible text (DESIGN 7.1: "text kept; edge in document_links"); page
mentions also add the target's external id to `links`. So do plain text hyperlinks that point
at a Notion page (notion.so / notion.site URLs, or the relative `/<id>` hrefs Notion emits for
in-workspace links): to a reader they are the same edge as a mention. Inline equations become
`$...$`.

Link hygiene (best effort, NOT the security boundary): only http(s) hrefs become Markdown links;
the href's whitespace, `(`, `)`, `<` and `>` are percent-encoded so it cannot close the link
early and start a second one, and `[`/`]` in the link text are escaped. Ordinary text is NOT
Markdown-escaped (that would mangle real content such as `[x]`), so a page can still contain the
literal text `[x](javascript:...)`. URL-scheme safety of rendered answers is owned by the output
sanitizer (module E's OutputSanitizer), which must treat all indexed text as untrusted.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from typing import Any
from urllib.parse import parse_qs, quote, urlsplit

from ragforge_core.adapters.connectors.notion.ids import canonical_id

_SAFE_SCHEMES = ("https://", "http://")
_NOTION_HOSTS = ("notion.so", "notion.site")
_HREF_SPECIALS = re.compile(r"[\s()<>]")
_LINK_TEXT_SPECIALS = re.compile(r"[\[\]]")
# A Notion URL path ends with the 32-hex id ("Title-<id>") or, rarely, a dashed UUID.
_TRAILING_ID = re.compile(
    r"([0-9a-f]{32}|[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})$", re.IGNORECASE
)


class LinkCollector:
    """Ordered, de-duplicated external ids of linked pages."""

    def __init__(self) -> None:
        self._ids: dict[str, None] = {}

    def add(self, raw_id: object) -> None:
        if not isinstance(raw_id, str):
            return
        try:
            self._ids[canonical_id(raw_id)] = None
        except ValueError:
            return

    def ids(self) -> tuple[str, ...]:
        return tuple(self._ids)


def plain_text(items: Iterable[Mapping[str, Any]] | None) -> str:
    return "".join(str(item.get("plain_text") or "") for item in items or ())


def rich_text_md(items: Iterable[Mapping[str, Any]] | None, links: LinkCollector) -> str:
    return "".join(_item_md(item, links) for item in items or ())


def _item_md(item: Mapping[str, Any], links: LinkCollector) -> str:
    kind = item.get("type")
    text = str(item.get("plain_text") or "")
    if kind == "equation":
        expression = str((item.get("equation") or {}).get("expression") or text)
        return f"${expression}$" if expression else ""
    if kind == "mention":
        _collect_mention(item.get("mention") or {}, links)
        return _annotate(text, item.get("annotations") or {})
    href = item.get("href")
    if isinstance(href, str):
        links.add(notion_page_id_from_href(href))
    styled = _annotate(text, item.get("annotations") or {})
    if isinstance(href, str) and href.startswith(_SAFE_SCHEMES) and text.strip():
        return f"[{_escape_link_text(styled)}]({_encode_href(href)})"
    return styled


def _escape_link_text(text: str) -> str:
    return _LINK_TEXT_SPECIALS.sub(lambda m: "\\" + m.group(0), text)


def _encode_href(href: str) -> str:
    """Percent-encode the characters that would end a Markdown link destination early."""
    return _HREF_SPECIALS.sub(lambda m: quote(m.group(0), safe=""), href)


def notion_page_id_from_href(href: str) -> str | None:
    """The page id a Notion URL points at, or None for any other URL.

    `?p=<id>` (a database row opened as a peek) wins over the path, which then names the
    database view. A `#fragment` names a BLOCK inside the page, so it is ignored: the edge is
    page-to-page. Only Notion hosts (and relative hrefs, which Notion uses for workspace links)
    count, so an arbitrary site whose path happens to end in 32 hex is never mistaken for one.
    """
    try:
        parts = urlsplit(href.strip())
    except ValueError:
        return None
    if parts.scheme or parts.netloc:
        host = (parts.hostname or "").lower()
        is_notion = parts.scheme in ("http", "https") and any(
            host == h or host.endswith("." + h) for h in _NOTION_HOSTS
        )
        if not is_notion:
            return None
    elif not parts.path.startswith("/"):
        return None
    peek = parse_qs(parts.query).get("p")
    if peek and _TRAILING_ID.fullmatch(peek[0]):
        return peek[0]
    segment = parts.path.rstrip("/").rsplit("/", 1)[-1]
    match = _TRAILING_ID.search(segment)
    return match.group(1) if match else None


def _collect_mention(mention: Mapping[str, Any], links: LinkCollector) -> None:
    kind = mention.get("type")
    if kind == "page":
        links.add((mention.get("page") or {}).get("id"))
    elif kind == "link_to_page":  # defensive: some payloads nest link_to_page in mentions
        links.add((mention.get("link_to_page") or {}).get("page_id"))


def _annotate(text: str, annotations: Mapping[str, Any]) -> str:
    """Wrap the non-whitespace core so `**bold **` never renders as broken Markdown."""
    core = text.strip()
    if not core:
        return text
    lead = text[: len(text) - len(text.lstrip())]
    trail = text[len(text.rstrip()) :]
    if annotations.get("code"):
        core = f"`{core}`"
    if annotations.get("bold"):
        core = f"**{core}**"
    if annotations.get("italic"):
        core = f"*{core}*"
    if annotations.get("strikethrough"):
        core = f"~~{core}~~"
    return f"{lead}{core}{trail}"
