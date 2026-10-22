"""Notion page + block tree -> domain NormalizedBlocks (DESIGN section 7.1). Pure: no I/O.

The fetcher (connector.py) builds the `BlockNode` tree (children fetched, synced blocks
resolved, depth capped); this module only decides how each block reads as Markdown.

Anchors: every NormalizedBlock's anchor is the dashless id of the Notion block it came from, so
`deep_link(page_url, anchor)` lands on that exact block. Container blocks that produce no text
of their own (columns, synced blocks) are transparent: their children keep their own anchors.

| Notion block                              | Result                                          |
|-------------------------------------------|-------------------------------------------------|
| heading_1..4 (incl. toggleable)           | `heading_N`, heading_level N, children inline   |
| paragraph                                 | `paragraph` (empty ones dropped)                |
| bulleted/numbered list, to_do             | `list` ("- ", "1. ", "- [x] "), nesting indent  |
| toggle                                    | `paragraph`, children expanded inline           |
| quote / callout                           | `quote` / `callout` ("> " prefixed)             |
| code                                      | `code`, fenced with language, never_split       |
| table                                     | `table` row groups, header repeated, never_split|
|                                           | (group 2+ anchored on its first row block)      |
| equation                                  | `equation` ($$ LaTeX $$)                        |
| image/video/file/pdf/audio/bookmark/embed | `media`: caption + file name (never signed URLs)|
| synced_block                              | resolved content; reference -> synced_source_ids|
| child_page / link_to_page / page mention  | links_to                                        |
| column_list / column                      | transparent                                     |
| divider / breadcrumb / table_of_contents  | dropped (no content)                            |
| anything else                             | skipped, counted in `unsupported`               |
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import unquote, urlsplit

from ragforge_core.adapters.connectors.notion.ids import anchor_id
from ragforge_core.adapters.connectors.notion.richtext import (
    LinkCollector,
    plain_text,
    rich_text_md,
)
from ragforge_core.domain.models import NormalizedBlock

# A table row group stays well inside one 512-token chunk (DESIGN 7.2) even with the header.
TABLE_GROUP_MAX_ROWS = 20
TABLE_GROUP_MAX_CHARS = 1500

# A plausible file name: no path separators or control characters, a short alphanumeric extension.
_FILE_NAME = re.compile(r"[^/\\\x00-\x1f]{1,200}\.[A-Za-z0-9]{1,8}")

# heading_4 is a newer Notion block type; unknown heading levels would silently drop the text.
HEADING_TYPES = {"heading_1": 1, "heading_2": 2, "heading_3": 3, "heading_4": 4}
MEDIA_TYPES = frozenset({"image", "video", "file", "pdf", "audio"})
LINK_MEDIA_TYPES = frozenset({"bookmark", "embed", "link_preview"})
TRANSPARENT_TYPES = frozenset({"column_list", "column", "synced_block"})
CONTENTLESS_TYPES = frozenset({"divider", "breadcrumb", "table_of_contents", "child_database"})
# Blocks whose children are rendered. The fetcher only spends rate budget on these.
CHILD_BEARING_TYPES = frozenset(
    {
        "paragraph",
        "bulleted_list_item",
        "numbered_list_item",
        "to_do",
        "toggle",
        "quote",
        "callout",
        "table",
        *HEADING_TYPES,
        *TRANSPARENT_TYPES,
    }
)


@dataclass(frozen=True, slots=True)
class BlockNode:
    """A fetched block with its (fetched) children.

    `synced_source_id`: set on a synced_block REFERENCE whose original's content was fetched
    into `children`. `truncated`: the block had children beyond the depth limit.
    """

    block: Mapping[str, Any]
    children: tuple[BlockNode, ...] = ()
    synced_source_id: str | None = None
    truncated: bool = False


@dataclass(frozen=True, slots=True)
class NormalizedPage:
    blocks: tuple[NormalizedBlock, ...]
    links_to: tuple[str, ...]
    synced_source_ids: tuple[str, ...]
    unsupported: dict[str, int] = field(default_factory=dict)
    truncated: bool = False


def normalize_blocks(nodes: Sequence[BlockNode], links: LinkCollector) -> NormalizedPage:
    """Normalize a page's top-level block nodes. `links` may already hold property links."""
    walker = _Walker(links)
    blocks = tuple(walker.walk(nodes, indent=0))
    return NormalizedPage(
        blocks=blocks,
        links_to=links.ids(),
        synced_source_ids=tuple(walker.synced),
        unsupported=dict(walker.unsupported),
        truncated=walker.truncated,
    )


class _Walker:
    def __init__(self, links: LinkCollector) -> None:
        self.links = links
        self.synced: dict[str, None] = {}
        self.unsupported: Counter[str] = Counter()
        self.truncated = False
        self._handlers: dict[str, Callable[[BlockNode, int, int], Iterator[NormalizedBlock]]] = {
            "paragraph": self._paragraph,
            "bulleted_list_item": self._list_item,
            "numbered_list_item": self._list_item,
            "to_do": self._list_item,
            "toggle": self._toggle,
            "quote": self._quote,
            "callout": self._quote,
            "code": self._code,
            "equation": self._equation,
            "table": self._table,
            "child_page": self._child_page,
            "link_to_page": self._link_to_page,
            **dict.fromkeys(HEADING_TYPES, self._heading),
            **dict.fromkeys(MEDIA_TYPES | LINK_MEDIA_TYPES, self._media),
            **dict.fromkeys(TRANSPARENT_TYPES, self._transparent),
            **dict.fromkeys(CONTENTLESS_TYPES, self._nothing),
        }

    def walk(self, nodes: Sequence[BlockNode], indent: int) -> Iterator[NormalizedBlock]:
        number = 0  # numbered-list counter: restarts after any non-numbered sibling
        for node in nodes:
            kind = str(node.block.get("type"))
            self.truncated = self.truncated or node.truncated
            if node.synced_source_id:
                self.synced[node.synced_source_id] = None
            if kind == "numbered_list_item":
                start = _body(node).get("list_start_index")
                number = int(start) if number == 0 and isinstance(start, int) else number + 1
            else:
                number = 0
            handler = self._handlers.get(kind)
            if handler is None:
                self.unsupported[kind] += 1
                continue
            yield from handler(node, indent, number)

    # ------------------------------------------------------------------ block handlers
    def _heading(self, node: BlockNode, indent: int, _: int) -> Iterator[NormalizedBlock]:
        kind = str(node.block["type"])
        level = HEADING_TYPES[kind]
        text = self._text(node)
        if text:
            yield _block(node, kind, f"{'#' * level} {text}", heading_level=level)
        yield from self.walk(node.children, indent)  # toggleable headings: expanded

    def _paragraph(self, node: BlockNode, indent: int, _: int) -> Iterator[NormalizedBlock]:
        text = self._text(node)
        if text.strip():
            yield _block(node, "paragraph", _indent(text, indent))
        yield from self.walk(node.children, indent)

    def _list_item(self, node: BlockNode, indent: int, number: int) -> Iterator[NormalizedBlock]:
        kind = node.block["type"]
        if kind == "numbered_list_item":
            marker = f"{max(number, 1)}. "
        elif kind == "to_do":
            marker = "- [x] " if _body(node).get("checked") else "- [ ] "
        else:
            marker = "- "
        yield _block(node, "list", _indent(marker + self._text(node), indent))
        yield from self.walk(node.children, indent + 1)

    def _toggle(self, node: BlockNode, indent: int, _: int) -> Iterator[NormalizedBlock]:
        text = self._text(node)
        if text.strip():
            yield _block(node, "paragraph", _indent(text, indent))
        yield from self.walk(node.children, indent)

    def _quote(self, node: BlockNode, indent: int, _: int) -> Iterator[NormalizedBlock]:
        kind = str(node.block["type"])
        text = self._text(node)
        icon = _body(node).get("icon") or {}
        if kind == "callout" and icon.get("type") == "emoji" and icon.get("emoji"):
            text = f"{icon['emoji']} {text}"
        if text.strip():
            quoted = "\n".join(f"> {line}" for line in text.splitlines())
            yield _block(node, kind, _indent(quoted, indent))
        yield from self.walk(node.children, indent)

    def _code(self, node: BlockNode, indent: int, _: int) -> Iterator[NormalizedBlock]:
        body = _body(node)
        code = plain_text(body.get("rich_text"))
        language = str(body.get("language") or "plain text")
        fence_lang = "" if language == "plain text" else language
        fence = "````" if "```" in code else "```"
        text = f"{fence}{fence_lang}\n{code}\n{fence}"
        yield _block(node, "code", text, language=language, never_split=True)

    def _equation(self, node: BlockNode, indent: int, _: int) -> Iterator[NormalizedBlock]:
        expression = str(_body(node).get("expression") or "").strip()
        if expression:
            yield _block(node, "equation", f"$$\n{expression}\n$$")

    def _table(self, node: BlockNode, indent: int, _: int) -> Iterator[NormalizedBlock]:
        """Row groups with the header repeated. The FIRST group anchors on the table itself (a
        citation there lands on the header too); every later group anchors on its own first
        table_row block, so a deep link to row 245 of a long table lands near row 245."""
        body = _body(node)
        row_nodes = [child for child in node.children if _is_row(child)]
        rows = [self._cells(child) for child in row_nodes]
        if not rows:
            return
        width = max(int(body.get("table_width") or 0), *(len(r) for r in rows))
        rows = [r + [""] * (width - len(r)) for r in rows]
        if body.get("has_column_header"):
            header, data, data_nodes = rows[0], rows[1:], row_nodes[1:]
        else:
            header, data, data_nodes = [""] * width, rows, row_nodes
        head = _row_md(header) + "\n" + _row_md(["---"] * width)
        start = 0
        for group in _row_groups(data, len(head)):
            text = head + "".join("\n" + _row_md(r) for r in group)
            anchor_node = node if start == 0 else data_nodes[start]
            yield _block(anchor_node, "table", text, never_split=True)
            start += len(group)
        if not data:  # header-only table still carries information
            yield _block(node, "table", head, never_split=True)

    def _child_page(self, node: BlockNode, indent: int, _: int) -> Iterator[NormalizedBlock]:
        self.links.add(node.block.get("id"))  # the child page is its own document
        return iter(())

    def _link_to_page(self, node: BlockNode, indent: int, _: int) -> Iterator[NormalizedBlock]:
        body = _body(node)
        if body.get("type") == "page_id":
            self.links.add(body.get("page_id"))
        return iter(())

    def _media(self, node: BlockNode, indent: int, _: int) -> Iterator[NormalizedBlock]:
        kind = str(node.block["type"])
        body = _body(node)
        caption = rich_text_md(body.get("caption"), self.links).strip()
        name = str(body.get("name") or "").strip() or _file_name_from_url(body)
        url = str(body.get("url") or "")
        parts = [p for p in (caption, name) if p]
        if not parts and kind in LINK_MEDIA_TYPES and url.startswith(("https://", "http://")):
            parts = [f"<{url}>"]  # a public bookmark URL is the content; uploads never are
        if parts:
            yield _block(node, "media", _indent(f"[{kind}] " + " — ".join(parts), indent))

    def _transparent(self, node: BlockNode, indent: int, _: int) -> Iterator[NormalizedBlock]:
        yield from self.walk(node.children, indent)

    def _nothing(self, node: BlockNode, indent: int, _: int) -> Iterator[NormalizedBlock]:
        return iter(())

    # ------------------------------------------------------------------ helpers
    def _text(self, node: BlockNode) -> str:
        return rich_text_md(_body(node).get("rich_text"), self.links)

    def _cells(self, row: BlockNode) -> list[str]:
        cells = _body(row).get("cells") or []
        return [_cell(rich_text_md(cell, self.links)) for cell in cells]


def _body(node: BlockNode) -> Mapping[str, Any]:
    body = node.block.get(str(node.block.get("type")))
    return body if isinstance(body, Mapping) else {}


def _block(
    node: BlockNode,
    kind: str,
    text: str,
    *,
    heading_level: int | None = None,
    language: str | None = None,
    never_split: bool = False,
) -> NormalizedBlock:
    return NormalizedBlock(
        anchor=anchor_id(str(node.block["id"])),
        kind=kind,
        text=text,
        heading_level=heading_level,
        language=language,
        never_split=never_split,
    )


def _file_name_from_url(body: Mapping[str, Any]) -> str:
    """The file name of an uploaded/external media file, from its URL PATH only.

    Never the URL itself and never its query string: Notion-hosted files are signed S3 URLs
    whose query carries the signature. Only a basename with an extension counts, so a page URL
    such as `youtube.com/watch` does not masquerade as a file name."""
    source = body.get(str(body.get("type")))
    url = source.get("url") if isinstance(source, Mapping) else None
    if not isinstance(url, str):
        return ""
    try:
        path = urlsplit(url).path
    except ValueError:
        return ""
    candidate = unquote(path.rsplit("/", 1)[-1]).strip()
    return candidate if _FILE_NAME.fullmatch(candidate) else ""


def _indent(text: str, indent: int) -> str:
    pad = "  " * indent
    return "\n".join(pad + line for line in text.splitlines()) if pad else text


def _is_row(node: BlockNode) -> bool:
    return node.block.get("type") == "table_row"


def _cell(text: str) -> str:
    return " ".join(text.replace("|", "\\|").split())


def _row_md(cells: Sequence[str]) -> str:
    return "| " + " | ".join(cells) + " |"


def _row_groups(rows: Sequence[list[str]], header_len: int) -> Iterator[list[list[str]]]:
    group: list[list[str]] = []
    size = header_len
    for row in rows:
        row_len = len(_row_md(row)) + 1
        if group and (len(group) >= TABLE_GROUP_MAX_ROWS or size + row_len > TABLE_GROUP_MAX_CHARS):
            yield group
            group, size = [], header_len
        group.append(row)
        size += row_len
    if group:
        yield group
