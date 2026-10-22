"""The structural map the guarantee is computed on (agent/structure.py, ADR-0015 section 4).

Review of the redesign, round 4: the structure used to come from regexes; it now comes from a
markdown-it-py parse aligned back onto the source. These properties check that alignment over
generated Markdown (containers, code at any depth, multi-backtick spans, links, autolinks,
escapes, entities, HTML, tables, line endings) against markdown-it-py itself:
- every plain-text atom is literally its source, and atoms and parts are ordered and disjoint;
- every code element markdown-it-py finds is in the map with the same content, unless its part
  was made opaque (fail closed);
- nothing that says something is outside a part except the numerals of a numbered list.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from markdown_it import MarkdownIt

import ragforge_core
from ragforge_core.agent.structure import LINKIFY, Block, Inline, Kind, states_nothing, structure

# The same configuration the reader's renderer is assumed to have (ADR-0015 section 4): this
# test checks the ALIGNMENT of the map onto the source, so it parses exactly as the map does.
_MD = MarkdownIt(
    "commonmark", {"html": True, "linkify": LINKIFY, "strikethrough_single_tilde": True}
).enable(["table", "strikethrough"])
if LINKIFY:
    _MD.enable("linkify")
    if _MD.linkify is not None:
        _MD.linkify.set({"fuzzy_link": False})

_PIECES = [
    "Kafka stores records",
    " [S1]",
    ". ",
    "\n",
    "\n\n",
    "> ",
    "- ",
    "1. ",
    "3) ",
    "    ",
    "```\n",
    "~~~py\n",
    "`",
    "``",
    "**",
    "_",
    "~~",
    "~",
    "[docs](https://x.example)",
    '[t](https://x.example "title")',
    "<https://x.example/a>",
    "https://x.example/b",
    "\\[",
    "&amp;",
    "&#91;",
    "<br>",
    "<!-- c -->",
    "<p class=x>",
    "| a | b |\n|---|---|\n| c | d |\n",
    "[r]: https://x.example\n",
    "![img](x.png)",
    "\r\n",
    "\t",
    "# ",
    "---\n",
    "\u3002",
    "\u200b",
    # blind review round 3 of the parser rewrite: Unicode whitespace str.strip removes, an
    # empty numbered item, a sentence cut inside a titled link
    "\u00a0",
    "\u202f",
    "\u3000",
    "2.\n",
    '[! a](https://x.example "t")',
]


@st.composite
def _markdown(draw: st.DrawFn) -> str:
    return "".join(draw(st.lists(st.sampled_from(_PIECES), min_size=1, max_size=14)))


def _md_code(text: str) -> list[str]:
    """Every code element markdown-it-py finds, by its own walk."""
    found: list[str] = []
    for token in _MD.parse(text):
        if token.type in ("fence", "code_block"):
            found.append(token.content)
        for child in token.children or []:
            if child.type == "code_inline":
                found.append(child.content)
    return found


@settings(max_examples=1500, deadline=None)
@given(text=_markdown())
def test_the_map_is_faithful_to_the_source(text: str) -> None:
    doc = structure(text)
    cursor = 0
    for part in doc.parts:
        assert cursor <= part.start <= part.end <= len(text), (text, part)
        cursor = part.end
        if isinstance(part, Inline):
            inner = part.start
            for atom in part.atoms:
                assert inner <= atom.start < atom.end <= part.end, (text, atom)
                inner = atom.end
                if atom.kind is Kind.text:
                    assert text[atom.start : atom.end] == atom.visible, (text, atom)
                if atom.kind is Kind.code:
                    assert atom.code is not None
                    assert text[atom.start] == "`", (text, atom)
                    assert text[atom.end - 1] == "`", (text, atom)
    # every code element markdown-it-py sees is in the map, unless a part was made opaque
    opaque = any(isinstance(p, Block) and p.kind == "opaque" for p in doc.parts)
    found = [c.text for c in doc.code]
    if not opaque:
        assert found == _md_code(text.replace("\r\n", "\n").replace("\r", "\n")), text
    # accounting: outside every part, only structure (and a numbered list's numerals)
    covered = [False] * len(text)
    for part in doc.parts:
        covered[part.start : part.end] = [True] * (part.end - part.start)
    for i, char in enumerate(text):
        if not covered[i]:
            assert states_nothing(char) or char == "#" or char.isdigit(), (text, i, char)


def test_a_part_that_cannot_be_aligned_is_opaque_never_guessed() -> None:
    """A tab inside a list item's content is expanded by the block parser, so the content no
    longer equals the source: the paragraph is opaque (residue), not mis-mapped."""
    text = "- a\n\n \tb [S1]"
    doc = structure(text)
    assert all(not isinstance(p, Inline) or text[p.start : p.end].strip() for p in doc.parts)
    for part in doc.parts:
        if isinstance(part, Inline):
            for atom in part.atoms:
                if atom.kind is Kind.text:
                    assert text[atom.start : atom.end] == atom.visible


@pytest.mark.parametrize("space", ["\u00a0", "\u202f", "\u2003", "\u3000"])
@pytest.mark.parametrize(
    "template",
    ["{s}Kafka stores records{s}", "## Kafka stores records{s}", "| a{s}|{s}b{s}|\n|---|---|"],
)
def test_unicode_whitespace_at_a_parts_edges_aligns(space: str, template: str) -> None:
    """markdown-it strips paragraph, heading and cell content with `str.strip()` (any Unicode
    whitespace); the alignment accepts the same set, so the part is mapped, not opaque (blind
    review round 3, P2: a trailing NBSP made a whole paragraph opaque)."""
    text = template.format(s=space)
    parts = structure(text).parts
    assert parts, text
    assert all(isinstance(p, Inline) for p in parts), (text, parts)
    for part in parts:
        assert isinstance(part, Inline)
        assert not text[part.start : part.end].startswith(space)
        assert not text[part.start : part.end].endswith(space)


@pytest.mark.parametrize(
    "text",
    [
        '[! Kafka stores records](https://x.example "NASA built Kafka")',
        "See [Kafka. Docs](https://x.example) now.",
        "[a `b` c](https://x.example)",
    ],
)
def test_every_atom_of_a_link_is_opaque(text: str) -> None:
    """A link hides its destination and title, so every atom from "[" to ")" is opaque, and a
    unit overlapping any part of it gets no inferred citation (blind review round 3, P2)."""
    (part,) = structure(text).parts
    assert isinstance(part, Inline)
    start, end = text.index("["), text.index(")") + 1
    inside = [a for a in part.atoms if start <= a.start < end]
    assert inside
    assert all(a.opaque for a in inside)
    assert not any(a.opaque for a in part.atoms if a.start >= end or a.end <= start)


@pytest.mark.parametrize(
    ("text", "numerals"),
    [
        ("1. a\n2.", ["2"]),
        ("1)\n2)\n3)", ["1", "2", "3"]),
        ("1. [S1]\n2. b", ["1"]),
        ("1. a\n2. - b", []),
        ("1. a\n\n   ```\n   x\n   ```\n2. b", []),
    ],
)
def test_an_empty_items_numeral_is_unclaimed(text: str, numerals: list[str]) -> None:
    """The numeral of an actual numbered list is structure only when its item says something
    besides labels; otherwise it is an opaque block (blind review round 3, P2)."""
    found = [text[p.start : p.end] for p in structure(text).parts if isinstance(p, Block)]
    assert [f for f in found if f.isdigit()] == numerals


def test_markdown_it_py_is_a_runtime_dependency_of_the_agent() -> None:
    """agent/structure.py imports markdown_it at module load, and the image installs with
    `uv sync --no-dev --extra agent --extra ingest`: markdown-it-py must be in ragforge-core's
    base dependencies or its `agent` extra, or a rebuilt api/worker image cannot import the
    agent (review of the parser rewrite, round 1, P0)."""
    pyproject = Path(ragforge_core.__file__).resolve().parents[2] / "pyproject.toml"
    project = tomllib.loads(pyproject.read_text())["project"]
    runtime = [*project["dependencies"], *project.get("optional-dependencies", {}).get("agent", [])]
    assert any(dep.replace(" ", "").lower().startswith("markdown-it-py") for dep in runtime)
