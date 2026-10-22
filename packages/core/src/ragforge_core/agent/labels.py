"""The canonical citation marker (ADR-0015).

One definition of what a label looks like, shared by the citation check (citations.py), the
coverage guarantee (coverage.py) and attribution. WHERE a marker counts as a label (only in
plain visible text: not in code, links, autolinks, strikethrough, HTML or escapes) is decided on
the CommonMark parse (`structure.Atom.eligible`, `coverage._labels`), never here: this module
used to mask inline code and HTML comments with its own scanner, a second Markdown grammar that
disagreed with the renderer (review of the redesign, round 4).
"""

from __future__ import annotations

import re
from typing import Final

CITATION_MARKER: Final = re.compile(r"\[([ST]\d{1,4}(?:\s*,\s*[ST]\d{1,4})*)\]")
_LABEL: Final = re.compile(r"([ST])(\d{1,4})")


def all_labels(text: str) -> list[tuple[str, int]]:
    """(kind, number) of every canonical label in `text`, in order, valid or not."""
    groups = CITATION_MARKER.findall(text)
    return [(k, int(n)) for group in groups for k, n in _LABEL.findall(group)]


def label_names(text: str) -> list[str]:
    """Canonical label names in order of appearance ("S1", "T2"), valid or not."""
    return [f"{kind}{n}" for kind, n in all_labels(text)]
