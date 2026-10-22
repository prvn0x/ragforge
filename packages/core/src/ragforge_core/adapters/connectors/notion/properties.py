"""Data-source row properties -> JSON-safe metadata (DESIGN 7.1: "properties -> metadata").

Each Notion property type is reduced to the value a person would read (names, not option ids;
ISO dates). Relations are also page links. Uploaded-file URLs are signed and expiring, so only
file NAMES are kept. Unknown property types are dropped rather than stored raw.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from ragforge_core.adapters.connectors.notion.richtext import LinkCollector, plain_text

Simplifier = Callable[[Mapping[str, Any], LinkCollector], Any]


def title_of(properties: Mapping[str, Any]) -> str:
    """Plain text of the (single) title property; "" when there is none."""
    for prop in properties.values():
        if isinstance(prop, Mapping) and prop.get("type") == "title":
            return plain_text(prop.get("title")).strip()
    return ""


def simplify_properties(properties: Mapping[str, Any], links: LinkCollector) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, prop in properties.items():
        if not isinstance(prop, Mapping):
            continue
        simplifier = _SIMPLIFIERS.get(str(prop.get("type")))
        if simplifier is None:
            continue
        value = simplifier(prop, links)
        if value is not None:
            out[str(name)] = value
    return out


def truncated_properties(properties: Mapping[str, Any]) -> list[str]:
    """Names of properties whose value the page object cut short (`has_more`).

    The page object returns at most 25 items of a relation, people, title or rich_text value.
    Paging each through `GET /pages/{id}/properties/{prop}` would cost one or more extra
    rate-limited requests per row, per property, on every sync. Recording the gap keeps the
    metadata honest (consumers know the list and the links_to edges are partial) at no cost.
    """
    return sorted(
        str(name)
        for name, prop in properties.items()
        if isinstance(prop, Mapping) and prop.get("has_more") is True
    )


def properties_markdown(values: Mapping[str, Any]) -> str:
    """`**Name:** value` lines, so row properties are searchable text, not only metadata."""
    lines = []
    for name, value in values.items():
        rendered = _render(value)
        if rendered:
            lines.append(f"**{name}:** {rendered}")
    return "\n".join(lines)


def _render(value: Any) -> str:
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, list):
        return ", ".join(_render(v) for v in value if _render(v))
    if isinstance(value, dict):
        start, end = value.get("start"), value.get("end")
        return f"{start} → {end}" if end else str(start or "")
    return "" if value is None else str(value)


def _named(obj: Any) -> str | None:
    return str(obj["name"]) if isinstance(obj, Mapping) and obj.get("name") else None


def _date(obj: Any) -> dict[str, Any] | None:
    if not isinstance(obj, Mapping) or not obj.get("start"):
        return None
    return {"start": obj.get("start"), "end": obj.get("end")}


def _person(obj: Any) -> str | None:
    if not isinstance(obj, Mapping):
        return None
    return str(obj.get("name") or obj.get("id") or "") or None


def _relation(prop: Mapping[str, Any], links: LinkCollector) -> list[str]:
    ids = [str(r.get("id")) for r in prop.get("relation") or [] if isinstance(r, Mapping)]
    for rid in ids:
        links.add(rid)
    return ids


def _formula(prop: Mapping[str, Any], _: LinkCollector) -> Any:
    formula = prop.get("formula") or {}
    kind = formula.get("type")
    return _date(formula.get(kind)) if kind == "date" else formula.get(kind)


def _rollup(prop: Mapping[str, Any], links: LinkCollector) -> Any:
    rollup = prop.get("rollup") or {}
    kind = rollup.get("type")
    if kind == "array":
        items = [simplify_properties({"_": v}, links).get("_") for v in rollup.get("array") or []]
        return [v for v in items if v is not None]
    if kind == "date":
        return _date(rollup.get("date"))
    return rollup.get(kind)


def _unique_id(prop: Mapping[str, Any], _: LinkCollector) -> str | None:
    uid = prop.get("unique_id") or {}
    if uid.get("number") is None:
        return None
    return f"{uid['prefix']}-{uid['number']}" if uid.get("prefix") else str(uid["number"])


_SIMPLIFIERS: dict[str, Simplifier] = {
    "title": lambda p, _: plain_text(p.get("title")),
    "rich_text": lambda p, _: plain_text(p.get("rich_text")),
    "number": lambda p, _: p.get("number"),
    "checkbox": lambda p, _: bool(p.get("checkbox")),
    "url": lambda p, _: p.get("url"),
    "email": lambda p, _: p.get("email"),
    "phone_number": lambda p, _: p.get("phone_number"),
    "select": lambda p, _: _named(p.get("select")),
    "status": lambda p, _: _named(p.get("status")),
    "multi_select": lambda p, _: [n for n in map(_named, p.get("multi_select") or []) if n],
    "date": lambda p, _: _date(p.get("date")),
    "people": lambda p, _: [n for n in map(_person, p.get("people") or []) if n],
    "created_by": lambda p, _: _person(p.get("created_by")),
    "last_edited_by": lambda p, _: _person(p.get("last_edited_by")),
    "created_time": lambda p, _: p.get("created_time"),
    "last_edited_time": lambda p, _: p.get("last_edited_time"),
    "files": lambda p, _: [n for n in map(_named, p.get("files") or []) if n],
    "relation": _relation,
    "formula": _formula,
    "rollup": _rollup,
    "unique_id": _unique_id,
}
