"""Versioned prompt templates (DESIGN section 8: "Prompts and outputs: versioned templates").

Every template is a Markdown file under `templates/` that starts with a mandatory header:

    ---
    name: answer
    version: 3
    ---
    <body, rendered with string.Template `$placeholders`>

WHY a header instead of a constant: the version travels with the text it versions, so editing a
prompt and forgetting to bump the version is visible in review of that one file. `PROMPT_VERSION`
(recorded in every TurnDone and in eval_runs) is the sorted `name@version` list of ALL templates,
so evals and usage rows can be grouped by the exact prompt set that produced them.

Per-surface sets (ADR-0017 §1): `PROMPT_VERSIONS` splits the same tags by the SURFACE they serve
(`chat`, `verifier`, `interview`), so an interview prompt edit never looks like a chat change to
the eval gate, whose baselines are keyed per surface. `PROMPT_VERSION` stays for compatibility.
The judge surface has no server template (the judge runs in the client, ADR-0017 §4).

WHY `string.Template`: retrieved text routinely contains `{}` (code, JSON); `str.format` on a body
that later gets user data spliced in is a classic injection/KeyError source. Templates are filled
only via `render()`, which substitutes values and never re-parses them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from importlib.resources import files
from string import Template
from typing import Final

__all__ = [
    "PROMPT_VERSION",
    "PROMPT_VERSIONS",
    "SURFACES",
    "PromptTemplate",
    "get_prompt",
    "load_templates",
    "render",
    "surface_of",
    "surface_versions",
]

_HEADER = re.compile(
    r"\A---\nname: (?P<name>[a-z_]+)\nversion: (?P<version>[1-9][0-9]*)\n---\n(?P<body>.*)\Z",
    re.DOTALL,
)


@dataclass(frozen=True, slots=True)
class PromptTemplate:
    name: str
    version: int
    body: str

    @property
    def tag(self) -> str:
        return f"{self.name}@{self.version}"


def parse_template(text: str, *, source: str) -> PromptTemplate:
    """Parse one template file; a missing or malformed header is a packaging bug (fail loudly)."""
    match = _HEADER.match(text)
    if match is None:
        raise ValueError(f"prompt template {source} is missing its version header")
    return PromptTemplate(
        name=match["name"], version=int(match["version"]), body=match["body"].rstrip("\n")
    )


def load_templates() -> dict[str, PromptTemplate]:
    root = files(__name__).joinpath("templates")
    out: dict[str, PromptTemplate] = {}
    for entry in sorted(root.iterdir(), key=lambda e: e.name):
        if not entry.name.endswith(".md"):
            continue
        tpl = parse_template(entry.read_text(encoding="utf-8"), source=entry.name)
        if tpl.name != entry.name.removesuffix(".md"):
            raise ValueError(f"prompt template {entry.name} declares name {tpl.name!r}")
        out[tpl.name] = tpl
    return out


_TEMPLATES: Final = load_templates()

PROMPT_VERSION: Final = ",".join(t.tag for t in sorted(_TEMPLATES.values(), key=lambda t: t.name))


# ADR-0017 §1 surfaces. The verifier's template is listed by name (it also serves interview
# verification: one verifier, one baseline); interview templates share a prefix; every other
# template belongs to the chat turn (answer, rewrite, contextualize, abstain, actions, extras).
SURFACES: Final = ("chat", "verifier", "interview")
_VERIFIER_TEMPLATES: Final = frozenset({"ground_check"})
_INTERVIEW_PREFIX: Final = "interview_"
_TAG: Final = re.compile(r"(?P<name>[a-z_]+)@[1-9][0-9]*")


def surface_of(name: str) -> str:
    """The surface a template NAME belongs to (see SURFACES)."""
    if name in _VERIFIER_TEMPLATES:
        return "verifier"
    if name.startswith(_INTERVIEW_PREFIX):
        return "interview"
    return "chat"


def surface_versions(*prompt_sets: str) -> dict[str, str]:
    """{surface: "name@v,..."} from one or more `PROMPT_VERSION`-shaped strings.

    WHY several: a turn resumed by a newer deploy ran under two prompt sets (TurnDone records
    both); each surface then reports every distinct set it used, joined by " + " in the order
    given, exactly like `prompt_version`. A surface with no template in any set is absent."""
    per: dict[str, list[str]] = {}
    for prompt_set in prompt_sets:
        split: dict[str, list[str]] = {}
        for match in _TAG.finditer(prompt_set):
            split.setdefault(surface_of(match["name"]), []).append(match.group(0))
        for surface, tags in split.items():
            joined = ",".join(sorted(tags))
            seen = per.setdefault(surface, [])
            if joined not in seen:
                seen.append(joined)
    return {s: " + ".join(per[s]) for s in SURFACES if s in per}


PROMPT_VERSIONS: Final = surface_versions(PROMPT_VERSION)


def get_prompt(name: str) -> PromptTemplate:
    return _TEMPLATES[name]


def render(name: str, **values: str) -> str:
    """Fill a template's `$placeholders`. Missing values raise (a typo must not ship silently)."""
    return Template(get_prompt(name).body).substitute(values)
