"""Convert a Phase 2b (v1) question file to the ADR-0017 v2 schema.

    uv run python evals/tools/convert_v1_questions.py IN.toml OUT.toml --titles titles.json

- `pages` (titles) stays, for citation grading; `relevance` maps each title to its stable page
  key "notion:<page id>" (grade 1 each: v1 listed "any of these", not an order; `need = "any"`).
- The burned `heldout` split becomes `dev` (ADR-0017 D1: it was revealed twice).
- `titles.json` is {title: page_key}, read from the SNAPSHOT (the eval tenant), so the keys are
  the ones the runs resolve. A title with no page is reported and the run refuses to write.
Writes OUT.toml (0600) and never prints question text.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tomllib
from pathlib import Path

from ragforge_client.evals.schema import QuestionSet


def _q(s: str) -> str:
    return json.dumps(s, ensure_ascii=False)  # a TOML basic string is a JSON string


def convert(v1: dict, titles: dict[str, str]) -> tuple[str, list[str]]:
    missing: list[str] = []
    out = ["schema_version = 2", 'name = "private"', ""]
    for q in v1["q"]:
        answerable = q.get("answerable", True)
        out.append("[[questions]]")
        out.append(f"id = {_q(q['id'])}")
        out.append(f"split = {_q('dev' if q.get('split') == 'heldout' else q.get('split', 'dev'))}")
        out.append(f"question = {_q(q['question'])}")
        if not answerable:
            out.append("answerable = false")
            out.append("")
            continue
        pages = q.get("pages", [])
        out.append("pages = [" + ", ".join(_q(p) for p in pages) + "]")
        rel = {}
        for title in pages:
            key = titles.get(title)
            if key is None:
                missing.append(f"{q['id']}: {title}")
            else:
                rel[key] = 1
        out.append("relevance = { " + ", ".join(f"{_q(k)} = 1" for k in rel) + " }")
        out.append('need = "any"')
        facts = [
            "{ fact = " + _q(f["fact"]) + ", match = [" + ", ".join(_q(m) for m in f["match"]) + "] }"
            for f in q.get("facts", [])
        ]
        out.append("facts = [\n  " + ",\n  ".join(facts) + ",\n]" if facts else "facts = []")
        out.append("")
    return "\n".join(out) + "\n", missing


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("input")
    p.add_argument("output")
    p.add_argument("--titles", required=True)
    a = p.parse_args()
    v1 = tomllib.loads(Path(a.input).read_text())
    titles = json.loads(Path(a.titles).read_text())
    text, missing = convert(v1, titles)
    if missing:
        print(f"{len(missing)} title(s) not in the snapshot:", *missing, sep="\n  ", file=sys.stderr)
        return 1
    QuestionSet.model_validate(tomllib.loads(text))  # the v2 schema accepts it
    fd = os.open(a.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(text)
    n = len(v1["q"])
    print(f"converted {n} questions -> {a.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
