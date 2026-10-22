"""Verification stays CPU-cheap on degenerate model output (ADR-0015 section 6).

Review of the redesign, round 2 (P1): three steps on the verification path were super-linear
(inline-code masking was cubic in a backtick run, the product-name check before "(S<n>)" and
the stray-parenthesis scan quadratic), and they run synchronously inside async graph nodes. A
small model stuck in a repetition loop could block the shared event loop for seconds. These
inputs are 20,000 characters, about 5x the longest draft `answer@11` produces; before the fix
the slowest took 4.5 to 10 s.

Review of the redesign, round 4: the structure now comes from a markdown-it-py parse, cached per
text. The stages are timed COLD and in production order (normalize parses the raw draft and its
normalised form, the later stages reuse that parse), so the cache cannot hide a slow parse;
markdown-it-py's own worst shapes ("[" runs, emphasis and link floods, nested containers) are
in the set.
"""

from __future__ import annotations

import time
from itertools import pairwise

import pytest

from module_ag.ag_hits import HITS, SHOWN, VALID
from ragforge_core.agent import coverage, structure
from ragforge_core.agent.attribution import attribute
from ragforge_core.agent.citations import check_citations, normalize_citations
from ragforge_core.agent.coverage import residue

N = 20_000
# the three shapes the review timed, each one stage at most this long
REVIEWED = {
    "backtick run": "`" * N,
    "capitalised run before (S1)": "Aa " * (N // 3) + "b (S1).",
    "unclosed label parenthesis": "(" + "S1-" * (N // 3),
}
STAGE_BUDGET_S = 0.05
# every other shape a regex or a per-unit loop could choke on: the whole pipeline
OTHERS = {
    "backticks with spaces": "`` ` " * (N // 5),
    "backtick lines": "`\n" * (N // 2),
    "bare label sentences": "(S1). " * (N // 6),
    "label ranges": "[S1-S3] " * (N // 8),
    "unclosed brackets": "[S1-" * (N // 4),
    "unclosed comments": "<!--" * (N // 4),
    "pipes": "|" * N,
    "table cells": "| a " * (N // 4) + "|",
    "html breaks": "<br>" * (N // 4),
    "dotted lower case": "ab.cd" * (N // 5),
    "source cues": ". Source S1" * (N // 11),
    "fence lines": "```\n" * (N // 4),
    "repeated sentence": "Kafka stores records in partitions. " * (N // 36),
    "cjk sentences": "它。" * (N // 2),
    "newlines": "\n" * N,
    # markdown-it-py's own worst shapes (round 4: the parse is on the path now)
    "open brackets": "[" * N,
    "emphasis flood": "**a " * (N // 4),
    "unclosed links": "[a](b " * (N // 6),
    "nested quotes": "> " * (N // 2) + "x",
    "nested lists": "- " * (N // 2) + "x",
    "code spans": "`a` " * (N // 4),
    "escapes": "\\[S1\\] " * (N // 8),
}
PIPELINE_BUDGET_S = 1.0


def _cold() -> None:
    for cached in (structure.structure, coverage.segment, coverage.visible_runs):
        cached.cache_clear()


def _stages(answer: str) -> dict[str, float]:
    """Each stage's best-of-three time, every run starting from EMPTY caches and running the
    stages in production order (nodes.verify_citations, then ground_check)."""
    best = dict.fromkeys(("normalize", "check", "attribute", "residue"), float("inf"))
    for _ in range(3):
        _cold()
        times = [time.perf_counter()]
        draft = normalize_citations(answer)
        times.append(time.perf_counter())
        check_citations(draft, SHOWN, strict=True)
        times.append(time.perf_counter())
        found = attribute(draft, HITS, SHOWN, VALID)
        times.append(time.perf_counter())
        residue(draft, VALID, [i.sentence for i in found.inferred])
        times.append(time.perf_counter())
        for stage, (a, b) in zip(best, pairwise(times), strict=True):
            best[stage] = min(best[stage], b - a)
    return best


@pytest.mark.parametrize("answer", REVIEWED.values(), ids=REVIEWED.keys())
def test_the_reviewed_degenerate_inputs_take_milliseconds_per_stage(answer: str) -> None:
    slow = {k: round(v, 3) for k, v in _stages(answer).items() if v > STAGE_BUDGET_S}
    assert slow == {}


@pytest.mark.parametrize("answer", OTHERS.values(), ids=OTHERS.keys())
def test_no_degenerate_input_is_super_linear(answer: str) -> None:
    assert sum(_stages(answer).values()) < PIPELINE_BUDGET_S
