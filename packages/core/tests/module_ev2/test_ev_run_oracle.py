"""EV-RUN end to end: a run written by the runners from the replayed fixtures, summarized by the
real `metrics.summarize`, gives the independent oracle's numbers (expected_summaries.json).

This closes the loop the golden fixtures exist for: if a runner reduced a server response
differently from the fixture (a page order, a span, an objection), these values would move.
E-I and E-G are covered row by row in test_ev_run_runners.py instead: the runner grades every
replicate the fixture dropped, and its judge is a fake.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from test_ev_run_fakes import (
    FIXTURES,
    FakeEvalClient,
    demo_datasets,
    load_questions,
    load_verifier,
)
from test_ev_run_runners import make_ctx

from ragforge_client.evals import metrics
from ragforge_client.evals.runfile import read_run
from ragforge_client.evals.runners import run_retrieval, run_verifier

EXPECTED: dict[str, float | None] = json.loads((FIXTURES / "expected_summaries.json").read_text())


@pytest.mark.parametrize("kind", ["retrieval", "verifier"])
async def test_runner_output_reproduces_the_oracle(tmp_path: Path, kind: str) -> None:
    ctx = make_ctx(tmp_path, FakeEvalClient())
    if kind == "retrieval":
        path = await run_retrieval(ctx, load_questions())
    else:
        path = await run_verifier(ctx, load_verifier())
    summary = metrics.summarize(read_run(path), demo_datasets(), [], [])
    got = {m.name: m.value for m in summary.metrics}
    expected = {k: v for k, v in EXPECTED.items() if k.startswith(f"{kind}.")}
    assert expected
    assert set(expected) <= set(got)
    for name, value in expected.items():
        assert got[name] == pytest.approx(value, abs=1e-9), name
    assert summary.stale_items == 0
    assert summary.errors_ours == 0
