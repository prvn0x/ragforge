"""Cross-builder seam (final review P0): a REAL runner's answer rows, labelled correct by the
owner, must feed the REAL D3 builder. Before the fix, rows recorded shown notes without stable
chunk keys and every one was skipped ("no_shown_keys"), so no private verifier set could exist."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

from test_ev_run_fakes import FakeEvalClient, load_questions
from test_ev_run_runners import make_ctx

from ragforge_client.evals import datasets
from ragforge_client.evals.runfile import read_run
from ragforge_client.evals.runners import run_application
from ragforge_client.evals.schema import AnswerOutput, Label, LabelKind


def test_runner_rows_carry_chunk_keys_and_build_a_private_verifier_set(tmp_path: Path) -> None:
    client = FakeEvalClient()
    qs = load_questions()
    path = asyncio.run(run_application(make_ctx(tmp_path, client), qs))
    run = read_run(path)
    answered = [
        r for r in run.rows if isinstance(r.output, AnswerOutput) and not r.output.abstained
    ]
    assert answered
    assert all(s.chunk_key is not None for r in answered for s in r.shown)

    by_id = {q.id: q for q in qs.questions}
    labels: list[Label] = []
    for row in answered:
        assert isinstance(row.output, AnswerOutput)
        ids = [UUID(s.chunk_id) for s in row.shown]
        chunks = asyncio.run(client.eval_chunks(ids)).chunks
        notes = tuple(c.display_text for c in chunks)
        q = by_id[row.item_id]
        digest = datasets.item_digest_of(
            q.question, row.output.answer, notes, [f.fact for f in q.facts]
        )
        labels.append(
            Label(item_digest=digest, item_id=row.item_id, kind=LabelKind.answer, correct=True,
                  invented=False, batch_id="b1", labelled_at=datetime.now(UTC))
        )  # fmt: skip
    built = asyncio.run(datasets.build_d3([run], qs, labels, client))
    assert "no_shown_keys" not in built.skipped
    assert built.verifier.items, built.skipped
