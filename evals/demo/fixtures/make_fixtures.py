"""Golden offline fixtures for the eval layer (ADR-0017 §2 D0, r2 P1-6).

    uv run python evals/demo/fixtures/make_fixtures.py

Writes deterministic, model-free run files (run file v2) for every eval kind on the D0 demo
set, plus synthetic owner labels and judge results. B2 (metrics/stats) and B4 (the Eval page)
build and test against these with no model and no stack. The EXPECTED numbers come from
`reference_metrics.py`, an independent stdlib-only script that never imports B2's code.

The outputs are fabricated on purpose to exercise every branch: hits at varying ranks and misses,
clean items rejected by coverage and by the verifier, planted errors caught by each layer, one
caught-but-unlocated and one missed, tied and non-monotone interview scores, judge disagreement.
"""

from __future__ import annotations

import json
import tomllib
from datetime import UTC, datetime
from pathlib import Path

from ragforge_client.evals.schema import (
    AnswerOutput,
    AnswerTier,
    Criterion,
    DatasetRef,
    EvalKind,
    GradeOutput,
    InterviewSet,
    JudgeKey,
    JudgeResult,
    Label,
    LabelKind,
    QuestionSet,
    RetrievalConfig,
    RetrievalOutput,
    RunFooter,
    RunHeader,
    RunRow,
    Span,
    Split,
    Surface,
    Timings,
    Tokens,
    VerifierSet,
    VerifyOutput,
    digest,
)

HERE = Path(__file__).resolve().parent
DEMO = HERE.parent
T0 = datetime(2026, 10, 21, 12, 0, tzinfo=UTC)
LEVELS = ["sde_1", "sde_2", "senior_sde_1", "senior_sde_2", "staff"]
RETRIEVAL = RetrievalConfig(
    embed_model="nomic-embed-text",
    space_id=1,
    chunker_version="heading-v1",
    tau=0.35,
    rerank_model="ms-marco-MiniLM-L-12-v2",
    rerank_top=20,
)
JUDGE = JudgeKey(
    provider="ollama",
    model_version="qwen3:8b-q4_K_M",
    decoding={"temperature": 0, "num_ctx": 8192},
    prompt="judge_qa@1",
)


def header(kind: EvalKind, surface: Surface, dataset: str, data: object) -> RunHeader:
    return RunHeader(
        run_id=f"golden-{kind.value}",
        eval_kind=kind,
        surface=surface,
        arm="golden",
        started_at=T0,
        dataset=DatasetRef(name=dataset, digest=digest(data), splits=[Split.dev]),
        retrieval=RETRIEVAL,
        chat_model="fake-chat",
        verifier_model="fake-verifier",
        prompt_versions={"chat": "answer@11", "verifier": "ground_check@3"},
        judge=JUDGE if kind is EvalKind.generator else None,
    )


def write_run(name: str, head: RunHeader, rows: list[RunRow]) -> None:
    lines = [head.model_dump_json(), *(r.model_dump_json() for r in rows)]
    lines.append(RunFooter(finished_at=T0, rows=len(rows)).model_dump_json())
    (HERE / name).write_text("\n".join(lines) + "\n")


# ------------------------------------------------------------------ E-R
def ranked(relevant: list[str], others: list[str], position: int | None) -> list[str]:
    """The primary page at `position` (1-based; None = absent); other relevant pages after it."""
    lst = others[:12]
    if position is not None:
        lst = [*lst[: position - 1], relevant[0], *lst[position - 1 :]]
    for extra in relevant[1:]:
        lst.insert(min(len(lst), (position or 3) + 1), extra)
    return lst


def retrieval(qs: QuestionSet) -> None:
    pages = sorted({k for q in qs.questions for k in q.relevance} | {"file:filler.md"})
    rows: list[RunRow] = []
    for i, q in enumerate(qs.questions):
        if not q.answerable:
            # unanswerable: nothing relevant; the top score alternates around tau
            noise = [p for p in pages if p != "file:filler.md"][:5]
            out = RetrievalOutput(
                stages={s: noise for s in ("keyword", "vector", "fused", "reranked")},
                top_rerank_score=0.2 if i % 2 else 0.5,
            )
        else:
            relevant = sorted(q.relevance, key=lambda k: -q.relevance[k])
            others = [p for p in pages if p not in relevant]

            out = RetrievalOutput(
                stages={
                    "keyword": ranked(relevant, others, None if i % 5 == 0 else (i % 7) + 1),
                    "vector": ranked(relevant, others, (i % 4) + 2),
                    "fused": ranked(relevant, others, (i % 3) + 1),
                    "reranked": ranked(relevant, others, 1 if i % 4 else 6),
                }
            )
        rows.append(
            RunRow(item_id=q.id, input_digest=digest(q.model_dump(mode="json")), output=out)
        )
    write_run(
        "run-retrieval.jsonl",
        header(EvalKind.retrieval, Surface.chat, "demo", qs.model_dump(mode="json")),
        rows,
    )


# ------------------------------------------------------------------ E-V
def verifier(vs: VerifierSet) -> None:
    rows: list[RunRow] = []
    for item in vs.items:
        if item.mutation is None:
            if item.id == "c-deploy":  # coverage-caused false reject (code fence lead-in)
                out = VerifyOutput(
                    status="partial",
                    verifier_ran=True,
                    verifier_objected=False,
                    coverage_spans=[Span(start=0, end=10)],
                    verifier_spans=[],
                    unlocated_verifier_claims=0,
                )
            elif item.id == "c-breaker":  # verifier-caused false reject
                out = VerifyOutput(
                    status="partial",
                    verifier_ran=True,
                    verifier_objected=True,
                    coverage_spans=[],
                    verifier_spans=[Span(start=0, end=20)],
                    unlocated_verifier_claims=0,
                )
            else:
                out = VerifyOutput(
                    status="verified",
                    verifier_ran=True,
                    verifier_objected=False,
                    coverage_spans=[],
                    verifier_spans=[],
                    unlocated_verifier_claims=0,
                )
        else:
            span = item.span
            assert span is not None
            hit = Span(start=span.start, end=span.end)
            if item.id == "p-entity-2":  # caught, unlocated: badge flipped, no overlap
                out = VerifyOutput(
                    status="partial",
                    verifier_ran=True,
                    verifier_objected=True,
                    coverage_spans=[],
                    verifier_spans=[],
                    unlocated_verifier_claims=1,
                )
            elif item.id == "p-negation-1":  # missed: still verified
                out = VerifyOutput(
                    status="verified",
                    verifier_ran=True,
                    verifier_objected=False,
                    coverage_spans=[],
                    verifier_spans=[],
                    unlocated_verifier_claims=0,
                )
            elif item.id == "p-moved-1":  # caught by both layers
                out = VerifyOutput(
                    status="partial",
                    verifier_ran=True,
                    verifier_objected=True,
                    coverage_spans=[hit],
                    verifier_spans=[hit],
                    unlocated_verifier_claims=0,
                )
            elif item.id == "p-number-2":  # flagged, but elsewhere: NOT a span catch
                out = VerifyOutput(
                    status="partial",
                    verifier_ran=True,
                    verifier_objected=True,
                    coverage_spans=[],
                    verifier_spans=[Span(start=0, end=3)],
                    unlocated_verifier_claims=0,
                )
            elif item.expected_layer and item.expected_layer.value == "coverage":
                out = VerifyOutput(
                    status="partial",
                    verifier_ran=False,
                    verifier_objected=False,
                    coverage_spans=[hit],
                    verifier_spans=[],
                    unlocated_verifier_claims=0,
                )
            else:
                out = VerifyOutput(
                    status="partial",
                    verifier_ran=True,
                    verifier_objected=True,
                    coverage_spans=[],
                    verifier_spans=[hit],
                    unlocated_verifier_claims=0,
                )
        rows.append(
            RunRow(item_id=item.id, input_digest=digest(item.model_dump(mode="json")), output=out)
        )
    write_run(
        "run-verifier.jsonl",
        header(EvalKind.verifier, Surface.verifier, "demo", vs.model_dump(mode="json")),
        rows,
    )


# ------------------------------------------------------------------ E-I
BASE = {AnswerTier.gold: 9, AnswerTier.partial: 5, AnswerTier.wrong: 1, AnswerTier.misconception: 6}


def interview(iv: InterviewSet) -> None:
    rows: list[RunRow] = []
    for si, seed in enumerate(iv.seeds):
        for tier in AnswerTier:
            for li, level in enumerate(LEVELS):
                if (
                    tier in (AnswerTier.wrong, AnswerTier.misconception)
                    and level != seed.seed_level
                ):
                    continue  # the level sweep covers gold and partial only (ADR-0017 §3)
                for rep in (0, 1):
                    if (
                        level != seed.seed_level
                        and rep == 1
                        and tier is AnswerTier.partial
                        and si == 3
                    ):
                        continue  # one missing replicate: the reference must cope
                    drift = (li - 2) * (1 if si % 2 == 0 else 0)  # seeds 0, 2 trend; 1, 3 flat
                    noise = (si + li + rep) % 3 - 1
                    score = max(0, min(10, BASE[tier] - drift + noise))
                    if (
                        seed.id == "s-ratelimit"
                        and tier is AnswerTier.partial
                        and level == seed.seed_level
                    ):
                        # exactly the gold score: a TIE, which ranking accuracy counts as wrong
                        score = max(0, min(10, BASE[AnswerTier.gold] - drift + noise))
                    given = (
                        [seed.misconception_sentence] if tier is AnswerTier.misconception else []
                    )
                    if tier is AnswerTier.misconception and si == 1:
                        given = ["consumers never see duplicates"]  # partial quote, still contained
                    kept = given if si != 2 else []  # seed 2: given but dropped by the server
                    out = GradeOutput(
                        seed_id=seed.id,
                        tier=tier,
                        level=level,
                        replicate=rep,
                        score=score,
                        misconceptions_given=given,
                        misconceptions_kept=kept,
                    )
                    rows.append(
                        RunRow(
                            item_id=f"{seed.id}/{tier.value}/{level}/{rep}",
                            input_digest=digest([seed.id, tier.value, level, rep]),
                            output=out,
                        )
                    )
    write_run(
        "run-interview.jsonl",
        header(EvalKind.interview, Surface.interview, "demo", iv.model_dump(mode="json")),
        rows,
    )


# ------------------------------------------------------------------ E-G + judge + labels
def generator(qs: QuestionSet) -> None:
    rows: list[RunRow] = []
    judged: list[JudgeResult] = []
    labels: list[Label] = []
    for i, q in enumerate(qs.questions):
        abstained = not q.answerable or i == 3
        out = AnswerOutput(
            answer="Your notes don't cover this." if abstained else f"Answer to {q.id} [S1].",
            status="unverified" if abstained else ("verified" if i % 3 else "partial"),
            cited_pages=[] if abstained else sorted(q.relevance)[:1],
            abstained=abstained,
        )
        item_digest = digest({"q": q.id, "answer": out.answer})
        refs: list[str] = []
        if not abstained:
            for c_i, crit in enumerate(
                (Criterion.relevance, Criterion.completeness, Criterion.faithfulness)
            ):
                judge_score = (i + c_i) % 3
                owner_score = judge_score if (i + c_i) % 4 else min(2, judge_score + 1)
                key = digest(
                    {
                        "judge": JUDGE.model_dump(mode="json"),
                        "item": item_digest,
                        "crit": crit.value,
                    }
                )
                refs.append(key)
                judged.append(
                    JudgeResult(
                        cache_key=key,
                        item_digest=item_digest,
                        criterion=crit,
                        score=judge_score,
                        reason="fixture",
                        judge=JUDGE,
                        scored_at=T0,
                    )
                )
                if c_i == 0:
                    labels.append(
                        Label(
                            item_digest=item_digest,
                            item_id=q.id,
                            kind=LabelKind.criteria,
                            scores={},
                            batch_id="b1",
                            labelled_at=T0,
                        )
                    )
                labels[-1] = labels[-1].model_copy(
                    update={"scores": {**labels[-1].scores, crit: owner_score}}
                )
        rows.append(
            RunRow(
                item_id=q.id,
                input_digest=digest(q.model_dump(mode="json")),
                output=out,
                judge_refs=refs,
                timings=Timings(ttft_s=1.0 + i / 10, total_s=2.0 + i / 5),
                tokens=Tokens(prompt=900 + i, completion=120 + i),
            )
        )
    write_run(
        "run-generator.jsonl",
        header(EvalKind.generator, Surface.chat, "demo", qs.model_dump(mode="json")),
        rows,
    )
    (HERE / "judge-results.jsonl").write_text("\n".join(j.model_dump_json() for j in judged) + "\n")
    (HERE / "labels.jsonl").write_text("\n".join(lb.model_dump_json() for lb in labels) + "\n")


def main() -> None:
    qs = QuestionSet.model_validate(tomllib.load((DEMO / "questions.toml").open("rb")))
    vs = VerifierSet.model_validate(json.loads((DEMO / "verifier.json").read_text()))
    iv = InterviewSet.model_validate(json.loads((DEMO / "interview.json").read_text()))
    retrieval(qs)
    verifier(vs)
    interview(iv)
    generator(qs)


if __name__ == "__main__":
    main()
