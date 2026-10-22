"""INDEPENDENT reference implementation of the ADR-0017 metric definitions (the golden oracle).

    python3 evals/demo/fixtures/reference_metrics.py      # writes expected_summaries.json

Standard library ONLY, and it never imports ragforge_client.evals: B2's metric code must
reproduce these numbers to 1e-9, so an oracle written with B2's code would prove nothing.
It is also the precise, executable definition of every metric NAME the Eval page and the
gate use. Point values only: intervals are B2's (their methods are named in ADR-0017).
"""

from __future__ import annotations

import json
import math
import re
import tomllib
from collections import defaultdict
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
DEMO = HERE.parent
LEVELS = ["sde_1", "sde_2", "senior_sde_1", "senior_sde_2", "staff"]
STAGES = ["keyword", "vector", "fused", "reranked"]


def rows(name: str) -> list[dict[str, Any]]:
    lines = [json.loads(x) for x in (HERE / name).read_text().splitlines() if x.strip()]
    return [r for r in lines if r["type"] == "row"]


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


# ------------------------------------------------------------------ E-R
def retrieval() -> dict[str, float | None]:
    """Page level. need=any: hit@k = any relevant page in the top k; MRR = 1/rank of the first
    relevant page (0 if none). need=all: recall@k = |relevant in top k| / |relevant|. All
    answerable: nDCG@10 with gain 2^rel - 1 and discount log2(rank + 1); the ideal ordering
    sorts every relevance grade of the question. fused_recall@20: |relevant in fused top 20| /
    |relevant|, over all answerable questions. Unanswerable: share whose top reranked score is
    below tau (0.35 in the fixture header)."""
    qs = {q["id"]: q for q in tomllib.load((DEMO / "questions.toml").open("rb"))["questions"]}
    out: dict[str, float | None] = {}
    acc: dict[str, list[float]] = defaultdict(list)
    below_tau: list[float] = []
    for r in rows("run-retrieval.jsonl"):
        q = qs[r["item_id"]]
        st = r["output"]["stages"]
        if not q.get("answerable", True):
            below_tau.append(1.0 if r["output"]["top_rerank_score"] < 0.35 else 0.0)
            continue
        rel = q["relevance"]
        need = q.get("need", "any")
        for stage in STAGES:
            ranked = st[stage]
            first = next((i + 1 for i, p in enumerate(ranked) if p in rel), None)
            if need == "any":
                for k in (5, 10):
                    acc[f"retrieval.{stage}.hit@{k}"].append(1.0 if first and first <= k else 0.0)
                acc[f"retrieval.{stage}.mrr"].append(1.0 / first if first else 0.0)
            else:
                for k in (5, 10):
                    got = len({p for p in ranked[:k] if p in rel})
                    acc[f"retrieval.{stage}.recall@{k}"].append(got / len(rel))
            dcg = sum(
                (2 ** rel.get(p, 0) - 1) / math.log2(i + 2) for i, p in enumerate(ranked[:10])
            )
            ideal = sorted(rel.values(), reverse=True)[:10]
            idcg = sum((2**g - 1) / math.log2(i + 2) for i, g in enumerate(ideal))
            acc[f"retrieval.{stage}.ndcg@10"].append(dcg / idcg if idcg else 0.0)
        fused20 = len({p for p in st["fused"][:20] if p in rel})
        acc["retrieval.fused.recall@20"].append(fused20 / len(rel))
    for name, vals in sorted(acc.items()):
        out[name] = mean(vals)
    out["retrieval.unanswerable.below_tau"] = mean(below_tau)
    return out


# ------------------------------------------------------------------ E-V
def overlaps(a: dict[str, int], b: dict[str, int]) -> bool:
    return bool(a["start"] < b["end"] and b["start"] < a["end"])


def verifier() -> dict[str, float | None]:
    """caught (span) = a coverage span or a located verifier span overlaps the mutation span.
    caught_by: coverage only / verifier only / both. caught_unlocated = not span-caught, status
    not verified, verifier ran with an objection and >= 1 unlocated claim; counted ONLY on
    sources whose clean item is verified in this run, as is badge_flip_recall (= planted items
    on those sources whose status is not verified). false_reject = clean items not verified;
    verifier-caused = the verifier objected (located or not, whatever coverage did);
    coverage-caused = not verified and the verifier did not object."""
    items = {i["id"]: i for i in json.loads((DEMO / "verifier.json").read_text())["items"]}
    out_by = {r["item_id"]: r["output"] for r in rows("run-verifier.jsonl")}
    clean = [i for i in items.values() if i["mutation"] is None]
    planted = [i for i in items.values() if i["mutation"] is not None]
    clean_verified = {i["id"] for i in clean if out_by[i["id"]]["status"] == "verified"}
    span_caught: list[float] = []
    by_layer: dict[str, int] = defaultdict(int)
    flips, unlocated = [], []
    per_mutation: dict[str, list[float]] = defaultdict(list)
    for i in planted:
        o, span = out_by[i["id"]], i["span"]
        cov = any(overlaps(s, span) for s in o["coverage_spans"])
        ver = any(overlaps(s, span) for s in o["verifier_spans"])
        caught = cov or ver
        span_caught.append(1.0 if caught else 0.0)
        per_mutation[i["mutation"]].append(1.0 if caught else 0.0)
        by_layer[
            "both" if cov and ver else "coverage" if cov else "verifier" if ver else "missed"
        ] += 1
        if i["source_id"] in clean_verified:
            flipped = o["status"] != "verified"
            flips.append(1.0 if flipped else 0.0)
            unloc = (
                not caught
                and flipped
                and o["verifier_ran"]
                and o["verifier_objected"]
                and o["unlocated_verifier_claims"] > 0
            )
            unlocated.append(1.0 if unloc else 0.0)
    out: dict[str, float | None] = {
        "verifier.span_recall": mean(span_caught),
        "verifier.badge_flip_recall": mean(flips),
        "verifier.caught_unlocated": mean(unlocated),
        "verifier.false_reject": mean(
            [0.0 if out_by[i["id"]]["status"] == "verified" else 1.0 for i in clean]
        ),
        "verifier.false_reject.verifier_caused": mean(
            [
                1.0
                if out_by[i["id"]]["verifier_objected"] and out_by[i["id"]]["status"] != "verified"
                else 0.0
                for i in clean
            ]
        ),
        "verifier.false_reject.coverage_caused": mean(
            [
                1.0
                if not out_by[i["id"]]["verifier_objected"]
                and out_by[i["id"]]["status"] != "verified"
                else 0.0
                for i in clean
            ]
        ),
    }
    for layer in ("coverage", "verifier", "both", "missed"):
        out[f"verifier.caught_by.{layer}"] = by_layer[layer] / len(planted)
    for mutation, vals in sorted(per_mutation.items()):
        out[f"verifier.span_recall.{mutation}"] = mean(vals)
    return out


# ------------------------------------------------------------------ E-I
def norm(text: str) -> str:
    return " ".join(re.sub(r"[^\w\s]", " ", text.casefold()).split())


def contains(planted: str, quote: str) -> bool:
    """Normalised containment either way; a quote shorter than 12 normalised chars never counts."""
    p, q = norm(planted), norm(quote)
    return len(q) >= 12 and (q in p or p in q)


def interview() -> dict[str, float | None]:
    """At the seed's own level, replicate 0: ranking accuracy over the pairs gold>partial,
    partial>wrong, gold>wrong, gold>misconception (strictly greater; a tie is wrong);
    detection given/kept = the planted sentence is contained in a misconception quote; bands:
    gold >= 7, wrong <= 3. Level sweep (gold + partial): the cell = mean of its replicates;
    Page's L per the levels ordered staff -> sde_1 (the predicted increasing order), ranks within
    each item block averaged on ties, L = sum(k * R_k); test-retest SD = sqrt(mean((r0 - r1)^2)
    / 2) over cells with both replicates."""
    seeds = {s["id"]: s for s in json.loads((DEMO / "interview.json").read_text())["seeds"]}
    grades = [r["output"] for r in rows("run-interview.jsonl")]
    at_seed = {
        (g["seed_id"], g["tier"]): g
        for g in grades
        if g["level"] == seeds[g["seed_id"]]["seed_level"] and g["replicate"] == 0
    }
    pairs = [
        ("gold", "partial"),
        ("partial", "wrong"),
        ("gold", "wrong"),
        ("gold", "misconception"),
    ]
    ranking = [
        1.0 if at_seed[(s, a)]["score"] > at_seed[(s, b)]["score"] else 0.0
        for s in seeds
        for a, b in pairs
    ]
    given, kept = [], []
    for s, seed in seeds.items():
        g = at_seed[(s, "misconception")]
        given.append(
            1.0
            if any(contains(seed["misconception_sentence"], q) for q in g["misconceptions_given"])
            else 0.0
        )
        kept.append(
            1.0
            if any(contains(seed["misconception_sentence"], q) for q in g["misconceptions_kept"])
            else 0.0
        )
    cells: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    for g in grades:
        if g["tier"] in ("gold", "partial"):
            cells[(g["seed_id"], g["tier"], g["level"])].append(g["score"])
    order = list(reversed(LEVELS))  # staff first: predicted lowest score
    total_l = 0.0
    for s in seeds:
        for tier in ("gold", "partial"):
            vals = [mean([float(x) for x in cells[(s, tier, lv)]]) for lv in order]
            ranks = average_ranks([v for v in vals if v is not None])
            total_l += sum((k + 1) * r for k, r in enumerate(ranks))
    sq = [float((c[0] - c[1]) ** 2) for c in cells.values() if len(c) == 2]
    sq_mean = mean(sq)
    return {
        "interview.ranking_accuracy": mean(ranking),
        "interview.detection_given": mean(given),
        "interview.detection_kept": mean(kept),
        "interview.band.gold_ge_7": mean(
            [1.0 if at_seed[(s, "gold")]["score"] >= 7 else 0.0 for s in seeds]
        ),
        "interview.band.wrong_le_3": mean(
            [1.0 if at_seed[(s, "wrong")]["score"] <= 3 else 0.0 for s in seeds]
        ),
        "interview.level.page_l": total_l,
        "interview.level.retest_sd": math.sqrt(sq_mean / 2) if sq_mean is not None else None,
    }


def average_ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


# ------------------------------------------------------------------ E-G + calibration
def generator() -> dict[str, float | None]:
    """Judge scores on ANSWERED rows only, mean per criterion (0..2), next to the answer rate.
    Calibration per criterion over items the owner labelled: raw agreement; linear-weighted kappa
    with weights w_ij = 1 - |i - j| / 2 over the 3x3 table: (Po - Pe) / (1 - Pe)."""
    rs = rows("run-generator.jsonl")
    judge = [json.loads(x) for x in (HERE / "judge-results.jsonl").read_text().splitlines() if x]
    labels = [json.loads(x) for x in (HERE / "labels.jsonl").read_text().splitlines() if x]
    answered = [r for r in rs if not r["output"]["abstained"]]
    by_key = {j["cache_key"]: j for j in judge}
    out: dict[str, float | None] = {"generator.answer_rate": len(answered) / len(rs)}
    owner = {lb["item_digest"]: lb["scores"] for lb in labels}
    for crit in ("relevance", "completeness", "faithfulness"):
        scores = [
            by_key[k]["score"]
            for r in answered
            for k in r["judge_refs"]
            if by_key[k]["criterion"] == crit
        ]
        out[f"generator.judge.{crit}"] = mean([float(s) for s in scores])
        pairs = [
            (owner[j["item_digest"]][crit], j["score"])
            for j in judge
            if j["criterion"] == crit
            and j["item_digest"] in owner
            and crit in owner[j["item_digest"]]
        ]
        out[f"calibration.{crit}.raw_agreement"] = mean([1.0 if a == b else 0.0 for a, b in pairs])
        out[f"calibration.{crit}.kappa_linear"] = weighted_kappa(pairs)
    return out


def weighted_kappa(pairs: list[tuple[int, int]]) -> float | None:
    n = len(pairs)
    if not n:
        return None
    table = [[0] * 3 for _ in range(3)]
    for a, b in pairs:
        table[a][b] += 1
    w = [[1 - abs(i - j) / 2 for j in range(3)] for i in range(3)]
    rows_ = [sum(table[i]) / n for i in range(3)]
    cols = [sum(table[i][j] for i in range(3)) / n for j in range(3)]
    po = sum(w[i][j] * table[i][j] / n for i in range(3) for j in range(3))
    pe = sum(w[i][j] * rows_[i] * cols[j] for i in range(3) for j in range(3))
    return None if pe == 1 else (po - pe) / (1 - pe)


def main() -> None:
    expected = {**retrieval(), **verifier(), **interview(), **generator()}
    (HERE / "expected_summaries.json").write_text(
        json.dumps(expected, indent=2, sort_keys=True) + "\n"
    )
    print(f"{len(expected)} expected metric values written")


if __name__ == "__main__":
    main()
