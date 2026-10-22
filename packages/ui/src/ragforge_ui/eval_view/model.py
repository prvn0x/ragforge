"""Pure view models of the Eval page: what to show, as plain values. No Streamlit, no I/O.

The page COMPUTES NO METRIC. Every number on it comes from a `RunSummary`, a `Calibration` or a
`GateVerdict` written by the eval layer (metrics/stats/gate are the definitions; see
`evals/demo/fixtures/reference_metrics.py`). What lives here is presentation and policy:
- formatting, and principle 1: a judge-derived number whose judge is not trusted is never shown
  as a result;
- held-out protection: `heldout2` rows are hidden unless revealed, and FAIL CLOSED (a row whose
  split can't be established in a run that declares `heldout2` counts as held out);
- the compare rules (same eval kind, dataset and snapshot content digest) and per-item flips,
  whose p-values are labelled exploratory (ADR-0017 §5: the gate is the decision, not the page);
- the blind label queue items, which by construction carry no judge score, reason or verdict.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime

from ragforge_client.evals.datasets import (
    CALIBRATION_N,
    calibration_candidates,
    calibration_sample,
)
from ragforge_client.evals.interfaces import Datasets
from ragforge_client.evals.runfile import RunFile
from ragforge_client.evals.schema import (
    E_G_CRITERIA,
    AnswerOutput,
    Calibration,
    Criterion,
    EvalKind,
    GradeOutput,
    JudgeKey,
    JudgeResult,
    Label,
    LabelKind,
    MetricValue,
    Need,
    Question,
    QuestionSet,
    RetrievalOutput,
    RunHeader,
    RunRow,
    RunSummary,
    ShownRef,
    Span,
    Split,
    VerifierItem,
    VerifyOutput,
)

NONE = "-"
NOT_TRUSTED = "not trusted"
UNCALIBRATED = "uncalibrated"
EXPLORATORY = "exploratory: not a gate decision (ADR-0017 §5)"
NO_TRUSTED_JUDGE = (
    "No judge is trusted. Only the deterministic checks and your own labels count; judge "
    "scores are shown for diagnosis only (ADR-0017 §4)."
)
LEVELS = ("sde_1", "sde_2", "senior_sde_1", "senior_sde_2", "staff")
_QUESTION_KINDS = frozenset(
    {EvalKind.retrieval, EvalKind.generator, EvalKind.application, EvalKind.latency}
)

# The headline metrics per eval kind (names: reference_metrics.py). Others stay in the full table.
HEADLINE: dict[EvalKind, tuple[str, ...]] = {
    EvalKind.retrieval: (
        "retrieval.reranked.hit@5",
        "retrieval.reranked.mrr",
        "retrieval.reranked.ndcg@10",
        "retrieval.fused.recall@20",
        "retrieval.unanswerable.below_tau",
    ),
    EvalKind.generator: (
        "generator.answer_rate",
        "generator.judge.relevance",
        "generator.judge.completeness",
        "generator.judge.faithfulness",
    ),
    EvalKind.verifier: (
        "verifier.badge_flip_recall",
        "verifier.span_recall",
        "verifier.false_reject.verifier_caused",
        "verifier.caught_unlocated",
    ),
    EvalKind.interview: (
        "interview.ranking_accuracy",
        "interview.detection_given",
        "interview.detection_kept",
        "interview.level.retest_sd",
    ),
}

# ADR-0017 §3 E-V SLO, per verifier model (r4 P2: badge-flip recall is what the user sees).
SLO_RECALL_LOW = 0.75
SLO_FALSE_REJECT_HIGH = 0.20


# ------------------------------------------------------------------ formatting
def fmt(value: float | None) -> str:
    """Three significant digits; large values (Page's L) with one decimal, never exponents."""
    if value is None or math.isnan(value):
        return NONE
    return f"{value:.1f}" if abs(value) >= 100 else f"{value:.3g}"


def ci_text(metric: MetricValue) -> str:
    if metric.ci_low is None or metric.ci_high is None:
        return NONE
    return f"[{fmt(metric.ci_low)}, {fmt(metric.ci_high)}]"


CALIBRATION_PREFIX = "calibration."


def hides_value(metric: MetricValue) -> bool:
    """Principle 1: an untrusted judge's RESULT is never shown. A `calibration.*` metric (kappa,
    raw agreement) is not a judge result but the measure of the judge's trust: hiding it for an
    untrusted criterion would hide exactly the numbers that explain why it is untrusted."""
    return metric.trusted is False and not metric.name.startswith(CALIBRATION_PREFIX)


def ci_cell(metric: MetricValue) -> str:
    """Principle 1 for intervals: an untrusted judge's CI pins its number down as surely as the
    number itself, so it is hidden wherever a value is (tiles, tables, captions)."""
    return NONE if hides_value(metric) else ci_text(metric)


def trust_text(metric: MetricValue) -> str:
    if metric.trusted is None:
        return ""  # not judge-derived
    return "trusted" if metric.trusted else NOT_TRUSTED


def value_text(metric: MetricValue) -> str:
    """Principle 1: an untrusted judge number is replaced by its trust state, not shown."""
    if hides_value(metric):
        return NOT_TRUSTED
    return fmt(metric.value)


def metric_row(metric: MetricValue) -> dict[str, str]:
    return {
        "metric": metric.name,
        "value": value_text(metric),
        "95% CI": ci_cell(metric),
        "n": str(metric.n),
        "method": metric.method.value,
        "judge": trust_text(metric),
        "note": metric.note,
    }


def by_name(summary: RunSummary | None) -> dict[str, MetricValue]:
    return {m.name: m for m in summary.metrics} if summary else {}


def headline(summary: RunSummary) -> list[MetricValue]:
    metrics = by_name(summary)
    names = HEADLINE.get(summary.eval_kind, ())
    chosen = [metrics[n] for n in names if n in metrics]
    return chosen or summary.metrics[:4]


def with_prefix(summary: RunSummary | None, prefix: str) -> list[MetricValue]:
    return [m for m in (summary.metrics if summary else []) if m.name.startswith(prefix)]


def k_of_n(metric: MetricValue) -> str:
    """ "15/16" when value x n is a whole count (a k/n proportion), else the value and n."""
    if metric.value is None:
        return f"{NONE} (n {metric.n})"
    k = metric.value * metric.n
    if abs(k - round(k)) < 1e-6:
        return f"{round(k)}/{metric.n}"
    return f"{fmt(metric.value)} (n {metric.n})"


def short(digest_hex: str | None) -> str:
    return digest_hex[:12] if digest_hex else NONE


# ------------------------------------------------------------------ run header strip
def header_facts(header: RunHeader) -> list[tuple[str, str]]:
    """The facts that make a run comparable, in the order an owner checks them."""
    snap = header.snapshot
    facts = [
        ("Surface", header.surface.value),
        ("Kind", header.eval_kind.value),
        ("Arm", header.arm),
        ("Dataset", f"{header.dataset.name} {short(header.dataset.digest)}"),
        ("Splits", ", ".join(s.value for s in header.dataset.splits)),
        ("Snapshot content", short(snap.content_digest) if snap else "demo (no snapshot)"),
        ("Snapshot index", short(snap.index_digest) if snap else NONE),
        ("Chat model", header.chat_model or NONE),
        ("Verifier model", header.verifier_model or NONE),
        ("Judge", judge_text(header.judge) if header.judge else NONE),
        ("Prompts", prompts_text(header.prompt_versions)),
    ]
    if header.retrieval is not None:
        r = header.retrieval
        facts.append(
            ("Retrieval", f"{r.embed_model} · {r.chunker_version} · τ {r.tau} · {r.rerank_model}")
        )
    return facts


def judge_text(judge: JudgeKey) -> str:
    return f"{judge.provider} {judge.model_version} · {judge.prompt}"


def prompts_text(versions: Mapping[str, str]) -> str:
    return ", ".join(f"{s}: {v}" for s, v in sorted(versions.items())) or NONE


# ------------------------------------------------------------------ held-out rows
@dataclass(frozen=True, slots=True)
class Visible:
    rows: list[RunRow]
    hidden: int


def _question_map(datasets: Datasets) -> dict[str, Question]:
    qs = datasets.questions
    return {q.id: q for q in qs.questions} if qs else {}


def row_split(row: RunRow, header: RunHeader, datasets: Datasets) -> Split | None:
    """The split of a row's item, or None when it can't be established from the datasets.

    D3 items are built from Q&A answers, so a verifier item whose id or source id is a held-out
    question is held out, one linked to a dev question is dev, and one linked to NO question is
    unknown (None): `is_heldout` then fails closed in a run that declares heldout2. D4 seeds
    have no split (the interview set has none)."""
    questions = _question_map(datasets)
    if header.eval_kind in _QUESTION_KINDS:
        q = questions.get(row.item_id)
        return q.split if q else None
    if header.eval_kind is EvalKind.verifier:
        return _verifier_split(row.item_id, datasets, questions)
    return Split.dev  # interview: seeds carry no split


def _verifier_split(
    item_id: str, datasets: Datasets, questions: Mapping[str, Question]
) -> Split | None:
    vs = datasets.verifier
    item = next((i for i in vs.items if i.id == item_id), None) if vs else None
    if item is None:
        return None
    linked = [questions[k].split for k in (item.id, item.source_id) if k in questions]
    if Split.heldout2 in linked:
        return Split.heldout2
    return Split.dev if linked else None


def is_heldout(row: RunRow, header: RunHeader, datasets: Datasets) -> bool:
    """FAIL CLOSED: unknown split + a run that declares heldout2 = held out."""
    split = row_split(row, header, datasets)
    if split is None:
        return Split.heldout2 in header.dataset.splits
    return split is Split.heldout2


def visible_rows(run: RunFile, datasets: Datasets, *, reveal: bool) -> Visible:
    if reveal:
        return Visible(list(run.rows), 0)
    shown = [r for r in run.rows if not is_heldout(r, run.header, datasets)]
    return Visible(shown, len(run.rows) - len(shown))


# ------------------------------------------------------------------ compare
def comparability(base: RunHeader, cand: RunHeader) -> str | None:
    """Why two runs can NOT be compared, or None. Mirrors the gate's refusal (§2 D-snap): runs
    on different corpora compare corpora, not code."""
    if base.eval_kind is not cand.eval_kind:
        return f"They are different eval kinds ({base.eval_kind.value} vs {cand.eval_kind.value})."
    if base.dataset.digest != cand.dataset.digest:
        return (
            "They ran on different datasets (digest "
            f"{short(base.dataset.digest)} vs {short(cand.dataset.digest)}), so no item pairs up."
        )
    bs, cs = base.snapshot, cand.snapshot
    if (bs is None) != (cs is None):
        return "One ran on the eval snapshot and the other on the demo set: different corpora."
    if bs is not None and cs is not None and bs.content_digest != cs.content_digest:
        return (
            "They ran on different snapshot content (digest "
            f"{short(bs.content_digest)} vs {short(cs.content_digest)}): the difference would "
            "measure the notes, not the code. Re-run both arms on one snapshot."
        )
    return None


def metric_deltas(base: RunSummary, cand: RunSummary) -> list[dict[str, str]]:
    """Point deltas (candidate - baseline) per shared metric; an untrusted judge metric shows
    its trust state instead of a delta."""
    b, c = by_name(base), by_name(cand)
    rows: list[dict[str, str]] = []
    for name in sorted(b.keys() & c.keys()):
        mb, mc = b[name], c[name]
        untrusted = hides_value(mb) or hides_value(mc)
        delta = None if mb.value is None or mc.value is None or untrusted else mc.value - mb.value
        rows.append(
            {
                "metric": name,
                "baseline": value_text(mb),
                "candidate": value_text(mc),
                "delta": NOT_TRUSTED if untrusted else _signed(delta),
            }
        )
    return rows


def _signed(value: float | None) -> str:
    return NONE if value is None else ("+" if value > 0 else "") + fmt(value)


_CAUGHT = frozenset({"coverage", "verifier", "both"})


def binary_outcomes(
    row: RunRow,
    questions: Mapping[str, Question],
    items: Mapping[str, VerifierItem] | None = None,
    clean_verified: Collection[str] = frozenset(),
) -> dict[str, bool]:
    """Per-item pass/fail measures read straight off a row (display only, for flips).

    "pass" is always the GOOD outcome, so pass -> fail reads as a regression everywhere. That is
    why verifier rows split by item kind: on a planted item "verified" is a miss, on a clean one
    it is the right answer; pooling them would make a better verifier look worse."""
    out = row.output
    if row.error is not None or out is None:
        return {}
    if isinstance(out, AnswerOutput):
        found = {"verified": out.status == "verified", "answered": not out.abstained}
        correct = row.grades.get("correct")
        if isinstance(correct, bool):
            found["correct"] = correct
        return found
    if isinstance(out, VerifyOutput):
        return _verify_outcomes(out, (items or {}).get(row.item_id), clean_verified)
    if isinstance(out, RetrievalOutput):
        return _retrieval_outcomes(out, questions.get(row.item_id))
    return {}


def _verify_outcomes(
    out: VerifyOutput, item: VerifierItem | None, clean_verified: Collection[str]
) -> dict[str, bool]:
    """Planted: caught where the error sits (span) and badge flipped; clean: verified clean.
    An item missing from the verifier set has no known kind, so no measure.

    "badge flipped" exists ONLY when the item's clean source was verified clean in the SAME run
    (ADR-0017 E-V, r4; reference_metrics.verifier): otherwise a verifier that rejects
    everything would read as a gain (integrator fix of review P1)."""
    if item is None:
        return {}
    if item.mutation is None:
        return {"clean: verified clean": out.status == "verified"}
    found = {"planted: caught at the span": caught_layer(out, item) in _CAUGHT}
    if item.source_id in clean_verified:
        found["planted: badge flipped"] = out.status != "verified"
    return found


def clean_verified_ids(
    rows: Sequence[RunRow], items: Mapping[str, VerifierItem] | None
) -> frozenset[str]:
    """Clean items this run verified clean: the sources whose planted items may count."""
    known = items or {}
    return frozenset(
        r.item_id
        for r in rows
        if isinstance(r.output, VerifyOutput)
        and r.output.status == "verified"
        and (item := known.get(r.item_id)) is not None
        and item.mutation is None
    )


def _retrieval_outcomes(out: RetrievalOutput, q: Question | None) -> dict[str, bool]:
    """The per-item counterparts of reranked hit@5 (need=any) and of full recall@5 (need=all),
    kept apart so each measure pairs the same questions as its metric (reference_metrics)."""
    if q is None or not q.answerable:
        return {}
    top = out.stages.get("reranked", [])[:5]
    if q.need is Need.any:
        return {"reranked: relevant page in top 5": any(p in q.relevance for p in top)}
    return {"reranked: all relevant pages in top 5": all(p in top for p in q.relevance)}


@dataclass(frozen=True, slots=True)
class Flip:
    item_id: str
    measure: str
    baseline: str
    candidate: str


@dataclass(frozen=True, slots=True)
class Discordance:
    measure: str
    pairs: int
    lost: int  # pass -> fail
    gained: int  # fail -> pass
    p_value: float  # exact McNemar, two-sided: EXPLORATORY


@dataclass(frozen=True, slots=True)
class Paired:
    flips: list[Flip]
    discordance: list[Discordance]
    only_one: int  # items present in one run only (errored or hidden in the other)


def paired(
    base: Sequence[RunRow],
    cand: Sequence[RunRow],
    questions: Mapping[str, Question],
    items: Mapping[str, VerifierItem] | None = None,
) -> Paired:
    """Per-item flips between two comparable runs, keyed by item id (both runs' visible rows)."""
    b = {r.item_id: r for r in base}
    c = {r.item_id: r for r in cand}
    common = sorted(b.keys() & c.keys())
    base_clean, cand_clean = clean_verified_ids(base, items), clean_verified_ids(cand, items)
    flips: list[Flip] = []
    counts: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])  # pairs, lost, gained
    for item in common:
        ob = binary_outcomes(b[item], questions, items, base_clean)
        oc = binary_outcomes(c[item], questions, items, cand_clean)
        for measure in sorted(ob.keys() & oc.keys()):
            tally = counts[measure]
            tally[0] += 1
            if ob[measure] != oc[measure]:
                tally[1 if ob[measure] else 2] += 1
                flips.append(Flip(item, measure, _yes(ob[measure]), _yes(oc[measure])))
        flips += _score_flip(item, b[item], c[item])
    discordance = [
        Discordance(m, t[0], t[1], t[2], mcnemar_exact(t[1], t[2]))
        for m, t in sorted(counts.items())
    ]
    return Paired(flips, discordance, len(b.keys() ^ c.keys()))


def _yes(value: bool) -> str:
    return "pass" if value else "fail"


def _score_flip(item: str, base: RunRow, cand: RunRow) -> list[Flip]:
    ob, oc = base.output, cand.output
    if isinstance(ob, GradeOutput) and isinstance(oc, GradeOutput) and ob.score != oc.score:
        return [Flip(item, "grade score", str(ob.score), str(oc.score))]
    return []


def mcnemar_exact(lost: int, gained: int) -> float:
    """Two-sided exact McNemar: a binomial(n = discordant pairs, 1/2) test on the smaller count.
    Exact because these n are small; shown as exploratory only (the gate owns decisions)."""
    n = lost + gained
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(lost, gained) + 1)) / (1 << n)
    return min(1.0, 2 * tail)


# ------------------------------------------------------------------ verifier (drill-down)
def overlaps(a: Span, b: Span) -> bool:
    """The server's positional overlap rule: half-open intervals that share a character."""
    return a.start < b.end and b.start < a.end


def caught_layer(out: VerifyOutput, item: VerifierItem | None) -> str:
    """Which layer's flag overlaps the planted span (ADR-0017 §3 E-V: span-based, a flag
    elsewhere is not a catch); "caught, unlocated" when only an unplaced objection flipped it."""
    if item is None:
        return "unknown item (not in the verifier set)"
    if item.span is None:
        return "clean item" if out.status == "verified" else "clean item, rejected"
    cov = any(overlaps(s, item.span) for s in out.coverage_spans)
    ver = any(overlaps(s, item.span) for s in out.verifier_spans)
    if cov and ver:
        return "both"
    if cov or ver:
        return "coverage" if cov else "verifier"
    unlocated = (
        out.status != "verified"
        and out.verifier_ran
        and out.verifier_objected
        and out.unlocated_verifier_claims > 0
    )
    return "caught, unlocated" if unlocated else "missed"


@dataclass(frozen=True, slots=True)
class SloVerdict:
    state: str  # "pass" | "fail" | "not evaluable"
    text: str


def slo_verdict(summary: RunSummary | None) -> SloVerdict:
    """ADR-0017 §3 E-V: badge-flip recall LOWER bound >= 0.75 AND verifier-caused false-reject
    UPPER bound <= 0.20. Judged on the interval bounds only: a point estimate never passes."""
    m = by_name(summary)
    recall = m.get("verifier.badge_flip_recall")
    reject = m.get("verifier.false_reject.verifier_caused")
    if recall is None or reject is None:
        return SloVerdict("not evaluable", "the summary lacks badge-flip recall or false-reject")
    if recall.ci_low is None or reject.ci_high is None:
        return SloVerdict("not evaluable", "no confidence interval yet (point values only)")
    ok = recall.ci_low >= SLO_RECALL_LOW and reject.ci_high <= SLO_FALSE_REJECT_HIGH
    if ok:
        return SloVerdict("pass", "this verifier model meets the SLO")
    return SloVerdict(
        "fail", "below the SLO: this verifier model should be `verifier_trusted=false`"
    )


# ------------------------------------------------------------------ interview
def level_means(rows: Iterable[RunRow]) -> list[dict[str, str]]:
    """Mean grade per (tier, level) for the gold and partial sweep: the trend the Page's L test
    in the summary tests. Display only; the test itself is the summary's."""
    cells: dict[tuple[str, str], list[int]] = defaultdict(list)
    for row in rows:
        out = row.output
        if isinstance(out, GradeOutput) and out.tier.value in ("gold", "partial"):
            cells[(out.tier.value, out.level)].append(out.score)
    table: list[dict[str, str]] = []
    for tier in ("gold", "partial"):
        line = {"tier": tier}
        for level in LEVELS:
            scores = cells.get((tier, level), [])
            line[level] = fmt(sum(scores) / len(scores)) if scores else NONE
        table.append(line)
    return table


# ------------------------------------------------------------------ calibration
@dataclass(frozen=True, slots=True)
class CalibrationState:
    current: dict[Criterion, Calibration]
    other_judges: int  # records for another judge key (stale for this judge)

    @property
    def untrusted(self) -> list[Criterion]:
        return [c for c, cal in self.current.items() if not cal.trusted]

    @property
    def any_trusted(self) -> bool:
        return any(cal.trusted for cal in self.current.values())


def calibration_state(records: Sequence[Calibration], judge: JudgeKey | None) -> CalibrationState:
    """The newest record per criterion for `judge` (a new judge key makes old ones stale, §4)."""
    current: dict[Criterion, Calibration] = {}
    other = 0
    for cal in records:
        if judge is not None and cal.judge != judge:
            other += 1
            continue
        best = current.get(cal.criterion)
        if best is None or cal.computed_at >= best.computed_at:
            current[cal.criterion] = cal
    return CalibrationState(dict(sorted(current.items())), other)


def judge_keys(records: Sequence[Calibration]) -> list[JudgeKey]:
    keys: list[JudgeKey] = []
    for cal in records:
        if cal.judge not in keys:
            keys.append(cal.judge)
    return keys


def confusion_rows(cal: Calibration) -> list[dict[str, str]]:
    """Rows = the owner's score 0..2, columns = the judge's (the LOCKED orientation)."""
    return [
        {"owner": f"owner {i}", **{f"judge {j}": str(n) for j, n in enumerate(row)}}
        for i, row in enumerate(cal.confusion)
    ]


def calibration_row(cal: Calibration) -> dict[str, str]:
    kappa_ci = cal.kappa_ci
    return {
        "criterion": cal.criterion.value,
        "kappa (linear)": fmt(cal.kappa_linear),
        "95% CI": f"[{fmt(kappa_ci[0])}, {fmt(kappa_ci[1])}]" if kappa_ci else NONE,
        "raw agreement": fmt(cal.raw_agreement),
        "leniency p": fmt(cal.leniency_p),
        "n": str(cal.n),
        "trusted": "trusted" if cal.trusted else NOT_TRUSTED,
        "label batch": cal.label_batch_id,
    }


def trust_for(criterion: Criterion, state: CalibrationState, summary: RunSummary | None) -> str:
    """A judge criterion's trust: the calibration record first, else the summary's flag."""
    cal = state.current.get(criterion)
    if cal is not None:
        return "trusted" if cal.trusted else NOT_TRUSTED
    metric = by_name(summary).get(f"generator.judge.{criterion.value}")
    if metric is not None and metric.trusted is not None:
        return "trusted" if metric.trusted else NOT_TRUSTED
    return UNCALIBRATED


def summary_trust(summary: RunSummary | None) -> dict[str, bool]:
    """Per E-G criterion, whether the summary says its judge is trusted (absent: not judged).

    The calibration metric's flag is the trust verdict itself; the judge metric's flag is a copy
    of it, used only when the summary has no calibration metric for that criterion."""
    metrics = by_name(summary)
    trust: dict[str, bool] = {}
    for criterion in E_G_CRITERIA:
        c = criterion.value
        for name in (f"{CALIBRATION_PREFIX}{c}.kappa_linear", f"generator.judge.{c}"):
            flag = metrics[name].trusted if name in metrics else None
            if flag is not None:
                trust[c] = flag
                break
    return trust


# ------------------------------------------------------------------ judge results and labels
def judge_index(results: Iterable[JudgeResult]) -> dict[str, JudgeResult]:
    return {r.cache_key: r for r in results}


def row_judgements(row: RunRow, index: Mapping[str, JudgeResult]) -> list[JudgeResult]:
    return [index[k] for k in row.judge_refs if k in index]


def owner_labels(digests: Iterable[str], labels: Iterable[Label]) -> list[Label]:
    wanted = set(digests)
    return [lb for lb in labels if lb.item_digest in wanted]


# ------------------------------------------------------------------ the blind label queue
@dataclass(frozen=True, slots=True)
class QueueItem:
    """Exactly what the owner sees while labelling: question, shown notes, answer.

    No field can hold a judge score, a judge reason or the server's verdict: blindness is a
    property of the type, not of the view's discipline (ADR-0017 §2 D5)."""

    item_id: str
    item_digest: str  # the judged item's digest, so a label pairs with its judge results
    question: str
    answer: str
    shown: tuple[ShownRef, ...]


@dataclass(frozen=True, slots=True)
class Queue:
    items: list[QueueItem]  # the calibration sample, in the sample's (digest) order
    skipped: dict[str, int] = field(default_factory=dict)
    outside_sample: int = 0  # labelable items the stratified sample did not draw


def queue_items(
    run: RunFile,
    questions: QuestionSet | None,
    index: Mapping[str, JudgeResult],
    *,
    sample_n: int = CALIBRATION_N,
    seed: int = 0,
) -> Queue:
    """The calibration sample of one E-G run (ADR-0017 §4): its labelable answers, drawn by
    EV-JUDGE's `calibration_sample` (stratified by answer tier and verdict, never heldout2), so
    UI labels and CLI labels come from the same design rather than a convenience sample.

    The sample is drawn from ALL labelable items, without excluding labelled ones, so it stays
    the same set while the owner works through it (progress is k of n, and the queue ends).
    Only the newest row per item counts (a resumed run may repeat an item, as in the CLI). A
    row the judge has not scored yet has no item digest to pair a label with, so it waits."""
    if questions is None:
        return Queue([], {"not in the question set (split unknown)": len(run.rows)})
    qmap = {q.id: q for q in questions.questions}
    latest = list({r.item_id: r for r in run.rows}.values())
    pool: dict[str, QueueItem] = {}
    digest_of: dict[int, str] = {}
    skipped: dict[str, int] = defaultdict(int)
    for row in latest:
        why = _unlabelable(row, qmap)
        digests = {r.item_digest for r in row_judgements(row, index)}
        if why is None and len(digests) != 1:
            why = "not judged yet" if not digests else "judge results disagree on the item"
        if why is not None:
            skipped[why] += 1
            continue
        out = row.output
        assert isinstance(out, AnswerOutput)  # checked by _unlabelable
        digest = digests.pop()
        digest_of[id(row)] = digest
        pool[digest] = QueueItem(
            row.item_id, digest, qmap[row.item_id].question, out.answer, tuple(row.shown)
        )
    candidates = calibration_candidates(
        [replace(run, rows=latest)], questions, lambda r: digest_of.get(id(r))
    )
    picked = calibration_sample(candidates, n=sample_n, seed=seed)
    items = [pool[c.item_digest] for c in picked]
    return Queue(items, dict(skipped), len(pool) - len(items))


def _unlabelable(row: RunRow, questions: Mapping[str, Question]) -> str | None:
    if row.error is not None or not isinstance(row.output, AnswerOutput):
        return "no answer (errored, or not a Q&A run)"
    if row.stale:
        return "stale (its notes changed)"
    if row.output.abstained:
        return "abstained (the judge scores answered rows only)"
    if not row.shown:
        return "with no shown notes recorded (faithfulness needs them)"
    q = questions.get(row.item_id)
    if q is None:
        return "not in the question set (split unknown)"
    if q.split is Split.heldout2:
        return "held out (never labelled)"
    return None


def labelled_digests(labels: Iterable[Label]) -> set[str]:
    """Items whose newest criteria label scores every E-G criterion."""
    return {
        lb.item_digest
        for lb in labels
        if lb.kind is LabelKind.criteria and all(c in lb.scores for c in E_G_CRITERIA)
    }


def criteria_label(
    item: QueueItem,
    scores: Mapping[Criterion, int],
    *,
    note: str,
    batch_id: str,
    at: datetime,
) -> Label:
    return Label(
        item_digest=item.item_digest,
        item_id=item.item_id,
        kind=LabelKind.criteria,
        scores=dict(scores),
        note=note.strip(),
        batch_id=batch_id,
        labelled_at=at,
    )
