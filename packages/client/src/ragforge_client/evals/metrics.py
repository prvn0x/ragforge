"""Pure metric functions of the eval layer (ADR-0017 §3, §4): a run file in, a `RunSummary` out.

The NAMES and point values are defined by `evals/demo/fixtures/reference_metrics.py` (the golden
oracle, stdlib only, never imports this module); this module must reproduce them to 1e-9. What is
added here is everything the oracle leaves out on purpose: the n, the interval and its method,
the judge trust state, and the hygiene rules a real run needs:

- one row per item: the LATEST attempt wins (a resumed run re-runs transient errors);
- stale rows (principle 6) are excluded and counted, never re-mapped;
- errored rows are excluded and counted by class (`ours` vs `provider_transient`);
- a run made from a different dataset than the one passed in is refused (digest check).

Intervals: Wilson for simple proportions; a bootstrap clustered by source question for E-V (and,
with one cluster per question, for non-binary means such as MRR), widened to the exact
Clopper-Pearson bound for binary items when that is more conservative (a degenerate bootstrap is
never reported as a CI); Clopper-Pearson for interview ranking k/n; an iid pairs bootstrap for
kappa. Every value carries its method.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel

from ragforge_client.evals import stats
from ragforge_client.evals.interfaces import Datasets
from ragforge_client.evals.runfile import RunFile
from ragforge_client.evals.schema import (
    E_G_CRITERIA,
    E_I_CRITERIA,
    AnswerOutput,
    AnswerTier,
    Calibration,
    Criterion,
    ErrorClass,
    EvalKind,
    GradeOutput,
    InterviewSeed,
    JudgeKey,
    JudgeResult,
    Label,
    LabelKind,
    Method,
    MetricValue,
    Question,
    QuestionOutput,
    RetrievalOutput,
    RunRow,
    RunSummary,
    Span,
    VerifierItem,
    VerifyOutput,
    digest,
)

Stage = Literal["keyword", "vector", "fused", "reranked"]
STAGES: tuple[Stage, ...] = ("keyword", "vector", "fused", "reranked")
LEVELS = ("sde_1", "sde_2", "senior_sde_1", "senior_sde_2", "staff")
RANKING_PAIRS = (
    (AnswerTier.gold, AnswerTier.partial),
    (AnswerTier.partial, AnswerTier.wrong),
    (AnswerTier.gold, AnswerTier.wrong),
    (AnswerTier.gold, AnswerTier.misconception),
)
SWEEP_TIERS = (AnswerTier.gold, AnswerTier.partial)
MIN_QUOTE_CHARS = 12  # a misconception quote shorter than this (normalised) never matches
COLD_QUESTIONS = 2  # the latency pass excludes the first 2 (cold) questions
KAPPA_TRUSTED = 0.6
KAPPA_CI_LOW_TRUSTED = 0.4
LENIENCY_ALPHA = 0.05
QUESTION_QUALITY_LABELS_NEEDED = 40  # r2 P1-7: below this the E-I judge stays "uncalibrated"
# No judge is trusted on fewer labelled items than this, whatever its kappa and CI say: at tiny n
# the bootstrap has a handful of distinct resamples and the sign test cannot fire. The ADR's
# calibration set is 50 per criterion, and that is the floor (owner decision 2026-10-21, on the
# design reviewer's simulation: at n=20 a good judge is trusted only ~39% of the time, so fewer
# labels make "trusted" a coin flip).
MIN_CALIBRATION_LABELS = 50
FALSE_VERIFIED_BASE_RATES = (0.05, 0.10, 0.20, 0.30)  # ADR-0017 §3: base rates of 5-30%


class MetricsError(ValueError):
    """The run cannot be summarised: wrong dataset, unknown item, or a malformed output."""


# ------------------------------------------------------------------ row hygiene
@dataclass(frozen=True)
class Selection:
    """The usable rows of a run, by item id, and what was excluded."""

    rows: dict[str, RunRow]
    stale: int
    errors_ours: int
    errors_transient: int
    # items with an error of ours on ANY attempt: a resume that retries one must not hide it
    ours_any_attempt: int = 0


def select_rows(run: RunFile) -> Selection:
    """Latest attempt per item; then drop stale and errored rows, counting each kind.

    A stale row is counted as stale even if it also errored: staleness is about the dataset, and
    the owner fixes it by re-approving the item, not by re-running."""
    latest: dict[str, RunRow] = {}
    for row in run.rows:
        held = latest.get(row.item_id)
        if held is None or row.attempt >= held.attempt:
            latest[row.item_id] = row
    ours_ever = {
        r.item_id
        for r in run.rows
        if r.error is not None and r.error.cls is not ErrorClass.provider_transient
    }
    usable: dict[str, RunRow] = {}
    stale = ours = transient = 0
    for item_id, row in latest.items():
        if row.stale:
            stale += 1
        elif row.error is not None and row.error.cls is ErrorClass.provider_transient:
            transient += 1
        elif row.error is not None or row.output is None:
            ours += 1  # an output-less row without an error class is our bug too
        else:
            usable[item_id] = row
    return Selection(usable, stale, ours, transient, len(ours_ever))


def _outputs[O](selection: Selection, kind: type[O]) -> dict[str, O]:
    out: dict[str, O] = {}
    for item_id, row in selection.rows.items():
        if not isinstance(row.output, kind):
            raise MetricsError(f"{item_id}: expected a {kind.__name__} output")
        out[item_id] = row.output
    return out


def _require[D: BaseModel](dataset: D | None, what: str, run: RunFile) -> D:
    """The dataset the run names, digest-checked: a summary of the wrong dataset is a lie."""
    if dataset is None:
        raise MetricsError(f"{run.header.run_id}: a {run.header.eval_kind} run needs the {what}")
    if digest(dataset.model_dump(mode="json")) != run.header.dataset.digest:
        raise MetricsError(
            f"{run.header.run_id}: made from a different {what} (dataset digest mismatch)"
        )
    return dataset


def _lookup[V](table: Mapping[str, V], item_id: str, what: str) -> V:
    try:
        return table[item_id]
    except KeyError:
        raise MetricsError(f"{item_id}: not in the {what}") from None


# ------------------------------------------------------------------ metric value builders
def _mean(values: Sequence[float]) -> float | None:
    return math.fsum(values) / len(values) if values else None


def proportion(name: str, values: Sequence[float], *, note: str = "") -> MetricValue:
    """A share of binary outcomes with its Wilson interval."""
    if not values:
        return MetricValue(name=name, value=None, n=0, method=Method.point, note=note)
    k = round(math.fsum(values))
    low, high = stats.wilson(k, len(values))
    return MetricValue(
        name=name,
        value=_mean(values),
        ci_low=low,
        ci_high=high,
        n=len(values),
        method=Method.wilson,
        note=note,
    )


def exact_proportion(name: str, values: Sequence[float], *, note: str = "") -> MetricValue:
    """A share of binary outcomes with its Clopper-Pearson interval (too few clusters to boot)."""
    if not values:
        return MetricValue(name=name, value=None, n=0, method=Method.point, note=note)
    k = round(math.fsum(values))
    low, high = stats.clopper_pearson(k, len(values))
    return MetricValue(
        name=name,
        value=_mean(values),
        ci_low=low,
        ci_high=high,
        n=len(values),
        method=Method.clopper_pearson,
        note=note,
    )


def _binary_counts(name: str, values: Sequence[float]) -> tuple[int, int]:
    """(k, n) of a metric its caller declared binary; a non-0/1 value there is a bug upstream."""
    if not all(v in (0.0, 1.0) for v in values):
        raise MetricsError(f"{name}: declared binary but holds a value other than 0 or 1")
    return round(math.fsum(values)), len(values)


def _conservative_ci(
    name: str, clusters: Sequence[Sequence[float]], flat: Sequence[float], *, binary: bool
) -> tuple[tuple[float, float] | None, Method, str]:
    """The cluster-bootstrap CI, made honest at small n.

    A percentile bootstrap of an all-ones (or all-zeros) sample is [1, 1] (or [0, 0]): every
    resample is identical, so it reports certainty from three items. For binary values each bound
    is therefore the more conservative of the bootstrap bound and the exact Clopper-Pearson bound
    on the item counts (the bootstrap keeps the cluster correlation at large n, CP keeps coverage
    at small n). A non-binary degenerate bootstrap has no exact fallback and reports no interval.

    Binary is DECLARED by the caller, never inferred from the values: a 0..2 judge mean whose
    scores all happen to be 0 or 1 is still not a binomial proportion (its CP bound would cap at 1
    on a 0..2 scale).
    """
    boot = stats.cluster_bootstrap([c for c in clusters if c])
    degenerate = boot is None or boot[0] == boot[1]
    if not binary:
        if degenerate:
            return None, Method.point, "bootstrap degenerate (every resample equal): no interval"
        return boot, Method.cluster_bootstrap, ""
    cp_low, cp_high = stats.clopper_pearson(*_binary_counts(name, flat))
    if degenerate or boot is None:
        return (cp_low, cp_high), Method.clopper_pearson, "exact CP: bootstrap degenerate"
    low, high = min(boot[0], cp_low), max(boot[1], cp_high)
    widened = (low, high) != boot
    note = "a bound widened to the exact Clopper-Pearson bound" if widened else ""
    return (low, high), Method.cluster_bootstrap, note


def clustered_mean(
    name: str,
    clusters: Sequence[Sequence[float]],
    *,
    binary: bool,
    note: str = "",
    trusted: bool | None = None,
) -> MetricValue:
    """An item mean with a bootstrap CI over clusters (one cluster per independent unit),
    never narrower than the exact binomial interval when the caller declares the items binary."""
    flat = [v for c in clusters for v in c]
    if not flat:
        return MetricValue(
            name=name, value=None, n=0, method=Method.point, trusted=trusted, note=note
        )
    ci, method, why = _conservative_ci(name, clusters, flat, binary=binary)
    return MetricValue(
        name=name,
        value=_mean(flat),
        ci_low=ci[0] if ci else None,
        ci_high=ci[1] if ci else None,
        n=len(flat),
        method=method,
        trusted=trusted,
        note="; ".join(x for x in (note, why) if x),
    )


def iid_mean(name: str, values: Sequence[float], *, note: str = "") -> MetricValue:
    """A non-binary item mean (MRR, nDCG, recall): bootstrap with one cluster per question."""
    extra = "bootstrap over questions" if len(values) > 1 else ""
    return clustered_mean(name, [[v] for v in values], binary=False, note=note or extra)


def point(name: str, value: float | None, n: int, *, note: str = "") -> MetricValue:
    return MetricValue(name=name, value=value, n=n, method=Method.point, note=note)


# ------------------------------------------------------------------ E-R
def _first_relevant(ranked: Sequence[str], relevance: Mapping[str, int]) -> int | None:
    return next((i + 1 for i, page in enumerate(ranked) if page in relevance), None)


def _ndcg10(ranked: Sequence[str], relevance: Mapping[str, int]) -> float:
    """Gain 2^rel - 1, discount log2(rank + 1); the ideal ordering sorts every grade."""
    dcg = sum((2 ** relevance.get(p, 0) - 1) / math.log2(i + 2) for i, p in enumerate(ranked[:10]))
    ideal = sorted(relevance.values(), reverse=True)[:10]
    idcg = sum((2**g - 1) / math.log2(i + 2) for i, g in enumerate(ideal))
    return dcg / idcg if idcg else 0.0


def _recall(ranked: Sequence[str], relevance: Mapping[str, int], k: int) -> float:
    return len({p for p in ranked[:k] if p in relevance}) / len(relevance)


def question_retrieval_values(
    question: Question, output: RetrievalOutput, tau: float | None
) -> dict[str, float]:
    """Every per-question E-R value, keyed by metric name (the oracle's definitions).

    An unanswerable question contributes only `retrieval.unanswerable.below_tau`. A missing top
    score means nothing was retrieved at all, which is below any tau. Unknown tau (no retrieval
    config in the header) contributes nothing."""
    values: dict[str, float] = {}
    if not question.answerable:
        if tau is not None:
            top = output.top_rerank_score
            values["retrieval.unanswerable.below_tau"] = 1.0 if top is None or top < tau else 0.0
        return values
    rel = question.relevance
    for stage in STAGES:
        ranked = output.stages.get(stage, [])
        if question.need.value == "any":
            first = _first_relevant(ranked, rel)
            for k in (5, 10):
                values[f"retrieval.{stage}.hit@{k}"] = 1.0 if first and first <= k else 0.0
            values[f"retrieval.{stage}.mrr"] = 1.0 / first if first else 0.0
        else:
            for k in (5, 10):
                values[f"retrieval.{stage}.recall@{k}"] = _recall(ranked, rel, k)
        values[f"retrieval.{stage}.ndcg@10"] = _ndcg10(ranked, rel)
    values["retrieval.fused.recall@20"] = _recall(output.stages.get("fused", []), rel, 20)
    return values


def retrieval_item_values(run: RunFile, datasets: Datasets) -> dict[str, dict[str, float]]:
    """metric name -> item id -> value, over the usable rows (the gate pairs on this)."""
    qs = _require(datasets.questions, "question set", run)
    by_id = {q.id: q for q in qs.questions}
    tau = run.header.retrieval.tau if run.header.retrieval else None
    table: dict[str, dict[str, float]] = defaultdict(dict)
    for item_id, out in _outputs(select_rows(run), RetrievalOutput).items():
        q = _lookup(by_id, item_id, "question set")
        for name, value in question_retrieval_values(q, out, tau).items():
            table[name][item_id] = value
    return table


_BINARY_RETRIEVAL = re.compile(r"\.hit@\d+$|^retrieval\.unanswerable\.below_tau$")


def retrieval_metrics(run: RunFile, datasets: Datasets) -> list[MetricValue]:
    table = retrieval_item_values(run, datasets)
    out: list[MetricValue] = []
    for name in sorted(table):
        values = [table[name][i] for i in sorted(table[name])]
        out.append(
            proportion(name, values) if _BINARY_RETRIEVAL.search(name) else iid_mean(name, values)
        )
    if "retrieval.unanswerable.below_tau" not in table and run.header.retrieval is None:
        out.append(point("retrieval.unanswerable.below_tau", None, 0, note="no tau in header"))
    return out


# ------------------------------------------------------------------ E-V
def _overlaps(a: Span, b: Span) -> bool:
    """The server's positional overlap rule: half-open intervals that share a character."""
    return a.start < b.end and b.start < a.end


@dataclass(frozen=True)
class VerifierRow:
    item: VerifierItem
    output: VerifyOutput

    @property
    def verified(self) -> bool:
        return self.output.status == "verified"

    @property
    def by_coverage(self) -> bool:
        span = self.item.span
        return span is not None and any(_overlaps(s, span) for s in self.output.coverage_spans)

    @property
    def by_verifier(self) -> bool:
        span = self.item.span
        return span is not None and any(_overlaps(s, span) for s in self.output.verifier_spans)

    @property
    def span_caught(self) -> bool:
        return self.by_coverage or self.by_verifier

    @property
    def layer(self) -> str:
        if self.by_coverage and self.by_verifier:
            return "both"
        if self.by_coverage:
            return "coverage"
        return "verifier" if self.by_verifier else "missed"

    @property
    def caught_unlocated(self) -> bool:
        """Badge flipped by a verifier objection that nothing places on the span (ADR-0015 4a)."""
        o = self.output
        return (
            not self.span_caught
            and not self.verified
            and o.verifier_ran
            and o.verifier_objected
            and o.unlocated_verifier_claims > 0
        )


@dataclass(frozen=True)
class VerifierRun:
    clean: list[VerifierRow]
    planted: list[VerifierRow]

    @property
    def clean_verified_sources(self) -> set[str]:
        return {r.item.source_id for r in self.clean if r.verified}

    @property
    def flip_eligible(self) -> list[VerifierRow]:
        """Planted items whose clean source the SAME run verified (the verifier is stochastic)."""
        ok = self.clean_verified_sources
        return [r for r in self.planted if r.item.source_id in ok]


def verifier_rows(run: RunFile, datasets: Datasets) -> VerifierRun:
    vs = _require(datasets.verifier, "verifier set", run)
    by_id = {i.id: i for i in vs.items}
    rows = [
        VerifierRow(_lookup(by_id, item_id, "verifier set"), out)
        for item_id, out in sorted(_outputs(select_rows(run), VerifyOutput).items())
    ]
    return VerifierRun(
        clean=[r for r in rows if r.item.mutation is None],
        planted=[r for r in rows if r.item.mutation is not None],
    )


def _by_source(
    rows: Iterable[VerifierRow], value: Callable[[VerifierRow], bool]
) -> list[list[float]]:
    """Per-item 0/1 values grouped into clusters by source question (the bootstrap unit)."""
    clusters: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        clusters[r.item.source_id].append(1.0 if value(r) else 0.0)
    return [clusters[k] for k in sorted(clusters)]


def _in_layer(layer: str) -> Callable[[VerifierRow], bool]:
    return lambda r: r.layer == layer


def false_reject_verifier_caused(r: VerifierRow) -> bool:
    """r4 P2: the verifier objected (located or not), whatever coverage did."""
    return not r.verified and r.output.verifier_objected


def verifier_metrics(run: RunFile, datasets: Datasets) -> list[MetricValue]:
    vr = verifier_rows(run, datasets)

    def cb(name: str, clusters: Sequence[Sequence[float]], *, note: str = "") -> MetricValue:
        return clustered_mean(name, clusters, binary=True, note=note)

    eligible = vr.flip_eligible
    flip_note = f"on {len(vr.clean_verified_sources)} clean-verified sources"
    out = [
        cb("verifier.span_recall", _by_source(vr.planted, lambda r: r.span_caught)),
        cb(
            "verifier.badge_flip_recall",
            _by_source(eligible, lambda r: not r.verified),
            note=flip_note,
        ),
        cb(
            "verifier.caught_unlocated",
            _by_source(eligible, lambda r: r.caught_unlocated),
            note=flip_note,
        ),
        cb("verifier.false_reject", _by_source(vr.clean, lambda r: not r.verified)),
        cb(
            "verifier.false_reject.verifier_caused",
            _by_source(vr.clean, false_reject_verifier_caused),
        ),
        cb(
            "verifier.false_reject.coverage_caused",
            _by_source(vr.clean, lambda r: not r.verified and not r.output.verifier_objected),
        ),
    ]
    for layer in ("coverage", "verifier", "both", "missed"):
        out.append(cb(f"verifier.caught_by.{layer}", _by_source(vr.planted, _in_layer(layer))))
    mutations = sorted({r.item.mutation for r in vr.planted if r.item.mutation is not None})
    for m in mutations:
        rows = [r for r in vr.planted if r.item.mutation is m]
        out.append(cb(f"verifier.span_recall.{m.value}", _by_source(rows, lambda r: r.span_caught)))
    by_name = {m.name: m for m in out}
    out.extend(
        false_verified_metrics(
            by_name["verifier.badge_flip_recall"], by_name["verifier.false_reject"]
        )
    )
    return out


@dataclass(frozen=True)
class FalseVerifiedPoint:
    """P(an answer is wrong | it shows the verified badge) at one base rate of wrong answers.

    `low`/`high` plug the component CI bounds in at their worst/best corner (the curve is
    monotone in both), so they are conservative bounds, not a joint confidence interval."""

    base_rate: float
    value: float | None
    low: float | None
    high: float | None


def false_verified_rate(base_rate: float, miss: float, clean_pass: float) -> float | None:
    """Bayes: wrong-and-verified over all verified. `miss` = P(verified | wrong) = 1 - badge-flip
    recall; `clean_pass` = P(verified | clean) = 1 - false-reject. None when nothing verifies."""
    wrong = base_rate * miss
    verified = wrong + (1 - base_rate) * clean_pass
    return wrong / verified if verified > 0 else None


def false_verified_curve(
    badge_flip_recall: MetricValue,
    false_reject: MetricValue,
    base_rates: Sequence[float] = FALSE_VERIFIED_BASE_RATES,
) -> list[FalseVerifiedPoint]:
    """ADR-0017 §3 E-V false-verified curve. It uses BADGE-FLIP recall, not span recall (r4 P2):
    the user sees the badge, and a caught-unlocated flip still removes it. False-reject is the
    total one (any layer), because a clean answer rejected by coverage is not verified either.

    The rate rises as recall falls and as false-reject rises, so the upper bound takes recall's
    CI low with false-reject's CI high, and the lower bound the opposite corner."""

    def at(r: float, recall: float | None, reject: float | None) -> float | None:
        if recall is None or reject is None:
            return None
        return false_verified_rate(r, 1 - recall, 1 - reject)

    rec, fr = badge_flip_recall, false_reject
    return [
        FalseVerifiedPoint(
            base_rate=r,
            value=at(r, rec.value, fr.value),
            low=at(r, rec.ci_high, fr.ci_low),
            high=at(r, rec.ci_low, fr.ci_high),
        )
        for r in base_rates
    ]


def false_verified_metrics(
    badge_flip_recall: MetricValue, false_reject: MetricValue
) -> list[MetricValue]:
    """The curve as `verifier.false_verified@<rate>` values. Method `point`: the locked Method
    enum has no 'plug-in' kind, so the conservative bounds are stated in the note, not as a CI."""
    n = min(badge_flip_recall.n, false_reject.n)
    out: list[MetricValue] = []
    for pt in false_verified_curve(badge_flip_recall, false_reject):
        bounds = (
            f"conservative plug-in bounds [{_fmt(pt.low)}, {_fmt(pt.high)}] from the recall and "
            "false-reject CI bounds (not a joint CI)"
            if pt.low is not None and pt.high is not None
            else "no bounds: a component has no interval"
        )
        out.append(
            point(
                f"verifier.false_verified@{pt.base_rate:.2f}",
                pt.value,
                n,
                note=f"P(wrong | verified) at base rate {pt.base_rate:.0%}; {bounds}",
            )
        )
    return out


# ------------------------------------------------------------------ E-I
def normalise(text: str) -> str:
    return " ".join(re.sub(r"[^\w\s]", " ", text.casefold()).split())


def quote_matches(planted: str, quote: str) -> bool:
    """Normalised containment either way; a quote under 12 normalised chars never counts, so a
    one-word quote cannot match every sentence."""
    p, q = normalise(planted), normalise(quote)
    return len(q) >= MIN_QUOTE_CHARS and (q in p or p in q)


@dataclass(frozen=True)
class InterviewRun:
    seeds: dict[str, InterviewSeed]
    grades: list[GradeOutput]
    questions: list[tuple[RunRow, QuestionOutput]]

    def at_seed_level(self) -> dict[tuple[str, AnswerTier], GradeOutput]:
        """Replicate 0 at each seed's own level: the grading-quality view."""
        return {
            (g.seed_id, g.tier): g
            for g in self.grades
            if g.level == self.seeds[g.seed_id].seed_level and g.replicate == 0
        }

    def cells(self) -> dict[tuple[str, AnswerTier, str], list[int]]:
        """(seed, tier, level) -> replicate scores, gold and partial only (the level sweep)."""
        cells: dict[tuple[str, AnswerTier, str], list[int]] = defaultdict(list)
        for g in sorted(self.grades, key=lambda g: g.replicate):
            if g.tier in SWEEP_TIERS:
                cells[(g.seed_id, g.tier, g.level)].append(g.score)
        return cells


def interview_rows(run: RunFile, datasets: Datasets) -> InterviewRun:
    iv = _require(datasets.interview, "interview set", run)
    seeds = {s.id: s for s in iv.seeds}
    grades: list[GradeOutput] = []
    questions: list[tuple[RunRow, QuestionOutput]] = []
    seen: dict[tuple[str, AnswerTier, str, int], str] = {}
    for item_id, row in sorted(select_rows(run).rows.items()):
        out = row.output
        if isinstance(out, GradeOutput):
            _lookup(seeds, out.seed_id, "interview set")
            key = (out.seed_id, out.tier, out.level, out.replicate)
            if key in seen:
                # two item ids grading one cell: a runner bug that last-wins would hide
                raise MetricsError(f"{item_id}: duplicates the grade of {seen[key]} ({key})")
            seen[key] = item_id
            grades.append(out)
        elif isinstance(out, QuestionOutput):
            _lookup(seeds, out.seed_id, "interview set")
            questions.append((row, out))
        else:
            raise MetricsError(f"{item_id}: expected a grade or question output")
    return InterviewRun(seeds, grades, questions)


def ranking_outcomes(ivr: InterviewRun) -> dict[str, float]:
    """'<seed>:<better>><worse>' -> 1 if strictly ranked right (a tie is wrong)."""
    at = ivr.at_seed_level()
    out: dict[str, float] = {}
    for s in sorted(ivr.seeds):
        for a, b in RANKING_PAIRS:
            if (s, a) in at and (s, b) in at:
                out[f"{s}:{a.value}>{b.value}"] = (
                    1.0 if at[(s, a)].score > at[(s, b)].score else 0.0
                )
    return out


def detection_outcomes(ivr: InterviewRun, *, kept: bool) -> dict[str, float]:
    """seed -> 1 if the planted sentence is among the grade's misconception quotes."""
    at = ivr.at_seed_level()
    out: dict[str, float] = {}
    for s, seed in sorted(ivr.seeds.items()):
        g = at.get((s, AnswerTier.misconception))
        if g is not None:
            quotes = g.misconceptions_kept if kept else g.misconceptions_given
            hit = any(quote_matches(seed.misconception_sentence, q) for q in quotes)
            out[s] = 1.0 if hit else 0.0
    return out


def level_blocks(ivr: InterviewRun) -> dict[str, list[float]]:
    """'<seed>/<tier>' -> cell means ordered staff -> sde_1 (the predicted INCREASING order).

    A block with a missing cell is dropped: Page's L compares rank sums of equal-length blocks,
    so a shorter block would shift the weights (the fixture has none; see the handoff)."""
    cells = ivr.cells()
    order = list(reversed(LEVELS))
    blocks: dict[str, list[float]] = {}
    for s in sorted(ivr.seeds):
        for tier in SWEEP_TIERS:
            vals = [_mean([float(x) for x in cells.get((s, tier, lv), [])]) for lv in order]
            present = [v for v in vals if v is not None]
            if len(present) == len(order):
                blocks[f"{s}/{tier.value}"] = present
    return blocks


def retest_sd(ivr: InterviewRun) -> tuple[float | None, int]:
    """sqrt(mean((r0 - r1)^2) / 2) over cells graded twice: the per-grade noise SD."""
    sq = [float((c[0] - c[1]) ** 2) for c in ivr.cells().values() if len(c) == 2]
    m = _mean(sq)
    return (math.sqrt(m / 2) if m is not None else None), len(sq)


def interview_metrics(
    run: RunFile, datasets: Datasets, judge_results: Sequence[JudgeResult], labels: Sequence[Label]
) -> list[MetricValue]:
    ivr = interview_rows(run, datasets)
    at = ivr.at_seed_level()
    ranking = ranking_outcomes(ivr)
    gold = [
        1.0 if g.score >= 7 else 0.0 for (_, t), g in sorted(at.items()) if t is AnswerTier.gold
    ]
    wrong = [
        1.0 if g.score <= 3 else 0.0 for (_, t), g in sorted(at.items()) if t is AnswerTier.wrong
    ]
    out = [
        exact_proportion(
            "interview.ranking_accuracy",
            [ranking[k] for k in sorted(ranking)],
            note="tier pairs at the seed level; a tie counts as wrong",
        ),
        proportion("interview.detection_given", list(detection_outcomes(ivr, kept=False).values())),
        proportion("interview.detection_kept", list(detection_outcomes(ivr, kept=True).values())),
        proportion("interview.band.gold_ge_7", gold),
        proportion("interview.band.wrong_le_3", wrong),
    ]
    out.extend(_level_metrics(ivr))
    out.extend(_question_quality(run, ivr, judge_results, labels))
    return out


def _level_metrics(ivr: InterviewRun) -> list[MetricValue]:
    blocks = level_blocks(ivr)
    sd, n_sd = retest_sd(ivr)
    out = [point("interview.level.retest_sd", sd, n_sd, note="cells graded twice")]
    if not blocks:
        out.append(point("interview.level.page_l", None, 0, note="no complete level block"))
        return out
    test = stats.page_l_test(list(blocks.values()))
    out.append(
        MetricValue(
            name="interview.level.page_l",
            value=test.statistic,
            n=len(blocks),
            method=Method.exact,
            note=f"exact permutation p={test.p_value:.4g} (one-sided, staff -> sde_1 increasing)",
        )
    )
    out.append(
        MetricValue(
            name="interview.level.page_l.p_value",
            value=test.p_value,
            n=len(blocks),
            method=Method.exact,
            note="P(L* >= L) under within-block exchangeability",
        )
    )
    diffs = [b[-1] - b[0] for b in blocks.values()]  # sde_1 - staff
    sign = stats.sign_test(diffs)
    out.append(
        MetricValue(
            name="interview.level.sde1_vs_staff.sign_p",
            value=sign.p_value,
            n=len(diffs),
            method=Method.exact,
            note=f"pooled paired sign test: +{sign.positive} -{sign.negative} ={sign.zeros}",
        )
    )
    return out


def _question_quality(
    run: RunFile, ivr: InterviewRun, judge_results: Sequence[JudgeResult], labels: Sequence[Label]
) -> list[MetricValue]:
    """E-I question quality: deterministic grades the runner recorded (any boolean `grades`
    key, e.g. `leak`, `code_ref`, `seed_chunk_included`) and the judge criteria, which stay
    'uncalibrated' (trusted False) until 40 questions are labelled AND calibration trusts them."""
    if not ivr.questions:
        return []
    out: list[MetricValue] = []
    graded: dict[str, list[float]] = defaultdict(list)
    for row, _ in ivr.questions:
        for key, value in row.grades.items():
            if isinstance(value, bool):
                graded[key].append(1.0 if value else 0.0)
    for key in sorted(graded):
        out.append(proportion(f"interview.question.{key}", graded[key]))
    rows = [row for row, _ in ivr.questions]
    out.extend(
        _judge_means(
            run,
            rows,
            E_I_CRITERIA,
            "interview.question.judge",
            judge_results,
            labels,
            min_labels=QUESTION_QUALITY_LABELS_NEEDED,
        )
    )
    return out


# ------------------------------------------------------------------ E-G + calibration
def latest_labels(labels: Iterable[Label]) -> list[Label]:
    """The newest label per (item_digest, kind); later lines win ties (append-only file)."""
    newest: dict[tuple[str, LabelKind], Label] = {}
    for lb in labels:
        key = (lb.item_digest, lb.kind)
        held = newest.get(key)
        if held is None or lb.labelled_at >= held.labelled_at:
            newest[key] = lb
    return list(newest.values())


def _judge_by_item(
    results: Iterable[JudgeResult], judge: JudgeKey, criterion: Criterion
) -> tuple[dict[str, int], int]:
    """item digest -> the newest valid score of THIS judge key; plus the invalid-reply count."""
    newest: dict[str, JudgeResult] = {}
    for jr in results:
        if jr.judge == judge and jr.criterion is criterion:
            held = newest.get(jr.item_digest)
            if held is None or jr.scored_at >= held.scored_at:
                newest[jr.item_digest] = jr
    valid = {d: jr.score for d, jr in newest.items() if jr.score is not None}
    return valid, len(newest) - len(valid)


def calibration_pairs(
    criterion: Criterion,
    judge: JudgeKey,
    judge_results: Sequence[JudgeResult],
    labels: Sequence[Label],
) -> list[tuple[int, int]]:
    """(owner, judge) score pairs over items the owner labelled on this criterion."""
    owner = {
        lb.item_digest: lb.scores[criterion]
        for lb in latest_labels(labels)
        if lb.kind is LabelKind.criteria and criterion in lb.scores
    }
    scored, _ = _judge_by_item(judge_results, judge, criterion)
    return [(owner[d], scored[d]) for d in sorted(owner) if d in scored]


def calibrate(
    criterion: Criterion,
    judge: JudgeKey,
    judge_results: Sequence[JudgeResult],
    labels: Sequence[Label],
    *,
    computed_at: datetime | None = None,
    seed: int = 0,
) -> Calibration:
    """ADR-0017 §4: kappa (linear) with a bootstrap CI, raw agreement, confusion, leniency.

    Trusted = kappa >= 0.6 AND CI low >= 0.4 AND no significant bias of judge - owner (exact sign
    test, either direction: a harsh judge misleads as much as a lenient one) AND at least
    `MIN_CALIBRATION_LABELS` labelled pairs. A degenerate kappa bootstrap gives no CI
    (`stats.kappa_ci`), which also means untrusted."""
    pairs = calibration_pairs(criterion, judge, judge_results, labels)
    kappa = stats.linear_weighted_kappa(pairs)
    ci = stats.kappa_ci(pairs, seed=seed)
    confusion = [[0] * 3 for _ in range(3)]
    for owner, judged in pairs:
        confusion[owner][judged] += 1
    sign = stats.sign_test([j - o for o, j in pairs])
    leniency_p = sign.p_value if sign.positive + sign.negative else None
    trusted = (
        len(pairs) >= MIN_CALIBRATION_LABELS
        and kappa is not None
        and kappa >= KAPPA_TRUSTED
        and ci is not None
        and ci[0] >= KAPPA_CI_LOW_TRUSTED
        and (leniency_p is None or leniency_p >= LENIENCY_ALPHA)
    )
    batches = sorted({lb.batch_id for lb in latest_labels(labels) if lb.kind is LabelKind.criteria})
    return Calibration(
        criterion=criterion,
        judge=judge,
        label_batch_id="+".join(batches),
        n=len(pairs),
        kappa_linear=kappa,
        kappa_ci=ci,
        raw_agreement=_mean([1.0 if a == b else 0.0 for a, b in pairs]),
        confusion=confusion,
        leniency_p=leniency_p,
        trusted=trusted,
        computed_at=computed_at or datetime.now(UTC),
    )


def _judge_means(
    run: RunFile,
    rows: Sequence[RunRow],
    criteria: Sequence[Criterion],
    prefix: str,
    judge_results: Sequence[JudgeResult],
    labels: Sequence[Label],
    *,
    min_labels: int = 0,
) -> list[MetricValue]:
    """Mean judge score (0..2) per criterion over `rows`' judge refs, with its trust state.

    Only results of the run header's judge key count: trust comes from THAT key's calibration,
    so a ref resolving to another judge (stale cache, moved provider alias) is excluded and
    counted rather than mixed into the mean."""
    judge = run.header.judge
    by_key = {jr.cache_key: jr for jr in judge_results}
    out: list[MetricValue] = []
    for crit in criteria:
        clusters: list[list[float]] = []
        invalid = missing = mismatched = 0
        for row in rows:
            scores: list[float] = []
            for ref in row.judge_refs:
                jr = by_key.get(ref)
                if jr is None:
                    missing += 1
                elif jr.criterion is not crit:
                    continue
                elif judge is not None and jr.judge != judge:
                    mismatched += 1
                else:
                    if jr.score is None:
                        invalid += 1
                    else:
                        scores.append(float(jr.score))
            clusters.append(scores)
        trusted, why = False, "no judge key in the run header"
        if judge is not None:
            cal = calibrate(crit, judge, judge_results, labels)
            trusted = cal.trusted and cal.n >= min_labels
            why = f"calibration n={cal.n}, kappa={_fmt(cal.kappa_linear)}"
            needed = max(min_labels, MIN_CALIBRATION_LABELS)
            if not trusted:
                why += f", uncalibrated (< {needed} labels)" if cal.n < needed else ", not trusted"
        notes = [why]
        if mismatched:
            notes.append(f"{mismatched} judge refs from another judge key excluded")
        if invalid:
            notes.append(f"{invalid} invalid judge replies excluded")
        if missing:
            notes.append(f"{missing} judge refs without a result")
        out.append(
            clustered_mean(
                f"{prefix}.{crit.value}",
                clusters,
                binary=False,
                note="; ".join(notes),
                trusted=trusted,
            )
        )
    return out


def _fmt(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.3f}"


def _calibration_metrics(
    run: RunFile, judge_results: Sequence[JudgeResult], labels: Sequence[Label]
) -> list[MetricValue]:
    judge = run.header.judge
    if judge is None:
        return []
    out: list[MetricValue] = []
    for crit in E_G_CRITERIA:
        cal = calibrate(crit, judge, judge_results, labels)
        pairs = calibration_pairs(crit, judge, judge_results, labels)
        raw = proportion(
            f"calibration.{crit.value}.raw_agreement", [1.0 if a == b else 0.0 for a, b in pairs]
        )
        out.append(raw.model_copy(update={"trusted": cal.trusted}))
        ci = cal.kappa_ci
        out.append(
            MetricValue(
                name=f"calibration.{crit.value}.kappa_linear",
                value=cal.kappa_linear,
                ci_low=ci[0] if ci else None,
                ci_high=ci[1] if ci else None,
                n=cal.n,
                method=Method.bootstrap if ci else Method.point,
                trusted=cal.trusted,
                note=f"iid pairs bootstrap over labelled items; "
                f"leniency p={_fmt(cal.leniency_p)}; trust needs n >= {MIN_CALIBRATION_LABELS}"
                + ("" if ci else "; no CI: bootstrap degenerate or too few defined resamples"),
            )
        )
    return out


def answer_rows(run: RunFile) -> dict[str, AnswerOutput]:
    return _outputs(select_rows(run), AnswerOutput)


def _deterministic_answers(
    prefix: str, run: RunFile, outs: Mapping[str, AnswerOutput], questions: Mapping[str, Question]
) -> list[MetricValue]:
    """correct% (the runner's deterministic grade), verified% and abstain correctness."""
    sel = select_rows(run)
    correct = [
        1.0 if sel.rows[i].grades["correct"] else 0.0
        for i in sorted(outs)
        if isinstance(sel.rows[i].grades.get("correct"), bool)
    ]
    abstain_ok = [
        1.0 if outs[i].abstained != _lookup(questions, i, "question set").answerable else 0.0
        for i in sorted(outs)
    ]
    out = [
        proportion(
            f"{prefix}.verified",
            [1.0 if outs[i].status == "verified" else 0.0 for i in sorted(outs)],
        ),
        proportion(f"{prefix}.abstain_correct", abstain_ok),
    ]
    if correct:
        out.append(proportion(f"{prefix}.correct", correct))
    return out


def generator_metrics(
    run: RunFile, datasets: Datasets, judge_results: Sequence[JudgeResult], labels: Sequence[Label]
) -> list[MetricValue]:
    """Judge means on ANSWERED rows only, next to the answer rate (an abstain-heavy arm cannot
    look good), plus calibration per criterion and the deterministic answer metrics."""
    qs = _require(datasets.questions, "question set", run)
    outs = answer_rows(run)
    sel = select_rows(run)
    answered = [sel.rows[i] for i in sorted(outs) if not outs[i].abstained]
    out = [
        proportion("generator.answer_rate", [0.0 if o.abstained else 1.0 for o in outs.values()])
    ]
    out.extend(_judge_means(run, answered, E_G_CRITERIA, "generator.judge", judge_results, labels))
    out.extend(_calibration_metrics(run, judge_results, labels))
    out.extend(_deterministic_answers("generator", run, outs, {q.id: q for q in qs.questions}))
    confirmed = confirmed_false_verified(run, judge_results, labels)
    if confirmed is not None:
        out.append(confirmed)
    return out


def confirmed_false_verified(
    run: RunFile, judge_results: Sequence[JudgeResult], labels: Sequence[Label]
) -> MetricValue | None:
    """ADR-0017 §3 E-G real-distribution false-verified: k owner-CONFIRMED adjudications over
    the n verified answers of the run, with an exact Clopper-Pearson CI.

    The judge is a screen, not an estimator, so k counts only what the owner confirmed; an
    unflagged wrong answer is invisible, which makes k/n a lower bound. An adjudication label
    belongs to a row when its `item_digest` is the digest a judge result of that row scored
    (the flagged answer itself, not merely the same question id). None when no adjudication
    label belongs to this run: 'not adjudicated yet' must not print as 0/n."""
    sel = select_rows(run)
    by_key = {jr.cache_key: jr for jr in judge_results}
    verified: dict[str, set[str]] = {}
    for item_id, out in answer_rows(run).items():
        if out.status == "verified":
            refs = (by_key.get(ref) for ref in sel.rows[item_id].judge_refs)
            verified[item_id] = {jr.item_digest for jr in refs if jr is not None}
    adjudicated = {
        lb.item_digest: lb.confirmed
        for lb in latest_labels(labels)
        if lb.kind is LabelKind.adjudication and lb.confirmed is not None
    }
    found = [
        any(adjudicated.get(d) is True for d in digests)
        for digests in verified.values()
        if any(d in adjudicated for d in digests)
    ]
    if not found:
        return None
    k = sum(found)
    return exact_proportion(
        "generator.false_verified.confirmed",
        [1.0] * k + [0.0] * (len(verified) - k),
        note=f"{len(found)} judge-flagged answers adjudicated, {k} confirmed; a lower bound "
        "(the judge screen misses some)",
    )


# ------------------------------------------------------------------ E-A + latency
def application_metrics(run: RunFile, datasets: Datasets) -> list[MetricValue]:
    qs = _require(datasets.questions, "question set", run)
    outs = answer_rows(run)
    sel = select_rows(run)
    tokens = [float(sel.rows[i].tokens.prompt + sel.rows[i].tokens.completion) for i in outs]
    out = _deterministic_answers("application", run, outs, {q.id: q for q in qs.questions})
    out.append(iid_mean("application.tokens_per_question", tokens))
    return out


def cold_and_warm_rows(run: RunFile) -> tuple[list[RunRow], list[RunRow]]:
    """The usable rows in the order the questions were FIRST asked, split into the first 2
    (cold) and the rest. A resumed run appends its retries at the end of the file, so file order
    of the latest attempt is not the order asked: each item keeps its first attempt's position
    (the run file has no explicit order field) while its values come from the latest attempt."""
    sel = select_rows(run)
    first_seen = list(dict.fromkeys(r.item_id for r in run.rows))
    ordered = [sel.rows[i] for i in first_seen if i in sel.rows]
    return ordered[:COLD_QUESTIONS], ordered[COLD_QUESTIONS:]


def latency_metrics(run: RunFile) -> list[MetricValue]:
    """TTFT and total p50/p95 in seconds, the first 2 (cold) questions reported apart."""
    cold, warm = cold_and_warm_rows(run)
    out: list[MetricValue] = []
    for field in ("ttft_s", "total_s"):
        values = sorted(v for r in warm if (v := getattr(r.timings, field)) is not None)
        for label, q in (("p50", 0.5), ("p95", 0.95)):
            value = stats.quantile(values, q) if values else None
            out.append(point(f"latency.{field[:-2]}.{label}", value, len(values)))
        cold_values = [v for r in cold if (v := getattr(r.timings, field)) is not None]
        out.append(
            point(
                f"latency.cold.{field[:-2]}.mean",
                _mean(cold_values),
                len(cold_values),
                note="first questions of the pass, excluded above",
            )
        )
    return out


# ------------------------------------------------------------------ entry point
def summarize(
    run: RunFile,
    datasets: Datasets,
    labels: Sequence[Label],
    judge_results: Sequence[JudgeResult],
) -> RunSummary:
    """The run's summary: every metric of its eval kind, sorted by name, with n/CI/method/trust,
    plus the excluded stale and errored counts. Pure: same inputs, same summary."""
    kind = run.header.eval_kind
    if kind is EvalKind.retrieval:
        metrics = retrieval_metrics(run, datasets)
    elif kind is EvalKind.verifier:
        metrics = verifier_metrics(run, datasets)
    elif kind is EvalKind.interview:
        metrics = interview_metrics(run, datasets, judge_results, labels)
    elif kind is EvalKind.generator:
        metrics = generator_metrics(run, datasets, judge_results, labels)
    elif kind is EvalKind.application:
        metrics = application_metrics(run, datasets)
    else:
        metrics = latency_metrics(run)
    sel = select_rows(run)
    return RunSummary(
        run_id=run.header.run_id,
        eval_kind=kind,
        header_digest=run.header_digest,
        metrics=sorted(metrics, key=lambda m: m.name),
        stale_items=sel.stale,
        errors_ours=sel.errors_ours,
        errors_transient=sel.errors_transient,
    )


def metric_map(summary: RunSummary) -> dict[str, MetricValue]:
    return {m.name: m for m in summary.metrics}
