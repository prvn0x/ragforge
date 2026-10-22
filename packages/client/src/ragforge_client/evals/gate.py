"""The regression gate (ADR-0017 §5): baseline vs candidate, paired per item, with the A/A pair
setting the MDE every check prints.

Order of business:
1. **Comparable?** Same eval kind, surface and dataset; the same snapshot CONTENT digest (the index
   digest may differ: a re-embed is evaluable); the same chunker (a chunker change cannot be
   evaluated on a snapshot of chunks). Otherwise the verdict is `not_evaluable`, never a pass.
2. **Complete?** Both arms finished (a footer) and the candidate has a row for every baseline
   item. A crashed, unfinished or item-dropping candidate is `not_evaluable`: it measured less,
   so pairing on the intersection would pass it silently, and no override can excuse that.
   **Pairing.** Latest attempt per item; stale rows and rows with a provider-transient error in
   EITHER arm are excluded (each counted in the note), so a 429 storm cannot fake a regression.
   A baseline item the candidate has no value for (an error of our own, a missing grade) is
   counted as missing and BLOCKS a block-severity check; so does a check that paired nothing.
3. **Checks per kind** (one paired test per metric, Holm within a family):
   - retrieval: exact McNemar on reranked hit@5 + sign-flip on reranked MRR, Holm; block when
     adjusted p < 0.025 AND the drop > 2 points, or a metric falls below its ABSOLUTE (frozen,
     configured) floor - 3 points; no configured floor, no floor block;
   - chat (generator / application): exact McNemar on correct% and verified%, Holm at 0.05;
     hard floor: no errors of our own. The E-V hard floor is a separate verifier-kind verdict
     that the caller ANDs in on any chat or verifier change (`all_pass`);
   - verifier: exact McNemar on the planted items (badge flipped) + the absolute SLO on the
     candidate (badge-flip recall CI low >= 0.75, verifier-caused false-reject CI high <= 0.20,
     each bound the more conservative of the cluster bootstrap and exact Clopper-Pearson);
   - interview and latency: WARN only (10 seeds give huge MDEs; latency is noisy).
4. **Outcome.** not_evaluable > block > warn > pass. An override is recorded, never hidden: the
   outcome stays what the evidence says, and `passes()` honours the override.
"""

from __future__ import annotations

import itertools
import math
import statistics
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

from ragforge_client.evals import metrics, stats
from ragforge_client.evals.interfaces import Datasets
from ragforge_client.evals.runfile import RunFile
from ragforge_client.evals.schema import (
    ErrorClass,
    EvalKind,
    GateCheck,
    GateOutcome,
    GateVerdict,
    RetrievalOutput,
    RunRow,
)

RETRIEVAL_ALPHA = 0.025
RETRIEVAL_MIN_DROP = 0.02  # "the drop is > 2 points"
FLOOR_MARGIN = 0.03  # "or a metric is below floor - 3"
CHAT_ALPHA = 0.05
VERIFIER_ALPHA = 0.05
WARN_ALPHA = 0.05
SLO_BADGE_FLIP_LOW = 0.75
SLO_VERIFIER_FALSE_REJECT_HIGH = 0.20
RETRIEVAL_HIT = "retrieval.reranked.hit@5"
RETRIEVAL_MRR = "retrieval.reranked.mrr"

Values = dict[str, float]  # item id -> per-item value
Extractor = Callable[[RunFile, Datasets], dict[str, Values]]


@dataclass(frozen=True)
class _Spec:
    """One paired check: which per-item values, which test, and how a regression is judged."""

    name: str
    binary: bool  # True: exact McNemar; False: paired sign-flip permutation
    family: str  # Holm adjusts within a family
    alpha: float
    severity: GateOutcome  # block, or warn for warn-only surfaces
    min_drop: float = 0.0  # the effect must also be worse than -min_drop
    higher_is_better: bool = True


@dataclass
class _Result:
    spec: _Spec
    n: int
    effect: float | None
    p: float | None
    method: str
    mde: float | None
    missing: int = 0  # baseline items (after exclusions) with no candidate value
    notes: list[str] = field(default_factory=list)
    # LAST, keyword-only in spirit: transients left too little measured (integrator fix, P1)
    unmeasured: bool = False


# ------------------------------------------------------------------ comparability
def comparability_problems(baseline: RunFile, candidate: RunFile) -> list[str]:
    """Why two runs cannot be compared at all (empty when they can)."""
    b, c = baseline.header, candidate.header
    problems: list[str] = []
    if b.eval_kind is not c.eval_kind:
        problems.append(f"eval kind differs ({b.eval_kind} vs {c.eval_kind})")
    if b.surface is not c.surface:
        problems.append(f"surface differs ({b.surface} vs {c.surface})")
    if b.dataset.digest != c.dataset.digest:
        problems.append("dataset digest differs")
    if (b.snapshot is None) != (c.snapshot is None):
        problems.append("only one arm ran on a snapshot")
    elif b.snapshot and c.snapshot and b.snapshot.content_digest != c.snapshot.content_digest:
        problems.append("snapshot content digest differs (refresh = new baseline)")
    if b.eval_kind is EvalKind.retrieval and (b.retrieval is None) != (c.retrieval is None):
        problems.append("retrieval config in only one arm: the chunker cannot be compared")
    if b.retrieval and c.retrieval and b.retrieval.chunker_version != c.retrieval.chunker_version:
        problems.append("chunker changed: not evaluable on a snapshot of chunks (ADR-0017 §1)")
    return problems


# ------------------------------------------------------------------ per-item extractors
def _bool(x: bool) -> float:
    return 1.0 if x else 0.0


def _retrieval_values(run: RunFile, datasets: Datasets) -> dict[str, Values]:
    table = metrics.retrieval_item_values(run, datasets)
    return {name: table.get(name, {}) for name in (RETRIEVAL_HIT, RETRIEVAL_MRR)}


def _chat_values(run: RunFile, datasets: Datasets) -> dict[str, Values]:
    sel = metrics.select_rows(run)
    outs = metrics.answer_rows(run)
    correct = {
        i: _bool(g) for i in outs if isinstance(g := sel.rows[i].grades.get("correct"), bool)
    }
    verified = {i: _bool(o.status == "verified") for i, o in outs.items()}
    return {"chat.correct": correct, "chat.verified": verified}


def _verifier_values(run: RunFile, datasets: Datasets) -> dict[str, Values]:
    vr = metrics.verifier_rows(run, datasets)
    return {
        "verifier.planted.badge_flipped": {r.item.id: _bool(not r.verified) for r in vr.planted}
    }


def _interview_values(run: RunFile, datasets: Datasets) -> dict[str, Values]:
    ivr = metrics.interview_rows(run, datasets)
    return {
        "interview.ranking": metrics.ranking_outcomes(ivr),
        "interview.detection_kept": metrics.detection_outcomes(ivr, kept=True),
    }


def _latency_values(run: RunFile, datasets: Datasets) -> dict[str, Values]:
    _, warm = metrics.cold_and_warm_rows(run)
    return {
        "latency.total_s": {
            r.item_id: r.timings.total_s for r in warm if r.timings.total_s is not None
        }
    }


_RETRIEVAL_SPECS = (
    _Spec(RETRIEVAL_HIT, True, "retrieval", RETRIEVAL_ALPHA, GateOutcome.block, RETRIEVAL_MIN_DROP),
    _Spec(
        RETRIEVAL_MRR, False, "retrieval", RETRIEVAL_ALPHA, GateOutcome.block, RETRIEVAL_MIN_DROP
    ),
)
_CHAT_SPECS = (
    _Spec("chat.correct", True, "chat", CHAT_ALPHA, GateOutcome.block),
    _Spec("chat.verified", True, "chat", CHAT_ALPHA, GateOutcome.block),
)
_VERIFIER_SPECS = (
    _Spec("verifier.planted.badge_flipped", True, "verifier", VERIFIER_ALPHA, GateOutcome.block),
)
_INTERVIEW_SPECS = (
    _Spec("interview.ranking", True, "interview", WARN_ALPHA, GateOutcome.warn),
    _Spec("interview.detection_kept", True, "interview", WARN_ALPHA, GateOutcome.warn),
)
_LATENCY_SPECS = (
    _Spec(
        "latency.total_s", False, "latency", WARN_ALPHA, GateOutcome.warn, higher_is_better=False
    ),
)

_PLAN: dict[EvalKind, tuple[Extractor, tuple[_Spec, ...]]] = {
    EvalKind.retrieval: (_retrieval_values, _RETRIEVAL_SPECS),
    EvalKind.generator: (_chat_values, _CHAT_SPECS),
    EvalKind.application: (_chat_values, _CHAT_SPECS),
    EvalKind.verifier: (_verifier_values, _VERIFIER_SPECS),
    EvalKind.interview: (_interview_values, _INTERVIEW_SPECS),
    EvalKind.latency: (_latency_values, _LATENCY_SPECS),
}


# ------------------------------------------------------------------ pairing and tests
def _latest_rows(run: RunFile) -> dict[str, RunRow]:
    latest: dict[str, RunRow] = {}
    for row in run.rows:
        if row.item_id not in latest or row.attempt >= latest[row.item_id].attempt:
            latest[row.item_id] = row
    return latest


def _transient_ids(run: RunFile) -> set[str]:
    return {
        i
        for i, r in _latest_rows(run).items()
        if not r.stale and r.error is not None and r.error.cls is ErrorClass.provider_transient
    }


def _stale_ids(run: RunFile) -> set[str]:
    return {i for i, r in _latest_rows(run).items() if r.stale}


def _paired(base: Values, cand: Values) -> list[str]:
    return sorted(set(base) & set(cand))


def _paired_test(spec: _Spec, base: Values, cand: Values) -> tuple[int, float | None, float, str]:
    """(n, effect = candidate - baseline mean, raw p, method) over the paired items."""
    ids = _paired(base, cand)
    if not ids:
        return 0, None, 1.0, "none (no paired items)"
    effect = statistics.fmean(cand[i] - base[i] for i in ids)
    if spec.binary:
        only_base = sum(1 for i in ids if base[i] > cand[i])
        only_cand = sum(1 for i in ids if cand[i] > base[i])
        p = stats.mcnemar_exact(only_base, only_cand)
        return len(ids), effect, p, f"exact McNemar (b={only_base}, c={only_cand})"
    res = stats.sign_flip_test([cand[i] - base[i] for i in ids])
    how = "exact" if res.exact else "Monte Carlo 20000, seed 0"
    return len(ids), effect, res.p_value, f"paired sign-flip permutation ({how})"


def _aa_pairs(aa_runs: Sequence[RunFile], candidate: RunFile) -> list[tuple[RunFile, RunFile]]:
    """Consecutive A/A pairs that are comparable with the candidate (others are ignored)."""
    usable = [r for r in aa_runs if not comparability_problems(r, candidate)]
    return list(itertools.pairwise(usable))


def _mde(
    spec: _Spec, n: int, extract: Extractor, aa: Sequence[tuple[RunFile, RunFile]], ds: Datasets
) -> tuple[float | None, str]:
    """MDE at this check's n from the A/A noise: the discordance rate for a binary metric, the
    SD of per-item differences otherwise (floored at the exact test's discreteness)."""
    if not aa or n == 0:
        return None, "MDE unknown: no comparable A/A pair"
    diffs: list[float] = []
    for first, second in aa:
        a, b = extract(first, ds)[spec.name], extract(second, ds)[spec.name]
        diffs.extend(b[i] - a[i] for i in _paired(a, b))
    if not diffs:
        return None, "MDE unknown: the A/A pair shares no items"
    if spec.binary:
        psi = sum(1 for d in diffs if d != 0) / len(diffs)
        return stats.mde_from_discordance(psi, n, alpha=spec.alpha), f"A/A discordance {psi:.3f}"
    sd = statistics.pstdev(diffs)
    if sd == 0:
        # no A/A noise to scale by; a continuous effect has no discreteness floor in its units
        m = stats.min_discordant_for_significance(spec.alpha)
        return None, f"MDE n/a: A/A identical (the test needs >= {m} items moving one way)"
    return stats.mde_from_paired_sd(sd, n, alpha=spec.alpha), f"A/A per-item SD {sd:.3g}"


# Above this share of a check's baseline items lost to provider-transient errors (after the
# runner's retries), the candidate is unmeasured, not passed (integrator fix of review P1).
MAX_TRANSIENT_SHARE = 0.2


def _decide(r: _Result, adjusted_p: float | None) -> GateOutcome:
    if r.unmeasured:
        return GateOutcome.not_evaluable
    if r.missing:
        return r.spec.severity  # the candidate measured less than the baseline
    if r.n == 0 or r.effect is None or adjusted_p is None:
        # nothing paired and nothing missing: neither arm records this metric. Never a silent
        # pass, never a false block (a baseline-only metric is `missing` above and blocks)
        return GateOutcome.warn
    worse = r.effect < -r.spec.min_drop if r.spec.higher_is_better else r.effect > r.spec.min_drop
    if worse and adjusted_p < r.spec.alpha:
        return r.spec.severity
    return GateOutcome.pass_


def _paired_checks(
    baseline: RunFile,
    candidate: RunFile,
    aa: Sequence[tuple[RunFile, RunFile]],
    datasets: Datasets,
) -> list[GateCheck]:
    extract, specs = _PLAN[candidate.header.eval_kind]
    transient = _transient_ids(baseline) | _transient_ids(candidate)
    stale = (_stale_ids(baseline) | _stale_ids(candidate)) - transient
    excluded = transient | stale
    base_all, cand_all = extract(baseline, datasets), extract(candidate, datasets)
    results: list[_Result] = []
    for spec in specs:
        base = {i: v for i, v in base_all[spec.name].items() if i not in excluded}
        cand = {i: v for i, v in cand_all[spec.name].items() if i not in excluded}
        n, effect, p, method = _paired_test(spec, base, cand)
        mde, mde_note = _mde(spec, n, extract, aa, datasets)
        missing = len(set(base) - set(cand))
        r = _Result(spec, n, effect, p if n else None, method, mde, missing, [mde_note])
        base_raw = base_all[spec.name]
        lost = sum(1 for i in base_raw if i in transient)
        paired = set(base) & set(cand)  # items measured in BOTH arms (not `n`, the test count)
        if base_raw and (not paired or lost / len(base_raw) > MAX_TRANSIENT_SHARE):
            # transients that survived the retries are missing measurement: never a pass
            r.unmeasured = True
            r.notes.append(
                f"{lost}/{len(base_raw)} item(s) lost to provider-transient errors: unmeasured"
            )
        if missing:
            r.notes.append(f"{missing} baseline item(s) have no candidate value")
        if transient:
            r.notes.append(f"{len(transient)} provider-transient item(s) excluded")
        if stale:
            r.notes.append(f"{len(stale)} stale item(s) excluded (stale in either arm)")
        if n == 0:
            r.notes.append(
                "no paired items: nothing to compare"
                + (" (neither arm records this metric)" if not base_all[spec.name] else "")
            )
        results.append(r)
    checks: list[GateCheck] = []
    for family in dict.fromkeys(r.spec.family for r in results):
        members = [r for r in results if r.spec.family == family and r.p is not None]
        adjusted = dict(
            zip(
                map(id, members), stats.holm([r.p for r in members if r.p is not None]), strict=True
            )
        )
        for r in (r for r in results if r.spec.family == family):
            adj = adjusted.get(id(r))
            if adj is not None and len(members) > 1:
                r.notes.append(f"Holm-adjusted p={adj:.4g} (family of {len(members)})")
            checks.append(
                GateCheck(
                    name=r.spec.name,
                    outcome=_decide(r, adj),
                    method=r.method,
                    n=r.n,
                    effect=r.effect,
                    p_value=r.p,
                    mde=r.mde,
                    note="; ".join(r.notes),
                )
            )
    return checks


# ------------------------------------------------------------------ completeness
def _coverage_check(baseline: RunFile, candidate: RunFile) -> GateCheck:
    """Both runs finished and the candidate covers every baseline item (any row counts: an
    errored or stale row is accounted for by the checks; an ABSENT row is not). Not evaluable
    otherwise, because the paired tests only see the intersection of the two arms."""
    base_ids = {r.item_id for r in baseline.rows}
    missing = base_ids - {r.item_id for r in candidate.rows}
    problems: list[str] = []
    if candidate.footer is None:
        problems.append("candidate run has no footer (crashed or unfinished)")
    if baseline.footer is None:
        problems.append("baseline run has no footer (unfinished)")
    if not base_ids:
        problems.append("baseline has no rows: nothing to compare against")
    if missing:
        problems.append(f"{len(missing)} of {len(base_ids)} baseline item(s) absent from candidate")
    return GateCheck(
        name="coverage",
        outcome=GateOutcome.not_evaluable if problems else GateOutcome.pass_,
        method="item-set and footer completeness",
        n=len(base_ids),
        effect=float(len(missing)),
        note=("; ".join(problems) or "every baseline item present, both runs finished")
        + "; MDE n/a",
    )


# ------------------------------------------------------------------ absolute checks
def _errors_check(candidate: RunFile) -> GateCheck:
    sel = metrics.select_rows(candidate)
    warn_only = candidate.header.eval_kind in (EvalKind.interview, EvalKind.latency)
    bad = GateOutcome.warn if warn_only else GateOutcome.block
    return GateCheck(
        name="errors.ours",
        outcome=bad if sel.ours_any_attempt else GateOutcome.pass_,
        method="count (hard floor: none of our own, on any attempt)",
        n=len(sel.rows) + sel.errors_ours + sel.errors_transient,
        effect=float(sel.ours_any_attempt),
        note=f"{sel.errors_transient} provider-transient, {sel.stale} stale (excluded); MDE n/a",
    )


def _floor_checks(
    candidate: RunFile, datasets: Datasets, floors: Mapping[str, float] | None
) -> list[GateCheck]:
    """Retrieval floors (ADR-0012 `frozen.json`): block when a metric falls below floor - 3.

    The floor is ABSOLUTE and configured, never the baseline's own value: a baseline-relative
    floor turns every drop over 3 points into a block whatever its p (at n = 12 one flipped
    question is 8.3 points), which defeats the significance test and makes a noisy A/A fail.
    With no floor configured the check passes and says so in its note."""
    cand = metrics.metric_map(metrics.summarize(candidate, datasets, [], []))
    checks: list[GateCheck] = []
    for name in (RETRIEVAL_HIT, RETRIEVAL_MRR):
        value = cand[name].value if name in cand else None
        floor = floors.get(name) if floors else None
        effect: float | None = None
        if floor is None:
            outcome, note = GateOutcome.pass_, "no absolute floor configured; MDE n/a"
        elif value is None:
            outcome, note = GateOutcome.warn, "no value to compare with the floor; MDE n/a"
        else:
            effect = value - floor
            below = value < floor - FLOOR_MARGIN
            outcome = GateOutcome.block if below else GateOutcome.pass_
            note = f"candidate {value:.4f} vs floor {floor:.4f} - {FLOOR_MARGIN:.2f}; MDE n/a"
        checks.append(
            GateCheck(
                name=f"floor.{name}",
                outcome=outcome,
                method="absolute floor (frozen)",
                n=cand[name].n if name in cand else 0,
                effect=effect,
                note=note,
            )
        )
    return checks


def _slo_checks(candidate: RunFile, datasets: Datasets) -> list[GateCheck]:
    """The absolute E-V SLO on the candidate (ADR-0017 §3). The bounds come from
    `metrics.clustered_mean`, which widens a cluster-bootstrap bound to the exact Clopper-Pearson
    bound whenever that is more conservative: a percentile bootstrap of 3/3 says [1, 1], which
    would certify a verifier model on almost no evidence."""
    m = metrics.metric_map(metrics.summarize(candidate, datasets, [], []))
    flip = m["verifier.badge_flip_recall"]
    reject = m["verifier.false_reject.verifier_caused"]
    flip_ok = flip.ci_low is not None and flip.ci_low >= SLO_BADGE_FLIP_LOW
    reject_ok = reject.ci_high is not None and reject.ci_high <= SLO_VERIFIER_FALSE_REJECT_HIGH
    return [
        GateCheck(
            name="slo.verifier.badge_flip_recall.ci_low",
            outcome=GateOutcome.pass_ if flip_ok else GateOutcome.block,
            method=f"absolute SLO: CI low >= {SLO_BADGE_FLIP_LOW} ({flip.method.value})",
            n=flip.n,
            effect=flip.ci_low,
            note=f"recall {flip.value} [{flip.ci_low}, {flip.ci_high}]; MDE n/a",
        ),
        GateCheck(
            name="slo.verifier.false_reject.verifier_caused.ci_high",
            outcome=GateOutcome.pass_ if reject_ok else GateOutcome.block,
            method=f"absolute SLO: CI high <= {SLO_VERIFIER_FALSE_REJECT_HIGH} "
            f"({reject.method.value})",
            n=reject.n,
            effect=reject.ci_high,
            note=f"false-reject {reject.value} [{reject.ci_low}, {reject.ci_high}]; MDE n/a",
        ),
    ]


def _level_trend_check(baseline: RunFile, candidate: RunFile, datasets: Datasets) -> GateCheck:
    """Warn when the baseline's level trend was significant AND the candidate's per-block L is
    significantly lower (one-sided paired sign-flip over blocks: the question is directional).
    'Not significant' alone never warns."""
    base = metrics.level_blocks(metrics.interview_rows(baseline, datasets))
    cand = metrics.level_blocks(metrics.interview_rows(candidate, datasets))
    ids = sorted(set(base) & set(cand))
    if not ids:
        return GateCheck(
            name="interview.level.page_l",
            outcome=GateOutcome.pass_,
            method="Page's L",
            n=0,
            note="no common complete blocks; MDE n/a",
        )
    base_test = stats.page_l_test([base[i] for i in ids])
    diffs = [stats.page_l([cand[i]]) - stats.page_l([base[i]]) for i in ids]
    lower = stats.sign_flip_test(diffs, alternative="less")
    effect = math.fsum(diffs)
    warn = base_test.p_value < WARN_ALPHA and effect < 0 and lower.p_value < WARN_ALPHA
    return GateCheck(
        name="interview.level.page_l",
        outcome=GateOutcome.warn if warn else GateOutcome.pass_,
        method="Page's L exact permutation (baseline) + one-sided paired sign-flip over blocks",
        n=len(ids),
        effect=effect,
        p_value=lower.p_value,
        note=f"baseline trend p={base_test.p_value:.4g}; MDE n/a for a rank statistic",
    )


def _aa_retrieval_identical(aa: Sequence[tuple[RunFile, RunFile]]) -> GateCheck | None:
    """§5: with chunk_id tie-breaks an A/A pair must retrieve identically; a difference means
    host near-ties leak nondeterminism into every comparison, so it is surfaced (warn)."""
    if not aa:
        return None
    differing = 0
    total = 0
    for first, second in aa:
        a = {i: r.output for i, r in metrics.select_rows(first).rows.items()}
        b = {i: r.output for i, r in metrics.select_rows(second).rows.items()}
        for i in sorted(set(a) & set(b)):
            total += 1
            oa, ob = a[i], b[i]
            if not (isinstance(oa, RetrievalOutput) and isinstance(ob, RetrievalOutput)):
                continue
            differing += oa.stages != ob.stages
    return GateCheck(
        name="aa.retrieval_identical",
        outcome=GateOutcome.warn if differing else GateOutcome.pass_,
        method="A/A equality of every stage ranking",
        n=total,
        effect=float(differing),
        note=f"{differing} of {total} questions retrieved differently; MDE n/a",
    )


# ------------------------------------------------------------------ entry point
def _overall(checks: Sequence[GateCheck]) -> GateOutcome:
    for outcome in (GateOutcome.not_evaluable, GateOutcome.block, GateOutcome.warn):
        if any(c.outcome is outcome for c in checks):
            return outcome
    return GateOutcome.pass_


def evaluate(
    baseline: RunFile,
    candidate: RunFile,
    aa_runs: Sequence[RunFile],
    datasets: Datasets,
    *,
    floors: Mapping[str, float] | None = None,
) -> GateVerdict:
    """The gate verdict for `candidate` against `baseline`. `aa_runs` are two (or more) runs of
    one build on the same snapshot; consecutive pairs give the MDE. Pure and deterministic."""
    problems = comparability_problems(baseline, candidate)
    if not problems:
        try:
            return _evaluate(baseline, candidate, aa_runs, datasets, floors)
        except metrics.MetricsError as exc:
            problems = [str(exc)]
    return GateVerdict(
        surface=candidate.header.surface,
        baseline_run=baseline.header.run_id,
        candidate_run=candidate.header.run_id,
        checks=[
            GateCheck(
                name="comparable",
                outcome=GateOutcome.not_evaluable,
                method="header comparison",
                n=0,
                note="; ".join(problems),
            )
        ],
        outcome=GateOutcome.not_evaluable,
        override=candidate.header.override,
    )


def _evaluate(
    baseline: RunFile,
    candidate: RunFile,
    aa_runs: Sequence[RunFile],
    datasets: Datasets,
    floors: Mapping[str, float] | None,
) -> GateVerdict:
    kind = candidate.header.eval_kind
    aa = _aa_pairs(aa_runs, candidate)
    checks = [
        _coverage_check(baseline, candidate),
        _errors_check(candidate),
        *_paired_checks(baseline, candidate, aa, datasets),
    ]
    if kind is EvalKind.retrieval:
        checks.extend(_floor_checks(candidate, datasets, floors))
        identical = _aa_retrieval_identical(aa)
        if identical is not None:
            checks.append(identical)
    elif kind is EvalKind.verifier:
        checks.extend(_slo_checks(candidate, datasets))
    elif kind is EvalKind.interview:
        checks.append(_level_trend_check(baseline, candidate, datasets))
    return GateVerdict(
        surface=candidate.header.surface,
        baseline_run=baseline.header.run_id,
        candidate_run=candidate.header.run_id,
        checks=checks,
        outcome=_overall(checks),
        override=candidate.header.override,
    )


def passes(verdict: GateVerdict) -> bool:
    """Exit-code semantics: pass/warn pass; a block passes only with a recorded override; a
    not_evaluable comparison never passes (an override cannot make incomparable runs comparable)."""
    if verdict.outcome in (GateOutcome.pass_, GateOutcome.warn):
        return True
    return verdict.outcome is GateOutcome.block and bool(verdict.override)


def all_pass(verdicts: Sequence[GateVerdict]) -> bool:
    """ADR-0017 §5: on a chat or verifier change the chat verdict and the verifier-surface verdict
    (the E-V hard floor) must BOTH pass; `evaluate` judges one run pair of one kind, so the CLI
    composes them here. An empty sequence is not a pass (nothing was judged)."""
    return bool(verdicts) and all(passes(v) for v in verdicts)


def format_verdict(verdict: GateVerdict) -> str:
    """One line per check with its n, method, effect, p and MDE (ADR-0017 §5 'Output')."""

    def num(x: float | None) -> str:
        return "-" if x is None else f"{x:.4g}"

    lines = [
        f"gate {verdict.surface.value}: {verdict.outcome.value} "
        f"({verdict.candidate_run} vs {verdict.baseline_run})"
    ]
    if verdict.override:
        lines.append(f"  override: {verdict.override}")
    for c in verdict.checks:
        lines.append(
            f"  [{c.outcome.value}] {c.name}: n={c.n} effect={num(c.effect)} p={num(c.p_value)} "
            f"MDE={num(c.mde)} method={c.method}" + (f" ({c.note})" if c.note else "")
        )
    return "\n".join(lines)
