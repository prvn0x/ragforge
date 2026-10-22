"""Pure statistics for the eval layer (ADR-0017 §3-§5). Standard library only: numpy/scipy are not
dependencies, and at this project's n (tens of items) exact enumeration beats asymptotics anyway.

Conventions:
- Every interval is two-sided at `alpha` (default 0.05) and returned as `(low, high)`.
- Every resampling function takes a `seed` and uses its own `random.Random`, so a summary or a gate
  verdict is reproducible bit for bit (the Eval page and the gate must agree on the same file).
- Exact methods are preferred whenever the enumeration is cheap; the Monte Carlo fallback says so
  in its result, because the gate prints the method next to every p-value.
"""

from __future__ import annotations

import itertools
import math
import random
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from statistics import NormalDist
from typing import Literal

ALPHA = 0.05
N_RESAMPLES = 2000
# A kappa bootstrap where under half the resamples are defined (Pe = 1 when a resample holds one
# category) describes the surviving resamples, not the data: no CI rather than a biased one.
KAPPA_MIN_DEFINED_SHARE = 0.5
Alternative = Literal["two-sided", "less", "greater"]
_STD_NORMAL = NormalDist()


def z_for(alpha: float) -> float:
    """The two-sided normal critical value (1.959964 at alpha 0.05)."""
    return _STD_NORMAL.inv_cdf(1 - alpha / 2)


# ------------------------------------------------------------------ binomial intervals
def wilson(k: int, n: int, alpha: float = ALPHA) -> tuple[float, float]:
    """Wilson score interval for k/n. Chosen over Wald because it stays inside [0, 1] and keeps
    near-nominal coverage at the small n (10-50) every eval here has."""
    _check_counts(k, n)
    z = z_for(alpha)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    # the closed forms are exactly 0 / 1 at the edges; pin them so rounding never prints 0.9999
    low = 0.0 if k == 0 else max(0.0, centre - half)
    high = 1.0 if k == n else min(1.0, centre + half)
    return low, high


def binom_cdf(k: int, n: int, p: float) -> float:
    """P(X <= k) for X ~ Binomial(n, p), summed in log space so n in the thousands is safe."""
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    if p <= 0.0:
        return 1.0
    if p >= 1.0:
        return 0.0
    log_p, log_q = math.log(p), math.log1p(-p)
    terms = [_log_comb(n, i) + i * log_p + (n - i) * log_q for i in range(k + 1)]
    top = max(terms)
    return min(1.0, math.exp(top) * math.fsum(math.exp(t - top) for t in terms))


def clopper_pearson(k: int, n: int, alpha: float = ALPHA) -> tuple[float, float]:
    """Exact (Clopper-Pearson) interval for k/n, by inverting the binomial tails with bisection.

    Used where clusters are too few for a bootstrap (E-I ranking over ~10 seeds) and for
    confirmed false-verified k/n. Conservative by construction: coverage is never below 1-alpha.
    """
    _check_counts(k, n)
    tail = alpha / 2
    # lower: P(X >= k | p) = tail, i.e. 1 - cdf(k - 1) = tail; increasing in p
    low = 0.0 if k == 0 else _bisect(lambda p: 1 - binom_cdf(k - 1, n, p) - tail)
    # upper: P(X <= k | p) = tail; decreasing in p
    high = 1.0 if k == n else _bisect(lambda p: tail - binom_cdf(k, n, p))
    return low, high


def _bisect(f: Callable[[float], float]) -> float:
    """Root of an increasing f on (0, 1); 200 halvings reach float resolution."""
    lo, hi = 0.0, 1.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if f(mid) < 0:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-15:
            break
    return (lo + hi) / 2


def _log_comb(n: int, k: int) -> float:
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def _check_counts(k: int, n: int) -> None:
    if n <= 0 or not 0 <= k <= n:
        raise ValueError(f"need 0 <= k <= n and n > 0, got k={k}, n={n}")


# ------------------------------------------------------------------ bootstrap
def quantile(sorted_values: Sequence[float], q: float) -> float:
    """Linear-interpolation quantile (Hyndman-Fan type 7, numpy's default) of sorted values."""
    if not sorted_values:
        raise ValueError("quantile of an empty sequence")
    pos = (len(sorted_values) - 1) * q
    lo = math.floor(pos)
    hi = min(lo + 1, len(sorted_values) - 1)
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (pos - lo)


def bootstrap_ci[T](
    units: Sequence[T],
    statistic: Callable[[Sequence[T]], float | None],
    *,
    n_resamples: int = N_RESAMPLES,
    seed: int = 0,
    alpha: float = ALPHA,
    min_defined: int = 1,
) -> tuple[float, float] | None:
    """Percentile bootstrap: resample whole `units` with replacement, recompute `statistic`.

    A unit is whatever is independent: a source question's cluster of mutations (E-V), one
    labelled item's (owner, judge) pair (kappa). Resamples where the statistic is undefined (kappa
    with no spread) are dropped; None when fewer than 2 units or no resample is defined, because
    an interval from one unit would only restate the point value. `min_defined` raises the bar:
    fewer defined resamples than that also gives None.
    """
    if len(units) < 2:
        return None
    rng = random.Random(seed)  # noqa: S311 - seeded resampling, not crypto
    stats: list[float] = []
    for _ in range(n_resamples):
        value = statistic(rng.choices(units, k=len(units)))
        if value is not None:
            stats.append(value)
    if not stats or len(stats) < min_defined:
        return None
    stats.sort()
    return quantile(stats, alpha / 2), quantile(stats, 1 - alpha / 2)


def pooled_mean(clusters: Sequence[Sequence[float]]) -> float | None:
    """The item-level mean over every item of every cluster (a ratio estimator, not a mean of
    cluster means: big clusters weigh more, exactly as in the point estimate)."""
    total = sum(len(c) for c in clusters)
    return math.fsum(v for c in clusters for v in c) / total if total else None


def cluster_bootstrap(
    clusters: Sequence[Sequence[float]],
    *,
    n_resamples: int = N_RESAMPLES,
    seed: int = 0,
    alpha: float = ALPHA,
) -> tuple[float, float] | None:
    """Bootstrap CI of the pooled item mean, resampling CLUSTERS (ADR-0017 D3): the mutations of
    one answer are correlated, so iid resampling of items would make the CI falsely narrow."""
    return bootstrap_ci(clusters, pooled_mean, n_resamples=n_resamples, seed=seed, alpha=alpha)


# ------------------------------------------------------------------ paired tests
def mcnemar_exact(only_a: int, only_b: int) -> float:
    """Exact two-sided McNemar: a binomial(b + c, 1/2) test on the discordant pairs only.

    Exact integer arithmetic: at the gate's n the chi-square approximation is badly off."""
    if only_a < 0 or only_b < 0:
        raise ValueError("discordant counts must be non-negative")
    m = only_a + only_b
    if m == 0:
        return 1.0
    tail = sum(math.comb(m, i) for i in range(min(only_a, only_b) + 1))
    return min(1.0, 2 * tail / (1 << m))


@dataclass(frozen=True)
class SignTest:
    positive: int
    negative: int
    zeros: int
    p_value: float


def sign_test(differences: Sequence[float]) -> SignTest:
    """Exact paired sign test (zeros dropped). Used for judge leniency (judge - owner) and the
    pooled SDE-1 vs Staff level check: only the direction of each pair is trusted, not its size."""
    pos = sum(1 for d in differences if d > 0)
    neg = sum(1 for d in differences if d < 0)
    return SignTest(pos, neg, len(differences) - pos - neg, mcnemar_exact(pos, neg))


@dataclass(frozen=True)
class PermutationResult:
    statistic: float
    p_value: float
    exact: bool  # False: Monte Carlo with the stated seed and resamples
    n_nonzero: int


def sign_flip_test(
    differences: Sequence[float],
    *,
    exact_limit: int = 16,
    n_resamples: int = 20000,
    seed: int = 0,
    alternative: Alternative = "two-sided",
) -> PermutationResult:
    """Paired sign-flip permutation test of mean difference 0.

    Under H0 each paired difference is symmetric about 0, so flipping its sign is free. Exact
    (all 2^m sign patterns of the m non-zero differences) when m <= `exact_limit`; otherwise
    Monte Carlo with the add-one estimator, which never reports p = 0. Two-sided on |sum| by
    default; `less` / `greater` give the one-sided P(S* <= S) / P(S* >= S) for a directional
    question (a warn on "lower" should not pay the two-sided price).
    """
    nonzero = [d for d in differences if d != 0]
    total = math.fsum(nonzero)
    m = len(nonzero)
    if m == 0:
        return PermutationResult(0.0, 1.0, True, 0)
    tol = 1e-9 * max(1.0, abs(total))
    observed = abs(total) if alternative == "two-sided" else total

    def extreme(s: float) -> bool:
        if alternative == "two-sided":
            return abs(s) >= observed - tol
        if alternative == "greater":
            return s >= observed - tol
        return s <= observed + tol

    if m <= exact_limit:
        sums = [0.0]
        for d in nonzero:
            sums = [s + d for s in sums] + [s - d for s in sums]
        count = sum(1 for s in sums if extreme(s))
        return PermutationResult(observed, count / len(sums), True, m)
    rng = random.Random(seed)  # noqa: S311 - seeded resampling, not crypto
    count = sum(
        1
        for _ in range(n_resamples)
        if extreme(math.fsum(d if rng.random() < 0.5 else -d for d in nonzero))
    )
    return PermutationResult(observed, (count + 1) / (n_resamples + 1), False, m)


def holm(p_values: Sequence[float]) -> list[float]:
    """Holm step-down adjusted p-values (same order as given): controls the family-wise error
    without Bonferroni's waste, and needs no independence assumption."""
    m = len(p_values)
    order = sorted(range(m), key=lambda i: p_values[i])
    adjusted = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p_values[i])
        adjusted[i] = min(1.0, running)
    return adjusted


# ------------------------------------------------------------------ Page's L
def average_ranks(values: Sequence[float]) -> list[float]:
    """1-based ranks, ties given their average rank (integer interview scores tie often)."""
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


def page_l(blocks: Sequence[Sequence[float]]) -> float:
    """Page's L = sum over blocks of sum_k k * R_k, with treatments listed in the PREDICTED
    increasing order and ranks averaged on ties."""
    return math.fsum(
        (k + 1) * r for block in blocks for k, r in enumerate(average_ranks(list(block)))
    )


def page_l_test(blocks: Sequence[Sequence[float]]) -> PermutationResult:
    """Page's L with its EXACT one-sided permutation p-value P(L* >= L_obs).

    Under H0 the values are exchangeable within each block, so a block's L contribution is
    uniform over the k! permutations of its (tie-averaged) ranks, and blocks are independent:
    the null distribution is the convolution of the per-block distributions. Ranks are multiples
    of 1/2, so 2L is an integer and the convolution is exact. Cheap for k <= 8 at any number of
    blocks, which is why no Monte Carlo fallback is needed (the ADR's 5 levels give 120 perms).
    """
    if not blocks:
        raise ValueError("Page's L needs at least one block")
    k = len(blocks[0])
    if k < 2 or k > 8 or any(len(b) != k for b in blocks):
        raise ValueError("Page's L needs blocks of one common length 2..8")
    observed2 = round(2 * page_l(blocks))
    dist: Counter[int] = Counter({0: 1})
    for block in blocks:
        ranks2 = [round(2 * r) for r in average_ranks(list(block))]
        per_block: Counter[int] = Counter(
            sum((pos + 1) * ranks2[src] for pos, src in enumerate(perm))
            for perm in itertools.permutations(range(k))
        )
        nxt: Counter[int] = Counter()
        for a, ca in dist.items():
            for b, cb in per_block.items():
                nxt[a + b] += ca * cb
        dist = nxt
    total = sum(dist.values())
    extreme = sum(c for v, c in dist.items() if v >= observed2)
    return PermutationResult(observed2 / 2, extreme / total, True, len(blocks))


# ------------------------------------------------------------------ agreement
def linear_weighted_kappa(pairs: Sequence[tuple[int, int]], categories: int = 3) -> float | None:
    """Cohen's kappa with linear weights w_ij = 1 - |i - j| / (c - 1) over (owner, judge) scores.

    Linear, not quadratic: the 0/1/2 anchors are ordinal steps of equal cost. None when n = 0 or
    when chance agreement is total (Pe = 1, kappa undefined), never a silent 0 or 1."""
    n = len(pairs)
    if not n:
        return None
    table = [[0] * categories for _ in range(categories)]
    for a, b in pairs:
        table[a][b] += 1
    span = categories - 1
    w = [[1 - abs(i - j) / span for j in range(categories)] for i in range(categories)]
    rows = [sum(table[i]) / n for i in range(categories)]
    cols = [sum(table[i][j] for i in range(categories)) / n for j in range(categories)]
    cells = [(i, j) for i in range(categories) for j in range(categories)]
    po = sum(w[i][j] * table[i][j] / n for i, j in cells)
    pe = sum(w[i][j] * rows[i] * cols[j] for i, j in cells)
    return None if pe == 1 else (po - pe) / (1 - pe)


def kappa_ci(
    pairs: Sequence[tuple[int, int]],
    *,
    n_resamples: int = N_RESAMPLES,
    seed: int = 0,
    alpha: float = ALPHA,
) -> tuple[float, float] | None:
    """Percentile bootstrap CI of the linear-weighted kappa, resampling labelled items.

    None when the bootstrap cannot speak: under half the resamples define kappa, or the interval
    has zero width. With 2-3 perfectly agreeing labels every defined resample is 1.0, and a
    [1, 1] interval would pass any 'CI low >= 0.4' rule on no evidence (review r2 P1)."""
    ci = bootstrap_ci(
        pairs,
        linear_weighted_kappa,
        n_resamples=n_resamples,
        seed=seed,
        alpha=alpha,
        min_defined=math.ceil(KAPPA_MIN_DEFINED_SHARE * n_resamples),
    )
    if ci is None or ci[0] == ci[1]:
        return None
    return ci


# ------------------------------------------------------------------ MDE
def min_discordant_for_significance(alpha: float) -> int:
    """The fewest discordant pairs, all one way, for which the exact McNemar can reach p < alpha.
    Below it the test cannot fire however large the effect: the gate's discreteness floor."""
    m = 1
    while mcnemar_exact(m, 0) >= alpha:
        m += 1
    return m


def mde_from_discordance(
    discordance: float, n: int, *, alpha: float = ALPHA, power: float = 0.8
) -> float | None:
    """Minimum detectable paired difference in a proportion, from the A/A discordance rate.

    Normal approximation for McNemar: delta = (z_{1-a/2} + z_power) * sqrt(psi / n), where psi is
    the share of items that flip between two runs of the SAME build (pure noise). It is floored
    at the exact test's discreteness (`min_discordant_for_significance / n`) because an A/A pair
    with no flips would otherwise claim an MDE of 0, and capped at 1. None when n = 0.
    """
    if n <= 0:
        return None
    normal = (z_for(alpha) + _STD_NORMAL.inv_cdf(power)) * math.sqrt(max(0.0, discordance) / n)
    floor = min_discordant_for_significance(alpha) / n
    return min(1.0, max(normal, floor))


def mde_from_paired_sd(
    sd: float, n: int, *, alpha: float = ALPHA, power: float = 0.8
) -> float | None:
    """MDE of a paired mean difference (e.g. MRR) from the SD of A/A per-item differences."""
    if n <= 0:
        return None
    return (z_for(alpha) + _STD_NORMAL.inv_cdf(power)) * sd / math.sqrt(n)
