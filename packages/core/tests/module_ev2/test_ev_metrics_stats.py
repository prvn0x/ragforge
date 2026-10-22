"""EV-METRICS statistics: textbook values, exactness against brute force at small n, and
hypothesis properties (bounds, symmetry, monotonicity, determinism)."""

from __future__ import annotations

import itertools
import math
from fractions import Fraction

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from test_ev_metrics_support import reference

from ragforge_client.evals import stats

counts = st.integers(min_value=1, max_value=200).flatmap(
    lambda n: st.tuples(st.integers(min_value=0, max_value=n), st.just(n))
)
score_pairs = st.lists(st.tuples(st.integers(0, 2), st.integers(0, 2)), min_size=1, max_size=40)
small_diffs = st.lists(st.integers(-5, 5).map(float), min_size=0, max_size=10)


# ------------------------------------------------------------------ binomial intervals
# Newcombe (1998), "Two-sided confidence intervals for the single proportion", Table I:
# method 3 (Wilson score, no continuity correction) and method 5 (exact, Clopper-Pearson).
NEWCOMBE = [
    # k, n, wilson, clopper-pearson
    (15, 148, (0.0624, 0.1605), (0.0578, 0.1617)),
    (0, 20, (0.0, 0.1611), (0.0, 0.1684)),
    (1, 29, (0.0061, 0.1718), (0.0009, 0.1776)),
]


@pytest.mark.parametrize(("k", "n", "wil", "cp"), NEWCOMBE)
def test_intervals_match_newcombe_1998(
    k: int, n: int, wil: tuple[float, float], cp: tuple[float, float]
) -> None:
    assert stats.wilson(k, n) == pytest.approx(wil, abs=5e-5)
    assert stats.clopper_pearson(k, n) == pytest.approx(cp, abs=5e-5)


def test_wilson_newcombe_81_of_263() -> None:
    assert stats.wilson(81, 263) == pytest.approx((0.2553, 0.3662), abs=5e-5)


@settings(max_examples=40, deadline=None)
@given(st.integers(1, 40).flatmap(lambda n: st.tuples(st.integers(0, n), st.just(n))))
def test_clopper_pearson_bounds_solve_the_exact_tail_equations(kn: tuple[int, int]) -> None:
    """The definition itself, checked in exact rational arithmetic: P(X >= k | low) = alpha/2
    and P(X <= k | high) = alpha/2 (independent of the bisection inside the implementation)."""
    k, n = kn
    low, high = stats.clopper_pearson(k, n)

    def cdf(j: int, p: Fraction) -> Fraction:
        return sum((math.comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(j + 1)), Fraction(0))

    if k > 0:
        assert float(1 - cdf(k - 1, Fraction(low))) == pytest.approx(0.025, abs=1e-9)
    if k < n:
        assert float(cdf(k, Fraction(high))) == pytest.approx(0.025, abs=1e-9)


def test_clopper_pearson_closed_forms_at_the_edges() -> None:
    # k = 0: upper = 1 - (alpha/2)^(1/n); k = n: lower = (alpha/2)^(1/n)
    assert stats.clopper_pearson(0, 10)[1] == pytest.approx(1 - 0.025 ** (1 / 10), abs=1e-12)
    assert stats.clopper_pearson(10, 10)[0] == pytest.approx(0.025 ** (1 / 10), abs=1e-12)
    assert stats.clopper_pearson(5, 10) == pytest.approx((0.187086, 0.812914), abs=1e-6)


@settings(deadline=None)  # exact intervals at n=200 are slow under a loaded machine
@given(counts)
def test_intervals_are_ordered_bounded_and_contain_the_point(kn: tuple[int, int]) -> None:
    k, n = kn
    for low, high in (stats.wilson(k, n), stats.clopper_pearson(k, n)):
        assert 0.0 <= low <= k / n <= high <= 1.0
    assert stats.wilson(k, n)[0] == (0.0 if k == 0 else stats.wilson(k, n)[0])
    assert stats.clopper_pearson(k, n)[1] == (1.0 if k == n else stats.clopper_pearson(k, n)[1])


@settings(deadline=None)  # exact intervals at n=200 are slow under a loaded machine
@given(counts)
def test_intervals_are_symmetric_under_k_to_n_minus_k(kn: tuple[int, int]) -> None:
    k, n = kn
    for f in (stats.wilson, stats.clopper_pearson):
        low, high = f(k, n)
        mlow, mhigh = f(n - k, n)
        assert low == pytest.approx(1 - mhigh, abs=1e-9)
        assert high == pytest.approx(1 - mlow, abs=1e-9)


@given(
    st.integers(1, 30).flatmap(lambda n: st.tuples(st.integers(0, n), st.just(n))),
    st.fractions(min_value=Fraction(1, 100), max_value=Fraction(99, 100)),
)
def test_binomial_cdf_matches_exact_rational_arithmetic(kn: tuple[int, int], p: Fraction) -> None:
    k, n = kn
    exact = sum(math.comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(k + 1))
    assert stats.binom_cdf(k, n, float(p)) == pytest.approx(float(exact), abs=1e-12)


def test_bad_counts_are_refused() -> None:
    for k, n in ((1, 0), (-1, 5), (6, 5)):
        with pytest.raises(ValueError, match="0 <= k <= n"):
            stats.wilson(k, n)


# ------------------------------------------------------------------ McNemar / sign / Holm
def test_exact_mcnemar_known_values() -> None:
    assert stats.mcnemar_exact(0, 0) == 1.0
    assert stats.mcnemar_exact(0, 6) == pytest.approx(2 / 64)
    assert stats.mcnemar_exact(0, 5) == pytest.approx(2 / 32)
    assert stats.mcnemar_exact(3, 3) == 1.0
    # binomial(10, 1/2): P(X <= 1) = 11/1024, two-sided 22/1024
    assert stats.mcnemar_exact(1, 9) == pytest.approx(22 / 1024)


@given(st.integers(0, 60), st.integers(0, 60))
def test_mcnemar_is_symmetric_and_a_probability(b: int, c: int) -> None:
    p = stats.mcnemar_exact(b, c)
    assert p == stats.mcnemar_exact(c, b)
    assert 0 < p <= 1


@given(st.integers(0, 30), st.integers(0, 30))
def test_mcnemar_never_gets_more_significant_as_the_split_evens_out(b: int, c: int) -> None:
    if b < c:
        assert stats.mcnemar_exact(b, c) <= stats.mcnemar_exact(b + 1, c - 1) + 1e-15


def test_min_discordant_for_significance() -> None:
    assert stats.min_discordant_for_significance(0.05) == 6  # 2/64 < 0.05 <= 2/32
    assert stats.min_discordant_for_significance(0.025) == 7  # 2/128 < 0.025 <= 2/64


def test_sign_test_drops_zeros() -> None:
    res = stats.sign_test([1, 2, 0, 0, -1, 3])
    assert (res.positive, res.negative, res.zeros) == (3, 1, 2)
    assert res.p_value == stats.mcnemar_exact(3, 1)


def test_holm_textbook_example() -> None:
    assert stats.holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert stats.holm([0.5, 0.9]) == pytest.approx([1.0, 1.0])  # monotone: max(2 * 0.5, 0.9)
    assert stats.holm([0.2, 0.01]) == pytest.approx([0.2, 0.02])
    assert stats.holm([]) == []


@given(st.lists(st.floats(0, 1), min_size=1, max_size=12))
def test_holm_is_monotone_bounded_and_never_below_raw(ps: list[float]) -> None:
    adj = stats.holm(ps)
    assert all(a >= p - 1e-15 and a <= 1.0 for a, p in zip(adj, ps, strict=True))
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    assert all(adj[order[i]] <= adj[order[i + 1]] + 1e-15 for i in range(len(ps) - 1))
    assert min(adj) == pytest.approx(min(1.0, len(ps) * min(ps)))


# ------------------------------------------------------------------ sign-flip permutation
def _brute_sign_flip(diffs: list[float]) -> float:
    observed = abs(sum(diffs))
    hits = total = 0
    for signs in itertools.product((1, -1), repeat=len(diffs)):
        total += 1
        hits += abs(sum(s * d for s, d in zip(signs, diffs, strict=True))) >= observed - 1e-9
    return hits / total


@given(small_diffs)
def test_sign_flip_is_exact_at_small_n(diffs: list[float]) -> None:
    """Zeros cannot change |sum| under a flip, so dropping them leaves the exact p unchanged."""
    res = stats.sign_flip_test(diffs)
    assert res.exact
    assert res.p_value == pytest.approx(_brute_sign_flip(diffs) if diffs else 1.0)


@given(small_diffs)
def test_sign_flip_is_symmetric_and_a_probability(diffs: list[float]) -> None:
    p = stats.sign_flip_test(diffs).p_value
    assert 0 < p <= 1
    assert p == pytest.approx(stats.sign_flip_test([-d for d in diffs]).p_value)


def test_sign_flip_known_values_and_monte_carlo_agreement() -> None:
    assert stats.sign_flip_test([1.0] * 6).p_value == pytest.approx(2 / 64)
    assert stats.sign_flip_test([0.0, 0.0]).p_value == 1.0
    diffs = [0.5, 0.25, 1.0, 0.2, 0.1, 0.33, -0.1, 0.5, 0.25, 0.4, 0.2, 0.1, 0.3, -0.2, 0.6, 0.7]
    exact = stats.sign_flip_test(diffs).p_value
    mc = stats.sign_flip_test(diffs, exact_limit=0, n_resamples=20000, seed=7)
    assert not mc.exact
    assert mc.p_value == pytest.approx(exact, abs=0.01)
    again = stats.sign_flip_test(diffs, exact_limit=0, n_resamples=20000, seed=7)
    assert again == mc  # seeded: reproducible bit for bit


# ------------------------------------------------------------------ Page's L
def _brute_page(blocks: list[list[float]]) -> float:
    """P(L* >= L) by enumerating every within-block permutation of the VALUES."""
    observed = stats.page_l(blocks)
    hits = total = 0
    for perms in itertools.product(*(itertools.permutations(b) for b in blocks)):
        total += 1
        hits += stats.page_l([list(p) for p in perms]) >= observed - 1e-9
    return hits / total


block_sets = st.integers(2, 4).flatmap(
    lambda k: st.lists(
        st.lists(st.integers(0, 3).map(float), min_size=k, max_size=k), min_size=1, max_size=3
    )
)


@settings(max_examples=60, deadline=None)
@given(block_sets)
def test_page_l_permutation_p_is_exact_with_ties(blocks: list[list[float]]) -> None:
    res = stats.page_l_test(blocks)
    assert res.exact
    assert res.p_value == pytest.approx(_brute_page(blocks), abs=1e-12)


def test_page_l_textbook_values() -> None:
    # one block of 3 in the predicted order: L = 1*1 + 2*2 + 3*3 = 14, the max of 6 perms
    res = stats.page_l_test([[1.0, 2.0, 3.0]])
    assert res.statistic == 14.0
    assert res.p_value == pytest.approx(1 / 6)
    # all tied: every rank is 2, L = 2 * (1+2+3) = 12 under every perm -> p = 1
    assert stats.page_l_test([[5.0, 5.0, 5.0]]).p_value == 1.0
    # Page (1963): with no ties, E[L] = b k (k+1)^2 / 4; reversed order gives the minimum
    assert stats.page_l([[3.0, 2.0, 1.0]]) == 10.0


@given(block_sets)
def test_page_l_matches_the_oracle_and_average_ranks_sum_correctly(
    blocks: list[list[float]],
) -> None:
    ref = reference()
    for b in blocks:
        ranks = stats.average_ranks(b)
        assert ranks == ref.average_ranks(b)
        assert sum(ranks) == pytest.approx(len(b) * (len(b) + 1) / 2)


def test_page_l_refuses_ragged_blocks() -> None:
    with pytest.raises(ValueError, match="common length"):
        stats.page_l_test([[1.0, 2.0], [1.0, 2.0, 3.0]])


# ------------------------------------------------------------------ kappa
@given(score_pairs)
def test_kappa_matches_the_independent_oracle(pairs: list[tuple[int, int]]) -> None:
    got = stats.linear_weighted_kappa(pairs)
    want = reference().weighted_kappa(pairs)
    assert (got is None) == (want is None)
    if got is not None:
        assert got == pytest.approx(want, abs=1e-12)


@given(score_pairs)
def test_kappa_is_symmetric_and_bounded(pairs: list[tuple[int, int]]) -> None:
    k = stats.linear_weighted_kappa(pairs)
    swapped = stats.linear_weighted_kappa([(b, a) for a, b in pairs])
    assert (k is None) == (swapped is None)
    if k is not None and swapped is not None:
        assert k == pytest.approx(swapped, abs=1e-12)
        assert -1 - 1e-9 <= k <= 1 + 1e-9


def test_kappa_known_values() -> None:
    assert stats.linear_weighted_kappa([(0, 0), (1, 1), (2, 2)]) == pytest.approx(1.0)
    assert stats.linear_weighted_kappa([(1, 1), (1, 1)]) is None  # Pe = 1: undefined
    assert stats.linear_weighted_kappa([]) is None
    # perfect disagreement at the extremes: (0,2),(2,0): Po = 0, Pe = 0.5 -> kappa = -1
    assert stats.linear_weighted_kappa([(0, 2), (2, 0)]) == pytest.approx(-1.0)


def test_kappa_ci_is_seeded_and_brackets_a_stable_kappa() -> None:
    pairs = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 1), (2, 2), (1, 1), (0, 0)] * 3
    ci = stats.kappa_ci(pairs, seed=3)
    assert ci == stats.kappa_ci(pairs, seed=3)
    k = stats.linear_weighted_kappa(pairs)
    assert ci is not None
    assert k is not None
    assert ci[0] <= k <= ci[1]
    assert stats.kappa_ci(pairs[:1]) is None


# ------------------------------------------------------------------ bootstrap
@given(
    st.lists(
        st.lists(st.sampled_from([0.0, 1.0]), min_size=1, max_size=5), min_size=2, max_size=12
    ),
    st.integers(0, 10),
)
@settings(max_examples=40, deadline=None)
def test_cluster_bootstrap_is_deterministic_and_inside_the_data_range(
    clusters: list[list[float]], seed: int
) -> None:
    ci = stats.cluster_bootstrap(clusters, n_resamples=300, seed=seed)
    assert ci == stats.cluster_bootstrap(clusters, n_resamples=300, seed=seed)
    flat = [v for c in clusters for v in c]
    assert ci is not None
    assert min(flat) <= ci[0] <= ci[1] <= max(flat)


def test_cluster_bootstrap_is_wider_than_iid_when_clusters_are_correlated() -> None:
    """Perfectly correlated clusters carry one observation each: resampling items as if iid would
    understate the uncertainty, which is the reason D3 uses a clustered bootstrap."""
    clusters = [[1.0] * 5, [0.0] * 5, [1.0] * 5, [1.0] * 5, [0.0] * 5, [1.0] * 5]
    clustered = stats.cluster_bootstrap(clusters, seed=1)
    iid = stats.cluster_bootstrap([[v] for c in clusters for v in c], seed=1)
    assert clustered is not None
    assert iid is not None
    assert clustered[1] - clustered[0] > iid[1] - iid[0]


def test_bootstrap_needs_two_units() -> None:
    assert stats.cluster_bootstrap([[1.0, 0.0]]) is None
    assert stats.pooled_mean([[1.0, 0.0], [1.0]]) == pytest.approx(2 / 3)


def test_quantile_is_type_7() -> None:
    assert stats.quantile([1.0, 2.0, 3.0, 4.0], 0.5) == 2.5
    assert stats.quantile([1.0, 2.0, 3.0, 4.0], 0.95) == pytest.approx(3.85)
    assert stats.quantile([7.0], 0.3) == 7.0


# ------------------------------------------------------------------ MDE
@given(st.floats(0, 1), st.integers(1, 500))
def test_mde_is_bounded_and_never_below_the_discreteness_floor(psi: float, n: int) -> None:
    mde = stats.mde_from_discordance(psi, n)
    assert mde is not None
    floor = min(1.0, stats.min_discordant_for_significance(0.05) / n)
    assert floor - 1e-12 <= mde <= 1.0


def test_mde_shrinks_with_n_and_grows_with_noise() -> None:
    a = stats.mde_from_discordance(0.2, 50)
    b = stats.mde_from_discordance(0.2, 200)
    c = stats.mde_from_discordance(0.4, 200)
    assert a is not None
    assert b is not None
    assert c is not None
    assert a > b
    assert c > b
    # (1.959964 + 0.841621) * sqrt(0.2 / 200)
    assert b == pytest.approx(2.801585 * math.sqrt(0.001), abs=1e-6)
    assert stats.mde_from_discordance(0.2, 0) is None
    assert stats.mde_from_paired_sd(0.5, 25) == pytest.approx(2.801585 * 0.5 / 5, abs=1e-6)
