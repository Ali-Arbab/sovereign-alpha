"""Tests for Module II statistical rigor -- PSR / DSR / bootstrap CIs."""

from __future__ import annotations

import math

import numpy as np
import pytest

from modules.module_2_quant.statistics import (
    benjamini_hochberg,
    bonferroni_correction,
    bootstrap_metric,
    deflated_sharpe_ratio,
    expected_max_sharpe,
    holm_bonferroni,
    probabilistic_sharpe_ratio,
)


def test_psr_at_benchmark_is_one_half() -> None:
    # When observed = benchmark, PSR = Phi(0) = 0.5
    assert probabilistic_sharpe_ratio(0.5, n=1000, benchmark_sr=0.5) == pytest.approx(0.5)


def test_psr_above_benchmark_exceeds_one_half() -> None:
    assert probabilistic_sharpe_ratio(1.0, n=1000, benchmark_sr=0.0) > 0.5


def test_psr_below_benchmark_under_one_half() -> None:
    assert probabilistic_sharpe_ratio(-0.5, n=1000, benchmark_sr=0.0) < 0.5


def test_psr_grows_with_sample_size_for_positive_signal() -> None:
    p_small = probabilistic_sharpe_ratio(0.5, n=50, benchmark_sr=0.0)
    p_large = probabilistic_sharpe_ratio(0.5, n=5000, benchmark_sr=0.0)
    assert p_large > p_small


def test_psr_returns_nan_for_short_series() -> None:
    assert math.isnan(probabilistic_sharpe_ratio(0.5, n=1))


def test_psr_kurtosis_penalizes_fat_tails() -> None:
    """For positive Sharpe, a fatter tail should reduce PSR -- the kurtosis
    correction shrinks the z-score's denominator. Parameters chosen so that
    PSR is not saturated at 1.0 in floating point."""
    p_normal = probabilistic_sharpe_ratio(0.5, n=10, kurtosis=3.0)
    p_fat = probabilistic_sharpe_ratio(0.5, n=10, kurtosis=10.0)
    assert p_fat < p_normal


def test_expected_max_sharpe_zero_for_one_trial() -> None:
    assert expected_max_sharpe(1) == 0.0


def test_expected_max_sharpe_grows_with_trials() -> None:
    a = expected_max_sharpe(10)
    b = expected_max_sharpe(1000)
    assert b > a > 0


def test_expected_max_sharpe_rejects_negative_variance() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        expected_max_sharpe(10, sr_variance=-1.0)


def test_dsr_below_psr_for_many_trials() -> None:
    """DSR deflates PSR by the expected-max correction. With many trials and
    nonzero SR variance, DSR < PSR."""
    p = probabilistic_sharpe_ratio(1.0, n=1000, benchmark_sr=0.0)
    d = deflated_sharpe_ratio(1.0, n=1000, n_trials=100, sr_variance=1.0)
    assert d < p


def test_dsr_equals_psr_at_one_trial() -> None:
    """One trial = no selection bias = no deflation."""
    p = probabilistic_sharpe_ratio(0.8, n=500, benchmark_sr=0.0)
    d = deflated_sharpe_ratio(0.8, n=500, n_trials=1, sr_variance=1.0)
    assert d == pytest.approx(p)


def test_bootstrap_iid_ci_contains_point_estimate() -> None:
    rng = np.random.default_rng(0)
    arr = rng.normal(0.05, 1.0, size=500)
    lo, point, hi = bootstrap_metric(arr, np.mean, n_iter=500, alpha=0.10, seed=0)
    assert lo <= point <= hi


def test_bootstrap_block_ci_for_serially_correlated_returns() -> None:
    rng = np.random.default_rng(1)
    raw = rng.normal(0, 1, size=500)
    # AR(1) with phi=0.5 -- introduces serial correlation
    series = np.empty(500)
    series[0] = raw[0]
    for i in range(1, 500):
        series[i] = 0.5 * series[i - 1] + raw[i]
    lo, point, hi = bootstrap_metric(
        series, np.mean, n_iter=500, alpha=0.10, block_size=20, seed=0
    )
    assert lo <= point <= hi


def test_bootstrap_invalid_args() -> None:
    arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    with pytest.raises(ValueError, match="1-D"):
        bootstrap_metric(np.zeros((5, 5)), np.mean)
    with pytest.raises(ValueError, match="2 observations"):
        bootstrap_metric(np.array([1.0]), np.mean)
    with pytest.raises(ValueError, match="alpha"):
        bootstrap_metric(arr, np.mean, alpha=0.0)
    with pytest.raises(ValueError, match="alpha"):
        bootstrap_metric(arr, np.mean, alpha=1.0)
    with pytest.raises(ValueError, match="n_iter"):
        bootstrap_metric(arr, np.mean, n_iter=10)
    with pytest.raises(ValueError, match="block_size"):
        bootstrap_metric(arr, np.mean, n_iter=200, block_size=0)


def test_bootstrap_seed_determinism() -> None:
    arr = np.array([0.1, -0.2, 0.05, 0.0, 0.15, -0.1, 0.08])
    a = bootstrap_metric(arr, np.mean, n_iter=200, seed=42)
    b = bootstrap_metric(arr, np.mean, n_iter=200, seed=42)
    assert a == b


# --- multiple-hypothesis corrections -----------------------------------


def test_bonferroni_per_test_threshold_is_alpha_over_n() -> None:
    p = np.array([0.001, 0.01, 0.04, 0.06, 0.5])
    sig, corrected = bonferroni_correction(p, alpha=0.05)
    assert corrected == pytest.approx(0.05 / 5)
    # Only p < 0.01 should pass
    assert sig.tolist() == [True, False, False, False, False]


def test_bonferroni_empty_input_returns_empty() -> None:
    sig, corrected = bonferroni_correction([], alpha=0.05)
    assert sig.size == 0
    assert corrected == 0.05


def test_bonferroni_rejects_invalid_p() -> None:
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        bonferroni_correction([0.1, 1.5])


def test_holm_no_more_conservative_than_bonferroni() -> None:
    """Holm rejects at least as many hypotheses as Bonferroni (less
    conservative for the same FWER bound)."""
    p = np.array([0.001, 0.01, 0.03, 0.04, 0.05, 0.5])
    h = holm_bonferroni(p, alpha=0.05)
    b, _ = bonferroni_correction(p, alpha=0.05)
    assert h.sum() >= b.sum()


def test_holm_step_down_stops_at_first_failure() -> None:
    """Once a sorted p-value fails its threshold, everything after must be
    non-significant -- regardless of its value.

    With N=2 and alpha=0.05, sorted thresholds are [0.025, 0.05]. For
    p=[0.5, 0.001]: sort to [0.001, 0.5]; 0.001 passes 0.025 (significant);
    0.5 fails 0.05 (non-significant). Original order: [0.5, 0.001] ->
    [False, True]."""
    h = holm_bonferroni(np.array([0.5, 0.001]), alpha=0.05)
    assert h.tolist() == [False, True]


def test_bh_less_conservative_than_holm() -> None:
    """BH controls FDR (looser than FWER), so it rejects at least as many."""
    p = np.array([0.001, 0.005, 0.01, 0.02, 0.04, 0.06, 0.5])
    bh = benjamini_hochberg(p, alpha=0.05)
    h = holm_bonferroni(p, alpha=0.05)
    assert bh.sum() >= h.sum()


def test_bh_classic_example() -> None:
    """A worked-through BH example to confirm the largest-k rule.

    p = [0.001, 0.008, 0.039, 0.041, 0.042, 0.060, 0.074, 0.205].
    With alpha=0.05, thresholds (k/N)*alpha = [0.00625, 0.0125, 0.01875,
    0.025, 0.03125, 0.0375, 0.04375, 0.05]. We compare each sorted p to
    its threshold; the largest k where p_(k) <= threshold is k=2 (0.008
    <= 0.0125). Reject the first 2.
    """
    p = np.array([0.001, 0.008, 0.039, 0.041, 0.042, 0.060, 0.074, 0.205])
    bh = benjamini_hochberg(p, alpha=0.05)
    assert bh.tolist() == [True, True, False, False, False, False, False, False]


def test_bh_all_below_threshold_rejects_all() -> None:
    p = np.array([0.001, 0.002, 0.003])
    bh = benjamini_hochberg(p, alpha=0.05)
    assert bh.tolist() == [True, True, True]


def test_bh_no_signal_rejects_none() -> None:
    p = np.array([0.5, 0.6, 0.7, 0.8])
    bh = benjamini_hochberg(p, alpha=0.05)
    assert not bh.any()


def test_corrections_argument_validation() -> None:
    with pytest.raises(ValueError, match="alpha"):
        bonferroni_correction([0.5], alpha=0.0)
    with pytest.raises(ValueError, match="alpha"):
        holm_bonferroni([0.5], alpha=1.0)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        benjamini_hochberg([0.5, -0.1])
