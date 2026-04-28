"""Tests for Module II metrics -- Sharpe, Sortino, drawdown, capture ratio."""

from __future__ import annotations

import math

import numpy as np
import polars as pl
import pytest

from modules.module_2_quant.metrics import (
    capture_ratio,
    drawdown_report,
    sharpe_ratio,
    sortino_ratio,
)


def test_sharpe_zero_returns_is_nan() -> None:
    assert math.isnan(sharpe_ratio([0.0] * 10))


def test_sharpe_positive_constant_excess_diverges_to_nan() -> None:
    # Constant returns have zero stdev -> NaN by construction
    assert math.isnan(sharpe_ratio([0.001] * 50))


def test_sharpe_basic_value_sanity() -> None:
    # Strong positive signal: daily SNR = 0.5 -> annualized Sharpe ~ 7.9.
    # With N = 252*10 the sample-Sharpe std is small enough that "> 5" is
    # robust to seed without becoming a tautology.
    rng = np.random.default_rng(0)
    daily = rng.normal(loc=0.005, scale=0.01, size=252 * 10)
    assert sharpe_ratio(daily) > 5.0


def test_sharpe_zero_mean_noise_is_near_zero() -> None:
    rng = np.random.default_rng(0)
    arr = rng.normal(0.0, 0.01, size=10_000)
    assert abs(sharpe_ratio(arr)) < 1.0


def test_sharpe_accepts_polars_series() -> None:
    s = pl.Series([0.001, 0.002, -0.001, 0.0005, 0.0015])
    assert not math.isnan(sharpe_ratio(s))


def test_sortino_no_downside_returns_inf_when_mean_positive() -> None:
    assert sortino_ratio([0.001] * 50) == float("inf")


def test_sortino_with_downside() -> None:
    rng = np.random.default_rng(1)
    daily = rng.normal(loc=0.0005, scale=0.01, size=1000)
    so = sortino_ratio(daily)
    sr = sharpe_ratio(daily)
    # Sortino >= Sharpe for the same mean: downside-only stdev <= total stdev
    assert so >= sr - 1e-6


def test_drawdown_monotonic_up_is_zero() -> None:
    eq = [100.0 + i for i in range(20)]
    rep = drawdown_report(eq)
    assert rep.max_drawdown == 0.0
    assert rep.underwater_periods == 0


def test_drawdown_simple_case() -> None:
    # Peak at 200 (idx 1), trough at 100 (idx 3), recover to 150
    eq = [100.0, 200.0, 150.0, 100.0, 130.0, 150.0]
    rep = drawdown_report(eq)
    assert rep.max_drawdown == pytest.approx(-0.5)
    assert rep.peak_index == 1
    assert rep.trough_index == 3
    # Underwater: idx 2 (below 200), idx 3 (below 200), idx 4 (below 200), idx 5 (below 200) = 4
    assert rep.underwater_periods == 4


def test_drawdown_rejects_non_positive_equity() -> None:
    with pytest.raises(ValueError, match="positive"):
        drawdown_report([100.0, 0.0, 50.0])


def test_drawdown_short_series() -> None:
    rep = drawdown_report([100.0])
    assert rep.max_drawdown == 0.0


def test_capture_ratio_against_self_is_one() -> None:
    rng = np.random.default_rng(2)
    benchmark = rng.normal(loc=0.0, scale=0.01, size=500)
    up, down = capture_ratio(benchmark, benchmark)
    assert up == pytest.approx(1.0)
    assert down == pytest.approx(1.0)


def test_capture_ratio_double_benchmark_is_two() -> None:
    rng = np.random.default_rng(3)
    benchmark = rng.normal(loc=0.0, scale=0.01, size=500)
    strategy = 2 * benchmark
    up, down = capture_ratio(strategy, benchmark)
    assert up == pytest.approx(2.0)
    assert down == pytest.approx(2.0)


def test_capture_ratio_mismatched_lengths_raises() -> None:
    with pytest.raises(ValueError, match="length"):
        capture_ratio([0.0, 0.1], [0.0, 0.1, 0.2])


def test_capture_ratio_empty_inputs_yields_nan() -> None:
    up, down = capture_ratio([], [])
    assert math.isnan(up) and math.isnan(down)
