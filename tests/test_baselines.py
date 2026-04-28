"""Tests for the baseline strategies."""

from __future__ import annotations

import math
from datetime import date

import polars as pl
import pytest

from modules.module_2_quant.backtest import run_backtest
from modules.module_2_quant.baselines import (
    buy_and_hold_config,
    mean_reversion_config,
    mean_reversion_signal,
    momentum_config,
    momentum_signal,
)
from modules.module_2_quant.friction import FrictionModel
from modules.module_2_quant.synthetic_ohlcv import generate_synthetic_ohlcv


def _zero_friction() -> FrictionModel:
    return FrictionModel(
        commission_per_share=0.0, max_volume_pct=1.0, slippage_quadratic_coef=0.0
    )


def _bars(tmp_path) -> pl.DataFrame:
    paths = generate_synthetic_ohlcv(
        tmp_path,
        date(2024, 1, 2),
        date(2024, 1, 12),
        seed=0,
        bars_per_day=20,
    )
    return pl.read_parquet(paths).sort("epoch_ns")


# --- momentum_signal ---------------------------------------------------


def test_momentum_signal_fires_on_uptrend() -> None:
    """Strictly increasing close => positive momentum across the window."""
    df = pl.DataFrame({"close": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]})
    sig = momentum_signal(window=3)
    out = sig.evaluate(df).to_list()
    # Indices 0..2 are warmup (null -> False); 3 onward should fire
    assert out[:3] == [False, False, False]
    assert all(v for v in out[3:])


def test_momentum_signal_does_not_fire_on_downtrend() -> None:
    df = pl.DataFrame({"close": [10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]})
    sig = momentum_signal(window=3)
    out = sig.evaluate(df).to_list()
    assert all(not v for v in out)


def test_momentum_signal_validates_window() -> None:
    with pytest.raises(ValueError, match="window"):
        momentum_signal(window=0)


# --- mean_reversion_signal --------------------------------------------


def test_mean_reversion_fires_on_strong_downward_spike() -> None:
    """Insert a single 5-sigma negative spike at index 25 -- it should fire."""
    base = [100.0] * 24 + [50.0] + [100.0] * 5
    df = pl.DataFrame({"close": base})
    sig = mean_reversion_signal(window=20, threshold_z=2.0)
    out = sig.evaluate(df).to_list()
    # The spike at index 24 is wildly below the rolling mean -- must fire
    assert out[24], "mean-reversion did not fire on a 50% drawdown spike"


def test_mean_reversion_does_not_fire_on_steady_prices() -> None:
    df = pl.DataFrame({"close": [100.0] * 50})
    sig = mean_reversion_signal(window=20, threshold_z=2.0)
    out = sig.evaluate(df).to_list()
    # Constant prices -> rolling stdev is zero -> z is NaN -> coerced to False
    assert not any(out)


def test_mean_reversion_validates_args() -> None:
    with pytest.raises(ValueError, match="window"):
        mean_reversion_signal(window=1)
    with pytest.raises(ValueError, match="threshold_z"):
        mean_reversion_signal(threshold_z=0.0)


# --- end-to-end backtests on the OHLCV synthetic ----------------------


def _fused_with_required_cols(bars: pl.DataFrame) -> pl.DataFrame:
    """Baselines do not use the Alpha Ledger; we just need the bar columns
    plus an empty doc_hash so backtest's attribution path is satisfied."""
    return bars.with_columns(pl.lit("").alias("doc_hash"))


def test_buy_and_hold_baseline_runs_end_to_end(tmp_path) -> None:
    bars = _bars(tmp_path)
    fused = _fused_with_required_cols(bars)
    cfg = buy_and_hold_config(
        target_pct=1.0, initial_capital=10_000.0, friction=_zero_friction()
    )
    result = run_backtest(fused, cfg)
    assert result.equity_curve.height > 0
    assert result.trades.height >= 1
    assert math.isfinite(result.final_equity)


def test_momentum_baseline_runs_end_to_end(tmp_path) -> None:
    bars = _bars(tmp_path)
    fused = _fused_with_required_cols(bars)
    cfg = momentum_config(
        window=10,
        target_pct=0.5,
        initial_capital=10_000.0,
        friction=_zero_friction(),
    )
    result = run_backtest(fused, cfg)
    assert math.isfinite(result.final_equity)


def test_mean_reversion_baseline_runs_end_to_end(tmp_path) -> None:
    bars = _bars(tmp_path)
    fused = _fused_with_required_cols(bars)
    cfg = mean_reversion_config(
        window=10,
        threshold_z=1.5,
        target_pct=0.5,
        initial_capital=10_000.0,
        friction=_zero_friction(),
    )
    result = run_backtest(fused, cfg)
    assert math.isfinite(result.final_equity)


def test_three_baselines_diverge_on_same_bars(tmp_path) -> None:
    """The three baselines must produce non-identical outcomes on the same
    bars -- otherwise they are not three distinct null hypotheses."""
    bars = _bars(tmp_path)
    fused = _fused_with_required_cols(bars)
    bh = run_backtest(
        fused,
        buy_and_hold_config(initial_capital=10_000.0, friction=_zero_friction()),
    )
    mom = run_backtest(
        fused,
        momentum_config(
            window=10, initial_capital=10_000.0, friction=_zero_friction()
        ),
    )
    mr = run_backtest(
        fused,
        mean_reversion_config(
            window=10,
            threshold_z=1.5,
            initial_capital=10_000.0,
            friction=_zero_friction(),
        ),
    )
    eqs = {bh.final_equity, mom.final_equity, mr.final_equity}
    # Three distinct strategies on the same bars must not all coincide
    assert len(eqs) >= 2, "all three baselines produced identical equity"
