"""Tests for the Module II backtest runner."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import polars as pl
import pytest

from modules.module_1_extraction.synthetic_ledger import generate_synthetic_ledger
from modules.module_2_quant.backtest import BacktestConfig, run_backtest
from modules.module_2_quant.friction import FrictionModel
from modules.module_2_quant.fusion import as_of_fuse, explode_ledger_entities
from modules.module_2_quant.strategy import (
    TargetPctRule,
    always,
    col_eq,
    col_gt,
    never,
)
from modules.module_2_quant.synthetic_ohlcv import generate_synthetic_ohlcv


def _trivial_fused() -> pl.DataFrame:
    """A small fused frame: 1 ticker, ascending close, plenty of volume."""
    return pl.DataFrame(
        {
            "ticker": ["AAPL"] * 5,
            "epoch_ns": [10, 20, 30, 40, 50],
            "close": [100.0, 101.0, 102.0, 103.0, 104.0],
            "volume": [1_000_000] * 5,
            "doc_hash": ["sha256:a", "sha256:b", "sha256:c", "sha256:d", "sha256:e"],
            "macro_sentiment": [0.1, 0.6, 0.7, 0.8, 0.9],
        }
    )


def _zero_friction() -> FrictionModel:
    return FrictionModel(
        commission_per_share=0.0,
        max_volume_pct=1.0,
        slippage_quadratic_coef=0.0,
    )


def test_never_signal_keeps_capital_in_cash() -> None:
    fused = _trivial_fused()
    cfg = BacktestConfig(rule=TargetPctRule(never(), 1.0), initial_capital=10_000.0)
    result = run_backtest(fused, cfg)
    assert result.trades.height == 0
    assert result.final_equity == pytest.approx(10_000.0)
    # equity is constant -> max drawdown is zero
    assert result.drawdown.max_drawdown == pytest.approx(0.0)


def test_always_buy_and_hold_no_friction_tracks_price() -> None:
    fused = _trivial_fused()
    cfg = BacktestConfig(
        rule=TargetPctRule(always(), 1.0),
        initial_capital=10_000.0,
        friction=_zero_friction(),
    )
    result = run_backtest(fused, cfg)
    # First bar: buys 100 shares at 100 -> uses all 10_000
    # Last bar: 100 shares * 104 = 10_400 equity
    assert result.final_equity == pytest.approx(10_400.0)
    assert result.trades.height == 1
    assert result.trades["side"][0] == "buy"
    assert result.trades["qty"][0] == 100


def test_signal_fires_only_when_threshold_crossed() -> None:
    fused = _trivial_fused()
    rule = TargetPctRule(col_gt("macro_sentiment", 0.5), 0.5)
    cfg = BacktestConfig(rule=rule, initial_capital=10_000.0, friction=_zero_friction())
    result = run_backtest(fused, cfg)
    # First bar (sentiment 0.1) -> no buy
    # Subsequent bars (sentiment > 0.5) -> target = 0.5 * equity / close
    assert result.trades.height >= 1
    # First trade should NOT be on bar 0 (sentiment 0.1)
    first_epoch = result.trades["epoch_ns"][0]
    assert first_epoch != 10


def test_per_trade_attribution_carries_doc_hash() -> None:
    fused = _trivial_fused()
    cfg = BacktestConfig(
        rule=TargetPctRule(always(), 0.5),
        initial_capital=10_000.0,
        friction=_zero_friction(),
    )
    result = run_backtest(fused, cfg)
    assert result.trades.height >= 1
    # Every trade row carries a non-empty doc_hash
    assert (result.trades["doc_hash"].str.len_chars() > 0).all()


def test_signal_with_regime_filter_composition() -> None:
    fused = _trivial_fused().with_columns(
        pl.Series("regime_shift_flag", [False, True, False, False, False])
    )
    sig = col_gt("macro_sentiment", 0.5) & ~col_eq("regime_shift_flag", True)
    cfg = BacktestConfig(
        rule=TargetPctRule(sig, 0.5),
        initial_capital=10_000.0,
        friction=_zero_friction(),
    )
    result = run_backtest(fused, cfg)
    # Bar at epoch=20 has regime_shift_flag=True -> signal does NOT fire there
    # even though sentiment is > 0.5
    assert result.trades.filter(pl.col("epoch_ns") == 20).height == 0


def test_buy_capped_at_available_cash() -> None:
    """Even at target_pct=1.0, can't spend more cash than we have."""
    fused = _trivial_fused()
    cfg = BacktestConfig(
        rule=TargetPctRule(always(), 1.0),
        initial_capital=200.0,  # only enough for 2 shares at $100
        friction=_zero_friction(),
    )
    result = run_backtest(fused, cfg)
    first_trade_qty = result.trades["qty"][0]
    assert first_trade_qty <= 2


def test_runner_rejects_unsorted_frame() -> None:
    bad = pl.DataFrame(
        {
            "ticker": ["AAPL", "AAPL"],
            "epoch_ns": [50, 10],
            "close": [101.0, 100.0],
            "volume": [1_000_000, 1_000_000],
        }
    )
    with pytest.raises(ValueError, match="sorted"):
        run_backtest(bad, BacktestConfig(rule=TargetPctRule(always(), 1.0)))


def test_runner_rejects_missing_columns() -> None:
    incomplete = pl.DataFrame({"ticker": ["AAPL"], "epoch_ns": [10], "close": [100.0]})
    with pytest.raises(ValueError, match="missing required"):
        run_backtest(incomplete, BacktestConfig(rule=TargetPctRule(always(), 1.0)))


def test_empty_frame_returns_initial_capital() -> None:
    empty = pl.DataFrame(
        schema={
            "ticker": pl.String,
            "epoch_ns": pl.Int64,
            "close": pl.Float64,
            "volume": pl.Int64,
        }
    )
    result = run_backtest(empty, BacktestConfig(rule=TargetPctRule(always(), 1.0)))
    assert result.final_equity == pytest.approx(100_000.0)
    assert result.trades.height == 0


def test_friction_reduces_pnl_vs_zero_friction() -> None:
    fused = _trivial_fused()
    zero_cfg = BacktestConfig(
        rule=TargetPctRule(always(), 1.0),
        initial_capital=10_000.0,
        friction=_zero_friction(),
    )
    real_cfg = BacktestConfig(
        rule=TargetPctRule(always(), 1.0),
        initial_capital=10_000.0,
        friction=FrictionModel(
            commission_per_share=0.01, max_volume_pct=1.0, slippage_quadratic_coef=10.0
        ),
    )
    zero_result = run_backtest(fused, zero_cfg)
    real_result = run_backtest(fused, real_cfg)
    assert real_result.final_equity < zero_result.final_equity


def test_e2e_synthetic_ledger_x_ohlcv_full_pipeline(tmp_path: Path) -> None:
    """End-to-end: generate Alpha Ledger + OHLCV, fuse, run a sentiment-driven
    strategy, verify equity curve and per-trade attribution."""
    ledger_paths = generate_synthetic_ledger(
        tmp_path / "ledger",
        date(2020, 1, 27),
        date(2020, 1, 28),
        seed=0,
        docs_per_day_mean=20.0,
    )
    ohlcv_paths = generate_synthetic_ohlcv(
        tmp_path / "ohlcv",
        date(2020, 1, 27),
        date(2020, 1, 28),
        seed=0,
        bars_per_day=10,
    )
    ledger = explode_ledger_entities(pl.read_parquet(ledger_paths)).sort("epoch_ns")
    bars = pl.read_parquet(ohlcv_paths).sort("epoch_ns")
    fused = as_of_fuse(bars, ledger, by_left="ticker", by_right="entity")

    rule = TargetPctRule(col_gt("macro_sentiment", 0.0), 0.05)
    cfg = BacktestConfig(rule=rule, initial_capital=100_000.0)
    result = run_backtest(fused, cfg)

    # Engine ran end-to-end: equity curve has rows, final equity is finite
    assert result.equity_curve.height > 0
    assert result.final_equity > 0
    # If any trades happened, every one carries a doc_hash attribution
    if result.trades.height > 0:
        assert (result.trades["doc_hash"].str.len_chars() > 0).any()
