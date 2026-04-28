"""Tests for the mock-state publisher -- BacktestResult -> Module III messages."""

from __future__ import annotations

import polars as pl
import pytest
from pydantic import BaseModel

from modules.module_2_quant.backtest import BacktestConfig, run_backtest
from modules.module_2_quant.friction import FrictionModel
from modules.module_2_quant.strategy import TargetPctRule, always
from modules.module_3_twin.messages import (
    PORTFOLIO_STATE,
    PRICES,
    TRADES,
    PortfolioStateMessage,
    PriceTickMessage,
    TradeMessage,
)
from modules.module_3_twin.publisher import publish_backtest_state


def _trivial_fused() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "ticker": ["AAPL"] * 5,
            "epoch_ns": [10, 20, 30, 40, 50],
            "open": [100.0, 101.0, 102.0, 103.0, 104.0],
            "high": [101.0, 102.0, 103.0, 104.0, 105.0],
            "low": [99.0, 100.0, 101.0, 102.0, 103.0],
            "close": [100.5, 101.5, 102.5, 103.5, 104.5],
            "volume": [1_000_000] * 5,
            "doc_hash": ["sha256:a", "sha256:b", "sha256:c", "sha256:d", "sha256:e"],
        }
    )


def _zero_friction() -> FrictionModel:
    return FrictionModel(
        commission_per_share=0.0, max_volume_pct=1.0, slippage_quadratic_coef=0.0
    )


def test_publish_emits_one_price_tick_per_bar() -> None:
    fused = _trivial_fused()
    cfg = BacktestConfig(
        rule=TargetPctRule(always(), 0.5),
        initial_capital=10_000.0,
        friction=_zero_friction(),
    )
    result = run_backtest(fused, cfg)

    captured: list[tuple[str, BaseModel]] = []
    publish_backtest_state(
        lambda topic, msg: captured.append((topic, msg)),
        fused,
        result,
    )

    price_msgs = [m for t, m in captured if t == PRICES]
    assert len(price_msgs) == fused.height
    assert all(isinstance(m, PriceTickMessage) for m in price_msgs)


def test_publish_emits_trades_with_doc_hash_attribution() -> None:
    fused = _trivial_fused()
    cfg = BacktestConfig(
        rule=TargetPctRule(always(), 0.5),
        initial_capital=10_000.0,
        friction=_zero_friction(),
    )
    result = run_backtest(fused, cfg)

    captured: list[tuple[str, BaseModel]] = []
    publish_backtest_state(
        lambda topic, msg: captured.append((topic, msg)), fused, result
    )

    trade_msgs = [m for t, m in captured if t == TRADES]
    assert len(trade_msgs) == result.trades.height
    assert all(isinstance(m, TradeMessage) for m in trade_msgs)
    if trade_msgs:
        assert all(m.doc_hash for m in trade_msgs)


def test_publish_emits_portfolio_state_per_unique_epoch() -> None:
    fused = _trivial_fused()
    cfg = BacktestConfig(
        rule=TargetPctRule(always(), 0.5),
        initial_capital=10_000.0,
        friction=_zero_friction(),
    )
    result = run_backtest(fused, cfg)

    captured: list[tuple[str, BaseModel]] = []
    publish_backtest_state(
        lambda topic, msg: captured.append((topic, msg)), fused, result
    )

    portfolio_msgs = [m for t, m in captured if t == PORTFOLIO_STATE]
    unique_epochs = len({int(e) for e in fused["epoch_ns"]})
    assert len(portfolio_msgs) == unique_epochs
    assert all(isinstance(m, PortfolioStateMessage) for m in portfolio_msgs)


def test_publish_returns_total_message_count() -> None:
    fused = _trivial_fused()
    cfg = BacktestConfig(
        rule=TargetPctRule(always(), 0.5),
        initial_capital=10_000.0,
        friction=_zero_friction(),
    )
    result = run_backtest(fused, cfg)
    captured: list[tuple[str, BaseModel]] = []
    sent = publish_backtest_state(
        lambda topic, msg: captured.append((topic, msg)), fused, result
    )
    assert sent == len(captured)


def test_publish_rejects_missing_columns() -> None:
    bad = pl.DataFrame({"ticker": ["AAPL"], "epoch_ns": [1]})
    cfg = BacktestConfig(rule=TargetPctRule(always(), 1.0))
    fused = _trivial_fused()
    result = run_backtest(fused, cfg)
    with pytest.raises(ValueError, match="missing required"):
        publish_backtest_state(lambda t, m: None, bad, result)
