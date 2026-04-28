"""Tests for the Module II friction layer."""

from __future__ import annotations

import pytest

from modules.module_2_quant.friction import FrictionModel, Side


def test_small_buy_no_partial_fill() -> None:
    model = FrictionModel(commission_per_share=0.005, max_volume_pct=0.05)
    fr = model.fill(side=Side.BUY, qty_requested=100, bar_volume=10_000, bar_price=100.0)
    assert fr.filled_qty == 100
    assert fr.rejected_qty == 0
    assert fr.commission == pytest.approx(0.5)
    # Slippage is positive on a buy
    assert fr.avg_fill_price > 100.0


def test_large_buy_partial_fill_capped_at_volume_pct() -> None:
    model = FrictionModel(max_volume_pct=0.05)
    fr = model.fill(side=Side.BUY, qty_requested=10_000, bar_volume=100_000, bar_price=50.0)
    # Cap = 5% of 100_000 = 5000
    assert fr.filled_qty == 5000
    assert fr.rejected_qty == 5000


def test_sell_slippage_below_bar_price() -> None:
    model = FrictionModel()
    fr = model.fill(side=Side.SELL, qty_requested=200, bar_volume=10_000, bar_price=200.0)
    assert fr.avg_fill_price < 200.0
    assert fr.slippage_cost > 0  # cost is always positive (magnitude)


def test_quadratic_slippage_scaling() -> None:
    model = FrictionModel(slippage_quadratic_coef=1.0, max_volume_pct=0.5)
    small = model.fill(side=Side.BUY, qty_requested=100, bar_volume=10_000, bar_price=100.0)
    large = model.fill(side=Side.BUY, qty_requested=1000, bar_volume=10_000, bar_price=100.0)
    # 10x participation -> 100x per-share slippage
    small_per_share = small.avg_fill_price - 100.0
    large_per_share = large.avg_fill_price - 100.0
    assert large_per_share == pytest.approx(small_per_share * 100, rel=1e-6)


def test_zero_qty_or_zero_volume_yields_no_fill() -> None:
    model = FrictionModel()
    a = model.fill(side=Side.BUY, qty_requested=0, bar_volume=10_000, bar_price=100.0)
    b = model.fill(side=Side.BUY, qty_requested=100, bar_volume=0, bar_price=100.0)
    assert a.filled_qty == 0 and a.commission == 0.0
    assert b.filled_qty == 0 and b.rejected_qty == 100


def test_total_cost_and_notional_helpers() -> None:
    model = FrictionModel(commission_per_share=0.01)
    fr = model.fill(side=Side.BUY, qty_requested=500, bar_volume=100_000, bar_price=20.0)
    assert fr.total_cost == pytest.approx(fr.slippage_cost + fr.commission)
    assert fr.total_notional == pytest.approx(fr.filled_qty * fr.avg_fill_price)


def test_borrow_cost_stub() -> None:
    model = FrictionModel(annual_borrow_rate=0.02)
    cost = model.borrow_cost(qty_short=100, price=50.0, days=365)
    # 100 shares * $50 * 2% * 1 year = $100
    assert cost == pytest.approx(100.0)


def test_borrow_cost_zero_for_non_short_or_zero_days() -> None:
    model = FrictionModel()
    assert model.borrow_cost(qty_short=0, price=50.0, days=10) == 0.0
    assert model.borrow_cost(qty_short=-5, price=50.0, days=10) == 0.0
    assert model.borrow_cost(qty_short=100, price=50.0, days=0) == 0.0


def test_invalid_args() -> None:
    model = FrictionModel()
    with pytest.raises(ValueError, match="qty_requested"):
        model.fill(side=Side.BUY, qty_requested=-1, bar_volume=100, bar_price=10.0)
    with pytest.raises(ValueError, match="bar_volume"):
        model.fill(side=Side.BUY, qty_requested=10, bar_volume=-1, bar_price=10.0)
    with pytest.raises(ValueError, match="bar_price"):
        model.fill(side=Side.BUY, qty_requested=10, bar_volume=100, bar_price=0.0)


def test_invalid_friction_params() -> None:
    with pytest.raises(ValueError, match="commission"):
        FrictionModel(commission_per_share=-0.001)
    with pytest.raises(ValueError, match="max_volume_pct"):
        FrictionModel(max_volume_pct=0.0)
    with pytest.raises(ValueError, match="max_volume_pct"):
        FrictionModel(max_volume_pct=1.5)
    with pytest.raises(ValueError, match="slippage"):
        FrictionModel(slippage_quadratic_coef=-1.0)
    with pytest.raises(ValueError, match="annual_borrow_rate"):
        FrictionModel(annual_borrow_rate=-0.01)
