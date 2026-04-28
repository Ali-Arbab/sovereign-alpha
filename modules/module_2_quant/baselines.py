"""Baseline strategies -- the null hypotheses Semantic Alpha must clear.

Per master directive section 0.5.1.B. None of these read the Alpha
Ledger; they only consume the OHLCV frame. A research claim that a
persona-driven strategy "works" is meaningless unless it outperforms
these baselines on a friction-adjusted basis.

Three baselines:

1. **Buy-and-hold** -- the trivial null. Long the asset for the full
   window at `target_pct` of equity. Tracks SPX-style index returns.

2. **Time-series momentum** -- long when the rolling log-return over
   `window` bars is positive. Captures trend persistence.

3. **Mean-reversion** -- long when price is `threshold_z` standard
   deviations BELOW the rolling mean. Buys oversold, expects bounce.

All three return `BacktestConfig` objects that plug into the standard
`run_backtest` runner from `backtest.py`.
"""

from __future__ import annotations

import polars as pl

from modules.module_2_quant.backtest import BacktestConfig
from modules.module_2_quant.friction import FrictionModel
from modules.module_2_quant.strategy import Signal, TargetPctRule, always


def momentum_signal(window: int = 20, *, on: str = "close") -> Signal:
    """Long when the rolling log-return over `window` bars is positive.

    Implementation: log price diffed by `window`. Null in the warmup
    window; Signal.evaluate coerces nulls to False so the strategy
    holds no position during warmup.
    """
    if window < 1:
        raise ValueError("window must be >= 1")
    return Signal(pl.col(on).log().diff(window) > 0)


def mean_reversion_signal(
    window: int = 20,
    *,
    threshold_z: float = 2.0,
    on: str = "close",
) -> Signal:
    """Long when price is at least `threshold_z` standard deviations BELOW
    the rolling mean over `window` bars.

    Buys oversold, anticipates reversion. Threshold is negative-side
    only (no symmetric short rule in v0; long-only is the runner contract)."""
    if window < 2:
        raise ValueError("window must be >= 2")
    if threshold_z <= 0:
        raise ValueError("threshold_z must be positive")
    rolling_mean = pl.col(on).rolling_mean(window_size=window)
    rolling_std = pl.col(on).rolling_std(window_size=window)
    z = (pl.col(on) - rolling_mean) / rolling_std
    return Signal(z < -threshold_z)


def buy_and_hold_config(
    *,
    target_pct: float = 1.0,
    initial_capital: float = 100_000.0,
    friction: FrictionModel | None = None,
) -> BacktestConfig:
    """A buy-and-hold-the-full-equity baseline."""
    rule = TargetPctRule(always(), target_pct)
    return BacktestConfig(
        rule=rule,
        initial_capital=initial_capital,
        friction=friction or FrictionModel(),
    )


def momentum_config(
    *,
    window: int = 20,
    target_pct: float = 0.5,
    initial_capital: float = 100_000.0,
    friction: FrictionModel | None = None,
    on: str = "close",
) -> BacktestConfig:
    """Time-series momentum: long `target_pct` when momentum is positive."""
    rule = TargetPctRule(momentum_signal(window=window, on=on), target_pct)
    return BacktestConfig(
        rule=rule,
        initial_capital=initial_capital,
        friction=friction or FrictionModel(),
    )


def mean_reversion_config(
    *,
    window: int = 20,
    threshold_z: float = 2.0,
    target_pct: float = 0.5,
    initial_capital: float = 100_000.0,
    friction: FrictionModel | None = None,
    on: str = "close",
) -> BacktestConfig:
    """Mean-reversion: long `target_pct` when price is below band by threshold_z."""
    rule = TargetPctRule(
        mean_reversion_signal(window=window, threshold_z=threshold_z, on=on),
        target_pct,
    )
    return BacktestConfig(
        rule=rule,
        initial_capital=initial_capital,
        friction=friction or FrictionModel(),
    )
