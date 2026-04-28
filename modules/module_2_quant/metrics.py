"""Module II metrics -- friction-adjusted Sharpe / Sortino, drawdowns, capture.

Per master directive section 4.5. All ratios are annualized assuming
`periods_per_year` periods (default 252 for daily; pass 252*390 for M1 bars).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

ArrayLike = pl.Series | np.ndarray | list[float]


def _as_numpy(x: ArrayLike) -> np.ndarray:
    if isinstance(x, pl.Series):
        return x.to_numpy()
    if isinstance(x, np.ndarray):
        return x
    return np.asarray(x, dtype=float)


def sharpe_ratio(
    returns: ArrayLike,
    *,
    rf_annual: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """Annualized Sharpe ratio. NaN for fewer than 2 observations or zero stdev."""
    arr = _as_numpy(returns)
    if arr.size < 2:
        return float("nan")
    excess = arr - (rf_annual / periods_per_year)
    std = excess.std(ddof=1)
    if std == 0 or np.isnan(std):
        return float("nan")
    return float(excess.mean() / std * np.sqrt(periods_per_year))


def sortino_ratio(
    returns: ArrayLike,
    *,
    mar_annual: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """Annualized Sortino ratio (downside-only volatility denominator).

    Returns +inf when no downside observations exist and the mean is positive
    (a strategy with no drawdown vs MAR), NaN when ambiguous.
    """
    arr = _as_numpy(returns)
    if arr.size < 2:
        return float("nan")
    excess = arr - (mar_annual / periods_per_year)
    downside = excess[excess < 0]
    if downside.size == 0:
        return float("inf") if excess.mean() > 0 else float("nan")
    downside_std = float(np.sqrt(np.mean(downside * downside)))
    if downside_std == 0:
        return float("nan")
    return float(excess.mean() / downside_std * np.sqrt(periods_per_year))


@dataclass(frozen=True)
class DrawdownReport:
    max_drawdown: float
    peak_index: int
    trough_index: int
    underwater_periods: int


def drawdown_report(equity: ArrayLike) -> DrawdownReport:
    """Max drawdown (negative fraction), the peak/trough indices around it,
    and the longest contiguous underwater stretch in periods.
    """
    arr = _as_numpy(equity)
    if arr.size < 2:
        return DrawdownReport(0.0, 0, 0, 0)
    if (arr <= 0).any():
        raise ValueError("equity series must be strictly positive")

    running_max = np.maximum.accumulate(arr)
    dd = (arr - running_max) / running_max
    trough_idx = int(np.argmin(dd))
    peak_idx = int(np.argmax(arr[: trough_idx + 1])) if trough_idx > 0 else 0
    max_dd = float(dd[trough_idx])

    underwater = arr < running_max
    max_run = 0
    cur_run = 0
    for flag in underwater:
        if flag:
            cur_run += 1
            if cur_run > max_run:
                max_run = cur_run
        else:
            cur_run = 0

    return DrawdownReport(max_dd, peak_idx, trough_idx, max_run)


def capture_ratio(
    strategy_returns: ArrayLike,
    benchmark_returns: ArrayLike,
) -> tuple[float, float]:
    """Return (upside capture, downside capture) vs benchmark.

    Upside capture = mean(strategy_returns where benchmark > 0) / mean(benchmark > 0).
    Downside likewise on benchmark < 0. NaN when either side has no observations
    or zero mean benchmark return.
    """
    s = _as_numpy(strategy_returns)
    b = _as_numpy(benchmark_returns)
    if s.size != b.size:
        raise ValueError("strategy and benchmark series must be the same length")
    if s.size == 0:
        return float("nan"), float("nan")

    def _ratio(mask: np.ndarray) -> float:
        if not mask.any():
            return float("nan")
        b_mean = float(b[mask].mean())
        if b_mean == 0.0:
            return float("nan")
        return float(s[mask].mean() / b_mean)

    return _ratio(b > 0), _ratio(b < 0)
