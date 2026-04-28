"""Synthetic OHLCV bar generator -- bootstrap phase only.

Geometric-Brownian-motion log-price walk per ticker. Companion to the synthetic
Alpha Ledger generator (modules.module_1_extraction.synthetic_ledger). NOT a
research artifact -- every bar is deterministic per `(seed, ticker, bar_idx)`.

Output: Hive-partitioned Parquet at `{out_dir}/year=YYYY/month=MM/part-0.parquet`,
multi-ticker per file, sorted ascending on epoch_ns within each partition.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl

from modules.module_1_extraction.bootstrap_universe import DEFAULT_BOOTSTRAP_UNIVERSE
from shared.schemas.ohlcv_bar import SCHEMA_VERSION

_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)
ANNUAL_TRADING_MINUTES: Final[int] = 252 * 390  # ~98k

_COLUMN_ORDER: Final[tuple[str, ...]] = (
    "ticker",
    "timestamp",
    "epoch_ns",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "schema_version",
)


def _epoch_ns(ts: datetime) -> int:
    delta = ts - _EPOCH
    return (delta.days * 86_400 + delta.seconds) * 1_000_000_000 + delta.microseconds * 1_000


def _gbm_path(
    rng: np.random.Generator,
    n: int,
    *,
    initial: float,
    mu_annual: float = 0.08,
    sigma_annual: float = 0.20,
    bar_minutes: int = 1,
) -> np.ndarray:
    if n <= 0:
        return np.zeros(0)
    dt = bar_minutes / ANNUAL_TRADING_MINUTES
    drift = (mu_annual - 0.5 * sigma_annual**2) * dt
    diffusion = sigma_annual * np.sqrt(dt)
    log_returns = drift + diffusion * rng.standard_normal(n)
    log_prices = np.log(initial) + np.cumsum(log_returns)
    return np.exp(log_prices)


def generate_synthetic_ohlcv(
    out_dir: Path,
    start_date: date,
    end_date: date,
    *,
    tickers: dict[str, str] | None = None,
    seed: int = 0,
    bar_minutes: int = 1,
    bars_per_day: int = 390,
    initial_price_range: tuple[float, float] = (50.0, 500.0),
) -> list[Path]:
    """Generate synthetic OHLCV bars per ticker over [start_date, end_date].

    `bar_minutes` is bar resolution (1 = M1). `bars_per_day` defaults to 390
    (US regular trading hours). Weekends are skipped. Per-ticker initial prices
    are sampled uniformly from `initial_price_range`. Returns the list of
    written Parquet file paths in ascending (year, month) order.
    """
    if start_date > end_date:
        raise ValueError("start_date must be <= end_date")
    if bar_minutes <= 0:
        raise ValueError("bar_minutes must be positive")
    if bars_per_day <= 0:
        raise ValueError("bars_per_day must be positive")
    if tickers is None:
        tickers = DEFAULT_BOOTSTRAP_UNIVERSE
    if not tickers:
        raise ValueError("tickers must be a non-empty mapping")
    lo, hi = initial_price_range
    if not (0.0 < lo <= hi):
        raise ValueError("initial_price_range must satisfy 0 < lo <= hi")

    rng = np.random.default_rng(seed)
    ticker_list = list(tickers.keys())

    days: list[date] = []
    d = start_date
    while d <= end_date:
        if d.weekday() < 5:
            days.append(d)
        d += timedelta(days=1)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not days:
        return []

    bars_per_ticker = len(days) * bars_per_day
    initials = rng.uniform(lo, hi, size=len(ticker_list))

    rows: list[dict] = []
    for ti, ticker in enumerate(ticker_list):
        prices = _gbm_path(
            rng, bars_per_ticker, initial=float(initials[ti]), bar_minutes=bar_minutes
        )
        volumes = rng.lognormal(mean=np.log(100_000), sigma=0.5, size=bars_per_ticker)
        high_offsets = np.abs(rng.normal(0, 0.001, size=bars_per_ticker)) * prices
        low_offsets = np.abs(rng.normal(0, 0.001, size=bars_per_ticker)) * prices

        bar_idx = 0
        for d_in in days:
            for b in range(bars_per_day):
                ts = datetime(
                    d_in.year, d_in.month, d_in.day, 9, 30, tzinfo=UTC
                ) + timedelta(minutes=bar_minutes * b)
                close = float(prices[bar_idx])
                open_p = float(prices[bar_idx - 1]) if bar_idx > 0 else close
                high_p = max(open_p, close) + float(high_offsets[bar_idx])
                low_p = min(open_p, close) - float(low_offsets[bar_idx])
                if low_p <= 0:
                    low_p = min(open_p, close) * 0.999
                rows.append(
                    {
                        "ticker": ticker,
                        "timestamp": ts.isoformat().replace("+00:00", "Z"),
                        "epoch_ns": _epoch_ns(ts),
                        "open": open_p,
                        "high": high_p,
                        "low": low_p,
                        "close": close,
                        "volume": int(volumes[bar_idx]),
                        "schema_version": SCHEMA_VERSION,
                        "_year": d_in.year,
                        "_month": d_in.month,
                    }
                )
                bar_idx += 1

    if not rows:
        return []

    df = pl.DataFrame(rows).select([*_COLUMN_ORDER, "_year", "_month"]).sort("epoch_ns")

    written: list[Path] = []
    for sub in df.partition_by(["_year", "_month"], maintain_order=True):
        year = int(sub["_year"][0])
        month = int(sub["_month"][0])
        part_dir = out_dir / f"year={year}" / f"month={month:02d}"
        part_dir.mkdir(parents=True, exist_ok=True)
        out_path = part_dir / "part-0.parquet"
        sub.drop(["_year", "_month"]).write_parquet(out_path)
        written.append(out_path)

    return sorted(written)
