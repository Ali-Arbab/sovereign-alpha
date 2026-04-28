"""Persona-tournament harness -- elimination brackets across rolling windows.

Per master directive section 8.2: "Personas compete in elimination brackets
across rolling 6-month windows. Survivor cohorts inform meta-strategies."

The harness orchestrates per-window backtests and aggregates rankings
across windows. The actual backtest runner is pluggable (BacktestRunner
callable), so the same harness drives both synthetic-data tournaments
in the bootstrap phase and real-Alpha-Ledger tournaments post-hardware.

`make_synthetic_runner` returns a runner that, for each `(persona_id,
window)`, generates a synthetic Alpha Ledger seeded by the persona id
(so each persona has a distinct sentiment distribution), fuses against
synthetic OHLCV, and runs a default sentiment-driven backtest. This
unblocks the tournament harness in the bootstrap phase.
"""

from __future__ import annotations

import math
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import polars as pl

from modules.module_1_extraction.synthetic_ledger import generate_synthetic_ledger
from modules.module_2_quant.backtest import BacktestConfig, run_backtest
from modules.module_2_quant.fusion import as_of_fuse, explode_ledger_entities
from modules.module_2_quant.strategy import TargetPctRule, col_gt
from modules.module_2_quant.synthetic_ohlcv import generate_synthetic_ohlcv

RANKABLE_METRICS: frozenset[str] = frozenset(
    {"sharpe", "sortino", "final_equity", "max_drawdown", "n_trades"}
)


@dataclass(frozen=True)
class TournamentWindow:
    """A contiguous date range over which each persona is evaluated once."""

    start_date: date
    end_date: date

    def __post_init__(self) -> None:
        if self.start_date > self.end_date:
            raise ValueError("start_date must be <= end_date")

    def label(self) -> str:
        return f"{self.start_date.isoformat()}_{self.end_date.isoformat()}"


@dataclass(frozen=True)
class WindowResult:
    """One persona's metrics for one tournament window.

    `max_drawdown` is a negative fraction (e.g. -0.07 means a 7% drawdown).
    Sorting descending naturally ranks "less drawdown" (closer to 0) first.
    """

    window: TournamentWindow
    persona_id: str
    sharpe: float
    sortino: float
    max_drawdown: float
    final_equity: float
    n_trades: int


@dataclass(frozen=True)
class TournamentReport:
    """Aggregate output of `run_tournament`."""

    windows: list[TournamentWindow]
    results: list[WindowResult]
    rankings: dict[str, list[str]]  # window_label -> persona_ids best-to-worst
    survivors: list[str]
    metric: str
    elimination_fraction: float

    def results_for(self, persona_id: str) -> list[WindowResult]:
        return [r for r in self.results if r.persona_id == persona_id]


BacktestRunner = Callable[[str, TournamentWindow], WindowResult]


def rolling_windows(
    start: date,
    end: date,
    *,
    window_days: int = 180,
    step_days: int = 30,
) -> list[TournamentWindow]:
    """Generate non-empty rolling windows of `window_days`, advancing `step_days`.

    The last window is the latest `[w_start, w_end]` with `w_end <= end`.
    """
    if start > end:
        raise ValueError("start must be <= end")
    if window_days <= 0 or step_days <= 0:
        raise ValueError("window_days and step_days must be positive")

    windows: list[TournamentWindow] = []
    w_start = start
    while True:
        w_end = w_start + timedelta(days=window_days - 1)
        if w_end > end:
            break
        windows.append(TournamentWindow(start_date=w_start, end_date=w_end))
        w_start = w_start + timedelta(days=step_days)
    return windows


def _metric_value(result: WindowResult, metric: str) -> float:
    if metric not in RANKABLE_METRICS:
        raise ValueError(f"metric must be one of {sorted(RANKABLE_METRICS)}")
    return float(getattr(result, metric))


def _rank_descending(
    results: list[WindowResult], metric: str
) -> list[WindowResult]:
    """Sort results best-first by `metric`. NaN metric values sort last."""

    def key(r: WindowResult) -> tuple[int, float]:
        v = _metric_value(r, metric)
        nan_flag = 1 if math.isnan(v) else 0  # NaN gets pushed to the bottom
        return (nan_flag, -v)

    return sorted(results, key=key)


def run_tournament(
    *,
    persona_ids: list[str],
    windows: list[TournamentWindow],
    runner: BacktestRunner,
    metric: str = "sharpe",
    elimination_fraction: float = 0.5,
    min_survivors_per_window: int = 1,
) -> TournamentReport:
    """Orchestrate an elimination-bracket tournament.

    For each window: run every persona, rank by `metric`, mark the top
    `1 - elimination_fraction` (rounded up, but never below
    `min_survivors_per_window`) as that window's survivors. Personas
    that survive every window are in `report.survivors`.
    """
    if not persona_ids:
        raise ValueError("persona_ids must be non-empty")
    if not windows:
        raise ValueError("windows must be non-empty")
    if not (0.0 < elimination_fraction < 1.0):
        raise ValueError("elimination_fraction must be in (0, 1)")
    if min_survivors_per_window < 1:
        raise ValueError("min_survivors_per_window must be >= 1")
    if metric not in RANKABLE_METRICS:
        raise ValueError(f"metric must be one of {sorted(RANKABLE_METRICS)}")

    rankings: dict[str, list[str]] = {}
    all_results: list[WindowResult] = []
    survival_count: dict[str, int] = dict.fromkeys(persona_ids, 0)

    for window in windows:
        window_results = [runner(p, window) for p in persona_ids]
        all_results.extend(window_results)

        ranked = _rank_descending(window_results, metric)
        rankings[window.label()] = [r.persona_id for r in ranked]

        n_keep = max(
            min_survivors_per_window,
            math.ceil(len(persona_ids) * (1.0 - elimination_fraction)),
        )
        for r in ranked[:n_keep]:
            survival_count[r.persona_id] += 1

    survivors = [p for p in persona_ids if survival_count[p] == len(windows)]

    return TournamentReport(
        windows=windows,
        results=all_results,
        rankings=rankings,
        survivors=survivors,
        metric=metric,
        elimination_fraction=elimination_fraction,
    )


def make_synthetic_runner(
    *,
    ohlcv_seed: int = 0,
    bars_per_day: int = 20,
    docs_per_day_mean: float = 20.0,
    initial_capital: float = 100_000.0,
    target_pct: float = 0.05,
    sentiment_threshold: float = 0.0,
) -> BacktestRunner:
    """Build a runner that drives the bootstrap-phase synthetic pipeline.

    For each `(persona_id, window)` the runner:
      1. Seeds the Alpha Ledger generator with `hash(persona_id)` so each
         persona produces a distinct (deterministic) sentiment distribution.
      2. Generates synthetic OHLCV with `ohlcv_seed` (shared across
         personas, since the bars are the market and don't depend on
         persona).
      3. Fuses + runs a sentiment-threshold backtest.
      4. Returns a WindowResult.

    Use a real BacktestRunner once Module I inference produces actual
    Alpha Ledgers per persona.
    """

    def _runner(persona_id: str, window: TournamentWindow) -> WindowResult:
        ledger_seed = abs(hash(persona_id)) % (2**32)
        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)
            ledger_paths = generate_synthetic_ledger(
                tmp / "ledger",
                window.start_date,
                window.end_date,
                seed=ledger_seed,
                docs_per_day_mean=docs_per_day_mean,
            )
            ohlcv_paths = generate_synthetic_ohlcv(
                tmp / "ohlcv",
                window.start_date,
                window.end_date,
                seed=ohlcv_seed,
                bars_per_day=bars_per_day,
            )
            ledger = explode_ledger_entities(pl.read_parquet(ledger_paths)).sort(
                "epoch_ns"
            )
            bars = pl.read_parquet(ohlcv_paths).sort("epoch_ns")
            fused = as_of_fuse(bars, ledger, by_left="ticker", by_right="entity")

            cfg = BacktestConfig(
                rule=TargetPctRule(
                    col_gt("macro_sentiment", sentiment_threshold), target_pct
                ),
                initial_capital=initial_capital,
            )
            result = run_backtest(fused, cfg)

        return WindowResult(
            window=window,
            persona_id=persona_id,
            sharpe=result.sharpe,
            sortino=result.sortino,
            max_drawdown=result.drawdown.max_drawdown,
            final_equity=result.final_equity,
            n_trades=result.trades.height,
        )

    return _runner
