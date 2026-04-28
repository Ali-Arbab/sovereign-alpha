"""Module II backtest runner -- monotonic, friction-adjusted, attribution-bearing.

Per master directive section 4. Runs a strategy over a fused (bars + ledger)
frame under a strictly-monotonic cursor over `epoch_ns`. Per-trade attribution
links each trade back to the matched Alpha Ledger `doc_hash` (directive §4.5).

This is a v0 single-pass runner, long-only, with mark-to-market on close. Order
sizing for a buy is capped to available cash; order sizing for a sell is capped
to currently-held shares (no shorting in v0). Borrow costs are reachable via
FrictionModel.borrow_cost but not yet wired -- they will land alongside the
short-side rule.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import polars as pl

from modules.module_2_quant.friction import FrictionModel, Side
from modules.module_2_quant.metrics import (
    DrawdownReport,
    drawdown_report,
    sharpe_ratio,
    sortino_ratio,
)
from modules.module_2_quant.strategy import TargetPctRule


@dataclass(frozen=True)
class BacktestResult:
    equity_curve: pl.DataFrame  # one row per timestep: epoch_ns, equity, cash
    trades: pl.DataFrame  # one row per fill: epoch_ns, ticker, side, qty, ...
    final_equity: float
    sharpe: float
    sortino: float
    drawdown: DrawdownReport
    initial_capital: float
    periods_per_year: int


@dataclass(frozen=True)
class BacktestConfig:
    rule: TargetPctRule
    initial_capital: float = 100_000.0
    friction: FrictionModel = field(default_factory=FrictionModel)
    periods_per_year: int = 252


def _is_sorted_ascending(df: pl.DataFrame, on: str) -> bool:
    if df.is_empty():
        return True
    diffs = df[on].diff().drop_nulls()
    return diffs.is_empty() or bool(diffs.min() >= 0)


def _empty_equity_df() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "epoch_ns": pl.Int64,
            "cash": pl.Float64,
            "equity": pl.Float64,
        }
    )


def _empty_trades_df() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "epoch_ns": pl.Int64,
            "ticker": pl.String,
            "side": pl.String,
            "qty": pl.Int64,
            "avg_fill_price": pl.Float64,
            "slippage_cost": pl.Float64,
            "commission": pl.Float64,
            "doc_hash": pl.String,
        }
    )


def run_backtest(fused: pl.DataFrame, config: BacktestConfig) -> BacktestResult:
    """Run a backtest over a fused (bars + ledger) DataFrame.

    Required columns on `fused`: ticker, epoch_ns, close, volume. Optional:
    doc_hash (for per-trade attribution) plus any columns referenced by the
    signal in `config.rule`.
    """
    required = {"ticker", "epoch_ns", "close", "volume"}
    missing = required - set(fused.columns)
    if missing:
        raise ValueError(f"fused frame missing required columns: {sorted(missing)}")
    if not _is_sorted_ascending(fused, "epoch_ns"):
        raise ValueError("fused frame must be sorted ascending on epoch_ns")

    if fused.is_empty():
        return BacktestResult(
            equity_curve=_empty_equity_df(),
            trades=_empty_trades_df(),
            final_equity=config.initial_capital,
            sharpe=float("nan"),
            sortino=float("nan"),
            drawdown=DrawdownReport(0.0, 0, 0, 0),
            initial_capital=config.initial_capital,
            periods_per_year=config.periods_per_year,
        )

    signal_values = config.rule.signal.evaluate(fused).to_list()

    cash: float = config.initial_capital
    positions: dict[str, int] = {}
    last_close: dict[str, float] = {}

    equity_rows: list[dict] = []
    trade_rows: list[dict] = []

    for i, row in enumerate(fused.iter_rows(named=True)):
        epoch = int(row["epoch_ns"])
        ticker = str(row["ticker"])
        close = float(row["close"])
        volume = int(row["volume"])
        last_close[ticker] = close

        equity_now = cash + sum(positions.get(t, 0) * last_close[t] for t in last_close)

        signal_fires = bool(signal_values[i]) if signal_values[i] is not None else False
        target_position = (
            int((config.rule.target_pct * equity_now) // close) if signal_fires else 0
        )

        current = positions.get(ticker, 0)
        order_qty = target_position - current

        if order_qty > 0:
            max_affordable = int(cash // close)
            order_qty = min(order_qty, max_affordable)
        elif order_qty < 0:
            order_qty = max(order_qty, -current)

        if order_qty != 0:
            side = Side.BUY if order_qty > 0 else Side.SELL
            qty_request = abs(order_qty)
            fill = config.friction.fill(
                side=side,
                qty_requested=qty_request,
                bar_volume=volume,
                bar_price=close,
            )
            if fill.filled_qty > 0:
                if side is Side.BUY:
                    cash -= fill.filled_qty * fill.avg_fill_price + fill.commission
                    positions[ticker] = current + fill.filled_qty
                else:
                    cash += fill.filled_qty * fill.avg_fill_price - fill.commission
                    positions[ticker] = current - fill.filled_qty

                trade_rows.append(
                    {
                        "epoch_ns": epoch,
                        "ticker": ticker,
                        "side": side.value,
                        "qty": fill.filled_qty,
                        "avg_fill_price": fill.avg_fill_price,
                        "slippage_cost": fill.slippage_cost,
                        "commission": fill.commission,
                        "doc_hash": str(row.get("doc_hash") or ""),
                    }
                )

        equity_after = cash + sum(
            positions.get(t, 0) * last_close[t] for t in last_close
        )
        equity_rows.append(
            {"epoch_ns": epoch, "cash": cash, "equity": equity_after}
        )

    equity_df = pl.DataFrame(equity_rows)
    equity_timeline = equity_df.group_by("epoch_ns", maintain_order=True).agg(
        pl.col("cash").last(), pl.col("equity").last()
    )

    if equity_timeline.height > 1:
        equity_arr = equity_timeline["equity"].to_numpy()
        if (equity_arr <= 0).any():
            sr = float("nan")
            so = float("nan")
            dr = DrawdownReport(0.0, 0, 0, 0)
        else:
            returns = np.diff(equity_arr) / equity_arr[:-1]
            sr = sharpe_ratio(returns, periods_per_year=config.periods_per_year)
            so = sortino_ratio(returns, periods_per_year=config.periods_per_year)
            dr = drawdown_report(equity_arr)
        final_equity = float(equity_arr[-1])
    else:
        sr = float("nan")
        so = float("nan")
        dr = DrawdownReport(0.0, 0, 0, 0)
        final_equity = (
            float(equity_timeline["equity"][-1])
            if equity_timeline.height
            else config.initial_capital
        )

    trades_df = pl.DataFrame(trade_rows) if trade_rows else _empty_trades_df()

    return BacktestResult(
        equity_curve=equity_timeline,
        trades=trades_df,
        final_equity=final_equity,
        sharpe=sr,
        sortino=so,
        drawdown=dr,
        initial_capital=config.initial_capital,
        periods_per_year=config.periods_per_year,
    )
