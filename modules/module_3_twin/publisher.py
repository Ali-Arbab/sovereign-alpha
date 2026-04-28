"""Mock state publisher -- emits Module II backtest state over the UE5 bridge.

Per master directive section 0.5.1.C. Walks a `BacktestResult` plus its source
fused frame and emits prices / trades / portfolio_state messages in epoch_ns
order. Lets UE5 development proceed against a deterministic synthetic stream
without needing a live backtest each session.

The `send` callback decouples this from any specific transport -- pass
`Publisher.publish` in production, or a list-appending closure in tests.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import polars as pl
from pydantic import BaseModel

from modules.module_2_quant.backtest import BacktestResult
from modules.module_3_twin.messages import (
    PORTFOLIO_STATE,
    PRICES,
    TRADES,
    PortfolioStateMessage,
    PriceTickMessage,
    TradeMessage,
)

SendFn = Callable[[str, BaseModel], None]


def _index_by_epoch(df: pl.DataFrame) -> dict[int, list[dict[str, Any]]]:
    out: dict[int, list[dict[str, Any]]] = {}
    if df.is_empty():
        return out
    for row in df.iter_rows(named=True):
        out.setdefault(int(row["epoch_ns"]), []).append(row)
    return out


def publish_backtest_state(
    send: SendFn,
    fused: pl.DataFrame,
    result: BacktestResult,
    *,
    delay_per_epoch_s: float = 0.0,
) -> int:
    """Emit prices / trades / portfolio_state in epoch_ns order. Returns the
    total number of messages sent.

    Required `fused` columns: ticker, epoch_ns, open, high, low, close, volume.
    `result.trades` columns are taken from the Module II runner contract.
    `result.equity_curve` columns: epoch_ns, cash, equity.
    """
    required_bar_cols = {"ticker", "epoch_ns", "open", "high", "low", "close", "volume"}
    missing = required_bar_cols - set(fused.columns)
    if missing:
        raise ValueError(f"fused frame missing required columns: {sorted(missing)}")

    trades_by_epoch = _index_by_epoch(result.trades)
    equity_by_epoch: dict[int, dict[str, Any]] = {}
    if not result.equity_curve.is_empty():
        for row in result.equity_curve.iter_rows(named=True):
            equity_by_epoch[int(row["epoch_ns"])] = row

    sent = 0
    seen_epochs: set[int] = set()
    for row in fused.iter_rows(named=True):
        epoch = int(row["epoch_ns"])
        send(
            PRICES,
            PriceTickMessage(
                epoch_ns=epoch,
                ticker=str(row["ticker"]),
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=int(row["volume"]),
            ),
        )
        sent += 1

        if epoch in seen_epochs:
            continue
        seen_epochs.add(epoch)

        for trade in trades_by_epoch.get(epoch, []):
            send(
                TRADES,
                TradeMessage(
                    epoch_ns=epoch,
                    ticker=str(trade["ticker"]),
                    side=str(trade["side"]),
                    qty=int(trade["qty"]),
                    avg_fill_price=float(trade["avg_fill_price"]),
                    slippage_cost=float(trade["slippage_cost"]),
                    commission=float(trade["commission"]),
                    doc_hash=str(trade.get("doc_hash") or ""),
                ),
            )
            sent += 1

        if epoch in equity_by_epoch:
            eq = equity_by_epoch[epoch]
            send(
                PORTFOLIO_STATE,
                PortfolioStateMessage(
                    epoch_ns=epoch,
                    cash=float(eq["cash"]),
                    equity=float(eq["equity"]),
                ),
            )
            sent += 1

        if delay_per_epoch_s > 0:
            time.sleep(delay_per_epoch_s)

    return sent
