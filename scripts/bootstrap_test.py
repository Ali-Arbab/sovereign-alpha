"""End-to-end bootstrap test -- the §0.5.3 definition-of-done entrypoint.

Stitches every Module of the bootstrap-phase pipeline together:

1. Synthetic Alpha Ledger    (Module I substitute)
2. Synthetic OHLCV           (Module II input)
3. as_of_fuse + temporal cursor (Module II data fusion + firewall)
4. Sentiment-driven backtest (Module II runner)
5. Mock UE5 publisher        (Module III)

Asserts every stage produces non-trivial output, prints a summary, exits 0
on success and 1 on any failure. Should complete in well under 5 minutes on
commodity hardware (typical: 5-15 seconds).

Run with:
    make bootstrap-test
or
    uv run python scripts/bootstrap_test.py
"""

from __future__ import annotations

import sys
import tempfile
import time
from datetime import date
from pathlib import Path

import polars as pl

from modules.module_1_extraction.synthetic_ledger import generate_synthetic_ledger
from modules.module_2_quant.backtest import BacktestConfig, run_backtest
from modules.module_2_quant.fusion import as_of_fuse, explode_ledger_entities
from modules.module_2_quant.strategy import TargetPctRule, col_gt
from modules.module_2_quant.synthetic_ohlcv import generate_synthetic_ohlcv
from modules.module_3_twin.publisher import publish_backtest_state


def main() -> int:
    print("=" * 64)
    print("Sovereign Alpha bootstrap-test (master directive §0.5.3)")
    print("=" * 64)
    t0 = time.monotonic()

    with tempfile.TemporaryDirectory() as tmp_str:
        tmp = Path(tmp_str)

        print("\n[1/5] generate synthetic Alpha Ledger ...")
        ledger_paths = generate_synthetic_ledger(
            tmp / "ledger",
            date(2024, 1, 2),
            date(2024, 1, 12),
            seed=42,
            docs_per_day_mean=25.0,
        )
        assert ledger_paths, "synthetic Alpha Ledger produced no partitions"
        print(f"      -> {len(ledger_paths)} parquet partitions")

        print("\n[2/5] generate synthetic OHLCV ...")
        ohlcv_paths = generate_synthetic_ohlcv(
            tmp / "ohlcv",
            date(2024, 1, 2),
            date(2024, 1, 12),
            seed=42,
            bars_per_day=20,
        )
        assert ohlcv_paths, "synthetic OHLCV produced no partitions"
        print(f"      -> {len(ohlcv_paths)} parquet partitions")

        print("\n[3/5] as_of_fuse(bars, exploded_ledger) ...")
        ledger = (
            explode_ledger_entities(pl.read_parquet(ledger_paths)).sort("epoch_ns")
        )
        bars = pl.read_parquet(ohlcv_paths).sort("epoch_ns")
        fused = as_of_fuse(bars, ledger, by_left="ticker", by_right="entity")
        assert fused.height == bars.height, "fusion lost rows"
        print(f"      -> {fused.height} fused rows; "
              f"{fused['macro_sentiment'].drop_nulls().len()} carry sentiment")

        print("\n[4/5] backtest sentiment-driven strategy ...")
        rule = TargetPctRule(col_gt("macro_sentiment", 0.0), 0.05)
        cfg = BacktestConfig(rule=rule, initial_capital=100_000.0)
        result = run_backtest(fused, cfg)
        assert result.equity_curve.height > 0
        print(f"      -> {result.trades.height} trades, "
              f"final equity = ${result.final_equity:,.2f}")
        print(f"      -> Sharpe = {result.sharpe:.3f}, "
              f"Sortino = {result.sortino:.3f}, "
              f"max DD = {result.drawdown.max_drawdown * 100:.2f}%")

        print("\n[5/5] publish backtest state to Module III mock bus ...")
        captured: list[tuple[str, object]] = []
        sent = publish_backtest_state(
            lambda topic, msg: captured.append((topic, msg)), fused, result
        )
        topics = {t for t, _ in captured}
        assert sent > 0, "publisher emitted no messages"
        print(f"      -> {sent} messages across topics: {sorted(topics)}")

    elapsed = time.monotonic() - t0
    print("\n" + "=" * 64)
    print(f"bootstrap-test PASSED in {elapsed:.2f}s")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    sys.exit(main())
