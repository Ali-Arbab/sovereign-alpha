# ADR-0001: Polars over Pandas for Module II

**Status:** Accepted
**Date:** 2026-04-28

## Context
Module II processes ~1.5M OHLCV bars per backtest and joins against a ~1.8M-row Alpha Ledger. Master directive §10 mandates "vectorized data handling" and "zero-copy, memory-mapped I/O." Backtests must run end-to-end inside a single overnight research loop, so a 10× performance differential at the data-frame layer compounds across iterations.

## Decision
**Use Polars (LazyFrames + memory-mapped Parquet) for every Module II data path.** Pandas is explicitly rejected.

## Alternatives considered
- **Pandas.** Default in the Python data ecosystem, but: chained operations create intermediate copies; no native lazy/optimized query plan; per-cell Python overhead dominates on the row-counts at play here.
- **DuckDB / SQLite.** Excellent query optimization, but pulls SQL into the strategy DSL boundary and breaks the Polars-native expression API used in `strategy.py`.
- **NumPy + hand-rolled joins.** Maximum control, minimum ergonomics. Unmaintainable as the join graph grows.

## Consequences
- API has a steeper learning curve for contributors used to Pandas.
- Tools that assume Pandas (some plotting libs, some sklearn integrations) need explicit `.to_pandas()` conversion at the call site — kept narrow and intentional.
- Polars' Rust-backed engine eliminates GIL contention for CPU-bound stages.
- `as_of_join` (ADR-0002) is a Polars-native operation, so the firewall and the engine are coherent.
