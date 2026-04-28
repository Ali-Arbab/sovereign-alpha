"""Module II data fusion -- `as_of_join` over the Alpha Ledger and OHLCV bars.

Strict invariant (master directive sections 4.1 / 4.2): every temporal merge
MUST be a backwards `as_of_join` on a monotonic timestamp column. Naive
`inner_join` is build-broken -- it silently rounds timestamps and can leak
future ledger rows into past pricing observations.

Both inputs are required to be sorted ascending on the join column. This is
checked, not assumed, on every call.
"""

from __future__ import annotations

import polars as pl

DEFAULT_JOIN_KEY = "epoch_ns"


def _is_sorted_ascending(df: pl.DataFrame, on: str) -> bool:
    if df.is_empty():
        return True
    col = df[on]
    diffs = col.diff().drop_nulls()
    if diffs.is_empty():
        return True
    return bool(diffs.min() >= 0)


def explode_ledger_entities(
    ledger: pl.DataFrame,
    *,
    list_col: str = "entities",
    flat_col: str = "entity",
) -> pl.DataFrame:
    """Explode a list-typed entities column so each row carries a single entity.

    Required before `as_of_fuse(by_left='ticker', by_right='entity')` when the
    ledger comes from Module I at the `(document, entity)` granularity but
    stores entities as a list per row.
    """
    if list_col not in ledger.columns:
        raise ValueError(f"ledger must have a {list_col!r} list column")
    return ledger.explode(list_col).rename({list_col: flat_col})


def as_of_fuse(
    bars: pl.DataFrame,
    ledger: pl.DataFrame,
    *,
    on: str = DEFAULT_JOIN_KEY,
    by_left: str | None = None,
    by_right: str | None = None,
) -> pl.DataFrame:
    """Backwards as_of join -- for each bar, attach the most recent ledger row
    with `<= bar.epoch_ns` (matched on `by_left == by_right` when both given).

    Use `by_left='ticker', by_right='entity'` when fusing per-ticker bars
    against an entity-exploded Alpha Ledger. Pass both as None for a global
    as_of fuse (every bar receives the latest ledger row regardless of ticker).
    """
    if not _is_sorted_ascending(bars, on):
        raise ValueError(f"bars must be sorted ascending on {on!r}")
    if not _is_sorted_ascending(ledger, on):
        raise ValueError(f"ledger must be sorted ascending on {on!r}")
    if (by_left is None) != (by_right is None):
        raise ValueError("by_left and by_right must both be set or both None")

    if by_left is None:
        return bars.join_asof(ledger, on=on, strategy="backward")
    return bars.join_asof(
        ledger, on=on, strategy="backward", by_left=by_left, by_right=by_right
    )
