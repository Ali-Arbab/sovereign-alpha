"""Strictly monotonic backtest cursor.

Sentinel for the temporal firewall: state at simulation time `t` may evaluate
ONLY rows where `epoch_ns <= t`. The cursor enforces both directions:

1. `advance_to(t)` returns a snapshot filtered to past-only rows.
2. Going backwards (`t < previous_t`) is a programmer error and raises.

Every Module II strategy sees the world through a cursor; raw frame access is
structurally invisible to strategy code.
"""

from __future__ import annotations

import polars as pl


class MonotonicCursor:
    """Cursor over a Polars DataFrame keyed on a monotonic integer column."""

    def __init__(self, frame: pl.DataFrame, *, on: str = "epoch_ns") -> None:
        if on not in frame.columns:
            raise ValueError(f"cursor key {on!r} not in frame columns")
        col = frame[on]
        if not col.is_empty():
            diffs = col.diff().drop_nulls()
            if not diffs.is_empty() and diffs.min() < 0:
                raise ValueError(f"frame must be sorted ascending on {on!r}")
        self._frame = frame
        self._on = on
        self._t = -1

    def advance_to(self, t: int) -> pl.DataFrame:
        """Return rows with `epoch_ns <= t`. Raises if t is less than previous t."""
        if t < self._t:
            raise ValueError(
                f"cursor cannot move backwards: {self._t} -> {t} "
                f"(temporal firewall, directive section 6.4)"
            )
        self._t = t
        return self._frame.filter(pl.col(self._on) <= t)

    @property
    def current_time(self) -> int:
        return self._t
