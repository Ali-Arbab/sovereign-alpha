"""Module II strategy DSL -- composable boolean signals over a fused frame.

Per master directive section 4.3: strategy logic is declarative and composable.
Friction, sizing, and risk limits are layered as separate functions, never
inlined into the signal definition.

A Signal wraps a Polars expression that evaluates to a boolean column over a
fused (bars + ledger) DataFrame. Signals compose with `&`, `|`, and `~`.
A TargetPctRule pairs a signal with a sizing fraction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import polars as pl


class Signal:
    """A composable boolean predicate over a fused frame.

    Wraps a Polars `Expr` that, when evaluated against a frame, yields a
    boolean column the same height as the frame. Compose with `&`, `|`, `~`.
    """

    __slots__ = ("expr",)

    def __init__(self, expr: pl.Expr) -> None:
        self.expr = expr

    def __and__(self, other: Signal) -> Signal:
        return Signal(self.expr & other.expr)

    def __or__(self, other: Signal) -> Signal:
        return Signal(self.expr | other.expr)

    def __invert__(self) -> Signal:
        return Signal(~self.expr)

    def evaluate(self, frame: pl.DataFrame) -> pl.Series:
        """Evaluate the signal against `frame` and return a same-height boolean
        Series. Null values are coerced to False (no signal)."""
        result = frame.with_columns(self.expr.fill_null(False).alias("__sig__"))
        return result["__sig__"]


def col_gt(col: str, threshold: float) -> Signal:
    return Signal(pl.col(col) > threshold)


def col_ge(col: str, threshold: float) -> Signal:
    return Signal(pl.col(col) >= threshold)


def col_lt(col: str, threshold: float) -> Signal:
    return Signal(pl.col(col) < threshold)


def col_le(col: str, threshold: float) -> Signal:
    return Signal(pl.col(col) <= threshold)


def col_eq(col: str, value: Any) -> Signal:
    return Signal(pl.col(col) == value)


def always() -> Signal:
    """Signal that fires on every row -- useful for buy-and-hold baselines."""
    return Signal(pl.lit(True))


def never() -> Signal:
    return Signal(pl.lit(False))


@dataclass(frozen=True)
class TargetPctRule:
    """Allocate `target_pct` of current equity to instruments where `signal` fires.

    `target_pct` is a long-only fraction in [0, 1]. When `signal` fires on a
    row for ticker T, the runner targets `target_pct * equity_now / close`
    shares of T. When the signal does not fire, the target is zero (flat).
    """

    signal: Signal
    target_pct: float

    def __post_init__(self) -> None:
        if not (0.0 <= self.target_pct <= 1.0):
            raise ValueError("target_pct must be in [0, 1]")
