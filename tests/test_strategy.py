"""Tests for the Module II strategy DSL -- Signal composition + TargetPctRule."""

from __future__ import annotations

import polars as pl
import pytest

from modules.module_2_quant.strategy import (
    Signal,
    TargetPctRule,
    always,
    col_eq,
    col_ge,
    col_gt,
    col_le,
    col_lt,
    never,
)


def _frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "x": [1, 2, 3, 4, 5],
            "y": [10.0, 20.0, 30.0, 40.0, 50.0],
            "regime": [False, True, False, True, False],
        }
    )


def test_col_gt_basic() -> None:
    sig = col_gt("x", 2)
    assert sig.evaluate(_frame()).to_list() == [False, False, True, True, True]


def test_col_lt_basic() -> None:
    sig = col_lt("y", 30.0)
    assert sig.evaluate(_frame()).to_list() == [True, True, False, False, False]


def test_col_eq_basic() -> None:
    assert col_eq("regime", True).evaluate(_frame()).to_list() == [
        False, True, False, True, False,
    ]


def test_col_ge_and_le() -> None:
    assert col_ge("x", 3).evaluate(_frame()).to_list() == [False, False, True, True, True]
    assert col_le("y", 30.0).evaluate(_frame()).to_list() == [True, True, True, False, False]


def test_signal_and() -> None:
    sig = col_gt("x", 1) & col_lt("y", 40.0)
    assert sig.evaluate(_frame()).to_list() == [False, True, True, False, False]


def test_signal_or() -> None:
    sig = col_lt("x", 2) | col_gt("y", 40.0)
    assert sig.evaluate(_frame()).to_list() == [True, False, False, False, True]


def test_signal_not() -> None:
    sig = ~col_gt("x", 3)
    assert sig.evaluate(_frame()).to_list() == [True, True, True, False, False]


def test_always_and_never() -> None:
    assert always().evaluate(_frame()).to_list() == [True] * 5
    assert never().evaluate(_frame()).to_list() == [False] * 5


def test_null_values_coerced_to_false() -> None:
    f = pl.DataFrame({"x": [1, None, 3]})
    sig = col_gt("x", 0)
    # x[1] = None -> comparison yields None -> coerced to False
    assert sig.evaluate(f).to_list() == [True, False, True]


def test_complex_composition() -> None:
    sig = (col_gt("x", 1) & col_lt("y", 50.0)) | col_eq("regime", True)
    assert sig.evaluate(_frame()).to_list() == [False, True, True, True, False]


def test_target_pct_rule_validates_bounds() -> None:
    sig = always()
    TargetPctRule(sig, target_pct=0.0)
    TargetPctRule(sig, target_pct=1.0)
    TargetPctRule(sig, target_pct=0.5)
    with pytest.raises(ValueError, match="target_pct"):
        TargetPctRule(sig, target_pct=-0.01)
    with pytest.raises(ValueError, match="target_pct"):
        TargetPctRule(sig, target_pct=1.01)


def test_signal_is_pure_polars_expr() -> None:
    """Signal must wrap a pl.Expr -- the public surface other code relies on."""
    sig = col_gt("x", 0)
    assert isinstance(sig, Signal)
    assert hasattr(sig, "expr")
