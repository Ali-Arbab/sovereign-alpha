"""Tests for MonotonicCursor -- the strictly-monotonic backtest cursor."""

from __future__ import annotations

import polars as pl
import pytest

from modules.module_2_quant.cursor import MonotonicCursor


def _frame(epochs: list[int]) -> pl.DataFrame:
    return pl.DataFrame({"epoch_ns": epochs, "v": list(range(len(epochs)))})


def test_advance_to_filters_to_past_only() -> None:
    cursor = MonotonicCursor(_frame([10, 20, 30, 40, 50]))
    snap = cursor.advance_to(25)
    assert snap["epoch_ns"].to_list() == [10, 20]


def test_advance_to_inclusive_at_exact_match() -> None:
    cursor = MonotonicCursor(_frame([10, 20, 30]))
    snap = cursor.advance_to(20)
    assert snap["epoch_ns"].to_list() == [10, 20]


def test_advance_forward_succeeds() -> None:
    cursor = MonotonicCursor(_frame([10, 20, 30, 40]))
    cursor.advance_to(15)
    snap2 = cursor.advance_to(35)
    assert snap2["epoch_ns"].to_list() == [10, 20, 30]
    assert cursor.current_time == 35


def test_backwards_movement_raises() -> None:
    cursor = MonotonicCursor(_frame([10, 20, 30]))
    cursor.advance_to(25)
    with pytest.raises(ValueError, match="backwards"):
        cursor.advance_to(15)


def test_unsorted_frame_rejected() -> None:
    bad = pl.DataFrame({"epoch_ns": [30, 10, 20], "v": [0, 0, 0]})
    with pytest.raises(ValueError, match="sorted"):
        MonotonicCursor(bad)


def test_missing_key_column_raises() -> None:
    df = pl.DataFrame({"x": [1, 2, 3]})
    with pytest.raises(ValueError, match="cursor key"):
        MonotonicCursor(df)


def test_empty_frame_is_ok() -> None:
    empty = pl.DataFrame({"epoch_ns": [], "v": []}, schema={"epoch_ns": pl.Int64, "v": pl.Int64})
    cursor = MonotonicCursor(empty)
    assert cursor.advance_to(100).is_empty()
