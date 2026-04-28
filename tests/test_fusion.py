"""Tests for as_of_fuse -- Module II's only sanctioned temporal merge."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import polars as pl
import pytest

from modules.module_1_extraction.synthetic_ledger import generate_synthetic_ledger
from modules.module_2_quant.fusion import as_of_fuse, explode_ledger_entities
from modules.module_2_quant.synthetic_ohlcv import generate_synthetic_ohlcv


def _bars(rows: list[tuple[str, int, float]]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "ticker": [r[0] for r in rows],
            "epoch_ns": [r[1] for r in rows],
            "close": [r[2] for r in rows],
        }
    )


def _ledger(rows: list[tuple[str, int, float]]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "entity": [r[0] for r in rows],
            "epoch_ns": [r[1] for r in rows],
            "macro_sentiment": [r[2] for r in rows],
        }
    )


def test_basic_backwards_join() -> None:
    bars = _bars([("AAPL", 10, 100.0), ("AAPL", 20, 101.0), ("AAPL", 30, 102.0)])
    ledger = _ledger([("AAPL", 5, -0.5), ("AAPL", 15, 0.3), ("AAPL", 25, -0.1)])
    fused = as_of_fuse(bars, ledger, by_left="ticker", by_right="entity")
    assert fused["macro_sentiment"].to_list() == [-0.5, 0.3, -0.1]


def test_global_no_by_columns() -> None:
    bars = _bars([("AAPL", 10, 100.0), ("MSFT", 20, 200.0)])
    ledger = pl.DataFrame({"epoch_ns": [5], "macro_sentiment": [0.7]})
    fused = as_of_fuse(bars, ledger)
    assert fused["macro_sentiment"].to_list() == [0.7, 0.7]


def test_unsorted_bars_rejected() -> None:
    bars = _bars([("AAPL", 30, 102.0), ("AAPL", 10, 100.0)])
    ledger = _ledger([("AAPL", 5, -0.5)])
    with pytest.raises(ValueError, match="bars"):
        as_of_fuse(bars, ledger, by_left="ticker", by_right="entity")


def test_unsorted_ledger_rejected() -> None:
    bars = _bars([("AAPL", 10, 100.0)])
    ledger = _ledger([("AAPL", 30, -0.5), ("AAPL", 5, 0.0)])
    with pytest.raises(ValueError, match="ledger"):
        as_of_fuse(bars, ledger, by_left="ticker", by_right="entity")


def test_by_args_must_be_paired() -> None:
    bars = _bars([("AAPL", 10, 100.0)])
    ledger = _ledger([("AAPL", 5, 0.0)])
    with pytest.raises(ValueError, match="by_left and by_right"):
        as_of_fuse(bars, ledger, by_left="ticker", by_right=None)


def test_no_match_yields_null_sentiment() -> None:
    bars = _bars([("AAPL", 10, 100.0)])
    ledger = _ledger([("AAPL", 20, 0.5)])
    fused = as_of_fuse(bars, ledger, by_left="ticker", by_right="entity")
    assert fused["macro_sentiment"].to_list() == [None]


def test_explode_ledger_entities_round_trip() -> None:
    multi = pl.DataFrame(
        {
            "doc_hash": ["sha256:a", "sha256:b"],
            "epoch_ns": [1, 2],
            "entities": [["AAPL", "MSFT"], ["AAPL"]],
        }
    )
    flat = explode_ledger_entities(multi)
    assert flat.height == 3
    assert "entity" in flat.columns
    assert flat["entity"].to_list() == ["AAPL", "MSFT", "AAPL"]


def test_explode_ledger_entities_requires_list_column() -> None:
    bad = pl.DataFrame({"doc_hash": ["sha256:a"], "epoch_ns": [1]})
    with pytest.raises(ValueError, match="entities"):
        explode_ledger_entities(bad)


def test_e2e_synthetic_ledger_x_ohlcv_fuses(tmp_path: Path) -> None:
    """End-to-end: generate Alpha Ledger + OHLCV, explode, fuse, verify shape."""
    ledger_paths = generate_synthetic_ledger(
        tmp_path / "ledger",
        date(2020, 1, 27),
        date(2020, 1, 28),
        seed=0,
        docs_per_day_mean=20,
    )
    ohlcv_paths = generate_synthetic_ohlcv(
        tmp_path / "ohlcv",
        date(2020, 1, 27),
        date(2020, 1, 28),
        seed=0,
        bars_per_day=10,
    )
    ledger = (
        explode_ledger_entities(pl.read_parquet(ledger_paths)).sort("epoch_ns")
    )
    bars = pl.read_parquet(ohlcv_paths).sort("epoch_ns")

    fused = as_of_fuse(bars, ledger, by_left="ticker", by_right="entity")
    assert fused.height == bars.height
    # Most bars near the end of the day should have a ledger row attached
    assert fused["macro_sentiment"].drop_nulls().len() > 0
