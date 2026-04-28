"""Tests for the bootstrap-phase synthetic OHLCV generator."""

from datetime import date
from pathlib import Path

import polars as pl
import pytest

from modules.module_2_quant.synthetic_ohlcv import generate_synthetic_ohlcv
from shared.schemas.ohlcv_bar import OHLCVBar


def test_generates_partitioned_parquet(tmp_path: Path) -> None:
    paths = generate_synthetic_ohlcv(
        tmp_path,
        date(2020, 1, 28),
        date(2020, 2, 4),
        seed=0,
        bars_per_day=10,
    )
    rels = sorted(str(p.relative_to(tmp_path)).replace("\\", "/") for p in paths)
    assert rels == [
        "year=2020/month=01/part-0.parquet",
        "year=2020/month=02/part-0.parquet",
    ]


def test_skips_weekends(tmp_path: Path) -> None:
    paths = generate_synthetic_ohlcv(
        tmp_path,
        date(2020, 1, 24),
        date(2020, 1, 27),
        seed=0,
        bars_per_day=5,
    )
    df = pl.read_parquet(paths)
    days = {t[:10] for t in df["timestamp"].to_list()}
    assert days == {"2020-01-24", "2020-01-27"}


def test_records_validate_against_pydantic_schema(tmp_path: Path) -> None:
    paths = generate_synthetic_ohlcv(
        tmp_path, date(2020, 1, 27), date(2020, 1, 28), seed=42, bars_per_day=5
    )
    df = pl.read_parquet(paths)
    for row in df.head(50).iter_rows(named=True):
        OHLCVBar(**row)


def test_determinism_same_seed_byte_identical(tmp_path: Path) -> None:
    p_a = generate_synthetic_ohlcv(
        tmp_path / "a", date(2020, 1, 27), date(2020, 1, 28), seed=7, bars_per_day=5
    )
    p_b = generate_synthetic_ohlcv(
        tmp_path / "b", date(2020, 1, 27), date(2020, 1, 28), seed=7, bars_per_day=5
    )
    assert pl.read_parquet(p_a).equals(pl.read_parquet(p_b))


def test_epoch_ns_monotonic_within_each_partition(tmp_path: Path) -> None:
    paths = generate_synthetic_ohlcv(
        tmp_path, date(2020, 1, 28), date(2020, 2, 3), seed=0, bars_per_day=5
    )
    for p in paths:
        epoch = pl.read_parquet(p)["epoch_ns"].to_list()
        assert epoch == sorted(epoch), f"epoch_ns not monotonic in {p}"


def test_ohlcv_geometry_invariants(tmp_path: Path) -> None:
    paths = generate_synthetic_ohlcv(
        tmp_path, date(2020, 1, 27), date(2020, 1, 28), seed=3, bars_per_day=10
    )
    df = pl.read_parquet(paths)
    assert (df["high"] >= df["open"]).all()
    assert (df["high"] >= df["close"]).all()
    assert (df["low"] <= df["open"]).all()
    assert (df["low"] <= df["close"]).all()
    assert (df["low"] > 0).all()
    assert (df["volume"] >= 0).all()


def test_invalid_args(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="start_date"):
        generate_synthetic_ohlcv(tmp_path, date(2020, 1, 5), date(2020, 1, 1), seed=0)
    with pytest.raises(ValueError, match="bar_minutes"):
        generate_synthetic_ohlcv(
            tmp_path, date(2020, 1, 27), date(2020, 1, 28), seed=0, bar_minutes=0
        )
    with pytest.raises(ValueError, match="initial_price_range"):
        generate_synthetic_ohlcv(
            tmp_path,
            date(2020, 1, 27),
            date(2020, 1, 28),
            seed=0,
            initial_price_range=(100.0, 50.0),
        )
