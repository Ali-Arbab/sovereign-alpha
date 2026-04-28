"""Tests for the bootstrap-phase synthetic Alpha Ledger generator."""

from datetime import date
from pathlib import Path

import polars as pl
import pytest

from modules.module_1_extraction.synthetic_ledger import (
    MODEL_ID,
    PERSONA_ID,
    generate_synthetic_ledger,
)
from shared.schemas.alpha_ledger import AlphaLedgerRecord


def test_generates_partitioned_parquet(tmp_path: Path) -> None:
    paths = generate_synthetic_ledger(
        tmp_path,
        date(2020, 1, 28),
        date(2020, 2, 3),
        seed=0,
        docs_per_day_mean=10.0,
    )
    rels = sorted(str(p.relative_to(tmp_path)).replace("\\", "/") for p in paths)
    assert rels == [
        "year=2020/month=01/part-0.parquet",
        "year=2020/month=02/part-0.parquet",
    ]


def test_records_validate_against_pydantic_schema(tmp_path: Path) -> None:
    paths = generate_synthetic_ledger(
        tmp_path,
        date(2020, 1, 1),
        date(2020, 1, 3),
        seed=42,
        docs_per_day_mean=8.0,
    )
    df = pl.read_parquet(paths)
    assert df.height > 0
    for row in df.head(50).iter_rows(named=True):
        AlphaLedgerRecord(**row)


def test_determinism_same_seed_byte_identical(tmp_path: Path) -> None:
    paths_a = generate_synthetic_ledger(
        tmp_path / "a", date(2020, 1, 1), date(2020, 1, 5), seed=7, docs_per_day_mean=8.0
    )
    paths_b = generate_synthetic_ledger(
        tmp_path / "b", date(2020, 1, 1), date(2020, 1, 5), seed=7, docs_per_day_mean=8.0
    )
    assert pl.read_parquet(paths_a).equals(pl.read_parquet(paths_b))


def test_different_seeds_differ(tmp_path: Path) -> None:
    paths_a = generate_synthetic_ledger(
        tmp_path / "a", date(2020, 1, 1), date(2020, 1, 2), seed=1, docs_per_day_mean=10.0
    )
    paths_b = generate_synthetic_ledger(
        tmp_path / "b", date(2020, 1, 1), date(2020, 1, 2), seed=2, docs_per_day_mean=10.0
    )
    assert not pl.read_parquet(paths_a).equals(pl.read_parquet(paths_b))


def test_epoch_ns_monotonic_within_each_partition(tmp_path: Path) -> None:
    paths = generate_synthetic_ledger(
        tmp_path, date(2020, 1, 28), date(2020, 2, 2), seed=0, docs_per_day_mean=12.0
    )
    for p in paths:
        epoch = pl.read_parquet(p)["epoch_ns"].to_list()
        assert epoch == sorted(epoch), f"epoch_ns not monotonic in {p}"


def test_sentiment_and_confidence_bounds(tmp_path: Path) -> None:
    paths = generate_synthetic_ledger(
        tmp_path, date(2020, 1, 1), date(2020, 1, 5), seed=3, docs_per_day_mean=20.0
    )
    df = pl.read_parquet(paths)
    assert df["macro_sentiment"].min() >= -1.0
    assert df["macro_sentiment"].max() <= 1.0
    assert df["sector_sentiment"].min() >= -1.0
    assert df["sector_sentiment"].max() <= 1.0
    assert df["confidence_score"].min() >= 0.0
    assert df["confidence_score"].max() <= 1.0


def test_synthetic_tagging_so_records_are_unmistakable(tmp_path: Path) -> None:
    paths = generate_synthetic_ledger(
        tmp_path, date(2020, 1, 1), date(2020, 1, 2), seed=0, docs_per_day_mean=10.0
    )
    df = pl.read_parquet(paths)
    assert (df["persona_id"] == PERSONA_ID).all()
    assert (df["model_id"] == MODEL_ID).all()
    assert PERSONA_ID == "bootstrap_synthetic_v1"
    assert MODEL_ID == "synthetic"


def test_invalid_args(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="start_date"):
        generate_synthetic_ledger(tmp_path, date(2020, 1, 5), date(2020, 1, 1), seed=0)
    with pytest.raises(ValueError, match="docs_per_day_mean"):
        generate_synthetic_ledger(
            tmp_path, date(2020, 1, 1), date(2020, 1, 2), seed=0, docs_per_day_mean=0.0
        )
    with pytest.raises(ValueError, match="tickers"):
        generate_synthetic_ledger(
            tmp_path, date(2020, 1, 1), date(2020, 1, 2), seed=0, tickers={}
        )
