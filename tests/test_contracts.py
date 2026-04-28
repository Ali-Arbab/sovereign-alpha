"""Tests for the Pandera DataFrame contracts."""

from __future__ import annotations

from datetime import date

import pandera.errors as pa_errors
import polars as pl
import pytest

from modules.module_1_extraction.synthetic_ledger import generate_synthetic_ledger
from modules.module_2_quant.synthetic_ohlcv import generate_synthetic_ohlcv
from shared.schemas.contracts import (
    validate_alpha_ledger_frame,
    validate_ohlcv_frame,
)


def _valid_alpha_ledger_row() -> dict:
    return {
        "doc_hash": "sha256:" + "0" * 64,
        "timestamp": "2024-05-01T12:00:00Z",
        "epoch_ns": 1_700_000_000_000_000_000,
        "entities": ["AAPL"],
        "sector_tags": ["consumer_electronics"],
        "macro_sentiment": 0.5,
        "sector_sentiment": 0.4,
        "confidence_interval": [0.85, 0.95],
        "confidence_score": 0.9,
        "regime_shift_flag": False,
        "horizon_days": 90,
        "reasoning_trace": "x",
        "persona_id": "p_v1",
        "model_id": "m",
        "schema_version": "1.0.0",
    }


def _valid_ohlcv_row() -> dict:
    return {
        "ticker": "AAPL",
        "timestamp": "2024-05-01T12:00:00Z",
        "epoch_ns": 1_700_000_000_000_000_000,
        "open": 100.0,
        "high": 101.0,
        "low": 99.0,
        "close": 100.5,
        "volume": 1_000_000,
        "schema_version": "1.0.0",
    }


def test_validate_alpha_ledger_accepts_valid_frame() -> None:
    df = pl.DataFrame([_valid_alpha_ledger_row()])
    out = validate_alpha_ledger_frame(df)
    assert out.height == 1


def test_validate_alpha_ledger_rejects_sentiment_out_of_bounds() -> None:
    bad = _valid_alpha_ledger_row() | {"macro_sentiment": 1.5}
    df = pl.DataFrame([bad])
    with pytest.raises(pa_errors.SchemaError):
        validate_alpha_ledger_frame(df)


def test_validate_alpha_ledger_rejects_negative_horizon() -> None:
    bad = _valid_alpha_ledger_row() | {"horizon_days": 0}
    df = pl.DataFrame([bad])
    with pytest.raises(pa_errors.SchemaError):
        validate_alpha_ledger_frame(df)


def test_validate_alpha_ledger_rejects_confidence_above_one() -> None:
    bad = _valid_alpha_ledger_row() | {"confidence_score": 1.5}
    df = pl.DataFrame([bad])
    with pytest.raises(pa_errors.SchemaError):
        validate_alpha_ledger_frame(df)


def test_validate_ohlcv_accepts_valid_frame() -> None:
    df = pl.DataFrame([_valid_ohlcv_row()])
    out = validate_ohlcv_frame(df)
    assert out.height == 1


def test_validate_ohlcv_rejects_zero_price() -> None:
    bad = _valid_ohlcv_row() | {"open": 0.0}
    df = pl.DataFrame([bad])
    with pytest.raises(pa_errors.SchemaError):
        validate_ohlcv_frame(df)


def test_validate_ohlcv_rejects_low_above_high() -> None:
    bad = _valid_ohlcv_row() | {"low": 102.0}  # low > high
    df = pl.DataFrame([bad])
    with pytest.raises(ValueError, match="low > high"):
        validate_ohlcv_frame(df)


def test_validate_ohlcv_rejects_open_outside_low_high() -> None:
    bad = _valid_ohlcv_row() | {"open": 200.0}  # open > high
    df = pl.DataFrame([bad])
    with pytest.raises(ValueError, match="open"):
        validate_ohlcv_frame(df)


def test_validate_ohlcv_rejects_close_outside_low_high() -> None:
    bad = _valid_ohlcv_row() | {"close": 50.0}  # close < low
    df = pl.DataFrame([bad])
    with pytest.raises(ValueError, match="close"):
        validate_ohlcv_frame(df)


def test_validate_synthetic_ledger_passes(tmp_path) -> None:
    """The synthetic-ledger generator must produce frames the contract accepts."""
    paths = generate_synthetic_ledger(
        tmp_path,
        date(2024, 1, 2),
        date(2024, 1, 4),
        seed=0,
        docs_per_day_mean=10.0,
    )
    df = pl.read_parquet(paths)
    assert validate_alpha_ledger_frame(df).height == df.height


def test_validate_synthetic_ohlcv_passes(tmp_path) -> None:
    """The synthetic-OHLCV generator must produce frames the contract accepts."""
    paths = generate_synthetic_ohlcv(
        tmp_path,
        date(2024, 1, 2),
        date(2024, 1, 3),
        seed=0,
        bars_per_day=5,
    )
    df = pl.read_parquet(paths)
    assert validate_ohlcv_frame(df).height == df.height


def test_validate_ohlcv_empty_frame_short_circuits() -> None:
    """Geometry checks must not crash on an empty frame."""
    empty = pl.DataFrame(
        schema={
            "ticker": pl.String,
            "timestamp": pl.String,
            "epoch_ns": pl.Int64,
            "open": pl.Float64,
            "high": pl.Float64,
            "low": pl.Float64,
            "close": pl.Float64,
            "volume": pl.Int64,
            "schema_version": pl.String,
        }
    )
    assert validate_ohlcv_frame(empty).height == 0
