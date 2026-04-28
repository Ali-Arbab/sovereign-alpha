"""Pandera DataFrame contracts -- schema-as-contract for in-flight Polars frames.

Per master directive section 0.5.1.A and 6.5. Pydantic schemas (in
`alpha_ledger.py`, `ohlcv_bar.py`) validate single-row records at the
serialization boundary; pandera contracts validate whole Polars
DataFrames in flight between Module II stages. Both must agree --
schema drift between them is a build-failure surface.
"""

from __future__ import annotations

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import Series


class AlphaLedgerFrame(pa.DataFrameModel):
    """In-flight Polars-frame contract for the Alpha Ledger (directive §3.4)."""

    doc_hash: Series[str]
    timestamp: Series[str]
    epoch_ns: Series[int] = pa.Field(ge=0)
    macro_sentiment: Series[float] = pa.Field(ge=-1.0, le=1.0)
    sector_sentiment: Series[float] = pa.Field(ge=-1.0, le=1.0)
    confidence_score: Series[float] = pa.Field(ge=0.0, le=1.0)
    regime_shift_flag: Series[bool]
    horizon_days: Series[int] = pa.Field(gt=0)
    reasoning_trace: Series[str]
    persona_id: Series[str]
    model_id: Series[str]
    schema_version: Series[str]

    class Config:
        strict = False  # tolerate the entities / sector_tags list columns


class OHLCVFrame(pa.DataFrameModel):
    """In-flight Polars-frame contract for OHLCV bars (directive §4.1)."""

    ticker: Series[str]
    timestamp: Series[str]
    epoch_ns: Series[int] = pa.Field(ge=0)
    open: Series[float] = pa.Field(gt=0.0)
    high: Series[float] = pa.Field(gt=0.0)
    low: Series[float] = pa.Field(gt=0.0)
    close: Series[float] = pa.Field(gt=0.0)
    volume: Series[int] = pa.Field(ge=0)
    schema_version: Series[str]

    class Config:
        strict = False


def validate_alpha_ledger_frame(df: pl.DataFrame) -> pl.DataFrame:
    """Validate a Polars Alpha Ledger frame; raises on schema drift or
    bound violations. Returns the same frame on success (chainable)."""
    return AlphaLedgerFrame.validate(df)


def validate_ohlcv_frame(df: pl.DataFrame) -> pl.DataFrame:
    """Validate a Polars OHLCV frame plus bar-geometry invariants
    (low <= open <= high, low <= close <= high). Raises ValueError on
    geometry breach -- pandera's per-column checks cannot express
    cross-column invariants."""
    OHLCVFrame.validate(df)
    # Cross-column geometry check (pandera-polars does not expose
    # @pa.dataframe_check at this version's stability level)
    if not df.is_empty():
        if (df["low"] > df["high"]).any():
            raise ValueError("OHLCV invariant: low > high in at least one row")
        if (df["open"] < df["low"]).any() or (df["open"] > df["high"]).any():
            raise ValueError("OHLCV invariant: open outside [low, high]")
        if (df["close"] < df["low"]).any() or (df["close"] > df["high"]).any():
            raise ValueError("OHLCV invariant: close outside [low, high]")
    return df
