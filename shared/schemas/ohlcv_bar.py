"""OHLCV bar schema -- Module II's primary pricing-input contract.

One row per (ticker, timestamp) pair. M1 (1-minute) bars by default; S1 variants
share this schema with finer-grained timestamps. Schema drift is a build failure
(directive section 6.5).
"""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, model_validator

SCHEMA_VERSION = "1.0.0"


class OHLCVBar(BaseModel):
    """Single OHLCV bar -- open/high/low/close + volume for one (ticker, timestamp)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    ticker: str
    timestamp: str
    epoch_ns: Annotated[int, Field(ge=0)]
    open: Annotated[float, Field(gt=0.0)]
    high: Annotated[float, Field(gt=0.0)]
    low: Annotated[float, Field(gt=0.0)]
    close: Annotated[float, Field(gt=0.0)]
    volume: Annotated[int, Field(ge=0)]
    schema_version: Annotated[str, Field(pattern=r"^\d+\.\d+\.\d+$")] = SCHEMA_VERSION

    @model_validator(mode="after")
    def _bar_geometry(self) -> OHLCVBar:
        if not (self.low <= self.open <= self.high):
            raise ValueError(
                f"OHLCV invariant violated: open ({self.open}) outside "
                f"[low ({self.low}), high ({self.high})]"
            )
        if not (self.low <= self.close <= self.high):
            raise ValueError(
                f"OHLCV invariant violated: close ({self.close}) outside "
                f"[low ({self.low}), high ({self.high})]"
            )
        return self
