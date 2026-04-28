"""Alpha Ledger record schema — Module I → Module II contract.

Mechanically transcribed from master directive §3.4. Granularity is one row
per (document, entity) pair; sector-level aggregations are computed downstream
in Polars, not via a second LLM pass. Schema drift is a build-breaking defect.
"""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

SCHEMA_VERSION = "1.0.0"


class AlphaLedgerRecord(BaseModel):
    """Single Alpha Ledger record — one (document, entity) inference output."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    doc_hash: Annotated[str, Field(pattern=r"^sha256:[0-9a-f]{64}$")]
    timestamp: str
    epoch_ns: Annotated[int, Field(ge=0)]
    entities: list[str]
    sector_tags: list[str]
    macro_sentiment: Annotated[float, Field(ge=-1.0, le=1.0)]
    sector_sentiment: Annotated[float, Field(ge=-1.0, le=1.0)]
    confidence_interval: tuple[float, float]
    confidence_score: Annotated[float, Field(ge=0.0, le=1.0)]
    regime_shift_flag: bool
    horizon_days: Annotated[int, Field(gt=0)]
    reasoning_trace: str
    persona_id: str
    model_id: str
    schema_version: Annotated[str, Field(pattern=r"^\d+\.\d+\.\d+$")] = SCHEMA_VERSION
