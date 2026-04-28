"""Module III message schemas -- ZMQ PUB/SUB topics over MessagePack.

Per master directive section 5.1. Topic taxonomy:
- "prices"           -> PriceTickMessage      (one per OHLCV bar)
- "trades"           -> TradeMessage          (one per fill)
- "sentiment"        -> SentimentMessage      (one per ledger row applied)
- "portfolio_state"  -> PortfolioStateMessage (one per simulation timestep)
- "regime_events"    -> RegimeEventMessage    (one on regime_shift_flag transition)

Every message is Pydantic-validated on the Python side and serialized via
MessagePack. The UE5 client deserializes into fixed C++ structs; schema drift
is a build failure (directive section 6.5).
"""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

SCHEMA_VERSION = "1.0.0"

PRICES = "prices"
TRADES = "trades"
SENTIMENT = "sentiment"
PORTFOLIO_STATE = "portfolio_state"
REGIME_EVENTS = "regime_events"

ALL_TOPICS: tuple[str, ...] = (
    PRICES,
    TRADES,
    SENTIMENT,
    PORTFOLIO_STATE,
    REGIME_EVENTS,
)


class _BridgeMessage(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Annotated[str, Field(pattern=r"^\d+\.\d+\.\d+$")] = SCHEMA_VERSION
    epoch_ns: Annotated[int, Field(ge=0)]


class PriceTickMessage(_BridgeMessage):
    ticker: str
    open: Annotated[float, Field(gt=0.0)]
    high: Annotated[float, Field(gt=0.0)]
    low: Annotated[float, Field(gt=0.0)]
    close: Annotated[float, Field(gt=0.0)]
    volume: Annotated[int, Field(ge=0)]


class TradeMessage(_BridgeMessage):
    ticker: str
    side: str  # "buy" or "sell"
    qty: Annotated[int, Field(gt=0)]
    avg_fill_price: Annotated[float, Field(gt=0.0)]
    slippage_cost: Annotated[float, Field(ge=0.0)]
    commission: Annotated[float, Field(ge=0.0)]
    doc_hash: str = ""


class SentimentMessage(_BridgeMessage):
    entity: str
    sector: str
    macro_sentiment: Annotated[float, Field(ge=-1.0, le=1.0)]
    sector_sentiment: Annotated[float, Field(ge=-1.0, le=1.0)]
    confidence_score: Annotated[float, Field(ge=0.0, le=1.0)]
    persona_id: str


class PortfolioStateMessage(_BridgeMessage):
    cash: float
    equity: float
    positions: dict[str, int] = Field(default_factory=dict)


class RegimeEventMessage(_BridgeMessage):
    flag: bool
    description: str = ""
