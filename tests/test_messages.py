"""Tests for Module III message schemas + msgpack round-trip."""

from __future__ import annotations

import msgpack
import pytest
from pydantic import ValidationError

from modules.module_3_twin.messages import (
    ALL_TOPICS,
    SCHEMA_VERSION,
    PortfolioStateMessage,
    PriceTickMessage,
    RegimeEventMessage,
    SentimentMessage,
    TradeMessage,
)


def _round_trip(msg) -> dict:
    payload = msgpack.packb(msg.model_dump(), use_bin_type=True)
    return msgpack.unpackb(payload, raw=False)


def test_all_topic_constants_present() -> None:
    assert set(ALL_TOPICS) == {"prices", "trades", "sentiment", "portfolio_state", "regime_events"}


def test_price_tick_round_trip() -> None:
    msg = PriceTickMessage(
        epoch_ns=1_700_000_000_000_000_000,
        ticker="AAPL",
        open=150.0,
        high=152.5,
        low=149.0,
        close=151.0,
        volume=1_500_000,
    )
    rt = _round_trip(msg)
    PriceTickMessage(**rt)
    assert rt["ticker"] == "AAPL"
    assert rt["schema_version"] == SCHEMA_VERSION


def test_trade_round_trip() -> None:
    msg = TradeMessage(
        epoch_ns=1,
        ticker="MSFT",
        side="buy",
        qty=100,
        avg_fill_price=300.0,
        slippage_cost=0.5,
        commission=0.5,
        doc_hash="sha256:abc",
    )
    rt = _round_trip(msg)
    TradeMessage(**rt)
    assert rt["doc_hash"] == "sha256:abc"


def test_sentiment_round_trip() -> None:
    msg = SentimentMessage(
        epoch_ns=1,
        entity="NVDA",
        sector="semiconductors",
        macro_sentiment=-0.3,
        sector_sentiment=-0.4,
        confidence_score=0.85,
        persona_id="hawkish_fed_v1",
    )
    rt = _round_trip(msg)
    SentimentMessage(**rt)


def test_portfolio_state_round_trip() -> None:
    msg = PortfolioStateMessage(
        epoch_ns=1,
        cash=50_000.0,
        equity=110_000.0,
        positions={"AAPL": 100, "MSFT": 50},
    )
    rt = _round_trip(msg)
    rebuilt = PortfolioStateMessage(**rt)
    assert rebuilt.positions == {"AAPL": 100, "MSFT": 50}


def test_regime_event_round_trip() -> None:
    msg = RegimeEventMessage(epoch_ns=1, flag=True, description="hawkish pivot")
    rt = _round_trip(msg)
    RegimeEventMessage(**rt)


def test_sentiment_bounds_enforced() -> None:
    with pytest.raises(ValidationError):
        SentimentMessage(
            epoch_ns=1,
            entity="AAPL",
            sector="x",
            macro_sentiment=1.5,
            sector_sentiment=0.0,
            confidence_score=0.5,
            persona_id="p",
        )


def test_trade_qty_must_be_positive() -> None:
    with pytest.raises(ValidationError):
        TradeMessage(
            epoch_ns=1,
            ticker="AAPL",
            side="buy",
            qty=0,
            avg_fill_price=100.0,
            slippage_cost=0.0,
            commission=0.0,
        )


def test_extra_fields_rejected() -> None:
    with pytest.raises(ValidationError):
        PriceTickMessage(
            epoch_ns=1,
            ticker="AAPL",
            open=1.0,
            high=2.0,
            low=0.5,
            close=1.5,
            volume=10,
            unsanctioned="leak",
        )
