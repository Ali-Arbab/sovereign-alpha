"""Tests for the ZMQ + MessagePack transport layer."""

from __future__ import annotations

import time

import pytest
import zmq

from modules.module_3_twin.messages import PRICES, TRADES, PriceTickMessage, TradeMessage
from modules.module_3_twin.transport import (
    Publisher,
    Subscriber,
    pack_message,
    unpack_message,
)


def _price_tick(ticker: str = "AAPL", epoch_ns: int = 1) -> PriceTickMessage:
    return PriceTickMessage(
        epoch_ns=epoch_ns, ticker=ticker, open=1.0, high=2.0, low=0.5, close=1.5, volume=10
    )


def test_pack_unpack_round_trip_pure_function() -> None:
    msg = _price_tick()
    buf = pack_message(PRICES, msg)
    topic, payload = unpack_message(buf)
    assert topic == PRICES
    rebuilt = PriceTickMessage(**payload)
    assert rebuilt == msg


def test_pack_rejects_empty_topic() -> None:
    with pytest.raises(ValueError, match="topic"):
        pack_message("", _price_tick())


def test_pack_rejects_topic_with_delimiter() -> None:
    with pytest.raises(ValueError, match="must not contain"):
        pack_message("foo|bar", _price_tick())


def test_unpack_rejects_frame_without_delimiter() -> None:
    with pytest.raises(ValueError, match="delimiter"):
        unpack_message(b"no-delim-here")


def test_zmq_inproc_publisher_subscriber_round_trip() -> None:
    ctx = zmq.Context()
    endpoint = "inproc://sovereign-alpha-test"

    pub = Publisher(endpoint, context=ctx)
    sub = Subscriber(endpoint, topics=None, context=ctx)

    # Slow-joiner: even on inproc, the SUB filter handshake takes a moment.
    # Loop a sentinel send until SUB sees one, then send the real messages.
    deadline = time.monotonic() + 1.0
    while time.monotonic() < deadline:
        pub.publish("__handshake__", _price_tick(epoch_ns=0))
        if sub.recv(timeout_ms=20) is not None:
            break
    else:
        pub.close()
        sub.close()
        ctx.term()
        pytest.fail("SUB never received the handshake -- ZMQ inproc misconfigured")

    try:
        pub.publish(PRICES, _price_tick(ticker="AAPL", epoch_ns=10))
        pub.publish(
            TRADES,
            TradeMessage(
                epoch_ns=10,
                ticker="AAPL",
                side="buy",
                qty=1,
                avg_fill_price=1.0,
                slippage_cost=0.0,
                commission=0.0,
            ),
        )

        topics_seen: set[str] = set()
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline and len(topics_seen) < 2:
            recv = sub.recv(timeout_ms=50)
            if recv is None:
                continue
            topic, _ = recv
            topics_seen.add(topic)

        assert PRICES in topics_seen
        assert TRADES in topics_seen
    finally:
        pub.close()
        sub.close()
        ctx.term()


def test_zmq_topic_filter_only_passes_matching() -> None:
    ctx = zmq.Context()
    endpoint = "inproc://sovereign-alpha-filter-test"

    pub = Publisher(endpoint, context=ctx)
    sub = Subscriber(endpoint, topics=[TRADES], context=ctx)

    deadline = time.monotonic() + 1.0
    while time.monotonic() < deadline:
        pub.publish(
            TRADES,
            TradeMessage(
                epoch_ns=0,
                ticker="AAPL",
                side="buy",
                qty=1,
                avg_fill_price=1.0,
                slippage_cost=0.0,
                commission=0.0,
            ),
        )
        if sub.recv(timeout_ms=20) is not None:
            break

    try:
        # Send a filtered-out topic and a passing topic
        pub.publish(PRICES, _price_tick(epoch_ns=1))
        pub.publish(
            TRADES,
            TradeMessage(
                epoch_ns=2,
                ticker="MSFT",
                side="sell",
                qty=1,
                avg_fill_price=2.0,
                slippage_cost=0.0,
                commission=0.0,
            ),
        )

        # We should never see PRICES; should see at least one TRADES
        topics: list[str] = []
        deadline = time.monotonic() + 0.5
        while time.monotonic() < deadline:
            recv = sub.recv(timeout_ms=50)
            if recv is None:
                continue
            topics.append(recv[0])

        assert PRICES not in topics
        assert TRADES in topics
    finally:
        pub.close()
        sub.close()
        ctx.term()
