"""Schema validation for AlphaLedgerRecord — directive §3.4 example must validate."""

import pytest
from pydantic import ValidationError

from shared.schemas.alpha_ledger import SCHEMA_VERSION, AlphaLedgerRecord


def _example() -> dict:
    return {
        "doc_hash": "sha256:" + "0" * 64,
        "timestamp": "2018-04-12T09:30:00Z",
        "epoch_ns": 1523525400000000000,
        "entities": ["AAPL", "TSM"],
        "sector_tags": ["semiconductors", "consumer_electronics"],
        "macro_sentiment": -0.65,
        "sector_sentiment": -0.80,
        "confidence_interval": (0.85, 0.95),
        "confidence_score": 0.92,
        "regime_shift_flag": True,
        "horizon_days": 90,
        "reasoning_trace": "Tariff implementation disrupts primary fabrication nodes.",
        "persona_id": "supply_chain_analyst_v3",
        "model_id": "deepseek-r1-32b-q6_k",
    }


def test_directive_example_validates() -> None:
    record = AlphaLedgerRecord(**_example())
    assert record.schema_version == SCHEMA_VERSION
    assert record.entities == ["AAPL", "TSM"]
    assert record.regime_shift_flag is True


def test_doc_hash_pattern_is_enforced() -> None:
    bad = _example() | {"doc_hash": "not-a-sha256"}
    with pytest.raises(ValidationError):
        AlphaLedgerRecord(**bad)


def test_sentiment_bounds_are_enforced() -> None:
    bad = _example() | {"macro_sentiment": 1.5}
    with pytest.raises(ValidationError):
        AlphaLedgerRecord(**bad)


def test_extra_fields_are_rejected() -> None:
    bad = _example() | {"unsanctioned_field": "leak"}
    with pytest.raises(ValidationError):
        AlphaLedgerRecord(**bad)
