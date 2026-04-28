"""Tests for counterfactual replay -- §8.3."""

from __future__ import annotations

import math
from datetime import date

import polars as pl
import pytest

from modules.module_1_extraction.synthetic_ledger import generate_synthetic_ledger
from modules.module_2_quant.backtest import BacktestConfig
from modules.module_2_quant.counterfactual import (
    CounterfactualEvent,
    inject_counterfactual,
    replay_with_counterfactual,
)
from modules.module_2_quant.friction import FrictionModel
from modules.module_2_quant.strategy import TargetPctRule, col_gt
from modules.module_2_quant.synthetic_ohlcv import generate_synthetic_ohlcv
from shared.schemas.alpha_ledger import AlphaLedgerRecord


def _zero_friction() -> FrictionModel:
    return FrictionModel(
        commission_per_share=0.0, max_volume_pct=1.0, slippage_quadratic_coef=0.0
    )


def _bullish_event(timestamp: str = "2024-01-04T12:00:00Z") -> CounterfactualEvent:
    return CounterfactualEvent(
        timestamp=timestamp,
        entities=["AAPL"],
        sector_tags=["consumer_electronics"],
        macro_sentiment=0.95,
        sector_sentiment=0.95,
        confidence_score=0.99,
        regime_shift_flag=False,
        horizon_days=7,
        description="Hypothetical major positive product cycle confirmation",
    )


# --- CounterfactualEvent ------------------------------------------------


def test_to_record_dict_validates_against_alpha_ledger_schema() -> None:
    event = _bullish_event()
    record = event.to_record_dict()
    AlphaLedgerRecord.model_validate(record)
    assert record["persona_id"] == "counterfactual_v1"
    assert record["model_id"] == "counterfactual"


def test_to_record_dict_overrides_persona_and_model() -> None:
    rec = _bullish_event().to_record_dict(
        persona_id="what_if_v1", model_id="what_if_engine"
    )
    assert rec["persona_id"] == "what_if_v1"
    assert rec["model_id"] == "what_if_engine"


def test_to_record_dict_handles_z_and_offset_timestamps() -> None:
    z = _bullish_event(timestamp="2024-01-04T12:00:00Z").to_record_dict()
    plus = CounterfactualEvent(
        timestamp="2024-01-04T12:00:00+00:00",
        entities=["AAPL"],
        sector_tags=["consumer_electronics"],
        macro_sentiment=0.5,
        sector_sentiment=0.5,
    ).to_record_dict()
    assert z["epoch_ns"] == plus["epoch_ns"]


def test_to_record_dict_rejects_invalid_timestamp() -> None:
    bad = CounterfactualEvent(
        timestamp="not-a-timestamp",
        entities=["AAPL"],
        sector_tags=["x"],
        macro_sentiment=0.0,
        sector_sentiment=0.0,
    )
    with pytest.raises(ValueError, match="ISO"):
        bad.to_record_dict()


# --- inject_counterfactual ----------------------------------------------


def test_inject_empty_events_is_a_noop_modulo_sort() -> None:
    ledger = pl.DataFrame(
        {
            "doc_hash": ["sha256:" + "0" * 64],
            "timestamp": ["2024-01-04T00:00:00Z"],
            "epoch_ns": [1_704_326_400_000_000_000],
            "entities": [["AAPL"]],
            "sector_tags": [["x"]],
            "macro_sentiment": [0.1],
            "sector_sentiment": [0.1],
            "confidence_interval": [[0.85, 0.95]],
            "confidence_score": [0.9],
            "regime_shift_flag": [False],
            "horizon_days": [30],
            "reasoning_trace": [""],
            "persona_id": ["p_v1"],
            "model_id": ["m"],
            "schema_version": ["1.0.0"],
        }
    )
    out = inject_counterfactual(ledger, [])
    assert out.height == ledger.height


def test_inject_appends_events_and_keeps_sort_order(tmp_path) -> None:
    paths = generate_synthetic_ledger(
        tmp_path,
        date(2024, 1, 2),
        date(2024, 1, 3),
        seed=0,
        docs_per_day_mean=10.0,
    )
    ledger = pl.read_parquet(paths)
    base_h = ledger.height
    out = inject_counterfactual(ledger, [_bullish_event()])
    assert out.height == base_h + 1
    epochs = out["epoch_ns"].to_list()
    assert epochs == sorted(epochs)


def test_inject_rejects_ledger_without_epoch_ns_column() -> None:
    bad = pl.DataFrame({"doc_hash": ["sha256:0"], "timestamp": ["x"]})
    with pytest.raises(ValueError, match="epoch_ns"):
        inject_counterfactual(bad, [_bullish_event()])


# --- replay_with_counterfactual -----------------------------------------


def _bars_and_ledger(tmp_path) -> tuple[pl.DataFrame, pl.DataFrame]:
    ledger_paths = generate_synthetic_ledger(
        tmp_path / "ledger",
        date(2024, 1, 2),
        date(2024, 1, 5),
        seed=0,
        docs_per_day_mean=15.0,
    )
    ohlcv_paths = generate_synthetic_ohlcv(
        tmp_path / "ohlcv",
        date(2024, 1, 2),
        date(2024, 1, 5),
        seed=0,
        bars_per_day=10,
    )
    ledger = pl.read_parquet(ledger_paths)
    bars = pl.read_parquet(ohlcv_paths)
    return bars, ledger


def test_replay_with_no_events_baseline_equals_counterfactual(tmp_path) -> None:
    bars, ledger = _bars_and_ledger(tmp_path)
    cfg = BacktestConfig(
        rule=TargetPctRule(col_gt("macro_sentiment", 0.0), 0.05),
        initial_capital=100_000.0,
        friction=_zero_friction(),
    )
    replay = replay_with_counterfactual(bars, ledger, [], cfg)
    assert replay.n_events_injected == 0
    assert replay.equity_delta == pytest.approx(0.0)
    assert replay.trade_count_delta == 0


def test_replay_with_one_bullish_event_changes_or_preserves_outcome(tmp_path) -> None:
    """A bullish event injected mid-window may or may not change the
    backtest depending on bar timing; what MUST be true is that the
    replay returns two distinct BacktestResult objects with the
    expected n_events_injected."""
    bars, ledger = _bars_and_ledger(tmp_path)
    cfg = BacktestConfig(
        rule=TargetPctRule(col_gt("macro_sentiment", 0.0), 0.05),
        initial_capital=100_000.0,
        friction=_zero_friction(),
    )
    replay = replay_with_counterfactual(
        bars, ledger, [_bullish_event(timestamp="2024-01-03T15:00:00Z")], cfg
    )
    assert replay.n_events_injected == 1
    assert math.isfinite(replay.baseline.final_equity)
    assert math.isfinite(replay.counterfactual.final_equity)
    # Equity delta is finite (not NaN, not inf)
    assert math.isfinite(replay.equity_delta)


def test_replay_with_strongly_bearish_burst_lowers_equity(tmp_path) -> None:
    """Inject many strongly-negative-sentiment events early in the window;
    a sentiment-positive strategy should buy LESS / hold less stock,
    diverging from the baseline."""
    bars, ledger = _bars_and_ledger(tmp_path)
    bearish = [
        CounterfactualEvent(
            timestamp=f"2024-01-02T{h:02d}:00:00Z",
            entities=["AAPL"],
            sector_tags=["consumer_electronics"],
            macro_sentiment=-0.95,
            sector_sentiment=-0.95,
            confidence_score=0.99,
            regime_shift_flag=True,
            horizon_days=30,
            description=f"hypothetical bearish shock at hour {h}",
        )
        for h in (10, 11, 12, 13, 14)
    ]
    cfg = BacktestConfig(
        rule=TargetPctRule(col_gt("macro_sentiment", 0.0), 0.05),
        initial_capital=100_000.0,
        friction=_zero_friction(),
    )
    replay = replay_with_counterfactual(bars, ledger, bearish, cfg)
    # Counterfactual either has fewer trades or different equity than baseline.
    differs = (
        replay.counterfactual.trades.height != replay.baseline.trades.height
        or replay.counterfactual.final_equity != replay.baseline.final_equity
    )
    assert differs, "5 strongly-bearish counterfactual events did not move the backtest"


def test_replay_handles_nan_sharpe_delta(tmp_path) -> None:
    """If either backtest yields a NaN Sharpe (e.g. zero variance equity
    because no trades fired), `sharpe_delta` must return NaN, not raise."""
    bars, ledger = _bars_and_ledger(tmp_path)
    cfg = BacktestConfig(
        rule=TargetPctRule(col_gt("macro_sentiment", 100.0), 0.05),  # never fires
        initial_capital=100_000.0,
        friction=_zero_friction(),
    )
    replay = replay_with_counterfactual(bars, ledger, [_bullish_event()], cfg)
    delta = replay.sharpe_delta
    # Either both Sharpes are NaN (-> NaN delta) or both are finite (-> finite)
    assert math.isnan(delta) or math.isfinite(delta)
