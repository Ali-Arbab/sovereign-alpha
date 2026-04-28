"""Tests for the multi-agent runner -- §8.1."""

from __future__ import annotations

import math
from datetime import date

import polars as pl
import pytest

from modules.module_1_extraction.synthetic_ledger import generate_synthetic_ledger
from modules.module_2_quant.backtest import BacktestConfig
from modules.module_2_quant.friction import FrictionModel
from modules.module_2_quant.fusion import as_of_fuse, explode_ledger_entities
from modules.module_2_quant.multi_agent import (
    Agent,
    MultiAgentResult,
    run_multi_agent,
)
from modules.module_2_quant.strategy import TargetPctRule, col_gt
from modules.module_2_quant.synthetic_ohlcv import generate_synthetic_ohlcv


def _zero_friction() -> FrictionModel:
    return FrictionModel(
        commission_per_share=0.0, max_volume_pct=1.0, slippage_quadratic_coef=0.0
    )


def _agent_for_persona(
    persona_id: str,
    *,
    ledger_seed: int,
    bars: pl.DataFrame,
    tmp_path,
    sentiment_threshold: float = 0.0,
) -> Agent:
    """Build an Agent by generating a per-persona synthetic ledger and fusing."""
    paths = generate_synthetic_ledger(
        tmp_path / persona_id,
        date(2024, 1, 2),
        date(2024, 1, 8),
        seed=ledger_seed,
        docs_per_day_mean=15.0,
    )
    ledger = explode_ledger_entities(pl.read_parquet(paths)).sort("epoch_ns")
    fused = as_of_fuse(bars, ledger, by_left="ticker", by_right="entity")
    cfg = BacktestConfig(
        rule=TargetPctRule(col_gt("macro_sentiment", sentiment_threshold), 0.05),
        initial_capital=100_000.0,
        friction=_zero_friction(),
    )
    return Agent(persona_id=persona_id, fused=fused, config=cfg)


def _bars(tmp_path) -> pl.DataFrame:
    paths = generate_synthetic_ohlcv(
        tmp_path / "ohlcv",
        date(2024, 1, 2),
        date(2024, 1, 8),
        seed=0,
        bars_per_day=15,
    )
    return pl.read_parquet(paths).sort("epoch_ns")


def test_run_multi_agent_returns_per_persona_results(tmp_path) -> None:
    bars = _bars(tmp_path)
    agents = [
        _agent_for_persona("alpha_v1", ledger_seed=1, bars=bars, tmp_path=tmp_path),
        _agent_for_persona("beta_v1", ledger_seed=2, bars=bars, tmp_path=tmp_path),
        _agent_for_persona("gamma_v1", ledger_seed=3, bars=bars, tmp_path=tmp_path),
    ]
    result = run_multi_agent(agents)
    assert isinstance(result, MultiAgentResult)
    assert result.n_agents == 3
    assert set(result.agents.keys()) == {"alpha_v1", "beta_v1", "gamma_v1"}
    for pid, r in result.agents.items():
        assert math.isfinite(r.final_equity), f"{pid} produced non-finite equity"


def test_total_final_equity_sums_per_agent_equity(tmp_path) -> None:
    bars = _bars(tmp_path)
    agents = [
        _agent_for_persona("alpha_v1", ledger_seed=1, bars=bars, tmp_path=tmp_path),
        _agent_for_persona("beta_v1", ledger_seed=2, bars=bars, tmp_path=tmp_path),
    ]
    result = run_multi_agent(agents)
    expected = sum(r.final_equity for r in result.agents.values())
    assert result.total_final_equity == pytest.approx(expected)


def test_total_trades_aggregates_correctly(tmp_path) -> None:
    bars = _bars(tmp_path)
    agents = [
        _agent_for_persona("alpha_v1", ledger_seed=1, bars=bars, tmp_path=tmp_path),
        _agent_for_persona("beta_v1", ledger_seed=2, bars=bars, tmp_path=tmp_path),
    ]
    result = run_multi_agent(agents)
    expected = sum(r.trades.height for r in result.agents.values())
    assert result.total_trades == expected


def test_survivor_personas_are_those_above_initial_capital(tmp_path) -> None:
    bars = _bars(tmp_path)
    agents = [
        _agent_for_persona("alpha_v1", ledger_seed=1, bars=bars, tmp_path=tmp_path),
        _agent_for_persona("beta_v1", ledger_seed=2, bars=bars, tmp_path=tmp_path),
    ]
    result = run_multi_agent(agents)
    # Property: survivors are EXACTLY those with final > initial
    expected = {
        pid for pid, r in result.agents.items() if r.final_equity > r.initial_capital
    }
    assert set(result.survivor_personas) == expected


def test_ranked_by_final_equity_is_descending(tmp_path) -> None:
    bars = _bars(tmp_path)
    agents = [
        _agent_for_persona(f"agent_{i}_v1", ledger_seed=i, bars=bars, tmp_path=tmp_path)
        for i in range(1, 5)
    ]
    result = run_multi_agent(agents)
    ranked = result.ranked_by("final_equity")
    values = [v for _, v in ranked]
    assert values == sorted(values, reverse=True)


def test_ranked_by_invalid_metric_raises(tmp_path) -> None:
    bars = _bars(tmp_path)
    agents = [_agent_for_persona("alpha_v1", ledger_seed=1, bars=bars, tmp_path=tmp_path)]
    result = run_multi_agent(agents)
    with pytest.raises(ValueError, match="metric must be"):
        result.ranked_by("not_a_metric")


def test_run_multi_agent_rejects_empty_agents() -> None:
    with pytest.raises(ValueError, match="agents must be non-empty"):
        run_multi_agent([])


def test_run_multi_agent_rejects_duplicate_persona_ids(tmp_path) -> None:
    bars = _bars(tmp_path)
    agents = [
        _agent_for_persona("alpha_v1", ledger_seed=1, bars=bars, tmp_path=tmp_path),
        _agent_for_persona("alpha_v1", ledger_seed=2, bars=bars, tmp_path=tmp_path),
    ]
    with pytest.raises(ValueError, match="unique"):
        run_multi_agent(agents)


def test_distinct_personas_produce_distinct_results(tmp_path) -> None:
    """Different ledger seeds must produce different per-persona equity --
    otherwise the multi-agent harness collapses to a single agent."""
    bars = _bars(tmp_path)
    agents = [
        _agent_for_persona("alpha_v1", ledger_seed=1, bars=bars, tmp_path=tmp_path),
        _agent_for_persona("beta_v1", ledger_seed=999, bars=bars, tmp_path=tmp_path),
    ]
    result = run_multi_agent(agents)
    a = result.agents["alpha_v1"]
    b = result.agents["beta_v1"]
    differs = (
        a.final_equity != b.final_equity
        or a.trades.height != b.trades.height
    )
    assert differs, "multi-agent: distinct personas yielded identical results"
