"""Multi-agent simulation -- N distinct traders on the same historical timeline.

Per master directive section 8.1: "The architecture supports N distinct
virtual traders operating on the same historical timeline simultaneously,
each driven by a different persona-derived Alpha Ledger." This is the
infrastructure layer; aggregation across agents (which persona made the
most money in 2020-Q1?) is the natural research question downstream.

Inter-agent market impact -- where the aggregate order flow of all agents
reduces effective bar volume for everyone -- is sketched but not yet
implemented; it requires a two-pass runner (project orders, adjust
volumes, re-run with adjusted friction). v0 runs each agent in isolation
on the same bars, so per-agent metrics are independent. The API is
shaped so the impact step can drop in without breaking callers.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import polars as pl

from modules.module_2_quant.backtest import BacktestConfig, BacktestResult, run_backtest


@dataclass(frozen=True)
class Agent:
    """One simulated trader: a persona id + a fully-fused frame + a backtest config.

    `fused` MUST be the result of `as_of_fuse(bars, this_agent's_ledger, ...)`;
    different agents will typically share `bars` but consume per-persona
    ledgers, so each agent's `fused` is distinct.
    """

    persona_id: str
    fused: pl.DataFrame
    config: BacktestConfig


@dataclass(frozen=True)
class MultiAgentResult:
    """Per-persona BacktestResults plus aggregate views."""

    agents: Mapping[str, BacktestResult]

    @property
    def n_agents(self) -> int:
        return len(self.agents)

    @property
    def total_final_equity(self) -> float:
        return float(sum(r.final_equity for r in self.agents.values()))

    @property
    def total_trades(self) -> int:
        return int(sum(r.trades.height for r in self.agents.values()))

    @property
    def survivor_personas(self) -> list[str]:
        """Personas whose final equity exceeded their starting capital."""
        return sorted(
            pid
            for pid, r in self.agents.items()
            if r.final_equity > r.initial_capital
        )

    def ranked_by(self, metric: str = "final_equity") -> list[tuple[str, float]]:
        """Return [(persona_id, value)] sorted descending by `metric` on each
        agent's BacktestResult.
        """
        if metric == "final_equity":
            extract = lambda r: float(r.final_equity)  # noqa: E731
        elif metric == "sharpe":
            extract = lambda r: float(r.sharpe)  # noqa: E731
        elif metric == "sortino":
            extract = lambda r: float(r.sortino)  # noqa: E731
        elif metric == "max_drawdown":
            extract = lambda r: float(r.drawdown.max_drawdown)  # noqa: E731
        elif metric == "n_trades":
            extract = lambda r: float(r.trades.height)  # noqa: E731
        else:
            raise ValueError(
                "metric must be one of: final_equity, sharpe, sortino, "
                "max_drawdown, n_trades"
            )
        scored = [(pid, extract(r)) for pid, r in self.agents.items()]
        # NaN values sort to the bottom
        return sorted(scored, key=lambda t: (1 if t[1] != t[1] else 0, -t[1]))


def run_multi_agent(agents: list[Agent]) -> MultiAgentResult:
    """Run every agent's backtest in lockstep against its fused frame.

    Each agent's results are independent in v0 -- no cross-agent market
    impact yet. To add impact: project order flow per agent, aggregate
    per bar, reduce effective volume, re-run friction. Out of scope for
    this commit.
    """
    if not agents:
        raise ValueError("agents must be non-empty")
    persona_ids = [a.persona_id for a in agents]
    if len(set(persona_ids)) != len(persona_ids):
        raise ValueError(f"persona_ids must be unique, got {persona_ids}")

    results: dict[str, BacktestResult] = {}
    for agent in agents:
        results[agent.persona_id] = run_backtest(agent.fused, agent.config)
    return MultiAgentResult(agents=results)
