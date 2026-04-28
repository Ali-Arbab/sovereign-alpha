"""Counterfactual replay -- inject hypothetical events into the Alpha Ledger.

Per master directive section 8.3: "Inject hypothetical news events into the
corpus (e.g., 'What if SVB had been bailed out 48 hours earlier?') and
replay the backtest to quantify event sensitivity."

Workflow:
1. Take an existing list-typed Alpha Ledger frame (synthetic or real).
2. Author one or more `CounterfactualEvent`s -- typed wrappers that
   produce schema-conformant AlphaLedgerRecord rows.
3. Inject them into the ledger via `inject_counterfactual`.
4. Run two backtests (baseline + counterfactual) via
   `replay_with_counterfactual`, get a `CounterfactualReplay` holding
   both results plus equity / Sharpe deltas.

Injected rows are tagged `persona_id="counterfactual_v1"` and
`model_id="counterfactual"` by default so they are structurally
distinguishable from research output.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import UTC, datetime

import polars as pl
from pydantic import BaseModel, ConfigDict, Field

from modules.module_2_quant.backtest import BacktestConfig, BacktestResult, run_backtest
from modules.module_2_quant.fusion import as_of_fuse, explode_ledger_entities
from shared.schemas.alpha_ledger import SCHEMA_VERSION


class CounterfactualEvent(BaseModel):
    """One hypothetical event injected into the Alpha Ledger.

    Defaults are chosen so the only required arguments are timestamp,
    entities, sector_tags, and the two sentiment scalars; the rest get
    sensible counterfactual-typical values (high confidence, no regime
    shift, 30-day horizon).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    timestamp: str
    entities: list[str]
    sector_tags: list[str]
    macro_sentiment: float = Field(ge=-1.0, le=1.0)
    sector_sentiment: float = Field(ge=-1.0, le=1.0)
    confidence_score: float = Field(default=0.95, ge=0.0, le=1.0)
    regime_shift_flag: bool = False
    horizon_days: int = Field(default=30, gt=0)
    description: str = "Counterfactual event"

    def to_record_dict(
        self,
        *,
        persona_id: str = "counterfactual_v1",
        model_id: str = "counterfactual",
    ) -> dict:
        """Produce an AlphaLedgerRecord-shaped dict ready for ledger concat."""
        ts = self.timestamp
        # Parse ISO -- accept either Z suffix or +00:00
        normalized = ts.replace("Z", "+00:00") if ts.endswith("Z") else ts
        try:
            dt = datetime.fromisoformat(normalized)
        except ValueError as e:
            raise ValueError(f"invalid ISO timestamp: {ts!r}") from e
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        epoch = dt - datetime(1970, 1, 1, tzinfo=UTC)
        epoch_ns = (
            (epoch.days * 86_400 + epoch.seconds) * 1_000_000_000
            + epoch.microseconds * 1_000
        )
        doc_hash = (
            "sha256:"
            + hashlib.sha256(
                f"counterfactual|{self.timestamp}|{self.description}".encode()
            ).hexdigest()
        )
        ci_low = max(0.0, self.confidence_score - 0.05)
        ci_high = min(1.0, self.confidence_score + 0.05)
        return {
            "doc_hash": doc_hash,
            "timestamp": self.timestamp,
            "epoch_ns": epoch_ns,
            "entities": list(self.entities),
            "sector_tags": list(self.sector_tags),
            "macro_sentiment": float(self.macro_sentiment),
            "sector_sentiment": float(self.sector_sentiment),
            "confidence_interval": [ci_low, ci_high],
            "confidence_score": float(self.confidence_score),
            "regime_shift_flag": bool(self.regime_shift_flag),
            "horizon_days": int(self.horizon_days),
            "reasoning_trace": f"[counterfactual] {self.description}",
            "persona_id": persona_id,
            "model_id": model_id,
            "schema_version": SCHEMA_VERSION,
        }


def inject_counterfactual(
    ledger: pl.DataFrame,
    events: list[CounterfactualEvent],
    *,
    persona_id: str = "counterfactual_v1",
    model_id: str = "counterfactual",
) -> pl.DataFrame:
    """Concatenate counterfactual rows onto a ledger frame and re-sort.

    `ledger` MUST be in the list-typed form (entities/sector_tags as
    list columns). Output is sorted ascending on epoch_ns so the
    downstream as_of_fuse precondition is satisfied.
    """
    if not events:
        return ledger.sort("epoch_ns") if ledger.height else ledger
    if "epoch_ns" not in ledger.columns:
        raise ValueError("ledger must have an 'epoch_ns' column")

    rows = [
        e.to_record_dict(persona_id=persona_id, model_id=model_id) for e in events
    ]
    injected = pl.DataFrame(rows)
    combined = pl.concat(
        [ledger.select(injected.columns), injected], how="vertical_relaxed"
    )
    return combined.sort("epoch_ns")


@dataclass(frozen=True)
class CounterfactualReplay:
    """Two backtests run on identical bars: baseline and counterfactual."""

    baseline: BacktestResult
    counterfactual: BacktestResult
    n_events_injected: int

    @property
    def equity_delta(self) -> float:
        return self.counterfactual.final_equity - self.baseline.final_equity

    @property
    def sharpe_delta(self) -> float:
        import math

        a = self.baseline.sharpe
        b = self.counterfactual.sharpe
        if math.isnan(a) or math.isnan(b):
            return float("nan")
        return b - a

    @property
    def trade_count_delta(self) -> int:
        return self.counterfactual.trades.height - self.baseline.trades.height


def replay_with_counterfactual(
    bars: pl.DataFrame,
    ledger: pl.DataFrame,
    events: list[CounterfactualEvent],
    config: BacktestConfig,
) -> CounterfactualReplay:
    """Run the same strategy twice -- once on the unmodified ledger, once
    with `events` injected -- and return both results.

    `ledger` is the list-typed form; this function performs the explode +
    sort + as_of_fuse on each path, so the caller works with raw ledger
    output rather than pre-fused frames.
    """
    baseline_ledger = explode_ledger_entities(ledger).sort("epoch_ns")
    baseline_fused = as_of_fuse(
        bars.sort("epoch_ns"), baseline_ledger, by_left="ticker", by_right="entity"
    )
    baseline_result = run_backtest(baseline_fused, config)

    cf_ledger_listed = inject_counterfactual(ledger, events)
    cf_ledger = explode_ledger_entities(cf_ledger_listed).sort("epoch_ns")
    cf_fused = as_of_fuse(
        bars.sort("epoch_ns"), cf_ledger, by_left="ticker", by_right="entity"
    )
    cf_result = run_backtest(cf_fused, config)

    return CounterfactualReplay(
        baseline=baseline_result,
        counterfactual=cf_result,
        n_events_injected=len(events),
    )
