"""Module II friction layer -- slippage, commission, partial fills, borrow costs.

Per master directive section 4.4. Friction is layered as separate functions
applied to fill requests, never inlined into strategy signal logic. A reported
PnL number is research-grade only when every fill has been routed through here.

Models in scope:
- **Slippage**: volume-weighted heuristic. Larger participation in a bar's
  volume incurs quadratic-in-participation slippage on the fill price.
- **Commission**: per-share, configurable per broker.
- **Partial fills**: an order capped at `max_volume_pct` of the bar's volume.
  Excess quantity is reported as `rejected_qty` and must be re-queued upstream.
- **Borrow costs**: stub model -- fixed annual rate against the short notional.
  Real model awaits historical borrow-rate curves (directive §0.5.1.D blocked).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Final

EPS: Final[float] = 1e-12


class Side(StrEnum):
    BUY = "buy"
    SELL = "sell"


@dataclass(frozen=True)
class FillResult:
    """Outcome of a single fill request after friction has been applied."""

    side: Side
    filled_qty: int
    rejected_qty: int
    avg_fill_price: float
    slippage_cost: float
    commission: float

    @property
    def total_cost(self) -> float:
        return self.slippage_cost + self.commission

    @property
    def total_notional(self) -> float:
        return self.filled_qty * self.avg_fill_price


@dataclass(frozen=True)
class FrictionModel:
    """Friction parameters applied to every fill in the backtest."""

    commission_per_share: float = 0.005
    max_volume_pct: float = 0.05
    slippage_quadratic_coef: float = 0.5
    annual_borrow_rate: float = 0.02

    def __post_init__(self) -> None:
        if self.commission_per_share < 0:
            raise ValueError("commission_per_share must be non-negative")
        if not (0.0 < self.max_volume_pct <= 1.0):
            raise ValueError("max_volume_pct must be in (0, 1]")
        if self.slippage_quadratic_coef < 0:
            raise ValueError("slippage_quadratic_coef must be non-negative")
        if self.annual_borrow_rate < 0:
            raise ValueError("annual_borrow_rate must be non-negative")

    def fill(
        self,
        *,
        side: Side,
        qty_requested: int,
        bar_volume: int,
        bar_price: float,
    ) -> FillResult:
        """Apply friction to a fill request against a single OHLCV bar.

        Quantity above `max_volume_pct * bar_volume` is rejected (partial fill).
        Slippage is `slippage_quadratic_coef * participation^2 * bar_price`
        per filled share; sign convention pushes BUY fills above the bar price
        and SELL fills below (slippage cost is always >= 0).
        """
        if qty_requested < 0:
            raise ValueError("qty_requested must be non-negative")
        if bar_volume < 0:
            raise ValueError("bar_volume must be non-negative")
        if bar_price <= 0:
            raise ValueError("bar_price must be positive")

        if qty_requested == 0 or bar_volume == 0:
            return FillResult(
                side=side,
                filled_qty=0,
                rejected_qty=qty_requested,
                avg_fill_price=bar_price,
                slippage_cost=0.0,
                commission=0.0,
            )

        max_fillable = int(self.max_volume_pct * bar_volume)
        filled_qty = min(qty_requested, max_fillable)
        rejected_qty = qty_requested - filled_qty

        if filled_qty == 0:
            return FillResult(
                side=side,
                filled_qty=0,
                rejected_qty=rejected_qty,
                avg_fill_price=bar_price,
                slippage_cost=0.0,
                commission=0.0,
            )

        participation = filled_qty / max(bar_volume, EPS)
        slippage_per_share = self.slippage_quadratic_coef * participation**2 * bar_price
        sign = 1.0 if side is Side.BUY else -1.0
        avg_fill_price = bar_price + sign * slippage_per_share
        slippage_cost = slippage_per_share * filled_qty
        commission = self.commission_per_share * filled_qty

        return FillResult(
            side=side,
            filled_qty=filled_qty,
            rejected_qty=rejected_qty,
            avg_fill_price=avg_fill_price,
            slippage_cost=slippage_cost,
            commission=commission,
        )

    def borrow_cost(self, *, qty_short: int, price: float, days: int) -> float:
        """Stub borrow cost: notional * annual_rate * (days / 365).

        Returns zero for non-positive short qty. Real borrow-rate curves arrive
        with corpus acquisition (directive §0.5.1.D); this is the API hook.
        """
        if qty_short <= 0 or days <= 0:
            return 0.0
        if price <= 0:
            raise ValueError("price must be positive")
        notional = qty_short * price
        return notional * self.annual_borrow_rate * (days / 365.0)
