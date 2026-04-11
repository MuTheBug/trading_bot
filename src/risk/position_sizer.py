"""Fixed-fractional position sizing with exchange-filter compliance."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from ..exchange.base import SymbolFilters


@dataclass
class SizingResult:
    qty: float
    notional: float
    feasible: bool
    reason: str = ""


def _round_step(value: float, step: float) -> float:
    """Floor `value` to the nearest multiple of `step`.

    A small epsilon (1e-9 × step) is added before flooring so that values
    which should land exactly on a step boundary but have tiny float drift
    (e.g. 49.99999999999987 instead of 50.0) round up to the correct multiple.
    """
    if step <= 0:
        return value
    return math.floor(value / step + 1e-9) * step


def compute_position_size(
    equity: float,
    risk_pct: float,
    entry_price: float,
    stop_price: float,
    filters: SymbolFilters,
    leverage: int = 1,
) -> SizingResult:
    """Fixed-fractional sizing.

    qty = (equity * risk_pct/100) / |entry - stop|

    Then rounded down to LOT_SIZE step and validated against:
      - min_qty
      - min_notional
      - leverage-bounded margin (qty * entry / leverage <= equity)
    """
    if entry_price <= 0 or stop_price <= 0:
        return SizingResult(0.0, 0.0, False, "invalid prices")
    stop_distance = abs(entry_price - stop_price)
    if stop_distance == 0:
        return SizingResult(0.0, 0.0, False, "zero stop distance")

    risk_amount = equity * (risk_pct / 100.0)
    raw_qty = risk_amount / stop_distance

    qty = _round_step(raw_qty, filters.qty_step)
    if qty < filters.min_qty:
        return SizingResult(0.0, 0.0, False, f"below min_qty {filters.min_qty}")

    notional = qty * entry_price
    if notional < filters.min_notional:
        return SizingResult(
            qty, notional, False,
            f"notional {notional:.4f} < min_notional {filters.min_notional}"
        )

    # Margin check: notional / leverage must fit in available equity.
    required_margin = notional / max(leverage, 1)
    if required_margin > equity:
        # Cap qty so margin fits. This happens when risk-based qty is too large
        # for the account (unlikely with 1% risk and 3x lev but guard anyway).
        capped_qty = _round_step(equity * leverage / entry_price, filters.qty_step)
        if capped_qty < filters.min_qty:
            return SizingResult(0.0, 0.0, False, "insufficient margin even capped")
        capped_notional = capped_qty * entry_price
        if capped_notional < filters.min_notional:
            return SizingResult(
                capped_qty, capped_notional, False,
                "margin-capped notional below min_notional"
            )
        return SizingResult(capped_qty, capped_notional, True, "margin-capped")

    return SizingResult(qty, notional, True)


__all__ = ["compute_position_size", "SizingResult"]
