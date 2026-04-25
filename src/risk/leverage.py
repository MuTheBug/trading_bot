"""Adaptive leverage & position sizing for small-capital directional trading.

Design goals
------------
- Risk a fixed % of equity per trade regardless of leverage used.
- For small accounts (< ~$50) leverage must be raised high enough that the
  required position meets the exchange's min-notional filter; otherwise the
  trade would simply be rejected. We scale leverage up to that minimum and
  then cap at the configured max.
- Scale leverage DOWN in high-volatility regimes and UP in low-vol,
  high-confidence regimes so $-risk stays roughly constant.
- All sizing is margin-aware: required margin must fit in available equity.

A single entry point ``compute_trade_plan`` returns a feasible (qty, leverage,
margin) tuple or an infeasibility reason.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from ..exchange.base import SymbolFilters


@dataclass
class TradePlan:
    qty: float
    leverage: int
    notional: float
    margin: float
    risk_dollars: float        # expected $ loss if SL hits
    feasible: bool
    reason: str = ""


def _round_step(value: float, step: float) -> float:
    if step <= 0:
        return value
    return math.floor(value / step + 1e-9) * step


def adaptive_leverage(
    confidence: float,
    atr_pct: float,
    base_leverage: int,
    max_leverage: int,
    min_leverage: int = 1,
) -> int:
    """Pick a leverage based on regime confidence and realised volatility.

    Higher confidence + lower vol -> more leverage (up to cap).
    Lower confidence + higher vol -> less leverage.
    """
    conf = max(0.0, min(1.0, confidence))
    # Normalise ATR% to a 0..1 "vol factor": 0.3% -> 0, 3% -> 1.
    vol_factor = max(0.0, min(1.0, (atr_pct - 0.3) / 2.7))
    span = max(max_leverage - min_leverage, 0)
    # Multiplier: 1.0 at perfect conditions, ~0.25 at worst.
    mult = 0.25 + 0.75 * conf * (1.0 - vol_factor)
    lev = min_leverage + int(round(span * mult))
    # Never below base unless base > cap.
    lev = max(lev, min(base_leverage, max_leverage))
    return max(min(lev, max_leverage), min_leverage)


def compute_trade_plan(
    equity: float,
    entry_price: float,
    stop_price: float,
    filters: SymbolFilters,
    risk_pct: float,
    confidence: float,
    atr_pct: float,
    base_leverage: int,
    max_leverage: int,
    max_margin_pct: float = 90.0,
    min_leverage: int = 1,
    hard_margin_pct: float = 98.0,
) -> TradePlan:
    """Compute qty + leverage so $-risk == equity * risk_pct / 100.

    Steps:
    1. Pick leverage from confidence/vol.
    2. qty = risk$ / |entry - stop|, floored to LOT_SIZE.
    3. If notional < min_notional OR qty < min_qty, bump leverage (up to cap)
       so the smallest allowed qty still fits within max_margin_pct of equity.
    4. Expensive-asset fallback: if the exchange's min_notional forces a
       margin above ``max_margin_pct`` even at max leverage, accept it up
       to ``hard_margin_pct`` of equity (leaves a small buffer for fees
       and slippage). Reject only if we'd blow through that hard cap.
    5. Re-check all filters; return infeasible if anything fails.
    """
    if equity <= 0 or entry_price <= 0 or stop_price <= 0:
        return TradePlan(0, 1, 0, 0, 0, False, "invalid inputs")
    stop_distance = abs(entry_price - stop_price)
    if stop_distance <= 0:
        return TradePlan(0, 1, 0, 0, 0, False, "zero stop distance")

    leverage = adaptive_leverage(
        confidence, atr_pct, base_leverage, max_leverage, min_leverage
    )

    risk_dollars = equity * (risk_pct / 100.0)
    raw_qty = risk_dollars / stop_distance
    qty = _round_step(raw_qty, filters.qty_step)

    min_qty_from_filter = max(filters.min_qty, filters.min_notional / entry_price)
    # Round min required qty UP to lot step.
    if filters.qty_step > 0:
        min_qty_from_filter = (
            math.ceil(min_qty_from_filter / filters.qty_step - 1e-9)
            * filters.qty_step
        )

    # If risk-based qty is below exchange minimum, bump qty up to the minimum
    # and compensate by raising leverage (so margin still fits). Risk will
    # exceed the target — caller can inspect risk_dollars and accept or reject.
    bumped = False
    if qty < min_qty_from_filter:
        qty = min_qty_from_filter
        bumped = True

    notional = qty * entry_price
    soft_cap = equity * (max_margin_pct / 100.0)
    hard_cap = equity * (max(hard_margin_pct, max_margin_pct) / 100.0)

    # Raise leverage so margin fits under the soft cap when possible.
    needed_lev_soft = math.ceil(notional / max(soft_cap, 1e-9))
    if needed_lev_soft > leverage:
        leverage = min(max_leverage, int(needed_lev_soft))

    # Still too expensive at soft cap? Try harder — push leverage up so
    # margin fits under the hard cap. This is the "expensive asset" path.
    margin = notional / max(leverage, 1)
    elevated_margin = False
    if margin > soft_cap:
        needed_lev_hard = math.ceil(notional / max(hard_cap, 1e-9))
        if needed_lev_hard > leverage:
            leverage = min(max_leverage, int(needed_lev_hard))
            margin = notional / max(leverage, 1)
        if margin > soft_cap:
            elevated_margin = True  # accepted, but over the soft cap

    if margin > hard_cap:
        return TradePlan(
            qty, leverage, notional, margin, 0.0, False,
            f"min lot margin {margin:.4f} > {hard_margin_pct:.0f}% of equity "
            f"({hard_cap:.4f}) even at {leverage}x",
        )

    # Verify notional meets exchange minimum.
    if notional < filters.min_notional:
        return TradePlan(
            qty, leverage, notional, margin, 0.0, False,
            f"notional {notional:.4f} < min_notional {filters.min_notional}",
        )

    # Effective $-risk at this qty.
    effective_risk = qty * stop_distance
    notes: list[str] = []
    if bumped:
        notes.append(
            f"qty bumped to exchange minimum; risk {effective_risk:.4f} "
            f"> target {risk_dollars:.4f}"
        )
    if elevated_margin:
        notes.append(
            f"expensive asset: margin {margin/equity*100:.1f}% of equity "
            f"(> soft cap {max_margin_pct:.0f}%) at {leverage}x"
        )
    reason = "; ".join(notes)
    return TradePlan(qty, leverage, notional, margin, effective_risk, True, reason)


__all__ = ["TradePlan", "adaptive_leverage", "compute_trade_plan"]
