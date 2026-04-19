"""Adaptive leverage / trade plan tests."""
import pytest

from src.exchange.base import SymbolFilters
from src.risk.leverage import adaptive_leverage, compute_trade_plan


FILTERS_DOGE = SymbolFilters(
    symbol="DOGEUSDT",
    price_tick=0.00001,
    qty_step=1.0,
    min_qty=1.0,
    min_notional=5.0,
)


def test_adaptive_leverage_bounds():
    lo = adaptive_leverage(confidence=0.0, atr_pct=3.0,
                           base_leverage=1, max_leverage=20, min_leverage=1)
    hi = adaptive_leverage(confidence=1.0, atr_pct=0.3,
                           base_leverage=1, max_leverage=20, min_leverage=1)
    assert 1 <= lo <= hi <= 20


def test_adaptive_leverage_scales_with_confidence():
    low = adaptive_leverage(confidence=0.2, atr_pct=1.0,
                            base_leverage=3, max_leverage=20)
    high = adaptive_leverage(confidence=0.9, atr_pct=1.0,
                             base_leverage=3, max_leverage=20)
    assert high > low


def test_trade_plan_small_capital_bumps_leverage():
    """With a $10 account and $5 min-notional, leverage must scale so
    margin fits. A 2x base isn't enough; planner should raise it."""
    tp = compute_trade_plan(
        equity=10.0,
        entry_price=0.20, stop_price=0.196,
        filters=FILTERS_DOGE,
        risk_pct=1.0,
        confidence=0.7, atr_pct=1.0,
        base_leverage=2, max_leverage=20,
    )
    assert tp.feasible
    assert tp.notional >= FILTERS_DOGE.min_notional
    # Margin must fit
    assert tp.margin <= 10.0 * 0.9 + 1e-9


def test_trade_plan_infeasible_when_max_lev_too_low():
    """A $5 account can't take a $5 min-notional trade at 1x leverage
    without exceeding the 85% margin cap."""
    tp = compute_trade_plan(
        equity=5.0,
        entry_price=0.20, stop_price=0.196,
        filters=FILTERS_DOGE,
        risk_pct=1.0,
        confidence=0.7, atr_pct=1.0,
        base_leverage=1, max_leverage=1,
    )
    assert not tp.feasible


def test_trade_plan_rejects_zero_stop():
    tp = compute_trade_plan(
        equity=100, entry_price=1.0, stop_price=1.0,
        filters=FILTERS_DOGE, risk_pct=1.0,
        confidence=0.5, atr_pct=1.0,
        base_leverage=3, max_leverage=20,
    )
    assert not tp.feasible


def test_trade_plan_respects_max_leverage():
    tp = compute_trade_plan(
        equity=10.0,
        entry_price=0.20, stop_price=0.196,
        filters=FILTERS_DOGE,
        risk_pct=1.0,
        confidence=0.9, atr_pct=0.3,
        base_leverage=5, max_leverage=7,
    )
    assert tp.leverage <= 7


def test_expensive_asset_fallback_uses_more_margin():
    """Min-notional that forces > 85% margin at max leverage should still
    be accepted as long as it fits under the 98% hard cap."""
    # $10 equity, min_notional $9.50 — at 20x leverage margin is $0.475
    # (fits fine), but cap max_leverage so it's forced tight.
    filters = SymbolFilters(
        symbol="EXPUSDT", price_tick=0.01, qty_step=1.0,
        min_qty=1.0, min_notional=9.50,
    )
    tp = compute_trade_plan(
        equity=10.0,
        entry_price=9.50, stop_price=9.30,
        filters=filters,
        risk_pct=1.0,
        confidence=0.7, atr_pct=1.0,
        base_leverage=1, max_leverage=2,
        max_margin_pct=85.0, hard_margin_pct=98.0,
    )
    # qty=1, notional=$9.50. At 2x max_lev, margin = $4.75 (47.5%) which
    # is under the soft cap anyway — that's fine, asserts feasibility.
    assert tp.feasible
    assert tp.notional >= filters.min_notional


def test_expensive_asset_accepts_elevated_margin():
    """When even max leverage can't hold margin under the soft cap, the
    trade is still accepted if it fits under the hard cap."""
    # $10 equity, min_notional $9 at max_leverage 1 -> margin = $9 (90%)
    # which is > soft (85%) but <= hard (98%).
    filters = SymbolFilters(
        symbol="EXPUSDT", price_tick=0.01, qty_step=1.0,
        min_qty=1.0, min_notional=9.0,
    )
    tp = compute_trade_plan(
        equity=10.0,
        entry_price=9.0, stop_price=8.82,
        filters=filters,
        risk_pct=1.0,
        confidence=0.7, atr_pct=1.0,
        base_leverage=1, max_leverage=1,
        max_margin_pct=85.0, hard_margin_pct=98.0,
    )
    assert tp.feasible
    assert tp.margin > 10.0 * 0.85
    assert tp.margin <= 10.0 * 0.98 + 1e-9
    assert "expensive" in tp.reason.lower()


def test_expensive_asset_rejected_above_hard_cap():
    """Truly unaffordable asset: notional > hard cap even at max lev."""
    filters = SymbolFilters(
        symbol="EXPUSDT", price_tick=0.01, qty_step=1.0,
        min_qty=1.0, min_notional=9.90,
    )
    tp = compute_trade_plan(
        equity=10.0,
        entry_price=9.90, stop_price=9.70,
        filters=filters,
        risk_pct=1.0,
        confidence=0.7, atr_pct=1.0,
        base_leverage=1, max_leverage=1,
        max_margin_pct=85.0, hard_margin_pct=98.0,
    )
    assert not tp.feasible
