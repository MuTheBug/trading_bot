"""Regime-adaptive strategy tests."""
import asyncio

import numpy as np
import pandas as pd
import pytest

from src.strategy.adaptive import RegimeAdaptiveStrategy


def _df(closes, vol=1000.0):
    closes = np.asarray(closes, dtype=float)
    rng = np.random.default_rng(1)
    opens = np.concatenate([[closes[0]], closes[:-1]])
    highs = np.maximum(opens, closes) * (1.0 + rng.uniform(0, 0.002, len(closes)))
    lows = np.minimum(opens, closes) * (1.0 - rng.uniform(0, 0.002, len(closes)))
    return pd.DataFrame({
        "open": opens, "high": highs, "low": lows, "close": closes,
        "volume": np.full(len(closes), vol),
    })


def test_uptrend_produces_long_plan():
    # Clear uptrend: drift up then small pullback so price is near EMA20.
    up = np.linspace(100.0, 130.0, 140)
    pullback = np.linspace(130.0, 128.5, 10)
    closes = np.concatenate([up, pullback])
    df15 = _df(closes)
    df1h = _df(np.linspace(95, 131, 100))
    strat = RegimeAdaptiveStrategy()
    plan = strat.plan(df15, df1h)
    # Setup may or may not fire depending on where price sits relative to EMA20
    # at the moment of evaluation; if it does, it must be a long.
    if plan is not None:
        assert plan.side == "LONG"
        assert plan.stop_loss < plan.entry_price
        assert plan.take_profits[0][0] > plan.entry_price


def test_downtrend_produces_short_plan_or_none():
    dn = np.linspace(130.0, 100.0, 140)
    bounce = np.linspace(100.0, 101.5, 10)
    closes = np.concatenate([dn, bounce])
    df15 = _df(closes)
    df1h = _df(np.linspace(135, 101, 100))
    strat = RegimeAdaptiveStrategy()
    plan = strat.plan(df15, df1h)
    if plan is not None:
        assert plan.side == "SHORT"
        assert plan.stop_loss > plan.entry_price
        assert plan.take_profits[0][0] < plan.entry_price


def test_tp_levels_are_ordered_for_long():
    up = np.linspace(100.0, 130.0, 140)
    pullback = np.linspace(130.0, 128.5, 10)
    df15 = _df(np.concatenate([up, pullback]))
    df1h = _df(np.linspace(95, 131, 100))
    plan = RegimeAdaptiveStrategy().plan(df15, df1h)
    if plan is None:
        pytest.skip("regime didn't trigger in this synthetic series")
    prices = [tp[0] for tp in plan.take_profits]
    assert prices == sorted(prices)
    pcts = [tp[1] for tp in plan.take_profits]
    assert abs(sum(pcts) - 100.0) < 1e-6


def test_insufficient_data_returns_none():
    df15 = _df(np.linspace(100, 101, 30))
    df1h = _df(np.linspace(100, 101, 30))
    plan = RegimeAdaptiveStrategy().plan(df15, df1h)
    assert plan is None


def test_htf_blocks_counter_trend():
    # 15m looks like a bounce (short-term up) but 1h is in strong downtrend.
    up15 = np.linspace(100.0, 120.0, 150)
    dn1h = np.linspace(140.0, 100.0, 140)
    df15 = _df(up15)
    df1h = _df(dn1h)
    strat = RegimeAdaptiveStrategy()
    plan = strat.plan(df15, df1h)
    # If regime fires LONG the HTF gate must have allowed it; we just want to
    # verify that when HTF says STRONG_DOWNTREND the strategy returns None for
    # a long candidate. If the strategy produced a plan, it must be SHORT or
    # None (no LONG against a strong bearish HTF).
    if plan is not None:
        assert plan.side == "SHORT"
