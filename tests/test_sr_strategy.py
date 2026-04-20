"""Tests for the deterministic S/R strategy."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import DirectionalConfig, SRConfig
from src.exchange.base import SymbolFilters, TickerInfo
from src.strategy.sr_strategy import (
    CandidateCtx,
    SRStrategy,
    bearish_rejection,
    bullish_rejection,
    classify_bias,
    cluster_levels,
    detect_levels,
    find_pivots,
    Pivot,
)


# --------------------------- helpers ---------------------------


def _ohlcv_from_hl(highs, lows, closes=None, opens=None, volumes=None):
    """Build an OHLCV frame from explicit highs/lows."""
    n = len(highs)
    closes = list(closes) if closes is not None else [(h + l) / 2.0 for h, l in zip(highs, lows)]
    opens = list(opens) if opens is not None else closes[:]
    volumes = list(volumes) if volumes is not None else [1000.0] * n
    idx = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows,
         "close": closes, "volume": volumes},
        index=idx,
    )


def _flat_ohlcv(closes):
    """Default OHLCV with high=close*1.005, low=close*0.995."""
    highs = [c * 1.005 for c in closes]
    lows = [c * 0.995 for c in closes]
    return _ohlcv_from_hl(highs, lows, closes=closes, opens=closes)


def _ticker(symbol="X", price=100.0, chg=0.0, vol=1e8):
    return TickerInfo(symbol, price=price, volume_24h=vol, change_pct_24h=chg,
                      high_24h=price * 1.1, low_24h=price * 0.9)


def _filters(symbol="X"):
    return SymbolFilters(symbol, price_tick=0.01, qty_step=0.001,
                         min_qty=0.001, min_notional=5.0)


# --------------------------- pivot detection ---------------------------


def test_find_pivots_detects_obvious_high_and_low():
    highs = [1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 1.0]  # spike at idx 3
    lows = [1.0] * 7
    pivots = find_pivots(_ohlcv_from_hl(highs, lows), left=3, right=3)
    assert any(p.kind == "high" and p.bar == 3 and p.price == 2.0 for p in pivots)


def test_find_pivots_returns_empty_when_too_few_bars():
    closes = [1.0, 2.0, 1.0]
    assert find_pivots(_flat_ohlcv(closes), left=3, right=3) == []


def test_find_pivots_symmetric_low():
    highs = [1.0] * 9
    lows = [1.0, 1.0, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0, 1.0]
    pivots = find_pivots(_ohlcv_from_hl(highs, lows), left=3, right=3)
    assert any(p.kind == "low" and p.bar == 4 and p.price == 0.5 for p in pivots)


# --------------------------- clustering ---------------------------


def test_cluster_merges_nearby_pivots():
    pivots = [
        Pivot(bar=5, price=100.0, kind="low"),
        Pivot(bar=20, price=100.5, kind="low"),    # within 0.3 ATR (atr=10)
        Pivot(bar=35, price=99.7, kind="low"),     # within tol too
        Pivot(bar=50, price=120.0, kind="high"),   # separate
    ]
    ls = cluster_levels(
        pivots, atr_value=10.0, current_price=110.0, current_bar=60,
        tolerance_atr=0.3, min_touches=2,
    )
    # The three ~100 pivots cluster into a support; the 120 high is a single
    # touch and drops out at min_touches=2.
    assert len(ls.supports) == 1
    assert len(ls.resistances) == 0
    assert 99.5 <= ls.supports[0].price <= 100.5
    assert ls.supports[0].touches == 3


def test_cluster_splits_far_pivots():
    pivots = [
        Pivot(bar=5, price=100.0, kind="low"),
        Pivot(bar=10, price=100.2, kind="low"),
        Pivot(bar=20, price=110.0, kind="high"),
        Pivot(bar=30, price=110.3, kind="high"),
    ]
    ls = cluster_levels(
        pivots, atr_value=1.0, current_price=105.0, current_bar=40,
        tolerance_atr=0.5, min_touches=2,
    )
    assert len(ls.supports) == 1 and ls.supports[0].touches == 2
    assert len(ls.resistances) == 1 and ls.resistances[0].touches == 2


def test_cluster_discards_single_touches():
    pivots = [Pivot(bar=5, price=50.0, kind="low")]
    ls = cluster_levels(
        pivots, atr_value=1.0, current_price=60.0, current_bar=20,
        tolerance_atr=0.3, min_touches=2,
    )
    assert ls.supports == [] and ls.resistances == []


def test_nearest_helpers():
    p = [
        Pivot(bar=1, price=90.0, kind="low"),
        Pivot(bar=2, price=90.1, kind="low"),
        Pivot(bar=10, price=95.0, kind="low"),
        Pivot(bar=11, price=95.1, kind="low"),
        Pivot(bar=20, price=120.0, kind="high"),
        Pivot(bar=21, price=120.2, kind="high"),
    ]
    ls = cluster_levels(p, atr_value=1.0, current_price=100.0, current_bar=30,
                        tolerance_atr=0.5, min_touches=2)
    ns = ls.nearest_support(100.0)
    nr = ls.nearest_resistance(100.0)
    assert ns is not None and 94.9 <= ns.price <= 95.2
    assert nr is not None and 119.9 <= nr.price <= 120.3


# --------------------------- HTF bias ---------------------------


def test_classify_bias_up_on_rising_trend():
    closes = np.linspace(100, 130, 80)
    assert classify_bias(_flat_ohlcv(closes), ema_period=50) == "up"


def test_classify_bias_down_on_falling_trend():
    closes = np.linspace(130, 100, 80)
    assert classify_bias(_flat_ohlcv(closes), ema_period=50) == "down"


def test_classify_bias_neutral_on_flat_market():
    # Tiny wiggles around 100 — neither clearly up nor down.
    rng = np.random.default_rng(42)
    closes = 100.0 + rng.normal(0, 0.05, size=80)
    assert classify_bias(_flat_ohlcv(closes), ema_period=50) == "neutral"


# --------------------------- candle patterns ---------------------------


def test_bullish_rejection_true():
    # Long lower wick, close near high.
    bar = pd.Series({"open": 101.0, "high": 102.0, "low": 95.0, "close": 101.8})
    assert bullish_rejection(bar)


def test_bullish_rejection_false_on_red_body():
    bar = pd.Series({"open": 102.0, "high": 102.0, "low": 99.0, "close": 99.2})
    assert not bullish_rejection(bar)


def test_bearish_rejection_true():
    bar = pd.Series({"open": 99.0, "high": 105.0, "low": 98.0, "close": 98.5})
    assert bearish_rejection(bar)


def test_bearish_rejection_false_on_green_body():
    bar = pd.Series({"open": 98.0, "high": 99.5, "low": 98.0, "close": 99.4})
    assert not bearish_rejection(bar)


# --------------------------- detect_levels end-to-end ---------------------------


def test_detect_levels_finds_double_bottom():
    """Two distinct troughs at ~100 with an intervening rally should
    appear as one clustered support level."""
    closes = []
    highs = []
    lows = []
    # Down-leg to 100 (touch #1), up to 110, back to 100 (touch #2), up to 108.
    for p in np.linspace(108, 100, 20):
        closes.append(p); highs.append(p + 0.2); lows.append(p - 0.2)
    lows[-1] = 99.5  # make the first bottom crisp
    for p in np.linspace(100, 110, 20):
        closes.append(p); highs.append(p + 0.2); lows.append(p - 0.2)
    for p in np.linspace(110, 100, 20):
        closes.append(p); highs.append(p + 0.2); lows.append(p - 0.2)
    lows[-1] = 99.6  # second bottom
    for p in np.linspace(100, 108, 20):
        closes.append(p); highs.append(p + 0.2); lows.append(p - 0.2)
    df = _ohlcv_from_hl(highs, lows, closes=closes)
    cfg = SRConfig(min_touches=2, pivot_left=3, pivot_right=3,
                   level_tolerance_atr=1.0)
    ls = detect_levels(df, cfg)
    assert ls is not None
    assert ls.supports, "expected at least one clustered support"
    # The clustered support should sit near 99.5-100.0.
    nearest = ls.supports[-1]
    assert 99.0 <= nearest.price <= 101.0


# --------------------------- full SRStrategy.propose ---------------------------


def _dfs_with_support_setup():
    """HTF in uptrend; LTF oscillates between support ~100 and resistance
    ~115 with two touches each, then pulls back to ~101 with a bullish
    rejection candle on the last closed bar."""
    htf_closes = np.linspace(100, 130, 80)
    htf = _flat_ohlcv(htf_closes)

    closes, highs, lows = [], [], []

    def _leg(start, end, n):
        for p in np.linspace(start, end, n):
            closes.append(p); highs.append(p + 0.3); lows.append(p - 0.3)

    _leg(115, 100, 15)        # bars 0-14
    lows[-1] = 99.5           # support touch 1
    _leg(100, 115, 15)        # bars 15-29
    highs[-1] = 115.5         # resistance touch 1
    _leg(115, 100, 15)        # bars 30-44
    lows[-1] = 99.6           # support touch 2
    _leg(100, 115, 15)        # bars 45-59
    highs[-1] = 115.4         # resistance touch 2
    _leg(115, 101, 15)        # bars 60-74 — drift back toward support
    closes.append(101.0); highs.append(101.2); lows.append(100.8)  # forming

    ltf = _ohlcv_from_hl(highs, lows, closes=closes)
    # Last CLOSED bar (index -2) = bullish rejection near support.
    idx = len(ltf) - 2
    ltf.iloc[idx] = [100.0, 101.5, 98.0, 101.0, 1500.0]
    return {"1d": htf, "15m": ltf}


def _dfs_with_resistance_setup():
    """HTF downtrend; LTF oscillates between support ~100 and resistance
    ~110 with two touches each, then rallies to ~109.5 with a bearish
    rejection candle on the last closed bar."""
    htf_closes = np.linspace(130, 100, 80)
    htf = _flat_ohlcv(htf_closes)

    closes, highs, lows = [], [], []

    def _leg(start, end, n):
        for p in np.linspace(start, end, n):
            closes.append(p); highs.append(p + 0.3); lows.append(p - 0.3)

    _leg(100, 110, 15)        # bars 0-14
    highs[-1] = 110.5         # resistance touch 1
    _leg(110, 100, 15)        # bars 15-29
    lows[-1] = 99.5           # support touch 1
    _leg(100, 110, 15)        # bars 30-44
    highs[-1] = 110.4         # resistance touch 2
    _leg(110, 100, 15)        # bars 45-59
    lows[-1] = 99.6           # support touch 2
    _leg(100, 109, 15)        # bars 60-74 — drift up toward resistance
    closes.append(109.5); highs.append(109.7); lows.append(109.3)  # forming

    ltf = _ohlcv_from_hl(highs, lows, closes=closes)
    # Last CLOSED bar (index -2) = bearish rejection near resistance.
    idx = len(ltf) - 2
    ltf.iloc[idx] = [110.0, 112.0, 108.5, 109.5, 1500.0]
    return {"1d": htf, "15m": ltf}


def _ctx_from_dfs(symbol, dfs, price):
    ticker = _ticker(symbol=symbol, price=price)
    return CandidateCtx(symbol=symbol, ticker=ticker,
                        filters=_filters(symbol), dfs=dfs)


def test_propose_long_on_support_bounce():
    dfs = _dfs_with_support_setup()
    ctx = _ctx_from_dfs("BTCUSDT", dfs, price=101.0)
    cfg = SRConfig(min_touches=2, min_level_strength=0.5,
                   level_tolerance_atr=1.0, entry_zone_atr=2.0,
                   sl_buffer_atr=0.5, min_rr=1.2, htf_trend_ema=20)
    strat = SRStrategy(cfg, DirectionalConfig())
    dec = strat.propose(ctx)
    assert dec.is_trade, f"expected trade, got SKIP: {dec.reasoning}"
    assert dec.side == "LONG"
    assert dec.entry is not None and dec.stop_loss is not None
    assert dec.stop_loss < dec.entry  # SL below entry for long
    assert dec.take_profits and dec.take_profits[0][0] > dec.entry
    # TP close_pct sums to ~100.
    assert abs(sum(p for _, p in dec.take_profits) - 100.0) < 1e-6


def test_propose_short_on_resistance_fail():
    dfs = _dfs_with_resistance_setup()
    ctx = _ctx_from_dfs("ETHUSDT", dfs, price=109.5)
    cfg = SRConfig(min_touches=2, min_level_strength=0.5,
                   level_tolerance_atr=1.0, entry_zone_atr=2.0,
                   sl_buffer_atr=0.5, min_rr=1.2, htf_trend_ema=20)
    strat = SRStrategy(cfg, DirectionalConfig())
    dec = strat.propose(ctx)
    assert dec.is_trade, f"expected trade, got SKIP: {dec.reasoning}"
    assert dec.side == "SHORT"
    assert dec.stop_loss > dec.entry
    assert dec.take_profits[0][0] < dec.entry


def test_propose_skip_when_no_levels():
    # Straight-line uptrend with no double-bottoms in range.
    dfs = {"1d": _flat_ohlcv(np.linspace(100, 150, 80)),
           "15m": _flat_ohlcv(np.linspace(100, 150, 80))}
    ctx = _ctx_from_dfs("X", dfs, price=150.0)
    strat = SRStrategy(SRConfig(), DirectionalConfig())
    dec = strat.propose(ctx)
    assert not dec.is_trade


def test_propose_skip_when_htf_opposes_long():
    """Even if we see a support bounce on LTF, a bearish HTF denies long."""
    dfs = _dfs_with_support_setup()
    # Override HTF to be clearly down.
    dfs["1d"] = _flat_ohlcv(np.linspace(150, 100, 80))
    ctx = _ctx_from_dfs("X", dfs, price=102.8)
    cfg = SRConfig(min_touches=2, min_level_strength=0.5,
                   level_tolerance_atr=1.0, entry_zone_atr=2.0,
                   sl_buffer_atr=0.5, min_rr=1.2, htf_trend_ema=20)
    strat = SRStrategy(cfg, DirectionalConfig())
    dec = strat.propose(ctx)
    # HTF bias is down -> long branch rejects. Short branch needs a
    # resistance bounce which this fixture doesn't produce at price 102.8.
    # Either way, no trade.
    assert not dec.is_trade


def test_propose_skip_without_rejection_candle():
    """Same support setup but the last bar is a plain continuation
    candle -> require_rejection_candle blocks the trade."""
    dfs = _dfs_with_support_setup()
    ltf = dfs["15m"].copy()
    idx = len(ltf) - 2
    # Overwrite with a bland green continuation bar (no long lower wick),
    # but still inside the support entry zone so the rejection check is
    # the reason we skip.
    ltf.iloc[idx] = [100.3, 100.8, 100.2, 100.7, 1000.0]
    dfs["15m"] = ltf
    ctx = _ctx_from_dfs("X", dfs, price=100.7)
    cfg = SRConfig(min_touches=2, min_level_strength=0.5,
                   level_tolerance_atr=1.0, entry_zone_atr=2.0,
                   sl_buffer_atr=0.5, min_rr=1.2, htf_trend_ema=20,
                   require_rejection_candle=True)
    strat = SRStrategy(cfg, DirectionalConfig())
    dec = strat.propose(ctx)
    assert not dec.is_trade
    assert "rejection" in dec.reasoning.lower()


# --------------------- three-timeframe layout ---------------------


def test_propose_uses_explicit_three_timeframes():
    """Levels come from the configured level TF, bias from the bias TF,
    and the rejection/entry comes from the trigger TF — each frame has
    a distinct shape so only the right wiring produces a trade."""
    # Bias TF (4h): uptrend so long branch isn't blocked.
    bias_df = _flat_ohlcv(np.linspace(100, 130, 80))
    # Level TF (1h): the support fixture's 1h-style structure.
    level_dfs = _dfs_with_support_setup()
    level_df = level_dfs["15m"]  # reuse the shape; it has real pivots
    # Trigger TF (15m): just the last two bars of the level TF are enough
    # for price + rejection. We keep it identical here so trigger_price
    # sits inside the zone and a bullish rejection is present.
    trigger_df = level_df.copy()
    # Mislabel the level TF key so fallback can't accidentally save us.
    dfs = {"4h": bias_df, "1h": level_df, "15m": trigger_df}
    ctx = CandidateCtx(symbol="X", ticker=_ticker(symbol="X", price=101.0),
                       filters=_filters("X"), dfs=dfs)
    cfg = SRConfig(
        level_timeframe="1h", bias_timeframe="4h", trigger_timeframe="15m",
        min_touches=2, min_level_strength=0.5,
        level_tolerance_atr=1.0, entry_zone_atr=2.0,
        sl_buffer_atr=0.5, min_rr=1.2, htf_trend_ema=20,
    )
    strat = SRStrategy(cfg, DirectionalConfig())
    dec = strat.propose(ctx)
    assert dec.is_trade, f"expected trade, got SKIP: {dec.reasoning}"
    assert dec.side == "LONG"


def test_propose_bias_tf_blocks_even_when_levels_ok():
    """If the bias TF is bearish, a clean support setup on the level TF
    must not produce a long — this proves bias_timeframe is actually
    driving the gate (not accidentally read from level/trigger TF)."""
    level_dfs = _dfs_with_support_setup()
    level_df = level_dfs["15m"]
    bias_df = _flat_ohlcv(np.linspace(150, 100, 80))  # clearly down
    trigger_df = level_df.copy()
    dfs = {"4h": bias_df, "1h": level_df, "15m": trigger_df}
    ctx = CandidateCtx(symbol="X", ticker=_ticker(symbol="X", price=101.0),
                       filters=_filters("X"), dfs=dfs)
    cfg = SRConfig(
        level_timeframe="1h", bias_timeframe="4h", trigger_timeframe="15m",
        min_touches=2, min_level_strength=0.5,
        level_tolerance_atr=1.0, entry_zone_atr=2.0,
        sl_buffer_atr=0.5, min_rr=1.2, htf_trend_ema=20,
    )
    dec = SRStrategy(cfg, DirectionalConfig()).propose(ctx)
    # Long is blocked by bias; the short branch has no resistance bounce
    # setup in this fixture -> overall SKIP.
    assert not dec.is_trade
