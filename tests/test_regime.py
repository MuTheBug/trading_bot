"""Regime classifier tests — synthesized price series for each regime."""
import numpy as np
import pandas as pd
import pytest

from src.strategy.regime import classify_regime


def _make_df(closes, vol=1000.0):
    n = len(closes)
    closes = np.asarray(closes, dtype=float)
    # Generate plausible OHLC around close; small 0.2% noise for highs/lows.
    rng = np.random.default_rng(42)
    noise_up = rng.uniform(0.0, 0.003, n)
    noise_dn = rng.uniform(0.0, 0.003, n)
    opens = np.concatenate([[closes[0]], closes[:-1]])
    highs = np.maximum(opens, closes) * (1.0 + noise_up)
    lows = np.minimum(opens, closes) * (1.0 - noise_dn)
    return pd.DataFrame({
        "open": opens, "high": highs, "low": lows, "close": closes,
        "volume": np.full(n, vol),
    })


def test_regime_uptrend():
    closes = np.linspace(100.0, 130.0, 150)
    df = _make_df(closes)
    snap = classify_regime(df)
    assert snap is not None
    assert snap.regime in ("STRONG_UPTREND", "WEAK_UPTREND", "BREAKOUT")
    assert snap.direction == "LONG"


def test_regime_downtrend():
    closes = np.linspace(130.0, 100.0, 150)
    df = _make_df(closes)
    snap = classify_regime(df)
    assert snap is not None
    assert snap.regime in ("STRONG_DOWNTREND", "WEAK_DOWNTREND", "BREAKOUT")
    assert snap.direction == "SHORT"


def test_regime_range():
    # Tight oscillation, ADX will be very low
    t = np.arange(200)
    closes = 100.0 + 0.3 * np.sin(t / 4.0)
    df = _make_df(closes)
    snap = classify_regime(df)
    assert snap is not None
    # Tight oscillation should not classify as a strong trend.
    assert snap.regime not in ("STRONG_UPTREND", "STRONG_DOWNTREND")


def test_regime_chop():
    # Pure noise around a fixed level — neither trend nor tight squeeze.
    rng = np.random.default_rng(0)
    closes = 100.0 + rng.normal(0, 0.6, 200)
    df = _make_df(closes)
    snap = classify_regime(df)
    assert snap is not None
    assert snap.regime != "STRONG_UPTREND"
    assert snap.regime != "STRONG_DOWNTREND"


def test_regime_too_short_returns_none():
    df = _make_df(np.linspace(100, 101, 20))
    assert classify_regime(df) is None
