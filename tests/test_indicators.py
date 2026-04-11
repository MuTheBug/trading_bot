"""Indicator sanity checks against hand-computed values."""
import numpy as np
import pandas as pd
import pytest

from src.indicators import ema, sma, rsi, atr, adx


def _df(highs, lows, closes, volumes=None):
    n = len(closes)
    opens = [c for c in closes]
    if volumes is None:
        volumes = [100] * n
    return pd.DataFrame({
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": volumes,
    })


def test_sma_basic():
    s = pd.Series([1, 2, 3, 4, 5], dtype=float)
    out = sma(s, 3)
    assert pd.isna(out.iloc[0])
    assert pd.isna(out.iloc[1])
    assert out.iloc[2] == pytest.approx(2.0)
    assert out.iloc[3] == pytest.approx(3.0)
    assert out.iloc[4] == pytest.approx(4.0)


def test_ema_converges():
    s = pd.Series([10.0] * 50)
    out = ema(s, 9)
    # EMA of a constant series equals the constant
    assert out.iloc[-1] == pytest.approx(10.0)


def test_rsi_bounds():
    # monotonically increasing prices -> RSI approaches 100
    s = pd.Series(np.linspace(100, 200, 60))
    out = rsi(s, 14)
    assert out.iloc[-1] > 90
    # monotonically decreasing -> RSI approaches 0
    s2 = pd.Series(np.linspace(200, 100, 60))
    out2 = rsi(s2, 14)
    assert out2.iloc[-1] < 10


def test_atr_positive_on_volatility():
    highs = list(np.linspace(100, 120, 30))
    lows = [h - 2 for h in highs]
    closes = [(h + l) / 2 for h, l in zip(highs, lows)]
    df = _df(highs, lows, closes)
    out = atr(df, 14)
    # ATR should be positive and roughly reflect the 2-unit daily range
    assert out.iloc[-1] > 0
    assert out.iloc[-1] < 10


def test_adx_rises_in_trend():
    # Synthetic uptrend: widening highs and lows
    n = 60
    highs = [100 + i * 1.0 for i in range(n)]
    lows = [99 + i * 1.0 for i in range(n)]
    closes = [99.5 + i * 1.0 for i in range(n)]
    df = _df(highs, lows, closes)
    out = adx(df, 14)
    assert out["adx"].iloc[-1] > 20
    assert out["plus_di"].iloc[-1] > out["minus_di"].iloc[-1]
