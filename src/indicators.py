"""Technical indicators — pure pandas/numpy, no external TA dep.

All functions take and return pandas Series/DataFrames so they are easy to
compose and unit-test against hand-verified values.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential moving average (Wilder-less, standard pandas ewm)."""
    return series.ewm(span=period, adjust=False, min_periods=period).mean()


def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(window=period, min_periods=period).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's RSI.

    Implemented as `100 * avg_gain / (avg_gain + avg_loss)` which is
    mathematically equivalent to the classic `100 - 100/(1+rs)` form but
    avoids division-by-zero when prices move only in one direction.
    """
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    # Wilder smoothing == EMA with alpha = 1/period
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    denom = avg_gain + avg_loss
    out = 100.0 * avg_gain / denom.replace(0, np.nan)
    # When both gain and loss are zero (no movement) there's no meaningful RSI;
    # fall back to the neutral 50. NaNs from the warm-up period propagate through.
    out = out.where(denom.notna(), 50.0)
    out = out.where(~((denom == 0) & denom.notna()), 50.0)
    return out


def true_range(df: pd.DataFrame) -> pd.Series:
    """True Range. df must have columns: high, low, close."""
    high = df["high"]
    low = df["low"]
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range (Wilder)."""
    tr = true_range(df)
    return tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


def adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Wilder's ADX. Returns DataFrame with columns: adx, plus_di, minus_di."""
    high = df["high"]
    low = df["low"]

    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr = true_range(df)
    atr_series = tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()

    plus_dm_s = pd.Series(plus_dm, index=df.index).ewm(
        alpha=1.0 / period, adjust=False, min_periods=period
    ).mean()
    minus_dm_s = pd.Series(minus_dm, index=df.index).ewm(
        alpha=1.0 / period, adjust=False, min_periods=period
    ).mean()

    plus_di = 100.0 * plus_dm_s / atr_series.replace(0, np.nan)
    minus_di = 100.0 * minus_dm_s / atr_series.replace(0, np.nan)

    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx_series = dx.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()

    return pd.DataFrame(
        {"adx": adx_series, "plus_di": plus_di, "minus_di": minus_di}
    )


def enrich(df: pd.DataFrame, cfg) -> pd.DataFrame:
    """Attach all strategy indicators to a kline DataFrame.

    `cfg` is a StrategyConfig instance (duck-typed).
    `df` must have columns: open, high, low, close, volume.
    """
    out = df.copy()
    out["ema_fast"] = ema(out["close"], cfg.ema_fast)
    out["ema_slow"] = ema(out["close"], cfg.ema_slow)
    out["rsi"] = rsi(out["close"], cfg.rsi_period)
    out["atr"] = atr(out, cfg.atr_period)
    adx_df = adx(out, cfg.adx_period)
    out["adx"] = adx_df["adx"]
    out["plus_di"] = adx_df["plus_di"]
    out["minus_di"] = adx_df["minus_di"]
    out["vol_sma"] = sma(out["volume"], cfg.volume_sma_period)
    return out


__all__ = ["ema", "sma", "rsi", "true_range", "atr", "adx", "enrich"]
