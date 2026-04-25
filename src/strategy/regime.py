"""Market regime classification.

Classifies the current market state into one of seven regimes using ADX,
EMA slope, Bollinger width, RSI, and ATR%. Each regime maps to a specific
playbook in the adaptive strategy.

Regimes
-------
STRONG_UPTREND   : ADX >= 25, price above rising EMA stack, DI+ > DI-
STRONG_DOWNTREND : ADX >= 25, price below falling EMA stack, DI- > DI+
WEAK_UPTREND     : 18 <= ADX < 25, positive EMA slope
WEAK_DOWNTREND   : 18 <= ADX < 25, negative EMA slope
RANGE            : ADX < 18, BB width below recent median -> mean reversion
BREAKOUT         : BB width expanding sharply, ATR% spike, direction via last bar
CHOP             : undecided - skip trading
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
import pandas as pd

from ..indicators import adx, atr, ema, rsi, sma


Regime = Literal[
    "STRONG_UPTREND",
    "STRONG_DOWNTREND",
    "WEAK_UPTREND",
    "WEAK_DOWNTREND",
    "RANGE",
    "BREAKOUT",
    "CHOP",
]


@dataclass
class RegimeSnapshot:
    regime: Regime
    direction: Literal["LONG", "SHORT", "BOTH", "NONE"]
    confidence: float          # 0..1 — how decisive the regime signature is
    adx: float
    ema_slope_pct: float       # EMA50 slope over last 10 bars, % of price
    bb_width_pct: float        # (upper - lower) / middle, percent
    bb_width_pctile: float     # 0..1 rank of bb_width in lookback window
    atr_pct: float             # ATR / close, percent
    rsi: float
    price_vs_ema50: float      # % distance, signed
    last_close: float
    reason: str

    def is_tradable(self) -> bool:
        return self.regime != "CHOP" and self.direction != "NONE"


def _bollinger(close: pd.Series, period: int = 20, mult: float = 2.0):
    mid = sma(close, period)
    std = close.rolling(window=period, min_periods=period).std()
    return mid - mult * std, mid, mid + mult * std


def classify_regime(
    df: pd.DataFrame,
    adx_strong: float = 25.0,
    adx_weak: float = 18.0,
    bb_squeeze_pctile: float = 0.30,
    bb_expand_pctile: float = 0.80,
    ema_fast: int = 20,
    ema_slow: int = 50,
    lookback: int = 100,
) -> Optional[RegimeSnapshot]:
    """Return a RegimeSnapshot for the last CLOSED bar.

    `df` must have columns: open, high, low, close, volume. At least
    `ema_slow + lookback` rows are recommended for stable signals.
    Returns None when there are too few bars.
    """
    if df is None or len(df) < max(ema_slow + 5, 60):
        return None

    d = df.copy()
    d["ema_fast"] = ema(d["close"], ema_fast)
    d["ema_slow"] = ema(d["close"], ema_slow)
    d["rsi"] = rsi(d["close"], 14)
    d["atr"] = atr(d, 14)
    adx_df = adx(d, 14)
    d["adx"] = adx_df["adx"]
    d["plus_di"] = adx_df["plus_di"]
    d["minus_di"] = adx_df["minus_di"]
    lo, mid, up = _bollinger(d["close"], 20, 2.0)
    d["bb_low"] = lo
    d["bb_mid"] = mid
    d["bb_up"] = up
    d["bb_width"] = (up - lo) / mid.replace(0, np.nan) * 100.0

    # Use second-to-last bar (last closed).
    if len(d) < 3:
        return None
    last = d.iloc[-2]
    needed = ["ema_fast", "ema_slow", "rsi", "atr", "adx", "plus_di",
              "minus_di", "bb_width"]
    if last[needed].isna().any():
        return None

    close = float(last["close"])
    atr_val = float(last["atr"])
    adx_val = float(last["adx"])
    plus_di = float(last["plus_di"])
    minus_di = float(last["minus_di"])
    rsi_val = float(last["rsi"])
    bb_w = float(last["bb_width"])

    # EMA slope: %-change of EMA50 over last 10 bars relative to price.
    ema_slow_s = d["ema_slow"].dropna()
    if len(ema_slow_s) < 12:
        return None
    ema_now = float(ema_slow_s.iloc[-2])
    ema_past = float(ema_slow_s.iloc[-12])
    ema_slope_pct = (ema_now - ema_past) / close * 100.0
    price_vs_ema50 = (close - ema_now) / ema_now * 100.0

    # BB width percentile within lookback window.
    bb_w_hist = d["bb_width"].dropna().iloc[-lookback:]
    if len(bb_w_hist) < 10:
        return None
    bb_pctile = float((bb_w_hist < bb_w).mean())
    atr_pct = atr_val / close * 100.0

    # --- Classify ---
    reason_parts = [
        f"adx={adx_val:.1f}",
        f"slope={ema_slope_pct:+.2f}%",
        f"bbw={bb_w:.2f}% (p{bb_pctile*100:.0f})",
        f"atr%={atr_pct:.2f}",
        f"rsi={rsi_val:.0f}",
    ]

    # Breakout: sharp expansion + strong directional close.
    if bb_pctile >= bb_expand_pctile and atr_pct > 0.8:
        last_bar = d.iloc[-2]
        bar_range = max(float(last_bar["high"] - last_bar["low"]), 1e-9)
        body = float(last_bar["close"] - last_bar["open"])
        body_frac = body / bar_range
        if body_frac > 0.5:
            conf = min(1.0, bb_pctile)
            return RegimeSnapshot(
                "BREAKOUT", "LONG", conf, adx_val, ema_slope_pct,
                bb_w, bb_pctile, atr_pct, rsi_val, price_vs_ema50,
                close, "breakout-up " + " ".join(reason_parts),
            )
        if body_frac < -0.5:
            conf = min(1.0, bb_pctile)
            return RegimeSnapshot(
                "BREAKOUT", "SHORT", conf, adx_val, ema_slope_pct,
                bb_w, bb_pctile, atr_pct, rsi_val, price_vs_ema50,
                close, "breakout-dn " + " ".join(reason_parts),
            )

    # Strong trends.
    if adx_val >= adx_strong and plus_di > minus_di and ema_slope_pct > 0.10:
        conf = min(1.0, (adx_val - adx_strong) / 20.0 + 0.6)
        return RegimeSnapshot(
            "STRONG_UPTREND", "LONG", conf, adx_val, ema_slope_pct,
            bb_w, bb_pctile, atr_pct, rsi_val, price_vs_ema50,
            close, " ".join(reason_parts),
        )
    if adx_val >= adx_strong and minus_di > plus_di and ema_slope_pct < -0.10:
        conf = min(1.0, (adx_val - adx_strong) / 20.0 + 0.6)
        return RegimeSnapshot(
            "STRONG_DOWNTREND", "SHORT", conf, adx_val, ema_slope_pct,
            bb_w, bb_pctile, atr_pct, rsi_val, price_vs_ema50,
            close, " ".join(reason_parts),
        )

    # Weak trends.
    if adx_weak <= adx_val < adx_strong and ema_slope_pct > 0.05:
        return RegimeSnapshot(
            "WEAK_UPTREND", "LONG", 0.5, adx_val, ema_slope_pct,
            bb_w, bb_pctile, atr_pct, rsi_val, price_vs_ema50,
            close, " ".join(reason_parts),
        )
    if adx_weak <= adx_val < adx_strong and ema_slope_pct < -0.05:
        return RegimeSnapshot(
            "WEAK_DOWNTREND", "SHORT", 0.5, adx_val, ema_slope_pct,
            bb_w, bb_pctile, atr_pct, rsi_val, price_vs_ema50,
            close, " ".join(reason_parts),
        )

    # Range (low ADX + squeezed BB).
    if adx_val < adx_weak and bb_pctile <= bb_squeeze_pctile:
        conf = 1.0 - bb_pctile
        return RegimeSnapshot(
            "RANGE", "BOTH", conf, adx_val, ema_slope_pct,
            bb_w, bb_pctile, atr_pct, rsi_val, price_vs_ema50,
            close, "squeeze " + " ".join(reason_parts),
        )

    return RegimeSnapshot(
        "CHOP", "NONE", 0.0, adx_val, ema_slope_pct,
        bb_w, bb_pctile, atr_pct, rsi_val, price_vs_ema50,
        close, "undecided " + " ".join(reason_parts),
    )


__all__ = ["Regime", "RegimeSnapshot", "classify_regime"]
