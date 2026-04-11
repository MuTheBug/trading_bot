"""Trend-momentum hybrid: EMA cross + ADX + RSI + volume + HTF filter."""
from __future__ import annotations

from typing import Optional

import pandas as pd

from ..config import StrategyConfig
from ..indicators import enrich, ema
from .base import Signal, Strategy


class TrendMomentumStrategy(Strategy):
    """Signal stack (all must agree):

    1. HTF filter:      close > EMA(htf) for LONG, < for SHORT
    2. Primary trigger: EMA_fast crosses EMA_slow in trend direction on the
                        last closed candle
    3. Trend strength:  ADX >= adx_threshold
    4. Momentum:        RSI within [40, 70] for LONG, [30, 60] for SHORT
    5. Volume:          volume > SMA20(volume)
    """

    def __init__(self, cfg: StrategyConfig) -> None:
        self.cfg = cfg

    def evaluate(
        self, df15: pd.DataFrame, df1h: pd.DataFrame
    ) -> Optional[Signal]:
        c = self.cfg
        min_bars = max(c.ema_slow, c.adx_period, c.rsi_period, c.atr_period, c.volume_sma_period) + 5
        if len(df15) < min_bars or len(df1h) < c.ema_htf + 2:
            return None

        d = enrich(df15, c)
        # Use second-to-last row = last CLOSED candle. -1 is the still-forming one.
        last = d.iloc[-2]
        prev = d.iloc[-3]
        if last[["ema_fast", "ema_slow", "adx", "rsi", "atr", "vol_sma"]].isna().any():
            return None

        # HTF trend
        df1h = df1h.copy()
        df1h["ema_htf"] = ema(df1h["close"], c.ema_htf)
        htf_last = df1h.iloc[-2]
        if pd.isna(htf_last["ema_htf"]):
            return None
        htf_bull = htf_last["close"] > htf_last["ema_htf"]
        htf_bear = htf_last["close"] < htf_last["ema_htf"]

        cross_up = (prev["ema_fast"] <= prev["ema_slow"]) and (last["ema_fast"] > last["ema_slow"])
        cross_dn = (prev["ema_fast"] >= prev["ema_slow"]) and (last["ema_fast"] < last["ema_slow"])

        adx_ok = last["adx"] >= c.adx_threshold
        vol_ok = last["volume"] > last["vol_sma"]

        if cross_up and htf_bull and adx_ok and vol_ok:
            if c.rsi_long_min <= last["rsi"] <= c.rsi_long_max:
                return Signal(
                    side="LONG",
                    entry_price=float(last["close"]),
                    atr=float(last["atr"]),
                    reason=f"EMA{c.ema_fast}x{c.ema_slow} bull cross, ADX={last['adx']:.1f}, "
                           f"RSI={last['rsi']:.1f}, HTF bull",
                )

        if cross_dn and htf_bear and adx_ok and vol_ok:
            if c.rsi_short_min <= last["rsi"] <= c.rsi_short_max:
                return Signal(
                    side="SHORT",
                    entry_price=float(last["close"]),
                    atr=float(last["atr"]),
                    reason=f"EMA{c.ema_fast}x{c.ema_slow} bear cross, ADX={last['adx']:.1f}, "
                           f"RSI={last['rsi']:.1f}, HTF bear",
                )

        return None


__all__ = ["TrendMomentumStrategy"]
