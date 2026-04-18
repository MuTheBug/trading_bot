"""Regime-adaptive directional strategy.

Picks a playbook per market regime:

* STRONG_UPTREND   -> pullback-long to EMA20, ride with trailing stop
* STRONG_DOWNTREND -> rally-short to EMA20, ride with trailing stop
* WEAK_UPTREND     -> long only when RSI < 55 and price > EMA20 (small size)
* WEAK_DOWNTREND   -> short only when RSI > 45 and price < EMA20 (small size)
* RANGE            -> long at BB lower / RSI < 30, short at BB upper / RSI > 70
* BREAKOUT         -> trade with last bar's direction when BB expanding
* CHOP             -> skip

Stops and take-profits are ATR-based; take-profit ratios vary per regime to
match its edge characteristics (trend = let runner fly, range = quick scalps).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import pandas as pd

from ..indicators import atr, ema, rsi, sma
from .base import Side, Signal, Strategy
from .regime import RegimeSnapshot, classify_regime


@dataclass
class DirectionalPlan:
    """Full trade plan: regime + signal + SL/TP grid."""
    regime: RegimeSnapshot
    side: Side
    entry_price: float
    stop_loss: float
    take_profits: List[Tuple[float, float]]  # (price, close_pct)
    atr: float
    confidence: float
    reason: str

    def to_signal(self, leverage: Optional[int] = None) -> Signal:
        return Signal(
            side=self.side,
            entry_price=self.entry_price,
            atr=self.atr,
            reason=self.reason,
            stop_loss=self.stop_loss,
            take_profits=self.take_profits,
            leverage=leverage,
            confidence=self.confidence,
        )


def _sl_tp(
    side: Side,
    entry: float,
    atr_val: float,
    sl_mult: float,
    tp_mults: List[Tuple[float, float]],
) -> Tuple[float, List[Tuple[float, float]]]:
    if side == "LONG":
        sl = entry - sl_mult * atr_val
        tps = [(entry + m * atr_val, pct) for m, pct in tp_mults]
    else:
        sl = entry + sl_mult * atr_val
        tps = [(entry - m * atr_val, pct) for m, pct in tp_mults]
    return sl, tps


class RegimeAdaptiveStrategy(Strategy):
    """Regime -> playbook dispatcher."""

    def __init__(
        self,
        adx_strong: float = 25.0,
        adx_weak: float = 18.0,
        rsi_overbought: float = 70.0,
        rsi_oversold: float = 30.0,
        # Per-regime SL / TP multiples of ATR. Trend regimes get wider stops
        # and runners; range regimes get tight stops and quick scalps.
        trend_sl_mult: float = 2.0,
        trend_tps: Optional[List[Tuple[float, float]]] = None,
        range_sl_mult: float = 1.2,
        range_tps: Optional[List[Tuple[float, float]]] = None,
        breakout_sl_mult: float = 1.5,
        breakout_tps: Optional[List[Tuple[float, float]]] = None,
        weak_sl_mult: float = 1.5,
        weak_tps: Optional[List[Tuple[float, float]]] = None,
    ) -> None:
        self.adx_strong = adx_strong
        self.adx_weak = adx_weak
        self.rsi_ob = rsi_overbought
        self.rsi_os = rsi_oversold
        self.trend_sl = trend_sl_mult
        self.trend_tps = trend_tps or [(1.0, 30.0), (2.0, 30.0), (4.0, 40.0)]
        self.range_sl = range_sl_mult
        self.range_tps = range_tps or [(0.8, 50.0), (1.5, 50.0)]
        self.breakout_sl = breakout_sl_mult
        self.breakout_tps = breakout_tps or [(1.0, 40.0), (2.5, 60.0)]
        self.weak_sl = weak_sl_mult
        self.weak_tps = weak_tps or [(1.0, 50.0), (2.0, 50.0)]

    async def evaluate(
        self, symbol: str, df15: pd.DataFrame, df1h: pd.DataFrame
    ) -> Optional[Signal]:
        plan = self.plan(df15, df1h)
        if plan is None:
            return None
        return plan.to_signal()

    def plan(
        self,
        df15: pd.DataFrame,
        df1h: pd.DataFrame,
    ) -> Optional[DirectionalPlan]:
        """Produce a full directional plan or None.

        Two-timeframe gate: the 1h regime must not contradict the 15m trade
        direction. A 15m long counter to a strong 1h downtrend is filtered
        out. Range and breakout signals on 15m bypass this filter when the
        1h is non-trending.
        """
        if df15 is None or len(df15) < 60:
            return None

        regime15 = classify_regime(
            df15, adx_strong=self.adx_strong, adx_weak=self.adx_weak
        )
        if regime15 is None or not regime15.is_tradable():
            return None

        # Higher-timeframe confirmation (directional only).
        regime1h = None
        if df1h is not None and len(df1h) >= 60:
            regime1h = classify_regime(
                df1h, adx_strong=self.adx_strong, adx_weak=self.adx_weak
            )

        # Enriched last bar from 15m for playbook conditions.
        d = df15.copy()
        d["ema20"] = ema(d["close"], 20)
        d["ema50"] = ema(d["close"], 50)
        d["rsi"] = rsi(d["close"], 14)
        d["atr"] = atr(d, 14)
        last = d.iloc[-2]
        if last[["ema20", "ema50", "rsi", "atr"]].isna().any():
            return None

        close = float(last["close"])
        atr_val = float(last["atr"])
        ema20 = float(last["ema20"])
        rsi_val = float(last["rsi"])

        regime = regime15.regime
        # Bollinger bands for range/breakout.
        bb_mid = sma(d["close"], 20).iloc[-2]
        bb_std = d["close"].rolling(20, min_periods=20).std().iloc[-2]
        if pd.isna(bb_mid) or pd.isna(bb_std):
            return None
        bb_low = float(bb_mid - 2.0 * bb_std)
        bb_up = float(bb_mid + 2.0 * bb_std)

        side: Optional[Side] = None
        sl_mult = self.trend_sl
        tps = self.trend_tps
        reason = ""

        if regime == "STRONG_UPTREND":
            if self._htf_blocks(regime1h, "LONG"):
                return None
            # Pullback entry: close above EMA20 after dipping within 1 ATR of it.
            if close > ema20 and (close - ema20) < 1.2 * atr_val and rsi_val < 65:
                side = "LONG"
                sl_mult = self.trend_sl
                tps = self.trend_tps
                reason = f"strong-uptrend pullback rsi={rsi_val:.0f}"

        elif regime == "STRONG_DOWNTREND":
            if self._htf_blocks(regime1h, "SHORT"):
                return None
            if close < ema20 and (ema20 - close) < 1.2 * atr_val and rsi_val > 35:
                side = "SHORT"
                sl_mult = self.trend_sl
                tps = self.trend_tps
                reason = f"strong-downtrend rally rsi={rsi_val:.0f}"

        elif regime == "WEAK_UPTREND":
            if self._htf_blocks(regime1h, "LONG"):
                return None
            if close > ema20 and rsi_val < 55:
                side = "LONG"
                sl_mult = self.weak_sl
                tps = self.weak_tps
                reason = f"weak-uptrend rsi={rsi_val:.0f}"

        elif regime == "WEAK_DOWNTREND":
            if self._htf_blocks(regime1h, "SHORT"):
                return None
            if close < ema20 and rsi_val > 45:
                side = "SHORT"
                sl_mult = self.weak_sl
                tps = self.weak_tps
                reason = f"weak-downtrend rsi={rsi_val:.0f}"

        elif regime == "RANGE":
            # Mean reversion at band edges or RSI extremes.
            if close <= bb_low or rsi_val <= self.rsi_os:
                side = "LONG"
                sl_mult = self.range_sl
                tps = self.range_tps
                reason = f"range-bottom rsi={rsi_val:.0f}"
            elif close >= bb_up or rsi_val >= self.rsi_ob:
                side = "SHORT"
                sl_mult = self.range_sl
                tps = self.range_tps
                reason = f"range-top rsi={rsi_val:.0f}"

        elif regime == "BREAKOUT":
            side = regime15.direction if regime15.direction in ("LONG", "SHORT") else None
            sl_mult = self.breakout_sl
            tps = self.breakout_tps
            reason = f"breakout-{side} bbp={regime15.bb_width_pctile:.2f}"

        if side is None:
            return None

        entry = close
        stop, take_profits = _sl_tp(side, entry, atr_val, sl_mult, tps)

        # Minimum distance sanity: stop must be > 0 and at least 2 ticks away.
        if stop <= 0 or abs(entry - stop) < atr_val * 0.2:
            return None

        return DirectionalPlan(
            regime=regime15,
            side=side,
            entry_price=entry,
            stop_loss=stop,
            take_profits=take_profits,
            atr=atr_val,
            confidence=regime15.confidence,
            reason=f"{regime} {reason}",
        )

    @staticmethod
    def _htf_blocks(htf: Optional[RegimeSnapshot], side: Side) -> bool:
        """Return True if the higher timeframe strongly contradicts `side`."""
        if htf is None:
            return False
        if side == "LONG" and htf.regime == "STRONG_DOWNTREND":
            return True
        if side == "SHORT" and htf.regime == "STRONG_UPTREND":
            return True
        return False


__all__ = ["RegimeAdaptiveStrategy", "DirectionalPlan"]
