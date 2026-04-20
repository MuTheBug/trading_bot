"""Deterministic Support/Resistance directional strategy.

The strategy replaces the old AI-driven directional decision with a
reproducible rule-based system:

1. Pivot detection on each configured timeframe (fractal-style swings).
2. Cluster pivots whose prices sit within `tolerance * ATR` into
   Levels — each Level remembers how many times price has touched it,
   when it was first/last touched, and its "strength".
3. Classify HTF bias (uptrend / downtrend / neutral) from EMA stack.
4. For each symbol, look for a tradable setup:
   - LONG: HTF not-bearish + price within `entry_zone_atr` of a strong
     support + bullish rejection candle. SL below the support minus a
     buffer, TP ladder toward the next resistances.
   - SHORT: inverse.
5. Require minimum reward:risk (to the final TP) and minimum level
   strength; otherwise SKIP.

The strategy returns an `SRDecision` that mirrors the old `AIDecision`
interface so the trader, position manager, telegram notifier and tests
all continue to work unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..config import DirectionalConfig, SRConfig
from ..exchange.base import SymbolFilters, TickerInfo
from ..indicators import atr as atr_ind, ema, rsi as rsi_ind


# ---------------- data types ----------------


@dataclass
class Pivot:
    bar: int
    price: float
    kind: str  # "high" or "low"


@dataclass
class Level:
    """A clustered price level built from one or more swing pivots."""
    price: float
    touches: int
    first_bar: int        # index of earliest touch
    last_bar: int         # index of most recent touch
    kind: str             # "support" (below current price) or "resistance" (above)
    strength: float       # composite score

    def age_bars(self, current_bar: int) -> int:
        return max(0, current_bar - self.last_bar)


@dataclass
class LevelSet:
    supports: List[Level]       # sorted by price ascending
    resistances: List[Level]    # sorted by price ascending
    atr: float
    last_price: float
    last_bar: int

    def nearest_support(self, price: float) -> Optional[Level]:
        below = [lv for lv in self.supports if lv.price < price]
        return max(below, key=lambda lv: lv.price) if below else None

    def nearest_resistance(self, price: float) -> Optional[Level]:
        above = [lv for lv in self.resistances if lv.price > price]
        return min(above, key=lambda lv: lv.price) if above else None


@dataclass
class SRDecision:
    """Return type matching the old AIDecision interface.

    The trader inspects `.is_trade`, `.side`, `.entry`, `.stop_loss`,
    `.take_profits`, `.confidence`, `.reasoning` — all identical fields.
    """
    action: str              # OPEN_LONG / OPEN_SHORT / SKIP
    reasoning: str
    symbol: Optional[str] = None
    entry: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profits: List[Tuple[float, float]] = field(default_factory=list)
    leverage: int = 0
    confidence: float = 0.0
    reference_level: Optional[float] = None

    @property
    def is_trade(self) -> bool:
        return self.action in ("OPEN_LONG", "OPEN_SHORT")

    @property
    def side(self) -> Optional[str]:
        if self.action == "OPEN_LONG":
            return "LONG"
        if self.action == "OPEN_SHORT":
            return "SHORT"
        return None


@dataclass
class CandidateCtx:
    """Per-symbol context: ticker, filters, and MTF OHLCV frames.

    Kept as a plain carrier so tests and the trader share the same
    structure. `dfs` preserves insertion order (top-down MTF).
    """
    symbol: str
    ticker: TickerInfo
    filters: SymbolFilters
    dfs: Dict[str, pd.DataFrame] = field(default_factory=dict)


# ---------------- pivot detection ----------------


def find_pivots(df: pd.DataFrame, left: int = 3, right: int = 3) -> List[Pivot]:
    """Classic fractal pivot detection.

    A bar at index `i` is a swing HIGH if its high is strictly greater
    than the highs of the `left` bars before it and greater-than-or-equal
    to the highs of the `right` bars after. Swing LOW is the inverse.

    The right-side comparison uses `>=` / `<=` so that flat tops/bottoms
    still register (otherwise a double-top bar against an identical
    follower would be lost).
    """
    if df is None or len(df) < left + right + 1:
        return []
    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()
    n = len(df)
    pivots: List[Pivot] = []
    for i in range(left, n - right):
        h = highs[i]
        if (
            all(h > highs[j] for j in range(i - left, i))
            and all(h >= highs[j] for j in range(i + 1, i + right + 1))
        ):
            pivots.append(Pivot(bar=i, price=float(h), kind="high"))
            continue
        l = lows[i]
        if (
            all(l < lows[j] for j in range(i - left, i))
            and all(l <= lows[j] for j in range(i + 1, i + right + 1))
        ):
            pivots.append(Pivot(bar=i, price=float(l), kind="low"))
    return pivots


# ---------------- clustering ----------------


def cluster_levels(
    pivots: List[Pivot],
    atr_value: float,
    current_price: float,
    current_bar: int,
    tolerance_atr: float = 0.3,
    min_touches: int = 2,
) -> LevelSet:
    """Group nearby pivots into clustered levels.

    Two pivots belong to the same level when their prices differ by less
    than `tolerance_atr * ATR`. A cluster's price is the mean of its
    pivot prices; its `touches` count is the pivot count; its age is
    measured against the current bar.

    Levels are split into supports (price below current) and resistances
    (above). A cluster can include BOTH swing highs and swing lows —
    that's a classic "flipped" level, and it counts as either support or
    resistance depending on where price is right now.
    """
    if not pivots or atr_value <= 0:
        return LevelSet([], [], atr_value, current_price, current_bar)

    tol = tolerance_atr * atr_value
    # Sort pivots by price so we can sweep and group neighbours.
    sorted_pivots = sorted(pivots, key=lambda p: p.price)

    clusters: List[List[Pivot]] = []
    current: List[Pivot] = [sorted_pivots[0]]
    for p in sorted_pivots[1:]:
        # Compare to the cluster's current mean price so a long run of
        # slightly-drifting pivots doesn't all collapse into one.
        cluster_mean = sum(x.price for x in current) / len(current)
        if abs(p.price - cluster_mean) <= tol:
            current.append(p)
        else:
            clusters.append(current)
            current = [p]
    clusters.append(current)

    levels: List[Level] = []
    for group in clusters:
        touches = len(group)
        if touches < min_touches:
            continue
        price = float(np.mean([x.price for x in group]))
        bars = [x.bar for x in group]
        first_bar = min(bars)
        last_bar = max(bars)
        kind = "support" if price <= current_price else "resistance"
        # Strength: more touches + recent + long-held -> stronger.
        age = max(0, current_bar - last_bar)
        held = max(1, last_bar - first_bar)
        recency = max(0.0, 1.0 - min(1.0, age / 100.0))
        strength = touches * (0.6 + 0.4 * recency) * (1.0 + min(held / 100.0, 1.0))
        levels.append(Level(
            price=price, touches=touches, first_bar=first_bar,
            last_bar=last_bar, kind=kind, strength=strength,
        ))

    supports = sorted([lv for lv in levels if lv.kind == "support"],
                     key=lambda lv: lv.price)
    resistances = sorted([lv for lv in levels if lv.kind == "resistance"],
                         key=lambda lv: lv.price)
    return LevelSet(supports, resistances, atr_value, current_price, current_bar)


def detect_levels(
    df: pd.DataFrame,
    cfg: SRConfig,
) -> Optional[LevelSet]:
    """End-to-end: pivots -> clustered levels for a single timeframe."""
    if df is None or len(df) < cfg.pivot_left + cfg.pivot_right + 20:
        return None
    # Keep only the last `max_lookback_bars` closed candles so old
    # levels that have been violated don't dominate.
    use = df.iloc[-cfg.max_lookback_bars:] if len(df) > cfg.max_lookback_bars else df
    use = use.reset_index(drop=True)
    # Work off closed bars; drop the last (potentially forming) bar.
    closed = use.iloc[:-1] if len(use) > 1 else use
    atr_series = atr_ind(closed, 14)
    if atr_series.isna().iloc[-1]:
        return None
    atr_value = float(atr_series.iloc[-1])
    last_price = float(closed["close"].iloc[-1])
    last_bar = len(closed) - 1
    pivots = find_pivots(closed, cfg.pivot_left, cfg.pivot_right)
    return cluster_levels(
        pivots, atr_value, last_price, last_bar,
        tolerance_atr=cfg.level_tolerance_atr,
        min_touches=cfg.min_touches,
    )


# ---------------- HTF bias ----------------


def classify_bias(df: pd.DataFrame, ema_period: int = 50) -> str:
    """Return 'up' / 'down' / 'neutral' for the HTF using an EMA stack.

    Rule: price above a rising EMA(period) by at least ~0.3% -> up.
    Price below a falling EMA by >=0.3% -> down. Otherwise neutral.
    The 0.3% buffer keeps us out of sideways markets.
    """
    if df is None or len(df) < ema_period + 5:
        return "neutral"
    close = df["close"]
    e = ema(close, ema_period)
    if e.isna().iloc[-1]:
        return "neutral"
    last_close = float(close.iloc[-1])
    last_e = float(e.iloc[-1])
    past_e = float(e.iloc[-6]) if len(e) >= 6 else last_e
    slope = (last_e - past_e) / last_e if last_e > 0 else 0.0
    dist = (last_close - last_e) / last_e if last_e > 0 else 0.0
    if dist > 0.003 and slope > -0.0005:
        return "up"
    if dist < -0.003 and slope < 0.0005:
        return "down"
    return "neutral"


# ---------------- candle patterns ----------------


def bullish_rejection(bar: pd.Series) -> bool:
    """Last closed bar shows a lower-wick rejection and closes near the high."""
    o, h, l, c = (float(bar["open"]), float(bar["high"]),
                  float(bar["low"]), float(bar["close"]))
    rng = max(h - l, 1e-12)
    body = abs(c - o)
    lower_wick = min(o, c) - l
    # Close in upper 40% of the range AND lower wick >= 50% of the range.
    close_pos = (c - l) / rng
    return close_pos >= 0.6 and lower_wick >= 0.5 * rng and body <= 0.6 * rng


def bearish_rejection(bar: pd.Series) -> bool:
    """Last closed bar shows an upper-wick rejection and closes near the low."""
    o, h, l, c = (float(bar["open"]), float(bar["high"]),
                  float(bar["low"]), float(bar["close"]))
    rng = max(h - l, 1e-12)
    body = abs(c - o)
    upper_wick = h - max(o, c)
    close_pos = (c - l) / rng
    return close_pos <= 0.4 and upper_wick >= 0.5 * rng and body <= 0.6 * rng


# ---------------- setup proposal ----------------


class SRStrategy:
    """Produce an SRDecision for a single candidate.

    The trader calls `propose(ctx)` once per candidate. No I/O, no LLM —
    pure math over the OHLCV frames in `ctx.dfs`.
    """

    def __init__(self, sr_cfg: SRConfig, dir_cfg: DirectionalConfig) -> None:
        self.sr = sr_cfg
        self.dir = dir_cfg

    def propose(self, ctx: CandidateCtx) -> SRDecision:
        if not ctx.dfs:
            return SRDecision(action="SKIP", reasoning="no MTF data")
        tfs = list(ctx.dfs.keys())
        ltf_name = tfs[-1]
        htf_name = tfs[0]
        ltf = ctx.dfs[ltf_name]
        htf = ctx.dfs[htf_name]

        levels = detect_levels(ltf, self.sr)
        if levels is None:
            return SRDecision(
                action="SKIP",
                reasoning=f"insufficient data on {ltf_name} for level detection",
            )
        bias = classify_bias(htf, self.sr.htf_trend_ema)

        closed = ltf.iloc[:-1] if len(ltf) > 1 else ltf
        last_bar = closed.iloc[-1]
        price = float(last_bar["close"])
        atr_val = levels.atr
        if atr_val <= 0 or price <= 0:
            return SRDecision(action="SKIP", reasoning="non-positive atr or price")

        rsi_s = rsi_ind(closed["close"], 14)
        last_rsi = float(rsi_s.iloc[-1]) if not rsi_s.isna().iloc[-1] else 50.0

        # Try LONG first (support bounce).
        long_sig = self._try_long(
            ctx, levels, bias, last_bar, price, atr_val, last_rsi,
        )
        if long_sig.is_trade:
            return long_sig

        # Then SHORT (resistance fail).
        short_sig = self._try_short(
            ctx, levels, bias, last_bar, price, atr_val, last_rsi,
        )
        if short_sig.is_trade:
            return short_sig

        # Neither side qualified — surface the most informative skip reason.
        reason = long_sig.reasoning or short_sig.reasoning or "no setup"
        return SRDecision(action="SKIP", reasoning=reason)

    # ---- LONG branch ----
    def _try_long(
        self,
        ctx: CandidateCtx,
        levels: LevelSet,
        bias: str,
        last_bar: pd.Series,
        price: float,
        atr_val: float,
        last_rsi: float,
    ) -> SRDecision:
        if bias == "down":
            return SRDecision(action="SKIP",
                             reasoning="HTF bias down — no long setup")
        support = levels.nearest_support(price)
        if support is None:
            return SRDecision(action="SKIP",
                             reasoning="no support below current price")
        if support.strength < self.sr.min_level_strength:
            return SRDecision(
                action="SKIP",
                reasoning=f"support {support.price:.6f} too weak "
                          f"(strength {support.strength:.2f})",
            )
        # Distance to the level in ATR units. Must be close (in the zone)
        # but not already blown through (don't long below the level).
        dist_atr = (price - support.price) / atr_val
        if dist_atr < 0 or dist_atr > self.sr.entry_zone_atr:
            return SRDecision(
                action="SKIP",
                reasoning=f"price {price:.6f} not in long zone of "
                          f"support {support.price:.6f} "
                          f"({dist_atr:+.2f} ATR)",
            )
        if self.sr.require_rejection_candle and not bullish_rejection(last_bar):
            return SRDecision(
                action="SKIP",
                reasoning=f"no bullish rejection candle at support "
                          f"{support.price:.6f}",
            )
        if last_rsi > self.sr.max_rsi_for_long:
            return SRDecision(
                action="SKIP",
                reasoning=f"RSI {last_rsi:.1f} too hot for long bounce",
            )

        entry = price
        stop_loss = support.price - self.sr.sl_buffer_atr * atr_val
        risk = entry - stop_loss
        if risk <= 0:
            return SRDecision(action="SKIP", reasoning="non-positive risk for long")

        resistances = [lv for lv in levels.resistances if lv.price > entry]
        if not resistances:
            return SRDecision(
                action="SKIP",
                reasoning="no resistance above entry for long TP",
            )
        # TP ladder: up to three closest resistances.
        targets = resistances[: self.sr.max_tp_levels]
        tps = self._build_tp_ladder(
            entry=entry, risk=risk, targets=[lv.price for lv in targets],
            side="LONG",
        )
        if not tps:
            return SRDecision(
                action="SKIP",
                reasoning="TP ladder fails min R:R for long",
            )

        conf = self._confidence(support.strength, bias, "LONG", last_rsi, dist_atr)
        reasoning = (
            f"LONG bounce off support {support.price:.6f} "
            f"(strength {support.strength:.2f}, touches {support.touches}), "
            f"HTF {bias}, rsi {last_rsi:.1f}, R:R "
            f"to final TP {(tps[-1][0] - entry) / risk:.2f}"
        )
        return SRDecision(
            action="OPEN_LONG", symbol=ctx.symbol, entry=entry,
            stop_loss=stop_loss, take_profits=tps, confidence=conf,
            reasoning=reasoning, reference_level=support.price,
        )

    # ---- SHORT branch ----
    def _try_short(
        self,
        ctx: CandidateCtx,
        levels: LevelSet,
        bias: str,
        last_bar: pd.Series,
        price: float,
        atr_val: float,
        last_rsi: float,
    ) -> SRDecision:
        if bias == "up":
            return SRDecision(action="SKIP",
                             reasoning="HTF bias up — no short setup")
        resistance = levels.nearest_resistance(price)
        if resistance is None:
            return SRDecision(action="SKIP",
                             reasoning="no resistance above current price")
        if resistance.strength < self.sr.min_level_strength:
            return SRDecision(
                action="SKIP",
                reasoning=f"resistance {resistance.price:.6f} too weak "
                          f"(strength {resistance.strength:.2f})",
            )
        dist_atr = (resistance.price - price) / atr_val
        if dist_atr < 0 or dist_atr > self.sr.entry_zone_atr:
            return SRDecision(
                action="SKIP",
                reasoning=f"price {price:.6f} not in short zone of "
                          f"resistance {resistance.price:.6f} "
                          f"({dist_atr:+.2f} ATR)",
            )
        if self.sr.require_rejection_candle and not bearish_rejection(last_bar):
            return SRDecision(
                action="SKIP",
                reasoning=f"no bearish rejection candle at resistance "
                          f"{resistance.price:.6f}",
            )
        if last_rsi < self.sr.min_rsi_for_short:
            return SRDecision(
                action="SKIP",
                reasoning=f"RSI {last_rsi:.1f} too cold for short bounce",
            )

        entry = price
        stop_loss = resistance.price + self.sr.sl_buffer_atr * atr_val
        risk = stop_loss - entry
        if risk <= 0:
            return SRDecision(action="SKIP", reasoning="non-positive risk for short")

        supports_below = [lv for lv in levels.supports if lv.price < entry]
        if not supports_below:
            return SRDecision(
                action="SKIP",
                reasoning="no support below entry for short TP",
            )
        # Descending: nearest below first, furthest last.
        supports_below.sort(key=lambda lv: lv.price, reverse=True)
        targets = supports_below[: self.sr.max_tp_levels]
        tps = self._build_tp_ladder(
            entry=entry, risk=risk, targets=[lv.price for lv in targets],
            side="SHORT",
        )
        if not tps:
            return SRDecision(
                action="SKIP",
                reasoning="TP ladder fails min R:R for short",
            )

        conf = self._confidence(
            resistance.strength, bias, "SHORT", last_rsi, dist_atr,
        )
        reasoning = (
            f"SHORT bounce off resistance {resistance.price:.6f} "
            f"(strength {resistance.strength:.2f}, touches {resistance.touches}), "
            f"HTF {bias}, rsi {last_rsi:.1f}, R:R "
            f"to final TP {(entry - tps[-1][0]) / risk:.2f}"
        )
        return SRDecision(
            action="OPEN_SHORT", symbol=ctx.symbol, entry=entry,
            stop_loss=stop_loss, take_profits=tps, confidence=conf,
            reasoning=reasoning, reference_level=resistance.price,
        )

    # ---- helpers ----
    def _build_tp_ladder(
        self,
        entry: float,
        risk: float,
        targets: List[float],
        side: str,
    ) -> List[Tuple[float, float]]:
        """Split close_pct across up to N targets; require min R:R on the
        FINAL target. Returns [(price, close_pct), ...] summing to 100."""
        if not targets or risk <= 0:
            return []
        # Keep only targets on the correct side of entry.
        if side == "LONG":
            valid = [t for t in targets if t > entry]
            if not valid:
                return []
            final_rr = (max(valid) - entry) / risk
        else:
            valid = [t for t in targets if t < entry]
            if not valid:
                return []
            final_rr = (entry - min(valid)) / risk
        if final_rr < self.sr.min_rr:
            return []
        # Allocation: front-load the first TP a bit so we lock in quickly.
        n = len(valid)
        if n == 1:
            weights = [100.0]
        elif n == 2:
            weights = [50.0, 50.0]
        else:
            weights = [40.0, 35.0, 25.0][:n]
            # Normalise if fewer than 3 stored.
            total = sum(weights)
            weights = [w * 100.0 / total for w in weights]
        # Sort targets away from entry.
        if side == "LONG":
            valid.sort()
        else:
            valid.sort(reverse=True)
        return [(float(p), float(w)) for p, w in zip(valid, weights)]

    def _confidence(
        self,
        level_strength: float,
        bias: str,
        side: str,
        rsi_val: float,
        dist_atr: float,
    ) -> float:
        """Compose a 0..1 confidence from the setup's quality.

        Heavier-weighted inputs: level strength and HTF alignment. The
        RSI and zone-distance terms provide small bonuses.
        """
        base = min(0.8, 0.35 + 0.08 * level_strength)
        if (side == "LONG" and bias == "up") or (side == "SHORT" and bias == "down"):
            base += 0.1
        # Closer to the level is better (smaller dist_atr -> bonus).
        base += max(0.0, 0.05 * (1.0 - dist_atr / max(self.sr.entry_zone_atr, 1e-6)))
        # RSI comfort zone.
        if side == "LONG" and 35.0 <= rsi_val <= 55.0:
            base += 0.03
        if side == "SHORT" and 45.0 <= rsi_val <= 65.0:
            base += 0.03
        return float(max(0.0, min(1.0, base)))


__all__ = [
    "Pivot",
    "Level",
    "LevelSet",
    "SRDecision",
    "CandidateCtx",
    "SRStrategy",
    "find_pivots",
    "cluster_levels",
    "detect_levels",
    "classify_bias",
    "bullish_rejection",
    "bearish_rejection",
]
