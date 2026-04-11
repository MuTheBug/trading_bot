"""Risk manager: SL/TP ladder, trailing stops, circuit breakers, funding guard."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Tuple

from loguru import logger

from ..config import ExitsConfig, RiskConfig
from ..state import Position, StateStore, TakeProfitLevel


@dataclass
class ExitDecision:
    should_close: bool
    close_qty_pct: float          # 0..100
    reason: str                   # SL | TP1 | TP2 | TP3 | TRAIL | TIME | FUNDING
    new_stop_loss: Optional[float] = None  # set if SL should move (BE / trailing)
    enable_trailing: bool = False


class RiskManager:
    def __init__(self, risk_cfg: RiskConfig, exits_cfg: ExitsConfig) -> None:
        self.risk = risk_cfg
        self.exits = exits_cfg

    # ---------- pre-trade gating ----------

    def can_open_new(
        self,
        state: StateStore,
        equity: float,
        symbol: str,
    ) -> Tuple[bool, str]:
        if state.state.paused:
            return False, "bot paused"

        if len(state.state.positions) >= self.risk.max_concurrent_positions:
            return False, "max concurrent positions reached"

        if symbol in state.state.positions:
            return False, "already in position"

        if state.is_in_cooldown(symbol):
            return False, "symbol in cooldown after loss"

        # Daily loss circuit breaker
        if equity > 0:
            daily_loss_pct = (state.state.daily.realized_pnl / equity) * 100.0
            if daily_loss_pct <= -self.risk.daily_loss_limit_pct:
                return False, f"daily loss limit hit ({daily_loss_pct:.2f}%)"

        # Max drawdown from peak equity
        peak = max(state.state.peak_equity, equity)
        if peak > 0:
            dd_pct = (peak - equity) / peak * 100.0
            if dd_pct >= self.risk.max_drawdown_pct:
                return False, f"max drawdown hit ({dd_pct:.2f}%)"

        return True, ""

    def funding_action(self, funding_pct_annualized: float, side: str) -> str:
        """Return 'skip' / 'exit' / 'ok' based on annualized funding rate."""
        abs_funding = abs(funding_pct_annualized)
        if abs_funding >= self.risk.funding_rate_exit_pct:
            # Exit only if we're on the paying side (positive funding -> longs pay shorts)
            if (funding_pct_annualized > 0 and side == "LONG") or (
                funding_pct_annualized < 0 and side == "SHORT"
            ):
                return "exit"
        if abs_funding >= self.risk.funding_rate_skip_pct:
            return "skip"
        return "ok"

    # ---------- exit planning at entry ----------

    def build_exit_ladder(
        self, entry: float, side: str, atr: float
    ) -> Tuple[float, List[TakeProfitLevel]]:
        """Return (initial_sl, [tp1, tp2, tp3])."""
        direction = 1 if side == "LONG" else -1
        sl = entry - direction * self.exits.sl_atr_mult * atr
        tp1 = entry + direction * self.exits.tp1_atr_mult * atr
        tp2 = entry + direction * self.exits.tp2_atr_mult * atr
        tp3 = entry + direction * self.exits.tp3_atr_mult * atr
        tp_remaining = 100.0 - self.exits.tp1_close_pct - self.exits.tp2_close_pct
        return sl, [
            TakeProfitLevel(price=tp1, close_pct=self.exits.tp1_close_pct),
            TakeProfitLevel(price=tp2, close_pct=self.exits.tp2_close_pct),
            TakeProfitLevel(price=tp3, close_pct=max(tp_remaining, 0.0)),
        ]

    # ---------- per-candle management ----------

    def manage_position(
        self, pos: Position, mark_price: float, now_utc: Optional[datetime] = None
    ) -> Optional[ExitDecision]:
        """Evaluate an open position against the current mark price.

        Returns an ExitDecision when something should happen, else None.
        The caller applies the close to the exchange and updates state.
        Positions are managed one-step-at-a-time: on each call at most one
        TP level or stop event is acted on.
        """
        now = now_utc or datetime.now(timezone.utc)

        # Time stop
        try:
            opened = datetime.fromisoformat(pos.opened_at)
            if now - opened >= timedelta(hours=self.risk.time_stop_hours):
                return ExitDecision(True, 100.0, "TIME")
        except ValueError:
            pass

        # Track extremes for trailing stop
        if pos.side == "LONG":
            pos.highest_since_entry = max(pos.highest_since_entry or mark_price, mark_price)
        else:
            pos.lowest_since_entry = min(pos.lowest_since_entry or mark_price, mark_price)

        # Stop loss hit?
        if pos.side == "LONG" and mark_price <= pos.stop_loss:
            return ExitDecision(True, 100.0, "SL")
        if pos.side == "SHORT" and mark_price >= pos.stop_loss:
            return ExitDecision(True, 100.0, "SL")

        # TP1
        tp1 = pos.take_profits[0]
        if not tp1.hit:
            hit = mark_price >= tp1.price if pos.side == "LONG" else mark_price <= tp1.price
            if hit:
                tp1.hit = True
                pos.tp1_hit = True
                new_sl = pos.entry_price if self.exits.move_sl_to_be_after_tp1 else None
                return ExitDecision(
                    should_close=True,
                    close_qty_pct=tp1.close_pct,
                    reason="TP1",
                    new_stop_loss=new_sl,
                )

        # TP2
        tp2 = pos.take_profits[1]
        if not tp2.hit and tp1.hit:
            hit = mark_price >= tp2.price if pos.side == "LONG" else mark_price <= tp2.price
            if hit:
                tp2.hit = True
                pos.tp2_hit = True
                return ExitDecision(
                    should_close=True,
                    close_qty_pct=tp2.close_pct,
                    reason="TP2",
                    enable_trailing=True,
                )

        # TP3
        tp3 = pos.take_profits[2]
        if not tp3.hit and tp2.hit and tp3.close_pct > 0:
            hit = mark_price >= tp3.price if pos.side == "LONG" else mark_price <= tp3.price
            if hit:
                tp3.hit = True
                return ExitDecision(True, 100.0, "TP3")

        # Trailing stop (after trailing activated)
        if pos.trailing_active:
            trail_dist = self.exits.trail_atr_mult * pos.atr_at_entry
            if pos.side == "LONG":
                trail_sl = pos.highest_since_entry - trail_dist
                if trail_sl > pos.stop_loss:
                    pos.stop_loss = trail_sl
                if mark_price <= pos.stop_loss:
                    return ExitDecision(True, 100.0, "TRAIL")
            else:
                trail_sl = pos.lowest_since_entry + trail_dist
                if trail_sl < pos.stop_loss:
                    pos.stop_loss = trail_sl
                if mark_price >= pos.stop_loss:
                    return ExitDecision(True, 100.0, "TRAIL")

        return None


__all__ = ["RiskManager", "ExitDecision"]
