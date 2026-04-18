"""Position manager for a single directional long/short trade.

Features
--------
- Market entry with stop-loss pre-computed.
- Tiered take-profits: partial-close at each TP price; remainder rides.
- Breakeven move: after TP1 hits, stop snaps to entry (+/- buffer).
- ATR trailing stop: activated after TP1 (or after `trail_arm_atr` of profit),
  trails the best price by `trail_atr_mult * ATR`.
- Hard time-stop: close if still open after `time_stop_hours`.
- Hard max-loss: belt-and-braces cap on $-loss regardless of price action.

The manager stores one active position keyed by symbol. It is designed to be
driven from the bot's tick loop: call ``on_tick(mark_price)`` on each tick
and react to the returned action (``HOLD`` / ``PARTIAL_CLOSE`` / ``EXIT``).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Awaitable, Callable, List, Literal, Optional, Tuple

from loguru import logger

from ..exchange.base import (
    ExchangeInterface,
    OrderResult,
    OrderSide,
    SymbolFilters,
)


Side = Literal["LONG", "SHORT"]
ExitReason = Literal[
    "STOP_LOSS", "TRAILING_STOP", "TAKE_PROFIT_FINAL",
    "TIME_STOP", "MAX_LOSS", "MANUAL", "REGIME_FLIP",
]
Action = Literal["HOLD", "PARTIAL_CLOSE", "EXIT"]


@dataclass
class TpLevel:
    price: float
    close_pct: float        # 0..100, fraction of ORIGINAL qty to close
    hit: bool = False


@dataclass
class ManagedPosition:
    symbol: str
    side: Side
    entry_price: float
    original_qty: float
    remaining_qty: float
    leverage: int
    stop_loss: float
    take_profits: List[TpLevel]
    atr_at_entry: float
    opened_at: str
    qty_step: float
    price_tick: float
    # Live state
    best_price: float = 0.0              # peak (LONG) or trough (SHORT)
    trailing_armed: bool = False
    breakeven_moved: bool = False
    realised_pnl: float = 0.0
    fees_paid: float = 0.0

    def unrealised_pnl(self, mark: float) -> float:
        if self.side == "LONG":
            return (mark - self.entry_price) * self.remaining_qty
        return (self.entry_price - mark) * self.remaining_qty

    def unrealised_pct(self, mark: float) -> float:
        if self.entry_price <= 0:
            return 0.0
        if self.side == "LONG":
            return (mark - self.entry_price) / self.entry_price * 100.0
        return (self.entry_price - mark) / self.entry_price * 100.0


@dataclass
class TickResult:
    action: Action
    reason: Optional[ExitReason] = None
    close_qty: float = 0.0                  # requested close size (0 for HOLD)
    new_stop: Optional[float] = None        # non-None when stop moved this tick


def _round_step(value: float, step: float) -> float:
    if step <= 0:
        return value
    return math.floor(value / step + 1e-9) * step


def _round_tick(value: float, tick: float) -> float:
    if tick <= 0:
        return value
    return round(round(value / tick) * tick, 12)


class PositionManager:
    """Single-position directional trader.

    Call order:
        await pm.open(...)
        while pm.has_position():
            tick = pm.on_tick(mark_price)
            if tick.action == "PARTIAL_CLOSE":
                await pm.apply_partial(tick.close_qty, mark_price)
            elif tick.action == "EXIT":
                await pm.close(tick.reason, mark_price)
    """

    def __init__(
        self,
        exchange: ExchangeInterface,
        *,
        trail_atr_mult: float = 1.5,
        trail_arm_atr: float = 1.0,           # arm trailing after +1 ATR of profit
        breakeven_buffer_atr: float = 0.1,    # BE stop at entry +/- this*ATR
        time_stop_hours: float = 24.0,
        max_loss_pct: float = 6.0,            # hard cap as % of equity at open
        breakeven_after_tp1: bool = True,
    ) -> None:
        self.ex = exchange
        self.trail_atr_mult = trail_atr_mult
        self.trail_arm_atr = trail_arm_atr
        self.be_buffer_atr = breakeven_buffer_atr
        self.time_stop_hours = time_stop_hours
        self.max_loss_pct = max_loss_pct
        self.breakeven_after_tp1 = breakeven_after_tp1
        self._position: Optional[ManagedPosition] = None
        self._equity_at_open: float = 0.0

    # ------------ lifecycle ------------

    def has_position(self) -> bool:
        return self._position is not None and self._position.remaining_qty > 0

    @property
    def position(self) -> Optional[ManagedPosition]:
        return self._position

    async def open(
        self,
        symbol: str,
        side: Side,
        qty: float,
        entry_price: float,
        stop_loss: float,
        take_profits: List[Tuple[float, float]],
        leverage: int,
        atr_at_entry: float,
        filters: SymbolFilters,
        equity_at_open: float,
        margin_type: str = "ISOLATED",
    ) -> ManagedPosition:
        if self.has_position():
            raise RuntimeError("position already open")

        qty = _round_step(qty, filters.qty_step)
        if qty < filters.min_qty:
            raise ValueError(f"qty {qty} below min_qty {filters.min_qty}")

        try:
            await self.ex.set_margin_type(symbol, margin_type)
        except Exception:
            pass  # most exchanges reject if unchanged; safe to ignore
        try:
            await self.ex.set_leverage(symbol, leverage)
        except Exception as e:
            logger.warning("set_leverage({}, {}) failed: {}", symbol, leverage, e)

        order_side: OrderSide = "BUY" if side == "LONG" else "SELL"
        result = await self.ex.market_open(symbol, order_side, qty)
        fill_price = result.avg_price or entry_price

        tps = [
            TpLevel(price=_round_tick(p, filters.price_tick), close_pct=pct)
            for p, pct in take_profits
        ]
        stop = _round_tick(stop_loss, filters.price_tick)

        pos = ManagedPosition(
            symbol=symbol,
            side=side,
            entry_price=fill_price,
            original_qty=qty,
            remaining_qty=qty,
            leverage=leverage,
            stop_loss=stop,
            take_profits=tps,
            atr_at_entry=atr_at_entry,
            opened_at=datetime.now(timezone.utc).isoformat(),
            qty_step=filters.qty_step,
            price_tick=filters.price_tick,
            best_price=fill_price,
            fees_paid=result.fee,
        )
        self._position = pos
        self._equity_at_open = equity_at_open
        logger.info(
            "Opened {} {} {:.8f} @ {:.6f} lev={}x SL={:.6f} TPs={}",
            side, symbol, qty, fill_price, leverage, stop,
            [(round(tp.price, 6), tp.close_pct) for tp in tps],
        )
        return pos

    async def apply_partial(self, qty: float, mark: float) -> OrderResult:
        """Close ``qty`` of the current position at market."""
        pos = self._position
        if pos is None or qty <= 0:
            raise RuntimeError("no position / zero qty")
        qty = min(qty, pos.remaining_qty)
        qty = _round_step(qty, pos.qty_step)
        if qty < pos.qty_step:
            return OrderResult("noop", pos.symbol, "BUY", 0.0, mark, 0.0, "SKIPPED")
        close_side: OrderSide = "SELL" if pos.side == "LONG" else "BUY"
        result = await self.ex.market_close(pos.symbol, close_side, qty)
        filled = result.avg_price or mark
        pnl = (
            (filled - pos.entry_price) * qty
            if pos.side == "LONG"
            else (pos.entry_price - filled) * qty
        )
        pos.realised_pnl += pnl
        pos.fees_paid += result.fee
        pos.remaining_qty = _round_step(pos.remaining_qty - qty, pos.qty_step)
        logger.info(
            "Partial close {} {:.8f} @ {:.6f} pnl={:+.4f} remaining={:.8f}",
            pos.symbol, qty, filled, pnl, pos.remaining_qty,
        )
        if pos.remaining_qty <= 0:
            self._position = None
        return result

    async def close(self, reason: ExitReason, mark: float) -> OrderResult:
        """Close the entire remaining position."""
        pos = self._position
        if pos is None:
            raise RuntimeError("no position")
        qty = pos.remaining_qty
        result = await self.apply_partial(qty, mark)
        logger.info("Closed {} ({}) total_pnl={:+.4f}", pos.symbol, reason,
                    pos.realised_pnl)
        self._position = None
        return result

    # ------------ per-tick logic ------------

    def on_tick(self, mark: float) -> TickResult:
        """Run the management rules for this tick and return the action.

        The bot is expected to perform the actual orders based on the result
        using ``apply_partial`` / ``close``.
        """
        pos = self._position
        if pos is None or pos.remaining_qty <= 0:
            return TickResult("HOLD")

        # Track best excursion.
        if pos.side == "LONG":
            if mark > pos.best_price:
                pos.best_price = mark
        else:
            if mark < pos.best_price or pos.best_price == 0:
                pos.best_price = mark

        # 1) Hard stop-loss.
        if pos.side == "LONG" and mark <= pos.stop_loss:
            return TickResult("EXIT", "STOP_LOSS", pos.remaining_qty)
        if pos.side == "SHORT" and mark >= pos.stop_loss:
            return TickResult("EXIT", "STOP_LOSS", pos.remaining_qty)

        # 2) Hard max-loss guard (belt-and-braces).
        if self._equity_at_open > 0 and self.max_loss_pct > 0:
            upnl = pos.unrealised_pnl(mark)
            if upnl < 0 and abs(upnl) >= self._equity_at_open * (self.max_loss_pct / 100.0):
                return TickResult("EXIT", "MAX_LOSS", pos.remaining_qty)

        # 3) Time stop.
        if self.time_stop_hours > 0:
            opened = datetime.fromisoformat(pos.opened_at)
            if datetime.now(timezone.utc) - opened > _hours(self.time_stop_hours):
                return TickResult("EXIT", "TIME_STOP", pos.remaining_qty)

        # 4) Take-profits (first unhit tp whose price is reached).
        for i, tp in enumerate(pos.take_profits):
            if tp.hit:
                continue
            reached = mark >= tp.price if pos.side == "LONG" else mark <= tp.price
            if not reached:
                continue
            tp.hit = True
            # Close requested fraction of ORIGINAL qty.
            close_qty = _round_step(
                pos.original_qty * (tp.close_pct / 100.0), pos.qty_step
            )
            close_qty = min(close_qty, pos.remaining_qty)
            is_final = i == len(pos.take_profits) - 1
            # Move SL to breakeven after first TP.
            new_stop: Optional[float] = None
            if i == 0 and self.breakeven_after_tp1 and not pos.breakeven_moved:
                new_stop = self._breakeven_stop(pos)
                if new_stop is not None:
                    pos.stop_loss = new_stop
                    pos.breakeven_moved = True
                    pos.trailing_armed = True
            if is_final or close_qty >= pos.remaining_qty:
                return TickResult("EXIT", "TAKE_PROFIT_FINAL",
                                  pos.remaining_qty, new_stop=new_stop)
            return TickResult("PARTIAL_CLOSE", None, close_qty, new_stop=new_stop)

        # 5) Arm trailing stop once we're +trail_arm_atr in profit.
        if not pos.trailing_armed:
            if pos.side == "LONG":
                if mark - pos.entry_price >= self.trail_arm_atr * pos.atr_at_entry:
                    pos.trailing_armed = True
            else:
                if pos.entry_price - mark >= self.trail_arm_atr * pos.atr_at_entry:
                    pos.trailing_armed = True

        # 6) Update trailing stop.
        if pos.trailing_armed:
            new_stop = self._trailing_stop(pos)
            if new_stop is not None:
                # Only ever tighten — never loosen.
                if pos.side == "LONG" and new_stop > pos.stop_loss:
                    pos.stop_loss = new_stop
                    return TickResult("HOLD", new_stop=new_stop)
                if pos.side == "SHORT" and new_stop < pos.stop_loss:
                    pos.stop_loss = new_stop
                    return TickResult("HOLD", new_stop=new_stop)

        return TickResult("HOLD")

    # ------------ helpers ------------

    def _breakeven_stop(self, pos: ManagedPosition) -> Optional[float]:
        buffer = self.be_buffer_atr * pos.atr_at_entry
        if pos.side == "LONG":
            price = pos.entry_price + buffer
        else:
            price = pos.entry_price - buffer
        return _round_tick(price, pos.price_tick)

    def _trailing_stop(self, pos: ManagedPosition) -> Optional[float]:
        dist = self.trail_atr_mult * pos.atr_at_entry
        if pos.side == "LONG":
            return _round_tick(pos.best_price - dist, pos.price_tick)
        return _round_tick(pos.best_price + dist, pos.price_tick)


def _hours(h: float):
    from datetime import timedelta
    return timedelta(seconds=h * 3600)


__all__ = ["PositionManager", "ManagedPosition", "TpLevel", "TickResult",
           "ExitReason", "Action"]
