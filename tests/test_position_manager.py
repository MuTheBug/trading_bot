"""Position manager tests — stub exchange, deterministic price feed."""
import asyncio
from dataclasses import dataclass, field
from typing import List

import pytest

from src.exchange.base import (
    ExchangeInterface, LimitOrder, LivePosition, OrderResult, OrderSide,
    SymbolFilters, TickerInfo,
)
from src.position.manager import PositionManager


class StubExchange(ExchangeInterface):
    """Minimal in-memory exchange for position-manager tests."""
    mode = "sim"

    def __init__(self):
        self.opens: List[tuple] = []
        self.closes: List[tuple] = []
        self.fills_at = None

    async def connect(self): pass
    async def close(self): pass
    async def get_balance(self): return 100.0
    async def get_klines(self, *a, **kw): return None
    async def get_mark_price(self, symbol): return 100.0
    async def get_funding_rate(self, symbol): return 0.0
    async def get_symbol_filters(self, symbol): return None
    async def set_leverage(self, symbol, leverage): pass
    async def set_margin_type(self, symbol, margin_type): pass
    async def market_open(self, symbol, side, qty):
        self.opens.append((symbol, side, qty))
        return OrderResult("o1", symbol, side, qty,
                           self.fills_at or 100.0, 0.01, "FILLED")
    async def market_close(self, symbol, side, qty):
        self.closes.append((symbol, side, qty))
        return OrderResult("c1", symbol, side, qty,
                           self.fills_at or 100.0, 0.01, "FILLED")
    async def get_open_positions(self): return []
    async def limit_order(self, *a, **kw): return "nop"
    async def get_open_orders(self, symbol): return []
    async def cancel_order(self, symbol, order_id): return True
    async def cancel_all_orders(self, symbol): return 0
    async def get_all_tickers(self): return []
    async def get_all_symbol_filters(self): return {}


FILTERS = SymbolFilters("X", price_tick=0.01, qty_step=0.001,
                        min_qty=0.001, min_notional=5.0)


def run(coro):
    return asyncio.get_event_loop().run_until_complete(coro) if False \
        else asyncio.run(coro)


async def _open(pm, ex, side="LONG", entry=100.0, sl=95.0, tps=None):
    ex.fills_at = entry
    tps = tps or [(102.0, 50.0), (110.0, 50.0)]
    return await pm.open(
        symbol="XUSDT", side=side, qty=1.0,
        entry_price=entry, stop_loss=sl,
        take_profits=tps, leverage=5, atr_at_entry=2.0,
        filters=FILTERS, equity_at_open=100.0,
    )


def test_stop_loss_triggers_exit():
    async def go():
        ex = StubExchange()
        pm = PositionManager(ex)
        await _open(pm, ex)
        res = pm.on_tick(95.0)
        assert res.action == "EXIT"
        assert res.reason == "STOP_LOSS"
    run(go())


def test_first_tp_partial_close_and_move_to_breakeven():
    async def go():
        ex = StubExchange()
        pm = PositionManager(ex, breakeven_buffer_atr=0.0)
        pos = await _open(pm, ex)
        res = pm.on_tick(102.0)
        assert res.action == "PARTIAL_CLOSE"
        # SL should have moved to breakeven (entry +/- 0 buffer).
        assert pos.stop_loss == pytest.approx(pos.entry_price)
        assert pos.trailing_armed is True
    run(go())


def test_final_tp_exits_fully():
    async def go():
        ex = StubExchange()
        pm = PositionManager(ex)
        await _open(pm, ex, tps=[(110.0, 100.0)])
        res = pm.on_tick(110.0)
        assert res.action == "EXIT"
    run(go())


def test_trailing_stop_tightens_only():
    async def go():
        ex = StubExchange()
        # Disable giveback + %-breakeven so this test isolates trailing logic.
        pm = PositionManager(
            ex, trail_atr_mult=1.0, trail_arm_atr=0.5,
            giveback_arm_pct=0.0, giveback_exit_pct=0.0,
            breakeven_profit_pct=0.0,
        )
        pos = await _open(pm, ex, tps=[(200.0, 100.0)])  # won't hit TP
        pm.on_tick(105.0)   # arms trailing (+5 > 0.5*2 ATR)
        assert pos.trailing_armed
        pm.on_tick(110.0)   # best=110, stop should be around 108
        new_sl1 = pos.stop_loss
        assert new_sl1 > 95.0
        # Price dips but best stays at 110 -> stop must not loosen.
        pm.on_tick(108.5)
        assert pos.stop_loss == new_sl1
    run(go())


def test_max_loss_guard_fires():
    async def go():
        ex = StubExchange()
        pm = PositionManager(ex, max_loss_pct=2.0)
        await _open(pm, ex, sl=50.0)  # wide SL so max_loss fires first
        # 2% of 100 = $2 loss on 1 qty means $2 adverse price move.
        res = pm.on_tick(97.9)
        assert res.action == "EXIT"
        assert res.reason == "MAX_LOSS"
    run(go())


def test_short_position_stop_and_tp():
    async def go():
        ex = StubExchange()
        pm = PositionManager(ex)
        await _open(
            pm, ex, side="SHORT", entry=100.0, sl=105.0,
            tps=[(95.0, 100.0)],
        )
        # Price drops to TP -> full exit.
        res = pm.on_tick(95.0)
        assert res.action == "EXIT"
        ex.fills_at = 95.0
        await pm.close(res.reason, 95.0)

        # Re-open to test stop side.
        await _open(pm, ex, side="SHORT", entry=100.0, sl=105.0,
                    tps=[(90.0, 100.0)])
        res = pm.on_tick(105.0)
        assert res.action == "EXIT"
        assert res.reason == "STOP_LOSS"
    run(go())


def test_hold_when_inside_range():
    async def go():
        ex = StubExchange()
        # Disable %-based breakeven so a 0.5% move doesn't tighten the stop.
        pm = PositionManager(ex, breakeven_profit_pct=0.0)
        await _open(pm, ex)
        res = pm.on_tick(100.5)
        assert res.action == "HOLD"
        assert res.new_stop is None
    run(go())


def test_breakeven_on_profit_pct_before_tp1():
    """A +0.5% move (below TP1 at 102) must still snap SL to entry."""
    async def go():
        ex = StubExchange()
        pm = PositionManager(
            ex, breakeven_profit_pct=0.4, breakeven_buffer_atr=0.0,
            giveback_arm_pct=0.0,  # isolate BE behavior
        )
        pos = await _open(pm, ex)  # entry 100, SL 95, TP1 102
        res = pm.on_tick(100.5)
        assert res.action == "HOLD"
        assert res.new_stop is not None
        assert pos.stop_loss == pytest.approx(100.0)
        assert pos.breakeven_moved
        # And it must not loosen on a subsequent smaller gain.
        pm.on_tick(100.2)
        assert pos.stop_loss == pytest.approx(100.0)
    run(go())


def test_giveback_exit_fires_after_peak():
    """Position hits +1.5% then retraces to +0.8% -> exit, don't round-trip."""
    async def go():
        ex = StubExchange()
        pm = PositionManager(
            ex, breakeven_profit_pct=0.0,  # disable BE to isolate giveback
            giveback_arm_pct=1.0, giveback_exit_pct=0.6,
        )
        await _open(pm, ex, tps=[(200.0, 100.0)])  # TPs out of reach
        # Ride to +1.5% peak.
        pm.on_tick(101.5)
        # Give back more than 0.6% from peak -> should exit.
        res = pm.on_tick(100.8)
        assert res.action == "EXIT"
        assert res.reason == "GIVEBACK"
    run(go())


def test_giveback_disarmed_below_arm_threshold():
    """If peak never reached arm_pct, giveback must NOT fire."""
    async def go():
        ex = StubExchange()
        pm = PositionManager(
            ex, breakeven_profit_pct=0.0,
            giveback_arm_pct=2.0, giveback_exit_pct=0.6,
        )
        await _open(pm, ex, tps=[(200.0, 100.0)])
        pm.on_tick(101.0)   # peak +1%, below 2% arm
        res = pm.on_tick(100.2)  # gave back 0.8%
        assert res.action == "HOLD"
    run(go())


def test_trailing_tightens_after_deep_profit():
    """At +3 ATR profit the trail mult tightens from 1.5 -> 0.75 ATR."""
    async def go():
        ex = StubExchange()
        pm = PositionManager(
            ex, trail_atr_mult=1.5, trail_arm_atr=0.5,
            trail_tighten_atr=2.0, trail_tighten_mult=0.75,
            breakeven_profit_pct=0.0, giveback_arm_pct=0.0,
        )
        pos = await _open(pm, ex, tps=[(500.0, 100.0)])  # out-of-reach TP
        # ATR = 2.0. Push price to +6 (>= 2 ATR profit) to arm tightened trail.
        pm.on_tick(106.0)
        # Expected tightened stop = best - 0.75*ATR = 106 - 1.5 = 104.5
        assert pos.stop_loss == pytest.approx(104.5)
    run(go())


def test_short_giveback_exit():
    async def go():
        ex = StubExchange()
        pm = PositionManager(
            ex, breakeven_profit_pct=0.0,
            giveback_arm_pct=1.0, giveback_exit_pct=0.6,
        )
        # Short at 100, SL 105, TPs far.
        await _open(pm, ex, side="SHORT", entry=100.0, sl=105.0,
                    tps=[(50.0, 100.0)])
        pm.on_tick(98.5)  # peak +1.5% short profit
        res = pm.on_tick(99.2)  # give back 0.7% from peak
        assert res.action == "EXIT"
        assert res.reason == "GIVEBACK"
    run(go())
