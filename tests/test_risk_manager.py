"""Risk manager tests: exit ladder, stop management, circuit breakers."""
from datetime import datetime, timedelta, timezone

import pytest

from src.config import ExitsConfig, RiskConfig
from src.risk.risk_manager import RiskManager
from src.state import BotState, DailyStats, Position, StateStore, TakeProfitLevel


@pytest.fixture
def rm():
    return RiskManager(RiskConfig(), ExitsConfig())


def make_long_position(entry=100.0, atr=2.0) -> Position:
    sl = entry - 1.5 * atr
    tps = [
        TakeProfitLevel(price=entry + 0.75 * atr, close_pct=40.0),
        TakeProfitLevel(price=entry + 2.0 * atr, close_pct=30.0),
        TakeProfitLevel(price=entry + 4.0 * atr, close_pct=30.0),
    ]
    return Position(
        symbol="TEST",
        side="LONG",
        entry_price=entry,
        original_qty=1.0,
        remaining_qty=1.0,
        leverage=3,
        stop_loss=sl,
        take_profits=tps,
        opened_at=datetime.now(timezone.utc).isoformat(),
        atr_at_entry=atr,
        highest_since_entry=entry,
        lowest_since_entry=entry,
    )


def test_build_exit_ladder_long(rm):
    sl, tps = rm.build_exit_ladder(entry=100.0, side="LONG", atr=2.0)
    assert sl == pytest.approx(97.0)
    assert tps[0].price == pytest.approx(101.5)
    assert tps[1].price == pytest.approx(104.0)
    assert tps[2].price == pytest.approx(108.0)
    assert tps[0].close_pct + tps[1].close_pct + tps[2].close_pct == pytest.approx(100.0)


def test_build_exit_ladder_short(rm):
    sl, tps = rm.build_exit_ladder(entry=100.0, side="SHORT", atr=2.0)
    assert sl == pytest.approx(103.0)
    assert tps[0].price == pytest.approx(98.5)
    assert tps[1].price == pytest.approx(96.0)
    assert tps[2].price == pytest.approx(92.0)


def test_stop_loss_hit_long(rm):
    pos = make_long_position()
    d = rm.manage_position(pos, mark_price=96.0)
    assert d is not None
    assert d.reason == "SL"
    assert d.close_qty_pct == 100.0


def test_tp1_moves_sl_to_be(rm):
    pos = make_long_position()
    d = rm.manage_position(pos, mark_price=101.5)
    assert d is not None
    assert d.reason == "TP1"
    assert d.close_qty_pct == 40.0
    assert d.new_stop_loss == pytest.approx(100.0)


def test_tp2_enables_trailing(rm):
    pos = make_long_position()
    pos.take_profits[0].hit = True
    pos.tp1_hit = True
    d = rm.manage_position(pos, mark_price=104.0)
    assert d is not None
    assert d.reason == "TP2"
    assert d.enable_trailing is True


def test_time_stop(rm):
    pos = make_long_position()
    pos.opened_at = (datetime.now(timezone.utc) - timedelta(hours=25)).isoformat()
    d = rm.manage_position(pos, mark_price=100.5)
    assert d is not None
    assert d.reason == "TIME"


def test_daily_loss_circuit_breaker(rm, tmp_path):
    store = StateStore(tmp_path / "state.json")
    store.state.daily = DailyStats(date="2026-04-11", realized_pnl=-0.25)
    ok, reason = rm.can_open_new(store, equity=10.0, symbol="DOGEUSDT")
    assert not ok
    assert "daily" in reason.lower()


def test_max_concurrent_positions(rm, tmp_path):
    store = StateStore(tmp_path / "state.json")
    store.state.positions["AAA"] = make_long_position()
    store.state.positions["BBB"] = make_long_position()
    ok, reason = rm.can_open_new(store, equity=10.0, symbol="CCC")
    assert not ok
    assert "concurrent" in reason.lower()


def test_funding_guard_skip_and_exit(rm):
    # Skip on any side when near skip threshold
    assert rm.funding_action(16.0, "LONG") == "skip"
    assert rm.funding_action(-16.0, "SHORT") == "skip"
    # Exit only on the paying side
    assert rm.funding_action(21.0, "LONG") == "exit"
    assert rm.funding_action(21.0, "SHORT") == "skip"
    assert rm.funding_action(-21.0, "SHORT") == "exit"
    # OK below thresholds
    assert rm.funding_action(5.0, "LONG") == "ok"
