"""Risk manager tests for grid trading: circuit breakers and funding guard."""
import pytest

from src.config import GridConfig, RiskConfig
from src.risk.risk_manager import RiskManager
from src.state import DailyStats, StateStore


@pytest.fixture
def rm():
    return RiskManager(RiskConfig(), GridConfig())


def test_can_run_grid_ok(rm, tmp_path):
    store = StateStore(tmp_path / "state.json")
    ok, reason = rm.can_run_grid(store, equity=10.0)
    assert ok
    assert reason == ""


def test_can_run_grid_paused(rm, tmp_path):
    store = StateStore(tmp_path / "state.json")
    store.state.paused = True
    ok, reason = rm.can_run_grid(store, equity=10.0)
    assert not ok
    assert "paused" in reason


def test_daily_loss_circuit_breaker(rm, tmp_path):
    store = StateStore(tmp_path / "state.json")
    store.state.daily = DailyStats(date="2026-04-11", realized_pnl=-0.25)
    ok, reason = rm.can_run_grid(store, equity=10.0)
    assert not ok
    assert "daily" in reason.lower()


def test_max_drawdown_circuit_breaker(rm, tmp_path):
    store = StateStore(tmp_path / "state.json")
    store.state.peak_equity = 10.0
    ok, reason = rm.can_run_grid(store, equity=8.9)
    assert not ok
    assert "drawdown" in reason.lower()


def test_funding_action_flat(rm):
    assert rm.funding_action(25.0, "FLAT") == "ok"


def test_funding_action_exit_long(rm):
    assert rm.funding_action(25.0, "LONG") == "exit"


def test_funding_action_exit_short(rm):
    assert rm.funding_action(-25.0, "SHORT") == "exit"


def test_funding_action_ok_below_threshold(rm):
    assert rm.funding_action(5.0, "LONG") == "ok"
    assert rm.funding_action(-5.0, "SHORT") == "ok"


def test_funding_action_no_exit_opposite_side(rm):
    # Positive funding + short = collecting, not paying
    assert rm.funding_action(25.0, "SHORT") == "ok"
    assert rm.funding_action(-25.0, "LONG") == "ok"
