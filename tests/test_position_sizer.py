"""Position sizer tests."""
import pytest

from src.exchange.base import SymbolFilters
from src.risk.position_sizer import compute_position_size


FILTERS = SymbolFilters(
    symbol="DOGEUSDT",
    price_tick=0.00001,
    qty_step=1.0,
    min_qty=1.0,
    min_notional=5.0,
)


def test_sizing_basic_feasible():
    # equity=10, risk 1% => risk $0.10; stop distance $0.002 on $0.20 price
    # raw_qty = 0.10 / 0.002 = 50 DOGE; notional = 50 * 0.20 = $10 -> above min_notional
    res = compute_position_size(
        equity=10.0, risk_pct=1.0,
        entry_price=0.20, stop_price=0.198,
        filters=FILTERS, leverage=3,
    )
    assert res.feasible
    assert res.qty == 50
    assert res.notional == pytest.approx(10.0)


def test_sizing_respects_min_notional():
    # Very tight stop => tiny qty => notional below 5.0 => infeasible
    res = compute_position_size(
        equity=10.0, risk_pct=0.1,
        entry_price=0.20, stop_price=0.199,
        filters=FILTERS, leverage=3,
    )
    assert not res.feasible
    assert "min_notional" in res.reason


def test_sizing_rounds_down_to_step():
    fil = SymbolFilters("X", price_tick=0.01, qty_step=0.001, min_qty=0.001, min_notional=5.0)
    res = compute_position_size(
        equity=100.0, risk_pct=1.0,
        entry_price=100.0, stop_price=99.0,
        filters=fil, leverage=3,
    )
    # raw_qty = 1.0, step 0.001 => 1.000 exactly
    assert res.feasible
    # qty must be a multiple of 0.001
    q_scaled = round(res.qty / 0.001)
    assert abs(res.qty - q_scaled * 0.001) < 1e-9


def test_sizing_margin_cap():
    # equity=10, leverage=1, huge risk => qty capped by margin
    res = compute_position_size(
        equity=10.0, risk_pct=100.0,  # wildly over-risk
        entry_price=0.20, stop_price=0.01,  # giant stop distance
        filters=FILTERS, leverage=1,
    )
    assert res.feasible
    assert res.qty * 0.20 <= 10.0 * 1 + 1e-9


def test_sizing_invalid_prices():
    res = compute_position_size(
        equity=10.0, risk_pct=1.0,
        entry_price=0.0, stop_price=0.198,
        filters=FILTERS, leverage=3,
    )
    assert not res.feasible
