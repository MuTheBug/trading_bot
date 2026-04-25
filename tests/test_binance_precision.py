"""Tests for Binance precision formatting helpers.

These guard against the -1111 "Precision is over the maximum defined for
this asset" API error caused by Python float -> str conversion bleeding
trailing digits (e.g. 0.3330000000000001).
"""
from src.exchange.binance_live import (
    _decimals_for_step,
    _fmt_price,
    _fmt_quantity,
)


def test_decimals_for_step_common():
    assert _decimals_for_step(1.0) == 0
    assert _decimals_for_step(0.1) == 1
    assert _decimals_for_step(0.01) == 2
    assert _decimals_for_step(0.001) == 3
    assert _decimals_for_step(0.0001) == 4
    assert _decimals_for_step(0.00001) == 5


def test_decimals_for_step_zero_safe():
    # Pathological fallback — we default to 8 digits which is still within
    # Binance's tolerance and prevents a crash.
    assert _decimals_for_step(0.0) == 8


def test_fmt_quantity_strips_float_noise():
    # 0.333 / step 0.001 repeated adds would produce 0.333000...0001.
    qty = 0.1 + 0.2 + 0.033  # = 0.33300000000000007
    assert _fmt_quantity(qty, 0.001) == "0.333"


def test_fmt_quantity_floors_to_step():
    # 0.3337 with step 0.001 should floor to 0.333 (never exceed intent).
    assert _fmt_quantity(0.3337, 0.001) == "0.333"


def test_fmt_quantity_integer_step():
    assert _fmt_quantity(42.9, 1.0) == "42"


def test_fmt_price_rounds_to_tick():
    assert _fmt_price(27123.457, 0.1) == "27123.5"
    assert _fmt_price(27123.44, 0.1) == "27123.4"


def test_fmt_price_exact_decimals():
    # Must not emit scientific notation or trailing noise.
    s = _fmt_price(0.00012345, 0.00000001)
    assert "e" not in s.lower()
    assert s.count(".") == 1
    assert len(s.split(".")[1]) == 8


def test_fmt_quantity_small_step():
    # BTC steps are often 0.001; ensure tiny qty renders cleanly.
    assert _fmt_quantity(0.000999999, 0.001) == "0.000"
    assert _fmt_quantity(0.001, 0.001) == "0.001"
