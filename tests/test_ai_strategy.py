"""AI grid strategy tests — symbol selection, math-based grid params,
rebalance, and JSON parsing helpers. Stubbed Anthropic client for no network.
"""
import json
from types import SimpleNamespace

import pytest

from src.config import AIConfig, GridConfig
from src.exchange.base import SymbolFilters, TickerInfo
from src.strategy.ai_strategy import (
    AIGridStrategy, _extract_json, score_symbol, compute_grid_params,
    is_grid_friendly, MAKER_FEE, MIN_FEE_MULT,
)


class _StubMessages:
    def __init__(self, reply: str):
        self._reply = reply
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text=self._reply)]
        )


class _StubClient:
    def __init__(self, reply: str):
        self.messages = _StubMessages(reply)


def _strategy_with_reply(reply: str) -> AIGridStrategy:
    return AIGridStrategy(
        ai_cfg=AIConfig(retries=0),
        grid_cfg=GridConfig(),
        api_key="test",
        base_url="https://example.invalid",
        client=_StubClient(reply),
    )


def _make_tickers():
    return [
        TickerInfo(
            symbol="DOGEUSDT", price=0.15, volume_24h=200_000_000,
            change_pct_24h=2.5, high_24h=0.155, low_24h=0.145,
        ),
        TickerInfo(
            symbol="BTCUSDT", price=60000.0, volume_24h=500_000_000,
            change_pct_24h=-0.5, high_24h=61000.0, low_24h=59000.0,
        ),
        TickerInfo(
            symbol="1000PEPEUSDT", price=0.008, volume_24h=100_000_000,
            change_pct_24h=5.0, high_24h=0.0085, low_24h=0.0075,
        ),
    ]


def _make_filters():
    return {
        "DOGEUSDT": SymbolFilters("DOGEUSDT", 0.00001, 1.0, 1.0, 5.0),
        "BTCUSDT": SymbolFilters("BTCUSDT", 0.01, 0.001, 0.001, 5.0),
        "1000PEPEUSDT": SymbolFilters("1000PEPEUSDT", 0.0000001, 1.0, 1.0, 5.0),
    }


# ---- JSON extraction ----

def test_extract_json_plain():
    assert _extract_json('{"a": 1}') == {"a": 1}


def test_extract_json_with_fence():
    text = "sure:\n```json\n{\"side\": \"LONG\"}\n```\nthanks"
    assert _extract_json(text) == {"side": "LONG"}


def test_extract_json_with_prose():
    text = 'here is my decision {"side": "SHORT", "confidence": 0.8}. thanks!'
    out = _extract_json(text)
    assert out is not None and out["side"] == "SHORT"


def test_extract_json_garbage():
    assert _extract_json("not json at all") is None


# ---- Symbol scoring ----

def test_score_symbol_oscillating_beats_trending():
    # High range, low net change -> high score (oscillating)
    oscillating = TickerInfo("A", 1.0, 100_000_000, 0.5, 1.05, 0.95)
    # High range, high net change -> lower score (trending)
    trending = TickerInfo("B", 1.0, 100_000_000, 8.0, 1.10, 1.00)
    assert score_symbol(oscillating) > score_symbol(trending)


def test_score_symbol_zero_volume():
    t = TickerInfo("X", 1.0, 0, 1.0, 1.01, 0.99)
    assert score_symbol(t) == 0.0


def test_score_symbol_zero_price():
    t = TickerInfo("X", 0.0, 1e6, 1.0, 0.01, 0.0)
    assert score_symbol(t) == 0.0


def test_is_grid_friendly_rejects_at_24h_high():
    """A symbol sitting at its 24h high is about to break — not a grid setup."""
    t = TickerInfo(
        symbol="X", price=1.10, volume_24h=1e8, change_pct_24h=0.5,
        high_24h=1.10, low_24h=1.00,
    )
    assert not is_grid_friendly(t)


def test_is_grid_friendly_rejects_at_24h_low():
    t = TickerInfo(
        symbol="X", price=1.00, volume_24h=1e8, change_pct_24h=-0.5,
        high_24h=1.10, low_24h=1.00,
    )
    assert not is_grid_friendly(t)


def test_is_grid_friendly_accepts_middle_of_range():
    t = TickerInfo(
        symbol="X", price=1.05, volume_24h=1e8, change_pct_24h=0.5,
        high_24h=1.10, low_24h=1.00,
    )
    assert is_grid_friendly(t)


def test_is_grid_friendly_rejects_strong_trend():
    """Tightened MAX_TREND_PCT to 3% so grids don't fight already-moving markets."""
    t = TickerInfo(
        symbol="X", price=1.05, volume_24h=1e8, change_pct_24h=4.0,
        high_24h=1.10, low_24h=1.00,
    )
    assert not is_grid_friendly(t)


# ---- Symbol selection ----

@pytest.mark.asyncio
async def test_select_symbol_happy_path():
    reply = json.dumps({
        "symbol": "DOGEUSDT",
        "reasoning": "good volume and range",
    })
    s = _strategy_with_reply(reply)
    choice = await s.select_symbol(_make_tickers(), _make_filters())
    assert choice is not None
    assert choice.symbol == "DOGEUSDT"
    assert choice.score > 0


@pytest.mark.asyncio
async def test_select_symbol_invalid_pick_falls_back():
    """AI picks a symbol not in the top list — fallback to highest score."""
    reply = json.dumps({
        "symbol": "INVALIDUSDT",
        "reasoning": "this does not exist",
    })
    s = _strategy_with_reply(reply)
    choice = await s.select_symbol(_make_tickers(), _make_filters())
    # Falls back to highest-scored symbol instead of None
    assert choice is not None
    assert choice.score > 0


@pytest.mark.asyncio
async def test_select_symbol_bad_json_falls_back():
    """AI returns garbage — fallback to highest-scored symbol."""
    s = _strategy_with_reply("sorry, can't help")
    choice = await s.select_symbol(_make_tickers(), _make_filters())
    assert choice is not None
    assert choice.score > 0


# ---- Math-based grid parameter computation ----

def test_compute_params_happy_path():
    s = _strategy_with_reply("")
    ticker = _make_tickers()[0]  # DOGEUSDT price=0.15
    filters = _make_filters()["DOGEUSDT"]
    decision = s.compute_params(
        symbol="DOGEUSDT",
        current_price=0.15,
        ticker=ticker,
        filters=filters,
        balance=10.0,
    )
    assert decision is not None
    assert decision.lower_price < 0.15 < decision.upper_price
    assert decision.num_grids >= 3
    assert decision.spacing > 0
    assert decision.profit_per_trip > 0
    assert decision.leverage >= 5


def test_compute_params_profit_guaranteed():
    """Each round trip must be profitable after fees."""
    s = _strategy_with_reply("")
    ticker = _make_tickers()[0]
    filters = _make_filters()["DOGEUSDT"]
    decision = s.compute_params("DOGEUSDT", 0.15, ticker, filters, 10.0)
    assert decision is not None
    # Verify the math: spacing * qty > 2 * fee * price * qty
    fee_cost = 2 * MAKER_FEE * 0.15 * decision.qty_per_grid
    gross = decision.spacing * decision.qty_per_grid
    assert gross > fee_cost


def test_compute_params_zero_balance():
    s = _strategy_with_reply("")
    ticker = _make_tickers()[0]
    filters = _make_filters()["DOGEUSDT"]
    decision = s.compute_params("DOGEUSDT", 0.15, ticker, filters, 0.0)
    assert decision is None


def test_compute_params_zero_price():
    s = _strategy_with_reply("")
    ticker = _make_tickers()[0]
    filters = _make_filters()["DOGEUSDT"]
    decision = s.compute_params("DOGEUSDT", 0.0, ticker, filters, 10.0)
    assert decision is None


def test_compute_params_btc():
    """BTC has high price — ensure grid still works with $10."""
    s = _strategy_with_reply("")
    ticker = _make_tickers()[1]  # BTCUSDT price=60000
    filters = _make_filters()["BTCUSDT"]
    decision = s.compute_params("BTCUSDT", 60000.0, ticker, filters, 10.0)
    # May or may not be viable with $10 — just ensure no crash
    if decision is not None:
        assert decision.profit_per_trip > 0


# ---- compute_grid_params standalone ----

def test_compute_grid_params_min_spacing_respected():
    """Spacing must be at least MIN_FEE_MULT * 2 * fee * price."""
    ticker = TickerInfo("TEST", 100.0, 1e9, 1.0, 105.0, 95.0)
    filters = SymbolFilters("TEST", 0.01, 0.01, 0.01, 5.0)
    decision = compute_grid_params(100.0, ticker, filters, 100.0, 20, 15, 3)
    assert decision is not None
    min_spacing = 100.0 * 2 * MAKER_FEE * MIN_FEE_MULT
    assert decision.spacing >= min_spacing * 0.99  # allow tiny float imprecision


def test_compute_grid_params_scales_qty_to_capital_cap():
    """Previously the solver always picked the minimum qty that met
    min_notional, so with a $10 DOGEUSDT grid each round trip yielded
    sub-cent profits. The fix scales qty UP to the capital cap.
    """
    ticker = TickerInfo(
        "DOGEUSDT", price=0.15, volume_24h=200_000_000,
        change_pct_24h=2.5, high_24h=0.155, low_24h=0.145,
    )
    filters = SymbolFilters("DOGEUSDT", 0.00001, 1.0, 1.0, 5.0)
    decision = compute_grid_params(
        price=0.15, ticker=ticker, filters=filters, balance=10.0,
        max_leverage=10, max_grids=15, min_grids=3, max_capital_pct=75.0,
    )
    assert decision is not None
    # Floor qty (min_notional at lower_price, rounded up to step) is 1.
    # New solver should pick >> 1 to use the $7.50 margin budget.
    assert decision.qty_per_grid > 10, (
        f"qty {decision.qty_per_grid} is still near the min_notional floor"
    )
    # And the full-cycle profit should be meaningful (>1% of balance)
    total_trip_profit = decision.num_grids * decision.profit_per_trip
    assert total_trip_profit > 0.10, (
        f"total trip profit {total_trip_profit:.4f} is < 1% of balance"
    )


def test_compute_grid_params_grid_width_floor():
    """Grid width must be at least 3% of price so it's wider than the
    2.5-3% drift_exit threshold. If the grid is narrower than drift,
    every setup would drift-exit on the first meaningful move.
    """
    # Low-range ticker: only 1% daily range, used to have 40% grid = 0.4% wide
    ticker = TickerInfo(
        "SLEEPY", price=1.00, volume_24h=1e8, change_pct_24h=0.1,
        high_24h=1.005, low_24h=0.995,
    )
    filters = SymbolFilters("SLEEPY", 0.0001, 0.01, 0.01, 5.0)
    decision = compute_grid_params(
        price=1.00, ticker=ticker, filters=filters, balance=100.0,
        max_leverage=5, max_grids=15, min_grids=3, max_capital_pct=50.0,
    )
    assert decision is not None
    width_pct = (decision.upper_price - decision.lower_price) / 1.00 * 100.0
    assert width_pct >= 3.0 - 1e-6, f"grid width {width_pct:.3f}% is below 3% floor"


def test_compute_grid_params_respects_capital_cap():
    """Total margin must stay under max_capital_pct of balance."""
    ticker = TickerInfo(
        "DOGEUSDT", price=0.15, volume_24h=200_000_000,
        change_pct_24h=2.5, high_24h=0.155, low_24h=0.145,
    )
    filters = SymbolFilters("DOGEUSDT", 0.00001, 1.0, 1.0, 5.0)
    decision = compute_grid_params(
        price=0.15, ticker=ticker, filters=filters, balance=10.0,
        max_leverage=10, max_grids=15, min_grids=3, max_capital_pct=50.0,
    )
    assert decision is not None
    avg_price = (decision.upper_price + decision.lower_price) / 2
    total_margin = decision.num_grids * decision.qty_per_grid * avg_price / decision.leverage
    # Allow 2% headroom for step rounding
    assert total_margin <= 10.0 * 0.50 * 1.02


# ---- Rebalance (math-based) ----

def test_compute_rebalance_returns_new_params():
    s = _strategy_with_reply("")
    ticker = _make_tickers()[0]
    filters = _make_filters()["DOGEUSDT"]
    summary = {
        "symbol": "DOGEUSDT", "upper": 0.155, "lower": 0.145,
        "num_grids": 5, "spacing": 0.002, "leverage": 10,
        "qty_per_grid": 50.0, "total_profit": 0.001, "total_fees": 0.0002,
        "round_trips": 3, "net_qty": 50.0, "unrealized_pnl": -0.002,
        "mark_price": 0.16,
    }
    decision = s.compute_rebalance(
        symbol="DOGEUSDT",
        current_price=0.16,
        ticker=ticker,
        filters=filters,
        balance=10.0,
        grid_summary=summary,
    )
    assert decision.action == "REBALANCE"
    assert decision.new_params is not None
    # New grid should be centered around current price
    p = decision.new_params
    assert p.lower_price < 0.16 < p.upper_price


def test_compute_rebalance_hold_when_no_viable_params():
    """If compute_grid_params returns None, decision should be HOLD."""
    s = _strategy_with_reply("")
    # Ticker with zero price — no viable grid
    ticker = TickerInfo("BAD", 0.0, 1e6, 0.0, 0.0, 0.0)
    filters = _make_filters()["DOGEUSDT"]
    summary = {
        "symbol": "BAD", "upper": 0.01, "lower": 0.005,
        "num_grids": 3, "spacing": 0.001, "leverage": 5,
        "qty_per_grid": 1.0, "total_profit": 0.0, "total_fees": 0.0,
        "round_trips": 0, "net_qty": 0.0, "unrealized_pnl": 0.0,
        "mark_price": 0.0,
    }
    decision = s.compute_rebalance("BAD", 0.0, ticker, filters, 10.0, summary)
    assert decision.action == "HOLD"
