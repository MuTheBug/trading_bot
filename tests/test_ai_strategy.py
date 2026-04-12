"""AI grid strategy tests — exercise parsing, symbol selection, and grid param
decisions with a stubbed Anthropic client so no network calls happen.
"""
import json
from types import SimpleNamespace

import pytest

from src.config import AIConfig, GridConfig
from src.exchange.base import SymbolFilters, TickerInfo
from src.strategy.ai_strategy import AIGridStrategy, _extract_json


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
    assert "volume" in choice.reasoning.lower() or len(choice.reasoning) > 0


@pytest.mark.asyncio
async def test_select_symbol_invalid_pick():
    reply = json.dumps({
        "symbol": "INVALIDUSDT",
        "reasoning": "this does not exist",
    })
    s = _strategy_with_reply(reply)
    choice = await s.select_symbol(_make_tickers(), _make_filters())
    assert choice is None


@pytest.mark.asyncio
async def test_select_symbol_bad_json():
    s = _strategy_with_reply("sorry, can't help")
    choice = await s.select_symbol(_make_tickers(), _make_filters())
    assert choice is None


# ---- Grid parameter decisions ----

@pytest.mark.asyncio
async def test_decide_grid_params_happy_path():
    reply = json.dumps({
        "upper_price": 0.155,
        "lower_price": 0.145,
        "num_grids": 5,
        "leverage": 10,
        "qty_per_grid": 50.0,
        "reasoning": "tight range around current price",
    })
    s = _strategy_with_reply(reply)
    ticker = _make_tickers()[0]  # DOGEUSDT
    filters = _make_filters()["DOGEUSDT"]
    decision = await s.decide_grid_params(
        symbol="DOGEUSDT",
        current_price=0.15,
        ticker=ticker,
        filters=filters,
        balance=10.0,
    )
    assert decision is not None
    assert decision.upper_price == 0.155
    assert decision.lower_price == 0.145
    assert decision.num_grids == 5
    assert decision.leverage == 10


@pytest.mark.asyncio
async def test_decide_grid_params_price_outside_range():
    reply = json.dumps({
        "upper_price": 0.10,
        "lower_price": 0.09,
        "num_grids": 5,
        "leverage": 10,
        "qty_per_grid": 50.0,
        "reasoning": "wrong range",
    })
    s = _strategy_with_reply(reply)
    ticker = _make_tickers()[0]
    filters = _make_filters()["DOGEUSDT"]
    decision = await s.decide_grid_params(
        symbol="DOGEUSDT",
        current_price=0.15,
        ticker=ticker,
        filters=filters,
        balance=10.0,
    )
    assert decision is None  # current price outside range


@pytest.mark.asyncio
async def test_decide_grid_params_too_many_grids():
    reply = json.dumps({
        "upper_price": 0.155,
        "lower_price": 0.145,
        "num_grids": 50,  # exceeds max_grids
        "leverage": 10,
        "qty_per_grid": 50.0,
        "reasoning": "too many",
    })
    s = _strategy_with_reply(reply)
    ticker = _make_tickers()[0]
    filters = _make_filters()["DOGEUSDT"]
    decision = await s.decide_grid_params(
        symbol="DOGEUSDT",
        current_price=0.15,
        ticker=ticker,
        filters=filters,
        balance=10.0,
    )
    assert decision is None


# ---- Rebalance evaluation ----

@pytest.mark.asyncio
async def test_evaluate_rebalance_hold():
    reply = json.dumps({
        "action": "HOLD",
        "reasoning": "price will return",
        "new_upper": None,
        "new_lower": None,
        "new_num_grids": None,
        "new_leverage": None,
        "new_qty_per_grid": None,
    })
    s = _strategy_with_reply(reply)
    summary = {
        "symbol": "DOGEUSDT", "upper": 0.155, "lower": 0.145,
        "num_grids": 5, "spacing": 0.002, "leverage": 10,
        "qty_per_grid": 50.0, "total_profit": 0.001, "round_trips": 3,
        "net_qty": 50.0, "unrealized_pnl": -0.002, "mark_price": 0.157,
    }
    filters = _make_filters()["DOGEUSDT"]
    decision = await s.evaluate_rebalance("DOGEUSDT", summary, 10.0, filters)
    assert decision.action == "HOLD"


@pytest.mark.asyncio
async def test_evaluate_rebalance_rebalance():
    reply = json.dumps({
        "action": "REBALANCE",
        "reasoning": "price moved too far",
        "new_upper": 0.165,
        "new_lower": 0.155,
        "new_num_grids": 5,
        "new_leverage": 10,
        "new_qty_per_grid": 50.0,
    })
    s = _strategy_with_reply(reply)
    summary = {
        "symbol": "DOGEUSDT", "upper": 0.155, "lower": 0.145,
        "num_grids": 5, "spacing": 0.002, "leverage": 10,
        "qty_per_grid": 50.0, "total_profit": 0.001, "round_trips": 3,
        "net_qty": 50.0, "unrealized_pnl": -0.002, "mark_price": 0.16,
    }
    filters = _make_filters()["DOGEUSDT"]
    decision = await s.evaluate_rebalance("DOGEUSDT", summary, 10.0, filters)
    assert decision.action == "REBALANCE"
    assert decision.new_params is not None
    assert decision.new_params.upper_price == 0.165
