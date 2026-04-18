"""Tests for AI directional strategy parser / validator."""
import pytest

from src.strategy.ai_directional import (
    AIDecision,
    _extract_json,
    _parse_decision,
)


def test_parse_valid_long():
    obj = {
        "action": "OPEN_LONG",
        "symbol": "btcusdt",
        "entry": 100.0,
        "stop_loss": 98.0,
        "take_profits": [[102.0, 50], [106.0, 50]],
        "leverage": 5,
        "confidence": 0.7,
        "reasoning": "trend",
    }
    d = _parse_decision(obj)
    assert d.is_trade
    assert d.side == "LONG"
    assert d.symbol == "BTCUSDT"
    assert d.leverage == 5
    # close_pct rescaled to sum to 100
    assert abs(sum(p for _, p in d.take_profits) - 100.0) < 1e-6
    # Ordered away from entry (ascending for long)
    prices = [p for p, _ in d.take_profits]
    assert prices == sorted(prices)


def test_parse_valid_short_rescales_tp():
    obj = {
        "action": "OPEN_SHORT",
        "symbol": "ETHUSDT",
        "entry": 100.0,
        "stop_loss": 103.0,
        "take_profits": [
            {"price": 97.0, "close_pct": 30},
            {"price": 90.0, "close_pct": 30},
        ],
        "leverage": 0,
        "confidence": 0.6,
    }
    d = _parse_decision(obj)
    assert d.is_trade
    assert d.side == "SHORT"
    prices = [p for p, _ in d.take_profits]
    # Short: TPs ordered descending (further below entry = later).
    assert prices == sorted(prices, reverse=True)
    assert abs(sum(p for _, p in d.take_profits) - 100.0) < 1e-6


def test_parse_skip():
    d = _parse_decision({"action": "SKIP", "reasoning": "choppy"})
    assert d.action == "SKIP"
    assert not d.is_trade


def test_parse_rejects_wrong_side_sl_for_long():
    d = _parse_decision({
        "action": "OPEN_LONG", "symbol": "X", "entry": 100, "stop_loss": 105,
        "take_profits": [[110, 100]], "confidence": 0.5,
    })
    assert d.action == "SKIP"


def test_parse_rejects_wrong_side_sl_for_short():
    d = _parse_decision({
        "action": "OPEN_SHORT", "symbol": "X", "entry": 100, "stop_loss": 95,
        "take_profits": [[90, 100]], "confidence": 0.5,
    })
    assert d.action == "SKIP"


def test_parse_drops_tps_on_wrong_side():
    # LONG with entry 100, one TP below entry (invalid) and one above.
    d = _parse_decision({
        "action": "OPEN_LONG", "symbol": "X", "entry": 100, "stop_loss": 98,
        "take_profits": [[99, 50], [110, 50]], "confidence": 0.5,
    })
    assert d.is_trade
    assert len(d.take_profits) == 1
    assert d.take_profits[0][0] == 110.0


def test_parse_skip_when_no_valid_tps():
    d = _parse_decision({
        "action": "OPEN_LONG", "symbol": "X", "entry": 100, "stop_loss": 98,
        "take_profits": [[99, 50]], "confidence": 0.5,
    })
    assert d.action == "SKIP"


def test_parse_invalid_entry():
    d = _parse_decision({"action": "OPEN_LONG", "symbol": "X"})
    assert d.action == "SKIP"


def test_parse_clamps_confidence():
    d = _parse_decision({
        "action": "OPEN_LONG", "symbol": "X", "entry": 100, "stop_loss": 98,
        "take_profits": [[110, 100]], "confidence": 5.0,
    })
    assert d.confidence == 1.0


def test_extract_json_from_fence():
    text = 'Here is my answer:\n```json\n{"action": "SKIP"}\n```\nThanks.'
    obj = _extract_json(text)
    assert obj == {"action": "SKIP"}


def test_extract_json_bare():
    text = 'Preface text. {"action": "OPEN_LONG", "symbol": "X"} trailing.'
    obj = _extract_json(text)
    assert obj is not None
    assert obj["action"] == "OPEN_LONG"


def test_extract_json_garbage_returns_none():
    assert _extract_json("no json here at all") is None
