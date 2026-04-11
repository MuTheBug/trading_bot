"""AI strategy tests — exercise parsing, validation and the evaluate() pipeline
with a stubbed Anthropic client so no network calls happen.
"""
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.config import AIConfig, StrategyConfig
from src.strategy.ai_strategy import AIStrategy, _extract_json, _AIDecision


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


def _make_df(n: int = 100, start: float = 100.0, step: float = 0.1) -> pd.DataFrame:
    idx = pd.date_range("2026-01-01", periods=n, freq="15min", tz="UTC")
    closes = np.array([start + i * step for i in range(n)])
    highs = closes + 0.2
    lows = closes - 0.2
    opens = closes - 0.05
    volumes = np.full(n, 1000.0)
    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes, "volume": volumes},
        index=idx,
    )


def _make_htf(n: int = 80, start: float = 100.0) -> pd.DataFrame:
    idx = pd.date_range("2026-01-01", periods=n, freq="1h", tz="UTC")
    closes = np.array([start + i * 0.5 for i in range(n)])
    return pd.DataFrame(
        {
            "open": closes - 0.1,
            "high": closes + 0.3,
            "low": closes - 0.3,
            "close": closes,
            "volume": np.full(n, 2000.0),
        },
        index=idx,
    )


def _strategy_with_reply(reply: str) -> AIStrategy:
    return AIStrategy(
        ai_cfg=AIConfig(min_confidence=0.5, retries=0),
        strategy_cfg=StrategyConfig(),
        api_key="test",
        base_url="https://example.invalid",
        client=_StubClient(reply),
    )


def test_extract_json_plain():
    assert _extract_json('{"a": 1}') == {"a": 1}


def test_extract_json_with_fence():
    text = "sure:\n```json\n{\"side\": \"LONG\"}\n```\nthanks"
    assert _extract_json(text) == {"side": "LONG"}


def test_extract_json_with_prose():
    text = "here is my decision {\"side\": \"SHORT\", \"confidence\": 0.8}. thanks!"
    out = _extract_json(text)
    assert out is not None and out["side"] == "SHORT"


def test_extract_json_garbage():
    assert _extract_json("not json at all") is None


def test_parse_decision_rejects_invalid_side():
    assert AIStrategy._parse_decision('{"side": "MAYBE"}') is None


def test_sanity_check_rejects_sl_on_wrong_side():
    s = _strategy_with_reply("{}")
    bad = _AIDecision(
        side="LONG", confidence=0.9, entry_price=100.0, stop_loss=105.0,
        take_profits=[(101.0, 40), (102.0, 30), (103.0, 30)], leverage=3, reasoning="x",
    )
    assert s._sanity_check(bad) is False


def test_sanity_check_accepts_unordered_tps():
    # AI provides TPs in any order — they are still structurally valid.
    # The strategy will sort them by distance before handing to the bot.
    s = _strategy_with_reply("{}")
    mixed = _AIDecision(
        side="LONG", confidence=0.9, entry_price=100.0, stop_loss=98.0,
        take_profits=[(108.0, 30), (101.5, 40), (104.0, 30)], leverage=3, reasoning="x",
    )
    assert s._sanity_check(mixed) is True


def test_sanity_check_rejects_close_pct_sum_mismatch():
    s = _strategy_with_reply("{}")
    bad = _AIDecision(
        side="LONG", confidence=0.9, entry_price=100.0, stop_loss=98.0,
        take_profits=[(101.0, 40), (102.0, 30), (103.0, 20)], leverage=3, reasoning="x",
    )
    assert s._sanity_check(bad) is False


def test_sanity_check_accepts_valid_long():
    s = _strategy_with_reply("{}")
    good = _AIDecision(
        side="LONG", confidence=0.9, entry_price=100.0, stop_loss=98.0,
        take_profits=[(101.5, 40), (104.0, 30), (108.0, 30)], leverage=3, reasoning="x",
    )
    assert s._sanity_check(good) is True


def test_sanity_check_accepts_low_rr():
    # A tight TP1 is the AI's choice — we no longer enforce any R:R floor.
    s = _strategy_with_reply("{}")
    tight = _AIDecision(
        side="LONG", confidence=0.9, entry_price=100.0, stop_loss=98.0,
        take_profits=[(100.5, 40), (104.0, 30), (108.0, 30)], leverage=3, reasoning="x",
    )
    assert s._sanity_check(tight) is True


def test_sanity_check_accepts_single_tp():
    # 1 TP at 100% close is a valid all-in exit plan.
    s = _strategy_with_reply("{}")
    single = _AIDecision(
        side="SHORT", confidence=0.9, entry_price=100.0, stop_loss=102.0,
        take_profits=[(95.0, 100)], leverage=2, reasoning="x",
    )
    assert s._sanity_check(single) is True


def test_sanity_check_rejects_tp_on_wrong_side_for_long():
    # TP below entry on a LONG is physically impossible to hit as profit.
    s = _strategy_with_reply("{}")
    bad = _AIDecision(
        side="LONG", confidence=0.9, entry_price=100.0, stop_loss=98.0,
        take_profits=[(95.0, 40), (104.0, 30), (108.0, 30)], leverage=3, reasoning="x",
    )
    assert s._sanity_check(bad) is False


@pytest.mark.asyncio
async def test_evaluate_happy_path_long():
    reply = json.dumps({
        "side": "LONG",
        "confidence": 0.8,
        "entry_price": 109.9,
        "stop_loss": 108.4,
        "take_profits": [
            {"price": 112.0, "close_pct": 40},
            {"price": 115.0, "close_pct": 30},
            {"price": 120.0, "close_pct": 30},
        ],
        "leverage": 3,
        "reasoning": "clean uptrend",
    })
    s = _strategy_with_reply(reply)
    sig = await s.evaluate("DOGEUSDT", _make_df(), _make_htf())
    assert sig is not None
    assert sig.side == "LONG"
    assert sig.leverage == 3
    assert sig.stop_loss == 108.4
    assert sig.take_profits is not None and len(sig.take_profits) == 3
    assert sig.confidence == 0.8
    assert "clean uptrend" in sig.reason


@pytest.mark.asyncio
async def test_evaluate_none_decision_returns_none():
    reply = json.dumps({
        "side": "NONE", "confidence": 0.0,
        "entry_price": 0, "stop_loss": 0,
        "take_profits": [],
        "leverage": 1, "reasoning": "no edge",
    })
    s = _strategy_with_reply(reply)
    sig = await s.evaluate("DOGEUSDT", _make_df(), _make_htf())
    assert sig is None


@pytest.mark.asyncio
async def test_evaluate_low_confidence_skipped():
    reply = json.dumps({
        "side": "LONG", "confidence": 0.3,
        "entry_price": 109.9, "stop_loss": 108.4,
        "take_profits": [
            {"price": 112.0, "close_pct": 40},
            {"price": 115.0, "close_pct": 30},
            {"price": 120.0, "close_pct": 30},
        ],
        "leverage": 3, "reasoning": "meh",
    })
    s = _strategy_with_reply(reply)
    sig = await s.evaluate("DOGEUSDT", _make_df(), _make_htf())
    assert sig is None


@pytest.mark.asyncio
async def test_evaluate_clamps_to_max_leverage():
    reply = json.dumps({
        "side": "LONG", "confidence": 0.9,
        "entry_price": 109.9, "stop_loss": 108.4,
        "take_profits": [
            {"price": 112.0, "close_pct": 40},
            {"price": 115.0, "close_pct": 30},
            {"price": 120.0, "close_pct": 30},
        ],
        "leverage": 50, "reasoning": "aggressive",
    })
    s = _strategy_with_reply(reply)
    sig = await s.evaluate("DOGEUSDT", _make_df(), _make_htf())
    assert sig is not None
    assert sig.leverage == s.ai.max_leverage


@pytest.mark.asyncio
async def test_evaluate_bad_json_returns_none():
    s = _strategy_with_reply("sorry I cannot help")
    sig = await s.evaluate("DOGEUSDT", _make_df(), _make_htf())
    assert sig is None


@pytest.mark.asyncio
async def test_evaluate_sorts_tps_by_distance():
    # AI returns TPs in arbitrary order — strategy must sort them by
    # distance from entry so TP1 is nearest, TP3 is furthest.
    reply = json.dumps({
        "side": "LONG", "confidence": 0.9,
        "entry_price": 109.9, "stop_loss": 108.4,
        "take_profits": [
            {"price": 120.0, "close_pct": 30},   # furthest
            {"price": 112.0, "close_pct": 40},   # nearest
            {"price": 115.0, "close_pct": 30},   # middle
        ],
        "leverage": 3, "reasoning": "three targets out of order",
    })
    s = _strategy_with_reply(reply)
    sig = await s.evaluate("DOGEUSDT", _make_df(), _make_htf())
    assert sig is not None
    assert sig.take_profits is not None
    prices = [p for p, _ in sig.take_profits]
    assert prices == [112.0, 115.0, 120.0]


@pytest.mark.asyncio
async def test_evaluate_pads_single_tp_to_three_slots():
    reply = json.dumps({
        "side": "LONG", "confidence": 0.9,
        "entry_price": 109.9, "stop_loss": 108.4,
        "take_profits": [{"price": 115.0, "close_pct": 100}],
        "leverage": 3, "reasoning": "single target",
    })
    s = _strategy_with_reply(reply)
    sig = await s.evaluate("DOGEUSDT", _make_df(), _make_htf())
    assert sig is not None
    assert sig.take_profits is not None
    assert len(sig.take_profits) == 3
    # First TP carries the full 100% close; the fillers are 0%.
    assert sig.take_profits[0] == (115.0, 100)
    assert sig.take_profits[1][1] == 0
    assert sig.take_profits[2][1] == 0
