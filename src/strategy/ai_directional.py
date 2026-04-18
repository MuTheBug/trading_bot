"""AI-driven directional trading strategy.

The model receives rich market context for a shortlist of candidates
(OHLCV tails, indicator snapshots, regime classification, account size)
and returns a single decision:

    * pick ONE symbol, OPEN_LONG or OPEN_SHORT, with entry/SL/TPs/leverage
    * or SKIP (no trade this scan)

Everything the model needs to reason about direction, sizing and exits
is in the payload. The bot validates the output (sides match, SL on the
right side of entry, TPs monotone, leverage within cap) before handing
off to the position manager.

Context fed to the AI per symbol
--------------------------------
* Ticker: price, 24h range/change, volume.
* Exchange filters: min_qty, min_notional, price_tick, qty_step.
* 30 most recent 15m candles (o/h/l/c/vol).
* Last-row indicator snapshot: EMA20/50, RSI14, ADX14, DI+/DI-, ATR14,
  Bollinger upper/mid/lower, BB width, ATR%.
* Regime classifier result: regime label + confidence.
* Higher-timeframe (1h) regime + last 10 candles for context.

Account context
---------------
* Available balance in USDT.
* Hard caps: max_leverage, max_margin_pct, risk_per_trade_pct.
* Recent trade outcomes (last 5 wins/losses) so the model can adjust
  aggression if we've been on a losing streak.
"""
from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

try:
    from anthropic import AsyncAnthropic, APIError, APIStatusError
except ImportError:  # pragma: no cover
    AsyncAnthropic = None  # type: ignore
    APIError = Exception  # type: ignore
    APIStatusError = Exception  # type: ignore

import pandas as pd

from ..config import AIConfig, DirectionalConfig
from ..exchange.base import SymbolFilters, TickerInfo
from ..indicators import adx, atr, ema, rsi, sma
from .regime import classify_regime


AI_SYSTEM_PROMPT = """\
You are the brain of a Binance USDT-M Futures autonomous trading bot.

You receive a shortlist of USDT perpetual candidates with their OHLCV \
history, indicator snapshots, regime classification, and the account \
balance + caps. Decide ONE action:

 1. Open a LONG or SHORT on one symbol with explicit stop-loss and \
    take-profit prices.
 2. Or SKIP — no trade if setups are weak, ambiguous, or dangerous \
    (choppy/low-volume/news-driven).

You can freely take long or short positions. Pick the side with the \
clearest edge, not a default direction.

Rules you MUST follow:
- Price must be close to current price (fills at market). Use the \
  symbol's current `price` as entry.
- stop_loss must be on the LOSS side of entry (below for long, above \
  for short). Distance should reflect actual risk (usually 1.2-2.5x ATR).
- Each take_profit entry is [price, close_pct]. Prices must be ordered \
  away from entry (ascending for long, descending for short). close_pct \
  values sum to 100. Reward:risk on the final TP should be >= 1.5.
- leverage: 0 = let the bot auto-size based on account and SL distance. \
  Otherwise must be in [min_leverage, max_leverage] from the payload.
- confidence: 0.0-1.0 self-assessment. Scores below 0.45 should SKIP.

Think about:
- Market regime (trend vs range vs breakout vs chop) in BOTH 15m and 1h.
- Recent volatility (ATR%) — high vol = tighter size via lower leverage.
- Recent trade outcomes — if on a losing streak, be more selective.
- The symbol's liquidity (volume_M) — low-volume symbols have worse \
  slippage; prefer top-tier pairs for small trade sizes.

Reply with EXACTLY ONE JSON object, no commentary:

{
  "action": "OPEN_LONG" | "OPEN_SHORT" | "SKIP",
  "symbol": "<SYMBOL>",          // required unless SKIP
  "entry": <float>,               // current price
  "stop_loss": <float>,
  "take_profits": [[<price>, <close_pct>], ...],
  "leverage": <int>,              // 0 = auto
  "confidence": <0..1>,
  "reasoning": "<one sentence>"
}

If SKIP: reply with {"action": "SKIP", "reasoning": "<why>"}.
"""


@dataclass
class AIDecision:
    action: str              # OPEN_LONG / OPEN_SHORT / SKIP
    reasoning: str
    symbol: Optional[str] = None
    entry: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profits: List[Tuple[float, float]] = field(default_factory=list)
    leverage: int = 0
    confidence: float = 0.0

    @property
    def is_trade(self) -> bool:
        return self.action in ("OPEN_LONG", "OPEN_SHORT")

    @property
    def side(self) -> Optional[str]:
        if self.action == "OPEN_LONG":
            return "LONG"
        if self.action == "OPEN_SHORT":
            return "SHORT"
        return None


@dataclass
class _CandidateCtx:
    symbol: str
    ticker: TickerInfo
    filters: SymbolFilters
    df15: pd.DataFrame
    df1h: pd.DataFrame

    def payload(self) -> Dict[str, Any]:
        """Build the compact per-symbol payload sent to the AI."""
        d15 = _enrich(self.df15)
        d1h = _enrich(self.df1h)
        if d15 is None or d1h is None:
            return {}
        # Last closed row (second-to-last; -1 is the forming bar).
        last15 = d15.iloc[-2]
        last1h = d1h.iloc[-2]

        def _candles(df: pd.DataFrame, n: int):
            # Take the n most recently CLOSED candles.
            tail = df.iloc[-(n + 1):-1]
            return [
                [
                    round(float(r.open), 8),
                    round(float(r.high), 8),
                    round(float(r.low), 8),
                    round(float(r.close), 8),
                    round(float(r.volume), 2),
                ]
                for r in tail.itertuples(index=False)
            ]

        reg15 = classify_regime(self.df15)
        reg1h = classify_regime(self.df1h)

        return {
            "symbol": self.symbol,
            "price": round(self.ticker.price, 8),
            "change_24h": round(self.ticker.change_pct_24h, 2),
            "high_24h": round(self.ticker.high_24h, 8),
            "low_24h": round(self.ticker.low_24h, 8),
            "volume_M": round(self.ticker.volume_24h / 1e6, 1),
            "filters": {
                "price_tick": self.filters.price_tick,
                "qty_step": self.filters.qty_step,
                "min_qty": self.filters.min_qty,
                "min_notional": self.filters.min_notional,
            },
            "ind_15m": _ind_snapshot(last15),
            "ind_1h": _ind_snapshot(last1h),
            "regime_15m": _regime_payload(reg15),
            "regime_1h": _regime_payload(reg1h),
            "candles_15m": _candles(self.df15, 30),
            "candles_1h": _candles(self.df1h, 20),
        }


def _enrich(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    if df is None or len(df) < 60:
        return None
    d = df.copy()
    d["ema20"] = ema(d["close"], 20)
    d["ema50"] = ema(d["close"], 50)
    d["rsi"] = rsi(d["close"], 14)
    d["atr"] = atr(d, 14)
    adx_df = adx(d, 14)
    d["adx"] = adx_df["adx"]
    d["plus_di"] = adx_df["plus_di"]
    d["minus_di"] = adx_df["minus_di"]
    d["bb_mid"] = sma(d["close"], 20)
    std = d["close"].rolling(20, min_periods=20).std()
    d["bb_low"] = d["bb_mid"] - 2.0 * std
    d["bb_up"] = d["bb_mid"] + 2.0 * std
    return d


def _ind_snapshot(row) -> Dict[str, float]:
    def _f(name, digits=4):
        v = row.get(name)
        if v is None or pd.isna(v):
            return None
        return round(float(v), digits)
    price = _f("close", 8) or 0.0
    atr_val = _f("atr", 8) or 0.0
    return {
        "close": price,
        "ema20": _f("ema20", 8),
        "ema50": _f("ema50", 8),
        "rsi": _f("rsi", 2),
        "adx": _f("adx", 2),
        "plus_di": _f("plus_di", 2),
        "minus_di": _f("minus_di", 2),
        "atr": atr_val,
        "atr_pct": round(atr_val / price * 100.0, 3) if price > 0 else 0.0,
        "bb_low": _f("bb_low", 8),
        "bb_mid": _f("bb_mid", 8),
        "bb_up": _f("bb_up", 8),
    }


def _regime_payload(snap) -> Optional[Dict[str, Any]]:
    if snap is None:
        return None
    return {
        "regime": snap.regime,
        "direction": snap.direction,
        "confidence": round(snap.confidence, 2),
        "adx": round(snap.adx, 2),
        "ema_slope_pct": round(snap.ema_slope_pct, 3),
        "bb_width_pct": round(snap.bb_width_pct, 3),
        "atr_pct": round(snap.atr_pct, 3),
        "rsi": round(snap.rsi, 2),
    }


class AIDirectionalStrategy:
    """Pure-AI trade decision for a shortlist of candidates."""

    def __init__(
        self,
        ai_cfg: AIConfig,
        dir_cfg: DirectionalConfig,
        api_key: str,
        base_url: str,
        client: Any = None,
    ) -> None:
        self.ai = ai_cfg
        self.dir = dir_cfg
        if client is not None:
            self._client = client
        elif AsyncAnthropic is None:
            raise RuntimeError(
                "`anthropic` package not installed. Run ./install.sh or "
                "`pip install anthropic>=0.39`."
            )
        elif not api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY missing. Re-run ./install.sh and provide "
                "your MiniMax Token Plan key."
            )
        else:
            self._client = AsyncAnthropic(api_key=api_key, base_url=base_url)

    async def decide(
        self,
        candidates: List[_CandidateCtx],
        balance: float,
        recent_outcomes: List[Dict[str, Any]],
    ) -> Optional[AIDecision]:
        """Send a shortlist of candidates + account context; return the
        model's decision or None if the call failed."""
        if not candidates:
            return None
        account_payload = {
            "balance_usdt": round(balance, 4),
            "risk_per_trade_pct": self.dir.risk_per_trade_pct,
            "min_leverage": self.dir.min_leverage,
            "max_leverage": self.dir.max_leverage,
            "max_margin_pct": self.dir.max_margin_pct,
            "min_confidence": self.dir.min_confidence,
            "recent_trades": recent_outcomes[-5:],
        }
        cand_payloads = []
        for c in candidates:
            p = c.payload()
            if p:
                cand_payloads.append(p)
        if not cand_payloads:
            return None

        user_payload = {
            "account": account_payload,
            "candidates": cand_payloads,
        }
        prompt = json.dumps(user_payload, separators=(",", ":"))
        raw = await self._call_ai(AI_SYSTEM_PROMPT, prompt)
        if raw is None:
            return None
        obj = _extract_json(raw)
        if obj is None:
            logger.warning("AI directional: unparseable response: {!r}", raw[:200])
            return None
        return _parse_decision(obj)

    async def _call_ai(self, system_prompt: str, user_prompt: str) -> Optional[str]:
        last_err: Optional[Exception] = None
        for attempt in range(self.ai.retries + 1):
            try:
                kwargs: Dict[str, Any] = dict(
                    model=self.ai.model,
                    max_tokens=self.ai.max_tokens,
                    system=system_prompt,
                    messages=[{
                        "role": "user",
                        "content": [{"type": "text", "text": user_prompt}],
                    }],
                )
                if self.ai.thinking:
                    kwargs["thinking"] = {"type": "enabled", "budget_tokens": 1024}
                msg = await asyncio.wait_for(
                    self._client.messages.create(**kwargs),
                    timeout=self.ai.request_timeout_s,
                )
                return _extract_text(msg)
            except (APIStatusError, APIError) as e:
                last_err = e
                logger.warning(
                    "AI call error (attempt {}/{}): {}",
                    attempt + 1, self.ai.retries + 1, e,
                )
            except asyncio.TimeoutError as e:
                last_err = e
                logger.warning(
                    "AI call timed out (attempt {}/{})",
                    attempt + 1, self.ai.retries + 1,
                )
            except Exception as e:
                last_err = e
                logger.exception("Unexpected AI call error: {}", e)
                break
            if attempt < self.ai.retries:
                await asyncio.sleep(2 ** attempt)
        if last_err is not None:
            logger.error("AI directional call failed: {}", last_err)
        return None


# ---------- parsing / validation ----------

_JSON_FENCE = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)


def _extract_text(msg: Any) -> str:
    content = getattr(msg, "content", None) or []
    parts: List[str] = []
    for block in content:
        if getattr(block, "type", None) == "text":
            parts.append(getattr(block, "text", "") or "")
    return "".join(parts).strip()


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    candidates: List[str] = []
    m = _JSON_FENCE.search(text)
    if m:
        candidates.append(m.group(1).strip())
    first = text.find("{")
    last = text.rfind("}")
    if first != -1 and last != -1 and last > first:
        candidates.append(text[first:last + 1])
    candidates.append(text.strip())
    for c in candidates:
        try:
            obj = json.loads(c)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue
    return None


def _parse_decision(obj: Dict[str, Any]) -> AIDecision:
    """Parse + validate the AI response. Invalid fields demote to SKIP."""
    action = str(obj.get("action", "SKIP")).upper().strip()
    reasoning = str(obj.get("reasoning", ""))[:240]

    if action == "SKIP" or action not in ("OPEN_LONG", "OPEN_SHORT"):
        return AIDecision(action="SKIP", reasoning=reasoning or "skip")

    symbol = obj.get("symbol")
    if not isinstance(symbol, str) or not symbol.strip():
        return AIDecision(action="SKIP", reasoning="invalid symbol")
    symbol = symbol.upper()

    try:
        entry = float(obj["entry"])
        stop_loss = float(obj["stop_loss"])
    except (KeyError, TypeError, ValueError):
        return AIDecision(action="SKIP", reasoning="invalid entry/SL")
    if entry <= 0 or stop_loss <= 0:
        return AIDecision(action="SKIP", reasoning="non-positive price")

    is_long = action == "OPEN_LONG"
    if is_long and stop_loss >= entry:
        return AIDecision(action="SKIP", reasoning="SL on wrong side for long")
    if not is_long and stop_loss <= entry:
        return AIDecision(action="SKIP", reasoning="SL on wrong side for short")

    raw_tps = obj.get("take_profits") or []
    tps: List[Tuple[float, float]] = []
    for item in raw_tps:
        try:
            if isinstance(item, dict):
                price = float(item.get("price"))
                pct = float(item.get("close_pct", item.get("pct", 0.0)))
            else:
                price = float(item[0])
                pct = float(item[1])
        except (TypeError, ValueError, IndexError):
            continue
        if price <= 0 or pct <= 0:
            continue
        if is_long and price <= entry:
            continue
        if not is_long and price >= entry:
            continue
        tps.append((price, pct))
    if not tps:
        return AIDecision(action="SKIP", reasoning="no valid take-profits")

    # Sort TPs by increasing distance from entry; rescale close_pct to 100.
    tps.sort(key=lambda p: p[0], reverse=not is_long)
    total = sum(p for _, p in tps)
    if total <= 0:
        return AIDecision(action="SKIP", reasoning="zero close-pct sum")
    tps = [(p, round(pct * 100.0 / total, 4)) for p, pct in tps]

    try:
        leverage = int(obj.get("leverage", 0))
    except (TypeError, ValueError):
        leverage = 0
    leverage = max(0, leverage)

    try:
        confidence = float(obj.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))

    return AIDecision(
        action=action,
        symbol=symbol,
        entry=entry,
        stop_loss=stop_loss,
        take_profits=tps,
        leverage=leverage,
        confidence=confidence,
        reasoning=reasoning,
    )


__all__ = [
    "AIDirectionalStrategy",
    "AIDecision",
    "_CandidateCtx",
    "_parse_decision",
    "_extract_json",
]
