"""AI-driven strategy.

The full trading decision is delegated to MiniMax-M2.7 via the Anthropic-compatible
API at https://api.minimax.io/anthropic. The model receives recent OHLCV,
pre-computed indicators, funding rate and account context, and returns a
strict-JSON decision object which is parsed into a Signal.

The caller (bot.py) throttles invocation to once per closed 15-minute candle per
symbol so the 1500-req / 5h quota is never exceeded.

Safety net: whatever the AI returns is still clamped by the RiskManager
(position sizing, daily loss, max drawdown, cooldown, funding guards). The AI
cannot bypass those rules.
"""
from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from loguru import logger

try:
    from anthropic import AsyncAnthropic
    from anthropic import APIError, APIStatusError
except ImportError:  # pragma: no cover
    AsyncAnthropic = None  # type: ignore
    APIError = Exception  # type: ignore
    APIStatusError = Exception  # type: ignore

from ..config import AIConfig, StrategyConfig
from ..indicators import ema, enrich
from .base import Signal, Strategy


SYSTEM_PROMPT = """\
You are the trading brain of an autonomous Binance USDT-M Futures bot. You \
have complete discretion over whether to trade and how. No rules are imposed \
on your strategy — pick whatever approach you think works best for the \
current market.

Environment facts (not strategy instructions):
- The symbol, recent OHLCV, pre-computed indicators, funding rate and account \
context are provided in the user message as JSON.
- The account is very small (~10 USDT). Symbols have minimum notionals \
(~5 USDT) — if no workable plan exists for this size, return "NONE".
- Leverage cap for this request is in `max_leverage`. You may choose anything \
from 1 up to that value.
- Position sizing, daily loss limits, max drawdown and cooldowns are enforced \
by a separate risk layer; you do not need to reason about them.
- For any open trade the bot will execute exactly the stop-loss and \
take-profit ladder you provide. Place them wherever your analysis says.

Output format — reply with a SINGLE JSON object and nothing else. No markdown \
fences, no prose outside the JSON:

{
  "side": "LONG" | "SHORT" | "NONE",
  "confidence": 0.0-1.0,
  "entry_price": <float>,
  "stop_loss": <float>,
  "take_profits": [
      {"price": <float>, "close_pct": <number 1-100>},
      ...
  ],
  "leverage": <int 1..max_leverage>,
  "reasoning": "<one short sentence>"
}

Structural requirements (not strategy rules, just so the executor can run \
your plan):
- If side is LONG, stop_loss must be below entry_price and every take-profit \
price must be above entry_price.
- If side is SHORT, stop_loss must be above entry_price and every take-profit \
price must be below entry_price.
- take_profits may contain 1, 2 or 3 entries. The close_pct values must sum \
to 100 (±1 rounding).
- If you do not want to trade, return side="NONE" and leave numeric fields as 0.
"""


@dataclass
class _AIDecision:
    side: str                           # LONG / SHORT / NONE
    confidence: float
    entry_price: float
    stop_loss: float
    take_profits: List[Tuple[float, float]]   # [(price, close_pct), ...]
    leverage: int
    reasoning: str


class AIStrategy(Strategy):
    """Delegates the entire entry decision to an LLM.

    Retry-safe: transient API errors are retried with exponential backoff.
    Validation-strict: any malformed decision falls back to "NONE" (no trade)
    rather than opening a bad position.
    """

    def __init__(
        self,
        ai_cfg: AIConfig,
        strategy_cfg: StrategyConfig,
        api_key: str,
        base_url: str,
        client: Any = None,  # dependency injection for tests
    ) -> None:
        self.ai = ai_cfg
        self.strat = strategy_cfg
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

    # ---------------- public API ----------------

    async def evaluate(
        self, symbol: str, df15: pd.DataFrame, df1h: pd.DataFrame
    ) -> Optional[Signal]:
        min_bars = max(
            self.strat.ema_slow, self.strat.adx_period,
            self.strat.rsi_period, self.strat.atr_period,
            self.strat.volume_sma_period,
        ) + 5
        if len(df15) < min_bars or len(df1h) < self.strat.ema_htf + 2:
            return None

        d15 = enrich(df15, self.strat)
        d1h = df1h.copy()
        d1h["ema_htf"] = ema(d1h["close"], self.strat.ema_htf)

        # Use the LAST CLOSED candle for "now"; -1 is the forming one.
        last = d15.iloc[-2]
        if last[["ema_fast", "ema_slow", "adx", "rsi", "atr", "vol_sma"]].isna().any():
            return None

        prompt = self._build_user_prompt(symbol, d15, d1h)
        raw = await self._call_ai(prompt)
        if raw is None:
            return None

        decision = self._parse_decision(raw)
        if decision is None:
            logger.warning("{}: AI response did not parse as valid decision", symbol)
            return None

        if decision.side == "NONE":
            logger.info("{}: AI says NO TRADE ({})", symbol, decision.reasoning)
            return None

        if decision.confidence < self.ai.min_confidence:
            logger.info(
                "{}: AI confidence {:.2f} below threshold {:.2f}, skipping",
                symbol, decision.confidence, self.ai.min_confidence,
            )
            return None

        if not self._sanity_check(decision):
            logger.warning(
                "{}: AI decision failed structural check ({}), skipping",
                symbol, decision,
            )
            return None

        # Sort TPs by distance from entry so the risk manager processes them
        # nearest-first (TP1 = closest). The AI is free to supply them in any
        # order; we just normalise. Also pad to 3 entries so the rest of the
        # bot (which indexes tp[0..2]) doesn't need to branch.
        tps_sorted = sorted(
            decision.take_profits, key=lambda t: abs(t[0] - decision.entry_price)
        )
        # Guarantee the furthest real TP fully closes the remainder. Without
        # this, 1- or 2-TP plans could leave a residual position hanging until
        # the trailing stop / time stop / SL fires. Setting close_pct=100 on
        # the last real TP makes bot.py's `close_qty_pct >= 100` branch
        # trigger a full close.
        last_price, _ = tps_sorted[-1]
        tps_sorted[-1] = (last_price, 100.0)
        # Pad to 3 slots so the risk manager's tp[0..2] indexing always works.
        while len(tps_sorted) < 3:
            tps_sorted.append((last_price, 0.0))

        atr_val = float(last["atr"])
        return Signal(
            side=decision.side,  # type: ignore[arg-type]
            entry_price=decision.entry_price,
            atr=atr_val,
            reason=f"AI {self.ai.model}: {decision.reasoning}",
            stop_loss=decision.stop_loss,
            take_profits=tps_sorted,
            leverage=min(decision.leverage, self.ai.max_leverage),
            confidence=decision.confidence,
        )

    # ---------------- prompt ----------------

    def _build_user_prompt(
        self, symbol: str, d15: pd.DataFrame, d1h: pd.DataFrame
    ) -> str:
        # Trim to what the model actually needs; keep token usage bounded.
        n15 = min(self.ai.kline_history, len(d15) - 1)  # -1 to drop forming candle
        n1h = min(self.ai.htf_history, len(d1h) - 1)

        recent15 = d15.iloc[-(n15 + 1):-1]  # drop forming bar
        recent1h = d1h.iloc[-(n1h + 1):-1]

        def _fmt_row(r: pd.Series, has_ind: bool) -> Dict[str, Any]:
            base = {
                "t": r.name.strftime("%Y-%m-%dT%H:%M"),  # type: ignore[union-attr]
                "o": round(float(r["open"]), 8),
                "h": round(float(r["high"]), 8),
                "l": round(float(r["low"]), 8),
                "c": round(float(r["close"]), 8),
                "v": round(float(r["volume"]), 4),
            }
            if has_ind:
                base.update({
                    "ema_f": _safe_round(r.get("ema_fast")),
                    "ema_s": _safe_round(r.get("ema_slow")),
                    "rsi": _safe_round(r.get("rsi"), 2),
                    "adx": _safe_round(r.get("adx"), 2),
                    "atr": _safe_round(r.get("atr"), 8),
                })
            return base

        klines15 = [_fmt_row(r, has_ind=True) for _, r in recent15.iterrows()]
        klines1h = [_fmt_row(r, has_ind=False) for _, r in recent1h.iterrows()]
        # HTF EMA50 context
        last_1h = recent1h.iloc[-1]
        htf_price = float(last_1h["close"])
        htf_ema = _safe_round(last_1h.get("ema_htf"), 8)

        last15 = recent15.iloc[-1]
        current_price = float(last15["close"])
        atr_val = float(last15["atr"])

        payload = {
            "symbol": symbol,
            "timeframe": "15m",
            "current_price": round(current_price, 8),
            "atr_15m": round(atr_val, 8),
            "rsi_15m": _safe_round(last15.get("rsi"), 2),
            "adx_15m": _safe_round(last15.get("adx"), 2),
            "htf_timeframe": "1h",
            "htf_price": round(htf_price, 8),
            "htf_ema50": htf_ema,
            "htf_trend": "bull" if (htf_ema is not None and htf_price > htf_ema)
                         else "bear" if (htf_ema is not None and htf_price < htf_ema)
                         else "flat",
            "max_leverage": self.ai.max_leverage,
            "recent_15m": klines15,
            "recent_1h": klines1h,
        }
        return json.dumps(payload, separators=(",", ":"))

    # ---------------- API call ----------------

    async def _call_ai(self, user_prompt: str) -> Optional[str]:
        last_err: Optional[Exception] = None
        for attempt in range(self.ai.retries + 1):
            try:
                kwargs: Dict[str, Any] = dict(
                    model=self.ai.model,
                    max_tokens=self.ai.max_tokens,
                    system=SYSTEM_PROMPT,
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
            except Exception as e:  # noqa: BLE001
                last_err = e
                logger.exception("Unexpected AI call error: {}", e)
                break
            if attempt < self.ai.retries:
                await asyncio.sleep(2 ** attempt)
        if last_err is not None:
            logger.error("AI call failed after retries: {}", last_err)
        return None

    # ---------------- parsing & validation ----------------

    @staticmethod
    def _parse_decision(raw: str) -> Optional[_AIDecision]:
        obj = _extract_json(raw)
        if obj is None:
            return None
        try:
            side = str(obj.get("side", "NONE")).upper()
            if side not in ("LONG", "SHORT", "NONE"):
                return None
            tps_raw = obj.get("take_profits", []) or []
            tps: List[Tuple[float, float]] = []
            for tp in tps_raw:
                if not isinstance(tp, dict):
                    continue
                tps.append((float(tp["price"]), float(tp["close_pct"])))
            return _AIDecision(
                side=side,
                confidence=float(obj.get("confidence", 0.0)),
                entry_price=float(obj.get("entry_price", 0.0)),
                stop_loss=float(obj.get("stop_loss", 0.0)),
                take_profits=tps,
                leverage=int(obj.get("leverage", 1)),
                reasoning=str(obj.get("reasoning", ""))[:240],
            )
        except (ValueError, TypeError, KeyError) as e:
            logger.warning("AI decision parse error: {}", e)
            return None

    def _sanity_check(self, d: _AIDecision) -> bool:
        """Structural checks only — not strategy rules.

        These are purely physical constraints required for the bot to be able
        to execute the plan at all: prices must be positive, stop-loss must be
        on the losing side of entry, take-profits must be on the winning side,
        and close percentages must add up. Nothing about R:R, trend alignment
        or anything else the model is choosing is enforced here.
        """
        if d.entry_price <= 0 or d.stop_loss <= 0:
            return False
        if d.leverage < 1:
            return False
        if not d.take_profits or not (1 <= len(d.take_profits) <= 3):
            return False
        if any(price <= 0 or pct <= 0 for price, pct in d.take_profits):
            return False
        total = sum(pct for _, pct in d.take_profits)
        if not (99.0 <= total <= 101.0):
            return False

        if d.side == "LONG":
            if d.stop_loss >= d.entry_price:
                return False
            if any(p <= d.entry_price for p, _ in d.take_profits):
                return False
        elif d.side == "SHORT":
            if d.stop_loss <= d.entry_price:
                return False
            if any(p >= d.entry_price for p, _ in d.take_profits):
                return False
        return True


# ---------------- helpers ----------------

def _safe_round(x: Any, digits: int = 8) -> Optional[float]:
    try:
        if x is None:
            return None
        xf = float(x)
        if xf != xf:  # NaN
            return None
        return round(xf, digits)
    except (TypeError, ValueError):
        return None


def _extract_text(msg: Any) -> str:
    """Pull the first 'text' block out of an Anthropic Message, skipping thinking."""
    content = getattr(msg, "content", None) or []
    out_parts: List[str] = []
    for block in content:
        btype = getattr(block, "type", None)
        if btype == "text":
            out_parts.append(getattr(block, "text", "") or "")
    return "".join(out_parts).strip()


_JSON_FENCE = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    """Best-effort JSON extractor. Tolerates markdown fences and surrounding prose."""
    if not text:
        return None
    candidates: List[str] = []
    m = _JSON_FENCE.search(text)
    if m:
        candidates.append(m.group(1).strip())
    # Fallback: substring between first '{' and last '}'
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


__all__ = ["AIStrategy"]
