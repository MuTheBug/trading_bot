"""AI-driven grid strategy.

The AI (MiniMax-M2.7 via Anthropic-compatible API) performs two jobs:

1. **Symbol selection** — scans all Binance USDT-M perpetual tickers, picks
   the best symbol for grid trading based on volatility, volume, and price
   range.

2. **Grid parameter decision** — given the chosen symbol's recent price data,
   the AI decides upper/lower bounds, number of grid levels, leverage, and
   qty per grid, respecting the $10 account size and Binance minimum notional
   constraints.

3. **Rebalance evaluation** — when the current price moves out of the grid
   range, the AI is asked whether to tear down and rebuild with new params
   or to hold.
"""
from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

try:
    from anthropic import AsyncAnthropic
    from anthropic import APIError, APIStatusError
except ImportError:  # pragma: no cover
    AsyncAnthropic = None  # type: ignore
    APIError = Exception  # type: ignore
    APIStatusError = Exception  # type: ignore

from ..config import AIConfig, GridConfig
from ..exchange.base import TickerInfo, SymbolFilters
from ..grid.manager import GridSetupParams


# ---- system prompts ----

SYMBOL_SCAN_PROMPT = """\
You are the brain of an autonomous Binance USDT-M Futures grid trading bot.

Your task: pick THE SINGLE BEST symbol for a neutral grid strategy right now.

A neutral grid places BUY limit orders below the current price and SELL limit \
orders above. Profit comes from price oscillating within the grid range. The \
ideal symbol has:
- HIGH 24h volume (>50M USDT preferred) for fills
- MODERATE volatility — price oscillates within a range, not trending hard
- Price and filters that work with a $10 account (min notional ~5 USDT)

You will receive a JSON list of the top symbols by volume with their 24h stats.

Reply with a SINGLE JSON object and nothing else:
{
  "symbol": "<SYMBOL>",
  "reasoning": "<one sentence why this symbol is best for grid trading right now>"
}
"""

GRID_PARAMS_PROMPT = """\
You are the brain of an autonomous Binance USDT-M Futures grid trading bot.

Your task: decide the grid parameters for {symbol}.

Account context:
- Available balance: {balance:.4f} USDT
- Maximum leverage allowed: {max_leverage}x
- With leverage, max notional = balance * leverage
- Each grid level needs qty * price >= min_notional ({min_notional} USDT)
- Minimum order qty: {min_qty}, qty step: {qty_step}, price tick: {price_tick}
- Grid levels allowed: {min_grids} to {max_grids}

Current market data:
- Current price: {current_price}
- 24h high: {high_24h}
- 24h low: {low_24h}
- 24h volume: {volume_24h:.0f} USDT
- 24h change: {change_pct:.2f}%

Recent 15m OHLCV (last {n_candles} candles):
{klines_json}

IMPORTANT constraints for a $10 account:
- Total margin used = num_grids * qty_per_grid * avg_price / leverage
- This total margin MUST be <= {balance:.4f} USDT (your available balance)
- Each grid level: qty_per_grid * grid_price >= {min_notional} USDT
- Choose leverage wisely: higher leverage = more grid levels possible but more \
risk. For a $10 account, 5-10x leverage is typical.
- qty_per_grid must be a multiple of {qty_step} and >= {min_qty}
- Prices must be multiples of {price_tick}
- Grid range should capture the likely oscillation range (often 1-3% for \
15m-1h timeframes on crypto)

Reply with a SINGLE JSON object and nothing else:
{{
  "upper_price": <float>,
  "lower_price": <float>,
  "num_grids": <int {min_grids}-{max_grids}>,
  "leverage": <int 1-{max_leverage}>,
  "qty_per_grid": <float>,
  "reasoning": "<one sentence explaining your grid setup>"
}}
"""

REBALANCE_PROMPT = """\
You are the brain of an autonomous Binance USDT-M Futures grid trading bot.

The current grid on {symbol} may need rebalancing. Price has moved {direction} \
of the grid range.

Current grid:
- Range: {lower_price} - {upper_price}
- Levels: {num_grids}, Spacing: {spacing:.8f}
- Leverage: {leverage}x, Qty/grid: {qty_per_grid}
- Realized profit: {total_profit:.6f} USDT
- Round trips: {round_trips}
- Net position qty: {net_qty:+.8f}
- Unrealized PnL: {unrealized_pnl:+.6f} USDT

Current price: {current_price}
Balance: {balance:.4f} USDT

Should we:
1. "REBALANCE" — tear down and set up a new grid around the current price
2. "HOLD" — keep the current grid (price might return to range)

Reply with a SINGLE JSON object:
{{
  "action": "REBALANCE" | "HOLD",
  "reasoning": "<one sentence>",
  "new_upper": <float or null>,
  "new_lower": <float or null>,
  "new_num_grids": <int or null>,
  "new_leverage": <int or null>,
  "new_qty_per_grid": <float or null>
}}

If action is HOLD, set all new_* fields to null.
If action is REBALANCE, provide new grid parameters following the same \
constraints as initial setup. Balance={balance:.4f}, min_notional={min_notional}, \
min_qty={min_qty}, qty_step={qty_step}, price_tick={price_tick}, \
max_leverage={max_leverage}.
"""


@dataclass
class SymbolChoice:
    symbol: str
    reasoning: str


@dataclass
class GridDecision:
    upper_price: float
    lower_price: float
    num_grids: int
    leverage: int
    qty_per_grid: float
    reasoning: str


@dataclass
class RebalanceDecision:
    action: str  # REBALANCE or HOLD
    reasoning: str
    new_params: Optional[GridSetupParams] = None


class AIGridStrategy:
    """AI-driven grid parameter selection."""

    def __init__(
        self,
        ai_cfg: AIConfig,
        grid_cfg: GridConfig,
        api_key: str,
        base_url: str,
        client: Any = None,
    ) -> None:
        self.ai = ai_cfg
        self.grid = grid_cfg
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

    # ---- 1. Symbol selection ----

    async def select_symbol(
        self, tickers: List[TickerInfo], all_filters: Dict[str, SymbolFilters],
    ) -> Optional[SymbolChoice]:
        """Ask AI to pick the best symbol from the scanned tickers."""
        # Pre-filter: only USDT perpetuals with decent volume
        candidates = [
            t for t in tickers
            if t.volume_24h > 10_000_000
            and t.symbol in all_filters
            and t.price > 0
        ]
        # Sort by volume descending, take top 30
        candidates.sort(key=lambda t: t.volume_24h, reverse=True)
        candidates = candidates[:30]

        if not candidates:
            logger.error("No viable symbols found after filtering")
            return None

        # Build compact JSON for the AI
        ticker_data = [
            {
                "s": t.symbol,
                "p": t.price,
                "v24h": round(t.volume_24h),
                "chg": round(t.change_pct_24h, 2),
                "h": t.high_24h,
                "l": t.low_24h,
                "range_pct": round(
                    (t.high_24h - t.low_24h) / t.price * 100, 2
                ) if t.price > 0 else 0,
                "min_notional": all_filters[t.symbol].min_notional,
                "min_qty": all_filters[t.symbol].min_qty,
            }
            for t in candidates
        ]

        prompt = json.dumps(ticker_data, separators=(",", ":"))
        raw = await self._call_ai(SYMBOL_SCAN_PROMPT, prompt)
        if raw is None:
            return None

        obj = _extract_json(raw)
        if obj is None or "symbol" not in obj:
            logger.warning("AI symbol selection response did not parse")
            return None

        symbol = str(obj["symbol"]).upper()
        # Validate the AI actually picked one of our candidates
        valid_symbols = {t.symbol for t in candidates}
        if symbol not in valid_symbols:
            logger.warning("AI picked {} which is not in candidates", symbol)
            return None

        return SymbolChoice(
            symbol=symbol,
            reasoning=str(obj.get("reasoning", ""))[:240],
        )

    # ---- 2. Grid parameter decision ----

    async def decide_grid_params(
        self,
        symbol: str,
        current_price: float,
        ticker: TickerInfo,
        filters: SymbolFilters,
        balance: float,
        klines_15m: Optional[Any] = None,
    ) -> Optional[GridDecision]:
        """Ask AI to decide grid parameters for the chosen symbol."""
        # Build klines snippet
        klines_json = "[]"
        n_candles = 0
        if klines_15m is not None and len(klines_15m) > 2:
            n = min(self.ai.kline_history, len(klines_15m) - 1)
            recent = klines_15m.iloc[-(n + 1):-1]
            n_candles = len(recent)
            rows = []
            for _, r in recent.iterrows():
                rows.append({
                    "t": r.name.strftime("%H:%M") if hasattr(r.name, "strftime") else str(r.name),
                    "o": round(float(r["open"]), 8),
                    "h": round(float(r["high"]), 8),
                    "l": round(float(r["low"]), 8),
                    "c": round(float(r["close"]), 8),
                    "v": round(float(r["volume"]), 2),
                })
            klines_json = json.dumps(rows, separators=(",", ":"))

        prompt = GRID_PARAMS_PROMPT.format(
            symbol=symbol,
            balance=balance,
            max_leverage=self.ai.max_leverage,
            min_notional=filters.min_notional,
            min_qty=filters.min_qty,
            qty_step=filters.qty_step,
            price_tick=filters.price_tick,
            min_grids=self.grid.min_grids,
            max_grids=self.grid.max_grids,
            current_price=current_price,
            high_24h=ticker.high_24h,
            low_24h=ticker.low_24h,
            volume_24h=ticker.volume_24h,
            change_pct=ticker.change_pct_24h,
            n_candles=n_candles,
            klines_json=klines_json,
        )

        raw = await self._call_ai(prompt, "Decide the grid parameters.")
        if raw is None:
            return None

        obj = _extract_json(raw)
        if obj is None:
            logger.warning("AI grid params response did not parse")
            return None

        try:
            decision = GridDecision(
                upper_price=float(obj["upper_price"]),
                lower_price=float(obj["lower_price"]),
                num_grids=int(obj["num_grids"]),
                leverage=int(obj["leverage"]),
                qty_per_grid=float(obj["qty_per_grid"]),
                reasoning=str(obj.get("reasoning", ""))[:240],
            )
        except (ValueError, TypeError, KeyError) as e:
            logger.warning("AI grid params parse error: {}", e)
            return None

        # Validate
        if not self._validate_grid_decision(decision, filters, balance, current_price):
            return None

        return decision

    def _validate_grid_decision(
        self,
        d: GridDecision,
        filters: SymbolFilters,
        balance: float,
        current_price: float,
    ) -> bool:
        """Structural validation of AI grid decision."""
        if d.upper_price <= d.lower_price:
            logger.warning("AI grid: upper <= lower ({} <= {})", d.upper_price, d.lower_price)
            return False
        if d.num_grids < self.grid.min_grids or d.num_grids > self.grid.max_grids:
            logger.warning("AI grid: num_grids {} out of range [{}, {}]",
                          d.num_grids, self.grid.min_grids, self.grid.max_grids)
            return False
        if d.leverage < 1 or d.leverage > self.ai.max_leverage:
            logger.warning("AI grid: leverage {} out of range [1, {}]",
                          d.leverage, self.ai.max_leverage)
            return False
        if d.qty_per_grid < filters.min_qty:
            logger.warning("AI grid: qty {} below min_qty {}", d.qty_per_grid, filters.min_qty)
            return False
        # Check that the grid range contains the current price
        if current_price < d.lower_price or current_price > d.upper_price:
            logger.warning("AI grid: current price {} outside grid [{}, {}]",
                          current_price, d.lower_price, d.upper_price)
            return False
        # Check min notional at lowest price
        if d.lower_price * d.qty_per_grid < filters.min_notional * 0.9:
            logger.warning("AI grid: notional at lower bound too small")
            return False
        # Check total margin doesn't exceed balance
        avg_price = (d.upper_price + d.lower_price) / 2
        total_margin = d.num_grids * d.qty_per_grid * avg_price / d.leverage
        if total_margin > balance * 1.5:  # allow some slack
            logger.warning("AI grid: total margin {:.4f} exceeds balance {:.4f}",
                          total_margin, balance)
            return False
        return True

    # ---- 3. Rebalance evaluation ----

    async def evaluate_rebalance(
        self,
        symbol: str,
        grid_summary: dict,
        balance: float,
        filters: SymbolFilters,
    ) -> RebalanceDecision:
        """Ask AI whether to rebalance the grid."""
        current_price = grid_summary["mark_price"]
        direction = "above" if current_price > grid_summary["upper"] else "below"
        spacing = grid_summary.get("spacing", 0)

        prompt = REBALANCE_PROMPT.format(
            symbol=symbol,
            direction=direction,
            lower_price=grid_summary["lower"],
            upper_price=grid_summary["upper"],
            num_grids=grid_summary["num_grids"],
            spacing=spacing,
            leverage=grid_summary["leverage"],
            qty_per_grid=grid_summary["qty_per_grid"],
            total_profit=grid_summary["total_profit"],
            round_trips=grid_summary["round_trips"],
            net_qty=grid_summary["net_qty"],
            unrealized_pnl=grid_summary["unrealized_pnl"],
            current_price=current_price,
            balance=balance,
            min_notional=filters.min_notional,
            min_qty=filters.min_qty,
            qty_step=filters.qty_step,
            price_tick=filters.price_tick,
            max_leverage=self.ai.max_leverage,
        )

        raw = await self._call_ai(prompt, "Evaluate rebalance.")
        if raw is None:
            return RebalanceDecision(action="HOLD", reasoning="AI call failed")

        obj = _extract_json(raw)
        if obj is None:
            return RebalanceDecision(action="HOLD", reasoning="AI response did not parse")

        action = str(obj.get("action", "HOLD")).upper()
        reasoning = str(obj.get("reasoning", ""))[:240]

        if action == "REBALANCE":
            try:
                new_params = GridSetupParams(
                    symbol=symbol,
                    upper_price=float(obj["new_upper"]),
                    lower_price=float(obj["new_lower"]),
                    num_grids=int(obj["new_num_grids"]),
                    leverage=int(obj["new_leverage"]),
                    qty_per_grid=float(obj["new_qty_per_grid"]),
                    reasoning=reasoning,
                )
                return RebalanceDecision(
                    action="REBALANCE",
                    reasoning=reasoning,
                    new_params=new_params,
                )
            except (ValueError, TypeError, KeyError) as e:
                logger.warning("AI rebalance params parse error: {}", e)
                return RebalanceDecision(action="HOLD", reasoning=f"Parse error: {e}")

        return RebalanceDecision(action="HOLD", reasoning=reasoning)

    # ---- API call ----

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
            logger.error("AI call failed after retries: {}", last_err)
        return None


# ---- helpers ----

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


__all__ = ["AIGridStrategy", "SymbolChoice", "GridDecision", "RebalanceDecision"]
