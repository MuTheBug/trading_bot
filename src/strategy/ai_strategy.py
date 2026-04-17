"""AI-assisted grid strategy with mathematical profit guarantees.

The strategy enforces hard mathematical constraints so every round-trip is
profitable after fees. The AI only picks the symbol — all grid parameters
are computed from market data and account constraints.

Profit guarantee per round-trip:
    profit = spacing * qty - 2 * maker_fee * price * qty
    We enforce spacing >= price * fee_mult * maker_fee
    where fee_mult >= 10 (default 12), so each trip nets >= 10x the fee cost.

Symbol selection scoring (computed, not AI-guessed):
    score = (range_pct / max(abs(change_pct), 1.0)) * log10(volume_24h)
    High score = oscillating (high range, low net change) + liquid.
    Trending symbols get penalized by the change_pct denominator AND are
    hard-rejected when abs(change_24h) exceeds the trend threshold.
"""
from __future__ import annotations

import asyncio
import json
import math
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

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
from ..grid.manager import GridSetupParams, round_price, round_qty


# Maker fee on Binance Futures (0.02%)
MAKER_FEE = 0.0002

# Minimum spacing multiplier over fees — each round trip must net at least
# this many times the fee cost. 12x gives ~83% of gross spacing as profit
# even after accounting for slippage and adverse selection.
MIN_FEE_MULT = 12

# Hard rejection: if |24h change| exceeds this, symbol is considered too
# trendy for a neutral grid regardless of range.
MAX_TREND_PCT = 8.0

# Require range to exceed change by at least this factor (oscillation check)
MIN_RANGE_TREND_RATIO = 1.8


SYMBOL_SCAN_PROMPT = """\
You are the brain of an autonomous Binance USDT-M Futures grid trading bot.

Pick THE SINGLE BEST symbol from this pre-scored list for grid trading.
The "score" already ranks symbols by grid-friendliness (oscillation vs trend, \
volume, range). Higher score = better for grids.

Consider:
- Prefer score > 15, but also consider practical factors
- Avoid symbols with extreme recent pumps/dumps (high |change_24h|)
- Prefer symbols where min_notional works with a $10 account
- Prefer symbols you recognize as typically range-bound

Reply with ONLY a JSON object:
{"symbol": "<SYMBOL>", "reasoning": "<one sentence>"}
"""


@dataclass
class SymbolChoice:
    symbol: str
    reasoning: str
    score: float


@dataclass
class GridDecision:
    upper_price: float
    lower_price: float
    num_grids: int
    leverage: int
    qty_per_grid: float
    spacing: float
    profit_per_trip: float
    reasoning: str


@dataclass
class RebalanceDecision:
    action: str  # REBALANCE / HOLD / EXIT
    reasoning: str
    new_params: Optional[GridSetupParams] = None


def score_symbol(t: TickerInfo) -> float:
    """Score a symbol for grid-friendliness.

    High score = high oscillation (range), low trend (net change), high volume.
    """
    if t.price <= 0 or t.volume_24h <= 0:
        return 0.0
    range_pct = (t.high_24h - t.low_24h) / t.price * 100.0
    abs_change = max(abs(t.change_pct_24h), 1.0)  # floor at 1% for stability
    # Oscillation ratio: high range + low net move = good
    osc_ratio = range_pct / abs_change
    # Volume factor: log scale so 100M and 1B aren't worlds apart
    vol_factor = math.log10(max(t.volume_24h, 1))
    return osc_ratio * vol_factor


def is_grid_friendly(t: TickerInfo) -> bool:
    """Hard filter: reject symbols that are clearly trending."""
    if t.price <= 0 or t.volume_24h <= 0:
        return False
    if abs(t.change_pct_24h) > MAX_TREND_PCT:
        return False
    range_pct = (t.high_24h - t.low_24h) / t.price * 100.0
    # Need the range to substantially exceed the net move
    if range_pct < max(abs(t.change_pct_24h) * MIN_RANGE_TREND_RATIO, 1.0):
        return False
    return True


def compute_grid_params(
    price: float,
    ticker: TickerInfo,
    filters: SymbolFilters,
    balance: float,
    max_leverage: int,
    max_grids: int,
    min_grids: int,
    max_capital_pct: float = 75.0,
) -> Optional[GridDecision]:
    """Compute grid parameters mathematically — no AI involved.

    Profitability strategy: previous version always picked the SMALLEST
    qty that met min_notional and then stopped at the LOWEST leverage
    that fit. Result: with a $10 balance on a $0.04 symbol, each trip
    netted ~$0.000004 — below the rounding error of the exchange.

    New approach:
      1. Pick the HIGHEST allowed leverage (bigger qty for same margin)
      2. Scale qty UP to use the full ``max_capital_pct`` capital budget
      3. Round down to lot step, never below min_qty / min_notional

    With per-tick 2% stop-loss, 10x leverage caps max loss at ~20% of
    margin = ~15% of equity at 75% cap — acceptable with tight stops.
    """
    if price <= 0 or balance <= 0:
        return None

    tick = filters.price_tick
    step = filters.qty_step

    # --- 1. Minimum profitable spacing (fee-multiple above maker fee) ---
    min_spacing = price * 2 * MAKER_FEE * MIN_FEE_MULT
    min_spacing = max(min_spacing, tick * 2)

    # --- 2. Grid width from 24h volatility ---
    range_24h = ticker.high_24h - ticker.low_24h
    if range_24h <= 0:
        range_24h = price * 0.02  # fallback: 2%

    # 40% of 24h range — keeps most movements inside the grid while
    # giving enough oscillation room to catch round trips.
    grid_width = range_24h * 0.40
    grid_width = max(grid_width, min_spacing * min_grids)

    # --- 3. Number of grids ---
    num_grids = int(grid_width / min_spacing)
    num_grids = max(min_grids, min(num_grids, max_grids))

    spacing = grid_width / num_grids
    if spacing < min_spacing:
        spacing = min_spacing
        grid_width = spacing * num_grids

    spacing = round_price(spacing, tick)
    if spacing < tick:
        spacing = tick

    # --- 4. Grid bounds centered on price ---
    half_width = (spacing * num_grids) / 2
    lower_price = round_price(price - half_width, tick)
    upper_price = round_price(price + half_width, tick)
    if lower_price <= 0:
        lower_price = tick
        upper_price = round_price(lower_price + spacing * num_grids, tick)

    avg_price = (upper_price + lower_price) / 2
    max_margin = balance * (max_capital_pct / 100.0)

    # --- 5. Leverage + qty ---
    # Use the MAX leverage allowed. Higher leverage means the same margin
    # budget funds a bigger qty — which scales profit linearly. With a
    # 2% hard stop-loss that triggers per-tick, runaway drawdowns are
    # bounded at 2% * leverage of margin, so the leverage ladder now
    # aggressively uses the configured cap.
    leverage = max(1, max_leverage)

    # Floor: smallest qty that meets both min_qty and min_notional
    min_qty_for_notional = (
        filters.min_notional / lower_price if lower_price > 0 else filters.min_qty
    )
    min_qty_for_notional = math.ceil(min_qty_for_notional / step) * step
    qty_floor = max(min_qty_for_notional, filters.min_qty)

    # Ceiling: the biggest qty where total_margin still fits
    # total_margin = num_grids * qty * avg_price / leverage <= max_margin
    qty_ceiling = max_margin * leverage / (num_grids * avg_price)
    qty_ceiling = math.floor(qty_ceiling / step) * step

    if qty_ceiling < qty_floor:
        # Even the smallest viable qty exceeds budget — try shrinking
        # num_grids so capital goes further (at min qty the cheapest
        # config is the one with the fewest levels).
        shrunk = False
        for n in range(num_grids - 1, min_grids - 1, -1):
            qc = max_margin * leverage / (n * avg_price)
            qc = math.floor(qc / step) * step
            if qc >= qty_floor:
                num_grids = n
                grid_width = spacing * num_grids
                half_width = grid_width / 2
                lower_price = round_price(price - half_width, tick)
                upper_price = round_price(price + half_width, tick)
                if lower_price <= 0:
                    lower_price = tick
                    upper_price = round_price(lower_price + spacing * num_grids, tick)
                avg_price = (upper_price + lower_price) / 2
                qty_ceiling = qc
                shrunk = True
                break
        if not shrunk:
            return None

    # Take the largest qty that fits — this is the profit multiplier.
    qty = max(qty_ceiling, qty_floor)
    qty = round_qty(qty, step)
    if qty < filters.min_qty:
        qty = filters.min_qty

    # --- 6. Final validation ---
    if lower_price * qty < filters.min_notional:
        return None
    if upper_price <= lower_price or num_grids < min_grids:
        return None

    fee_per_trip = 2 * MAKER_FEE * price * qty
    profit_per_trip = spacing * qty - fee_per_trip
    if profit_per_trip <= 0:
        return None

    total_margin = num_grids * qty * avg_price / leverage
    # Allow 2% over-budget for rounding; if much more, back off qty by one step
    while total_margin > max_margin * 1.02 and qty - step >= qty_floor:
        qty = round_qty(qty - step, step)
        total_margin = num_grids * qty * avg_price / leverage
    if total_margin > max_margin * 1.02:
        return None

    capital_used_pct = total_margin / balance * 100.0 if balance > 0 else 0.0
    reasoning = (
        f"Grid: {num_grids} lvls, spacing {spacing:.6f} "
        f"({spacing/price*100:.3f}%), qty {qty:g} "
        f"(profit/trip {profit_per_trip:.6f} USDT), "
        f"leverage {leverage}x, margin {total_margin:.2f}/{max_margin:.2f} USDT "
        f"({capital_used_pct:.0f}% of balance)"
    )

    return GridDecision(
        upper_price=upper_price,
        lower_price=lower_price,
        num_grids=num_grids,
        leverage=leverage,
        qty_per_grid=qty,
        spacing=spacing,
        profit_per_trip=profit_per_trip,
        reasoning=reasoning,
    )


class AIGridStrategy:
    """Math-first grid strategy. AI only assists with symbol selection."""

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
        """Score all symbols mathematically, then let AI pick from the top candidates."""
        # Pre-filter: USDT perpetuals with decent volume, tradeable, and
        # not clearly trending.
        candidates = [
            t for t in tickers
            if t.volume_24h > 20_000_000
            and t.symbol in all_filters
            and t.price > 0
            and t.symbol.endswith("USDT")
            and is_grid_friendly(t)
        ]

        if not candidates:
            logger.warning(
                "No grid-friendly symbols after filtering — every candidate is trending."
            )
            return None

        # Score each symbol
        scored = [(t, score_symbol(t)) for t in candidates]
        scored.sort(key=lambda x: x[1], reverse=True)

        # Take top 15 for AI to choose from
        top = scored[:15]

        # Log the scores
        for t, s in top[:5]:
            logger.info("  {} score={:.1f} range={:.2f}% chg={:.2f}% vol={:.0f}M",
                        t.symbol, s,
                        (t.high_24h - t.low_24h) / t.price * 100,
                        t.change_pct_24h,
                        t.volume_24h / 1e6)

        # Build compact data for AI
        ticker_data = [
            {
                "s": t.symbol,
                "price": t.price,
                "score": round(s, 1),
                "range_pct": round((t.high_24h - t.low_24h) / t.price * 100, 2),
                "change_24h": round(t.change_pct_24h, 2),
                "vol_M": round(t.volume_24h / 1e6),
                "min_notional": all_filters[t.symbol].min_notional,
            }
            for t, s in top
        ]

        prompt = json.dumps(ticker_data, separators=(",", ":"))
        raw = await self._call_ai(SYMBOL_SCAN_PROMPT, prompt)

        # Fallback: if AI fails, just pick the highest-scored symbol
        if raw is None:
            best_t, best_s = top[0]
            logger.warning("AI call failed, using top-scored symbol: {}", best_t.symbol)
            return SymbolChoice(
                symbol=best_t.symbol,
                reasoning=f"Highest grid score ({best_s:.1f})",
                score=best_s,
            )

        obj = _extract_json(raw)
        if obj is None or "symbol" not in obj:
            best_t, best_s = top[0]
            return SymbolChoice(
                symbol=best_t.symbol,
                reasoning=f"Highest grid score ({best_s:.1f})",
                score=best_s,
            )

        symbol = str(obj["symbol"]).upper()
        valid = {t.symbol: (t, s) for t, s in top}
        if symbol not in valid:
            best_t, best_s = top[0]
            return SymbolChoice(
                symbol=best_t.symbol,
                reasoning=f"Highest grid score ({best_s:.1f})",
                score=best_s,
            )

        t, s = valid[symbol]
        return SymbolChoice(
            symbol=symbol,
            reasoning=str(obj.get("reasoning", ""))[:240],
            score=s,
        )

    # ---- 2. Grid parameter computation (math, no AI) ----

    def compute_params(
        self,
        symbol: str,
        current_price: float,
        ticker: TickerInfo,
        filters: SymbolFilters,
        balance: float,
    ) -> Optional[GridDecision]:
        """Compute grid parameters mathematically."""
        return compute_grid_params(
            price=current_price,
            ticker=ticker,
            filters=filters,
            balance=balance,
            max_leverage=min(self.ai.max_leverage, self.grid.max_leverage),
            max_grids=self.grid.max_grids,
            min_grids=self.grid.min_grids,
            max_capital_pct=self.grid.max_capital_pct,
        )

    # ---- 3. Rebalance (pure math — just re-center the grid) ----

    def compute_rebalance(
        self,
        symbol: str,
        current_price: float,
        ticker: TickerInfo,
        filters: SymbolFilters,
        balance: float,
        grid_summary: dict,
    ) -> RebalanceDecision:
        """Decide whether to rebalance, hold, or exit.

        Rules:
        - If the symbol has flipped into a strong trend (|24h change|
          above threshold), EXIT — don't fight the trend.
        - If compute_grid_params can't find viable params, HOLD.
        - If price is in range, HOLD — existing grid orders handle it.
        - Otherwise REBALANCE around current price. Closing the naked
          inventory is handled in GridManager.teardown() so each
          rebalance starts flat.
        """
        upper = grid_summary["upper"]
        lower = grid_summary["lower"]
        total_profit = grid_summary["total_profit"]
        total_fees = grid_summary["total_fees"]
        # total_profit is already net of fees (see state.record_grid_fill)
        net_profit = total_profit

        # Strong trend → EXIT rather than keep grid-trading into a loss
        if abs(ticker.change_pct_24h) > MAX_TREND_PCT:
            return RebalanceDecision(
                action="EXIT",
                reasoning=(
                    f"Symbol now trending ({ticker.change_pct_24h:+.2f}% 24h). "
                    f"Exiting to avoid fighting the move. Net so far: {net_profit:+.6f}"
                ),
            )

        # Still in range? Let the existing orders work.
        if lower < current_price < upper:
            return RebalanceDecision(
                action="HOLD",
                reasoning="Price back inside grid range; no rebalance needed.",
            )

        decision = compute_grid_params(
            price=current_price,
            ticker=ticker,
            filters=filters,
            balance=balance,
            max_leverage=min(self.ai.max_leverage, self.grid.max_leverage),
            max_grids=self.grid.max_grids,
            min_grids=self.grid.min_grids,
            max_capital_pct=self.grid.max_capital_pct,
        )

        if decision is None:
            return RebalanceDecision(
                action="HOLD",
                reasoning="Cannot compute viable grid params at current price",
            )

        new_params = GridSetupParams(
            symbol=symbol,
            upper_price=decision.upper_price,
            lower_price=decision.lower_price,
            num_grids=decision.num_grids,
            leverage=decision.leverage,
            qty_per_grid=decision.qty_per_grid,
            reasoning=(
                f"Re-centered grid. Previous net: {net_profit:+.6f} USDT. "
                f"{decision.reasoning}"
            ),
        )

        return RebalanceDecision(
            action="REBALANCE",
            reasoning=(
                f"Re-centering around {current_price:.6f}. "
                f"Net profit so far: {net_profit:+.6f}"
            ),
            new_params=new_params,
        )

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
    content = getattr(msg, "content", None) or []
    out_parts: List[str] = []
    for block in content:
        btype = getattr(block, "type", None)
        if btype == "text":
            out_parts.append(getattr(block, "text", "") or "")
    return "".join(out_parts).strip()


_JSON_FENCE = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)


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


__all__ = [
    "AIGridStrategy", "SymbolChoice", "GridDecision", "RebalanceDecision",
    "score_symbol", "is_grid_friendly", "compute_grid_params",
]
