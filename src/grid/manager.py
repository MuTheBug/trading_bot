"""Neutral grid manager — places and manages a grid of limit orders.

The grid is a set of evenly-spaced price levels between a lower and upper bound.
Below the current price, BUY limit orders are placed; above, SELL limit orders.
When a BUY fills, a corresponding SELL is placed one level up (take profit).
When a SELL fills, a corresponding BUY is placed one level down.
Each round-trip between adjacent levels earns the grid spacing minus fees.

The grid parameters (symbol, bounds, levels, leverage, qty) are decided by the
AI strategy. This module handles only the mechanical execution.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from loguru import logger

from ..exchange.base import ExchangeInterface, LimitOrder, SymbolFilters
from ..state import GridState, GridLevelState, StateStore, TradeRecord, _now_iso


@dataclass
class GridSetupParams:
    """AI-provided grid parameters."""
    symbol: str
    upper_price: float
    lower_price: float
    num_grids: int
    leverage: int
    qty_per_grid: float
    reasoning: str = ""


def compute_grid_prices(lower: float, upper: float, num_grids: int) -> List[float]:
    """Return num_grids + 1 evenly-spaced price levels."""
    if num_grids < 1 or upper <= lower:
        return []
    step = (upper - lower) / num_grids
    return [lower + i * step for i in range(num_grids + 1)]


def round_price(price: float, tick: float) -> float:
    """Round a price to the nearest tick size."""
    if tick <= 0:
        return price
    return round(round(price / tick) * tick, 12)


def round_qty(qty: float, step: float) -> float:
    """Round qty down to the nearest step size."""
    if step <= 0:
        return qty
    return math.floor(qty / step + 1e-9) * step


class GridManager:
    """Manages a single grid on one symbol."""

    def __init__(self, exchange: ExchangeInterface, state: StateStore) -> None:
        self.exchange = exchange
        self.state = state

    @property
    def grid(self) -> GridState:
        return self.state.state.grid

    @property
    def active(self) -> bool:
        return self.grid.active and self.grid.symbol != ""

    async def setup_grid(
        self,
        params: GridSetupParams,
        filters: SymbolFilters,
        current_price: float,
    ) -> bool:
        """Set up a new grid. Cancels any existing grid first.

        Returns True if the grid was set up successfully.
        """
        if self.active:
            await self.teardown()

        tick = filters.price_tick
        step = filters.qty_step

        prices = compute_grid_prices(params.lower_price, params.upper_price, params.num_grids)
        if len(prices) < 2:
            logger.error("Grid needs at least 2 price levels")
            return False

        # Round prices and qty
        prices = [round_price(p, tick) for p in prices]
        qty = round_qty(params.qty_per_grid, step)

        if qty < filters.min_qty:
            qty = filters.min_qty

        # Auto-bump qty to meet min notional at the lowest grid price
        min_price = min(prices)
        if min_price > 0 and min_price * qty < filters.min_notional:
            needed = math.ceil(filters.min_notional / min_price / step) * step
            if needed >= filters.min_qty:
                qty = needed
                logger.info(
                    "Bumped qty_per_grid to {:.8f} to meet min_notional", qty,
                )

        if min_price * qty < filters.min_notional:
            logger.error(
                "Grid notional {:.4f} below min_notional {:.4f}",
                min_price * qty, filters.min_notional,
            )
            return False

        # Set leverage
        await self.exchange.set_leverage(params.symbol, params.leverage)
        await self.exchange.set_margin_type(params.symbol, "ISOLATED")

        # Build grid state
        levels: List[GridLevelState] = []
        buy_count = 0
        sell_count = 0

        for i, price in enumerate(prices):
            level = GridLevelState(index=i, price=price)

            if price < current_price:
                # Place BUY below current price
                try:
                    oid = await self.exchange.limit_order(
                        params.symbol, "BUY", qty, price
                    )
                    level.buy_order_id = oid
                    buy_count += 1
                except Exception as e:
                    logger.warning("Failed to place BUY at {}: {}", price, e)
            elif price > current_price:
                # Place SELL above current price
                try:
                    oid = await self.exchange.limit_order(
                        params.symbol, "SELL", qty, price
                    )
                    level.sell_order_id = oid
                    sell_count += 1
                except Exception as e:
                    logger.warning("Failed to place SELL at {}: {}", price, e)

            levels.append(level)

        # Persist grid state
        gs = self.state.state.grid
        gs.symbol = params.symbol
        gs.upper_price = params.upper_price
        gs.lower_price = params.lower_price
        gs.num_grids = params.num_grids
        gs.leverage = params.leverage
        gs.qty_per_grid = qty
        gs.levels = levels
        gs.active = True
        gs.total_profit = 0.0
        gs.total_fees = 0.0
        gs.round_trips = 0
        gs.setup_at = _now_iso()
        gs.ai_reasoning = params.reasoning
        gs.net_qty = 0.0
        gs.avg_entry = 0.0
        self.state.save()

        grid_spacing = prices[1] - prices[0] if len(prices) >= 2 else 0
        logger.info(
            "Grid set up: {} | {:.8f} - {:.8f} | {} levels | spacing {:.8f} | "
            "qty {:.8f} | lev {}x | {} buys + {} sells placed",
            params.symbol, params.lower_price, params.upper_price,
            params.num_grids, grid_spacing, qty, params.leverage,
            buy_count, sell_count,
        )
        return True

    async def check_fills_and_reorder(self, mark_price: float) -> List[dict]:
        """Check for filled orders and place counter orders.

        Called every tick. For the simulator, we call check_limit_fills first.
        Returns a list of fill event dicts for logging/alerting.
        """
        if not self.active:
            return []

        gs = self.grid
        symbol = gs.symbol
        qty = gs.qty_per_grid
        events: List[dict] = []

        # For simulator: trigger fill checks
        if self.exchange.mode == "sim":
            sim = self.exchange  # type: ignore
            if hasattr(sim, "check_limit_fills"):
                sim.check_limit_fills(symbol, mark_price)

        # Check each level for fills
        for level in gs.levels:
            filled_side: Optional[str] = None

            # Check BUY order fill
            if level.buy_order_id is not None:
                if self.exchange.mode == "sim":
                    # In sim, check_limit_fills already removed filled orders
                    sim = self.exchange  # type: ignore
                    if level.buy_order_id not in getattr(sim, "_pending_orders", {}):
                        filled_side = "BUY"
                        level.buy_order_id = None
                else:
                    # In live, check order status
                    orders = await self.exchange.get_open_orders(symbol)
                    open_ids = {o.order_id for o in orders}
                    if level.buy_order_id not in open_ids:
                        filled_side = "BUY"
                        level.buy_order_id = None

            # Check SELL order fill
            if filled_side is None and level.sell_order_id is not None:
                if self.exchange.mode == "sim":
                    sim = self.exchange  # type: ignore
                    if level.sell_order_id not in getattr(sim, "_pending_orders", {}):
                        filled_side = "SELL"
                        level.sell_order_id = None
                else:
                    orders = await self.exchange.get_open_orders(symbol)
                    open_ids = {o.order_id for o in orders}
                    if level.sell_order_id not in open_ids:
                        filled_side = "SELL"
                        level.sell_order_id = None

            if filled_side is None:
                continue

            # A fill happened at this level
            fill_price = level.price
            notional = fill_price * qty
            fee = notional * 0.0002  # maker fee estimate

            # Update net position tracking
            if filled_side == "BUY":
                # We bought — going more long / reducing short
                self._update_net_position(fill_price, qty, "BUY")
                logger.info(
                    "[GRID] BUY filled at {:.8f} (level {}), placing SELL at next level up",
                    fill_price, level.index,
                )
                # Place counter SELL one level up
                next_idx = level.index + 1
                if next_idx < len(gs.levels):
                    next_level = gs.levels[next_idx]
                    if next_level.sell_order_id is None:
                        try:
                            oid = await self.exchange.limit_order(
                                symbol, "SELL", qty, next_level.price
                            )
                            next_level.sell_order_id = oid
                        except Exception as e:
                            logger.warning("Failed to place counter SELL: {}", e)

            elif filled_side == "SELL":
                # We sold — going more short / reducing long
                pnl = self._calc_grid_pnl(fill_price, qty, "SELL")
                self._update_net_position(fill_price, qty, "SELL")
                logger.info(
                    "[GRID] SELL filled at {:.8f} (level {}), pnl={:+.5f}, placing BUY at next level down",
                    fill_price, level.index, pnl,
                )
                # Record the round-trip profit
                self.state.record_grid_fill(pnl, fee)

                # Place counter BUY one level down
                prev_idx = level.index - 1
                if prev_idx >= 0:
                    prev_level = gs.levels[prev_idx]
                    if prev_level.buy_order_id is None:
                        try:
                            oid = await self.exchange.limit_order(
                                symbol, "BUY", qty, prev_level.price
                            )
                            prev_level.buy_order_id = oid
                        except Exception as e:
                            logger.warning("Failed to place counter BUY: {}", e)

            events.append({
                "side": filled_side,
                "price": fill_price,
                "qty": qty,
                "level": level.index,
                "fee": fee,
            })

        if events:
            self.state.save()

        return events

    def _update_net_position(self, price: float, qty: float, side: str) -> None:
        """Track net position for unrealized PnL calculation."""
        gs = self.grid
        if side == "BUY":
            if gs.net_qty >= 0:
                # Adding to long — update average
                total_cost = gs.avg_entry * gs.net_qty + price * qty
                gs.net_qty += qty
                gs.avg_entry = total_cost / gs.net_qty if gs.net_qty > 0 else 0
            else:
                # Reducing short
                gs.net_qty += qty
                if gs.net_qty >= 0:
                    gs.avg_entry = price if gs.net_qty > 0 else 0
        elif side == "SELL":
            if gs.net_qty <= 0:
                # Adding to short — update average
                total_cost = gs.avg_entry * abs(gs.net_qty) + price * qty
                gs.net_qty -= qty
                gs.avg_entry = total_cost / abs(gs.net_qty) if gs.net_qty != 0 else 0
            else:
                # Reducing long
                gs.net_qty -= qty
                if gs.net_qty <= 0:
                    gs.avg_entry = price if gs.net_qty < 0 else 0

    def _calc_grid_pnl(self, sell_price: float, qty: float, side: str) -> float:
        """Estimate PnL for a grid sell (rough — based on grid spacing)."""
        gs = self.grid
        if gs.num_grids <= 0:
            return 0.0
        spacing = (gs.upper_price - gs.lower_price) / gs.num_grids
        # Each sell is roughly one spacing above the corresponding buy
        return spacing * qty

    def unrealized_pnl(self, mark_price: float) -> float:
        """Calculate unrealized PnL of the net grid position."""
        gs = self.grid
        if abs(gs.net_qty) < 1e-12 or gs.avg_entry <= 0:
            return 0.0
        if gs.net_qty > 0:  # net long
            return (mark_price - gs.avg_entry) * gs.net_qty
        else:  # net short
            return (gs.avg_entry - mark_price) * abs(gs.net_qty)

    def is_price_out_of_range(self, mark_price: float, threshold_pct: float) -> bool:
        """Check if price has moved outside the grid range by threshold_pct."""
        gs = self.grid
        if not gs.active or gs.upper_price <= gs.lower_price:
            return False
        grid_range = gs.upper_price - gs.lower_price
        margin = grid_range * threshold_pct / 100.0
        return mark_price < gs.lower_price - margin or mark_price > gs.upper_price + margin

    async def teardown(self) -> None:
        """Cancel all grid orders and deactivate."""
        gs = self.grid
        if not gs.symbol:
            return

        logger.info("Tearing down grid on {}", gs.symbol)
        try:
            cancelled = await self.exchange.cancel_all_orders(gs.symbol)
            logger.info("Cancelled {} pending grid orders", cancelled)
        except Exception as e:
            logger.warning("Error cancelling grid orders: {}", e)

        gs.active = False
        gs.levels.clear()
        self.state.save()

    def grid_summary(self, mark_price: float) -> dict:
        """Return a summary dict for Telegram / logging."""
        gs = self.grid
        upnl = self.unrealized_pnl(mark_price)
        spacing = (gs.upper_price - gs.lower_price) / gs.num_grids if gs.num_grids > 0 else 0
        buy_orders = sum(1 for lv in gs.levels if lv.buy_order_id)
        sell_orders = sum(1 for lv in gs.levels if lv.sell_order_id)
        return {
            "symbol": gs.symbol,
            "active": gs.active,
            "upper": gs.upper_price,
            "lower": gs.lower_price,
            "num_grids": gs.num_grids,
            "spacing": spacing,
            "leverage": gs.leverage,
            "qty_per_grid": gs.qty_per_grid,
            "mark_price": mark_price,
            "net_qty": gs.net_qty,
            "unrealized_pnl": upnl,
            "total_profit": gs.total_profit,
            "total_fees": gs.total_fees,
            "round_trips": gs.round_trips,
            "buy_orders": buy_orders,
            "sell_orders": sell_orders,
            "reasoning": gs.ai_reasoning,
        }


__all__ = ["GridManager", "GridSetupParams", "compute_grid_prices"]
