"""Neutral grid manager — places and manages a grid of limit orders.

The grid is a set of evenly-spaced price levels between a lower and upper bound.
Below the current price, BUY limit orders are placed; above, SELL limit orders.
When a BUY fills, a corresponding SELL is placed one level up (take profit).
When a SELL fills, a corresponding BUY is placed one level down.

Realized PnL is computed using weighted-average cost basis on the net
inventory: each fill that REDUCES the current inventory realizes
(close_price - avg_entry) * qty for longs (or the inverse for shorts).
This gives a correct running PnL instead of the phantom
`spacing * qty` that was previously booked on every SELL fill.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from loguru import logger

from ..exchange.base import ExchangeInterface, LimitOrder, SymbolFilters
from ..state import GridState, GridLevelState, StateStore, TradeRecord, _now_iso


# Binance Futures maker fee (0.02%)
_MAKER_FEE = 0.0002


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


def _precision(tick: float) -> int:
    """Derive decimal precision from a tick/step size (e.g. 0.001 -> 3)."""
    s = f"{tick:.12f}".rstrip("0")
    if "." in s:
        return len(s.split(".")[1])
    return 0


def round_price(price: float, tick: float) -> float:
    """Round a price to the nearest tick size with correct precision."""
    if tick <= 0:
        return price
    p = _precision(tick)
    return round(round(price / tick) * tick, p)


def round_qty(qty: float, step: float) -> float:
    """Round qty down to the nearest step size with correct precision."""
    if step <= 0:
        return qty
    p = _precision(step)
    return round(math.floor(qty / step + 1e-9) * step, p)


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
        """Set up a new grid. Cancels any existing grid first and closes
        any naked net position so the new grid starts flat.

        Returns True if the grid was set up successfully.
        """
        if self.active:
            await self.teardown(close_position=True)

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

        # Snapshot open orders once per tick (live) instead of per level
        open_ids: Optional[set] = None
        if self.exchange.mode != "sim":
            orders = await self.exchange.get_open_orders(symbol)
            open_ids = {o.order_id for o in orders}

        # Check each level for fills
        for level in gs.levels:
            filled_side: Optional[str] = None

            # Check BUY order fill
            if level.buy_order_id is not None:
                if self.exchange.mode == "sim":
                    sim = self.exchange  # type: ignore
                    if level.buy_order_id not in getattr(sim, "_pending_orders", {}):
                        filled_side = "BUY"
                        level.buy_order_id = None
                else:
                    assert open_ids is not None
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
                    assert open_ids is not None
                    if level.sell_order_id not in open_ids:
                        filled_side = "SELL"
                        level.sell_order_id = None

            if filled_side is None:
                continue

            # A fill happened at this level
            fill_price = level.price
            notional = fill_price * qty
            fee = notional * _MAKER_FEE

            # Compute realized PnL BEFORE updating net position
            realized = self._realized_pnl(filled_side, fill_price, qty)
            # Always subtract fee from realized so wins/losses reflect net PnL
            realized_after_fee = realized - fee

            # Update net position tracking (after PnL calc)
            self._update_net_position(fill_price, qty, filled_side)

            if filled_side == "BUY":
                logger.info(
                    "[GRID] BUY filled @ {:.8f} (lvl {}) realized={:+.5f} "
                    "net_qty={:.6f} avg={:.8f}",
                    fill_price, level.index, realized_after_fee,
                    gs.net_qty, gs.avg_entry,
                )
                # Only record PnL when BUY actually CLOSES short inventory
                if realized != 0.0:
                    self.state.record_grid_fill(realized_after_fee, fee)
                else:
                    # Inventory-opening fill: just book the fee as cost
                    self.state.record_grid_fee(fee)

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
                logger.info(
                    "[GRID] SELL filled @ {:.8f} (lvl {}) realized={:+.5f} "
                    "net_qty={:.6f} avg={:.8f}",
                    fill_price, level.index, realized_after_fee,
                    gs.net_qty, gs.avg_entry,
                )
                if realized != 0.0:
                    self.state.record_grid_fill(realized_after_fee, fee)
                else:
                    self.state.record_grid_fee(fee)

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
                "pnl": realized_after_fee,
            })

        if events:
            self.state.save()

        return events

    # ---- realized PnL & position tracking ----

    def _realized_pnl(self, side: str, price: float, qty: float) -> float:
        """Realized PnL for a fill, using weighted-average cost basis.

        Returns the gross PnL (before fees) realized by this fill. A fill
        that only OPENS or ADDS to inventory returns 0.0; a fill that
        REDUCES inventory returns (close_price - avg_entry) * closed_qty
        (or the inverse for shorts). A fill that FLIPS sides realizes
        PnL on the closed portion and leaves the opening portion with
        avg_entry = fill price.
        """
        gs = self.grid
        net = gs.net_qty
        avg = gs.avg_entry

        if side == "BUY":
            if net >= 0:
                # Adding to long — no realized
                return 0.0
            # Net short, BUY closes part/all of short
            close_qty = min(qty, -net)
            return (avg - price) * close_qty
        else:  # SELL
            if net <= 0:
                # Adding to short — no realized
                return 0.0
            # Net long, SELL closes part/all of long
            close_qty = min(qty, net)
            return (price - avg) * close_qty

    def _update_net_position(self, price: float, qty: float, side: str) -> None:
        """Update weighted-average entry and net inventory after a fill."""
        gs = self.grid
        net = gs.net_qty
        avg = gs.avg_entry

        if side == "BUY":
            if net >= 0:
                # Adding to (or opening) long
                new_net = net + qty
                total_cost = avg * net + price * qty
                gs.avg_entry = total_cost / new_net if new_net > 0 else 0.0
                gs.net_qty = new_net
            else:
                # Net short: BUY reduces / flips
                if qty <= -net:
                    gs.net_qty = net + qty
                    if abs(gs.net_qty) < 1e-12:
                        gs.net_qty = 0.0
                        gs.avg_entry = 0.0
                    # avg stays on remaining short
                else:
                    # Flipped from short to long
                    leftover = qty - (-net)
                    gs.net_qty = leftover
                    gs.avg_entry = price
        else:  # SELL
            if net <= 0:
                # Adding to (or opening) short
                new_abs = abs(net) + qty
                total_cost = avg * abs(net) + price * qty
                gs.avg_entry = total_cost / new_abs if new_abs > 0 else 0.0
                gs.net_qty = -new_abs
            else:
                # Net long: SELL reduces / flips
                if qty <= net:
                    gs.net_qty = net - qty
                    if abs(gs.net_qty) < 1e-12:
                        gs.net_qty = 0.0
                        gs.avg_entry = 0.0
                    # avg stays on remaining long
                else:
                    # Flipped from long to short
                    leftover = qty - net
                    gs.net_qty = -leftover
                    gs.avg_entry = price

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

    async def close_net_position(self) -> float:
        """Market-close the net inventory so we don't leave a naked position.

        Returns the realized PnL recognised by the close (gross of fees).
        """
        gs = self.grid
        if abs(gs.net_qty) < 1e-12 or not gs.symbol:
            return 0.0

        symbol = gs.symbol
        qty = abs(gs.net_qty)
        side = "SELL" if gs.net_qty > 0 else "BUY"

        try:
            result = await self.exchange.market_close(symbol, side, qty)
        except Exception as e:
            # Fallback: some exchanges need market_open with reduce_only off
            logger.warning("market_close failed ({}), trying market_open: {}", side, e)
            try:
                result = await self.exchange.market_open(symbol, side, qty)
            except Exception as e2:
                logger.error("Failed to close net position: {}", e2)
                return 0.0

        fill_price = result.avg_price or 0.0
        fee = result.fee or (fill_price * qty * _MAKER_FEE * 2)  # fallback taker-ish
        if fill_price <= 0:
            # Can't price — just zero out tracking
            realized = 0.0
        else:
            realized = self._realized_pnl(side, fill_price, qty)
        self._update_net_position(fill_price or gs.avg_entry, qty, side)
        # Safety: force to zero since we just closed all of it
        gs.net_qty = 0.0
        gs.avg_entry = 0.0

        net_realized = realized - fee
        logger.info(
            "[GRID] Closed naked position: {} {} @ {:.8f} realized={:+.5f} fee={:.5f}",
            side, qty, fill_price, net_realized, fee,
        )
        if realized != 0.0 or fee != 0.0:
            self.state.record_grid_fill(net_realized, fee)
        self.state.save()
        return net_realized

    async def teardown(self, close_position: bool = True) -> None:
        """Cancel all grid orders and deactivate.

        If ``close_position`` is True (default), any accumulated net
        inventory is market-closed so the bot doesn't hold a naked,
        unmanaged position across rebalances.
        """
        gs = self.grid
        if not gs.symbol:
            return

        logger.info("Tearing down grid on {}", gs.symbol)
        try:
            cancelled = await self.exchange.cancel_all_orders(gs.symbol)
            logger.info("Cancelled {} pending grid orders", cancelled)
        except Exception as e:
            logger.warning("Error cancelling grid orders: {}", e)

        if close_position:
            try:
                await self.close_net_position()
            except Exception as e:
                logger.exception("Error closing net position on teardown: {}", e)

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
            "avg_entry": gs.avg_entry,
            "unrealized_pnl": upnl,
            "total_profit": gs.total_profit,
            "total_fees": gs.total_fees,
            "round_trips": gs.round_trips,
            "buy_orders": buy_orders,
            "sell_orders": sell_orders,
            "reasoning": gs.ai_reasoning,
        }


__all__ = ["GridManager", "GridSetupParams", "compute_grid_prices"]
