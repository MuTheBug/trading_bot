"""Paper-trading exchange: real market data in, virtual fills out.

The simulator pulls live klines from Binance public REST (no API key required)
so the user sees identical signals to live mode. Orders are tracked in-memory
with realistic fees and slippage but never reach Binance. Balance/positions
persist via StateStore in the main bot loop.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import aiohttp
import pandas as pd
from loguru import logger

from .base import (
    ExchangeInterface,
    LimitOrder,
    LivePosition,
    OrderResult,
    OrderSide,
    SymbolFilters,
    TickerInfo,
)


BINANCE_FAPI = "https://fapi.binance.com"


@dataclass
class _SimPosition:
    symbol: str
    side: str            # LONG / SHORT
    qty: float
    entry_price: float
    leverage: int


@dataclass
class _SimLimitOrder:
    order_id: str
    symbol: str
    side: OrderSide
    qty: float
    price: float


class SimulatorExchange(ExchangeInterface):
    """Virtual exchange that reads live Binance data but simulates fills."""

    mode = "sim"

    def __init__(
        self,
        starting_balance: float = 10.0,
        taker_fee_pct: float = 0.04,
        slippage_ticks: int = 1,
    ) -> None:
        self._balance = float(starting_balance)
        self._initial_balance = float(starting_balance)
        self._taker_fee = taker_fee_pct / 100.0
        self._slippage_ticks = slippage_ticks
        self._positions: Dict[str, _SimPosition] = {}
        self._pending_orders: Dict[str, _SimLimitOrder] = {}
        self._filters_cache: Dict[str, SymbolFilters] = {}
        self._all_filters_cache: Optional[Dict[str, SymbolFilters]] = None
        self._session: Optional[aiohttp.ClientSession] = None
        self._order_seq = 0

    async def connect(self) -> None:
        self._session = aiohttp.ClientSession()
        logger.info(
            "Simulator exchange started with starting balance {:.2f} USDT",
            self._balance,
        )

    async def close(self) -> None:
        if self._session is not None:
            await self._session.close()
            self._session = None

    def _s(self) -> aiohttp.ClientSession:
        if self._session is None:
            raise RuntimeError("Simulator not connected. Call connect() first.")
        return self._session

    async def _get(self, path: str, params: Optional[dict] = None) -> dict:
        async with self._s().get(BINANCE_FAPI + path, params=params) as r:
            r.raise_for_status()
            return await r.json()

    async def _get_json(self, path: str, params: Optional[dict] = None):
        """Like _get but returns the raw JSON (may be list or dict)."""
        async with self._s().get(BINANCE_FAPI + path, params=params) as r:
            r.raise_for_status()
            return await r.json()

    # --- ExchangeInterface ---

    async def get_balance(self) -> float:
        # For sim, equity = cash + unrealized on open positions (we return cash here;
        # the bot loop can compute unrealized from mark prices when needed).
        return self._balance

    @property
    def initial_balance(self) -> float:
        return self._initial_balance

    async def get_klines(
        self, symbol: str, interval: str, limit: int = 200
    ) -> pd.DataFrame:
        raw = await self._get(
            "/fapi/v1/klines",
            params={"symbol": symbol, "interval": interval, "limit": limit},
        )
        cols = [
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "trades",
            "taker_buy_base", "taker_buy_quote", "_ignore",
        ]
        df = pd.DataFrame(raw, columns=cols)
        for c in ("open", "high", "low", "close", "volume"):
            df[c] = df[c].astype(float)
        df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
        df = df.set_index("close_time")
        return df[["open", "high", "low", "close", "volume"]]

    async def get_mark_price(self, symbol: str) -> float:
        r = await self._get("/fapi/v1/premiumIndex", params={"symbol": symbol})
        return float(r["markPrice"])

    async def get_funding_rate(self, symbol: str) -> float:
        r = await self._get("/fapi/v1/premiumIndex", params={"symbol": symbol})
        rate = float(r.get("lastFundingRate", 0.0))
        return rate * 3.0 * 365.0 * 100.0

    async def get_symbol_filters(self, symbol: str) -> SymbolFilters:
        if symbol in self._filters_cache:
            return self._filters_cache[symbol]
        info = await self._get("/fapi/v1/exchangeInfo")
        for s in info["symbols"]:
            if s["symbol"] != symbol:
                continue
            tick = 0.0001
            step = 0.001
            min_qty = 0.001
            min_notional = 5.0
            for f in s["filters"]:
                if f["filterType"] == "PRICE_FILTER":
                    tick = float(f["tickSize"])
                elif f["filterType"] == "LOT_SIZE":
                    step = float(f["stepSize"])
                    min_qty = float(f["minQty"])
                elif f["filterType"] in ("MIN_NOTIONAL", "NOTIONAL"):
                    min_notional = float(
                        f.get("notional") or f.get("minNotional") or 5.0
                    )
            sf = SymbolFilters(
                symbol=symbol,
                price_tick=tick,
                qty_step=step,
                min_qty=min_qty,
                min_notional=min_notional,
            )
            self._filters_cache[symbol] = sf
            return sf
        raise ValueError(f"Symbol {symbol} not found on Binance Futures")

    async def set_leverage(self, symbol: str, leverage: int) -> None:
        # No-op in sim; we track the chosen leverage per position when opening.
        self._default_leverage = leverage  # type: ignore[attr-defined]

    async def set_margin_type(self, symbol: str, margin_type: str) -> None:
        return None

    async def market_open(
        self, symbol: str, side: OrderSide, qty: float
    ) -> OrderResult:
        mark = await self.get_mark_price(symbol)
        filters = await self.get_symbol_filters(symbol)
        slip = filters.price_tick * self._slippage_ticks
        fill_price = mark + slip if side == "BUY" else mark - slip
        notional = fill_price * qty
        fee = notional * self._taker_fee
        self._balance -= fee
        pos_side = "LONG" if side == "BUY" else "SHORT"
        self._positions[symbol] = _SimPosition(
            symbol=symbol,
            side=pos_side,
            qty=qty,
            entry_price=fill_price,
            leverage=getattr(self, "_default_leverage", 3),
        )
        self._order_seq += 1
        logger.info(
            "[SIM] OPEN {} {} qty={} @ {:.6f}  fee={:.5f}  bal={:.4f}",
            pos_side, symbol, qty, fill_price, fee, self._balance,
        )
        return OrderResult(
            order_id=f"SIM-{self._order_seq}",
            symbol=symbol,
            side=side,
            qty=qty,
            avg_price=fill_price,
            fee=fee,
            status="FILLED",
        )

    async def market_close(
        self, symbol: str, side: OrderSide, qty: float
    ) -> OrderResult:
        pos = self._positions.get(symbol)
        if pos is None:
            raise RuntimeError(f"[SIM] No open position for {symbol} to close")
        mark = await self.get_mark_price(symbol)
        filters = await self.get_symbol_filters(symbol)
        slip = filters.price_tick * self._slippage_ticks
        fill_price = mark + slip if side == "BUY" else mark - slip
        close_qty = min(qty, pos.qty)
        notional = fill_price * close_qty
        fee = notional * self._taker_fee

        if pos.side == "LONG":
            pnl = (fill_price - pos.entry_price) * close_qty
        else:
            pnl = (pos.entry_price - fill_price) * close_qty

        self._balance += pnl - fee
        pos.qty -= close_qty
        if pos.qty <= 1e-12:
            del self._positions[symbol]

        self._order_seq += 1
        logger.info(
            "[SIM] CLOSE {} {} qty={} @ {:.6f}  pnl={:+.5f}  fee={:.5f}  bal={:.4f}",
            pos.side, symbol, close_qty, fill_price, pnl, fee, self._balance,
        )
        return OrderResult(
            order_id=f"SIM-{self._order_seq}",
            symbol=symbol,
            side=side,
            qty=close_qty,
            avg_price=fill_price,
            fee=fee,
            status="FILLED",
        )

    async def get_open_positions(self) -> List[LivePosition]:
        out: List[LivePosition] = []
        for p in self._positions.values():
            mark = await self.get_mark_price(p.symbol)
            if p.side == "LONG":
                upnl = (mark - p.entry_price) * p.qty
            else:
                upnl = (p.entry_price - mark) * p.qty
            out.append(LivePosition(
                symbol=p.symbol,
                side=p.side,  # type: ignore[arg-type]
                qty=p.qty,
                entry_price=p.entry_price,
                mark_price=mark,
                unrealized_pnl=upnl,
                leverage=p.leverage,
            ))
        return out

    # ---- limit order interface ----

    _MAKER_FEE = 0.0002  # 0.02%

    async def limit_order(
        self, symbol: str, side: OrderSide, qty: float, price: float,
    ) -> str:
        self._order_seq += 1
        order_id = f"SIM-L-{self._order_seq}"
        self._pending_orders[order_id] = _SimLimitOrder(
            order_id=order_id,
            symbol=symbol,
            side=side,
            qty=qty,
            price=price,
        )
        logger.info(
            "[SIM] LIMIT {} {} qty={} @ {:.6f}  id={}",
            side, symbol, qty, price, order_id,
        )
        return order_id

    async def get_open_orders(self, symbol: str) -> List[LimitOrder]:
        return [
            LimitOrder(
                order_id=o.order_id,
                symbol=o.symbol,
                side=o.side,
                qty=o.qty,
                price=o.price,
                status="NEW",
            )
            for o in self._pending_orders.values()
            if o.symbol == symbol
        ]

    async def cancel_order(self, symbol: str, order_id: str) -> bool:
        order = self._pending_orders.get(order_id)
        if order is None or order.symbol != symbol:
            return False
        del self._pending_orders[order_id]
        logger.info("[SIM] CANCEL order {}  {}", order_id, symbol)
        return True

    async def cancel_all_orders(self, symbol: str) -> int:
        to_remove = [
            oid for oid, o in self._pending_orders.items()
            if o.symbol == symbol
        ]
        for oid in to_remove:
            del self._pending_orders[oid]
        if to_remove:
            logger.info(
                "[SIM] CANCEL ALL {} orders for {}", len(to_remove), symbol,
            )
        return len(to_remove)

    def check_limit_fills(
        self, symbol: str, mark_price: float,
    ) -> List[LimitOrder]:
        """Check pending limit orders against *mark_price* and fill those
        that would have been matched.  Called by the bot's main loop each tick.

        Returns a list of :class:`LimitOrder` objects with status ``FILLED``.
        """
        filled: List[LimitOrder] = []
        to_remove: List[str] = []

        for oid, order in self._pending_orders.items():
            if order.symbol != symbol:
                continue

            should_fill = (
                (order.side == "BUY" and mark_price <= order.price)
                or (order.side == "SELL" and mark_price >= order.price)
            )
            if not should_fill:
                continue

            notional = order.price * order.qty
            fee = notional * self._MAKER_FEE
            self._balance -= fee

            filled.append(LimitOrder(
                order_id=order.order_id,
                symbol=order.symbol,
                side=order.side,
                qty=order.qty,
                price=order.price,
                status="FILLED",
                filled_qty=order.qty,
                filled_price=order.price,
                fee=fee,
            ))
            to_remove.append(oid)

            logger.info(
                "[SIM] LIMIT FILL {} {} qty={} @ {:.6f}  fee={:.5f}  bal={:.4f}",
                order.side, order.symbol, order.qty, order.price,
                fee, self._balance,
            )

        for oid in to_remove:
            del self._pending_orders[oid]

        return filled

    # ---- symbol scanning ----

    async def get_all_tickers(self) -> List[TickerInfo]:
        raw = await self._get_json("/fapi/v1/ticker/24hr")
        tickers: List[TickerInfo] = []
        for t in raw:
            sym = t.get("symbol", "")
            if not sym.endswith("USDT"):
                continue
            try:
                tickers.append(TickerInfo(
                    symbol=sym,
                    price=float(t["lastPrice"]),
                    volume_24h=float(t["quoteVolume"]),
                    change_pct_24h=float(t["priceChangePercent"]),
                    high_24h=float(t["highPrice"]),
                    low_24h=float(t["lowPrice"]),
                ))
            except (KeyError, ValueError):
                continue
        return tickers

    async def get_all_symbol_filters(self) -> Dict[str, SymbolFilters]:
        if self._all_filters_cache is not None:
            return self._all_filters_cache

        info = await self._get("/fapi/v1/exchangeInfo")
        result: Dict[str, SymbolFilters] = {}
        for s in info["symbols"]:
            sym = s.get("symbol", "")
            if s.get("contractType") != "PERPETUAL":
                continue
            if s.get("status") != "TRADING":
                continue
            tick = 0.0001
            step = 0.001
            min_qty = 0.001
            min_notional = 5.0
            for f in s.get("filters", []):
                if f["filterType"] == "PRICE_FILTER":
                    tick = float(f["tickSize"])
                elif f["filterType"] == "LOT_SIZE":
                    step = float(f["stepSize"])
                    min_qty = float(f["minQty"])
                elif f["filterType"] in ("MIN_NOTIONAL", "NOTIONAL"):
                    min_notional = float(
                        f.get("notional") or f.get("minNotional") or 5.0
                    )
            sf = SymbolFilters(
                symbol=sym,
                price_tick=tick,
                qty_step=step,
                min_qty=min_qty,
                min_notional=min_notional,
            )
            result[sym] = sf
            # Also populate per-symbol cache for get_symbol_filters()
            self._filters_cache[sym] = sf

        self._all_filters_cache = result
        logger.info(
            "[SIM] Cached symbol filters for {} perpetual contracts",
            len(result),
        )
        return result


__all__ = ["SimulatorExchange"]
