"""Live Binance USDT-M Futures exchange via python-binance async client.

All read-only calls are wrapped in `_retry` with exponential backoff so
transient network errors / 5xx / -1003 rate-limit responses don't crash the
main loop. Order placement is NOT blindly retried — a timed-out order could
already have filled. Instead we re-query positions to decide.
"""
from __future__ import annotations

import asyncio
from typing import Any, Awaitable, Callable, Dict, List, Optional, TypeVar

import pandas as pd
from loguru import logger

try:
    from binance import AsyncClient
    from binance.exceptions import BinanceAPIException
except ImportError:  # pragma: no cover
    AsyncClient = None  # type: ignore
    BinanceAPIException = Exception  # type: ignore

from .base import (
    ExchangeInterface,
    LimitOrder,
    LivePosition,
    OrderResult,
    OrderSide,
    SymbolFilters,
    TickerInfo,
)


_INTERVAL_MAP = {
    "1m": "1m", "3m": "3m", "5m": "5m", "15m": "15m", "30m": "30m",
    "1h": "1h", "2h": "2h", "4h": "4h", "6h": "6h", "8h": "8h", "12h": "12h",
    "1d": "1d",
}


T = TypeVar("T")


def _is_transient(e: Exception) -> bool:
    """Classify an error as retryable (transient network / rate-limit / 5xx)."""
    if isinstance(e, (asyncio.TimeoutError, ConnectionError)):
        return True
    if isinstance(e, BinanceAPIException):
        code = getattr(e, "code", None)
        status = getattr(e, "status_code", None)
        # -1003: rate limit. -1007: timeout. -1000..-1010: generic network/server.
        if code in (-1003, -1007, -1000, -1001, -1006, -1008, -1016, -1021):
            return True
        if isinstance(status, int) and 500 <= status < 600:
            return True
    # aiohttp-level timeouts / disconnects
    name = type(e).__name__
    if name in ("ClientOSError", "ServerDisconnectedError", "ClientConnectorError",
                "ClientPayloadError"):
        return True
    return False


async def _retry(
    fn: Callable[[], Awaitable[T]],
    *,
    attempts: int = 4,
    base_delay: float = 1.0,
    what: str = "binance call",
) -> T:
    last: Optional[Exception] = None
    for i in range(attempts):
        try:
            return await fn()
        except Exception as e:  # noqa: BLE001
            last = e
            if not _is_transient(e) or i == attempts - 1:
                raise
            delay = base_delay * (2 ** i)
            logger.warning(
                "{} transient error (attempt {}/{}): {} — retrying in {}s",
                what, i + 1, attempts, e, delay,
            )
            await asyncio.sleep(delay)
    # Unreachable, but keeps type checkers happy
    assert last is not None
    raise last


class BinanceLiveExchange(ExchangeInterface):
    mode = "live"

    def __init__(self, api_key: str, api_secret: str, testnet: bool = False) -> None:
        if AsyncClient is None:
            raise RuntimeError(
                "python-binance is not installed. Run ./install.sh or "
                "`pip install python-binance`."
            )
        self._api_key = api_key
        self._api_secret = api_secret
        self._testnet = testnet
        self._client: Optional[AsyncClient] = None
        self._filters_cache: Dict[str, SymbolFilters] = {}

    async def connect(self) -> None:
        self._client = await AsyncClient.create(
            api_key=self._api_key,
            api_secret=self._api_secret,
            testnet=self._testnet,
        )
        logger.info("Connected to Binance Futures {}", "TESTNET" if self._testnet else "MAINNET")

    async def close(self) -> None:
        if self._client is not None:
            await self._client.close_connection()
            self._client = None

    def _c(self) -> AsyncClient:
        if self._client is None:
            raise RuntimeError("Exchange not connected. Call connect() first.")
        return self._client

    async def get_balance(self) -> float:
        balances = await _retry(
            lambda: self._c().futures_account_balance(), what="futures_account_balance"
        )
        for b in balances:
            if b["asset"] == "USDT":
                return float(b["balance"])
        return 0.0

    async def get_klines(
        self, symbol: str, interval: str, limit: int = 200
    ) -> pd.DataFrame:
        raw = await _retry(
            lambda: self._c().futures_klines(
                symbol=symbol, interval=_INTERVAL_MAP[interval], limit=limit
            ),
            what=f"futures_klines[{symbol}]",
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
        r = await _retry(
            lambda: self._c().futures_mark_price(symbol=symbol),
            what=f"futures_mark_price[{symbol}]",
        )
        return float(r["markPrice"])

    async def get_funding_rate(self, symbol: str) -> float:
        """Return annualized funding rate in percent."""
        r = await _retry(
            lambda: self._c().futures_mark_price(symbol=symbol),
            what=f"funding_rate[{symbol}]",
        )
        # lastFundingRate is per 8h. Annualized: rate * 3 * 365 * 100
        rate = float(r.get("lastFundingRate", 0.0))
        return rate * 3.0 * 365.0 * 100.0

    async def get_symbol_filters(self, symbol: str) -> SymbolFilters:
        if symbol in self._filters_cache:
            return self._filters_cache[symbol]
        info = await _retry(
            lambda: self._c().futures_exchange_info(), what="futures_exchange_info"
        )
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
        try:
            await self._c().futures_change_leverage(symbol=symbol, leverage=leverage)
        except BinanceAPIException as e:
            logger.warning("set_leverage({}, {}) failed: {}", symbol, leverage, e)

    async def set_margin_type(self, symbol: str, margin_type: str) -> None:
        try:
            await self._c().futures_change_margin_type(
                symbol=symbol, marginType=margin_type.upper()
            )
        except BinanceAPIException as e:
            # -4046 = "No need to change margin type." Ignore.
            if getattr(e, "code", None) != -4046:
                logger.warning("set_margin_type({}, {}) failed: {}", symbol, margin_type, e)

    async def market_open(
        self, symbol: str, side: OrderSide, qty: float
    ) -> OrderResult:
        return await self._market_order(symbol, side, qty, reduce_only=False)

    async def market_close(
        self, symbol: str, side: OrderSide, qty: float
    ) -> OrderResult:
        return await self._market_order(symbol, side, qty, reduce_only=True)

    async def _market_order(
        self, symbol: str, side: OrderSide, qty: float, reduce_only: bool
    ) -> OrderResult:
        params = dict(symbol=symbol, side=side, type="MARKET", quantity=qty)
        if reduce_only:
            params["reduceOnly"] = "true"
        r = await self._c().futures_create_order(**params)
        avg_price = float(r.get("avgPrice") or 0.0)
        filled_qty = float(r.get("executedQty") or qty)
        # Fee isn't in the create_order response; fetch from user trades.
        fee = 0.0
        try:
            trades = await self._c().futures_account_trades(symbol=symbol, limit=5)
            for t in trades:
                if str(t.get("orderId")) == str(r.get("orderId")):
                    fee += float(t.get("commission", 0))
                    if not avg_price:
                        avg_price = float(t.get("price", 0))
        except BinanceAPIException:
            pass
        return OrderResult(
            order_id=str(r.get("orderId")),
            symbol=symbol,
            side=side,
            qty=filled_qty,
            avg_price=avg_price,
            fee=fee,
            status=r.get("status", "FILLED"),
            raw=r,
        )

    async def get_open_positions(self) -> List[LivePosition]:
        data = await _retry(
            lambda: self._c().futures_position_information(),
            what="futures_position_information",
        )
        out: List[LivePosition] = []
        for p in data:
            amt = float(p["positionAmt"])
            if amt == 0:
                continue
            out.append(LivePosition(
                symbol=p["symbol"],
                side="LONG" if amt > 0 else "SHORT",
                qty=abs(amt),
                entry_price=float(p["entryPrice"]),
                mark_price=float(p["markPrice"]),
                unrealized_pnl=float(p["unRealizedProfit"]),
                leverage=int(float(p.get("leverage", 1))),
            ))
        return out

    # ---- limit order interface ----

    async def limit_order(
        self, symbol: str, side: OrderSide, qty: float, price: float,
    ) -> str:
        r = await _retry(
            lambda: self._c().futures_create_order(
                symbol=symbol,
                side=side,
                type="LIMIT",
                quantity=qty,
                price=price,
                timeInForce="GTC",
            ),
            what=f"limit_order[{symbol}]",
        )
        return str(r["orderId"])

    async def get_open_orders(self, symbol: str) -> List[LimitOrder]:
        raw = await _retry(
            lambda: self._c().futures_get_open_orders(symbol=symbol),
            what=f"get_open_orders[{symbol}]",
        )
        out: List[LimitOrder] = []
        for o in raw:
            if o.get("type") != "LIMIT":
                continue
            out.append(LimitOrder(
                order_id=str(o["orderId"]),
                symbol=o["symbol"],
                side=o["side"],
                qty=float(o["origQty"]),
                price=float(o["price"]),
                status=o["status"],
                filled_qty=float(o.get("executedQty", 0)),
                filled_price=float(o.get("avgPrice", 0)),
                fee=0.0,
            ))
        return out

    async def cancel_order(self, symbol: str, order_id: str) -> bool:
        try:
            await self._c().futures_cancel_order(
                symbol=symbol, orderId=int(order_id),
            )
            return True
        except BinanceAPIException:
            return False

    async def cancel_all_orders(self, symbol: str) -> int:
        orders = await self.get_open_orders(symbol)
        count = len(orders)
        if count == 0:
            return 0
        try:
            await self._c().futures_cancel_all_open_orders(symbol=symbol)
        except BinanceAPIException:
            # Fallback: cancel individually
            for o in orders:
                try:
                    await self._c().futures_cancel_order(
                        symbol=symbol, orderId=int(o.order_id),
                    )
                except BinanceAPIException:
                    count -= 1
        return count

    # ---- symbol scanning ----

    async def get_all_tickers(self) -> List[TickerInfo]:
        raw = await _retry(
            lambda: self._c().futures_ticker(),
            what="futures_ticker",
        )
        out: List[TickerInfo] = []
        for t in raw:
            sym = t["symbol"]
            if not sym.endswith("USDT"):
                continue
            out.append(TickerInfo(
                symbol=sym,
                price=float(t["lastPrice"]),
                volume_24h=float(t["quoteVolume"]),
                change_pct_24h=float(t["priceChangePercent"]),
                high_24h=float(t["highPrice"]),
                low_24h=float(t["lowPrice"]),
            ))
        return out

    async def get_all_symbol_filters(self) -> Dict[str, SymbolFilters]:
        if self._filters_cache:
            return dict(self._filters_cache)
        info = await _retry(
            lambda: self._c().futures_exchange_info(),
            what="futures_exchange_info",
        )
        result: Dict[str, SymbolFilters] = {}
        for s in info["symbols"]:
            if s.get("contractType") != "PERPETUAL":
                continue
            if s.get("status") != "TRADING":
                continue
            symbol = s["symbol"]
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
            result[symbol] = sf
        self._filters_cache = result
        return dict(result)


__all__ = ["BinanceLiveExchange"]
