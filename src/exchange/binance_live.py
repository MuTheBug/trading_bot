"""Live Binance USDT-M Futures exchange via python-binance async client."""
from __future__ import annotations

import asyncio
from typing import Dict, List, Optional

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
    LivePosition,
    OrderResult,
    OrderSide,
    SymbolFilters,
)


_INTERVAL_MAP = {
    "1m": "1m", "3m": "3m", "5m": "5m", "15m": "15m", "30m": "30m",
    "1h": "1h", "2h": "2h", "4h": "4h", "6h": "6h", "8h": "8h", "12h": "12h",
    "1d": "1d",
}


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
        balances = await self._c().futures_account_balance()
        for b in balances:
            if b["asset"] == "USDT":
                return float(b["balance"])
        return 0.0

    async def get_klines(
        self, symbol: str, interval: str, limit: int = 200
    ) -> pd.DataFrame:
        raw = await self._c().futures_klines(
            symbol=symbol, interval=_INTERVAL_MAP[interval], limit=limit
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
        r = await self._c().futures_mark_price(symbol=symbol)
        return float(r["markPrice"])

    async def get_funding_rate(self, symbol: str) -> float:
        """Return annualized funding rate in percent."""
        r = await self._c().futures_mark_price(symbol=symbol)
        # lastFundingRate is per 8h. Annualized: rate * 3 * 365 * 100
        rate = float(r.get("lastFundingRate", 0.0))
        return rate * 3.0 * 365.0 * 100.0

    async def get_symbol_filters(self, symbol: str) -> SymbolFilters:
        if symbol in self._filters_cache:
            return self._filters_cache[symbol]
        info = await self._c().futures_exchange_info()
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
        data = await self._c().futures_position_information()
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


__all__ = ["BinanceLiveExchange"]
