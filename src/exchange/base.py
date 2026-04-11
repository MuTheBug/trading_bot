"""Abstract exchange interface shared by live and simulator backends."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from decimal import Decimal
from typing import List, Literal, Optional

import pandas as pd


OrderSide = Literal["BUY", "SELL"]
PositionSide = Literal["LONG", "SHORT"]


@dataclass
class SymbolFilters:
    """Exchange filters for a symbol."""
    symbol: str
    price_tick: float          # PRICE_FILTER tickSize
    qty_step: float            # LOT_SIZE stepSize
    min_qty: float             # LOT_SIZE minQty
    min_notional: float        # MIN_NOTIONAL


@dataclass
class OrderResult:
    order_id: str
    symbol: str
    side: OrderSide
    qty: float
    avg_price: float
    fee: float                 # in quote (USDT)
    status: str                # FILLED / NEW / CANCELED
    raw: dict | None = None


@dataclass
class LivePosition:
    symbol: str
    side: PositionSide
    qty: float
    entry_price: float
    mark_price: float
    unrealized_pnl: float
    leverage: int


class ExchangeInterface(ABC):
    """Every method is async so live and sim backends share one contract."""

    mode: Literal["sim", "live"]

    @abstractmethod
    async def connect(self) -> None: ...

    @abstractmethod
    async def close(self) -> None: ...

    @abstractmethod
    async def get_balance(self) -> float:
        """Available USDT balance (equity for sim)."""

    @abstractmethod
    async def get_klines(
        self, symbol: str, interval: str, limit: int = 200
    ) -> pd.DataFrame:
        """Return a DataFrame indexed by close_time with OHLCV columns."""

    @abstractmethod
    async def get_mark_price(self, symbol: str) -> float: ...

    @abstractmethod
    async def get_funding_rate(self, symbol: str) -> float:
        """Return the annualized funding rate in percent (e.g. 18.5 = 18.5%)."""

    @abstractmethod
    async def get_symbol_filters(self, symbol: str) -> SymbolFilters: ...

    @abstractmethod
    async def set_leverage(self, symbol: str, leverage: int) -> None: ...

    @abstractmethod
    async def set_margin_type(self, symbol: str, margin_type: str) -> None: ...

    @abstractmethod
    async def market_open(
        self, symbol: str, side: OrderSide, qty: float
    ) -> OrderResult: ...

    @abstractmethod
    async def market_close(
        self, symbol: str, side: OrderSide, qty: float
    ) -> OrderResult:
        """side here is the closing side (opposite of position side)."""

    @abstractmethod
    async def get_open_positions(self) -> List[LivePosition]: ...


__all__ = [
    "ExchangeInterface",
    "SymbolFilters",
    "OrderResult",
    "LivePosition",
    "OrderSide",
    "PositionSide",
]
