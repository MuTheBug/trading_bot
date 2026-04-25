"""Strategy base types."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Literal, Optional, Tuple

import pandas as pd


Side = Literal["LONG", "SHORT"]


@dataclass
class Signal:
    """A trade decision produced by a Strategy.

    Classical strategies (e.g. trend_momentum) only set side/entry/atr/reason
    and let RiskManager.build_exit_ladder() derive SL/TP from ATR. The AI
    strategy additionally sets stop_loss / take_profits / leverage so the
    bot trades the exact plan the model chose.
    """

    side: Side
    entry_price: float
    atr: float
    reason: str

    # Optional model-chosen plan. When set, bot.py trades these directly.
    stop_loss: Optional[float] = None
    take_profits: Optional[List[Tuple[float, float]]] = None  # (price, close_pct)
    leverage: Optional[int] = None
    confidence: Optional[float] = None  # 0..1


class Strategy(ABC):
    """Contract: given enriched 15m + 1h dataframes, return a Signal or None."""

    @abstractmethod
    async def evaluate(
        self, symbol: str, df15: pd.DataFrame, df1h: pd.DataFrame
    ) -> Optional[Signal]:
        ...


__all__ = ["Strategy", "Signal", "Side"]
