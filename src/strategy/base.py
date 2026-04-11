"""Strategy base types."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal, Optional

import pandas as pd


Side = Literal["LONG", "SHORT"]


@dataclass
class Signal:
    side: Side
    entry_price: float
    atr: float
    reason: str


class Strategy(ABC):
    """Contract: given enriched 15m + 1h dataframes, return a Signal or None."""

    @abstractmethod
    def evaluate(
        self, df15: pd.DataFrame, df1h: pd.DataFrame
    ) -> Optional[Signal]:
        ...


__all__ = ["Strategy", "Signal", "Side"]
