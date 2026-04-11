"""Persistent runtime state: positions, daily PnL, trade history."""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Dict, List, Literal, Optional


Side = Literal["LONG", "SHORT"]


@dataclass
class TakeProfitLevel:
    price: float
    close_pct: float       # percent of original position to close at this level
    hit: bool = False


@dataclass
class Position:
    symbol: str
    side: Side
    entry_price: float
    original_qty: float
    remaining_qty: float
    leverage: int
    stop_loss: float
    take_profits: List[TakeProfitLevel]
    opened_at: str         # ISO8601 UTC
    atr_at_entry: float
    tp1_hit: bool = False
    tp2_hit: bool = False
    highest_since_entry: float = 0.0   # for trailing (LONG)
    lowest_since_entry: float = 0.0    # for trailing (SHORT)
    trailing_active: bool = False

    def unrealized_pnl_pct(self, mark: float) -> float:
        if self.side == "LONG":
            return (mark - self.entry_price) / self.entry_price * 100.0
        return (self.entry_price - mark) / self.entry_price * 100.0


@dataclass
class TradeRecord:
    symbol: str
    side: Side
    entry_price: float
    exit_price: float
    qty: float
    pnl: float             # in USDT (after fees)
    fees: float
    opened_at: str
    closed_at: str
    exit_reason: str       # SL, TP1, TP2, TP3, TRAIL, TIME, MANUAL, FUNDING


@dataclass
class DailyStats:
    date: str              # YYYY-MM-DD UTC
    realized_pnl: float = 0.0
    fees_paid: float = 0.0
    trades: int = 0
    wins: int = 0
    losses: int = 0

    @property
    def win_rate(self) -> float:
        return (self.wins / self.trades * 100.0) if self.trades else 0.0


@dataclass
class BotState:
    positions: Dict[str, Position] = field(default_factory=dict)          # keyed by symbol
    history: List[TradeRecord] = field(default_factory=list)
    daily: DailyStats = field(default_factory=lambda: DailyStats(date=_today_utc()))
    paused: bool = False
    peak_equity: float = 0.0
    cooldown_until: Dict[str, str] = field(default_factory=dict)          # symbol -> ISO ts


def _today_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class StateStore:
    """JSON-backed state persistence with thread-safe writes."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()
        self.state: BotState = self._load()

    def _load(self) -> BotState:
        if not self.path.exists():
            return BotState()
        try:
            raw = json.loads(self.path.read_text())
            positions = {
                sym: Position(
                    **{**p, "take_profits": [TakeProfitLevel(**tp) for tp in p["take_profits"]]}
                )
                for sym, p in raw.get("positions", {}).items()
            }
            history = [TradeRecord(**t) for t in raw.get("history", [])]
            daily_raw = raw.get("daily") or {"date": _today_utc()}
            daily = DailyStats(**daily_raw)
            return BotState(
                positions=positions,
                history=history,
                daily=daily,
                paused=raw.get("paused", False),
                peak_equity=raw.get("peak_equity", 0.0),
                cooldown_until=raw.get("cooldown_until", {}),
            )
        except (json.JSONDecodeError, TypeError, KeyError):
            # Corrupt state — start fresh but keep the bad file for debugging.
            backup = self.path.with_suffix(".corrupt.json")
            self.path.rename(backup)
            return BotState()

    def save(self) -> None:
        with self._lock:
            raw = {
                "positions": {sym: asdict(p) for sym, p in self.state.positions.items()},
                "history": [asdict(t) for t in self.state.history],
                "daily": asdict(self.state.daily),
                "paused": self.state.paused,
                "peak_equity": self.state.peak_equity,
                "cooldown_until": self.state.cooldown_until,
            }
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(raw, indent=2, default=str))
            tmp.replace(self.path)

    # --- convenience mutators ---

    def roll_daily_if_needed(self) -> bool:
        """Reset the daily stats at UTC midnight. Returns True if a roll happened."""
        today = _today_utc()
        if self.state.daily.date != today:
            self.state.daily = DailyStats(date=today)
            self.save()
            return True
        return False

    def add_position(self, pos: Position) -> None:
        self.state.positions[pos.symbol] = pos
        self.save()

    def remove_position(self, symbol: str) -> Optional[Position]:
        pos = self.state.positions.pop(symbol, None)
        self.save()
        return pos

    def record_trade(self, trade: TradeRecord) -> None:
        self.state.history.append(trade)
        self.state.daily.trades += 1
        self.state.daily.realized_pnl += trade.pnl
        self.state.daily.fees_paid += trade.fees
        if trade.pnl > 0:
            self.state.daily.wins += 1
        else:
            self.state.daily.losses += 1
        self.save()

    def set_cooldown(self, symbol: str, until_iso: str) -> None:
        self.state.cooldown_until[symbol] = until_iso
        self.save()

    def is_in_cooldown(self, symbol: str) -> bool:
        until = self.state.cooldown_until.get(symbol)
        if not until:
            return False
        try:
            return datetime.fromisoformat(until) > datetime.now(timezone.utc)
        except ValueError:
            return False


__all__ = [
    "Position",
    "TakeProfitLevel",
    "TradeRecord",
    "DailyStats",
    "BotState",
    "StateStore",
    "Side",
]
