"""Persistent runtime state: grid config, fills, daily PnL, trade history."""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Literal, Optional


Side = Literal["LONG", "SHORT"]


# ---- Grid state ----

@dataclass
class GridLevelState:
    """Persisted state for one grid level."""
    index: int
    price: float
    buy_order_id: Optional[str] = None
    sell_order_id: Optional[str] = None
    filled_side: Optional[str] = None    # "BUY" or "SELL" or None

    @property
    def has_pending(self) -> bool:
        return self.buy_order_id is not None or self.sell_order_id is not None


@dataclass
class GridState:
    """Full grid configuration and runtime state."""
    symbol: str = ""
    upper_price: float = 0.0
    lower_price: float = 0.0
    num_grids: int = 0
    leverage: int = 1
    qty_per_grid: float = 0.0
    levels: List[GridLevelState] = field(default_factory=list)
    active: bool = False
    total_profit: float = 0.0
    total_fees: float = 0.0
    round_trips: int = 0
    setup_at: str = ""
    ai_reasoning: str = ""
    net_qty: float = 0.0        # positive = net long, negative = net short
    avg_entry: float = 0.0      # weighted average entry price of net position


# ---- Legacy position (kept for compatibility with old state files) ----

@dataclass
class TakeProfitLevel:
    price: float
    close_pct: float
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
    opened_at: str
    atr_at_entry: float
    tp1_hit: bool = False
    tp2_hit: bool = False
    highest_since_entry: float = 0.0
    lowest_since_entry: float = 0.0
    trailing_active: bool = False

    def unrealized_pnl_pct(self, mark: float) -> float:
        if self.side == "LONG":
            return (mark - self.entry_price) / self.entry_price * 100.0
        return (self.entry_price - mark) / self.entry_price * 100.0


@dataclass
class TradeRecord:
    symbol: str
    side: str
    entry_price: float
    exit_price: float
    qty: float
    pnl: float
    fees: float
    opened_at: str
    closed_at: str
    exit_reason: str


@dataclass
class DailyStats:
    date: str
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
    grid: GridState = field(default_factory=GridState)
    positions: Dict[str, Position] = field(default_factory=dict)
    history: List[TradeRecord] = field(default_factory=list)
    daily: DailyStats = field(default_factory=lambda: DailyStats(date=_today_utc()))
    paused: bool = False
    peak_equity: float = 0.0
    cooldown_until: Dict[str, str] = field(default_factory=dict)


def _today_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_grid_state(raw: dict) -> GridState:
    """Deserialize grid state from JSON dict."""
    levels = [GridLevelState(**lv) for lv in raw.get("levels", [])]
    return GridState(
        symbol=raw.get("symbol", ""),
        upper_price=raw.get("upper_price", 0.0),
        lower_price=raw.get("lower_price", 0.0),
        num_grids=raw.get("num_grids", 0),
        leverage=raw.get("leverage", 1),
        qty_per_grid=raw.get("qty_per_grid", 0.0),
        levels=levels,
        active=raw.get("active", False),
        total_profit=raw.get("total_profit", 0.0),
        total_fees=raw.get("total_fees", 0.0),
        round_trips=raw.get("round_trips", 0),
        setup_at=raw.get("setup_at", ""),
        ai_reasoning=raw.get("ai_reasoning", ""),
        net_qty=raw.get("net_qty", 0.0),
        avg_entry=raw.get("avg_entry", 0.0),
    )


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
                    **{**p, "take_profits": [TakeProfitLevel(**tp) for tp in p.get("take_profits", [])]}
                )
                for sym, p in raw.get("positions", {}).items()
            }
            history = [TradeRecord(**t) for t in raw.get("history", [])]
            daily_raw = raw.get("daily") or {"date": _today_utc()}
            daily = DailyStats(**daily_raw)
            grid_raw = raw.get("grid") or {}
            grid = _parse_grid_state(grid_raw) if grid_raw else GridState()
            return BotState(
                grid=grid,
                positions=positions,
                history=history,
                daily=daily,
                paused=raw.get("paused", False),
                peak_equity=raw.get("peak_equity", 0.0),
                cooldown_until=raw.get("cooldown_until", {}),
            )
        except (json.JSONDecodeError, TypeError, KeyError):
            backup = self.path.with_suffix(".corrupt.json")
            self.path.rename(backup)
            return BotState()

    def save(self) -> None:
        with self._lock:
            raw = {
                "grid": asdict(self.state.grid),
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

    def record_grid_fill(self, pnl: float, fee: float) -> None:
        """Record a grid fill that realized PnL (closed inventory).

        ``pnl`` should already be net of ``fee`` so totals stay consistent.
        """
        self.state.grid.total_profit += pnl
        self.state.grid.total_fees += fee
        self.state.grid.round_trips += 1
        self.state.daily.realized_pnl += pnl
        self.state.daily.fees_paid += fee
        self.state.daily.trades += 1
        if pnl > 0:
            self.state.daily.wins += 1
        elif pnl < 0:
            self.state.daily.losses += 1
        self.save()

    def record_grid_fee(self, fee: float) -> None:
        """Record an inventory-opening fill: only fee cost, no win/loss."""
        self.state.grid.total_fees += fee
        self.state.grid.total_profit -= fee  # net profit includes paid fees
        self.state.daily.fees_paid += fee
        self.state.daily.realized_pnl -= fee
        self.save()


__all__ = [
    "Position",
    "TakeProfitLevel",
    "TradeRecord",
    "DailyStats",
    "BotState",
    "GridState",
    "GridLevelState",
    "StateStore",
    "Side",
    "_now_iso",
    "_today_utc",
]
