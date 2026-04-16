"""Risk manager for grid trading: circuit breakers and safety checks.

The grid bot doesn't use SL/TP ladders or trailing stops — those are
handled by the grid manager's fill-and-reorder logic. This module provides:
- Pre-trade gating (pause, daily loss, drawdown)
- Funding rate checks
"""
from __future__ import annotations

from typing import Tuple

from loguru import logger

from ..config import GridConfig, RiskConfig
from ..state import StateStore


class RiskManager:
    def __init__(self, risk_cfg: RiskConfig, grid_cfg: GridConfig) -> None:
        self.risk = risk_cfg
        self.grid = grid_cfg

    def can_run_grid(
        self,
        state: StateStore,
        equity: float,
    ) -> Tuple[bool, str]:
        """Check if the grid bot is allowed to operate.

        Returns (ok, reason). If not ok, the main loop should skip the tick.
        """
        if state.state.paused:
            return False, "bot paused"

        # Daily loss circuit breaker
        if equity > 0:
            daily_loss_pct = (state.state.daily.realized_pnl / equity) * 100.0
            if daily_loss_pct <= -self.risk.daily_loss_limit_pct:
                return False, f"daily loss limit hit ({daily_loss_pct:.2f}%)"

        # Max drawdown from peak equity
        peak = max(state.state.peak_equity, equity)
        if peak > 0:
            dd_pct = (peak - equity) / peak * 100.0
            if dd_pct >= self.risk.max_drawdown_pct:
                return False, f"max drawdown hit ({dd_pct:.2f}%)"

        return True, ""

    def funding_action(self, funding_pct_annualized: float, net_side: str) -> str:
        """Return 'exit' / 'ok' based on annualized funding rate and net exposure.

        net_side: 'LONG' if net_qty > 0, 'SHORT' if net_qty < 0, 'FLAT' if ~0.
        """
        if net_side == "FLAT":
            return "ok"
        abs_funding = abs(funding_pct_annualized)
        if abs_funding >= self.risk.funding_rate_exit_pct:
            if (funding_pct_annualized > 0 and net_side == "LONG") or (
                funding_pct_annualized < 0 and net_side == "SHORT"
            ):
                return "exit"
        return "ok"


__all__ = ["RiskManager"]
