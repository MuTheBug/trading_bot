"""Directional trading loop: regime-aware long/short with adaptive leverage.

Sits beside the grid manager as an alternative trading mode. Each tick:

1. If no open position, scan symbols for the strongest regime/setup, compute
   an adaptive trade plan (qty + leverage), and open it with a tiered SL/TP.
2. If a position is open, let the PositionManager run its rules and react to
   the returned action (partial close / full close / hold).

The scan uses the same scoring approach the AI strategy already has: it ranks
symbols by recent activity (range_pct & volume) then classifies the regime of
each candidate to pick the one with the highest confidence.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Tuple

from loguru import logger

from . import trade_log
from .config import BotConfig, Secrets
from .exchange.base import ExchangeInterface, SymbolFilters, TickerInfo
from .position.manager import PositionManager
from .risk.leverage import TradePlan, compute_trade_plan
from .risk.risk_manager import RiskManager
from .state import StateStore
from .strategy.adaptive import DirectionalPlan, RegimeAdaptiveStrategy
from .strategy.regime import RegimeSnapshot


@dataclass
class _Candidate:
    symbol: str
    ticker: TickerInfo
    filters: SymbolFilters
    score: float           # pre-regime liquidity/activity score


def _prescreen(
    tickers: List[TickerInfo],
    all_filters: dict,
    min_volume_usd: float,
    top_n: int = 25,
) -> List[_Candidate]:
    """Filter & rank symbols by activity before running the regime classifier.

    Regime classification pulls klines per candidate, so we restrict to a
    manageable top-N. Scoring favours liquid symbols with real movement.
    """
    out: List[_Candidate] = []
    for t in tickers:
        if not t.symbol.endswith("USDT"):
            continue
        if t.volume_24h < min_volume_usd:
            continue
        filt = all_filters.get(t.symbol)
        if filt is None:
            continue
        price_range = max(t.high_24h - t.low_24h, 0.0)
        if price_range <= 0 or t.price <= 0:
            continue
        range_pct = price_range / t.price * 100.0
        # Slight bias toward movers; log-scale volume so a $1B coin doesn't
        # completely drown a $100M gem.
        import math
        score = range_pct * math.log10(max(t.volume_24h, 1.0))
        out.append(_Candidate(t.symbol, t, filt, score))
    out.sort(key=lambda c: c.score, reverse=True)
    return out[:top_n]


class DirectionalTrader:
    """Scan + trade one directional position at a time."""

    def __init__(
        self,
        exchange: ExchangeInterface,
        state: StateStore,
        config: BotConfig,
    ) -> None:
        self.ex = exchange
        self.state = state
        self.cfg = config
        self.risk = RiskManager(config.risk, config.grid)
        self.strategy = RegimeAdaptiveStrategy(
            adx_strong=config.directional.adx_strong,
            adx_weak=config.directional.adx_weak,
        )
        self.pm = PositionManager(
            exchange,
            trail_atr_mult=config.directional.trail_atr_mult,
            trail_arm_atr=config.directional.trail_arm_atr,
            breakeven_buffer_atr=config.directional.breakeven_buffer_atr,
            time_stop_hours=config.directional.time_stop_hours,
            max_loss_pct=config.directional.max_loss_pct,
            breakeven_after_tp1=config.directional.breakeven_after_tp1,
        )
        self._last_scan: Optional[datetime] = None

    def has_position(self) -> bool:
        return self.pm.has_position()

    async def start(self) -> None:
        # Reconcile: if the exchange holds a position but we don't know
        # about it, close it defensively. We only own one position at a time.
        if self.pm.has_position():
            return
        try:
            opens = await self.ex.get_open_positions()
        except Exception as e:  # pragma: no cover — defensive
            logger.warning("get_open_positions failed on start: {}", e)
            return
        for p in opens:
            side = "SELL" if p.side == "LONG" else "BUY"
            try:
                await self.ex.market_close(p.symbol, side, p.qty)
                logger.warning("Closed orphan {} position on {}", p.side, p.symbol)
            except Exception as e:  # pragma: no cover — defensive
                logger.warning("Failed to close orphan {}: {}", p.symbol, e)

    async def equity(self) -> float:
        bal = await self.ex.get_balance()
        if self.pm.has_position():
            try:
                mark = await self.ex.get_mark_price(self.pm.position.symbol)
                return bal + self.pm.position.unrealised_pnl(mark)
            except Exception:
                pass
        return bal

    async def tick(self) -> None:
        """One heartbeat of the directional loop."""
        if self.state.state.paused:
            logger.warning("Bot PAUSED — skipping directional tick")
            return

        equity = await self.equity()
        ok, reason = self.risk.can_run_grid(self.state, equity)
        if not ok and not self.pm.has_position():
            logger.warning("Risk manager blocks new entries: {}", reason)
            return

        if self.pm.has_position():
            await self._manage_position()
            return

        # Cooldown between scans so we don't hammer exchange APIs.
        now = datetime.now(timezone.utc)
        if self._last_scan is not None:
            gap = (now - self._last_scan).total_seconds()
            if gap < self.cfg.directional.scan_interval_seconds:
                return
        self._last_scan = now
        await self._scan_and_open(equity)

    async def force_close(self, reason: str = "MANUAL") -> None:
        if not self.pm.has_position():
            raise RuntimeError("no position open")
        mark = await self.ex.get_mark_price(self.pm.position.symbol)
        await self.pm.close("MANUAL", mark)  # type: ignore[arg-type]
        trade_log.log("close", why=reason)

    # ------------ management ------------

    async def _manage_position(self) -> None:
        pos = self.pm.position
        assert pos is not None
        try:
            mark = await self.ex.get_mark_price(pos.symbol)
        except Exception as e:
            logger.warning("get_mark_price({}) failed: {}", pos.symbol, e)
            return
        result = self.pm.on_tick(mark)
        if result.action == "HOLD":
            if result.new_stop is not None:
                logger.info(
                    "Trailing stop tightened on {}: {:.6f}",
                    pos.symbol, result.new_stop,
                )
            return
        if result.action == "PARTIAL_CLOSE":
            await self.pm.apply_partial(result.close_qty, mark)
            trade_log.log(
                "tp", s=pos.symbol, p=mark, q=result.close_qty,
                rem=pos.remaining_qty,
            )
            return
        if result.action == "EXIT":
            rp = pos.realised_pnl
            await self.pm.close(result.reason or "MANUAL", mark)
            # After the close, measure true realised PnL from state not memory.
            self._record_exit(pos.symbol, mark, result.reason, rp)

    def _record_exit(
        self, symbol: str, mark: float, reason, prior_realised: float,
    ) -> None:
        trade_log.log("close", s=symbol, p=mark, why=str(reason),
                      rp=prior_realised)
        self.state.state.daily.trades += 1
        if prior_realised > 0:
            self.state.state.daily.wins += 1
        elif prior_realised < 0:
            self.state.state.daily.losses += 1
        self.state.state.daily.realized_pnl += prior_realised
        self.state.save()

    # ------------ scan & open ------------

    async def _scan_and_open(self, equity: float) -> None:
        try:
            tickers = await self.ex.get_all_tickers()
            all_filters = await self.ex.get_all_symbol_filters()
        except Exception as e:
            logger.warning("scan fetch failed: {}", e)
            return
        candidates = _prescreen(
            tickers, all_filters,
            min_volume_usd=self.cfg.directional.min_volume_usd,
            top_n=self.cfg.directional.scan_top_n,
        )
        if not candidates:
            logger.info("No candidates after prescreen")
            return

        # Filter out cooldown symbols.
        candidates = [
            c for c in candidates if not self.state.is_in_cooldown(c.symbol)
        ]
        if not candidates:
            logger.info("All candidates in cooldown")
            return

        best: Optional[Tuple[_Candidate, DirectionalPlan]] = None
        # Scan candidates sequentially but cap how many we inspect — each
        # iteration does 2 kline fetches.
        limit = min(len(candidates), self.cfg.directional.scan_top_n)
        for cand in candidates[:limit]:
            plan = await self._evaluate_candidate(cand)
            if plan is None:
                continue
            if best is None or plan.confidence > best[1].confidence:
                best = (cand, plan)
            # Early exit: if we find a high-confidence setup, stop scanning.
            if plan.confidence >= 0.85:
                break

        if best is None:
            logger.info("No tradable regime found across {} candidates", limit)
            return

        cand, plan = best
        # Cap check: don't trade below confidence threshold.
        if plan.confidence < self.cfg.directional.min_confidence:
            logger.info(
                "Best candidate {} below confidence threshold ({:.2f} < {:.2f})",
                cand.symbol, plan.confidence,
                self.cfg.directional.min_confidence,
            )
            return

        await self._open_trade(cand, plan, equity)

    async def _evaluate_candidate(
        self, cand: _Candidate,
    ) -> Optional[DirectionalPlan]:
        try:
            df15 = await self.ex.get_klines(cand.symbol, "15m", 150)
            df1h = await self.ex.get_klines(cand.symbol, "1h", 100)
        except Exception as e:
            logger.debug("klines failed for {}: {}", cand.symbol, e)
            return None
        try:
            return self.strategy.plan(df15, df1h)
        except Exception as e:  # pragma: no cover — defensive
            logger.debug("plan failed for {}: {}", cand.symbol, e)
            return None

    async def _open_trade(
        self, cand: _Candidate, plan: DirectionalPlan, equity: float,
    ) -> None:
        atr_pct = plan.atr / plan.entry_price * 100.0 if plan.entry_price > 0 else 0
        tp = compute_trade_plan(
            equity=equity,
            entry_price=plan.entry_price,
            stop_price=plan.stop_loss,
            filters=cand.filters,
            risk_pct=self.cfg.directional.risk_per_trade_pct,
            confidence=plan.confidence,
            atr_pct=atr_pct,
            base_leverage=self.cfg.directional.base_leverage,
            max_leverage=self.cfg.directional.max_leverage,
            max_margin_pct=self.cfg.directional.max_margin_pct,
            min_leverage=self.cfg.directional.min_leverage,
        )
        if not tp.feasible:
            logger.info(
                "Trade plan infeasible for {}: {}", cand.symbol, tp.reason,
            )
            trade_log.log("skip", s=cand.symbol, why=f"infeasible:{tp.reason}")
            return

        try:
            await self.pm.open(
                symbol=cand.symbol,
                side=plan.side,
                qty=tp.qty,
                entry_price=plan.entry_price,
                stop_loss=plan.stop_loss,
                take_profits=plan.take_profits,
                leverage=tp.leverage,
                atr_at_entry=plan.atr,
                filters=cand.filters,
                equity_at_open=equity,
                margin_type=self.cfg.margin_type,
            )
        except Exception as e:
            logger.exception("open failed for {}: {}", cand.symbol, e)
            trade_log.log("skip", s=cand.symbol, why=f"open_fail:{e}")
            return

        trade_log.log(
            "setup", s=cand.symbol, sd=plan.side[:1],
            p=plan.entry_price, q=tp.qty, lev=tp.leverage,
            sl=plan.stop_loss, reg=plan.regime.regime,
            conf=f"{plan.confidence:.2f}",
            why=plan.reason[:60],
        )


__all__ = ["DirectionalTrader"]
