"""AI-driven directional trading loop (long / short / skip).

Each scan:

1. Prescreen tickers by liquidity + activity (math, no AI) to cut the
   universe to a manageable shortlist.
2. Fetch 15m + 1h klines for each shortlisted symbol.
3. Build a rich payload (OHLCV tails, indicators, regime classification,
   account balance, risk caps, recent trade outcomes) and send the WHOLE
   shortlist to the AI.
4. The AI returns ONE decision: OPEN_LONG, OPEN_SHORT (with symbol +
   entry/SL/TPs/leverage) or SKIP.
5. If a trade, validate + size with the adaptive leverage planner and
   hand off to the PositionManager.

The deterministic regime classifier is still run but only to enrich the
AI's context — it does not constrain direction or setup choice.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

from . import trade_log
from .config import BotConfig, Secrets
from .exchange.base import ExchangeInterface, SymbolFilters, TickerInfo
from .position.manager import PositionManager
from .risk.leverage import TradePlan, adaptive_leverage, compute_trade_plan
from .risk.risk_manager import RiskManager
from .state import StateStore
from .strategy.ai_directional import (
    AIDecision,
    AIDirectionalStrategy,
    _CandidateCtx,
)


@dataclass
class _Candidate:
    symbol: str
    ticker: TickerInfo
    filters: SymbolFilters
    score: float


def _prescreen(
    tickers: List[TickerInfo],
    all_filters: Dict[str, SymbolFilters],
    min_volume_usd: float,
    top_n: int,
) -> List[_Candidate]:
    """Filter + rank candidates by activity before calling the AI.

    The AI can only reason about the data we feed it, so this is just
    liquidity + "something's moving" gating. No directional bias here.
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
        if t.price <= 0:
            continue
        rng = max(t.high_24h - t.low_24h, 0.0)
        range_pct = rng / t.price * 100.0 if t.price > 0 else 0.0
        # Favour symbols that actually moved (either direction).
        activity = abs(t.change_pct_24h) + 0.5 * range_pct
        score = activity * math.log10(max(t.volume_24h, 1.0))
        out.append(_Candidate(t.symbol, t, filt, score))
    out.sort(key=lambda c: c.score, reverse=True)
    return out[:top_n]


class DirectionalTrader:
    """AI-driven long/short trader. One position at a time."""

    def __init__(
        self,
        exchange: ExchangeInterface,
        state: StateStore,
        config: BotConfig,
        secrets: Secrets,
        notifier: Any = None,
    ) -> None:
        self.ex = exchange
        self.state = state
        self.cfg = config
        self.notifier = notifier
        self.risk = RiskManager(config.risk, config.grid)
        self.ai_strategy = AIDirectionalStrategy(
            ai_cfg=config.ai,
            dir_cfg=config.directional,
            api_key=secrets.ai_api_key,
            base_url=secrets.ai_base_url,
        )
        d = config.directional
        self.pm = PositionManager(
            exchange,
            trail_atr_mult=d.trail_atr_mult,
            trail_arm_atr=d.trail_arm_atr,
            trail_tighten_atr=d.trail_tighten_atr,
            trail_tighten_mult=d.trail_tighten_mult,
            breakeven_buffer_atr=d.breakeven_buffer_atr,
            breakeven_profit_pct=d.breakeven_profit_pct,
            breakeven_after_tp1=d.breakeven_after_tp1,
            giveback_arm_pct=d.giveback_arm_pct,
            giveback_exit_pct=d.giveback_exit_pct,
            time_stop_hours=d.time_stop_hours,
            max_loss_pct=d.max_loss_pct,
        )
        self._last_scan: Optional[datetime] = None
        self._recent_outcomes: List[Dict[str, Any]] = []

    # ---------------- lifecycle ----------------

    def has_position(self) -> bool:
        return self.pm.has_position()

    async def start(self) -> None:
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

    # ---------------- management ----------------

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
            entry = pos.entry_price
            peak_pct = pos.peak_profit_pct
            symbol = pos.symbol
            side = pos.side
            qty = pos.original_qty
            # Close at market; pm.close updates pos.realised_pnl with the
            # *actual* fill price and tracks fees across all partials.
            close_order = await self.pm.close(result.reason or "MANUAL", mark)
            exit_price = close_order.avg_price or mark
            gross_pnl = pos.realised_pnl
            fees = pos.fees_paid
            net_pnl = gross_pnl - fees
            self._record_exit(symbol, side, exit_price, result.reason, net_pnl)
            await self._notify_close(
                symbol, side, qty, entry, exit_price, result.reason,
                net_pnl, peak_pct, gross_pnl=gross_pnl, fees=fees,
            )

    def _record_exit(
        self, symbol: str, side: str, mark: float, reason, realised_pnl: float,
    ) -> None:
        trade_log.log("close", s=symbol, sd=side, p=mark, why=str(reason),
                      rp=realised_pnl)
        self.state.state.daily.trades += 1
        if realised_pnl > 0:
            self.state.state.daily.wins += 1
        elif realised_pnl < 0:
            self.state.state.daily.losses += 1
        self.state.state.daily.realized_pnl += realised_pnl
        self.state.save()
        self._recent_outcomes.append({
            "symbol": symbol,
            "side": side,
            "pnl": round(realised_pnl, 4),
            "reason": str(reason),
        })
        self._recent_outcomes = self._recent_outcomes[-10:]

    # ---------------- scan & open ----------------

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
        candidates = [
            c for c in candidates if not self.state.is_in_cooldown(c.symbol)
        ]
        if not candidates:
            logger.info("All candidates in cooldown")
            return

        # Build AI context — fetch klines for each shortlisted symbol.
        ctxs: List[_CandidateCtx] = []
        for cand in candidates:
            try:
                df15 = await self.ex.get_klines(cand.symbol, "15m", 150)
                df1h = await self.ex.get_klines(cand.symbol, "1h", 100)
            except Exception as e:
                logger.debug("klines failed for {}: {}", cand.symbol, e)
                continue
            if df15 is None or df1h is None or len(df15) < 60 or len(df1h) < 60:
                continue
            ctxs.append(_CandidateCtx(
                symbol=cand.symbol, ticker=cand.ticker, filters=cand.filters,
                df15=df15, df1h=df1h,
            ))

        if not ctxs:
            logger.info("No candidates with sufficient kline history")
            return

        logger.info(
            "AI directional scan: {} candidates, balance={:.4f}",
            len(ctxs), equity,
        )
        decision = await self.ai_strategy.decide(
            candidates=ctxs,
            balance=equity,
            recent_outcomes=self._recent_outcomes,
        )
        if decision is None:
            logger.warning("AI returned no decision — skipping")
            trade_log.log("skip", why="ai_no_decision")
            return
        if not decision.is_trade:
            logger.info("AI decided SKIP: {}", decision.reasoning)
            trade_log.log("skip", why=f"ai_skip:{decision.reasoning[:80]}")
            return
        if decision.confidence < self.cfg.directional.min_confidence:
            logger.info(
                "AI confidence {:.2f} below threshold {:.2f} — skipping",
                decision.confidence, self.cfg.directional.min_confidence,
            )
            trade_log.log(
                "skip", s=decision.symbol,
                why=f"ai_low_conf:{decision.confidence:.2f}",
            )
            return

        # Resolve the context for the chosen symbol.
        chosen = next((c for c in ctxs if c.symbol == decision.symbol), None)
        if chosen is None:
            logger.warning(
                "AI chose {} which is not in the shortlist; skipping",
                decision.symbol,
            )
            trade_log.log(
                "skip", s=decision.symbol, why="ai_symbol_off_list",
            )
            return

        await self._open_from_decision(chosen, decision, equity)

    async def _open_from_decision(
        self, cand: _CandidateCtx, decision: AIDecision, equity: float,
    ) -> None:
        assert decision.entry is not None and decision.stop_loss is not None
        # Fills at market. Anchor to the live mark price, not the AI's
        # remembered entry, to avoid a stale quote from the prompt.
        try:
            entry = await self.ex.get_mark_price(cand.symbol)
        except Exception:
            entry = decision.entry

        # Rescale SL + TPs if the AI's entry was different — preserve its
        # intended R-multiples relative to the real fill price.
        if decision.entry > 0 and abs(entry - decision.entry) / decision.entry > 0.002:
            ratio = entry / decision.entry
            stop_loss = decision.stop_loss * ratio
            take_profits = [(p * ratio, pct) for p, pct in decision.take_profits]
        else:
            stop_loss = decision.stop_loss
            take_profits = list(decision.take_profits)

        # Sanity: SL on right side of real entry.
        if decision.action == "OPEN_LONG" and stop_loss >= entry:
            logger.warning("Skipping: SL >= entry for long after rescale")
            return
        if decision.action == "OPEN_SHORT" and stop_loss <= entry:
            logger.warning("Skipping: SL <= entry for short after rescale")
            return

        # ATR estimate for trailing stop distance: derive from SL distance
        # as a reasonable proxy — if we don't have an ATR from the payload
        # we can scale from the chosen SL (usually ~1.5-2x ATR).
        sl_distance = abs(entry - stop_loss)
        atr_at_entry = sl_distance / 1.8 if sl_distance > 0 else 0.0
        atr_pct = atr_at_entry / entry * 100.0 if entry > 0 else 1.0

        # Leverage: use AI's choice if provided, else adaptive.
        max_lev = self.cfg.directional.max_leverage
        min_lev = self.cfg.directional.min_leverage
        if decision.leverage and decision.leverage > 0:
            leverage = max(min_lev, min(decision.leverage, max_lev))
        else:
            leverage = adaptive_leverage(
                decision.confidence, atr_pct,
                base_leverage=self.cfg.directional.base_leverage,
                max_leverage=max_lev, min_leverage=min_lev,
            )

        tp = compute_trade_plan(
            equity=equity,
            entry_price=entry, stop_price=stop_loss,
            filters=cand.filters,
            risk_pct=self.cfg.directional.risk_per_trade_pct,
            confidence=decision.confidence,
            atr_pct=atr_pct,
            base_leverage=leverage, max_leverage=max_lev,
            max_margin_pct=self.cfg.directional.max_margin_pct,
            min_leverage=min_lev,
        )
        if not tp.feasible:
            logger.info(
                "Trade plan infeasible for {}: {}", cand.symbol, tp.reason,
            )
            trade_log.log("skip", s=cand.symbol, why=f"infeasible:{tp.reason}")
            return

        side = "LONG" if decision.action == "OPEN_LONG" else "SHORT"
        try:
            await self.pm.open(
                symbol=cand.symbol,
                side=side,  # type: ignore[arg-type]
                qty=tp.qty,
                entry_price=entry,
                stop_loss=stop_loss,
                take_profits=take_profits,
                leverage=tp.leverage,
                atr_at_entry=atr_at_entry,
                filters=cand.filters,
                equity_at_open=equity,
                margin_type=self.cfg.margin_type,
            )
        except Exception as e:
            logger.exception("open failed for {}: {}", cand.symbol, e)
            trade_log.log("skip", s=cand.symbol, why=f"open_fail:{e}")
            return

        trade_log.log(
            "setup", s=cand.symbol, sd=side[:1],
            p=entry, q=tp.qty, lev=tp.leverage,
            sl=stop_loss,
            conf=f"{decision.confidence:.2f}",
            why=decision.reasoning[:80],
        )
        await self._notify_open(
            cand.symbol, side, tp.qty, entry, stop_loss,
            take_profits, tp.leverage, decision.confidence,
            decision.reasoning,
        )

    # ---------------- notifications ----------------

    async def _notify_open(
        self, symbol: str, side: str, qty: float, entry: float,
        stop_loss: float, take_profits: List[Tuple[float, float]],
        leverage: int, confidence: float, reasoning: str,
    ) -> None:
        if self.notifier is None:
            return
        sl_pct = abs(entry - stop_loss) / entry * 100.0 if entry > 0 else 0.0
        arrow = "\U0001f7e2" if side == "LONG" else "\U0001f534"
        tp_lines = "\n".join(
            f"  TP{i+1}: {p:.6f} ({pct:.0f}%)"
            for i, (p, pct) in enumerate(take_profits)
        )
        text = (
            f"{arrow} <b>OPEN {side}</b> {symbol}\n"
            f"Entry: <code>{entry:.6f}</code>  Qty: <code>{qty:.6f}</code>\n"
            f"Leverage: <b>{leverage}x</b>  Conf: {confidence:.2f}\n"
            f"SL: <code>{stop_loss:.6f}</code> ({sl_pct:.2f}%)\n"
            f"{tp_lines}\n"
            f"<i>{reasoning[:160]}</i>"
        )
        try:
            await self.notifier.send(text)
        except Exception as e:
            logger.debug("notify_open failed: {}", e)

    async def _notify_close(
        self, symbol: str, side: str, qty: float, entry: float, exit_price: float,
        reason: Any, pnl: float, peak_pct: float,
        gross_pnl: Optional[float] = None, fees: Optional[float] = None,
    ) -> None:
        if self.notifier is None:
            return
        move_pct = (
            (exit_price - entry) / entry * 100.0 if side == "LONG"
            else (entry - exit_price) / entry * 100.0
        ) if entry > 0 else 0.0
        emoji = "\U0001f7e2" if pnl > 0 else ("\U0001f534" if pnl < 0 else "\u26aa")
        breakdown = ""
        if gross_pnl is not None and fees is not None:
            breakdown = f"\n(gross {gross_pnl:+.4f} − fees {fees:.4f})"
        text = (
            f"{emoji} <b>CLOSE {side}</b> {symbol} ({reason})\n"
            f"Entry: <code>{entry:.6f}</code>  Exit: <code>{exit_price:.6f}</code>\n"
            f"Move: {move_pct:+.2f}%  Peak: {peak_pct:+.2f}%\n"
            f"Net PnL: <b>{pnl:+.4f} USDT</b>{breakdown}"
        )
        try:
            await self.notifier.send(text)
        except Exception as e:
            logger.debug("notify_close failed: {}", e)


__all__ = ["DirectionalTrader"]
