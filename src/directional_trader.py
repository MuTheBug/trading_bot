"""Directional trading loop (long / short / skip) driven by the
deterministic support/resistance strategy.

Each scan:

1. Prescreen all USDT-M tickers by liquidity + activity. With
   ``scan_top_n <= 0`` no cap is applied, so every liquid symbol is
   considered.
2. For each candidate (one by one, highest activity first) fetch MTF
   klines and build a `CandidateCtx`.
3. The `SRStrategy` inspects the frames, detects clustered pivots into
   support/resistance levels, classifies HTF bias, and proposes an
   entry at a pullback to a strong level with a rejection candle.
4. Remaining vetoes (HTF alignment, pullback-location sanity) then
   fire; accepted setups go through the adaptive leverage planner and
   on to the PositionManager.

No LLM call is made — the signal is fully reproducible.
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
from .indicators import atr as atr_ind, rsi as rsi_ind
from .position.manager import PositionManager
from .risk.leverage import TradePlan, adaptive_leverage, compute_trade_plan
from .risk.risk_manager import RiskManager
from .state import StateStore
from .strategy.sr_strategy import (
    CandidateCtx,
    SRDecision,
    SRStrategy,
)

# Public re-exports — tests and older callers expect these names here.
AIDecision = SRDecision  # backwards-compat alias
_CandidateCtx = CandidateCtx


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

    Set ``top_n <= 0`` to disable the cap and return every liquid symbol
    (still sorted by activity score so the most interesting ones are
    scanned first).

    The AI can only reason about the data we feed it, so this is just
    liquidity + "something's moving" gating. No directional bias here.

    Symbols that have already moved very far in 24h are penalized: a
    -20% day is usually an exhausted move, not an opportunity.
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
        # Favour symbols that actually moved (either direction), but cap
        # each contribution so extreme 24h movers don't dominate — they
        # tend to be exhausted by the time the scan sees them.
        abs_chg = abs(t.change_pct_24h)
        activity = min(abs_chg, 10.0) + 0.5 * min(range_pct, 15.0)
        # Additional multiplicative penalty for clearly-blown-off symbols.
        if abs_chg > 20.0:
            activity *= 0.3
        elif abs_chg > 12.0:
            activity *= 0.6
        score = activity * math.log10(max(t.volume_24h, 1.0))
        out.append(_Candidate(t.symbol, t, filt, score))
    out.sort(key=lambda c: c.score, reverse=True)
    if top_n is None or top_n <= 0:
        return out
    return out[:top_n]


class DirectionalTrader:
    """Deterministic S/R long/short trader. One position at a time."""

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
        self.strategy = SRStrategy(
            sr_cfg=config.sr,
            dir_cfg=config.directional,
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

    async def _build_mtf_ctx(
        self, cand: _Candidate,
    ) -> Optional[_CandidateCtx]:
        """Fetch klines for every configured MTF timeframe for one symbol.

        Returns None if any timeframe fails or has too few bars.
        """
        mtf_tfs = self.cfg.directional.mtf_timeframes or ["1h", "15m"]
        dfs: Dict[str, Any] = {}
        for tf in mtf_tfs:
            try:
                df = await self.ex.get_klines(cand.symbol, tf, 150)
            except Exception as e:
                logger.debug("klines failed for {} {}: {}", cand.symbol, tf, e)
                return None
            if df is None or len(df) < 60:
                return None
            dfs[tf] = df
        if not dfs:
            return None
        return _CandidateCtx(
            symbol=cand.symbol, ticker=cand.ticker, filters=cand.filters,
            dfs=dfs,
        )

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

        max_calls = self.cfg.directional.scan_max_ai_calls
        if max_calls and max_calls > 0:
            candidates = candidates[:max_calls]

        logger.info(
            "S/R scan: iterating {} candidates one-by-one, balance={:.4f}",
            len(candidates), equity,
        )

        # Walk candidates in score order. First one to produce a
        # qualifying S/R setup that also clears the vetoes wins.
        min_conf = self.cfg.directional.min_confidence
        for cand in candidates:
            ctx = await self._build_mtf_ctx(cand)
            if ctx is None:
                continue

            decision = self.strategy.propose(ctx)
            if not decision.is_trade:
                logger.debug(
                    "S/R skip on {}: {}", cand.symbol, decision.reasoning[:120],
                )
                continue
            if decision.confidence < min_conf:
                logger.debug(
                    "S/R conf {:.2f} on {} below {:.2f} — next",
                    decision.confidence, cand.symbol, min_conf,
                )
                continue

            veto, veto_reason = self._mtf_veto(ctx, decision)
            if veto:
                logger.info(
                    "MTF veto {} {}: {}",
                    decision.action, cand.symbol, veto_reason,
                )
                trade_log.log(
                    "skip", s=cand.symbol,
                    why=f"mtf_veto:{veto_reason[:80]}",
                )
                continue

            veto, veto_reason = self._pullback_veto(ctx, decision)
            if veto:
                logger.info(
                    "Pullback veto {} {}: {}",
                    decision.action, cand.symbol, veto_reason,
                )
                trade_log.log(
                    "skip", s=cand.symbol,
                    why=f"pullback_veto:{veto_reason[:80]}",
                )
                continue

            await self._open_from_decision(ctx, decision, equity)
            return

        logger.info("Scan finished: no tradeable setup across {} candidates",
                    len(candidates))

    def _mtf_veto(
        self, cand: _CandidateCtx, decision: AIDecision,
    ) -> Tuple[bool, str]:
        """Reject trades that fight both the 24h tape and the HTF regime.

        The veto only fires when the evidence is strong AND the AI didn't
        explicitly flag a reversal. High-confidence reversal calls are
        still allowed through.
        """
        from .strategy.regime import classify_regime  # local to avoid cycle

        wants_long = decision.action == "OPEN_LONG"
        reasoning = (decision.reasoning or "").lower()
        reversal_flag = any(
            w in reasoning for w in
            ("reversal", "oversold", "overbought", "capitulat",
             "exhaust", "bounce", "mean revert")
        )

        # 24h tape check. A LONG into a symbol down >10% over 24h is a
        # "falling knife"; a SHORT into a +10% day is "shorting the tape".
        chg24 = cand.ticker.change_pct_24h
        if wants_long and chg24 <= -10.0 and not reversal_flag:
            return True, f"24h {chg24:+.1f}% (falling knife)"
        if (not wants_long) and chg24 >= 10.0 and not reversal_flag:
            return True, f"24h {chg24:+.1f}% (chasing pump)"

        # HTF + next-TF regime check. Need both top TFs to disagree.
        opposing = 0
        opposing_tfs: List[str] = []
        for tf in list(cand.dfs.keys())[:2]:  # top two TFs (top-down)
            snap = classify_regime(cand.dfs[tf])
            if snap is None:
                continue
            dir_ok = (
                snap.direction == "BOTH"
                or (wants_long and snap.direction == "LONG")
                or ((not wants_long) and snap.direction == "SHORT")
            )
            if not dir_ok and snap.confidence >= 0.5:
                opposing += 1
                opposing_tfs.append(f"{tf}:{snap.regime}")
        if opposing >= 2 and not reversal_flag and decision.confidence < 0.75:
            return True, f"HTF+MTF oppose ({', '.join(opposing_tfs)})"

        return False, ""

    def _pullback_veto(
        self, cand: _CandidateCtx, decision: AIDecision,
    ) -> Tuple[bool, str]:
        """Require the entry to sit at a pullback bottom (LONG) or top (SHORT).

        Uses the LOWEST timeframe in the payload (the entry-trigger TF) to
        measure where price sits within the recent swing:

        * pullback depth  = recent_high - recent_low (over last N closed bars)
        * distance to extreme we're buying/selling, in ATR units

        A LONG is accepted only when price has pulled back at least 0.5 ATR
        from the recent high AND now sits within ~1 ATR of the recent low,
        with LTF RSI not already overbought. SHORT is the inverse.

        Reversal calls (AI flagged the setup as oversold/exhaustion/etc) are
        exempt — they're explicitly not trend-continuation pullbacks.
        """
        wants_long = decision.action == "OPEN_LONG"
        reasoning = (decision.reasoning or "").lower()
        reversal_flag = any(
            w in reasoning for w in
            ("reversal", "capitulat", "exhaust", "bottoming", "topping out")
        )
        if reversal_flag:
            return False, ""

        if not cand.dfs:
            return False, ""
        ltf_name = list(cand.dfs.keys())[-1]
        df = cand.dfs[ltf_name]
        if df is None or len(df) < 20:
            return False, ""

        # Work off the last CLOSED bar to keep the check reproducible.
        closed = df.iloc[:-1] if len(df) > 1 else df
        if len(closed) < 15:
            return False, ""

        atr_series = atr_ind(closed, 14)
        rsi_series = rsi_ind(closed["close"], 14)
        if atr_series.iloc[-1] != atr_series.iloc[-1]:  # NaN check
            return False, ""
        atr_val = float(atr_series.iloc[-1])
        last_rsi = float(rsi_series.iloc[-1])
        last_close = float(closed["close"].iloc[-1])
        if atr_val <= 0 or last_close <= 0:
            return False, ""

        window = closed.iloc[-10:]
        recent_high = float(window["high"].max())
        recent_low = float(window["low"].min())
        dist_from_high_atr = (recent_high - last_close) / atr_val
        dist_from_low_atr = (last_close - recent_low) / atr_val

        if wants_long:
            if dist_from_high_atr < 0.5:
                return True, (
                    f"no pullback: only {dist_from_high_atr:.2f} ATR below "
                    f"recent high on {ltf_name}"
                )
            if dist_from_low_atr > 1.2:
                return True, (
                    f"chasing: {dist_from_low_atr:.2f} ATR above recent "
                    f"{ltf_name} low — wait for next pullback"
                )
            if last_rsi > 65.0:
                return True, (
                    f"LTF RSI {last_rsi:.1f} overbought — not a pullback low"
                )
        else:
            if dist_from_low_atr < 0.5:
                return True, (
                    f"no pullback: only {dist_from_low_atr:.2f} ATR above "
                    f"recent low on {ltf_name}"
                )
            if dist_from_high_atr > 1.2:
                return True, (
                    f"chasing: {dist_from_high_atr:.2f} ATR below recent "
                    f"{ltf_name} high — wait for next bounce"
                )
            if last_rsi < 35.0:
                return True, (
                    f"LTF RSI {last_rsi:.1f} oversold — not a pullback high"
                )

        return False, ""

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
            hard_margin_pct=self.cfg.directional.hard_margin_pct,
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
