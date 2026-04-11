"""Main trading bot loop."""
from __future__ import annotations

import asyncio
import signal
from datetime import datetime, timedelta, timezone
from typing import Dict, Literal, Optional

from loguru import logger

from .config import BotConfig, Secrets
from .exchange.base import ExchangeInterface, LivePosition
from .exchange.binance_live import BinanceLiveExchange
from .exchange.simulator import SimulatorExchange
from .risk.position_sizer import compute_position_size
from .risk.risk_manager import ExitDecision, RiskManager
from .state import (
    Position,
    StateStore,
    TakeProfitLevel,
    TradeRecord,
)
from .strategy.base import Signal
from .strategy.trend_momentum import TrendMomentumStrategy
from .telegram.bot import TelegramNotifier


Mode = Literal["sim", "live"]


class TradingBot:
    def __init__(
        self,
        mode: Mode,
        config: BotConfig,
        secrets: Secrets,
    ) -> None:
        self.mode: Mode = mode
        self.config = config
        self.secrets = secrets
        self.state = StateStore(config.state_file)
        self.strategy = TrendMomentumStrategy(config.strategy)
        self.risk = RiskManager(config.risk, config.exits)
        self.exchange: ExchangeInterface = self._build_exchange()
        self.telegram = TelegramNotifier(
            token=secrets.telegram_bot_token,
            chat_id=secrets.telegram_chat_id,
            state=self.state,
            bot_ref=self,
            enabled=config.telegram.enabled,
        )
        self._running = False
        self._stop_event = asyncio.Event()
        # Protects state mutations that can race between loop tick & /close command
        self._lock = asyncio.Lock()

    def _build_exchange(self) -> ExchangeInterface:
        if self.mode == "live":
            if not self.secrets.binance_api_key or not self.secrets.binance_api_secret:
                raise RuntimeError(
                    "Live mode requires BINANCE_API_KEY and BINANCE_API_SECRET in .env"
                )
            return BinanceLiveExchange(
                api_key=self.secrets.binance_api_key,
                api_secret=self.secrets.binance_api_secret,
                testnet=self.secrets.binance_testnet,
            )
        return SimulatorExchange(
            starting_balance=self.config.simulator.starting_balance,
            taker_fee_pct=self.config.simulator.taker_fee_pct,
            slippage_ticks=self.config.simulator.slippage_ticks,
        )

    # ---------- lifecycle ----------

    async def start(self) -> None:
        logger.info("Starting trading bot in {} mode", self.mode.upper())
        await self.exchange.connect()
        await self.telegram.start()

        # Initial setup: leverage + margin type per symbol
        for sym in self.config.symbols:
            try:
                await self.exchange.set_margin_type(sym, self.config.margin_type)
                await self.exchange.set_leverage(sym, self.config.leverage)
            except Exception as e:
                logger.warning("Setup {}: {}", sym, e)

        equity = await self.get_equity()
        if self.state.state.peak_equity < equity:
            self.state.state.peak_equity = equity
            self.state.save()

        self._running = True
        logger.info("Bot ready. Equity={:.4f} USDT  Symbols={}",
                    equity, self.config.symbols)

    async def stop(self) -> None:
        logger.info("Stopping bot...")
        self._running = False
        self._stop_event.set()
        try:
            await self.telegram.send("🛑 Bot shutting down")
            await self.telegram.stop()
        except Exception:
            pass
        try:
            await self.exchange.close()
        except Exception:
            pass
        self.state.save()
        logger.info("Bot stopped.")

    # ---------- public helpers used by Telegram commands ----------

    async def get_equity(self) -> float:
        bal = await self.exchange.get_balance()
        # Add unrealized PnL
        try:
            positions = await self.exchange.get_open_positions()
            upnl = sum(p.unrealized_pnl for p in positions)
        except Exception:
            upnl = 0.0
        return bal + upnl

    async def force_close(self, symbol: str, reason: str = "MANUAL") -> None:
        async with self._lock:
            pos = self.state.state.positions.get(symbol)
            if not pos:
                raise RuntimeError(f"No tracked position for {symbol}")
            close_side = "SELL" if pos.side == "LONG" else "BUY"
            order = await self.exchange.market_close(
                symbol=symbol, side=close_side, qty=pos.remaining_qty
            )
            self._finalize_close(pos, order.avg_price, order.fee, reason,
                                 closed_qty=pos.remaining_qty, full_close=True)

    # ---------- main loop ----------

    async def run(self) -> None:
        await self.start()
        try:
            while self._running:
                try:
                    await self._tick()
                except Exception as e:
                    logger.exception("Tick error: {}", e)
                    if self.config.telegram.alerts_on_error:
                        await self.telegram.send(f"⚠️ Tick error: <code>{e}</code>")
                try:
                    await asyncio.wait_for(
                        self._stop_event.wait(),
                        timeout=self.config.loop_interval_seconds,
                    )
                except asyncio.TimeoutError:
                    pass
        finally:
            await self.stop()

    async def _tick(self) -> None:
        async with self._lock:
            self.state.roll_daily_if_needed()
            equity = await self.get_equity()
            if equity > self.state.state.peak_equity:
                self.state.state.peak_equity = equity

            # 1) manage existing positions
            for sym in list(self.state.state.positions.keys()):
                await self._manage_position(sym)

            # 2) scan for new entries
            for sym in self.config.symbols:
                if sym in self.state.state.positions:
                    continue
                ok, reason = self.risk.can_open_new(self.state, equity, sym)
                if not ok:
                    logger.debug("Skip {}: {}", sym, reason)
                    continue
                await self._try_enter(sym, equity)

    async def _manage_position(self, symbol: str) -> None:
        pos = self.state.state.positions.get(symbol)
        if pos is None:
            return
        mark = await self.exchange.get_mark_price(symbol)

        # Funding-rate forced exit
        try:
            funding = await self.exchange.get_funding_rate(symbol)
            action = self.risk.funding_action(funding, pos.side)
            if action == "exit":
                logger.warning("{} funding {:.2f}% -> forced exit", symbol, funding)
                close_side = "SELL" if pos.side == "LONG" else "BUY"
                order = await self.exchange.market_close(
                    symbol, close_side, pos.remaining_qty
                )
                self._finalize_close(pos, order.avg_price, order.fee, "FUNDING",
                                     closed_qty=pos.remaining_qty, full_close=True)
                return
        except Exception as e:
            logger.debug("funding check {}: {}", symbol, e)

        decision = self.risk.manage_position(pos, mark)
        if decision is None:
            self.state.save()
            return

        close_side = "SELL" if pos.side == "LONG" else "BUY"
        close_qty = pos.remaining_qty * (decision.close_qty_pct / 100.0)

        # Snap to step size
        try:
            filters = await self.exchange.get_symbol_filters(symbol)
            import math
            if filters.qty_step > 0:
                close_qty = math.floor(close_qty / filters.qty_step) * filters.qty_step
            if close_qty < filters.min_qty:
                close_qty = pos.remaining_qty
        except Exception:
            pass

        order = await self.exchange.market_close(symbol, close_side, close_qty)

        full_close = (
            abs(pos.remaining_qty - close_qty) < 1e-12
            or decision.close_qty_pct >= 100.0
            or decision.reason in ("SL", "TP3", "TRAIL", "TIME")
        )
        self._finalize_close(
            pos, order.avg_price, order.fee, decision.reason,
            closed_qty=close_qty, full_close=full_close,
            new_stop_loss=decision.new_stop_loss,
            enable_trailing=decision.enable_trailing,
        )

    def _finalize_close(
        self,
        pos: Position,
        exit_price: float,
        fee: float,
        reason: str,
        closed_qty: float,
        full_close: bool,
        new_stop_loss: Optional[float] = None,
        enable_trailing: bool = False,
    ) -> None:
        # PnL on the closed portion
        if pos.side == "LONG":
            pnl = (exit_price - pos.entry_price) * closed_qty
        else:
            pnl = (pos.entry_price - exit_price) * closed_qty
        pnl_net = pnl - fee

        trade = TradeRecord(
            symbol=pos.symbol,
            side=pos.side,
            entry_price=pos.entry_price,
            exit_price=exit_price,
            qty=closed_qty,
            pnl=pnl_net,
            fees=fee,
            opened_at=pos.opened_at,
            closed_at=datetime.now(timezone.utc).isoformat(),
            exit_reason=reason,
        )
        self.state.record_trade(trade)

        if full_close:
            self.state.remove_position(pos.symbol)
            if pnl_net < 0:
                until = datetime.now(timezone.utc) + timedelta(
                    minutes=self.config.risk.cooldown_after_loss_minutes
                )
                self.state.set_cooldown(pos.symbol, until.isoformat())
        else:
            pos.remaining_qty -= closed_qty
            if new_stop_loss is not None:
                pos.stop_loss = new_stop_loss
            if enable_trailing:
                pos.trailing_active = True
            self.state.save()

        # Alert
        if self.config.telegram.alerts_on_exit:
            asyncio.create_task(self.telegram.send(
                f"{'🟢' if pnl_net > 0 else '🔴'} <b>{reason}</b> {pos.symbol} {pos.side}\n"
                f"qty={closed_qty:g} @ {exit_price:.6f}\n"
                f"PnL: <b>{pnl_net:+.4f} USDT</b>"
            ))

    async def _try_enter(self, symbol: str, equity: float) -> None:
        # Fetch candles
        df15 = await self.exchange.get_klines(
            symbol, self.config.timeframe, limit=self.config.kline_history
        )
        df1h = await self.exchange.get_klines(
            symbol, self.config.htf_timeframe, limit=max(self.config.strategy.ema_htf + 20, 100)
        )

        signal: Optional[Signal] = self.strategy.evaluate(df15, df1h)
        if signal is None:
            return

        # Funding check
        try:
            funding = await self.exchange.get_funding_rate(symbol)
            action = self.risk.funding_action(funding, signal.side)
            if action == "skip":
                logger.info("Skip {}: funding {:.2f}% too high", symbol, funding)
                return
        except Exception as e:
            logger.debug("funding fetch {}: {}", symbol, e)

        # Build SL / TPs
        sl, tps = self.risk.build_exit_ladder(
            entry=signal.entry_price, side=signal.side, atr=signal.atr
        )

        # Size
        filters = await self.exchange.get_symbol_filters(symbol)
        sizing = compute_position_size(
            equity=equity,
            risk_pct=self.config.risk.risk_per_trade_pct,
            entry_price=signal.entry_price,
            stop_price=sl,
            filters=filters,
            leverage=self.config.leverage,
        )
        if not sizing.feasible:
            logger.info("Skip {}: sizing infeasible ({})", symbol, sizing.reason)
            return

        # Enter
        order_side = "BUY" if signal.side == "LONG" else "SELL"
        order = await self.exchange.market_open(
            symbol=symbol, side=order_side, qty=sizing.qty
        )
        entry_price = order.avg_price or signal.entry_price

        # Rebuild TP/SL around the actual fill price to stay precise
        sl, tps = self.risk.build_exit_ladder(entry_price, signal.side, signal.atr)

        pos = Position(
            symbol=symbol,
            side=signal.side,
            entry_price=entry_price,
            original_qty=sizing.qty,
            remaining_qty=sizing.qty,
            leverage=self.config.leverage,
            stop_loss=sl,
            take_profits=tps,
            opened_at=datetime.now(timezone.utc).isoformat(),
            atr_at_entry=signal.atr,
            highest_since_entry=entry_price,
            lowest_since_entry=entry_price,
        )
        self.state.add_position(pos)

        # Record the entry fee as a separate "0 pnl" trade so daily fees accounting stays
        # correct without double-counting; we attribute it to the eventual exit's PnL minus
        # this fee implicitly — instead we just log it now and let exit trades carry fee.
        logger.info(
            "OPEN {} {} qty={} entry={:.6f} SL={:.6f} TP1={:.6f} reason='{}'",
            signal.side, symbol, sizing.qty, entry_price, sl,
            tps[0].price, signal.reason,
        )

        if self.config.telegram.alerts_on_entry:
            await self.telegram.send(
                f"🚀 <b>OPEN {signal.side}</b> {symbol}\n"
                f"entry={entry_price:.6f}  qty={sizing.qty:g}\n"
                f"SL={sl:.6f}\n"
                f"TP1={tps[0].price:.6f}  TP2={tps[1].price:.6f}  TP3={tps[2].price:.6f}\n"
                f"<i>{signal.reason}</i>"
            )


# ---------- entry-point helper ----------

async def run_bot(mode: Mode, config: BotConfig, secrets: Secrets) -> None:
    bot = TradingBot(mode=mode, config=config, secrets=secrets)

    loop = asyncio.get_running_loop()

    def _signal_handler() -> None:
        logger.info("Received shutdown signal")
        bot._stop_event.set()
        bot._running = False

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _signal_handler)
        except NotImplementedError:  # pragma: no cover
            pass

    await bot.run()
