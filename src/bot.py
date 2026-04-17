"""Main grid trading bot loop.

The bot lifecycle:
1. Start up, connect exchange, scan symbols via AI
2. AI picks the best symbol and decides grid parameters
3. Set up the grid (place limit orders)
4. Main loop: check fills, place counter orders, periodically re-evaluate
5. If price moves out of range, ask AI to rebalance or hold
"""
from __future__ import annotations

import asyncio
import signal
from datetime import datetime, timezone
from typing import Literal, Optional

from loguru import logger

from . import trade_log
from .config import BotConfig, Secrets
from .exchange.base import ExchangeInterface
from .exchange.binance_live import BinanceLiveExchange
from .exchange.simulator import SimulatorExchange
from .grid.manager import GridManager, GridSetupParams
from .state import StateStore, _now_iso, _today_utc
from .strategy.ai_strategy import AIGridStrategy
from .telegram.bot import TelegramNotifier


_MAX_CONSECUTIVE_ERRORS = 5

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
        self.exchange: ExchangeInterface = self._build_exchange()
        self.grid_manager = GridManager(self.exchange, self.state)
        self.ai_strategy = self._build_ai_strategy()
        self.telegram = TelegramNotifier(
            token=secrets.telegram_bot_token,
            chat_id=secrets.telegram_chat_id,
            state=self.state,
            bot_ref=self,
            enabled=config.telegram.enabled,
        )
        self._running = False
        self._stop_event = asyncio.Event()
        self._lock = asyncio.Lock()
        self._consecutive_errors = 0
        self._last_rebalance_check: Optional[datetime] = None
        self._last_rebalance_action: Optional[datetime] = None
        self._ticks_since_start = 0

    def _build_ai_strategy(self) -> AIGridStrategy:
        logger.info("AI grid strategy: {}", self.config.ai.model)
        return AIGridStrategy(
            ai_cfg=self.config.ai,
            grid_cfg=self.config.grid,
            api_key=self.secrets.ai_api_key,
            base_url=self.secrets.ai_base_url,
        )

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
        logger.info("Starting grid trading bot in {} mode", self.mode.upper())
        trade_log.configure(self.config.trade_log_file)
        await self.exchange.connect()
        await self.telegram.start()

        # Clear auto-pause from previous session on fresh restart
        if self.state.state.paused:
            logger.info("Clearing paused state from previous session")
            self.state.state.paused = False
            self.state.save()

        # Clear stale grid from previous session — orders are gone after restart
        if self.state.state.grid.active:
            logger.info("Clearing stale grid from previous session (orders no longer exist)")
            self.state.state.grid.active = False
            self.state.state.grid.levels.clear()
            self.state.save()

        equity = await self.get_equity()
        if self.state.state.peak_equity < equity:
            self.state.state.peak_equity = equity
            self.state.save()

        self._running = True
        logger.info("Bot ready. Equity={:.4f} USDT", equity)

    async def stop(self) -> None:
        logger.info("Stopping bot...")
        self._running = False
        self._stop_event.set()
        try:
            await self.telegram.send("\U0001f6d1 Bot shutting down")
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
        if self.grid_manager.active:
            try:
                mark = await self.exchange.get_mark_price(self.grid_manager.grid.symbol)
                upnl = self.grid_manager.unrealized_pnl(mark)
                return bal + upnl
            except Exception:
                pass
        return bal

    async def force_teardown(self, reason: str = "MANUAL") -> None:
        """Tear down the active grid (used by Telegram commands)."""
        async with self._lock:
            if not self.grid_manager.active:
                raise RuntimeError("No active grid to tear down")
            await self.grid_manager.teardown()
            await self.telegram.send(
                f"\U0001f6d1 Grid torn down ({reason})"
            )

    # ---------- main loop ----------

    async def run(self) -> None:
        await self.start()
        try:
            while self._running:
                try:
                    await self._tick()
                    self._consecutive_errors = 0
                except Exception as e:
                    self._consecutive_errors += 1
                    logger.exception(
                        "Tick error ({}/{}): {}",
                        self._consecutive_errors, _MAX_CONSECUTIVE_ERRORS, e,
                    )
                    if self.config.telegram.alerts_on_error:
                        await self.telegram.send(
                            f"\u26a0\ufe0f Tick error ({self._consecutive_errors}/"
                            f"{_MAX_CONSECUTIVE_ERRORS}): <code>{e}</code>"
                        )
                    if self._consecutive_errors >= _MAX_CONSECUTIVE_ERRORS:
                        if not self.state.state.paused:
                            self.state.state.paused = True
                            self.state.save()
                            await self.telegram.send(
                                "\U0001f6d1 <b>Auto-paused</b> after "
                                f"{_MAX_CONSECUTIVE_ERRORS} consecutive errors. "
                                "Investigate logs and /resume when ready."
                            )
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
            # Log the outgoing day's totals right before they get reset.
            if self.state.state.daily.date != _today_utc():
                d = self.state.state.daily
                try:
                    eq = await self.get_equity()
                except Exception:
                    eq = 0.0
                trade_log.log(
                    "daily", date=d.date, pnl=d.realized_pnl, fees=d.fees_paid,
                    trades=d.trades, wins=d.wins, losses=d.losses, eq=eq,
                )
            self.state.roll_daily_if_needed()
            self._ticks_since_start += 1
            logger.info("Tick #{}", self._ticks_since_start)

            if self.state.state.paused:
                logger.warning("Bot is PAUSED. Use /resume to unpause.")
                return

            # If no active grid, try to set one up
            if not self.grid_manager.active:
                logger.info("No active grid, scanning for setup...")
                await self._scan_and_setup_grid()
                return

            gs = self.grid_manager.grid
            symbol = gs.symbol

            # Get current mark price
            mark_price = await self.exchange.get_mark_price(symbol)
            logger.info("Tick #{} | {} | mark={:.6f} | buys={} sells={} | trips={} | profit={:.6f}",
                        self._ticks_since_start, symbol, mark_price,
                        sum(1 for lv in gs.levels if lv.buy_order_id),
                        sum(1 for lv in gs.levels if lv.sell_order_id),
                        gs.round_trips, gs.total_profit)

            # Update peak equity
            equity = await self.get_equity()
            if equity > self.state.state.peak_equity:
                self.state.state.peak_equity = equity

            # Check for filled orders and place counter orders
            events = await self.grid_manager.check_fills_and_reorder(mark_price)

            for ev in events:
                side_emoji = "\U0001f7e2" if ev["side"] == "BUY" else "\U0001f534"
                await self.telegram.send(
                    f"{side_emoji} Grid <b>{ev['side']}</b> filled\n"
                    f"{symbol} @ {ev['price']:.8f}\n"
                    f"qty={ev['qty']:.8f} level={ev['level']}"
                )

            # Check max unrealized loss — only tear down on LOSSES, not
            # on winning trades that happen to have a large uPnL. Also
            # close the naked position so we don't keep bleeding after
            # teardown.
            upnl = self.grid_manager.unrealized_pnl(mark_price)
            if (
                equity > 0
                and upnl < 0
                and abs(upnl) / equity * 100 > self.config.grid.max_unrealized_loss_pct
            ):
                pct = abs(upnl) / equity * 100
                logger.warning(
                    "Grid uPnL {:.4f} exceeds max_unrealized_loss_pct {:.1f}%, closing grid",
                    upnl, self.config.grid.max_unrealized_loss_pct,
                )
                trade_log.log("stop", s=symbol, upnl=upnl, eq=equity, pct=pct)
                await self.telegram.send(
                    f"\U0001f6a8 <b>Max unrealized loss</b> exceeded ({upnl:+.4f} USDT)\n"
                    "Closing position and tearing down grid for safety."
                )
                await self.grid_manager.teardown(close_position=True)
                return

            # Check if price is out of range -> AI rebalance check
            if self.grid_manager.is_price_out_of_range(mark_price, self.config.grid.out_of_range_pct):
                await self._check_rebalance(mark_price)

            # Periodic rebalance check (even if price is in range)
            now = datetime.now(timezone.utc)
            if self._last_rebalance_check is None:
                self._last_rebalance_check = now
            minutes_since = (now - self._last_rebalance_check).total_seconds() / 60
            if minutes_since >= self.config.grid.rebalance_check_minutes:
                self._last_rebalance_check = now
                # Only do periodic check if price is actually wandering near edges
                gs = self.grid_manager.grid
                grid_range = gs.upper_price - gs.lower_price
                if grid_range > 0:
                    mid = (gs.upper_price + gs.lower_price) / 2
                    dist_from_mid_pct = abs(mark_price - mid) / grid_range * 100
                    if dist_from_mid_pct > 40:  # price in outer 20% of range
                        await self._check_rebalance(mark_price)

    async def _scan_and_setup_grid(self) -> None:
        """Scan symbols, let AI pick one, compute grid params, set up the grid."""
        logger.info("Scanning symbols for grid trading...")
        await self.telegram.send("\U0001f50d Scanning symbols for grid trading...")

        try:
            # 1. Fetch all tickers and filters
            logger.info("Fetching all tickers...")
            tickers = await self.exchange.get_all_tickers()
            logger.info("Fetching all symbol filters...")
            all_filters = await self.exchange.get_all_symbol_filters()
            balance = await self.exchange.get_balance()

            logger.info("Found {} USDT tickers, {} filters, balance={:.4f}",
                        len(tickers), len(all_filters), balance)

            # 2. AI picks the best symbol (scoring is math-based, AI just confirms)
            logger.info("Selecting best symbol for grid trading...")
            choice = await self.ai_strategy.select_symbol(tickers, all_filters)
            if choice is None:
                logger.error("Could not select a symbol")
                trade_log.log("skip", why="no_grid_friendly_symbol")
                await self.telegram.send("\u274c Could not select a symbol. Will retry.")
                return

            logger.info("Selected: {} score={:.1f} ({})", choice.symbol, choice.score, choice.reasoning)
            trade_log.log("sel", s=choice.symbol, score=choice.score,
                          why=choice.reasoning)
            await self.telegram.send(
                f"\U0001f3af Selected <b>{choice.symbol}</b> (score {choice.score:.1f})\n"
                f"<i>{choice.reasoning}</i>"
            )

            # 3. Get current price and filters for the chosen symbol
            mark_price = await self.exchange.get_mark_price(choice.symbol)
            filters = all_filters.get(choice.symbol)
            if filters is None:
                filters = await self.exchange.get_symbol_filters(choice.symbol)

            ticker = next((t for t in tickers if t.symbol == choice.symbol), None)
            if ticker is None:
                logger.error("Ticker for {} not found", choice.symbol)
                return

            # 4. Compute grid parameters mathematically (no AI)
            logger.info("Computing grid parameters for {}...", choice.symbol)
            decision = self.ai_strategy.compute_params(
                symbol=choice.symbol,
                current_price=mark_price,
                ticker=ticker,
                filters=filters,
                balance=balance,
            )
            if decision is None:
                logger.error("Cannot compute viable grid parameters for {}", choice.symbol)
                trade_log.log("skip", s=choice.symbol, why="no_viable_params")
                await self.telegram.send(
                    f"\u274c Cannot compute viable grid for {choice.symbol}. Will retry."
                )
                return

            # 5. Set up the grid
            params = GridSetupParams(
                symbol=choice.symbol,
                upper_price=decision.upper_price,
                lower_price=decision.lower_price,
                num_grids=decision.num_grids,
                leverage=decision.leverage,
                qty_per_grid=decision.qty_per_grid,
                reasoning=decision.reasoning,
            )

            success = await self.grid_manager.setup_grid(params, filters, mark_price)
            if success:
                await self.telegram.send(
                    f"\u2705 <b>Grid active</b> on {choice.symbol}\n"
                    f"Range: {decision.lower_price:.8f} - {decision.upper_price:.8f}\n"
                    f"Levels: {decision.num_grids} | Spacing: {decision.spacing:.8f}\n"
                    f"Leverage: {decision.leverage}x | Qty: {decision.qty_per_grid:.8f}\n"
                    f"Profit/trip: {decision.profit_per_trip:.6f} USDT\n"
                    f"<i>{decision.reasoning}</i>"
                )
                self._last_rebalance_check = datetime.now(timezone.utc)
            else:
                await self.telegram.send(
                    f"\u274c Grid setup failed for {choice.symbol}. Will retry."
                )

        except Exception as e:
            logger.exception("Error during symbol scan / grid setup: {}", e)
            await self.telegram.send(
                f"\u274c Grid setup error: <code>{e}</code>"
            )

    async def _check_rebalance(self, mark_price: float) -> None:
        """Check whether to rebalance the grid (pure math, no AI)."""
        gs = self.grid_manager.grid
        symbol = gs.symbol

        # Cooldown: don't rebalance more often than the configured interval.
        # Constant teardown/rebuild locks in losses and pays a full set of
        # new entry fees for no edge.
        now = datetime.now(timezone.utc)
        if self._last_rebalance_action is not None:
            minutes = (now - self._last_rebalance_action).total_seconds() / 60
            if minutes < self.config.grid.rebalance_check_minutes:
                logger.info(
                    "Rebalance cooldown active ({:.1f} of {} min) — skipping",
                    minutes, self.config.grid.rebalance_check_minutes,
                )
                return

        balance = await self.exchange.get_balance()
        filters = await self.exchange.get_symbol_filters(symbol)

        # Need ticker data for the rebalance computation
        tickers = await self.exchange.get_all_tickers()
        ticker = next((t for t in tickers if t.symbol == symbol), None)
        if ticker is None:
            logger.warning("Cannot get ticker for {} — skipping rebalance", symbol)
            return

        summary = self.grid_manager.grid_summary(mark_price)
        decision = self.ai_strategy.compute_rebalance(
            symbol=symbol,
            current_price=mark_price,
            ticker=ticker,
            filters=filters,
            balance=balance,
            grid_summary=summary,
        )

        logger.info("Rebalance decision: {} ({})", decision.action, decision.reasoning)
        trade_log.log("rebal", s=symbol, a=decision.action, why=decision.reasoning)

        if decision.action == "EXIT":
            await self.telegram.send(
                f"\U0001f6d1 <b>Exiting grid</b> on {symbol}\n"
                f"<i>{decision.reasoning}</i>"
            )
            await self.grid_manager.teardown(close_position=True)
            self._last_rebalance_action = now
            return

        if decision.action == "REBALANCE" and decision.new_params is not None:
            await self.telegram.send(
                f"\U0001f504 <b>Rebalancing grid</b> on {symbol}\n"
                f"<i>{decision.reasoning}</i>"
            )
            # setup_grid internally tears down + closes naked inventory
            success = await self.grid_manager.setup_grid(
                decision.new_params, filters, mark_price
            )
            if success:
                p = decision.new_params
                spacing = (p.upper_price - p.lower_price) / p.num_grids
                await self.telegram.send(
                    f"\u2705 <b>Grid rebalanced</b> on {symbol}\n"
                    f"Range: {p.lower_price:.8f} - {p.upper_price:.8f}\n"
                    f"Levels: {p.num_grids} | Spacing: {spacing:.8f}\n"
                    f"Leverage: {p.leverage}x | Qty: {p.qty_per_grid:.8f}"
                )
            else:
                await self.telegram.send(
                    f"\u274c Grid rebalance failed for {symbol}"
                )
            self._last_rebalance_check = now
            self._last_rebalance_action = now
        else:
            logger.info("Holding grid: {}", decision.reasoning)

    def build_daily_summary(self) -> str:
        """Build daily summary text for Telegram / logging."""
        d = self.state.state.daily
        gs = self.state.state.grid
        lines = [
            "\U0001f4ca <b>Daily Summary</b>",
            f"Date: {d.date}",
            f"Realized PnL: <b>{d.realized_pnl:+.6f} USDT</b>",
            f"Fees paid: {d.fees_paid:.6f} USDT",
            f"Trades: {d.trades} (W:{d.wins}/L:{d.losses})",
            f"Win rate: {d.win_rate:.1f}%",
        ]
        if gs.active:
            lines.extend([
                f"\nGrid: <b>{gs.symbol}</b>",
                f"Range: {gs.lower_price:.8f} - {gs.upper_price:.8f}",
                f"Round trips: {gs.round_trips}",
                f"Grid profit: {gs.total_profit:.6f} USDT",
                f"Grid fees: {gs.total_fees:.6f} USDT",
            ])
        return "\n".join(lines)


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
