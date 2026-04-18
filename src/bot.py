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
from datetime import datetime, timedelta, timezone
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
        # After any close, pause scans for post_exit_cooldown_minutes so
        # we don't immediately re-enter at a worse price / re-pay fees.
        self._pause_scans_until: Optional[datetime] = None

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
            await self._cleanup_stale_orders(self.state.state.grid.symbol)
            self.state.state.grid.active = False
            self.state.state.grid.levels.clear()
            self.state.state.grid.net_qty = 0.0
            self.state.state.grid.avg_entry = 0.0
            self.state.save()
        elif self.state.state.grid.symbol:
            # No active grid but a previous symbol is remembered — make sure
            # no stray orders are still sitting on the exchange (e.g. the bot
            # crashed mid-teardown).
            await self._cleanup_stale_orders(self.state.state.grid.symbol)

        # Close any orphaned positions ACROSS THE WHOLE ACCOUNT. If the
        # previous session lost a position during a TP or stop (or the
        # user fat-fingered a manual trade), it would otherwise sit
        # bleeding while a new grid stacks on top.
        try:
            closed = await self.grid_manager.reconcile_positions()
            if closed:
                logger.warning("Reconciled {} orphaned position(s) on startup", closed)
        except Exception as e:
            logger.warning("Startup reconcile failed: {}", e)

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

    async def _cleanup_stale_orders(self, symbol: str) -> None:
        """Cancel any leftover orders for ``symbol`` on the exchange.

        Addresses the "54 stray open orders" case where previous grid
        setups were interrupted before their orders got cancelled.
        """
        if not symbol:
            return
        try:
            orders = await self.exchange.get_open_orders(symbol)
            if not orders:
                return
            logger.warning("Found {} stale open orders on {}, cancelling", len(orders), symbol)
            n = await self.exchange.cancel_all_orders(symbol)
            trade_log.log("cleanup", s=symbol, n=n)
        except Exception as e:  # pragma: no cover — defensive
            logger.warning("Stale-order cleanup failed for {}: {}", symbol, e)

    def _set_symbol_cooldown(self, symbol: str, minutes: int) -> None:
        """Set a per-symbol cooldown so we don't immediately re-pick it."""
        if not symbol or minutes <= 0:
            return
        until = datetime.now(timezone.utc) + timedelta(minutes=minutes)
        self.state.set_cooldown(symbol, until.isoformat())

    def _set_post_exit_cooldown(self) -> None:
        """After any close, don't start a new grid for a few minutes.

        Prevents the pattern observed in real trades where the bot
        tore down a grid at a small loss, immediately scanned, picked
        the same or similar symbol, and paid a fresh set of entry fees
        before the market had stabilized.
        """
        mins = self.config.grid.post_exit_cooldown_minutes
        if mins <= 0:
            return
        self._pause_scans_until = (
            datetime.now(timezone.utc) + timedelta(minutes=mins)
        )
        logger.info("Post-exit cooldown: scanning paused for {} min", mins)

    async def _equity_now(self) -> float:
        """Fetch current balance + uPnL as a float (0.0 on failure)."""
        try:
            return await self.get_equity()
        except Exception as e:  # pragma: no cover — defensive
            logger.warning("get_equity failed: {}", e)
            return 0.0

    async def _balance_now(self) -> float:
        """Fetch spot balance (cash, excluding uPnL). 0.0 on failure."""
        try:
            return await self.exchange.get_balance()
        except Exception as e:  # pragma: no cover — defensive
            logger.warning("get_balance failed: {}", e)
            return 0.0

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

            # If no active grid, try to set one up — but honor the
            # post-exit cooldown so we don't immediately re-enter after
            # a losing trade.
            if not self.grid_manager.active:
                if (
                    self._pause_scans_until is not None
                    and datetime.now(timezone.utc) < self._pause_scans_until
                ):
                    remaining = (
                        self._pause_scans_until - datetime.now(timezone.utc)
                    ).total_seconds()
                    logger.info(
                        "Post-exit cooldown: {:.0f}s remaining before next scan",
                        remaining,
                    )
                    return
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

            upnl = self.grid_manager.unrealized_pnl(mark_price)

            # Track peak equity since setup (anchors trailing TP)
            if equity > gs.peak_equity_since_setup:
                gs.peak_equity_since_setup = equity
                self.state.save()

            # --- Drift exit: price has trended far from grid center ---
            # Gets us out before the position stop fires on a runaway
            # move. Even if avg_entry hasn't been hit hard yet, a big
            # drift means the grid will keep loading into the trend and
            # the equity stop will fire soon anyway — better to exit here
            # with a smaller loss than ride it down.
            if await self._check_drift_exit(mark_price, symbol):
                return

            # --- Per-tick position stop-loss on the net inventory ---
            stopped = await self._check_position_stop(mark_price, symbol)
            if stopped:
                return

            # --- Trailing take-profit: once we've been meaningfully up,
            # don't let the gain collapse back to zero. Fires faster than
            # the fixed take_profit_pct and captures wins that would
            # otherwise be given back on the next adverse wick.
            if await self._check_trailing_tp(equity, symbol):
                return

            # --- Take-profit on the aggregate grid PnL ---
            # If (realized + unrealized) gain has reached the configured %
            # of equity, lock it in. Grids that don't harvest wins
            # eventually give them back.
            tp_hit = await self._check_take_profit(mark_price, upnl, equity, symbol)
            if tp_hit:
                return

            # Check max unrealized loss — only tear down on LOSSES, not
            # on winning trades that happen to have a large uPnL. Report
            # the actual equity delta after close, not a pre-close estimate.
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
                await self._close_and_report(
                    symbol,
                    reason=f"upnl {upnl:+.4f} ({pct:.1f}% > {self.config.grid.max_unrealized_loss_pct:.1f}%)",
                    tag="stop",
                )
                return

            # Heartbeat: periodic compact snapshot of grid health for AI review
            hb = self.config.grid.heartbeat_ticks
            if hb > 0 and self._ticks_since_start % hb == 0:
                trade_log.log(
                    "tick", s=symbol, p=mark_price, nq=gs.net_qty,
                    ae=gs.avg_entry, upnl=upnl, rp=gs.total_profit,
                    rt=gs.round_trips, eq=equity,
                )

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

    def _grid_age_minutes(self) -> float:
        """Minutes since the current grid was set up (0 if unknown)."""
        gs = self.grid_manager.grid
        if not gs.setup_at:
            return 0.0
        try:
            t0 = datetime.fromisoformat(gs.setup_at)
        except ValueError:
            return 0.0
        return (datetime.now(timezone.utc) - t0).total_seconds() / 60.0

    async def _close_and_report(
        self, symbol: str, reason: str, tag: str,
    ) -> float:
        """Tear down the grid and report the REAL equity delta afterwards.

        This is the fix for the "Telegram said I won +$0.13 while I
        lost" bug. Previously the bot computed a pre-close estimate
        (realized + uPnL) and broadcast that number, then proceeded to
        close at taker fees that ate most or all of the estimated gain.
        Now we close first, measure (post_equity - starting_equity),
        and use that single exchange-truth number for logs, trade_log,
        and Telegram. If the close fails, the real delta is whatever
        the exchange produced and we still report it honestly.
        """
        gs = self.grid_manager.grid
        starting_equity = gs.starting_equity or 0.0

        await self.grid_manager.teardown(close_position=True)

        post_balance = await self._balance_now()
        # After teardown, grid is flat so equity == balance. Use balance
        # directly — it's exchange truth, not a derived number.
        realized_delta = (
            post_balance - starting_equity
            if starting_equity > 0 else 0.0
        )
        delta_pct = (
            realized_delta / starting_equity * 100.0
            if starting_equity > 0 else 0.0
        )

        outcome = "WIN" if realized_delta > 0 else (
            "LOSS" if realized_delta < 0 else "FLAT"
        )
        trade_log.log(
            tag, s=symbol, pnl=realized_delta, pct=delta_pct,
            se=starting_equity, eq=post_balance, why=reason,
        )
        emoji = "\U0001f3af" if realized_delta > 0 else "\U0001f6a8"
        await self.telegram.send(
            f"{emoji} <b>{tag.upper()} — {outcome}</b> on {symbol}\n"
            f"Reason: {reason}\n"
            f"Equity: {starting_equity:.4f} → {post_balance:.4f} USDT\n"
            f"Realized: <b>{realized_delta:+.4f}</b> USDT ({delta_pct:+.2f}%)"
        )
        self._set_symbol_cooldown(symbol, self.config.grid.symbol_cooldown_minutes)
        self._set_post_exit_cooldown()
        return realized_delta

    async def _check_drift_exit(self, mark_price: float, symbol: str) -> bool:
        """Early exit when price has drifted too far from grid center.

        Neutral grids lose in trending markets because every fill
        stacks more naked inventory into the direction the market is
        leaving. By the time the 3%-of-equity stop fires, we've
        already paid a full set of adverse fills. This check bails
        BEFORE the grid has a chance to load up further.
        """
        pct = self.config.grid.drift_exit_pct
        gs = self.grid_manager.grid
        if pct <= 0 or gs.upper_price <= gs.lower_price:
            return False
        if self._grid_age_minutes() < self.config.grid.min_hold_minutes:
            return False
        center = (gs.upper_price + gs.lower_price) / 2
        if center <= 0:
            return False
        drift = abs(mark_price - center) / center * 100.0
        if drift < pct:
            return False
        logger.warning(
            "Drift exit on {}: mark {:.8f} is {:.2f}% from center {:.8f} (>= {:.2f}%)",
            symbol, mark_price, drift, center, pct,
        )
        await self._close_and_report(
            symbol, reason=f"drift {drift:.2f}% from center", tag="drift",
        )
        return True

    async def _check_trailing_tp(self, equity: float, symbol: str) -> bool:
        """Trailing take-profit anchored to start equity.

        Arms once gain reaches ``trailing_tp_arm_pct`` of start equity.
        Once armed, exits as soon as the gain gives back
        ``trailing_tp_giveback_pct`` of start equity from its peak.
        """
        gs = self.grid_manager.grid
        starting = gs.starting_equity
        if starting <= 0 or equity <= 0:
            return False
        if self._grid_age_minutes() < self.config.grid.min_hold_minutes:
            return False

        arm_pct = self.config.grid.trailing_tp_arm_pct
        give_pct = self.config.grid.trailing_tp_giveback_pct
        if arm_pct <= 0 or give_pct <= 0:
            return False

        gain_pct = (equity - starting) / starting * 100.0
        peak_gain_pct = (gs.peak_equity_since_setup - starting) / starting * 100.0

        if not gs.trailing_armed and gain_pct >= arm_pct:
            gs.trailing_armed = True
            self.state.save()
            logger.info(
                "Trailing TP armed on {}: gain {:.2f}% of start equity", symbol, gain_pct,
            )

        if not gs.trailing_armed:
            return False

        giveback_pct = peak_gain_pct - gain_pct
        if giveback_pct < give_pct:
            return False

        logger.info(
            "Trailing TP firing on {}: peak {:.2f}% -> now {:.2f}% (gave back {:.2f}%)",
            symbol, peak_gain_pct, gain_pct, giveback_pct,
        )
        await self._close_and_report(
            symbol,
            reason=f"trail: peak {peak_gain_pct:.2f}% -> {gain_pct:.2f}%",
            tag="trail_tp",
        )
        return True

    async def _check_position_stop(self, mark_price: float, symbol: str) -> bool:
        """Direct stop-loss on the weighted-average entry price.

        Unlike the equity-ratio check, this fires even on moderate equity
        when the naked position is moving fast against us. Returns True
        if the stop fired and the grid was torn down.

        Requires ``min_hold_minutes`` to have elapsed since setup — we
        don't want a single ugly tick right after setup to unwind a grid
        that hasn't had a chance to even place its counter orders.
        """
        gs = self.grid_manager.grid
        pct = self.config.grid.position_stop_loss_pct
        if pct <= 0 or abs(gs.net_qty) < 1e-12 or gs.avg_entry <= 0:
            return False
        if self._grid_age_minutes() < self.config.grid.min_hold_minutes:
            return False

        if gs.net_qty > 0:
            adverse = gs.avg_entry * (1 - pct / 100.0)
            tripped = mark_price <= adverse
        else:
            adverse = gs.avg_entry * (1 + pct / 100.0)
            tripped = mark_price >= adverse
        if not tripped:
            return False

        logger.warning(
            "Position stop-loss tripped on {}: mark {:.8f} vs avg {:.8f} "
            "({:.2f}% adverse)",
            symbol, mark_price, gs.avg_entry, pct,
        )
        await self._close_and_report(
            symbol, reason=f"pos_sl {pct:.1f}% vs avg", tag="stop",
        )
        return True

    async def _check_take_profit(
        self, mark_price: float, upnl: float, equity: float, symbol: str,
    ) -> bool:
        """Close the grid once actual (exchange-truth) gain >= threshold.

        Gain is measured as (current_equity - starting_equity), NOT as
        synthetic accounting (gs.total_profit + upnl). The synthetic
        number was booking maker-fee fills but every teardown paid
        taker fees, so on every TP the Telegram reported a gain that
        never actually hit the account.

        A streak filter requires N consecutive ticks above threshold
        before firing, so a volatile wick doesn't collapse a winning
        grid at a momentary mark spike.
        """
        pct = self.config.grid.take_profit_pct
        gs = self.grid_manager.grid
        starting_equity = gs.starting_equity
        if pct <= 0 or starting_equity <= 0 or equity <= 0:
            return False
        if self._grid_age_minutes() < self.config.grid.min_hold_minutes:
            return False

        actual_gain = equity - starting_equity
        gain_pct = actual_gain / starting_equity * 100.0
        if gain_pct < pct:
            # Reset streak so only CONSECUTIVE over-threshold ticks count
            if gs.tp_streak != 0:
                gs.tp_streak = 0
                self.state.save()
            return False

        gs.tp_streak += 1
        self.state.save()
        required = max(1, self.config.grid.take_profit_streak)
        if gs.tp_streak < required:
            logger.info(
                "TP candidate on {}: gain {:.4f} ({:.2f}%) streak {}/{}",
                symbol, actual_gain, gain_pct, gs.tp_streak, required,
            )
            return False

        logger.info(
            "Take-profit firing on {}: actual_gain={:.4f} ({:.2f}% of start eq)",
            symbol, actual_gain, gain_pct,
        )
        await self._close_and_report(
            symbol, reason=f"tp_streak {gs.tp_streak}x @ {gain_pct:.2f}%",
            tag="tp",
        )
        return True

    async def _scan_and_setup_grid(self) -> None:
        """Scan symbols, let AI pick one, compute grid params, set up the grid."""
        logger.info("Scanning symbols for grid trading...")
        await self.telegram.send("\U0001f50d Scanning symbols for grid trading...")

        # Belt-and-suspenders: before we commit to a new grid, make sure
        # no residual position is sitting open on any symbol. This closes
        # the "TP leaves a bag, next grid stacks on top" bug class.
        try:
            closed = await self.grid_manager.reconcile_positions()
            if closed:
                logger.warning(
                    "Pre-setup reconcile closed {} orphan position(s)", closed,
                )
        except Exception as e:
            logger.warning("Pre-setup reconcile failed: {}", e)

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
            # Filter out symbols that are still in cooldown from a recent
            # bad exit — re-entering the same losing symbol is a leading
            # cause of PnL spiraling.
            cooling = [
                sym for sym in list(self.state.state.cooldown_until.keys())
                if self.state.is_in_cooldown(sym)
            ]
            if cooling:
                before = len(tickers)
                tickers = [t for t in tickers if t.symbol not in set(cooling)]
                logger.info(
                    "Filtered {} cooldown symbol(s): {} — {}/{} candidates remain",
                    len(cooling), ",".join(cooling[:5]), len(tickers), before,
                )
            logger.info("Selecting best symbol for grid trading...")
            choice = await self.ai_strategy.select_symbol(
                tickers, all_filters, exchange=self.exchange,
            )
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

            success = await self.grid_manager.setup_grid(
                params, filters, mark_price,
                starting_equity=balance,
            )
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
            await self._close_and_report(
                symbol, reason=f"rebal_exit: {decision.reasoning}"[:120],
                tag="exit",
            )
            self._last_rebalance_action = now
            return

        if decision.action == "REBALANCE" and decision.new_params is not None:
            await self.telegram.send(
                f"\U0001f504 <b>Rebalancing grid</b> on {symbol}\n"
                f"<i>{decision.reasoning}</i>"
            )
            # Carry the original starting_equity into the new grid so
            # TP stays anchored to the true entry equity — not whatever
            # the balance happens to be mid-rebalance.
            carry_equity = self.grid_manager.grid.starting_equity
            if carry_equity <= 0:
                carry_equity = await self._balance_now()
            # setup_grid internally tears down + closes naked inventory
            success = await self.grid_manager.setup_grid(
                decision.new_params, filters, mark_price,
                starting_equity=carry_equity,
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
