"""Telegram bot: outbound alerts + inbound command handlers for grid trading.

Uses python-telegram-bot v21 (async). The bot runs concurrently with the
trading loop via its own asyncio task. All commands are ACL-restricted to
the configured TELEGRAM_CHAT_ID.
"""
from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Optional

from loguru import logger

try:
    from telegram import Update
    from telegram.constants import ParseMode
    from telegram.ext import (
        Application,
        CommandHandler,
        ContextTypes,
    )
except ImportError:  # pragma: no cover
    Update = None  # type: ignore
    Application = None  # type: ignore

from ..state import StateStore

if TYPE_CHECKING:
    from ..bot import TradingBot


class TelegramNotifier:
    def __init__(
        self,
        token: str,
        chat_id: str,
        state: StateStore,
        bot_ref: "TradingBot",
        enabled: bool = True,
    ) -> None:
        self.token = token
        self.chat_id = str(chat_id) if chat_id else ""
        self.state = state
        self.bot_ref = bot_ref
        self.enabled = enabled and bool(token) and bool(chat_id) and Application is not None
        self._app: Optional[Application] = None

    async def start(self) -> None:
        if not self.enabled:
            logger.info("Telegram disabled (missing token/chat_id or library).")
            return
        self._app = Application.builder().token(self.token).build()
        self._register_handlers()
        await self._app.initialize()
        await self._app.start()
        await self._app.updater.start_polling(drop_pending_updates=True)
        logger.info("Telegram bot started (chat_id={})", self.chat_id)
        await self.send(
            f"\U0001f916 Grid trading bot online in <b>{self.bot_ref.mode.upper()}</b> mode"
        )

    async def stop(self) -> None:
        if self._app is None:
            return
        try:
            await self._app.updater.stop()
            await self._app.stop()
            await self._app.shutdown()
        except Exception as e:  # pragma: no cover
            logger.warning("Telegram shutdown error: {}", e)
        self._app = None

    async def send(self, text: str) -> None:
        if not self.enabled or self._app is None:
            return
        try:
            await self._app.bot.send_message(
                chat_id=self.chat_id, text=text, parse_mode=ParseMode.HTML
            )
        except Exception as e:
            logger.warning("Telegram send failed: {}", e)

    # ---------- handlers ----------

    def _register_handlers(self) -> None:
        app = self._app
        assert app is not None
        app.add_handler(CommandHandler("start", self._cmd_start))
        app.add_handler(CommandHandler("help", self._cmd_help))
        app.add_handler(CommandHandler("status", self._cmd_status))
        app.add_handler(CommandHandler("balance", self._cmd_balance))
        app.add_handler(CommandHandler("stats", self._cmd_stats))
        app.add_handler(CommandHandler("grid", self._cmd_grid))
        app.add_handler(CommandHandler("summary", self._cmd_summary))
        app.add_handler(CommandHandler("pause", self._cmd_pause))
        app.add_handler(CommandHandler("resume", self._cmd_resume))
        app.add_handler(CommandHandler("teardown", self._cmd_teardown))
        app.add_handler(CommandHandler("mode", self._cmd_mode))

    def _is_authorized(self, update: Update) -> bool:
        uid = str(update.effective_chat.id) if update.effective_chat else ""
        return uid == self.chat_id

    async def _guard(self, update: Update) -> bool:
        if not self._is_authorized(update):
            if update.message:
                await update.message.reply_text("\u274c Unauthorized.")
            return False
        return True

    async def _cmd_start(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        if not await self._guard(update):
            return
        await update.message.reply_html(
            f"\U0001f44b Welcome. Grid bot running in <b>{self.bot_ref.mode.upper()}</b> mode.\n"
            f"Use /help for commands."
        )

    async def _cmd_help(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        if not await self._guard(update):
            return
        await update.message.reply_html(
            "<b>Grid Bot Commands</b>\n"
            "/status - grid status &amp; equity\n"
            "/balance - wallet + daily PnL\n"
            "/stats - trade statistics\n"
            "/grid - detailed grid info\n"
            "/summary - daily summary\n"
            "/pause - stop grid operations\n"
            "/resume - re-enable grid\n"
            "/teardown - force tear down the active grid\n"
            "/mode - show current run mode"
        )

    async def _cmd_status(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        if not await self._guard(update):
            return
        eq = await self.bot_ref.get_equity()
        gs = self.state.state.grid
        lines = [f"\U0001f4b0 Equity: <b>{eq:.4f} USDT</b>"]

        if gs.active:
            mark = await self.bot_ref.exchange.get_mark_price(gs.symbol)
            upnl = self.bot_ref.grid_manager.unrealized_pnl(mark)
            buy_orders = sum(1 for lv in gs.levels if lv.buy_order_id)
            sell_orders = sum(1 for lv in gs.levels if lv.sell_order_id)
            lines.extend([
                f"\n\U0001f4ca <b>Grid: {gs.symbol}</b>",
                f"Range: {gs.lower_price:.8f} - {gs.upper_price:.8f}",
                f"Mark: {mark:.8f}",
                f"Leverage: {gs.leverage}x",
                f"Orders: {buy_orders} buys + {sell_orders} sells",
                f"Round trips: {gs.round_trips}",
                f"Profit: {gs.total_profit:+.6f} USDT",
                f"uPnL: {upnl:+.6f} USDT",
            ])
        else:
            lines.append("\nNo active grid.")

        if self.state.state.paused:
            lines.append("\n\u23f8 <b>PAUSED</b>")

        await update.message.reply_html("\n".join(lines))

    async def _cmd_balance(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        if not await self._guard(update):
            return
        eq = await self.bot_ref.get_equity()
        d = self.state.state.daily
        await update.message.reply_html(
            f"\U0001f4b0 Equity: <b>{eq:.4f} USDT</b>\n"
            f"\U0001f4c5 Daily PnL: <b>{d.realized_pnl:+.4f}</b>\n"
            f"\U0001f4b8 Fees today: {d.fees_paid:.4f}\n"
            f"\U0001f4ca Trades today: {d.trades} (W:{d.wins}/L:{d.losses})\n"
            f"\U0001f3af Win rate: {d.win_rate:.1f}%"
        )

    async def _cmd_stats(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        if not await self._guard(update):
            return
        gs = self.state.state.grid
        if gs.round_trips == 0:
            await update.message.reply_text("No grid trades yet.")
            return
        net_profit = gs.total_profit - gs.total_fees
        await update.message.reply_html(
            f"\U0001f4ca <b>Grid Statistics</b>\n"
            f"Symbol: {gs.symbol}\n"
            f"Round trips: {gs.round_trips}\n"
            f"Gross profit: {gs.total_profit:+.6f} USDT\n"
            f"Total fees: {gs.total_fees:.6f} USDT\n"
            f"Net profit: <b>{net_profit:+.6f} USDT</b>\n"
            f"Avg profit/trip: {gs.total_profit / gs.round_trips:.6f} USDT"
        )

    async def _cmd_grid(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        if not await self._guard(update):
            return
        gs = self.state.state.grid
        if not gs.active:
            await update.message.reply_text("No active grid.")
            return

        mark = await self.bot_ref.exchange.get_mark_price(gs.symbol)
        summary = self.bot_ref.grid_manager.grid_summary(mark)

        lines = [
            f"\U0001f4ca <b>Grid Detail: {gs.symbol}</b>",
            f"Active: {'Yes' if gs.active else 'No'}",
            f"Range: {gs.lower_price:.8f} - {gs.upper_price:.8f}",
            f"Levels: {gs.num_grids}",
            f"Spacing: {summary['spacing']:.8f}",
            f"Leverage: {gs.leverage}x",
            f"Qty/level: {gs.qty_per_grid:.8f}",
            f"Mark price: {mark:.8f}",
            f"Net qty: {gs.net_qty:+.8f}",
            f"uPnL: {summary['unrealized_pnl']:+.6f}",
            f"Profit: {gs.total_profit:+.6f}",
            f"Fees: {gs.total_fees:.6f}",
            f"Round trips: {gs.round_trips}",
            f"Buy orders: {summary['buy_orders']}",
            f"Sell orders: {summary['sell_orders']}",
            f"Setup at: {gs.setup_at}",
        ]
        if gs.ai_reasoning:
            lines.append(f"\nAI: <i>{gs.ai_reasoning}</i>")

        await update.message.reply_html("\n".join(lines))

    async def _cmd_summary(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        if not await self._guard(update):
            return
        text = self.bot_ref.build_daily_summary()
        await update.message.reply_html(text)

    async def _cmd_pause(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        if not await self._guard(update):
            return
        self.state.state.paused = True
        self.state.save()
        await update.message.reply_text("\u23f8 Paused. Grid will stop checking fills.")

    async def _cmd_resume(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        if not await self._guard(update):
            return
        self.state.state.paused = False
        self.state.save()
        await update.message.reply_text("\u25b6\ufe0f Resumed.")

    async def _cmd_teardown(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        if not await self._guard(update):
            return
        try:
            await self.bot_ref.force_teardown(reason="MANUAL via /teardown")
            await update.message.reply_text(
                "\u2705 Grid torn down. Bot will scan for a new symbol on next tick."
            )
        except Exception as e:
            await update.message.reply_text(f"\u274c {e}")

    async def _cmd_mode(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        if not await self._guard(update):
            return
        await update.message.reply_html(
            f"Current mode: <b>{self.bot_ref.mode.upper()}</b>\n"
            f"(Restart with --mode to switch.)"
        )


__all__ = ["TelegramNotifier"]
