"""Telegram bot: outbound alerts + inbound command handlers.

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
        await self.send(f"🤖 Trading bot online in <b>{self.bot_ref.mode.upper()}</b> mode")

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
        app.add_handler(CommandHandler("pause", self._cmd_pause))
        app.add_handler(CommandHandler("resume", self._cmd_resume))
        app.add_handler(CommandHandler("close", self._cmd_close))
        app.add_handler(CommandHandler("closeall", self._cmd_closeall))
        app.add_handler(CommandHandler("mode", self._cmd_mode))

    def _is_authorized(self, update: Update) -> bool:
        uid = str(update.effective_chat.id) if update.effective_chat else ""
        return uid == self.chat_id

    async def _guard(self, update: Update) -> bool:
        if not self._is_authorized(update):
            if update.message:
                await update.message.reply_text("❌ Unauthorized.")
            return False
        return True

    async def _cmd_start(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        if not await self._guard(update):
            return
        await update.message.reply_html(
            f"👋 Welcome. Bot is running in <b>{self.bot_ref.mode.upper()}</b> mode.\n"
            f"Use /help for commands."
        )

    async def _cmd_help(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        if not await self._guard(update):
            return
        await update.message.reply_html(
            "<b>Commands</b>\n"
            "/status - positions &amp; PnL\n"
            "/balance - wallet + daily PnL\n"
            "/stats - trade statistics\n"
            "/pause - stop new entries\n"
            "/resume - re-enable entries\n"
            "/close &lt;SYMBOL&gt; - force-close one position\n"
            "/closeall - force-close everything\n"
            "/mode - show current run mode"
        )

    async def _cmd_status(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        if not await self._guard(update):
            return
        eq = await self.bot_ref.get_equity()
        lines = [f"💰 Equity: <b>{eq:.4f} USDT</b>"]
        if not self.state.state.positions:
            lines.append("No open positions.")
        else:
            for sym, p in self.state.state.positions.items():
                mark = await self.bot_ref.exchange.get_mark_price(sym)
                pnl_pct = p.unrealized_pnl_pct(mark)
                lines.append(
                    f"• <b>{sym}</b> {p.side} qty={p.remaining_qty:g}\n"
                    f"   entry={p.entry_price:.6f}  mark={mark:.6f}  "
                    f"uPnL={pnl_pct:+.2f}%\n"
                    f"   SL={p.stop_loss:.6f}  TP1={'✅' if p.tp1_hit else '⏳'} "
                    f"TP2={'✅' if p.tp2_hit else '⏳'}"
                )
        if self.state.state.paused:
            lines.append("\n⏸ <b>PAUSED</b> - no new entries")
        await update.message.reply_html("\n".join(lines))

    async def _cmd_balance(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        if not await self._guard(update):
            return
        eq = await self.bot_ref.get_equity()
        d = self.state.state.daily
        await update.message.reply_html(
            f"💰 Equity: <b>{eq:.4f} USDT</b>\n"
            f"📅 Daily PnL: <b>{d.realized_pnl:+.4f}</b>\n"
            f"💸 Fees today: {d.fees_paid:.4f}\n"
            f"📊 Trades today: {d.trades} (W:{d.wins}/L:{d.losses})\n"
            f"🎯 Win rate: {d.win_rate:.1f}%"
        )

    async def _cmd_stats(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        if not await self._guard(update):
            return
        h = self.state.state.history
        if not h:
            await update.message.reply_text("No trades yet.")
            return
        wins = [t for t in h if t.pnl > 0]
        losses = [t for t in h if t.pnl <= 0]
        gross_win = sum(t.pnl for t in wins)
        gross_loss = -sum(t.pnl for t in losses)
        pf = (gross_win / gross_loss) if gross_loss > 0 else float("inf")
        total = sum(t.pnl for t in h)
        await update.message.reply_html(
            f"📊 <b>Lifetime stats</b>\n"
            f"Trades: {len(h)}\n"
            f"Wins: {len(wins)}  Losses: {len(losses)}\n"
            f"Win rate: {len(wins)/len(h)*100:.1f}%\n"
            f"Profit factor: {pf:.2f}\n"
            f"Net PnL: <b>{total:+.4f} USDT</b>"
        )

    async def _cmd_pause(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        if not await self._guard(update):
            return
        self.state.state.paused = True
        self.state.save()
        await update.message.reply_text("⏸ Paused. No new entries will be taken.")

    async def _cmd_resume(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        if not await self._guard(update):
            return
        self.state.state.paused = False
        self.state.save()
        await update.message.reply_text("▶️ Resumed.")

    async def _cmd_close(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        if not await self._guard(update):
            return
        if not ctx.args:
            await update.message.reply_text("Usage: /close <SYMBOL>")
            return
        symbol = ctx.args[0].upper()
        try:
            await self.bot_ref.force_close(symbol, reason="MANUAL")
            await update.message.reply_text(f"✅ Closed {symbol}.")
        except Exception as e:
            await update.message.reply_text(f"❌ {e}")

    async def _cmd_closeall(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        if not await self._guard(update):
            return
        closed = 0
        for sym in list(self.state.state.positions.keys()):
            try:
                await self.bot_ref.force_close(sym, reason="MANUAL")
                closed += 1
            except Exception as e:
                logger.warning("closeall {}: {}", sym, e)
        await update.message.reply_text(f"✅ Closed {closed} position(s).")

    async def _cmd_mode(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        if not await self._guard(update):
            return
        await update.message.reply_html(
            f"Current mode: <b>{self.bot_ref.mode.upper()}</b>\n"
            f"(Restart with --mode to switch.)"
        )


__all__ = ["TelegramNotifier"]
