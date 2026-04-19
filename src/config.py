"""Configuration loader: merges .env (secrets) with config.yaml (strategy params)."""
from __future__ import annotations

from pathlib import Path
from typing import List, Literal

import yaml
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# --- Secrets from .env ---

class Secrets(BaseSettings):
    """Loaded from environment / .env file."""

    binance_api_key: str = Field(default="", alias="BINANCE_API_KEY")
    binance_api_secret: str = Field(default="", alias="BINANCE_API_SECRET")
    binance_testnet: bool = Field(default=False, alias="BINANCE_TESTNET")
    telegram_bot_token: str = Field(default="", alias="TELEGRAM_BOT_TOKEN")
    telegram_chat_id: str = Field(default="", alias="TELEGRAM_CHAT_ID")
    run_mode: Literal["sim", "live"] = Field(default="sim", alias="RUN_MODE")

    # MiniMax (Anthropic-compatible) API — the AI brain driving trade decisions
    ai_api_key: str = Field(default="", alias="ANTHROPIC_API_KEY")
    ai_base_url: str = Field(
        default="https://api.minimax.io/anthropic", alias="ANTHROPIC_BASE_URL"
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )


# --- Strategy / risk config from config.yaml ---

class RiskConfig(BaseModel):
    risk_per_trade_pct: float = 1.0
    max_concurrent_positions: int = 2
    daily_loss_limit_pct: float = 2.0
    max_drawdown_pct: float = 10.0
    funding_rate_skip_pct: float = 15.0
    funding_rate_exit_pct: float = 20.0
    time_stop_hours: int = 24
    cooldown_after_loss_minutes: int = 15


class StrategyConfig(BaseModel):
    name: str = "trend_momentum"
    ema_fast: int = 9
    ema_slow: int = 21
    ema_htf: int = 50
    adx_period: int = 14
    adx_threshold: float = 20.0
    rsi_period: int = 14
    rsi_long_min: float = 40.0
    rsi_long_max: float = 70.0
    rsi_short_min: float = 30.0
    rsi_short_max: float = 60.0
    atr_period: int = 14
    volume_sma_period: int = 20


class ExitsConfig(BaseModel):
    sl_atr_mult: float = 1.5
    tp1_atr_mult: float = 0.75
    tp1_close_pct: float = 40.0
    tp2_atr_mult: float = 2.0
    tp2_close_pct: float = 30.0
    tp3_atr_mult: float = 4.0
    trail_atr_mult: float = 1.5
    move_sl_to_be_after_tp1: bool = True


class SimulatorConfig(BaseModel):
    starting_balance: float = 10.0
    taker_fee_pct: float = 0.04
    maker_fee_pct: float = 0.02
    slippage_ticks: int = 1


class TelegramConfig(BaseModel):
    enabled: bool = True
    alerts_on_entry: bool = True
    alerts_on_exit: bool = True
    alerts_on_error: bool = True
    daily_summary_utc_hour: int = 0


class AIConfig(BaseModel):
    """AI-driven strategy configuration (MiniMax M2.7 via Anthropic-compatible API)."""

    enabled: bool = True
    model: str = "MiniMax-M2.7"
    max_tokens: int = 4096  # MiniMax writes a lot of reasoning text before JSON
    thinking: bool = False  # set true to use extended thinking mode
    max_leverage: int = 20  # hard cap on leverage the AI may request
    kline_history: int = 50      # candles included in the prompt
    htf_history: int = 30        # HTF candles included in the prompt
    request_timeout_s: float = 90.0
    retries: int = 2


class GridConfig(BaseModel):
    """Grid-specific settings. The AI sets the actual grid parameters
    (symbol, upper/lower bounds, levels, leverage) at runtime; these are
    constraints and defaults the AI must stay within.

    Defaults are tuned from 20 real $10 trades: tight stops + 15x lev
    + sub-minute churn produced consistent -$0.01..-$0.20 losses that
    dwarfed the few winners. These wider parameters favour letting
    winners run and cutting losers only on genuine break-down, not on
    spread noise.
    """

    max_grids: int = 15               # max grid levels AI may create
    min_grids: int = 3                # minimum grid levels
    rebalance_check_minutes: int = 15 # how often to ask AI to re-evaluate
    out_of_range_pct: float = 1.5     # % outside grid to trigger AI re-eval
    max_unrealized_loss_pct: float = 5.0  # force-close grid if uPnL loss exceeds this (of equity)
    max_capital_pct: float = 50.0     # max % of balance the grid may commit as margin (lower = more headroom for loaded grid to breathe)
    max_leverage: int = 5             # hard cap on grid leverage (best winner was 5x)
    # Per-tick position safety
    position_stop_loss_pct: float = 4.0   # close grid if price moves this far adverse from avg_entry
    drift_exit_pct: float = 3.0           # exit if price drifts this far from grid center (trend emerging)
    take_profit_pct: float = 2.5          # lock in when actual gain (equity delta) >= this % of start equity
    trailing_tp_arm_pct: float = 1.5      # arm trailing TP once gain reaches this % of start equity
    trailing_tp_giveback_pct: float = 0.5 # from peak gain, exit if we give back this much of start equity
    take_profit_streak: int = 2           # require this many consecutive ticks over TP to actually fire
    min_hold_minutes: int = 5             # don't TP/stop before grid has had time to breathe
    symbol_cooldown_minutes: int = 120    # after EXIT/stop, don't re-pick the same symbol for N min
    post_exit_cooldown_minutes: int = 3   # after any close, pause bot entirely before starting new grid
    min_volume_usd: float = 50_000_000    # reject illiquid symbols from scan
    heartbeat_ticks: int = 30             # trade_log heartbeat every N ticks (0 = off)


class DirectionalConfig(BaseModel):
    """Regime-adaptive directional trading (long/short).

    Enable by setting ``trading_mode: directional`` at the top level of
    config.yaml. The bot scans for the highest-confidence market regime on
    each tick and opens a single position with adaptive leverage sized so
    the SL risks ``risk_per_trade_pct`` of current equity.
    """

    # Sizing / leverage
    risk_per_trade_pct: float = 1.0        # % of equity risked per trade at SL
    base_leverage: int = 3                 # preferred leverage when conditions neutral
    min_leverage: int = 1
    max_leverage: int = 20                 # hard cap
    max_margin_pct: float = 85.0           # margin must not exceed this % of equity

    # Scanning
    min_volume_usd: float = 50_000_000
    scan_top_n: int = 10                   # candidates sent to the AI per scan
    scan_interval_seconds: int = 60        # min gap between scans when flat
    min_confidence: float = 0.45           # reject if the AI's self-confidence is below this
    # Multi-timeframe analysis: top-down ordered list of timeframes the
    # AI receives per candidate. Must be ordered HIGHEST -> LOWEST.
    # Supported: 1d, 4h, 1h, 15m, 5m, 3m, 1m.
    mtf_timeframes: List[str] = Field(
        default_factory=lambda: ["1d", "4h", "1h", "15m"]
    )

    # Regime detection thresholds
    adx_strong: float = 25.0
    adx_weak: float = 18.0

    # Position management
    trail_atr_mult: float = 1.5
    trail_arm_atr: float = 1.0             # arm trailing after +1 ATR profit
    trail_tighten_atr: float = 2.0         # once profit >= N ATR, use tighter trail
    trail_tighten_mult: float = 0.75       # trailing mult after tightening
    breakeven_profit_pct: float = 0.4      # move SL to BE once uPnL% >= this
    breakeven_buffer_atr: float = 0.1
    breakeven_after_tp1: bool = True
    giveback_arm_pct: float = 1.0          # arm giveback protection at this peak uPnL%
    giveback_exit_pct: float = 0.6         # exit if we give back this much from peak
    time_stop_hours: float = 24.0
    max_loss_pct: float = 6.0              # hard cap % of open-equity


TradingMode = Literal["grid", "directional"]


class BotConfig(BaseModel):
    timeframe: str = "15m"
    htf_timeframe: str = "1h"
    margin_type: Literal["ISOLATED", "CROSSED"] = "ISOLATED"
    trading_mode: TradingMode = "directional"
    risk: RiskConfig = Field(default_factory=RiskConfig)
    strategy: StrategyConfig = Field(default_factory=StrategyConfig)
    exits: ExitsConfig = Field(default_factory=ExitsConfig)
    simulator: SimulatorConfig = Field(default_factory=SimulatorConfig)
    telegram: TelegramConfig = Field(default_factory=TelegramConfig)
    ai: AIConfig = Field(default_factory=AIConfig)
    grid: GridConfig = Field(default_factory=GridConfig)
    directional: DirectionalConfig = Field(default_factory=DirectionalConfig)
    loop_interval_seconds: int = 10
    kline_history: int = 200
    log_level: str = "INFO"
    log_file: str = "logs/bot.log"
    trade_log_file: str = "logs/trades.log"
    state_file: str = "state/bot_state.json"


# --- Loader ---

def load_config(config_path: str | Path = "config.yaml") -> BotConfig:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Config file not found: {path}. Run ./install.sh or copy config.example.yaml."
        )
    with path.open("r") as f:
        raw = yaml.safe_load(f) or {}
    return BotConfig(**raw)


def load_secrets() -> Secrets:
    return Secrets()  # type: ignore[call-arg]
