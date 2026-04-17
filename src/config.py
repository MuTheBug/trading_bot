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
    max_tokens: int = 2048
    thinking: bool = False  # set true to use extended thinking mode
    max_leverage: int = 20  # hard cap on leverage the AI may request
    kline_history: int = 50      # candles included in the prompt
    htf_history: int = 30        # HTF candles included in the prompt
    request_timeout_s: float = 90.0
    retries: int = 2


class GridConfig(BaseModel):
    """Grid-specific settings. The AI sets the actual grid parameters
    (symbol, upper/lower bounds, levels, leverage) at runtime; these are
    constraints and defaults the AI must stay within."""

    max_grids: int = 15               # max grid levels AI may create
    min_grids: int = 3                # minimum grid levels
    rebalance_check_minutes: int = 60 # how often to ask AI to re-evaluate
    out_of_range_pct: float = 2.0     # % outside grid to trigger AI re-eval
    max_unrealized_loss_pct: float = 3.0  # force-close grid if uPnL loss exceeds this (of equity)
    max_capital_pct: float = 40.0     # max % of balance the grid may commit as margin
    max_leverage: int = 5             # hard cap on grid leverage (safer than AI cap)


class BotConfig(BaseModel):
    timeframe: str = "15m"
    htf_timeframe: str = "1h"
    margin_type: Literal["ISOLATED", "CROSSED"] = "ISOLATED"
    risk: RiskConfig = Field(default_factory=RiskConfig)
    strategy: StrategyConfig = Field(default_factory=StrategyConfig)
    exits: ExitsConfig = Field(default_factory=ExitsConfig)
    simulator: SimulatorConfig = Field(default_factory=SimulatorConfig)
    telegram: TelegramConfig = Field(default_factory=TelegramConfig)
    ai: AIConfig = Field(default_factory=AIConfig)
    grid: GridConfig = Field(default_factory=GridConfig)
    loop_interval_seconds: int = 10
    kline_history: int = 200
    log_level: str = "INFO"
    log_file: str = "logs/bot.log"
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
