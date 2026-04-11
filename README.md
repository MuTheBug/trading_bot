# Binance USDT-M Futures Trading Bot

A self-contained Python trading bot for Binance USDT-M Futures, designed to
**survive and grow a $10 starting balance**. It manages everything: symbol
selection, leverage, position sizing, stops, take-profit ladder, trailing
stops, daily loss circuit breaker, funding-rate guard, and Telegram control.

> ⚠️ **Trading futures is risky.** This software is provided as-is with no
> warranty. Start in simulation mode. Never risk money you cannot afford to
> lose.

## Features

- **Two run modes**
  - `sim` — paper trading on **live** Binance market data (no API key needed, no risk)
  - `live` — real orders on Binance USDT-M Futures
- **Strategy** — Trend-momentum hybrid on 15m with 1h HTF filter:
  EMA 9/21 cross + ADX ≥ 20 + RSI zone + volume + HTF EMA50 direction
- **Exits** — ATR-based SL, three-tier TP ladder (0.75 / 2.0 / 4.0 × ATR),
  break-even move after TP1, trailing stop after TP2, 24h time stop
- **Risk** — 1 % risk per trade (fixed-fractional), max 2 concurrent positions,
  2 % daily loss limit, max-drawdown breaker, cooldown after losses,
  funding-rate skip/exit guards
- **Telegram** — real-time entry/exit alerts and an authorised command set
  (`/status`, `/balance`, `/stats`, `/pause`, `/resume`, `/close`, `/closeall`, `/mode`)
- **State persistence** — positions, daily PnL and trade history survive restarts
- **One-shot installer** — interactively asks for API keys and writes `.env`

## Install

```bash
git clone <your-repo-url> trading_bot
cd trading_bot
./install.sh
```

The installer creates a venv, installs dependencies, copies `config.example.yaml`
to `config.yaml`, and prompts for:

- Binance API key + secret (Futures trading enabled, withdrawals disabled, IP-whitelisted)
- Whether to use Binance Futures Testnet
- Telegram bot token (from `@BotFather`) + numeric chat id (from `@userinfobot`)
- Default run mode

## Run

```bash
./run.sh                 # uses RUN_MODE from .env (default sim)
./run.sh --mode sim      # paper trading on live data
./run.sh --mode live     # REAL orders — test in sim first
```

## Configuration

Edit `config.yaml` to tune symbols, leverage, risk parameters and strategy
values. Defaults are the **balanced profile** recommended for a micro account:

- 3× isolated leverage
- 1 % risk per trade
- Max 2 concurrent positions
- 2 % daily loss circuit breaker
- Symbols: `DOGEUSDT`, `1000PEPEUSDT`, `XRPUSDT`, `SUIUSDT`, `1000SHIBUSDT`

## Directory layout

```
trading_bot/
├── install.sh          # interactive installer
├── run.sh              # launcher (activates venv + runs)
├── run.py              # CLI entry
├── config.example.yaml # template config
├── .env.example        # template env
├── src/
│   ├── config.py       # pydantic settings
│   ├── logger.py       # loguru setup
│   ├── state.py        # JSON-backed runtime state
│   ├── indicators.py   # EMA / ADX / ATR / RSI (pandas/numpy only)
│   ├── exchange/       # live + simulator backends (same interface)
│   ├── risk/           # position sizer + risk manager
│   ├── strategy/       # trend_momentum hybrid
│   ├── telegram/       # notifier + command handlers
│   └── bot.py          # main async loop
└── tests/              # unit tests
```

## Testing

```bash
source .venv/bin/activate
pytest -q
```

## Verification checklist before going live

1. `pytest -q` — all unit tests green
2. `./run.sh --mode sim` for **at least 24 h** on live market — watch logs and
   Telegram alerts; verify signals trigger, SL/TPs fill, daily stats reset at
   UTC midnight, state persists across a restart
3. Create a dedicated Binance sub-account, fund with the exact amount you are
   willing to lose, enable Futures only, disable withdrawals, IP-whitelist the
   server
4. `./run.sh --mode live` — watch the first few trades closely

## Caveats

- **$10 is genuinely hard.** Minimum notional (5 USDT on most alts,
  50–100 USDT on BTC/ETH) plus fees plus slippage make micro-profitability
  difficult. The bot is built as a framework that survives at $10 and
  scales cleanly as the balance grows.
- **No strategy is guaranteed profitable.** Crypto regimes shift. Sim-test
  first, always.
- **Leverage kills.** Even 3× can blow up an account on a news spike. The
  daily circuit breaker exists for a reason — do not disable it.
