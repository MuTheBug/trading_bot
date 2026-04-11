# AI-Driven Binance USDT-M Futures Trading Bot

A self-contained Python trading bot for Binance USDT-M Futures, designed to
**survive and grow a $10 starting balance**. The entry decisions are delegated
entirely to **MiniMax M2.7** (via the Anthropic-compatible API provided by the
MiniMax Token Plan). The AI decides side, entry, stop-loss, take-profit ladder
and leverage — on its own terms, with no hard-coded strategy imposed on it.

A deterministic risk layer enforces position sizing, daily loss limits,
drawdown, cooldowns and funding-rate guards on top of the AI's plan, so a bad
call by the model can't blow up the account.

> ⚠️ **Trading futures is risky.** This software is provided as-is with no
> warranty. Start in simulation mode. Never risk money you cannot afford to
> lose.

## Features

- **AI brain — MiniMax M2.7** (Anthropic-compatible API)
  - Full control over each trade: side, entry, SL, 1–3 take-profits, leverage
  - No prescribed strategy — the model picks whatever approach fits the market
  - Structural validation only (SL on losing side, TPs on winning side,
    close-percentages sum to 100). The model is not second-guessed beyond that.
  - Called exactly once per closed 15-minute candle per symbol, so the Starter
    plan's 1500-req / 5h budget is never hit
- **Two run modes**
  - `sim` — paper trading on **live** Binance market data (no API key needed, no risk)
  - `live` — real orders on Binance USDT-M Futures
- **Risk layer (not overridable by the AI)**
  - Fixed-fractional position sizing on a ~$10 account (1% risk per trade)
  - Max 2 concurrent positions, 2% daily loss circuit breaker, 10% drawdown breaker
  - Cooldown after losses, funding-rate skip/exit guards
  - Consecutive-error auto-pause after 5 failed ticks
- **Telegram** — real-time entry/exit alerts and an authorised command set
  (`/status`, `/balance`, `/stats`, `/pause`, `/resume`, `/close`, `/closeall`, `/mode`)
- **State persistence** — positions, daily PnL and trade history survive restarts
- **Hardened live exchange** — transient Binance API errors are retried with
  exponential backoff; rate limits and 5xx responses auto-heal
- **One-shot installer** — interactively asks for API keys, writes `.env`
  (chmod 600), and runs an optional connectivity check against Binance,
  Telegram and MiniMax

## Install

```bash
git clone <your-repo-url> trading_bot
cd trading_bot
./install.sh
```

The installer creates a venv, installs dependencies, copies `config.example.yaml`
to `config.yaml`, and prompts for:

- **Binance API key + secret** (Futures trading enabled, withdrawals disabled,
  IP-whitelisted)
- **Whether to use Binance Futures Testnet**
- **Telegram bot token** (from `@BotFather`) + **numeric chat id** (from `@userinfobot`)
- **MiniMax Token Plan API key** — get one at
  [platform.minimax.io/user-center/basic-information/interface-key](https://platform.minimax.io/user-center/basic-information/interface-key).
  Subscribe to a Token Plan first (Starter / Plus / Max).
- **Default run mode**

After writing `.env`, the installer offers a connectivity check that:
1. Pings Binance Futures with your keys
2. Calls Telegram's `getMe` on your bot token
3. Sends a one-token "OK" round-trip to MiniMax M2.7

Any failure is reported immediately so you don't discover typos an hour into
your first run.

## Run

```bash
./run.sh                 # uses RUN_MODE from .env (default sim)
./run.sh --mode sim      # paper trading on live data
./run.sh --mode live     # REAL orders — test in sim first
```

## Configuration

`config.yaml` controls symbols, timeframes, leverage cap, risk parameters and
the AI settings. Defaults are the **balanced profile** recommended for a micro
account:

- Symbols: `DOGEUSDT`, `1000PEPEUSDT`, `XRPUSDT`, `SUIUSDT`, `1000SHIBUSDT`
- 15-minute signal timeframe with a 1-hour higher-timeframe context fed to the AI
- 3× isolated leverage cap (AI may request up to `ai.max_leverage = 5`)
- 1% risk per trade
- Max 2 concurrent positions
- 2% daily loss circuit breaker
- `ai.min_confidence = 0.6` — trades below this confidence are skipped
- `ai.kline_history = 80` / `htf_history = 40` candles passed to the model per call

To fall back to the classical trend-momentum strategy, set `ai.enabled: false`
in `config.yaml`. The code path is still there and uses the same risk layer.

## How the AI path works

Every tick (`loop_interval_seconds = 15`), for each symbol without an open
position, the bot:

1. Pulls the latest 15m + 1h klines from Binance
2. Checks whether a new 15m candle has closed since the last AI call for that
   symbol. If not, it skips (quota-friendly throttling).
3. Enriches the data with EMA / RSI / ADX / ATR / volume SMA
4. Serialises recent OHLCV + indicators + current price + funding + leverage
   cap into a compact JSON payload
5. Sends it to MiniMax M2.7 with a short system prompt describing the output
   format and the environment (not the strategy)
6. Parses the model's JSON response. Any malformed reply, low-confidence
   signal, or structurally-impossible plan is discarded.
7. Applies the model-chosen SL/TP/leverage, runs it through the risk layer
   (sizing, daily loss, max drawdown, funding guard, cooldown)
8. If everything clears, places the market order and persists the position

Existing positions are managed by the deterministic risk manager using the
stop-loss and take-profit ladder the AI chose at entry — no further AI calls
are needed unless you open another trade.

Token-budget math for the Starter plan (1500 req / 5h ≈ 5 / minute):

- 5 symbols × 4 candles/hour = 20 AI calls / hour
- Well under the 300 / hour cap — leaves 90% headroom for re-entries and retries

## Telegram commands

All commands are ACL-restricted to the configured `TELEGRAM_CHAT_ID`.

| Command | Effect |
|---|---|
| `/status` | Open positions, unrealized PnL, SL, TP progress |
| `/balance` | Equity, daily PnL, fees, trade count, win rate |
| `/stats` | Lifetime trade count, win rate, profit factor, net PnL |
| `/pause` | Halts new entries (open positions still managed) |
| `/resume` | Re-enables entries |
| `/close <SYMBOL>` | Force-close one position at market |
| `/closeall` | Force-close every open position |
| `/mode` | Reports current run mode |

You will also receive push alerts on: entry, exit (with colour + PnL), tick
errors, auto-pause after 5 consecutive errors, and one-shot alerts when the
daily-loss or drawdown circuit breaker first trips.

## Directory layout

```
trading_bot/
├── install.sh          # interactive installer
├── run.sh              # launcher (activates venv + runs)
├── run.py              # CLI entry
├── config.example.yaml # template config (copied to config.yaml on install)
├── .env.example        # template env
├── requirements.txt
├── src/
│   ├── config.py       # pydantic settings (incl. AIConfig)
│   ├── logger.py       # loguru setup
│   ├── state.py        # JSON-backed runtime state
│   ├── indicators.py   # EMA / ADX / ATR / RSI (pandas/numpy only)
│   ├── exchange/       # live + simulator backends (same interface)
│   │   ├── base.py
│   │   ├── binance_live.py   # retry-wrapped python-binance AsyncClient
│   │   └── simulator.py       # paper trading on live Binance public data
│   ├── risk/           # position sizer + risk manager
│   ├── strategy/
│   │   ├── base.py            # Strategy ABC + Signal dataclass
│   │   ├── ai_strategy.py     # ← MiniMax M2.7 brain
│   │   └── trend_momentum.py  # fallback classical strategy
│   ├── telegram/       # notifier + command handlers
│   └── bot.py          # main async loop
└── tests/              # unit tests
```

## Testing

```bash
source .venv/bin/activate
pytest -q
```

The test suite covers: indicators, risk manager, position sizer, simulator,
and the AI strategy (parsing, structural validation, JSON extraction, TP
sorting, padding for 1/2 TP plans — all driven via a stubbed Anthropic client
so no network calls happen).

## Verification checklist before going live

1. `pytest -q` — all unit tests green
2. Export your keys and run the installer connectivity check
3. `./run.sh --mode sim` for **at least 24 h** on live market — watch logs and
   Telegram alerts; verify the AI returns sensible decisions, SL/TPs fill,
   daily stats reset at UTC midnight, state persists across a restart
4. Create a dedicated Binance sub-account, fund with the exact amount you are
   willing to lose, enable Futures only, disable withdrawals, IP-whitelist the
   server
5. `./run.sh --mode live` — watch the first few trades closely

## Caveats

- **$10 is genuinely hard.** Minimum notional (5 USDT on most alts,
  50–100 USDT on BTC/ETH) plus fees plus slippage make micro-profitability
  difficult. The bot is built as a framework that survives at $10 and
  scales cleanly as the balance grows.
- **The AI is not magic.** It will take bad trades sometimes. The risk layer
  (1% risk, 2% daily loss breaker, 10% max drawdown breaker, cooldowns) is
  what keeps the account alive through those.
- **Leverage kills.** The cap is 5× in the config; the AI is told that number
  and stays under it. Do not raise it without thinking hard.
- **Your MiniMax API key is exclusive to the Token Plan** and is only valid
  while the plan is active. If your subscription lapses, AI calls fail and
  the bot auto-pauses after 5 consecutive errors.
