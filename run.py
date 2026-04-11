"""CLI entry point: `python run.py --mode sim|live`."""
from __future__ import annotations

import argparse
import asyncio
import sys

from src.bot import run_bot
from src.config import load_config, load_secrets
from src.logger import setup_logger


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="trading_bot",
        description="Binance USDT-M Futures trading bot (sim or live).",
    )
    p.add_argument(
        "--mode",
        choices=["sim", "live"],
        default=None,
        help="Run mode. Overrides RUN_MODE in .env. Default: value from .env (fallback sim).",
    )
    p.add_argument(
        "--config",
        default="config.yaml",
        help="Path to config.yaml (default: ./config.yaml)",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    try:
        config = load_config(args.config)
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    setup_logger(log_file=config.log_file, level=config.log_level)

    secrets = load_secrets()
    mode = args.mode or secrets.run_mode

    if mode == "live":
        print("⚠️  LIVE MODE — real money will be at risk.")
        print("    Make sure you tested in --mode sim first.")
        print()

    try:
        asyncio.run(run_bot(mode=mode, config=config, secrets=secrets))
    except KeyboardInterrupt:
        print("\nInterrupted.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
