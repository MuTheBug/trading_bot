"""Centralized loguru configuration."""
from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger


def setup_logger(log_file: str = "logs/bot.log", level: str = "INFO") -> None:
    """Configure loguru with stdout + rotating file sinks."""
    logger.remove()

    logger.add(
        sys.stdout,
        level=level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}:{line}</cyan> - <level>{message}</level>"
        ),
        colorize=True,
    )

    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger.add(
        str(log_path),
        level=level,
        rotation="10 MB",
        retention="14 days",
        compression="zip",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{line} - {message}",
        enqueue=True,
    )


__all__ = ["logger", "setup_logger"]
