"""Compact, AI-friendly activity log for strategy review.

Writes one line per event in pipe-delimited ``key=value`` format to a
dedicated file. Designed to be copy-pasted wholesale into an AI prompt
for strategy analysis. The header auto-written on first use explains
the schema so the AI can parse it without extra context, and each row
is ~15-25 tokens to keep long sessions affordable.

Usage::

    from src import trade_log
    trade_log.configure("logs/trades.log")
    trade_log.log("setup", s="BLURUSDT", p=0.02591, ...)
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Optional


_HEADER = """\
# Trading bot activity log — paste into an AI for strategy review.
# FMT: ISO_TIME | event | key=value key=value ...
# events:
#   sel     symbol selected     s=sym score=X why=reason
#   skip    setup rejected      why=reason
#   setup   grid placed         s=sym p=price l=low u=up n=levels lev=lev q=qty sp=spacing pt=profit/trip
#   fill    order filled        s=sym sd=B|S p=price q=qty lv=level nq=net_qty ae=avg_entry pnl=realized_net fe=fee
#   close   position closed     s=sym sd=B|S p=price q=qty pnl=realized why=reason
#   rebal   rebalance decision  s=sym a=HOLD|REBAL|EXIT why=reason
#   stop    safety stop tripped s=sym p=mark ae=avg_entry nq=net_qty pct=X why=pos_sl|upnl
#   tp      take-profit hit     s=sym p=mark rp=realized upnl=unreal eq=equity pct=gain_pct
#   tick    heartbeat snapshot  s=sym p=mark nq=net_qty ae=avg_entry upnl=X rp=realized rt=trips eq=equity
#   cleanup stale orders killed s=sym n=cancelled
#   daily   day summary         pnl=realized fees=paid trades=n wins=n losses=n eq=equity
# sd: B=buy S=sell. All prices in quote (USDT). pnl already net of fee.
"""


_LOCK = Lock()
_PATH: Optional[Path] = None


def configure(path: str | Path) -> None:
    """Point the logger at ``path``. Writes the schema header if empty."""
    global _PATH
    _PATH = Path(path)
    _PATH.parent.mkdir(parents=True, exist_ok=True)
    if not _PATH.exists() or _PATH.stat().st_size == 0:
        _PATH.write_text(_HEADER)


def _fmt(v: Any) -> str:
    if isinstance(v, bool):
        return "1" if v else "0"
    if isinstance(v, float):
        # %g drops trailing zeros & picks sci notation for tiny numbers.
        s = f"{v:.6g}"
        return s
    if isinstance(v, str):
        if not v:
            return '""'
        if any(c in v for c in ' |"'):
            return '"' + v.replace('"', "'") + '"'
        return v
    return str(v)


def log(event: str, **fields: Any) -> None:
    """Append one event line. Silently no-ops if not configured."""
    if _PATH is None:
        return
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    kvs = " ".join(f"{k}={_fmt(v)}" for k, v in fields.items())
    line = f"{ts} | {event} | {kvs}\n"
    with _LOCK:
        with _PATH.open("a") as f:
            f.write(line)


def tail(n: int = 200) -> str:
    """Return the last ``n`` lines — useful for Telegram /log commands."""
    if _PATH is None or not _PATH.exists():
        return ""
    with _PATH.open("rb") as f:
        try:
            f.seek(0, 2)
            size = f.tell()
            block = 4096
            data = b""
            while size > 0 and data.count(b"\n") <= n:
                step = min(block, size)
                size -= step
                f.seek(size)
                data = f.read(step) + data
        except OSError:
            return ""
    lines = data.decode(errors="replace").splitlines()[-n:]
    return "\n".join(lines)


__all__ = ["configure", "log", "tail"]
