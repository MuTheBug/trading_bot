"""Tests for the compact AI-friendly trade log."""
from pathlib import Path

from src import trade_log


def test_header_written_on_configure(tmp_path: Path):
    p = tmp_path / "trades.log"
    trade_log.configure(p)
    content = p.read_text()
    assert content.startswith("# Trading bot activity log")
    # Schema covers every event we emit
    for ev in ("sel", "setup", "fill", "close", "rebal", "stop", "daily"):
        assert ev in content


def test_log_writes_pipe_delimited_kv(tmp_path: Path):
    p = tmp_path / "trades.log"
    trade_log.configure(p)
    trade_log.log("setup", s="BLURUSDT", p=0.02591, n=11, lev=5)
    lines = p.read_text().splitlines()
    last = lines[-1]
    # Structure: ISO_TIME | event | kvs
    parts = [x.strip() for x in last.split("|")]
    assert len(parts) == 3
    assert parts[1] == "setup"
    assert "s=BLURUSDT" in parts[2]
    assert "p=0.02591" in parts[2]
    assert "n=11" in parts[2]
    assert "lev=5" in parts[2]


def test_log_quotes_strings_with_spaces(tmp_path: Path):
    p = tmp_path / "trades.log"
    trade_log.configure(p)
    trade_log.log("rebal", s="ETHUSDT", a="EXIT", why="trend flip up")
    last = p.read_text().splitlines()[-1]
    # Space-containing value must be quoted so the format stays parseable
    assert 'why="trend flip up"' in last


def test_log_is_noop_when_not_configured(tmp_path: Path):
    # Reset to unconfigured state
    trade_log._PATH = None
    # Must not raise
    trade_log.log("fill", s="BTCUSDT", p=60000.0)


def test_float_formatting_drops_trailing_zeros(tmp_path: Path):
    p = tmp_path / "trades.log"
    trade_log.configure(p)
    trade_log.log("fill", pnl=0.00010000, fe=0.00002000)
    last = p.read_text().splitlines()[-1]
    # %.6g drops trailing zeros — saves tokens
    assert "pnl=0.0001" in last
    assert "fe=2e-05" in last or "fe=0.00002" in last


def test_tail_returns_last_n_lines(tmp_path: Path):
    p = tmp_path / "trades.log"
    trade_log.configure(p)
    for i in range(10):
        trade_log.log("fill", s="X", i=i)
    out = trade_log.tail(3)
    lines = out.splitlines()
    assert len(lines) == 3
    assert "i=9" in lines[-1]
