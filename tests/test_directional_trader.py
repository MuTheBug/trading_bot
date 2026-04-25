"""Tests for DirectionalTrader's MTF guard + prescreen."""
import numpy as np
import pandas as pd

from src.directional_trader import DirectionalTrader, _prescreen
from src.exchange.base import SymbolFilters, TickerInfo
from src.strategy.sr_strategy import SRDecision as AIDecision, CandidateCtx as _CandidateCtx


def _ohlcv(closes):
    n = len(closes)
    idx = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    return pd.DataFrame({
        "open": closes, "high": closes * 1.01, "low": closes * 0.99,
        "close": closes, "volume": np.full(n, 1000.0),
    }, index=idx)


def _ticker(symbol="X", price=100.0, chg=0.0, vol=1e8):
    return TickerInfo(symbol, price=price, volume_24h=vol, change_pct_24h=chg,
                      high_24h=price * 1.1, low_24h=price * 0.9)


def _filters(symbol="X"):
    return SymbolFilters(symbol, price_tick=0.01, qty_step=0.001,
                         min_qty=0.001, min_notional=5.0)


class _Trader(DirectionalTrader):
    """Construct without calling parent __init__ — we only need _mtf_veto."""
    def __init__(self):
        pass


def test_prescreen_penalizes_extreme_moves():
    """A -25% symbol should rank below a -5% symbol at equal volume."""
    tickers = [
        _ticker("ADUMPUSDT", price=1.0, chg=-25.0, vol=1e9),
        _ticker("BQUIETUSDT", price=1.0, chg=-5.0, vol=1e9),
    ]
    # Give both the same high-low range so only change_pct differs.
    for t in tickers:
        t.high_24h = 1.10
        t.low_24h = 0.90
    fils = {t.symbol: _filters(t.symbol) for t in tickers}
    ranked = _prescreen(tickers, fils, min_volume_usd=1e7, top_n=5)
    # The less-extreme mover should rank ahead.
    assert ranked[0].symbol == "BQUIETUSDT"
    assert ranked[1].symbol == "ADUMPUSDT"


def test_prescreen_top_n_zero_returns_all():
    """top_n <= 0 disables the cap — every liquid symbol passes through."""
    tickers = [_ticker(f"S{i}USDT", price=1.0, chg=1.0 + i * 0.1, vol=1e9)
               for i in range(25)]
    fils = {t.symbol: _filters(t.symbol) for t in tickers}
    ranked_all = _prescreen(tickers, fils, min_volume_usd=1e7, top_n=0)
    ranked_capped = _prescreen(tickers, fils, min_volume_usd=1e7, top_n=5)
    assert len(ranked_all) == 25
    assert len(ranked_capped) == 5


def test_veto_falling_knife_long():
    """LONG a -19% day with no reversal flag -> vetoed."""
    ctx = _CandidateCtx(
        symbol="X", ticker=_ticker(chg=-19.0), filters=_filters(),
        dfs={"1d": _ohlcv(np.linspace(130, 100, 80))},  # clearly down
    )
    dec = AIDecision(action="OPEN_LONG", symbol="X", entry=100.0,
                     stop_loss=98.0, confidence=0.5, reasoning="trend up")
    veto, why = _Trader()._mtf_veto(ctx, dec)
    assert veto
    assert "falling knife" in why.lower()


def test_veto_allows_explicit_reversal_call():
    """Same setup but reasoning mentions 'reversal' -> pass through."""
    ctx = _CandidateCtx(
        symbol="X", ticker=_ticker(chg=-19.0), filters=_filters(),
        dfs={"1d": _ohlcv(np.linspace(130, 100, 80))},
    )
    dec = AIDecision(
        action="OPEN_LONG", symbol="X", entry=100.0, stop_loss=98.0,
        confidence=0.5, reasoning="capitulation wick + oversold reversal",
    )
    veto, _ = _Trader()._mtf_veto(ctx, dec)
    assert not veto


def test_veto_chasing_pump_short():
    """SHORT into +12% day, no reversal flag -> vetoed."""
    ctx = _CandidateCtx(
        symbol="X", ticker=_ticker(chg=12.0), filters=_filters(),
        dfs={"1d": _ohlcv(np.linspace(80, 100, 80))},
    )
    dec = AIDecision(action="OPEN_SHORT", symbol="X", entry=100.0,
                     stop_loss=102.0, confidence=0.5, reasoning="top")
    veto, why = _Trader()._mtf_veto(ctx, dec)
    assert veto
    assert "pump" in why.lower()


def test_veto_htf_mtf_oppose():
    """HTF + MTF both clearly down, LONG with moderate conf -> vetoed."""
    dn = np.linspace(130, 100, 80)
    ctx = _CandidateCtx(
        symbol="X", ticker=_ticker(chg=-5.0), filters=_filters(),
        dfs={"1d": _ohlcv(dn), "4h": _ohlcv(dn)},
    )
    dec = AIDecision(action="OPEN_LONG", symbol="X", entry=100.0,
                     stop_loss=98.0, confidence=0.6, reasoning="15m bullish")
    veto, why = _Trader()._mtf_veto(ctx, dec)
    assert veto
    assert "oppose" in why.lower()


def test_veto_passes_aligned_setup():
    """HTF + MTF up, LONG -> no veto."""
    up = np.linspace(100, 130, 80)
    ctx = _CandidateCtx(
        symbol="X", ticker=_ticker(chg=3.0), filters=_filters(),
        dfs={"1d": _ohlcv(up), "4h": _ohlcv(up)},
    )
    dec = AIDecision(action="OPEN_LONG", symbol="X", entry=130.0,
                     stop_loss=125.0, confidence=0.6, reasoning="trend continuation")
    veto, _ = _Trader()._mtf_veto(ctx, dec)
    assert not veto


# --- pullback veto ----------------------------------------------------------


def _pullback_long_closes():
    """80 bars: long uptrend, then a small pullback in the last ~4 bars."""
    trend = np.linspace(100, 120, 76)
    pull = np.array([120.0, 118.5, 117.0, 116.2])
    return np.concatenate([trend, pull])


def _pullback_short_closes():
    """80 bars: long downtrend, then a small bounce in the last ~4 bars."""
    trend = np.linspace(120, 100, 76)
    bounce = np.array([100.0, 101.5, 103.0, 103.8])
    return np.concatenate([trend, bounce])


def test_pullback_allows_long_at_pullback_low():
    """LONG where LTF just pulled back ~4 bars -> allowed."""
    closes = _pullback_long_closes()
    ctx = _CandidateCtx(
        symbol="X", ticker=_ticker(price=116.2, chg=3.0), filters=_filters(),
        dfs={"1d": _ohlcv(np.linspace(100, 120, 80)),
             "15m": _ohlcv(closes)},
    )
    dec = AIDecision(action="OPEN_LONG", symbol="X", entry=116.2,
                     stop_loss=114.0, confidence=0.6,
                     reasoning="HTF up, LTF pulled back to EMA20")
    veto, _ = _Trader()._pullback_veto(ctx, dec)
    assert not veto


def test_pullback_blocks_long_chasing_breakout():
    """LONG at the very top of a straight-line rally -> vetoed (chasing)."""
    closes = np.linspace(100, 130, 80)
    ctx = _CandidateCtx(
        symbol="X", ticker=_ticker(price=130.0, chg=3.0), filters=_filters(),
        dfs={"1d": _ohlcv(closes), "15m": _ohlcv(closes)},
    )
    dec = AIDecision(action="OPEN_LONG", symbol="X", entry=130.0,
                     stop_loss=125.0, confidence=0.6,
                     reasoning="trend continuation")
    veto, why = _Trader()._pullback_veto(ctx, dec)
    assert veto
    assert "pullback" in why.lower() or "chas" in why.lower()


def test_pullback_allows_short_at_bounce_high():
    """SHORT where LTF just bounced ~4 bars -> allowed."""
    closes = _pullback_short_closes()
    ctx = _CandidateCtx(
        symbol="X", ticker=_ticker(price=103.8, chg=-3.0), filters=_filters(),
        dfs={"1d": _ohlcv(np.linspace(120, 100, 80)),
             "15m": _ohlcv(closes)},
    )
    dec = AIDecision(action="OPEN_SHORT", symbol="X", entry=103.8,
                     stop_loss=106.0, confidence=0.6,
                     reasoning="HTF down, LTF bounced to EMA20")
    veto, _ = _Trader()._pullback_veto(ctx, dec)
    assert not veto


def test_pullback_blocks_short_at_bottom_of_dump():
    """SHORT at the lowest print of a straight-line dump -> vetoed."""
    closes = np.linspace(120, 100, 80)
    ctx = _CandidateCtx(
        symbol="X", ticker=_ticker(price=100.0, chg=-5.0), filters=_filters(),
        dfs={"1d": _ohlcv(closes), "15m": _ohlcv(closes)},
    )
    dec = AIDecision(action="OPEN_SHORT", symbol="X", entry=100.0,
                     stop_loss=103.0, confidence=0.6,
                     reasoning="trend continuation")
    veto, why = _Trader()._pullback_veto(ctx, dec)
    assert veto
    assert "pullback" in why.lower() or "chas" in why.lower()


def test_pullback_exempts_reversal_call():
    """Reversal-flagged entries bypass the pullback gate."""
    closes = np.linspace(120, 100, 80)
    ctx = _CandidateCtx(
        symbol="X", ticker=_ticker(price=100.0, chg=-15.0), filters=_filters(),
        dfs={"1d": _ohlcv(closes), "15m": _ohlcv(closes)},
    )
    dec = AIDecision(action="OPEN_LONG", symbol="X", entry=100.0,
                     stop_loss=98.0, confidence=0.7,
                     reasoning="capitulation wick + oversold reversal")
    veto, _ = _Trader()._pullback_veto(ctx, dec)
    assert not veto
