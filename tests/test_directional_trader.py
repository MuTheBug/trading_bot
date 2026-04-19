"""Tests for DirectionalTrader's MTF guard + prescreen."""
import numpy as np
import pandas as pd

from src.directional_trader import DirectionalTrader, _prescreen
from src.exchange.base import SymbolFilters, TickerInfo
from src.strategy.ai_directional import AIDecision, _CandidateCtx


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
