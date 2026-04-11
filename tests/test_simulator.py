"""Simulator exchange tests — pure logic, no network calls.

We monkey-patch the two methods that hit Binance public REST so the simulator
can run entirely offline.
"""
import pytest

from src.exchange.base import SymbolFilters
from src.exchange.simulator import SimulatorExchange


class FakeSim(SimulatorExchange):
    """Simulator with network calls stubbed."""

    def __init__(self, price: float = 0.20):
        super().__init__(starting_balance=10.0, taker_fee_pct=0.04, slippage_ticks=0)
        self._price = price

    async def connect(self) -> None:
        return None  # skip aiohttp session

    async def close(self) -> None:
        return None

    async def get_mark_price(self, symbol: str) -> float:
        return self._price

    async def get_symbol_filters(self, symbol: str) -> SymbolFilters:
        return SymbolFilters(
            symbol=symbol,
            price_tick=0.00001,
            qty_step=1.0,
            min_qty=1.0,
            min_notional=5.0,
        )


@pytest.mark.asyncio
async def test_sim_long_profit():
    sim = FakeSim(price=0.20)
    await sim.connect()
    start = await sim.get_balance()

    order = await sim.market_open("DOGEUSDT", "BUY", 50.0)
    assert order.status == "FILLED"
    assert order.qty == 50.0

    # Price moves up 10%
    sim._price = 0.22
    close = await sim.market_close("DOGEUSDT", "SELL", 50.0)
    assert close.status == "FILLED"

    end = await sim.get_balance()
    # gross pnl = (0.22 - 0.20) * 50 = 1.0
    # fees: open 50*0.20*0.0004=0.004, close 50*0.22*0.0004=0.0044
    # net ~= 1.0 - 0.004 - 0.0044 = 0.9916
    assert end - start == pytest.approx(0.9916, abs=1e-6)
    positions = await sim.get_open_positions()
    assert positions == []


@pytest.mark.asyncio
async def test_sim_short_loss():
    sim = FakeSim(price=0.20)
    await sim.connect()
    await sim.market_open("DOGEUSDT", "SELL", 50.0)
    sim._price = 0.21  # price up, short loses
    await sim.market_close("DOGEUSDT", "BUY", 50.0)
    end = await sim.get_balance()
    # gross pnl = (0.20 - 0.21) * 50 = -0.5
    # fees: 0.004 + 0.0042 = 0.0082
    # balance = 10 - 0.5 - 0.0082 = 9.4918
    assert end == pytest.approx(9.4918, abs=1e-6)


@pytest.mark.asyncio
async def test_sim_partial_close():
    sim = FakeSim(price=1.0)
    await sim.connect()
    await sim.market_open("XRPUSDT", "BUY", 10.0)
    sim._price = 1.10
    await sim.market_close("XRPUSDT", "SELL", 4.0)
    positions = await sim.get_open_positions()
    assert len(positions) == 1
    assert positions[0].qty == pytest.approx(6.0)


@pytest.mark.asyncio
async def test_sim_close_without_open_raises():
    sim = FakeSim()
    await sim.connect()
    with pytest.raises(RuntimeError):
        await sim.market_close("DOGEUSDT", "SELL", 1.0)
