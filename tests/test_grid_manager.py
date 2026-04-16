"""Grid manager tests — grid setup, fill detection, counter orders, teardown.

Uses a stubbed SimulatorExchange with no network calls.
"""
import pytest

from src.exchange.base import SymbolFilters
from src.exchange.simulator import SimulatorExchange
from src.grid.manager import (
    GridManager,
    GridSetupParams,
    compute_grid_prices,
    round_price,
    round_qty,
)
from src.state import StateStore


# ---- helper functions ----

def test_compute_grid_prices_basic():
    prices = compute_grid_prices(100.0, 110.0, 5)
    assert len(prices) == 6  # num_grids + 1
    assert prices[0] == pytest.approx(100.0)
    assert prices[-1] == pytest.approx(110.0)
    assert prices[1] - prices[0] == pytest.approx(2.0)


def test_compute_grid_prices_single_level():
    prices = compute_grid_prices(100.0, 110.0, 1)
    assert len(prices) == 2
    assert prices == [100.0, 110.0]


def test_compute_grid_prices_invalid():
    assert compute_grid_prices(110.0, 100.0, 5) == []  # upper < lower
    assert compute_grid_prices(100.0, 110.0, 0) == []  # zero grids


def test_round_price():
    assert round_price(0.12345, 0.001) == pytest.approx(0.123)
    assert round_price(0.12350, 0.001) == pytest.approx(0.124)
    assert round_price(100.0, 0.01) == pytest.approx(100.0)
    assert round_price(5.0, 0) == 5.0  # zero tick passthrough


def test_round_qty():
    assert round_qty(10.7, 1.0) == pytest.approx(10.0)
    assert round_qty(10.0, 1.0) == pytest.approx(10.0)
    assert round_qty(0.5, 1.0) == pytest.approx(0.0)
    assert round_qty(5.0, 0) == 5.0  # zero step passthrough


# ---- FakeSim: no network ----

class FakeSim(SimulatorExchange):
    """Simulator with network calls stubbed out."""

    def __init__(self, price: float = 0.15):
        super().__init__(starting_balance=10.0, taker_fee_pct=0.04, slippage_ticks=0)
        self._price = price

    async def connect(self) -> None:
        return None

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


def _make_filters():
    return SymbolFilters(
        symbol="DOGEUSDT",
        price_tick=0.00001,
        qty_step=1.0,
        min_qty=1.0,
        min_notional=5.0,
    )


def _make_params(**overrides):
    defaults = dict(
        symbol="DOGEUSDT",
        upper_price=0.16,
        lower_price=0.14,
        num_grids=4,
        leverage=10,
        qty_per_grid=50.0,
        reasoning="test grid",
    )
    defaults.update(overrides)
    return GridSetupParams(**defaults)


# ---- Grid setup tests ----

@pytest.mark.asyncio
async def test_grid_setup_creates_orders():
    sim = FakeSim(price=0.15)
    await sim.connect()
    store = StateStore("/tmp/test_grid_state.json")
    gm = GridManager(sim, store)
    filters = _make_filters()

    ok = await gm.setup_grid(_make_params(), filters, current_price=0.15)
    assert ok
    assert gm.active
    gs = gm.grid
    assert gs.symbol == "DOGEUSDT"
    assert gs.num_grids == 4
    assert gs.leverage == 10
    assert len(gs.levels) == 5  # num_grids + 1

    # Count placed orders
    buy_ids = [lv for lv in gs.levels if lv.buy_order_id is not None]
    sell_ids = [lv for lv in gs.levels if lv.sell_order_id is not None]
    # Price=0.15, lower=0.14, upper=0.16, spacing=0.005
    # Levels: 0.14, 0.145, 0.15, 0.155, 0.16
    # Below 0.15: 0.14 and 0.145 -> 2 buys
    # Above 0.15: 0.155 and 0.16 -> 2 sells
    # 0.15 itself gets neither (not < or > current_price)
    assert len(buy_ids) == 2
    assert len(sell_ids) == 2

    # Verify orders are in the sim
    orders = await sim.get_open_orders("DOGEUSDT")
    assert len(orders) == 4


@pytest.mark.asyncio
async def test_grid_setup_auto_bumps_below_min_qty():
    sim = FakeSim(price=0.15)
    await sim.connect()
    store = StateStore("/tmp/test_grid_state2.json")
    gm = GridManager(sim, store)
    filters = _make_filters()

    params = _make_params(qty_per_grid=0.5)  # below min_qty of 1.0
    ok = await gm.setup_grid(params, filters, current_price=0.15)
    assert ok  # auto-bumped to meet min_qty and min_notional
    assert gm.grid.qty_per_grid >= filters.min_qty


@pytest.mark.asyncio
async def test_grid_setup_auto_bumps_below_min_notional():
    sim = FakeSim(price=0.001)
    await sim.connect()
    store = StateStore("/tmp/test_grid_state3.json")
    gm = GridManager(sim, store)
    filters = _make_filters()

    # price 0.001 * qty 1 = 0.001 < min_notional 5.0, auto-bumped
    params = _make_params(
        lower_price=0.0009, upper_price=0.0011,
        qty_per_grid=1.0,
    )
    ok = await gm.setup_grid(params, filters, current_price=0.001)
    assert ok
    assert gm.grid.qty_per_grid * 0.0009 >= filters.min_notional


# ---- Fill detection and counter orders ----

@pytest.mark.asyncio
async def test_buy_fill_places_counter_sell():
    sim = FakeSim(price=0.15)
    await sim.connect()
    store = StateStore("/tmp/test_grid_fills.json")
    gm = GridManager(sim, store)
    filters = _make_filters()

    await gm.setup_grid(_make_params(), filters, current_price=0.15)

    # Move price down to fill BUY at 0.145
    sim._price = 0.144
    events = await gm.check_fills_and_reorder(0.144)

    # At least one buy should have filled
    buy_fills = [e for e in events if e["side"] == "BUY"]
    assert len(buy_fills) >= 1

    # After a BUY fill, a counter SELL should be placed one level up
    gs = gm.grid
    # The level above the filled buy should have a sell_order_id
    filled_level = buy_fills[0]["level"]
    if filled_level + 1 < len(gs.levels):
        next_level = gs.levels[filled_level + 1]
        assert next_level.sell_order_id is not None


@pytest.mark.asyncio
async def test_sell_fill_places_counter_buy():
    sim = FakeSim(price=0.15)
    await sim.connect()
    store = StateStore("/tmp/test_grid_sells.json")
    gm = GridManager(sim, store)
    filters = _make_filters()

    await gm.setup_grid(_make_params(), filters, current_price=0.15)

    # Move price up to fill SELL at 0.155
    sim._price = 0.156
    events = await gm.check_fills_and_reorder(0.156)

    sell_fills = [e for e in events if e["side"] == "SELL"]
    assert len(sell_fills) >= 1

    # After a SELL fill, a counter BUY should be placed one level down
    filled_level = sell_fills[0]["level"]
    if filled_level - 1 >= 0:
        prev_level = gm.grid.levels[filled_level - 1]
        assert prev_level.buy_order_id is not None


# ---- Teardown ----

@pytest.mark.asyncio
async def test_teardown_cancels_orders():
    sim = FakeSim(price=0.15)
    await sim.connect()
    store = StateStore("/tmp/test_grid_teardown.json")
    gm = GridManager(sim, store)
    filters = _make_filters()

    await gm.setup_grid(_make_params(), filters, current_price=0.15)
    assert gm.active

    await gm.teardown()
    assert not gm.active
    assert gm.grid.levels == []

    orders = await sim.get_open_orders("DOGEUSDT")
    assert len(orders) == 0


# ---- Unrealized PnL ----

def test_unrealized_pnl_no_position():
    sim = FakeSim()
    store = StateStore("/tmp/test_grid_upnl.json")
    gm = GridManager(sim, store)
    assert gm.unrealized_pnl(0.15) == 0.0


@pytest.mark.asyncio
async def test_unrealized_pnl_after_buy():
    sim = FakeSim(price=0.15)
    await sim.connect()
    store = StateStore("/tmp/test_grid_upnl2.json")
    gm = GridManager(sim, store)
    filters = _make_filters()

    await gm.setup_grid(_make_params(), filters, current_price=0.15)

    # Simulate a BUY fill to create a net long position
    gs = gm.grid
    gs.net_qty = 50.0
    gs.avg_entry = 0.145

    # Price goes up -> positive PnL
    upnl = gm.unrealized_pnl(0.155)
    assert upnl == pytest.approx((0.155 - 0.145) * 50.0)


# ---- Out of range check ----

@pytest.mark.asyncio
async def test_is_price_out_of_range():
    sim = FakeSim(price=0.15)
    await sim.connect()
    store = StateStore("/tmp/test_grid_oor.json")
    gm = GridManager(sim, store)
    filters = _make_filters()

    await gm.setup_grid(_make_params(), filters, current_price=0.15)

    # Grid range is 0.14 - 0.16, range = 0.02, 2% margin = 0.0004
    assert not gm.is_price_out_of_range(0.15, 2.0)  # in range
    assert not gm.is_price_out_of_range(0.16, 2.0)  # at edge
    assert gm.is_price_out_of_range(0.165, 2.0)      # above + margin
    assert gm.is_price_out_of_range(0.135, 2.0)      # below - margin


# ---- Realized PnL correctness ----

@pytest.mark.asyncio
async def test_realized_pnl_matches_avg_entry_not_spacing():
    """A SELL that closes a long must realize (sell_price - avg_entry) * qty,
    not the fake `spacing * qty` profit the old code always booked.
    """
    sim = FakeSim(price=0.15)
    await sim.connect()
    store = StateStore("/tmp/test_grid_pnl_real.json")
    gm = GridManager(sim, store)
    filters = _make_filters()

    await gm.setup_grid(_make_params(), filters, current_price=0.15)

    # Price drops — BUY at 0.145 fills (level 1) then at 0.14 (level 0)
    sim._price = 0.139
    await gm.check_fills_and_reorder(0.139)

    gs = gm.grid
    # Net long of 100 qty with avg between 0.14 and 0.145
    assert gs.net_qty == pytest.approx(100.0)
    assert 0.14 <= gs.avg_entry <= 0.145
    avg_entry = gs.avg_entry

    # Price pops back up — counter SELL at 0.145 fills
    sim._price = 0.146
    events = await gm.check_fills_and_reorder(0.146)
    sell_events = [e for e in events if e["side"] == "SELL"]
    assert sell_events

    # The realized PnL reported must reflect the true cost basis.
    ev = sell_events[0]
    qty = 50.0
    fee = 0.145 * qty * 0.0002
    expected = (0.145 - avg_entry) * qty - fee
    assert ev["pnl"] == pytest.approx(expected, rel=1e-6)


@pytest.mark.asyncio
async def test_loss_is_recorded_when_sell_below_avg_entry():
    """If price pumps so that sells fill at a loss relative to avg_entry
    (after the grid flips short), the loss must show up in daily stats.
    """
    sim = FakeSim(price=0.15)
    await sim.connect()
    store = StateStore("/tmp/test_grid_pnl_loss.json")
    gm = GridManager(sim, store)
    filters = _make_filters()

    await gm.setup_grid(_make_params(), filters, current_price=0.15)

    # Push price up so SELLs fill first (at 0.155, 0.16)
    sim._price = 0.161
    await gm.check_fills_and_reorder(0.161)
    gs = gm.grid
    # Net short now
    assert gs.net_qty < 0
    short_avg = gs.avg_entry

    # Price keeps going up — no counter BUY fills, but also confirm
    # _realized_pnl gives negative when we BUY above short_avg.
    pnl = gm._realized_pnl("BUY", short_avg + 0.002, 50.0)
    # Closing short at a higher price is a loss
    assert pnl < 0


@pytest.mark.asyncio
async def test_teardown_closes_naked_position():
    """teardown(close_position=True) must zero the net_qty, not leave it."""
    sim = FakeSim(price=0.15)
    await sim.connect()
    store = StateStore("/tmp/test_grid_teardown_close.json")
    gm = GridManager(sim, store)
    filters = _make_filters()

    await gm.setup_grid(_make_params(), filters, current_price=0.15)
    # Manually simulate an accumulated long inventory
    gs = gm.grid
    gs.net_qty = 50.0
    gs.avg_entry = 0.144
    store.save()

    # Open a matching sim position so market_close has something to close
    await sim.market_open("DOGEUSDT", "BUY", 50.0)

    await gm.teardown(close_position=True)
    assert gm.grid.net_qty == 0.0
    assert gm.grid.avg_entry == 0.0
    assert not gm.active


# ---- Grid summary ----

@pytest.mark.asyncio
async def test_grid_summary():
    sim = FakeSim(price=0.15)
    await sim.connect()
    store = StateStore("/tmp/test_grid_summary.json")
    gm = GridManager(sim, store)
    filters = _make_filters()

    await gm.setup_grid(_make_params(), filters, current_price=0.15)
    summary = gm.grid_summary(0.15)
    assert summary["symbol"] == "DOGEUSDT"
    assert summary["active"] is True
    assert summary["num_grids"] == 4
    assert summary["buy_orders"] == 2
    assert summary["sell_orders"] == 2
