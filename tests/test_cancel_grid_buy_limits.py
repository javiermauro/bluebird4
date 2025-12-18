import asyncio
import pytest

from types import SimpleNamespace


def test_cancel_grid_buy_limits_cancels_only_tracked_buy_limits():
    """Wrapper to run async test with asyncio.run()."""
    asyncio.run(_test_cancel_grid_buy_limits_cancels_only_tracked_buy_limits())


async def _test_cancel_grid_buy_limits_cancels_only_tracked_buy_limits():
    """
    cancel_grid_buy_limits() should:
    - cancel ONLY BUY limits that match an order_id in grid_strategy.open_limit_orders
    - remove the matching entry from open_limit_orders
    - record notional in overlay telemetry
    """
    from src.execution.bot_grid import cancel_grid_buy_limits
    from src.strategy.grid_trading import OpenLimitOrder

    cancelled = []
    untracked = []

    class FakeTradingClient:
        def cancel_order_by_id(self, order_id):
            cancelled.append(order_id)

    class FakeClient:
        def __init__(self, open_orders):
            self._open_orders = open_orders
            self.trading_client = FakeTradingClient()

        def get_open_orders(self, symbols=None):
            return self._open_orders

    class FakeOverlay:
        def __init__(self):
            self.cancelled = []
            self.untracked = []

        def record_cancelled_limit(self, symbol, notional):
            self.cancelled.append((symbol, notional))

        def record_untracked_buy(self, symbol, order_id):
            self.untracked.append((symbol, order_id))

    class FakeGridStrategy:
        def __init__(self, open_limit_orders):
            self.open_limit_orders = open_limit_orders
            self.saved = False

        def save_state(self):
            self.saved = True

    tracked_order_id = "order-123"
    tracked_key = "BTC/USD:buy:level-1"

    bot = SimpleNamespace(
        risk_overlay=FakeOverlay(),
        grid_strategy=FakeGridStrategy(
            open_limit_orders={
                tracked_key: OpenLimitOrder(
                    order_id=tracked_order_id,
                    symbol="BTC/USD",
                    side="buy",
                    level_id="level-1",
                    level_price=100.0,
                    limit_price=100.0,
                    qty=2.0,
                )
            }
        ),
    )

    open_orders = [
        # Should cancel (tracked BUY limit)
        {"id": tracked_order_id, "side": "buy", "type": "limit", "qty": 2.0, "limit_price": 100.0},
        # Should NOT cancel (untracked BUY limit)
        {"id": "order-999", "side": "buy", "type": "limit", "qty": 1.0, "limit_price": 50.0},
        # Should ignore (sell)
        {"id": "order-sell", "side": "sell", "type": "limit", "qty": 1.0, "limit_price": 120.0},
    ]

    client = FakeClient(open_orders=open_orders)
    count = await cancel_grid_buy_limits(bot, ["BTC/USD"], client)

    assert count == 1
    assert cancelled == [tracked_order_id]
    assert tracked_key not in bot.grid_strategy.open_limit_orders
    assert bot.grid_strategy.saved is True

    # Telemetry recorded for cancellation
    assert bot.risk_overlay.cancelled == [("BTC/USD", 200.0)]
    # Untracked buy should be noted but not cancelled
    assert bot.risk_overlay.untracked == [("BTC/USD", "order-999")]


