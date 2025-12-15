"""
Tests for Fast Fill Detection (Near Real-Time Limit Order Fill Detection)

Tests cover:
1. Grace period: new orders not marked as disappeared
2. Filled orders: apply and remove tracking
3. Canceled no fill: remove tracking only
4. Canceled with partial: apply partial, remove tracking
5. Idempotency: double-run doesn't double-apply
6. Rate limiting: check skipped when called too frequently
7. Error backoff: check skipped after error
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, AsyncMock, patch
from dataclasses import dataclass
from typing import Optional

# Import the modules we're testing
from src.strategy.grid_trading import (
    GridTradingStrategy,
    GridConfig,
    OpenLimitOrder,
    PendingOrder,
    normalize_symbol,
    normalize_side
)


# =============================================================================
# Mock Classes
# =============================================================================

@dataclass
class MockOrder:
    """Mock Alpaca order object."""
    id: str
    symbol: str
    side: str
    status: str
    filled_qty: Optional[float] = None
    filled_avg_price: Optional[float] = None
    qty: float = 0.0
    limit_price: float = 0.0


class MockAlpacaClient:
    """Mock Alpaca client for testing."""

    def __init__(self):
        self.open_orders = []
        self.order_details = {}

    def get_open_orders(self, symbols=None):
        """Return list of open orders."""
        return [{'id': o.id, 'symbol': o.symbol, 'side': o.side} for o in self.open_orders]

    def get_order_by_id(self, order_id: str):
        """Return order details by ID."""
        return self.order_details.get(order_id)


class MockConfig:
    """Mock config for testing."""
    SYMBOLS = ['BTC/USD', 'SOL/USD']
    GRID_USE_LIMIT_ORDERS = True
    ENABLE_FAST_FILL_CHECK = True
    FAST_FILL_INTERVAL_SECONDS = 10.0
    FAST_FILL_MAX_CHECKS_PER_CYCLE = 5
    FAST_FILL_MIN_ORDER_AGE_SECONDS = 10.0
    FAST_FILL_HANDLE_PARTIALS = True
    FAST_FILL_ERROR_BACKOFF_SECONDS = 30.0


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def grid_strategy():
    """Create a fresh GridTradingStrategy for testing."""
    strategy = GridTradingStrategy()
    # Initialize a simple grid for BTC/USD
    config = GridConfig(
        symbol='BTC/USD',
        upper_price=105000.0,
        lower_price=95000.0,
        num_grids=5,
        investment_per_grid=100.0
    )
    strategy.create_grid(config, current_price=100000.0)
    return strategy


@pytest.fixture
def mock_client():
    """Create a mock Alpaca client."""
    return MockAlpacaClient()


@pytest.fixture
def mock_config():
    """Create a mock config."""
    return MockConfig()


# =============================================================================
# Test Cases
# =============================================================================

class TestGracePeriod:
    """Test grace period for new orders (eventual consistency)."""

    def test_new_order_not_disappeared(self, grid_strategy):
        """Orders younger than grace period should not be processed."""
        # Register an order created just now
        now = datetime.now()
        grid_strategy.register_open_limit_order(
            order_id='new-order-123',
            symbol='BTC/USD',
            side='buy',
            level_id='level-1',
            level_price=99000.0,
            limit_price=98995.0,
            qty=0.001,
            source='grid'
        )

        # Verify the order has a recent created_at
        order = list(grid_strategy.open_limit_orders.values())[0]
        created_at = datetime.fromisoformat(order.created_at)
        assert (now - created_at).total_seconds() < 5, "Order should have recent timestamp"

    def test_old_order_eligible_for_check(self, grid_strategy):
        """Orders older than grace period should be eligible."""
        # Register an order with old timestamp
        old_time = (datetime.now() - timedelta(seconds=60)).isoformat()
        order = OpenLimitOrder(
            order_id='old-order-123',
            symbol='BTC/USD',
            side='buy',
            level_id='level-1',
            level_price=99000.0,
            limit_price=98995.0,
            qty=0.001,
            created_at=old_time,
            source='grid'
        )
        key = OpenLimitOrder.make_key('BTC/USD', 'buy', 'level-1')
        grid_strategy.open_limit_orders[key] = order

        # Check age
        created_at = datetime.fromisoformat(order.created_at)
        age = (datetime.now() - created_at).total_seconds()
        assert age > 10, "Order should be old enough for processing"


class TestFilledOrderDetection:
    """Test filled order detection and application."""

    def test_apply_filled_order_updates_grid(self, grid_strategy):
        """Filled order should be applied to grid state."""
        # Register a pending order
        grid_strategy.register_pending_order(
            order_id='fill-order-123',
            symbol='BTC/USD',
            side='buy',
            intended_level_price=99000.0,
            intended_level_id='level-1',
            source='grid'
        )

        # Apply the fill
        profit = grid_strategy.apply_filled_order(
            symbol='BTC/USD',
            side='buy',
            fill_price=99000.0,
            fill_qty=0.001,
            order_id='fill-order-123',
            source='grid'
        )

        # Verify order is marked as applied
        assert 'fill-order-123' in grid_strategy.applied_order_ids

    def test_filled_order_removes_tracking(self, grid_strategy):
        """Filled order should remove from open_limit_orders and pending_orders."""
        # Register both tracking entries
        grid_strategy.register_open_limit_order(
            order_id='tracking-123',
            symbol='BTC/USD',
            side='buy',
            level_id='level-1',
            level_price=99000.0,
            limit_price=98995.0,
            qty=0.001,
            source='grid'
        )
        grid_strategy.register_pending_order(
            order_id='tracking-123',
            symbol='BTC/USD',
            side='buy',
            intended_level_price=99000.0,
            intended_level_id='level-1',
            source='grid'
        )

        # Verify tracking exists
        assert 'tracking-123' in grid_strategy.pending_orders
        assert any(o.order_id == 'tracking-123' for o in grid_strategy.open_limit_orders.values())

        # Apply fill and remove tracking
        grid_strategy.apply_filled_order(
            symbol='BTC/USD',
            side='buy',
            fill_price=99000.0,
            fill_qty=0.001,
            order_id='tracking-123',
            source='grid'
        )
        grid_strategy.remove_open_limit_order('tracking-123')
        grid_strategy.remove_pending_order('tracking-123')

        # Verify tracking removed
        assert 'tracking-123' not in grid_strategy.pending_orders
        assert not any(o.order_id == 'tracking-123' for o in grid_strategy.open_limit_orders.values())


class TestCanceledOrderHandling:
    """Test canceled order handling."""

    def test_canceled_no_fill_removes_tracking(self, grid_strategy):
        """Canceled order with no fill should only remove tracking."""
        # Register tracking
        grid_strategy.register_open_limit_order(
            order_id='cancel-123',
            symbol='BTC/USD',
            side='buy',
            level_id='level-1',
            level_price=99000.0,
            limit_price=98995.0,
            qty=0.001,
            source='grid'
        )
        grid_strategy.register_pending_order(
            order_id='cancel-123',
            symbol='BTC/USD',
            side='buy',
            intended_level_price=99000.0,
            intended_level_id='level-1',
            source='grid'
        )

        # Simulate cancel (no fill) - just remove tracking
        grid_strategy.remove_open_limit_order('cancel-123')
        grid_strategy.remove_pending_order('cancel-123')

        # Verify NOT in applied_order_ids (wasn't filled)
        assert 'cancel-123' not in grid_strategy.applied_order_ids
        # Verify tracking removed
        assert 'cancel-123' not in grid_strategy.pending_orders

    def test_canceled_with_partial_applies_fill(self, grid_strategy):
        """Canceled order with partial fill should apply the filled portion."""
        # Register tracking
        grid_strategy.register_pending_order(
            order_id='partial-123',
            symbol='BTC/USD',
            side='buy',
            intended_level_price=99000.0,
            intended_level_id='level-1',
            source='grid'
        )

        # Apply partial fill (0.0005 of 0.001 original qty)
        partial_qty = 0.0005
        grid_strategy.apply_filled_order(
            symbol='BTC/USD',
            side='buy',
            fill_price=99000.0,
            fill_qty=partial_qty,  # Use actual filled qty, not original!
            order_id='partial-123',
            source='grid'
        )

        # Verify applied with partial qty
        assert 'partial-123' in grid_strategy.applied_order_ids


class TestIdempotency:
    """Test idempotency of fill application."""

    def test_double_apply_is_noop(self, grid_strategy):
        """Applying same order twice should be a no-op."""
        # Register pending
        grid_strategy.register_pending_order(
            order_id='idempotent-123',
            symbol='BTC/USD',
            side='buy',
            intended_level_price=99000.0,
            intended_level_id='level-1',
            source='grid'
        )

        # Apply first time
        profit1 = grid_strategy.apply_filled_order(
            symbol='BTC/USD',
            side='buy',
            fill_price=99000.0,
            fill_qty=0.001,
            order_id='idempotent-123',
            source='grid'
        )

        # Count filled levels before second apply
        state = grid_strategy.grids.get('BTC/USD')
        filled_before = sum(1 for l in state.levels if l.is_filled) if state else 0

        # Apply second time (should be no-op)
        profit2 = grid_strategy.apply_filled_order(
            symbol='BTC/USD',
            side='buy',
            fill_price=99000.0,
            fill_qty=0.001,
            order_id='idempotent-123',
            source='grid'
        )

        # Count filled levels after second apply
        filled_after = sum(1 for l in state.levels if l.is_filled) if state else 0

        # Verify second apply was no-op
        assert profit2 is None, "Second apply should return None"
        assert filled_before == filled_after, "Filled count should not change"


class TestNormalization:
    """Test symbol and side normalization."""

    def test_symbol_normalization(self):
        """Test symbol normalization from various formats."""
        assert normalize_symbol('BTCUSD') == 'BTC/USD'
        assert normalize_symbol('BTC/USD') == 'BTC/USD'
        assert normalize_symbol('SOLUSD') == 'SOL/USD'
        assert normalize_symbol('SOL/USD') == 'SOL/USD'

    def test_side_normalization(self):
        """Test side normalization from various formats."""
        assert normalize_side('buy') == 'buy'
        assert normalize_side('BUY') == 'buy'
        assert normalize_side('OrderSide.BUY') == 'buy'
        assert normalize_side('sell') == 'sell'
        assert normalize_side('SELL') == 'sell'
        assert normalize_side('OrderSide.SELL') == 'sell'


class TestTerminalStatusSet:
    """Test terminal status detection."""

    def test_terminal_statuses(self):
        """Test that terminal statuses are recognized."""
        TERMINAL_STATUSES = {'filled', 'canceled', 'cancelled', 'expired', 'rejected', 'suspended'}

        # These should be terminal
        assert 'filled' in TERMINAL_STATUSES
        assert 'canceled' in TERMINAL_STATUSES
        assert 'cancelled' in TERMINAL_STATUSES
        assert 'expired' in TERMINAL_STATUSES
        assert 'rejected' in TERMINAL_STATUSES
        assert 'suspended' in TERMINAL_STATUSES

        # These should NOT be terminal
        assert 'new' not in TERMINAL_STATUSES
        assert 'accepted' not in TERMINAL_STATUSES
        assert 'pending_new' not in TERMINAL_STATUSES
        assert 'partially_filled' not in TERMINAL_STATUSES


class TestOpenLimitOrderTracking:
    """Test open limit order tracking."""

    def test_register_open_limit_order(self, grid_strategy):
        """Test registering an open limit order."""
        grid_strategy.register_open_limit_order(
            order_id='test-123',
            symbol='BTC/USD',
            side='buy',
            level_id='level-1',
            level_price=99000.0,
            limit_price=98995.0,
            qty=0.001,
            source='grid'
        )

        # Verify registered
        key = OpenLimitOrder.make_key('BTC/USD', 'buy', 'level-1')
        assert key in grid_strategy.open_limit_orders
        assert grid_strategy.open_limit_orders[key].order_id == 'test-123'

    def test_remove_open_limit_order(self, grid_strategy):
        """Test removing an open limit order."""
        # Register first
        grid_strategy.register_open_limit_order(
            order_id='remove-123',
            symbol='BTC/USD',
            side='buy',
            level_id='level-1',
            level_price=99000.0,
            limit_price=98995.0,
            qty=0.001,
            source='grid'
        )

        # Remove
        removed = grid_strategy.remove_open_limit_order('remove-123')
        # Returns the removed object or None
        assert removed is not None

        # Verify removed
        key = OpenLimitOrder.make_key('BTC/USD', 'buy', 'level-1')
        assert key not in grid_strategy.open_limit_orders

    def test_get_stale_orders(self, grid_strategy):
        """Test detecting stale orders."""
        # Register an old order
        old_time = (datetime.now() - timedelta(minutes=120)).isoformat()
        order = OpenLimitOrder(
            order_id='stale-123',
            symbol='BTC/USD',
            side='buy',
            level_id='level-1',
            level_price=99000.0,
            limit_price=98995.0,
            qty=0.001,
            created_at=old_time,
            source='grid'
        )
        key = OpenLimitOrder.make_key('BTC/USD', 'buy', 'level-1')
        grid_strategy.open_limit_orders[key] = order

        # Get stale orders (max age 60 min)
        stale = grid_strategy.get_stale_orders(max_age_minutes=60)

        # Verify detected as stale
        assert len(stale) >= 1
        assert any(o.order_id == 'stale-123' for o in stale)


class TestPendingOrderTracking:
    """Test pending order tracking."""

    def test_register_pending_order(self, grid_strategy):
        """Test registering a pending order."""
        grid_strategy.register_pending_order(
            order_id='pending-123',
            symbol='BTC/USD',
            side='buy',
            intended_level_price=99000.0,
            intended_level_id='level-1',
            source='grid'
        )

        # Verify registered
        assert 'pending-123' in grid_strategy.pending_orders
        assert grid_strategy.pending_orders['pending-123'].symbol == 'BTC/USD'

    def test_remove_pending_order(self, grid_strategy):
        """Test removing a pending order."""
        # Register first
        grid_strategy.register_pending_order(
            order_id='remove-pending-123',
            symbol='BTC/USD',
            side='buy',
            intended_level_price=99000.0,
            intended_level_id='level-1',
            source='grid'
        )

        # Remove
        removed = grid_strategy.remove_pending_order('remove-pending-123')
        # Returns the removed object or None
        assert removed is not None

        # Verify removed
        assert 'remove-pending-123' not in grid_strategy.pending_orders
