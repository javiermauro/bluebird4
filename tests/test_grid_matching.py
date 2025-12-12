"""
Unit tests for grid matching correctness.

Tests the deterministic fill-to-level matching logic, including:
- Normalization helpers
- Idempotency (same order_id applied twice is a no-op)
- Side-safe matching (BUY fills can't mark SELL levels)
- Pending order mapping
- Verify timeout recovery via reconciliation
- completed_cycles only for grid sells
"""

import pytest
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.strategy.grid_trading import (
    GridTradingStrategy,
    GridConfig,
    GridLevel,
    GridOrderSide,
    PendingOrder,
    normalize_side,
    normalize_symbol
)


class TestNormalization:
    """Tests for normalize_side() and normalize_symbol()."""

    def test_normalize_side_lowercase(self):
        assert normalize_side("buy") == "buy"
        assert normalize_side("sell") == "sell"

    def test_normalize_side_uppercase(self):
        assert normalize_side("BUY") == "buy"
        assert normalize_side("SELL") == "sell"

    def test_normalize_side_enum_value(self):
        """Test with Alpaca OrderSide enum-like values."""
        assert normalize_side("OrderSide.BUY") == "buy"
        assert normalize_side("OrderSide.SELL") == "sell"

    def test_normalize_side_with_enum(self):
        """Test with actual GridOrderSide enum."""
        assert normalize_side(GridOrderSide.BUY) == "buy"
        assert normalize_side(GridOrderSide.SELL) == "sell"

    def test_normalize_side_invalid(self):
        with pytest.raises(ValueError):
            normalize_side("hold")

    def test_normalize_symbol_with_slash(self):
        assert normalize_symbol("BTC/USD") == "BTC/USD"
        assert normalize_symbol("btc/usd") == "BTC/USD"

    def test_normalize_symbol_without_slash(self):
        assert normalize_symbol("BTCUSD") == "BTC/USD"
        assert normalize_symbol("btcusd") == "BTC/USD"
        assert normalize_symbol("SOLUSD") == "SOL/USD"
        assert normalize_symbol("LTCUSD") == "LTC/USD"
        assert normalize_symbol("AVAXUSD") == "AVAX/USD"

    def test_normalize_symbol_unknown(self):
        # Unknown base currencies pass through unchanged
        assert normalize_symbol("XYZUSD") == "XYZUSD"


def make_test_config(symbol="BTC/USD", upper=100000, lower=90000, num_grids=10, investment=100):
    """Helper to create test GridConfig."""
    return GridConfig(
        symbol=symbol,
        upper_price=upper,
        lower_price=lower,
        num_grids=num_grids,
        investment_per_grid=investment
    )


def make_test_strategy(symbol="BTC/USD", upper=100000, lower=90000, current_price=95000):
    """Helper to create a test strategy with initialized grid."""
    strategy = GridTradingStrategy()
    config = make_test_config(symbol=symbol, upper=upper, lower=lower)
    strategy.create_grid(config, current_price)
    return strategy


class TestIdempotency:
    """Tests for idempotent fill application."""

    def test_same_order_id_applied_twice_is_noop(self):
        """Applying the same order_id twice should be a no-op."""
        strategy = make_test_strategy()

        # First apply - use 94000 which is a BUY level (current_price=95000, so levels below are BUY)
        result1 = strategy.apply_filled_order(
            "BTC/USD", "buy", 94000, 0.01, "order-123"
        )

        # Should have one fill
        assert strategy.grids["BTC/USD"].total_fills == 1

        # Second apply (same order_id)
        result2 = strategy.apply_filled_order(
            "BTC/USD", "buy", 94000, 0.01, "order-123"
        )

        # Should be no-op (None return, still 1 fill)
        assert result2 is None
        assert strategy.grids["BTC/USD"].total_fills == 1
        assert "order-123" in strategy.applied_order_ids

    def test_different_order_ids_both_applied(self):
        """Different order_ids should both be applied."""
        strategy = make_test_strategy()

        # Use 93000 and 94000 which are BUY levels
        strategy.apply_filled_order("BTC/USD", "buy", 93000, 0.01, "order-1")
        strategy.apply_filled_order("BTC/USD", "buy", 94000, 0.01, "order-2")

        assert strategy.grids["BTC/USD"].total_fills == 2


class TestSideSafety:
    """Tests for side-safe matching."""

    def test_side_mismatch_rejected(self):
        """Pending BUY order should not match SELL fill."""
        strategy = make_test_strategy()

        # Register pending as BUY at a BUY level (94000)
        strategy.register_pending_order(
            order_id="order-123",
            symbol="BTC/USD",
            side="buy",
            intended_level_price=94000.0
        )

        # Try to apply as SELL (mismatch in side)
        result = strategy.apply_filled_order(
            "BTC/USD", "sell", 94000, 0.01, "order-123"
        )

        # Should be rejected due to side mismatch
        assert result is None
        assert "order-123" in strategy.unmatched_fills
        assert strategy.unmatched_fills["order-123"]["reason"] == "side_mismatch"
        # Pending should NOT be deleted on mismatch
        assert "order-123" in strategy.pending_orders

    def test_fallback_is_side_gated(self):
        """Fallback matching (no pending) should still respect side."""
        strategy = make_test_strategy()

        # Get initial state - there should be buy levels
        state = strategy.grids["BTC/USD"]
        buy_levels_before = sum(1 for l in state.levels if l.side == GridOrderSide.BUY and not l.is_filled)

        # Try to apply a SELL fill at a BUY level price (94000)
        # This should NOT match any BUY level because side doesn't match
        result = strategy.apply_filled_order(
            "BTC/USD", "sell", 94000, 0.01, "orphan-sell"
        )

        # Should be unmatched (94000 is a BUY level, not a SELL level)
        assert result is None
        assert "orphan-sell" in strategy.unmatched_fills

        # Buy levels should be unchanged
        buy_levels_after = sum(1 for l in state.levels if l.side == GridOrderSide.BUY and not l.is_filled)
        assert buy_levels_before == buy_levels_after


class TestPendingOrderMatching:
    """Tests for pending order registration and matching."""

    def test_register_pending_order(self):
        """Test that pending orders are registered correctly."""
        strategy = make_test_strategy()

        strategy.register_pending_order(
            order_id="order-123",
            symbol="BTC/USD",
            side="buy",
            intended_level_price=94000.0,
            intended_level_id="abc123",
            source="grid"
        )

        assert "order-123" in strategy.pending_orders
        pending = strategy.pending_orders["order-123"]
        assert pending.symbol == "BTC/USD"
        assert pending.side == "buy"
        assert pending.intended_level_price == 94000.0
        assert pending.intended_level_id == "abc123"
        assert pending.source == "grid"

    def test_pending_mapping_deleted_on_successful_apply(self):
        """Pending order should be deleted after successful apply."""
        strategy = make_test_strategy()

        # Use 94000 which is a BUY level
        strategy.register_pending_order(
            order_id="order-123",
            symbol="BTC/USD",
            side="buy",
            intended_level_price=94000.0
        )

        # Apply the fill at the same price
        strategy.apply_filled_order("BTC/USD", "buy", 94000, 0.01, "order-123")

        # Pending should be deleted
        assert "order-123" not in strategy.pending_orders
        assert "order-123" in strategy.applied_order_ids


class TestVerifyTimeoutRecovery:
    """Tests for verify timeout recovery via reconciliation."""

    def test_verify_timeout_recovery(self):
        """Test that reconciliation can apply a fill after verify times out."""
        strategy = make_test_strategy()

        # Register pending (simulates submit before verify times out)
        # Use 94000 which is a BUY level
        strategy.register_pending_order(
            order_id="order-123",
            symbol="BTC/USD",
            side="buy",
            intended_level_price=94000.0
        )

        # Pending is registered but not yet applied
        assert "order-123" in strategy.pending_orders
        assert "order-123" not in strategy.applied_order_ids

        # Later, reconciliation discovers the fill (slightly different price)
        result = strategy.apply_filled_order(
            "BTC/USD", "buy", 94010, 0.01, "order-123"
        )

        # Should have been applied successfully
        assert "order-123" in strategy.applied_order_ids
        assert "order-123" not in strategy.pending_orders
        assert strategy.grids["BTC/USD"].total_fills == 1


class TestCompletedCycles:
    """Tests for accurate completed_cycles counting."""

    def test_completed_cycles_only_grid_sells(self):
        """Only grid sells should increment completed_cycles."""
        strategy = make_test_strategy()

        # First, do a buy to create a sell level
        strategy.apply_filled_order("BTC/USD", "buy", 95000, 0.01, "buy-1", source="grid")

        # Now there should be a sell level at upper price - grid_spacing
        state = strategy.grids["BTC/USD"]
        sell_levels = [l for l in state.levels if l.side == GridOrderSide.SELL and not l.is_filled]
        assert len(sell_levels) >= 1

        # Grid sell (should increment cycles) - use the actual sell level price
        sell_price = sell_levels[0].price
        strategy.apply_filled_order("BTC/USD", "sell", sell_price, 0.01, "sell-1", source="grid")
        assert strategy.grids["BTC/USD"].completed_cycles == 1

        # Stop-loss sell (should NOT increment cycles)
        # First, create another position
        strategy.apply_filled_order("BTC/USD", "buy", 94000, 0.01, "buy-2", source="grid")
        # Get the new sell level
        sell_levels2 = [l for l in state.levels if l.side == GridOrderSide.SELL and not l.is_filled]
        if sell_levels2:
            sell_price2 = sell_levels2[0].price
            # Register and apply as stop_loss
            strategy.register_pending_order("sell-2", "BTC/USD", "sell", sell_price2, source="stop_loss")
            strategy.apply_filled_order("BTC/USD", "sell", sell_price2, 0.01, "sell-2", source="stop_loss")
            # Cycles should still be 1
            assert strategy.grids["BTC/USD"].completed_cycles == 1

    def test_total_fills_includes_all(self):
        """total_fills should count all fills regardless of source."""
        strategy = make_test_strategy()

        # Use BUY level prices (below 95000)
        strategy.apply_filled_order("BTC/USD", "buy", 92000, 0.01, "order-1", source="grid")
        strategy.apply_filled_order("BTC/USD", "buy", 93000, 0.01, "order-2", source="stop_loss")
        strategy.apply_filled_order("BTC/USD", "buy", 94000, 0.01, "order-3", source="windfall")

        # All three should be counted
        assert strategy.grids["BTC/USD"].total_fills == 3


class TestGridLevel:
    """Tests for GridLevel dataclass."""

    def test_level_has_uuid(self):
        """Each GridLevel should have a unique level_id."""
        level1 = GridLevel(price=95000, side=GridOrderSide.BUY)
        level2 = GridLevel(price=95000, side=GridOrderSide.BUY)

        assert level1.level_id is not None
        assert level2.level_id is not None
        assert level1.level_id != level2.level_id

    def test_level_id_persisted_in_dict(self):
        """level_id should be persisted in to_dict/from_dict."""
        level = GridLevel(price=95000, side=GridOrderSide.BUY)
        original_id = level.level_id

        level_dict = level.to_dict()
        assert "level_id" in level_dict
        assert level_dict["level_id"] == original_id

        restored = GridLevel.from_dict(level_dict)
        assert restored.level_id == original_id


class TestPersistence:
    """Tests for state persistence."""

    def test_pending_orders_persisted(self):
        """Pending orders should survive save/load cycle."""
        strategy = make_test_strategy()

        strategy.register_pending_order(
            order_id="order-123",
            symbol="BTC/USD",
            side="buy",
            intended_level_price=95000.0
        )

        # Save state
        strategy.save_state()

        # Create new strategy and load
        new_strategy = GridTradingStrategy()
        new_strategy.load_state()

        assert "order-123" in new_strategy.pending_orders

    def test_applied_order_ids_persisted(self):
        """Applied order IDs should survive save/load cycle."""
        strategy = make_test_strategy()

        # Use 94000 which is a BUY level
        strategy.apply_filled_order("BTC/USD", "buy", 94000, 0.01, "order-123")

        # Save state
        strategy.save_state()

        # Create new strategy and load
        new_strategy = GridTradingStrategy()
        new_strategy.load_state()

        assert "order-123" in new_strategy.applied_order_ids


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
