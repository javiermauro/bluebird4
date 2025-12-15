"""
Grid Trading Strategy for BLUEBIRD 4.0

This strategy profits from sideways market volatility without needing predictions.
It places buy/sell orders at regular price intervals (the "grid") and captures
profit from every price oscillation within the defined range.

Why Grid Trading works for us:
1. Our model predicts SIDEWAYS 90% of the time (correctly!)
2. Grid trading thrives in sideways/ranging markets
3. No predictions needed - just profits from natural price volatility
4. The more the price oscillates, the more profit we capture

Key concepts:
- Grid Range: The price range we operate in (e.g., $85,000 - $90,000)
- Grid Levels: The number of buy/sell price points within the range
- Grid Spacing: The distance between grid levels (profit per trade)
- Upper/Lower Limits: Safety boundaries for the grid
"""

import json
import logging
import os
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
from enum import Enum

import config_ultra as config

logger = logging.getLogger("GridTrading")


# =============================================================================
# Normalization Helpers
# =============================================================================

def normalize_side(side: Any) -> str:
    """
    Normalize side to canonical 'buy' | 'sell'.

    Accepts: 'buy', 'sell', 'OrderSide.BUY', enum values, etc.
    """
    if hasattr(side, 'value'):  # Enum
        side = side.value
    side_str = str(side).lower()
    if 'buy' in side_str:
        return 'buy'
    if 'sell' in side_str:
        return 'sell'
    raise ValueError(f"Unknown side: {side}")


def normalize_symbol(symbol: str) -> str:
    """
    Normalize symbol to canonical 'BTC/USD' format.

    Handles: 'BTCUSD' <-> 'BTC/USD'
    """
    s = symbol.upper().replace(' ', '')
    if '/' not in s:
        for base in ['BTC', 'ETH', 'SOL', 'LTC', 'AVAX', 'LINK']:
            if s.startswith(base):
                return f"{base}/{s[len(base):]}"
    return s


def normalize_fee_type(value: Any) -> str:
    """
    Normalize fee_type to canonical 'maker' or 'taker'.

    Handles: None, 'Maker', 'TAKER', etc.
    Returns 'taker' for unknown/None values (conservative default).
    """
    if value is None:
        return "taker"
    s = str(value).lower().strip()
    if s == "maker":
        return "maker"
    return "taker"


# Grid state persistence file
GRID_STATE_FILE = "/tmp/bluebird-grid-state.json"


class GridOrderSide(Enum):
    BUY = "buy"
    SELL = "sell"


@dataclass
class GridLevel:
    """Represents a single grid level with its order status."""
    price: float
    side: GridOrderSide
    is_filled: bool = False
    order_id: Optional[str] = None
    filled_at: Optional[datetime] = None
    quantity: float = 0.0
    level_id: str = field(default_factory=lambda: uuid.uuid4().hex)  # Stable identity (full 32-char UUID)
    # Fee model: origin buy info for sell levels (set when buy creates corresponding sell)
    origin_buy_price: float = 0.0       # Price of the buy that created this sell
    origin_buy_fee_type: str = "taker"  # Fee type of that buy ("maker" or "taker")

    def to_dict(self) -> dict:
        """Serialize grid level for persistence."""
        return {
            'price': self.price,
            'side': self.side.value,
            'is_filled': self.is_filled,
            'order_id': self.order_id,
            'filled_at': self.filled_at.isoformat() if self.filled_at else None,
            'quantity': self.quantity,
            'level_id': self.level_id,
            'origin_buy_price': self.origin_buy_price,
            'origin_buy_fee_type': self.origin_buy_fee_type
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'GridLevel':
        """Deserialize grid level from persistence."""
        return cls(
            price=data['price'],
            side=GridOrderSide(data['side']),
            is_filled=data['is_filled'],
            order_id=data.get('order_id'),
            filled_at=datetime.fromisoformat(data['filled_at']) if data.get('filled_at') else None,
            quantity=data.get('quantity', 0.0),
            level_id=data.get('level_id', uuid.uuid4().hex),  # Generate if missing
            origin_buy_price=data.get('origin_buy_price', 0.0),
            origin_buy_fee_type=normalize_fee_type(data.get('origin_buy_fee_type'))
        )


@dataclass
class GridConfig:
    """Configuration for a grid trading setup on a single symbol."""
    symbol: str
    upper_price: float  # Top of grid range
    lower_price: float  # Bottom of grid range
    num_grids: int = 10  # Number of grid levels
    investment_per_grid: float = 100.0  # $ per grid level

    # Safety settings
    stop_loss_pct: float = 0.05  # Stop loss if price drops 5% below grid
    take_profit_pct: float = 0.10  # Take profit if price rises 10% above grid

    # Auto-adjust settings
    auto_rebalance: bool = True  # Rebalance grid when price breaks range
    rebalance_threshold: float = 0.02  # Rebalance if price is 2% outside range

    def __post_init__(self):
        """Calculate grid spacing after initialization."""
        self.grid_spacing = (self.upper_price - self.lower_price) / self.num_grids
        self.profit_per_grid_pct = self.grid_spacing / self.lower_price * 100


@dataclass
class PendingOrder:
    """
    Tracks an order awaiting fill confirmation.

    Used for deterministic order->level matching and verify-timeout recovery.
    """
    order_id: str
    symbol: str                    # Canonical: "BTC/USD"
    side: str                      # Canonical: "buy" | "sell"
    intended_level_price: float
    intended_level_id: Optional[str] = None  # For stable matching
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    source: str = "grid"           # "grid" | "windfall" | "stop_loss"
    client_order_id: Optional[str] = None
    fee_type: str = "taker"        # "maker" or "taker" - for fee attribution

    def to_dict(self) -> dict:
        """Serialize for persistence."""
        return {
            'order_id': self.order_id,
            'symbol': self.symbol,
            'side': self.side,
            'intended_level_price': self.intended_level_price,
            'intended_level_id': self.intended_level_id,
            'created_at': self.created_at,
            'source': self.source,
            'client_order_id': self.client_order_id,
            'fee_type': self.fee_type
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'PendingOrder':
        """Deserialize from persistence."""
        return cls(
            order_id=data['order_id'],
            symbol=data['symbol'],
            side=data['side'],
            intended_level_price=data['intended_level_price'],
            intended_level_id=data.get('intended_level_id'),
            created_at=data.get('created_at', datetime.now().isoformat()),
            source=data.get('source', 'grid'),
            client_order_id=data.get('client_order_id'),
            fee_type=normalize_fee_type(data.get('fee_type'))
        )


@dataclass
class OpenLimitOrder:
    """
    Tracks an open limit order on the exchange.

    Used to prevent duplicate orders for the same level and
    to cancel stale orders that haven't filled.
    """
    order_id: str
    symbol: str                    # Canonical: "BTC/USD"
    side: str                      # Canonical: "buy" | "sell"
    level_id: str                  # Grid level ID
    level_price: float             # Grid level price (where we wanted to trade)
    limit_price: float             # Actual limit price submitted
    qty: float
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    source: str = "grid"           # "grid" | "windfall" | "stop_loss"
    fee_type: str = "maker"        # "maker" for limit orders (default)

    def to_dict(self) -> dict:
        return {
            'order_id': self.order_id,
            'symbol': self.symbol,
            'side': self.side,
            'level_id': self.level_id,
            'level_price': self.level_price,
            'limit_price': self.limit_price,
            'qty': self.qty,
            'created_at': self.created_at,
            'source': self.source,
            'fee_type': self.fee_type
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'OpenLimitOrder':
        return cls(
            order_id=data['order_id'],
            symbol=data['symbol'],
            side=data['side'],
            level_id=data['level_id'],
            level_price=data['level_price'],
            limit_price=data['limit_price'],
            qty=data['qty'],
            created_at=data.get('created_at', datetime.now().isoformat()),
            source=data.get('source', 'grid'),
            fee_type=normalize_fee_type(data.get('fee_type', 'maker'))  # Default maker for limit orders
        )

    @staticmethod
    def make_key(symbol: str, side: str, level_id: str) -> str:
        """Create unique key for order tracking."""
        return f"{symbol}:{side}:{level_id}"


@dataclass
class GridState:
    """Tracks the current state of a grid trading setup."""
    config: GridConfig
    levels: List[GridLevel] = field(default_factory=list)
    is_active: bool = False
    total_profit: float = 0.0
    completed_trades: int = 0  # KEEP for backward compat (alias to total_fills)
    created_at: datetime = field(default_factory=datetime.now)
    last_trade_at: Optional[datetime] = None

    # Performance tracking
    total_buys: int = 0
    total_sells: int = 0
    avg_buy_price: float = 0.0
    avg_sell_price: float = 0.0

    # NEW: Accurate fill tracking
    total_fills: int = 0           # Every applied fill (buy or sell)
    completed_cycles: int = 0      # Only grid sells (not stop-loss/windfall)

    # Fee model diagnostics: track how profit is calculated
    fee_origin_paired_count: int = 0    # Sells with paired buy metadata
    fee_origin_estimated_count: int = 0  # Sells using estimated buy price

    def to_dict(self) -> dict:
        """Serialize grid state for persistence."""
        return {
            'symbol': self.config.symbol,
            'config': {
                'symbol': self.config.symbol,
                'upper_price': self.config.upper_price,
                'lower_price': self.config.lower_price,
                'num_grids': self.config.num_grids,
                'investment_per_grid': self.config.investment_per_grid
            },
            'levels': [level.to_dict() for level in self.levels],
            'is_active': self.is_active,
            'total_profit': self.total_profit,
            'completed_trades': self.completed_trades,
            'total_buys': self.total_buys,
            'total_sells': self.total_sells,
            'avg_buy_price': self.avg_buy_price,
            'avg_sell_price': self.avg_sell_price,
            'created_at': self.created_at.isoformat(),
            'last_trade_at': self.last_trade_at.isoformat() if self.last_trade_at else None,
            'total_fills': self.total_fills,
            'completed_cycles': self.completed_cycles,
            'fee_origin_paired_count': self.fee_origin_paired_count,
            'fee_origin_estimated_count': self.fee_origin_estimated_count
        }


class GridTradingStrategy:
    """
    Grid Trading Strategy Implementation.

    Creates a grid of buy/sell orders within a price range and automatically
    executes trades as price oscillates, capturing profit from each move.
    """

    def __init__(self):
        self.grids: Dict[str, GridState] = {}  # symbol -> GridState

        # NEW: Pending order mappings (order_id -> PendingOrder)
        self.pending_orders: Dict[str, PendingOrder] = {}

        # NEW: Applied order IDs for idempotency (order_id -> applied_at ISO)
        self.applied_order_ids: Dict[str, str] = {}

        # NEW: Unmatched fills for diagnostics (order_id -> fill details)
        self.unmatched_fills: Dict[str, dict] = {}

        # NEW: Open limit order tracking (level_key -> OpenLimitOrder)
        # Key format: "{symbol}:{side}:{level_id}"
        self.open_limit_orders: Dict[str, OpenLimitOrder] = {}

        logger.info("GridTradingStrategy initialized")

    def save_state(self) -> None:
        """Save all grid states to file AND database for persistence across restarts."""
        try:
            data = {
                'date': datetime.now().strftime('%Y-%m-%d'),
                'saved_at': datetime.now().isoformat(),
                'grids': {symbol: state.to_dict() for symbol, state in self.grids.items()},
                # NEW: Persist pending/applied/unmatched
                'pending_orders': {oid: p.to_dict() for oid, p in self.pending_orders.items()},
                'applied_order_ids': self.applied_order_ids,
                'unmatched_fills': self.unmatched_fills,
                # NEW: Persist open limit orders
                'open_limit_orders': {key: o.to_dict() for key, o in self.open_limit_orders.items()}
            }
            with open(GRID_STATE_FILE, 'w') as f:
                json.dump(data, f, indent=2)
            logger.debug(f"Grid state saved for {len(self.grids)} symbols")

            # Also save to DB for long-term persistence (survives date changes)
            self.save_to_db()
        except Exception as e:
            logger.error(f"Failed to save grid state: {e}")

    def load_state(self) -> bool:
        """
        Load grid states from file. Returns True if valid state was loaded.

        Only loads state from the same day to avoid stale grids after overnight price changes.
        """
        try:
            if not os.path.exists(GRID_STATE_FILE):
                logger.info("No grid state file found, will create fresh grids")
                return False

            with open(GRID_STATE_FILE, 'r') as f:
                data = json.load(f)

            # Only load if from today (grids may be stale after overnight)
            saved_date = data.get('date')
            today = datetime.now().strftime('%Y-%m-%d')
            if saved_date != today:
                logger.info(f"Grid state from {saved_date} is stale, will create fresh grids")
                return False

            # Restore each grid
            for symbol, grid_data in data.get('grids', {}).items():
                config = GridConfig(
                    symbol=grid_data['config']['symbol'],
                    upper_price=grid_data['config']['upper_price'],
                    lower_price=grid_data['config']['lower_price'],
                    num_grids=grid_data['config']['num_grids'],
                    investment_per_grid=grid_data['config']['investment_per_grid']
                )

                levels = [GridLevel.from_dict(l) for l in grid_data['levels']]

                state = GridState(config=config, levels=levels)
                state.is_active = grid_data['is_active']
                state.total_profit = grid_data['total_profit']
                state.completed_trades = grid_data['completed_trades']
                state.total_buys = grid_data['total_buys']
                state.total_sells = grid_data['total_sells']
                state.avg_buy_price = grid_data['avg_buy_price']
                state.avg_sell_price = grid_data['avg_sell_price']
                state.created_at = datetime.fromisoformat(grid_data['created_at'])
                if grid_data.get('last_trade_at'):
                    state.last_trade_at = datetime.fromisoformat(grid_data['last_trade_at'])

                # NEW: Load new fields with backward-compat defaults
                state.total_fills = grid_data.get('total_fills', state.completed_trades)
                state.completed_cycles = grid_data.get('completed_cycles', 0)
                state.fee_origin_paired_count = grid_data.get('fee_origin_paired_count', 0)
                state.fee_origin_estimated_count = grid_data.get('fee_origin_estimated_count', 0)

                self.grids[symbol] = state

                filled_count = sum(1 for l in levels if l.is_filled)
                logger.info(f"Restored grid for {symbol}: {filled_count}/{len(levels)} levels filled")

            # NEW: Restore pending orders
            for oid, pending_data in data.get('pending_orders', {}).items():
                self.pending_orders[oid] = PendingOrder.from_dict(pending_data)
            if self.pending_orders:
                logger.info(f"Restored {len(self.pending_orders)} pending orders")

            # NEW: Restore applied order IDs (with pruning)
            self.applied_order_ids = data.get('applied_order_ids', {})
            # Prune to last 5000 entries
            if len(self.applied_order_ids) > 5000:
                sorted_ids = sorted(self.applied_order_ids.items(), key=lambda x: x[1], reverse=True)
                self.applied_order_ids = dict(sorted_ids[:5000])
                logger.info(f"Pruned applied_order_ids to 5000 entries")

            # NEW: Restore unmatched fills (with pruning)
            self.unmatched_fills = data.get('unmatched_fills', {})
            # Prune to last 500 entries
            if len(self.unmatched_fills) > 500:
                sorted_fills = sorted(
                    self.unmatched_fills.items(),
                    key=lambda x: x[1].get('recorded_at', ''),
                    reverse=True
                )
                self.unmatched_fills = dict(sorted_fills[:500])

            # NEW: Restore open limit orders
            for key, order_data in data.get('open_limit_orders', {}).items():
                self.open_limit_orders[key] = OpenLimitOrder.from_dict(order_data)
            if self.open_limit_orders:
                logger.info(f"Restored {len(self.open_limit_orders)} open limit orders")

            # NEW: Cleanup stale pending orders (>24h)
            self._cleanup_stale_pendings()

            logger.info(f"Grid state restored for {len(self.grids)} symbols")
            return True

        except Exception as e:
            logger.error(f"Failed to load grid state: {e}")
            return False

    def _cleanup_stale_pendings(self, max_age_hours: int = 72) -> None:
        """
        Remove pending mappings older than max_age_hours.

        Conservative approach: 72h default (3 days) to avoid dropping mappings
        for orders that might still be pending on Alpaca. Market orders should
        fill immediately, but we keep a long window for safety.

        NOTE: For stricter compliance, the bot layer could verify against Alpaca
        before calling this method and pass confirmed-dead order IDs.
        """
        now = datetime.now()
        stale = []
        for order_id, pending in self.pending_orders.items():
            try:
                created = datetime.fromisoformat(pending.created_at)
                age_hours = (now - created).total_seconds() / 3600
                if age_hours > max_age_hours:
                    stale.append((order_id, pending, age_hours))
            except (ValueError, TypeError):
                pass  # Skip malformed entries

        for order_id, pending, age_hours in stale:
            logger.warning(
                f"[GRID] Removing stale pending order {order_id[:8]} "
                f"(age: {age_hours:.1f}h, symbol: {pending.symbol}, side: {pending.side}). "
                f"If this order filled on Alpaca, the fill will be recovered via reconciliation."
            )
            # Move to unmatched_fills for audit trail before deleting
            self.unmatched_fills[order_id] = {
                'symbol': pending.symbol,
                'side': pending.side,
                'price': pending.intended_level_price,
                'reason': f'stale_pending_removed_after_{age_hours:.1f}h',
                'recorded_at': now.isoformat()
            }
            del self.pending_orders[order_id]

        if stale:
            logger.warning(f"[GRID] Cleaned up {len(stale)} stale pending orders (>72h old)")

    def save_to_db(self) -> None:
        """Save grid state to database for persistence across restarts."""
        try:
            from src.database.db import save_grid_state

            state_dict = {
                'saved_at': datetime.now().isoformat(),
                'grids': {symbol: state.to_dict() for symbol, state in self.grids.items()},
                'pending_orders': {oid: p.to_dict() for oid, p in self.pending_orders.items()},
                'applied_order_ids': self.applied_order_ids,
                'unmatched_fills': self.unmatched_fills,
                'open_limit_orders': {key: o.to_dict() for key, o in self.open_limit_orders.items()}
            }
            save_grid_state(state_dict)
            logger.debug(f"Grid state saved to DB for {len(self.grids)} symbols")
        except Exception as e:
            logger.error(f"Failed to save grid state to DB: {e}")

    def load_from_db(self) -> bool:
        """
        Load grid state from database.

        Unlike load_state() which only loads same-day state from /tmp,
        this loads from DB regardless of date (grids persist across restarts).

        Returns:
            True if state was loaded successfully
        """
        try:
            from src.database.db import get_latest_grid_state

            data = get_latest_grid_state()
            if not data:
                logger.info("No grid state in database, will create fresh grids")
                return False

            # Restore grids
            for symbol, grid_data in data.get('grids', {}).items():
                config = GridConfig(
                    symbol=grid_data['config']['symbol'],
                    upper_price=grid_data['config']['upper_price'],
                    lower_price=grid_data['config']['lower_price'],
                    num_grids=grid_data['config']['num_grids'],
                    investment_per_grid=grid_data['config']['investment_per_grid']
                )

                levels = [GridLevel.from_dict(l) for l in grid_data['levels']]
                state = GridState(config=config, levels=levels)
                state.is_active = grid_data.get('is_active', True)
                state.total_profit = grid_data.get('total_profit', 0.0)
                state.completed_trades = grid_data.get('completed_trades', 0)
                state.total_buys = grid_data.get('total_buys', 0)
                state.total_sells = grid_data.get('total_sells', 0)
                state.avg_buy_price = grid_data.get('avg_buy_price', 0.0)
                state.avg_sell_price = grid_data.get('avg_sell_price', 0.0)
                state.total_fills = grid_data.get('total_fills', 0)
                state.completed_cycles = grid_data.get('completed_cycles', 0)
                state.fee_origin_paired_count = grid_data.get('fee_origin_paired_count', 0)
                state.fee_origin_estimated_count = grid_data.get('fee_origin_estimated_count', 0)

                # Restore timestamps
                if grid_data.get('created_at'):
                    state.created_at = datetime.fromisoformat(grid_data['created_at'])
                if grid_data.get('last_trade_at'):
                    state.last_trade_at = datetime.fromisoformat(grid_data['last_trade_at'])

                self.grids[symbol] = state
                logger.info(f"Restored grid for {symbol}: {len(levels)} levels, profit=${state.total_profit:.2f}")

            # Restore pending orders
            for oid, pending_data in data.get('pending_orders', {}).items():
                self.pending_orders[oid] = PendingOrder.from_dict(pending_data)

            # Restore applied order IDs
            self.applied_order_ids = data.get('applied_order_ids', {})

            # Restore unmatched fills
            self.unmatched_fills = data.get('unmatched_fills', {})

            # Restore open limit orders
            for key, order_data in data.get('open_limit_orders', {}).items():
                self.open_limit_orders[key] = OpenLimitOrder.from_dict(order_data)

            # Cleanup stale pending orders
            self._cleanup_stale_pendings()

            logger.info(f"Loaded grid state from DB: {len(self.grids)} grids, "
                       f"{len(self.pending_orders)} pending, {len(self.open_limit_orders)} open limits")
            return True

        except Exception as e:
            logger.error(f"Failed to load grid state from DB: {e}")
            return False

    def register_pending_order(
        self,
        order_id: str,
        symbol: str,
        side: str,
        intended_level_price: float,
        intended_level_id: Optional[str] = None,
        source: str = "grid",
        client_order_id: Optional[str] = None
    ) -> bool:
        """
        Register order->level mapping immediately after submission.

        Must be called BEFORE verify_order_fill() to enable deterministic matching.

        Args:
            order_id: Alpaca order ID
            symbol: Trading symbol
            side: "buy" or "sell"
            intended_level_price: The grid level price this order was for
            intended_level_id: Optional stable level ID for best matching
            source: "grid" | "windfall" | "stop_loss"
            client_order_id: Optional Alpaca client order ID

        Returns:
            True if registered successfully
        """
        symbol = normalize_symbol(symbol)
        side = normalize_side(side)

        if order_id in self.pending_orders:
            logger.warning(f"[GRID] Order {order_id[:8]} already registered as pending")
            return False

        pending = PendingOrder(
            order_id=order_id,
            symbol=symbol,
            side=side,
            intended_level_price=intended_level_price,
            intended_level_id=intended_level_id,
            created_at=datetime.now().isoformat(),
            source=source,
            client_order_id=client_order_id
        )
        self.pending_orders[order_id] = pending
        logger.info(f"[GRID] Registered pending {side} order {order_id[:8]} for {symbol} @ ${intended_level_price:.2f}")
        self.save_state()  # Persist immediately for crash recovery
        return True

    # =========================================================================
    # Open Limit Order Management (for duplicate prevention & stale cancellation)
    # =========================================================================

    def register_open_limit_order(
        self,
        order_id: str,
        symbol: str,
        side: str,
        level_id: str,
        level_price: float,
        limit_price: float,
        qty: float,
        source: str = "grid"
    ) -> bool:
        """
        Register a newly submitted limit order for tracking.

        Returns False if an order already exists for this level.
        """
        symbol = normalize_symbol(symbol)
        side = normalize_side(side)
        key = OpenLimitOrder.make_key(symbol, side, level_id)

        if key in self.open_limit_orders:
            existing = self.open_limit_orders[key]
            logger.warning(
                f"[GRID] Duplicate limit order for {key}: "
                f"existing={existing.order_id[:8]}, new={order_id[:8]}"
            )
            return False

        order = OpenLimitOrder(
            order_id=order_id,
            symbol=symbol,
            side=side,
            level_id=level_id,
            level_price=level_price,
            limit_price=limit_price,
            qty=qty,
            source=source
        )
        self.open_limit_orders[key] = order
        logger.info(f"[GRID] Registered open limit {side} @ ${limit_price:.2f} ({order_id[:8]})")
        self.save_state()
        return True

    def has_open_order_for_level(self, symbol: str, side: str, level_id: str) -> bool:
        """Check if there's already an open order for this level."""
        key = OpenLimitOrder.make_key(
            normalize_symbol(symbol),
            normalize_side(side),
            level_id
        )
        return key in self.open_limit_orders

    def get_open_order_for_level(self, symbol: str, side: str, level_id: str) -> Optional[OpenLimitOrder]:
        """Get open order for a specific level, if any."""
        key = OpenLimitOrder.make_key(
            normalize_symbol(symbol),
            normalize_side(side),
            level_id
        )
        return self.open_limit_orders.get(key)

    def remove_open_limit_order(self, order_id: str) -> Optional[OpenLimitOrder]:
        """Remove an open order by ID (when filled or cancelled)."""
        for key, order in list(self.open_limit_orders.items()):
            if order.order_id == order_id:
                del self.open_limit_orders[key]
                logger.info(f"[GRID] Removed open limit order {order_id[:8]}")
                self.save_state()
                return order
        return None

    def remove_pending_order(self, order_id: str) -> Optional[PendingOrder]:
        """Remove a pending order by ID (cleanup on cancel/expire/reject)."""
        if order_id in self.pending_orders:
            removed = self.pending_orders.pop(order_id)
            logger.info(f"[GRID] Removed pending order {order_id[:8]}")
            self.save_state()
            return removed
        return None

    def get_stale_orders(self, max_age_minutes: int = 60) -> List[OpenLimitOrder]:
        """Get open limit orders older than max_age_minutes."""
        now = datetime.now()
        stale = []
        for order in self.open_limit_orders.values():
            try:
                created = datetime.fromisoformat(order.created_at)
                age_minutes = (now - created).total_seconds() / 60
                if age_minutes > max_age_minutes:
                    stale.append(order)
            except (ValueError, TypeError):
                pass
        return stale

    def _record_unmatched(
        self,
        order_id: str,
        symbol: str,
        side: str,
        price: float,
        reason: str
    ) -> None:
        """Record unmatched fill for diagnostics."""
        self.unmatched_fills[order_id] = {
            'symbol': symbol,
            'side': side,
            'price': price,
            'reason': reason,
            'recorded_at': datetime.now().isoformat()
        }
        logger.error(f"[GRID] Unmatched fill: {order_id[:8]} {side} {symbol} @ ${price:.2f} ({reason})")

    def apply_filled_order(
        self,
        symbol: str,
        side: str,
        fill_price: float,
        fill_qty: float,
        order_id: str,
        source: str = "reconcile"
    ) -> Optional[float]:
        """
        Single entry point for applying fills. Returns profit for sells.

        This is the deterministic, idempotent fill matching algorithm.

        Matching rules (strict order):
        0. Idempotency: Skip if order_id already applied
        1. Deterministic: Match by order_id from pending_orders (level_id first, then price)
        2. Fallback: Match by price-proximity WITH side validation

        Args:
            symbol: Trading symbol
            side: "buy" or "sell"
            fill_price: Actual fill price from Alpaca
            fill_qty: Fill quantity
            order_id: Alpaca order ID
            source: "grid" | "reconcile" | "stop_loss" | "windfall"

        Returns:
            Profit from this trade (if grid sell), else None
        """
        symbol = normalize_symbol(symbol)
        side = normalize_side(side)

        # Step 0: Idempotency guard
        if order_id in self.applied_order_ids:
            logger.debug(f"[GRID] Skip duplicate: order {order_id[:8]} already applied")
            return None

        if symbol not in self.grids:
            logger.error(f"[GRID] No grid for {symbol}")
            return None

        state = self.grids[symbol]
        matched_level = None
        match_method = None
        pending = self.pending_orders.get(order_id)

        # Step 1: Try pending mapping (deterministic match)
        if pending:
            # Validate symbol matches
            if pending.symbol != symbol:
                self._record_unmatched(order_id, symbol, side, fill_price, "symbol_mismatch")
                return None  # Keep pending, don't apply

            # Validate side matches (HARD requirement - don't apply on mismatch)
            if pending.side != side:
                self._record_unmatched(order_id, symbol, side, fill_price, "side_mismatch")
                return None  # Keep pending, don't apply

            # Match by level_id first (most stable)
            if pending.intended_level_id:
                for level in state.levels:
                    if level.level_id == pending.intended_level_id and not level.is_filled:
                        matched_level = level
                        match_method = "level_id"
                        break

            # Fallback to price within tolerance
            if not matched_level:
                tolerance = state.config.grid_spacing * 0.5
                for level in state.levels:
                    if (not level.is_filled and
                        level.side.value == side and
                        abs(level.price - pending.intended_level_price) < tolerance):
                        matched_level = level
                        match_method = "pending_price"
                        break

            # Use pending's source for cycle counting
            source = pending.source

        # Step 2: Fallback (no pending mapping) - side-gated + bounded
        if not matched_level:
            tolerance = state.config.grid_spacing * 0.5
            min_diff = float('inf')

            for level in state.levels:
                if not level.is_filled and level.side.value == side:
                    diff = abs(level.price - fill_price)
                    if diff < tolerance and diff < min_diff:
                        min_diff = diff
                        matched_level = level
                        match_method = "fallback"

            if matched_level:
                logger.warning(f"[GRID] Fallback match: {side} ${fill_price:.2f} -> level ${matched_level.price:.2f}")

        # No match found - record as unmatched but do NOT mark as applied
        if not matched_level:
            self._record_unmatched(order_id, symbol, side, fill_price, "no_level_match")
            return None

        # === Apply the fill ===
        matched_level.is_filled = True
        matched_level.order_id = order_id
        matched_level.filled_at = datetime.now()
        matched_level.quantity = fill_qty

        state.last_trade_at = datetime.now()

        # Update counters
        state.total_fills += 1
        state.completed_trades = state.total_fills  # Backward compat alias

        # Add to applied (idempotency) ONLY after successful apply
        self.applied_order_ids[order_id] = datetime.now().isoformat()

        # Remove pending mapping ONLY after successful apply
        if pending:
            del self.pending_orders[order_id]

        logger.info(f"[GRID] Applied {side} {symbol} @ ${fill_price:.2f} (method: {match_method})")

        # Handle buy/sell specific logic
        profit = None
        if side == "buy":
            state.total_buys += 1
            # Update average buy price
            if state.avg_buy_price == 0:
                state.avg_buy_price = fill_price
            else:
                state.avg_buy_price = (state.avg_buy_price + fill_price) / 2

            # Create corresponding sell level with origin buy metadata
            sell_price = fill_price + state.config.grid_spacing
            if sell_price <= state.config.upper_price:
                # Get buy fee type from pending order (for accurate sell-side profit calc)
                buy_fee_type = pending.fee_type if pending else "taker"
                self._add_sell_level(
                    state,
                    sell_price,
                    fill_qty,
                    origin_buy_price=fill_price,
                    origin_buy_fee_type=buy_fee_type
                )

        else:  # sell
            state.total_sells += 1
            # Update average sell price
            if state.avg_sell_price == 0:
                state.avg_sell_price = fill_price
            else:
                state.avg_sell_price = (state.avg_sell_price + fill_price) / 2

            # Only count grid sells as completed_cycles (not stop-loss/windfall)
            if source == "grid":
                state.completed_cycles += 1

            # Calculate profit using two-notional fee model
            # Get fee rates from config (with fallback defaults)
            maker_fee = getattr(config, 'MAKER_FEE_PCT', 0.0015)
            taker_fee = getattr(config, 'TAKER_FEE_PCT', 0.0025)

            # Get sell fee type from pending order (local var holds ref even after dict removal)
            sell_fee_type = pending.fee_type if pending else "taker"
            sell_fee_rate = maker_fee if sell_fee_type == "maker" else taker_fee

            # Get buy fee type/price from matched SELL level's origin fields
            # matched_level is the sell level that triggered this fill
            has_paired_origin = (
                hasattr(matched_level, 'origin_buy_price') and
                matched_level.origin_buy_price > 0
            )
            if has_paired_origin:
                buy_price = matched_level.origin_buy_price
                buy_fee_type = matched_level.origin_buy_fee_type
                state.fee_origin_paired_count += 1
            else:
                # Fallback: estimate buy price from grid spacing (initial grid sells, etc.)
                buy_price = fill_price - state.config.grid_spacing
                buy_fee_type = "taker"  # Conservative default for untracked buys
                state.fee_origin_estimated_count += 1

            buy_fee_rate = maker_fee if buy_fee_type == "maker" else taker_fee

            # Compute fees using BOTH notionals (buy leg + sell leg)
            sell_price = fill_price
            buy_fee_usd = buy_price * fill_qty * buy_fee_rate
            sell_fee_usd = sell_price * fill_qty * sell_fee_rate
            fees = buy_fee_usd + sell_fee_usd

            # Gross profit is price difference * quantity
            if has_paired_origin:
                gross_profit = (sell_price - buy_price) * fill_qty
            else:
                gross_profit = state.config.grid_spacing * fill_qty

            profit = gross_profit - fees
            cycle_fee_type = f"{buy_fee_type}+{sell_fee_type}"  # e.g., "maker+maker"

            if profit > 0:
                state.total_profit += profit
                logger.info(
                    f"[PROFIT] {symbol}: +${profit:.2f} "
                    f"(buy=${buy_price:.2f}, sell=${sell_price:.2f}, qty={fill_qty:.4f}, "
                    f"fees=${fees:.2f} [{cycle_fee_type}])"
                )
            else:
                logger.warning(f"[PROFIT] {symbol}: Negative profit ${profit:.2f} - possibly forced sell")
                profit = 0.0

            # Create corresponding buy level
            new_buy_price = fill_price - state.config.grid_spacing
            if new_buy_price >= state.config.lower_price:
                self._add_buy_level(state, new_buy_price, fill_qty)

        self.save_state()
        return profit if profit and profit > 0 else None

    def create_grid(self, config: GridConfig, current_price: float) -> GridState:
        """
        Create a new grid for a symbol.

        Args:
            config: Grid configuration
            current_price: Current market price

        Returns:
            GridState with initialized grid levels
        """
        logger.info(f"Creating grid for {config.symbol}")
        logger.info(f"  Range: ${config.lower_price:,.2f} - ${config.upper_price:,.2f}")
        logger.info(f"  Grids: {config.num_grids}, Spacing: ${config.grid_spacing:,.2f}")
        logger.info(f"  Profit per grid: {config.profit_per_grid_pct:.2f}%")

        # Create grid levels
        levels = []
        for i in range(config.num_grids + 1):
            level_price = config.lower_price + (i * config.grid_spacing)

            # Levels below current price are BUY, above are SELL
            side = GridOrderSide.BUY if level_price < current_price else GridOrderSide.SELL

            # Calculate quantity for this level
            quantity = config.investment_per_grid / level_price

            level = GridLevel(
                price=level_price,
                side=side,
                quantity=quantity
            )
            levels.append(level)

        state = GridState(
            config=config,
            levels=levels,
            is_active=True
        )

        self.grids[config.symbol] = state

        # Log grid structure
        buy_levels = [l for l in levels if l.side == GridOrderSide.BUY]
        sell_levels = [l for l in levels if l.side == GridOrderSide.SELL]
        logger.info(f"  Buy levels: {len(buy_levels)}, Sell levels: {len(sell_levels)}")

        return state

    def auto_configure_grid(
        self,
        symbol: str,
        current_price: float,
        atr: float,  # Average True Range for volatility
        total_investment: float = 1000.0,
        num_grids: int = 10
    ) -> GridConfig:
        """
        Automatically configure a grid based on current market conditions.

        Uses ATR (volatility) to set appropriate grid range.

        Args:
            symbol: Trading symbol
            current_price: Current market price
            atr: Average True Range (volatility indicator)
            total_investment: Total $ to deploy across grid
            num_grids: Number of grid levels

        Returns:
            Optimized GridConfig
        """
        # Use 3x ATR for grid range (captures typical daily moves)
        range_width = atr * 3

        # Set grid boundaries
        upper_price = current_price + range_width
        lower_price = current_price - range_width

        # Investment per grid level
        investment_per_grid = total_investment / num_grids

        config = GridConfig(
            symbol=symbol,
            upper_price=upper_price,
            lower_price=lower_price,
            num_grids=num_grids,
            investment_per_grid=investment_per_grid
        )

        logger.info(f"Auto-configured grid for {symbol}:")
        logger.info(f"  Current: ${current_price:,.2f}, ATR: ${atr:,.2f}")
        logger.info(f"  Range: ${lower_price:,.2f} - ${upper_price:,.2f}")
        logger.info(f"  Spacing: ${config.grid_spacing:,.2f} ({config.profit_per_grid_pct:.2f}%)")

        return config

    def evaluate_grid(
        self,
        symbol: str,
        current_price: float,
        current_position_qty: float = 0.0
    ) -> Dict:
        """
        Evaluate the grid and determine what actions to take.

        Args:
            symbol: Trading symbol
            current_price: Current market price
            current_position_qty: Current position quantity

        Returns:
            Dict with trading signals and grid state info
        """
        if symbol not in self.grids:
            return {
                "action": "NONE",
                "reason": "No grid configured for this symbol",
                "grid_active": False
            }

        state = self.grids[symbol]
        config = state.config

        # Check if price is outside grid range
        price_above_range = current_price > config.upper_price * (1 + config.rebalance_threshold)
        price_below_range = current_price < config.lower_price * (1 - config.rebalance_threshold)

        if price_above_range or price_below_range:
            # AUTO-REBALANCE: Automatically recenter grid around current price
            if config.auto_rebalance:
                old_lower = config.lower_price
                old_upper = config.upper_price

                # Rebalance the grid around current price
                self.rebalance_grid(symbol, current_price, preserve_positions=True)

                new_state = self.grids[symbol]
                new_config = new_state.config

                direction = "UP" if price_above_range else "DOWN"
                logger.info(f"AUTO-REBALANCED {symbol} grid {direction}")
                logger.info(f"  Old: ${old_lower:,.2f} - ${old_upper:,.2f}")
                logger.info(f"  New: ${new_config.lower_price:,.2f} - ${new_config.upper_price:,.2f}")

                return {
                    "action": f"REBALANCED_{direction}",
                    "reason": f"Auto-rebalanced grid around ${current_price:,.2f}",
                    "grid_active": True,
                    "old_range": {"lower": old_lower, "upper": old_upper},
                    "new_range": {"lower": new_config.lower_price, "upper": new_config.upper_price},
                    "auto_rebalanced": True
                }
            else:
                # Manual rebalance mode - just notify
                if price_above_range:
                    return {
                        "action": "REBALANCE_UP",
                        "reason": f"Price ${current_price:,.2f} above grid upper ${config.upper_price:,.2f}",
                        "grid_active": True,
                        "recommendation": "Consider moving grid up or taking profits"
                    }
                else:
                    return {
                        "action": "REBALANCE_DOWN",
                        "reason": f"Price ${current_price:,.2f} below grid lower ${config.lower_price:,.2f}",
                        "grid_active": True,
                        "recommendation": "Consider moving grid down or stopping loss"
                    }

        # Find triggered levels
        triggered_buys = []
        triggered_sells = []

        for level in state.levels:
            if level.is_filled:
                continue

            if level.side == GridOrderSide.BUY and current_price <= level.price:
                triggered_buys.append(level)
            elif level.side == GridOrderSide.SELL and current_price >= level.price:
                triggered_sells.append(level)

        # Determine action
        action = "HOLD"
        order_details = None

        if triggered_buys:
            # Buy at the highest triggered buy level
            best_buy = max(triggered_buys, key=lambda l: l.price)
            action = "BUY"
            order_details = {
                "price": best_buy.price,
                "quantity": best_buy.quantity,
                "grid_level": state.levels.index(best_buy),  # Keep for UI/logging
                "level_id": best_buy.level_id  # NEW: Stable identity for matching
            }
        elif triggered_sells and current_position_qty > 0:
            # Sell at the lowest triggered sell level
            best_sell = min(triggered_sells, key=lambda l: l.price)
            action = "SELL"
            order_details = {
                "price": best_sell.price,
                "quantity": min(best_sell.quantity, current_position_qty),
                "grid_level": state.levels.index(best_sell),  # Keep for UI/logging
                "level_id": best_sell.level_id  # NEW: Stable identity for matching
            }

        return {
            "action": action,
            "order_details": order_details,
            "grid_active": True,
            "current_price": current_price,
            "grid_range": f"${config.lower_price:,.2f} - ${config.upper_price:,.2f}",
            "pending_buys": len([l for l in state.levels if l.side == GridOrderSide.BUY and not l.is_filled]),
            "pending_sells": len([l for l in state.levels if l.side == GridOrderSide.SELL and not l.is_filled]),
            "completed_trades": state.completed_trades,
            "total_profit": state.total_profit
        }

    def get_desired_limit_orders(
        self,
        symbol: str,
        current_price: float,
        position_qty: float,
        maker_buffer_bps: int = 5
    ) -> Dict[str, Any]:
        """
        Identify maker-safe grid levels for resting limit orders.

        This is the PRIMARY method for limit order mode. Unlike evaluate_grid()
        which triggers AFTER price crosses a level (suitable for market orders),
        this method identifies levels where resting limit orders should be placed
        AHEAD of price movement.

        The key difference:
        - evaluate_grid(): "Price crossed $95,000, trigger BUY at $95,000"
          → But placing limit at $95,000 when market is below would be taker!
        - get_desired_limit_orders(): "Market is $94,000, place resting BUY at $93,950"
          → Order rests on book, fills when price drops to it (maker)

        Args:
            symbol: Trading symbol (e.g., "BTC/USD")
            current_price: Current market price
            position_qty: Current position quantity (needed for sell eligibility)
            maker_buffer_bps: Buffer in basis points to ensure maker execution (default 5 = 0.05%)

        Returns:
            Dict with:
                - 'desired_buy': Dict with {price, quantity, level_id, grid_level} or None
                - 'desired_sell': Dict with {price, quantity, level_id, grid_level} or None
                - 'buy_candidates': int - count of eligible buy levels
                - 'sell_candidates': int - count of eligible sell levels
                - 'no_eligible_levels': bool - True if no maker-safe levels exist (overshoot)
                - 'reason': str - explanation of selection
        """
        symbol = normalize_symbol(symbol)

        result = {
            'desired_buy': None,
            'desired_sell': None,
            'buy_candidates': 0,
            'sell_candidates': 0,
            'no_eligible_levels': False,
            'reason': ''
        }

        if symbol not in self.grids:
            result['reason'] = f"No grid configured for {symbol}"
            result['no_eligible_levels'] = True
            return result

        state = self.grids[symbol]
        config = state.config
        buffer = maker_buffer_bps / 10000

        # Calculate price thresholds for maker safety
        # BUY must be below this price to stay on maker side of book
        max_buy_price = current_price * (1 - buffer)
        # SELL must be above this price to stay on maker side of book
        min_sell_price = current_price * (1 + buffer)

        # === FIND BUY CANDIDATES ===
        # Unfilled BUY levels that are sufficiently below market
        buy_candidates = []
        for level in state.levels:
            if (level.side == GridOrderSide.BUY
                and not level.is_filled
                and level.price <= max_buy_price
                and not self.has_open_order_for_level(symbol, 'buy', level.level_id)):
                buy_candidates.append(level)

        result['buy_candidates'] = len(buy_candidates)

        # Pick the HIGHEST buy candidate (closest to market = fills first when price drops)
        if buy_candidates:
            best_buy = max(buy_candidates, key=lambda l: l.price)
            result['desired_buy'] = {
                'price': best_buy.price,
                'quantity': best_buy.quantity,
                'level_id': best_buy.level_id,
                'grid_level': state.levels.index(best_buy)
            }

        # === FIND SELL CANDIDATES ===
        # Unfilled SELL levels that are sufficiently above market
        # AND we have position to sell
        sell_candidates = []
        if position_qty > 0:
            for level in state.levels:
                if (level.side == GridOrderSide.SELL
                    and not level.is_filled
                    and level.price >= min_sell_price
                    and not self.has_open_order_for_level(symbol, 'sell', level.level_id)):
                    sell_candidates.append(level)

        result['sell_candidates'] = len(sell_candidates)

        # Pick the LOWEST sell candidate (closest to market = fills first when price rises)
        if sell_candidates:
            best_sell = min(sell_candidates, key=lambda l: l.price)
            # Cap quantity to available position
            sell_qty = min(best_sell.quantity, position_qty)
            result['desired_sell'] = {
                'price': best_sell.price,
                'quantity': sell_qty,
                'level_id': best_sell.level_id,
                'grid_level': state.levels.index(best_sell)
            }

        # Determine if we're in an overshoot situation (no eligible levels)
        unfilled_buys = [l for l in state.levels if l.side == GridOrderSide.BUY and not l.is_filled]
        unfilled_sells = [l for l in state.levels if l.side == GridOrderSide.SELL and not l.is_filled]

        # Overshoot = have unfilled levels but none are maker-safe
        if unfilled_buys and result['buy_candidates'] == 0:
            result['no_eligible_levels'] = True
        if position_qty > 0 and unfilled_sells and result['sell_candidates'] == 0:
            result['no_eligible_levels'] = True

        # Build reason string for logging
        reasons = []
        if result['desired_buy']:
            reasons.append(f"BUY @ ${result['desired_buy']['price']:,.2f}")
        elif result['buy_candidates'] == 0 and unfilled_buys:
            reasons.append(f"No eligible BUY ({len(unfilled_buys)} levels above max ${max_buy_price:,.2f})")
        elif not unfilled_buys:
            reasons.append("No unfilled BUY levels")
        else:
            reasons.append(f"BUY: {result['buy_candidates']} candidates")

        if result['desired_sell']:
            reasons.append(f"SELL @ ${result['desired_sell']['price']:,.2f}")
        elif position_qty <= 0:
            reasons.append("No SELL (no position)")
        elif result['sell_candidates'] == 0 and unfilled_sells:
            reasons.append(f"No eligible SELL ({len(unfilled_sells)} levels below min ${min_sell_price:,.2f})")
        elif not unfilled_sells:
            reasons.append("No unfilled SELL levels")
        else:
            reasons.append(f"SELL: {result['sell_candidates']} candidates")

        result['reason'] = "; ".join(reasons)

        return result

    def record_fill(
        self,
        symbol: str,
        side: str,  # "buy" or "sell"
        price: float,
        quantity: float,
        order_id: str,
        source: str = "grid"
    ) -> Optional[float]:
        """
        Record that a grid order was filled.

        LEGACY METHOD: Delegates to apply_filled_order() for deterministic matching.

        Args:
            symbol: Trading symbol
            side: "buy" or "sell"
            price: Fill price
            quantity: Fill quantity
            order_id: Order ID
            source: "grid", "stop_loss", or "windfall"

        Returns:
            Profit from this trade (if closing a grid cycle)
        """
        return self.apply_filled_order(symbol, side, price, quantity, order_id, source=source)

    def _add_sell_level(
        self,
        state: GridState,
        price: float,
        quantity: float,
        origin_buy_price: float = 0.0,
        origin_buy_fee_type: str = "taker"
    ):
        """
        Add a new sell level to the grid.

        Args:
            state: Grid state to modify
            price: Sell level price
            quantity: Order quantity
            origin_buy_price: Price of the buy that created this sell (for fee calc)
            origin_buy_fee_type: Fee type of the originating buy ("maker" or "taker")
        """
        level = GridLevel(
            price=price,
            side=GridOrderSide.SELL,
            quantity=quantity,
            origin_buy_price=origin_buy_price,
            origin_buy_fee_type=origin_buy_fee_type
        )
        state.levels.append(level)
        state.levels.sort(key=lambda l: l.price)

    def _add_buy_level(self, state: GridState, price: float, quantity: float):
        """Add a new buy level to the grid."""
        level = GridLevel(
            price=price,
            side=GridOrderSide.BUY,
            quantity=quantity
        )
        state.levels.append(level)
        state.levels.sort(key=lambda l: l.price)

    def rebalance_grid(
        self,
        symbol: str,
        new_center_price: float,
        preserve_positions: bool = True
    ) -> GridState:
        """
        Rebalance the grid around a new center price.

        Args:
            symbol: Trading symbol
            new_center_price: New center price for the grid
            preserve_positions: Whether to keep track of existing positions

        Returns:
            Updated GridState
        """
        if symbol not in self.grids:
            raise ValueError(f"No grid for {symbol}")

        old_state = self.grids[symbol]
        config = old_state.config

        # Calculate new range (same width, new center)
        range_width = config.upper_price - config.lower_price
        new_lower = new_center_price - (range_width / 2)
        new_upper = new_center_price + (range_width / 2)

        # Create new config
        new_config = GridConfig(
            symbol=symbol,
            upper_price=new_upper,
            lower_price=new_lower,
            num_grids=config.num_grids,
            investment_per_grid=config.investment_per_grid
        )

        # Create new grid
        new_state = self.create_grid(new_config, new_center_price)

        # Preserve stats if requested
        if preserve_positions:
            new_state.total_profit = old_state.total_profit
            new_state.completed_trades = old_state.completed_trades
            new_state.total_buys = old_state.total_buys
            new_state.total_sells = old_state.total_sells

        logger.info(f"Rebalanced grid for {symbol}")
        logger.info(f"  Old range: ${config.lower_price:,.2f} - ${config.upper_price:,.2f}")
        logger.info(f"  New range: ${new_lower:,.2f} - ${new_upper:,.2f}")

        return new_state

    def get_grid_summary(self, symbol: str) -> Dict:
        """Get a summary of the grid state for display/logging."""
        if symbol not in self.grids:
            return {"error": f"No grid for {symbol}"}

        state = self.grids[symbol]
        grid_config = state.config

        filled_levels = [l for l in state.levels if l.is_filled]
        unfilled_buys = [l for l in state.levels if not l.is_filled and l.side == GridOrderSide.BUY]
        unfilled_sells = [l for l in state.levels if not l.is_filled and l.side == GridOrderSide.SELL]

        # Calculate estimated profit from grid spacing if total_profit is 0 but we have sells
        estimated_profit = state.total_profit
        if state.total_profit == 0 and state.total_sells > 0:
            # Use grid spacing for more reliable profit estimation
            # Each sell should profit by approximately: grid_spacing * quantity - fees
            grid_spacing = grid_config.grid_spacing
            # Use config fee rates for round-trip estimate (buy + sell)
            # Default to maker+maker assumption for limit orders
            import config_ultra
            maker_fee = getattr(config_ultra, 'MAKER_FEE_PCT', 0.0015)
            fee_pct = maker_fee * 2  # Round-trip (buy maker + sell maker)

            # Estimate average quantity per trade
            avg_price = state.avg_sell_price if state.avg_sell_price > 0 else grid_config.upper_price
            avg_qty_per_trade = grid_config.investment_per_grid / avg_price

            # Gross profit from grid spacing
            gross_profit_per_sell = grid_spacing * avg_qty_per_trade
            # Estimated fees per trade
            fees_per_trade = avg_price * avg_qty_per_trade * fee_pct
            # Net profit per sell
            net_profit_per_sell = gross_profit_per_sell - fees_per_trade

            if net_profit_per_sell > 0:
                estimated_profit = net_profit_per_sell * state.total_sells
                logger.debug(f"Estimated profit for {symbol}: spacing=${grid_spacing:.2f}, qty={avg_qty_per_trade:.4f}, sells={state.total_sells} -> ${estimated_profit:.2f}")

        return {
            "symbol": symbol,
            "is_active": state.is_active,
            "range": {
                "lower": grid_config.lower_price,
                "upper": grid_config.upper_price,
                "spacing": grid_config.grid_spacing,
                "spacing_pct": grid_config.profit_per_grid_pct
            },
            "levels": {
                "total": len(state.levels),
                "filled": len(filled_levels),
                "pending_buys": len(unfilled_buys),
                "pending_sells": len(unfilled_sells)
            },
            "performance": {
                "total_profit": estimated_profit,
                # NOTE: completed_trades is backward-compat alias for total_fills
                "completed_trades": state.completed_trades,
                "completed_cycles": state.completed_cycles,
                "total_buys": state.total_buys,
                "total_sells": state.total_sells,
                "avg_buy_price": state.avg_buy_price,
                "avg_sell_price": state.avg_sell_price,
                # Fee model diagnostics
                "fee_origin_paired": state.fee_origin_paired_count,
                "fee_origin_estimated": state.fee_origin_estimated_count
            },
            "timestamps": {
                "created_at": state.created_at.isoformat() if state.created_at else None,
                "last_trade_at": state.last_trade_at.isoformat() if state.last_trade_at else None
            }
        }

    def calculate_optimal_grid_params(
        self,
        symbol: str,
        current_price: float,
        historical_volatility: float,  # Daily volatility as decimal (e.g., 0.03 = 3%)
        available_capital: float,
        risk_tolerance: str = "medium"  # "low", "medium", "high"
    ) -> GridConfig:
        """
        Calculate optimal grid parameters based on market conditions and risk preference.

        Args:
            symbol: Trading symbol
            current_price: Current market price
            historical_volatility: Historical daily volatility
            available_capital: Capital available for this grid
            risk_tolerance: Risk preference

        Returns:
            Optimized GridConfig
        """
        # Risk-based multipliers
        risk_multipliers = {
            "low": {"range": 1.5, "grids": 15},
            "medium": {"range": 2.0, "grids": 10},
            "high": {"range": 3.0, "grids": 8}
        }

        multipliers = risk_multipliers.get(risk_tolerance, risk_multipliers["medium"])

        # Calculate range based on volatility
        daily_range = current_price * historical_volatility
        grid_range = daily_range * multipliers["range"]

        upper_price = current_price + (grid_range / 2)
        lower_price = current_price - (grid_range / 2)
        num_grids = multipliers["grids"]

        investment_per_grid = available_capital / num_grids

        config = GridConfig(
            symbol=symbol,
            upper_price=upper_price,
            lower_price=lower_price,
            num_grids=num_grids,
            investment_per_grid=investment_per_grid
        )

        logger.info(f"Calculated optimal grid for {symbol} ({risk_tolerance} risk):")
        logger.info(f"  Volatility: {historical_volatility*100:.1f}%, Range: ${grid_range:,.2f}")
        logger.info(f"  Grid: ${lower_price:,.2f} - ${upper_price:,.2f}")
        logger.info(f"  Levels: {num_grids}, Investment/level: ${investment_per_grid:,.2f}")
        logger.info(f"  Expected profit/cycle: {config.profit_per_grid_pct:.2f}%")

        return config


# Pre-configured grid setups for our symbols
# Optimized for diversification: BTC + SOL + LTC + AVAX (low correlation portfolio)
DEFAULT_GRID_CONFIGS = {
    "BTC/USD": {
        "num_grids": 5,
        "range_pct": 0.08,  # 8% range - tighter for less volatile BTC
        "investment_ratio": 0.35  # 35% - highest volume, most reliable
    },
    "SOL/USD": {
        "num_grids": 5,
        "range_pct": 0.12,  # 12% range - SOL more volatile
        "investment_ratio": 0.30  # 30% - good volume
    },
    "LTC/USD": {
        "num_grids": 5,
        "range_pct": 0.10,  # 10% range - payment coin, moderate volatility
        "investment_ratio": 0.20  # 20% - lower volume, reduce exposure
    },
    "AVAX/USD": {
        "num_grids": 5,
        "range_pct": 0.15,  # 15% range - most volatile, lowest BTC correlation (0.738)
        "investment_ratio": 0.15  # 15% - lowest volume, smallest positions
    }
}


def create_default_grids(
    strategy: GridTradingStrategy,
    prices: Dict[str, float],
    total_capital: float
) -> Dict[str, GridState]:
    """
    Create default grid setups for all configured symbols.

    Args:
        strategy: GridTradingStrategy instance
        prices: Current prices for each symbol
        total_capital: Total capital available

    Returns:
        Dict of symbol -> GridState
    """
    grids = {}

    for symbol, config_template in DEFAULT_GRID_CONFIGS.items():
        if symbol not in prices:
            continue

        current_price = prices[symbol]
        range_pct = config_template["range_pct"]
        num_grids = config_template["num_grids"]
        capital = total_capital * config_template["investment_ratio"]

        config = GridConfig(
            symbol=symbol,
            upper_price=current_price * (1 + range_pct / 2),
            lower_price=current_price * (1 - range_pct / 2),
            num_grids=num_grids,
            investment_per_grid=capital / num_grids
        )

        grids[symbol] = strategy.create_grid(config, current_price)

    return grids
