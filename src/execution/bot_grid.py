"""
Grid Trading Bot for BLUEBIRD 4.0

This bot replaces the prediction-based trading with a Grid Trading strategy
that profits from sideways market volatility.

WHY GRID TRADING?
- Our AI model correctly predicts SIDEWAYS 90%+ of the time
- Previous prediction-based trading: 21% win rate (worse than random!)
- Grid trading THRIVES in sideways markets - no predictions needed
- Every price oscillation = profit

HOW IT WORKS:
1. Set price range based on current price and volatility (ATR)
2. Create grid levels (buy below current, sell above current)
3. When price drops to a buy level -> BUY
4. When price rises to a sell level -> SELL
5. Repeat = profit from every oscillation
"""

import asyncio
import logging
import os
import random
from datetime import datetime, timedelta
from decimal import Decimal, ROUND_DOWN
from typing import Dict, Any, Optional, List, Tuple
import numpy as np

import pandas as pd

import config_ultra as config
from config_ultra import UltraConfig
from src.execution.alpaca_client import AlpacaClient
from src.strategy.grid_trading import (
    GridTradingStrategy,
    GridConfig,
    GridState,
    DEFAULT_GRID_CONFIGS,
    create_default_grids,
    normalize_side,
    normalize_symbol
)
from src.strategy.regime_detector import RegimeDetector, MarketRegime
from src.strategy.feature_calculator import FeatureCalculator
from src.strategy.risk_overlay import RiskOverlay, RiskMode

from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

# Database for persistent storage
from src.database import db as database

logger = logging.getLogger("GridBot")


# =============================================================================
# Limit Order Helper Functions
# =============================================================================

def can_place_limit_order(grid_price: float, current_price: float, side: str, buffer_bps: int = 5) -> bool:
    """
    Guard against crossing the spread (becoming a taker).
    Limit price stays at grid_price - this just checks if it's safe to place.

    BUY: Only place if grid_price <= current_price * (1 - buffer)
         (grid level is sufficiently below market)
    SELL: Only place if grid_price >= current_price * (1 + buffer)
          (grid level is sufficiently above market)

    If guard fails, skip placing and wait for next bar.
    """
    buffer = buffer_bps / 10000
    if side.lower() == 'buy':
        return grid_price <= current_price * (1 - buffer)
    else:
        return grid_price >= current_price * (1 + buffer)


def round_limit_price(symbol: str, price: float, config) -> float:
    """Round price to valid tick size for symbol."""
    precision = getattr(config, 'SYMBOL_PRECISION', {})
    price_decimals, _ = precision.get(symbol, (2, 6))
    return round(price, price_decimals)


def round_qty(symbol: str, qty: float, config) -> float:
    """Round quantity to valid precision for symbol."""
    precision = getattr(config, 'SYMBOL_PRECISION', {})
    _, qty_decimals = precision.get(symbol, (2, 6))
    return round(qty, qty_decimals)


def floor_qty(symbol: str, qty: float, config) -> float:
    """Floor quantity to valid precision for symbol (never rounds up).

    Uses Decimal + ROUND_DOWN to avoid float precision artifacts.
    Used when capping to available inventory to ensure we never exceed it.
    """
    precision = getattr(config, 'SYMBOL_PRECISION', {})
    _, qty_decimals = precision.get(symbol, (2, 6))

    # Use Decimal to avoid float precision artifacts
    step = Decimal(f'1e-{qty_decimals}')
    floored = Decimal(str(qty)).quantize(step, rounding=ROUND_DOWN)
    # Ensure non-negative (edge case protection)
    return max(float(floored), 0.0)


# =============================================================================
# Grid Sizing Validation Constants
# =============================================================================

# Alpaca minimum notional for crypto orders
MIN_NOTIONAL = 10.0

# Minimum investment per grid level (with buffer for rounding/fees)
# Must exceed MIN_NOTIONAL to avoid "notional < $10" rejections
MIN_INVESTMENT_PER_GRID = 12.0  # $12 floor ensures ~$10+ after rounding

# If current investment_per_grid is below this fraction of expected,
# consider the grid "stale" and trigger rebuild
STALE_GRID_THRESHOLD = 0.25  # Rebuild if current < 25% of expected


class GridTradingBot:
    """
    Grid Trading Bot Implementation.

    Replaces prediction-based trading with a simpler, more profitable
    strategy that exploits sideways market volatility.

    NO AI PREDICTIONS NEEDED - just profits from price oscillations!

    RISK CONTROLS:
    - Stop-loss: Close all positions if price drops 10% below grid
    - Daily loss limit: Stop trading if daily loss exceeds 5%
    - Max drawdown: Stop trading if drawdown exceeds 10%
    """

    def __init__(self, config: UltraConfig, client: AlpacaClient):
        self.config = config
        self.client = client
        self.symbols = getattr(config, 'SYMBOLS', ['BTC/USD', 'ETH/USD'])

        # Grid strategy engine
        self.grid_strategy = GridTradingStrategy()

        # Historical bars for each symbol (for tracking only)
        self.bars: Dict[str, List[Any]] = {s: [] for s in self.symbols}

        # Track if grids are initialized
        self.grids_initialized = False

        # Trade tracking
        self.total_trades = 0
        self.total_profit = 0.0

        # === REGIME DETECTION (Smart Grid) ===
        # Detects market regime to PAUSE grid during strong trends
        self.regime_detector = RegimeDetector(config)
        self.regime_pause: Dict[str, bool] = {s: False for s in self.symbols}  # Per-symbol pause state
        self.current_regime: Dict[str, str] = {s: MarketRegime.UNKNOWN for s in self.symbols}
        self.regime_metrics: Dict[str, Dict] = {}  # Store latest regime metrics per symbol
        self.regime_adx_threshold = getattr(config, 'REGIME_ADX_THRESHOLD', 40)  # ADX threshold for "strong trend"
        logger.info(f"  Regime Detection: ENABLED (ADX threshold={self.regime_adx_threshold})")

        # === RISK MANAGEMENT STATE ===
        # Track peak equity for drawdown calculation
        self.peak_equity = 0.0
        self.starting_equity = 0.0  # Session starting equity (resets on restart)

        # === ALL-TIME PERFORMANCE TRACKING ===
        # These are PERMANENT values that persist across restarts
        self.alltime_equity_file = "/tmp/bluebird-alltime-equity.json"
        self.alltime_starting_equity = 0.0  # When trading first started (Nov 24)
        self.alltime_starting_date = None
        self.grid_starting_equity = 0.0  # When grid trading started (Dec 2)
        self.grid_starting_date = None
        self._load_alltime_equity()  # Load persistent all-time stats

        # Track daily P&L (with persistence for true daily tracking)
        self.daily_pnl = 0.0
        self.daily_start_equity = 0.0
        self.last_trading_day = None
        self.daily_equity_file = "/tmp/bluebird-daily-equity.json"
        self._load_daily_equity()  # Load persisted daily equity on startup

        # Circuit breaker flags (persist across restarts for safety)
        self.circuit_breaker_file = "/tmp/bluebird-circuit-breaker.json"
        self.daily_limit_hit = False
        self.max_drawdown_hit = False
        self.stop_loss_triggered: Dict[str, bool] = {}  # Per-symbol stop loss
        self._load_circuit_breaker_state()  # Load persisted state on startup

        # Risk thresholds from config
        self.stop_loss_pct = getattr(config, 'GRID_STOP_LOSS_PCT', 0.10)
        self.daily_loss_limit = getattr(config, 'DAILY_LOSS_LIMIT', 0.05)
        self.max_drawdown = getattr(config, 'MAX_DRAWDOWN', 0.10)

        # === RISK OVERLAY (Crash Protection State Machine) ===
        # Blocks buys and rebalance-down during crashes, gradual re-entry on recovery
        self.risk_overlay = RiskOverlay(config)
        logger.info(f"  Risk Overlay: {'ENABLED' if getattr(config, 'RISK_OVERLAY_ENABLED', True) else 'DISABLED'}")

        # === LIMIT ORDER SETTINGS (Maker Fee Optimization) ===
        self.use_limit_orders = getattr(config, 'GRID_USE_LIMIT_ORDERS', True)
        self.maker_buffer_bps = getattr(config, 'MAKER_BUFFER_BPS', 5)
        self.max_order_age_minutes = getattr(config, 'MAX_ORDER_AGE_MINUTES', 60)

        logger.info(f"GridTradingBot initialized for {len(self.symbols)} symbols")
        logger.info(f"  Risk controls: SL={self.stop_loss_pct:.0%}, Daily={self.daily_loss_limit:.0%}, DD={self.max_drawdown:.0%}")
        logger.info(f"  Limit orders: {'ENABLED' if self.use_limit_orders else 'DISABLED'}")
        if self.use_limit_orders:
            logger.info(f"    Maker buffer: {self.maker_buffer_bps} bps, Max age: {self.max_order_age_minutes} min")

        # Time filtering settings
        self.use_time_filter = getattr(config, 'USE_TIME_FILTER', True)
        self.optimal_hours = getattr(config, 'OPTIMAL_HOURS', [(13, 17), (1, 4), (7, 9)])
        self.avoid_hours = getattr(config, 'AVOID_HOURS', [(22, 24), (5, 7)])
        self.weekend_size_mult = getattr(config, 'WEEKEND_SIZE_MULT', 0.5)

        # Correlation monitoring
        self.price_history: Dict[str, List[float]] = {s: [] for s in self.symbols}
        self.correlation_window = 20  # Number of bars for correlation calculation
        self.high_correlation_threshold = 0.85  # Reduce exposure when correlation > this

        # === WINDFALL PROFIT-TAKING STATE ===
        # Captures profits when positions show significant unrealized gains
        self.windfall_log_file = "/tmp/bluebird-windfall-log.json"
        self.windfall_cooldowns: Dict[str, datetime] = {}  # Per-symbol cooldown
        self.windfall_stats = {
            "total_captures": 0,
            "total_profit": 0.0,
            "transactions": []
        }
        self._load_windfall_log()  # Load persistent windfall stats

        # Performance tracking
        self.expected_profit_per_trade: Dict[str, float] = {}
        self.actual_profits: List[float] = []
        self.trade_count_by_hour: Dict[int, int] = {h: 0 for h in range(24)}
        self.profit_by_hour: Dict[int, float] = {h: 0.0 for h in range(24)}

        # === ORDER CONFIRMATION TRACKING ===
        # Confirmed orders from Alpaca (verified fills)
        self.confirmed_orders: List[Dict] = []
        # Orders pending verification
        self.pending_verification: List[str] = []
        # Last reconciliation timestamp
        self.last_reconciliation: Optional[datetime] = None
        # Reconciliation status
        self.reconciliation_status: Dict[str, Any] = {
            'synced': True,
            'last_check': None,
            'matched': 0,
            'total': 0,
            'discrepancies': []
        }
        # Last order tracking health check result (for dashboard display)
        self._last_health_check: Dict[str, Any] = {
            'healthy': True,
            'alpaca_count': 0,
            'tracked_count': 0,
            'mismatch': 0,
            'checked_at': None
        }

        # === FAST FILL DETECTION STATE ===
        # Lightweight per-tick check for limit order fills (~10-15s detection)
        self._last_fast_fill_check: Optional[datetime] = None
        self._fast_fill_backoff_until: Optional[datetime] = None
        self.fast_fill_stats: Dict[str, Any] = {
            'last_check_at': None,
            'open_tracked_count': 0,
            'disappeared_count': 0,
            'orders_checked_count': 0,
            'filled_applied_count': 0,
            'partial_applied_count': 0,
            'terminal_removed_count': 0,
            'unknown_status_count': 0,
            'errors_count': 0,
            'last_error_at': None,
            'last_result': ''
        }

        logger.info(f"  Time filter: {'ENABLED' if self.use_time_filter else 'DISABLED'}")
        logger.info(f"  Optimal hours (UTC): {self.optimal_hours}")

        # NOTE: Alpaca API calls moved to run_grid_bot() async init
        # to avoid blocking the event loop during constructor.
        # - _load_confirmed_orders_from_alpaca()
        # - _run_reconciliation()

    def _load_confirmed_orders_from_alpaca(self) -> None:
        """Load recent filled orders from Alpaca on startup."""
        try:
            orders = self.client.get_order_history(days=7, status='closed')
            # Status can be 'filled', 'OrderStatus.FILLED', etc.
            filled_orders = [o for o in orders if 'filled' in str(o.get('status', '')).lower()]

            for order in filled_orders:
                # Convert to our confirmed order format
                # Normalize symbol: BTCUSD -> BTC/USD, BTC/USD stays as is
                raw_symbol = order.get('symbol', '')
                if '/' not in raw_symbol:
                    symbol = raw_symbol.replace('USD', '/USD')
                else:
                    symbol = raw_symbol
                confirmed = {
                    'order_id': order.get('id'),
                    'symbol': symbol,
                    'side': order.get('side'),
                    'filled_qty': order.get('filled_qty') or order.get('qty', 0),
                    'filled_avg_price': order.get('filled_avg_price') or order.get('price', 0),
                    'filled_at': order.get('filled_at'),
                    'status': 'filled',
                    'confirmed': True,
                    'recorded_at': order.get('filled_at')
                }
                self.confirmed_orders.append(confirmed)

            # Sort by filled_at (most recent first)
            self.confirmed_orders.sort(key=lambda x: x.get('filled_at', ''), reverse=True)

            # Keep only last 500
            if len(self.confirmed_orders) > 500:
                self.confirmed_orders = self.confirmed_orders[:500]

            logger.info(f"Loaded {len(self.confirmed_orders)} confirmed orders from Alpaca")

        except Exception as e:
            logger.warning(f"Could not load orders from Alpaca: {e}")

    def _run_reconciliation(self) -> None:
        """
        Run reconciliation between database and Alpaca.
        Called on startup and periodically (every 5 minutes).

        Also applies any filled orders to grid state (for verify-timeout recovery).
        """
        try:
            # Get recent orders from Alpaca
            alpaca_orders = self.client.get_order_history(days=7, status='closed')

            def _normalize_order_status(raw_status: str) -> str:
                """
                Normalize Alpaca status strings like:
                - "filled"
                - "OrderStatus.FILLED"
                - "orderstatus.partially_filled"
                """
                s = (raw_status or "").strip().lower()
                # Common prefixes
                s = s.replace("orderstatus.", "")
                s = s.replace("querystatus.", "")
                return s

            filled_orders = []
            for o in alpaca_orders:
                status_norm = _normalize_order_status(str(o.get('status', '')))
                # IMPORTANT: do not treat "partially_filled" as "filled"
                if status_norm == 'filled':
                    filled_orders.append(o)

            # NEW: Clean up open limit order tracking for ALL closed orders
            # (filled, canceled, expired, rejected)
            if self.use_limit_orders:
                for order in alpaca_orders:
                    order_id = str(order.get('id', ''))
                    status_norm = _normalize_order_status(str(order.get('status', '')))
                    # Only remove tracking for terminal statuses; keep partials tracked.
                    if order_id and status_norm in {'filled', 'canceled', 'cancelled', 'expired', 'rejected'}:
                        # Clean up BOTH tracking dicts
                        removed_open = self.grid_strategy.remove_open_limit_order(order_id)
                        if removed_open:
                            logger.info(f"[RECONCILE] Removed {status_norm} order from open tracking: {order_id[:8]}")
                        # Only remove pending for non-fills (fills will be applied below)
                        if status_norm != 'filled':
                            removed_pending = self.grid_strategy.remove_pending_order(order_id)
                            if removed_pending:
                                logger.info(f"[RECONCILE] Removed {status_norm} order from pending: {order_id[:8]}")

            # Run reconciliation with database
            results = database.reconcile_with_alpaca(filled_orders)

            # Auto-sync any missing orders
            if results['missing_in_db']:
                synced = database.sync_missing_orders(results['missing_in_db'])
                logger.info(f"[RECONCILE] Auto-synced {synced} missing orders to database")

            # NEW: Apply filled orders to grid state (idempotent - safe to replay)
            # This recovers fills that timed out during verify_order_fill()
            grid_applied = 0
            for order in filled_orders:
                try:
                    # Alpaca uses 'id' not 'order_id'
                    order_id = str(order.get('id', ''))
                    if not order_id:
                        continue

                    # Normalize symbol and side from Alpaca format
                    raw_symbol = str(order.get('symbol', ''))
                    raw_side = order.get('side', '')
                    if not raw_symbol or not raw_side:
                        continue

                    symbol = normalize_symbol(raw_symbol)
                    side = normalize_side(raw_side)
                    fill_price = float(order.get('filled_avg_price', 0) or 0)
                    fill_qty = float(order.get('filled_qty', 0) or 0)

                    if fill_price > 0 and fill_qty > 0 and symbol in self.grid_strategy.grids:
                        # Check if already applied BEFORE calling (to count correctly)
                        was_already_applied = order_id in self.grid_strategy.applied_order_ids

                        # apply_filled_order is idempotent - checks applied_order_ids
                        self.grid_strategy.apply_filled_order(
                            symbol, side, fill_price, fill_qty, order_id,
                            source="reconcile"
                        )

                        # Count if newly applied (not already in applied_order_ids before)
                        if not was_already_applied and order_id in self.grid_strategy.applied_order_ids:
                            grid_applied += 1

                except Exception as e:
                    logger.debug(f"[RECONCILE] Could not apply order to grid: {e}")

            if grid_applied > 0:
                logger.info(f"[RECONCILE] Applied {grid_applied} fills to grid state")

            # Update status
            self.reconciliation_status = {
                'synced': results['synced'] or len(results['missing_in_db']) == 0,
                'last_check': datetime.now().isoformat(),
                'matched': results['matched'],
                'total_alpaca': results['total_alpaca'],
                'total_db': results['total_db'],
                'discrepancies': results['mismatched'][:5],  # Keep first 5
                'grid_fills_applied': grid_applied  # NEW
            }

            if results['synced']:
                logger.info(f"[RECONCILE] ✓ Database synced with Alpaca: {results['matched']} orders matched")
            else:
                logger.warning(f"[RECONCILE] ⚠ Discrepancies found: {len(results['missing_in_db'])} missing, {len(results['mismatched'])} mismatched")

            self.last_reconciliation = datetime.now()

        except Exception as e:
            logger.warning(f"Reconciliation failed: {e}")
            self.reconciliation_status['synced'] = False
            self.reconciliation_status['last_check'] = datetime.now().isoformat()

    def check_periodic_reconciliation(self) -> None:
        """Check if it's time to run periodic reconciliation (every 5 minutes)."""
        if self.last_reconciliation is None:
            self._run_reconciliation()
            # Also cancel stale limit orders on first reconciliation
            if self.use_limit_orders:
                self.cancel_stale_orders()
        elif (datetime.now() - self.last_reconciliation).seconds >= 300:  # 5 minutes
            self._run_reconciliation()
            # Also cancel stale limit orders periodically
            if self.use_limit_orders:
                self.cancel_stale_orders()

    def cancel_stale_orders(self) -> int:
        """
        Cancel orders older than MAX_ORDER_AGE_MINUTES.

        Returns number of orders cancelled.
        """
        if not self.use_limit_orders:
            return 0

        stale_orders = self.grid_strategy.get_stale_orders(self.max_order_age_minutes)
        cancelled = 0

        for order in stale_orders:
            # 1. Cancel on Alpaca
            if self.client.cancel_order(order.order_id):
                # 2. Remove from open_limit_orders
                self.grid_strategy.remove_open_limit_order(order.order_id)
                # 3. Remove from pending_orders (don't add to applied_order_ids - it wasn't filled)
                self.grid_strategy.remove_pending_order(order.order_id)
                cancelled += 1
                logger.info(f"[STALE] Cancelled {order.side} {order.symbol} @ ${order.limit_price:.2f}")

        if cancelled > 0:
            logger.info(f"[STALE] Cancelled {cancelled} stale limit orders (>{self.max_order_age_minutes} min old)")

        return cancelled

    async def check_limit_order_fills_fast(
        self,
        client: 'AlpacaClient',
        max_checks_per_cycle: int = 5,
        min_interval_seconds: float = 10.0,
        min_order_age_seconds: float = 10.0
    ) -> Dict[str, Any]:
        """
        Lightweight per-tick check for limit order fills.

        Compares tracked order IDs against Alpaca's open orders to detect
        fills/cancellations quickly without the overhead of full reconciliation.

        Critical implementation notes (from review):
        1. Call ONLY apply_filled_order(), NOT record_fill() (avoid double application)
        2. Grace period uses tracked created_at, not wall-clock
        3. Terminal + partial: apply partial, ALWAYS remove open tracking
        4. Normalize symbol/side BEFORE matching
        5. Terminal status set: filled, canceled, cancelled, expired, rejected, suspended

        Args:
            client: AlpacaClient instance for API calls
            max_checks_per_cycle: Maximum individual orders to verify per call (rate limiting)
            min_interval_seconds: Minimum time between checks (prevents over-polling)
            min_order_age_seconds: Grace period for new orders (eventual consistency)

        Returns:
            Dict with fills_detected, cancels_detected, orders_checked, etc.
        """
        # run_blocking is defined in this module (line ~2305), no import needed

        result = {
            'fills_detected': 0,
            'partials_detected': 0,
            'cancels_detected': 0,
            'orders_checked': 0,
            'api_calls': 0,
            'skipped': False,
            'skipped_reason': None,
            'errors': []
        }

        # Terminal status set (normalized lowercase)
        TERMINAL_STATUSES = {'filled', 'canceled', 'cancelled', 'expired', 'rejected', 'suspended'}

        # Early exit if feature disabled
        if not self.use_limit_orders:
            result['skipped'] = True
            result['skipped_reason'] = 'limit_orders_disabled'
            return result

        if not getattr(self.config, 'ENABLE_FAST_FILL_CHECK', True):
            result['skipped'] = True
            result['skipped_reason'] = 'feature_disabled'
            return result

        now = datetime.now()

        # Check error backoff
        if self._fast_fill_backoff_until and now < self._fast_fill_backoff_until:
            result['skipped'] = True
            result['skipped_reason'] = 'error_backoff'
            return result

        # Rate limiting - check interval
        if self._last_fast_fill_check:
            elapsed = (now - self._last_fast_fill_check).total_seconds()
            if elapsed < min_interval_seconds:
                result['skipped'] = True
                result['skipped_reason'] = 'rate_limited'
                return result

        self._last_fast_fill_check = now

        # Early exit if no open limit orders tracked
        if not self.grid_strategy.open_limit_orders:
            self.fast_fill_stats['last_check_at'] = now.isoformat()
            self.fast_fill_stats['open_tracked_count'] = 0
            self.fast_fill_stats['last_result'] = 'no_tracked_orders'
            return result

        try:
            # Step 1: Get current open orders from Alpaca (single API call)
            alpaca_open = await run_blocking(client.get_open_orders, self.symbols)
            result['api_calls'] += 1

            alpaca_open_ids = {o['id'] for o in alpaca_open}

            # Step 2: Find "disappeared" orders (tracked but not in Alpaca open list)
            tracked_orders = list(self.grid_strategy.open_limit_orders.values())
            tracked_ids = {order.order_id for order in tracked_orders}
            disappeared_ids = tracked_ids - alpaca_open_ids

            self.fast_fill_stats['open_tracked_count'] = len(tracked_ids)
            self.fast_fill_stats['disappeared_count'] = len(disappeared_ids)

            if not disappeared_ids:
                # All tracked orders still open - nothing to do
                self.fast_fill_stats['last_check_at'] = now.isoformat()
                self.fast_fill_stats['last_result'] = f'all_open tracked={len(tracked_ids)}'
                return result

            logger.info(f"[FAST-FILL] {len(disappeared_ids)} orders disappeared from Alpaca open list")

            # Step 3: Filter by grace period (using tracked created_at)
            # Build map of order_id -> OpenLimitOrder for quick lookup
            order_map = {o.order_id: o for o in tracked_orders}

            eligible_disappeared = []
            for order_id in disappeared_ids:
                tracked_order = order_map.get(order_id)
                if not tracked_order:
                    continue

                # Parse created_at safely
                try:
                    created_at = datetime.fromisoformat(tracked_order.created_at.replace('Z', '+00:00'))
                    # Remove timezone for comparison if needed
                    if created_at.tzinfo:
                        created_at = created_at.replace(tzinfo=None)
                    age_seconds = (now - created_at).total_seconds()

                    if age_seconds < min_order_age_seconds:
                        # Too young - grace period, skip
                        logger.debug(f"[FAST-FILL] Skipping {order_id[:8]}: age={age_seconds:.1f}s < grace={min_order_age_seconds}s")
                        continue
                except Exception as e:
                    # If parse fails, default to SKIP (avoid false deletes)
                    logger.warning(f"[FAST-FILL] Could not parse created_at for {order_id[:8]}: {e}, skipping")
                    continue

                eligible_disappeared.append((order_id, tracked_order, age_seconds))

            if not eligible_disappeared:
                self.fast_fill_stats['last_check_at'] = now.isoformat()
                self.fast_fill_stats['last_result'] = f'all_young disappeared={len(disappeared_ids)}'
                return result

            # Step 4: Sort by age (oldest first) for better latency
            eligible_disappeared.sort(key=lambda x: x[2], reverse=True)

            # Step 5: Check each disappeared order (rate-limited)
            orders_to_check = eligible_disappeared[:max_checks_per_cycle]

            for order_id, tracked_order, age_seconds in orders_to_check:
                result['orders_checked'] += 1
                self.fast_fill_stats['orders_checked_count'] += 1

                try:
                    # Fetch full order details from Alpaca
                    order = await run_blocking(client.get_order_by_id, order_id)
                    result['api_calls'] += 1

                    if order is None:
                        logger.warning(f"[FAST-FILL] Order {order_id[:8]} not found on Alpaca")
                        # Terminal (order doesn't exist) - remove tracking
                        self.grid_strategy.remove_open_limit_order(order_id)
                        self.grid_strategy.remove_pending_order(order_id)
                        self.fast_fill_stats['terminal_removed_count'] += 1
                        continue

                    # Normalize status (handle OrderStatus.FILLED, "filled", etc.)
                    raw_status = str(getattr(order, 'status', '') or '')
                    status = raw_status.lower().replace('orderstatus.', '').replace('querystatus.', '').strip()

                    # Normalize filled_qty and filled_avg_price (handle None, string, float)
                    filled_qty_raw = getattr(order, 'filled_qty', None)
                    filled_price_raw = getattr(order, 'filled_avg_price', None)

                    try:
                        filled_qty = float(filled_qty_raw) if filled_qty_raw is not None else 0.0
                    except (ValueError, TypeError):
                        filled_qty = 0.0

                    try:
                        filled_price = float(filled_price_raw) if filled_price_raw is not None else 0.0
                    except (ValueError, TypeError):
                        filled_price = 0.0

                    # Normalize symbol (BTCUSD -> BTC/USD)
                    raw_symbol = str(getattr(order, 'symbol', '') or '')
                    symbol = normalize_symbol(raw_symbol)

                    # Normalize side to canonical "buy" | "sell"
                    raw_side = str(getattr(order, 'side', '') or '')
                    side = normalize_side(raw_side)

                    # Branch on status
                    if status == 'filled':
                        # Full fill - apply and remove tracking
                        if filled_qty > 0 and filled_price > 0:
                            source = tracked_order.source if tracked_order else "grid"

                            # Apply fill (idempotent via applied_order_ids)
                            # NOTE: Call ONLY apply_filled_order, NOT record_fill
                            profit = self.grid_strategy.apply_filled_order(
                                symbol=symbol,
                                side=side,
                                fill_price=filled_price,
                                fill_qty=filled_qty,
                                order_id=order_id,
                                source=source
                            )

                            # Always remove tracking (it's terminal)
                            self.grid_strategy.remove_open_limit_order(order_id)
                            # Remove pending only if applied or recorded unmatched
                            # apply_filled_order handles pending removal internally
                            # but we ensure it here too
                            self.grid_strategy.remove_pending_order(order_id)

                            result['fills_detected'] += 1
                            self.fast_fill_stats['filled_applied_count'] += 1
                            logger.info(f"[FAST-FILL] Detected fill: {side} {symbol} @ ${filled_price:,.2f} qty={filled_qty:.6f}")

                    elif status in TERMINAL_STATUSES:
                        # Terminal but not 'filled' - check for partial fill
                        if filled_qty > 0 and filled_price > 0:
                            # Partial fill on cancel - apply the filled portion
                            source = tracked_order.source if tracked_order else "grid"

                            # Apply partial (use filled_qty, not original qty!)
                            profit = self.grid_strategy.apply_filled_order(
                                symbol=symbol,
                                side=side,
                                fill_price=filled_price,
                                fill_qty=filled_qty,  # Use actual filled qty
                                order_id=order_id,
                                source=source
                            )

                            result['partials_detected'] += 1
                            self.fast_fill_stats['partial_applied_count'] += 1
                            logger.info(f"[FAST-FILL] Partial fill on {status}: {side} {symbol} @ ${filled_price:,.2f} qty={filled_qty:.6f}")

                        # ALWAYS remove open tracking (it's terminal)
                        self.grid_strategy.remove_open_limit_order(order_id)
                        # Remove pending if applied OR recorded unmatched (don't spin forever)
                        # If it was unmatched, grid_strategy._record_unmatched was called
                        self.grid_strategy.remove_pending_order(order_id)

                        result['cancels_detected'] += 1
                        self.fast_fill_stats['terminal_removed_count'] += 1
                        logger.info(f"[FAST-FILL] Order {order_id[:8]} {status}, removed from tracking")

                    else:
                        # Non-terminal status (new, accepted, pending_new, etc.)
                        # Do NOT remove tracking - increment unknown and try again later
                        self.fast_fill_stats['unknown_status_count'] += 1
                        logger.warning(f"[FAST-FILL] Order {order_id[:8]} has non-terminal status: {status}, keeping tracked")

                except Exception as e:
                    error_msg = f"Error checking order {order_id[:8]}: {str(e)}"
                    result['errors'].append(error_msg)
                    self.fast_fill_stats['errors_count'] += 1
                    logger.error(f"[FAST-FILL] {error_msg}")

            # Save state if any changes
            if result['fills_detected'] > 0 or result['partials_detected'] > 0 or result['cancels_detected'] > 0:
                self.grid_strategy.save_state()

            # Update stats
            self.fast_fill_stats['last_check_at'] = now.isoformat()
            self.fast_fill_stats['last_result'] = (
                f"filled={result['fills_detected']} partial={result['partials_detected']} "
                f"removed={result['cancels_detected']} checked={result['orders_checked']}"
            )

        except Exception as e:
            error_msg = f"Fast fill check failed: {str(e)}"
            result['errors'].append(error_msg)
            self.fast_fill_stats['errors_count'] += 1
            self.fast_fill_stats['last_error_at'] = now.isoformat()
            logger.error(f"[FAST-FILL] {error_msg}")

            # Set error backoff
            backoff_seconds = getattr(self.config, 'FAST_FILL_ERROR_BACKOFF_SECONDS', 30.0)
            self._fast_fill_backoff_until = now + timedelta(seconds=backoff_seconds)
            logger.warning(f"[FAST-FILL] Backing off for {backoff_seconds}s after error")

        return result

    def sync_open_orders_from_alpaca(self) -> None:
        """
        Sync open orders from Alpaca with our tracking on startup.

        Handles the case where bot crashed with open limit orders.
        """
        if not self.use_limit_orders:
            return

        try:
            alpaca_open = self.client.get_open_orders(self.symbols)

            our_open_ids = {o.order_id for o in self.grid_strategy.open_limit_orders.values()}
            alpaca_open_ids = {o['id'] for o in alpaca_open}

            # Orders we're tracking but Alpaca says aren't open (filled or cancelled)
            stale_tracked = our_open_ids - alpaca_open_ids
            for order_id in stale_tracked:
                # These might have filled - reconciliation will pick them up
                removed = self.grid_strategy.remove_open_limit_order(order_id)
                if removed:
                    logger.info(f"[SYNC] Removed tracked order not open on Alpaca: {order_id[:8]}")

            # Orders open on Alpaca but we're not tracking (submitted before crash?)
            untracked = alpaca_open_ids - our_open_ids
            if untracked:
                logger.warning(f"[SYNC] Found {len(untracked)} open orders on Alpaca not tracked locally")
                cancel_untracked = getattr(self.config, 'CANCEL_UNTRACKED_OPEN_ORDERS_ON_STARTUP', True)
                if cancel_untracked:
                    cancelled = 0
                    for order_id in untracked:
                        if self.client.cancel_order(order_id):
                            cancelled += 1
                            logger.warning(f"[SYNC] Cancelled untracked open order on Alpaca: {order_id[:8]}")
                    if cancelled:
                        logger.warning(f"[SYNC] Cancelled {cancelled}/{len(untracked)} untracked open orders on startup")
                else:
                    logger.warning("[SYNC] Leaving untracked open orders on Alpaca (config disables auto-cancel)")

            logger.info(f"[SYNC] Open order sync: {len(alpaca_open)} on Alpaca, "
                       f"{len(self.grid_strategy.open_limit_orders)} tracked")

        except Exception as e:
            logger.warning(f"Could not sync open orders: {e}")

    def _load_alltime_equity(self) -> None:
        """Load all-time equity tracking (persists across all restarts)."""
        import json
        try:
            if os.path.exists(self.alltime_equity_file):
                with open(self.alltime_equity_file, 'r') as f:
                    data = json.load(f)
                    self.alltime_starting_equity = data.get('alltime_starting_equity', 0)
                    self.alltime_starting_date = data.get('alltime_starting_date')
                    self.grid_starting_equity = data.get('grid_starting_equity', 0)
                    self.grid_starting_date = data.get('grid_starting_date')
                    logger.info(f"Loaded all-time equity: ${self.alltime_starting_equity:,.2f} from {self.alltime_starting_date}")
                    logger.info(f"Loaded grid starting equity: ${self.grid_starting_equity:,.2f} from {self.grid_starting_date}")
            else:
                # First time running - will be set from verified API data
                # These are the CORRECT historical values from Alpaca portfolio history
                self.alltime_starting_equity = 100000.00  # TRUE starting equity
                self.alltime_starting_date = "2025-11-24"
                self.grid_starting_equity = 90276.26  # Dec 2, 2025 - grid trading started
                self.grid_starting_date = "2025-12-02"
                self._save_alltime_equity()
                logger.info(f"Initialized all-time equity tracking")
                logger.info(f"  All-time start: ${self.alltime_starting_equity:,.2f} ({self.alltime_starting_date})")
                logger.info(f"  Grid start: ${self.grid_starting_equity:,.2f} ({self.grid_starting_date})")
        except Exception as e:
            logger.error(f"Could not load all-time equity: {e}")
            # Use verified defaults
            self.alltime_starting_equity = 100000.00
            self.alltime_starting_date = "2025-11-24"
            self.grid_starting_equity = 90276.26
            self.grid_starting_date = "2025-12-02"

    def _save_alltime_equity(self) -> None:
        """Save all-time equity tracking to file."""
        import json
        try:
            data = {
                'alltime_starting_equity': self.alltime_starting_equity,
                'alltime_starting_date': self.alltime_starting_date,
                'grid_starting_equity': self.grid_starting_equity,
                'grid_starting_date': self.grid_starting_date,
                'last_updated': datetime.now().isoformat()
            }
            with open(self.alltime_equity_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Could not save all-time equity: {e}")

    def _load_circuit_breaker_state(self) -> None:
        """Load circuit breaker state from disk - PERSISTS across restarts for safety."""
        import json
        try:
            if os.path.exists(self.circuit_breaker_file):
                with open(self.circuit_breaker_file, 'r') as f:
                    data = json.load(f)

                    # Load max drawdown state (persists until manually reset)
                    if data.get('max_drawdown_hit', False):
                        self.max_drawdown_hit = True
                        triggered_at = data.get('max_drawdown_triggered_at', 'unknown')
                        logger.warning(f"CIRCUIT BREAKER ACTIVE: Max drawdown hit at {triggered_at}")
                        logger.warning("Trading is HALTED. Use /api/risk/reset to resume.")

                    # Load stop-loss states (persist until manually reset)
                    stop_losses = data.get('stop_losses_triggered', {})
                    if stop_losses:
                        self.stop_loss_triggered = stop_losses
                        logger.warning(f"STOP-LOSSES ACTIVE for: {list(stop_losses.keys())}")
                        logger.warning("These symbols won't trade. Use /api/risk/reset to resume.")

                    # Daily limit resets automatically at midnight, but check if still today
                    if data.get('daily_limit_hit', False):
                        saved_date = data.get('daily_limit_date')
                        today = datetime.now().strftime('%Y-%m-%d')
                        if saved_date == today:
                            self.daily_limit_hit = True
                            logger.warning("DAILY LIMIT still active (same day). Trading halted until midnight.")

                    logger.info("Loaded circuit breaker state from disk")
        except Exception as e:
            logger.debug(f"Could not load circuit breaker state: {e}")

    def _save_circuit_breaker_state(self) -> None:
        """Save circuit breaker state to disk for persistence across restarts."""
        import json
        try:
            data = {
                'max_drawdown_hit': self.max_drawdown_hit,
                'max_drawdown_triggered_at': datetime.now().isoformat() if self.max_drawdown_hit else None,
                'daily_limit_hit': self.daily_limit_hit,
                'daily_limit_date': datetime.now().strftime('%Y-%m-%d') if self.daily_limit_hit else None,
                'stop_losses_triggered': self.stop_loss_triggered,
                'last_updated': datetime.now().isoformat()
            }
            with open(self.circuit_breaker_file, 'w') as f:
                json.dump(data, f, indent=2)
            logger.debug("Circuit breaker state saved to disk")
        except Exception as e:
            logger.error(f"Could not save circuit breaker state: {e}")

    def reset_circuit_breakers(self, reset_type: str = 'all') -> Dict[str, Any]:
        """
        Manually reset circuit breakers. Call via /api/risk/reset endpoint.

        Args:
            reset_type: 'all', 'drawdown', 'daily', or symbol name for stop-loss

        Returns:
            Dict with reset status
        """
        result = {'success': True, 'reset': [], 'message': ''}

        if reset_type == 'all':
            if self.max_drawdown_hit:
                result['reset'].append('max_drawdown')
            if self.daily_limit_hit:
                result['reset'].append('daily_limit')
            if self.stop_loss_triggered:
                result['reset'].extend(list(self.stop_loss_triggered.keys()))

            self.max_drawdown_hit = False
            self.daily_limit_hit = False
            self.stop_loss_triggered = {}
            result['message'] = 'All circuit breakers reset. Trading resumed.'

        elif reset_type == 'drawdown':
            self.max_drawdown_hit = False
            result['reset'].append('max_drawdown')
            result['message'] = 'Max drawdown circuit breaker reset.'

        elif reset_type == 'daily':
            self.daily_limit_hit = False
            result['reset'].append('daily_limit')
            result['message'] = 'Daily limit reset.'

        elif reset_type in self.stop_loss_triggered:
            del self.stop_loss_triggered[reset_type]
            result['reset'].append(reset_type)
            result['message'] = f'Stop-loss for {reset_type} reset.'

        else:
            result['success'] = False
            result['message'] = f'Unknown reset type: {reset_type}'
            return result

        # Save the cleared state
        self._save_circuit_breaker_state()
        logger.warning(f"CIRCUIT BREAKERS RESET: {result['reset']}")

        return result

    def _load_daily_equity(self) -> None:
        """Load daily equity using Alpaca's last_equity for true daily P/L tracking."""
        import json
        try:
            # First, try to get Alpaca's last_equity (previous day's close)
            # This is the most accurate source for true daily P/L
            alpaca_last_equity = self._get_alpaca_last_equity()

            if alpaca_last_equity and alpaca_last_equity > 0:
                self.daily_start_equity = alpaca_last_equity
                self.last_trading_day = datetime.now().date()
                logger.info(f"Using Alpaca last_equity for daily P/L: ${self.daily_start_equity:,.2f}")

                # Also load peak equity from file if available
                if os.path.exists(self.daily_equity_file):
                    with open(self.daily_equity_file, 'r') as f:
                        data = json.load(f)
                        saved_date = data.get('date')
                        today = datetime.now().strftime('%Y-%m-%d')
                        if saved_date == today:
                            self.peak_equity = data.get('peak_equity', self.daily_start_equity)
                return

            # Fallback to file-based tracking if Alpaca unavailable
            if os.path.exists(self.daily_equity_file):
                with open(self.daily_equity_file, 'r') as f:
                    data = json.load(f)
                    saved_date = data.get('date')
                    today = datetime.now().strftime('%Y-%m-%d')

                    if saved_date == today:
                        self.daily_start_equity = data.get('starting_equity', 0)
                        self.last_trading_day = datetime.now().date()
                        self.peak_equity = data.get('peak_equity', self.daily_start_equity)
                        logger.info(f"Loaded daily equity from file: ${self.daily_start_equity:,.2f} (today)")
                    else:
                        logger.info(f"Daily equity file is from {saved_date}, will reset for today")
        except Exception as e:
            logger.debug(f"Could not load daily equity: {e}")

    def _get_alpaca_last_equity(self) -> Optional[float]:
        """Get Alpaca's last_equity (previous day's closing equity) for accurate daily P/L."""
        try:
            account = self.client.trading_client.get_account()
            last_equity = float(account.last_equity)
            logger.debug(f"Alpaca last_equity: ${last_equity:,.2f}")
            return last_equity
        except Exception as e:
            logger.debug(f"Could not fetch Alpaca last_equity: {e}")
            return None

    def _save_daily_equity(self, equity: float, is_new_day: bool = False) -> None:
        """Save daily equity to file for persistence across restarts."""
        import json
        try:
            today = datetime.now().strftime('%Y-%m-%d')
            data = {
                'date': today,
                'starting_equity': self.daily_start_equity,
                'peak_equity': self.peak_equity,
                'current_equity': equity,
                'last_updated': datetime.now().isoformat()
            }

            # On new day, capture the starting equity
            if is_new_day:
                data['starting_equity'] = equity

            with open(self.daily_equity_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.debug(f"Could not save daily equity file: {e}")

    def add_bar(self, symbol: str, bar: Any) -> None:
        """Add a bar to history (for tracking only - grids don't need features)."""
        if symbol not in self.bars:
            self.bars[symbol] = []

        self.bars[symbol].append(bar)

        # Keep last 100 bars
        if len(self.bars[symbol]) > 100:
            self.bars[symbol].pop(0)

        # Update price history for correlation monitoring
        price = float(bar.close) if hasattr(bar, 'close') else float(bar.get('close', 0))
        if symbol not in self.price_history:
            self.price_history[symbol] = []
        self.price_history[symbol].append(price)
        if len(self.price_history[symbol]) > self.correlation_window:
            self.price_history[symbol].pop(0)

    def detect_regime(self, symbol: str) -> Dict[str, Any]:
        """
        Detect market regime for a symbol using stored bars.

        Returns:
            Dict with regime info, whether to allow buys/sells, and ADX value
        """
        result = {
            'regime': MarketRegime.UNKNOWN,
            'allow_buy': True,
            'allow_sell': True,
            'is_strong_trend': False,
            'adx': 0,
            'confidence': 0,
            'reason': 'Insufficient data'
        }

        bars = self.bars.get(symbol, [])
        if len(bars) < 50:
            return result

        try:
            # Build DataFrame from bars
            data = []
            for bar in bars:
                data.append({
                    'open': float(bar.open) if hasattr(bar, 'open') else float(bar.get('open', 0)),
                    'high': float(bar.high) if hasattr(bar, 'high') else float(bar.get('high', 0)),
                    'low': float(bar.low) if hasattr(bar, 'low') else float(bar.get('low', 0)),
                    'close': float(bar.close) if hasattr(bar, 'close') else float(bar.get('close', 0)),
                    'volume': float(bar.volume) if hasattr(bar, 'volume') else float(bar.get('volume', 0))
                })

            df = pd.DataFrame(data)

            # Calculate features for regime detection
            df = FeatureCalculator.calculate_features(df, drop_warmup=False)

            # Run regime detection
            regime_result = self.regime_detector.detect(df)

            regime = regime_result.get('regime', MarketRegime.UNKNOWN)
            adx = regime_result.get('metrics', {}).get('adx', 0)
            confidence = regime_result.get('confidence', 0)

            # Store metrics
            self.current_regime[symbol] = regime
            self.regime_metrics[symbol] = regime_result.get('metrics', {})

            # Check for strong trend
            is_strong_trend = (
                regime in [MarketRegime.TRENDING_UP, MarketRegime.TRENDING_DOWN]
                and adx > self.regime_adx_threshold
            )

            # Check for developing downtrend (ADX between developing and strong threshold)
            developing_threshold = getattr(self.config, 'REGIME_ADX_DEVELOPING', 25)
            is_developing_downtrend = (
                regime == MarketRegime.TRENDING_DOWN
                and developing_threshold <= adx <= self.regime_adx_threshold
            )

            # Determine whether to allow buys/sells based on regime
            allow_buy = True
            allow_sell = True
            size_mult = 1.0  # Default: full size
            reason = f"Regime: {regime}"

            if is_strong_trend:
                self.regime_pause[symbol] = True

                if regime == MarketRegime.TRENDING_DOWN:
                    # Strong downtrend: PAUSE buys (don't catch falling knife)
                    # ALLOW sells (take profit on existing positions)
                    allow_buy = False
                    allow_sell = True
                    size_mult = 0.0  # No buys
                    reason = f"STRONG DOWNTREND (ADX={adx:.0f}) - Pausing buys"

                elif regime == MarketRegime.TRENDING_UP:
                    # Strong uptrend: PAUSE sells (let winners run)
                    # ALLOW buys (can still accumulate)
                    allow_buy = True
                    allow_sell = False
                    reason = f"STRONG UPTREND (ADX={adx:.0f}) - Holding sells"

            elif is_developing_downtrend:
                # Developing downtrend: REDUCE buy size (don't catch falling knife early)
                self.regime_pause[symbol] = False
                size_mult = getattr(self.config, 'DEVELOPING_DOWNTREND_SIZE_MULT', 0.5)
                reason = f"DEVELOPING DOWNTREND (ADX={adx:.0f}) - Size reduced to {size_mult:.0%}"
                logger.info(f"[REGIME] {symbol}: {reason}")

            else:
                self.regime_pause[symbol] = False
                reason = f"{regime} (ADX={adx:.0f}) - Grid active"

            result = {
                'regime': regime,
                'allow_buy': allow_buy,
                'allow_sell': allow_sell,
                'is_strong_trend': is_strong_trend,
                'is_developing_downtrend': is_developing_downtrend,
                'size_mult': size_mult,
                'adx': adx,
                'confidence': confidence,
                'reason': reason,
                'strategy_hint': regime_result.get('strategy_hint', 'WAIT'),
                'metrics': regime_result.get('metrics', {})
            }

        except Exception as e:
            logger.warning(f"Regime detection failed for {symbol}: {e}")

        return result

    def is_optimal_trading_time(self) -> Dict[str, Any]:
        """
        Check if current time is within optimal trading hours.

        Returns:
            Dict with 'should_trade', 'reason', 'time_quality' (0-1 score)
        """
        if not self.use_time_filter:
            return {'should_trade': True, 'reason': 'Time filter disabled', 'time_quality': 1.0}

        now = datetime.utcnow()
        hour = now.hour
        day = now.weekday()  # 0=Monday, 6=Sunday

        result = {
            'should_trade': True,
            'reason': '',
            'time_quality': 0.5,
            'hour': hour,
            'day': day,
            'is_weekend': day >= 5
        }

        # Check if in avoid hours
        for start, end in self.avoid_hours:
            if start <= hour < end:
                result['should_trade'] = False
                result['reason'] = f"Avoiding low-liquidity hour {hour}:00 UTC"
                result['time_quality'] = 0.2
                return result

        # Check if in optimal hours
        for start, end in self.optimal_hours:
            if start <= hour < end:
                result['should_trade'] = True
                result['reason'] = f"Optimal trading hour {hour}:00 UTC"
                result['time_quality'] = 1.0

                # Weekend adjustment
                if day >= 5:
                    result['time_quality'] *= self.weekend_size_mult
                    result['reason'] += f" (weekend: {self.weekend_size_mult:.0%} size)"

                return result

        # Default: trade but with reduced confidence
        result['should_trade'] = True
        result['reason'] = f"Standard hour {hour}:00 UTC"
        result['time_quality'] = 0.7

        if day >= 5:
            result['time_quality'] *= self.weekend_size_mult
            result['reason'] += f" (weekend: {self.weekend_size_mult:.0%} size)"

        return result

    def check_consecutive_down_bars(self, symbol: str) -> Tuple[bool, int]:
        """
        Check for consecutive down (red) bars as a simple crash guard.

        Args:
            symbol: The trading symbol to check

        Returns:
            Tuple of (allow_buy, consecutive_down_count)
            - allow_buy: True if buys are allowed, False if blocked
            - consecutive_down_count: Number of consecutive down bars
        """
        if not getattr(self.config, 'CONSECUTIVE_DOWN_BARS_ENABLED', True):
            return True, 0

        bars = self.bars.get(symbol, [])
        threshold = getattr(self.config, 'CONSECUTIVE_DOWN_BARS_BLOCK', 3)

        if len(bars) < threshold:
            return True, 0

        # Count consecutive down bars from most recent
        count = 0
        for bar in reversed(bars[-threshold:]):
            bar_open = float(bar.open) if hasattr(bar, 'open') else float(bar.get('open', 0))
            bar_close = float(bar.close) if hasattr(bar, 'close') else float(bar.get('close', 0))
            if bar_close < bar_open:
                count += 1
            else:
                break

        allow_buy = count < threshold
        if not allow_buy:
            logger.info(f"[DOWN] {symbol}: {count} consecutive down bars - blocking buys")

        return allow_buy, count

    def calculate_correlations(self) -> Dict[str, float]:
        """
        Calculate pairwise correlations between all tracked symbols.

        Returns:
            Dict with correlation pairs (e.g., 'BTC/USD-ETH/USD': 0.92)
        """
        correlations = {}

        symbols_with_data = [s for s in self.symbols if len(self.price_history.get(s, [])) >= self.correlation_window]

        if len(symbols_with_data) < 2:
            return correlations

        for i, sym1 in enumerate(symbols_with_data):
            for sym2 in symbols_with_data[i+1:]:
                prices1 = np.array(self.price_history[sym1][-self.correlation_window:])
                prices2 = np.array(self.price_history[sym2][-self.correlation_window:])

                # Calculate returns for correlation
                returns1 = np.diff(prices1) / prices1[:-1]
                returns2 = np.diff(prices2) / prices2[:-1]

                if len(returns1) > 1 and len(returns2) > 1:
                    corr = np.corrcoef(returns1, returns2)[0, 1]
                    if not np.isnan(corr):
                        correlations[f"{sym1}-{sym2}"] = float(corr)

        return correlations

    def get_correlation_risk_adjustment(self, symbol: str) -> float:
        """
        Get position size adjustment based on correlation risk.

        If assets are highly correlated (>0.85), reduce position sizes
        to avoid simultaneous losses.

        Returns:
            Multiplier for position size (0.5 to 1.0)
        """
        correlations = self.calculate_correlations()

        if not correlations:
            return 1.0

        # Find correlations involving this symbol
        symbol_correlations = []
        for pair, corr in correlations.items():
            if symbol in pair:
                symbol_correlations.append(abs(corr))

        if not symbol_correlations:
            return 1.0

        max_corr = max(symbol_correlations)
        avg_corr = np.mean(symbol_correlations)

        # If highly correlated, reduce position size
        if max_corr > self.high_correlation_threshold:
            # Linear reduction: 0.85 corr = 0.75x, 0.95 corr = 0.5x
            adjustment = 1.0 - (max_corr - self.high_correlation_threshold) * 2.5
            adjustment = max(0.5, min(1.0, adjustment))
            logger.info(f"[CORR] {symbol} high correlation ({max_corr:.2f}), reducing size to {adjustment:.0%}")
            return adjustment

        return 1.0

    def track_trade_performance(self, symbol: str, profit: float, hour: int) -> None:
        """Track trade performance by hour for optimization."""
        self.actual_profits.append(profit)
        self.trade_count_by_hour[hour] = self.trade_count_by_hour.get(hour, 0) + 1
        self.profit_by_hour[hour] = self.profit_by_hour.get(hour, 0.0) + profit

    def get_momentum_filter(self, symbol: str) -> Dict[str, Any]:
        """
        Check momentum to avoid buying in strong downtrends.

        Uses simple price momentum over last N bars to detect trends.

        Returns:
            Dict with 'allow_buy', 'allow_sell', 'momentum', 'reason'
        """
        result = {
            'allow_buy': True,
            'allow_sell': True,
            'momentum': 0.0,
            'reason': 'Neutral momentum'
        }

        prices = self.price_history.get(symbol, [])
        if len(prices) < 10:
            return result

        # Calculate short-term momentum (last 5 vs previous 5)
        recent = np.mean(prices[-5:])
        previous = np.mean(prices[-10:-5])

        MIN_PRICE_FOR_MOMENTUM = 0.01  # Avoid division by tiny values
        if previous > MIN_PRICE_FOR_MOMENTUM:
            # Result is in PERCENT units: -2.5 means -2.5% (converted to decimal at handoff)
            momentum = (recent - previous) / previous * 100
            momentum = np.clip(momentum, -100.0, 100.0)  # Clamp to [-100%, +100%]
            result['momentum'] = round(momentum, 2)      # Percent units

            # Strong downtrend: don't buy
            if momentum < -1.5:  # More than 1.5% down
                result['allow_buy'] = False
                result['reason'] = f"Strong downtrend ({momentum:.1f}%), avoiding buys"

            # Strong uptrend: don't sell (let winners run)
            elif momentum > 1.5:  # More than 1.5% up
                result['allow_sell'] = False
                result['reason'] = f"Strong uptrend ({momentum:.1f}%), holding position"

            elif momentum > 0.5:
                result['reason'] = f"Mild uptrend ({momentum:.1f}%)"
            elif momentum < -0.5:
                result['reason'] = f"Mild downtrend ({momentum:.1f}%)"
            else:
                result['reason'] = f"Sideways ({momentum:.1f}%)"
        else:
            # Denominator too small - set momentum to neutral 0.0 (percent)
            result['momentum'] = 0.0

        return result

    # === WINDFALL PROFIT-TAKING METHODS ===

    def _load_windfall_log(self) -> None:
        """Load windfall stats from persistent file."""
        import json
        try:
            if os.path.exists(self.windfall_log_file):
                with open(self.windfall_log_file, 'r') as f:
                    data = json.load(f)
                    self.windfall_stats = {
                        "total_captures": data.get("total_captures", 0),
                        "total_profit": data.get("total_profit", 0.0),
                        "transactions": data.get("transactions", [])
                    }
                    logger.info(f"Loaded windfall stats: {self.windfall_stats['total_captures']} captures, ${self.windfall_stats['total_profit']:.2f} profit")
        except Exception as e:
            logger.debug(f"Could not load windfall log: {e}")

    def _save_windfall_log(self) -> None:
        """Save windfall stats to persistent file."""
        import json
        try:
            data = {
                "total_captures": self.windfall_stats["total_captures"],
                "total_profit": self.windfall_stats["total_profit"],
                "transactions": self.windfall_stats["transactions"][-100:],  # Keep last 100
                "last_updated": datetime.now().isoformat()
            }
            with open(self.windfall_log_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Could not save windfall log: {e}")

    def calculate_rsi(self, symbol: str, period: int = 14) -> float:
        """
        Calculate RSI from recent price history.

        Args:
            symbol: Trading symbol
            period: RSI period (default 14)

        Returns:
            RSI value (0-100), or 50.0 if insufficient data
        """
        prices = self.price_history.get(symbol, [])
        if len(prices) < period + 1:
            return 50.0  # Neutral if not enough data

        # Calculate price changes
        changes = [prices[i] - prices[i-1] for i in range(1, len(prices))]
        recent_changes = changes[-period:]

        # Separate gains and losses
        gains = [c if c > 0 else 0 for c in recent_changes]
        losses = [-c if c < 0 else 0 for c in recent_changes]

        # Calculate average gain/loss
        avg_gain = sum(gains) / period
        avg_loss = sum(losses) / period

        if avg_loss == 0:
            return 100.0 if avg_gain > 0 else 50.0

        # Calculate RSI
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return round(rsi, 2)

    def check_windfall_opportunity(self, symbol: str, position: Dict, current_price: float) -> Dict[str, Any]:
        """
        Check if position qualifies for windfall profit-taking.

        Logic (Option B - Momentum-Triggered Exit):
        - Soft trigger: unrealized > 4% AND RSI > 70
        - Hard trigger: unrealized > 6% (regardless of RSI)

        Args:
            symbol: Trading symbol
            position: Position dict with unrealized_plpc
            current_price: Current market price

        Returns:
            Dict with should_sell, trigger_type, rsi, sell_portion, etc.
        """
        # Get windfall config
        windfall_config = getattr(self.config, 'WINDFALL_PROFIT_CONFIG', {})
        if not windfall_config.get('enabled', False):
            return {'should_sell': False, 'reason': 'disabled'}

        # Get unrealized P/L percentage
        unrealized_pct = float(position.get('unrealized_plpc', 0)) * 100

        # Check cooldown for this symbol
        if symbol in self.windfall_cooldowns:
            if datetime.now() < self.windfall_cooldowns[symbol]:
                remaining = (self.windfall_cooldowns[symbol] - datetime.now()).seconds // 60
                return {'should_sell': False, 'reason': f'cooldown_active ({remaining}m remaining)'}

        # Calculate RSI
        rsi = self.calculate_rsi(symbol)

        # Get thresholds from config
        soft_threshold = windfall_config.get('soft_threshold_pct', 4.0)
        hard_threshold = windfall_config.get('hard_threshold_pct', 6.0)
        rsi_threshold = windfall_config.get('rsi_threshold', 70)
        sell_portion = windfall_config.get('sell_portion', 0.70)

        should_sell = False
        trigger_type = None

        # Hard trigger: > 6% regardless of RSI
        if unrealized_pct > hard_threshold:
            should_sell = True
            trigger_type = 'hard_threshold'

        # Soft trigger: > 4% AND RSI > 70
        elif unrealized_pct > soft_threshold and rsi > rsi_threshold:
            should_sell = True
            trigger_type = 'soft_threshold_rsi'

        return {
            'should_sell': should_sell,
            'trigger_type': trigger_type,
            'unrealized_pct': round(unrealized_pct, 2),
            'rsi': rsi,
            'sell_portion': sell_portion,
            'soft_threshold': soft_threshold,
            'hard_threshold': hard_threshold,
            'rsi_threshold': rsi_threshold
        }

    def log_windfall_transaction(self, symbol: str, qty: float, price: float,
                                  profit: float, trigger_type: str, unrealized_pct: float, rsi: float) -> None:
        """
        Log windfall transaction for weekly review.

        Args:
            symbol: Trading symbol
            qty: Quantity sold
            price: Sell price
            profit: Profit captured (after fees)
            trigger_type: 'hard_threshold' or 'soft_threshold_rsi'
            unrealized_pct: Unrealized P/L % that triggered the sell
            rsi: RSI value at time of sell
        """
        transaction = {
            'timestamp': datetime.now().isoformat(),
            'symbol': symbol,
            'qty_sold': round(qty, 6),
            'sell_price': round(price, 2),
            'profit': round(profit, 2),
            'trigger_type': trigger_type,
            'unrealized_pct': round(unrealized_pct, 2),
            'rsi': round(rsi, 2)
        }

        self.windfall_stats['transactions'].append(transaction)
        self.windfall_stats['total_captures'] += 1
        self.windfall_stats['total_profit'] += profit

        # Keep only last 100 transactions in memory
        if len(self.windfall_stats['transactions']) > 100:
            self.windfall_stats['transactions'] = self.windfall_stats['transactions'][-100:]

        self._save_windfall_log()
        logger.info(f"[WINDFALL] Logged: {symbol} +${profit:.2f} ({trigger_type})")

    def set_windfall_cooldown(self, symbol: str) -> None:
        """Set cooldown for a symbol after windfall sell."""
        windfall_config = getattr(self.config, 'WINDFALL_PROFIT_CONFIG', {})
        cooldown_minutes = windfall_config.get('cooldown_minutes', 30)
        self.windfall_cooldowns[symbol] = datetime.now() + timedelta(minutes=cooldown_minutes)
        logger.info(f"[WINDFALL] Cooldown set for {symbol}: {cooldown_minutes} minutes")

    def get_windfall_stats(self) -> Dict[str, Any]:
        """Get windfall statistics for dashboard display."""
        windfall_config = getattr(self.config, 'WINDFALL_PROFIT_CONFIG', {})
        return {
            'enabled': windfall_config.get('enabled', False),
            'total_captures': self.windfall_stats['total_captures'],
            'total_profit': round(self.windfall_stats['total_profit'], 2),
            'transactions': self.windfall_stats['transactions'][-20:],  # Last 20 for dashboard
            'config': {
                'soft_threshold_pct': windfall_config.get('soft_threshold_pct', 4.0),
                'hard_threshold_pct': windfall_config.get('hard_threshold_pct', 6.0),
                'rsi_threshold': windfall_config.get('rsi_threshold', 70),
                'sell_portion': windfall_config.get('sell_portion', 0.70),
                'cooldown_minutes': windfall_config.get('cooldown_minutes', 30)
            },
            'active_cooldowns': {
                s: self.windfall_cooldowns[s].isoformat()
                for s in self.windfall_cooldowns
                if datetime.now() < self.windfall_cooldowns[s]
            }
        }

    def get_performance_report(self) -> Dict[str, Any]:
        """Get performance tracking report."""
        total_trades = len(self.actual_profits)
        total_profit = sum(self.actual_profits) if self.actual_profits else 0

        # Best and worst hours
        best_hour = max(self.profit_by_hour, key=self.profit_by_hour.get) if any(self.profit_by_hour.values()) else 0
        worst_hour = min(self.profit_by_hour, key=self.profit_by_hour.get) if any(self.profit_by_hour.values()) else 0

        # Win rate
        wins = sum(1 for p in self.actual_profits if p > 0)
        win_rate = wins / total_trades if total_trades > 0 else 0

        # Average profit per trade
        avg_profit = total_profit / total_trades if total_trades > 0 else 0

        # Expected vs actual
        expected_total = sum(self.expected_profit_per_trade.values()) * total_trades / len(self.expected_profit_per_trade) if self.expected_profit_per_trade else 0

        return {
            'total_trades': total_trades,
            'total_profit': round(total_profit, 2),
            'avg_profit_per_trade': round(avg_profit, 2),
            'win_rate': round(win_rate * 100, 1),
            'best_hour': best_hour,
            'best_hour_profit': round(self.profit_by_hour.get(best_hour, 0), 2),
            'worst_hour': worst_hour,
            'worst_hour_profit': round(self.profit_by_hour.get(worst_hour, 0), 2),
            'trades_by_hour': dict(self.trade_count_by_hour),
            'profit_by_hour': {k: round(v, 2) for k, v in self.profit_by_hour.items()},
            'expected_vs_actual': round((total_profit / expected_total - 1) * 100, 1) if expected_total > 0 else 0
        }

    # === ORDER CONFIRMATION METHODS ===

    def add_confirmed_order(self, order_data: Dict) -> None:
        """
        Add a confirmed order from Alpaca to the tracking list AND database.

        Args:
            order_data: Dict with order details from verify_order_fill()
        """
        # Add timestamp if not present
        if 'recorded_at' not in order_data:
            order_data['recorded_at'] = datetime.now().isoformat()

        self.confirmed_orders.append(order_data)

        # Keep last 500 orders in memory
        if len(self.confirmed_orders) > 500:
            self.confirmed_orders = self.confirmed_orders[-500:]

        # Also save to database for persistence
        try:
            database.record_order(order_data)
        except Exception as e:
            logger.warning(f"Failed to save order to database: {e}")

        logger.info(f"[CONFIRMED] Order added: {order_data.get('side', 'N/A')} {order_data.get('symbol', 'N/A')} "
                    f"qty={order_data.get('filled_qty', 0):.6f} @ ${order_data.get('filled_avg_price', 0):,.2f}")

    def get_confirmed_orders(self, limit: int = 50, symbol: str = None) -> List[Dict]:
        """
        Get list of confirmed orders for dashboard display.

        Args:
            limit: Maximum number of orders to return
            symbol: Optional filter by symbol

        Returns:
            List of confirmed order dicts, most recent first
        """
        orders = self.confirmed_orders.copy()

        if symbol:
            # Normalize symbol format for comparison
            symbol_normalized = symbol.replace('/', '')
            orders = [o for o in orders if o.get('symbol', '').replace('/', '') == symbol_normalized]

        # Sort by filled_at (most recent first)
        orders.sort(key=lambda x: x.get('filled_at', ''), reverse=True)

        return orders[:limit]

    def get_order_stats(self) -> Dict[str, Any]:
        """
        Get statistics about confirmed orders.

        Returns:
            Dict with order stats: total, by_symbol, by_side, fill_rate
        """
        if not self.confirmed_orders:
            return {
                'total_confirmed': 0,
                'by_symbol': {},
                'by_side': {'buy': 0, 'sell': 0},
                'avg_fill_price_by_symbol': {},
                'total_volume': 0.0
            }

        by_symbol = {}
        by_side = {'buy': 0, 'sell': 0}
        volume_by_symbol = {}
        price_sum_by_symbol = {}

        for order in self.confirmed_orders:
            symbol = order.get('symbol', 'UNKNOWN')
            side = order.get('side', '').lower()
            qty = order.get('filled_qty', 0)
            price = order.get('filled_avg_price', 0)

            # Count by symbol
            by_symbol[symbol] = by_symbol.get(symbol, 0) + 1

            # Count by side
            if 'buy' in side:
                by_side['buy'] += 1
            elif 'sell' in side:
                by_side['sell'] += 1

            # Track volume
            volume_by_symbol[symbol] = volume_by_symbol.get(symbol, 0) + (qty * price)

            # Track for average price
            if symbol not in price_sum_by_symbol:
                price_sum_by_symbol[symbol] = {'sum': 0, 'count': 0}
            price_sum_by_symbol[symbol]['sum'] += price
            price_sum_by_symbol[symbol]['count'] += 1

        # Calculate average fill prices
        avg_fill_price = {}
        for symbol, data in price_sum_by_symbol.items():
            if data['count'] > 0:
                avg_fill_price[symbol] = round(data['sum'] / data['count'], 2)

        return {
            'total_confirmed': len(self.confirmed_orders),
            'by_symbol': by_symbol,
            'by_side': by_side,
            'avg_fill_price_by_symbol': avg_fill_price,
            'total_volume': round(sum(volume_by_symbol.values()), 2)
        }

    async def reconcile_with_alpaca(self, client: 'AlpacaClient', days: int = 7) -> Dict[str, Any]:
        """
        Compare internal confirmed orders with Alpaca's order history.

        Args:
            client: AlpacaClient instance
            days: Number of days of history to check

        Returns:
            Dict with reconciliation results
        """
        try:
            # Fetch Alpaca order history
            alpaca_orders = client.get_order_history(days=days, symbols=self.symbols, status='closed')

            # Filter for filled orders only
            alpaca_filled = [o for o in alpaca_orders if 'filled' in o.get('status', '').lower()]

            # Get our confirmed order IDs
            our_order_ids = {o.get('order_id') for o in self.confirmed_orders if o.get('order_id')}

            # Get Alpaca order IDs
            alpaca_order_ids = {o.get('id') for o in alpaca_filled}

            # Find matches and discrepancies
            matched = our_order_ids & alpaca_order_ids
            missing_from_bot = alpaca_order_ids - our_order_ids
            missing_from_alpaca = our_order_ids - alpaca_order_ids

            # Build discrepancy list
            discrepancies = []
            for order_id in missing_from_bot:
                order = next((o for o in alpaca_filled if o.get('id') == order_id), None)
                if order:
                    discrepancies.append({
                        'type': 'missing_from_bot',
                        'order_id': order_id,
                        'symbol': order.get('symbol'),
                        'side': order.get('side'),
                        'filled_at': order.get('filled_at')
                    })

            for order_id in missing_from_alpaca:
                order = next((o for o in self.confirmed_orders if o.get('order_id') == order_id), None)
                if order:
                    discrepancies.append({
                        'type': 'missing_from_alpaca',
                        'order_id': order_id,
                        'symbol': order.get('symbol'),
                        'side': order.get('side'),
                        'filled_at': order.get('filled_at')
                    })

            # Update reconciliation status
            self.reconciliation_status = {
                'synced': len(discrepancies) == 0,
                'last_check': datetime.now().isoformat(),
                'matched': len(matched),
                'total_bot': len(our_order_ids),
                'total_alpaca': len(alpaca_order_ids),
                'discrepancies': discrepancies[:10]  # Limit to 10
            }
            self.last_reconciliation = datetime.now()

            logger.info(f"[RECONCILE] Matched: {len(matched)}, "
                        f"Missing from bot: {len(missing_from_bot)}, "
                        f"Missing from Alpaca: {len(missing_from_alpaca)}")

            return self.reconciliation_status

        except Exception as e:
            logger.error(f"[RECONCILE] Error during reconciliation: {e}")
            return {
                'synced': False,
                'last_check': datetime.now().isoformat(),
                'error': str(e)
            }

    async def reconcile_open_orders_on_startup(self, client: 'AlpacaClient') -> Dict[str, Any]:
        """
        Reconcile locally-tracked open limit orders with Alpaca on startup.

        Handles:
        1. Orders filled while bot was down -> apply_filled_order()
        2. Orders cancelled/expired -> remove from tracking
        3. Orders still open -> keep tracking
        4. Untracked orders on Alpaca -> optionally cancel (safety)

        Args:
            client: AlpacaClient instance

        Returns:
            Reconciliation summary
        """
        results = {
            'filled': [],
            'cancelled': [],
            'still_open': [],
            'untracked_cancelled': [],
            'errors': []
        }

        try:
            # Get all open orders from Alpaca
            alpaca_open_orders = client.get_open_orders(symbols=self.symbols)
            alpaca_open_ids = {o.get('id') for o in alpaca_open_orders if o.get('id')}

            # Get all filled orders from last 7 days (in case bot was down longer)
            # NOTE: No symbols filter - Alpaca API has format sensitivity issues
            # The filled_by_id lookup handles filtering by our tracked orders
            filled_orders = client.get_order_history(
                days=7,
                status='closed'
            )
            filled_by_id = {
                o.get('id'): o
                for o in filled_orders
                if o.get('id') and 'filled' in str(o.get('status', '')).lower()
            }

            # Check each locally-tracked open limit order
            for key, tracked in list(self.grid_strategy.open_limit_orders.items()):
                order_id = tracked.order_id

                if order_id in filled_by_id:
                    # Order filled while bot was down
                    filled_order = filled_by_id[order_id]
                    logger.info(f"[STARTUP] Order {order_id[:8]} filled while offline, applying...")

                    filled_qty = float(filled_order.get('filled_qty') or 0)
                    filled_price = float(filled_order.get('filled_avg_price') or 0)

                    if filled_qty > 0 and filled_price > 0:
                        self.grid_strategy.apply_filled_order(
                            symbol=tracked.symbol,
                            side=tracked.side,
                            fill_price=filled_price,
                            fill_qty=filled_qty,
                            order_id=order_id,
                            source=tracked.source
                        )
                        results['filled'].append(order_id)
                    else:
                        logger.warning(f"[STARTUP] Order {order_id[:8]} has invalid fill data")

                elif order_id in alpaca_open_ids:
                    # Order still open on Alpaca, keep tracking
                    results['still_open'].append(order_id)
                    logger.debug(f"[STARTUP] Order {order_id[:8]} still open on Alpaca")

                else:
                    # Order not found - probably cancelled or expired
                    logger.warning(f"[STARTUP] Order {order_id[:8]} not found on Alpaca, removing tracking")
                    self.grid_strategy.remove_open_limit_order(order_id)
                    results['cancelled'].append(order_id)

            # Handle untracked open orders (safety - cancel orphan limit orders)
            # Only cancel LIMIT orders for our grid symbols that we don't track
            cancel_untracked = getattr(self.config, 'CANCEL_UNTRACKED_OPEN_ORDERS_ON_STARTUP', True)
            if cancel_untracked:
                tracked_ids = {o.order_id for o in self.grid_strategy.open_limit_orders.values()}
                # Normalize our symbols for comparison
                our_symbols_normalized = {s.replace('/', '') for s in self.symbols}

                for alpaca_order in alpaca_open_orders:
                    order_id = alpaca_order.get('id')
                    # Normalize symbol for comparison
                    raw_symbol = alpaca_order.get('symbol', '')
                    symbol_normalized = raw_symbol.replace('/', '')
                    order_type = alpaca_order.get('type', '').lower()

                    # Only cancel LIMIT orders for our symbols that we don't track
                    is_our_symbol = symbol_normalized in our_symbols_normalized
                    is_limit_order = 'limit' in order_type
                    is_untracked = order_id not in tracked_ids

                    if is_our_symbol and is_limit_order and is_untracked:
                        logger.warning(f"[STARTUP] Orphan limit order {order_id[:8]} for {raw_symbol} not tracked, cancelling")
                        try:
                            client.cancel_order(order_id)
                            results['untracked_cancelled'].append(order_id)
                        except Exception as e:
                            logger.error(f"Failed to cancel orphan order {order_id[:8]}: {e}")
                            results['errors'].append(f"cancel_{order_id[:8]}: {str(e)}")

            # Save state after reconciliation
            self.grid_strategy.save_state()

            logger.info(f"[STARTUP] Reconciliation complete: "
                       f"{len(results['filled'])} filled, "
                       f"{len(results['still_open'])} open, "
                       f"{len(results['cancelled'])} removed, "
                       f"{len(results['untracked_cancelled'])} untracked cancelled")

        except Exception as e:
            logger.error(f"[STARTUP] Reconciliation error: {e}")
            results['errors'].append(str(e))

        return results

    def get_reconciliation_status(self) -> Dict[str, Any]:
        """Get current reconciliation status for dashboard."""
        status = self.reconciliation_status.copy()
        if self.last_reconciliation:
            # Add time since last check
            seconds_ago = (datetime.now() - self.last_reconciliation).total_seconds()
            status['seconds_ago'] = int(seconds_ago)
            status['minutes_ago'] = int(seconds_ago / 60)
        return status

    def get_order_health_summary(self) -> Dict[str, Any]:
        """
        Get order tracking health summary for dashboard display.

        Returns cached health check result with current tracked order count.
        Before first health check runs, assumes Alpaca matches tracked (optimistic).
        """
        # Get current tracked count (always fresh)
        current_tracked = len(self.grid_strategy.open_limit_orders)

        # If no health check has run yet, assume Alpaca matches tracked (optimistic)
        has_run_check = self._last_health_check.get('checked_at') is not None
        alpaca_count = self._last_health_check.get('alpaca_count', current_tracked) if has_run_check else current_tracked

        # Build summary from last health check
        return {
            'tracked_open_limits': current_tracked,
            'alpaca_open_grid_limits': alpaca_count,
            'mismatch': self._last_health_check.get('mismatch', 0) if has_run_check else 0,
            'healthy': self._last_health_check.get('healthy', True),
            'last_check_at': self._last_health_check.get('checked_at'),
            'orphan_ids': self._last_health_check.get('orphan_ids', []),
            'stale_ids': self._last_health_check.get('stale_ids', []),
            'error': self._last_health_check.get('error')
        }

    async def check_order_tracking_health(self, client: 'AlpacaClient') -> Dict[str, Any]:
        """
        Runtime health check to detect mismatch between Alpaca open orders and local tracking.

        Should be called periodically (every N minutes) to catch drift.

        Args:
            client: AlpacaClient instance

        Returns:
            Health check results with mismatch detection
        """
        try:
            # Fetch all open orders from Alpaca
            alpaca_orders = client.get_open_orders()  # Fetches all, we filter locally

            # Normalize our symbols for comparison
            our_symbols_normalized = {s.replace('/', '') for s in self.symbols}

            # Filter to grid limit orders for our symbols
            alpaca_grid_orders = [
                o for o in alpaca_orders
                if o.get('symbol', '').replace('/', '') in our_symbols_normalized
                and 'limit' in o.get('type', '').lower()
            ]

            tracked_count = len(self.grid_strategy.open_limit_orders)
            alpaca_count = len(alpaca_grid_orders)

            # Build result
            result = {
                'healthy': True,
                'alpaca_count': alpaca_count,
                'tracked_count': tracked_count,
                'mismatch': abs(alpaca_count - tracked_count),
                'checked_at': datetime.now().isoformat()
            }

            # Flag unhealthy if mismatch exceeds threshold
            mismatch_threshold = 2
            if abs(alpaca_count - tracked_count) > mismatch_threshold:
                result['healthy'] = False
                logger.error(
                    f"[HEALTH] Order tracking mismatch! "
                    f"Alpaca={alpaca_count}, Tracked={tracked_count}, "
                    f"Diff={alpaca_count - tracked_count}"
                )

                # Log details for debugging
                tracked_ids = {o.order_id for o in self.grid_strategy.open_limit_orders.values()}
                alpaca_ids = {o.get('id') for o in alpaca_grid_orders}

                orphans = alpaca_ids - tracked_ids
                stale = tracked_ids - alpaca_ids

                if orphans:
                    logger.warning(f"[HEALTH] Orphan orders on Alpaca (not tracked): {len(orphans)}")
                    result['orphan_ids'] = list(orphans)[:5]  # First 5 for logging
                if stale:
                    logger.warning(f"[HEALTH] Stale tracking (not on Alpaca): {len(stale)}")
                    result['stale_ids'] = list(stale)[:5]
            else:
                logger.debug(f"[HEALTH] Order tracking OK: Alpaca={alpaca_count}, Tracked={tracked_count}")

            # Store result for dashboard display
            self._last_health_check = result
            return result

        except Exception as e:
            logger.error(f"[HEALTH] Health check failed: {e}")
            error_result = {
                'healthy': False,
                'alpaca_count': 0,
                'tracked_count': len(self.grid_strategy.open_limit_orders),
                'mismatch': 0,
                'error': str(e),
                'checked_at': datetime.now().isoformat()
            }
            # Store error result for dashboard display
            self._last_health_check = error_result
            return error_result

    def initialize_grids(self, prices: Dict[str, float], equity: float) -> None:
        """
        Initialize grids for all symbols based on current prices.

        Restoration priority:
        1. Try load_from_db() - Database persistence (survives date changes/restarts)
        2. Try load_state() - Same-day /tmp JSON (backward compatibility)
        3. Create fresh grids for any symbols in GRID_CONFIGS that weren't restored

        Args:
            prices: Current price for each symbol
            equity: Total account equity
        """
        restored_symbols = []
        created_symbols = []
        skipped_symbols = []  # Configured but no price data

        # Try to restore from database first
        if self.grid_strategy.load_from_db():
            restored_symbols = list(self.grid_strategy.grids.keys())
            logger.info(f"Restored grids from DB: {restored_symbols}")
        # Fallback: Try to restore from /tmp JSON
        elif self.grid_strategy.load_state():
            restored_symbols = list(self.grid_strategy.grids.keys())
            logger.info(f"Restored grids from file: {restored_symbols}")

        # Compute totals deterministically (OVERWRITE, not increment)
        # This prevents double-counting on restart
        self.total_trades = 0
        self.total_profit = 0.0
        for symbol in restored_symbols:
            summary = self.grid_strategy.get_grid_summary(symbol)
            if summary.get('is_active'):
                perf = summary.get('performance', {})
                self.total_trades += perf.get('completed_trades', 0)
                self.total_profit += perf.get('total_profit', 0.0)

        # Build target symbol universe: union of self.symbols AND GRID_CONFIGS keys
        # This ensures we don't miss symbols if config changes
        grid_configs = getattr(self.config, 'GRID_CONFIGS', DEFAULT_GRID_CONFIGS)
        target_symbols = sorted(set(self.symbols) | set(grid_configs.keys()))

        # Create fresh grids for any missing symbols
        for symbol in target_symbols:
            if symbol in self.grid_strategy.grids:
                continue  # Already restored

            if symbol not in prices:
                skipped_symbols.append(symbol)
                logger.warning(f"[GRID] Cannot create grid for {symbol}: no price data available")
                continue

            current_price = prices[symbol]
            template = grid_configs.get(symbol, {
                "num_grids": 10,
                "range_pct": 0.05,
                "investment_ratio": 0.25
            })

            range_pct = template["range_pct"]
            num_grids = template["num_grids"]
            investment_ratio = template["investment_ratio"]
            capital_for_grid = equity * investment_ratio

            config = GridConfig(
                symbol=symbol,
                upper_price=current_price * (1 + range_pct / 2),
                lower_price=current_price * (1 - range_pct / 2),
                num_grids=num_grids,
                investment_per_grid=capital_for_grid / num_grids
            )

            self.grid_strategy.create_grid(config, current_price)
            created_symbols.append(symbol)

            logger.info(f"Created grid for {symbol}: ${config.lower_price:,.2f} - ${config.upper_price:,.2f}")

        # Log summary
        logger.info("=" * 50)
        logger.info("GRID INITIALIZATION COMPLETE")
        if restored_symbols:
            logger.info(f"  Restored from DB/file: {restored_symbols}")
        if created_symbols:
            logger.info(f"  Created (missing): {created_symbols}")
        if skipped_symbols:
            logger.error(f"  SKIPPED (no price data): {skipped_symbols}")
        logger.info(f"  Totals: {self.total_trades} trades, ${self.total_profit:.2f} profit")
        logger.info("=" * 50)

        # Save state to ensure DB has all grids (including newly created)
        if created_symbols:
            self.grid_strategy.save_state()
            logger.info("Saved updated grid state with new grids to DB")

        # Validate and rebuild any undersized grids (prevents $10 min notional skips)
        # This must run BEFORE grids_initialized=True so grids are final before trading
        rebuilt_symbols = self._validate_and_rebuild_undersized_grids(prices, equity)
        if rebuilt_symbols:
            logger.info(f"  Rebuilt undersized grids: {rebuilt_symbols}")

        # Mark initialized only after all validation/rebuilds complete
        self.grids_initialized = True

    def _validate_and_rebuild_undersized_grids(
        self,
        prices: Dict[str, float],
        equity: float
    ) -> List[str]:
        """
        Validate grid sizing and rebuild grids with investment_per_grid too small.

        This prevents the "notional < $10" skip issue where restored grids have
        stale/tiny investment_per_grid values that cause all orders to be skipped.

        Rebuild triggers:
        1. current_investment_per_grid < MIN_INVESTMENT_PER_GRID ($12)
        2. current_investment_per_grid < 25% of expected (stale grid detection)

        Args:
            prices: Current price for each symbol
            equity: Total account equity

        Returns:
            List of symbols that were rebuilt
        """
        rebuilt_symbols = []
        grid_configs = getattr(self.config, 'GRID_CONFIGS', DEFAULT_GRID_CONFIGS)

        for symbol, state in list(self.grid_strategy.grids.items()):
            if symbol not in prices:
                logger.warning(f"[GRID-VALIDATE] Cannot validate {symbol}: no price data")
                continue

            current_price = prices[symbol]
            current_investment = state.config.investment_per_grid

            # Calculate expected investment_per_grid from current equity
            template = grid_configs.get(symbol, {
                "num_grids": 10,
                "range_pct": 0.05,
                "investment_ratio": 0.25
            })
            num_grids = template.get("num_grids", 10)
            investment_ratio = template.get("investment_ratio", 0.25)
            expected_investment = (equity * investment_ratio) / num_grids

            # Check trigger conditions
            below_minimum = current_investment < MIN_INVESTMENT_PER_GRID
            below_expected = current_investment < (expected_investment * STALE_GRID_THRESHOLD)

            if not below_minimum and not below_expected:
                # Grid sizing is OK - log verification
                example_qty = current_investment / current_price
                example_notional = example_qty * current_price
                logger.debug(
                    f"[GRID-VALIDATE] {symbol} OK: investment=${current_investment:.2f}, "
                    f"expected=${expected_investment:.2f}, notional=${example_notional:.2f}"
                )
                continue

            # Rebuild needed
            reason = "below_minimum" if below_minimum else "stale_grid"
            logger.warning(
                f"[GRID] Rebuilding {symbol}: investment_per_grid too small "
                f"(current=${current_investment:.2f}, expected=${expected_investment:.2f}, "
                f"min=${MIN_INVESTMENT_PER_GRID:.2f}, reason={reason})"
            )

            # Step 1: Cancel open orders for this symbol on Alpaca
            self._cancel_symbol_orders(symbol)

            # Step 2: Clear local tracking for this symbol
            self._clear_symbol_tracking(symbol)

            # Step 3: Preserve performance stats before rebuild
            old_profit = state.total_profit
            old_trades = state.completed_trades
            old_cycles = state.completed_cycles
            old_buys = state.total_buys
            old_sells = state.total_sells

            # Step 4: Create fresh grid with proper sizing
            range_pct = template.get("range_pct", 0.05)
            new_config = GridConfig(
                symbol=symbol,
                upper_price=current_price * (1 + range_pct / 2),
                lower_price=current_price * (1 - range_pct / 2),
                num_grids=num_grids,
                investment_per_grid=expected_investment
            )

            new_state = self.grid_strategy.create_grid(new_config, current_price)

            # Step 5: Restore preserved stats
            new_state.total_profit = old_profit
            new_state.completed_trades = old_trades
            new_state.completed_cycles = old_cycles
            new_state.total_buys = old_buys
            new_state.total_sells = old_sells

            # Step 6: Log verification
            example_qty = expected_investment / current_price
            example_notional = example_qty * current_price
            logger.info(
                f"[GRID-REBUILD] {symbol} rebuilt: "
                f"num_grids={num_grids}, investment=${expected_investment:.2f}, "
                f"example_qty={example_qty:.6f}, example_notional=${example_notional:.2f}"
            )

            rebuilt_symbols.append(symbol)

        # Save state if any grids were rebuilt
        if rebuilt_symbols:
            self.grid_strategy.save_state()
            logger.info(f"[GRID-REBUILD] Saved state after rebuilding {len(rebuilt_symbols)} grids")

        return rebuilt_symbols

    def _cancel_symbol_orders(self, symbol: str) -> int:
        """
        Cancel all open orders for a symbol on Alpaca.

        Returns number of orders cancelled.
        """
        cancelled = 0
        try:
            # Get all open orders for this symbol from Alpaca
            open_orders = self.client.get_open_orders(symbols=[symbol])

            for order in open_orders:
                order_id = order.get('id')
                if order_id:
                    if self.client.cancel_order(order_id):
                        cancelled += 1
                        logger.info(f"[GRID-REBUILD] Cancelled order {order_id[:8]} for {symbol}")

            if cancelled > 0:
                logger.info(f"[GRID-REBUILD] Cancelled {cancelled} open orders for {symbol}")

        except Exception as e:
            logger.error(f"[GRID-REBUILD] Error cancelling orders for {symbol}: {e}")

        return cancelled

    def _clear_symbol_tracking(self, symbol: str) -> None:
        """
        Clear local order tracking for a symbol.

        Removes entries from:
        - grid_strategy.open_limit_orders
        - grid_strategy.pending_orders
        """
        symbol_normalized = normalize_symbol(symbol)

        # Clear open limit orders for this symbol
        keys_to_remove = [
            key for key, order in self.grid_strategy.open_limit_orders.items()
            if normalize_symbol(order.symbol) == symbol_normalized
        ]
        for key in keys_to_remove:
            del self.grid_strategy.open_limit_orders[key]
        if keys_to_remove:
            logger.info(f"[GRID-REBUILD] Cleared {len(keys_to_remove)} open limit order entries for {symbol}")

        # Clear pending orders for this symbol
        pending_to_remove = [
            oid for oid, pending in self.grid_strategy.pending_orders.items()
            if normalize_symbol(pending.symbol) == symbol_normalized
        ]
        for oid in pending_to_remove:
            del self.grid_strategy.pending_orders[oid]
        if pending_to_remove:
            logger.info(f"[GRID-REBUILD] Cleared {len(pending_to_remove)} pending order entries for {symbol}")

    def evaluate(self, symbol: str, current_price: float, position_qty: float) -> Dict:
        """
        Evaluate grid and determine trading action.

        Args:
            symbol: Trading symbol
            current_price: Current market price
            position_qty: Current position quantity

        Returns:
            Dict with action, order details, and grid state
        """
        if not self.grids_initialized:
            return {
                "action": "WAIT",
                "reason": "Grids not yet initialized",
                "grid_active": False
            }

        return self.grid_strategy.evaluate_grid(symbol, current_price, position_qty)

    def record_fill(self, symbol: str, side: str, price: float, qty: float, order_id: str, source: str = "grid") -> Optional[float]:
        """Record that an order was filled and return profit if applicable.

        Args:
            source: "grid", "stop_loss", or "windfall" - affects cycle counting
        """
        profit = self.grid_strategy.record_fill(symbol, side, price, qty, order_id, source=source)

        if profit:
            self.total_profit += profit
            self.total_trades += 1

        # Log to database for persistent history
        try:
            database.record_trade(
                symbol=symbol,
                side=side,
                quantity=qty,
                price=price,
                order_id=order_id,
                profit=profit or 0,
                source=source
            )
        except Exception as e:
            logger.warning(f"Failed to log trade to database: {e}")

        # Save grid state after each fill for persistence across restarts
        self.grid_strategy.save_state()

        return profit

    def get_grid_summary(self, symbol: str) -> Dict:
        """Get summary of grid state for dashboard."""
        return self.grid_strategy.get_grid_summary(symbol)

    def get_all_grid_summaries(self) -> Dict[str, Dict]:
        """Get summaries for all grids."""
        return {symbol: self.get_grid_summary(symbol) for symbol in self.symbols}

    def update_risk_state(self, equity: float) -> None:
        """
        Update risk management state with current equity.
        Called on each bar to track drawdown and daily P&L.
        """
        today = datetime.now().date()

        # Reset daily tracking at start of new day
        if self.last_trading_day != today:
            self.daily_pnl = 0.0
            self.last_trading_day = today
            self.daily_limit_hit = False

            # Use Alpaca's last_equity for true daily P/L (previous day's close)
            alpaca_last_equity = self._get_alpaca_last_equity()
            if alpaca_last_equity and alpaca_last_equity > 0:
                self.daily_start_equity = alpaca_last_equity
                logger.info(f"New trading day - using Alpaca last_equity: ${self.daily_start_equity:,.2f}")
            else:
                self.daily_start_equity = equity
                logger.info(f"New trading day - using current equity: ${self.daily_start_equity:,.2f}")

            self._save_daily_equity(equity, is_new_day=True)  # Persist for tracking

        # Initialize starting equity
        if self.starting_equity == 0:
            self.starting_equity = equity
            self.peak_equity = equity
            logger.info(f"Initial equity set: ${equity:,.2f}")

        # Update peak equity for drawdown calculation
        if equity > self.peak_equity:
            self.peak_equity = equity

        # Calculate current daily P&L
        if self.daily_start_equity > 0:
            self.daily_pnl = (equity - self.daily_start_equity) / self.daily_start_equity

        # Periodically save daily equity state (every update to ensure persistence)
        self._save_daily_equity(equity)

        # Log equity snapshot to database (every minute based on timestamp)
        now = datetime.now()
        if not hasattr(self, '_last_equity_snapshot') or (now - self._last_equity_snapshot).seconds >= 60:
            try:
                # Get account details for full snapshot
                account = self.client.trading_client.get_account()
                database.record_equity_snapshot(
                    equity=equity,
                    cash=float(account.cash) if account.cash else None,
                    buying_power=float(account.buying_power) if account.buying_power else None,
                    daily_pnl=self.daily_pnl * self.daily_start_equity if self.daily_start_equity else None,
                    daily_pnl_pct=self.daily_pnl * 100 if self.daily_pnl else None
                )
                self._last_equity_snapshot = now
            except Exception as e:
                logger.debug(f"Failed to log equity snapshot: {e}")

    def check_circuit_breakers(self, equity: float) -> Dict[str, Any]:
        """
        Check all circuit breakers and return status.

        Returns:
            Dict with 'should_stop', 'reason', and detailed status
        """
        result = {
            'should_stop': False,
            'reason': None,
            'daily_pnl': self.daily_pnl,
            'drawdown': 0.0,
            'daily_limit_hit': self.daily_limit_hit,
            'max_drawdown_hit': self.max_drawdown_hit
        }

        # Check daily loss limit
        if self.daily_start_equity > 0:
            if self.daily_pnl < -self.daily_loss_limit:
                if not self.daily_limit_hit:  # Only save on first trigger
                    self.daily_limit_hit = True
                    self._save_circuit_breaker_state()  # Persist to disk
                    logger.warning("CIRCUIT BREAKER: Daily loss limit triggered and saved to disk")
                result['should_stop'] = True
                result['reason'] = f"Daily loss limit hit: {self.daily_pnl:.2%} (limit: -{self.daily_loss_limit:.0%})"

        # Check max drawdown from peak
        if self.peak_equity > 0:
            drawdown = (self.peak_equity - equity) / self.peak_equity
            result['drawdown'] = drawdown

            if drawdown > self.max_drawdown:
                if not self.max_drawdown_hit:  # Only save on first trigger
                    self.max_drawdown_hit = True
                    self._save_circuit_breaker_state()  # Persist to disk
                    logger.warning("CIRCUIT BREAKER: Max drawdown triggered and saved to disk")
                result['should_stop'] = True
                result['reason'] = f"Max drawdown hit: {drawdown:.2%} (limit: {self.max_drawdown:.0%})"

        return result

    def check_stop_loss(self, symbol: str, current_price: float) -> Dict[str, Any]:
        """
        Check if stop-loss should be triggered for a symbol.

        Stop-loss triggers when price drops 10% below the grid's lower boundary.
        This protects against trend breakdowns.

        Returns:
            Dict with 'triggered', 'stop_price', 'reason'
        """
        result = {
            'triggered': False,
            'stop_price': 0.0,
            'reason': None
        }

        # Check if already triggered for this symbol
        if self.stop_loss_triggered.get(symbol, False):
            result['triggered'] = True
            result['reason'] = "Stop-loss already triggered"
            return result

        # Get grid state
        if symbol not in self.grid_strategy.grids:
            return result

        grid_state = self.grid_strategy.grids[symbol]
        lower_price = grid_state.config.lower_price

        # Stop price is 10% below the grid's lower boundary
        stop_price = lower_price * (1 - self.stop_loss_pct)
        result['stop_price'] = stop_price

        if current_price < stop_price:
            self.stop_loss_triggered[symbol] = True
            self._save_circuit_breaker_state()  # Persist to disk
            result['triggered'] = True
            result['reason'] = f"Price ${current_price:,.2f} below stop ${stop_price:,.2f} (grid lower: ${lower_price:,.2f})"
            logger.warning(f"STOP-LOSS TRIGGERED for {symbol}: {result['reason']}")
            logger.warning(f"Stop-loss saved to disk. Use /api/risk/reset to resume trading {symbol}")

        return result

    def get_risk_status(self, equity: float) -> Dict[str, Any]:
        """Get comprehensive risk status for dashboard."""
        circuit = self.check_circuit_breakers(equity)

        # Calculate daily P/L in dollars
        daily_pnl_dollars = equity - self.daily_start_equity if self.daily_start_equity > 0 else 0

        # Calculate all-time P/L (since Nov 24)
        alltime_pnl = equity - self.alltime_starting_equity if self.alltime_starting_equity > 0 else 0
        alltime_pnl_pct = (alltime_pnl / self.alltime_starting_equity * 100) if self.alltime_starting_equity > 0 else 0

        # Calculate grid P/L (since Dec 2)
        grid_pnl = equity - self.grid_starting_equity if self.grid_starting_equity > 0 else 0
        grid_pnl_pct = (grid_pnl / self.grid_starting_equity * 100) if self.grid_starting_equity > 0 else 0

        return {
            'peak_equity': self.peak_equity,
            'starting_equity': self.alltime_starting_equity,  # Use all-time as primary
            'starting_date': self.alltime_starting_date,
            'daily_start_equity': self.daily_start_equity,  # Alpaca's last_equity
            'daily_pnl': daily_pnl_dollars,  # Dollar amount
            'daily_pnl_pct': self.daily_pnl * 100,
            # All-time performance (since Nov 24)
            'alltime_starting_equity': self.alltime_starting_equity,
            'alltime_starting_date': self.alltime_starting_date,
            'alltime_pnl': alltime_pnl,
            'alltime_pnl_pct': alltime_pnl_pct,
            # Grid trading performance (since Dec 2)
            'grid_starting_equity': self.grid_starting_equity,
            'grid_starting_date': self.grid_starting_date,
            'grid_pnl': grid_pnl,
            'grid_pnl_pct': grid_pnl_pct,
            # Risk metrics
            'drawdown_pct': circuit['drawdown'] * 100,
            'daily_limit_hit': self.daily_limit_hit,
            'max_drawdown_hit': self.max_drawdown_hit,
            'stop_losses_triggered': list(self.stop_loss_triggered.keys()),
            'trading_halted': circuit['should_stop'],
            'halt_reason': circuit['reason'],
            'thresholds': {
                'stop_loss_pct': self.stop_loss_pct * 100,
                'daily_loss_limit_pct': self.daily_loss_limit * 100,
                'max_drawdown_pct': self.max_drawdown * 100
            }
        }


async def run_blocking(func, *args, **kwargs):
    """
    Run a blocking function in a thread pool to avoid blocking the event loop.

    This prevents Alpaca API calls from freezing the FastAPI server when
    the Alpaca API is slow or unresponsive.
    """
    return await asyncio.to_thread(func, *args, **kwargs)


async def cancel_grid_buy_limits(bot: GridTradingBot, symbols: list, client) -> int:
    """
    Cancel grid-owned BUY limit orders on RISK_OFF entry.

    SAFETY: Only cancels orders that are provably grid-owned by matching
    Alpaca order IDs against local open_limit_orders tracking.
    Orders not in local tracking are logged but NOT cancelled.

    Returns:
        Number of orders cancelled
    """
    cancelled = 0
    total_notional = 0.0

    for symbol in symbols:
        try:
            # Get Alpaca open orders for this symbol
            alpaca_orders = await run_blocking(client.get_open_orders, [symbol])

            for order in alpaca_orders:
                # Only process BUYs
                # NOTE: AlpacaClient.get_open_orders returns dicts (not Alpaca order objects)
                order_side = str(order.get('side', '')).lower() if isinstance(order, dict) else str(getattr(order, 'side', '')).lower()
                if 'buy' not in order_side:
                    continue

                order_id = str(order.get('id')) if isinstance(order, dict) else str(getattr(order, 'id', ''))
                if not order_id:
                    continue

                # SAFETY: Search open_limit_orders VALUES for this order_id
                # Keys are symbol:side:level_id, values are OpenLimitOrder dataclass
                is_grid_owned = False
                matching_key = None

                for key, tracked in bot.grid_strategy.open_limit_orders.items():
                    # Use attribute access (dataclass), not dict .get()
                    if tracked.order_id == order_id and tracked.side == "buy":
                        is_grid_owned = True
                        matching_key = key
                        break

                if is_grid_owned:
                    # Cancel the order
                    try:
                        await run_blocking(client.trading_client.cancel_order_by_id, order_id)
                        cancelled += 1

                        # Calculate notional for telemetry
                        if isinstance(order, dict):
                            order_qty = float(order.get('qty') or order.get('filled_qty') or 0)
                            order_price = float(order.get('limit_price') or 0)
                        else:
                            order_qty = float(getattr(order, 'qty', 0) or getattr(order, 'filled_qty', 0) or 0)
                            order_price = float(getattr(order, 'limit_price', 0) or 0)
                        notional = order_qty * order_price
                        total_notional += notional

                        # Record in overlay telemetry
                        bot.risk_overlay.record_cancelled_limit(symbol, notional)

                        # Remove from local tracking
                        del bot.grid_strategy.open_limit_orders[matching_key]

                        logger.info(f"[RISK] Cancelled grid BUY limit {order_id[:8]} for {symbol} (${notional:.2f})")

                    except Exception as e:
                        logger.error(f"[RISK] Failed to cancel order {order_id}: {e}")
                else:
                    # NOT in local tracking - could be manual order or legacy
                    bot.risk_overlay.record_untracked_buy(symbol, order_id)

        except Exception as e:
            logger.error(f"[RISK] Error cancelling buy limits for {symbol}: {e}")

    if cancelled > 0:
        logger.warning(f"[RISK_OFF] Cancelled {cancelled} grid BUY limits (${total_notional:.2f} notional)")
        # Persist state after cancellations
        bot.grid_strategy.save_state()

    return cancelled


async def run_grid_bot(broadcast_update, broadcast_log):
    """Run the Grid Trading bot."""
    logger.info("Starting BlueBird Grid Trading Bot...")
    await broadcast_log("Starting GRID TRADING BOT")
    await broadcast_log("=" * 50)
    await broadcast_log("Strategy: Grid Trading (NO predictions)")
    await broadcast_log("Why: Model predicts SIDEWAYS 90%+ = perfect for grids")
    await broadcast_log("How: Buy dips, sell rips, profit from oscillations")
    await broadcast_log("=" * 50)

    try:
        config = UltraConfig()
        symbols = getattr(config, 'SYMBOLS', ['BTC/USD', 'ETH/USD'])

        await broadcast_log(f"Trading {len(symbols)} assets: {', '.join(symbols)}")

        # Initialize (skip_verify=True to avoid blocking event loop)
        client = AlpacaClient(config, skip_verify=True)
        bot = GridTradingBot(config, client)

        # Async initialization (moved from constructor to avoid blocking event loop)
        await broadcast_log("Loading order history from Alpaca...")
        await run_blocking(bot._load_confirmed_orders_from_alpaca)
        await run_blocking(bot._run_reconciliation)
        await broadcast_log("Order history loaded and reconciled")

        # Get account info (wrapped in run_blocking to avoid freezing event loop)
        try:
            account = await run_blocking(client.trading_client.get_account)
            equity = float(account.equity)
            buying_power = float(account.buying_power)
            await broadcast_log(f"Account Equity: ${equity:,.2f}")
            await broadcast_log(f"Buying Power: ${buying_power:,.2f}")
        except Exception as e:
            await broadcast_log(f"Failed to get account: {e}")
            equity = 90000  # Fallback
            buying_power = 180000

        # Warm up with historical data
        await broadcast_log("Warming up with historical data...")

        current_prices = {}

        for symbol in symbols:
            try:
                req = CryptoBarsRequest(
                    symbol_or_symbols=[symbol],
                    timeframe=TimeFrame.Minute,
                    start=datetime.now() - timedelta(minutes=100),
                    limit=100
                )
                # Wrap in run_blocking to avoid freezing event loop during warmup
                bars = await run_blocking(lambda: client.data_client.get_crypto_bars(req))
                if symbol in bars.data:
                    for bar in bars.data[symbol]:
                        bot.add_bar(symbol, bar)

                    # Get current price from last bar
                    last_bar = bars.data[symbol][-1]
                    current_prices[symbol] = float(last_bar.close)

                    await broadcast_log(f"  {symbol}: {len(bars.data[symbol])} bars, price ${current_prices[symbol]:,.2f}")
            except Exception as e:
                await broadcast_log(f"  {symbol}: Failed ({e})")

        # Initialize grids with current prices
        await broadcast_log("Initializing grid levels...")
        bot.initialize_grids(current_prices, equity)

        # Log grid configuration
        for symbol in symbols:
            summary = bot.get_grid_summary(symbol)
            if 'range' in summary:
                r = summary['range']
                await broadcast_log(f"  {symbol}: ${r['lower']:,.2f} - ${r['upper']:,.2f}")
                await broadcast_log(f"    {summary['levels']['pending_buys']} buy levels, {summary['levels']['pending_sells']} sell levels")

        await broadcast_log("Grid initialization complete. Starting live trading...")

        # Reconcile open limit orders with Alpaca (crash recovery)
        # This handles:
        # 1. Orders filled while bot was down -> applies fills
        # 2. Orders cancelled -> removes tracking
        # 3. Untracked orders -> optionally cancels
        if bot.use_limit_orders:
            await broadcast_log("Reconciling open orders with Alpaca...")
            recon_results = await bot.reconcile_open_orders_on_startup(client)
            if recon_results.get('filled'):
                await broadcast_log(f"  Applied {len(recon_results['filled'])} fills from offline period")
            if recon_results.get('cancelled'):
                await broadcast_log(f"  Removed {len(recon_results['cancelled'])} stale order trackings")
            if recon_results.get('still_open'):
                await broadcast_log(f"  {len(recon_results['still_open'])} orders still open on Alpaca")
            await broadcast_log("Open order reconciliation complete")

        # Main trading loop
        from alpaca.data.live import CryptoDataStream

        # Heartbeat tracking for stream health monitoring
        last_bar_time = [datetime.now()]  # Use list for mutable reference in nested function
        STALE_THRESHOLD_SECONDS = 180  # 3 minutes without a bar = stale

        # NOTE: Stream is created inside the reconnection loop below.
        # We define handle_bar here so it's available for subscription.

        async def handle_bar(bar):
            """Handle incoming bar with grid evaluation and risk management."""
            # Update heartbeat timestamp
            last_bar_time[0] = datetime.now()

            symbol = bar.symbol
            current_price = float(bar.close)

            logger.info(f"[GRID] {symbol} @ ${current_price:,.2f}")

            if symbol not in symbols:
                return

            # Periodic reconciliation check (every 5 minutes)
            # Run in thread to avoid blocking event loop during Alpaca API calls
            await run_blocking(bot.check_periodic_reconciliation)

            # === FAST FILL DETECTION (Near Real-Time) ===
            # Check for limit order fills every ~10s (rate-limited internally)
            # Run EARLY in tick so grid state is current before evaluating new orders
            if bot.use_limit_orders and getattr(config, 'ENABLE_FAST_FILL_CHECK', True):
                fill_result = await bot.check_limit_order_fills_fast(
                    client=client,
                    max_checks_per_cycle=getattr(config, 'FAST_FILL_MAX_CHECKS_PER_CYCLE', 5),
                    min_interval_seconds=getattr(config, 'FAST_FILL_INTERVAL_SECONDS', 10.0),
                    min_order_age_seconds=getattr(config, 'FAST_FILL_MIN_ORDER_AGE_SECONDS', 10.0)
                )
                if fill_result.get('fills_detected', 0) > 0 or fill_result.get('partials_detected', 0) > 0:
                    fills = fill_result.get('fills_detected', 0)
                    partials = fill_result.get('partials_detected', 0)
                    await broadcast_log(f"[FAST-FILL] Detected {fills} fill(s), {partials} partial(s)")

            # Add bar to bot
            bot.add_bar(symbol, bar)

            # === TIME FILTER CHECK ===
            time_status = bot.is_optimal_trading_time()
            current_hour = time_status.get('hour', 12)
            skip_trading = not time_status['should_trade']

            if skip_trading:
                await broadcast_log(f"[TIME] Skipping trade: {time_status['reason']}")

            # === CORRELATION RISK CHECK ===
            corr_adjustment = bot.get_correlation_risk_adjustment(symbol)
            correlations = bot.calculate_correlations()

            # === MOMENTUM FILTER ===
            momentum_status = bot.get_momentum_filter(symbol)

            # === REGIME DETECTION (Smart Grid) ===
            regime_status = bot.detect_regime(symbol)
            regime_allow_buy = regime_status.get('allow_buy', True)
            regime_allow_sell = regime_status.get('allow_sell', True)
            current_regime = regime_status.get('regime', 'UNKNOWN')
            regime_adx = regime_status.get('adx', 0)

            if regime_status.get('is_strong_trend'):
                await broadcast_log(f"[REGIME] {symbol}: {regime_status['reason']}")

            # === RISK OVERLAY EVALUATION (Crash Protection) ===
            # Check for API command files (manual overrides)
            bot.risk_overlay.check_command_file()

            # Collect signals and evaluate state machine
            max_corr = max(correlations.values()) if correlations else 0.0

            # Determine ADX direction from regime (TRENDING_DOWN = "down", TRENDING_UP = "up")
            if current_regime == MarketRegime.TRENDING_DOWN:
                adx_direction = "down"
            elif current_regime == MarketRegime.TRENDING_UP:
                adx_direction = "up"
            else:
                adx_direction = "neutral"

            # Build current prices dict for all symbols
            current_prices = {}
            for s in bot.symbols:
                if s in bot.bars and len(bot.bars[s]) > 0:
                    current_prices[s] = float(bot.bars[s][-1].close)

            # momentum_status['momentum'] is in PERCENT units (e.g., -2.5 means -2.5%)
            # risk_overlay expects DECIMAL units (e.g., -0.025 means -2.5%)
            # Convert at this boundary to keep internal logic unchanged
            momentum_pct = momentum_status.get("momentum", 0)  # Percent: -2.5 = -2.5%
            momentum_decimal = momentum_pct / 100.0            # Decimal: -0.025 = -2.5%

            overlay_signals = {
                "momentum": momentum_decimal,  # risk_overlay expects decimal
                "adx": regime_adx,
                "adx_direction": adx_direction,
                "max_correlation": max_corr,
                "current_prices": current_prices,
            }

            previous_overlay_mode = bot.risk_overlay.mode
            overlay_mode = bot.risk_overlay.evaluate(overlay_signals)

            # Log mode transitions
            if overlay_mode != previous_overlay_mode:
                await broadcast_log(f"[RISK] Mode changed: {previous_overlay_mode.value} -> {overlay_mode.value}")
                if overlay_mode == RiskMode.RISK_OFF:
                    # Cancel grid-owned BUY limits on RISK_OFF entry
                    await cancel_grid_buy_limits(bot, bot.symbols, client)

            # Get overlay gates
            overlay_allow_buy, overlay_buy_reason = bot.risk_overlay.allows_buy(symbol)
            overlay_allow_sell, overlay_sell_reason = bot.risk_overlay.allows_sell(symbol)
            overlay_position_mult = bot.risk_overlay.get_position_multiplier()

            # In RISK_OFF: override other filters to ALLOW sells (exit opportunities)
            if overlay_mode == RiskMode.RISK_OFF:
                regime_allow_sell = True  # Override regime block on sells

            # Get current positions (run in thread to avoid blocking event loop)
            try:
                positions = await run_blocking(client.get_positions)
                current_positions = {p.symbol: p for p in positions}
                num_positions = len(positions)
            except:
                current_positions = {}
                num_positions = 0

            # Get Alpaca open orders (for inventory gating - single API call per bar)
            try:
                alpaca_open_orders = await run_blocking(client.get_open_orders, symbols)
            except:
                alpaca_open_orders = []

            # Get account (run in thread to avoid blocking event loop)
            try:
                account = await run_blocking(client.trading_client.get_account)
                equity = float(account.equity)
            except:
                equity = 90000

            # Check position quantity for this symbol
            alpaca_symbol = symbol.replace('/', '')
            position_qty = 0.0
            position_data = None
            if alpaca_symbol in current_positions:
                position_qty = float(current_positions[alpaca_symbol].qty)
                position_data = current_positions[alpaca_symbol]
            elif symbol in current_positions:
                position_qty = float(current_positions[symbol].qty)
                position_data = current_positions[symbol]

            # === WINDFALL PROFIT-TAKING CHECK ===
            # Check if any position qualifies for windfall profit capture
            windfall_executed = None
            if position_data and position_qty > 0:
                # Build position dict for windfall check
                pos_dict = {
                    'symbol': symbol,
                    'qty': position_qty,
                    'unrealized_plpc': float(position_data.unrealized_plpc) if hasattr(position_data, 'unrealized_plpc') else 0,
                    'unrealized_pl': float(position_data.unrealized_pl) if hasattr(position_data, 'unrealized_pl') else 0,
                    'avg_entry_price': float(position_data.avg_entry_price) if hasattr(position_data, 'avg_entry_price') else 0,
                    'current_price': current_price
                }

                windfall_check = bot.check_windfall_opportunity(symbol, pos_dict, current_price)

                if windfall_check.get('should_sell'):
                    # Execute windfall partial sell
                    sell_portion = windfall_check.get('sell_portion', 0.70)
                    sell_qty = position_qty * sell_portion
                    trigger_type = windfall_check.get('trigger_type', 'unknown')
                    unrealized_pct = windfall_check.get('unrealized_pct', 0)
                    rsi = windfall_check.get('rsi', 50)

                    await broadcast_log(f"[WINDFALL] {symbol} triggered! {unrealized_pct:.1f}% gain, RSI={rsi:.0f}")
                    await broadcast_log(f"[WINDFALL] Selling {sell_portion*100:.0f}% ({sell_qty:.6f}) - Trigger: {trigger_type}")

                    try:
                        order = MarketOrderRequest(
                            symbol=symbol,
                            qty=round(sell_qty, 6),
                            side=OrderSide.SELL,
                            time_in_force=TimeInForce.GTC
                        )
                        result = await run_blocking(client.trading_client.submit_order, order)

                        # Verify order fill (async version avoids blocking thread pool)
                        verification = await client.verify_order_fill_async(str(result.id), max_wait_seconds=5)

                        if verification['confirmed']:
                            fill_price = verification['filled_avg_price']
                            fill_qty = verification['filled_qty']

                            # Calculate profit (entry to sell price, minus fees)
                            entry_price = pos_dict['avg_entry_price']
                            gross_profit = (fill_price - entry_price) * fill_qty
                            # Windfall sells are market orders (taker fee)
                            taker_fee = getattr(config, 'TAKER_FEE_PCT', 0.0025)
                            fee = fill_price * fill_qty * taker_fee
                            net_profit = gross_profit - fee

                            # Log the windfall transaction
                            bot.log_windfall_transaction(
                                symbol=symbol,
                                qty=fill_qty,
                                price=fill_price,
                                profit=net_profit,
                                trigger_type=trigger_type,
                                unrealized_pct=unrealized_pct,
                                rsi=rsi
                            )

                            # Set cooldown for this symbol
                            bot.set_windfall_cooldown(symbol)

                            # Add to confirmed orders
                            verification['windfall'] = True
                            verification['profit'] = net_profit
                            bot.add_confirmed_order(verification)

                            windfall_executed = {
                                'action': 'WINDFALL_SELL',
                                'symbol': symbol,
                                'qty': fill_qty,
                                'price': fill_price,
                                'profit': net_profit,
                                'trigger_type': trigger_type,
                                'unrealized_pct': unrealized_pct,
                                'rsi': rsi,
                                'timestamp': datetime.now().isoformat()
                            }

                            await broadcast_log(f"[WINDFALL] SUCCESS: Sold {fill_qty:.6f} {symbol} @ ${fill_price:,.2f}")
                            await broadcast_log(f"[WINDFALL] Captured: ${net_profit:+,.2f} profit!")

                            # Update position qty for remaining grid evaluation
                            position_qty = position_qty - fill_qty

                        else:
                            await broadcast_log(f"[WINDFALL] Order not confirmed: {verification.get('status', 'unknown')}")

                    except Exception as e:
                        logger.error(f"[WINDFALL] Order failed: {e}")
                        await broadcast_log(f"[WINDFALL] Order failed: {e}")

            # === RISK MANAGEMENT CHECKS ===
            # Update risk state (wrapped to avoid blocking - contains Alpaca API call)
            await run_blocking(bot.update_risk_state, equity)

            # Check circuit breakers (daily loss, max drawdown)
            circuit_status = bot.check_circuit_breakers(equity)
            if circuit_status['should_stop']:
                await broadcast_log(f"[RISK] CIRCUIT BREAKER: {circuit_status['reason']}")
                await broadcast_log(f"[RISK] Trading halted. Manual intervention required.")
                # Still broadcast status but don't trade
                evaluation = {'action': 'HALTED', 'reason': circuit_status['reason']}
            else:
                # Check stop-loss for this symbol
                stop_loss_status = bot.check_stop_loss(symbol, current_price)

                if stop_loss_status['triggered'] and position_qty > 0:
                    # EMERGENCY SELL - close entire position
                    await broadcast_log(f"[RISK] STOP-LOSS TRIGGERED: {stop_loss_status['reason']}")
                    await broadcast_log(f"[RISK] Closing position: {position_qty} {symbol}")

                    try:
                        order = MarketOrderRequest(
                            symbol=symbol,
                            qty=round(position_qty, 6),
                            side=OrderSide.SELL,
                            time_in_force=TimeInForce.GTC
                        )
                        result = await run_blocking(client.trading_client.submit_order, order)
                        await broadcast_log(f"[RISK] STOP-LOSS EXECUTED: Sold {position_qty:.6f} {symbol} @ ${current_price:,.2f}")

                        # Record as emergency sell (source="stop_loss" prevents inflating completed_cycles)
                        bot.record_fill(symbol, "sell", current_price, position_qty, str(result.id), source="stop_loss")

                        evaluation = {
                            'action': 'STOP_LOSS',
                            'reason': stop_loss_status['reason'],
                            'order_details': {'quantity': position_qty, 'price': current_price}
                        }
                    except Exception as e:
                        await broadcast_log(f"[RISK] STOP-LOSS ORDER FAILED: {e}")
                        evaluation = {'action': 'STOP_LOSS_FAILED', 'reason': str(e)}
                else:
                    # === RESTING LIMIT MODE (New - proactive maker orders) ===
                    if bot.use_limit_orders:
                        # Use new method that finds maker-safe levels AHEAD of price
                        desired = bot.grid_strategy.get_desired_limit_orders(
                            symbol=symbol,
                            current_price=current_price,
                            position_qty=position_qty,
                            maker_buffer_bps=bot.maker_buffer_bps
                        )

                        logger.info(f"[GRID] {symbol} @ ${current_price:,.2f} (resting limit mode) - {desired['reason']}")

                        # Get target allocation for this symbol from config
                        grid_configs = getattr(bot.config, 'GRID_CONFIGS', DEFAULT_GRID_CONFIGS)
                        target_allocation = grid_configs.get(symbol, {}).get('investment_ratio', 0.25)
                        max_allocation = target_allocation * 1.1  # Allow 10% buffer
                        current_position_value = position_qty * current_price
                        current_allocation = current_position_value / equity if equity > 0 else 0

                        # Apply time quality adjustment
                        time_quality = time_status.get('time_quality', 1.0)

                        # === PLACE RESTING BUY LIMIT ORDER ===
                        if desired['desired_buy'] and not skip_trading:
                            buy_details = desired['desired_buy']
                            level_id = buy_details['level_id']
                            grid_price = buy_details['price']
                            base_qty = buy_details['quantity']

                            # Check consecutive down bars (crash guard)
                            down_bar_allow, down_bar_count = bot.check_consecutive_down_bars(symbol)

                            # Apply filters (regime, momentum, allocation, RISK OVERLAY, consecutive down bars)
                            if not overlay_allow_buy:
                                # RISK OVERLAY GATE - highest priority
                                notional = buy_details['quantity'] * buy_details['price']
                                bot.risk_overlay._record_blocked_buy(symbol, notional, overlay_buy_reason)
                                logger.info(f"[RISK] Blocked resting BUY {symbol}: {overlay_buy_reason}")
                            elif not regime_allow_buy:
                                logger.debug(f"[REGIME] Skipping resting BUY {symbol}: {regime_status['reason']}")
                            elif not momentum_status['allow_buy']:
                                logger.debug(f"[MOM] Skipping resting BUY {symbol}: {momentum_status['reason']}")
                            elif not down_bar_allow:
                                logger.info(f"[DOWN] Skipping resting BUY {symbol}: {down_bar_count} consecutive down bars")
                            elif current_allocation >= max_allocation:
                                logger.debug(f"[ALLOC] Skipping resting BUY {symbol}: {current_allocation:.1%} >= {max_allocation:.1%}")
                            else:
                                # Apply adjustments (include overlay position multiplier and regime size multiplier)
                                regime_size_mult = regime_status.get('size_mult', 1.0)
                                qty = base_qty * time_quality * corr_adjustment * overlay_position_mult * regime_size_mult

                                try:
                                    limit_price = round_limit_price(symbol, grid_price, bot.config)
                                    rounded_qty = round_qty(symbol, qty, bot.config)

                                    # Check minimum notional ($10 for Alpaca crypto)
                                    MIN_NOTIONAL = 10.0
                                    notional = rounded_qty * limit_price
                                    if rounded_qty > 0 and notional < MIN_NOTIONAL:
                                        logger.info(f"[GRID] Skipping resting BUY {symbol}: notional ${notional:.2f} < ${MIN_NOTIONAL}")
                                        rounded_qty = 0  # Skip by zeroing qty

                                    if rounded_qty > 0:
                                        order = LimitOrderRequest(
                                            symbol=symbol,
                                            qty=rounded_qty,
                                            side=OrderSide.BUY,
                                            time_in_force=TimeInForce.GTC,
                                            limit_price=limit_price
                                        )
                                        result = await run_blocking(client.trading_client.submit_order, order)
                                        order_id = str(result.id)

                                        # Register for tracking
                                        bot.grid_strategy.register_open_limit_order(
                                            order_id=order_id, symbol=symbol, side="buy",
                                            level_id=level_id, level_price=grid_price,
                                            limit_price=limit_price, qty=rounded_qty, source="grid"
                                        )
                                        bot.grid_strategy.register_pending_order(
                                            order_id=order_id, symbol=symbol, side="buy",
                                            intended_level_price=grid_price,
                                            intended_level_id=level_id, source="grid",
                                            fee_type="maker"  # Limit orders are maker
                                        )

                                        logger.info(f"[BUY] Resting LIMIT: {rounded_qty:.6f} {symbol} @ ${limit_price:,.2f}")
                                        await broadcast_log(f"[GRID] Resting BUY {symbol}: {rounded_qty:.6f} @ ${limit_price:,.2f}")
                                        await broadcast_log(f"  Level: {level_id[:8]}...")

                                except Exception as e:
                                    logger.error(f"[BUY] Resting LIMIT failed for {symbol}: {e}")
                                    await broadcast_log(f"[ERROR] Resting BUY failed: {e}")

                        # === PLACE RESTING SELL LIMIT ORDER ===
                        if desired['desired_sell'] and not skip_trading:
                            sell_details = desired['desired_sell']
                            level_id = sell_details['level_id']
                            grid_price = sell_details['price']
                            base_qty = sell_details['quantity']

                            # Apply filters (regime, momentum, duplicate)
                            if not regime_allow_sell:
                                logger.debug(f"[REGIME] Skipping resting SELL {symbol}: {regime_status['reason']}")
                            elif not momentum_status['allow_sell']:
                                logger.debug(f"[MOM] Skipping resting SELL {symbol}: {momentum_status['reason']}")
                            elif bot.grid_strategy.has_open_order_for_level(symbol, "sell", level_id):
                                logger.info(f"[GRID] Skipping resting SELL {symbol}: open order exists for level {level_id[:8]}")
                            else:
                                # Apply adjustments
                                qty = base_qty * time_quality * corr_adjustment

                                try:
                                    limit_price = round_limit_price(symbol, grid_price, bot.config)
                                    rounded_qty = round_qty(symbol, qty, bot.config)

                                    # === INVENTORY GATING FOR SELL ORDERS (Alpaca source of truth) ===
                                    # Calculate reserved base from Alpaca open SELL limit orders
                                    # This is more accurate than internal tracking due to race conditions
                                    alpaca_symbol_normalized = symbol.replace('/', '')

                                    # Calculate reserved from Alpaca open SELL orders (source of truth)
                                    reserved_sells_alpaca = sum(
                                        (o.get('qty', 0) - o.get('filled_qty', 0))
                                        for o in alpaca_open_orders
                                        if o.get('symbol', '').replace('/', '') == alpaca_symbol_normalized
                                        and 'sell' in o.get('side', '').lower()
                                    )
                                    # Apply 1% safety buffer to prevent edge cases
                                    safety_buffer = 0.01 * position_qty if position_qty > 0 else 0
                                    effective_available = position_qty - reserved_sells_alpaca - safety_buffer

                                    # Cap order size to effective available inventory
                                    if rounded_qty > effective_available:
                                        capped_qty = floor_qty(symbol, effective_available, bot.config)
                                        if capped_qty > 0 and capped_qty < rounded_qty:
                                            logger.info(f"[GRID] Capping resting SELL {symbol}: pos={position_qty:.6f} reserved_alpaca={reserved_sells_alpaca:.6f} available={effective_available:.6f} desired={rounded_qty:.6f} capped={capped_qty:.6f}")
                                            rounded_qty = capped_qty
                                        else:
                                            # effective_available <= 0 OR capped_qty rounds to 0 - skip entirely
                                            logger.info(f"[GRID] Skipping resting SELL {symbol}: pos={position_qty:.6f} reserved_alpaca={reserved_sells_alpaca:.6f} available={effective_available:.6f} capped_to={capped_qty:.6f}")
                                            rounded_qty = 0  # Skip this order

                                    if rounded_qty > 0:
                                        order = LimitOrderRequest(
                                            symbol=symbol,
                                            qty=rounded_qty,
                                            side=OrderSide.SELL,
                                            time_in_force=TimeInForce.GTC,
                                            limit_price=limit_price
                                        )
                                        result = await run_blocking(client.trading_client.submit_order, order)
                                        order_id = str(result.id)

                                        # Register for tracking
                                        bot.grid_strategy.register_open_limit_order(
                                            order_id=order_id, symbol=symbol, side="sell",
                                            level_id=level_id, level_price=grid_price,
                                            limit_price=limit_price, qty=rounded_qty, source="grid"
                                        )
                                        bot.grid_strategy.register_pending_order(
                                            order_id=order_id, symbol=symbol, side="sell",
                                            intended_level_price=grid_price,
                                            intended_level_id=level_id, source="grid",
                                            fee_type="maker"  # Limit orders are maker
                                        )

                                        logger.info(f"[SELL] Resting LIMIT: {rounded_qty:.6f} {symbol} @ ${limit_price:,.2f}")
                                        await broadcast_log(f"[GRID] Resting SELL {symbol}: {rounded_qty:.6f} @ ${limit_price:,.2f}")
                                        await broadcast_log(f"  Level: {level_id[:8]}...")

                                except Exception as e:
                                    logger.error(f"[SELL] Resting LIMIT failed for {symbol}: {e}")
                                    await broadcast_log(f"[ERROR] Resting SELL failed: {e}")

                        # === OVERSHOOT HANDLING ===
                        if desired['no_eligible_levels']:
                            overshoot_mode = getattr(bot.config, 'LIMIT_ORDER_OVERSHOOT_MODE', 'wait')

                            # Track consecutive bars with no eligible levels
                            if not hasattr(bot, '_no_eligible_bars'):
                                bot._no_eligible_bars = {}
                            bot._no_eligible_bars[symbol] = bot._no_eligible_bars.get(symbol, 0) + 1

                            bars_threshold = getattr(bot.config, 'OVERSHOOT_BARS_THRESHOLD', 5)
                            cooldown_minutes = getattr(bot.config, 'OVERSHOOT_REBALANCE_COOLDOWN', 30)

                            # Check if we should rebalance
                            if overshoot_mode == 'rebalance' and bot._no_eligible_bars[symbol] >= bars_threshold:
                                # Check cooldown
                                if not hasattr(bot, '_last_rebalance_at'):
                                    bot._last_rebalance_at = {}
                                last_rebalance = bot._last_rebalance_at.get(symbol)
                                now = datetime.now()

                                can_rebalance = (
                                    last_rebalance is None or
                                    (now - last_rebalance).total_seconds() / 60 >= cooldown_minutes
                                )

                                if can_rebalance:
                                    # === RISK OVERLAY: Check if this is a rebalance-DOWN ===
                                    # Get current grid center to determine direction
                                    rebalance_blocked = False
                                    grid = bot.grid_strategy.grids.get(symbol)
                                    if grid:
                                        grid_center = (grid.config.upper_price + grid.config.lower_price) / 2
                                        is_rebalance_down = current_price < grid_center

                                        # Block rebalance-down in RISK_OFF and RECOVERY
                                        if is_rebalance_down:
                                            overlay_allow_rebalance, rebalance_reason = bot.risk_overlay.allows_rebalance_down(symbol)
                                            if not overlay_allow_rebalance:
                                                logger.warning(f"[RISK] Blocked rebalance-DOWN for {symbol}: {rebalance_reason}")
                                                await broadcast_log(f"[RISK] Blocked rebalance-DOWN: {rebalance_reason}")
                                                # Don't reset counter - keep waiting
                                                rebalance_blocked = True

                                if can_rebalance and not rebalance_blocked:
                                    # Cancel existing open orders for this symbol before rebalancing
                                    cancelled = 0
                                    for key, order in list(bot.grid_strategy.open_limit_orders.items()):
                                        if order.symbol == symbol:
                                            try:
                                                await run_blocking(client.trading_client.cancel_order_by_id, order.order_id)
                                                bot.grid_strategy.remove_open_limit_order(order.order_id)
                                                bot.grid_strategy.remove_pending_order(order.order_id)
                                                cancelled += 1
                                            except Exception as e:
                                                logger.warning(f"[REBALANCE] Failed to cancel order {order.order_id[:8]}: {e}")

                                    if cancelled > 0:
                                        logger.info(f"[REBALANCE] Cancelled {cancelled} open orders for {symbol}")

                                    # Trigger rebalance
                                    logger.info(f"[OVERSHOOT] {symbol} - No eligible levels for {bot._no_eligible_bars[symbol]} bars, rebalancing grid")
                                    await broadcast_log(f"[OVERSHOOT] {symbol} - Rebalancing grid around ${current_price:,.2f}")

                                    bot.grid_strategy.rebalance_grid(symbol, current_price, preserve_positions=True)
                                    bot._last_rebalance_at[symbol] = now
                                    bot._no_eligible_bars[symbol] = 0

                                    # Persist immediately after rebalance to avoid crash window
                                    bot.grid_strategy.save_state()
                                else:
                                    remaining = cooldown_minutes - (now - last_rebalance).total_seconds() / 60
                                    logger.debug(f"[OVERSHOOT] {symbol} - Rebalance on cooldown ({remaining:.1f}m remaining)")
                            elif overshoot_mode == 'wait':
                                logger.debug(f"[OVERSHOOT] {symbol} - Waiting for price to return ({bot._no_eligible_bars[symbol]} bars)")
                        else:
                            # Reset counter when we have eligible levels
                            if hasattr(bot, '_no_eligible_bars') and symbol in bot._no_eligible_bars:
                                bot._no_eligible_bars[symbol] = 0

                        # Set evaluation to HOLD for resting limit mode (no action-based handling needed)
                        evaluation = {'action': 'HOLD', 'mode': 'resting_limit'}

                    else:
                        # === MARKET ORDER MODE (Original - trigger after crossing) ===
                        evaluation = bot.evaluate(symbol, current_price, position_qty)

            action = evaluation.get('action', 'HOLD')
            order_details = evaluation.get('order_details')

            # DEBUG: Log the evaluation result (skip for resting limit mode)
            if action != 'HOLD' and evaluation.get('mode') != 'resting_limit':
                logger.info(f"[DEBUG] {symbol} action={action}, order_details={order_details is not None}, skip_trading={skip_trading}")

            trade_executed = None

            # Skip BUY/SELL action handling in resting limit mode (already handled above)
            if evaluation.get('mode') == 'resting_limit':
                pass  # Orders already placed above
            elif action == 'BUY' and order_details and not skip_trading:
                # Check consecutive down bars (crash guard)
                down_bar_allow, down_bar_count = bot.check_consecutive_down_bars(symbol)

                # RISK OVERLAY GATE - highest priority (blocks buys in RISK_OFF)
                if not overlay_allow_buy:
                    notional = order_details.get('quantity', 0) * order_details.get('price', current_price)
                    bot.risk_overlay._record_blocked_buy(symbol, notional, overlay_buy_reason)
                    logger.info(f"[RISK] Blocked market BUY {symbol}: {overlay_buy_reason}")
                    await broadcast_log(f"[RISK] Blocked BUY: {overlay_buy_reason}")
                # Check regime filter (Smart Grid - avoid buying in strong downtrend)
                elif not regime_allow_buy:
                    logger.info(f"[REGIME] Skipping BUY {symbol}: {regime_status['reason']}")
                    await broadcast_log(f"[REGIME] Skipping BUY: {regime_status['reason']}")
                # Check momentum filter
                elif not momentum_status['allow_buy']:
                    logger.info(f"[MOM] Skipping BUY {symbol}: {momentum_status['reason']}")
                    await broadcast_log(f"[MOM] Skipping BUY: {momentum_status['reason']}")
                # Check consecutive down bars
                elif not down_bar_allow:
                    logger.info(f"[DOWN] Skipping BUY {symbol}: {down_bar_count} consecutive down bars")
                    await broadcast_log(f"[DOWN] Skipping BUY: {down_bar_count} consecutive down bars")
                # Check allocation limit - prevent over-buying
                else:
                    # Get target allocation for this symbol from config
                    grid_configs = getattr(bot.config, 'GRID_CONFIGS', DEFAULT_GRID_CONFIGS)
                    target_allocation = grid_configs.get(symbol, {}).get('investment_ratio', 0.25)
                    max_allocation = target_allocation * 1.1  # Allow 10% buffer

                    # Calculate current allocation
                    current_position_value = position_qty * current_price
                    current_allocation = current_position_value / equity if equity > 0 else 0

                    if current_allocation >= max_allocation:
                        logger.info(f"[ALLOC] Skipping BUY {symbol}: Already at {current_allocation:.1%} (max: {max_allocation:.1%})")
                        await broadcast_log(f"[ALLOC] Skipping BUY {symbol}: {current_allocation:.1%} >= {max_allocation:.1%} limit")
                    else:
                        grid_price = order_details['price']
                        base_qty = order_details['quantity']
                        level_id = order_details.get('level_id')  # May be None for old grids

                        # Apply time quality, correlation, and regime size adjustments
                        time_quality = time_status.get('time_quality', 1.0)
                        regime_size_mult = regime_status.get('size_mult', 1.0)
                        qty = base_qty * time_quality * corr_adjustment * regime_size_mult

                        # Log adjustments if applied
                        if time_quality < 1.0 or corr_adjustment < 1.0 or regime_size_mult < 1.0:
                            await broadcast_log(f"[ADJ] Qty adjusted: {base_qty:.4f} -> {qty:.4f} (time: {time_quality:.0%}, corr: {corr_adjustment:.0%}, regime: {regime_size_mult:.0%})")

                        # === LIMIT ORDER PATH (Maker Fee Optimization) ===
                        if bot.use_limit_orders and level_id:
                            # IMPORTANT: If limit-orders are enabled but we skip due to guard/duplicate,
                            # do NOT fall back to market orders (that would negate the maker-fee intent).
                            if bot.grid_strategy.has_open_order_for_level(symbol, "buy", level_id):
                                logger.info(f"[GRID] Skipping BUY {symbol}: Open order exists for level {level_id[:8]}")
                                await broadcast_log(f"[GRID] Skipping duplicate order for level")
                            elif not can_place_limit_order(grid_price, current_price, "buy", bot.maker_buffer_bps):
                                logger.info(f"[GRID] Skipping BUY {symbol}: Grid price ${grid_price:,.2f} too close to market ${current_price:,.2f}")
                                await broadcast_log(f"[GRID] Skipping BUY: crossing guard (wait for next bar)")
                            else:
                                try:
                                    # Limit price = grid level price (preserve grid integrity)
                                    limit_price = round_limit_price(symbol, grid_price, bot.config)
                                    rounded_qty = round_qty(symbol, qty, bot.config)
                                    if rounded_qty <= 0:
                                        logger.warning(f"[BUY] Skipping LIMIT order: rounded_qty<=0 for {symbol} (raw={qty})")
                                        await broadcast_log(f"[GRID] BUY LIMIT skipped: qty too small after rounding")
                                    else:
                                        order = LimitOrderRequest(
                                            symbol=symbol,
                                            qty=rounded_qty,
                                            side=OrderSide.BUY,
                                            time_in_force=TimeInForce.GTC,
                                            limit_price=limit_price
                                        )
                                        result = await run_blocking(client.trading_client.submit_order, order)
                                        order_id = str(result.id)

                                        # Register open limit order (for duplicate prevention)
                                        bot.grid_strategy.register_open_limit_order(
                                            order_id=order_id,
                                            symbol=symbol,
                                            side="buy",
                                            level_id=level_id,
                                            level_price=grid_price,
                                            limit_price=limit_price,
                                            qty=rounded_qty,
                                            source="grid"
                                        )

                                        # Register pending order (for reconciliation)
                                        bot.grid_strategy.register_pending_order(
                                            order_id=order_id,
                                            symbol=symbol,
                                            side="buy",
                                            intended_level_price=grid_price,
                                            intended_level_id=level_id,
                                            source="grid",
                                            fee_type="maker"  # Limit orders are maker
                                        )

                                        logger.info(f"[BUY] LIMIT order: {rounded_qty:.6f} {symbol} @ ${limit_price:,.2f} (grid: ${grid_price:,.2f})")
                                        await broadcast_log(f"[GRID] BUY LIMIT {symbol}: {rounded_qty:.6f} @ ${limit_price:,.2f}")
                                        await broadcast_log(f"  Level: {level_id[:8]}... | Order: {order_id[:8]}...")

                                        # Note: Don't verify_order_fill for limit orders - they may not fill immediately
                                        # Fills will be detected via reconciliation

                                except Exception as e:
                                    logger.error(f"[BUY] LIMIT order failed for {symbol}: {e}")
                                    await broadcast_log(f"[ERROR] BUY LIMIT failed: {e}")

                        # === MARKET ORDER PATH (original behavior - taker fees) ===
                        elif not (bot.use_limit_orders and level_id):
                            try:
                                order = MarketOrderRequest(
                                    symbol=symbol,
                                    qty=round(qty, 6),
                                    side=OrderSide.BUY,
                                    time_in_force=TimeInForce.GTC
                                )
                                result = await run_blocking(client.trading_client.submit_order, order)
                                order_id = str(result.id)

                                # Register pending order BEFORE verify (for deterministic matching)
                                level_price = order_details['price']
                                bot.grid_strategy.register_pending_order(
                                    order_id=order_id,
                                    symbol=symbol,
                                    side="buy",
                                    intended_level_price=level_price,
                                    intended_level_id=level_id,
                                    source="grid",
                                    fee_type="maker"  # Limit orders are maker
                                )

                                # Verify order fill (async version avoids blocking thread pool)
                                verification = await client.verify_order_fill_async(order_id, max_wait_seconds=5)

                                if verification['confirmed']:
                                    # Use actual fill price from Alpaca
                                    fill_price = verification['filled_avg_price']
                                    fill_qty = verification['filled_qty']

                                    # Record fill with verified data (idempotent - uses apply_filled_order)
                                    bot.record_fill(symbol, "buy", fill_price, fill_qty, order_id)

                                    # Add to confirmed orders
                                    bot.add_confirmed_order(verification)

                                    trade_executed = {
                                        'action': 'BUY',
                                        'symbol': symbol,
                                        'price': fill_price,
                                        'qty': fill_qty,
                                        'grid_level': order_details.get('grid_level'),
                                        'timestamp': datetime.now().isoformat(),
                                        'order_id': order_id,
                                        'verified': True
                                    }

                                    logger.info(f"[BUY] {symbol}: {fill_qty:.4f} @ ${fill_price:,.2f} [VERIFIED]")
                                    await broadcast_log(f"[GRID] BUY {symbol}: {fill_qty:.4f} @ ${fill_price:,.2f}")
                                    await broadcast_log(f"  Grid level: {order_details.get('grid_level')} | Order ID: {order_id[:8]}...")
                                else:
                                    # Verify timed out - pending order remains for reconciliation
                                    logger.warning(f"[BUY] Order {order_id} not confirmed: {verification.get('reason', 'Unknown')} - pending for reconciliation")
                                    await broadcast_log(f"[GRID] Order not confirmed: {verification.get('status', 'pending')} - will reconcile later")

                            except Exception as e:
                                logger.error(f"[BUY ERROR] {symbol}: {e}")
                                await broadcast_log(f"[GRID] Order failed: {e}")

            elif action == 'SELL' and order_details and not skip_trading:
                # Check regime filter first (Smart Grid - hold sells in strong uptrend)
                if not regime_allow_sell:
                    logger.info(f"[REGIME] Holding SELL {symbol}: {regime_status['reason']}")
                    await broadcast_log(f"[REGIME] Holding SELL: {regime_status['reason']}")
                # Check momentum filter (optional for sells - let profits run in uptrend)
                elif not momentum_status['allow_sell']:
                    await broadcast_log(f"[MOM] Delaying SELL: {momentum_status['reason']}")
                else:
                    grid_price = order_details['price']
                    base_qty = order_details['quantity']
                    level_id = order_details.get('level_id')  # May be None for old grids

                    # Apply time quality and correlation adjustments
                    time_quality = time_status.get('time_quality', 1.0)
                    qty = base_qty * time_quality * corr_adjustment

                    # Log adjustments if applied
                    if time_quality < 1.0 or corr_adjustment < 1.0:
                        await broadcast_log(f"[ADJ] Qty adjusted: {base_qty:.4f} -> {qty:.4f} (time: {time_quality:.0%}, corr: {corr_adjustment:.0%})")

                    # === LIMIT ORDER PATH (Maker Fee Optimization) ===
                    if bot.use_limit_orders and level_id:
                        # IMPORTANT: If limit-orders are enabled but we skip due to guard/duplicate,
                        # do NOT fall back to market orders (that would negate the maker-fee intent).
                        if bot.grid_strategy.has_open_order_for_level(symbol, "sell", level_id):
                            logger.info(f"[GRID] Skipping SELL {symbol}: Open order exists for level {level_id[:8]}")
                            await broadcast_log(f"[GRID] Skipping duplicate order for level")
                        elif not can_place_limit_order(grid_price, current_price, "sell", bot.maker_buffer_bps):
                            logger.info(f"[GRID] Skipping SELL {symbol}: Grid price ${grid_price:,.2f} too close to market ${current_price:,.2f}")
                            await broadcast_log(f"[GRID] Skipping SELL: crossing guard (wait for next bar)")
                        else:
                            try:
                                # Limit price = grid level price (preserve grid integrity)
                                limit_price = round_limit_price(symbol, grid_price, bot.config)
                                rounded_qty = round_qty(symbol, qty, bot.config)
                                if rounded_qty <= 0:
                                    logger.warning(f"[SELL] Skipping LIMIT order: rounded_qty<=0 for {symbol} (raw={qty})")
                                    await broadcast_log(f"[GRID] SELL LIMIT skipped: qty too small after rounding")
                                else:
                                    order = LimitOrderRequest(
                                        symbol=symbol,
                                        qty=rounded_qty,
                                        side=OrderSide.SELL,
                                        time_in_force=TimeInForce.GTC,
                                        limit_price=limit_price
                                    )
                                    result = await run_blocking(client.trading_client.submit_order, order)
                                    order_id = str(result.id)

                                    # Register open limit order (for duplicate prevention)
                                    bot.grid_strategy.register_open_limit_order(
                                        order_id=order_id,
                                        symbol=symbol,
                                        side="sell",
                                        level_id=level_id,
                                        level_price=grid_price,
                                        limit_price=limit_price,
                                        qty=rounded_qty,
                                        source="grid"
                                    )

                                    # Register pending order (for reconciliation)
                                    bot.grid_strategy.register_pending_order(
                                        order_id=order_id,
                                        symbol=symbol,
                                        side="sell",
                                        intended_level_price=grid_price,
                                        intended_level_id=level_id,
                                        source="grid",
                                        fee_type="maker"  # Limit orders are maker
                                    )

                                    logger.info(f"[SELL] LIMIT order: {rounded_qty:.6f} {symbol} @ ${limit_price:,.2f} (grid: ${grid_price:,.2f})")
                                    await broadcast_log(f"[GRID] SELL LIMIT {symbol}: {rounded_qty:.6f} @ ${limit_price:,.2f}")
                                    await broadcast_log(f"  Level: {level_id[:8]}... | Order: {order_id[:8]}...")

                                    # Note: Don't verify_order_fill for limit orders - they may not fill immediately
                                    # Fills will be detected via reconciliation

                            except Exception as e:
                                logger.error(f"[SELL] LIMIT order failed for {symbol}: {e}")
                                await broadcast_log(f"[ERROR] SELL LIMIT failed: {e}")

                    # === MARKET ORDER PATH (original behavior - taker fees) ===
                    elif not (bot.use_limit_orders and level_id):
                        try:
                            order = MarketOrderRequest(
                                symbol=symbol,
                                qty=round(qty, 6),
                                side=OrderSide.SELL,
                                time_in_force=TimeInForce.GTC
                            )
                            result = await run_blocking(client.trading_client.submit_order, order)
                            order_id = str(result.id)

                            # Register pending order BEFORE verify (for deterministic matching)
                            level_price = order_details['price']
                            bot.grid_strategy.register_pending_order(
                                order_id=order_id,
                                symbol=symbol,
                                side="sell",
                                intended_level_price=level_price,
                                intended_level_id=level_id,
                                source="grid",
                                fee_type="maker"  # Limit orders are maker
                            )

                            # Verify order fill (async version avoids blocking thread pool)
                            verification = await client.verify_order_fill_async(order_id, max_wait_seconds=5)

                            if verification['confirmed']:
                                # Use actual fill price from Alpaca
                                fill_price = verification['filled_avg_price']
                                fill_qty = verification['filled_qty']

                                # Record fill and get profit with verified data (idempotent)
                                profit = bot.record_fill(symbol, "sell", fill_price, fill_qty, order_id)

                                # Add to confirmed orders (include profit)
                                verification['profit'] = profit
                                bot.add_confirmed_order(verification)

                                # Track performance by hour
                                if profit:
                                    bot.track_trade_performance(symbol, profit, current_hour)

                                trade_executed = {
                                    'action': 'SELL',
                                    'symbol': symbol,
                                    'price': fill_price,
                                    'qty': fill_qty,
                                    'profit': profit,
                                    'grid_level': order_details.get('grid_level'),
                                    'timestamp': datetime.now().isoformat(),
                                    'time_quality': time_quality,
                                    'corr_adjustment': corr_adjustment,
                                    'order_id': order_id,
                                    'verified': True
                                }

                                logger.info(f"[SELL] {symbol}: {fill_qty:.4f} @ ${fill_price:,.2f} [VERIFIED]")
                                await broadcast_log(f"[GRID] SELL {symbol}: {fill_qty:.4f} @ ${fill_price:,.2f}")
                                if profit:
                                    await broadcast_log(f"  Profit: ${profit:.2f} (Hour {current_hour}:00 UTC)")
                                await broadcast_log(f"  Grid level: {order_details.get('grid_level')} | Order ID: {order_id[:8]}...")
                            else:
                                # Verify timed out - pending order remains for reconciliation
                                logger.warning(f"[SELL] Order {order_id} not confirmed: {verification.get('reason', 'Unknown')} - pending for reconciliation")
                                await broadcast_log(f"[GRID] Order not confirmed: {verification.get('status', 'pending')} - will reconcile later")

                        except Exception as e:
                            logger.error(f"[SELL ERROR] {symbol}: {e}")
                            await broadcast_log(f"[GRID] Order failed: {e}")

            elif action == 'REBALANCED_UP':
                # Grid was auto-rebalanced upward
                old_range = evaluation.get('old_range', {})
                new_range = evaluation.get('new_range', {})
                await broadcast_log(f"[GRID] {symbol} AUTO-REBALANCED UP")
                await broadcast_log(f"  Old: ${old_range.get('lower', 0):,.2f} - ${old_range.get('upper', 0):,.2f}")
                await broadcast_log(f"  New: ${new_range.get('lower', 0):,.2f} - ${new_range.get('upper', 0):,.2f}")

            elif action == 'REBALANCED_DOWN':
                # Grid was auto-rebalanced downward
                old_range = evaluation.get('old_range', {})
                new_range = evaluation.get('new_range', {})
                await broadcast_log(f"[GRID] {symbol} AUTO-REBALANCED DOWN")
                await broadcast_log(f"  Old: ${old_range.get('lower', 0):,.2f} - ${old_range.get('upper', 0):,.2f}")
                await broadcast_log(f"  New: ${new_range.get('lower', 0):,.2f} - ${new_range.get('upper', 0):,.2f}")

            elif action == 'REBALANCE_UP':
                # Manual rebalance mode (auto_rebalance=False)
                await broadcast_log(f"[GRID] {symbol} ABOVE grid range - manual rebalance needed")

            elif action == 'REBALANCE_DOWN':
                # Manual rebalance mode (auto_rebalance=False)
                await broadcast_log(f"[GRID] {symbol} BELOW grid range - manual rebalance needed")

            # Prepare positions data
            positions_data = []
            for p in current_positions.values():
                positions_data.append({
                    "symbol": p.symbol,
                    "qty": float(p.qty),
                    "avg_entry_price": float(p.avg_entry_price),
                    "current_price": float(p.current_price),
                    "unrealized_pl": float(p.unrealized_pl),
                    "unrealized_plpc": float(p.unrealized_plpc)
                })

            # Get grid summaries for dashboard
            grid_summaries = bot.get_all_grid_summaries()

            # Get current grid summary for this symbol
            current_grid = grid_summaries.get(symbol, {})
            grid_range = current_grid.get('range', {})
            grid_levels = current_grid.get('levels', {})
            grid_perf = current_grid.get('performance', {})

            # Broadcast comprehensive update
            await broadcast_update({
                "status": "running",
                "price": current_price,
                "timestamp": str(bar.timestamp),
                "symbol": symbol,
                "multi_asset": {
                    "symbols": symbols,
                    "signals": {s: "GRID" for s in symbols},
                    "confidences": {s: 100 for s in symbols},
                    "active_symbol": symbol
                },
                "ai": {
                    'prediction': None,
                    'confidence': 100,
                    'signal': 'GRID',
                    'reasoning': [
                        f"Grid range: ${grid_range.get('lower', 0):,.2f} - ${grid_range.get('upper', 0):,.2f}",
                        f"Pending: {grid_levels.get('pending_buys', 0)} buys, {grid_levels.get('pending_sells', 0)} sells",
                        f"Completed trades: {grid_perf.get('completed_trades', 0)}",
                        f"Total profit: ${grid_perf.get('total_profit', 0):,.2f}",
                        f"Action: {action}"
                    ],
                    'features': {
                        'price': current_price,
                        'grid_lower': grid_range.get('lower', 0),
                        'grid_upper': grid_range.get('upper', 0),
                        'grid_spacing': grid_range.get('spacing', 0),
                        'pending_buys': grid_levels.get('pending_buys', 0),
                        'pending_sells': grid_levels.get('pending_sells', 0)
                    },
                    'thresholds': {
                        'grid_spacing_pct': grid_range.get('spacing_pct', 0)
                    },
                    'multi_timeframe': {},
                    'feature_importance': {}
                },
                "ultra": {
                    "regime": "GRID_TRADING",
                    "strategy": f"Grid {grid_levels.get('filled', 0)}/{grid_levels.get('total', 0)} filled",
                    "signal": action,
                    "confidence": 1.0,
                    "should_trade": action in ['BUY', 'SELL'],
                    "trade_reason": evaluation.get('reason', ''),
                    "metrics": {
                        "total_trades": bot.total_trades,
                        "total_profit": bot.total_profit,
                        "avg_buy_price": grid_perf.get('avg_buy_price', 0),
                        "avg_sell_price": grid_perf.get('avg_sell_price', 0)
                    },
                    "time_filter": {"window_name": "GRID_ACTIVE", "score": 1.0},
                    "kelly": {"kelly_fraction": 0.25}
                },
                "grid": {
                    "active": True,
                    "summaries": grid_summaries,
                    "total_trades": bot.total_trades,
                    "total_profit": bot.total_profit
                },
                "time_filter": {
                    "enabled": bot.use_time_filter,
                    "should_trade": time_status.get('should_trade', True),
                    "time_quality": time_status.get('time_quality', 1.0),
                    "reason": time_status.get('reason', ''),
                    "hour": time_status.get('hour', 0),
                    "is_weekend": time_status.get('is_weekend', False)
                },
                "correlations": correlations,
                "momentum": {
                    "status": momentum_status,
                    "allow_buy": momentum_status.get('allow_buy', True),
                    "allow_sell": momentum_status.get('allow_sell', True)
                },
                "regime": {
                    "current": current_regime,
                    "adx": regime_adx,
                    "allow_buy": regime_allow_buy,
                    "allow_sell": regime_allow_sell,
                    "is_strong_trend": regime_status.get('is_strong_trend', False),
                    "paused": bot.regime_pause.get(symbol, False),
                    "reason": regime_status.get('reason', ''),
                    "strategy_hint": regime_status.get('strategy_hint', 'WAIT'),
                    "confidence": regime_status.get('confidence', 0),
                    "all_regimes": dict(bot.current_regime),
                    "all_paused": dict(bot.regime_pause)
                },
                "performance": bot.get_performance_report(),
                "risk": bot.get_risk_status(equity),
                "account": {
                    "equity": equity,
                    "buying_power": float(account.buying_power) if account else 0,
                    "balance": float(account.cash) if account else 0
                },
                "positions": positions_data,
                "market": {
                    "high": float(bar.high),
                    "low": float(bar.low),
                    "volume": float(bar.volume),
                    "change": 0
                },
                "last_trade": windfall_executed or trade_executed,
                "orders": {
                    "confirmed": bot.get_confirmed_orders(limit=20),
                    "stats": bot.get_order_stats(),
                    "reconciliation": bot.get_reconciliation_status()
                },
                "windfall": bot.get_windfall_stats(),
                "health": {
                    "stream": {
                        "last_bar_at": last_bar_time[0].isoformat(),
                        "seconds_since_bar": int((datetime.now() - last_bar_time[0]).total_seconds()),
                        "stale_threshold_seconds": STALE_THRESHOLD_SECONDS,
                        "status": "connected" if (datetime.now() - last_bar_time[0]).total_seconds() < 90 else (
                            "degraded" if (datetime.now() - last_bar_time[0]).total_seconds() < STALE_THRESHOLD_SECONDS else "stale"
                        )
                    },
                    "orders": bot.get_order_health_summary()
                },
                "risk_overlay": bot.risk_overlay.get_status()
            })

        await broadcast_log(f"Will subscribe to: {', '.join(symbols)}")

        # Stream watchdog - monitors for stale data and forces reconnect
        stream_should_restart = [False]

        async def stream_watchdog():
            """Monitor stream health and trigger restart if stale."""
            while True:
                await asyncio.sleep(30)  # Check every 30 seconds
                seconds_since_bar = (datetime.now() - last_bar_time[0]).total_seconds()

                if seconds_since_bar > STALE_THRESHOLD_SECONDS:
                    logger.warning(f"[WATCHDOG] Stream stale! No bar for {seconds_since_bar:.0f}s")
                    await broadcast_log(f"[WATCHDOG] Stream stale ({seconds_since_bar:.0f}s) - forcing reconnect...")
                    stream_should_restart[0] = True

                    # Force close the stream to trigger reconnection
                    try:
                        await stream.close()
                    except Exception as e:
                        logger.error(f"[WATCHDOG] Error closing stream: {e}")

                    # Reset timer to avoid repeated triggers
                    last_bar_time[0] = datetime.now()

        # Start watchdog task
        watchdog_task = asyncio.create_task(stream_watchdog())
        await broadcast_log("[WATCHDOG] Stream health monitor started (3 min threshold)")

        # Order tracking health check task - runs every 5 minutes
        HEALTH_CHECK_INTERVAL_SECONDS = 300  # 5 minutes

        async def order_health_check():
            """Periodic health check for order tracking mismatch detection."""
            while True:
                await asyncio.sleep(HEALTH_CHECK_INTERVAL_SECONDS)
                try:
                    health = await bot.check_order_tracking_health(client)
                    if not health.get('healthy', True):
                        await broadcast_log(
                            f"[HEALTH] Order tracking issue detected! "
                            f"Alpaca={health.get('alpaca_count', '?')}, "
                            f"Tracked={health.get('tracked_count', '?')}"
                        )
                except Exception as e:
                    logger.error(f"[HEALTH] Health check error: {e}")

        # Start health check task
        health_task = asyncio.create_task(order_health_check())
        await broadcast_log("[HEALTH] Order tracking monitor started (5 min interval)")

        # Run stream with reconnection and proper cleanup
        backoff = 1
        stream = None  # Track current stream for cleanup

        while True:
            try:
                stream_should_restart[0] = False

                # Clean up any existing stream BEFORE creating new one
                if stream is not None:
                    try:
                        stream.stop()
                        await stream.close()
                    except Exception as cleanup_err:
                        logger.debug(f"Stream cleanup error (ignored): {cleanup_err}")
                    stream = None

                await broadcast_log("Connecting to Alpaca stream...")

                stream = CryptoDataStream(
                    config.API_KEY,
                    config.SECRET_KEY
                )
                stream.subscribe_bars(handle_bar, *symbols)

                # Note: run() is not async, must use _run_forever() for async context
                await stream._run_forever()
                backoff = 1  # Reset on clean exit

                await broadcast_log("[STREAM] Connection ended cleanly - reconnecting...")

            except Exception as e:
                error_str = str(e).lower()
                is_rate_limit = '429' in error_str or 'connection limit' in error_str or 'rate' in error_str

                if stream_should_restart[0]:
                    await broadcast_log(f"[WATCHDOG] Reconnecting after forced restart...")
                    backoff = 5  # Moderate backoff for watchdog restarts
                elif is_rate_limit:
                    # Aggressive backoff for rate limits: 15s -> 30s -> 60s -> 120s (cap)
                    backoff = min(backoff * 2, 120) if backoff >= 15 else 15
                    await broadcast_log(f"[STREAM] Rate limited - backing off {backoff}s...")
                else:
                    await broadcast_log(f"Stream disconnected: {e}")
                    backoff = min(backoff * 2, 60)  # Normal backoff: 1 -> 2 -> 4 -> ... -> 60

                # Add jitter to prevent thundering herd
                jitter = random.uniform(0, backoff * 0.2)
                await asyncio.sleep(backoff + jitter)

            finally:
                # ALWAYS clean up stream on any exit path
                if stream is not None:
                    try:
                        stream.stop()
                        await stream.close()
                    except Exception:
                        pass
                    stream = None

    except Exception as e:
        logger.error(f"Grid Bot Error: {e}")
        await broadcast_log(f"Error: {e}")
        raise