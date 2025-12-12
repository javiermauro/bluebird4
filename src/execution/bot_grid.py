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
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
import numpy as np

import pandas as pd

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

        logger.info(f"  Time filter: {'ENABLED' if self.use_time_filter else 'DISABLED'}")
        logger.info(f"  Optimal hours (UTC): {self.optimal_hours}")

        # Load confirmed orders from Alpaca on startup
        self._load_confirmed_orders_from_alpaca()

        # Run initial reconciliation with database
        self._run_reconciliation()

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

            # Determine whether to allow buys/sells based on regime
            allow_buy = True
            allow_sell = True
            reason = f"Regime: {regime}"

            if is_strong_trend:
                self.regime_pause[symbol] = True

                if regime == MarketRegime.TRENDING_DOWN:
                    # Strong downtrend: PAUSE buys (don't catch falling knife)
                    # ALLOW sells (take profit on existing positions)
                    allow_buy = False
                    allow_sell = True
                    reason = f"STRONG DOWNTREND (ADX={adx:.0f}) - Pausing buys"

                elif regime == MarketRegime.TRENDING_UP:
                    # Strong uptrend: PAUSE sells (let winners run)
                    # ALLOW buys (can still accumulate)
                    allow_buy = True
                    allow_sell = False
                    reason = f"STRONG UPTREND (ADX={adx:.0f}) - Holding sells"
            else:
                self.regime_pause[symbol] = False
                reason = f"{regime} (ADX={adx:.0f}) - Grid active"

            result = {
                'regime': regime,
                'allow_buy': allow_buy,
                'allow_sell': allow_sell,
                'is_strong_trend': is_strong_trend,
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

        if previous > 0:
            momentum = (recent - previous) / previous * 100
            result['momentum'] = round(momentum, 2)

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

    def get_reconciliation_status(self) -> Dict[str, Any]:
        """Get current reconciliation status for dashboard."""
        status = self.reconciliation_status.copy()
        if self.last_reconciliation:
            # Add time since last check
            seconds_ago = (datetime.now() - self.last_reconciliation).total_seconds()
            status['seconds_ago'] = int(seconds_ago)
            status['minutes_ago'] = int(seconds_ago / 60)
        return status

    def initialize_grids(self, prices: Dict[str, float], equity: float) -> None:
        """
        Initialize grids for all symbols based on current prices.

        First tries to restore saved grid state from file (same-day only).
        If no valid saved state, creates fresh grids.

        Uses config-defined ranges - no complex ATR calculations needed.
        Grid trading is SIMPLE: buy low, sell high within the range.

        Args:
            prices: Current price for each symbol
            equity: Total account equity
        """
        # Try to restore saved grid state first (prevents duplicate orders on restart)
        if self.grid_strategy.load_state():
            self.grids_initialized = True
            logger.info("=" * 50)
            logger.info("GRID STATE RESTORED FROM FILE")
            logger.info("=" * 50)

            # Restore total_trades and total_profit from grid state
            for symbol in self.symbols:
                summary = self.grid_strategy.get_grid_summary(symbol)
                if summary.get('is_active'):
                    filled = summary['levels']['filled']
                    total = summary['levels']['total']
                    perf = summary.get('performance', {})
                    self.total_trades += perf.get('completed_trades', 0)
                    self.total_profit += perf.get('total_profit', 0.0)
                    logger.info(f"  {symbol}: {filled}/{total} levels filled, {perf.get('completed_trades', 0)} trades, ${perf.get('total_profit', 0):.2f} profit")

            logger.info(f"  TOTALS: {self.total_trades} trades, ${self.total_profit:.2f} profit")
            return

        # No valid saved state - create fresh grids
        logger.info("Creating fresh grids for all symbols...")

        for symbol in self.symbols:
            if symbol not in prices:
                continue

            current_price = prices[symbol]

            # Get grid config from UltraConfig (or use defaults)
            grid_configs = getattr(self.config, 'GRID_CONFIGS', DEFAULT_GRID_CONFIGS)
            template = grid_configs.get(symbol, {
                "num_grids": 10,
                "range_pct": 0.05,
                "investment_ratio": 0.25
            })

            # Create grid config using config-defined range percentage
            range_pct = template["range_pct"]
            num_grids = template["num_grids"]
            investment_ratio = template["investment_ratio"]

            # Calculate capital for this symbol's grid
            capital_for_grid = equity * investment_ratio

            config = GridConfig(
                symbol=symbol,
                upper_price=current_price * (1 + range_pct / 2),
                lower_price=current_price * (1 - range_pct / 2),
                num_grids=num_grids,
                investment_per_grid=capital_for_grid / num_grids
            )

            # Create the grid
            self.grid_strategy.create_grid(config, current_price)

            logger.info(f"Grid created for {symbol}:")
            logger.info(f"  Range: ${config.lower_price:,.2f} - ${config.upper_price:,.2f}")
            logger.info(f"  Levels: {num_grids}, Investment/level: ${config.investment_per_grid:,.2f}")
            logger.info(f"  Profit per cycle: {config.profit_per_grid_pct:.2f}%")

            # Calculate and store expected profit per trade (after fees)
            # Fees: ~0.5% round trip (0.25% taker + spread + slippage)
            fee_pct = 0.005  # 0.5% round trip
            expected_profit_pct = (config.profit_per_grid_pct / 100) - fee_pct
            expected_profit = config.investment_per_grid * expected_profit_pct
            self.expected_profit_per_trade[symbol] = expected_profit
            logger.info(f"  Expected profit/trade (after fees): ${expected_profit:.2f} ({expected_profit_pct*100:.2f}%)")

        self.grids_initialized = True

        # Save initial grid state for persistence across restarts
        self.grid_strategy.save_state()
        logger.info("Initial grid state saved to file")

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

        # Initialize
        client = AlpacaClient(config)
        bot = GridTradingBot(config, client)

        # Get account info
        try:
            account = client.trading_client.get_account()
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
                bars = client.data_client.get_crypto_bars(req)
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

        # Sync open limit orders from Alpaca (crash recovery)
        if bot.use_limit_orders:
            bot.sync_open_orders_from_alpaca()
            await broadcast_log("Open limit order sync complete")

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
            bot.check_periodic_reconciliation()

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

            # Get current positions
            try:
                positions = client.get_positions()
                current_positions = {p.symbol: p for p in positions}
                num_positions = len(positions)
            except:
                current_positions = {}
                num_positions = 0

            # Get account
            try:
                account = client.trading_client.get_account()
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
                        result = client.trading_client.submit_order(order)

                        # Verify order fill
                        verification = client.verify_order_fill(str(result.id), max_wait_seconds=5)

                        if verification['confirmed']:
                            fill_price = verification['filled_avg_price']
                            fill_qty = verification['filled_qty']

                            # Calculate profit (entry to sell price, minus fees)
                            entry_price = pos_dict['avg_entry_price']
                            gross_profit = (fill_price - entry_price) * fill_qty
                            fee = fill_price * fill_qty * 0.0025  # 0.25% taker fee
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
            # Update risk state
            bot.update_risk_state(equity)

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
                        result = client.trading_client.submit_order(order)
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
                    # Normal grid evaluation
                    evaluation = bot.evaluate(symbol, current_price, position_qty)

            action = evaluation.get('action', 'HOLD')
            order_details = evaluation.get('order_details')

            # DEBUG: Log the evaluation result
            if action != 'HOLD':
                logger.info(f"[DEBUG] {symbol} action={action}, order_details={order_details is not None}, skip_trading={skip_trading}")

            trade_executed = None

            if action == 'BUY' and order_details and not skip_trading:
                # Check regime filter first (Smart Grid - avoid buying in strong downtrend)
                if not regime_allow_buy:
                    logger.info(f"[REGIME] Skipping BUY {symbol}: {regime_status['reason']}")
                    await broadcast_log(f"[REGIME] Skipping BUY: {regime_status['reason']}")
                # Check momentum filter
                elif not momentum_status['allow_buy']:
                    logger.info(f"[MOM] Skipping BUY {symbol}: {momentum_status['reason']}")
                    await broadcast_log(f"[MOM] Skipping BUY: {momentum_status['reason']}")
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
                                        result = client.trading_client.submit_order(order)
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
                                            source="grid"
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
                                result = client.trading_client.submit_order(order)
                                order_id = str(result.id)

                                # Register pending order BEFORE verify (for deterministic matching)
                                level_price = order_details['price']
                                bot.grid_strategy.register_pending_order(
                                    order_id=order_id,
                                    symbol=symbol,
                                    side="buy",
                                    intended_level_price=level_price,
                                    intended_level_id=level_id,
                                    source="grid"
                                )

                                # Verify order fill with Alpaca
                                verification = client.verify_order_fill(order_id, max_wait_seconds=5)

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
                                    result = client.trading_client.submit_order(order)
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
                                        source="grid"
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
                            result = client.trading_client.submit_order(order)
                            order_id = str(result.id)

                            # Register pending order BEFORE verify (for deterministic matching)
                            level_price = order_details['price']
                            bot.grid_strategy.register_pending_order(
                                order_id=order_id,
                                symbol=symbol,
                                side="sell",
                                intended_level_price=level_price,
                                intended_level_id=level_id,
                                source="grid"
                            )

                            # Verify order fill with Alpaca
                            verification = client.verify_order_fill(order_id, max_wait_seconds=5)

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
                "windfall": bot.get_windfall_stats()
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

        # Run stream with reconnection
        backoff = 1
        max_backoff = 60

        while True:
            try:
                stream_should_restart[0] = False
                await broadcast_log("Connecting to Alpaca stream...")

                # Create fresh stream on reconnect
                stream = CryptoDataStream(
                    config.API_KEY,
                    config.SECRET_KEY
                )
                stream.subscribe_bars(handle_bar, *symbols)

                await stream._run_forever()
                backoff = 1

                # If we exit cleanly, log it (unusual)
                await broadcast_log("[STREAM] Connection ended cleanly - reconnecting...")

            except Exception as e:
                if stream_should_restart[0]:
                    await broadcast_log(f"[WATCHDOG] Reconnecting after forced restart...")
                    backoff = 1  # Reset backoff for watchdog-triggered restarts
                else:
                    await broadcast_log(f"Stream disconnected: {e}")

                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, max_backoff)

    except Exception as e:
        logger.error(f"Grid Bot Error: {e}")
        await broadcast_log(f"Error: {e}")
        raise