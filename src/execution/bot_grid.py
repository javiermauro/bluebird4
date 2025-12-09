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

from config_ultra import UltraConfig
from src.execution.alpaca_client import AlpacaClient
from src.strategy.grid_trading import (
    GridTradingStrategy,
    GridConfig,
    GridState,
    DEFAULT_GRID_CONFIGS,
    create_default_grids
)

from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

logger = logging.getLogger("GridBot")


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

        # === RISK MANAGEMENT STATE ===
        # Track peak equity for drawdown calculation
        self.peak_equity = 0.0
        self.starting_equity = 0.0

        # Track daily P&L (with persistence for true daily tracking)
        self.daily_pnl = 0.0
        self.daily_start_equity = 0.0
        self.last_trading_day = None
        self.daily_equity_file = "/tmp/bluebird-daily-equity.json"
        self._load_daily_equity()  # Load persisted daily equity on startup

        # Circuit breaker flags
        self.daily_limit_hit = False
        self.max_drawdown_hit = False
        self.stop_loss_triggered: Dict[str, bool] = {}  # Per-symbol stop loss

        # Risk thresholds from config
        self.stop_loss_pct = getattr(config, 'GRID_STOP_LOSS_PCT', 0.10)
        self.daily_loss_limit = getattr(config, 'DAILY_LOSS_LIMIT', 0.05)
        self.max_drawdown = getattr(config, 'MAX_DRAWDOWN', 0.10)

        logger.info(f"GridTradingBot initialized for {len(self.symbols)} symbols")
        logger.info(f"  Risk controls: SL={self.stop_loss_pct:.0%}, Daily={self.daily_loss_limit:.0%}, DD={self.max_drawdown:.0%}")

        # Time filtering settings
        self.use_time_filter = getattr(config, 'USE_TIME_FILTER', True)
        self.optimal_hours = getattr(config, 'OPTIMAL_HOURS', [(13, 17), (1, 4), (7, 9)])
        self.avoid_hours = getattr(config, 'AVOID_HOURS', [(22, 24), (5, 7)])
        self.weekend_size_mult = getattr(config, 'WEEKEND_SIZE_MULT', 0.5)

        # Correlation monitoring
        self.price_history: Dict[str, List[float]] = {s: [] for s in self.symbols}
        self.correlation_window = 20  # Number of bars for correlation calculation
        self.high_correlation_threshold = 0.85  # Reduce exposure when correlation > this

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
        Add a confirmed order from Alpaca to the tracking list.

        Args:
            order_data: Dict with order details from verify_order_fill()
        """
        # Add timestamp if not present
        if 'recorded_at' not in order_data:
            order_data['recorded_at'] = datetime.now().isoformat()

        self.confirmed_orders.append(order_data)

        # Keep last 500 orders
        if len(self.confirmed_orders) > 500:
            self.confirmed_orders = self.confirmed_orders[-500:]

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

            # Log restored state summary
            for symbol in self.symbols:
                summary = self.grid_strategy.get_grid_summary(symbol)
                if summary.get('is_active'):
                    filled = summary['levels']['filled']
                    total = summary['levels']['total']
                    logger.info(f"  {symbol}: {filled}/{total} levels filled, ${summary['performance']['total_profit']:.2f} profit")
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

    def record_fill(self, symbol: str, side: str, price: float, qty: float, order_id: str) -> Optional[float]:
        """Record that an order was filled and return profit if applicable."""
        profit = self.grid_strategy.record_fill(symbol, side, price, qty, order_id)

        if profit:
            self.total_profit += profit
            self.total_trades += 1

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
                self.daily_limit_hit = True
                result['should_stop'] = True
                result['reason'] = f"Daily loss limit hit: {self.daily_pnl:.2%} (limit: -{self.daily_loss_limit:.0%})"

        # Check max drawdown from peak
        if self.peak_equity > 0:
            drawdown = (self.peak_equity - equity) / self.peak_equity
            result['drawdown'] = drawdown

            if drawdown > self.max_drawdown:
                self.max_drawdown_hit = True
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
            result['triggered'] = True
            result['reason'] = f"Price ${current_price:,.2f} below stop ${stop_price:,.2f} (grid lower: ${lower_price:,.2f})"
            logger.warning(f"STOP-LOSS TRIGGERED for {symbol}: {result['reason']}")

        return result

    def get_risk_status(self, equity: float) -> Dict[str, Any]:
        """Get comprehensive risk status for dashboard."""
        circuit = self.check_circuit_breakers(equity)

        # Calculate daily P/L in dollars
        daily_pnl_dollars = equity - self.daily_start_equity if self.daily_start_equity > 0 else 0

        return {
            'peak_equity': self.peak_equity,
            'starting_equity': self.starting_equity,
            'daily_start_equity': self.daily_start_equity,  # Alpaca's last_equity
            'daily_pnl': daily_pnl_dollars,  # Dollar amount
            'daily_pnl_pct': self.daily_pnl * 100,
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

        # Main trading loop
        from alpaca.data.live import CryptoDataStream

        stream = CryptoDataStream(
            config.API_KEY,
            config.SECRET_KEY
        )

        async def handle_bar(bar):
            """Handle incoming bar with grid evaluation and risk management."""
            symbol = bar.symbol
            current_price = float(bar.close)

            logger.info(f"[GRID] {symbol} @ ${current_price:,.2f}")

            if symbol not in symbols:
                return

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
            if alpaca_symbol in current_positions:
                position_qty = float(current_positions[alpaca_symbol].qty)
            elif symbol in current_positions:
                position_qty = float(current_positions[symbol].qty)

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

                        # Record as emergency sell
                        bot.record_fill(symbol, "sell", current_price, position_qty, str(result.id))

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
                # Check momentum filter
                if not momentum_status['allow_buy']:
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

                        # Apply time quality and correlation adjustments
                        time_quality = time_status.get('time_quality', 1.0)
                        qty = base_qty * time_quality * corr_adjustment

                        # Log adjustments if applied
                        if time_quality < 1.0 or corr_adjustment < 1.0:
                            await broadcast_log(f"[ADJ] Qty adjusted: {base_qty:.4f} -> {qty:.4f} (time: {time_quality:.0%}, corr: {corr_adjustment:.0%})")

                        try:
                            order = MarketOrderRequest(
                                symbol=symbol,
                                qty=round(qty, 6),
                                side=OrderSide.BUY,
                                time_in_force=TimeInForce.GTC
                            )
                            result = client.trading_client.submit_order(order)
                            order_id = str(result.id)

                            # Verify order fill with Alpaca
                            verification = client.verify_order_fill(order_id, max_wait_seconds=5)

                            if verification['confirmed']:
                                # Use actual fill price from Alpaca
                                fill_price = verification['filled_avg_price']
                                fill_qty = verification['filled_qty']

                                # Record fill with verified data
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
                                logger.warning(f"[BUY] Order {order_id} not confirmed: {verification.get('reason', 'Unknown')}")
                                await broadcast_log(f"[GRID] Order not confirmed: {verification.get('status', 'pending')}")

                        except Exception as e:
                            logger.error(f"[BUY ERROR] {symbol}: {e}")
                            await broadcast_log(f"[GRID] Order failed: {e}")

            elif action == 'SELL' and order_details and not skip_trading:
                # Check momentum filter (optional for sells - let profits run in uptrend)
                if not momentum_status['allow_sell']:
                    await broadcast_log(f"[MOM] Delaying SELL: {momentum_status['reason']}")
                else:
                    base_qty = order_details['quantity']

                    # Apply time quality and correlation adjustments
                    time_quality = time_status.get('time_quality', 1.0)
                    qty = base_qty * time_quality * corr_adjustment

                    # Log adjustments if applied
                    if time_quality < 1.0 or corr_adjustment < 1.0:
                        await broadcast_log(f"[ADJ] Qty adjusted: {base_qty:.4f} -> {qty:.4f} (time: {time_quality:.0%}, corr: {corr_adjustment:.0%})")

                    try:
                        order = MarketOrderRequest(
                            symbol=symbol,
                            qty=round(qty, 6),
                            side=OrderSide.SELL,
                            time_in_force=TimeInForce.GTC
                        )
                        result = client.trading_client.submit_order(order)
                        order_id = str(result.id)

                        # Verify order fill with Alpaca
                        verification = client.verify_order_fill(order_id, max_wait_seconds=5)

                        if verification['confirmed']:
                            # Use actual fill price from Alpaca
                            fill_price = verification['filled_avg_price']
                            fill_qty = verification['filled_qty']

                            # Record fill and get profit with verified data
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
                            logger.warning(f"[SELL] Order {order_id} not confirmed: {verification.get('reason', 'Unknown')}")
                            await broadcast_log(f"[GRID] Order not confirmed: {verification.get('status', 'pending')}")

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
                "last_trade": trade_executed,
                "orders": {
                    "confirmed": bot.get_confirmed_orders(limit=20),
                    "stats": bot.get_order_stats(),
                    "reconciliation": bot.get_reconciliation_status()
                }
            })

        # Subscribe to all symbols
        stream.subscribe_bars(handle_bar, *symbols)
        await broadcast_log(f"Subscribed to: {', '.join(symbols)}")

        # Run stream with reconnection
        backoff = 1
        max_backoff = 60

        while True:
            try:
                await broadcast_log("Connecting to Alpaca stream...")
                await stream._run_forever()
                backoff = 1
            except Exception as e:
                await broadcast_log(f"Stream disconnected: {e}")
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, max_backoff)

    except Exception as e:
        logger.error(f"Grid Bot Error: {e}")
        await broadcast_log(f"Error: {e}")
        raise
