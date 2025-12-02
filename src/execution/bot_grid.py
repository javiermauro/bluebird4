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
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List

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

        # Track daily P&L
        self.daily_pnl = 0.0
        self.daily_start_equity = 0.0
        self.last_trading_day = None

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

    def add_bar(self, symbol: str, bar: Any) -> None:
        """Add a bar to history (for tracking only - grids don't need features)."""
        if symbol not in self.bars:
            self.bars[symbol] = []

        self.bars[symbol].append(bar)

        # Keep last 100 bars
        if len(self.bars[symbol]) > 100:
            self.bars[symbol].pop(0)

    def initialize_grids(self, prices: Dict[str, float], equity: float) -> None:
        """
        Initialize grids for all symbols based on current prices.

        Uses config-defined ranges - no complex ATR calculations needed.
        Grid trading is SIMPLE: buy low, sell high within the range.

        Args:
            prices: Current price for each symbol
            equity: Total account equity
        """
        logger.info("Initializing grids for all symbols...")

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

        self.grids_initialized = True

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
            self.daily_start_equity = equity
            self.last_trading_day = today
            self.daily_limit_hit = False
            logger.info(f"New trading day - daily P&L reset. Starting equity: ${equity:,.2f}")

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

        return {
            'peak_equity': self.peak_equity,
            'starting_equity': self.starting_equity,
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

            trade_executed = None

            if action == 'BUY' and order_details:
                grid_price = order_details['price']
                qty = order_details['quantity']

                try:
                    order = MarketOrderRequest(
                        symbol=symbol,
                        qty=round(qty, 6),
                        side=OrderSide.BUY,
                        time_in_force=TimeInForce.GTC
                    )
                    result = client.trading_client.submit_order(order)

                    # Record fill
                    bot.record_fill(symbol, "buy", current_price, qty, str(result.id))

                    trade_executed = {
                        'action': 'BUY',
                        'symbol': symbol,
                        'price': current_price,
                        'qty': qty,
                        'grid_level': order_details.get('grid_level'),
                        'timestamp': datetime.now().isoformat()
                    }

                    await broadcast_log(f"[GRID] BUY {symbol}: {qty:.4f} @ ${current_price:,.2f}")
                    await broadcast_log(f"  Grid level: {order_details.get('grid_level')}")

                except Exception as e:
                    await broadcast_log(f"[GRID] Order failed: {e}")

            elif action == 'SELL' and order_details:
                qty = order_details['quantity']

                try:
                    order = MarketOrderRequest(
                        symbol=symbol,
                        qty=round(qty, 6),
                        side=OrderSide.SELL,
                        time_in_force=TimeInForce.GTC
                    )
                    result = client.trading_client.submit_order(order)

                    # Record fill and get profit
                    profit = bot.record_fill(symbol, "sell", current_price, qty, str(result.id))

                    trade_executed = {
                        'action': 'SELL',
                        'symbol': symbol,
                        'price': current_price,
                        'qty': qty,
                        'profit': profit,
                        'grid_level': order_details.get('grid_level'),
                        'timestamp': datetime.now().isoformat()
                    }

                    await broadcast_log(f"[GRID] SELL {symbol}: {qty:.4f} @ ${current_price:,.2f}")
                    if profit:
                        await broadcast_log(f"  Profit: ${profit:.2f}")
                    await broadcast_log(f"  Grid level: {order_details.get('grid_level')}")

                except Exception as e:
                    await broadcast_log(f"[GRID] Order failed: {e}")

            elif action == 'REBALANCE_UP':
                await broadcast_log(f"[GRID] {symbol} ABOVE grid range - consider rebalancing")

            elif action == 'REBALANCE_DOWN':
                await broadcast_log(f"[GRID] {symbol} BELOW grid range - consider rebalancing")

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
                "last_trade": trade_executed
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
