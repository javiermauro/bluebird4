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
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from enum import Enum

logger = logging.getLogger("GridTrading")

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

    def to_dict(self) -> dict:
        """Serialize grid level for persistence."""
        return {
            'price': self.price,
            'side': self.side.value,
            'is_filled': self.is_filled,
            'order_id': self.order_id,
            'filled_at': self.filled_at.isoformat() if self.filled_at else None,
            'quantity': self.quantity
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
            quantity=data['quantity']
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
class GridState:
    """Tracks the current state of a grid trading setup."""
    config: GridConfig
    levels: List[GridLevel] = field(default_factory=list)
    is_active: bool = False
    total_profit: float = 0.0
    completed_trades: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    last_trade_at: Optional[datetime] = None

    # Performance tracking
    total_buys: int = 0
    total_sells: int = 0
    avg_buy_price: float = 0.0
    avg_sell_price: float = 0.0

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
            'last_trade_at': self.last_trade_at.isoformat() if self.last_trade_at else None
        }


class GridTradingStrategy:
    """
    Grid Trading Strategy Implementation.

    Creates a grid of buy/sell orders within a price range and automatically
    executes trades as price oscillates, capturing profit from each move.
    """

    def __init__(self):
        self.grids: Dict[str, GridState] = {}  # symbol -> GridState
        logger.info("GridTradingStrategy initialized")

    def save_state(self) -> None:
        """Save all grid states to file for persistence across restarts."""
        try:
            data = {
                'date': datetime.now().strftime('%Y-%m-%d'),
                'saved_at': datetime.now().isoformat(),
                'grids': {symbol: state.to_dict() for symbol, state in self.grids.items()}
            }
            with open(GRID_STATE_FILE, 'w') as f:
                json.dump(data, f, indent=2)
            logger.debug(f"Grid state saved for {len(self.grids)} symbols")
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

                self.grids[symbol] = state

                filled_count = sum(1 for l in levels if l.is_filled)
                logger.info(f"Restored grid for {symbol}: {filled_count}/{len(levels)} levels filled")

            logger.info(f"Grid state restored for {len(self.grids)} symbols")
            return True

        except Exception as e:
            logger.error(f"Failed to load grid state: {e}")
            return False

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
                "grid_level": state.levels.index(best_buy)
            }
        elif triggered_sells and current_position_qty > 0:
            # Sell at the lowest triggered sell level
            best_sell = min(triggered_sells, key=lambda l: l.price)
            action = "SELL"
            order_details = {
                "price": best_sell.price,
                "quantity": min(best_sell.quantity, current_position_qty),
                "grid_level": state.levels.index(best_sell)
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

    def record_fill(
        self,
        symbol: str,
        side: str,  # "buy" or "sell"
        price: float,
        quantity: float,
        order_id: str
    ) -> Optional[float]:
        """
        Record that a grid order was filled.

        Args:
            symbol: Trading symbol
            side: "buy" or "sell"
            price: Fill price
            quantity: Fill quantity
            order_id: Order ID

        Returns:
            Profit from this trade (if closing a grid cycle)
        """
        if symbol not in self.grids:
            return None

        state = self.grids[symbol]

        # Find the closest grid level
        min_diff = float('inf')
        closest_level = None
        for level in state.levels:
            diff = abs(level.price - price)
            if diff < min_diff and not level.is_filled:
                min_diff = diff
                closest_level = level

        if closest_level:
            closest_level.is_filled = True
            closest_level.order_id = order_id
            closest_level.filled_at = datetime.now()

            state.last_trade_at = datetime.now()
            state.completed_trades += 1

            if side == "buy":
                state.total_buys += 1
                # Update average buy price
                if state.avg_buy_price == 0:
                    state.avg_buy_price = price
                else:
                    state.avg_buy_price = (state.avg_buy_price + price) / 2

                # Create corresponding sell level
                sell_price = price + state.config.grid_spacing
                if sell_price <= state.config.upper_price:
                    self._add_sell_level(state, sell_price, quantity)

            else:  # sell
                state.total_sells += 1
                # Update average sell price
                if state.avg_sell_price == 0:
                    state.avg_sell_price = price
                else:
                    state.avg_sell_price = (state.avg_sell_price + price) / 2

                # Calculate profit from this cycle
                profit = (price - state.avg_buy_price) * quantity
                state.total_profit += profit

                # Create corresponding buy level
                buy_price = price - state.config.grid_spacing
                if buy_price >= state.config.lower_price:
                    self._add_buy_level(state, buy_price, quantity)

                return profit

        return None

    def _add_sell_level(self, state: GridState, price: float, quantity: float):
        """Add a new sell level to the grid."""
        level = GridLevel(
            price=price,
            side=GridOrderSide.SELL,
            quantity=quantity
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
        config = state.config

        filled_levels = [l for l in state.levels if l.is_filled]
        unfilled_buys = [l for l in state.levels if not l.is_filled and l.side == GridOrderSide.BUY]
        unfilled_sells = [l for l in state.levels if not l.is_filled and l.side == GridOrderSide.SELL]

        return {
            "symbol": symbol,
            "is_active": state.is_active,
            "range": {
                "lower": config.lower_price,
                "upper": config.upper_price,
                "spacing": config.grid_spacing,
                "spacing_pct": config.profit_per_grid_pct
            },
            "levels": {
                "total": len(state.levels),
                "filled": len(filled_levels),
                "pending_buys": len(unfilled_buys),
                "pending_sells": len(unfilled_sells)
            },
            "performance": {
                "total_profit": state.total_profit,
                "completed_trades": state.completed_trades,
                "total_buys": state.total_buys,
                "total_sells": state.total_sells,
                "avg_buy_price": state.avg_buy_price,
                "avg_sell_price": state.avg_sell_price
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
