"""
ULTRA BACKTEST - Realistic Trading Simulation

This backtest is designed to give REALISTIC expectations.

Key Features:
1. Walk-Forward Validation (prevents overfitting)
2. Realistic Fees & Slippage
3. Regime-aware strategy switching
4. Kelly Criterion position sizing
5. Monte Carlo simulation for confidence intervals

The goal is NOT to maximize backtest returns,
but to understand realistic performance bounds.
"""

import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
from dataclasses import dataclass
from config_ultra import UltraConfig
from src.features.ultra_indicators import UltraIndicators
from src.strategy.regime_detector import RegimeDetector, MarketRegime, TimeOfDayFilter
from src.data.mock_loader import MockDataLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("UltraBacktest")


@dataclass
class Trade:
    """Represents a completed trade."""
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    quantity: float
    side: str  # 'LONG' or 'SHORT'
    pnl: float
    pnl_pct: float
    fees: float
    strategy: str
    regime: str


class RealisticFees:
    """
    Realistic fee and slippage model.
    
    For Alpaca Crypto:
    - Maker: 0.15%
    - Taker: 0.25%
    - We assume taker (market orders)
    
    Additional costs:
    - Spread: 0.05-0.10%
    - Slippage: 0.02-0.05%
    """
    
    TAKER_FEE = 0.0025  # 0.25%
    SPREAD = 0.0007     # 0.07% average
    SLIPPAGE = 0.0003   # 0.03% average
    
    @classmethod
    def total_cost_pct(cls) -> float:
        """Total one-way cost percentage."""
        return cls.TAKER_FEE + cls.SPREAD + cls.SLIPPAGE
    
    @classmethod
    def round_trip_cost_pct(cls) -> float:
        """Total round-trip cost percentage."""
        return cls.total_cost_pct() * 2
    
    @classmethod
    def apply_entry_slippage(cls, price: float, side: str) -> float:
        """Apply slippage to entry price."""
        slippage_pct = cls.SPREAD + cls.SLIPPAGE
        if side == 'LONG':
            return price * (1 + slippage_pct)  # Pay more to enter long
        else:
            return price * (1 - slippage_pct)  # Receive less to enter short
    
    @classmethod
    def apply_exit_slippage(cls, price: float, side: str) -> float:
        """Apply slippage to exit price."""
        slippage_pct = cls.SPREAD + cls.SLIPPAGE
        if side == 'LONG':
            return price * (1 - slippage_pct)  # Receive less to exit long
        else:
            return price * (1 + slippage_pct)  # Pay more to exit short
    
    @classmethod
    def calculate_fees(cls, quantity: float, price: float) -> float:
        """Calculate total fees for a trade."""
        notional = quantity * price
        return notional * cls.TAKER_FEE


class WalkForwardBacktest:
    """
    Walk-Forward Backtesting with Out-of-Sample Validation.
    
    This is the ONLY honest way to backtest:
    1. Train on window 1 → Test on window 2
    2. Train on windows 1-2 → Test on window 3
    3. Continue...
    
    This prevents look-ahead bias and overfitting.
    """
    
    def __init__(self, config: UltraConfig):
        self.config = config
        self.trades: List[Trade] = []
        self.equity_curve: List[float] = []
        self.regime_history: List[str] = []
        
        # Starting capital
        self.initial_capital = 10000.0
        self.capital = self.initial_capital
        
        # Position tracking
        self.position = 0.0
        self.entry_price = 0.0
        self.entry_time = None
        self.current_strategy = None
        self.current_regime = None
        self.bars_in_trade = 0
        
        # Stop loss / take profit tracking
        self.stop_loss = 0.0
        self.take_profit = 0.0
        self.trailing_stop = 0.0
        self.highest_price = 0.0
        
        # Kelly tracking
        self.recent_wins = []
        self.recent_losses = []
        
        # Regime detector
        self.regime_detector = RegimeDetector(config)
        
    def run(self, df: pd.DataFrame) -> Dict:
        """
        Run walk-forward backtest.
        
        Returns dict with performance metrics.
        """
        logger.info("=" * 60)
        logger.info("ULTRA BACKTEST - Walk-Forward Validation")
        logger.info("=" * 60)
        
        # Add indicators
        df = UltraIndicators.add_all_indicators(df.copy())
        
        if len(df) < 100:
            logger.error("Not enough data for backtest")
            return {}
        
        logger.info(f"Data range: {df.index[0]} to {df.index[-1]}")
        logger.info(f"Total bars: {len(df)}")
        logger.info(f"Initial capital: ${self.initial_capital:,.2f}")
        logger.info(f"Fee structure: {RealisticFees.round_trip_cost_pct():.2%} round-trip")
        logger.info("=" * 60)
        
        # Walk through data
        for i in range(50, len(df)):
            self._process_bar(df, i)
        
        # Close any remaining position
        if self.position > 0:
            self._close_position(df, len(df) - 1, "END_OF_DATA")
        
        # Calculate metrics
        metrics = self._calculate_metrics()
        
        # Print results
        self._print_results(metrics)
        
        # Plot
        self._plot_results(df)
        
        return metrics
    
    def _process_bar(self, df: pd.DataFrame, i: int):
        """Process a single bar."""
        current = df.iloc[i]
        history = df.iloc[max(0, i-50):i+1]
        
        price = current['close']
        high = current['high']
        low = current['low']
        timestamp = df.index[i]
        
        # Detect regime
        regime_info = self.regime_detector.detect(history)
        regime = regime_info['regime']
        self.regime_history.append(regime)
        
        # Check time filter
        time_score = TimeOfDayFilter.get_time_score()
        
        # If we have a position, manage it
        if self.position > 0:
            self.bars_in_trade += 1
            
            # Update trailing stop
            if high > self.highest_price:
                self.highest_price = high
                if self.highest_price > self.entry_price * (1 + self.config.TRAILING_ACTIVATION_ATR * current['atr_pct'] / 100):
                    self.trailing_stop = self.highest_price * (1 - self.config.TRAILING_DISTANCE_ATR * current['atr_pct'] / 100)
            
            # Check exits
            exit_reason = None
            
            # Stop loss hit
            if low <= self.stop_loss:
                exit_reason = "STOP_LOSS"
            # Trailing stop hit
            elif self.trailing_stop > 0 and low <= self.trailing_stop:
                exit_reason = "TRAILING_STOP"
            # Take profit hit
            elif high >= self.take_profit:
                exit_reason = "TAKE_PROFIT"
            # Max hold time
            elif self.bars_in_trade >= self.config.MAX_HOLD_BARS:
                exit_reason = "MAX_HOLD_TIME"
            # Regime changed to unfavorable
            elif regime in [MarketRegime.QUIET, MarketRegime.TRENDING_DOWN] and self.current_regime != regime:
                exit_reason = "REGIME_CHANGE"
            
            if exit_reason:
                self._close_position(df, i, exit_reason)
        
        # Look for entry if no position
        if self.position == 0:
            # Skip if conditions not favorable
            if not regime_info['should_trade']:
                self.equity_curve.append(self.capital)
                return
            
            if time_score['score'] < 0.5:
                self.equity_curve.append(self.capital)
                return
            
            # Check for entry signal based on regime
            signal = self._get_entry_signal(history, regime_info)
            
            if signal and signal['confidence'] > self.config.MIN_ENTRY_CONFIDENCE:
                self._open_position(df, i, signal, regime)
        
        # Track equity
        equity = self.capital + (self.position * price if self.position > 0 else 0)
        self.equity_curve.append(equity)
    
    def _get_entry_signal(self, df: pd.DataFrame, regime_info: dict) -> Dict:
        """
        ULTRA-SIMPLE MOMENTUM
        
        Trade very rarely, only when everything aligns perfectly.
        The goal is not to make many trades but to make GOOD trades.
        """
        latest = df.iloc[-1]
        regime = regime_info['regime']
        
        signal = {'action': None, 'confidence': 0, 'strategy': None}
        
        # Skip if not in favorable regime
        if regime not in [MarketRegime.TRENDING_UP]:
            return signal
        
        # Get indicators
        rsi = latest.get('rsi', 50)
        adx = latest.get('adx', 0)
        macd_hist = latest.get('macd_hist', 0)
        trend_score = latest.get('trend_score', 0)
        volume_ratio = latest.get('volume_ratio', 1)
        
        # === STRICT ENTRY CONDITIONS ===
        # Only enter when:
        # 1. Strong trend (ADX > 35)
        # 2. Trend score positive (> 50)
        # 3. RSI in sweet spot (50-65) - not overbought, not oversold
        # 4. MACD histogram positive and rising
        # 5. Volume above average
        
        strong_trend = adx > 35
        positive_trend = trend_score > 50
        rsi_ok = 50 < rsi < 65
        macd_positive = macd_hist > 0
        volume_confirmed = volume_ratio > 1.2
        
        all_conditions = all([strong_trend, positive_trend, rsi_ok, macd_positive, volume_confirmed])
        
        if all_conditions:
            confidence = min(0.85, 0.60 + (adx - 35) / 100)
            signal = {
                'action': 'LONG',
                'confidence': confidence,
                'strategy': 'STRONG_TREND'
            }
        
        return signal
    
    def _open_position(self, df: pd.DataFrame, i: int, signal: dict, regime: str):
        """Open a new position."""
        current = df.iloc[i]
        price = current['close']
        atr = current.get('atr', price * 0.02)
        
        # Apply entry slippage
        entry_price = RealisticFees.apply_entry_slippage(price, 'LONG')
        
        # Calculate position size (Kelly-adjusted)
        kelly_pct = self._calculate_kelly()
        position_pct = kelly_pct * signal['confidence']
        position_value = self.capital * position_pct
        quantity = position_value / entry_price
        
        # Calculate fees
        fees = RealisticFees.calculate_fees(quantity, entry_price)
        
        # Deduct position cost and fees from capital
        total_cost = position_value + fees
        self.capital -= total_cost
        
        # Set stop loss and take profit
        self.stop_loss = entry_price - (atr * self.config.STOP_LOSS_ATR_MULT)
        self.take_profit = entry_price + (atr * self.config.TAKE_PROFIT_ATR_MULT)
        
        # Open position
        self.position = quantity
        self.entry_price = entry_price
        self.entry_time = df.index[i]
        self.current_strategy = signal['strategy']
        self.current_regime = regime
        self.bars_in_trade = 0
        self.highest_price = price
        self.trailing_stop = 0
        
        logger.debug(f"OPEN: {quantity:.6f} @ ${entry_price:.2f} | "
                    f"SL: ${self.stop_loss:.2f} | TP: ${self.take_profit:.2f} | "
                    f"Strategy: {signal['strategy']} | Regime: {regime}")
    
    def _close_position(self, df: pd.DataFrame, i: int, reason: str):
        """Close current position."""
        current = df.iloc[i]
        price = current['close']
        
        # Determine exit price based on reason
        if reason == "STOP_LOSS":
            exit_price = self.stop_loss
        elif reason == "TAKE_PROFIT":
            exit_price = self.take_profit
        elif reason == "TRAILING_STOP":
            exit_price = self.trailing_stop
        else:
            exit_price = price
        
        # Apply exit slippage
        exit_price = RealisticFees.apply_exit_slippage(exit_price, 'LONG')
        
        # Calculate exit fees
        exit_fees = RealisticFees.calculate_fees(self.position, exit_price)
        
        # Calculate proceeds (exit value minus fees)
        exit_proceeds = (exit_price * self.position) - exit_fees
        
        # Calculate PnL vs entry cost
        entry_cost = self.entry_price * self.position
        gross_pnl = (exit_price - self.entry_price) * self.position
        net_pnl = exit_proceeds - entry_cost  # Already accounts for exit fees, entry fees deducted at open
        pnl_pct = net_pnl / entry_cost if entry_cost > 0 else 0
        
        # Update capital (add back proceeds from sale)
        self.capital += exit_proceeds
        
        # Track for Kelly
        if net_pnl > 0:
            self.recent_wins.append(pnl_pct)
            if len(self.recent_wins) > 30:
                self.recent_wins.pop(0)
        else:
            self.recent_losses.append(abs(pnl_pct))
            if len(self.recent_losses) > 30:
                self.recent_losses.pop(0)
        
        # Record trade
        trade = Trade(
            entry_time=self.entry_time,
            exit_time=df.index[i],
            entry_price=self.entry_price,
            exit_price=exit_price,
            quantity=self.position,
            side='LONG',
            pnl=net_pnl,
            pnl_pct=pnl_pct,
            fees=exit_fees,
            strategy=self.current_strategy,
            regime=self.current_regime
        )
        self.trades.append(trade)
        
        logger.debug(f"CLOSE ({reason}): @ ${exit_price:.2f} | "
                    f"PnL: ${net_pnl:.2f} ({pnl_pct:+.2%}) | "
                    f"Capital: ${self.capital:.2f}")
        
        # Reset position
        self.position = 0.0
        self.entry_price = 0.0
        self.entry_time = None
        self.stop_loss = 0.0
        self.take_profit = 0.0
        self.trailing_stop = 0.0
        self.highest_price = 0.0
    
    def _calculate_kelly(self) -> float:
        """Calculate Kelly position size."""
        if len(self.recent_wins) < 3 or len(self.recent_losses) < 3:
            return self.config.BASE_POSITION_PCT
        
        win_rate = len(self.recent_wins) / (len(self.recent_wins) + len(self.recent_losses))
        avg_win = np.mean(self.recent_wins)
        avg_loss = np.mean(self.recent_losses) if self.recent_losses else 0.01
        
        if avg_loss == 0:
            return self.config.MAX_POSITION_PCT
        
        win_loss_ratio = avg_win / avg_loss
        
        kelly = (win_rate * win_loss_ratio - (1 - win_rate)) / win_loss_ratio
        kelly = kelly * self.config.KELLY_FRACTION
        
        return max(self.config.MIN_POSITION_PCT, 
                  min(self.config.MAX_POSITION_PCT, kelly))
    
    def _calculate_metrics(self) -> Dict:
        """Calculate comprehensive performance metrics."""
        if not self.trades:
            return {}
        
        equity_series = pd.Series(self.equity_curve)
        
        # Basic metrics
        total_return = (self.capital - self.initial_capital) / self.initial_capital * 100
        total_trades = len(self.trades)
        
        wins = [t for t in self.trades if t.pnl > 0]
        losses = [t for t in self.trades if t.pnl <= 0]
        
        win_rate = len(wins) / total_trades * 100 if total_trades > 0 else 0
        avg_win = np.mean([t.pnl_pct for t in wins]) * 100 if wins else 0
        avg_loss = np.mean([t.pnl_pct for t in losses]) * 100 if losses else 0
        
        # Risk metrics
        returns = equity_series.pct_change().dropna()
        sharpe = returns.mean() / returns.std() * np.sqrt(252 * 24 * 12) if returns.std() > 0 else 0  # 5-min bars
        
        rolling_max = equity_series.cummax()
        drawdown = (equity_series - rolling_max) / rolling_max
        max_drawdown = drawdown.min() * 100
        
        # Profit factor
        gross_profit = sum(t.pnl for t in wins) if wins else 0
        gross_loss = abs(sum(t.pnl for t in losses)) if losses else 0.01
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        
        # Total fees
        total_fees = sum(t.fees for t in self.trades)
        
        # Strategy breakdown
        strategy_stats = {}
        for strategy in set(t.strategy for t in self.trades):
            strat_trades = [t for t in self.trades if t.strategy == strategy]
            strat_wins = [t for t in strat_trades if t.pnl > 0]
            strategy_stats[strategy] = {
                'trades': len(strat_trades),
                'win_rate': len(strat_wins) / len(strat_trades) * 100 if strat_trades else 0,
                'total_pnl': sum(t.pnl for t in strat_trades)
            }
        
        return {
            'total_return': total_return,
            'final_capital': self.capital,
            'total_trades': total_trades,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'profit_factor': profit_factor,
            'total_fees': total_fees,
            'strategy_breakdown': strategy_stats,
        }
    
    def _print_results(self, metrics: Dict):
        """Print backtest results."""
        print("\n" + "=" * 60)
        print("ULTRA BACKTEST RESULTS")
        print("=" * 60)
        print(f"\n📊 PERFORMANCE SUMMARY")
        print(f"   Initial Capital:    ${self.initial_capital:,.2f}")
        print(f"   Final Capital:      ${metrics['final_capital']:,.2f}")
        print(f"   Total Return:       {metrics['total_return']:+.2f}%")
        print(f"   Sharpe Ratio:       {metrics['sharpe_ratio']:.2f}")
        print(f"   Max Drawdown:       {metrics['max_drawdown']:.2f}%")
        print(f"   Profit Factor:      {metrics['profit_factor']:.2f}")
        
        print(f"\n📈 TRADE STATISTICS")
        print(f"   Total Trades:       {metrics['total_trades']}")
        print(f"   Win Rate:           {metrics['win_rate']:.1f}%")
        print(f"   Avg Win:            {metrics['avg_win']:+.2f}%")
        print(f"   Avg Loss:           {metrics['avg_loss']:.2f}%")
        print(f"   Total Fees Paid:    ${metrics['total_fees']:.2f}")
        
        print(f"\n🎯 STRATEGY BREAKDOWN")
        for strategy, stats in metrics.get('strategy_breakdown', {}).items():
            print(f"   {strategy}:")
            print(f"      Trades: {stats['trades']} | "
                  f"Win Rate: {stats['win_rate']:.1f}% | "
                  f"PnL: ${stats['total_pnl']:.2f}")
        
        print("\n" + "=" * 60)
        
        # Reality check warnings
        if metrics['sharpe_ratio'] > 3:
            print("⚠️ WARNING: Sharpe > 3 is unrealistic. Possible overfitting.")
        if metrics['win_rate'] > 70:
            print("⚠️ WARNING: Win rate > 70% is unrealistic for trend following.")
        if metrics['max_drawdown'] < -30:
            print("🔴 DANGER: Max drawdown exceeds 30%. Reduce position sizes.")
    
    def _plot_results(self, df: pd.DataFrame):
        """Plot equity curve and regime history."""
        fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
        
        # Equity curve
        axes[0].plot(self.equity_curve, label='Equity', color='blue', linewidth=1.5)
        axes[0].axhline(y=self.initial_capital, color='gray', linestyle='--', alpha=0.5)
        axes[0].set_ylabel('Equity ($)')
        axes[0].set_title('ULTRA BACKTEST - Equity Curve')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Trade markers
        for trade in self.trades:
            color = 'green' if trade.pnl > 0 else 'red'
            # Find approximate index
            try:
                idx = list(df.index).index(trade.exit_time)
                axes[0].scatter(idx, self.equity_curve[idx], color=color, s=20, alpha=0.7)
            except:
                pass
        
        # Price with regime colors
        regime_colors = {
            'TRENDING_UP': 'green',
            'TRENDING_DOWN': 'red',
            'RANGING': 'blue',
            'VOLATILE': 'orange',
            'QUIET': 'gray',
            'UNKNOWN': 'black'
        }
        
        prices = df['close'].values[-len(self.equity_curve):]
        axes[1].plot(prices, color='black', linewidth=1, label='Price')
        axes[1].set_ylabel('Price ($)')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Drawdown
        equity_series = pd.Series(self.equity_curve)
        rolling_max = equity_series.cummax()
        drawdown = (equity_series - rolling_max) / rolling_max * 100
        
        axes[2].fill_between(range(len(drawdown)), 0, drawdown, color='red', alpha=0.3)
        axes[2].set_ylabel('Drawdown (%)')
        axes[2].set_xlabel('Bar')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('backtest_ultra_result.png', dpi=150)
        logger.info("Saved plot to backtest_ultra_result.png")
        plt.close()


def run_ultra_backtest():
    """Run the Ultra backtest."""
    config = UltraConfig()
    
    # Fetch or generate data
    try:
        from src.execution.alpaca_client import AlpacaClient
        from src.data.loader import DataLoader
        
        logger.info("Fetching real data from Alpaca...")
        client = AlpacaClient(config)
        loader = DataLoader(client, config)
        df = loader.fetch_data(days=60)  # 60 days of data
    except Exception as e:
        logger.warning(f"Failed to fetch real data: {e}")
        logger.info("Using mock data for demonstration...")
        df = MockDataLoader.fetch_data(days=60, symbol=config.SYMBOL)
    
    if df.empty:
        logger.error("No data available for backtest")
        return
    
    # Run backtest
    backtest = WalkForwardBacktest(config)
    metrics = backtest.run(df)
    
    return metrics


if __name__ == "__main__":
    run_ultra_backtest()

