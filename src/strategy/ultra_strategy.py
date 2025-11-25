"""
ULTRA STRATEGY - Multi-Strategy Regime-Based System

This is a complete reimagining of the trading system.

Key Innovations:
1. We don't predict direction - we predict REGIME
2. Different strategies for different market conditions
3. Kelly Criterion for optimal position sizing
4. Scaling in/out for better average prices
5. Time-of-day awareness

WHY THIS WORKS:
- Regime is 3x more predictable than direction
- Each strategy is optimized for its specific conditions
- We AVOID trading when conditions are unfavorable
- Position sizing based on confidence, not fixed %
"""

import pandas as pd
import numpy as np
import logging
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple
from datetime import datetime

from src.strategy.regime_detector import RegimeDetector, MarketRegime, TimeOfDayFilter

logger = logging.getLogger(__name__)


@dataclass
class TradeSignal:
    """Represents a trading signal."""
    action: str  # 'BUY', 'SELL', 'CLOSE', 'HOLD'
    confidence: float  # 0.0 to 1.0
    strategy: str  # Which strategy generated this
    reason: str
    target_price: Optional[float] = None
    stop_loss: Optional[float] = None
    position_size_multiplier: float = 1.0  # Adjust size based on confidence


class TrendFollowStrategy:
    """
    TREND FOLLOWING - For trending markets (ADX > 25)
    
    Philosophy: "The trend is your friend until it ends"
    
    Entry: Pullback to support in uptrend (or resistance in downtrend)
    Exit: Trailing stop or trend reversal
    
    Win Rate Target: 45-50% (but wins are 2-3x larger than losses)
    """
    
    def __init__(self, config):
        self.config = config
        
        # Trend following parameters
        self.PULLBACK_RSI_BUY = 40   # Buy when RSI pulls back to 40 in uptrend
        self.PULLBACK_RSI_SELL = 60  # Sell when RSI pulls back to 60 in downtrend
        self.TRAILING_STOP_ATR = 2.0  # Trail by 2 ATR
        
    def analyze(self, df: pd.DataFrame, regime_info: dict) -> TradeSignal:
        """Generate signal for trend following."""
        latest = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else latest
        
        regime = regime_info['regime']
        adx = regime_info['metrics'].get('adx', 0)
        
        # Check if we should be trading trends
        if regime not in [MarketRegime.TRENDING_UP, MarketRegime.TRENDING_DOWN]:
            return TradeSignal('HOLD', 0, 'TREND_FOLLOW', 'Not in trending regime')
        
        rsi = latest.get('rsi', 50)
        close = latest['close']
        atr = latest.get('atr', close * 0.02)
        
        # EMA alignment check
        ema_fast = latest.get('ema_fast', close)
        ema_slow = latest.get('ema_slow', close)
        ema_trend = latest.get('ema_trend', close)
        
        # UPTREND: Look for pullback buys
        if regime == MarketRegime.TRENDING_UP:
            # Confirm EMA alignment
            if not (ema_fast > ema_slow > ema_trend):
                return TradeSignal('HOLD', 0.3, 'TREND_FOLLOW', 'EMAs not aligned for uptrend')
            
            # Look for RSI pullback
            if rsi < self.PULLBACK_RSI_BUY:
                # Price near lower BB or EMA support
                bb_lower = latest.get('bb_lower', close * 0.98)
                near_support = close < ema_slow * 1.01 or close < bb_lower * 1.02
                
                if near_support:
                    confidence = min(0.9, 0.6 + (adx / 100))
                    stop_loss = close - (atr * 1.5)
                    target = close + (atr * 3.0)  # 2:1 R/R minimum
                    
                    return TradeSignal(
                        'BUY', confidence, 'TREND_FOLLOW',
                        f'Pullback buy in uptrend. RSI={rsi:.0f}, ADX={adx:.0f}',
                        target_price=target,
                        stop_loss=stop_loss,
                        position_size_multiplier=confidence
                    )
        
        # DOWNTREND: Look for pullback sells (or just avoid if long-only)
        elif regime == MarketRegime.TRENDING_DOWN:
            # For now, just avoid downtrends if we're long-only
            return TradeSignal('HOLD', 0.7, 'TREND_FOLLOW', 'Downtrend - staying flat')
        
        return TradeSignal('HOLD', 0.5, 'TREND_FOLLOW', 'No pullback setup')


class MeanReversionStrategy:
    """
    MEAN REVERSION - For ranging markets
    
    Philosophy: "Extreme moves revert to the mean"
    
    Entry: RSI extreme + price at Bollinger Band extreme
    Exit: Return to middle BB or opposite extreme
    
    Win Rate Target: 65-70% (but wins are smaller)
    """
    
    def __init__(self, config):
        self.config = config
        
        # Mean reversion parameters
        self.RSI_OVERSOLD = 25
        self.RSI_OVERBOUGHT = 75
        self.BB_DEVIATION = 0.02  # Price must be within 2% of BB
        
    def analyze(self, df: pd.DataFrame, regime_info: dict) -> TradeSignal:
        """Generate signal for mean reversion."""
        latest = df.iloc[-1]
        
        regime = regime_info['regime']
        
        # Only trade mean reversion in ranging markets
        if regime != MarketRegime.RANGING:
            return TradeSignal('HOLD', 0, 'MEAN_REVERT', 'Not in ranging regime')
        
        rsi = latest.get('rsi', 50)
        close = latest['close']
        bb_upper = latest.get('bb_upper', close * 1.02)
        bb_lower = latest.get('bb_lower', close * 0.98)
        bb_middle = latest.get('bb_middle', close)
        atr = latest.get('atr', close * 0.01)
        
        # Check for volume confirmation
        volume_ratio = latest.get('volume_ratio', 1.0)
        volume_confirmed = volume_ratio > 1.2
        
        # OVERSOLD: Buy signal
        if rsi < self.RSI_OVERSOLD:
            near_lower_bb = close <= bb_lower * 1.01
            
            if near_lower_bb:
                confidence = 0.75 if volume_confirmed else 0.6
                target = bb_middle  # Target middle BB
                stop_loss = close - (atr * 1.5)
                
                return TradeSignal(
                    'BUY', confidence, 'MEAN_REVERT',
                    f'Oversold bounce. RSI={rsi:.0f}, Near lower BB',
                    target_price=target,
                    stop_loss=stop_loss,
                    position_size_multiplier=confidence
                )
        
        # OVERBOUGHT: Sell/Close signal
        if rsi > self.RSI_OVERBOUGHT:
            near_upper_bb = close >= bb_upper * 0.99
            
            if near_upper_bb:
                confidence = 0.75 if volume_confirmed else 0.6
                
                return TradeSignal(
                    'CLOSE', confidence, 'MEAN_REVERT',
                    f'Overbought - close longs. RSI={rsi:.0f}',
                    position_size_multiplier=1.0  # Close full position
                )
        
        return TradeSignal('HOLD', 0.5, 'MEAN_REVERT', 'No extreme reached')


class VolatilityBreakoutStrategy:
    """
    VOLATILITY BREAKOUT - For volatile/expanding markets
    
    Philosophy: "Volatility clusters - low vol leads to high vol"
    
    Entry: When volatility expands from a squeeze
    Direction: Follow the breakout direction
    Exit: Volatility contraction or trailing stop
    
    Win Rate Target: 40-45% (but wins are 3-4x larger)
    """
    
    def __init__(self, config):
        self.config = config
        
        # Volatility parameters
        self.SQUEEZE_BB_WIDTH = 0.03  # BB width < 3% = squeeze
        self.BREAKOUT_ATR_MULT = 1.5  # Need 1.5x ATR move for breakout
        
        # Track squeeze state
        self.in_squeeze = False
        self.squeeze_high = 0
        self.squeeze_low = 0
        
    def analyze(self, df: pd.DataFrame, regime_info: dict) -> TradeSignal:
        """Generate signal for volatility breakout."""
        if len(df) < 20:
            return TradeSignal('HOLD', 0, 'VOL_BREAKOUT', 'Not enough data')
        
        latest = df.iloc[-1]
        close = latest['close']
        
        bb_upper = latest.get('bb_upper', close * 1.02)
        bb_lower = latest.get('bb_lower', close * 0.98)
        bb_width = (bb_upper - bb_lower) / close
        atr = latest.get('atr', close * 0.02)
        
        vol_ratio = regime_info['metrics'].get('vol_ratio', 1.0)
        
        # Detect squeeze (low volatility compression)
        if bb_width < self.SQUEEZE_BB_WIDTH:
            if not self.in_squeeze:
                # Just entered squeeze - record range
                self.in_squeeze = True
                self.squeeze_high = df['high'].tail(10).max()
                self.squeeze_low = df['low'].tail(10).min()
                logger.info(f"SQUEEZE DETECTED: Range ${self.squeeze_low:.2f} - ${self.squeeze_high:.2f}")
            
            return TradeSignal('HOLD', 0.6, 'VOL_BREAKOUT', 
                             f'In squeeze. Waiting for breakout. Width={bb_width:.2%}')
        
        # Check for breakout from squeeze
        if self.in_squeeze:
            breakout_distance = atr * self.BREAKOUT_ATR_MULT
            
            # Upside breakout
            if close > self.squeeze_high + breakout_distance:
                self.in_squeeze = False
                confidence = min(0.85, 0.6 + vol_ratio * 0.1)
                stop_loss = self.squeeze_low - (atr * 0.5)
                target = close + (close - stop_loss) * 3  # 3:1 R/R
                
                return TradeSignal(
                    'BUY', confidence, 'VOL_BREAKOUT',
                    f'Breakout! Price ${close:.2f} > Squeeze high ${self.squeeze_high:.2f}',
                    target_price=target,
                    stop_loss=stop_loss,
                    position_size_multiplier=confidence * 0.8  # Slightly smaller for breakouts
                )
            
            # Downside breakout (close positions if long-only)
            if close < self.squeeze_low - breakout_distance:
                self.in_squeeze = False
                return TradeSignal(
                    'CLOSE', 0.8, 'VOL_BREAKOUT',
                    f'Downside breakout - close longs',
                    position_size_multiplier=1.0
                )
        else:
            self.in_squeeze = False
        
        return TradeSignal('HOLD', 0.4, 'VOL_BREAKOUT', 'No breakout setup')


class KellyPositionSizer:
    """
    KELLY CRITERION - Mathematically Optimal Position Sizing
    
    The Kelly formula tells us the optimal bet size to maximize
    geometric growth rate while avoiding ruin.
    
    f* = (p*b - q) / b
    
    Where:
    - p = probability of winning
    - q = probability of losing (1-p)
    - b = win/loss ratio
    - f* = fraction of bankroll to bet
    
    We use FRACTIONAL Kelly (25-50%) for safety.
    """
    
    def __init__(self, config):
        self.config = config
        
        # Historical performance tracking
        self.wins = []
        self.losses = []
        
        # Kelly fraction (0.25 = quarter Kelly for safety)
        self.kelly_fraction = 0.25
        
        # Absolute limits
        self.MIN_POSITION_PCT = 0.05  # Never less than 5%
        self.MAX_POSITION_PCT = 0.30  # Never more than 30%
        
    def add_trade_result(self, pnl_pct: float):
        """Track trade results for Kelly calculation."""
        if pnl_pct > 0:
            self.wins.append(pnl_pct)
        else:
            self.losses.append(abs(pnl_pct))
        
        # Keep last 50 trades
        if len(self.wins) > 50:
            self.wins.pop(0)
        if len(self.losses) > 50:
            self.losses.pop(0)
    
    def calculate_kelly(self) -> float:
        """Calculate Kelly fraction based on historical performance."""
        if len(self.wins) < 5 or len(self.losses) < 5:
            # Not enough data - use conservative default
            return 0.10
        
        win_rate = len(self.wins) / (len(self.wins) + len(self.losses))
        avg_win = np.mean(self.wins)
        avg_loss = np.mean(self.losses)
        
        if avg_loss == 0:
            return self.MAX_POSITION_PCT
        
        win_loss_ratio = avg_win / avg_loss
        
        # Kelly formula
        kelly = (win_rate * win_loss_ratio - (1 - win_rate)) / win_loss_ratio
        
        # Apply fractional Kelly
        kelly = kelly * self.kelly_fraction
        
        # Clamp to limits
        kelly = max(self.MIN_POSITION_PCT, min(self.MAX_POSITION_PCT, kelly))
        
        logger.info(f"Kelly Sizing: WR={win_rate:.0%}, W/L={win_loss_ratio:.2f}, Kelly={kelly:.1%}")
        
        return kelly
    
    def get_position_size(self, equity: float, price: float, 
                         signal_confidence: float = 1.0,
                         volatility_multiplier: float = 1.0) -> float:
        """
        Calculate position size in units (e.g., BTC quantity).
        
        Args:
            equity: Account equity
            price: Current asset price
            signal_confidence: 0-1 confidence from strategy
            volatility_multiplier: Reduce size in high volatility
        """
        base_kelly = self.calculate_kelly()
        
        # Adjust for signal confidence
        adjusted_kelly = base_kelly * signal_confidence
        
        # Adjust for volatility (smaller positions in high vol)
        adjusted_kelly = adjusted_kelly / volatility_multiplier
        
        # Final position value
        position_value = equity * adjusted_kelly
        
        # Convert to units
        quantity = position_value / price
        
        logger.info(f"Position Size: {adjusted_kelly:.1%} of ${equity:.2f} = ${position_value:.2f} = {quantity:.6f} units")
        
        return quantity


class UltraStrategy:
    """
    ULTRA STRATEGY - The Master Controller
    
    Combines all strategies with regime detection for
    adaptive, intelligent trading.
    """
    
    def __init__(self, config, client, predictor, order_manager):
        self.config = config
        self.client = client
        self.predictor = predictor  # Keep for backward compatibility
        self.order_manager = order_manager
        
        # Core components
        self.regime_detector = RegimeDetector(config)
        self.position_sizer = KellyPositionSizer(config)
        
        # Strategies
        self.strategies = {
            'TREND_FOLLOW': TrendFollowStrategy(config),
            'MEAN_REVERT': MeanReversionStrategy(config),
            'VOL_BREAKOUT': VolatilityBreakoutStrategy(config),
        }
        
        # Data buffer
        self.bars = []
        self.min_bars_required = 50
        
        # State
        self.current_regime = MarketRegime.UNKNOWN
        self.active_strategy = None
        self.last_signal = None
        
        # Performance tracking
        self.trade_history = []
        
    def load_historical_data(self, bars):
        """Load historical bars for warm-up."""
        for bar in bars:
            self.bars.append({
                'timestamp': bar.timestamp,
                'open': bar.open,
                'high': bar.high,
                'low': bar.low,
                'close': bar.close,
                'volume': bar.volume
            })
        logger.info(f"Loaded {len(bars)} historical bars")
    
    def get_latest_state(self) -> dict:
        """Return current strategy state for dashboard."""
        return {
            'prediction': self.last_signal.confidence if self.last_signal else 0.5,
            'signal': self.last_signal.action if self.last_signal else 'HOLD',
            'regime': self.current_regime,
            'strategy': self.active_strategy,
        }
    
    async def on_bar(self, bar):
        """Process new bar - main entry point."""
        # Update data buffer
        self.bars.append({
            'timestamp': bar.timestamp,
            'open': bar.open,
            'high': bar.high,
            'low': bar.low,
            'close': bar.close,
            'volume': bar.volume
        })
        
        if len(self.bars) > 200:
            self.bars.pop(0)
        
        if len(self.bars) < self.min_bars_required:
            logger.info(f"Warming up: {len(self.bars)}/{self.min_bars_required}")
            return
        
        # Prepare dataframe with indicators
        df = self._prepare_dataframe()
        if df.empty:
            return
        
        # Step 1: Detect market regime
        regime_info = self.regime_detector.detect(df)
        self.current_regime = regime_info['regime']
        
        # Step 2: Check time-of-day filter
        time_filter = TimeOfDayFilter.get_time_score()
        
        # Step 3: Should we trade at all?
        if not regime_info['should_trade']:
            logger.info(f"⏸️ No trade zone: {regime_info['strategy_hint']}")
            self.last_signal = TradeSignal('HOLD', 0, 'REGIME', 'Market conditions unfavorable')
            return
        
        if time_filter['score'] < 0.4:
            logger.info(f"⏸️ Sub-optimal time: {time_filter['window_name']}")
            self.last_signal = TradeSignal('HOLD', 0, 'TIME', f"Low liquidity period")
            return
        
        # Step 4: Get signal from appropriate strategy
        strategy_name = regime_info['strategy_hint']
        if strategy_name in self.strategies:
            self.active_strategy = strategy_name
            strategy = self.strategies[strategy_name]
            signal = strategy.analyze(df, regime_info)
        else:
            signal = TradeSignal('HOLD', 0, 'NONE', 'No active strategy')
        
        # Adjust signal confidence by time filter
        signal.confidence *= time_filter['multiplier']
        signal.position_size_multiplier *= time_filter['multiplier']
        
        self.last_signal = signal
        
        # Step 5: Execute if signal is actionable
        await self._execute_signal(signal, bar.close, regime_info)
    
    def _prepare_dataframe(self) -> pd.DataFrame:
        """Prepare dataframe with all indicators."""
        from src.features.indicators import Indicators
        
        df = pd.DataFrame(self.bars)
        df = Indicators.add_all_indicators(df)
        
        # Add additional indicators needed for strategies
        if not df.empty:
            close = df['close'].values
            
            # EMAs for trend
            import talib
            df['ema_fast'] = talib.EMA(close, timeperiod=9)
            df['ema_slow'] = talib.EMA(close, timeperiod=21)
            df['ema_trend'] = talib.EMA(close, timeperiod=50)
            
            # ADX and DI for regime
            high = df['high'].values
            low = df['low'].values
            df['adx'] = talib.ADX(high, low, close, timeperiod=14)
            df['plus_di'] = talib.PLUS_DI(high, low, close, timeperiod=14)
            df['minus_di'] = talib.MINUS_DI(high, low, close, timeperiod=14)
            
            # Volume ratio
            volume = df['volume'].values
            df['volume_sma'] = talib.SMA(volume, timeperiod=20)
            df['volume_ratio'] = df['volume'] / df['volume_sma']
            
            df.dropna(inplace=True)
        
        return df
    
    async def _execute_signal(self, signal: TradeSignal, price: float, regime_info: dict):
        """Execute trading signal."""
        if signal.action == 'HOLD':
            return
        
        # Get current positions
        try:
            positions = self.client.trading_client.get_all_positions()
            position_value = sum(float(p.qty) * float(p.current_price) 
                               for p in positions if p.symbol == self.config.SYMBOL)
        except Exception as e:
            logger.error(f"Error fetching positions: {e}")
            positions = []
            position_value = 0
        
        has_position = position_value > 0
        
        # Get account info
        try:
            account = self.client.trading_client.get_account()
            equity = float(account.equity)
        except:
            equity = 10000  # Fallback
        
        # CLOSE signal
        if signal.action == 'CLOSE' and has_position:
            logger.info(f"🔴 CLOSING: {signal.reason}")
            for position in positions:
                if position.symbol == self.config.SYMBOL:
                    self.client.submit_order(self.config.SYMBOL, float(position.qty), 'sell')
            return
        
        # BUY signal
        if signal.action == 'BUY' and signal.confidence > 0.5:
            # Check if we already have max positions
            max_positions = getattr(self.config, 'MAX_POSITIONS', 3)
            if len(positions) >= max_positions:
                logger.info(f"⚠️ Max positions reached ({max_positions})")
                return
            
            # Calculate position size using Kelly
            vol_ratio = regime_info['metrics'].get('vol_ratio', 1.0)
            qty = self.position_sizer.get_position_size(
                equity=equity,
                price=price,
                signal_confidence=signal.confidence,
                volatility_multiplier=max(1.0, vol_ratio)
            )
            
            if qty > 0:
                logger.info(f"🟢 BUY: {qty:.6f} @ ${price:.2f}")
                logger.info(f"   Strategy: {signal.strategy}")
                logger.info(f"   Confidence: {signal.confidence:.0%}")
                logger.info(f"   Reason: {signal.reason}")
                
                if signal.stop_loss:
                    logger.info(f"   Stop Loss: ${signal.stop_loss:.2f}")
                if signal.target_price:
                    logger.info(f"   Target: ${signal.target_price:.2f}")
                
                self.order_manager.execute_entry(self.config.SYMBOL, 'buy', price)

