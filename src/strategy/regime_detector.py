"""
REGIME DETECTOR - The Brain of the Ultra System

Instead of predicting price direction (nearly impossible),
we predict MARKET REGIME (much more predictable).

Regimes:
- TRENDING_UP: Strong uptrend, trade with momentum
- TRENDING_DOWN: Strong downtrend, trade with momentum (or stay out if long-only)
- RANGING: Sideways, mean reversion works best
- VOLATILE: High volatility expansion, breakout strategies
- QUIET: Low volatility, AVOID trading (no edge)

Key Insight: The regime is 3x more predictable than direction!
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


class MarketRegime:
    TRENDING_UP = "TRENDING_UP"
    TRENDING_DOWN = "TRENDING_DOWN"
    RANGING = "RANGING"
    VOLATILE = "VOLATILE"
    QUIET = "QUIET"
    UNKNOWN = "UNKNOWN"


class RegimeDetector:
    """
    Detects market regime using multiple timeframe analysis.
    
    This is the SECRET SAUCE - we don't predict direction,
    we predict the TYPE of market, then apply the right strategy.
    """
    
    def __init__(self, config):
        self.config = config
        
        # Regime detection thresholds
        self.ADX_TRENDING = 25      # ADX > 25 = trending
        self.ADX_STRONG_TREND = 40  # ADX > 40 = strong trend
        self.ATR_EXPANSION = 1.5    # ATR > 1.5x average = volatile
        self.ATR_CONTRACTION = 0.7  # ATR < 0.7x average = quiet
        self.BB_WIDTH_SQUEEZE = 0.02  # BB width < 2% = squeeze (pending breakout)
        
        # Track regime history for stability
        self.regime_history = []
        self.regime_stability_count = 3  # Need 3 consecutive same regime to confirm
        
    def detect(self, df: pd.DataFrame) -> dict:
        """
        Main detection function.
        
        Returns dict with:
        - regime: Current market regime
        - confidence: How confident we are (0-1)
        - strategy_hint: Which strategy to use
        - should_trade: Whether conditions favor trading at all
        """
        if len(df) < 50:
            return self._no_data_response()
        
        latest = df.iloc[-1]
        
        # Calculate regime indicators
        adx = latest.get('adx', 0)
        plus_di = latest.get('plus_di', 0)
        minus_di = latest.get('minus_di', 0)
        atr = latest.get('atr', 0)
        atr_sma = df['atr'].rolling(20).mean().iloc[-1] if 'atr' in df.columns else atr
        bb_upper = latest.get('bb_upper', 0)
        bb_lower = latest.get('bb_lower', 0)
        bb_width = (bb_upper - bb_lower) / latest['close'] if latest['close'] > 0 else 0
        rsi = latest.get('rsi', 50)
        
        # Volatility ratio
        vol_ratio = atr / atr_sma if atr_sma > 0 else 1.0
        
        # Determine regime
        regime, confidence = self._classify_regime(
            adx, plus_di, minus_di, vol_ratio, bb_width, rsi
        )
        
        # Get strategy hint based on regime
        strategy_hint, should_trade = self._get_strategy_hint(regime, confidence, vol_ratio)
        
        # Update regime history for stability
        self.regime_history.append(regime)
        if len(self.regime_history) > 10:
            self.regime_history.pop(0)
        
        # Check regime stability
        stable_regime = self._get_stable_regime()
        
        result = {
            'regime': stable_regime,
            'instant_regime': regime,
            'confidence': confidence,
            'strategy_hint': strategy_hint,
            'should_trade': should_trade,
            'metrics': {
                'adx': adx,
                'vol_ratio': vol_ratio,
                'bb_width': bb_width,
                'rsi': rsi,
                'plus_di': plus_di,
                'minus_di': minus_di
            }
        }
        
        logger.info(f"Regime: {stable_regime} | Confidence: {confidence:.0%} | Strategy: {strategy_hint}")
        
        return result
    
    def _classify_regime(self, adx, plus_di, minus_di, vol_ratio, bb_width, rsi):
        """
        Core classification logic.
        """
        confidence = 0.5
        
        # QUIET REGIME - Low volatility, avoid trading
        if vol_ratio < self.ATR_CONTRACTION:
            # Check if it's a squeeze (pending breakout)
            if bb_width < self.BB_WIDTH_SQUEEZE:
                return MarketRegime.QUIET, 0.7  # Squeeze - wait for breakout
            return MarketRegime.QUIET, 0.6
        
        # VOLATILE REGIME - High volatility expansion
        if vol_ratio > self.ATR_EXPANSION:
            confidence = min(0.9, 0.5 + (vol_ratio - 1.5) * 0.2)
            return MarketRegime.VOLATILE, confidence
        
        # TRENDING REGIMES
        if adx > self.ADX_TRENDING:
            if adx > self.ADX_STRONG_TREND:
                confidence = 0.85
            else:
                confidence = 0.65
            
            if plus_di > minus_di:
                return MarketRegime.TRENDING_UP, confidence
            else:
                return MarketRegime.TRENDING_DOWN, confidence
        
        # RANGING REGIME - Default when nothing else matches
        # RSI in middle range confirms ranging
        if 35 < rsi < 65:
            confidence = 0.7
        else:
            confidence = 0.55  # RSI at extremes might mean trend starting
        
        return MarketRegime.RANGING, confidence
    
    def _get_strategy_hint(self, regime, confidence, vol_ratio):
        """
        Returns which strategy to use and whether to trade at all.
        """
        strategies = {
            MarketRegime.TRENDING_UP: ("TREND_FOLLOW", True),
            MarketRegime.TRENDING_DOWN: ("TREND_FOLLOW", True),  # Can short or stay out
            MarketRegime.RANGING: ("MEAN_REVERT", True),
            MarketRegime.VOLATILE: ("VOLATILITY_BREAKOUT", True),
            MarketRegime.QUIET: ("WAIT", False),  # Don't trade quiet markets!
            MarketRegime.UNKNOWN: ("WAIT", False),
        }
        
        strategy, should_trade = strategies.get(regime, ("WAIT", False))
        
        # Override: Don't trade with low confidence
        if confidence < 0.5:
            should_trade = False
            strategy = "WAIT"
        
        return strategy, should_trade
    
    def _get_stable_regime(self):
        """
        Returns regime only if it's been stable for N periods.
        Prevents whipsawing between regimes.
        """
        if len(self.regime_history) < self.regime_stability_count:
            return MarketRegime.UNKNOWN
        
        recent = self.regime_history[-self.regime_stability_count:]
        if all(r == recent[0] for r in recent):
            return recent[0]
        
        # Return most common regime in recent history
        from collections import Counter
        return Counter(self.regime_history[-5:]).most_common(1)[0][0]
    
    def _no_data_response(self):
        return {
            'regime': MarketRegime.UNKNOWN,
            'instant_regime': MarketRegime.UNKNOWN,
            'confidence': 0,
            'strategy_hint': "WAIT",
            'should_trade': False,
            'metrics': {}
        }


class TimeOfDayFilter:
    """
    Filters trades based on optimal trading windows.
    
    Crypto trades 24/7, but not all hours are created equal.
    Institutional money moves during specific windows.
    """
    
    # Optimal windows (in UTC)
    OPTIMAL_WINDOWS = [
        (13, 17),  # US Market Hours (9AM-1PM ET = 1PM-5PM UTC)
        (1, 4),    # Asian Session Start (8PM-11PM ET = 1AM-4AM UTC)
        (7, 9),    # London Open (3AM-5AM ET = 7AM-9AM UTC)
    ]
    
    # Hours to AVOID
    AVOID_HOURS = [
        (22, 24),  # Low liquidity period
        (5, 7),    # Between sessions
    ]
    
    # Weekend penalty (Saturday 00:00 to Sunday 23:59 UTC)
    WEEKEND_MULTIPLIER = 0.5  # Reduce position size on weekends
    
    @classmethod
    def get_time_score(cls, timestamp=None) -> dict:
        """
        Returns a score for current time.
        
        Returns:
        - score: 0.0 to 1.0 (1.0 = optimal, 0.0 = avoid)
        - is_weekend: bool
        - window_name: str
        """
        from datetime import datetime
        
        if timestamp is None:
            timestamp = datetime.utcnow()
        
        hour = timestamp.hour
        weekday = timestamp.weekday()  # 0=Monday, 6=Sunday
        
        is_weekend = weekday >= 5
        
        # Check if in avoid hours
        for start, end in cls.AVOID_HOURS:
            if start <= hour < end:
                return {
                    'score': 0.2,
                    'is_weekend': is_weekend,
                    'window_name': 'LOW_LIQUIDITY',
                    'multiplier': 0.3 if is_weekend else 0.5
                }
        
        # Check if in optimal window
        for i, (start, end) in enumerate(cls.OPTIMAL_WINDOWS):
            if start <= hour < end:
                window_names = ['US_SESSION', 'ASIAN_SESSION', 'LONDON_OPEN']
                return {
                    'score': 1.0,
                    'is_weekend': is_weekend,
                    'window_name': window_names[i],
                    'multiplier': 0.7 if is_weekend else 1.0
                }
        
        # Default: neutral hours
        return {
            'score': 0.6,
            'is_weekend': is_weekend,
            'window_name': 'NEUTRAL',
            'multiplier': 0.5 if is_weekend else 0.8
        }

