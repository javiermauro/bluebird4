"""
ULTRA INDICATORS - Enhanced Technical Analysis

Key Innovations:
1. Volatility-adjusted indicators (normalize for market conditions)
2. Multi-timeframe indicators (trend on higher TF, entry on lower TF)
3. Custom composite indicators (combine multiple signals)
4. Regime-specific indicators (different for trending vs ranging)

Philosophy: LESS IS MORE
- We use fewer indicators, but they're more meaningful
- Each indicator serves a specific purpose
- No redundant/correlated indicators
"""

import pandas as pd
import numpy as np
import talib
import logging

logger = logging.getLogger(__name__)


class UltraIndicators:
    """
    Enhanced indicator suite for the Ultra Strategy System.
    """
    
    @staticmethod
    def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """
        Add all indicators needed for the Ultra system.
        
        Categories:
        1. Trend Indicators (direction)
        2. Momentum Indicators (speed/strength)
        3. Volatility Indicators (risk/regime)
        4. Volume Indicators (confirmation)
        5. Composite Indicators (multi-signal)
        """
        if df.empty or len(df) < 50:
            return df
        
        df.columns = [c.lower() for c in df.columns]
        
        close = df['close'].values.astype(float)
        high = df['high'].values.astype(float)
        low = df['low'].values.astype(float)
        volume = df['volume'].values.astype(float)
        
        # ============================================
        # 1. TREND INDICATORS
        # ============================================
        
        # EMAs for trend direction (multiple timeframes in one)
        df['ema_9'] = talib.EMA(close, timeperiod=9)
        df['ema_21'] = talib.EMA(close, timeperiod=21)
        df['ema_50'] = talib.EMA(close, timeperiod=50)
        
        # Trend strength (distance between fast and slow EMA)
        df['trend_strength'] = (df['ema_9'] - df['ema_50']) / df['ema_50'] * 100
        
        # ADX for trend existence (not direction)
        df['adx'] = talib.ADX(high, low, close, timeperiod=14)
        df['plus_di'] = talib.PLUS_DI(high, low, close, timeperiod=14)
        df['minus_di'] = talib.MINUS_DI(high, low, close, timeperiod=14)
        
        # Trend direction composite
        df['trend_direction'] = np.where(
            (df['ema_9'] > df['ema_21']) & (df['ema_21'] > df['ema_50']),
            1,  # Uptrend
            np.where(
                (df['ema_9'] < df['ema_21']) & (df['ema_21'] < df['ema_50']),
                -1,  # Downtrend
                0  # No clear trend
            )
        )
        
        # ============================================
        # 2. MOMENTUM INDICATORS
        # ============================================
        
        # RSI - classic momentum
        df['rsi'] = talib.RSI(close, timeperiod=14)
        
        # RSI with dynamic thresholds (normalized)
        rsi_mean = df['rsi'].rolling(50).mean()
        rsi_std = df['rsi'].rolling(50).std()
        df['rsi_zscore'] = (df['rsi'] - rsi_mean) / rsi_std
        
        # MACD - momentum with trend
        macd, macd_signal, macd_hist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
        df['macd'] = macd
        df['macd_signal'] = macd_signal
        df['macd_hist'] = macd_hist
        
        # Normalize MACD by price (percentage-based)
        df['macd_pct'] = df['macd'] / df['close'] * 100
        
        # Rate of Change - simple momentum
        df['roc_5'] = talib.ROC(close, timeperiod=5)
        df['roc_10'] = talib.ROC(close, timeperiod=10)
        
        # Stochastic RSI - momentum of momentum
        fastk, fastd = talib.STOCHRSI(close, timeperiod=14, fastk_period=3, fastd_period=3)
        df['stoch_rsi_k'] = fastk
        df['stoch_rsi_d'] = fastd
        
        # ============================================
        # 3. VOLATILITY INDICATORS
        # ============================================
        
        # ATR - absolute volatility
        df['atr'] = talib.ATR(high, low, close, timeperiod=14)
        
        # ATR as percentage of price (normalized)
        df['atr_pct'] = df['atr'] / df['close'] * 100
        
        # ATR ratio (current vs average) - volatility expansion/contraction
        atr_sma = df['atr'].rolling(20).mean()
        df['atr_ratio'] = df['atr'] / atr_sma
        
        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
        df['bb_upper'] = bb_upper
        df['bb_middle'] = bb_middle
        df['bb_lower'] = bb_lower
        
        # BB Width (volatility measure)
        df['bb_width'] = (bb_upper - bb_lower) / bb_middle
        
        # BB Position (where price is within bands: 0=lower, 1=upper)
        df['bb_position'] = (df['close'] - bb_lower) / (bb_upper - bb_lower)
        
        # Keltner Channels (for squeeze detection)
        keltner_middle = talib.EMA(close, timeperiod=20)
        keltner_range = df['atr'] * 1.5
        df['keltner_upper'] = keltner_middle + keltner_range
        df['keltner_lower'] = keltner_middle - keltner_range
        
        # Squeeze indicator (BB inside Keltner = squeeze)
        df['squeeze'] = ((df['bb_lower'] > df['keltner_lower']) & 
                        (df['bb_upper'] < df['keltner_upper'])).astype(int)
        
        # ============================================
        # 4. VOLUME INDICATORS
        # ============================================
        
        # Volume SMA
        df['volume_sma'] = talib.SMA(volume, timeperiod=20)
        
        # Volume ratio
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # On-Balance Volume (cumulative volume direction)
        df['obv'] = talib.OBV(close, volume)
        
        # OBV momentum
        df['obv_ema'] = talib.EMA(df['obv'].values, timeperiod=20)
        df['obv_momentum'] = (df['obv'] - df['obv_ema']) / abs(df['obv_ema']) * 100
        
        # Volume Price Trend
        df['vpt'] = (df['close'].pct_change() * df['volume']).cumsum()
        
        # ============================================
        # 5. COMPOSITE INDICATORS
        # ============================================
        
        # Trend Score (-100 to +100)
        # Combines: EMA alignment, ADX strength, MACD direction
        ema_score = df['trend_direction'] * 30
        adx_score = np.minimum(df['adx'], 50) * df['trend_direction'] * 0.6
        macd_score = np.sign(df['macd_hist']) * 20
        
        df['trend_score'] = ema_score + adx_score + macd_score
        
        # Mean Reversion Score (0 to 100)
        # High score = good mean reversion opportunity
        rsi_extreme = np.maximum(0, (25 - df['rsi']).clip(0, 25) + (df['rsi'] - 75).clip(0, 25))
        bb_extreme = np.maximum(0, (0.1 - df['bb_position']).clip(0, 0.1) * 10 + 
                                  (df['bb_position'] - 0.9).clip(0, 0.1) * 10) * 50
        
        df['mean_revert_score'] = rsi_extreme * 2 + bb_extreme
        
        # Volatility Regime (QUIET, NORMAL, VOLATILE)
        df['vol_regime'] = pd.cut(
            df['atr_ratio'],
            bins=[0, 0.7, 1.3, float('inf')],
            labels=['QUIET', 'NORMAL', 'VOLATILE']
        )
        
        # Trade Quality Score (0 to 100)
        # Higher = better trade conditions
        volume_quality = (df['volume_ratio'] - 0.5).clip(0, 2) * 25
        volatility_quality = 50 - abs(df['atr_ratio'] - 1) * 30
        trend_quality = abs(df['trend_score']) * 0.3
        
        df['trade_quality'] = (volume_quality + volatility_quality + trend_quality).clip(0, 100)
        
        # Drop NaN
        df.dropna(inplace=True)
        
        return df
    
    @staticmethod
    def get_regime_from_indicators(df: pd.DataFrame) -> str:
        """
        Quick regime detection from indicators.
        
        Returns: 'TRENDING_UP', 'TRENDING_DOWN', 'RANGING', 'VOLATILE', 'QUIET'
        """
        if df.empty:
            return 'UNKNOWN'
        
        latest = df.iloc[-1]
        
        vol_regime = latest.get('vol_regime', 'NORMAL')
        adx = latest.get('adx', 25)
        trend_direction = latest.get('trend_direction', 0)
        
        if vol_regime == 'QUIET':
            return 'QUIET'
        
        if vol_regime == 'VOLATILE':
            return 'VOLATILE'
        
        if adx > 25:
            if trend_direction > 0:
                return 'TRENDING_UP'
            elif trend_direction < 0:
                return 'TRENDING_DOWN'
        
        return 'RANGING'
    
    @staticmethod
    def get_trade_signals(df: pd.DataFrame) -> dict:
        """
        Generate trade signals from indicators.
        
        Returns dict with:
        - direction: 'LONG', 'SHORT', 'NEUTRAL'
        - strength: 0.0 to 1.0
        - reason: explanation
        """
        if df.empty:
            return {'direction': 'NEUTRAL', 'strength': 0, 'reason': 'No data'}
        
        latest = df.iloc[-1]
        
        trend_score = latest.get('trend_score', 0)
        mean_revert_score = latest.get('mean_revert_score', 0)
        trade_quality = latest.get('trade_quality', 50)
        rsi = latest.get('rsi', 50)
        
        # Low trade quality = no signal
        if trade_quality < 40:
            return {
                'direction': 'NEUTRAL',
                'strength': 0,
                'reason': f'Poor trade quality: {trade_quality:.0f}'
            }
        
        # Strong trend = trade with trend
        if abs(trend_score) > 50:
            direction = 'LONG' if trend_score > 0 else 'SHORT'
            strength = min(1.0, abs(trend_score) / 100)
            return {
                'direction': direction,
                'strength': strength,
                'reason': f'Strong trend: score={trend_score:.0f}'
            }
        
        # Mean reversion setup
        if mean_revert_score > 30:
            direction = 'LONG' if rsi < 40 else 'SHORT' if rsi > 60 else 'NEUTRAL'
            strength = min(1.0, mean_revert_score / 60)
            return {
                'direction': direction,
                'strength': strength,
                'reason': f'Mean reversion: RSI={rsi:.0f}, score={mean_revert_score:.0f}'
            }
        
        return {
            'direction': 'NEUTRAL',
            'strength': 0,
            'reason': 'No clear setup'
        }

