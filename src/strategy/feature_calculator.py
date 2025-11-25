"""
Feature Calculator Module

Standalone feature calculation that can be applied independently to any dataset.
This prevents data leakage by ensuring features are calculated ONLY on the data
provided, without any future information contamination.

Key Principles:
- Features are calculated ONLY on the provided data
- No bfill() - only forward fill is used
- Warmup period is dropped to prevent NaN contamination
- Can be applied independently to train, validation, and holdout sets
"""

import numpy as np
import pandas as pd
import talib
from typing import List, Tuple
import logging

logger = logging.getLogger(__name__)


class FeatureCalculator:
    """
    Calculate all 68 features for LightGBM training.

    This class is designed to be applied independently to separate datasets
    (train, validation, holdout) to prevent any data leakage.

    Features include:
    - 38 original technical indicators
    - 30 enhanced features (multi-timeframe, volatility regime, price position, etc.)
    """

    # Warmup period needed for indicators (300 bars for ROC_240)
    WARMUP_BARS = 300

    @staticmethod
    def get_feature_columns() -> List[str]:
        """Return list of all 68 feature column names (38 original + 30 enhanced)."""
        return [
            # Core (10)
            'rsi', 'macd', 'macd_signal', 'bb_upper', 'bb_middle', 'bb_lower',
            'atr', 'volume_ratio', 'momentum', 'adx',
            # Price patterns (7)
            'breakout_up', 'breakout_down', 'price_position',
            'body_size', 'upper_wick', 'lower_wick', 'bullish_candle',
            # Volatility (4)
            'atr_pct', 'volatility_regime', 'bb_width', 'bb_squeeze',
            # Trend/Momentum (5)
            'trend_alignment', 'roc_5', 'roc_10', 'roc_20', 'momentum_alignment',
            # Volume (3)
            'volume_trend', 'volume_spike', 'obv_momentum',
            # Oscillators (9)
            'rsi_momentum', 'macd_hist', 'macd_hist_momentum',
            'stoch_k', 'stoch_d', 'stoch_cross', 'willr', 'cci', 'mfi',
            # ===== ENHANCED FEATURES (30) =====
            # Multi-timeframe momentum (6)
            'roc_60', 'roc_240', 'mtf_momentum_align',
            'return_5', 'return_20', 'return_60',
            # Volatility regime (3)
            'atr_percentile', 'vol_expanding', 'vol_contracting',
            # Price position (4)
            'price_percentile_50', 'price_percentile_100',
            'dist_from_high_50', 'dist_from_low_50',
            # Acceleration (4)
            'momentum_accel', 'rsi_accel', 'roc_accel', 'volume_accel',
            # Time features (4)
            'hour', 'day_of_week', 'is_us_session', 'is_asia_session',
            # Trend strength (2)
            'strong_trend', 'weak_trend',
            # Mean reversion (4)
            'rsi_oversold', 'rsi_overbought', 'bb_oversold', 'bb_overbought',
            # Divergence (2)
            'bullish_divergence', 'bearish_divergence'
        ]

    @classmethod
    def calculate_features(cls, df: pd.DataFrame, drop_warmup: bool = True) -> pd.DataFrame:
        """
        Calculate all 38 features on the provided dataset.

        Args:
            df: DataFrame with OHLCV data (open, high, low, close, volume)
            drop_warmup: Whether to drop the first WARMUP_BARS rows (default True)

        Returns:
            DataFrame with all features calculated

        IMPORTANT: This method only uses data from the provided DataFrame.
        No future information is used (no bfill, no lookahead).
        """
        df = df.copy()
        df.columns = [c.lower() for c in df.columns]

        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        open_price = df['open'].values
        volume = df['volume'].values if 'volume' in df.columns else np.ones(len(close))

        # ============================================
        # CORE FEATURES (10)
        # ============================================
        df['rsi'] = talib.RSI(close, timeperiod=14)

        macd, macd_signal, macd_hist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
        df['macd'] = macd
        df['macd_signal'] = macd_signal
        df['macd_hist'] = macd_hist

        bb_upper, bb_middle, bb_lower = talib.BBANDS(close, timeperiod=20)
        df['bb_upper'] = bb_upper
        df['bb_middle'] = bb_middle
        df['bb_lower'] = bb_lower

        df['atr'] = talib.ATR(high, low, close, timeperiod=14)

        vol_ma = pd.Series(volume).rolling(window=20).mean()
        df['volume_ratio'] = volume / (vol_ma.values + 1e-10)

        df['momentum'] = talib.MOM(close, timeperiod=10)
        df['adx'] = talib.ADX(high, low, close, timeperiod=14)

        # ============================================
        # PRICE PATTERNS (7)
        # ============================================
        df['high_20'] = pd.Series(high).rolling(20).max()
        df['low_20'] = pd.Series(low).rolling(20).min()

        # Breakout signals
        df['breakout_up'] = (close > df['high_20'].shift(1)).astype(float)
        df['breakout_down'] = (close < df['low_20'].shift(1)).astype(float)

        # Price position in range
        range_20 = df['high_20'] - df['low_20']
        df['price_position'] = (close - df['low_20']) / (range_20 + 1e-10)

        # Candle patterns
        df['body_size'] = abs(close - open_price) / (df['atr'] + 1e-10)
        df['upper_wick'] = (high - np.maximum(close, open_price)) / (df['atr'] + 1e-10)
        df['lower_wick'] = (np.minimum(close, open_price) - low) / (df['atr'] + 1e-10)
        df['bullish_candle'] = (close > open_price).astype(float)

        # ============================================
        # VOLATILITY (4)
        # ============================================
        df['atr_pct'] = df['atr'] / close * 100
        df['atr_sma'] = df['atr_pct'].rolling(20).mean()
        df['volatility_regime'] = df['atr_pct'] / (df['atr_sma'] + 1e-10)

        df['bb_width'] = (bb_upper - bb_lower) / (bb_middle + 1e-10)
        df['bb_squeeze'] = (df['bb_width'] < df['bb_width'].rolling(50).quantile(0.2)).astype(float)

        # ============================================
        # TREND/MOMENTUM (5)
        # ============================================
        df['ema_8'] = talib.EMA(close, timeperiod=8)
        df['ema_21'] = talib.EMA(close, timeperiod=21)
        df['ema_50'] = talib.EMA(close, timeperiod=50)

        df['trend_alignment'] = (
            (df['ema_8'] > df['ema_21']).astype(float) +
            (df['ema_21'] > df['ema_50']).astype(float) +
            (close > df['ema_8']).astype(float)
        )

        df['roc_5'] = talib.ROC(close, timeperiod=5)
        df['roc_10'] = talib.ROC(close, timeperiod=10)
        df['roc_20'] = talib.ROC(close, timeperiod=20)

        df['momentum_alignment'] = (
            ((df['roc_5'] > 0) & (df['roc_10'] > 0) & (df['roc_20'] > 0)).astype(float) -
            ((df['roc_5'] < 0) & (df['roc_10'] < 0) & (df['roc_20'] < 0)).astype(float)
        )

        # ============================================
        # VOLUME (3)
        # ============================================
        df['volume_sma_10'] = pd.Series(volume).rolling(10).mean()
        df['volume_sma_50'] = pd.Series(volume).rolling(50).mean()
        df['volume_trend'] = df['volume_sma_10'] / (df['volume_sma_50'] + 1e-10)
        df['volume_spike'] = (df['volume_ratio'] > 2.0).astype(float)

        obv = talib.OBV(close, volume.astype(float))
        df['obv_momentum'] = pd.Series(obv).diff(10) / (pd.Series(obv).rolling(20).std() + 1e-10)

        # ============================================
        # OSCILLATORS (9)
        # ============================================
        df['macd_hist_momentum'] = pd.Series(macd_hist).diff(3)
        df['rsi_momentum'] = df['rsi'].diff(5)

        slowk, slowd = talib.STOCH(high, low, close, fastk_period=14, slowk_period=3, slowd_period=3)
        df['stoch_k'] = slowk
        df['stoch_d'] = slowd
        df['stoch_cross'] = (slowk > slowd).astype(float)

        df['willr'] = talib.WILLR(high, low, close, timeperiod=14)
        df['cci'] = talib.CCI(high, low, close, timeperiod=20)

        try:
            df['mfi'] = talib.MFI(high, low, close, volume.astype(float), timeperiod=14)
        except:
            df['mfi'] = 50.0

        # ============================================
        # ENHANCED FEATURES - Multi-timeframe & Advanced
        # ============================================

        # Multi-timeframe ROC (larger scales for trend detection)
        df['roc_60'] = talib.ROC(close, timeperiod=60)    # 1-hour momentum (60 x 1-min)
        df['roc_240'] = talib.ROC(close, timeperiod=240)  # 4-hour momentum (240 x 1-min)

        # Multi-timeframe momentum alignment (including larger timeframes)
        df['mtf_momentum_align'] = (
            ((df['roc_5'] > 0).astype(float) * 0.1) +
            ((df['roc_20'] > 0).astype(float) * 0.2) +
            ((df['roc_60'] > 0).astype(float) * 0.3) +
            ((df['roc_240'] > 0).astype(float) * 0.4)
        )

        # Returns at different scales (raw price changes)
        df['return_5'] = pd.Series(close).pct_change(5) * 100
        df['return_20'] = pd.Series(close).pct_change(20) * 100
        df['return_60'] = pd.Series(close).pct_change(60) * 100

        # ATR percentile (is current volatility high or low?)
        atr_series = df['atr']
        df['atr_percentile'] = atr_series.rolling(100).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else 0.5, raw=False
        )

        # Volatility expansion/contraction
        df['vol_expanding'] = (df['atr'] > df['atr'].rolling(20).mean()).astype(float)
        df['vol_contracting'] = (df['atr'] < df['atr'].rolling(20).mean() * 0.8).astype(float)

        # Price percentile (where is price relative to recent history?)
        df['price_percentile_50'] = pd.Series(close).rolling(50).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else 0.5, raw=False
        )
        df['price_percentile_100'] = pd.Series(close).rolling(100).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else 0.5, raw=False
        )

        # Distance from recent high/low (% away)
        high_50 = pd.Series(high).rolling(50).max()
        low_50 = pd.Series(low).rolling(50).min()
        df['dist_from_high_50'] = (high_50 - close) / (close + 1e-10) * 100
        df['dist_from_low_50'] = (close - low_50) / (close + 1e-10) * 100

        # Momentum acceleration (second derivative - is momentum speeding up?)
        df['momentum_accel'] = df['momentum'].diff(5)
        df['rsi_accel'] = df['rsi'].diff(5)
        df['roc_accel'] = df['roc_10'].diff(5)

        # Volume acceleration
        df['volume_accel'] = df['volume_ratio'].diff(5)

        # Time-based features (extract from index if datetime)
        try:
            if hasattr(df.index, 'hour'):
                df['hour'] = df.index.hour
                df['day_of_week'] = df.index.dayofweek
                # Session indicators
                df['is_us_session'] = ((df['hour'] >= 13) & (df['hour'] <= 21)).astype(float)  # ~9am-5pm ET in UTC
                df['is_asia_session'] = ((df['hour'] >= 0) & (df['hour'] <= 8)).astype(float)  # Asia morning in UTC
            else:
                df['hour'] = 12  # Default midday
                df['day_of_week'] = 2  # Default Wednesday
                df['is_us_session'] = 0.5
                df['is_asia_session'] = 0.5
        except:
            df['hour'] = 12
            df['day_of_week'] = 2
            df['is_us_session'] = 0.5
            df['is_asia_session'] = 0.5

        # Trend strength (ADX-based)
        df['strong_trend'] = (df['adx'] > 25).astype(float)
        df['weak_trend'] = (df['adx'] < 15).astype(float)

        # Mean reversion signals
        df['rsi_oversold'] = (df['rsi'] < 30).astype(float)
        df['rsi_overbought'] = (df['rsi'] > 70).astype(float)
        df['bb_oversold'] = (close < df['bb_lower']).astype(float)
        df['bb_overbought'] = (close > df['bb_upper']).astype(float)

        # Divergence detection (price vs RSI)
        price_higher = pd.Series(close).diff(20) > 0
        rsi_higher = df['rsi'].diff(20) > 0
        df['bullish_divergence'] = ((~price_higher) & rsi_higher).astype(float)
        df['bearish_divergence'] = (price_higher & (~rsi_higher)).astype(float)

        # ============================================
        # FILL NaN - FORWARD FILL ONLY (no bfill!)
        # ============================================
        # Only use forward fill to avoid future data leakage
        df = df.ffill()

        # Fill any remaining NaN with 0 (for columns that have no data at start)
        df = df.fillna(0)

        # Drop warmup period to ensure clean data
        if drop_warmup and len(df) > cls.WARMUP_BARS:
            df = df.iloc[cls.WARMUP_BARS:]

        return df

    @classmethod
    def prepare_training_data(cls, df: pd.DataFrame, horizon: int,
                              move_threshold: float = 0.005) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features and target for training.

        Args:
            df: Raw OHLCV data
            horizon: Prediction horizon in bars (5, 15, or 30)
            move_threshold: Threshold for significant move (default 0.5%)

        Returns:
            Tuple of (features DataFrame, target Series)
        """
        # Calculate features
        df_features = cls.calculate_features(df, drop_warmup=True)

        # Get feature columns only
        feature_cols = cls.get_feature_columns()
        features = df_features[[c for c in feature_cols if c in df_features.columns]]

        # Create target: 1 if price moves up > threshold, else 0
        future_close = df_features['close'].shift(-horizon)
        current_close = df_features['close']
        pct_change = (future_close - current_close) / current_close

        # Align indices and remove last `horizon` rows (no target available)
        valid_idx = features.index.intersection(future_close.dropna().index)
        X = features.loc[valid_idx].iloc[:-horizon]
        y = (pct_change.loc[valid_idx] > move_threshold).astype(int).iloc[:-horizon]

        return X, y

    @classmethod
    def add_buffer_between_splits(cls, train_end_idx: int, test_start_idx: int,
                                   buffer_bars: int = 50) -> Tuple[int, int]:
        """
        Add a buffer between train and test splits to prevent boundary contamination.

        Rolling window features (like EMA50) at the boundary of train/test can
        use data from the other set. This buffer ensures clean separation.

        Args:
            train_end_idx: Index where training data ends
            test_start_idx: Index where test data starts
            buffer_bars: Number of bars to skip between train and test

        Returns:
            Tuple of (new_train_end_idx, new_test_start_idx)
        """
        # Ensure at least buffer_bars gap
        if test_start_idx - train_end_idx < buffer_bars:
            # Add buffer by moving test start forward
            new_test_start = train_end_idx + buffer_bars
            return train_end_idx, new_test_start

        return train_end_idx, test_start_idx
