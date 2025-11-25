"""
Adaptive AI Engine for Multi-Asset Trading

This module replaces simple RSI-based logic with intelligent AI-driven decisions.
The AI evaluates EVERY bar for opportunities and only trades when confidence is high.

Key Features:
- Multi-timeframe analysis (1-min aggregated to 5-min and 15-min views)
- LightGBM model for price movement prediction
- Confidence scoring based on AI prediction + indicator alignment
- Opportunity detection (RSI extremes, volume spikes, breakouts)
- Full reasoning for every decision
"""

import logging
import os
import numpy as np
import pandas as pd
from collections import deque
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any

try:
    import lightgbm as lgb
except ImportError:
    lgb = None

try:
    import talib
except ImportError:
    talib = None

logger = logging.getLogger("AdaptiveAI")


class AdaptiveAI:
    """
    Adaptive AI Engine for intelligent trading decisions.
    
    This class:
    1. Maintains rolling windows of price data per symbol
    2. Calculates technical indicators using TA-Lib
    3. Runs XGBoost predictions for each bar
    4. Aggregates data into multiple timeframes
    5. Scores confidence based on indicator alignment
    6. Returns clear reasoning for every signal
    """
    
    # Feature importance weights (learned from backtesting)
    FEATURE_WEIGHTS = {
        'rsi': 0.35,
        'volume': 0.25,
        'macd': 0.18,
        'bollinger': 0.12,
        'price_action': 0.10
    }
    
    def __init__(self, config, model_path: str = None):
        """
        Initialize the Adaptive AI engine.

        Args:
            config: Configuration object with trading parameters
            model_path: Path to LightGBM model file
        """
        self.config = config
        self.model_path = model_path or getattr(config, 'MODEL_PATH', 'models/lightgbm_model.txt')

        # Load LightGBM model(s) - supports multi-horizon
        self.model = self._load_model()
        self.multi_horizon_models = self._load_multi_horizon_models()
        
        # Rolling windows of bars per symbol (1-min data)
        self.bars: Dict[str, deque] = {}
        
        # Aggregated timeframe data
        self.tf_5min: Dict[str, deque] = {}
        self.tf_15min: Dict[str, deque] = {}
        
        # Indicator cache per symbol
        self.indicators: Dict[str, Dict] = {}
        
        # Multi-timeframe signals
        self.tf_signals: Dict[str, Dict[str, str]] = {}
        
        # Last AI state per symbol (for dashboard)
        self.ai_state: Dict[str, Dict] = {}
        
        # Trade history for Kelly calculation
        self.trade_history: List[Dict] = []
        self.win_count = 0
        self.loss_count = 0
        
        # Thresholds (configurable)
        self.buy_threshold = getattr(config, 'BUY_THRESHOLD', 0.65)
        self.sell_threshold = getattr(config, 'SELL_THRESHOLD', 0.35)
        self.min_confidence = getattr(config, 'MIN_CONFIDENCE', 70)
        
        logger.info(f"AdaptiveAI initialized. Model: {'Loaded' if self.model else 'Not available'}")
        
    def _load_model(self) -> Optional[Any]:
        """Load LightGBM model from disk."""
        if lgb is None:
            logger.warning("LightGBM not installed")
            return None

        if not os.path.exists(self.model_path):
            logger.warning(f"Model not found at {self.model_path}")
            return None

        try:
            model = lgb.Booster(model_file=self.model_path)
            logger.info(f"Loaded LightGBM model from {self.model_path}")
            return model
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return None

    def _load_multi_horizon_models(self) -> Dict[int, Any]:
        """Load multi-horizon models if available (5, 15, 30 min)."""
        # This loads default models - symbol-specific models loaded on demand
        models = {}
        horizons = [5, 15, 30]

        if lgb is None:
            return models

        for horizon in horizons:
            model_path = f"models/lightgbm_{horizon}min.txt"
            if os.path.exists(model_path):
                try:
                    model = lgb.Booster(model_file=model_path)
                    models[horizon] = model
                    logger.info(f"Loaded {horizon}-min horizon model (default)")
                except Exception as e:
                    logger.warning(f"Failed to load {horizon}-min model: {e}")

        if models:
            logger.info(f"Multi-horizon models loaded: {list(models.keys())} minutes")

        return models

    def _load_symbol_models(self, symbol: str) -> Dict[int, Any]:
        """Load symbol-specific models if available."""
        models = {}
        horizons = [5, 15, 30]

        if lgb is None:
            return models

        # Convert symbol to filename format (e.g., "BTC/USD" -> "BTC_USD")
        symbol_safe = symbol.replace("/", "_")

        for horizon in horizons:
            model_path = f"models/{symbol_safe}_{horizon}min.txt"
            if os.path.exists(model_path):
                try:
                    model = lgb.Booster(model_file=model_path)
                    models[horizon] = model
                    logger.info(f"Loaded {symbol} {horizon}-min model")
                except Exception as e:
                    logger.warning(f"Failed to load {symbol} {horizon}-min model: {e}")

        return models

    def get_models_for_symbol(self, symbol: str) -> Dict[int, Any]:
        """Get the best available models for a symbol (symbol-specific or default)."""
        # Try symbol-specific models first
        if not hasattr(self, '_symbol_models_cache'):
            self._symbol_models_cache = {}

        if symbol not in self._symbol_models_cache:
            symbol_models = self._load_symbol_models(symbol)
            if symbol_models:
                self._symbol_models_cache[symbol] = symbol_models
                logger.info(f"Using symbol-specific models for {symbol}")
            else:
                self._symbol_models_cache[symbol] = self.multi_horizon_models
                logger.info(f"Using default models for {symbol}")

        return self._symbol_models_cache[symbol]
    
    def add_bar(self, symbol: str, bar: Any) -> None:
        """
        Add a new bar for a symbol and update all indicators.
        
        Args:
            symbol: Trading symbol (e.g., "BTC/USD")
            bar: Bar object with OHLCV data
        """
        # Initialize storage if needed
        if symbol not in self.bars:
            self.bars[symbol] = deque(maxlen=200)  # Keep more data for multi-TF
            self.tf_5min[symbol] = deque(maxlen=50)
            self.tf_15min[symbol] = deque(maxlen=30)
            self.indicators[symbol] = {}
            self.tf_signals[symbol] = {}
            self.ai_state[symbol] = {}
        
        # Convert bar to dict
        bar_data = {
            'timestamp': bar.timestamp if hasattr(bar, 'timestamp') else datetime.now(),
            'open': float(bar.open) if hasattr(bar, 'open') else float(bar.get('open', 0)),
            'high': float(bar.high) if hasattr(bar, 'high') else float(bar.get('high', 0)),
            'low': float(bar.low) if hasattr(bar, 'low') else float(bar.get('low', 0)),
            'close': float(bar.close) if hasattr(bar, 'close') else float(bar.get('close', 0)),
            'volume': float(bar.volume) if hasattr(bar, 'volume') else float(bar.get('volume', 0))
        }
        
        self.bars[symbol].append(bar_data)
        
        # Aggregate into higher timeframes
        self._aggregate_timeframes(symbol)
        
        # Calculate indicators for all timeframes
        self._calculate_all_indicators(symbol)
        
    def _aggregate_timeframes(self, symbol: str) -> None:
        """
        Aggregate 1-min bars into 5-min and 15-min bars using ROLLING windows.

        This creates virtual higher-timeframe bars on EVERY update, not just on
        exact multiples. This ensures multi-timeframe signals are always current.
        """
        bars = list(self.bars[symbol])

        # Create rolling 5-min bars (each bar represents last 5 1-min bars)
        if len(bars) >= 5:
            self.tf_5min[symbol].clear()
            # Create multiple 5-min bars from history
            for i in range(5, len(bars) + 1, 5):
                chunk = bars[i-5:i]
                agg_bar = {
                    'timestamp': chunk[-1]['timestamp'],
                    'open': chunk[0]['open'],
                    'high': max(b['high'] for b in chunk),
                    'low': min(b['low'] for b in chunk),
                    'close': chunk[-1]['close'],
                    'volume': sum(b['volume'] for b in chunk)
                }
                self.tf_5min[symbol].append(agg_bar)

            # Add current incomplete 5-min bar (most recent data)
            remaining = len(bars) % 5
            if remaining > 0:
                last_chunk = bars[-remaining:]
                current_bar = {
                    'timestamp': last_chunk[-1]['timestamp'],
                    'open': last_chunk[0]['open'],
                    'high': max(b['high'] for b in last_chunk),
                    'low': min(b['low'] for b in last_chunk),
                    'close': last_chunk[-1]['close'],
                    'volume': sum(b['volume'] for b in last_chunk)
                }
                self.tf_5min[symbol].append(current_bar)

        # Create rolling 15-min bars
        if len(bars) >= 15:
            self.tf_15min[symbol].clear()
            for i in range(15, len(bars) + 1, 15):
                chunk = bars[i-15:i]
                agg_bar = {
                    'timestamp': chunk[-1]['timestamp'],
                    'open': chunk[0]['open'],
                    'high': max(b['high'] for b in chunk),
                    'low': min(b['low'] for b in chunk),
                    'close': chunk[-1]['close'],
                    'volume': sum(b['volume'] for b in chunk)
                }
                self.tf_15min[symbol].append(agg_bar)

            # Add current incomplete 15-min bar
            remaining = len(bars) % 15
            if remaining > 0:
                last_chunk = bars[-remaining:]
                current_bar = {
                    'timestamp': last_chunk[-1]['timestamp'],
                    'open': last_chunk[0]['open'],
                    'high': max(b['high'] for b in last_chunk),
                    'low': min(b['low'] for b in last_chunk),
                    'close': last_chunk[-1]['close'],
                    'volume': sum(b['volume'] for b in last_chunk)
                }
                self.tf_15min[symbol].append(current_bar)
    
    def _calculate_all_indicators(self, symbol: str) -> None:
        """Calculate indicators for all timeframes."""
        bars_1min = list(self.bars[symbol])
        bars_5min = list(self.tf_5min[symbol])
        bars_15min = list(self.tf_15min[symbol])
        
        # Calculate for 1-min (primary)
        ind_1min = self._calculate_indicators(bars_1min)
        
        # Calculate for 5-min
        ind_5min = self._calculate_indicators(bars_5min) if len(bars_5min) >= 20 else {}
        
        # Calculate for 15-min
        ind_15min = self._calculate_indicators(bars_15min) if len(bars_15min) >= 20 else {}
        
        # Store all indicators
        self.indicators[symbol] = {
            '1min': ind_1min,
            '5min': ind_5min,
            '15min': ind_15min
        }
        
        # Generate timeframe signals
        self.tf_signals[symbol] = {
            '1min': self._get_timeframe_signal(ind_1min),
            '5min': self._get_timeframe_signal(ind_5min),
            '15min': self._get_timeframe_signal(ind_15min)
        }
    
    def _calculate_indicators(self, bars: List[Dict]) -> Dict:
        """Calculate technical indicators from bar data - ALL 38 features for XGBoost model."""
        if len(bars) < 50:  # Need more bars for full indicator calculation
            return {}

        closes = np.array([b['close'] for b in bars])
        highs = np.array([b['high'] for b in bars])
        lows = np.array([b['low'] for b in bars])
        opens = np.array([b['open'] for b in bars])
        volumes = np.array([b['volume'] for b in bars], dtype=float)

        indicators = {}

        try:
            if talib is not None:
                # ============================================
                # CORE FEATURES (10)
                # ============================================
                # RSI
                rsi = talib.RSI(closes, timeperiod=14)
                indicators['rsi'] = float(rsi[-1]) if not np.isnan(rsi[-1]) else 50.0

                # MACD
                macd, signal, hist = talib.MACD(closes, fastperiod=12, slowperiod=26, signalperiod=9)
                indicators['macd'] = float(macd[-1]) if not np.isnan(macd[-1]) else 0.0
                indicators['macd_signal'] = float(signal[-1]) if not np.isnan(signal[-1]) else 0.0
                indicators['macd_hist'] = float(hist[-1]) if not np.isnan(hist[-1]) else 0.0

                # Bollinger Bands
                upper, middle, lower = talib.BBANDS(closes, timeperiod=20)
                indicators['bb_upper'] = float(upper[-1]) if not np.isnan(upper[-1]) else closes[-1]
                indicators['bb_middle'] = float(middle[-1]) if not np.isnan(middle[-1]) else closes[-1]
                indicators['bb_lower'] = float(lower[-1]) if not np.isnan(lower[-1]) else closes[-1]

                # BB position (0 = at lower, 1 = at upper)
                bb_range = indicators['bb_upper'] - indicators['bb_lower']
                if bb_range > 0:
                    indicators['bb_position'] = (closes[-1] - indicators['bb_lower']) / bb_range
                else:
                    indicators['bb_position'] = 0.5

                # ATR
                atr = talib.ATR(highs, lows, closes, timeperiod=14)
                indicators['atr'] = float(atr[-1]) if not np.isnan(atr[-1]) else 0.0

                # ADX (trend strength)
                adx = talib.ADX(highs, lows, closes, timeperiod=14)
                indicators['adx'] = float(adx[-1]) if not np.isnan(adx[-1]) else 25.0

                # Momentum (talib)
                mom = talib.MOM(closes, timeperiod=10)
                indicators['momentum'] = float(mom[-1]) if not np.isnan(mom[-1]) else 0.0

                # Volume ratio
                avg_volume = np.mean(volumes[-20:])
                indicators['volume_ratio'] = float(volumes[-1] / avg_volume) if avg_volume > 0 else 1.0

                # ============================================
                # PRICE PATTERNS (7)
                # ============================================
                # Rolling highs/lows for breakout detection
                high_20 = pd.Series(highs).rolling(20).max().values
                low_20 = pd.Series(lows).rolling(20).min().values

                # Breakout signals
                indicators['breakout_up'] = 1.0 if closes[-1] > high_20[-2] else 0.0
                indicators['breakout_down'] = 1.0 if closes[-1] < low_20[-2] else 0.0

                # Price position in range
                range_20 = high_20[-1] - low_20[-1]
                indicators['price_position'] = (closes[-1] - low_20[-1]) / (range_20 + 1e-10)

                # Candle patterns
                atr_val = indicators['atr'] if indicators['atr'] > 0 else 1e-10
                indicators['body_size'] = abs(closes[-1] - opens[-1]) / atr_val
                indicators['upper_wick'] = (highs[-1] - max(closes[-1], opens[-1])) / atr_val
                indicators['lower_wick'] = (min(closes[-1], opens[-1]) - lows[-1]) / atr_val
                indicators['bullish_candle'] = 1.0 if closes[-1] > opens[-1] else 0.0

                # ============================================
                # VOLATILITY (4)
                # ============================================
                indicators['atr_pct'] = indicators['atr'] / closes[-1] * 100 if closes[-1] > 0 else 0.0

                # ATR SMA for volatility regime
                atr_series = talib.ATR(highs, lows, closes, timeperiod=14)
                atr_pct_series = atr_series / closes * 100
                atr_sma = np.nanmean(atr_pct_series[-20:])
                indicators['volatility_regime'] = indicators['atr_pct'] / (atr_sma + 1e-10)

                # BB width and squeeze
                indicators['bb_width'] = (indicators['bb_upper'] - indicators['bb_lower']) / (indicators['bb_middle'] + 1e-10)
                bb_width_series = (upper - lower) / (middle + 1e-10)
                bb_width_20 = bb_width_series[-50:] if len(bb_width_series) >= 50 else bb_width_series
                bb_squeeze_threshold = np.nanquantile(bb_width_20, 0.2)
                indicators['bb_squeeze'] = 1.0 if indicators['bb_width'] < bb_squeeze_threshold else 0.0

                # ============================================
                # TREND/MOMENTUM (5)
                # ============================================
                # EMAs for trend alignment
                ema_8 = talib.EMA(closes, timeperiod=8)
                ema_21 = talib.EMA(closes, timeperiod=21)
                ema_50 = talib.EMA(closes, timeperiod=50)

                # Trend alignment score (0-3)
                trend_score = 0.0
                if ema_8[-1] > ema_21[-1]:
                    trend_score += 1.0
                if ema_21[-1] > ema_50[-1]:
                    trend_score += 1.0
                if closes[-1] > ema_8[-1]:
                    trend_score += 1.0
                indicators['trend_alignment'] = trend_score

                # Multi-period ROC
                roc_5 = talib.ROC(closes, timeperiod=5)
                roc_10 = talib.ROC(closes, timeperiod=10)
                roc_20 = talib.ROC(closes, timeperiod=20)
                indicators['roc_5'] = float(roc_5[-1]) if not np.isnan(roc_5[-1]) else 0.0
                indicators['roc_10'] = float(roc_10[-1]) if not np.isnan(roc_10[-1]) else 0.0
                indicators['roc_20'] = float(roc_20[-1]) if not np.isnan(roc_20[-1]) else 0.0

                # Momentum alignment
                if indicators['roc_5'] > 0 and indicators['roc_10'] > 0 and indicators['roc_20'] > 0:
                    indicators['momentum_alignment'] = 1.0
                elif indicators['roc_5'] < 0 and indicators['roc_10'] < 0 and indicators['roc_20'] < 0:
                    indicators['momentum_alignment'] = -1.0
                else:
                    indicators['momentum_alignment'] = 0.0

                # ============================================
                # VOLUME (3)
                # ============================================
                vol_sma_10 = np.mean(volumes[-10:])
                vol_sma_50 = np.mean(volumes[-50:]) if len(volumes) >= 50 else vol_sma_10
                indicators['volume_trend'] = vol_sma_10 / (vol_sma_50 + 1e-10)
                indicators['volume_spike'] = 1.0 if indicators['volume_ratio'] > 2.0 else 0.0

                # OBV momentum
                obv = talib.OBV(closes, volumes)
                obv_diff = obv[-1] - obv[-11] if len(obv) >= 11 else 0
                obv_std = np.std(obv[-20:]) if len(obv) >= 20 else 1
                indicators['obv_momentum'] = obv_diff / (obv_std + 1e-10)

                # ============================================
                # OSCILLATORS (9)
                # ============================================
                # RSI momentum
                rsi_prev = float(rsi[-6]) if len(rsi) >= 6 and not np.isnan(rsi[-6]) else indicators['rsi']
                indicators['rsi_momentum'] = indicators['rsi'] - rsi_prev

                # MACD hist momentum
                hist_prev = float(hist[-4]) if len(hist) >= 4 and not np.isnan(hist[-4]) else indicators['macd_hist']
                indicators['macd_hist_momentum'] = indicators['macd_hist'] - hist_prev

                # Stochastic
                slowk, slowd = talib.STOCH(highs, lows, closes, fastk_period=14, slowk_period=3, slowd_period=3)
                indicators['stoch_k'] = float(slowk[-1]) if not np.isnan(slowk[-1]) else 50.0
                indicators['stoch_d'] = float(slowd[-1]) if not np.isnan(slowd[-1]) else 50.0
                indicators['stoch_cross'] = 1.0 if indicators['stoch_k'] > indicators['stoch_d'] else 0.0

                # Williams %R
                willr = talib.WILLR(highs, lows, closes, timeperiod=14)
                indicators['willr'] = float(willr[-1]) if not np.isnan(willr[-1]) else -50.0

                # CCI
                cci = talib.CCI(highs, lows, closes, timeperiod=20)
                indicators['cci'] = float(cci[-1]) if not np.isnan(cci[-1]) else 0.0

                # MFI
                try:
                    mfi = talib.MFI(highs, lows, closes, volumes, timeperiod=14)
                    indicators['mfi'] = float(mfi[-1]) if not np.isnan(mfi[-1]) else 50.0
                except:
                    indicators['mfi'] = 50.0

            else:
                # Fallback: manual calculations (basic features only)
                indicators = self._calculate_indicators_manual(closes, highs, lows, volumes)

        except Exception as e:
            logger.warning(f"Indicator calculation error: {e}")
            indicators = self._calculate_indicators_manual(closes, highs, lows, volumes)

        # Additional display fields (not for model)
        indicators['volume'] = float(volumes[-1])
        avg_volume = np.mean(volumes[-20:])
        indicators['avg_volume'] = float(avg_volume)
        indicators['price'] = float(closes[-1])
        indicators['price_change'] = float((closes[-1] - closes[-2]) / closes[-2] * 100) if len(closes) > 1 else 0.0
        indicators['sma_10'] = float(np.mean(closes[-10:])) if len(closes) >= 10 else closes[-1]
        indicators['sma_20'] = float(np.mean(closes[-20:])) if len(closes) >= 20 else closes[-1]
        indicators['trend'] = 'UP' if indicators['sma_10'] > indicators['sma_20'] else 'DOWN'

        return indicators
    
    def _calculate_indicators_manual(self, closes: np.ndarray, highs: np.ndarray, 
                                      lows: np.ndarray, volumes: np.ndarray) -> Dict:
        """Manual indicator calculations (fallback when TA-Lib unavailable)."""
        indicators = {}
        
        # RSI (manual)
        deltas = np.diff(closes)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[-14:]) if len(gains) >= 14 else 0.001
        avg_loss = np.mean(losses[-14:]) if len(losses) >= 14 else 0.001
        
        rs = avg_gain / avg_loss if avg_loss > 0 else 1
        indicators['rsi'] = 100 - (100 / (1 + rs))
        
        # Simple MACD approximation
        ema_12 = np.mean(closes[-12:]) if len(closes) >= 12 else closes[-1]
        ema_26 = np.mean(closes[-26:]) if len(closes) >= 26 else closes[-1]
        indicators['macd'] = ema_12 - ema_26
        indicators['macd_signal'] = indicators['macd'] * 0.9  # Simplified
        indicators['macd_hist'] = indicators['macd'] - indicators['macd_signal']
        
        # Bollinger Bands (manual)
        sma_20 = np.mean(closes[-20:])
        std_20 = np.std(closes[-20:])
        indicators['bb_upper'] = sma_20 + 2 * std_20
        indicators['bb_middle'] = sma_20
        indicators['bb_lower'] = sma_20 - 2 * std_20
        
        bb_range = indicators['bb_upper'] - indicators['bb_lower']
        indicators['bb_position'] = (closes[-1] - indicators['bb_lower']) / bb_range if bb_range > 0 else 0.5
        
        # ATR (manual)
        tr = np.maximum(highs[-14:] - lows[-14:], 
                        np.abs(highs[-14:] - np.roll(closes[-14:], 1)[1:]))
        indicators['atr'] = np.mean(tr) if len(tr) > 0 else 0
        indicators['atr_pct'] = indicators['atr'] / closes[-1] * 100 if closes[-1] > 0 else 0
        
        # ADX approximation
        indicators['adx'] = 25  # Default neutral
        
        return indicators
    
    def _get_timeframe_signal(self, indicators: Dict) -> str:
        """Get signal direction for a timeframe."""
        if not indicators:
            return 'NEUTRAL'
        
        rsi = indicators.get('rsi', 50)
        macd_hist = indicators.get('macd_hist', 0)
        trend = indicators.get('trend', 'NEUTRAL')
        
        bullish_count = 0
        bearish_count = 0
        
        # RSI
        if rsi < 35:
            bullish_count += 1
        elif rsi > 65:
            bearish_count += 1
        
        # MACD
        if macd_hist > 0:
            bullish_count += 1
        elif macd_hist < 0:
            bearish_count += 1
        
        # Trend
        if trend == 'UP':
            bullish_count += 1
        else:
            bearish_count += 1
        
        if bullish_count >= 2:
            return 'BULLISH'
        elif bearish_count >= 2:
            return 'BEARISH'
        return 'NEUTRAL'
    
    def evaluate(self, symbol: str) -> Dict:
        """
        Evaluate trading opportunity for a symbol.
        
        This is the main method that:
        1. Gets AI prediction from XGBoost
        2. Analyzes all indicators
        3. Checks multi-timeframe alignment
        4. Scores overall confidence
        5. Returns signal with full reasoning
        
        Returns:
            Dict with signal, confidence, reasoning, and all indicator values
        """
        if symbol not in self.indicators or not self.indicators[symbol].get('1min'):
            return self._empty_evaluation(symbol)
        
        ind = self.indicators[symbol]['1min']
        
        # Get AI prediction
        prediction = self._get_ai_prediction(symbol)
        
        # Build reasoning list
        reasoning = []
        confidence_scores = []
        
        # 1. AI Model prediction (35% weight)
        if prediction is not None:
            if prediction > 0.65:
                reasoning.append(f"Model bullish ({prediction:.0%})")
                confidence_scores.append(prediction * 100)
            elif prediction < 0.35:
                reasoning.append(f"Model bearish ({prediction:.0%})")
                confidence_scores.append((1 - prediction) * 100)
            else:
                reasoning.append(f"Model neutral ({prediction:.0%})")
                confidence_scores.append(50)
        else:
            reasoning.append("Model unavailable")
            confidence_scores.append(50)
        
        # 2. RSI analysis (25% weight)
        rsi = ind.get('rsi', 50)
        if rsi < 30:
            reasoning.append(f"RSI oversold ({rsi:.1f} < 30)")
            confidence_scores.append(75 + (30 - rsi))
        elif rsi < 35:
            reasoning.append(f"RSI near oversold ({rsi:.1f})")
            confidence_scores.append(65)
        elif rsi > 70:
            reasoning.append(f"RSI overbought ({rsi:.1f} > 70)")
            confidence_scores.append(75 + (rsi - 70))
        elif rsi > 65:
            reasoning.append(f"RSI near overbought ({rsi:.1f})")
            confidence_scores.append(65)
        else:
            reasoning.append(f"RSI neutral ({rsi:.1f})")
            confidence_scores.append(50)
        
        # 3. Volume analysis (20% weight)
        vol_ratio = ind.get('volume_ratio', 1.0)
        if vol_ratio > 2.0:
            reasoning.append(f"Volume spike ({vol_ratio:.1f}x avg)")
            confidence_scores.append(80)
        elif vol_ratio > 1.5:
            reasoning.append(f"High volume ({vol_ratio:.1f}x avg)")
            confidence_scores.append(70)
        elif vol_ratio < 0.5:
            reasoning.append(f"Low volume ({vol_ratio:.1f}x avg)")
            confidence_scores.append(40)
        else:
            reasoning.append(f"Normal volume ({vol_ratio:.1f}x)")
            confidence_scores.append(55)
        
        # 4. MACD analysis (10% weight)
        macd_hist = ind.get('macd_hist', 0)
        if macd_hist > 0:
            reasoning.append("MACD bullish")
            confidence_scores.append(65)
        elif macd_hist < 0:
            reasoning.append("MACD bearish")
            confidence_scores.append(65)
        else:
            reasoning.append("MACD neutral")
            confidence_scores.append(50)
        
        # 5. Bollinger Bands (10% weight)
        bb_pos = ind.get('bb_position', 0.5)
        if bb_pos < 0.2:
            reasoning.append(f"Near BB lower ({bb_pos:.0%})")
            confidence_scores.append(70)
        elif bb_pos > 0.8:
            reasoning.append(f"Near BB upper ({bb_pos:.0%})")
            confidence_scores.append(70)
        else:
            reasoning.append(f"Mid BB range ({bb_pos:.0%})")
            confidence_scores.append(50)
        
        # Calculate weighted average confidence
        weights = [0.35, 0.25, 0.20, 0.10, 0.10]
        overall_confidence = sum(s * w for s, w in zip(confidence_scores, weights))
        
        # Multi-timeframe alignment bonus
        tf_signals = self.tf_signals.get(symbol, {})
        bullish_tfs = sum(1 for s in tf_signals.values() if s == 'BULLISH')
        bearish_tfs = sum(1 for s in tf_signals.values() if s == 'BEARISH')
        
        alignment_bonus = 0
        if bullish_tfs >= 2:
            alignment_bonus = 10
            reasoning.append(f"{bullish_tfs}/3 timeframes bullish")
        elif bearish_tfs >= 2:
            alignment_bonus = 10
            reasoning.append(f"{bearish_tfs}/3 timeframes bearish")
        
        overall_confidence += alignment_bonus
        overall_confidence = min(overall_confidence, 95)  # Cap at 95%
        
        # Determine signal
        signal = 'HOLD'
        if prediction is not None:
            if prediction > self.buy_threshold and rsi < 60 and overall_confidence >= self.min_confidence:
                signal = 'BUY'
            elif prediction < self.sell_threshold and rsi > 40 and overall_confidence >= self.min_confidence:
                signal = 'SELL'
        else:
            # Fallback to indicator-based signals when model unavailable
            if rsi < 30 and vol_ratio > 0.5 and overall_confidence >= self.min_confidence:
                signal = 'BUY'
            elif rsi > 70 and vol_ratio > 0.5 and overall_confidence >= self.min_confidence:
                signal = 'SELL'
        
        # Build full AI state
        ai_state = {
            'prediction': prediction,
            'confidence': int(overall_confidence),
            'signal': signal,
            'reasoning': reasoning,
            'features': {
                # Core indicators
                'rsi': round(rsi, 1),
                'macd': round(ind.get('macd', 0), 4),
                'macd_hist': round(ind.get('macd_hist', 0), 4),
                'bb_position': round(bb_pos, 2),
                'atr_pct': round(ind.get('atr_pct', 0), 2),
                'volume_ratio': round(vol_ratio, 2),
                'momentum': round(ind.get('momentum', 0), 2),
                'adx': round(ind.get('adx', 0), 1),
                'price': round(ind.get('price', 0), 2),
                'trend': ind.get('trend', 'NEUTRAL'),
                # Oscillators
                'stoch_k': round(ind.get('stoch_k', 50), 1),
                'stoch_d': round(ind.get('stoch_d', 50), 1),
                'cci': round(ind.get('cci', 0), 1),
                'mfi': round(ind.get('mfi', 50), 1),
                'willr': round(ind.get('willr', -50), 1),
                # Trend/Momentum
                'trend_alignment': round(ind.get('trend_alignment', 0), 1),
                'momentum_alignment': round(ind.get('momentum_alignment', 0), 1),
                'roc_5': round(ind.get('roc_5', 0), 2),
                # Volatility
                'volatility_regime': round(ind.get('volatility_regime', 1), 2),
                'bb_width': round(ind.get('bb_width', 0), 4),
                'bb_squeeze': ind.get('bb_squeeze', 0),
                # Volume
                'volume_trend': round(ind.get('volume_trend', 1), 2),
                'volume_spike': ind.get('volume_spike', 0),
                'obv_momentum': round(ind.get('obv_momentum', 0), 2),
                # Price patterns
                'breakout_up': ind.get('breakout_up', 0),
                'breakout_down': ind.get('breakout_down', 0),
                'price_position': round(ind.get('price_position', 0.5), 2),
            },
            'thresholds': {
                'buy_threshold': self.buy_threshold,
                'sell_threshold': self.sell_threshold,
                'min_confidence': self.min_confidence
            },
            'multi_timeframe': tf_signals,
            'feature_importance': self.FEATURE_WEIGHTS,
            'timestamp': datetime.now().isoformat()
        }
        
        # Cache for dashboard
        self.ai_state[symbol] = ai_state
        
        return ai_state
    
    def _get_ai_prediction(self, symbol: str) -> Optional[float]:
        """
        Get prediction from XGBoost model(s).

        Uses SYMBOL-SPECIFIC models if available, otherwise falls back to defaults.

        Combines predictions with weights:
        - 5-min: 50% (short-term scalping)
        - 15-min: 30% (medium-term confirmation)
        - 30-min: 20% (trend direction)
        """
        # Get models for this specific symbol
        symbol_models = self.get_models_for_symbol(symbol)

        if self.model is None and not symbol_models:
            return None

        if symbol not in self.indicators:
            return None

        ind = self.indicators[symbol].get('1min', {})
        if not ind:
            return None

        # Check if we have all required features (need at least 38 features)
        required_features = [
            'rsi', 'macd', 'macd_signal', 'bb_upper', 'bb_middle', 'bb_lower',
            'atr', 'volume_ratio', 'momentum', 'adx',
            'breakout_up', 'breakout_down', 'price_position',
            'body_size', 'upper_wick', 'lower_wick', 'bullish_candle',
            'atr_pct', 'volatility_regime', 'bb_width', 'bb_squeeze',
            'trend_alignment', 'roc_5', 'roc_10', 'roc_20', 'momentum_alignment',
            'volume_trend', 'volume_spike', 'obv_momentum',
            'rsi_momentum', 'macd_hist', 'macd_hist_momentum',
            'stoch_k', 'stoch_d', 'stoch_cross', 'willr', 'cci', 'mfi'
        ]

        # Check if all features are available
        missing = [f for f in required_features if f not in ind]
        if missing:
            logger.debug(f"Missing features for {symbol}: {len(missing)} features")
            return None

        try:
            # Prepare ALL 38 features for model (MUST match training features)
            features = pd.DataFrame([{
                # Core (10)
                'rsi': ind.get('rsi', 50),
                'macd': ind.get('macd', 0),
                'macd_signal': ind.get('macd_signal', 0),
                'bb_upper': ind.get('bb_upper', 0),
                'bb_middle': ind.get('bb_middle', 0),
                'bb_lower': ind.get('bb_lower', 0),
                'atr': ind.get('atr', 0),
                'volume_ratio': ind.get('volume_ratio', 1),
                'momentum': ind.get('momentum', 0),
                'adx': ind.get('adx', 25),
                # Price patterns (7)
                'breakout_up': ind.get('breakout_up', 0),
                'breakout_down': ind.get('breakout_down', 0),
                'price_position': ind.get('price_position', 0.5),
                'body_size': ind.get('body_size', 0),
                'upper_wick': ind.get('upper_wick', 0),
                'lower_wick': ind.get('lower_wick', 0),
                'bullish_candle': ind.get('bullish_candle', 0),
                # Volatility (4)
                'atr_pct': ind.get('atr_pct', 0),
                'volatility_regime': ind.get('volatility_regime', 1),
                'bb_width': ind.get('bb_width', 0),
                'bb_squeeze': ind.get('bb_squeeze', 0),
                # Trend/Momentum (5)
                'trend_alignment': ind.get('trend_alignment', 0),
                'roc_5': ind.get('roc_5', 0),
                'roc_10': ind.get('roc_10', 0),
                'roc_20': ind.get('roc_20', 0),
                'momentum_alignment': ind.get('momentum_alignment', 0),
                # Volume (3)
                'volume_trend': ind.get('volume_trend', 1),
                'volume_spike': ind.get('volume_spike', 0),
                'obv_momentum': ind.get('obv_momentum', 0),
                # Oscillators (9)
                'rsi_momentum': ind.get('rsi_momentum', 0),
                'macd_hist': ind.get('macd_hist', 0),
                'macd_hist_momentum': ind.get('macd_hist_momentum', 0),
                'stoch_k': ind.get('stoch_k', 50),
                'stoch_d': ind.get('stoch_d', 50),
                'stoch_cross': ind.get('stoch_cross', 0),
                'willr': ind.get('willr', -50),
                'cci': ind.get('cci', 0),
                'mfi': ind.get('mfi', 50),
            }])

            # Use symbol-specific multi-horizon models if available
            if symbol_models:
                predictions = {}
                weights = {5: 0.50, 15: 0.30, 30: 0.20}

                for horizon, model in symbol_models.items():
                    pred = float(model.predict(features)[0])
                    predictions[horizon] = pred

                # Weighted average
                weighted_pred = sum(
                    predictions.get(h, 0.5) * w
                    for h, w in weights.items()
                    if h in predictions
                )
                total_weight = sum(w for h, w in weights.items() if h in predictions)
                prediction = weighted_pred / total_weight if total_weight > 0 else 0.5

                return prediction

            # Fallback to single model
            prediction = float(self.model.predict(features)[0])
            return prediction

        except Exception as e:
            logger.warning(f"Prediction error for {symbol}: {e}")
            return None
    
    def _empty_evaluation(self, symbol: str) -> Dict:
        """Return empty evaluation when not enough data."""
        return {
            'prediction': None,
            'confidence': 0,
            'signal': 'HOLD',
            'reasoning': ['Insufficient data for analysis'],
            'features': {},
            'thresholds': {
                'buy_threshold': self.buy_threshold,
                'sell_threshold': self.sell_threshold,
                'min_confidence': self.min_confidence
            },
            'multi_timeframe': {},
            'feature_importance': self.FEATURE_WEIGHTS,
            'timestamp': datetime.now().isoformat()
        }
    
    def get_all_evaluations(self) -> Dict[str, Dict]:
        """Evaluate all tracked symbols."""
        evaluations = {}
        for symbol in self.bars.keys():
            evaluations[symbol] = self.evaluate(symbol)
        return evaluations

    def get_warmup_status(self) -> Dict:
        """Get warmup status for all symbols."""
        required_bars = 50  # Minimum bars needed for full indicator calculation

        status = {
            'required_bars': required_bars,
            'symbols': {}
        }

        for symbol in self.bars.keys():
            bar_count = len(self.bars[symbol])
            is_ready = bar_count >= required_bars
            progress = min(100, int(bar_count / required_bars * 100))

            status['symbols'][symbol] = {
                'bars': bar_count,
                'ready': is_ready,
                'progress': progress
            }

        # Overall status
        if status['symbols']:
            total_bars = sum(s['bars'] for s in status['symbols'].values())
            avg_bars = total_bars // len(status['symbols']) if status['symbols'] else 0
            all_ready = all(s['ready'] for s in status['symbols'].values())
            avg_progress = sum(s['progress'] for s in status['symbols'].values()) // len(status['symbols']) if status['symbols'] else 0
        else:
            avg_bars = 0
            all_ready = False
            avg_progress = 0

        status['avg_bars'] = avg_bars
        status['all_ready'] = all_ready
        status['avg_progress'] = avg_progress

        return status

    def record_trade_result(self, profit_pct: float) -> None:
        """Record a trade result for Kelly calculations."""
        self.trade_history.append({
            'profit_pct': profit_pct,
            'timestamp': datetime.now()
        })
        
        if profit_pct > 0:
            self.win_count += 1
        else:
            self.loss_count += 1
    
    def get_kelly_stats(self) -> Dict:
        """Calculate Kelly Criterion statistics."""
        total_trades = self.win_count + self.loss_count
        if total_trades < 5:
            return {
                'win_rate': 0.5,
                'avg_win': 0,
                'avg_loss': 0,
                'kelly_fraction': 0.25,  # Default conservative
                'sample_size': total_trades
            }
        
        win_rate = self.win_count / total_trades
        
        wins = [t['profit_pct'] for t in self.trade_history if t['profit_pct'] > 0]
        losses = [abs(t['profit_pct']) for t in self.trade_history if t['profit_pct'] < 0]
        
        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0
        
        # Kelly formula: f = (bp - q) / b
        # where b = avg_win/avg_loss, p = win_rate, q = 1 - p
        if avg_loss > 0:
            b = avg_win / avg_loss
            kelly = (b * win_rate - (1 - win_rate)) / b
        else:
            kelly = 0.25
        
        # Apply half-Kelly for safety
        kelly = max(0, min(kelly * 0.5, 0.5))
        
        return {
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'kelly_fraction': kelly,
            'sample_size': total_trades
        }



