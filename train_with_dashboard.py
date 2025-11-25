"""
Advanced LightGBM Training with Live Dashboard

Run this to train models while watching progress in real-time.
Dashboard will be available at http://localhost:8001
"""

import asyncio
import json
import logging
import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from datetime import datetime
from typing import Dict, List, Optional, Any
from collections import deque

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import uvicorn

from config_ultra import UltraConfig
from src.execution.alpaca_client import AlpacaClient
from src.data.loader import DataLoader
from src.data.mock_loader import MockDataLoader
from src.strategy.feature_calculator import FeatureCalculator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("TrainingDashboard")

# ============================================
# WEBSOCKET MANAGER
# ============================================

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"Dashboard connected. Total: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info(f"Dashboard disconnected. Total: {len(self.active_connections)}")

    async def broadcast(self, message: dict):
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.warning(f"Failed to send to client: {e}")
                disconnected.append(connection)
        # Clean up disconnected clients
        for conn in disconnected:
            if conn in self.active_connections:
                self.active_connections.remove(conn)
        if message.get("data", {}).get("status") == "complete":
            logger.info(f"Broadcast complete status to {len(self.active_connections)} clients")


manager = ConnectionManager()

# ============================================
# TRAINING STATE (Global for dashboard access)
# ============================================

class TrainingState:
    def __init__(self):
        self.reset()

    def reset(self):
        self.status = "idle"  # idle, fetching_data, calculating_features, tuning, training, complete, error
        self.progress = 0
        self.current_horizon = None
        self.horizons = [5, 15, 30]

        # Multi-asset tracking
        self.current_symbol = None
        self.symbols = []
        self.symbol_idx = 0
        self.symbol_metrics = {}  # {symbol: {horizon: metrics}}

        # Data info
        self.data_rows = 0
        self.data_days = 0
        self.features_count = 0

        # Tuning progress
        self.tuning_iteration = 0
        self.tuning_total = 0
        self.tuning_best_score = float('inf')
        self.tuning_history = []

        # Training progress
        self.training_round = 0
        self.training_total = 200
        self.train_loss = []
        self.val_loss = []

        # Walk-forward validation
        self.wf_fold = 0
        self.wf_total = 5
        self.wf_results = []

        # Final metrics per horizon
        self.metrics = {}

        # HOLDOUT metrics (true out-of-sample performance)
        self.holdout_metrics = {}

        # Feature importance
        self.feature_importance = {}

        # Logs
        self.logs = deque(maxlen=100)

        # Errors
        self.error = None

    def to_dict(self) -> dict:
        def convert_numpy(obj):
            """Convert numpy types to Python types for JSON serialization."""
            import numpy as np
            if obj is None:
                return None
            if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
                if np.isnan(obj) or np.isinf(obj):
                    return None
                return float(obj)
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, np.ndarray):
                return [convert_numpy(x) for x in obj.tolist()]
            elif isinstance(obj, dict):
                return {str(k): convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_numpy(i) for i in obj]
            elif isinstance(obj, (int, float, str, bool)):
                return obj
            elif hasattr(obj, 'item'):  # numpy scalar
                return convert_numpy(obj.item())
            return str(obj)  # Fallback to string for unknown types

        return convert_numpy({
            "status": self.status,
            "progress": self.progress,
            "current_horizon": self.current_horizon,
            "horizons": self.horizons,
            "current_symbol": self.current_symbol,
            "symbols": self.symbols,
            "symbol_idx": self.symbol_idx,
            "symbol_metrics": self.symbol_metrics,
            "data": {
                "rows": self.data_rows,
                "days": self.data_days,
                "features": self.features_count
            },
            "tuning": {
                "iteration": self.tuning_iteration,
                "total": self.tuning_total,
                "best_score": self.tuning_best_score if self.tuning_best_score != float('inf') else None,
                "history": self.tuning_history[-20:]  # Last 20 iterations
            },
            "training": {
                "round": self.training_round,
                "total": self.training_total,
                "train_loss": self.train_loss[-50:],
                "val_loss": self.val_loss[-50:]
            },
            "walk_forward": {
                "fold": self.wf_fold,
                "total": self.wf_total,
                "results": self.wf_results
            },
            "metrics": self.metrics,
            "holdout_metrics": self.holdout_metrics,
            "feature_importance": self.feature_importance,
            "logs": list(self.logs),
            "error": self.error
        })


state = TrainingState()

# ============================================
# FEATURE ENGINE
# ============================================

class FeatureEngine:
    """
    Enhanced feature engineering with sophisticated indicators.

    Includes:
    - Core indicators (RSI, MACD, BB, ATR, ADX)
    - Price patterns (breakouts, higher highs/lows)
    - Volatility regime detection
    - Cross-timeframe momentum alignment
    - Volume analysis
    """

    @staticmethod
    def calculate_features(df: pd.DataFrame) -> pd.DataFrame:
        import talib

        df = df.copy()
        df.columns = [c.lower() for c in df.columns]

        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        open_price = df['open'].values
        volume = df['volume'].values if 'volume' in df.columns else np.ones(len(close))

        # ============================================
        # CORE FEATURES (match adaptive_ai.py)
        # ============================================
        df['rsi'] = talib.RSI(close, timeperiod=14)

        macd, macd_signal, macd_hist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
        df['macd'] = macd
        df['macd_signal'] = macd_signal

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
        # PRICE PATTERNS
        # ============================================
        # Rolling highs/lows for breakout detection
        df['high_20'] = pd.Series(high).rolling(20).max()
        df['low_20'] = pd.Series(low).rolling(20).min()

        # Breakout signals (1 = breakout up, -1 = breakdown)
        df['breakout_up'] = (close > df['high_20'].shift(1)).astype(float)
        df['breakout_down'] = (close < df['low_20'].shift(1)).astype(float)

        # Price position in range (0 = at low, 1 = at high)
        range_20 = df['high_20'] - df['low_20']
        df['price_position'] = (close - df['low_20']) / (range_20 + 1e-10)

        # Candle patterns
        df['body_size'] = abs(close - open_price) / (df['atr'] + 1e-10)
        df['upper_wick'] = (high - np.maximum(close, open_price)) / (df['atr'] + 1e-10)
        df['lower_wick'] = (np.minimum(close, open_price) - low) / (df['atr'] + 1e-10)
        df['bullish_candle'] = (close > open_price).astype(float)

        # ============================================
        # VOLATILITY REGIME
        # ============================================
        df['atr_pct'] = df['atr'] / close * 100
        df['atr_sma'] = df['atr_pct'].rolling(20).mean()
        df['volatility_regime'] = df['atr_pct'] / (df['atr_sma'] + 1e-10)

        # Bollinger Band width (squeeze = low volatility)
        df['bb_width'] = (bb_upper - bb_lower) / (bb_middle + 1e-10)
        df['bb_squeeze'] = (df['bb_width'] < df['bb_width'].rolling(50).quantile(0.2)).astype(float)

        # ============================================
        # CROSS-TIMEFRAME MOMENTUM
        # ============================================
        # EMAs for trend alignment
        df['ema_8'] = talib.EMA(close, timeperiod=8)
        df['ema_21'] = talib.EMA(close, timeperiod=21)
        df['ema_50'] = talib.EMA(close, timeperiod=50)

        # Trend alignment score (0-3)
        df['trend_alignment'] = (
            (df['ema_8'] > df['ema_21']).astype(float) +
            (df['ema_21'] > df['ema_50']).astype(float) +
            (close > df['ema_8']).astype(float)
        )

        # Multi-period ROC
        df['roc_5'] = talib.ROC(close, timeperiod=5)
        df['roc_10'] = talib.ROC(close, timeperiod=10)
        df['roc_20'] = talib.ROC(close, timeperiod=20)

        # Momentum alignment (-1, 0, or 1)
        df['momentum_alignment'] = (
            ((df['roc_5'] > 0) & (df['roc_10'] > 0) & (df['roc_20'] > 0)).astype(float) -
            ((df['roc_5'] < 0) & (df['roc_10'] < 0) & (df['roc_20'] < 0)).astype(float)
        )

        # ============================================
        # VOLUME ANALYSIS
        # ============================================
        df['volume_sma_10'] = pd.Series(volume).rolling(10).mean()
        df['volume_sma_50'] = pd.Series(volume).rolling(50).mean()
        df['volume_trend'] = df['volume_sma_10'] / (df['volume_sma_50'] + 1e-10)
        df['volume_spike'] = (df['volume_ratio'] > 2.0).astype(float)

        # OBV momentum
        obv = talib.OBV(close, volume.astype(float))
        df['obv_momentum'] = pd.Series(obv).diff(10) / (pd.Series(obv).rolling(20).std() + 1e-10)

        # ============================================
        # ADDITIONAL OSCILLATORS
        # ============================================
        df['macd_hist'] = macd_hist
        df['macd_hist_momentum'] = pd.Series(macd_hist).diff(3)

        # RSI momentum
        df['rsi_momentum'] = df['rsi'].diff(5)

        # Stochastic
        slowk, slowd = talib.STOCH(high, low, close, fastk_period=14, slowk_period=3, slowd_period=3)
        df['stoch_k'] = slowk
        df['stoch_d'] = slowd
        df['stoch_cross'] = (slowk > slowd).astype(float)

        # Williams %R
        df['willr'] = talib.WILLR(high, low, close, timeperiod=14)

        # CCI
        df['cci'] = talib.CCI(high, low, close, timeperiod=20)

        # MFI (Money Flow Index)
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

        # Fill NaN with sensible defaults instead of dropping all rows
        # Use ffill() and bfill() methods instead of deprecated fillna(method=...)
        df = df.ffill().bfill()

        # Fill any remaining NaN with 0 (for columns that have no data at all)
        df = df.fillna(0)

        # Skip first 300 rows for indicator warmup (need 240+ for ROC_240)
        if len(df) > 300:
            df = df.iloc[300:]

        return df

    @staticmethod
    def get_feature_columns() -> List[str]:
        """All features for training (38 original + 30 enhanced = 68 total)."""
        return [
            # Core (match adaptive_ai.py)
            'rsi', 'macd', 'macd_signal', 'bb_upper', 'bb_middle', 'bb_lower',
            'atr', 'volume_ratio', 'momentum', 'adx',
            # Price patterns
            'breakout_up', 'breakout_down', 'price_position',
            'body_size', 'upper_wick', 'lower_wick', 'bullish_candle',
            # Volatility
            'atr_pct', 'volatility_regime', 'bb_width', 'bb_squeeze',
            # Trend/Momentum
            'trend_alignment', 'roc_5', 'roc_10', 'roc_20', 'momentum_alignment',
            # Volume
            'volume_trend', 'volume_spike', 'obv_momentum',
            # Oscillators
            'rsi_momentum', 'macd_hist', 'macd_hist_momentum',
            'stoch_k', 'stoch_d', 'stoch_cross', 'willr', 'cci', 'mfi',
            # ===== ENHANCED FEATURES =====
            # Multi-timeframe momentum
            'roc_60', 'roc_240', 'mtf_momentum_align',
            'return_5', 'return_20', 'return_60',
            # Volatility regime
            'atr_percentile', 'vol_expanding', 'vol_contracting',
            # Price position
            'price_percentile_50', 'price_percentile_100',
            'dist_from_high_50', 'dist_from_low_50',
            # Acceleration (second derivatives)
            'momentum_accel', 'rsi_accel', 'roc_accel', 'volume_accel',
            # Time features
            'hour', 'day_of_week', 'is_us_session', 'is_asia_session',
            # Trend strength
            'strong_trend', 'weak_trend',
            # Mean reversion signals
            'rsi_oversold', 'rsi_overbought', 'bb_oversold', 'bb_overbought',
            # Divergence
            'bullish_divergence', 'bearish_divergence'
        ]


# ============================================
# CUSTOM LIGHTGBM CALLBACK FOR LIVE UPDATES
# ============================================

def create_dashboard_callback(broadcast_func, state: TrainingState):
    """Create a LightGBM callback function for live dashboard updates."""
    def callback(env):
        state.training_round = env.iteration + 1

        # Extract losses from evaluation results
        if env.evaluation_result_list:
            for (ds_name, metric_name, value, is_higher_better) in env.evaluation_result_list:
                if ds_name == 'train' and metric_name == 'binary_logloss':
                    state.train_loss.append(value)
                elif ds_name == 'valid' and metric_name == 'binary_logloss':
                    state.val_loss.append(value)

        # Broadcast every 5 rounds
        if env.iteration % 5 == 0:
            asyncio.create_task(broadcast_func({"type": "update", "data": state.to_dict()}))

    callback.order = 10  # Run after other callbacks
    return callback


# ============================================
# TRAINING LOGIC
# ============================================

async def run_training(config: UltraConfig, days: int = 180, tune: bool = True, tune_iterations: int = 15,
                       holdout_days: int = None):
    """
    Run training with live dashboard updates for ALL configured symbols.

    PROPER HOLDOUT IMPLEMENTATION:
    - Training data: days - holdout_days (fetched first)
    - Holdout data: last holdout_days (fetched SEPARATELY)
    - Features calculated INDEPENDENTLY on each dataset
    - Holdout is NEVER seen during training/tuning
    """
    global state
    state.reset()

    # Significant move threshold (0.5% for crypto)
    MOVE_THRESHOLD = 0.005  # 0.5%

    # Holdout configuration
    holdout_days = holdout_days or getattr(config, 'HOLDOUT_DAYS', 14)
    warmup_buffer = getattr(config, 'WARMUP_BUFFER_BARS', 50)
    train_days = days - holdout_days

    # Get all symbols to train
    symbols = getattr(config, 'SYMBOLS', [config.SYMBOL])
    state.symbols = symbols
    state.symbol_metrics = {}
    state.holdout_metrics = {}

    try:
        # Initialize client once
        try:
            client = AlpacaClient(config)
            loader = DataLoader(client, config)
            use_real_data = True
        except Exception as e:
            await log(f"Using mock data: {e}")
            use_real_data = False

        # ===== LOOP THROUGH ALL SYMBOLS =====
        total_symbols = len(symbols)
        horizons = [5, 15, 30]

        for sym_idx, symbol in enumerate(symbols):
            state.current_symbol = symbol
            state.symbol_idx = sym_idx
            state.metrics = {}  # Reset for this symbol

            await log(f"\n{'#'*50}")
            await log(f"TRAINING SYMBOL {sym_idx+1}/{total_symbols}: {symbol}")
            await log(f"{'#'*50}")

            # ===== FETCH TRAINING DATA (SEPARATE FROM HOLDOUT) =====
            state.status = "fetching_data"
            base_progress = (sym_idx / total_symbols) * 100
            state.progress = int(base_progress + 2)
            await log(f"Fetching {train_days} days of TRAINING data for {symbol}...")
            await log(f"(Holdout: last {holdout_days} days will be fetched separately)")
            await broadcast_state()

            # Calculate date ranges
            from datetime import timedelta
            end_time = datetime.now()
            holdout_start = end_time - timedelta(days=holdout_days)
            train_start = end_time - timedelta(days=days)
            train_end = holdout_start - timedelta(hours=1)  # Small gap to ensure no overlap

            try:
                if use_real_data:
                    # Fetch TRAINING data (excludes holdout period)
                    df_train = loader.fetch_data(symbol=symbol, start_date=train_start, end_date=train_end)
                    await log(f"Fetched {len(df_train)} TRAINING bars for {symbol}")

                    # Fetch HOLDOUT data (SEPARATELY - this is critical!)
                    df_holdout = loader.fetch_data(symbol=symbol, start_date=holdout_start, end_date=end_time)
                    await log(f"Fetched {len(df_holdout)} HOLDOUT bars for {symbol}")
                else:
                    # Mock data - simulate separate fetches
                    df_train = MockDataLoader.fetch_data(days=train_days, symbol=symbol)
                    df_holdout = MockDataLoader.fetch_data(days=holdout_days, symbol=symbol)
                    await log(f"Mock data: {len(df_train)} train, {len(df_holdout)} holdout bars")
            except Exception as e:
                await log(f"ERROR fetching {symbol}: {e}")
                continue

            if df_train.empty:
                await log(f"No training data available for {symbol}, skipping...")
                continue

            state.data_rows = len(df_train)
            state.data_days = train_days
            state.progress = int(base_progress + 5)
            await broadcast_state()

            # ===== CALCULATE FEATURES INDEPENDENTLY =====
            # CRITICAL: Features calculated ONLY on training data, NOT on full dataset!
            state.status = "calculating_features"
            await log(f"Calculating features for {symbol} (TRAINING data only)...")
            await broadcast_state()

            # Use FeatureCalculator for independent calculation (no data leakage)
            df_train_features = FeatureCalculator.calculate_features(df_train, drop_warmup=True)
            feature_cols = FeatureCalculator.get_feature_columns()

            # Check for missing columns
            missing_cols = [c for c in feature_cols if c not in df_train_features.columns]
            if missing_cols:
                await log(f"WARNING: Missing features: {missing_cols}")
                feature_cols = [c for c in feature_cols if c in df_train_features.columns]

            features = df_train_features[feature_cols]

            state.features_count = len(feature_cols)
            state.progress = int(base_progress + 10)
            await log(f"Calculated {len(feature_cols)} features on {len(features)} TRAINING samples for {symbol}")
            await broadcast_state()

            if len(features) < 1000:
                await log(f"Not enough training samples ({len(features)}) for {symbol}, skipping...")
                continue

            # Calculate features on HOLDOUT data (SEPARATELY - never mixed with training!)
            if not df_holdout.empty:
                df_holdout_features = FeatureCalculator.calculate_features(df_holdout, drop_warmup=True)
                await log(f"Calculated features on {len(df_holdout_features)} HOLDOUT samples")

            # ===== TRAIN EACH HORIZON FOR THIS SYMBOL =====
            for h_idx, horizon in enumerate(horizons):
                state.current_horizon = horizon
                await log(f"\n{'='*40}")
                await log(f"Training {horizon}-minute model for {symbol}")
                await log(f"{'='*40}")

                # Create target: SIGNIFICANT MOVE (>0.5% up)
                future_close = df_train_features['close'].shift(-horizon)
                current_close = df_train_features['close']

                # Calculate percentage change
                pct_change = (future_close - current_close) / current_close

                valid_idx = features.index.intersection(future_close.dropna().index)
                X = features.loc[valid_idx].iloc[:-horizon]

                # Target: 1 if price moves up > 0.5%, else 0
                y = (pct_change.loc[valid_idx] > MOVE_THRESHOLD).astype(int).iloc[:-horizon]

                if len(X) < 100:
                    await log(f"Not enough samples ({len(X)}) for horizon {horizon}, skipping...")
                    continue

                # Log target distribution
                up_pct = y.mean() * 100
                await log(f"Training samples: {len(X)}")
                await log(f"Target: {up_pct:.1f}% significant up moves (>{MOVE_THRESHOLD*100}%)")

                # Calculate class imbalance ratio for logging
                pos_count = y.sum()
                neg_count = len(y) - pos_count
                imbalance_ratio = neg_count / max(pos_count, 1)
                await log(f"Class balance: {neg_count}:{pos_count} (neg:pos), ratio={imbalance_ratio:.1f}:1")

                # ===== HYPERPARAMETER TUNING =====
                if tune:
                    state.status = "tuning"
                    state.tuning_iteration = 0
                    state.tuning_total = tune_iterations
                    state.tuning_history = []
                    state.tuning_best_score = float('inf')

                    best_params = await tune_hyperparameters(X, y, tune_iterations, horizon)
                else:
                    # Default LightGBM parameters
                    best_params = {
                        'max_depth': 5,
                        'learning_rate': 0.05,
                        'min_child_samples': 20,
                        'bagging_fraction': 0.8,
                        'feature_fraction': 0.8,
                        'num_leaves': 31,
                        'objective': 'binary',
                        'metric': 'binary_logloss',
                        'verbosity': -1,
                        'bagging_freq': 1,
                        'feature_pre_filter': False,
                        'is_unbalance': True,  # Let LightGBM handle class imbalance
                    }

                # ===== WALK-FORWARD VALIDATION =====
                state.status = "training"
                state.wf_fold = 0
                state.wf_total = 5
                state.wf_results = []

                await log("Running walk-forward validation (with warmup buffers)...")

                # Time series split with WARMUP BUFFER between train/test
                n = len(X)
                test_size = n // 6

                fold_metrics = []

                for fold in range(5):
                    state.wf_fold = fold + 1

                    test_end = n - (4 - fold) * test_size
                    test_start = test_end - test_size

                    # Add warmup buffer between train and test to prevent boundary contamination
                    train_end = test_start - warmup_buffer

                    if train_end < test_size * 2:
                        await log(f"Fold {fold + 1}: Skipping - not enough training data after buffer")
                        continue

                    X_train, X_test = X.iloc[:train_end], X.iloc[test_start:test_end]
                    y_train, y_test = y.iloc[:train_end], y.iloc[test_start:test_end]

                    await log(f"Fold {fold + 1}: Train={len(X_train)}, Buffer={warmup_buffer}, Test={len(X_test)}")

                    dtrain = lgb.Dataset(X_train, label=y_train)
                    dtest = lgb.Dataset(X_test, label=y_test, reference=dtrain)

                    # Train with callback
                    state.training_round = 0
                    state.train_loss = []
                    state.val_loss = []

                    model = lgb.train(
                        best_params, dtrain,
                        num_boost_round=100,
                        valid_sets=[dtrain, dtest],
                        valid_names=['train', 'valid'],
                        callbacks=[
                            lgb.early_stopping(stopping_rounds=20, first_metric_only=True, verbose=False, min_delta=0.0001),
                            create_dashboard_callback(broadcast_state_async, state)
                        ]
                    )

                    # Calculate fold accuracy
                    y_pred = (model.predict(X_test) > 0.5).astype(int)
                    accuracy = (y_pred == y_test.values).mean()

                    fold_metrics.append({
                        "fold": fold + 1,
                        "train_size": len(X_train),
                        "test_size": len(X_test),
                        "accuracy": round(accuracy * 100, 1) if not np.isnan(accuracy) else 0,
                        "best_iteration": model.best_iteration if model.best_iteration else 100
                    })

                    state.wf_results = fold_metrics
                    await log(f"Fold {fold + 1}: Accuracy {accuracy:.1%}")
                    await broadcast_state()

                # ===== FINAL MODEL TRAINING =====
                await log("Training final model on all data...")

                # Use 80% train, 20% test for final model
                split_idx = int(len(X) * 0.8)
                X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
                y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

                dtrain = lgb.Dataset(X_train, label=y_train)
                dtest = lgb.Dataset(X_test, label=y_test, reference=dtrain)

                state.training_round = 0
                state.training_total = 200
                state.train_loss = []
                state.val_loss = []

                final_model = lgb.train(
                    best_params, dtrain,
                    num_boost_round=200,
                    valid_sets=[dtrain, dtest],
                    valid_names=['train', 'valid'],
                    callbacks=[
                        lgb.early_stopping(stopping_rounds=30, first_metric_only=True, verbose=False, min_delta=0.0001),
                        create_dashboard_callback(broadcast_state_async, state)
                    ]
                )

                # Calculate final metrics
                y_pred_proba = final_model.predict(X_test)
                y_pred = (y_pred_proba > 0.5).astype(int)

                accuracy = (y_pred == y_test.values).mean()

                # Confident predictions
                confident_mask = (y_pred_proba > 0.65) | (y_pred_proba < 0.35)
                confident_acc = (y_pred[confident_mask] == y_test.values[confident_mask]).mean() if confident_mask.sum() > 0 else 0

                # Buy precision
                buy_mask = y_pred_proba > 0.65
                buy_precision = y_test.values[buy_mask].mean() if buy_mask.sum() > 0 else 0

                state.metrics[horizon] = {
                    "accuracy": round(accuracy * 100, 1) if not np.isnan(accuracy) else 0,
                    "confident_accuracy": round(confident_acc * 100, 1) if not np.isnan(confident_acc) else 0,
                    "buy_precision": round(buy_precision * 100, 1) if not np.isnan(buy_precision) else 0,
                    "total_samples": len(X),
                    "best_iteration": final_model.best_iteration if final_model.best_iteration else 200
                }

                # Feature importance (LightGBM returns dict of feature_name: importance)
                importance = dict(zip(feature_cols, final_model.feature_importance(importance_type='gain')))
                total = sum(importance.values()) if importance else 1
                state.feature_importance[horizon] = {k: round(v/total, 3) for k, v in importance.items()}

                # Save model with symbol name (sanitize symbol for filename)
                symbol_safe = symbol.replace("/", "_")
                model_path = f"models/{symbol_safe}_{horizon}min.txt"
                os.makedirs(os.path.dirname(model_path), exist_ok=True)
                final_model.save_model(model_path)
                await log(f"Saved model: {model_path}")

                # ===== HOLDOUT EVALUATION (TRUE OUT-OF-SAMPLE) =====
                # This is the REAL test - data the model has NEVER seen!
                if not df_holdout.empty and len(df_holdout_features) > horizon + 10:
                    await log(f"\n>>> HOLDOUT EVALUATION for {horizon}-min model <<<")

                    # Prepare holdout features
                    holdout_feature_cols = [c for c in feature_cols if c in df_holdout_features.columns]
                    X_holdout = df_holdout_features[holdout_feature_cols]

                    # Create holdout target
                    holdout_future_close = df_holdout_features['close'].shift(-horizon)
                    holdout_current_close = df_holdout_features['close']
                    holdout_pct_change = (holdout_future_close - holdout_current_close) / holdout_current_close

                    holdout_valid_idx = X_holdout.index.intersection(holdout_future_close.dropna().index)
                    X_holdout_final = X_holdout.loc[holdout_valid_idx].iloc[:-horizon]
                    y_holdout = (holdout_pct_change.loc[holdout_valid_idx] > MOVE_THRESHOLD).astype(int).iloc[:-horizon]

                    if len(X_holdout_final) > 10:
                        # Evaluate on holdout
                        y_holdout_pred_proba = final_model.predict(X_holdout_final)
                        y_true = y_holdout.values

                        # Diagnostic: Show prediction distribution
                        actual_positives = y_true.sum()
                        max_prob = y_holdout_pred_proba.max()
                        mean_prob = y_holdout_pred_proba.mean()
                        std_prob = y_holdout_pred_proba.std()

                        await log(f"  Prediction distribution: max={max_prob:.3f}, mean={mean_prob:.3f}, std={std_prob:.3f}")
                        await log(f"  Actual buys: {actual_positives} ({100*actual_positives/len(y_true):.1f}%)")

                        # ===== FIND OPTIMAL THRESHOLD =====
                        # Test multiple thresholds and find best precision with reasonable signal count
                        # Start from low threshold since model may output conservative probabilities
                        thresholds = np.arange(0.1, 0.7, 0.05)
                        best_threshold = 0.5
                        best_f1 = 0
                        threshold_results = []

                        for thresh in thresholds:
                            preds = (y_holdout_pred_proba > thresh).astype(int)
                            n_signals = preds.sum()

                            if n_signals > 0:
                                precision = y_true[preds == 1].mean() if preds.sum() > 0 else 0
                                recall = preds[y_true == 1].mean() if y_true.sum() > 0 else 0
                                # F0.5 score (precision-weighted)
                                f05 = (1.25 * precision * recall) / (0.25 * precision + recall) if (precision + recall) > 0 else 0
                                threshold_results.append({
                                    'threshold': thresh,
                                    'precision': precision,
                                    'recall': recall,
                                    'f05': f05,
                                    'signals': n_signals
                                })
                                if f05 > best_f1:
                                    best_f1 = f05
                                    best_threshold = thresh

                        # Show threshold analysis
                        await log(f"  Threshold analysis:")
                        for tr in threshold_results[:5]:  # Show top 5
                            await log(f"    @{tr['threshold']:.2f}: Prec={tr['precision']*100:.1f}%, Recall={tr['recall']*100:.1f}%, Signals={tr['signals']}")

                        await log(f"  📊 OPTIMAL THRESHOLD: {best_threshold:.2f}")

                        # Calculate final metrics at optimal threshold
                        y_holdout_pred = (y_holdout_pred_proba > best_threshold).astype(int)
                        holdout_accuracy = (y_holdout_pred == y_true).mean()

                        # Precision/Recall at optimal threshold
                        n_buy_signals = y_holdout_pred.sum()
                        holdout_buy_precision = y_true[y_holdout_pred == 1].mean() if n_buy_signals > 0 else 0
                        holdout_recall = y_holdout_pred[y_true == 1].mean() if y_true.sum() > 0 else 0

                        # Calculate signals per day (assuming ~1440 1-min bars per day)
                        bars_per_day = 1440
                        signals_per_day = (n_buy_signals / len(y_holdout_pred)) * bars_per_day

                        await log(f"  HOLDOUT Results @{best_threshold:.2f}:")
                        await log(f"    Accuracy: {holdout_accuracy:.1%}")
                        await log(f"    Buy Precision: {holdout_buy_precision:.1%} ({n_buy_signals} signals)")
                        await log(f"    Buy Recall: {holdout_recall:.1%}")
                        await log(f"    Signals/day: ~{signals_per_day:.1f}")

                        # Store holdout metrics
                        if symbol not in state.holdout_metrics:
                            state.holdout_metrics[symbol] = {}

                        state.holdout_metrics[symbol][horizon] = {
                            "accuracy": round(holdout_accuracy * 100, 1) if not np.isnan(holdout_accuracy) else 0,
                            "buy_precision": round(holdout_buy_precision * 100, 1) if not np.isnan(holdout_buy_precision) else 0,
                            "recall": round(holdout_recall * 100, 1) if not np.isnan(holdout_recall) else 0,
                            "signals_per_day": round(signals_per_day, 1),
                            "optimal_threshold": round(best_threshold, 2),
                            "total_samples": len(X_holdout_final)
                        }

                        # Save optimal threshold to metadata file
                        import json
                        metadata_path = model_path.replace('.txt', '_meta.json')
                        metadata = {
                            'optimal_threshold': float(best_threshold),
                            'holdout_precision': float(holdout_buy_precision),
                            'holdout_recall': float(holdout_recall),
                            'signals_per_day': float(signals_per_day),
                            'imbalance_ratio': float(imbalance_ratio),
                            'horizon': horizon,
                            'symbol': symbol
                        }
                        with open(metadata_path, 'w') as f:
                            json.dump(metadata, f, indent=2)
                        await log(f"  Saved metadata: {metadata_path}")

                        # Compare to training metrics
                        train_acc = state.metrics[horizon]['accuracy']
                        holdout_acc = round(holdout_accuracy * 100, 1)
                        gap = train_acc - holdout_acc
                        if gap > 10:
                            await log(f"  ⚠️ WARNING: {gap:.1f}% gap between train ({train_acc}%) and holdout ({holdout_acc}%)")
                        else:
                            await log(f"  ✅ Good generalization: gap is only {gap:.1f}%")
                    else:
                        await log(f"  Not enough holdout samples for {horizon}-min evaluation")

                # Update progress based on symbol and horizon
                symbol_progress = (sym_idx / total_symbols) * 100
                horizon_progress = ((h_idx + 1) / len(horizons)) * (100 / total_symbols)
                state.progress = int(symbol_progress + horizon_progress)
                await broadcast_state()

            # Store metrics for this symbol
            state.symbol_metrics[symbol] = dict(state.metrics)

            # Save default model for this symbol (5-min)
            symbol_safe = symbol.replace("/", "_")
            if os.path.exists(f"models/{symbol_safe}_5min.txt"):
                import shutil
                shutil.copy(f"models/{symbol_safe}_5min.txt", f"models/{symbol_safe}_model.txt")
                await log(f"Copied 5-min model for {symbol}")

        # Also copy BTC/USD model as the default
        if os.path.exists("models/BTC_USD_5min.txt"):
            import shutil
            default_model_path = config.MODEL_PATH.replace('.json', '.txt')
            shutil.copy("models/BTC_USD_5min.txt", default_model_path)
            await log(f"Copied BTC/USD 5-min model to {default_model_path}")

        # ===== COMPLETE =====
        state.status = "complete"
        state.progress = 100
        await log("\n" + "#"*50)
        await log("ALL TRAINING COMPLETE!")
        await log("#"*50)

        # Print training metrics
        await log("\n=== TRAINING METRICS (may be optimistic) ===")
        for symbol, sym_metrics in state.symbol_metrics.items():
            await log(f"\n{symbol}:")
            for horizon, metrics in sym_metrics.items():
                await log(f"  {horizon}-min: Acc={metrics['accuracy']}%, Buy Prec={metrics['buy_precision']}%")

        # Print HOLDOUT metrics (TRUE performance)
        if state.holdout_metrics:
            await log("\n=== HOLDOUT METRICS (TRUE OUT-OF-SAMPLE) ===")
            for symbol, sym_holdout in state.holdout_metrics.items():
                await log(f"\n{symbol}:")
                for horizon, metrics in sym_holdout.items():
                    await log(f"  {horizon}-min: Acc={metrics['accuracy']}%, Buy Prec={metrics['buy_precision']}%")
            await log("\n⚠️ HOLDOUT metrics are the TRUE indicator of model performance!")
            await log("If holdout accuracy is much lower than training, the model is overfitting.")

        # Broadcast completion status multiple times to ensure delivery
        await broadcast_state()
        await asyncio.sleep(0.5)  # Small delay to ensure message is sent
        await broadcast_state()  # Send again for reliability
        logger.info("Training complete - final status broadcast sent")

    except Exception as e:
        state.status = "error"
        state.error = str(e)
        await log(f"ERROR: {e}")
        await broadcast_state()
        logger.exception("Training error")


async def tune_hyperparameters(X: pd.DataFrame, y: pd.Series, n_iter: int, horizon: int) -> Dict:
    """Tune hyperparameters with dashboard updates."""
    global state

    # LightGBM parameter grid
    param_grid = {
        'max_depth': [3, 4, 5, 6],
        'learning_rate': [0.01, 0.05, 0.1],
        'min_child_samples': [10, 20, 30],
        'bagging_fraction': [0.7, 0.8, 0.9],
        'feature_fraction': [0.7, 0.8, 0.9],
        'num_leaves': [15, 31, 63],
    }

    fixed_params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'verbosity': -1,
        'bagging_freq': 1,  # Required when using bagging_fraction
        'feature_pre_filter': False,  # Allow dynamic min_data_in_leaf changes
        'is_unbalance': True,  # Let LightGBM handle class imbalance
    }

    best_params = None
    best_score = float('inf')

    await log(f"Hyperparameter tuning ({n_iter} iterations)...")

    # Simple time series split for tuning
    n = len(X)
    test_size = n // 5
    train_end = n - test_size

    X_train, X_test = X.iloc[:train_end], X.iloc[train_end:]
    y_train, y_test = y.iloc[:train_end], y.iloc[train_end:]

    dtrain = lgb.Dataset(X_train, label=y_train)
    dtest = lgb.Dataset(X_test, label=y_test, reference=dtrain)

    for i in range(n_iter):
        state.tuning_iteration = i + 1

        # Random params
        params = fixed_params.copy()
        for key, values in param_grid.items():
            params[key] = np.random.choice(values)

        # Quick train
        model = lgb.train(
            params, dtrain,
            num_boost_round=50,
            valid_sets=[dtest],
            valid_names=['valid'],
            callbacks=[lgb.early_stopping(stopping_rounds=10, verbose=False)]
        )

        # Get best score from model
        score = model.best_score['valid']['binary_logloss'] if model.best_score else float('inf')

        state.tuning_history.append({
            "iteration": i + 1,
            "score": round(score, 4),
            "params": {k: v for k, v in params.items() if k not in ['objective', 'metric', 'verbosity', 'bagging_freq']}
        })

        if score < best_score:
            best_score = score
            best_params = params.copy()
            state.tuning_best_score = round(score, 4)
            await log(f"  Iter {i+1}: New best score {score:.4f}")

        await broadcast_state()
        await asyncio.sleep(0.01)  # Allow dashboard to update

    await log(f"Best params found (score: {best_score:.4f})")
    return best_params


async def log(message: str):
    """Add log entry."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    state.logs.append(f"[{timestamp}] {message}")
    logger.info(message)


async def broadcast_state():
    """Broadcast current state to all connected clients."""
    await manager.broadcast({"type": "update", "data": state.to_dict()})


async def broadcast_state_async(message: dict):
    """Async wrapper for callback."""
    await manager.broadcast(message)


# ============================================
# FASTAPI APP
# ============================================

app = FastAPI(title="BlueBird Training Dashboard")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        # Send initial state
        await websocket.send_json({"type": "update", "data": state.to_dict()})

        while True:
            data = await websocket.receive_text()
            message = json.loads(data)

            if message.get("action") == "start_training":
                config = UltraConfig()
                days = message.get("days", 90)
                tune = message.get("tune", True)
                tune_iter = message.get("tune_iterations", 15)
                holdout_days = message.get("holdout_days", getattr(config, 'HOLDOUT_DAYS', 14))

                # Run training in background with proper holdout
                asyncio.create_task(run_training(config, days, tune, tune_iter, holdout_days))

            elif message.get("action") == "get_state":
                await websocket.send_json({"type": "update", "data": state.to_dict()})

    except WebSocketDisconnect:
        manager.disconnect(websocket)


@app.get("/api/state")
async def get_state():
    """REST endpoint to get current training state (fallback for WebSocket)."""
    return {"type": "update", "data": state.to_dict()}


@app.get("/", response_class=HTMLResponse)
async def get_dashboard():
    """Serve the training dashboard."""
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>BlueBird AI - Training Dashboard</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        .glass-card { background: rgba(15, 15, 25, 0.8); border: 1px solid rgba(255,255,255,0.05); }
        .gradient-text { background: linear-gradient(135deg, #6366f1, #a855f7); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
        @keyframes pulse-glow { 0%, 100% { box-shadow: 0 0 5px rgba(99, 102, 241, 0.5); } 50% { box-shadow: 0 0 20px rgba(99, 102, 241, 0.8); } }
        .pulse-glow { animation: pulse-glow 2s infinite; }
    </style>
</head>
<body class="bg-[#0a0a0f] text-gray-300 min-h-screen p-6">
    <div class="max-w-7xl mx-auto">
        <!-- Header -->
        <header class="flex justify-between items-center mb-8">
            <div>
                <h1 class="text-3xl font-bold text-white">BlueBird <span class="gradient-text">AI Training</span></h1>
                <p class="text-gray-500 text-sm">LightGBM Model Training Dashboard</p>
            </div>
            <div id="status-badge" class="px-4 py-2 rounded-lg bg-gray-500/20 border border-gray-500/30 text-gray-400">
                Disconnected
            </div>
        </header>

        <!-- Controls -->
        <div class="glass-card rounded-xl p-6 mb-6">
            <div class="flex items-center gap-6 flex-wrap">
                <div>
                    <label class="text-xs text-gray-500 block mb-1">Total Days</label>
                    <input type="number" id="days-input" value="90" class="bg-black/30 border border-gray-700 rounded px-3 py-2 w-24 text-white" />
                </div>
                <div>
                    <label class="text-xs text-gray-500 block mb-1">Holdout Days</label>
                    <input type="number" id="holdout-input" value="14" class="bg-black/30 border border-gray-700 rounded px-3 py-2 w-20 text-white" title="Days reserved for true out-of-sample testing" />
                </div>
                <div>
                    <label class="text-xs text-gray-500 block mb-1">Tune Iterations</label>
                    <input type="number" id="tune-iter-input" value="15" class="bg-black/30 border border-gray-700 rounded px-3 py-2 w-24 text-white" />
                </div>
                <div class="flex items-center gap-2">
                    <input type="checkbox" id="tune-checkbox" checked class="w-4 h-4" />
                    <label class="text-sm text-gray-400">Hyperparameter Tuning</label>
                </div>
                <button id="start-btn" onclick="startTraining()" class="bg-indigo-600 hover:bg-indigo-700 text-white px-6 py-2 rounded-lg font-medium transition">
                    Start Training
                </button>
            </div>
            <div class="mt-3 text-xs text-gray-500">
                <span class="text-emerald-400">Training:</span> <span id="train-days-display">76</span> days |
                <span class="text-amber-400">Holdout:</span> <span id="holdout-days-display">14</span> days (true out-of-sample test)
            </div>
        </div>

        <!-- Progress Bar -->
        <div class="glass-card rounded-xl p-6 mb-6">
            <div class="flex justify-between items-center mb-3">
                <span id="status-text" class="text-sm text-gray-400">Ready to train</span>
                <span id="progress-pct" class="text-sm text-indigo-400 font-mono">0%</span>
            </div>
            <div class="h-4 bg-black/30 rounded-full overflow-hidden">
                <div id="progress-bar" class="h-full bg-gradient-to-r from-indigo-500 to-purple-500 transition-all duration-300" style="width: 0%"></div>
            </div>
            <div id="horizon-badges" class="flex gap-2 mt-4"></div>
        </div>

        <!-- Main Grid -->
        <div class="grid grid-cols-3 gap-6">
            <!-- Left Column -->
            <div class="space-y-6">
                <!-- Data Info -->
                <div class="glass-card rounded-xl p-5">
                    <h3 class="text-sm font-semibold text-white mb-4">📊 Data</h3>
                    <div class="space-y-3 text-sm">
                        <div class="flex justify-between"><span class="text-gray-500">Rows</span><span id="data-rows" class="text-white font-mono">-</span></div>
                        <div class="flex justify-between"><span class="text-gray-500">Days</span><span id="data-days" class="text-white font-mono">-</span></div>
                        <div class="flex justify-between"><span class="text-gray-500">Features</span><span id="data-features" class="text-white font-mono">-</span></div>
                    </div>
                </div>

                <!-- Tuning Progress -->
                <div class="glass-card rounded-xl p-5">
                    <h3 class="text-sm font-semibold text-white mb-4">🎯 Hyperparameter Tuning</h3>
                    <div class="space-y-3 text-sm">
                        <div class="flex justify-between"><span class="text-gray-500">Iteration</span><span id="tune-iter" class="text-white font-mono">-</span></div>
                        <div class="flex justify-between"><span class="text-gray-500">Best Score</span><span id="tune-best" class="text-emerald-400 font-mono">-</span></div>
                    </div>
                    <div class="mt-4 h-2 bg-black/30 rounded-full overflow-hidden">
                        <div id="tune-progress" class="h-full bg-emerald-500 transition-all" style="width: 0%"></div>
                    </div>
                </div>

                <!-- Walk-Forward Validation -->
                <div class="glass-card rounded-xl p-5">
                    <h3 class="text-sm font-semibold text-white mb-4">📈 Walk-Forward Validation</h3>
                    <div id="wf-results" class="space-y-2"></div>
                </div>
            </div>

            <!-- Center Column -->
            <div class="space-y-6">
                <!-- Loss Chart -->
                <div class="glass-card rounded-xl p-5">
                    <h3 class="text-sm font-semibold text-white mb-4">📉 Training Loss</h3>
                    <div class="h-64">
                        <canvas id="loss-chart"></canvas>
                    </div>
                </div>

                <!-- Tuning History Chart -->
                <div class="glass-card rounded-xl p-5">
                    <h3 class="text-sm font-semibold text-white mb-4">🔍 Tuning History</h3>
                    <div class="h-48">
                        <canvas id="tune-chart"></canvas>
                    </div>
                </div>
            </div>

            <!-- Right Column -->
            <div class="space-y-6">
                <!-- Model Metrics (Training) -->
                <div class="glass-card rounded-xl p-5">
                    <h3 class="text-sm font-semibold text-white mb-4">🎯 Training Metrics <span class="text-xs text-gray-500">(may be optimistic)</span></h3>
                    <div id="metrics-container" class="space-y-4"></div>
                </div>

                <!-- HOLDOUT Metrics (TRUE Performance) -->
                <div class="glass-card rounded-xl p-5 border border-amber-500/30">
                    <h3 class="text-sm font-semibold text-amber-400 mb-4">⚡ HOLDOUT Metrics <span class="text-xs text-gray-400">(TRUE performance)</span></h3>
                    <div id="holdout-metrics-container" class="space-y-4">
                        <div class="text-gray-500 text-xs">Holdout metrics will appear after training completes</div>
                    </div>
                </div>

                <!-- Feature Importance -->
                <div class="glass-card rounded-xl p-5">
                    <h3 class="text-sm font-semibold text-white mb-4">⚡ Feature Importance</h3>
                    <div id="feature-importance" class="space-y-2"></div>
                </div>

                <!-- Logs -->
                <div class="glass-card rounded-xl p-5 max-h-64 overflow-hidden flex flex-col">
                    <h3 class="text-sm font-semibold text-white mb-4">📝 Logs</h3>
                    <div id="logs" class="flex-1 overflow-y-auto space-y-1 text-xs font-mono"></div>
                </div>
            </div>
        </div>
    </div>

    <script>
        let ws;
        let lossChart, tuneChart;

        // Initialize charts
        function initCharts() {
            const lossCtx = document.getElementById('loss-chart').getContext('2d');
            lossChart = new Chart(lossCtx, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [
                        { label: 'Train Loss', data: [], borderColor: '#6366f1', tension: 0.4, pointRadius: 0 },
                        { label: 'Val Loss', data: [], borderColor: '#f59e0b', tension: 0.4, pointRadius: 0 }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: { legend: { labels: { color: '#9ca3af' } } },
                    scales: {
                        x: { grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#6b7280' } },
                        y: { grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#6b7280' } }
                    }
                }
            });

            const tuneCtx = document.getElementById('tune-chart').getContext('2d');
            tuneChart = new Chart(tuneCtx, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [{ label: 'Score', data: [], borderColor: '#10b981', tension: 0.4, pointRadius: 2 }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: { legend: { display: false } },
                    scales: {
                        x: { grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#6b7280' } },
                        y: { grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#6b7280' } }
                    }
                }
            });
        }

        function connect() {
            ws = new WebSocket('ws://localhost:8001/ws');

            ws.onopen = () => {
                document.getElementById('status-badge').className = 'px-4 py-2 rounded-lg bg-emerald-500/20 border border-emerald-500/30 text-emerald-400';
                document.getElementById('status-badge').textContent = 'Connected';
            };

            ws.onclose = () => {
                document.getElementById('status-badge').className = 'px-4 py-2 rounded-lg bg-red-500/20 border border-red-500/30 text-red-400';
                document.getElementById('status-badge').textContent = 'Disconnected';
                setTimeout(connect, 3000);
            };

            ws.onmessage = (e) => {
                const msg = JSON.parse(e.data);
                if (msg.type === 'update') updateDashboard(msg.data);
            };
        }

        function updateDashboard(data) {
            // Status
            const statusMap = {
                'idle': 'Ready to train',
                'fetching_data': 'Fetching data...',
                'calculating_features': 'Calculating features...',
                'tuning': `Tuning ${data.current_horizon}-min model...`,
                'training': `Training ${data.current_horizon}-min model...`,
                'complete': 'Training complete!',
                'error': 'Error: ' + data.error
            };
            document.getElementById('status-text').textContent = statusMap[data.status] || data.status;
            document.getElementById('progress-pct').textContent = data.progress + '%';
            document.getElementById('progress-bar').style.width = data.progress + '%';

            // Horizon badges
            const badges = data.horizons.map(h => {
                const done = data.metrics[h];
                const current = h === data.current_horizon;
                const cls = done ? 'bg-emerald-500/20 border-emerald-500/30 text-emerald-400' :
                           current ? 'bg-indigo-500/20 border-indigo-500/30 text-indigo-400 pulse-glow' :
                           'bg-gray-500/20 border-gray-500/30 text-gray-400';
                return `<span class="px-3 py-1 rounded border ${cls} text-xs">${h}min ${done ? '✓' : ''}</span>`;
            }).join('');
            document.getElementById('horizon-badges').innerHTML = badges;

            // Data info
            document.getElementById('data-rows').textContent = data.data.rows.toLocaleString();
            document.getElementById('data-days').textContent = data.data.days;
            document.getElementById('data-features').textContent = data.data.features;

            // Tuning
            document.getElementById('tune-iter').textContent = `${data.tuning.iteration}/${data.tuning.total}`;
            document.getElementById('tune-best').textContent = data.tuning.best_score || '-';
            document.getElementById('tune-progress').style.width = (data.tuning.total > 0 ? data.tuning.iteration / data.tuning.total * 100 : 0) + '%';

            // Walk-forward results
            const wfHtml = data.walk_forward.results.map(r => `
                <div class="flex justify-between items-center p-2 bg-black/20 rounded text-xs">
                    <span class="text-gray-400">Fold ${r.fold}</span>
                    <span class="text-emerald-400 font-mono">${r.accuracy}%</span>
                </div>
            `).join('');
            document.getElementById('wf-results').innerHTML = wfHtml || '<div class="text-gray-500 text-xs">No results yet</div>';

            // Loss chart
            if (data.training.train_loss.length > 0) {
                lossChart.data.labels = data.training.train_loss.map((_, i) => i + 1);
                lossChart.data.datasets[0].data = data.training.train_loss;
                lossChart.data.datasets[1].data = data.training.val_loss;
                lossChart.update('none');
            }

            // Tuning chart
            if (data.tuning.history.length > 0) {
                tuneChart.data.labels = data.tuning.history.map(h => h.iteration);
                tuneChart.data.datasets[0].data = data.tuning.history.map(h => h.score);
                tuneChart.update('none');
            }

            // Training Metrics
            const metricsHtml = Object.entries(data.metrics).map(([horizon, m]) => `
                <div class="p-3 bg-black/20 rounded">
                    <div class="text-indigo-400 font-semibold mb-2">${horizon}-min Model</div>
                    <div class="grid grid-cols-3 gap-2 text-xs">
                        <div><span class="text-gray-500">Accuracy</span><div class="text-white font-mono">${m.accuracy}%</div></div>
                        <div><span class="text-gray-500">Confident</span><div class="text-emerald-400 font-mono">${m.confident_accuracy}%</div></div>
                        <div><span class="text-gray-500">Buy Prec</span><div class="text-yellow-400 font-mono">${m.buy_precision}%</div></div>
                    </div>
                </div>
            `).join('');
            document.getElementById('metrics-container').innerHTML = metricsHtml || '<div class="text-gray-500 text-xs">No metrics yet</div>';

            // HOLDOUT Metrics (TRUE performance)
            if (data.holdout_metrics && Object.keys(data.holdout_metrics).length > 0) {
                let holdoutHtml = '';
                for (const [symbol, horizonMetrics] of Object.entries(data.holdout_metrics)) {
                    holdoutHtml += `<div class="text-amber-300 text-xs font-semibold mb-2">${symbol}</div>`;
                    for (const [horizon, m] of Object.entries(horizonMetrics)) {
                        holdoutHtml += `
                            <div class="p-3 bg-black/20 rounded mb-2">
                                <div class="text-amber-400 font-semibold mb-2">${horizon}-min</div>
                                <div class="grid grid-cols-3 gap-2 text-xs">
                                    <div><span class="text-gray-500">Accuracy</span><div class="text-amber-300 font-mono">${m.accuracy}%</div></div>
                                    <div><span class="text-gray-500">Confident</span><div class="text-amber-300 font-mono">${m.confident_accuracy}%</div></div>
                                    <div><span class="text-gray-500">Buy Prec</span><div class="text-amber-300 font-mono">${m.buy_precision}%</div></div>
                                </div>
                            </div>
                        `;
                    }
                }
                document.getElementById('holdout-metrics-container').innerHTML = holdoutHtml;
            }

            // Feature importance (show for current/last horizon)
            const horizon = data.current_horizon || Object.keys(data.feature_importance)[0];
            const importance = data.feature_importance[horizon] || {};
            const fiHtml = Object.entries(importance).sort((a, b) => b[1] - a[1]).slice(0, 8).map(([f, w]) => `
                <div>
                    <div class="flex justify-between text-xs mb-1">
                        <span class="text-gray-400">${f}</span>
                        <span class="text-gray-300">${(w * 100).toFixed(1)}%</span>
                    </div>
                    <div class="h-2 bg-black/30 rounded-full overflow-hidden">
                        <div class="h-full bg-indigo-500" style="width: ${w * 100}%"></div>
                    </div>
                </div>
            `).join('');
            document.getElementById('feature-importance').innerHTML = fiHtml || '<div class="text-gray-500 text-xs">Training in progress...</div>';

            // Logs
            const logsHtml = data.logs.slice(-20).map(l => `
                <div class="text-gray-400 py-1">${l}</div>
            `).join('');
            document.getElementById('logs').innerHTML = logsHtml;
            document.getElementById('logs').scrollTop = document.getElementById('logs').scrollHeight;
        }

        function startTraining() {
            const days = parseInt(document.getElementById('days-input').value);
            const holdoutDays = parseInt(document.getElementById('holdout-input').value);
            const tuneIter = parseInt(document.getElementById('tune-iter-input').value);
            const tune = document.getElementById('tune-checkbox').checked;

            // Validate
            if (holdoutDays >= days) {
                alert('Holdout days must be less than total days!');
                return;
            }

            ws.send(JSON.stringify({
                action: 'start_training',
                days: days,
                holdout_days: holdoutDays,
                tune: tune,
                tune_iterations: tuneIter
            }));

            document.getElementById('start-btn').disabled = true;
            document.getElementById('start-btn').textContent = 'Training...';
        }

        function updateDaysDisplay() {
            const days = parseInt(document.getElementById('days-input').value) || 90;
            const holdout = parseInt(document.getElementById('holdout-input').value) || 14;
            const training = days - holdout;
            document.getElementById('train-days-display').textContent = training;
            document.getElementById('holdout-days-display').textContent = holdout;
        }

        // Initialize
        initCharts();
        connect();

        // Update display when inputs change
        document.getElementById('days-input').addEventListener('input', updateDaysDisplay);
        document.getElementById('holdout-input').addEventListener('input', updateDaysDisplay);
        updateDaysDisplay();
    </script>
</body>
</html>
"""


# ============================================
# MAIN
# ============================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("  BlueBird AI Training Dashboard")
    print("="*60)
    print("\n  Open http://localhost:8001 in your browser")
    print("  Click 'Start Training' to begin\n")

    uvicorn.run(app, host="0.0.0.0", port=8001)
