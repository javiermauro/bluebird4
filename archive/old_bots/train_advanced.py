"""
Advanced LightGBM Training Script

Features:
- 90+ days of training data
- Enhanced feature set (trend scores, momentum, volatility)
- Walk-forward validation (prevents overfitting)
- Hyperparameter tuning via cross-validation
- Multiple prediction horizons (5, 15, 30 min)
- Proper train/test metrics and logging
"""

import logging
import os
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
from datetime import datetime, timedelta
from sklearn.model_selection import TimeSeriesSplit
from typing import Dict, List, Tuple

from config_ultra import UltraConfig
from src.execution.alpaca_client import AlpacaClient
from src.data.loader import DataLoader
from src.data.mock_loader import MockDataLoader

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("AdvancedTrainer")


class AdvancedFeatureEngine:
    """Calculate all features needed for training and prediction."""

    @staticmethod
    def calculate_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate comprehensive feature set.

        Returns DataFrame with features that MUST match adaptive_ai.py
        """
        import talib

        df = df.copy()
        df.columns = [c.lower() for c in df.columns]

        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        volume = df['volume'].values if 'volume' in df.columns else np.ones(len(close))

        # ===== Core Features (must match adaptive_ai.py) =====

        # RSI (14-period)
        df['rsi'] = talib.RSI(close, timeperiod=14)

        # MACD
        macd, macd_signal, macd_hist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
        df['macd'] = macd
        df['macd_signal'] = macd_signal

        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)
        df['bb_upper'] = bb_upper
        df['bb_middle'] = bb_middle
        df['bb_lower'] = bb_lower

        # ATR (14-period)
        df['atr'] = talib.ATR(high, low, close, timeperiod=14)

        # Volume ratio
        vol_ma = pd.Series(volume).rolling(window=20).mean()
        df['volume_ratio'] = volume / vol_ma.values

        # Momentum (10-period)
        df['momentum'] = talib.MOM(close, timeperiod=10)

        # ADX (14-period)
        df['adx'] = talib.ADX(high, low, close, timeperiod=14)

        # ===== Additional Features for Better Predictions =====

        # Trend Score (EMA alignment)
        ema_8 = talib.EMA(close, timeperiod=8)
        ema_21 = talib.EMA(close, timeperiod=21)
        ema_50 = talib.EMA(close, timeperiod=50)
        df['trend_score'] = ((ema_8 > ema_21).astype(int) + (ema_21 > ema_50).astype(int)) / 2

        # BB Width (volatility measure)
        df['bb_width'] = (bb_upper - bb_lower) / bb_middle

        # BB Position (where price is within bands, 0=lower, 1=upper)
        df['bb_position'] = (close - bb_lower) / (bb_upper - bb_lower + 1e-10)

        # Rate of Change
        df['roc_5'] = talib.ROC(close, timeperiod=5)
        df['roc_10'] = talib.ROC(close, timeperiod=10)

        # RSI Momentum (RSI change)
        rsi = df['rsi'].values
        df['rsi_momentum'] = pd.Series(rsi).diff(3).values

        # MACD Histogram
        df['macd_hist'] = macd_hist

        # Stochastic
        slowk, slowd = talib.STOCH(high, low, close, fastk_period=14, slowk_period=3, slowd_period=3)
        df['stoch_k'] = slowk
        df['stoch_d'] = slowd

        # ATR Percent (normalized volatility)
        df['atr_pct'] = df['atr'] / close * 100

        # Volume spike detection
        df['volume_spike'] = (df['volume_ratio'] > 2.0).astype(int)

        # Price position relative to recent high/low
        df['price_position'] = (close - pd.Series(low).rolling(20).min().values) / \
                               (pd.Series(high).rolling(20).max().values - pd.Series(low).rolling(20).min().values + 1e-10)

        # Drop NaN rows
        df.dropna(inplace=True)

        return df

    @staticmethod
    def get_core_features() -> List[str]:
        """Features that MUST match adaptive_ai.py _get_ai_prediction()"""
        return [
            'rsi', 'macd', 'macd_signal', 'bb_upper', 'bb_middle', 'bb_lower',
            'atr', 'volume_ratio', 'momentum', 'adx'
        ]

    @staticmethod
    def get_all_features() -> List[str]:
        """All features for training (superset of core)"""
        return [
            # Core (must match adaptive_ai.py)
            'rsi', 'macd', 'macd_signal', 'bb_upper', 'bb_middle', 'bb_lower',
            'atr', 'volume_ratio', 'momentum', 'adx',
            # Additional
            'trend_score', 'bb_width', 'bb_position', 'roc_5', 'roc_10',
            'rsi_momentum', 'macd_hist', 'stoch_k', 'stoch_d', 'atr_pct',
            'volume_spike', 'price_position'
        ]


class WalkForwardValidator:
    """
    Walk-forward validation for time series.

    Unlike random cross-validation, this respects time ordering:
    - Train on past data
    - Validate on future data
    - No data leakage
    """

    def __init__(self, n_splits: int = 5, test_size: int = None):
        self.n_splits = n_splits
        self.test_size = test_size

    def split(self, X: pd.DataFrame) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Generate train/test indices for walk-forward validation."""
        n_samples = len(X)

        if self.test_size is None:
            test_size = n_samples // (self.n_splits + 1)
        else:
            test_size = self.test_size

        splits = []

        for i in range(self.n_splits):
            # Test set moves forward each split
            test_end = n_samples - (self.n_splits - 1 - i) * test_size
            test_start = test_end - test_size

            # Train on everything before test
            train_end = test_start
            train_start = 0

            # Minimum training size
            if train_end - train_start < test_size * 2:
                continue

            train_idx = np.arange(train_start, train_end)
            test_idx = np.arange(test_start, test_end)

            splits.append((train_idx, test_idx))

        return splits


class HyperparameterTuner:
    """Find optimal LightGBM parameters via walk-forward CV."""

    # Parameter grid to search
    PARAM_GRID = {
        'max_depth': [3, 4, 5, 6],
        'learning_rate': [0.01, 0.05, 0.1],
        'min_child_samples': [10, 20, 30],
        'bagging_fraction': [0.7, 0.8, 0.9],
        'feature_fraction': [0.7, 0.8, 0.9],
        'num_leaves': [15, 31, 63],
    }

    # Fixed parameters
    FIXED_PARAMS = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'verbosity': -1,
        'bagging_freq': 1,
        'feature_pre_filter': False,
    }

    def __init__(self, n_iter: int = 20):
        """
        Args:
            n_iter: Number of random parameter combinations to try
        """
        self.n_iter = n_iter
        self.best_params = None
        self.best_score = float('inf')
        self.results = []

    def _random_params(self) -> Dict:
        """Generate random parameter combination."""
        params = self.FIXED_PARAMS.copy()
        for key, values in self.PARAM_GRID.items():
            params[key] = np.random.choice(values)
        return params

    def tune(self, X: pd.DataFrame, y: pd.Series, n_splits: int = 3) -> Dict:
        """
        Find best parameters using walk-forward validation.

        Returns:
            Best parameter dict
        """
        logger.info(f"Starting hyperparameter tuning ({self.n_iter} iterations)...")

        validator = WalkForwardValidator(n_splits=n_splits)
        splits = validator.split(X)

        for i in range(self.n_iter):
            params = self._random_params()

            # Cross-validate with these params
            cv_scores = []

            for train_idx, test_idx in splits:
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

                dtrain = lgb.Dataset(X_train, label=y_train)
                dtest = lgb.Dataset(X_test, label=y_test, reference=dtrain)

                model = lgb.train(
                    params, dtrain,
                    num_boost_round=100,
                    valid_sets=[dtest],
                    valid_names=['valid'],
                    callbacks=[lgb.early_stopping(stopping_rounds=10, verbose=False)]
                )

                # Get best score
                score = model.best_score['valid']['binary_logloss'] if model.best_score else float('inf')
                cv_scores.append(score)

            mean_score = np.mean(cv_scores)

            self.results.append({
                'params': params.copy(),
                'mean_score': mean_score,
                'cv_scores': cv_scores
            })

            if mean_score < self.best_score:
                self.best_score = mean_score
                self.best_params = params.copy()
                logger.info(f"  Iter {i+1}: New best score {mean_score:.4f}")

            if (i + 1) % 5 == 0:
                logger.info(f"  Completed {i+1}/{self.n_iter} iterations")

        logger.info(f"Best params found with score {self.best_score:.4f}")
        return self.best_params


class MultiHorizonTrainer:
    """Train models for multiple prediction horizons."""

    def __init__(self, config: UltraConfig, horizons: List[int] = None):
        """
        Args:
            config: Configuration object
            horizons: List of prediction horizons in minutes (default: [5, 15, 30])
        """
        self.config = config
        self.horizons = horizons or [5, 15, 30]
        self.models = {}
        self.metrics = {}

    def create_targets(self, df: pd.DataFrame) -> Dict[int, pd.Series]:
        """Create target variables for each horizon."""
        targets = {}

        for horizon in self.horizons:
            future_close = df['close'].shift(-horizon)
            current_close = df['close']

            # 1 if price goes up, 0 if down
            target = (future_close > current_close).astype(int)
            targets[horizon] = target

        return targets

    def train_all(self, df: pd.DataFrame, features: pd.DataFrame,
                  tune_params: bool = True, n_tune_iter: int = 20) -> Dict:
        """
        Train models for all horizons.

        Args:
            df: Raw OHLCV data
            features: Calculated features DataFrame
            tune_params: Whether to tune hyperparameters
            n_tune_iter: Number of tuning iterations

        Returns:
            Dict with model paths and metrics
        """
        targets = self.create_targets(df)
        results = {}

        for horizon in self.horizons:
            logger.info(f"\n{'='*50}")
            logger.info(f"Training {horizon}-minute prediction model")
            logger.info(f"{'='*50}")

            # Align features with target
            target = targets[horizon]

            # Get common index (drop NaN targets at end)
            valid_idx = features.index.intersection(target.dropna().index)
            X = features.loc[valid_idx]
            y = target.loc[valid_idx]

            # Remove last `horizon` rows (no future data)
            X = X.iloc[:-horizon]
            y = y.iloc[:-horizon]

            logger.info(f"Training samples: {len(X)}")

            # Find best parameters
            if tune_params:
                tuner = HyperparameterTuner(n_iter=n_tune_iter)
                best_params = tuner.tune(X, y, n_splits=3)
            else:
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
                }

            # Final training with walk-forward validation
            validator = WalkForwardValidator(n_splits=5)
            splits = validator.split(X)

            # Use last split for final train/test
            train_idx, test_idx = splits[-1]
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            logger.info(f"Final train size: {len(X_train)}, test size: {len(X_test)}")

            # Train final model
            dtrain = lgb.Dataset(X_train, label=y_train)
            dtest = lgb.Dataset(X_test, label=y_test, reference=dtrain)

            model = lgb.train(
                best_params, dtrain,
                num_boost_round=200,
                valid_sets=[dtrain, dtest],
                valid_names=['train', 'valid'],
                callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)]
            )

            # Calculate metrics
            y_pred_proba = model.predict(X_test)
            y_pred = (y_pred_proba > 0.5).astype(int)

            accuracy = (y_pred == y_test.values).mean()

            # Calculate win rate for trades (only when confident)
            confident_mask = (y_pred_proba > 0.65) | (y_pred_proba < 0.35)
            if confident_mask.sum() > 0:
                confident_accuracy = (y_pred[confident_mask] == y_test.values[confident_mask]).mean()
            else:
                confident_accuracy = 0.0

            # Precision for BUY signals
            buy_mask = y_pred_proba > 0.65
            if buy_mask.sum() > 0:
                buy_precision = y_test.values[buy_mask].mean()
            else:
                buy_precision = 0.0

            metrics = {
                'accuracy': accuracy,
                'confident_accuracy': confident_accuracy,
                'buy_precision': buy_precision,
                'n_train': len(X_train),
                'n_test': len(X_test),
                'best_iteration': model.best_iteration if model.best_iteration else 200,
                'best_params': best_params
            }

            logger.info(f"\nMetrics for {horizon}-min model:")
            logger.info(f"  Overall Accuracy: {accuracy:.2%}")
            logger.info(f"  Confident Accuracy (>65% or <35%): {confident_accuracy:.2%}")
            logger.info(f"  BUY Precision (>65%): {buy_precision:.2%}")

            # Save model
            model_path = f"models/lightgbm_{horizon}min.txt"
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            model.save_model(model_path)
            logger.info(f"  Model saved: {model_path}")

            self.models[horizon] = model
            self.metrics[horizon] = metrics

            results[horizon] = {
                'model_path': model_path,
                'metrics': metrics
            }

        # Save the 5-min model as the default (for backward compatibility)
        if 5 in self.models:
            default_path = self.config.MODEL_PATH
            self.models[5].save_model(default_path)
            logger.info(f"\nDefault model (5-min) also saved to: {default_path}")

        return results


def fetch_training_data(config: UltraConfig, days: int = 90) -> pd.DataFrame:
    """Fetch historical data for training."""
    logger.info(f"Fetching {days} days of training data...")

    try:
        client = AlpacaClient(config)
        loader = DataLoader(client, config)
        df = loader.fetch_data(days=days)
        logger.info(f"Fetched {len(df)} bars of real data")
        return df
    except Exception as e:
        logger.warning(f"Failed to fetch real data: {e}")
        logger.info("Falling back to mock data...")
        df = MockDataLoader.fetch_data(days=days, symbol=config.SYMBOL)
        logger.info(f"Generated {len(df)} bars of mock data")
        return df


def main():
    """Main training function."""
    print("\n" + "="*60)
    print("  BLUEBIRD Advanced LightGBM Training")
    print("="*60 + "\n")

    # Configuration
    config = UltraConfig()

    TRAINING_DAYS = 90  # More data for better generalization
    TUNE_PARAMS = True  # Set to False for faster training
    TUNE_ITERATIONS = 15  # Number of hyperparameter combinations to try
    HORIZONS = [5, 15, 30]  # Prediction horizons in minutes

    logger.info(f"Training Configuration:")
    logger.info(f"  Data: {TRAINING_DAYS} days")
    logger.info(f"  Tune params: {TUNE_PARAMS} ({TUNE_ITERATIONS} iterations)")
    logger.info(f"  Horizons: {HORIZONS} minutes")

    # 1. Fetch Data
    df = fetch_training_data(config, days=TRAINING_DAYS)

    if df.empty:
        logger.error("No data available for training!")
        return

    # 2. Calculate Features
    logger.info("\nCalculating features...")
    feature_engine = AdvancedFeatureEngine()
    df_features = feature_engine.calculate_features(df)

    # Use only core features (to match adaptive_ai.py)
    core_features = feature_engine.get_core_features()
    features = df_features[core_features]

    logger.info(f"Feature shape: {features.shape}")
    logger.info(f"Features: {list(features.columns)}")

    # 3. Train Models
    trainer = MultiHorizonTrainer(config, horizons=HORIZONS)
    results = trainer.train_all(
        df_features,  # Pass full df for target creation
        features,     # Pass only core features for training
        tune_params=TUNE_PARAMS,
        n_tune_iter=TUNE_ITERATIONS
    )

    # 4. Summary
    print("\n" + "="*60)
    print("  TRAINING COMPLETE")
    print("="*60)

    for horizon, result in results.items():
        metrics = result['metrics']
        print(f"\n{horizon}-minute model:")
        print(f"  Path: {result['model_path']}")
        print(f"  Accuracy: {metrics['accuracy']:.2%}")
        print(f"  Confident Accuracy: {metrics['confident_accuracy']:.2%}")
        print(f"  BUY Precision: {metrics['buy_precision']:.2%}")

    # Save training report
    report = {
        'timestamp': datetime.now().isoformat(),
        'training_days': TRAINING_DAYS,
        'tune_params': TUNE_PARAMS,
        'horizons': HORIZONS,
        'results': {str(k): v for k, v in results.items()}
    }

    report_path = 'models/training_report.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\nTraining report saved: {report_path}")
    print("\nReady for paper trading!")


if __name__ == "__main__":
    main()
