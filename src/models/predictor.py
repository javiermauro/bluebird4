import lightgbm as lgb
import pandas as pd
import numpy as np
import logging
import os
import json

logger = logging.getLogger(__name__)

class Predictor:
    """
    LightGBM predictor for 3-class classification (DOWN/SIDEWAYS/UP).

    Model outputs probabilities for each class:
    - Class 0: DOWN (price drops > 0.5%)
    - Class 1: SIDEWAYS (price stays within ±0.5%)
    - Class 2: UP (price rises > 0.5%)
    """

    def __init__(self, config):
        self.config = config
        self.model = None
        self.metadata = None
        self._load_model()

    def _load_model(self):
        """Loads the LightGBM model from disk or initializes a new one."""
        model_path = self.config.MODEL_PATH.replace('.json', '.txt')
        meta_path = model_path.replace('.txt', '_meta.json')

        if os.path.exists(model_path):
            try:
                self.model = lgb.Booster(model_file=model_path)
                logger.info(f"Loaded model from {model_path}")

                # Load metadata if available
                if os.path.exists(meta_path):
                    with open(meta_path, 'r') as f:
                        self.metadata = json.load(f)
                    logger.info(f"Model type: {self.metadata.get('model_type', 'unknown')}")
                    logger.info(f"Optimal UP threshold: {self.metadata.get('optimal_up_probability', 0.33):.0%}")
                else:
                    self.metadata = {'model_type': '3class_classification', 'optimal_up_probability': 0.33}

            except Exception as e:
                logger.error(f"Failed to load model: {e}")
                self.model = None
        else:
            logger.warning(f"No model found at {model_path}. Model needs training.")
            self.model = None

    def predict(self, features):
        """
        Generates predictions from 3-class classification model.

        Returns:
            dict with:
                - up_probability: Probability of UP class (>0.5% move up)
                - down_probability: Probability of DOWN class (>0.5% move down)
                - sideways_probability: Probability of SIDEWAYS class
                - predicted_class: 0=DOWN, 1=SIDEWAYS, 2=UP
                - signal: 'BUY', 'SELL', or 'HOLD'
                - confidence: Confidence in the prediction (0-100)
        """
        if self.model is None:
            logger.warning("Model not trained. Returning neutral prediction.")
            return {
                'up_probability': 0.33,
                'down_probability': 0.33,
                'sideways_probability': 0.34,
                'predicted_class': 1,  # SIDEWAYS
                'signal': 'HOLD',
                'confidence': 0
            }

        # Get probabilities for all 3 classes
        proba = self.model.predict(features)

        # Handle single sample prediction
        if len(proba.shape) == 1:
            # Single sample - shape is (3,) for 3 classes
            if len(proba) == 3:
                down_prob, sideways_prob, up_prob = proba
            else:
                # Fallback for older binary models
                up_prob = proba[0]
                down_prob = 1 - up_prob
                sideways_prob = 0
        else:
            # Multiple samples - shape is (n_samples, 3)
            down_prob = proba[0, 0]
            sideways_prob = proba[0, 1]
            up_prob = proba[0, 2]

        # Determine predicted class
        predicted_class = int(np.argmax([down_prob, sideways_prob, up_prob]))

        # Get optimal threshold from metadata
        up_threshold = self.metadata.get('optimal_up_probability', 0.33) if self.metadata else 0.33
        down_threshold = 0.33  # Symmetric for now

        # Determine signal based on thresholds
        if up_prob > up_threshold and up_prob > down_prob:
            signal = 'BUY'
            confidence = int(up_prob * 100)
        elif down_prob > down_threshold and down_prob > up_prob:
            signal = 'SELL'
            confidence = int(down_prob * 100)
        else:
            signal = 'HOLD'
            confidence = int(sideways_prob * 100)

        return {
            'up_probability': float(up_prob),
            'down_probability': float(down_prob),
            'sideways_probability': float(sideways_prob),
            'predicted_class': predicted_class,
            'signal': signal,
            'confidence': confidence
        }

    def predict_up_probability(self, features):
        """
        Legacy method - returns just the UP probability.
        For backward compatibility with existing code.
        """
        result = self.predict(features)
        return result['up_probability']

    def train(self, X, y):
        """Trains the model (3-class classification)."""
        logger.info("Training 3-class classification model...")
        dtrain = lgb.Dataset(X, label=y)

        params = {
            'max_depth': 6,
            'learning_rate': 0.05,
            'min_child_samples': 20,
            'bagging_fraction': 0.8,
            'feature_fraction': 0.8,
            'num_leaves': 31,
            'objective': 'multiclass',
            'num_class': 3,
            'metric': 'multi_logloss',
            'verbosity': -1,
            'bagging_freq': 1,
            'feature_pre_filter': False,
            'is_unbalance': True,
        }

        self.model = lgb.train(params, dtrain, num_boost_round=200)

        # Save model
        model_path = self.config.MODEL_PATH.replace('.json', '.txt')
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        self.model.save_model(model_path)
        logger.info(f"Model saved to {model_path}")
