import lightgbm as lgb
import pandas as pd
import logging
import os

logger = logging.getLogger(__name__)

class Predictor:
    def __init__(self, config):
        self.config = config
        self.model = None
        self._load_model()

    def _load_model(self):
        """Loads the LightGBM model from disk or initializes a new one."""
        model_path = self.config.MODEL_PATH.replace('.json', '.txt')
        if os.path.exists(model_path):
            try:
                self.model = lgb.Booster(model_file=model_path)
                logger.info(f"Loaded model from {model_path}")
            except Exception as e:
                logger.error(f"Failed to load model: {e}")
                self.model = None
        else:
            logger.warning(f"No model found at {model_path}. Model needs training.")
            self.model = None

    def train(self, X, y):
        """Trains the model."""
        logger.info("Training model...")
        dtrain = lgb.Dataset(X, label=y)

        params = {
            'max_depth': 6,
            'learning_rate': 0.1,
            'objective': 'binary',
            'metric': 'binary_logloss',
            'verbosity': -1,
            'feature_pre_filter': False
        }

        self.model = lgb.train(params, dtrain, num_boost_round=100)

        # Save model
        model_path = self.config.MODEL_PATH.replace('.json', '.txt')
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        self.model.save_model(model_path)
        logger.info(f"Model saved to {model_path}")

    def predict(self, features):
        """Generates a prediction (probability of up move)."""
        if self.model is None:
            logger.warning("Model not trained. Returning neutral prediction.")
            return 0.5

        prediction = self.model.predict(features)
        return prediction[0]  # Return float
