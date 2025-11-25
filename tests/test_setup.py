import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TestSetup")

def test_imports():
    logger.info("Testing imports...")
    try:
        import alpaca.trading.client
        import pandas
        import xgboost
        import talib
        from config import Config
        from src.execution.alpaca_client import AlpacaClient
        from src.data.loader import DataLoader
        from src.features.indicators import Indicators
        from src.models.predictor import Predictor
        from src.risk.manager import RiskManager
        logger.info("All imports successful.")
    except ImportError as e:
        logger.error(f"Import failed: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error during imports: {e}")
        sys.exit(1)

def test_config():
    logger.info("Testing Config...")
    try:
        from config import Config
        # Mock env vars if needed, but better to check if they exist
        if not os.getenv("ALPACA_API_KEY"):
            logger.warning("ALPACA_API_KEY not set in environment. Config init might fail if not in .env")
        
        # We won't instantiate Config() here if it strictly requires keys, 
        # unless we want to fail fast.
        # c = Config() 
        logger.info("Config module loaded.")
    except Exception as e:
        logger.error(f"Config test failed: {e}")

if __name__ == "__main__":
    test_imports()
    test_config()
    logger.info("Setup test complete.")
