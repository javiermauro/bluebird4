import logging
import pandas as pd
from config_ultra import UltraConfig
from src.execution.alpaca_client import AlpacaClient
from src.data.loader import DataLoader
from src.data.mock_loader import MockDataLoader
from src.features.pipeline import DataPipeline
from src.models.predictor import Predictor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Trainer")

def train_model():
    logger.info("Starting Model Training...")
    config = UltraConfig()
    
    # 1. Fetch Data
    df = pd.DataFrame()
    try:
        logger.info("Attempting to fetch real data...")
        client = AlpacaClient(config)
        loader = DataLoader(client, config)
        df = loader.fetch_data(days=60) # Train on more data
    except Exception as e:
        logger.warning(f"Failed to fetch real data: {e}")
        logger.info("Falling back to MOCK DATA for training demonstration.")
        df = MockDataLoader.fetch_data(days=60, symbol=config.SYMBOL)
        
    if df.empty:
        logger.error("No data available for training.")
        return

    # 2. Prepare Features
    logger.info("Preparing features...")
    pipeline = DataPipeline(config)
    features = pipeline.prepare_data(df.copy())
    
    # 3. Create Target Variable
    # Target: 1 if price increases in next N minutes, else 0
    # Let's say we want to predict if price is higher in 5 minutes
    future_close = df['close'].shift(-5)
    current_close = df['close']
    
    # Align indices
    features = features.loc[features.index.intersection(future_close.index)]
    target = (future_close.loc[features.index] > current_close.loc[features.index]).astype(int)
    
    # Drop last 5 rows where target is NaN
    features = features.iloc[:-5]
    target = target.iloc[:-5]
    
    logger.info(f"Training data shape: {features.shape}")
    
    # 4. Train Model
    predictor = Predictor(config)
    predictor.train(features, target)
    
    logger.info("Training complete.")

if __name__ == "__main__":
    train_model()
