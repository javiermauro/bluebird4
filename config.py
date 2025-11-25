import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    # Alpaca API Settings
    API_KEY = os.getenv("ALPACA_API_KEY")
    SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
    BASE_URL = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
    
    # Trading Settings
    SYMBOL = "BTC/USD"
    TIMEFRAME = "1Min"  # 1 Minute candles for high frequency/scalping/swing
    USE_MOCK = False # Enable Mock Mode for demo/debugging
    MOCK_SPEED = 2 # Seconds between mock bars
    
    # Trading Parameters
    # TIMEFRAME = "1Min" # Already defined above, keeping it there.
    
    # Risk Management
    RISK_PER_TRADE = 0.02  # 2% of equity
    MAX_DRAWDOWN = 0.10    # Stop trading if 10% drawdown
    STOP_LOSS_PCT = 0.02
    TAKE_PROFIT_PCT = 0.04
    
    # Scalping Mode (DEFAULT: ON for safety)
    SCALPING_MODE = True  # Changed from False - ensures 10% positions by default
    SCALPING_TP = 0.02   # 2.0% - Profitable after 0.5% fees
    SCALPING_SL = 0.01   # 1.0% - 1:1 R/R ratio (50% breakeven WR)
    MAX_POSITIONS = 10   # Max concurrent positions (100% equity at 10% per trade)
    
    # Model Settings
    MODEL_PATH = "models/lightgbm_model.txt"
    FEATURE_COLUMNS = [
        "close", "volume", "rsi", "macd", "macd_signal", "bb_upper", "bb_lower"
    ]

    def __init__(self):
        if not self.API_KEY or not self.SECRET_KEY:
            # raise ValueError("API Keys not found. Please set ALPACA_API_KEY and ALPACA_SECRET_KEY in .env file.")
            print("WARNING: API Keys not found. Running in offline/mock mode only.")
