"""
ULTRA CONFIG - Optimized for Small Accounts

This configuration is designed for accounts under $10,000.

Key Principles:
1. PRESERVE CAPITAL - Don't blow up
2. QUALITY over QUANTITY - Fewer trades, higher win rate
3. ASYMMETRIC RISK - Risk 1 to make 2.5+
4. ADAPT to conditions - Different strategies for different regimes
5. ACCOUNT FOR FEES - They matter more for small accounts!

Alpaca Crypto Fees:
- Maker: 0.15%
- Taker: 0.25%
- Round trip (market orders): ~0.50%
- Spread: ~0.05-0.10%
- TOTAL per trade: ~0.60%

This means your minimum TP needs to be > 0.60% just to break even!
"""

import os
from dotenv import load_dotenv

load_dotenv()


class UltraConfig:
    """
    Ultra Strategy Configuration
    
    All settings are designed with small accounts in mind.
    """
    
    # =========================================
    # API SETTINGS
    # =========================================
    API_KEY = os.getenv("ALPACA_API_KEY")
    SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
    BASE_URL = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
    
    # =========================================
    # TRADING SETTINGS
    # =========================================
    SYMBOL = "BTC/USD"  # Primary symbol
    
    # Multi-asset diversification (reduces single-asset risk)
    # Top 4 most liquid Alpaca crypto pairs for diversification
    SYMBOLS = [
        "BTC/USD",   # Bitcoin - King, most liquid
        "ETH/USD",   # Ethereum - #2, smart contracts
        "SOL/USD",   # Solana - Fast L1, high volatility (good for trading)
        "LINK/USD",  # Chainlink - Oracles, different sector = less correlation
    ]
    MAX_EXPOSURE_PER_ASSET = 0.25  # Max 25% equity in any single asset
    
    # Timeframe: 5-minute bars are optimal balance of signal vs noise
    # 1-min = too noisy, 15-min = too slow for scalping
    TIMEFRAME = "5Min"
    
    # Mock mode for testing
    USE_MOCK = False
    MOCK_SPEED = 2
    
    # =========================================
    # RISK MANAGEMENT - CRITICAL FOR SMALL ACCOUNTS
    # =========================================
    
    # Maximum positions at once
    # 4 positions = good diversification without over-trading
    MAX_POSITIONS = 4  # 4 positions across different assets
    
    # Maximum risk per trade (% of equity)
    # Kelly Criterion will adjust this, but never exceed
    MAX_RISK_PER_TRADE = 0.015  # 1.5% max loss per trade (conservative)
    
    # Stop trading circuit breaker
    # If daily loss exceeds this, stop trading for the day
    DAILY_LOSS_LIMIT = 0.05  # 5% daily max loss
    
    # Maximum drawdown before stopping entirely
    MAX_DRAWDOWN = 0.10  # 10% max drawdown
    
    # =========================================
    # POSITION SIZING
    # =========================================
    
    # Base position size (before Kelly adjustment)
    # AGGRESSIVE when conditions are right!
    BASE_POSITION_PCT = 0.20  # 20% of equity - meaningful size
    
    # Kelly Criterion settings
    KELLY_FRACTION = 0.40  # Use 40% Kelly - trade with conviction
    MIN_POSITION_PCT = 0.10  # Minimum 10% - worth the fees
    MAX_POSITION_PCT = 0.35  # Up to 35% when confidence is HIGH
    
    # =========================================
    # STOP LOSS / TAKE PROFIT - ASYMMETRIC RISK/REWARD
    # =========================================
    
    # These are ATR-based, not fixed percentages
    # More adaptive to current volatility
    
    STOP_LOSS_ATR_MULT = 1.0  # TIGHT stop loss = 1 ATR (cut losers fast!)
    TAKE_PROFIT_ATR_MULT = 3.5  # Take profit = 3.5 ATR above entry
    # This gives 3.5:1 R/R ratio - need only 30% win rate to profit!
    
    # Trailing stop settings - LET WINNERS RUN
    TRAILING_ACTIVATION_ATR = 1.5  # Activate earlier (1.5 ATR profit)
    TRAILING_DISTANCE_ATR = 0.8  # Trail tighter (0.8 ATR) to lock in gains
    
    # Time-based exit - don't sit in dead trades
    MAX_HOLD_BARS = 30  # Exit after 30 bars (2.5hrs on 5-min) if nothing happening
    
    # =========================================
    # REGIME DETECTION
    # =========================================
    
    # ADX thresholds for trend strength
    ADX_TRENDING = 25  # ADX > 25 = trending
    ADX_STRONG = 40  # ADX > 40 = strong trend
    
    # Volatility thresholds
    VOL_QUIET = 0.7  # ATR ratio < 0.7 = quiet market
    VOL_HIGH = 1.5  # ATR ratio > 1.5 = volatile market
    
    # BB squeeze threshold
    BB_SQUEEZE_WIDTH = 0.02  # BB width < 2% = squeeze
    
    # =========================================
    # ENTRY THRESHOLDS - BE SELECTIVE, THEN AGGRESSIVE
    # =========================================
    
    # Minimum confidence to enter trade
    MIN_ENTRY_CONFIDENCE = 0.65  # 65% - only trade high conviction setups
    
    # AI Thresholds
    BUY_THRESHOLD = 0.65   # AI prediction > 0.65 = bullish
    SELL_THRESHOLD = 0.35  # AI prediction < 0.35 = bearish
    MIN_CONFIDENCE = 70    # Don't trade below 70% overall confidence
    
    # =========================================
    # AGGRESSIVE POSITION SIZING (3-5% BASED ON CONFIDENCE)
    # =========================================
    
    # Position sizing scales with AI confidence:
    # - 70% confidence -> 3% position
    # - 100% confidence -> 5% position
    MIN_POSITION_PCT_CONF = 0.03  # 3% at minimum confidence (70%)
    MAX_POSITION_PCT_CONF = 0.05  # 5% at maximum confidence (100%)
    
    # RSI thresholds for mean reversion
    RSI_OVERSOLD = 30  # More opportunities than 25
    RSI_OVERBOUGHT = 70  # More opportunities than 75
    
    # Volume confirmation - CRITICAL for small accounts
    MIN_VOLUME_RATIO = 0.8  # Need at least 0.8x average (was 1.2x - too restrictive)
    
    # NEW: Skip trades in dead markets
    SKIP_LOW_VOLUME = True
    LOW_VOLUME_THRESHOLD = 0.3  # Don't trade if volume < 30% of average
    
    # =========================================
    # TIME FILTERS
    # =========================================
    
    # Enable time-based filtering
    USE_TIME_FILTER = True
    
    # Optimal trading windows (UTC hours)
    # US Session: 13:00-17:00 UTC (9AM-1PM ET)
    # Asian Session: 01:00-04:00 UTC (8PM-11PM ET)
    # London: 07:00-09:00 UTC (3AM-5AM ET)
    OPTIMAL_HOURS = [(13, 17), (1, 4), (7, 9)]
    
    # Hours to avoid
    AVOID_HOURS = [(22, 24), (5, 7)]
    
    # Weekend trading (reduce size)
    WEEKEND_SIZE_MULT = 0.5  # Half size on weekends
    
    # =========================================
    # STRATEGY SELECTION
    # =========================================
    
    # Enable/disable specific strategies
    ENABLE_TREND_FOLLOW = True
    ENABLE_MEAN_REVERT = True
    ENABLE_VOL_BREAKOUT = True
    
    # Strategy weights (for blending signals)
    STRATEGY_WEIGHTS = {
        'TREND_FOLLOW': 1.0,
        'MEAN_REVERT': 0.8,
        'VOL_BREAKOUT': 0.7,
    }
    
    # =========================================
    # HOLDOUT & TRAINING SETTINGS
    # =========================================

    # Holdout settings to prevent overfitting
    HOLDOUT_DAYS = 14  # Days reserved for holdout (never seen during training)
    WARMUP_BUFFER_BARS = 50  # Bars to skip between train/validation splits
    MIN_TRAINING_DAYS = 30  # Minimum days for training data

    # Default training settings
    DEFAULT_TRAINING_DAYS = 90  # Total days to fetch for training

    # =========================================
    # MODEL SETTINGS
    # =========================================

    # We still use ML, but for regime detection, not price prediction
    MODEL_PATH = "models/lightgbm_model.txt"
    
    # Feature columns for ML - MUST MATCH adaptive_ai.py _get_ai_prediction()
    FEATURE_COLUMNS = [
        "rsi",           # RSI (14-period)
        "macd",          # MACD line
        "macd_signal",   # MACD signal line
        "bb_upper",      # Bollinger upper band
        "bb_middle",     # Bollinger middle band
        "bb_lower",      # Bollinger lower band
        "atr",           # ATR (14-period)
        "volume_ratio",  # Volume / 20-period average
        "momentum",      # Momentum (10-period)
        "adx",           # ADX (14-period)
    ]
    
    # =========================================
    # GRID TRADING SETTINGS
    # =========================================

    # Enable grid trading mode (replaces prediction-based trading)
    USE_GRID_TRADING = True

    # Grid configuration per symbol
    # range_pct: Total range as percentage of price (e.g., 0.05 = 5% total range)
    # num_grids: Number of grid levels (more = smaller profits per trade, more trades)
    # investment_ratio: Portion of equity allocated to this symbol's grid
    # OPTIMIZED FOR PROFITABILITY (accounting for ~0.5% round-trip fees)
    # Rule: Grid spacing must be > 1% to have meaningful profit after fees
    GRID_CONFIGS = {
        "BTC/USD": {
            "num_grids": 5,        # Fewer grids = wider spacing = more profit per trade
            "range_pct": 0.08,     # 8% range = 1.6% per grid (1.1% after fees)
            "investment_ratio": 0.25  # Reduced - BTC moves slower
        },
        "ETH/USD": {
            "num_grids": 6,        # 6 levels
            "range_pct": 0.10,     # 10% range = 1.67% per grid (1.17% after fees)
            "investment_ratio": 0.25
        },
        "SOL/USD": {
            "num_grids": 5,        # 5 levels - SOL is volatile, needs room
            "range_pct": 0.12,     # 12% range = 2.4% per grid (1.9% after fees)
            "investment_ratio": 0.25  # Increased - best profit margin
        },
        "LINK/USD": {
            "num_grids": 5,        # 5 levels
            "range_pct": 0.12,     # 12% range = 2.4% per grid (1.9% after fees)
            "investment_ratio": 0.25  # Increased - best profit margin
        }
    }

    # Grid safety settings
    GRID_STOP_LOSS_PCT = 0.10    # Stop loss if price drops 10% below grid
    GRID_REBALANCE_PCT = 0.03   # Rebalance if price moves 3% outside grid

    # =========================================
    # LOGGING & MONITORING
    # =========================================

    LOG_LEVEL = "INFO"
    LOG_TRADES = True
    LOG_SIGNALS = True
    LOG_REGIME_CHANGES = True
    
    # Performance tracking
    TRACK_KELLY_STATS = True
    SAVE_TRADE_HISTORY = True
    
    def __init__(self):
        """Validate configuration on init."""
        if not self.API_KEY or not self.SECRET_KEY:
            print("⚠️ WARNING: API Keys not found. Running in offline/mock mode only.")
        
        # Validate risk settings
        assert self.MAX_RISK_PER_TRADE > 0 and self.MAX_RISK_PER_TRADE <= 0.05, \
            "MAX_RISK_PER_TRADE should be between 0 and 5%"
        
        assert self.MAX_POSITIONS >= 1 and self.MAX_POSITIONS <= 5, \
            "MAX_POSITIONS should be between 1 and 5 for small accounts"
        
        assert self.STOP_LOSS_ATR_MULT > 0 and self.TAKE_PROFIT_ATR_MULT > self.STOP_LOSS_ATR_MULT, \
            "Take profit must be larger than stop loss"
        
        print("✅ Ultra Config validated successfully")
        print(f"   Symbol: {self.SYMBOL}")
        print(f"   Max Positions: {self.MAX_POSITIONS}")
        print(f"   Risk per Trade: {self.MAX_RISK_PER_TRADE:.1%}")
        print(f"   R/R Ratio: {self.TAKE_PROFIT_ATR_MULT / self.STOP_LOSS_ATR_MULT:.1f}:1")


# Create default instance
Config = UltraConfig

