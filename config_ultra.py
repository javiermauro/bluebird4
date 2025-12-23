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

# =========================================
# MODULE-LEVEL FEE CONSTANTS
# =========================================
# These are also defined in UltraConfig class, but exposed here for direct access
# via `import config_ultra; config_ultra.MAKER_FEE_PCT`
MAKER_FEE_PCT = 0.0015   # 0.15% - limit orders (add liquidity)
TAKER_FEE_PCT = 0.0025   # 0.25% - market orders (take liquidity)


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
        "BTC/USD",   # Bitcoin - King, most liquid (35% allocation)
        "SOL/USD",   # Solana - Fast L1, good volume (30% allocation)
        "LTC/USD",   # Litecoin - Payment narrative, lower correlation (20% allocation)
        "AVAX/USD",  # Avalanche - Lowest BTC correlation 0.738 (15% allocation)
    ]
    MAX_EXPOSURE_PER_ASSET = 0.35  # Max 35% equity in any single asset (BTC)
    
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
    # OPTIMIZED FOR MORE TRADES & FASTER RECOVERY (Dec 9, 2025)
    # Tighter ranges = more grid completions = more profits
    GRID_CONFIGS = {
        "BTC/USD": {
            "num_grids": 5,        # 5 grids = 6 levels, ~1.25% spacing (was 4 grids)
            "range_pct": 0.0625,   # 6.25% range for 1.25% per grid
            "investment_ratio": 0.30  # 30% - reduced from 35%, still largest
        },
        "SOL/USD": {
            "num_grids": 5,        # 5 grids = 6 levels, ~1.30% spacing (was 4 grids)
            "range_pct": 0.065,    # 6.5% range for 1.30% per grid
            "investment_ratio": 0.25  # 25% - reduced from 30%, less concentration
        },
        "LTC/USD": {
            "num_grids": 6,        # 6 grids = 7 levels, ~1.40% spacing (was 5 grids, 1.69%)
            "range_pct": 0.084,    # 8.4% range for 1.40% per grid - TIGHTENED
            "investment_ratio": 0.25  # INCREASED: 25% - best performer!
        },
        "AVAX/USD": {
            "num_grids": 6,        # 6 grids = 7 levels, ~1.45% spacing (was 5 grids, 1.68%)
            "range_pct": 0.087,    # 8.7% range for 1.45% per grid - TIGHTENED
            "investment_ratio": 0.20  # INCREASED: 20% - second best performer!
        }
    }

    # Grid safety settings
    GRID_STOP_LOSS_PCT = 0.10    # Stop loss if price drops 10% below grid
    GRID_REBALANCE_PCT = 0.03   # Rebalance if price moves 3% outside grid

    # =========================================
    # LIMIT ORDER SETTINGS (Maker Fee Optimization)
    # =========================================
    # Use limit orders for grid trades to reduce fees (~0.15% maker vs ~0.25% taker)
    # Emergency actions (stop-loss, windfall) still use market orders for speed

    # Master switch - set to False to revert to market orders
    GRID_USE_LIMIT_ORDERS = True

    # Maker buffer in basis points (1 bp = 0.01%)
    # Used as a crossing guard: only place limit if grid price is this far from market
    # For BUY: place only if grid_price <= current_price * (1 - buffer)
    # For SELL: place only if grid_price >= current_price * (1 + buffer)
    MAKER_BUFFER_BPS = 5  # 5 basis points = 0.05%

    # Maximum age for limit orders before cancellation (minutes)
    # Orders older than this are cancelled and level waits for natural re-trigger
    MAX_ORDER_AGE_MINUTES = 60

    # Overshoot handling - what to do when price moves beyond all maker-safe grid levels
    # "wait"     - Do nothing, wait for price to return to grid range (safest)
    # "rebalance" - Auto-rebalance grid after N bars with no eligible levels (recommended)
    # "fallback_market" - Use market order as last resort (negates maker fee savings)
    LIMIT_ORDER_OVERSHOOT_MODE = "rebalance"

    # Rebalance guardrails (only apply when LIMIT_ORDER_OVERSHOOT_MODE="rebalance")
    OVERSHOOT_BARS_THRESHOLD = 5       # Trigger after N consecutive bars with no eligible levels
    OVERSHOOT_REBALANCE_COOLDOWN = 30  # Minutes between rebalances per symbol (prevents churn)

    # =========================================
    # FAST FILL DETECTION (Near Real-Time)
    # =========================================
    # Lightweight per-tick check for limit order fills
    # Detects fills in ~10-15s vs 5-minute full reconciliation
    ENABLE_FAST_FILL_CHECK = True
    FAST_FILL_INTERVAL_SECONDS = 10.0       # Min seconds between checks
    FAST_FILL_MAX_CHECKS_PER_CYCLE = 5      # Max individual order lookups per cycle
    FAST_FILL_MIN_ORDER_AGE_SECONDS = 10.0  # Grace period for new orders (eventual consistency)
    FAST_FILL_HANDLE_PARTIALS = True        # Apply partial fills on canceled orders
    FAST_FILL_ERROR_BACKOFF_SECONDS = 30.0  # Skip checks for this long after an error

    # =========================================
    # FEE MODEL
    # =========================================
    # Accurate fee tracking for profit calculation
    # Alpaca crypto fees: maker ~0.15%, taker ~0.25%
    MAKER_FEE_PCT = 0.0015   # 0.15% - limit orders (add liquidity)
    TAKER_FEE_PCT = 0.0025   # 0.25% - market orders (take liquidity)

    # On startup, if Alpaca has OPEN orders for our symbols that we are not tracking locally,
    # cancel them to avoid "surprise fills" after a crash/restart.
    # Safer default for an automated grid bot.
    CANCEL_UNTRACKED_OPEN_ORDERS_ON_STARTUP = True

    # Symbol precision for price and quantity rounding
    # (price_decimals, qty_decimals)
    SYMBOL_PRECISION = {
        "BTC/USD": (2, 6),   # $0.01 price, 0.000001 qty
        "ETH/USD": (2, 5),   # $0.01 price, 0.00001 qty
        "SOL/USD": (2, 4),   # $0.01 price, 0.0001 qty
        "LTC/USD": (2, 5),   # $0.01 price, 0.00001 qty
        "AVAX/USD": (2, 4),  # $0.01 price, 0.0001 qty
    }

    # =========================================
    # WINDFALL PROFIT-TAKING SETTINGS
    # =========================================
    # Capture profits when positions show significant unrealized gains
    # Prevents profits from evaporating when price reverses before grid sells
    #
    # Logic (Option B - Momentum-Triggered Exit):
    # IF (unrealized > 4% AND RSI > 70) OR (unrealized > 6%):
    #     SELL 70% of position
    #     LOG transaction for weekly review
    WINDFALL_PROFIT_CONFIG = {
        "enabled": True,
        "soft_threshold_pct": 4.0,    # Sell if > 4% AND RSI > 70 (overbought)
        "hard_threshold_pct": 6.0,    # Sell if > 6% regardless of RSI
        "rsi_threshold": 70,          # RSI overbought level for soft trigger
        "sell_portion": 0.70,         # Sell 70%, keep 30% for continued upside
        "cooldown_minutes": 30,       # Wait 30 min between windfall sells per symbol
    }

    # =========================================
    # RISK OVERLAY SETTINGS (Crash Protection)
    # =========================================
    # State machine: NORMAL -> RISK_OFF -> RECOVERY -> NORMAL
    # Prevents "death spiral" of buying into crashes and rebalancing lower

    # Master switch - easy kill switch if anything unexpected
    RISK_OVERLAY_ENABLED = True

    # RISK_OFF trigger settings (2-of-3 signals required)
    RISK_OFF_TRIGGERS_REQUIRED = 2  # 2 signals must fire to enter RISK_OFF
    RISK_OFF_MIN_HOLD_MINUTES = 20  # Minimum time in RISK_OFF before RECOVERY

    # Trigger thresholds
    RISK_OFF_MOMENTUM_THRESHOLD = -0.015      # Momentum < -1.5% = shock
    RISK_OFF_ADX_THRESHOLD = 35               # ADX > 35 + direction=down = downtrend
    RISK_OFF_CORRELATION_THRESHOLD = 0.90     # Correlation > 0.90 = correlated selloff

    # Drawdown velocity (DISABLED by default - can be noisy)
    RISK_OFF_DRAWDOWN_VELOCITY_ENABLED = False
    RISK_OFF_DRAWDOWN_VELOCITY = 0.02  # 2% per hour (if enabled)

    # Exposure caps
    NORMAL_TOTAL_EXPOSURE_CAP = 0.70          # 70% of equity in NORMAL mode
    RISK_OFF_TOTAL_EXPOSURE_CAP = 0.40        # 40% of equity in RISK_OFF mode

    # RECOVERY settings (gradual re-entry after stability)
    RECOVERY_STABILITY_BARS = 10              # Bars of stability per stage
    RECOVERY_MIN_TOTAL_BARS = 30              # Minimum total bars in RECOVERY before NORMAL
    RECOVERY_ENTRY_MOMENTUM_MIN = -0.005      # Momentum > -0.5% to enter RECOVERY
    RECOVERY_ADVANCE_MOMENTUM_MIN = 0.0       # Momentum > 0% to advance stages
    RECOVERY_POSITION_RAMP = [0.25, 0.5, 0.75, 1.0]  # Position size multipliers per stage
    RECOVERY_NO_NEW_LOW = True                # Require price >= min(last N bars) to advance

    # =========================================
    # DEVELOPING DOWNTREND PROTECTION
    # =========================================
    # Reduce position size before trends become "strong" (ADX > 35)
    # This prevents inventory buildup during early downtrends

    REGIME_ADX_DEVELOPING = 25                # "Developing trend" ADX threshold
    DEVELOPING_DOWNTREND_SIZE_MULT = 0.50     # 50% size when ADX 25-35 + DOWN

    # =========================================
    # CONSECUTIVE DOWN BARS PROTECTION
    # =========================================
    # Block buys after N consecutive red candles (simple crash guard)

    CONSECUTIVE_DOWN_BARS_ENABLED = True
    CONSECUTIVE_DOWN_BARS_BLOCK = 3           # Block buys after 3 consecutive down bars

    # =========================================
    # ORCHESTRATOR SETTINGS (Meta-controller for inventory management)
    # =========================================
    # Thin meta-controller that consumes existing signals and adds:
    # - Inventory episode tracking (how long we've been "stuck" holding inventory)
    # - Staged LIMIT-sell liquidation (only in NORMAL overlay mode)
    # - Mode-based gates and size multipliers
    # Critical: Orchestrator never overrides RiskOverlay decisions.

    # Master switches
    ORCHESTRATOR_ENABLED = True          # Enable orchestrator evaluation
    ORCHESTRATOR_ENFORCE = True          # True = enforce mode (active)
    ORCHESTRATOR_LIQUIDATION_ENABLED = True   # Enable staged liquidation

    # Cooldowns and rate limits
    ORCHESTRATOR_COOLDOWN_MINUTES = 60   # Min time between mode changes
    LIQ_MIN_INTERVAL_MINUTES = 90        # Min time between liquidation orders per symbol

    # Episode tracking
    EPISODE_START_PCT = 30               # Start episode when inventory >= 30%
    EPISODE_RESET_PCT = 10               # Reset episode when inventory <= 10%

    # Mode thresholds (with hysteresis)
    DEFENSIVE_INVENTORY_PCT = 150        # Enter DEFENSIVE when inventory >= 150%
    DEFENSIVE_EXIT_PCT = 130             # Exit DEFENSIVE when inventory < 130%
    GRID_REDUCED_ENTER_PCT = 100         # Enter GRID_REDUCED when inventory >= 100%
    GRID_REDUCED_EXIT_PCT = 80           # Exit GRID_REDUCED when inventory < 80%

    # Liquidation triggers - TP Trim (take profit) - SAFEST
    LIQ_TP_HOURS = 24                    # Min episode age for TP
    LIQ_TP_MIN_PNL_PCT = 0.003           # Min unrealized P/L (+0.3%)
    LIQ_TP_INVENTORY_PCT = 120           # Min inventory level
    LIQ_TP_TARGET_INV_PCT = 100          # Target inventory after trim

    # Liquidation triggers - Loss Cut (conservative)
    LIQ_LOSS_HOURS = 48                  # Min episode age for loss cut
    LIQ_LOSS_CUT_PCT = -0.02             # Max unrealized loss (-2%)
    LIQ_LOSS_INVENTORY_PCT = 130         # Min inventory level
    LIQ_LOSS_REDUCE_MULT = 0.25          # Reduce 25% of excess inventory

    # Liquidation triggers - Max Age Stop
    LIQ_MAX_AGE_HOURS = 72               # Max episode age
    LIQ_MAX_AGE_INVENTORY_PCT = 120      # Min inventory level

    # Execution - Adaptive slippage based on stress proxy (ADX>30)
    # Note: ADX is trend-strength, not volatility. Consider ATR% later.
    LIQ_SLIPPAGE_NORMAL = 0.004          # 0.4% in calm conditions
    LIQ_SLIPPAGE_STRESSED = 0.008        # 0.8% when ADX > 30 (stress proxy)

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

