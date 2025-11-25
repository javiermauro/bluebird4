"""
BLUEBIRD ULTRA BOT

Main trading bot runner that integrates the Ultra strategy system
with the API server and dashboard.
"""

import asyncio
import logging
from datetime import datetime, timedelta

from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("BlueBird-ULTRA")


async def run_ultra_bot(broadcast_update, broadcast_log):
    """
    Main Ultra bot runner.
    
    Integrates:
    - Regime detection
    - Multi-strategy system
    - Kelly position sizing
    - Time-of-day filtering
    - Real-time dashboard updates
    """
    logger.info("=" * 50)
    logger.info("Starting BlueBird ULTRA Trading Bot...")
    logger.info("=" * 50)
    
    await broadcast_log("Initializing BlueBird ULTRA System...")
    
    try:
        # Import configuration
        try:
            from config_ultra import UltraConfig as Config
            config = Config()
            await broadcast_log("Loaded ULTRA configuration")
        except ImportError:
            from config import Config
            config = Config()
            await broadcast_log("Loaded standard configuration (Ultra config not found)")
        
        # Initialize components
        from src.execution.alpaca_client import AlpacaClient
        from src.risk.manager import RiskManager
        from src.execution.order_manager import OrderManager
        from src.strategy.regime_detector import RegimeDetector, TimeOfDayFilter
        from src.features.ultra_indicators import UltraIndicators
        
        # Try to use Ultra strategy, fall back to ML strategy
        try:
            from src.strategy.ultra_strategy import UltraStrategy
            USE_ULTRA = True
        except ImportError:
            from src.strategy.ml_strategy import MLStrategy as UltraStrategy
            USE_ULTRA = False
        
        from src.models.predictor import Predictor
        
        client = AlpacaClient(config)
        risk_manager = RiskManager(config, client)
        order_manager = OrderManager(client, risk_manager)
        predictor = Predictor(config)
        
        # Initialize strategy
        strategy = UltraStrategy(config, client, predictor, order_manager)
        
        # Initialize regime detector
        regime_detector = RegimeDetector(config)
        
        await broadcast_log(f"Strategy initialized: {'ULTRA' if USE_ULTRA else 'LEGACY'}")
        await broadcast_log(f"Symbol: {config.SYMBOL} | Timeframe: {getattr(config, 'TIMEFRAME', '5Min')}")
        
        # Warm up with historical data
        if not getattr(config, 'USE_MOCK', False):
            try:
                await broadcast_log("Fetching historical data for warm-up...")
                
                req = CryptoBarsRequest(
                    symbol_or_symbols=[config.SYMBOL],
                    timeframe=TimeFrame.Minute,
                    start=datetime.now() - timedelta(minutes=120),
                    limit=120
                )
                
                bars = client.get_historical_data(req)
                
                if config.SYMBOL in bars.data:
                    symbol_bars = bars.data[config.SYMBOL]
                    strategy.load_historical_data(symbol_bars)
                    await broadcast_log(f"Loaded {len(symbol_bars)} historical bars")
                else:
                    await broadcast_log("No historical data available")
                    
            except Exception as e:
                logger.error(f"Warm-up failed: {e}")
                await broadcast_log(f"Warm-up failed: {str(e)[:50]}")
        
        # Kelly tracking for position sizing
        kelly_stats = {
            'wins': [],
            'losses': [],
            'trades_count': 0
        }
        
        # Wrap the on_bar callback
        original_on_bar = strategy.on_bar
        
        async def on_bar_wrapper(bar):
            """Process bar and broadcast state to dashboard."""
            
            # Call strategy logic
            await original_on_bar(bar)
            
            # Get strategy state
            strategy_state = strategy.get_latest_state()
            
            # Get time filter
            time_info = TimeOfDayFilter.get_time_score()
            
            # Prepare features for regime detection
            import pandas as pd
            if hasattr(strategy, 'bars') and len(strategy.bars) > 50:
                df = pd.DataFrame(strategy.bars)
                df = UltraIndicators.add_all_indicators(df)
                
                if not df.empty:
                    # Get regime info
                    regime_info = regime_detector.detect(df)
                    latest = df.iloc[-1]
                    
                    # Extract metrics
                    metrics = {
                        'adx': float(latest.get('adx', 0)),
                        'rsi': float(latest.get('rsi', 50)),
                        'atr_pct': float(latest.get('atr_pct', 0)),
                        'volume_ratio': float(latest.get('volume_ratio', 1)),
                        'trend_score': float(latest.get('trend_score', 0))
                    }
                else:
                    regime_info = {'regime': 'UNKNOWN', 'confidence': 0, 'strategy_hint': 'WAIT', 'should_trade': False}
                    metrics = {'adx': 0, 'rsi': 50, 'atr_pct': 0, 'volume_ratio': 1, 'trend_score': 0}
            else:
                regime_info = {'regime': 'UNKNOWN', 'confidence': 0, 'strategy_hint': 'WAIT', 'should_trade': False}
                metrics = {'adx': 0, 'rsi': 50, 'atr_pct': 0, 'volume_ratio': 1, 'trend_score': 0}
            
            # Calculate Kelly
            if kelly_stats['trades_count'] > 5:
                win_rate = len(kelly_stats['wins']) / kelly_stats['trades_count']
            else:
                win_rate = 0.5
            
            kelly_pct = getattr(config, 'BASE_POSITION_PCT', 0.15)
            
            # Fetch account & positions
            try:
                account = client.trading_client.get_account()
                positions = client.get_positions()
                
                positions_data = []
                for p in positions:
                    positions_data.append({
                        "symbol": p.symbol,
                        "qty": float(p.qty),
                        "avg_entry_price": float(p.avg_entry_price),
                        "current_price": float(p.current_price),
                        "unrealized_pl": float(p.unrealized_pl),
                        "unrealized_plpc": float(p.unrealized_plpc)
                    })
                    
                account_data = {
                    "equity": float(account.equity),
                    "buying_power": float(account.buying_power),
                    "balance": float(account.cash)
                }
            except Exception as e:
                logger.error(f"Error fetching account: {e}")
                account_data = {"equity": 0, "buying_power": 0, "balance": 0}
                positions_data = []
            
            # Calculate market stats from bars
            if hasattr(strategy, 'bars') and strategy.bars:
                market_data = {
                    "high": max(b['high'] for b in strategy.bars[-60:]) if len(strategy.bars) >= 60 else max(b['high'] for b in strategy.bars),
                    "low": min(b['low'] for b in strategy.bars[-60:]) if len(strategy.bars) >= 60 else min(b['low'] for b in strategy.bars),
                    "volume": sum(b['volume'] for b in strategy.bars[-60:]) if len(strategy.bars) >= 60 else sum(b['volume'] for b in strategy.bars),
                    "change": ((bar.close - strategy.bars[0]['open']) / strategy.bars[0]['open'] * 100) if strategy.bars else 0
                }
            else:
                market_data = {"high": 0, "low": 0, "volume": 0, "change": 0}
            
            # Build Ultra state
            ultra_state = {
                "regime": regime_info.get('regime', 'UNKNOWN'),
                "strategy": regime_info.get('strategy_hint', 'WAIT'),
                "confidence": regime_info.get('confidence', 0),
                "signal": strategy_state.get('signal', 'HOLD'),
                "should_trade": regime_info.get('should_trade', False) and time_info['score'] > 0.4,
                "metrics": metrics,
                "time_filter": {
                    "score": time_info['score'],
                    "window_name": time_info['window_name'],
                    "is_weekend": time_info['is_weekend']
                },
                "kelly": {
                    "position_pct": kelly_pct,
                    "win_rate": win_rate,
                    "trades_count": kelly_stats['trades_count']
                }
            }
            
            # Broadcast full update
            await broadcast_update({
                "status": "running",
                "price": float(bar.close),
                "timestamp": str(bar.timestamp),
                "market": market_data,
                "account": account_data,
                "positions": positions_data,
                "ultra": ultra_state
            })
            
            # Log significant events
            if regime_info.get('regime', 'UNKNOWN') != 'UNKNOWN':
                if strategy_state.get('signal') == 'BUY':
                    await broadcast_log(f"BUY SIGNAL | Regime: {regime_info['regime']} | Conf: {regime_info['confidence']:.0%}")
                elif strategy_state.get('signal') in ['SELL', 'CLOSE']:
                    await broadcast_log(f"SELL SIGNAL | Closing positions")
        
        # Start data stream
        from src.data.stream import DataStream
        stream = DataStream(config, on_bar_wrapper)
        
        await broadcast_log("Connecting to real-time data stream...")
        await broadcast_log(f"REGIME DETECTION: Active")
        await broadcast_log(f"TIME FILTER: {TimeOfDayFilter.get_time_score()['window_name']}")
        
        await stream.start()
        
    except Exception as e:
        logger.error(f"Critical Bot Error: {e}", exc_info=True)
        await broadcast_log(f"CRITICAL ERROR: {str(e)[:100]}")


# For direct execution testing
if __name__ == "__main__":
    async def mock_broadcast_update(data):
        print(f"UPDATE: {data.get('ultra', {}).get('regime', 'N/A')}")
    
    async def mock_broadcast_log(msg):
        print(f"LOG: {msg}")
    
    asyncio.run(run_ultra_bot(mock_broadcast_update, mock_broadcast_log))

