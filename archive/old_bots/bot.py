import asyncio
import logging
from config import Config
from src.execution.alpaca_client import AlpacaClient
from src.data.stream import DataStream
from src.strategy.ml_strategy import MLStrategy
from src.models.predictor import Predictor
from src.risk.manager import RiskManager
from src.execution.order_manager import OrderManager

from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame
from datetime import datetime, timedelta

logger = logging.getLogger("BlueBird")

async def run_bot(broadcast_update, broadcast_log):
    logger.info("Starting BlueBird 4.0 Trading Bot...")
    await broadcast_log("Starting BlueBird 4.0 Trading Bot...")
    
    try:
        config = Config()
        await broadcast_log(f"Configuration loaded. Symbol: {config.SYMBOL}")
        
        # Initialize Components
        client = AlpacaClient(config)
        risk_manager = RiskManager(config, client)
        order_manager = OrderManager(client, risk_manager)
        predictor = Predictor(config)
        
        strategy = MLStrategy(config, client, predictor, order_manager)

        # WARM UP: Fetch historical data to fill the buffer
        if not getattr(config, 'USE_MOCK', False):
            try:
                await broadcast_log("Warming up with historical data...")
                req = CryptoBarsRequest(
                    symbol_or_symbols=[config.SYMBOL],
                    timeframe=TimeFrame.Minute,
                    start=datetime.now() - timedelta(minutes=60),
                    limit=60
                )
                bars = client.get_historical_data(req)
                if config.SYMBOL in bars.data:
                    symbol_bars = bars.data[config.SYMBOL]
                    strategy.load_historical_data(symbol_bars)
                    await broadcast_log(f"Warm-up complete. Loaded {len(symbol_bars)} bars.")
                else:
                    await broadcast_log("No historical data found for warm-up.")
            except Exception as e:
                logger.error(f"Warm-up failed: {e}")
                await broadcast_log(f"Warm-up failed: {e}")
        
        # Wrap the on_bar callback to also broadcast updates
        original_on_bar = strategy.on_bar
        
        async def on_bar_wrapper(bar):
            # Call original strategy logic
            await original_on_bar(bar)
            
            # Broadcast update to UI
            # Fetch latest state
            ml_state = strategy.get_latest_state()
            
            # Fetch Account & Positions (This might be slow for every bar, maybe throttle?)
            # For demo purposes, it's fine.
            try:
                account = client.trading_client.get_account()
                positions = client.get_positions()
                
                # Serialize positions
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
            except Exception as e:
                logger.error(f"Error fetching account data: {e}")
                account = None
                positions_data = []

            # Calculate Daily Stats (approximate from loaded bars)
            # In a real app, you'd fetch 24h ticker data from Alpaca
            daily_stats = {
                "high": max([b['high'] for b in strategy.bars]) if strategy.bars else 0,
                "low": min([b['low'] for b in strategy.bars]) if strategy.bars else 0,
                "volume": sum([b['volume'] for b in strategy.bars]) if strategy.bars else 0,
                "change": ((bar.close - strategy.bars[0]['open']) / strategy.bars[0]['open'] * 100) if strategy.bars else 0
            }

            await broadcast_update({
                "status": "running",
                "price": float(bar.close),
                "timestamp": str(bar.timestamp),
                "market": daily_stats,
                "ml": {
                    "prediction": float(ml_state.get("prediction", 0.5)),
                    "signal": str(ml_state.get("signal", "NEUTRAL")),
                    "buy_threshold": 0.52 if getattr(config, 'SCALPING_MODE', False) else 0.55,
                    "sell_threshold": 0.48 if getattr(config, 'SCALPING_MODE', False) else 0.45,
                    "scalping_mode": getattr(config, 'SCALPING_MODE', False)
                },
                "account": {
                    "equity": float(account.equity) if account else 0.0,
                    "buying_power": float(account.buying_power) if account else 0.0,
                    "balance": float(account.cash) if account else 0.0
                },
                "positions": positions_data
            })
            
        stream = DataStream(config, on_bar_wrapper)
        
        await broadcast_log("Connecting to Data Stream...")
        await stream.start()
            
    except Exception as e:
        logger.error(f"Critical Bot Error: {e}")
        await broadcast_log(f"Critical Error: {str(e)}")
        # Don't raise, just stop?
