"""
Multi-Asset AI Trading Bot
Trades up to 4 different crypto assets using Adaptive AI for decisions

Key Features:
- XGBoost AI model for predictions
- Multi-timeframe analysis (1-min, 5-min, 15-min)
- Confidence-based position sizing (3-5% per trade)
- Full reasoning for every decision
- Real-time dashboard updates with AI state
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional

from config_ultra import UltraConfig
from src.execution.alpaca_client import AlpacaClient
from src.strategy.adaptive_ai import AdaptiveAI

from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.requests import MarketOrderRequest, TakeProfitRequest, StopLossRequest
from alpaca.trading.enums import OrderSide, TimeInForce, OrderClass

logger = logging.getLogger("BlueBird-AI")


class AIMultiAssetBot:
    """
    AI-Powered Multi-Asset Trading Bot
    
    Uses AdaptiveAI for intelligent trade decisions with:
    - XGBoost predictions
    - Multi-timeframe analysis
    - Confidence-based sizing
    - Full reasoning trail
    """
    
    def __init__(self, config: UltraConfig, client: AlpacaClient):
        self.config = config
        self.client = client
        self.symbols = getattr(config, 'SYMBOLS', ['BTC/USD', 'ETH/USD'])
        self.max_positions = getattr(config, 'MAX_POSITIONS', 4)
        
        # Initialize Adaptive AI
        self.ai = AdaptiveAI(config)
        
        # Position sizing thresholds
        self.min_pos_pct = getattr(config, 'MIN_POSITION_PCT_CONF', 0.03)
        self.max_pos_pct = getattr(config, 'MAX_POSITION_PCT_CONF', 0.05)
        self.min_confidence = getattr(config, 'MIN_CONFIDENCE', 70)
        
        # Risk settings
        self.stop_loss_pct = 0.02  # 2% stop loss
        self.take_profit_pct = 0.06  # 6% take profit
        
        # Track last trade per symbol
        self.last_trade: Dict[str, Dict] = {}

        # Track open positions with their SL/TP targets (backup monitoring)
        self.position_targets: Dict[str, Dict] = {}

        logger.info(f"AIMultiAssetBot initialized for {len(self.symbols)} symbols")
        
    def add_bar(self, symbol: str, bar: Any) -> None:
        """Add a bar to the AI engine."""
        self.ai.add_bar(symbol, bar)
        
    def evaluate(self, symbol: str) -> Dict:
        """Get AI evaluation for a symbol."""
        return self.ai.evaluate(symbol)
    
    def calculate_position_size(self, confidence: int, equity: float, price: float) -> float:
        """
        Calculate position size based on AI confidence.
        
        Aggressive sizing:
        - 70% confidence -> 3% position
        - 100% confidence -> 5% position
        """
        if confidence < self.min_confidence:
            return 0.0
        
        # Scale from min to max based on confidence
        # confidence range: 70-100 maps to position 3-5%
        scale = (confidence - self.min_confidence) / (100 - self.min_confidence)
        scale = max(0, min(scale, 1))  # Clamp 0-1
        
        position_pct = self.min_pos_pct + (scale * (self.max_pos_pct - self.min_pos_pct))
        position_value = equity * position_pct
        qty = position_value / price
        
        return qty
    
    def should_trade(self, ai_state: Dict, has_position: bool, can_open_new: bool) -> tuple:
        """
        Determine if we should trade based on AI evaluation.
        
        Returns:
            (should_trade: bool, action: str, reason: str)
        """
        signal = ai_state.get('signal', 'HOLD')
        confidence = ai_state.get('confidence', 0)
        
        # Don't trade if confidence too low
        if confidence < self.min_confidence:
            return False, 'HOLD', f"Confidence {confidence}% < {self.min_confidence}% threshold"
        
        # BUY logic
        if signal == 'BUY':
            if has_position:
                return False, 'HOLD', "Already have position in this asset"
            if not can_open_new:
                return False, 'HOLD', f"Max positions ({self.max_positions}) reached"
            return True, 'BUY', f"AI signal BUY with {confidence}% confidence"
        
        # SELL logic
        if signal == 'SELL':
            if has_position:
                return True, 'SELL', f"AI signal SELL with {confidence}% confidence"
            return False, 'HOLD', "No position to sell"
        
        return False, 'HOLD', "AI signal is HOLD"
    
    def get_all_ai_states(self) -> Dict[str, Dict]:
        """Get AI state for all symbols."""
        return self.ai.get_all_evaluations()

    def check_position_targets(self, symbol: str, current_price: float, positions: Dict) -> Optional[str]:
        """
        Check if position has hit stop loss or take profit.

        This is BACKUP monitoring in case bracket orders don't work.
        Returns 'STOP_LOSS', 'TAKE_PROFIT', or None.
        """
        alpaca_symbol = symbol.replace('/', '')

        # Check if we have this position and its targets
        if alpaca_symbol not in self.position_targets:
            return None

        targets = self.position_targets[alpaca_symbol]
        stop_loss = targets.get('stop_loss')
        take_profit = targets.get('take_profit')
        entry_price = targets.get('entry_price')

        if stop_loss and current_price <= stop_loss:
            logger.warning(f"STOP LOSS HIT: {symbol} @ ${current_price:.2f} (SL: ${stop_loss:.2f})")
            return 'STOP_LOSS'

        if take_profit and current_price >= take_profit:
            logger.info(f"TAKE PROFIT HIT: {symbol} @ ${current_price:.2f} (TP: ${take_profit:.2f})")
            return 'TAKE_PROFIT'

        return None

    def record_position_targets(self, symbol: str, entry_price: float, stop_loss: float, take_profit: float):
        """Record SL/TP targets for a position."""
        alpaca_symbol = symbol.replace('/', '')
        self.position_targets[alpaca_symbol] = {
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'timestamp': datetime.now().isoformat()
        }

    def clear_position_targets(self, symbol: str):
        """Clear targets when position is closed."""
        alpaca_symbol = symbol.replace('/', '')
        if alpaca_symbol in self.position_targets:
            del self.position_targets[alpaca_symbol]


async def run_multi_bot(broadcast_update, broadcast_log):
    """Run the AI-powered multi-asset trading bot."""
    logger.info("Starting BlueBird AI Multi-Asset Bot...")
    await broadcast_log("🤖 Starting BlueBird AI Multi-Asset Bot...")
    
    try:
        config = UltraConfig()
        symbols = getattr(config, 'SYMBOLS', ['BTC/USD', 'ETH/USD'])
        
        await broadcast_log(f"🎯 Trading {len(symbols)} assets: {', '.join(symbols)}")
        await broadcast_log(f"📊 AI-Powered with XGBoost + Multi-Timeframe Analysis")
        await broadcast_log(f"💰 Position sizing: {config.MIN_POSITION_PCT_CONF*100:.0f}%-{config.MAX_POSITION_PCT_CONF*100:.0f}% based on confidence")
        
        # Initialize
        client = AlpacaClient(config)
        bot = AIMultiAssetBot(config, client)
        
        # Warm up with historical data
        await broadcast_log("🔄 Warming up AI with historical data...")
        
        for symbol in symbols:
            try:
                alpaca_symbol = symbol.replace('/', '')
                req = CryptoBarsRequest(
                    symbol_or_symbols=[alpaca_symbol],
                    timeframe=TimeFrame.Minute,
                    start=datetime.now() - timedelta(minutes=100),
                    limit=100
                )
                bars = client.data_client.get_crypto_bars(req)
                if alpaca_symbol in bars.data:
                    for bar in bars.data[alpaca_symbol]:
                        bot.add_bar(symbol, bar)
                    await broadcast_log(f"  ✓ {symbol}: {len(bars.data[alpaca_symbol])} bars loaded")
            except Exception as e:
                await broadcast_log(f"  ✗ {symbol}: Failed ({e})")
        
        await broadcast_log("✅ AI warm-up complete. Starting live trading...")
        
        # Main trading loop
        from alpaca.data.live import CryptoDataStream
        
        stream = CryptoDataStream(
            config.API_KEY,
            config.SECRET_KEY
        )
        
        async def handle_bar(bar):
            """Handle incoming bar with AI evaluation."""
            symbol = bar.symbol
            
            logger.info(f"📊 {symbol} @ ${bar.close:,.2f}")
            
            if symbol not in symbols:
                return
            
            # Add bar to AI
            bot.add_bar(symbol, bar)
            
            # Get AI evaluation
            ai_state = bot.evaluate(symbol)
            
            # Get all AI states for dashboard
            all_ai_states = bot.get_all_ai_states()
            
            # Get current positions
            try:
                positions = client.get_positions()
                current_positions = {p.symbol: p for p in positions}
                num_positions = len(positions)
            except:
                current_positions = {}
                num_positions = 0
            
            # Get account
            try:
                account = client.trading_client.get_account()
                equity = float(account.equity)
            except:
                account = None
                equity = 0
            
            # Check if we have position in this symbol
            alpaca_symbol = symbol.replace('/', '')
            has_position = alpaca_symbol in current_positions or symbol in current_positions
            can_open_new = num_positions < config.MAX_POSITIONS
            
            # Get trading decision from AI
            should_trade, action, reason = bot.should_trade(ai_state, has_position, can_open_new)
            
            trade_executed = None
            
            if should_trade and action == 'BUY':
                # Calculate position size based on confidence
                confidence = ai_state.get('confidence', 0)
                qty = bot.calculate_position_size(confidence, equity, float(bar.close))
                
                if qty > 0.0001:
                    try:
                        # Calculate targets BEFORE placing order
                        entry_price = float(bar.close)
                        stop_loss = round(entry_price * (1 - bot.stop_loss_pct), 2)
                        take_profit = round(entry_price * (1 + bot.take_profit_pct), 2)

                        # Use BRACKET order with stop loss and take profit
                        order = MarketOrderRequest(
                            symbol=alpaca_symbol,
                            qty=round(qty, 6),
                            side=OrderSide.BUY,
                            time_in_force=TimeInForce.GTC,
                            order_class=OrderClass.BRACKET,
                            stop_loss=StopLossRequest(stop_price=stop_loss),
                            take_profit=TakeProfitRequest(limit_price=take_profit)
                        )
                        client.trading_client.submit_order(order)
                        
                        trade_executed = {
                            'action': 'BUY',
                            'symbol': symbol,
                            'price': entry_price,
                            'qty': qty,
                            'confidence': confidence,
                            'stop_loss': stop_loss,
                            'take_profit': take_profit,
                            'reasoning': ai_state.get('reasoning', []),
                            'timestamp': datetime.now().isoformat()
                        }
                        
                        bot.last_trade[symbol] = trade_executed

                        # Record targets for backup monitoring
                        bot.record_position_targets(symbol, entry_price, stop_loss, take_profit)

                        await broadcast_log(f"🟢 BUY {symbol}: {qty:.4f} @ ${entry_price:,.2f}")
                        await broadcast_log(f"   Confidence: {confidence}% | SL: ${stop_loss:,.2f} | TP: ${take_profit:,.2f}")
                        for r in ai_state.get('reasoning', [])[:3]:
                            await broadcast_log(f"   • {r}")
                            
                    except Exception as e:
                        await broadcast_log(f"❌ Order failed: {e}")
                        
            elif should_trade and action == 'SELL':
                try:
                    client.trading_client.close_position(alpaca_symbol)

                    # Clear position targets
                    bot.clear_position_targets(symbol)

                    trade_executed = {
                        'action': 'SELL',
                        'symbol': symbol,
                        'price': float(bar.close),
                        'confidence': ai_state.get('confidence', 0),
                        'reasoning': ai_state.get('reasoning', []),
                        'timestamp': datetime.now().isoformat()
                    }

                    await broadcast_log(f"🔴 SELL {symbol} @ ${bar.close:,.2f}")
                    for r in ai_state.get('reasoning', [])[:3]:
                        await broadcast_log(f"   • {r}")

                except Exception as e:
                    await broadcast_log(f"❌ Close failed: {e}")

            # BACKUP: Check if any position hit SL/TP (in case bracket orders didn't work)
            if has_position:
                target_hit = bot.check_position_targets(symbol, float(bar.close), current_positions)
                if target_hit:
                    try:
                        client.trading_client.close_position(alpaca_symbol)
                        bot.clear_position_targets(symbol)
                        await broadcast_log(f"⚠️ {target_hit}: Closed {symbol} @ ${bar.close:,.2f}")
                    except Exception as e:
                        await broadcast_log(f"❌ Emergency close failed: {e}")
            
            # Prepare positions data
            positions_data = []
            for p in current_positions.values():
                positions_data.append({
                    "symbol": p.symbol,
                    "qty": float(p.qty),
                    "avg_entry_price": float(p.avg_entry_price),
                    "current_price": float(p.current_price),
                    "unrealized_pl": float(p.unrealized_pl),
                    "unrealized_plpc": float(p.unrealized_plpc)
                })
            
            # Build AI dashboard state
            warmup_status = bot.ai.get_warmup_status()
            ai_dashboard = {
                'prediction': ai_state.get('prediction'),
                'confidence': ai_state.get('confidence', 0),
                'signal': ai_state.get('signal', 'HOLD'),
                'reasoning': ai_state.get('reasoning', []),
                'features': ai_state.get('features', {}),
                'thresholds': ai_state.get('thresholds', {}),
                'multi_timeframe': ai_state.get('multi_timeframe', {}),
                'feature_importance': ai_state.get('feature_importance', {}),
                'warmup': warmup_status,
            }
            
            # Broadcast comprehensive update
            await broadcast_update({
                "status": "running",
                "price": float(bar.close),
                "timestamp": str(bar.timestamp),
                "symbol": symbol,
                "multi_asset": {
                    "symbols": symbols,
                    "signals": {s: all_ai_states[s]['signal'] for s in symbols if s in all_ai_states},
                    "confidences": {s: all_ai_states[s]['confidence'] for s in symbols if s in all_ai_states},
                    "active_symbol": symbol
                },
                "ai": ai_dashboard,
                "ultra": {
                    "regime": "AI_ADAPTIVE",
                    "strategy": f"{num_positions}/{config.MAX_POSITIONS} positions",
                    "signal": ai_state.get('signal', 'HOLD'),
                    "confidence": ai_state.get('confidence', 0) / 100,
                    "should_trade": should_trade,
                    "trade_reason": reason,
                    "metrics": ai_state.get('features', {}),
                    "time_filter": {"window_name": "ACTIVE", "score": 1.0},
                    "kelly": bot.ai.get_kelly_stats()
                },
                "account": {
                    "equity": equity,
                    "buying_power": float(account.buying_power) if account else 0,
                    "balance": float(account.cash) if account else 0
                },
                "positions": positions_data,
                "market": {
                    "high": float(bar.high),
                    "low": float(bar.low),
                    "volume": float(bar.volume),
                    "change": 0
                },
                "last_trade": trade_executed or bot.last_trade.get(symbol)
            })
        
        # Subscribe to all symbols
        stream.subscribe_bars(handle_bar, *symbols)
        await broadcast_log(f"📡 Subscribed to: {', '.join(symbols)}")
        
        # Run stream with reconnection
        backoff = 1
        max_backoff = 60
        
        while True:
            try:
                await broadcast_log("🔌 Connecting to Alpaca stream...")
                await stream._run_forever()
                backoff = 1
            except Exception as e:
                await broadcast_log(f"⚠️ Stream disconnected: {e}")
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, max_backoff)
    
    except Exception as e:
        logger.error(f"AI Bot Error: {e}")
        await broadcast_log(f"❌ Error: {e}")
        raise
