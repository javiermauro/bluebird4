import pandas as pd
import logging
from src.features.pipeline import DataPipeline
from datetime import datetime

logger = logging.getLogger(__name__)

class MLStrategy:
    def __init__(self, config, client, predictor, order_manager):
        self.config = config
        self.client = client
        self.predictor = predictor
        self.order_manager = order_manager
        self.pipeline = DataPipeline(config)
        
        # Keep track of recent bars to calculate indicators
        self.bars = [] 
        # Increased from 5 to 40: MACD needs 26, then dropna removes rows with NaN
        self.min_bars_required = 40
        
        # State tracking
        self.latest_prediction = 0.5
        self.latest_signal = "NEUTRAL"

    def get_latest_state(self):
        """Returns the latest strategy state."""
        return {
            "prediction": self.latest_prediction,
            "signal": self.latest_signal
        }

    def load_historical_data(self, bars):
        """Loads historical bars to warm up the strategy."""
        for bar in bars:
            self.bars.append({
                'timestamp': bar.timestamp,
                'open': bar.open,
                'high': bar.high,
                'low': bar.low,
                'close': bar.close,
                'volume': bar.volume
            })
        logger.info(f"Loaded {len(bars)} historical bars. Current buffer: {len(self.bars)}")

    async def on_bar(self, bar):
        """Called when a new bar is received."""
        logger.info(f"Processing bar: {bar.timestamp} - Close: {bar.close}")
        
        # 1. Update local history
        self.bars.append({
            'timestamp': bar.timestamp,
            'open': bar.open,
            'high': bar.high,
            'low': bar.low,
            'close': bar.close,
            'volume': bar.volume
        })
        
        # Keep list size manageable
        if len(self.bars) > 200:
            self.bars.pop(0)
            
        if len(self.bars) < self.min_bars_required:
            logger.info(f"Collecting data... {len(self.bars)}/{self.min_bars_required}")
            return

        # 2. Prepare Data
        df = pd.DataFrame(self.bars)
        features_df = self.pipeline.prepare_data(df)
        
        if features_df.empty:
            return

        # Get latest feature row
        current_features = features_df.iloc[[-1]]
        
        # 3. Generate Prediction
        # We need to drop non-feature columns if any remain, but pipeline should handle it.
        # XGBoost expects DMatrix or numpy array.
        prob_up = self.predictor.predict(current_features)
        
        prob_up = self.predictor.predict(current_features)
        self.latest_prediction = prob_up
        
        logger.info(f"Prediction (Prob Up): {prob_up:.4f}")
        
        # 4. Trading Logic
        # Determine thresholds based on mode
        if getattr(self.config, 'SCALPING_MODE', False):
            buy_threshold = 0.52
            sell_threshold = 0.48
        else:
            buy_threshold = 0.55
            sell_threshold = 0.45
        
        # Check all current positions and monitor SL/TP for each
        positions_list = []
        try:
            positions_list = self.client.trading_client.get_all_positions()
        except Exception as e:
            logger.error(f"Error fetching positions: {e}")
        
        # Monitor each position individually
        for position in positions_list:
            if position.symbol != self.config.SYMBOL:
                continue
                
            try:
                side = position.side
                qty = float(position.qty)
                entry_price = float(position.avg_entry_price)
                current_price = float(bar.close)
                
                # Calculate SL/TP for this specific position
                stop_loss, take_profit = self.order_manager.risk_manager.get_bracket_orders(entry_price, side)
                
                # Check if this position hit targets
                if side == 'long':
                    if current_price <= stop_loss:
                        logger.info(f"STOP LOSS HIT! Position @ ${entry_price:.2f}, Current: ${current_price:.2f}, SL: ${stop_loss:.2f}")
                        self.client.submit_order(self.config.SYMBOL, qty, 'sell')
                        continue
                    elif current_price >= take_profit:
                        logger.info(f"TAKE PROFIT HIT! Position @ ${entry_price:.2f}, Current: ${current_price:.2f}, TP: ${take_profit:.2f}")
                        self.client.submit_order(self.config.SYMBOL, qty, 'sell')
                        continue
            except Exception as e:
                logger.error(f"Error monitoring position: {e}")

        # Trading Logic - Allow multiple positions
        has_position = len(positions_list) > 0
        position_count = len(positions_list)
        max_positions = getattr(self.config, 'MAX_POSITIONS', 10)

        if prob_up > buy_threshold and position_count < max_positions:
            logger.info(f"Buy Signal detected! (Prob: {prob_up:.4f} > {buy_threshold}) | Positions: {position_count}/{max_positions}")
            self.latest_signal = "BUY"
            self.order_manager.execute_entry(self.config.SYMBOL, 'buy', bar.close)
            
        elif prob_up < sell_threshold and has_position:
            # Close ALL positions on strong sell signal
            logger.info(f"Sell Signal detected! (Prob: {prob_up:.4f} < {sell_threshold}) | Closing all {position_count} positions")
            self.latest_signal = "SELL"
            for position in positions_list:
                if position.symbol == self.config.SYMBOL:
                    self.client.submit_order(self.config.SYMBOL, float(position.qty), 'sell')
        else:
            self.latest_signal = "HOLD"
            
        # Shorting logic could be added here
