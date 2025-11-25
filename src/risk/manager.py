import logging

logger = logging.getLogger(__name__)

class RiskManager:
    def __init__(self, config, client):
        self.config = config
        self.client = client

    def calculate_position_size(self, price):
        """Calculates position size based on risk per trade."""
        try:
            account = self.client.trading_client.get_account()
            equity = float(account.equity)
            buying_power = float(account.buying_power)
            
            # Position sizing strategy depends on mode
            scalping_mode = getattr(self.config, 'SCALPING_MODE', False)
            
            if scalping_mode:
                # SCALPING MODE: Small positions, high frequency
                # Use 10% of equity per trade to allow multiple concurrent positions
                # This enables true scalping: make money through VOLUME, not SIZE
                position_pct = 0.10
                logger.info(f"🟢 SCALPING MODE: Using {position_pct*100}% position sizing")
            else:
                # STANDARD MODE: Larger positions, swing trading
                # Use 95% of equity for full deployment with strict stop loss
                position_pct = 0.95
                logger.info(f"🔵 STANDARD MODE: Using {position_pct*100}% position sizing")
            
            
            # CRITICAL FIX: Use equity * position_pct ONLY
            # Do NOT use buying_power (includes leverage and can exceed equity)
            trade_value = equity * position_pct
            qty = trade_value / price
            
            # DEBUG LOGGING
            logger.info(f"Position Sizing Calculation:")
            logger.info(f"  Equity: ${equity:,.2f}")
            logger.info(f"  Buying Power: ${buying_power:,.2f}")
            logger.info(f"  Mode: {'SCALPING' if scalping_mode else 'STANDARD'}")
            logger.info(f"  Position %: {position_pct*100}%")
            logger.info(f"  Price: ${price:,.2f}")
            logger.info(f"  Trade Value: ${trade_value:,.2f}")
            logger.info(f"  BTC Quantity: {qty:.6f}")
            
            # Ensure minimum order size (Alpaca crypto min is usually small, e.g. $1)
            if trade_value < 1.0:
                logger.warning("Trade value too small.")
                return 0
                
            return qty
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return 0

    def get_bracket_orders(self, price, side):
        """Calculates Stop Loss and Take Profit prices."""
        # Determine risk parameters based on mode
        if getattr(self.config, 'SCALPING_MODE', False):
            sl_pct = getattr(self.config, 'SCALPING_SL', 0.003)
            tp_pct = getattr(self.config, 'SCALPING_TP', 0.006)
        else:
            sl_pct = getattr(self.config, 'STOP_LOSS_PCT', 0.02)
            tp_pct = getattr(self.config, 'TAKE_PROFIT_PCT', 0.04)
        
        if side == 'buy':
            stop_loss = price * (1 - sl_pct)
            take_profit = price * (1 + tp_pct)
        else: # sell
            stop_loss = price * (1 + sl_pct)
            take_profit = price * (1 - tp_pct)
            
        return stop_loss, take_profit
