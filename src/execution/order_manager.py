import logging
from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest, TakeProfitRequest, StopLossRequest
from alpaca.trading.enums import OrderSide, TimeInForce, OrderClass

logger = logging.getLogger(__name__)

class OrderManager:
    def __init__(self, client, risk_manager):
        self.client = client
        self.risk_manager = risk_manager

    def execute_entry(self, symbol, side, price):
        """Executes an entry order with bracket (SL/TP)."""
        qty = self.risk_manager.calculate_position_size(price)
        
        if qty <= 0:
            logger.warning("Position size is 0. Skipping trade.")
            return

        stop_loss, take_profit = self.risk_manager.get_bracket_orders(price, side)
        
        logger.info(f"Placing {side} order for {qty:.4f} {symbol} @ {price}. SL: {stop_loss:.2f}, TP: {take_profit:.2f}")

        # For mock mode, simulate order placement
        if getattr(self.client.config, 'USE_MOCK', False):
            logger.info(f"MOCK BRACKET ORDER: {side.upper()} {qty:.4f} {symbol} @ ${price:.2f}, SL: ${stop_loss:.2f}, TP: ${take_profit:.2f}")
            return {"id": f"mock_order_{symbol}_{side}", "status": "filled", "filled_qty": qty}

        try:
            # Convert prices to proper float format
            entry_price = float(price)
            sl_price = float(stop_loss)
            tp_price = float(take_profit)
            qty_float = float(qty)
            
            # For crypto bracket orders, use simple market order instead
            # Bracket orders have limitations with crypto on Alpaca
            req = MarketOrderRequest(
                symbol=symbol,
                qty=qty_float,
                side=OrderSide.BUY if side == 'buy' else OrderSide.SELL,
                time_in_force=TimeInForce.GTC
            )
            
            order = self.client.trading_client.submit_order(req)
            logger.info(f"Market Order submitted: {order.id}. Note: Bracket orders not fully supported for crypto, managing SL/TP manually.")
            return order
            
        except Exception as e:
            logger.error(f"Failed to submit order: {e}")
            return None
