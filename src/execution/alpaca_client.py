from alpaca.trading.client import TradingClient
from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.live.crypto import CryptoDataStream
from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
import logging

logger = logging.getLogger(__name__)

class AlpacaClient:
    def __init__(self, config):
        self.config = config
        self.trading_client = TradingClient(config.API_KEY, config.SECRET_KEY, paper=True) # Default to paper
        self.data_client = CryptoHistoricalDataClient(config.API_KEY, config.SECRET_KEY)
        # self.stream_client = CryptoDataStream(config.API_KEY, config.SECRET_KEY) # REMOVED: Redundant connection causing 429s
        
        self._verify_connection()

    def _verify_connection(self):
        if getattr(self.config, 'USE_MOCK', False):
            logger.info("Running in MOCK MODE. Skipping connection verification.")
            return

        try:
            account = self.trading_client.get_account()
            if account.status == 'ACTIVE':
                logger.info(f"Connected to Alpaca. Account Status: {account.status}. Buying Power: {account.buying_power}")
            else:
                logger.error(f"Alpaca Account is not active: {account.status}")
        except Exception as e:
            logger.error(f"Failed to connect to Alpaca: {e}")
            raise

    def get_historical_data(self, request_params):
        """Fetches historical data using the data client."""
        # In mock mode, we could return mock history, but for now let's just return empty or let it fail if called
        return self.data_client.get_crypto_bars(request_params)

    def get_positions(self):
        """Returns all open positions."""
        if getattr(self.config, 'USE_MOCK', False):
            # Return dummy positions for demo
            return [] 

        try:
            return self.trading_client.get_all_positions()
        except Exception as e:
            logger.error(f"Failed to fetch positions: {e}")
            return []

    def submit_order(self, symbol, qty, side, type='market', limit_price=None):
        """Submits an order to Alpaca."""
        if getattr(self.config, 'USE_MOCK', False):
            logger.info(f"MOCK ORDER SUBMITTED: {side} {qty} {symbol} @ {type}")
            return "mock_order_id"

        try:
            if type == 'market':
                req = MarketOrderRequest(
                    symbol=symbol,
                    qty=qty,
                    side=side,
                    time_in_force=TimeInForce.GTC
                )
            elif type == 'limit':
                if limit_price is None:
                    raise ValueError("Limit price must be provided for limit orders")
                req = LimitOrderRequest(
                    symbol=symbol,
                    qty=qty,
                    side=side,
                    time_in_force=TimeInForce.GTC,
                    limit_price=limit_price
                )
            else:
                raise ValueError(f"Unsupported order type: {type}")

            order = self.trading_client.submit_order(req)
            logger.info(f"Order submitted: {side} {qty} {symbol} @ {type}")
            return order
        except Exception as e:
            logger.error(f"Failed to submit order: {e}")
            return None
