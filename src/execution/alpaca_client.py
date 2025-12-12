from alpaca.trading.client import TradingClient
from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.live.crypto import CryptoDataStream
from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest, GetOrdersRequest
from alpaca.trading.enums import OrderSide, TimeInForce, QueryOrderStatus
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import time
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

    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an open order by ID.

        Args:
            order_id: The Alpaca order ID to cancel

        Returns:
            True if cancellation succeeded or order was already closed
        """
        if getattr(self.config, 'USE_MOCK', False):
            logger.info(f"MOCK ORDER CANCELLED: {order_id}")
            return True

        try:
            self.trading_client.cancel_order_by_id(order_id)
            logger.info(f"Cancelled order: {order_id}")
            return True
        except Exception as e:
            error_msg = str(e).lower()
            # Already filled/cancelled is not an error
            if 'already' in error_msg or 'not found' in error_msg or 'filled' in error_msg:
                logger.debug(f"Order {order_id} already closed: {e}")
                return True
            logger.error(f"Failed to cancel order {order_id}: {e}")
            return False

    def get_open_orders(self, symbols: List[str] = None) -> List[Dict]:
        """
        Get all open orders, optionally filtered by symbols.

        Args:
            symbols: Optional list of symbols to filter (e.g., ['BTC/USD'])

        Returns:
            List of open order dicts
        """
        if getattr(self.config, 'USE_MOCK', False):
            return []

        try:
            request_params = GetOrdersRequest(
                status=QueryOrderStatus.OPEN,
                limit=500
            )

            if symbols:
                # Convert symbols to Alpaca format (remove /)
                alpaca_symbols = [s.replace('/', '') for s in symbols]
                request_params.symbols = alpaca_symbols

            orders = self.trading_client.get_orders(filter=request_params)

            return [{
                'id': str(order.id),
                'client_order_id': order.client_order_id,
                'symbol': order.symbol,
                'side': str(order.side),
                'type': str(order.type),
                'qty': float(order.qty) if order.qty else 0.0,
                'limit_price': float(order.limit_price) if order.limit_price else None,
                'status': str(order.status),
                'created_at': order.created_at.isoformat() if order.created_at else None,
            } for order in orders]

        except Exception as e:
            logger.error(f"Failed to get open orders: {e}")
            return []

    def get_order_by_id(self, order_id: str) -> Optional[Dict]:
        """
        Get order details by order ID from Alpaca.

        Returns order object with status, filled_qty, filled_avg_price, etc.
        """
        if getattr(self.config, 'USE_MOCK', False):
            return {
                'id': order_id,
                'status': 'filled',
                'filled_qty': '1.0',
                'filled_avg_price': '100.0',
                'filled_at': datetime.now().isoformat()
            }

        try:
            order = self.trading_client.get_order_by_id(order_id)
            return order
        except Exception as e:
            logger.error(f"Failed to get order {order_id}: {e}")
            return None

    def verify_order_fill(self, order_id: str, max_wait_seconds: int = 10, poll_interval: float = 0.5) -> Dict:
        """
        Verify an order was filled by polling Alpaca API.

        Args:
            order_id: The Alpaca order ID to verify
            max_wait_seconds: Maximum time to wait for fill confirmation
            poll_interval: Time between status checks

        Returns:
            Dict with: {
                'confirmed': bool,
                'status': str,
                'filled_qty': float,
                'filled_avg_price': float,
                'filled_at': str,
                'slippage_pct': float (if filled),
                'order_id': str
            }
        """
        if getattr(self.config, 'USE_MOCK', False):
            return {
                'confirmed': True,
                'status': 'filled',
                'filled_qty': 1.0,
                'filled_avg_price': 100.0,
                'filled_at': datetime.now().isoformat(),
                'slippage_pct': 0.0,
                'order_id': order_id
            }

        start_time = time.time()
        last_status = None

        while (time.time() - start_time) < max_wait_seconds:
            try:
                order = self.trading_client.get_order_by_id(order_id)
                status = str(order.status).lower()
                last_status = status

                if 'filled' in status:
                    filled_qty = float(order.filled_qty) if order.filled_qty else 0.0
                    filled_price = float(order.filled_avg_price) if order.filled_avg_price else 0.0
                    filled_at = order.filled_at.isoformat() if order.filled_at else None

                    logger.info(f"[VERIFIED] Order {order_id} FILLED: {filled_qty} @ ${filled_price:,.2f}")

                    return {
                        'confirmed': True,
                        'status': 'filled',
                        'filled_qty': filled_qty,
                        'filled_avg_price': filled_price,
                        'filled_at': filled_at,
                        'symbol': order.symbol,
                        'side': str(order.side),
                        'order_id': str(order.id),
                        'client_order_id': order.client_order_id
                    }

                elif any(s in status for s in ['canceled', 'expired', 'rejected', 'suspended']):
                    logger.warning(f"[VERIFY] Order {order_id} not filled: {status}")
                    return {
                        'confirmed': False,
                        'status': status,
                        'filled_qty': 0.0,
                        'filled_avg_price': 0.0,
                        'filled_at': None,
                        'order_id': str(order.id),
                        'reason': f"Order {status}"
                    }

                # Still pending, wait and retry
                time.sleep(poll_interval)

            except Exception as e:
                logger.error(f"[VERIFY] Error checking order {order_id}: {e}")
                time.sleep(poll_interval)

        # Timeout
        logger.warning(f"[VERIFY] Timeout waiting for order {order_id}, last status: {last_status}")
        return {
            'confirmed': False,
            'status': last_status or 'unknown',
            'filled_qty': 0.0,
            'filled_avg_price': 0.0,
            'filled_at': None,
            'order_id': order_id,
            'reason': 'Verification timeout'
        }

    def get_order_history(self, days: int = 7, symbols: List[str] = None, status: str = 'all') -> List[Dict]:
        """
        Fetch order history from Alpaca for reconciliation.

        Args:
            days: Number of days of history to fetch
            symbols: Optional list of symbols to filter
            status: 'all', 'open', or 'closed'

        Returns:
            List of order dicts with full details
        """
        if getattr(self.config, 'USE_MOCK', False):
            return []

        try:
            # Map status string to enum
            status_map = {
                'all': QueryOrderStatus.ALL,
                'open': QueryOrderStatus.OPEN,
                'closed': QueryOrderStatus.CLOSED
            }
            query_status = status_map.get(status, QueryOrderStatus.ALL)

            # Build request
            request_params = GetOrdersRequest(
                status=query_status,
                after=datetime.now() - timedelta(days=days),
                limit=500
            )

            if symbols:
                # Convert symbols to Alpaca format (remove /)
                alpaca_symbols = [s.replace('/', '') for s in symbols]
                request_params.symbols = alpaca_symbols

            orders = self.trading_client.get_orders(filter=request_params)

            # Convert to list of dicts
            order_list = []
            for order in orders:
                order_dict = {
                    'id': str(order.id),
                    'client_order_id': order.client_order_id,
                    'symbol': order.symbol,
                    'side': str(order.side),
                    'type': str(order.type),
                    'qty': float(order.qty) if order.qty else 0.0,
                    'filled_qty': float(order.filled_qty) if order.filled_qty else 0.0,
                    'filled_avg_price': float(order.filled_avg_price) if order.filled_avg_price else 0.0,
                    'status': str(order.status),
                    'created_at': order.created_at.isoformat() if order.created_at else None,
                    'filled_at': order.filled_at.isoformat() if order.filled_at else None,
                    'submitted_at': order.submitted_at.isoformat() if order.submitted_at else None
                }
                order_list.append(order_dict)

            logger.info(f"[HISTORY] Fetched {len(order_list)} orders from Alpaca (last {days} days)")
            return order_list

        except Exception as e:
            logger.error(f"Failed to fetch order history: {e}")
            return []

    def get_filled_orders(self, days: int = 7, symbols: List[str] = None) -> List[Dict]:
        """
        Get only filled orders for P&L tracking.
        Convenience method that filters for filled status.
        """
        all_orders = self.get_order_history(days=days, symbols=symbols, status='closed')
        return [o for o in all_orders if o['status'].lower() == 'orderstatus.filled' or o['status'].lower() == 'filled']

    def get_portfolio_history(self, period: str = "1M", timeframe: str = "1D") -> Dict:
        """
        Get portfolio equity history from Alpaca.

        Args:
            period: Time period - 1D, 1W, 1M, 3M, 1A (1 year), all
            timeframe: Resolution - 1Min, 5Min, 15Min, 1H, 1D

        Returns:
            Dict with equity curve data: {
                'timestamps': [],
                'equity': [],
                'profit_loss': [],
                'profit_loss_pct': [],
                'base_value': float
            }
        """
        if getattr(self.config, 'USE_MOCK', False):
            return {
                'timestamps': [],
                'equity': [],
                'profit_loss': [],
                'profit_loss_pct': [],
                'base_value': 100000
            }

        try:
            from alpaca.trading.requests import GetPortfolioHistoryRequest
            from datetime import datetime as dt

            # Build proper request object
            request = GetPortfolioHistoryRequest(
                period=period,
                timeframe=timeframe
            )

            history = self.trading_client.get_portfolio_history(history_filter=request)

            # Convert timestamps (unix epoch) to ISO format
            timestamps = []
            if history.timestamp:
                for ts in history.timestamp:
                    if isinstance(ts, (int, float)):
                        timestamps.append(dt.fromtimestamp(ts).isoformat())
                    elif hasattr(ts, 'isoformat'):
                        timestamps.append(ts.isoformat())
                    else:
                        timestamps.append(str(ts))

            # Convert to dict format
            return {
                'timestamps': timestamps,
                'equity': list(history.equity) if history.equity else [],
                'profit_loss': list(history.profit_loss) if history.profit_loss else [],
                'profit_loss_pct': list(history.profit_loss_pct) if history.profit_loss_pct else [],
                'base_value': float(history.base_value) if history.base_value else 0.0
            }

        except Exception as e:
            logger.error(f"Failed to fetch portfolio history: {e}")
            return {
                'timestamps': [],
                'equity': [],
                'profit_loss': [],
                'profit_loss_pct': [],
                'base_value': 0.0,
                'error': str(e)
            }

    def get_all_filled_orders(self, days: int = 30) -> List[Dict]:
        """
        Get all filled orders for comprehensive P/L calculation.
        Uses pagination to get all orders.

        Args:
            days: Number of days of history

        Returns:
            List of all filled orders
        """
        if getattr(self.config, 'USE_MOCK', False):
            return []

        try:
            all_orders = []
            after_date = datetime.now() - timedelta(days=days)

            # Fetch in batches
            request_params = GetOrdersRequest(
                status=QueryOrderStatus.CLOSED,
                after=after_date,
                limit=500
            )

            orders = self.trading_client.get_orders(filter=request_params)

            for order in orders:
                status = str(order.status).lower()
                if 'filled' in status:
                    order_dict = {
                        'id': str(order.id),
                        'symbol': order.symbol,
                        'side': str(order.side).replace('OrderSide.', '').lower(),
                        'qty': float(order.filled_qty) if order.filled_qty else 0.0,
                        'price': float(order.filled_avg_price) if order.filled_avg_price else 0.0,
                        'value': float(order.filled_qty or 0) * float(order.filled_avg_price or 0),
                        'status': 'filled',
                        'created_at': order.created_at.isoformat() if order.created_at else None,
                        'filled_at': order.filled_at.isoformat() if order.filled_at else None,
                    }
                    all_orders.append(order_dict)

            logger.info(f"[HISTORY] Fetched {len(all_orders)} filled orders (last {days} days)")
            return all_orders

        except Exception as e:
            logger.error(f"Failed to fetch all filled orders: {e}")
            return []
