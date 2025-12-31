from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame
from datetime import datetime, timedelta
import pandas as pd
import logging

logger = logging.getLogger(__name__)

class DataLoader:
    def __init__(self, client, config):
        self.client = client
        self.config = config

    def fetch_data(self, days=30, symbol=None, start_date=None, end_date=None):
        """
        Fetches historical data for a specific symbol or configured symbol.

        Args:
            days: Number of days to fetch (used if start_date/end_date not provided)
            symbol: Trading symbol (defaults to config.SYMBOL)
            start_date: Start datetime for data range (optional)
            end_date: End datetime for data range (optional)

        Returns:
            DataFrame with OHLCV data
        """
        target_symbol = symbol or self.config.SYMBOL

        # Use explicit date range if provided, otherwise use days
        if start_date is not None and end_date is not None:
            start_time = start_date
            end_time = end_date
            logger.info(f"Fetching data for {target_symbol} from {start_time} to {end_time}...")
        else:
            end_time = datetime.now()
            start_time = end_time - timedelta(days=days)
            logger.info(f"Fetching {days} days of historical data for {target_symbol}...")

        request_params = CryptoBarsRequest(
            symbol_or_symbols=[target_symbol],
            timeframe=TimeFrame.Minute,  # 1-minute bars
            start=start_time,
            end=end_time
        )

        try:
            bars = self.client.get_historical_data(request_params)
            df = bars.df

            # Reset index to make timestamp a column if needed, or keep as is.
            # Alpaca returns MultiIndex (symbol, timestamp).
            # We usually just want the dataframe for the specific symbol.
            if isinstance(df.index, pd.MultiIndex):
                df = df.xs(target_symbol)

            logger.info(f"Successfully fetched {len(df)} bars for {target_symbol}.")
            return df

        except Exception as e:
            logger.error(f"Error fetching historical data for {target_symbol}: {e}")
            return pd.DataFrame()
