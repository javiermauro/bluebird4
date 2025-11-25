import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class MockDataLoader:
    @staticmethod
    def fetch_data(days=30, symbol="BTC/USD"):
        """Generates synthetic crypto price data."""
        print(f"Generating {days} days of mock data for {symbol}...")
        
        # Generate timestamps (1 minute intervals)
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)
        timestamps = pd.date_range(start=start_time, end=end_time, freq='1min')
        
        n = len(timestamps)
        
        # Generate random walk for price
        # Start at 50000, random steps
        np.random.seed(42)
        returns = np.random.normal(loc=0.00001, scale=0.001, size=n) # Slight upward drift
        price_path = 50000 * np.cumprod(1 + returns)
        
        # Create OHLCV data
        # High/Low/Open/Close variations
        highs = price_path * (1 + np.abs(np.random.normal(0, 0.001, n)))
        lows = price_path * (1 - np.abs(np.random.normal(0, 0.001, n)))
        opens = price_path * (1 + np.random.normal(0, 0.0005, n))
        closes = price_path
        volumes = np.random.randint(1, 100, n) * np.random.uniform(0.5, 2.0, n)
        
        df = pd.DataFrame({
            'timestamp': timestamps,
            'open': opens,
            'high': highs,
            'low': lows,
            'close': closes,
            'volume': volumes
        })
        
        df.set_index('timestamp', inplace=True)
        return df
