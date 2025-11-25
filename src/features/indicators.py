import pandas as pd
import talib

class Indicators:
    @staticmethod
    def add_all_indicators(df):
        """Adds technical indicators to the dataframe."""
        if df.empty:
            return df
            
        # Ensure we have the necessary columns (lowercase)
        df.columns = [c.lower() for c in df.columns]
        
        # TA-Lib requires numpy arrays of floats
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        
        # RSI
        df['rsi'] = talib.RSI(close, timeperiod=14)
        
        # MACD
        macd, macdsignal, macdhist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
        df['macd'] = macd
        df['macd_signal'] = macdsignal
        df['macd_hist'] = macdhist
        
        # Bollinger Bands
        upper, middle, lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
        df['bb_upper'] = upper
        df['bb_middle'] = middle
        df['bb_lower'] = lower
        
        # ATR
        df['atr'] = talib.ATR(high, low, close, timeperiod=14)

        # ADX (Average Directional Index)
        df['adx'] = talib.ADX(high, low, close, timeperiod=14)

        # Momentum
        df['momentum'] = talib.MOM(close, timeperiod=10)

        # Volume ratio (current volume / 20-period average)
        if 'volume' in df.columns:
            vol = df['volume'].values
            df['volume_ratio'] = df['volume'] / df['volume'].rolling(window=20).mean()
        else:
            df['volume_ratio'] = 1.0

        # Drop NaN values created by indicators
        df.dropna(inplace=True)

        return df
