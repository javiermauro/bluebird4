import pandas as pd
from src.features.indicators import Indicators

class DataPipeline:
    def __init__(self, config):
        self.config = config

    def prepare_data(self, df):
        """Prepares raw data for the model."""
        # Add indicators
        df = Indicators.add_all_indicators(df)
        
        # Select features
        features = df[self.config.FEATURE_COLUMNS]
        
        return features
