import asyncio
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from config import Config
from src.execution.alpaca_client import AlpacaClient
from src.data.loader import DataLoader
from src.data.mock_loader import MockDataLoader
from src.features.pipeline import DataPipeline
from src.models.predictor import Predictor

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Backtest")

def calculate_metrics(equity_curve):
    """Calculates performance metrics."""
    equity_series = pd.Series(equity_curve)
    returns = equity_series.pct_change().dropna()
    
    # Total Return
    total_return = (equity_series.iloc[-1] - equity_series.iloc[0]) / equity_series.iloc[0] * 100
    
    # Sharpe Ratio (assuming risk-free rate = 0, annualized for 1min data is tricky, just simple sharpe)
    # Annualizing factor for 1min data (24/7) = 60*24*365 = 525600
    sharpe_ratio = returns.mean() / returns.std() * np.sqrt(525600) if returns.std() != 0 else 0
    
    # Max Drawdown
    rolling_max = equity_series.cummax()
    drawdown = (equity_series - rolling_max) / rolling_max
    max_drawdown = drawdown.min() * 100
    
    return total_return, sharpe_ratio, max_drawdown

def run_backtest():
    logger.info("Starting Backtest...")
    
    config = Config()
    
    # 1. Fetch Historical Data
    df = pd.DataFrame()
    try:
        logger.info("Attempting to fetch real data from Alpaca...")
        client = AlpacaClient(config)
        loader = DataLoader(client, config)
        df = loader.fetch_data(days=30)
    except Exception as e:
        logger.warning(f"Failed to fetch real data: {e}")
        logger.info("Falling back to MOCK DATA.")
        df = MockDataLoader.fetch_data(days=30, symbol=config.SYMBOL)

    if df.empty:
        logger.error("No data available.")
        return

    # 2. Prepare Data
    logger.info("Preparing data features...")
    pipeline = DataPipeline(config)
    features = pipeline.prepare_data(df.copy())
    
    # Align features with original df
    df = df.loc[features.index]
    
    # 3. Run Prediction & Simulation
    predictor = Predictor(config)
    
    balance = 10000.0
    position = 0.0
    equity_curve = []
    
    logger.info(f"Initial Balance: ${balance}")
    
    for i in range(len(features)):
        row = features.iloc[[i]]
        price = df.iloc[i]['close']
        
        # Predict
        prob_up = predictor.predict(row)
        
        # Logic
        buy_threshold = 0.65
        sell_threshold = 0.35
        
        # Buy
        if prob_up > buy_threshold and position == 0:
            qty = (balance * 0.95) / price
            cost = qty * price
            balance -= cost
            position = qty
            # logger.debug(f"BUY  @ {price:.2f} | Prob: {prob_up:.2f}")
            
        # Sell
        elif prob_up < sell_threshold and position > 0:
            revenue = position * price
            balance += revenue
            position = 0.0
            # logger.debug(f"SELL @ {price:.2f} | Prob: {prob_up:.2f}")
            
        # Mark to market
        equity = balance + (position * price)
        equity_curve.append(equity)

    # 4. Results
    total_return, sharpe, max_dd = calculate_metrics(equity_curve)
    
    logger.info("-" * 30)
    logger.info(f"Backtest Complete.")
    logger.info(f"Final Equity: ${equity_curve[-1]:.2f}")
    logger.info(f"Total Return: {total_return:.2f}%")
    logger.info(f"Sharpe Ratio: {sharpe:.2f}")
    logger.info(f"Max Drawdown: {max_dd:.2f}%")
    
    # Plot
    try:
        plt.figure(figsize=(12, 6))
        plt.plot(df.index, equity_curve, label='Equity')
        plt.title(f"Backtest Result: {config.SYMBOL}\nReturn: {total_return:.2f}% | Sharpe: {sharpe:.2f}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig("backtest_result.png")
        logger.info("Saved plot to backtest_result.png")
    except Exception as e:
        logger.warning(f"Could not save plot: {e}")

if __name__ == "__main__":
    run_backtest()
