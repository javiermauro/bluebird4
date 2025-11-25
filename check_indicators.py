#!/usr/bin/env python3
"""
Check current technical indicators to understand ML prediction
"""
import os
import sys
from datetime import datetime, timedelta
from dotenv import load_dotenv
import pandas as pd

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config
from src.execution.alpaca_client import AlpacaClient
from src.features.indicators import Indicators
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame

load_dotenv()

# Initialize
config = Config()
client = AlpacaClient(config)

# Fetch last 100 bars
print("Fetching recent price data...")
req = CryptoBarsRequest(
    symbol_or_symbols=[config.SYMBOL],
    timeframe=TimeFrame.Minute,
    start=datetime.now() - timedelta(minutes=100),
    limit=100
)

bars_data = client.get_historical_data(req)
symbol_bars = bars_data.data[config.SYMBOL]

# Convert to dict format
bars = []
for bar in symbol_bars:
    bars.append({
        'timestamp': bar.timestamp,
        'open': bar.open,
        'high': bar.high,
        'low': bar.low,
        'close': bar.close,
        'volume': bar.volume
    })

# Process through indicators
df = pd.DataFrame(bars)
df = Indicators.add_all_indicators(df)

if df.empty:
    print("No data after processing!")
    sys.exit(1)

# Get latest row
latest = df.iloc[-1]
prev = df.iloc[-2] if len(df) > 1 else None

print("\n" + "="*60)
print("🤖 CURRENT TECHNICAL INDICATORS (LIVE)")
print("="*60)
print(f"Timestamp: {latest['timestamp']}")
print(f"BTC Price: ${latest['close']:.2f}")
print()

# Price Action
print("📊 PRICE ACTION:")
print(f"  Open:   ${latest['open']:.2f}")
print(f"  High:   ${latest['high']:.2f}")
print(f"  Low:    ${latest['low']:.2f}")
print(f"  Close:  ${latest['close']:.2f}")
if latest['volume'] > 0:
    print(f"  Volume: {latest['volume']:.4f} BTC")
else:
    print(f" Volume: No volume data (crypto bars may not have volume)")
print()

# RSI
print("⚡ RSI (Relative Strength Index):")
rsi = latest['rsi']
if rsi > 70:
    rsi_signal = "🔴 OVERBOUGHT - Bearish signal"
    rsi_verdict = "BEARISH"
elif rsi > 60:
    rsi_signal = "🟡 Strong - Leaning bearish"
    rsi_verdict = "BEARISH"
elif rsi > 40:
    rsi_signal = "🟢 Neutral zone"
    rsi_verdict = "NEUTRAL"
elif rsi > 30:
    rsi_signal = "🟡 Weak - Leaning bullish"
    rsi_verdict = "BULLISH"
else:
    rsi_signal = "🔴 OVERSOLD - Bullish signal"
    rsi_verdict = "BULLISH"

print(f"  Value: {rsi:.2f}")
print(f"  Signal: {rsi_signal}")
print(f"  Verdict: {rsi_verdict}")
print()

# MACD
print("🔵 MACD (Momentum Indicator):")
macd = latest['macd']
macd_signal = latest['macd_signal']
macd_hist = macd - macd_signal

if macd > macd_signal:
    macd_trend = "🟢 BULLISH - MACD above signal"
    macd_verdict = "BULLISH"
else:
    macd_trend = "🔴 BEARISH - MACD below signal"
    macd_verdict = "BEARISH"

macd_strength = "Strong" if abs(macd_hist) > 50 else "Weak"

print(f"  MACD Line:      {macd:.2f}")
print(f"  Signal Line:    {macd_signal:.2f}")
print(f"  Histogram:      {macd_hist:.2f}")
print(f"  Crossover:      {macd_trend}")
print(f"  Strength:       {macd_strength}")
print(f"  Verdict:        {macd_verdict}")
print()

# Bollinger Bands
print("📏 BOLLINGER BANDS (Volatility Bands):")
bb_upper = latest['bb_upper']
bb_middle = (bb_upper + latest['bb_lower']) / 2
bb_lower = latest['bb_lower']
bb_width = bb_upper - bb_lower
price = latest['close']

# Calculate position within bands
bb_position_pct = ((price - bb_lower) / (bb_upper - bb_lower)) * 100

if price > bb_upper:
    bb_position = "🔴 Above upper band - Overbought"
    bb_verdict = "BEARISH"
elif bb_position_pct > 70:
    bb_position = "🟡 Upper region - Strong buying"
    bb_verdict = "NEUTRAL"
elif bb_position_pct > 30:
    bb_position = "🟢 Middle region - Neutral"
    bb_verdict = "NEUTRAL"
elif price < bb_lower:
    bb_position = "🔴 Below lower band - Oversold"
    bb_verdict = "BULLISH"
else:
    bb_position = "🟡 Lower region - Weak price"
    bb_verdict = "NEUTRAL"

print(f"  Upper Band:  ${bb_upper:.2f}")
print(f"  Middle:      ${bb_middle:.2f}")
print(f"  Lower Band:  ${bb_lower:.2f}")
print(f"  Band Width:  ${bb_width:.2f}")
print(f"  Price Position: {bb_position_pct:.1f}% ({bb_position})")
print(f"  Verdict: {bb_verdict}")
print()

# ATR (Volatility)
print("💨 ATR (Average True Range - Volatility):")
atr = latest['atr']
atr_pct = (atr / price) * 100
print(f"  Value: ${atr:.2f} ({atr_pct:.2f}% of price)")
if atr_pct > 2:
    print(f"  Verdict: HIGH volatility (risky for scalping)")
elif atr_pct > 1:
    print(f"  Verdict: MODERATE volatility")
else:
    print(f"  Verdict: LOW volatility (good for scalping)")
print()

# Summary
print("="*60)
print("📊 VERDICT SUMMARY")
print("="*60)

bearish_count = 0
bullish_count = 0
neutral_count = 0

verdicts = {
    'RSI': rsi_verdict,
    'MACD': macd_verdict,
    'Bollinger Bands': bb_verdict
}

for indicator, verdict in verdicts.items():
    icon = "🔴" if verdict == "BEARISH" else ("🟢" if verdict == "BULLISH" else "🟡")
    print(f"{icon} {indicator}: {verdict}")
    if verdict == "BEARISH":
        bearish_count += 1
    elif verdict == "BULLISH":
        bullish_count += 1
    else:
        neutral_count += 1

print()
print(f"Total: {bullish_count} Bullish | {neutral_count} Neutral | {bearish_count} Bearish")
print()

if bearish_count >= 2:
    overall = "🔴 OVERALL BEARISH - Model predicting low confidence"
    explain = "The ML model is correctly reading bearish signals. It's waiting for conditions to improve."
elif bullish_count >= 2:
    overall = "🟢 OVERALL BULLISH - Model should trigger buy soon (>0.52)"
    explain = "Conditions are favorable. Watch for ML prediction to cross 0.52 threshold."
else:
    overall = "🟡 OVERALL NEUTRAL - Mixed signals, model in wait mode"
    explain = "Market is choppy. Model waiting for clearer directional signals."

print(overall)
print(f"Explanation: {explain}")
print("="*60)
