#!/usr/bin/env python3
"""
Quick script to close all positions via Alpaca API
"""
import os
from dotenv import load_dotenv
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

load_dotenv()

API_KEY = os.getenv('ALPACA_API_KEY')
SECRET_KEY = os.getenv('ALPACA_SECRET_KEY')

client = TradingClient(API_KEY, SECRET_KEY, paper=True)

# Get all positions
positions = client.get_all_positions()

if not positions:
    print("No open positions to close.")
else:
    print(f"Found {len(positions)} position(s). Closing all...")
    
    for position in positions:
        symbol = position.symbol
        qty = abs(float(position.qty))
        side = position.side
        
        print(f"Closing {qty} {symbol} ({side})...")
        
        # Close position by placing opposite side order
        close_side = OrderSide.SELL if side == 'long' else OrderSide.BUY
        
        req = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=close_side,
            time_in_force=TimeInForce.GTC
        )
        
        order = client.submit_order(req)
        print(f"✅ Closed {symbol}: Order ID {order.id}")
    
    print(f"\nAll positions liquidated!")
