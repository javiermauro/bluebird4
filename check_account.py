#!/usr/bin/env python3
from alpaca.trading.client import TradingClient
import os
from dotenv import load_dotenv

load_dotenv()
client = TradingClient(os.getenv('ALPACA_API_KEY'), os.getenv('ALPACA_SECRET_KEY'), paper=True)

positions = client.get_all_positions()
print(f'Open Positions: {len(positions)}')
for p in positions:
    print(f'  {p.symbol}: {p.qty} @ ${p.avg_entry_price} | P&L: ${p.unrealized_pl}')

account = client.get_account()
print(f'\nAccount Equity: ${float(account.equity):.2f}')
print(f'Buying Power: ${float(account.buying_power):.2f}')
print(f'Cash Balance: ${float(account.cash):.2f}')
