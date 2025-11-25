#!/usr/bin/env python3
"""
Calculate trading costs and profitability for scalping strategy
"""

# Account Details
EQUITY = 97947.88
POSITION_SIZE_PCT = 0.10  # 10% per trade in scalping mode
BTC_PRICE = 88000  # Approximate current price

# Alpaca Crypto Fee Structure (as of 2024)
# https://alpaca.markets/support/crypto-fees-on-alpaca/
MAKER_FEE = 0.0025  # 0.25% - when you add liquidity (limit orders)
TAKER_FEE = 0.0025  # 0.25% - when you take liquidity (market orders)

# We use MARKET ORDERS, so we're always a TAKER
FEE_RATE = TAKER_FEE

# Scalping Parameters
TAKE_PROFIT_PCT = 0.006  # 0.6%
STOP_LOSS_PCT = 0.003    # 0.3%

print("="*70)
print("💰 TRADING COST ANALYSIS - SCALPING MODE")
print("="*70)
print()

# Position Size
position_value = EQUITY * POSITION_SIZE_PCT
btc_qty = position_value / BTC_PRICE

print("📊 POSITION DETAILS:")
print(f"  Account Equity:        ${EQUITY:,.2f}")
print(f"  Position Size:         {POSITION_SIZE_PCT*100}% = ${position_value:,.2f}")
print(f"  BTC Quantity:          {btc_qty:.6f} BTC")
print()

# Fee Calculation
entry_fee = position_value * FEE_RATE
exit_fee = position_value * FEE_RATE
total_fee_per_trade = entry_fee + exit_fee

print("💸 ALPACA FEES (Market Orders):")
print(f"  Fee Rate:              {FEE_RATE*100}% per side (taker)")
print(f"  Entry Fee (BUY):       ${entry_fee:.2f}")
print(f"  Exit Fee (SELL):       ${exit_fee:.2f}")
print(f"  Total Per Round Trip:  ${total_fee_per_trade:.2f}")
print(f"  Fee as % of Position:  {(total_fee_per_trade/position_value)*100:.3f}%")
print()

# Profitability Analysis
print("="*70)
print("📈 PROFITABILITY SCENARIOS")
print("="*70)
print()

# Scenario 1: Take Profit Hit
tp_profit_gross = position_value * TAKE_PROFIT_PCT
tp_profit_net = tp_profit_gross - total_fee_per_trade
tp_roi = (tp_profit_net / position_value) * 100

print("✅ SCENARIO 1: TAKE PROFIT HIT (+0.6%)")
print(f"  Entry Price:           ${BTC_PRICE:,.2f}")
print(f"  Exit Price:            ${BTC_PRICE * (1 + TAKE_PROFIT_PCT):,.2f}")
print(f"  Gross Profit:          ${tp_profit_gross:.2f}")
print(f"  Trading Fees:          -${total_fee_per_trade:.2f}")
print(f"  Net Profit:            ${tp_profit_net:.2f}")
print(f"  ROI (Net):             {tp_roi:.3f}%")
print()

# Scenario 2: Stop Loss Hit
sl_loss_gross = position_value * STOP_LOSS_PCT
sl_loss_net = sl_loss_gross + total_fee_per_trade
sl_roi = -(sl_loss_net / position_value) * 100

print("❌ SCENARIO 2: STOP LOSS HIT (-0.3%)")
print(f"  Entry Price:           ${BTC_PRICE:,.2f}")
print(f"  Exit Price:            ${BTC_PRICE * (1 - STOP_LOSS_PCT):,.2f}")
print(f"  Gross Loss:            -${sl_loss_gross:.2f}")
print(f"  Trading Fees:          -${total_fee_per_trade:.2f}")
print(f"  Net Loss:              -${sl_loss_net:.2f}")
print(f"  ROI (Net):             {sl_roi:.3f}%")
print()

# Risk/Reward Ratio
rr_ratio = tp_profit_net / sl_loss_net

print("⚖️  RISK/REWARD RATIO (After Fees):")
print(f"  Risk (Stop Loss):      ${sl_loss_net:.2f}")
print(f"  Reward (Take Profit):  ${tp_profit_net:.2f}")
print(f"  Ratio:                 1:{rr_ratio:.2f}")
print()

# Win Rate Needed to Break Even
# breakeven = (1 / (1 + RR)) * 100
breakeven_winrate = (1 / (1 + rr_ratio)) * 100

print(f"  Breakeven Win Rate:    {breakeven_winrate:.1f}%")
print(f"  Recommended Win Rate:  >{breakeven_winrate + 10:.1f}% (for profitability)")
print()

# Daily Trading Cost Scenarios
print("="*70)
print("📅 DAILY COST SCENARIOS")
print("="*70)
print()

scenarios = [
    ("Conservative (10 trades/day)", 10),
    ("Moderate (30 trades/day)", 30),
    ("Aggressive (50 trades/day)", 50),
]

for name, num_trades in scenarios:
    daily_fees = total_fee_per_trade * num_trades
    monthly_fees = daily_fees * 21  # Trading days per month
    daily_fees_pct = (daily_fees / EQUITY) * 100
    
    print(f"{name}:")
    print(f"  Daily Fees:            ${daily_fees:.2f} ({daily_fees_pct:.2f}% of equity)")
    print(f"  Monthly Fees:          ${monthly_fees:,.2f}")
    print()

# Profitability Example (Realistic Win Rate)
print("="*70)
print("💵 EXPECTED PROFITABILITY (60% Win Rate)")
print("="*70)
print()

WIN_RATE = 0.60
trades_per_day = 30

wins_per_day = trades_per_day * WIN_RATE
losses_per_day = trades_per_day * (1 - WIN_RATE)

daily_profit = (wins_per_day * tp_profit_net) - (losses_per_day * sl_loss_net)
daily_roi = (daily_profit / EQUITY) * 100
monthly_profit = daily_profit * 21

print(f"📊 ASSUMPTIONS:")
print(f"  Trades Per Day:        {trades_per_day}")
print(f"  Win Rate:              {WIN_RATE*100}%")
print(f"  Wins Per Day:          {wins_per_day:.0f}")
print(f"  Losses Per Day:        {losses_per_day:.0f}")
print()

print(f"💰 EXPECTED RETURNS:")
print(f"  Daily Profit:          ${daily_profit:.2f} ({daily_roi:.2f}% ROI)")
print(f"  Monthly Profit:        ${monthly_profit:,.2f} ({daily_roi*21:.1f}% ROI)")
print(f"  Annual Projection:     ${monthly_profit*12:,.2f} ({daily_roi*252:.0f}% ROI)")
print()

# Warning about fees
print("="*70)
print("⚠️  IMPORTANT NOTES")
print("="*70)
print()
print("1. FEES MATTER: At 0.5% per round trip, you need wins to be 2x larger")
print("   than losses just to break even at 50% win rate.")
print()
print("2. SLIPPAGE: Market orders may have additional slippage costs (1-5 basis points)")
print("   in volatile conditions.")
print()
print("3. ALPACA PAPER: Paper trading has $0 fees. LIVE trading will have these costs.")
print()
print("4. WIN RATE: Your ML model needs >35% win rate to be profitable after fees.")
print()
print("5. OPTIMIZATION: Consider using LIMIT orders (0.25% maker fee) instead of")
print("   market orders to reduce costs, though this adds execution risk.")
print()
print("="*70)
