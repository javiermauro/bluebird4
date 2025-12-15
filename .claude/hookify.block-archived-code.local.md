---
name: block-archived-code
enabled: true
event: file
pattern: archive/old_bots/
action: block
---

**ARCHIVED CODE - DO NOT MODIFY**

The `archive/old_bots/` directory contains deprecated prediction-based bots:
- `main.py`, `bot.py`, `bot_multi.py`, `bot_ultra.py`
- `train*.py`, `backtest*.py`

**These were archived because:**
- ML prediction achieved only 21% win rate
- Grid trading (current system) works better in sideways markets
- No predictions needed

**The active system is:**
- `src/execution/bot_grid.py` - Grid trading bot
- `src/strategy/grid_trading.py` - Grid strategy

Do not restore, modify, or use archived code. It's kept for reference only.
