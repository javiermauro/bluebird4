---
name: protect-state-files
enabled: true
event: bash
pattern: rm.*(bluebird-grid-state|bluebird-circuit-breaker|bluebird-daily-equity|bluebird-risk-state)\.json
action: block
---

**STATE FILE DELETION BLOCKED**

You attempted to delete critical state files:

| File | Contains |
|------|----------|
| `bluebird-grid-state.json` | Grid levels, pending orders, fill history |
| `bluebird-circuit-breaker.json` | Daily loss tracking, halt status |
| `bluebird-daily-equity.json` | Starting equity for P&L calculation |
| `bluebird-risk-state.json` | Position sizes, exposure limits |

**Deleting these causes:**
- Lost grid level tracking (orphaned orders)
- Circuit breaker resets (loses daily loss count)
- Daily P&L resets to zero

**If you need to reset state:**
1. Stop the bot first: `python start.py --stop`
2. Backup files before deleting
3. Restart bot to regenerate clean state
