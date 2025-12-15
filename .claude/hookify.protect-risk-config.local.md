---
name: protect-risk-config
enabled: true
event: file
pattern: (MAX_RISK_PER_TRADE|DAILY_LOSS_LIMIT|MAX_DRAWDOWN|GRID_STOP_LOSS_PCT)\s*=
action: warn
---

**RISK PARAMETER CHANGE DETECTED**

You are modifying critical risk controls in `config_ultra.py`:

| Parameter | Safe Range | Dangerous |
|-----------|------------|-----------|
| `MAX_RISK_PER_TRADE` | 0.01-0.02 (1-2%) | >0.05 (5%) |
| `DAILY_LOSS_LIMIT` | 0.03-0.05 (3-5%) | <0.02 or >0.10 |
| `MAX_DRAWDOWN` | 0.08-0.12 (8-12%) | >0.15 (15%) |
| `GRID_STOP_LOSS_PCT` | 0.08-0.12 | 0 (disabled) |

**Before changing:**
1. Calculate new max loss: `equity * new_value`
2. Ensure position sizes align with new limits
3. Test in paper trading first

These are your account's last line of defense.
