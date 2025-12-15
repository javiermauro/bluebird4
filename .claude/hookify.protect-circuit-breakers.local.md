---
name: protect-circuit-breakers
enabled: true
event: file
pattern: (check_circuit_breakers|should_stop.*False|circuit_status\[.should_stop.\]\s*=\s*False)
action: warn
---

**CIRCUIT BREAKER MODIFICATION DETECTED**

You are editing the circuit breaker system in `bot_grid.py`.

**Circuit breakers protect your account by:**
- Halting trading when daily loss exceeds 5%
- Stopping if drawdown exceeds 10% from peak
- Triggering per-symbol stop-losses at -10%

**The check pattern should always be:**
```python
circuit_status = bot.check_circuit_breakers(equity)
if circuit_status['should_stop']:
    # HALT ALL TRADING
    return
```

**Never:**
- Remove the `check_circuit_breakers()` call
- Force `should_stop` to False
- Skip the halt when triggered

This is your account's emergency stop button.
