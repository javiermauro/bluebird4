---
name: protect-idempotency
enabled: true
event: file
pattern: (applied_order_ids|register_pending_order|remove_pending_order)
action: warn
---

**IDEMPOTENCY SYSTEM MODIFICATION DETECTED**

You are editing the order idempotency system in `grid_trading.py` or `bot_grid.py`.

**This system prevents duplicate order fills:**
```python
if order_id in self.applied_order_ids:
    return None  # Already applied - skip!
```

**Order flow must be:**
1. `register_pending_order()` - before submitting
2. `submit_order()` - send to Alpaca
3. `apply_filled_order()` - check idempotency, then apply
4. `remove_pending_order()` - cleanup after fill

**Breaking idempotency causes:**
- Same fill applied twice = 2x position size
- Profit calculations doubled
- Grid state corrupted

**Never remove these checks "for performance."**
