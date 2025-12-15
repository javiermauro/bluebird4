---
name: warn-order-execution
enabled: true
event: file
pattern: (submit_order|alpaca_client\.py|place_order|create_order)
action: warn
---

**ORDER EXECUTION CODE MODIFICATION**

You are editing code that submits real orders to Alpaca.

**Critical files:**
- `src/execution/alpaca_client.py` - Direct API calls
- `src/execution/bot_grid.py` - Order logic

**Before modifying, verify:**
1. Order registration: `register_pending_order()` called before submit?
2. Idempotency: Duplicate fills prevented?
3. Error handling: Failed orders cleaned up?
4. Fill verification: `verify_order_fill()` called after submit?

**Common mistakes:**
- Calling `submit_order()` without registration = untracked orders
- Submitting in a loop without delays = API rate limits
- Not handling partial fills = incorrect position sizes

**Test changes in paper trading first.**
