# Phase 6: Error Rate Monitoring - Quick Start

## ✅ Status: IMPLEMENTED

Error rate monitoring is now active and will automatically alert if the bot logs >= 10 errors/hour.

## How It Works

Every 60 seconds, the notifier:
1. Parses `/tmp/bluebird-live-bot.log` for ERROR entries
2. Counts errors in the last hour
3. Alerts if count >= 10 (normal operation: 0-2 errors/hour)
4. Won't re-alert for 2 hours (prevents SMS spam)

## What You'll See

### Normal Operation
No alerts. Error rate monitoring runs silently in background.

Check notifier log:
```bash
tail -f /tmp/bluebird-live-notifier.log
# Look for: "Error rate normalized: X/hr" if previously alerted
```

### High Error Rate Alert (SMS)
```
[LIVE] 🚨 HIGH ERROR RATE

15 errors in last hour!
Threshold: 10/hr

Top sources:
GridBot:8, AlpacaClient:4, API:3

Recent samples:
  GridBot: NameError: name 'price_level' is not defined
  AlpacaClient: JSONDecodeError: Expecting value
  API: KeyError: 'unrealized_pl'

Check logs:
tail -100 /tmp/bluebird-live-bot.log
```

## Testing

### Quick Test (Inject Errors)
```bash
cd /Users/javierrodriguez/BLUEBIRD/bluebird-live

# Inject 15 test errors
python3 test_error_rate.py

# Wait for next poll cycle (60s)
sleep 70

# Check if alert was triggered
tail -30 /tmp/bluebird-live-notifier.log | grep "High error rate"

# Clean up test errors
grep -v 'TEST ERROR INJECTION' /tmp/bluebird-live-bot.log > /tmp/temp.log && mv /tmp/temp.log /tmp/bluebird-live-bot.log
```

### Verify State Persistence
```bash
# After triggering alert above, check state file
cat data/state/monitoring-state.json | python3 -m json.tool | grep -A 5 "error_rate"

# Should see:
# "error_rate": {
#   "alert_sent": true,
#   "last_alert_time": "2026-01-27T...",
#   "error_window": [ ... ]
# }
```

## Monitoring

### Check Current Error Rate
```bash
# Count errors in last hour (rough estimate)
tail -10000 /tmp/bluebird-live-bot.log | grep -E " - (ERROR|CRITICAL) - " | tail -20
```

### View Error Rate Status
```bash
# Check if error rate alert is active
cat data/state/monitoring-state.json | python3 -c "
import sys, json
state = json.load(sys.stdin)
er = state.get('error_rate', {})
if er.get('alert_sent'):
    print(f\"⚠️  Error rate alert ACTIVE\")
    print(f\"   Errors in window: {len(er.get('error_window', []))}\")
    print(f\"   Last alert: {er.get('last_alert_time')}\")
else:
    print('✓ No error rate alerts')
"
```

### Check Notifier Health
```bash
# Verify notifier is running and checking errors
tail -50 /tmp/bluebird-live-notifier.log | grep -E "(error rate|Error rate)"

# Should see periodic checks or status messages
```

## Configuration

Edit `/Users/javierrodriguez/BLUEBIRD/bluebird-live/src/notifications/notifier.py`:

```python
# Line 78-79
ERROR_RATE_THRESHOLD_PER_HOUR = 10  # Increase to reduce sensitivity
ERROR_RATE_GRACE_PERIOD_HOURS = 2    # Increase to reduce alert frequency
```

After changes:
```bash
python3 start.py --stop
python3 start.py --all
```

## What Errors Are Tracked?

**Included:**
- ERROR level log entries
- CRITICAL level log entries

**Excluded:**
- WARNING level (too noisy)
- INFO/DEBUG level (not actionable)
- Benign patterns: "connection limit exceeded" (Alpaca WebSocket)

To exclude more patterns, edit `src/utils/log_parser.py`:
```python
BENIGN_PATTERNS = [
    r'connection limit exceeded',
    r'your pattern here',  # Add more as needed
]
```

## Common Scenarios

### Scenario 1: Real Bug Detected
1. Receive SMS alert
2. Check logs: `tail -100 /tmp/bluebird-live-bot.log`
3. Identify root cause (e.g., NameError, KeyError)
4. Fix code and deploy
5. Error rate will normalize automatically
6. Alert clears, state resets

### Scenario 2: Transient Issue (Alpaca API)
1. Receive SMS alert
2. Check logs - see API errors
3. Verify bot still trading normally
4. Wait 2 hours (grace period prevents spam)
5. If API recovers, error rate normalizes

### Scenario 3: False Positive
1. Receive SMS alert
2. Check logs - errors are benign
3. Add pattern to `BENIGN_PATTERNS` in log_parser.py
4. Restart notifier
5. Pattern will be excluded from future checks

### Scenario 4: Grace Period Active
1. Error rate high, SMS sent
2. More errors occur in next hour
3. NO new SMS (grace period: 2 hours)
4. Check state file to confirm alert active
5. Fix issue, wait for normalization

## Troubleshooting

### No Alerts Received (When Expected)
```bash
# 1. Check notifier is running
python3 start.py --status

# 2. Check bot log exists
ls -lh /tmp/bluebird-live-bot.log

# 3. Check for errors manually
grep -c "ERROR" /tmp/bluebird-live-bot.log

# 4. Check notifier log for error rate checks
tail -50 /tmp/bluebird-live-notifier.log | grep -i "error"

# 5. Verify Twilio config
python3 -c "
from src.notifications.config import NotificationConfig
config = NotificationConfig.from_env()
print(f'Twilio configured: {config.is_configured()}')
"
```

### Alert Spam (Too Many SMS)
Grace period should prevent this. If still occurring:
1. Check grace period is 2 hours (line 79)
2. Check state file is being persisted
3. Increase grace period or threshold

### Log File Too Large
```bash
# Check log size
ls -lh /tmp/bluebird-live-bot.log

# If > 100MB, rotate manually
mv /tmp/bluebird-live-bot.log /tmp/bluebird-live-bot.log.old
# Bot will create new log automatically
```

### Parser Performance Issues
Parser is capped at 100 errors and has timeout protection.
If concerned:
```bash
# Time the parser
time python3 -c "
from pathlib import Path
from datetime import datetime, timedelta
from src.utils.log_parser import parse_log_errors

log = Path('/tmp/bluebird-live-bot.log')
since = datetime.now() - timedelta(hours=1)
errors = parse_log_errors(log, since=since)
print(f'Found {len(errors)} errors')
"
# Should be < 100ms
```

## Files to Know

| File | Purpose |
|------|---------|
| `/tmp/bluebird-live-bot.log` | Bot log (read-only, parsed for errors) |
| `data/state/monitoring-state.json` | Persistent state (error_rate section) |
| `src/notifications/notifier.py` | Main notifier (check_error_rate method) |
| `src/utils/log_parser.py` | Log parsing utility |
| `test_error_rate.py` | Test script for injecting errors |

## Rollback (If Needed)

If error rate monitoring causes issues:

```bash
cd /Users/javierrodriguez/BLUEBIRD/bluebird-live

# 1. Comment out the check in notifier.py (line 1233)
# Find: self.check_error_rate()  # Error rate monitoring (Phase 6)
# Replace with: # self.check_error_rate()  # Error rate monitoring (Phase 6) - DISABLED

# 2. Restart notifier
python3 start.py --stop
python3 start.py --all

# 3. Verify disabled
tail -50 /tmp/bluebird-live-notifier.log
# Should NOT see error rate checks
```

To re-enable: uncomment the line and restart.

## Key Benefits

✅ **Catches silent failures** - NameError bug would alert after ~2 hours (vs 2,254 occurrences)
✅ **No bot changes** - Zero performance impact on trading
✅ **Restart resilient** - State survives crashes
✅ **No alert spam** - 2-hour grace period
✅ **Clear alerts** - Shows sources, samples, actionable steps
✅ **Low overhead** - ~10ms every 60s

## Support

Questions or issues? Check the implementation docs:
- `PHASE6_IMPLEMENTATION.md` - Full technical details
- `PHASE6_CHANGES.md` - Exact code changes
- `PHASE6_QUICKSTART.md` - This file

Or check the logs:
```bash
tail -f /tmp/bluebird-live-notifier.log  # Notifier activity
tail -f /tmp/bluebird-live-bot.log       # Bot errors
```
