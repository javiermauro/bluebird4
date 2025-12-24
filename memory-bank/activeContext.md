# Active Context — Current Focus

## Now
- [2025-12-23 10:35] **Notification system overhauled** - database-backed, retry logic, watchdog monitoring
- [2025-12-23 10:35] **Notifier running healthy** - heartbeat updating, 20 trade IDs persisted

## Recent Developments
- [2025-12-23] Notification reliability overhaul: DB persistence, SMS retry, API resilience, watchdog cron
- [2025-12-23] Fixed notifier that was down 26+ hours due to file permission error
- [2025-12-21] Orchestrator meta-controller deployed and verified in production
- [2025-12-21] First real RISK_OFF trigger observed and handled correctly

## System Health
- **Bot**: Healthy, NORMAL mode, 100% position size
- **Notifier**: Running (PID in DB), heartbeat updating every 60s, watchdog active
- **Orchestrator**: ENFORCING, liquidation enabled, all symbols GRID_FULL
- **Database**: 4 new notification tables active (`sms_history`, `notified_trade_ids`, `sms_queue`, `notifier_status`)

## Risks / Gaps
- **Grid config pending**: New spacing/levels saved but grids using old state until next rebalance
- **Orchestrator untested at high inventory**: Zero interventions so far (inventory hasn't hit 100%+)
- **Bot auto-restart**: Not configured (LaunchAgent) - notifier has watchdog, bot does not

## Next Steps (Short List)
- Monitor notifier watchdog logs: `tail -f /tmp/bluebird-watchdog.log`
- Consider adding bot watchdog similar to notifier
- Watch for orchestrator interventions when inventory hits thresholds

## Key Metrics (Dec 21)
| Metric | Value |
|--------|-------|
| Daily P/L | +$1,685 (+1.83%) |
| Grid P/L | +$3,341 (+3.70%) |
| All-Time P/L | -$6,383 (-6.38%) |
| Current Equity | $93,617 |
| Days to Breakeven | ~4 at current pace |
