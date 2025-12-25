# Active Context — Current Focus

## Now
- [2025-12-25 11:55] **ALL 5 MAINTENANCE PHASES COMPLETE**
- [2025-12-25 11:55] **Maintenance Phase 5 Complete** - Log rotation daily at 5 AM
- [2025-12-25 11:53] **Maintenance Phase 4 Complete** - Database cleanup script (90-day retention)
- [2025-12-25 11:48] **Maintenance Phase 3 Complete** - Daily database backups at 3 AM
- [2025-12-25 11:42] **Maintenance Phase 2 Complete** - Bot watchdog with auto-restart on crash
- [2025-12-25 11:30] **Phase 1 Complete** - State files now in `data/state/` (survives reboot)

## Recent Developments
- [2025-12-25] Phase 5: Log rotation - `scripts/rotate_logs.sh` daily at 5 AM, 50 MB limit
- [2025-12-25] Phase 4: Database cleanup - `scripts/cleanup_db.py` with 90-day retention
- [2025-12-25] Phase 3: Database backup - daily at 3 AM, 7-day retention, integrity check
- [2025-12-25] Phase 2: Bot watchdog - auto-restart on crash via cron every 5 min
- [2025-12-25] Phase 1: State file persistence - moved 8 files from `/tmp` to `data/state/`
- [2025-12-23] Notification reliability overhaul: DB persistence, SMS retry, API resilience, watchdog cron
- [2025-12-21] Orchestrator meta-controller deployed and verified in production
- [2025-12-21] First real RISK_OFF trigger observed and handled correctly

## System Health
- **Bot**: Healthy, NORMAL mode, 100% position size, heartbeat updating every 60s
- **Notifier**: Running (PID in DB), heartbeat updating every 60s, watchdog active
- **Orchestrator**: ENFORCING, liquidation enabled, all symbols GRID_FULL
- **State Files**: Now persistent in `data/state/` (survives reboot)
- **Database Backup**: Daily at 3 AM, 7-day retention, first backup created (994 MB)

## Maintenance Plan (5 Phases) - ALL COMPLETE
| Phase | Status | Description |
|-------|--------|-------------|
| 1. State Persistence | ✅ DONE | Move `/tmp` state files to `data/state/` |
| 2. Bot Watchdog | ✅ DONE | Auto-restart bot on crash |
| 3. Database Backup | ✅ DONE | Daily automated backups at 3 AM |
| 4. DB Cleanup | ✅ DONE | Cleanup script with 90-day retention |
| 5. Log Rotation | ✅ DONE | Daily rotation at 5 AM, 50 MB limit |

## Active Cron Jobs
| Schedule | Script | Purpose |
|----------|--------|---------|
| `*/5 * * * *` | `check_notifier.sh` | Notifier watchdog |
| `*/5 * * * *` | `check_bot.sh` | Bot watchdog |
| `0 3 * * *` | `backup_db.sh` | Database backup |
| `0 5 * * *` | `rotate_logs.sh` | Log rotation |

## Monitoring Commands
```bash
tail -f /tmp/bluebird-watchdog.log    # Watchdog logs
tail -f /tmp/bluebird-backup.log      # Backup logs
tail -f /tmp/bluebird-logrotate.log   # Rotation logs
python3 scripts/cleanup_db.py         # DB cleanup (dry run)
```

## Key Metrics (Dec 21)
| Metric | Value |
|--------|-------|
| Daily P/L | +$1,685 (+1.83%) |
| Grid P/L | +$3,341 (+3.70%) |
| All-Time P/L | -$6,383 (-6.38%) |
| Current Equity | $93,617 |
| Days to Breakeven | ~4 at current pace |
