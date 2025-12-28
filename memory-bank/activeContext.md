# Active Context — Current Focus

## Now
- [2025-12-27] **TIER-CORRECT FEE MODELING COMPLETE** - Volume-based Alpaca crypto fee tiers, Gross vs Net P&L
- [2025-12-26 06:35] **WATCHDOG LAUNCHD MIGRATION COMPLETE** - Fixed EPERM on external volume, durable local state
- [2025-12-26 01:45] **TIMEOUT HARDENING COMPLETE** - All main loop Alpaca calls bounded with 10-15s timeouts
- [2025-12-25 18:30] **PHASE A ROBUSTNESS COMPLETE** - Crash loop detection, atomic writes, disk monitoring
- [2025-12-25 11:55] **ALL 5 MAINTENANCE PHASES COMPLETE**

## Recent Developments
- [2025-12-27] **Tier-Correct Fee Modeling**: Implemented 8-tier Alpaca crypto fee structure with 3am ET day bucketing. Dashboard now shows Gross vs Net P&L with fee tier card. Run `python -m src.utils.backfill_fees` to backfill historical trades.
- [2025-12-26] **Watchdog launchd Migration**: Fixed EPERM errors by moving watchdog scripts to local filesystem. Uses `~/Library/Application Support/BLUEBIRD/` for scripts and durable state. Run `scripts/sync-watchdog-scripts.sh` after editing repo watchdog scripts.
- [2025-12-26] **Timeout Hardening**: Prevents event loop hangs when Alpaca API is slow. All main loop Alpaca calls wrapped with `run_blocking_with_timeout()` (10-15s). Added `run_sync_with_timeout()` for sync contexts.
- [2025-12-25] **Phase A Robustness**: Crash loop detection (3 in 30 min = pause), atomic JSON writes, disk monitoring (90% alert)
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

## Scheduled Tasks

### LaunchAgents (launchd - for watchdogs)
| Agent | Interval | Purpose |
|-------|----------|---------|
| `com.bluebird.watchdog-bot` | 5 min | Bot auto-restart |
| `com.bluebird.watchdog-notifier` | 5 min | Notifier auto-restart |

**Note**: Watchdogs use launchd (not cron) because macOS can't execute scripts on external APFS volumes from cron. Scripts live in `~/Library/Application Support/BLUEBIRD/`.

### Cron Jobs (remaining)
| Schedule | Script | Purpose |
|----------|--------|---------|
| `0 3 * * *` | `backup_db.sh` | Database backup |
| `0 5 * * *` | `rotate_logs.sh` | Log rotation |

## Monitoring Commands
```bash
# LaunchAgents status
launchctl list | grep bluebird

# Watchdog logs
tail -f /tmp/bluebird-watchdog.log

# Manual watchdog run
bash "$HOME/Library/Application Support/BLUEBIRD/run-check-bot.sh"

# Reset crash loop pause (after fixing root cause)
rm "$HOME/Library/Application Support/BLUEBIRD/state/crash-loop-bot.json"

# Sync watchdog scripts (after editing repo scripts)
bash scripts/sync-watchdog-scripts.sh

# Other logs
tail -f /tmp/bluebird-backup.log      # Backup logs
tail -f /tmp/bluebird-logrotate.log   # Rotation logs
python3 scripts/cleanup_db.py         # DB cleanup (dry run)
```

## Key Metrics (Dec 25)
| Metric | Value |
|--------|-------|
| Daily P/L | +$2,512 (+2.63%) |
| Grid P/L | +$7,696 (+8.52%) |
| All-Time P/L | -$2,028 (-2.03%) |
| Current Equity | ~$97,972 |
| Buys Avoided | 452 ($877K notional protected) |

## Robustness Phase A - Complete
| Feature | Implementation |
|---------|---------------|
| Crash Loop Detection | Pause after 3 restarts in 30 min |
| Atomic JSON Writes | temp file + fsync + os.replace |
| Disk Monitoring | Alert at 90% capacity |
| Pending Alerts | Notifier processes on startup |

## Phase B (EUPHORIA Gate) - Deferred
Designed but not implemented. Would block upward grid rebalances during parabolic pumps (RSI > 75 + momentum > 2.5%). See plan at `/Users/javierrodriguez/.claude/plans/harmonic-herding-brook.md`.
