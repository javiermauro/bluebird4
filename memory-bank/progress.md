# Progress — Status & History

## Current Status
- [2025-12-25 11:55] System healthy, NORMAL mode, all protection layers active
- [2025-12-25 11:55] **ALL 5 MAINTENANCE PHASES COMPLETE**
- [2025-12-25 11:55] **Phase 5 Maintenance Complete**: Log rotation daily at 5 AM
- [2025-12-25 11:53] **Phase 4 Maintenance Complete**: Database cleanup script
- [2025-12-25 11:48] **Phase 3 Maintenance Complete**: Daily database backups at 3 AM
- [2025-12-25 11:42] **Phase 2 Maintenance Complete**: Bot watchdog with auto-restart
- [2025-12-25 11:30] **Phase 1 Maintenance Complete**: State files moved to persistent storage

## Recent Work (High Signal)

### Dec 25, 2025 — Maintenance Phase 5: Log Rotation
- **Goal**: Prevent unbounded log file growth
- **Changes**:
  - Created `scripts/rotate_logs.sh` log rotation script
  - Rotates logs when they exceed 50 MB
  - Keeps 3 compressed rotations per log file
  - Added cron job: `0 5 * * *` (daily at 5 AM)
- **Log Files Managed**:
  - `/tmp/bluebird-bot.log`
  - `/tmp/bluebird-notifier.log`
  - `/tmp/bluebird-watchdog.log`
  - `/tmp/bluebird-backup.log`
  - `/tmp/bluebird-cleanup.log`
- **Logs**: `/tmp/bluebird-logrotate.log`

### Dec 25, 2025 — Maintenance Phase 4: Database Cleanup
- **Goal**: Prevent unbounded database growth by cleaning old records
- **Changes**:
  - Created `scripts/cleanup_db.py` standalone cleanup script
  - Dry-run mode by default, `--execute` flag to actually delete
  - Cleans equity_snapshots (90 days) and sms_history (90 days)
  - Runs VACUUM after cleanup to reclaim disk space
- **Note**: Protected by `hookify.protect-database.local.md` hook - cleanup is manual/explicit
- **Usage**: `python3 scripts/cleanup_db.py --execute`
- **Cron** (optional): `0 4 * * 0` (weekly Sunday 4 AM)

### Dec 25, 2025 — Maintenance Phase 3: Database Backup
- **Goal**: Daily automated backups of `data/bluebird.db` (~1 GB)
- **Changes**:
  - Created `data/backups/` directory for backup storage
  - Created `scripts/backup_db.sh` backup script
  - Uses SQLite `.backup` command (safe while DB in use)
  - Automatic integrity check after backup
  - Keeps last 7 days of backups (auto-cleanup)
  - Added cron job: `0 3 * * * /bin/bash "...backup_db.sh"`
- **Testing**: Manual run created 994 MB backup, integrity check passed
- **Logs**: `/tmp/bluebird-backup.log`

### Dec 25, 2025 — Maintenance Phase 2: Bot Watchdog
- **Goal**: Auto-restart bot if it crashes (similar to existing notifier watchdog)
- **Changes**:
  - Added `bot_status` table to SQLite database
  - Added `update_bot_heartbeat()` and `get_bot_status()` to `src/database/db.py`
  - Added `bot_heartbeat_loop()` to `src/api/server.py` (60-second interval)
  - Created `scripts/check_bot.sh` watchdog script
  - Added cron job: `*/5 * * * * /bin/bash "...check_bot.sh"`
- **How It Works**:
  1. Bot writes heartbeat to database every 60 seconds
  2. Cron runs watchdog script every 5 minutes
  3. If heartbeat > 5 min old, script kills stale process and restarts bot
- **Testing**: Verified heartbeat updates, watchdog reports "OK: Bot is healthy"

### Dec 25, 2025 — Maintenance Phase 1: State File Persistence
- **Critical Fix**: State files moved from volatile `/tmp` to persistent `data/state/`
- **Problem**: System reboot would lose daily P&L tracking, risk overlay mode, circuit breaker flags
- **Solution**: All state files now in `data/state/` which survives reboot
- **Files Migrated** (8 files updated):
  - `src/execution/bot_grid.py` - daily equity, circuit breaker, windfall, alltime equity
  - `src/strategy/risk_overlay.py` - overlay state, command file
  - `src/strategy/orchestrator.py` - inventory episodes
  - `src/strategy/grid_trading.py` - grid state
  - `src/api/server.py` - API state file references
  - `src/notifications/notifier.py` - startup cooldown
  - `src/notifications/config.py` - settings file
- **State Files Now Persistent**:
  - `data/state/grid-state.json`
  - `data/state/risk-overlay.json`
  - `data/state/orchestrator.json`
  - `data/state/daily-equity.json`
  - `data/state/alltime-equity.json`
  - `data/state/circuit-breaker.json`
  - `data/state/windfall-log.json`
  - `data/state/watchdog.json`
- **Lock files remain in `/tmp/bluebird/`** (intentional - should clear on reboot)
- **Testing**: Bot restarted, all state restored correctly, trading continues

### Dec 23, 2025 — Notification System Reliability Overhaul
- **Critical Fix**: Notifier was down 26+ hours (file permission error) - fixed and restarted
- **Database Persistence**: All notification state now in SQLite (`data/bluebird.db`):
  - `sms_history` - Audit trail of all SMS sent
  - `notified_trade_ids` - Prevents duplicate alerts across restarts
  - `sms_queue` - Failed SMS retry queue
  - `notifier_status` - Heartbeat, status, API failure tracking
- **SMS Retry Logic**: 3 attempts with exponential backoff (5s, 10s, 20s), then queued for later
- **API Resilience**: Exponential backoff + circuit breaker (5 failures = SMS alert + 5min cooldown)
- **Watchdog Monitoring**: Cron job every 5 min checks DB heartbeat, auto-restarts if stale
- **Files Modified**: `src/database/db.py`, `src/notifications/notifier.py`, `scripts/check_notifier.sh`

### Dec 21, 2025 — Orchestrator Launch + Strong Performance
- **Orchestrator Go-Live (3 stages)**:
  1. Stage 1: Shadow mode (ENFORCE=False) - verified logs and API
  2. Stage 2: Enforce mode (ENFORCE=True) - verified blocking logic
  3. Stage 3: Liquidation enabled - full production mode
- **Dashboard Orchestrator Panel**: Added collapsible panel with mode badges, inventory gauges, per-symbol status cards, telemetry display. Teal/cyan color scheme.
- **Grid Config Tuning**:
  - BTC: 5 grids → 6 levels, 1.25% spacing
  - SOL: 5 grids → 6 levels, 1.30% spacing
  - LTC: 6 grids → 7 levels, 1.40% spacing (was 1.69%)
  - AVAX: 6 grids → 7 levels, 1.45% spacing (was 1.68%)
- **First RISK_OFF Event**: ADX downtrend (39.3) + correlation spike (0.91) triggered RISK_OFF for ~20 min. System recovered through RECOVERY stages back to NORMAL. Protected ~$200 of potential loss.
- **Performance Highlights**:
  - Daily P/L: +$1,685 (+1.83%)
  - Grid P/L: +$3,341 (+3.70% since Dec 2)
  - Fee efficiency: 46x profit/fee ratio
  - Cycle win rate: 100% (5/5 completed cycles profitable)

### Dec 20, 2025 — Observability & Crash Recovery
- Crash/outage review: ~23h downtime, limit orders filled while offline
- Observability upgrades: idempotent trade logging, positions_value snapshots
- DB reconciliation verified post-restart
- Decision: `orders` table is authoritative for fills

### Earlier — Foundation
- [2025-12-19] Memory Bank initialized
- [2025-12-02] Grid trading era began after prediction-based approach showed 21% win rate
- Risk Overlay (NORMAL/RISK_OFF/RECOVERY) implemented
- Downtrend protection (ADX 25-35 size reduction, consecutive down bars block)

## Known Issues / Follow-ups
- **P2**: New grid configs saved but grids using old state until price moves 3%+ (triggers rebalance)
- **P2**: Orchestrator hasn't been stress-tested with inventory >100% yet
- ~~P3: Log rotation~~ FIXED Dec 25 - daily rotation at 5 AM, 50 MB limit, 3 rotations
- ~~P3: Database cleanup for equity_snapshots~~ FIXED Dec 25 - cleanup script with 90-day retention
- ~~P3: Database backup~~ FIXED Dec 25 - daily backups at 3 AM, 7-day retention
- ~~P3: Bot auto-restart after reboot not configured~~ FIXED Dec 25 - bot watchdog with cron
- ~~P2: State files lost on reboot~~ FIXED Dec 25 - moved to `data/state/`
- ~~P2: Notifier state persistence~~ FIXED Dec 23 - now database-backed
- ~~P3: Notifier monitoring~~ FIXED Dec 23 - watchdog cron job active

## Performance Tracking
| Date | Daily P/L | Grid P/L | Notes |
|------|-----------|----------|-------|
| Dec 21 | +$1,685 (+1.83%) | +$3,341 | Best day, first RISK_OFF event |
| Dec 20 | ~+$200 | +$1,656 | Post-crash recovery |
| Dec 2 | Start | $0 | Grid trading era begins |
