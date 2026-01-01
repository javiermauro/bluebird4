# Active Context — Current Focus

## Now
- [2026-01-01 10:50] **DOGE/USD ADDED TO GRID TRADING** - Highest volatility (10.3% 7D range, 2.6% daily vol). Conservative 15% allocation for meme coin. New allocation: SOL 35%, LTC 25%, AVAX 25%, DOGE 15%.
- [2025-12-31 16:40] **BTC/USD REMOVED FROM GRID TRADING** - BTC underperformed by 5.7x ($763 vs $4,363 SOL). Removed from SYMBOLS/GRID_CONFIGS.
- [2025-12-31 16:00] **TIMEZONE + CONFIG BUGS FIXED** - Fixed naive datetime in warmup (5-hour offset). Fixed UltraConfig instantiation spam (12 per request → singleton). Fixed TIMEFRAME config "5Min" → "1Min".
- [2025-12-30 00:00] **BOT LAUNCHAGENT CREATED** - Created `com.bluebird.bot.plist` with `RunAtLoad=true` and `KeepAlive=true`. This is the proper macOS way to manage the bot daemon. The watchdog-started bot was dying due to shell signal propagation issues when the watchdog script exited. Now launchd directly manages the bot, and the watchdog serves as a backup health monitor.
- [2025-12-29 23:45] **WATCHDOG LSOF BUG FIX** - Fixed `lsof -ti :8000` killing notifier (had outgoing connections to bot). Changed to `lsof -ti TCP:8000 -sTCP:LISTEN` to only kill processes LISTENING on port 8000.
- [2025-12-27] **TIER-CORRECT FEE MODELING COMPLETE** - Volume-based Alpaca crypto fee tiers, Gross vs Net P&L
- [2025-12-26 06:35] **WATCHDOG LAUNCHD MIGRATION COMPLETE** - Fixed EPERM on external volume, durable local state
- [2025-12-26 01:45] **TIMEOUT HARDENING COMPLETE** - All main loop Alpaca calls bounded with 10-15s timeouts
- [2025-12-25 18:30] **PHASE A ROBUSTNESS COMPLETE** - Crash loop detection, atomic writes, disk monitoring
- [2025-12-25 11:55] **ALL 5 MAINTENANCE PHASES COMPLETE**
- [2025-12-30 02:10] **LAUNCHD DB ACCESS HARDENING** - Watchdogs support per-machine config (`config.env`), bot DB path override via `BLUEBIRD_DB_PATH`, notifier watchdog fallback when DB unreadable
- [2025-12-29 22:18] **INTERNAL IS SINGLE SOURCE OF TRUTH** - One final rsync `/Volumes/DOCK/BLUEBIRD 4.0/` → `~/BLUEBIRD/bluebird/`, watchdog scripts re-synced, bot restarted from internal, `/health` OK and monitor READY
- [2025-12-29 22:36] **FASTER UNATTENDED RECOVERY** - Watchdog LaunchAgents now run every 60s (was 300s) and stale heartbeat threshold reduced to 120s (was 300s) for faster bot/notifier restarts after reboot
- [2025-12-29 22:48] **WATCHDOG ANTI-FLAP FIX (BOT)** - Bot watchdog now uses `/health` as the primary liveness signal and will restart if `/health` is unreachable even if DB heartbeat looks recent; heartbeat threshold returned to 300s to avoid self-inflicted restarts
- [2025-12-29 23:05] **MONITOR UI REDESIGN** - `scripts/monitor_services.sh` now prints one line per component (bot API, bot listener, dashboard HTTP, each launchd job) with explicit YES/NO + WAITING list; added `--once` + `--no-clear` for easy snapshots/logging
- [2025-12-29 23:08] **AUTO MONITOR SNAPSHOTS ON RESTART** - Added `scripts/sync-monitor-scripts.sh` which installs `com.bluebird.monitor-status` LaunchAgent (60s interval) writing snapshots to `/tmp/bluebird-monitor-status.log` for post-reboot verification

## Recent Developments
- [2025-12-27] **Tier-Correct Fee Modeling**: Implemented 8-tier Alpaca crypto fee structure with 3am ET day bucketing. Dashboard now shows Gross vs Net P&L with fee tier card. Run `python -m src.utils.backfill_fees` to backfill historical trades.
- [2025-12-26] **Watchdog launchd Migration**: Fixed EPERM errors by moving watchdog scripts to local filesystem. Uses `~/Library/Application Support/BLUEBIRD/` for scripts and durable state. Run `scripts/sync-watchdog-scripts.sh` after editing repo watchdog scripts.
- [2025-12-26] **Timeout Hardening**: Prevents event loop hangs when Alpaca API is slow. All main loop Alpaca calls wrapped with `run_blocking_with_timeout()` (10-15s). Added `run_sync_with_timeout()` for sync contexts.
- [2025-12-25] **Phase A Robustness**: Crash loop detection (3 in 30 min = pause), atomic JSON writes, disk monitoring (90% alert)
- [2025-12-30 02:10] **Launchd DB Access Issue (External Volume)**: Observed `sqlite3 ... authorization denied` when watchdog runs under launchd against `/Volumes/DOCK/.../data/bluebird.db`. Implemented: (a) watchdog config file `~/Library/Application Support/BLUEBIRD/config.env` (project + optional DB override), (b) `/health`/process fallback behavior when DB unreadable, (c) bot supports `BLUEBIRD_DB_PATH` to run DB on internal disk for unattended reliability.
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

## Current Configuration (Jan 1, 2026)
| Setting | Value |
|---------|-------|
| **Symbols** | SOL/USD, LTC/USD, AVAX/USD, DOGE/USD |
| **Allocation** | SOL 35%, LTC 25%, AVAX 25%, DOGE 15% |
| **MAX_POSITIONS** | 4 |
| **TIMEFRAME** | 1Min |
| **BTC/USD** | REMOVED Dec 31 (5.7x underperformer) |
| **DOGE/USD** | ADDED Jan 1 (highest volatility) |

## Key Metrics (Dec 28 - Updated)
| Metric | Value |
|--------|-------|
| Start Equity (Dec 1) | $90,457.58 |
| Current Equity (Gross) | $99,879.30 |
| Gross P/L | +$9,421.72 (+10.4%) |
| Fees Expected | $1,210.48 |
| Fees Conservative | $2,312.21 |
| **Net P/L (Expected)** | **+$8,211.24 (+9.1%)** |
| **Net P/L (Conservative)** | **+$7,109.47 (+7.9%)** |
| 30-Day Volume | $1,095,209 |
| Current Fee Tier | Tier 4 (0.08%/0.18%) |
| **Test Status** | **25/25 PASS** |

## Fee Modeling - Live
| Component | Status |
|-----------|--------|
| Fee Tier Engine | ✅ `src/utils/crypto_fee_tiers.py` |
| Backfill Script | ✅ `python -m src.utils.backfill_fees` |
| DB Columns | ✅ 6 fee audit columns in trades |
| API Endpoints | ✅ `/api/profitability-report`, fee arrays in equity |
| Dashboard | ✅ Fee tier card, 3-line chart, warning banner |

## Robustness Phase A - Complete
| Feature | Implementation |
|---------|---------------|
| Crash Loop Detection | Pause after 3 restarts in 30 min |
| Atomic JSON Writes | temp file + fsync + os.replace |
| Disk Monitoring | Alert at 90% capacity |
| Pending Alerts | Notifier processes on startup |

## Phase B (EUPHORIA Gate) - Deferred
Designed but not implemented. Would block upward grid rebalances during parabolic pumps (RSI > 75 + momentum > 2.5%). See plan at `/Users/javierrodriguez/.claude/plans/harmonic-herding-brook.md`.
