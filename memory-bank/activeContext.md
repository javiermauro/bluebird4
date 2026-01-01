# Active Context — Current Focus

## Now
- [2026-01-01 17:20] **ALL 9 PROTECTION SYSTEMS VERIFIED** - Comprehensive verification: Windfall (20 captures, $864.52), Limit Orders (active), Fast Fill (10s checks), Time Filters, Grid Rebalancing, Consecutive Down Bars, Developing Downtrend, Risk Overlay (NORMAL), Orchestrator (LTC at GRID_REDUCED 111%).
- [2026-01-01 15:42] **SYSTEM CRASH RECOVERED** - Mac crashed, bot died at 15:42. Restarted successfully. All services healthy on ports 8000, 5173.
- [2026-01-01 15:17] **WINDFALL CAPTURED $368.07** - AVAX/USD windfall sell triggered at 9.78% unrealized. Total windfall profit today: $368.09.
- [2026-01-01 14:00] **DOGE DISPLAY FIX** - Fixed DOGE not appearing in dashboard orchestrator. Added pre-registration of all config symbols in orchestrator.py and multi_asset initialization in server.py. Commit: `ee3fbc3`.
- [2026-01-01 10:50] **DOGE/USD ADDED TO GRID TRADING** - Highest volatility (10.3% 7D range, 2.6% daily vol). Conservative 15% allocation for meme coin. New allocation: SOL 35%, LTC 25%, AVAX 25%, DOGE 15%.
- [2025-12-31 16:40] **BTC/USD REMOVED FROM GRID TRADING** - BTC underperformed by 5.7x ($763 vs $4,363 SOL). Removed from SYMBOLS/GRID_CONFIGS.
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
- **Bot**: Healthy, NORMAL mode, running on port 8000
- **Dashboard**: Running on port 5173
- **Notifier**: Running, watchdog active
- **Orchestrator**: ENFORCING, LTC/USD at GRID_REDUCED (111% inventory), others GRID_FULL
- **Risk Overlay**: NORMAL mode, all triggers inactive
- **State Files**: Persistent in `data/state/` (survives reboot)
- **Database**: Healthy, reconciled with Alpaca

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

## Key Metrics (Jan 1, 2026 - Updated)
| Metric | Value |
|--------|-------|
| Start Equity (Dec 1) | $90,457.58 |
| Current Equity | ~$103,000 |
| Grid Profit (All-Time) | +$12,555.74 |
| Windfall Captures | 20 captures, $864.52 |
| **Current Positions** | LTC/USD (111% inv), others minimal |
| **Protection Systems** | 9/9 verified working |
| **Test Status** | **101/101 PASS** |

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
