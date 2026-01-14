# Active Context — Current Focus

## Reminders
- **[2026-01-21] WEEK 2 REVIEW** — Continue monitoring 2.7% grid spacing performance. Compare daily averages, fee ratios, and total equity growth.

## Week 1 Analysis Results (Jan 14, 2026)
**Decision**: CONTINUE current config (AVAX 90%, LTC 10%, 2.7% spacing)

**Key Findings**:
- Net profit since config change (Jan 8): **+$15.13** (+0.76% on $2K)
- Fee ratio improved: **21%** of gross (was ~50% with 1.5% spacing)
- Daily average: ~$2.50/day
- AVAX generated 90% of profit (matches allocation)

**Why Keep LTC at 10%**:
- LTC is used for **correlation-based risk management**, not profit
- When AVAX-LTC correlation > 0.85 → position sizes reduced automatically
- When correlation > 0.90 → can trigger RISK_OFF mode
- Without 2nd pair: correlation features disabled, weaker crash protection
- $0.85/cycle from LTC is worth it for the risk signal it provides

## Now
- [2026-01-14 14:20] **PAPER BOT SERVICES DISABLED** - Stopped all paper bot services (port 8000) and disabled launchd agents permanently. Moved 7 plist files to `~/Library/LaunchAgents/disabled/`. Only live bot (port 8001) remains active.
- [2026-01-14 07:15] **CIRCUIT BREAKER SANITY CHECKS** - Added protection against false circuit breaker triggers from API timeouts. When Alpaca API times out and returns equity=$0, we now skip circuit breaker evaluation instead of triggering 100% drawdown halt. Also skips if equity drops >50% in single check.
- [2026-01-14 07:00] **ORPHAN ORDER AUTO-CANCEL** - Added `CANCEL_ORPHAN_ORDERS_ON_HEALTH_CHECK = True` to config. Health checks now auto-cancel orphan orders on Alpaca that we're not tracking internally. Also cleans up stale tracking entries.
- [2026-01-12 10:20] **TIME FILTER DISABLED** - Set `USE_TIME_FILTER = False` to allow 24/7 trading. Time filter was blocking ~6 hours/day during "low liquidity" windows (22-24 UTC, 5-7 UTC), severely limiting grid activity. Only 2 cycles completed in 5 days due to restrictions.
- [2026-01-12 10:10] **FALSE HALT BUG FIXED** - Corrupted `peak_equity: $90,000` in daily-equity.json caused false 97.78% drawdown calculation and trading halt. Fixed by resetting peak_equity to actual value (~$1,999). Root cause of corruption still needs investigation.
- [2026-01-08 12:30] **LIVE BOT CONFIG CHANGE: 2 Symbols + Wider Spacing** - Reduced from 3 symbols to 2 (removed DOGE). Widened grid spacing from ~1.5% to 2.7% to overcome friction costs. Previous config was losing money due to fees+slippage exceeding grid profits.
  - **Symbols**: AVAX/USD (90%), LTC/USD (10%)
  - **Grid spacing**: 2.67% (was 1.45-1.67%)
  - **Levels**: 5-6 per symbol (was 6-7)
  - **Rationale**: $2K account with 1.5% spacing = ~0% net after ~1.5-2% friction. 2.7% spacing should yield ~0.7-1% net per cycle.
- [2026-01-07 16:30] **DASHBOARD THEME OVERHAUL** - Deep Ocean teal/slate theme.
- [2026-01-07 16:00] **SMART GRID ADVISOR (Phase 1)** - Shadow-mode drift detection.
- [2026-01-06] **LIVE BOT LAUNCHED** - $2000 initial equity ($1000 deposit + $1000 existing).
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

## System Health (LIVE Instance) — Updated Jan 14, 2026 @ 14:35 EST
- **Bot**: Healthy, NORMAL mode, port 8001, stream connected
- **Notifier**: Running, watchdog active
- **Risk Overlay**: NORMAL, all triggers inactive (momentum +0.21%, correlation 0.75, ADX 15.5)
- **Equity**: $2,011.53 (started $2,000 = $1,000 original + $1,000 deposit)
- **Actual Trading P/L**: +$11.53 (+0.58% on $2K) since Jan 6
- **Daily P/L**: +$0.35 (+0.017%)
- **Positions**: LTC/USD (0.13 qty), DOGE/USD (0.10 qty), AVAX/USD (dust)
- **Paper Bot**: DISABLED (launchd agents moved to ~/Library/LaunchAgents/disabled/)

## Maintenance Plan (5 Phases) - ALL COMPLETE
| Phase | Status | Description |
|-------|--------|-------------|
| 1. State Persistence | ✅ DONE | Move `/tmp` state files to `data/state/` |
| 2. Bot Watchdog | ✅ DONE | Auto-restart bot on crash |
| 3. Database Backup | ✅ DONE | Daily automated backups at 3 AM |
| 4. DB Cleanup | ✅ DONE | Cleanup script with 90-day retention |
| 5. Log Rotation | ✅ DONE | Daily rotation at 5 AM, 50 MB limit |

## Scheduled Tasks

### LaunchAgents (launchd - LIVE instance only)
| Agent | Interval | Purpose |
|-------|----------|---------|
| `com.bluebird-live.bot` | RunAtLoad | Live bot (port 8001) |
| `com.bluebird-live.notifier` | RunAtLoad | Live notifier |
| `com.bluebird-live.watchdog-bot` | 5 min | Bot auto-restart |
| `com.bluebird-live.watchdog-notifier` | 5 min | Notifier auto-restart |

**Note**: Paper bot agents (`com.bluebird.*`) have been disabled and moved to `~/Library/LaunchAgents/disabled/` as of Jan 14, 2026.

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

## Current Configuration (Jan 14, 2026 - LIVE Instance)
| Setting | Value |
|---------|-------|
| **Symbols** | AVAX/USD (90%), LTC/USD (10%) |
| **LTC Purpose** | Correlation signal for risk management (not profit) |
| **Grid Spacing** | ~2.7% (AVAX 2.53%, LTC 2.78%) |
| **Levels** | 5-6 per symbol |
| **$/Level** | AVAX ~$360, LTC ~$40 |
| **MAX_POSITIONS** | 2 |
| **TIMEFRAME** | 1Min |

## Key Metrics (Jan 14, 2026 - LIVE Instance)
| Metric | Value |
|--------|-------|
| **Equity** | $2,011.53 |
| **Starting Capital** | $2,000.00 ($1K original + $1K deposit) |
| **Actual Trading P/L** | +$11.53 (+0.58%) |
| **P/L Since Config Change (Jan 8)** | +$15.13 (+0.76%) |
| **Daily P/L** | +$0.35 (+0.017%) |
| **Status** | Healthy, AVAX/USD (90%) + LTC/USD (10% for correlation) |

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
