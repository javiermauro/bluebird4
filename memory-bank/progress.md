# Progress — Status & History

## Current Status
- [2025-12-31 16:40] **BTC/USD REMOVED**: Underperformed 5.7x vs altcoins. New allocation: SOL 40%, LTC 30%, AVAX 30%. MAX_POSITIONS=4.
- [2025-12-31 16:00] **BUG FIXES**: Timezone (naive→UTC), config spam (singleton), TIMEFRAME ("5Min"→"1Min").
- [2025-12-30 00:00] **BOT LAUNCHAGENT CREATED**: Created `com.bluebird.bot.plist` with `RunAtLoad=true` and `KeepAlive=true`. Bot now managed directly by launchd for reliable auto-restart after reboot/power outage. Watchdog serves as backup monitor.
- [2025-12-29 23:45] **Watchdog lsof Bug Fixed**: Changed `lsof -ti :8000` to `lsof -ti TCP:8000 -sTCP:LISTEN` to only kill LISTENING processes (bot), not processes with outgoing connections (notifier).
- [2025-12-28] **Fee Modeling Tested & Verified**: 25/25 tests pass, all endpoints working
- [2025-12-27] **Tier-Correct Fee Modeling Complete**: Volume-based Alpaca crypto fee tiers, Gross vs Net P&L
- [2025-12-26 06:35] **Watchdog launchd Migration Complete**: Fixed EPERM on external volume
- [2025-12-30 02:10] **Launchd DB Access Hardening**: Added `BLUEBIRD_DB_PATH` override (run DB on internal disk), watchdog per-machine config `~/Library/Application Support/BLUEBIRD/config.env`, notifier watchdog fallback when DB unreadable to prevent restart flapping
- [2025-12-30 02:20] **Ops Realtime Monitor Added**: `scripts/monitor_services.sh` provides realtime status and exits once bot+dashboard+notifier are up
- [2025-12-29 22:18] **DOCK → INTERNAL FINAL SYNC + CLEAN RESTART**: rsync into `~/BLUEBIRD/bluebird/`, watchdog scripts/config re-synced, bot restarted from internal; `/health` OK and monitor reports READY
- [2025-12-29 22:36] **Watchdogs sped up**: LaunchAgents interval set to 60s (was 300s) and watchdog stale threshold set to 120s (was 300s) for faster unattended recovery after reboot
- [2025-12-29 22:48] **Bot watchdog anti-flap fix**: `/health` is now primary; restart if `/health` is unreachable even if heartbeat timestamp is recent; heartbeat threshold reset to 300s to avoid self-inflicted restarts
- [2025-12-29 23:05] **Monitor script redesigned**: `scripts/monitor_services.sh` now shows one line per component with clear YES/NO and a waiting list; supports `--once --no-clear` snapshots
- [2025-12-29 23:08] **Monitor runs automatically after reboot**: added `scripts/sync-monitor-scripts.sh` which installs `com.bluebird.monitor-status` (60s snapshots to `/tmp/bluebird-monitor-status.log`)
- [2025-12-26 01:45] **Timeout Hardening Complete**: All main loop Alpaca calls bounded with timeouts
- [2025-12-25 18:30] **Phase A Robustness Complete**: Crash loop detection, atomic writes, disk monitoring
- [2025-12-25 11:55] System healthy, NORMAL mode, all protection layers active
- [2025-12-25 11:55] **ALL 5 MAINTENANCE PHASES COMPLETE**
- [2025-12-25 11:55] **Phase 5 Maintenance Complete**: Log rotation daily at 5 AM
- [2025-12-25 11:53] **Phase 4 Maintenance Complete**: Database cleanup script
- [2025-12-25 11:48] **Phase 3 Maintenance Complete**: Daily database backups at 3 AM
- [2025-12-25 11:42] **Phase 2 Maintenance Complete**: Bot watchdog with auto-restart
- [2025-12-25 11:30] **Phase 1 Maintenance Complete**: State files moved to persistent storage

## Recent Work (High Signal)

### Dec 31, 2025 — BTC Removal + Bug Fixes
- **BTC/USD Removed from Grid Trading**: Analysis showed BTC underperformed altcoins by 5.7x ($763 profit vs $4,363 for SOL). Grid trading profits from volatility - BTC is too stable compared to altcoins.
- **New Allocation**: SOL 40%, LTC 30%, AVAX 30% (was BTC 30%, SOL 25%, LTC 25%, AVAX 20%)
- **MAX_POSITIONS**: Restored to 4 (allows multiple positions per symbol)
- **Commits**:
  - `6ad5dbe` - fix: timezone bugs + config validation spam + correct TIMEFRAME setting
  - `eee2700` - feat: remove BTC/USD from grid trading, reallocate to altcoins
  - `ac63b4e` - config: restore MAX_POSITIONS to 4
- **Bug Fixes**:
  1. **Timezone Bug**: Warmup used `datetime.now()` (naive/local), Alpaca interprets as UTC → 5-hour offset. Fixed with `datetime.now(timezone.utc)` in `bot_grid.py:3125` and `alpaca_client.py:399,529`.
  2. **Config Spam**: `UltraConfig()` instantiated 12+ times per request (prints validation each time). Fixed with singleton pattern in `server.py:26-28`.
  3. **TIMEFRAME Mismatch**: Config said "5Min" but Alpaca's `subscribe_bars()` defaults to 1-min. Fixed config to match reality: `TIMEFRAME = "1Min"`.
- **Dashboard Updated**: Removed BTC from symbol lists, changed default to SOL/USD.
- **Files Modified**: `config_ultra.py`, `src/execution/bot_grid.py`, `src/execution/alpaca_client.py`, `src/api/server.py`, `dashboard/src/App.jsx`

### Dec 27, 2025 — Tier-Correct Alpaca Crypto Fee Modeling
- **Goal**: Implement volume-based fee tier calculation and expose Gross vs Net equity/P&L
- **Problem**: Fees were not being persisted (`trades.fees` always 0), only Tier 1 rates hardcoded
- **Solution**: Full fee tier engine with dual-band tracking (expected vs conservative)

**Alpaca Fee Tier Table** (8 tiers based on 30-day rolling volume):
| Tier | 30D Volume | Maker | Taker |
|------|------------|-------|-------|
| 1 | $0-100K | 0.15% | 0.25% |
| 2 | $100K-500K | 0.12% | 0.22% |
| 3-8 | ... | ... | ... |

**Key Features**:
- **3am ET tier boundary**: Fee day runs 03:00:00 ET to 02:59:59 ET next day
- **Dual-band tracking**: Expected (maker for limits) and Conservative (taker for all)
- **Maker/Taker determination**: Market orders = taker, Limit orders = `maker_assumed`
- **Uncertain classification count**: Tracks how many orders are `maker_assumed`

**Files Created**:
1. `src/utils/crypto_fee_tiers.py` - Fee tier engine with 8-tier table
   - `get_fee_tier(volume_30d_usd)` - Get tier info from volume
   - `get_fee_day_bucket(ts)` - Convert timestamp to 3am ET fee day
   - `calculate_fee()` - Calculate both expected and conservative fees
   - `determine_fee_type()` - Market=taker, Limit=maker_assumed

2. `src/utils/backfill_fees.py` - Backfill historical fees since Dec 1, 2025
   - Fetches filled orders from Alpaca (bulk pagination)
   - Upserts into orders table
   - Creates trade records if missing
   - Recomputes fees chronologically using tier engine
   - Usage: `python -m src.utils.backfill_fees [--dry-run]`

**Database Changes** (6 new columns in trades table):
- `fee_rate` (REAL) - Applied rate (0.0015 for maker)
- `fee_type` (TEXT) - 'maker', 'taker', or 'maker_assumed'
- `fee_tier` (TEXT) - 'Tier 1', 'Tier 2', etc.
- `rolling_30d_volume` (REAL) - Volume at time of fill
- `fee_day_bucket` (TEXT) - '2025-12-10' (ET date)
- `fee_conservative` (REAL) - Worst-case taker fee

**API Changes**:
1. `GET /api/profitability-report` - NEW endpoint
   - Returns Gross P&L, Net P&L (expected + conservative)
   - Current tier, rates, rolling volume, tier progression
   - Fee stats by tier

2. `GET /api/history/equity` - Added `current_fee_tier` object
   - Tier info, rates, rolling volume, fee totals

**Dashboard Changes** (`HistoryDashboard.jsx`):
1. **Paper Trading Warning Banner** - Dismissible warning about fee/slippage differences
2. **Fee Tier Info Card** - Current tier, maker/taker rates, 30d volume, tier progression bar
3. **Updated Metrics Row** - Gross P/L, Net P/L (expected), Net P/L (conservative), Recovery
4. **CSS Styles** - Paper trading warning (amber), fee tier card, progress bars

**Code Flow**:
1. `grid_trading.py:apply_filled_order()` now returns dict with all fee fields
2. `bot_grid.py:record_trade()` calls pass fee fields to database
3. `db.py:record_trade()` accepts and stores all fee audit columns
4. `db.py:get_fee_stats()` aggregates fees for reporting

**Testing**:
- Run `python -m src.utils.backfill_fees --dry-run` to preview backfill
- Run `python -m src.utils.backfill_fees` to execute backfill
- `GET /api/profitability-report` should return fee data
- Dashboard History tab should show fee tier card and net P&L metrics

**Fallback**: `config_ultra.py` MAKER_FEE_PCT/TAKER_FEE_PCT kept as Tier 1 fallback

### Dec 26, 2025 — Watchdog launchd Migration (EPERM Fix)
- **Root Cause**: Cron couldn't execute scripts on external APFS volume (`/Volumes/DOCK`) due to macOS security restrictions (`noowners` mount flag + `com.apple.provenance` xattr).
- **Error**: `Operation not permitted` when cron ran `check_bot.sh` or `check_notifier.sh`
- **Solution**: Migrate from cron to launchd with full script copies on local filesystem.
- **Architecture**:
  - **Local scripts**: `~/Library/Application Support/BLUEBIRD/run-check-{bot|notifier}.sh`
  - **LaunchAgents**: `~/Library/LaunchAgents/com.bluebird.watchdog-{bot|notifier}.plist`
  - **Durable state**: `~/Library/Application Support/BLUEBIRD/state/` (survives reboot)
  - **State files**: `crash-loop-*.json`, `pending-alerts.txt`, `disk-alert.json`
- **Changes**:
  1. **Created sync script** (`scripts/sync-watchdog-scripts.sh`):
     - Copies repo scripts to local path with path transformations
     - Idempotent - run after editing repo watchdog scripts
  2. **Updated CLAUDE.md** with new watchdog architecture and commands
  3. **Simplified notifier** (`src/notifications/notifier.py`):
     - Pending alerts now only from local path (watchdogs can't write to external volume)
     - Removed dual-path check (eliminates double-send risk)
  4. **Removed cron entries** for watchdog jobs
- **Validation**:
  - Both local watchdog scripts execute successfully
  - Bot and notifier report healthy
  - LaunchAgents scheduled every 5 minutes
- **Exception Note**: Watchdog state is the ONLY state that lives outside `data/state/` due to macOS launchd restrictions.

### Dec 26, 2025 — Timeout Hardening for Alpaca API Calls
- **Root Cause**: Bot hung when Alpaca API was slow during RISK_OFF transition. Event loop blocked on synchronous API calls.
- **Solution**: Wrapped all Alpaca API calls with bounded timeouts to prevent event loop hangs.
- **Changes**:
  1. **Added `run_blocking_with_timeout()`** (async wrapper):
     - Uses `asyncio.wait_for(asyncio.to_thread(...), timeout=...)`
     - Default timeout: 10s, Critical (orders): 15s, Cancel: 8s
     - Returns safe default on timeout instead of crashing
  2. **Added `run_sync_with_timeout()`** (sync wrapper):
     - Uses `concurrent.futures.ThreadPoolExecutor` with timeout
     - For initialization and other sync contexts
  3. **Added `AlpacaTimeoutStats`** observability class:
     - Tracks timeout count, last timeout time/operation
     - Endpoint: `GET /api/alpaca/timeout-stats`
  4. **Config constants** added to `config_ultra.py`:
     - `ALPACA_API_TIMEOUT_SECONDS = 10.0`
     - `ALPACA_API_TIMEOUT_CRITICAL = 15.0`
     - `ALPACA_API_TIMEOUT_CANCEL = 8.0`
- **Calls Wrapped** (main loop - fully protected):
  - `get_positions` - with fail-closed pattern (skip buys on timeout)
  - `get_open_orders`
  - `get_account`
  - `update_risk_state` (outer call)
  - `check_periodic_reconciliation`
  - All `submit_order` calls (8 locations)
  - All `cancel_order` calls
- **Internal calls wrapped** (inside update_risk_state):
  - `equity_snapshot:get_account`
  - `equity_snapshot:get_positions`
- **Startup calls wrapped**:
  - `init:get_alpaca_last_equity`
  - `startup:load_orders`
  - `startup:reconciliation`
  - `startup:get_account`
- **Fixed 8 SyntaxErrors**: Invalid `continue` statements inside `handle_bar()` callback (not a loop). Fixed by wrapping success paths in `else:` blocks.
- **Remaining Gap**: Startup grid rebuild path (`_cancel_symbol_orders`) - rare, startup-only, watchdog provides safety net.
- **Files Modified**: `src/execution/bot_grid.py`, `src/api/server.py`, `config_ultra.py`

### Dec 25, 2025 — Phase A Robustness Improvements
- **Goal**: Improve system resilience against crash loops, state corruption, and disk exhaustion
- **Market Scenario Analysis**: Identified gaps in protection (parabolic pumps, crash loops, atomic writes, disk monitoring)
- **Changes**:
  1. **Crash Loop Detection** (both watchdog scripts):
     - Track restarts in `data/state/crash-loop-{bot|notifier}.json`
     - Pause after 3 restarts in 30 minutes
     - Write pending alerts to `data/state/pending-alerts.txt`
     - Manual clear: `rm data/state/crash-loop-*.json`
  2. **Atomic JSON Writes** (new utility + 4 modules):
     - Created `src/utils/atomic_io.py` with `atomic_write_json()`
     - Uses temp file + fsync + os.replace pattern
     - Failures log loudly but NEVER crash trading loop
     - Updated: `grid_trading.py`, `risk_overlay.py`, `orchestrator.py`, `bot_grid.py`
  3. **Disk Space Monitoring** (both watchdog scripts):
     - Alert when disk >= 90% capacity
     - Once-per-day limit (tracked in `data/state/disk-alert.json`)
  4. **Pending Alerts Processing** (notifier):
     - Processes `data/state/pending-alerts.txt` on startup and each poll
     - Atomic rename to `.processing.txt` before sending
- **State Files Added**:
  - `data/state/crash-loop-bot.json`
  - `data/state/crash-loop-notifier.json`
  - `data/state/pending-alerts.txt`
  - `data/state/disk-alert.json`
- **Phase B (EUPHORIA gate)**: Designed but deferred - would block upward rebalances in parabolic pumps
- **Plan Document**: `/Users/javierrodriguez/.claude/plans/harmonic-herding-brook.md`

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
