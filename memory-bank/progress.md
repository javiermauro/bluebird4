# Progress — Status & History

## Current Status
- [2026-01-22 PM] **✅ DECISION: HOLD AVAX** - After thorough analysis (web research + paper bot data), decided to HOLD. Paper bot shows AVAX outperforms SOL 3.3x ($4,428 vs $1,336 in Jan). Issue is timing not coin. Review Jan 29. Exit if AVAX < $11.
- [2026-01-21 PM] **COMPUTER RESTARTED - WATCHDOG RECOVERED BOT** - Mac restarted, launchd watchdog automatically recovered the live bot. Bot running on port 8001, stream initializing. All services healthy.
- [2026-01-21 PM] **AVAX POSITION UPDATE** - Position reduced from 119 to 75.30 qty (sells filled). Now @ $13.15 avg entry. Current price $12.14, unrealized -$76.38 (-7.7%). Need +8.3% to hit breakeven.
- [2026-01-21 AM] **CODE QUALITY IMPROVEMENTS DEPLOYED** - SQLite hardening (WAL mode, 30s timeout), exception handling fixes, stream watchdog lowered to 90s.
- [2026-01-21 AM] **BOT STABILITY** - Bot crashed multiple times during session but watchdog auto-recovered each time. Stream reconnection working.
- [2026-01-20 PM] **AVAX RECOVERY IN PROGRESS** - Placed manual breakeven limit sell: 35 AVAX @ $13.60 (Order ID: 4d3a11de-...). Position: 119.14 AVAX @ $13.58 avg, current ~$12.50, unrealized -$129. Grid top ($13.38) is below avg cost — grid can't recover alone. Need AVAX > $13.58 for order to fill.
- [2026-01-20 PM] **BOT RESTARTED WITH NEW CODE** - All protection improvements now active. ADX direction shows "up" (raw DI+/DI- working), stream state tracking visible in /health, AVAX in grid_reduced mode. Killed stray paper processes. Restarted notifier for 2-min stale threshold.
- [2026-01-20] **PROTECTION IMPROVEMENTS DEPLOYED TO LIVE** - All 9 phases of protection improvements deployed to LIVE instance with conservative settings. Changes take effect on next bot restart.
- [2026-01-19 19:00] **PROTECTION IMPROVEMENTS PLANNED** - Deep root cause analysis of Jan 17-18 incidents complete. Identified 5 protection gaps + 4 resilience gaps. Implementation plan at `~/.claude/plans/spicy-questing-bumblebee.md`. 9 phases of fixes planned (P0: ADX direction, orchestrator thresholds; P1: fast momentum, down bars, price drop, startup resilience; P2: stream tracking, stale detection, connection cleanup).
- [2026-01-19 18:59] **TRADING RESUMED** - New day reset daily loss limit. Equity $1,917.47. Today's P/L +$12.28 (+0.64%). AVAX recovering slightly ($12.66→$12.77). Still holding 119 AVAX at -6% unrealized (-$96). Risk overlay NORMAL, all signals healthy.
- [2026-01-18 22:12] **⚠️ DAILY LOSS LIMIT HIT — TRADING HALTED** - AVAX dropped ~8% ($13.76 → $12.66). Bot accumulated during decline (119 qty @ $13.58 avg). Unrealized -$111.17. Daily P/L -$105.84 (-5.27%) exceeded -5% limit. Circuit breaker triggered. Trading resumes at midnight or via manual reset.
- [2026-01-17 17:00] **BOT CRASH & RECOVERY** - Alpaca WebSocket rate limit (429 + "connection limit exceeded") caused crash. Fixed by killing zombie processes holding stale TCP connections, then starting server directly (bypassing start.py timeout). Exponential backoff in stream.py eventually reconnected.
- [2026-01-16 22:39] **PERFORMANCE UPDATE** - Bot healthy, NORMAL mode for 34+ hours. Equity $1,995.50. Today +$0.99 (4 trades). 7-day +$16.43. All-time +$18.74 (27 trades). Grid profit: AVAX $13.57, LTC $3.41. 5 open limit orders on AVAX.
- [2026-01-14 14:35] **WEEK 1 ANALYSIS COMPLETE** - Decision: CONTINUE current config (AVAX 90%, LTC 10%, 2.7% spacing). Net profit +$15.13 since Jan 8. Fee ratio improved from ~50% to 21%. LTC kept for correlation signal (risk management), not profit.
- [2026-01-14 14:20] **PAPER BOT SERVICES DISABLED** - Stopped all paper bot services and permanently disabled launchd agents. Moved 7 plist files to `~/Library/LaunchAgents/disabled/`. Only live bot (port 8001) remains active.
- [2026-01-14 14:20] **CORRECTED P/L** - Actual trading P/L: +$11.53 (+0.58% on $2K). Previous "+101%" was misleading (included $1K deposit).
- [2026-01-08 12:30] **LIVE BOT CONFIG CHANGE: Wider Grid Spacing** - Reduced from 3 symbols to 2, widened spacing from ~1.5% to 2.7% to overcome friction costs.
  - **Problem**: Previous 1.5% spacing was being eaten by fees (~0.25%) + slippage (~1-2%) = net ~0% or negative
  - **Solution**: 2 symbols (AVAX 90%, LTC 10%), 2.7% spacing, 5-6 levels
  - **Changes**: Updated `config_ultra.py`, cleared grid state, cancelled orphan DOGE orders
  - **Expected**: ~0.7-1% net profit per grid cycle instead of ~0%
- [2026-01-07 16:30] **DASHBOARD THEME OVERHAUL**: Deep Ocean teal/slate theme.
- [2026-01-07 16:00] **SMART GRID ADVISOR (Phase 1)**: Shadow-mode drift detection.
- [2026-01-06] **LIVE BOT LAUNCHED**: $2000 equity, 3 symbols (AVAX/LTC/DOGE), 1.5% spacing.
- [2025-12-31 16:40] **BTC/USD REMOVED**: Underperformed 5.7x vs altcoins.
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

## ✅ Coin Analysis Complete — DECISION: HOLD AVAX (Jan 22, 2026)

### Analysis Performed
Thorough research comparing AVAX vs SOL for grid trading:
1. Web search for price predictions and trends
2. Paper bot historical performance comparison
3. Volatility and risk metrics analysis

### Key Discovery: Paper Bot Data
| Symbol | Jan 2026 Sells | Total Profit | Avg/Trade |
|--------|----------------|--------------|-----------|
| **AVAX/USD** | 182 | **$4,428** | $24.33 |
| LTC/USD | 29 | $1,407 | $48.53 |
| SOL/USD | 107 | $1,336 | $12.49 |
| DOGE/USD | 91 | $1,266 | $13.92 |

**AVAX outperforms SOL by 3.3x on paper bot!**

### Web Research Summary
| Metric | AVAX | SOL |
|--------|------|-----|
| YTD 2026 Return | -19.26% | +21.05% |
| Volatility | 26.03% (higher = more grid cycles) | 20.78% |
| Sharpe Ratio | 0.10 | 0.70 |

### Why Live Bot Is Down (Not Coin's Fault)
1. **Bad timing**: Started Jan 6, crashed Jan 18 (-8%)
2. **Small capital**: $2K vs paper's $100K+ = less buffer
3. **Same strategy works**: Paper bot making $4,428 on AVAX

### Decision: HOLD
- Do NOT switch to SOL
- AVAX needs +8.3% to breakeven ($12.14 → $13.15)
- Grid profit +$44 proves strategy works
- Switching would lock in -$76 loss permanently

### Monitoring Plan
- **Review date**: Jan 29, 2026
- **Exit trigger**: If AVAX breaks below $11.00 (-10% more), reassess
- **Success trigger**: AVAX reaches $13.15+, position recovers

### Sources
- [PortfoliosLab SOL vs AVAX](https://portfolioslab.com/tools/stock-comparison/SOL-USD/AVAX-USD)
- [CoinCodex AVAX Prediction](https://coincodex.com/crypto/avalanche/price-prediction/)
- [BeInCrypto Grid Trading](https://beincrypto.com/learn/grid-trading-crypto-explained/)

---

## Recent Work (High Signal)

### Jan 21, 2026 AM — Code Quality & Robustness Improvements

**Session Focus**: Addressed critical code quality issues identified in system analysis.

**Changes Deployed** (both paper and live instances):

1. **SQLite Configuration** (`src/database/db.py` lines 41-46):
   ```python
   conn = sqlite3.connect(DB_PATH, timeout=30.0)
   conn.execute("PRAGMA journal_mode=WAL")
   conn.execute("PRAGMA synchronous=NORMAL")
   conn.execute("PRAGMA busy_timeout=30000")
   ```
   - Prevents "database is locked" errors under concurrent access
   - WAL mode enables concurrent reads
   - 30s timeout prevents indefinite hangs

2. **Exception Handling** (`src/execution/bot_grid.py` lines 3591, 3607, 3620):
   - Fixed 3 bare `except:` blocks that silently swallowed errors
   - Now logs: `logger.error(f"[ERROR] {operation} failed: {type(e).__name__}: {e}")`
   - Affects: positions fetch, open orders fetch, account/equity fetch
   - Maintains safe fallback behavior while providing visibility

3. **Stream Watchdog Threshold** (`src/execution/bot_grid.py` line 3410):
   - Changed `STALE_THRESHOLD_SECONDS` from 180 to 90
   - Triggers reconnection after ~1.5 missed bars instead of ~3
   - More aggressive recovery for live trading

4. **Stream Status Threshold** (`src/execution/bot_grid.py` line 4710):
   - Fixed hardcoded 90s to 60s for "connected" status
   - Status levels now: <60s=connected, 60-90s=degraded, >90s=stale
   - Provides accurate health reporting

**Bot Stability**: Bot crashed multiple times during session (cause unknown), but watchdog auto-recovered each time. Stream reconnection working correctly with new 90s threshold.

**AVAX Position**: Still underwater at -9.2% (-$147). Avg entry improved slightly ($13.58 → $13.45) due to cost basis averaging. Needs +7.5% to reach $13.13 breakeven sell.

**Files Modified**:
- `bluebird/src/database/db.py`
- `bluebird-live/src/database/db.py`
- `bluebird/src/execution/bot_grid.py`
- `bluebird-live/src/execution/bot_grid.py`

### Jan 20, 2026 PM — Bot Restart & AVAX Recovery Order

**Bot Restart**:
- Killed LIVE bot (PID 79619) to trigger restart with new protection code
- Verified new code loaded:
  - `/api/risk/overlay` shows `raw_plus_di` and `raw_minus_di` fields (ADX direction fix working)
  - ADX direction = "up" (previously stuck at "neutral" due to regime masking bug)
  - `/health` shows `stream_state` object with connection tracking
  - AVAX in `grid_reduced` mode (orchestrator thresholds working)
- Killed stray paper bot processes (PIDs 47868, 67890, 67891) that shouldn't have been running
- Restarted notifier to pick up 2-min stale threshold (was 5 min)

**AVAX Position Analysis**:
- **Core Problem**: Grid top ($13.38) is BELOW avg cost ($13.58)
- Grid sells can't recover position — they sell at a loss
- Position: 119.14 AVAX @ $13.58 avg, current ~$12.50
- Unrealized loss: **-$129** (-8.5%)
- 83 AVAX already allocated to 4 grid sell orders
- Only 36 AVAX available for new orders

**Recovery Strategy — Option C (Manual Breakeven Sell)**:
- First attempt: 40 AVAX @ $13.60 → REJECTED (insufficient available balance)
- Discovery: 83 AVAX allocated to grid orders, only 36 available
- Second attempt: 35 AVAX @ $13.60 → SUCCESS
- **Order ID**: `4d3a11de-6529-4ab5-9a07-92dac42da0a5`
- Expected: If AVAX > $13.60, sell 35 @ ~$0.02 profit each

**Current AVAX Order Stack**:
| Level | Qty | Price | Status |
|-------|-----|-------|--------|
| Breakeven | 35 | $13.60 | NEW (manual) |
| Grid L4 | ~20 | $14.02 | Grid sell |
| Grid L3 | ~21 | $13.71 | Grid sell |
| Grid L2 | ~21 | $13.38 | Grid sell |
| Grid L1 | ~21 | $13.04 | Grid sell |

**Market Conditions**: ADX 9.1 (very low = sideways), correlation -0.02, momentum -0.28%. Favorable for grid trading and recovery.

### Jan 20, 2026 — Protection Improvements Deployed to LIVE

**Deployment**: All 9 phases of protection improvements deployed to LIVE instance with **conservative settings** (more buffer than paper instance).

**Files Modified**:
1. `config_ultra.py` - Added new protection parameters with conservative values
2. `src/execution/bot_grid.py` - ADX direction fix, fast momentum, enhanced down bars, price drop protection, stream state updates
3. `src/api/server.py` - Added stream_state to system_state and /health endpoint
4. `src/notifications/notifier.py` - Faster stale detection (2 min threshold)
5. `start.py` - Added wait_for_health with 90s timeout, degraded mode support

**Conservative vs Paper Thresholds**:

| Parameter | Paper | LIVE | Rationale |
|-----------|-------|------|-----------|
| DEFENSIVE_INVENTORY_PCT | 110% | **130%** | More buffer before blocking |
| GRID_REDUCED_ENTER_PCT | 70% | **85%** | Catches buildup with more room |
| MOMENTUM_FAST_THRESHOLD | -2.0% | **-3.0%** | Only triggers on larger drops |
| PRICE_DROP_THRESHOLD_PCT | -5.0% | **-8.0%** | Won't block during normal volatility |
| ADX_DIRECTION_MIN_DI_DIFF_PCT | 5.0% | **5.0%** | Same (critical fix) |
| Stale threshold | 2 min | **2 min** | Same (faster alerts) |

**New Protection Features**:
- **ADX Direction Fix**: Uses raw DI+/DI- comparison instead of regime classification. Fixes bug where VOLATILE regime masked TRENDING_DOWN.
- **Fast Momentum**: 3-bar window catches flash crashes before 10-bar window would react.
- **Enhanced Down Bars**: Window analysis (70% red ratio OR 4% cumulative drop) doesn't reset on single green bar.
- **Price Drop Protection**: Blocks buys if price dropped >8% from 30-minute high.

**Observability Improvements**:
- `/health` endpoint now shows stream_state with connection_status, is_rate_limited, current_backoff_seconds, reconnect_count, last_error.
- Notifier alerts on stale data after 2 minutes (was 5 minutes).

**Startup Resilience**:
- `wait_for_health()` method with 90s timeout (was 30s port check)
- Accepts degraded mode (API up, stream reconnecting) during WebSocket backoff cycles

**Syntax Verification**: All 5 modified files pass `python -m py_compile`

**Note**: Changes take effect on next bot restart. Current bot continues running with old code until restart.

### Jan 19, 2026 — Root Cause Analysis & Protection Improvements Plan

**Incident Analysis (Jan 17-18)**:
Deep analysis of why protections failed during the AVAX 8% drop and WebSocket crash.

**Protection System Gaps Found**:

| Gap | Issue | Impact |
|-----|-------|--------|
| ADX Direction Bug | VOLATILE regime masks TRENDING_DOWN | adx_direction="neutral" even during clear downtrends |
| Orchestrator Dead Zone | 30-100% inventory has no restrictions | Inventory accumulated to 79% without any gates |
| Momentum Window | 10-bar window too large | -1.4% reading when actual drop was -8% |
| Down Bars Fragile | Resets on single green bar | Ineffective during sustained drops |

**Resilience Gaps Found**:

| Gap | Issue | Impact |
|-----|-------|--------|
| start.py Timeout | 30s too short for WebSocket backoff | Declares failure during normal reconnection |
| No Degraded Mode | Can't start API without stream | Bot completely down during rate limits |
| Slow Stale Detection | 5 min threshold in notifier | Late alerts on stream disconnect |
| Zombie Cleanup | No TCP connection cleanup | Manual intervention needed |

**Root Cause - ADX Direction**:
- `regime_detector.py` line 133-136 checks VOLATILE (vol_ratio > 1.5) BEFORE TRENDING (adx > 25)
- During volatile downtrends, regime = VOLATILE → adx_direction = "neutral"
- Raw DI+/DI- are calculated in `feature_calculator.py` but NOT used for direction
- Fix: Use raw `minus_di > plus_di` comparison instead of regime classification

**Root Cause - Orchestrator**:
- GRID_REDUCED threshold at 100% is too high
- At 79% inventory, orchestrator was still GRID_FULL (no size restrictions)
- Fix: Lower GRID_REDUCED to 70%, DEFENSIVE to 110%

**Implementation Plan**: `~/.claude/plans/spicy-questing-bumblebee.md`
- 9 phases of improvements
- P0: ADX direction fix, lower orchestrator thresholds
- P1: Fast momentum (3-bar), enhanced down bars, price drop protection, startup resilience
- P2: Stream state tracking, faster stale detection, connection cleanup

### Jan 14, 2026 — Week 1 Analysis + Paper Bot Disabled

**Week 1 Analysis (2.7% Grid Spacing)**:
Config changed on Jan 8. Analysis performed Jan 14 (6 days).

| Metric | Before (1.5% spacing) | After (2.7% spacing) |
|--------|----------------------|---------------------|
| Fee ratio | ~50% of gross | **21%** of gross |
| Net result | -$3.59 (2.5 days) | **+$15.13** (6 days) |
| Daily avg | -$1.44/day | **+$2.50/day** |

**Decision**: CONTINUE current config.

**Single Pair Consideration**:
- Analyzed going 100% AVAX (removing LTC)
- **Decision**: Keep LTC at 10% because it provides **correlation signal**
- Code uses 2nd pair for:
  1. `get_correlation_risk_adjustment()` — reduces position sizes when correlation > 0.85
  2. RISK_OFF trigger — correlation > 0.90 is one of 3 signals for crash protection
- Without 2nd pair: correlation always 0, crash protection weakened
- LTC profit (~$0.85/cycle) is minimal but correlation signal is valuable

**Corrected P/L** (was showing misleading +101%):
- Starting capital: $2,000 ($1K original + $1K deposit added)
- Current equity: $2,011.53
- **Actual trading P/L**: +$11.53 (+0.58% on $2K)
- **P/L since config change** (Jan 8): +$15.13 (+0.76%)

**Paper Bot Services Disabled (14:20)**:
- Stopped all paper bot services (port 8000) that were auto-restarting via launchd
- Moved 7 launchd plist files to `~/Library/LaunchAgents/disabled/`
- Paper bot will NOT auto-start on reboot anymore
- Only live bot agents (`com.bluebird-live.*`) remain active

### Jan 14, 2026 — Order Tracking & Circuit Breaker Hardening

**Issue 1: Orphan Orders Accumulating**
- Found 3 LTC limit orders on Alpaca not being tracked by bot
- Root cause: State save failure on Jan 12 restart left orders orphaned
- **Fix**: Added `CANCEL_ORPHAN_ORDERS_ON_HEALTH_CHECK = True` to config
- Health check (every 5 min) now auto-cancels orphan orders and cleans stale tracking
- **Commit**: `43193a3` - feat: auto-cancel orphan orders during health check

**Issue 2: False Circuit Breaker Trigger**
- Circuit breakers (max_drawdown + daily_limit) triggered during normal trading
- Root cause: Alpaca API timeout returned equity=0, causing false 100% drawdown calculation
- **Fix**: Added sanity checks in `check_circuit_breakers()`:
  - Skip if equity <= 0 (API timeout)
  - Skip if equity < 50% of daily start (likely API error)
- Also added sanity check in `update_risk_state()` for daily P&L calculation
- Reset circuit breakers via `/api/risk/reset`
- **Commit**: `5f58bb0` - Add sanity checks to prevent false circuit breaker triggers from API timeouts

**Files Modified**:
- `config_ultra.py`: Added `CANCEL_ORPHAN_ORDERS_ON_HEALTH_CHECK = True`
- `src/execution/bot_grid.py`:
  - `check_order_tracking_health()`: Auto-cancel orphans, clean stale tracking
  - `check_circuit_breakers()`: Sanity checks for invalid equity
  - `update_risk_state()`: Sanity check for daily P&L

### Jan 8, 2026 — LIVE Bot Grid Config Change (Wider Spacing)
**Problem Identified**: First 2.5 days of live trading showed:
- 10 trades, $2.31 realized profit, $0.67 fees
- BUT equity down $3.59 (-0.18%)
- Gap of $5.23 = hidden friction costs (spread + slippage)
- Grid spacing (1.5%) ≈ friction costs (1.5-2%) = no edge

**Solution Implemented**:
1. Reduced symbols from 3 to 2:
   - AVAX/USD: 90% allocation (was 45%)
   - LTC/USD: 10% allocation (for correlation signal)
   - DOGE/USD: REMOVED
2. Widened grid spacing to 2.5% target (actual: 2.67%):
   - `num_grids`: 5 (was 6)
   - `range_pct`: 0.125 (was 0.087)
   - Expected net profit: ~0.7-1% per cycle (was ~0%)
3. Concentrated capital: ~$300/level (was ~$130/level)

**Changes Made**:
- `config_ultra.py`: Updated SYMBOLS, GRID_CONFIGS, MAX_POSITIONS
- Cleared `data/state/grid-state.json` and `grid_snapshots` table
- Cancelled orphan DOGE limit order on Alpaca
- Restarted bot to apply new config

**Verification**:
- Grid status shows: AVAX 7 levels @ 2.67%, LTC 6 levels @ 2.68%
- 8 open orders (5 AVAX, 3 LTC) - no DOGE
- Bot healthy, stream healthy

**Test Plan**: Run for 1 week (through Jan 13) to validate wider spacing is profitable.

### Jan 7, 2026 — SmartGrid Advisor + Dashboard Theme
**SmartGrid Advisor (Phase 1 - Shadow Mode)**:
- Created `src/strategy/smart_grid_advisor.py` (~500 lines)
- Drift detection with hysteresis: 55% trigger / 40% clear / 60min cooldown
- DB-backed fill-rate sampling, ATR percentile ring buffer
- State persisted to `data/state/smart-grid-advisor.json`
- Background task for WS stall resilience (evaluates every 300s)
- API endpoint `GET /api/smartgrid/status`
- Gates: Risk Overlay must be NORMAL, Orchestrator not DEFENSIVE, not strong downtrend
- Respects existing rebalance mechanisms (won't compete)

**Dashboard Theme Overhaul**:
- Replaced harsh red/crimson "Control Room Alert" with calm teal/slate "Deep Ocean"
- New fonts: IBM Plex Sans (body) + IBM Plex Mono (data)
- Color palette: `--gold-primary: #14b8a6` (teal), `--gold-light: #2dd4bf`
- Added SmartGrid Advisor panel showing drift status per symbol
- Chart colors updated from amber to teal gradient

**Files Modified**:
- `src/strategy/smart_grid_advisor.py` (NEW)
- `config_ultra.py` (added SMART_GRID_* settings)
- `src/execution/bot_grid.py` (import, init, background task, handle_bar)
- `src/api/server.py` (added /api/smartgrid/status endpoint)
- `dashboard/src/index.css` (complete theme overhaul)
- `dashboard/src/App.jsx` (SmartGrid state, fetch, panel)

### Jan 1, 2026 — 🏆 BEST DAY EVER (+$6,283, +6.2%)
- **Peak Equity**: $107,138.56 (broke $107K!)
- **Daily P/L**: +$6,283.63 (+6.2%) — previous best was $1,685
- **Grid P/L (30 days)**: +$16,862.30 (+18.7%)
- **30D Volume**: $1,874,740

**What Happened**:
- Market rallied hard, grid sold into strength perfectly
- All 4 symbols cycled multiple times (buy dips → sell rallies)
- DOGE first day: accumulated 112K coins, +$334 unrealized
- Massive profit-taking waves locked in gains

**System Crash & Recovery (15:42)**:
- Mac crashed, bot died. Restarted successfully.
- DOGE display fix applied (commit `ee3fbc3`)
- All 9 protection systems verified working

**Key Insight**: Trusted the system, didn't add trailing stop complexity. System performed exactly as designed — buy low, sell high, repeat.

### Jan 1, 2026 — DOGE/USD Added
- **DOGE/USD Added to Grid Trading**: Highest volatility of all symbols (10.3% 7D range, 2.6% daily vol). Added with conservative 15% allocation since it's a meme coin.
- **New Allocation**: SOL 35%, LTC 25%, AVAX 25%, DOGE 15%
- **DOGE Grid Config**: 6 grids (7 levels), 10% range, ~1.67% spacing
- **Commit**: `bcb1811` - feat: add DOGE/USD to grid trading (15% allocation)
- **Dashboard Updated**: Added DOGE to symbol selector with Ð icon

### Dec 31, 2025 — BTC Removal + Bug Fixes
- **BTC/USD Removed from Grid Trading**: Analysis showed BTC underperformed altcoins by 5.7x ($763 profit vs $4,363 for SOL). Grid trading profits from volatility - BTC is too stable compared to altcoins.
- **Allocation After BTC Removal**: SOL 40%, LTC 30%, AVAX 30%
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

## Performance Tracking (LIVE Instance - $2K Account)

**NOTE**: Starting capital = $2,000 ($1K original + $1K deposit). Grid start = $1,000 (Jan 6). Current equity = $1,827.48.

| Date | Daily P/L | Cumulative | Equity | Notes |
|------|-----------|------------|--------|-------|
| **Jan 22** | -$36.11 | -$151.07 | $1,848.93 | ⚠️ Coin change evaluation |
| Jan 21 | +$20.22 | -$114.96 | $1,885.04 | Computer restarted, watchdog recovered bot |
| Jan 20 | — | — | ~$1,865 | Protections deployed, bot restarted, breakeven order placed |
| Jan 19 | +$12.28 | — | $1,917 | Trading resumed after daily reset |
| **Jan 18** | **-$99.29** | -$96.54 | $1,903.46 | ⚠️ HALTED: Daily loss limit (-5.27%) |
| Jan 17 | +$9.55 | +$2.75 | $2,012.85 | Bot crash & recovery (WS rate limit) |
| Jan 16 | -$0.09 | -$6.80 | $1,993.52 | — |
| Jan 15 | -$10.97 | -$6.71 | $1,996.35 | — |
| Jan 14 | -$4.56 | +$4.26 | $2,006.62 | Week 1 analysis: CONTINUE config |
| Jan 13 | +$16.53 | +$8.82 | $2,011.29 | Strong day |
| Jan 12 | -$4.46 | -$7.71 | $1,994.76 | False halt bug fixed |
| Jan 11 | +$2.81 | -$3.25 | $1,999.22 | — |
| Jan 10 | $0.00 | -$6.06 | $1,996.39 | — |
| Jan 9 | -$0.02 | -$6.06 | $1,996.39 | — |
| Jan 8 | -$3.45 | -$6.04 | $1,996.41 | Config change: 2 symbols, 2.7% spacing |
| Jan 7 | -$0.14 | -$2.59 | $1,999.86 | First full day, DOGE trades |
| Jan 6 | — | — | $2,000.00 | Start ($1K + $1K deposit) |

## Performance Tracking (PAPER Instance - $100K+ Account)
| Date | Daily P/L | Grid P/L | Notes |
|------|-----------|----------|-------|
| **Jan 1** | **+$6,283 (+6.2%)** | **+$16,862** | 🏆 BEST DAY EVER! Broke $107K |
| Dec 21 | +$1,685 (+1.83%) | +$3,341 | Previous best day, first RISK_OFF event |
| Dec 20 | ~+$200 | +$1,656 | Post-crash recovery |
| Dec 2 | Start | $0 | Grid trading era begins |
