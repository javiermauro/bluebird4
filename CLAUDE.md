# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## ⚠️ LIVE TRADING INSTANCE

**THIS IS THE LIVE TRADING INSTANCE** - Trades with REAL MONEY on Alpaca LIVE API.

| Setting | Value |
|---------|-------|
| **Mode** | LIVE (PAPER_TRADING = False) |
| **API Port** | 8001 |
| **Symbols** | AVAX/USD (90%), LTC/USD (10%) |
| **Lock Dir** | `/tmp/bluebird-live/` |
| **Logs** | `/tmp/bluebird-live-*.log` |
| **SMS Prefix** | `[LIVE]` |

Paper instance runs separately at port 8000 in `~/BLUEBIRD/bluebird`.

## Project Overview

BLUEBIRD 4.0 is a cryptocurrency grid trading bot that trades on Alpaca. It uses a grid trading strategy (not prediction-based) that profits from sideways market volatility by placing buy/sell orders at regular price intervals.

**Key Insight**: The system originally used ML predictions but achieved only 21% win rate. Grid trading was adopted because it thrives in sideways markets without needing predictions.

## Memory Bank (IMPORTANT)

The `memory-bank/` directory is **Claude's tool** for maintaining context and understanding across sessions. Since each conversation starts fresh, this is how you remember what happened before, what's working, what's broken, and what the system looks like.

**Maintain it properly** - your future self depends on it.

### At the START of every task:
1. Read `memory-bank/activeContext.md` - Current focus, system health, recent work
2. Read `memory-bank/progress.md` - Status history, known issues, what's been done

### At the END of every significant task:
1. Update `memory-bank/progress.md` with what was accomplished
2. Update `memory-bank/activeContext.md` with new status
3. Update other files if relevant:
   - `systemPatterns.md` - Architecture changes, new patterns
   - `techContext.md` - New dependencies, paths, scripts
   - `productContext.md` - Product behavior changes

### Memory Bank Files:
| File | Purpose |
|------|---------|
| `activeContext.md` | Current focus, system health, next steps |
| `progress.md` | Status history, completed work, known issues |
| `systemPatterns.md` | Architecture, design patterns, safety gates |
| `techContext.md` | Tech stack, paths, scripts, dependencies |
| `productContext.md` | Product goals, success metrics, UX |
| `projectbrief.md` | Foundation document (rarely changes) |

**Rule**: If you completed something important, document it in the memory bank before finishing.

## Running the System

### Start All Services
```bash
python start.py              # Bot + Dashboard
python start.py --all        # Bot + Dashboard + SMS Notifier
python start.py --status     # Check service status
python start.py --stop       # Stop all services
```

### Start Services Individually (with persistence)
```bash
# Grid Trading Bot (main service)
cd "/Volumes/DOCK/BLUEBIRD 4.0"
nohup caffeinate -i python3 -m src.api.server > /tmp/bluebird-bot.log 2>&1 &

# Dashboard
cd dashboard && npm run dev

# SMS Notifier
nohup caffeinate -i python3 src/notifications/notifier.py > /tmp/bluebird-notifier.log 2>&1 &
```

### Check Status
```bash
curl http://localhost:8001/health          # Bot health (LIVE)
curl http://localhost:8001/api/risk/status # Risk & P/L
ls /tmp/bluebird-live/*.pid                # Running services
```

## Architecture

### Services (3 total)
| Service | Entry Point | Port | Lock |
|---------|-------------|------|------|
| Grid Bot | `src/api/server.py` | 8001 | `bluebird-live-bot` |
| Dashboard | `dashboard/` (Vite/React) | 5174 | Port-based |
| Notifier | `src/notifications/notifier.py` | - | `bluebird-live-notifier` |

### Core Components
- **`src/api/server.py`** - FastAPI server with WebSocket for real-time dashboard updates. Runs the grid bot via `run_grid_bot()`.
- **`src/execution/bot_grid.py`** - Grid trading bot implementation. Handles order execution, risk controls, and state management.
- **`src/strategy/grid_trading.py`** - Grid trading strategy logic. Creates grid levels, tracks fills, calculates profits.
- **`src/strategy/risk_overlay.py`** - Risk overlay state machine (NORMAL/RISK_OFF/RECOVERY). Provides crash protection.
- **`src/strategy/orchestrator.py`** - Thin meta-controller for inventory management. Adds inventory episode tracking and staged liquidation.
- **`src/database/db.py`** - SQLite database for persistent trade/equity/order storage at `data/bluebird.db`.
- **`src/utils/process_lock.py`** - Single-instance protection using file locks in `/tmp/bluebird-live/`.
- **`config_ultra.py`** - All trading configuration (symbols, risk limits, grid settings).

### Data Flow
1. Alpaca WebSocket streams real-time price bars
2. `bot_grid.py` checks each bar against grid levels
3. If price hits a grid level → execute market order via `alpaca_client.py`
4. Record trade in SQLite database
5. Broadcast update to dashboard via WebSocket
6. Notifier polls API and sends SMS alerts for significant events

### State Persistence
**Persistent state files** (survive system reboot) in `data/state/`:
- `grid-state.json` - Grid levels, fills, pending orders
- `risk-overlay.json` - RISK_OFF/RECOVERY mode and triggers
- `orchestrator.json` - Inventory episode tracking
- `daily-equity.json` - Daily P&L tracking
- `alltime-equity.json` - All-time performance stats
- `circuit-breaker.json` - Max drawdown/stop-loss flags
- `windfall-log.json` - Windfall profit captures
- `watchdog.json` - Notifier watchdog state

**Process/lock files** (in `/tmp`, cleared on reboot - intentional):
- `/tmp/bluebird-live/*.lock`, `/tmp/bluebird-live/*.pid` - Single-instance protection
- `/tmp/bluebird-live-notifier.log` - Notifier log file
- `/tmp/bluebird-live-bot.log` - Bot log file

**Database**: `data/bluebird.db` (trades, equity, orders, notifications)

### Database Tables
**Trading tables**: `trades`, `equity_snapshots`, `orders`, `daily_summary`, `grid_snapshots`

**Notification tables** (added Dec 23, 2025):
- `sms_history` - Audit trail of all SMS sent
- `notified_trade_ids` - Prevents duplicate trade alerts across restarts
- `sms_queue` - Failed SMS queued for retry with exponential backoff
- `notifier_status` - Heartbeat, PID, API failure count, overlay mode

## Configuration

All settings in `config_ultra.py`:
- `TRADING_MODE`: "LIVE" (this is the live instance)
- `PAPER_TRADING`: False (uses Alpaca LIVE API)
- `SYMBOLS`: AVAX/USD (90%), LTC/USD (10%) - 2 symbols for correlation protection
- `GRID_CONFIGS`: Per-symbol grid settings (levels, spacing, size)
- Risk limits: `MAX_RISK_PER_TRADE=1.5%`, `DAILY_LOSS_LIMIT=5%`, `MAX_DRAWDOWN=10%`

Environment variables in `.env`:
- `ALPACA_API_KEY`, `ALPACA_SECRET_KEY` - **LIVE** trading credentials
- `ALPACA_BASE_URL`: `https://api.alpaca.markets` (LIVE API)
- `TWILIO_*` - SMS notification credentials

## Risk Overlay (Crash Protection)

The Risk Overlay is a state machine that provides crash protection by blocking risky actions during market crashes and gradually re-entering after stability returns.

### State Machine Modes

| Mode | Description | Buy Orders | Rebalance-Down | Position Size |
|------|-------------|------------|----------------|---------------|
| **NORMAL** | Full trading | Allowed | Allowed | 100% |
| **RISK_OFF** | Crash protection | BLOCKED | BLOCKED | 0% |
| **RECOVERY** | Gradual re-entry | Allowed (reduced) | BLOCKED | 25-100% (ramping) |

### Trigger Signals (2-of-3 required)

RISK_OFF activates when **2 or more** of these signals fire:

1. **Momentum Shock**: Returns < -1.5% (`RISK_OFF_MOMENTUM_THRESHOLD`)
2. **ADX Downtrend**: ADX > 35 AND direction == "down" (`RISK_OFF_ADX_THRESHOLD`)
3. **Correlation Spike**: Max correlation > 0.90 (`RISK_OFF_CORRELATION_THRESHOLD`)

Optional (disabled by default):
- **Drawdown Velocity**: Equity declining > 2%/hour (`RISK_OFF_DRAWDOWN_VELOCITY_ENABLED`)

### RISK_OFF Behavior

When RISK_OFF activates:
1. All new BUY orders are blocked
2. Grid-owned BUY limit orders are cancelled (safety: won't cancel untracked orders)
3. Rebalance-down is blocked
4. All SELL orders are ALLOWED (override other filters)
5. SMS alert is sent

### RECOVERY Ramp

After RISK_OFF holds for 20 minutes and stability gate passes:
1. **Stage 0**: 25% position size (10 bars to advance)
2. **Stage 1**: 50% position size (10 bars to advance)
3. **Stage 2**: 75% position size (10 bars to advance)
4. **Stage 3**: 100% position size → NORMAL

Stability gate requires:
- Momentum > -0.5%
- Correlation < 0.85
- No new price lows

**Relapse**: If triggers fire during RECOVERY → snap back to RISK_OFF immediately.

### API Endpoints

```bash
# Get current overlay status
curl http://localhost:8001/api/risk/overlay

# Get telemetry ($ amounts protected)
curl http://localhost:8001/api/risk/overlay/telemetry

# Manual override to RISK_OFF
curl -X POST http://localhost:8001/api/risk/overlay \
  -H "Content-Type: application/json" \
  -d '{"mode": "RISK_OFF", "reason": "Manual protection"}'

# Clear manual override (return to automatic)
curl -X POST http://localhost:8001/api/risk/overlay \
  -H "Content-Type: application/json" \
  -d '{"mode": "clear"}'
```

### Config Settings

In `config_ultra.py`:
```python
RISK_OVERLAY_ENABLED = True
RISK_OFF_TRIGGERS_REQUIRED = 2
RISK_OFF_MIN_HOLD_MINUTES = 20
RISK_OFF_MOMENTUM_THRESHOLD = -0.015
RISK_OFF_ADX_THRESHOLD = 35
RISK_OFF_CORRELATION_THRESHOLD = 0.90
RECOVERY_POSITION_RAMP = [0.25, 0.5, 0.75, 1.0]
```

### State Persistence

- **Risk overlay state**: `data/state/risk-overlay.json`
- Telemetry tracks $ amounts: `avoided_buys_notional`, `cancelled_limits_notional`

## Orchestrator (Inventory Management)

The Orchestrator is a thin meta-controller that adds inventory episode tracking and staged liquidation on top of existing signals. It runs AFTER windfall/stop-loss, BEFORE grid orders.

**Critical**: Orchestrator never overrides RiskOverlay decisions. It can only add restrictions.

### Orchestrator Modes

| Mode | Description | Buy Orders | Size Multiplier |
|------|-------------|------------|-----------------|
| **GRID_FULL** | Normal operation | Allowed | 100% |
| **GRID_REDUCED** | High inventory (≥100%) | Allowed | 50% |
| **DEFENSIVE** | Very high inventory (≥150%) | BLOCKED | 0% |

### Inventory Episode Tracking

Tracks how long we've been "stuck" holding meaningful inventory:
- Episode starts when `inventory_pct >= 30%`
- Episode resets when `inventory_pct <= 10%`
- Episode age used for liquidation trigger thresholds

### Staged Liquidation (NORMAL only)

Liquidation orders are ONLY placed in overlay mode NORMAL:

| Trigger | Conditions | Action |
|---------|------------|--------|
| **TP Trim** | Episode ≥24h, P/L ≥+0.3%, inventory ≥120% | Reduce to 100% |
| **Loss Cut** | Episode ≥48h, P/L ≤-2%, inventory ≥130% | Reduce 25% of excess |
| **Max Age** | Episode ≥72h, inventory ≥120% | Reduce to 100% |

### API Endpoints

```bash
# Get orchestrator status
curl http://localhost:8001/api/orchestrator/status

# Get symbol-specific status
curl http://localhost:8001/api/orchestrator/symbol/AVAX-USD

# Get telemetry ($ amounts)
curl http://localhost:8001/api/orchestrator/telemetry
```

### Config Settings

In `config_ultra.py`:
```python
ORCHESTRATOR_ENABLED = True           # Enable evaluation
ORCHESTRATOR_ENFORCE = False          # False = shadow mode
ORCHESTRATOR_LIQUIDATION_ENABLED = False  # Enable staged liquidation
ORCHESTRATOR_COOLDOWN_MINUTES = 60    # Min time between mode changes
DEFENSIVE_INVENTORY_PCT = 150         # Enter DEFENSIVE threshold
GRID_REDUCED_ENTER_PCT = 100          # Enter GRID_REDUCED threshold
```

### State Persistence

- **Orchestrator state**: `data/state/orchestrator.json`
- Telemetry tracks $ amounts: `buys_blocked_notional`, `liquidations_placed_notional`

## Downtrend Protection (Inventory Control)

Two additional layers prevent inventory buildup during downtrends:

### Developing Downtrend Size Multiplier

When ADX is between 25-35 (developing trend) AND direction is DOWN, buy size is reduced to 50%.

| ADX Level | Direction | Action |
|-----------|-----------|--------|
| < 25 | Any | Full size (100%) |
| 25-35 | DOWN | Reduced size (50%) |
| > 35 | DOWN | Buys blocked (STRONG DOWNTREND) |
| Any | UP/NEUTRAL | Full size (100%) |

**Config:**
```python
REGIME_ADX_DEVELOPING = 25              # "Developing trend" threshold
DEVELOPING_DOWNTREND_SIZE_MULT = 0.50   # 50% size when ADX 25-35 + DOWN
```

### Consecutive Down Bars Block

Simple crash guard: blocks buys after 3+ consecutive red (down) candles.

**How it works:**
1. Counts consecutive bars where `close < open`
2. If count >= threshold, blocks all buys for that symbol
3. Resets when a green bar appears

**Config:**
```python
CONSECUTIVE_DOWN_BARS_ENABLED = True
CONSECUTIVE_DOWN_BARS_BLOCK = 3         # Block after 3 consecutive down bars
```

**Log markers:**
- `[DOWN] Skipping BUY {symbol}: 3 consecutive down bars`
- `[REGIME] {symbol}: DEVELOPING DOWNTREND (ADX=30) - Size reduced to 50%`

## Notification System (SMS Alerts)

The notifier (`src/notifications/notifier.py`) polls the bot API and sends SMS alerts via Twilio.

### Reliability Features (Added Dec 23, 2025)

| Feature | Description |
|---------|-------------|
| **Database Persistence** | All state stored in SQLite, survives restarts |
| **SMS Retry** | 3 attempts with exponential backoff (5s, 10s, 20s) |
| **Failed SMS Queue** | Queued in DB for retry on next poll cycle |
| **API Resilience** | Exponential backoff + circuit breaker (5 failures = alert) |
| **Heartbeat** | Updates DB every 60s poll cycle |
| **Watchdog** | Cron job checks DB heartbeat, auto-restarts if stale |

### Alert Types
- **Trade alerts**: New order executions
- **Risk alerts**: Trading halted, drawdown threshold
- **Risk overlay**: RISK_OFF/RECOVERY mode transitions
- **Stale data**: Bot stream disconnected (5 min threshold)
- **API down**: Circuit breaker opened after 5 consecutive failures

### Watchdog Setup (Bot + Notifier)

Both bot and notifier have watchdog scripts that auto-restart on crash. These run via **launchd** (not cron) due to macOS security restrictions on external volumes.

**Architecture (LIVE Instance):**
| Component | Location |
|-----------|----------|
| Bot watchdog | `~/Library/Application Support/BLUEBIRD-LIVE/run-check-bot.sh` |
| Notifier watchdog | `~/Library/Application Support/BLUEBIRD-LIVE/run-check-notifier.sh` |
| Bot LaunchAgent | `~/Library/LaunchAgents/com.bluebird-live.watchdog-bot.plist` |
| Notifier LaunchAgent | `~/Library/LaunchAgents/com.bluebird-live.watchdog-notifier.plist` |
| Durable state | `~/Library/Application Support/BLUEBIRD-LIVE/state/` |
| Logs | `/tmp/bluebird-live-watchdog.log` |

**Why local filesystem?** macOS launchd cannot execute scripts on external APFS volumes with `noowners` flag (EPERM). The local watchdog scripts are full copies that read the database on the external volume but write state locally.

**State files (durable, survives reboot):**
- `crash-loop-bot.json` - Pauses restarts after 3 crashes in 30 min
- `crash-loop-notifier.json` - Same for notifier
- `pending-alerts.txt` - Alerts queued for SMS
- `disk-alert.json` - Tracks last disk space alert (once per day)

```bash
# Check LaunchAgents status
launchctl list | grep bluebird-live

# Manual checks (run local scripts directly)
bash "$HOME/Library/Application Support/BLUEBIRD-LIVE/run-check-bot.sh"
bash "$HOME/Library/Application Support/BLUEBIRD-LIVE/run-check-notifier.sh"

# View watchdog log
tail -f /tmp/bluebird-live-watchdog.log

# Reset crash loop pause (after fixing root cause)
rm "$HOME/Library/Application Support/BLUEBIRD-LIVE/state/crash-loop-bot.json"
rm "$HOME/Library/Application Support/BLUEBIRD-LIVE/state/crash-loop-notifier.json"
```

### Check Service Status
```bash
# Bot heartbeat (via database)
sqlite3 data/bluebird.db "SELECT pid, last_heartbeat, overlay_mode, status FROM bot_status"

# Notifier heartbeat (via database)
sqlite3 data/bluebird.db "SELECT pid, last_heartbeat, status, sms_today FROM notifier_status"

# Via start.py
python3 start.py --status
```

## Important Constraints

### Single Instance Protection
Both bot and notifier use process locks. If you try to start a second instance:
```
ERROR: BLUEBIRD is already running!
  PID: 12345
  Started: 2025-12-10T22:54:09
```

### Archived Code
Old prediction-based bots are in `archive/old_bots/`. The system now ONLY uses grid trading (`bot_grid.py`). Do not restore or use:
- `main.py`, `bot.py`, `bot_multi.py`, `bot_ultra.py`
- `train*.py`, `backtest*.py`

### Alpaca Fees
Round-trip cost is ~0.60% (0.25% taker + spread). Grid spacing must exceed this to be profitable.

### Grid Sizing Validation
On startup, `initialize_grids()` validates each grid's `investment_per_grid`:
- **MIN_INVESTMENT_PER_GRID = $12** (above Alpaca's $10 minimum notional)
- **STALE_GRID_THRESHOLD = 25%** (rebuild if < 25% of expected from equity)

If a restored grid has stale/tiny sizing, it's automatically rebuilt:
1. Cancel open Alpaca orders for that symbol
2. Clear local order tracking
3. Preserve performance stats (profit, trades, cycles)
4. Create fresh grid with proper sizing from current equity

### Unmatched Fill Handling
When a fill can't match any grid level (`no_level_match`), grace window quarantine prevents log spam:
- `source="reconcile"` → quarantine immediately
- Already seen before → quarantine on 2nd occurrence
- `pending.created_at > 2 min old` → quarantine immediately
- First fresh live occurrence → grace period (one more retry)

Quarantined fills are stored in `unmatched_fills` for audit but won't be reprocessed.

## API Endpoints

Key endpoints on `http://localhost:8001` (LIVE):
- `GET /health` - Service health and regime
- `GET /api/risk/status` - Current equity, daily P/L, drawdown
- `GET /api/risk/overlay` - Risk overlay mode, triggers, telemetry
- `GET /api/risk/overlay/telemetry` - $ amounts of buys avoided/cancelled
- `POST /api/risk/overlay` - Manual override (NORMAL, RISK_OFF, or clear)
- `GET /api/orchestrator/status` - Orchestrator mode, inventory %, episodes
- `GET /api/orchestrator/symbol/{symbol}` - Per-symbol orchestrator status
- `GET /api/orchestrator/telemetry` - $ amounts blocked/liquidated
- `GET /api/grid/status` - Grid levels and fill status
- `GET /api/db/stats` - Database statistics
- `GET /api/db/reconcile` - Sync database with Alpaca
- `WS /ws` - Real-time updates for dashboard

## Dashboard

React app in `dashboard/` using Vite (built with VITE_API_PORT=8001):
```bash
cd dashboard
npm install
VITE_API_PORT=8001 npm run dev -- --port 5174   # Development server on :5174
VITE_API_PORT=8001 npm run build                 # Production build to dist/
```

## Performance Updates

When user asks for a "performance update", follow this exact sequence:

### 1. Always check system time FIRST
```bash
date  # Get actual Mac system time - NEVER guess the date
```

### 2. Query data sources (use the RIGHT source for each metric)

```bash
# Bot health (LIVE on port 8001)
curl -s http://localhost:8001/health

# Equity & Daily P/L — USE DATABASE (accurate snapshots)
sqlite3 data/bluebird.db "SELECT timestamp, equity, daily_pnl, daily_pnl_pct FROM equity_snapshots ORDER BY timestamp DESC LIMIT 1;"

# Recent daily summaries — USE DATABASE
sqlite3 data/bluebird.db "SELECT date, realized_pnl, starting_equity, ending_equity FROM daily_summary ORDER BY date DESC LIMIT 3;"

# Current positions — USE ALPACA API (source of truth)
curl -s http://localhost:8001/api/positions

# Risk overlay status
curl -s http://localhost:8001/api/risk/overlay
```

**Data source accuracy:**
| Metric | Source | Why |
|--------|--------|-----|
| Positions | Alpaca API | DB trades incomplete (pre-Dec 10 missing) |
| Equity | DB equity_snapshots | Written every minute, accurate |
| Daily P/L | DB daily_summary | Aggregated correctly |
| Unrealized P/L | Alpaca API | Real-time from broker |

### 3. Calculate grid profit
Grid profit = Current Equity - Grid Starting Equity ($90,276.26 from Dec 2, 2025)

### 4. Format output with date header
Always include the actual date/time in the report header:
```
## Performance Update — [Day] [Month] [Date], [Year] @ [Time] [TZ]
```

**CRITICAL**: Never infer the date from conversation flow. Always run `date` command.
