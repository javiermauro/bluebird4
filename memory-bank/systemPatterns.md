# System Patterns — Architecture & Design

## Services
- **Grid Bot API**: `src/api/server.py` (FastAPI) — `http://localhost:8000`
  - WebSocket: `WS /ws` for real-time dashboard updates
- **Dashboard**: `dashboard/` (Vite/React) — `http://localhost:5173`
- **Notifier**: `src/notifications/notifier.py` (polls API, sends SMS)

## Core Modules (Backend)
- **Execution**: `src/execution/bot_grid.py`
  - Owns bar ingestion, order placement, safety gates, and reconciliation hooks.
- **Strategy**: `src/strategy/grid_trading.py`
  - Grid-level creation/tracking, profits, cycles.
- **Risk Overlay (Crash Protection)**: `src/strategy/risk_overlay.py`
  - State machine: NORMAL → RISK_OFF → RECOVERY → NORMAL
  - Triggers: 2-of-3 (momentum shock, ADX downtrend, correlation spike)
- **Orchestrator (Inventory Management)**: `src/strategy/orchestrator.py`
  - Meta-controller that runs AFTER windfall/stop-loss, BEFORE grid orders
  - Modes: GRID_FULL → GRID_REDUCED (100%+) → DEFENSIVE (150%+)
  - Episode tracking: starts at 30% inventory, resets at 10%
  - Staged liquidation: TP Trim (24h), Loss Cut (48h), Max Age (72h)
  - Critical: Never overrides Risk Overlay decisions
- **Database**: `src/database/db.py` using SQLite at `data/bluebird.db`
- **Single-instance / process locks**: `src/utils/process_lock.py` using `/tmp/bluebird/`

## State & Persistence
- **DB (durable)**: `data/bluebird.db`
- **Persistent JSON state** in `data/state/` (survives reboot):
  - `grid-state.json` - Grid levels, fills, pending orders
  - `risk-overlay.json` - RISK_OFF/RECOVERY mode and triggers
  - `orchestrator.json` - Inventory episode tracking
  - `daily-equity.json` - Daily P&L tracking
  - `alltime-equity.json` - All-time performance stats
  - `circuit-breaker.json` - Max drawdown/stop-loss flags
  - `windfall-log.json` - Windfall profit captures
- **Locks / PIDs** (ephemeral, intentionally cleared on reboot):
  - `/tmp/bluebird/*.lock`
  - `/tmp/bluebird/*.pid`

## Safety Gating Order (Conceptual)
When deciding whether to buy:
1. **Risk Overlay gate** (highest priority): blocks buys entirely in RISK_OFF
2. **Orchestrator gate**: blocks buys in DEFENSIVE mode, reduces size in GRID_REDUCED
3. **Downtrend protection**: 50% size when ADX 25-35 + DOWN direction
4. **Consecutive down bars**: blocks buys after 3+ red candles
5. **Regime/momentum/time gates**: reduce frequency / avoid poor conditions
6. **Allocation caps**: prevent concentration and overbuying

## Dashboard Panels
- **Risk Overlay Panel**: Red/gold color scheme, shows NORMAL/RISK_OFF/RECOVERY status
- **Orchestrator Panel**: Teal/cyan color scheme, shows inventory %, episode tracking, telemetry
- Both panels are collapsible with expanded detail views

## Operational Pattern: Reconcile on Recovery
- After restarts/outages, use `/api/db/reconcile` to align local DB/tracking with Alpaca fills
- Strategy is resilient to downtime because **existing limit orders live on the exchange**

## Reporting / "Day Boundary" Pattern
- Operator-facing rollups follow **Mac mini local time** for day boundaries
- Daily P&L matches how the operator thinks about trading days

## Grid Trading Metrics
- **Cycle win rate**: 100% by design (spacing > fees = guaranteed profit per cycle)
- **Fee efficiency**: Target 40x+ profit/fee ratio
- **Spacing requirements**: Must exceed ~0.60% round-trip fees to be profitable

## API Endpoints (Key)
- `GET /health` - Service health and regime
- `GET /api/risk/status` - Equity, daily P/L, drawdown
- `GET /api/risk/overlay` - Risk overlay mode, triggers, telemetry
- `GET /api/orchestrator/status` - Orchestrator mode, inventory, episodes
- `GET /api/grid/status` - Grid levels, fill status, profits
- `POST /api/risk/overlay` - Manual override (RISK_OFF, NORMAL, or clear)

## Deprecated Patterns
- Prediction-first bots under `archive/old_bots/` are considered legacy
- Current system is grid-first with protection layers

## Maintenance Infrastructure (Dec 25-30, 2025)

### LaunchAgents (Auto-Start After Reboot)
All services managed by launchd for reliable auto-restart:

| LaunchAgent | Purpose | KeepAlive |
|-------------|---------|-----------|
| `com.bluebird.bot` | Grid trading bot | Yes |
| `com.bluebird.notifier` | SMS notifications | Yes |
| `com.bluebird.dashboard` | Static dashboard server | No |
| `com.bluebird.watchdog-bot` | Backup bot health monitor (60s) | No |
| `com.bluebird.watchdog-notifier` | Backup notifier health monitor (60s) | No |
| `com.bluebird.monitor` | Opens Terminal with status monitor | No |

### Watchdog Pattern (Backup)
- Watchdogs run every 60 seconds via launchd (not cron - macOS security)
- Primary liveness signal: `/health` HTTP endpoint
- Secondary signal: DB heartbeat (written every 60s)
- If `/health` unreachable, watchdog restarts bot (with crash loop protection)
- Scripts live in `~/Library/Application Support/BLUEBIRD/`

### Cron Jobs (Maintenance Only)
| Schedule | Script | Purpose |
|----------|--------|---------|
| `0 3 * * *` | `backup_db.sh` | Daily DB backup |
| `0 5 * * *` | `rotate_logs.sh` | Daily log rotation |

### Maintenance Scripts (in `scripts/`)
- `sync-watchdog-scripts.sh` - Sync repo scripts to local LaunchAgent paths
- `check_bot.sh` / `check_notifier.sh` - Watchdogs (run from local copy)
- `backup_db.sh` - SQLite backup, 7-day retention
- `rotate_logs.sh` - Log rotation, 50 MB limit
- `cleanup_db.py` - Manual DB cleanup (90-day retention)
- `monitor_services.sh` - Real-time status display
