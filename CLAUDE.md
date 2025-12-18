# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

BLUEBIRD 4.0 is a cryptocurrency grid trading bot that trades on Alpaca. It uses a grid trading strategy (not prediction-based) that profits from sideways market volatility by placing buy/sell orders at regular price intervals.

**Key Insight**: The system originally used ML predictions but achieved only 21% win rate. Grid trading was adopted because it thrives in sideways markets without needing predictions.

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
curl http://localhost:8000/health          # Bot health
curl http://localhost:8000/api/risk/status # Risk & P/L
ls /tmp/bluebird/*.pid                     # Running services
```

## Architecture

### Services (3 total)
| Service | Entry Point | Port | Lock |
|---------|-------------|------|------|
| Grid Bot | `src/api/server.py` | 8000 | `bluebird-bot` |
| Dashboard | `dashboard/` (Vite/React) | 5173 | Port-based |
| Notifier | `src/notifications/notifier.py` | - | `bluebird-notifier` |

### Core Components
- **`src/api/server.py`** - FastAPI server with WebSocket for real-time dashboard updates. Runs the grid bot via `run_grid_bot()`.
- **`src/execution/bot_grid.py`** - Grid trading bot implementation. Handles order execution, risk controls, and state management.
- **`src/strategy/grid_trading.py`** - Grid trading strategy logic. Creates grid levels, tracks fills, calculates profits.
- **`src/strategy/risk_overlay.py`** - Risk overlay state machine (NORMAL/RISK_OFF/RECOVERY). Provides crash protection.
- **`src/database/db.py`** - SQLite database for persistent trade/equity/order storage at `data/bluebird.db`.
- **`src/utils/process_lock.py`** - Single-instance protection using file locks in `/tmp/bluebird/`.
- **`config_ultra.py`** - All trading configuration (symbols, risk limits, grid settings).

### Data Flow
1. Alpaca WebSocket streams real-time price bars
2. `bot_grid.py` checks each bar against grid levels
3. If price hits a grid level → execute market order via `alpaca_client.py`
4. Record trade in SQLite database
5. Broadcast update to dashboard via WebSocket
6. Notifier polls API and sends SMS alerts for significant events

### State Persistence
- **Grid state**: `/tmp/bluebird-grid-state.json`
- **Risk state**: `/tmp/bluebird-risk-state.json`
- **Risk overlay state**: `/tmp/bluebird-risk-overlay.json`
- **Daily equity**: `/tmp/bluebird-daily-equity.json`
- **Database**: `data/bluebird.db`
- **Lock files**: `/tmp/bluebird/*.lock`, `/tmp/bluebird/*.pid`

## Configuration

All settings in `config_ultra.py`:
- `SYMBOLS`: Trading pairs (BTC/USD, SOL/USD, LTC/USD, AVAX/USD)
- `GRID_CONFIGS`: Per-symbol grid settings (levels, spacing, size)
- Risk limits: `MAX_RISK_PER_TRADE=1.5%`, `DAILY_LOSS_LIMIT=5%`, `MAX_DRAWDOWN=10%`

Environment variables in `.env`:
- `ALPACA_API_KEY`, `ALPACA_SECRET_KEY` - Trading credentials
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
curl http://localhost:8000/api/risk/overlay

# Get telemetry ($ amounts protected)
curl http://localhost:8000/api/risk/overlay/telemetry

# Manual override to RISK_OFF
curl -X POST http://localhost:8000/api/risk/overlay \
  -H "Content-Type: application/json" \
  -d '{"mode": "RISK_OFF", "reason": "Manual protection"}'

# Clear manual override (return to automatic)
curl -X POST http://localhost:8000/api/risk/overlay \
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

- **Risk overlay state**: `/tmp/bluebird-risk-overlay.json`
- Telemetry tracks $ amounts: `avoided_buys_notional`, `cancelled_limits_notional`

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

Key endpoints on `http://localhost:8000`:
- `GET /health` - Service health and regime
- `GET /api/risk/status` - Current equity, daily P/L, drawdown
- `GET /api/risk/overlay` - Risk overlay mode, triggers, telemetry
- `GET /api/risk/overlay/telemetry` - $ amounts of buys avoided/cancelled
- `POST /api/risk/overlay` - Manual override (NORMAL, RISK_OFF, or clear)
- `GET /api/grid/status` - Grid levels and fill status
- `GET /api/db/stats` - Database statistics
- `GET /api/db/reconcile` - Sync database with Alpaca
- `WS /ws` - Real-time updates for dashboard

## Dashboard

React app in `dashboard/` using Vite:
```bash
cd dashboard
npm install
npm run dev      # Development server on :5173
npm run build    # Production build to dist/
```
