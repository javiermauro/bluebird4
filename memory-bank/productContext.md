# Product Context — Why BLUEBIRD exists

## Problem Being Solved
Directional prediction was not reliably profitable (21% win rate); the system pivoted to **grid trading** to monetize volatility and mean reversion without needing to forecast direction.

## Core UX / Operator Experience
- **One command to start/stop/status**: `python start.py`, `--all`, `--stop`, `--status`
- **Dashboard visibility** (Vite/React) for quick "is it alive?" + trading status checks
- **Real-time WebSocket updates** for live P/L tracking
- **API-first observability** for health and risk state:
  - `GET /health`
  - `GET /api/risk/status`
  - `GET /api/risk/overlay`
  - `GET /api/orchestrator/status`
  - `GET /api/grid/status`

## Product North Star
- The system's only goal is to **make money** (profitability). Everything else (risk overlay, orchestrator, sizing, uptime) exists to support sustainable profitability and avoid catastrophic loss.

## Key Product Behaviors
- **Orders persist on exchange**: resting limit orders can fill while the bot is offline
- **Multi-layer protection**:
  - Risk Overlay: crash protection (RISK_OFF blocks buys during market stress)
  - Orchestrator: inventory management (prevents bag accumulation)
  - Downtrend protection: reduces size in developing downtrends
- **State persistence**:
  - SQLite DB: `data/bluebird.db`
  - Persistent state files: `data/state/*.json` (survives reboot)
  - Lock files: `/tmp/bluebird/` (ephemeral, cleared on reboot)
- **Resilience guardrails**:
  - Single-instance locks in `/tmp/bluebird/`
  - Staged recovery (25% → 50% → 75% → 100%) after RISK_OFF
  - Startup reconciliation with Alpaca

## What "Good" Looks Like (Day-to-Day)
- Operator can quickly answer:
  - "Are we trading?" → Check dashboard or `/health`
  - "Are we in crash protection?" → Check Risk Overlay panel
  - "Is inventory building up?" → Check Orchestrator panel
  - "Do our DB records match Alpaca?" → Run `/api/db/reconcile`
  - "What's today's P/L?" → Dashboard or `/api/risk/status`

## Success Metrics
- **Daily P/L**: Target consistent positive days
- **Grid P/L**: Track cumulative since grid era start (Dec 2)
- **Fee efficiency**: Maintain 40x+ profit/fee ratio
- **Inventory health**: Keep below 100% to avoid GRID_REDUCED mode
- **Drawdown**: Minimize daily drawdown, especially during volatile periods

## Automated Maintenance (Dec 25, 2025)
The system now has comprehensive automated maintenance:
- **Watchdogs**: Bot and notifier auto-restart on crash (every 5 min check)
- **Backups**: Daily database backup at 3 AM (7-day retention)
- **Log rotation**: Daily at 5 AM (50 MB limit, 3 rotations)
- **DB cleanup**: Manual script for 90-day data retention

All maintenance is hands-off except for occasional `cleanup_db.py --execute`.
