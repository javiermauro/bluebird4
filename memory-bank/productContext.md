# Product Context — Why BLUEBIRD exists

## Problem Being Solved
Directional prediction was not reliably profitable; the system pivoted to **grid trading** to monetize volatility and mean reversion without needing to forecast direction.

## Core UX / Operator Experience
- **One command to start/stop/status**: `python start.py`, `--all`, `--stop`, `--status`.
- **Dashboard visibility** (Vite/React) for quick “is it alive?” + trading status checks.
- **API-first observability** for health and risk state:
  - `GET /health`
  - `GET /api/risk/status`
  - `GET /api/risk/overlay`
  - `GET /api/db/reconcile`
  - `GET /api/grid/status`

## Product North Star
- The system’s only goal is to **make money** (profitability). Everything else (risk overlay, reconciliation, sizing, uptime) exists to support sustainable profitability and avoid catastrophic loss.

## Key Product Behaviors
- **Orders persist on exchange**: resting limit orders can fill while the bot is offline.
- **State persistence**:
  - SQLite DB: `data/bluebird.db`
  - Runtime state files: `/tmp/bluebird-*.json` (grid/risk/overlay/equity)
- **Resilience guardrails**:
  - Single-instance locks in `/tmp/bluebird/`
  - Risk overlay state machine (NORMAL/RISK_OFF/RECOVERY)
  - Startup reconciliation and optional cancellation of untracked open orders

## What “Good” Looks Like (Day-to-Day)
- Operator can quickly answer:
  - “Are we trading?”
  - “Are we in crash protection?”
  - “Do our DB records match Alpaca?”
  - “What changed since last night?”


