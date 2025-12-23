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
- **Database**: `src/database/db.py` using SQLite at `data/bluebird.db`
- **Single-instance / process locks**: `src/utils/process_lock.py` using `/tmp/bluebird/`

## State & Persistence
- **DB (durable)**: `data/bluebird.db`
- **Runtime JSON state** (fast restore + operator visibility):
  - `/tmp/bluebird-grid-state.json`
  - `/tmp/bluebird-risk-state.json`
  - `/tmp/bluebird-risk-overlay.json`
  - `/tmp/bluebird-daily-equity.json`
- **Locks / PIDs**:
  - `/tmp/bluebird/*.lock`
  - `/tmp/bluebird/*.pid`

## Safety Gating Order (Conceptual)
When deciding whether to buy:
- **Risk overlay gate** (highest priority): can block buys entirely in RISK_OFF.
- **Regime/momentum/time gates**: reduce frequency / avoid poor conditions.
- **Allocation caps**: prevent concentration and overbuying.
- **Crash-guard heuristics**: e.g., consecutive down bars block (inventory control).

## Operational Pattern: Reconcile on Recovery
- After restarts/outages, use a reconcile endpoint (`/api/db/reconcile`) to align local DB/tracking with Alpaca fills.
- Strategy is resilient to downtime because **existing limit orders live on the exchange** and can fill while the bot is offline.

## Reporting / “Day Boundary” Pattern
- Operator-facing rollups (e.g., `daily_summary`) should follow **Mac mini local time** for day boundaries so daily P&L matches how the operator thinks.
- To avoid DST ambiguity, store the local timezone name (or offset) alongside the summary logic (either in config, logs, or an explicit DB field if added later).

## AI / ML Components Present (Reference Inventory)
These exist in the repo for experimentation but are **not the primary live edge** today (grid-first).
- **`src/strategy/adaptive_ai.py`**: `AdaptiveAI` — indicator computation + ML prediction + confidence/reasoning.
- **`src/strategy/ml_strategy.py`**: `MLStrategy` — legacy ML trading loop scaffold.
- **`src/strategy/ultra_strategy.py`**: `UltraStrategy` — regime-driven controller (multi-strategy concept).
- **`src/models/predictor.py`**: `Predictor` wrapper around a model’s `predict()` for probabilities.

Recommended exploration pattern:
- Start **shadow-mode** (log signals only), then allow AI to **only reduce risk** (pause buys / reduce size) if it proves it improves profitability/risk.

## Logging Pattern (Current Reality)
- Notifier writes to a file (e.g., `/tmp/bluebird-notifier.log`) via a `FileHandler`.
- Bot logging depends on how it is launched:
  - If launched using shell redirection (per `CLAUDE.md`), output goes to `/tmp/bluebird-bot.log`.
  - If launched via `start.py`, bot stdout/stderr is piped (`subprocess.PIPE`) and may not persist to a log file unless explicitly drained/redirected.

## Deprecated Patterns
- Prediction-first bots under `archive/old_bots/` are considered legacy; current system is grid-first.


