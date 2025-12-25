# Tech Context — Stack, Dependencies, Environment

## Runtime / Language
- **Python** (project root contains `start.py`, `src/` package)
- **Node.js** for dashboard (`dashboard/` Vite/React)

## Key Python Dependencies (from `requirements.txt`)
- `alpaca-py` (broker/exchange integration)
- `fastapi`, `uvicorn`, `websockets` (API + realtime)
- `requests` (API calls / internal client usage)
- `python-dotenv` (env var loading)
- `pandas`, `numpy` (data manipulation)
- `ta-lib` (technical indicators)
- `scikit-learn`, `lightgbm` (legacy/aux ML components; current edge is grid-first)
- `schedule` (job scheduling utilities)
- `twilio` (SMS notifications)

## Ports / Local Services
- Bot API: **8000**
- Dashboard dev server: **5173**

## Data Storage
- **SQLite Database**: `data/bluebird.db` (trades, equity, orders, notifications)
- **Persistent State Files**: `data/state/*.json` (survives reboot)
  - `grid-state.json` - Grid levels, fills, pending orders
  - `risk-overlay.json` - RISK_OFF/RECOVERY mode and triggers
  - `orchestrator.json` - Inventory episode tracking
  - `daily-equity.json` - Daily P&L tracking
  - `alltime-equity.json` - All-time performance stats
  - `circuit-breaker.json` - Max drawdown/stop-loss flags
  - `windfall-log.json` - Windfall profit captures
  - `watchdog.json` - Notifier watchdog state
- **Process/Lock Files**: `/tmp/bluebird/*.lock`, `/tmp/bluebird/*.pid` (cleared on reboot - intentional)
- **Log Files**: `/tmp/bluebird-bot.log`, `/tmp/bluebird-notifier.log`

## Operational Constraints / Notes
- **macOS** operators commonly run with `caffeinate -i` to prevent sleep.
- Avoid editing `.env` without explicit permission (contains secrets).
- State files in `data/state/` are now persistent across reboots (moved from /tmp Dec 25).
- Lock files in `/tmp/bluebird/` are intentionally ephemeral (prevent stale locks after reboot).



## Maintenance Scripts (added Dec 25, 2025)
All scripts in `scripts/` directory:
- `check_bot.sh` - Bot watchdog (cron every 5 min)
- `check_notifier.sh` - Notifier watchdog (cron every 5 min)
- `backup_db.sh` - Database backup (cron daily 3 AM)
- `rotate_logs.sh` - Log rotation (cron daily 5 AM)
- `cleanup_db.py` - Database cleanup (manual, use `--execute`)

## Backup Storage
- `data/backups/` - Database backups (7-day retention)
- Backup format: `bluebird-YYYYMMDD.db`

## Log Files
- `/tmp/bluebird-bot.log` - Bot log
- `/tmp/bluebird-notifier.log` - Notifier log
- `/tmp/bluebird-watchdog.log` - Watchdog logs (both)
- `/tmp/bluebird-backup.log` - Backup script logs
- `/tmp/bluebird-logrotate.log` - Rotation script logs
