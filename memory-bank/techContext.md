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
- SQLite: `data/bluebird.db`
- Runtime state/locks: `/tmp/bluebird/` and `/tmp/bluebird-*.json`

## Operational Constraints / Notes
- **macOS** operators commonly run with `caffeinate -i` to prevent sleep.
- Avoid editing `.env` without explicit permission (contains secrets).
- Prefer persisted logs/state outside ephemeral temp paths when doing incident forensics.


