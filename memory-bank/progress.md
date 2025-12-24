# Progress — Status & History

## Current Status
- [2025-12-23 10:35] System healthy, NORMAL mode, all protection layers active
- [2025-12-23 10:35] Notification system overhauled - now database-backed with retry logic

## Recent Work (High Signal)

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
- **P3**: Bot auto-restart after reboot not configured (LaunchAgent)
- ~~P2: Notifier state persistence~~ FIXED Dec 23 - now database-backed
- ~~P3: Notifier monitoring~~ FIXED Dec 23 - watchdog cron job active

## Performance Tracking
| Date | Daily P/L | Grid P/L | Notes |
|------|-----------|----------|-------|
| Dec 21 | +$1,685 (+1.83%) | +$3,341 | Best day, first RISK_OFF event |
| Dec 20 | ~+$200 | +$1,656 | Post-crash recovery |
| Dec 2 | Start | $0 | Grid trading era begins |
