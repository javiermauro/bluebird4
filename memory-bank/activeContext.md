# Active Context — Current Focus

## Reminders
- **[2026-01-29] AVAX REVIEW DATE** — Check if AVAX recovered. Exit trigger: below $11.00. Success: above $13.15.
- **[2026-01-22] DECISION: HOLD AVAX** — Analysis complete. Paper bot proves AVAX outperforms SOL 3.3x. Issue is timing, not coin. Do NOT switch.

## Now
- [2026-01-27 PM] **DECISION: MONITOR & WAIT** - Phase 1 complete (100%), Issue #5 complete, Phase 6 verified. All critical monitoring layers active. Focus on AVAX review (Jan 29) and system performance.
- [2026-01-27 PM] **Issue #5 Complete** - pytest verified working! All 101 tests passing (grid matching, risk overlay, fill detection).
- [2026-01-27 PM] **Phase 1 Complete (100%)** - All critical issues fixed! Circuit breaker age ✓, Error rate monitoring ✓, Zero fills alert ✓, State files verified ✓.
- [2026-01-27 PM] **Phase 6 Complete** - Error rate monitoring implemented and tested. Catches silent failures (10+ errors/hour triggers SMS alert).
- [2026-01-26 PM] **ALL FEATURES VERIFIED & COMMITTED** - Phase 1-5 + buy throttles + recovery trailing all deployed and verified working.
- [2026-01-26 PM] **BUY THROTTLE RESET** - 24h window rolled over, both AVAX and LTC now eligible for buys (0 fills in last 24h).

## System Health (LIVE Instance) — Updated Jan 26, 2026 @ 3:30 PM EST
- **Bot**: Healthy, NORMAL mode, port 8001
- **Position Value**: $959.59
- **Unrealized P/L**: **-$65.52** (AVAX -$66.79, LTC +$1.27)
- **AVAX Position**: 77.90 qty @ $12.72 avg (current $11.86)
- **LTC Position**: 0.51 qty @ $67.21 avg (current $69.70)
- **Breakeven Target**: $12.72 (+7.2% needed)
- **Risk Overlay**: NORMAL
- **Orchestrator**: AVAX grid_reduced (10 days episode), LTC grid_full

## Key Metrics (Jan 26, 2026 - LIVE Instance)
| Metric | Value |
|--------|-------|
| **Position Value** | $959.59 |
| **AVAX Market Value** | $923.88 |
| **LTC Market Value** | $35.70 |
| **AVAX Unrealized** | -$66.79 (-6.7%) |
| **LTC Unrealized** | +$1.27 (+3.7%) |
| **Breakeven Price** | $12.72 (+7.2% from $11.86) |

## All Protections Active (Jan 27, 2026)

### Phase 1-5 (Deployed Jan 26 AM)
- ✓ Config aligned with PAPER thresholds (GRID_REDUCED @ 70%, DEFENSIVE @ 110%)
- ✓ Graduated loss response (CAUTION @ 2%, DEFENSIVE @ 3.5%)
- ✓ SmartGrid enforcement enabled
- ✓ Grid quality in /health endpoint
- ✓ Escalation-based drawdown alerts
- ✓ Circuit breaker age monitoring (alerts if stuck >24h)
- ✓ Zero fills detection (alerts if no trades >12h)

### Phase 6: Error Rate Monitoring (Deployed Jan 27 PM)
- ✓ Parses bot log every 60s for ERROR/CRITICAL entries
- ✓ Alerts if >= 10 errors in 1-hour rolling window
- ✓ 2-hour grace period prevents SMS spam
- ✓ State persists across restarts
- ✓ Shows error sources + samples in alert
- ✓ Auto-clears when error rate normalizes

### Feature A: 24-Hour Buy Throttles (Deployed Jan 26 PM)
| Symbol | 24h Fills | 24h Notional | Status |
|--------|-----------|--------------|--------|
| AVAX/USD | 0/6 | $0/$250 | ✓ OK |
| LTC/USD | 0/6 | $0/$250 | ✓ OK |

### Feature B: Recovery Trailing Profit Trim (Deployed PM)
- Trigger: +2.5% unrealized P/L when inventory ≥ 100%
- Trail gap: 0.5%
- Sell portion: 40%
- Cooldown: 90 minutes
- Status: Ready (waiting for trigger conditions)

## Configuration (Jan 26, 2026 - LIVE Instance)
| Setting | Value |
|---------|-------|
| **Symbols** | AVAX/USD (90%), LTC/USD (10%) |
| **Grid Spacing** | ~2.7% |
| **Levels** | 5-6 per symbol |
| **Orchestrator Thresholds** | GRID_REDUCED @ 70%, DEFENSIVE @ 110% |
| **Graduated Loss** | CAUTION @ 2%, DEFENSIVE @ 3.5%, HALT @ 5% |
| **SmartGrid** | ENABLED + ENFORCE |

## Scheduled Tasks

### LaunchAgents (launchd - LIVE instance only)
| Agent | Interval | Purpose |
|-------|----------|---------|
| `com.bluebird-live.bot` | RunAtLoad | Live bot (port 8001) |
| `com.bluebird-live.notifier` | RunAtLoad | Live notifier |
| `com.bluebird-live.watchdog-bot` | 5 min | Bot auto-restart |
| `com.bluebird-live.watchdog-notifier` | 5 min | Notifier auto-restart |

### Cron Jobs
| Schedule | Script | Purpose |
|----------|--------|---------|
| `0 3 * * *` | `backup_db.sh` | Database backup |
| `0 5 * * *` | `rotate_logs.sh` | Log rotation |

## Monitoring Commands
```bash
# Health check (now includes grid_quality)
curl http://localhost:8001/health | python3 -m json.tool

# Positions
curl http://localhost:8001/api/positions

# Risk overlay
curl http://localhost:8001/api/risk/overlay

# Orchestrator (check new thresholds)
curl http://localhost:8001/api/orchestrator/status

# Recent logs
tail -50 /tmp/bluebird-live-bot.log
```
