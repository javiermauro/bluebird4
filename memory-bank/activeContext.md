# Active Context — Current Focus

## Reminders
- **[2026-01-29] AVAX REVIEW DATE** — Check if AVAX recovered. Exit trigger: below $11.00. Success: above $13.15.
- **[2026-01-22] DECISION: HOLD AVAX** — Analysis complete. Paper bot proves AVAX outperforms SOL 3.3x. Issue is timing, not coin. Do NOT switch.

## Now
- [2026-01-26 AM] **PHASE 1-5 IMPROVEMENTS DEPLOYED** - Major protection improvements aligned with paper bot thresholds. Bot restarted with new config.
- [2026-01-26 AM] **PERFORMANCE** - Equity $1,812.15 (-9.4% from $2K). AVAX unrealized -$81 (-8.2%). 5-day losing streak ended with +$4.74 today.
- [2026-01-26 AM] **AVAX EPISODE** - 238 hours (10 days) in inventory episode. Orchestrator in GRID_REDUCED mode.

## System Health (LIVE Instance) — Updated Jan 26, 2026 @ 8:54 AM EST
- **Bot**: Healthy, NORMAL mode, port 8001
- **Stream**: Healthy (88s since last bar)
- **Equity**: $1,812.15
- **Daily P/L**: +$3.71 (+0.21%)
- **Unrealized P/L**: **-$80** (AVAX -$81, LTC +$1)
- **AVAX Position**: 77.90 qty @ $12.72 avg (current $11.68)
- **Breakeven Target**: $12.72 (+8.9% needed)
- **Risk Overlay**: NORMAL (20+ hours)
- **Orchestrator**: AVAX grid_reduced (238h episode), LTC grid_full, DOGE grid_full

## Key Metrics (Jan 26, 2026 - LIVE Instance)
| Metric | Value |
|--------|-------|
| **Equity** | $1,812.15 |
| **Starting Capital** | $2,000.00 |
| **Total Return** | **-$187.85 (-9.4%)** |
| **Grid Profit** | +$812.15 (+81.2% since Jan 6) |
| **AVAX Position** | 77.90 qty @ $12.72 avg |
| **AVAX Unrealized** | -$80.99 (-8.2%) |
| **Breakeven Price** | $12.72 (+8.9% from $11.68) |

## New Protections Deployed (Jan 26, 2026)

### Phase 1: Config Alignment (LIVE with PAPER)
| Setting | Old | New |
|---------|-----|-----|
| GRID_REDUCED_ENTER_PCT | 85% | **70%** |
| DEFENSIVE_INVENTORY_PCT | 130% | **110%** |
| PRICE_DROP_LOOKBACK_MINUTES | 30 | **60** |
| PRICE_DROP_THRESHOLD_PCT | -8% | **-6%** |

### Phase 2: Graduated Loss Response
- **CAUTION** @ 2% daily loss → 50% size reduction
- **DEFENSIVE** @ 3.5% daily loss → block all buys
- **HALT** @ 5% daily loss → circuit breaker (existing)
- Hysteresis: recover when < 1.5% loss

### Phase 3: SmartGrid Enforcement
- `SMART_GRID_ENFORCE = True` - now executes drift-based rebalancing
- Safety gates still apply (overlay NORMAL, not DEFENSIVE)
- 60-min cooldown between rebalances

### Phase 4: Grid Quality Monitoring
- `/health` now includes `grid_quality` metrics per symbol
- Tracks: fills_last_hour, expected_fills_per_hour, fill_rate_pct, avg_fill_latency_ms
- Notifier alerts when fill rate < 50%

### Phase 5: Alert Improvements
- Drawdown alerts now escalation-based (fires on each 1% increase)
- sms_queue cleanup added to cleanup_db.py

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
