# Active Context — Current Focus

## Reminders
- **[2026-01-29] AVAX REVIEW DATE** — Check if AVAX recovered. Exit trigger: below $11.00. Success: above $13.15.
- **[2026-01-22] ✅ DECISION: HOLD AVAX** — Analysis complete. Paper bot proves AVAX outperforms SOL 3.3x. Issue is timing, not coin. Do NOT switch.

## Now
- [2026-01-22 PM] **✅ HOLD AVAX DECISION** - After analysis (web + paper bot data), decided to HOLD. Paper bot: AVAX $4,428 vs SOL $1,336 in Jan (3.3x better). Review Jan 29.
- [2026-01-22 PM] **PERFORMANCE** - Equity $1,848.93 (-7.6% from $2K). AVAX unrealized -$76 (-7.7%). Grid profit +$44. Needs +8.3% to breakeven.
- [2026-01-21 PM] **COMPUTER RESTARTED - WATCHDOG RECOVERED BOT** - System restarted, launchd watchdog automatically recovered the live bot.
- [2026-01-20] **PROTECTION IMPROVEMENTS DEPLOYED TO LIVE** - All 9 phases deployed with conservative settings.

## System Health (LIVE Instance) — Updated Jan 22, 2026 @ 7:26 PM EST
- **Bot**: Healthy, NORMAL mode, port 8001
- **Stream**: Degraded (100s since last bar)
- **Equity**: $1,848.93
- **Daily P/L**: -$36.11 (-1.92%)
- **Unrealized P/L**: **-$76** (AVAX -$76, LTC +$1)
- **AVAX Position**: 75.30 qty @ $13.15 avg (current $12.14)
- **Breakeven Target**: $13.15 (+8.3% needed)
- **Risk Overlay**: NORMAL (7.6 hours)
- **Orchestrator**: AVAX grid_full (152h episode), LTC grid_full (50h episode)

## Key Metrics (Jan 22, 2026 - LIVE Instance)
| Metric | Value |
|--------|-------|
| **Equity** | $1,848.93 |
| **Starting Capital** | $2,000.00 |
| **Total Return** | **-$151.07 (-7.6%)** |
| **AVAX Position** | 75.30 qty @ $13.15 avg |
| **AVAX Unrealized** | -$76.38 (-7.7%) |
| **Breakeven Price** | $13.15 (+8.3% from $12.14) |
| **Grid Profit** | +$44.04 (2 completed trades) |

## ✅ AVAX Decision: HOLD (Jan 22, 2026)

**Decision**: HOLD AVAX position. Do NOT switch to SOL.

**Key Finding**: Paper bot data proves AVAX is the BEST performer for grid trading:
| Symbol | Jan 2026 Profit |
|--------|-----------------|
| **AVAX** | **$4,428** (182 sells) |
| SOL | $1,336 (107 sells) |

AVAX outperforms SOL by **3.3x** on paper bot in same period!

**Why We're Down**: Bad timing, not bad coin choice.
- Live bot started Jan 6, right before Jan 18 crash (-8%)
- Small capital ($2K) vs paper ($100K+) = less buffer
- Same coin, same strategy = paper bot profiting, we're temporarily underwater

**Recovery Plan**:
- AVAX needs +8.3% to breakeven ($12.14 → $13.15)
- Grid profit: +$44 ✅ (strategy IS working)
- Monitor until Jan 29 - if AVAX breaks below $11.00, reassess
- Switching now would lock in -$76 loss permanently

## Configuration (Jan 14, 2026 - LIVE Instance)
| Setting | Value |
|---------|-------|
| **Symbols** | AVAX/USD (90%), LTC/USD (10%) |
| **Grid Spacing** | ~2.7% |
| **Levels** | 5-6 per symbol |
| **Stream Watchdog** | 90s (was 180s) |

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
# Health check
curl http://localhost:8001/health | python3 -m json.tool

# Positions
curl http://localhost:8001/api/positions

# Risk overlay
curl http://localhost:8001/api/risk/overlay

# Recent logs
tail -50 /tmp/bluebird-live-bot.log
```
