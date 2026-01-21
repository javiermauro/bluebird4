# Active Context — Current Focus

## Reminders
- **[2026-01-21] WEEK 2 REVIEW** — Continue monitoring 2.7% grid spacing performance. Compare daily averages, fee ratios, and total equity growth.

## Now
- [2026-01-21 PM] **COMPUTER RESTARTED - WATCHDOG RECOVERED BOT** - System restarted, launchd watchdog automatically recovered the live bot. Bot running on port 8001, stream initializing.
- [2026-01-21 PM] **AVAX POSITION STATUS** - Underwater at -9.6% (-$133), needs +10.6% to hit $13.26 breakeven. Position reduced from 119 to 104 qty (some sells filled). Grid strategy working as designed - waiting for bounce.
- [2026-01-21 AM] **CODE QUALITY IMPROVEMENTS DEPLOYED** - SQLite hardening (WAL mode, 30s timeout), exception handling fixes, stream watchdog lowered to 90s.
- [2026-01-20 PM] **AVAX BREAKEVEN ORDER PLACED** - Manual limit sell for 35 AVAX @ $13.60 (Order ID: 4d3a11de-...).
- [2026-01-20] **PROTECTION IMPROVEMENTS DEPLOYED TO LIVE** - All 9 phases of protection improvements deployed with conservative settings.

## System Health (LIVE Instance) — Updated Jan 21, 2026 @ 2:10 PM EST
- **Bot**: Healthy, NORMAL mode, port 8001
- **Stream**: Initializing (post-restart, watchdog recovering)
- **Equity**: $1,827.48
- **Daily P/L**: -$37.34 (-2.0%)
- **Unrealized P/L**: **-$133** (AVAX -$133, LTC -$1)
- **AVAX Position**: 104.47 qty @ $13.26 avg (current $11.99)
- **Breakeven Target**: $13.26 (+10.6% needed)
- **Risk Overlay**: NORMAL mode
- **Orchestrator**: AVAX in grid_reduced (123h episode), LTC in grid_full

## Key Metrics (Jan 21, 2026 - LIVE Instance)
| Metric | Value |
|--------|-------|
| **Equity** | $1,827.48 |
| **Starting Capital** | $2,000.00 |
| **Total Return** | **-$172.52 (-8.6%)** |
| **AVAX Position** | 104.47 qty @ $13.26 avg |
| **AVAX Unrealized** | -$132.69 (-9.6%) |
| **Breakeven Price** | $13.26 (+10.6% from $11.99) |
| **Grid Profit** | +$35.03 (2 completed trades) |

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
