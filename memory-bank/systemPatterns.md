# System Patterns — Architecture & Design

## System Overview

BlueBird is a **defensive grid trading system** for cryptocurrency. It's not just a simple grid bot — it's a multi-layered system designed to profit from sideways volatility while protecting capital during crashes.

### Why Grid Trading for Crypto?
- **High volatility**: Crypto moves 5-10% regularly = frequent grid fills
- **24/7 markets**: No gaps, continuous trading opportunities
- **No PDT rules**: Unlimited trades regardless of account size
- **Sideways markets**: Grid trading thrives when prices oscillate

---

## Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    Alpaca WebSocket                          │
│                   (1-min price bars)                         │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│              Layer 1: Risk Overlay Check                     │
│            Is market crashing? (2-of-3 triggers)            │
│                 NORMAL / RISK_OFF / RECOVERY                 │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│              Layer 2: Regime Detection                       │
│              ADX trend strength + direction                  │
│           Strong downtrend? Block/reduce buys               │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│              Layer 3: Orchestrator                           │
│              Check inventory % per symbol                    │
│            GRID_FULL / GRID_REDUCED / DEFENSIVE             │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│              Layer 4: Windfall Check                         │
│           Big unrealized profit? Take it!                   │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│              Layer 5: Grid Logic                             │
│         Price hit buy level? → Execute buy                  │
│         Price hit sell level? → Execute sell                │
│              (Using limit orders for maker fee)             │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                   Persistence                                │
│         SQLite DB (trades, equity, orders)                  │
│         State files (grid, overlay, orchestrator)           │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                  Notifications                               │
│            Dashboard (WebSocket real-time)                  │
│               SMS alerts (Twilio)                           │
└─────────────────────────────────────────────────────────────┘
```

---

## Services

| Service | Entry Point | Port | Description |
|---------|-------------|------|-------------|
| Grid Bot API | `src/api/server.py` | 8000 | FastAPI server, WebSocket for dashboard |
| Dashboard | `dashboard/` | 5173 | Vite/React real-time UI |
| Notifier | `src/notifications/notifier.py` | — | Polls API, sends SMS via Twilio |

---

## Core Components

### 1. Grid Trading Engine
**File:** `src/strategy/grid_trading.py`

```
Price
  │
  ├── $130 ── SELL level 4
  ├── $128 ── SELL level 3
  ├── $126 ── SELL level 2
  ├── $124 ── SELL level 1
  │           ─── Current Price ───
  ├── $122 ── BUY level 1
  ├── $120 ── BUY level 2
  ├── $118 ── BUY level 3
  └── $116 ── BUY level 4
```

- Places buy orders below price, sell orders above
- When price drops → buys triggered
- When price rises → sells triggered
- Profit = grid spacing minus fees (~1.5% spacing, ~0.6% fees = ~0.9% profit per cycle)

**Grid Rebalancing:**
- If price moves 3%+ outside grid range → rebuild grid centered on new price
- Preserves profit stats during rebalance

---

### 2. Risk Overlay (Crash Protection)
**File:** `src/strategy/risk_overlay.py`

State machine that stops buying during market crashes.

**States:**
| Mode | Buys | Sells | Position Size |
|------|------|-------|---------------|
| NORMAL | Allowed | Allowed | 100% |
| RISK_OFF | BLOCKED | Allowed | 0% |
| RECOVERY | Reduced | Allowed | 25%→50%→75%→100% |

**RISK_OFF Triggers (2-of-3 required):**
1. **Momentum shock**: Returns < -1.5%
2. **ADX downtrend**: ADX > 35 AND direction = DOWN
3. **Correlation spike**: Cross-symbol correlation > 0.90

**RECOVERY Ramp:**
- After 20 min stability in RISK_OFF → enter RECOVERY
- Stage 0: 25% size (10 bars to advance)
- Stage 1: 50% size (10 bars to advance)
- Stage 2: 75% size (10 bars to advance)
- Stage 3: 100% size → return to NORMAL

---

### 3. Orchestrator (Inventory Management)
**File:** `src/strategy/orchestrator.py`

Meta-controller that prevents over-accumulation in one direction.

**Modes based on inventory %:**
| Inventory | Mode | Action |
|-----------|------|--------|
| < 80% | GRID_FULL | Normal trading (100% size) |
| 80-150% | GRID_REDUCED | Reduced trading (50% size) |
| > 150% | DEFENSIVE | Buys blocked |

**Episode Tracking:**
- Episode starts when inventory ≥ 30%
- Episode resets when inventory ≤ 10%
- Age tracked for liquidation decisions

**Staged Liquidation (only in NORMAL overlay mode):**
| Trigger | Conditions | Action |
|---------|------------|--------|
| TP Trim | Episode ≥24h, P/L ≥+0.3%, inventory ≥120% | Reduce to 100% |
| Loss Cut | Episode ≥48h, P/L ≤-2%, inventory ≥130% | Reduce 25% of excess |
| Max Age | Episode ≥72h, inventory ≥120% | Reduce to 100% |

**Critical:** Orchestrator never overrides Risk Overlay decisions.

---

### 4. Windfall Profit Taking
**Config:** `WINDFALL_PROFIT_CONFIG` in `config_ultra.py`

Captures large unrealized gains before they evaporate:

```python
IF (unrealized P&L > 4% AND RSI > 70) OR (unrealized > 6%):
    SELL 70% of position
    LOG to windfall-log.json
```

- Soft threshold: 4% + overbought (RSI > 70)
- Hard threshold: 6% regardless of RSI
- Cooldown: 30 min between windfall sells per symbol

---

### 5. Limit Order Optimization
**Config:** `GRID_USE_LIMIT_ORDERS = True`

Instead of market orders (0.25% taker fee):
- Uses limit orders (0.15% maker fee)
- Saves ~0.10% per trade
- "Resting limit mode" - places orders and waits for fills

**Maker Buffer:** 5 basis points (0.05%) to ensure limit orders don't cross spread.

---

### 6. Fast Fill Detection
**Config:** `ENABLE_FAST_FILL_CHECK = True`

- Checks order status every 10 seconds
- Detects fills in ~10-15s vs 5-minute full reconciliation
- Faster grid level updates for continuous trading

---

### 7. Downtrend Protection Layers

**A. Regime Detection (ADX-based):**
| ADX | Direction | Action |
|-----|-----------|--------|
| < 25 | Any | Full trading |
| 25-35 | DOWN | 50% size (developing downtrend) |
| > 35 | DOWN | Buys blocked (strong downtrend) |

**B. Consecutive Down Bars:**
```
3+ red candles in a row → Block buys
```
Simple crash guard for rapid selloffs. Resets on first green candle.

---

### 8. Time Filters
**Config:** `USE_TIME_FILTER = True`

Optimal trading windows (higher volatility = more grid fills):
- **US Session**: 9AM-1PM ET (13:00-17:00 UTC)
- **Asian Session**: 8PM-11PM ET (01:00-04:00 UTC)
- **London**: 3AM-5AM ET (07:00-09:00 UTC)

**Avoid hours:** 10PM-12AM, 5AM-7AM (low volatility)
**Weekend:** 50% size reduction

---

### 9. Per-Symbol Stop Loss
**Config:** `GRID_STOP_LOSS_PCT = 0.10`

If price drops 10% below grid lower bound → stop loss triggers.

---

## Safety Gating Order

When deciding whether to place a buy order, gates are checked in this order:

1. **Risk Overlay** (highest priority): Blocks buys entirely in RISK_OFF
2. **Orchestrator**: Blocks buys in DEFENSIVE, reduces size in GRID_REDUCED
3. **Regime Detection**: 50% size when ADX 25-35 + DOWN
4. **Consecutive Down Bars**: Blocks after 3+ red candles
5. **Time Filter**: Avoids low-volatility hours
6. **Windfall Check**: Takes profit on big unrealized gains
7. **Grid Logic**: Finally, check if price hit a buy level

Sells are always allowed (protection layers don't block sells).

---

## State & Persistence

### Database (durable)
- **Path:** `data/bluebird.db` (SQLite)
- **Tables:** trades, equity_snapshots, orders, daily_summary, grid_snapshots, sms_history, notifier_status

### State Files (in `data/state/`, survives reboot)
| File | Purpose |
|------|---------|
| `grid-state.json` | Grid levels, fills, pending orders |
| `risk-overlay.json` | RISK_OFF/RECOVERY mode and triggers |
| `orchestrator.json` | Inventory episode tracking |
| `daily-equity.json` | Daily P&L tracking |
| `alltime-equity.json` | All-time performance stats |
| `circuit-breaker.json` | Max drawdown/stop-loss flags |
| `windfall-log.json` | Windfall profit captures |

### Ephemeral (cleared on reboot, intentional)
- `/tmp/bluebird/*.lock` - Process locks
- `/tmp/bluebird/*.pid` - PID files

---

## Dashboard Panels

| Panel | Color Scheme | Shows |
|-------|--------------|-------|
| Risk Overlay | Red/Gold | NORMAL/RISK_OFF/RECOVERY, triggers, telemetry |
| Orchestrator | Teal/Cyan | Inventory %, episode tracking, mode per symbol |
| Grid Status | Default | Levels, fills, pending orders, profit |

---

## API Endpoints (Key)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Service health, stream status |
| `/api/risk/status` | GET | Equity, daily P/L, drawdown |
| `/api/risk/overlay` | GET | Risk overlay mode, triggers, telemetry |
| `/api/risk/overlay` | POST | Manual override (RISK_OFF, NORMAL, clear) |
| `/api/orchestrator/status` | GET | Orchestrator mode, inventory, episodes |
| `/api/grid/status` | GET | Grid levels, fill status, profits |
| `/api/positions` | GET | Current positions from Alpaca |
| `/api/db/reconcile` | GET | Sync database with Alpaca fills |

---

## Operational Patterns

### Reconcile on Recovery
After restarts/outages:
- Use `/api/db/reconcile` to align local DB with Alpaca fills
- Strategy is resilient because limit orders live on the exchange

### Day Boundary Pattern
- Operator-facing rollups follow Mac mini local time
- Daily P&L matches how operator thinks about trading days

---

## Grid Trading Metrics

- **Cycle win rate**: 100% by design (spacing > fees = guaranteed profit)
- **Fee efficiency**: Target 40x+ profit/fee ratio
- **Spacing requirements**: Must exceed ~0.60% round-trip fees to be profitable
- **Typical spacing**: 1.3-1.7% per grid level

---

## Maintenance Infrastructure

### LaunchAgents (Auto-Start After Reboot)
| LaunchAgent | Purpose | KeepAlive |
|-------------|---------|-----------|
| `com.bluebird.bot` | Grid trading bot | Yes |
| `com.bluebird.notifier` | SMS notifications | Yes |
| `com.bluebird.dashboard` | Static dashboard server | No |
| `com.bluebird.watchdog-bot` | Backup bot health monitor (60s) | No |
| `com.bluebird.watchdog-notifier` | Backup notifier health monitor (60s) | No |

### Watchdog Pattern
- Run every 60 seconds via launchd
- Primary signal: `/health` HTTP endpoint
- Secondary signal: DB heartbeat (written every 60s)
- Crash loop protection: Pauses restarts after 3 crashes in 30 min
- Scripts in `~/Library/Application Support/BLUEBIRD/`

### Cron Jobs (Maintenance)
| Schedule | Script | Purpose |
|----------|--------|---------|
| `0 3 * * *` | `backup_db.sh` | Daily DB backup |
| `0 5 * * *` | `rotate_logs.sh` | Daily log rotation |

### Maintenance Scripts (in `scripts/`)
- `sync-watchdog-scripts.sh` - Sync repo scripts to local LaunchAgent paths
- `check_bot.sh` / `check_notifier.sh` - Watchdog scripts
- `backup_db.sh` - SQLite backup, 7-day retention
- `rotate_logs.sh` - Log rotation, 50 MB limit
- `cleanup_db.py` - Manual DB cleanup (90-day retention)
- `monitor_services.sh` - Real-time status display

---

## Component Summary Table

| Layer | Component | Purpose |
|-------|-----------|---------|
| 1 | Grid Trading | Core profit engine - buy low, sell high |
| 2 | Risk Overlay | Crash protection - stop buying in panics |
| 3 | Orchestrator | Inventory control - don't overload positions |
| 4 | Windfall | Profit capture - lock in big gains |
| 5 | Regime Detection | Trend awareness - reduce in downtrends |
| 6 | Limit Orders | Fee optimization - save 0.10% per trade |
| 7 | Fast Fill | Speed - detect fills quickly |
| 8 | Time Filters | Timing - trade in volatile hours |
| 9 | Stop Loss | Protection - cut losses at 10% |

---

## Deprecated Patterns
- Prediction-based bots in `archive/old_bots/` are legacy (21% win rate)
- Current system is grid-first with protection layers
- No ML predictions used for trade decisions
