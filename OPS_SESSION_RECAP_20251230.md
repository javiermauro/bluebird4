# BLUEBIRD Mac mini unattended boot — ops recap (2025-12-30)

## Goal
Make **internal** the only source of truth and stop the “two copies + launchd half-pointing” failure mode by doing **one final** `DOCK → internal` sync, then running **everything from** `~/BLUEBIRD/bluebird/` going forward.

## Source of Truth (from now on)
- **Internal repo (ONLY)**: `/Users/javierrodriguez/BLUEBIRD/bluebird`
- **External DOCK copy (backup ONLY)**: `/Volumes/DOCK/BLUEBIRD 4.0/`

## What we did (exact sequence)

### 0) Confirmed current running state (before changes)
- Found an existing bot process listening on **:8000**.
- Verified the bot’s **current working directory** was already internal:
  - `cwd = /Users/javierrodriguez/BLUEBIRD/bluebird`
- Verified watchdog config already pointed internal:
  - `~/Library/Application Support/BLUEBIRD/config.env`

### 1) One final sync from DOCK → internal
Ran:

```bash
rsync -a "/Volumes/DOCK/BLUEBIRD 4.0/" "$HOME/BLUEBIRD/bluebird/"
cd "$HOME/BLUEBIRD/bluebird" || exit 1
```

### 2) Rebuild local watchdog scripts/config so launchd points at internal
Ran:

```bash
cd "$HOME/BLUEBIRD/bluebird" || exit 1
bash scripts/sync-watchdog-scripts.sh
cat "$HOME/Library/Application Support/BLUEBIRD/config.env"
```

Verified output contained:

```bash
export BLUEBIRD_PROJECT_DIR="/Users/javierrodriguez/BLUEBIRD/bluebird"
```

### 3) Hard reset bot state (clean restart)
To avoid watchdog races, we **temporarily unloaded** the bot watchdog LaunchAgent, then reset crash-loop + lock/pid files and cleared port 8000:

```bash
launchctl unload -w "$HOME/Library/LaunchAgents/com.bluebird.watchdog-bot.plist" 2>/dev/null || true
rm -f "$HOME/Library/Application Support/BLUEBIRD/state/crash-loop-bot.json"
rm -f /tmp/bluebird/bluebird-bot.pid /tmp/bluebird/bluebird-bot.lock 2>/dev/null || true
kill -9 $(lsof -ti :8000 2>/dev/null) 2>/dev/null || true
```

### 4) Start bot manually from internal (deterministic)
Ran:

```bash
cd "$HOME/BLUEBIRD/bluebird" || exit 1
nohup caffeinate -i /usr/bin/python3 -m src.api.server > /tmp/bluebird-bot.log 2>&1 &
sleep 2
curl -sS --max-time 5 http://127.0.0.1:8000/health || true
tail -n 80 /tmp/bluebird-bot.log
```

Result:
- `/health` returned `status=healthy`
- Uvicorn came up on `0.0.0.0:8000`
- Bot reconciled successfully with Alpaca history and DB

### 5) Re-enable watchdog + run “wait until ready” monitor
Re-enabled watchdog-bot LaunchAgent:

```bash
launchctl load -w "$HOME/Library/LaunchAgents/com.bluebird.watchdog-bot.plist" 2>/dev/null || true
```

Started monitor in no-exit mode (logged to file):

```bash
nohup bash "$HOME/BLUEBIRD/bluebird/scripts/monitor_services.sh" --no-exit > /tmp/bluebird-monitor.log 2>&1 &
tail -n 120 /tmp/bluebird-monitor.log
```

Monitor showed:
- `Bot /health: YES`
- `Dashboard HTTP: YES`
- `READY: YES (bot + dashboard + notifier loaded)`

## Where to look (quick references)
- **Bot log**: `/tmp/bluebird-bot.log`
- **Monitor log**: `/tmp/bluebird-monitor.log`
- **Watchdog log**: `/tmp/bluebird-watchdog.log`
- **Notifier log**: `/tmp/bluebird-notifier.log`
- **Dashboard log**: `/tmp/bluebird-dashboard.log`

## Launchd / watchdog paths (local-only, by design)
- Runtime config: `~/Library/Application Support/BLUEBIRD/config.env`
- Local watchdog scripts:
  - `~/Library/Application Support/BLUEBIRD/run-check-bot.sh`
  - `~/Library/Application Support/BLUEBIRD/run-check-notifier.sh`
- LaunchAgents plists:
  - `~/Library/LaunchAgents/com.bluebird.watchdog-bot.plist`
  - `~/Library/LaunchAgents/com.bluebird.watchdog-notifier.plist`

## Operational rule (important)
- **Never edit** `/Volumes/DOCK/BLUEBIRD 4.0/` again — treat it as **backup only**.
- Do all work in Cursor from:
  - `~/BLUEBIRD/bluebird/`

## Verification checklist (minimal)
Run:

```bash
curl -sS --max-time 5 http://127.0.0.1:8000/health
launchctl list | grep -i bluebird
tail -n 80 /tmp/bluebird-monitor.log
```

Expected:
- `/health` returns JSON with `"status":"healthy"`
- launchctl shows watchdog + notifier/dashboard jobs
- monitor shows `Bot /health: YES` and `READY: YES`


