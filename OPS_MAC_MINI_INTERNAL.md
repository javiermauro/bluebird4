# Mac mini — Move BLUEBIRD to Internal Disk (Unattended-Safe)

This runbook moves BLUEBIRD off an external volume (e.g. `/Volumes/DOCK/...`) onto the Mac mini internal disk so `launchd` watchdogs can reliably read/write state and restart services after reboots.

## Why this is required
- `launchd` jobs can intermittently fail to access external volumes (observed: `sqlite3 ... authorization denied`), which can cause restart flapping and failed auto-recovery.
- Internal disk paths under `$HOME` are the most reliable for unattended operation.

## Target layout (recommended)
- **Runtime repo (internal)**: `~/BLUEBIRD/bluebird/`
- **Watchdog scripts/config (local)**: `~/Library/Application Support/BLUEBIRD/`
- **LaunchAgents**: `~/Library/LaunchAgents/com.bluebird.watchdog-*.plist`

## Step-by-step (run on the Mac mini)

### 0) (Optional) Realtime “wait until ready” monitor
This opens a terminal-style live status view and **exits automatically** once bot+dashboard+notifier are up:

```bash
bash "$HOME/BLUEBIRD/bluebird/scripts/monitor_services.sh"
```

### 1) Copy repo from external → internal

```bash
mkdir -p "$HOME/BLUEBIRD"
rsync -a "/Volumes/DOCK/BLUEBIRD 4.0/" "$HOME/BLUEBIRD/bluebird/"
cd "$HOME/BLUEBIRD/bluebird" || exit 1
```

### 2) Sync watchdog scripts + write config.env
This ensures LaunchAgents execute local copies and use the correct project path:

```bash
bash "scripts/sync-watchdog-scripts.sh"
```

This generates:
- `~/Library/Application Support/BLUEBIRD/run-check-bot.sh`
- `~/Library/Application Support/BLUEBIRD/run-check-notifier.sh`
- `~/Library/Application Support/BLUEBIRD/config.env` (project path; optional DB override)

### 3) Clear crash-loop pause files (if present)

```bash
rm -f "$HOME/Library/Application Support/BLUEBIRD/state/crash-loop-bot.json"
rm -f "$HOME/Library/Application Support/BLUEBIRD/state/crash-loop-notifier.json"
```

### 4) Kick watchdogs now (don’t wait the 5-minute interval)

```bash
UID=$(id -u)
launchctl kickstart -k "gui/${UID}/com.bluebird.watchdog-bot" || true
launchctl kickstart -k "gui/${UID}/com.bluebird.watchdog-notifier" || true
```

### 5) Verify services are actually up

```bash
sleep 3
lsof -nP -iTCP:8000 -sTCP:LISTEN || true
curl -sS --max-time 2 http://127.0.0.1:8000/health || true
tail -n 80 /tmp/bluebird-watchdog.log
tail -n 80 /tmp/bluebird-bot.log
tail -n 80 /tmp/bluebird-notifier.log
```

## Keeping the external drive as a backup (optional)
After you’re stable on internal, you can mirror internal → external occasionally:

```bash
rsync -a --delete "$HOME/BLUEBIRD/bluebird/" "/Volumes/DOCK/BLUEBIRD 4.0/"
```

## Unattended power-loss recovery requirements
### A) FileVault must be OFF
If FileVault is ON, the Mac will boot to the unlock screen and nothing starts unattended.

```bash
fdesetup status
sudo fdesetup disable
```

Wait until `fdesetup status` reports **Off**.

### B) Auto restart after power failure must be ON

```bash
sudo systemsetup -setrestartpowerfailure on
```

### C) Auto-login must be enabled (UI)
System Settings → Users & Groups → **Automatic login** → choose your runtime user.

## Real outage test procedure
1) Ensure the bot is healthy:
```bash
curl -sS --max-time 2 http://127.0.0.1:8000/health
```
2) Physically cut power (unplug or turn UPS output off) for 10–15 seconds, restore power.
3) After the Mac boots and auto-logs-in, verify:
```bash
launchctl list | grep bluebird
curl -sS --max-time 2 http://127.0.0.1:8000/health
tail -n 120 /tmp/bluebird-watchdog.log
```


