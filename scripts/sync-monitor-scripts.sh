#!/bin/bash
# BLUEBIRD Monitor Sync Script
#
# Installs a lightweight, restart-safe monitor snapshot job via launchd.
#
# Why:
# - The interactive monitor (Terminal UI) is nice, but not guaranteed to appear after reboot.
# - This installs a non-interactive snapshot job that writes a single-frame status to a log file,
#   so you can always `tail` it to see what's UP vs WAITING.
#
# Usage:
#   bash scripts/sync-monitor-scripts.sh
#
# Outputs:
# - ~/Library/Application Support/BLUEBIRD/run-monitor-status.sh
# - ~/Library/LaunchAgents/com.bluebird.monitor-status.plist
# - /tmp/bluebird-monitor-status.log

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

LOCAL_DIR="$HOME/Library/Application Support/BLUEBIRD"
LA_DIR="$HOME/Library/LaunchAgents"

RUN_SCRIPT="$LOCAL_DIR/run-monitor-status.sh"
PLIST="$LA_DIR/com.bluebird.monitor-status.plist"

mkdir -p "$LOCAL_DIR"
mkdir -p "$LA_DIR"

cat > "$RUN_SCRIPT" <<'EOF'
#!/bin/bash
set -euo pipefail

CONFIG_ENV="$HOME/Library/Application Support/BLUEBIRD/config.env"
if [ -f "$CONFIG_ENV" ]; then
  # shellcheck disable=SC1090
  source "$CONFIG_ENV"
fi

PROJECT_DIR="${BLUEBIRD_PROJECT_DIR:-$HOME/BLUEBIRD/bluebird}"
OUT="/tmp/bluebird-monitor-status.log"
TMP="/tmp/bluebird-monitor-status.log.tmp"

{
  echo ""
  echo "===== BLUEBIRD MONITOR SNAPSHOT $(date '+%Y-%m-%d %H:%M:%S') ====="
  bash "$PROJECT_DIR/scripts/monitor_services.sh" --once --no-clear
} >> "$OUT" 2>&1 || true

# Keep the log bounded
if [ -f "$OUT" ]; then
  tail -n 2000 "$OUT" > "$TMP" 2>/dev/null || true
  if [ -s "$TMP" ]; then
    mv "$TMP" "$OUT"
  else
    rm -f "$TMP" 2>/dev/null || true
  fi
fi
EOF

chmod 755 "$RUN_SCRIPT"

cat > "$PLIST" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>com.bluebird.monitor-status</string>

  <key>ProgramArguments</key>
  <array>
    <string>/bin/bash</string>
    <string>${RUN_SCRIPT}</string>
  </array>

  <key>RunAtLoad</key>
  <true/>

  <key>StartInterval</key>
  <integer>60</integer>

  <key>StandardOutPath</key>
  <string>/tmp/bluebird-monitor-status-launchd.log</string>

  <key>StandardErrorPath</key>
  <string>/tmp/bluebird-monitor-status-launchd.log</string>
</dict>
</plist>
EOF

# Reload agent (best-effort)
launchctl unload -w "$PLIST" 2>/dev/null || true
launchctl load -w "$PLIST" 2>/dev/null || true

echo "Installed monitor snapshot agent:"
echo "  Script: $RUN_SCRIPT"
echo "  Plist:  $PLIST"
echo "Tail snapshots with:"
echo "  tail -n 120 /tmp/bluebird-monitor-status.log"








