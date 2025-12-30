#!/bin/bash
# Move BLUEBIRD to internal disk and resync launchd watchdog scripts/config.
#
# Safe defaults:
# - Does NOT touch FileVault or auto-login (those are manual UI/security choices).
# - Does NOT write secrets.
#
# Usage:
#   bash scripts/install_macmini_internal.sh \
#     "/Volumes/DOCK/BLUEBIRD 4.0" \
#     "$HOME/BLUEBIRD/bluebird"

set -euo pipefail

SRC_DIR="${1:-}"
DST_DIR="${2:-}"

if [ -z "$SRC_DIR" ] || [ -z "$DST_DIR" ]; then
  echo "Usage: $0 <source_repo_dir> <dest_repo_dir>"
  echo "Example:"
  echo "  bash scripts/install_macmini_internal.sh \"/Volumes/DOCK/BLUEBIRD 4.0\" \"\$HOME/BLUEBIRD/bluebird\""
  exit 2
fi

if [ ! -d "$SRC_DIR" ]; then
  echo "ERROR: source dir not found: $SRC_DIR"
  exit 1
fi

mkdir -p "$(dirname "$DST_DIR")"

echo "=== BLUEBIRD internal install ==="
echo "Source: $SRC_DIR"
echo "Dest:   $DST_DIR"
echo ""

echo "1) Syncing repo to internal disk (rsync)..."
rsync -a "$SRC_DIR/" "$DST_DIR/"

echo ""
echo "2) Syncing watchdog scripts (writes config.env)..."
cd "$DST_DIR"
bash "scripts/sync-watchdog-scripts.sh"

echo ""
echo "3) Clearing crash-loop pause files (if any)..."
rm -f "$HOME/Library/Application Support/BLUEBIRD/state/crash-loop-bot.json"
rm -f "$HOME/Library/Application Support/BLUEBIRD/state/crash-loop-notifier.json"

echo ""
echo "4) Kickstarting watchdogs..."
USER_ID="$(id -u)"
launchctl kickstart -k "gui/${USER_ID}/com.bluebird.watchdog-bot" || true
launchctl kickstart -k "gui/${USER_ID}/com.bluebird.watchdog-notifier" || true

echo ""
echo "5) Verifying (best effort)..."
sleep 3
echo "- launchctl:"
launchctl list | grep bluebird || true
echo "- port 8000:"
lsof -nP -iTCP:8000 -sTCP:LISTEN || true
echo "- /health:"
curl -sS --max-time 2 http://127.0.0.1:8000/health || true
echo ""
echo "If /health is not healthy yet, check logs:"
echo "  tail -n 120 /tmp/bluebird-watchdog.log"
echo "  tail -n 120 /tmp/bluebird-bot.log"
echo "  tail -n 120 /tmp/bluebird-notifier.log"
echo ""
echo "Next manual steps for unattended recovery:"
echo "  fdesetup status"
echo "  sudo systemsetup -setrestartpowerfailure on"
echo "  (disable FileVault + enable auto-login in System Settings)"


