#!/bin/bash
# BLUEBIRD Watchdog Sync Script
#
# Copies watchdog scripts from the repo to the local filesystem and updates
# state paths. This is required because macOS launchd cannot execute scripts
# on external APFS volumes with 'noowners' flag (EPERM).
#
# Usage: bash scripts/sync-watchdog-scripts.sh
#
# This script is idempotent - safe to run multiple times.

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
LOCAL_DIR="$HOME/Library/Application Support/BLUEBIRD"
LOCAL_STATE_DIR="$LOCAL_DIR/state"

echo "=== BLUEBIRD Watchdog Sync ==="
echo "Source: $PROJECT_DIR/scripts/"
echo "Dest:   $LOCAL_DIR/"
echo ""

# Ensure local directories exist
mkdir -p "$LOCAL_DIR"
mkdir -p "$LOCAL_STATE_DIR"

# Function to sync a watchdog script
sync_script() {
    local src="$1"
    local dst="$2"
    local name="$3"

    if [ ! -f "$src" ]; then
        echo "ERROR: Source script not found: $src"
        return 1
    fi

    echo "Syncing $name..."

    # Copy and transform the script
    # Changes:
    # 1. Add LOCAL_STATE_DIR definition after STATE_DIR line
    # 2. Update CRASH_STATE, DISK_ALERT_STATE, PENDING_ALERTS to use LOCAL_STATE_DIR
    # Note: The repo scripts use STATE_DIR for external volume paths. We add LOCAL_STATE_DIR
    # for durable local state and redirect the state files there.
    sed -e '/^STATE_DIR=.*\/data\/state.*$/a\
# Durable state on LOCAL filesystem (launchd cannot write to external volumes)\
LOCAL_STATE_DIR="'"$LOCAL_STATE_DIR"'"' \
        -e 's|CRASH_STATE="${STATE_DIR}/|CRASH_STATE="${LOCAL_STATE_DIR}/|' \
        -e 's|DISK_ALERT_STATE="${STATE_DIR}/|DISK_ALERT_STATE="${LOCAL_STATE_DIR}/|' \
        -e 's|PENDING_ALERTS="${STATE_DIR}/|PENDING_ALERTS="${LOCAL_STATE_DIR}/|' \
        "$src" > "$dst"

    # Make executable
    chmod 755 "$dst"

    echo "  -> $dst ($(wc -l < "$dst") lines)"
}

# Sync bot watchdog
sync_script \
    "$PROJECT_DIR/scripts/check_bot.sh" \
    "$LOCAL_DIR/run-check-bot.sh" \
    "Bot watchdog"

# Sync notifier watchdog
sync_script \
    "$PROJECT_DIR/scripts/check_notifier.sh" \
    "$LOCAL_DIR/run-check-notifier.sh" \
    "Notifier watchdog"

echo ""
echo "=== Sync Complete ==="
echo ""
echo "Local scripts updated. LaunchAgents will use these on next run."
echo ""
echo "State directory: $LOCAL_STATE_DIR"
ls -la "$LOCAL_STATE_DIR" 2>/dev/null || echo "  (empty)"
echo ""
echo "To verify LaunchAgents are running:"
echo "  launchctl list | grep bluebird"
