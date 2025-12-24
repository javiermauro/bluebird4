#!/bin/bash
# BLUEBIRD Notifier Watchdog Script
#
# Checks the notifier heartbeat in the database and restarts if stale.
# Intended to be run via cron every 5 minutes:
#
#   */5 * * * * /Volumes/DOCK/BLUEBIRD\ 4.0/scripts/check_notifier.sh >> /tmp/bluebird-watchdog.log 2>&1
#

set -e

# Configuration
PROJECT_DIR="/Volumes/DOCK/BLUEBIRD 4.0"
DB_PATH="${PROJECT_DIR}/data/bluebird.db"
MAX_AGE_SECONDS=300  # 5 minutes
LOG_PREFIX="[$(date '+%Y-%m-%d %H:%M:%S')]"

# Check if database exists
if [ ! -f "$DB_PATH" ]; then
    echo "${LOG_PREFIX} ERROR: Database not found at $DB_PATH"
    exit 1
fi

# Query the heartbeat from database
HEARTBEAT=$(sqlite3 "$DB_PATH" "SELECT last_heartbeat FROM notifier_status WHERE id = 1" 2>/dev/null || echo "")

if [ -z "$HEARTBEAT" ]; then
    echo "${LOG_PREFIX} WARNING: No heartbeat record found in database"
    echo "${LOG_PREFIX} Notifier may have never started - attempting to start..."

    cd "$PROJECT_DIR"
    nohup caffeinate -i python3 -m src.notifications.notifier > /tmp/bluebird-notifier.log 2>&1 &
    NEW_PID=$!
    echo "${LOG_PREFIX} Started notifier with PID: $NEW_PID"
    exit 0
fi

# Parse the heartbeat timestamp and calculate age
# Format: 2025-12-23T10:23:18.123456
HEARTBEAT_EPOCH=$(python3 -c "from datetime import datetime; print(int(datetime.fromisoformat('${HEARTBEAT}').timestamp()))" 2>/dev/null || echo "0")
NOW_EPOCH=$(date +%s)
AGE=$((NOW_EPOCH - HEARTBEAT_EPOCH))

echo "${LOG_PREFIX} Heartbeat age: ${AGE}s (max: ${MAX_AGE_SECONDS}s)"

if [ "$AGE" -gt "$MAX_AGE_SECONDS" ]; then
    echo "${LOG_PREFIX} ALERT: Notifier heartbeat is STALE (${AGE}s > ${MAX_AGE_SECONDS}s)"

    # Get the PID from database
    OLD_PID=$(sqlite3 "$DB_PATH" "SELECT pid FROM notifier_status WHERE id = 1" 2>/dev/null || echo "")

    # Kill old process if still running
    if [ -n "$OLD_PID" ] && ps -p "$OLD_PID" > /dev/null 2>&1; then
        echo "${LOG_PREFIX} Killing stale notifier process (PID: $OLD_PID)"
        kill "$OLD_PID" 2>/dev/null || true
        sleep 2
    fi

    # Clean up stale files
    rm -f /tmp/bluebird/bluebird-notifier.pid 2>/dev/null || true
    rm -f /tmp/bluebird/bluebird-notifier.lock 2>/dev/null || true

    # Restart the notifier
    cd "$PROJECT_DIR"
    nohup caffeinate -i python3 -m src.notifications.notifier > /tmp/bluebird-notifier.log 2>&1 &
    NEW_PID=$!

    echo "${LOG_PREFIX} Restarted notifier with PID: $NEW_PID"

    # Optionally send alert via alternative method (uncomment if you have ntfy/pushover/etc)
    # curl -s -d "BLUEBIRD notifier restarted by watchdog" ntfy.sh/your-topic
else
    echo "${LOG_PREFIX} OK: Notifier is healthy"
fi
