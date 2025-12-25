#!/bin/bash
# BLUEBIRD Bot Watchdog Script
#
# Checks the bot heartbeat in the database and restarts if stale.
# Intended to be run via cron every 5 minutes:
#
#   */5 * * * * /bin/bash "/Volumes/DOCK/BLUEBIRD 4.0/scripts/check_bot.sh" >> /tmp/bluebird-watchdog.log 2>&1
#

set -e

# Configuration
PROJECT_DIR="/Volumes/DOCK/BLUEBIRD 4.0"
DB_PATH="${PROJECT_DIR}/data/bluebird.db"
MAX_AGE_SECONDS=300  # 5 minutes
LOG_PREFIX="[$(date '+%Y-%m-%d %H:%M:%S')] [BOT]"

# Check if database exists
if [ ! -f "$DB_PATH" ]; then
    echo "${LOG_PREFIX} ERROR: Database not found at $DB_PATH"
    exit 1
fi

# Query the heartbeat from database
HEARTBEAT=$(sqlite3 "$DB_PATH" "SELECT last_heartbeat FROM bot_status WHERE id = 1" 2>/dev/null || echo "")

if [ -z "$HEARTBEAT" ]; then
    echo "${LOG_PREFIX} WARNING: No heartbeat record found in database"
    echo "${LOG_PREFIX} Bot may have never started with heartbeat enabled - attempting to start..."

    cd "$PROJECT_DIR"

    # Clean up any stale lock files first
    rm -f /tmp/bluebird/bluebird-bot.pid 2>/dev/null || true
    rm -f /tmp/bluebird/bluebird-bot.lock 2>/dev/null || true

    # Start the bot
    nohup caffeinate -i python3 -m src.api.server > /tmp/bluebird-bot.log 2>&1 &
    NEW_PID=$!
    echo "${LOG_PREFIX} Started bot with PID: $NEW_PID"
    exit 0
fi

# Parse the heartbeat timestamp and calculate age
# Format: 2025-12-25T11:30:00.123456
HEARTBEAT_EPOCH=$(python3 -c "from datetime import datetime; print(int(datetime.fromisoformat('${HEARTBEAT}').timestamp()))" 2>/dev/null || echo "0")
NOW_EPOCH=$(date +%s)
AGE=$((NOW_EPOCH - HEARTBEAT_EPOCH))

echo "${LOG_PREFIX} Heartbeat age: ${AGE}s (max: ${MAX_AGE_SECONDS}s)"

if [ "$AGE" -gt "$MAX_AGE_SECONDS" ]; then
    echo "${LOG_PREFIX} ALERT: Bot heartbeat is STALE (${AGE}s > ${MAX_AGE_SECONDS}s)"

    # Get the PID from database
    OLD_PID=$(sqlite3 "$DB_PATH" "SELECT pid FROM bot_status WHERE id = 1" 2>/dev/null || echo "")

    # Kill old process if still running
    if [ -n "$OLD_PID" ] && ps -p "$OLD_PID" > /dev/null 2>&1; then
        echo "${LOG_PREFIX} Killing stale bot process (PID: $OLD_PID)"
        kill "$OLD_PID" 2>/dev/null || true
        sleep 3

        # Force kill if still running
        if ps -p "$OLD_PID" > /dev/null 2>&1; then
            echo "${LOG_PREFIX} Force killing bot process (PID: $OLD_PID)"
            kill -9 "$OLD_PID" 2>/dev/null || true
            sleep 2
        fi
    fi

    # Clean up stale lock files
    rm -f /tmp/bluebird/bluebird-bot.pid 2>/dev/null || true
    rm -f /tmp/bluebird/bluebird-bot.lock 2>/dev/null || true

    # Also kill any process holding port 8000
    PORT_PID=$(lsof -ti :8000 2>/dev/null || echo "")
    if [ -n "$PORT_PID" ]; then
        echo "${LOG_PREFIX} Killing process on port 8000 (PID: $PORT_PID)"
        kill -9 "$PORT_PID" 2>/dev/null || true
        sleep 2
    fi

    # Restart the bot
    cd "$PROJECT_DIR"
    nohup caffeinate -i python3 -m src.api.server > /tmp/bluebird-bot.log 2>&1 &
    NEW_PID=$!

    echo "${LOG_PREFIX} Restarted bot with PID: $NEW_PID"

    # Optionally send alert via SMS notifier or other method
    # The notifier should detect the restart and send an alert
else
    echo "${LOG_PREFIX} OK: Bot is healthy"
fi
