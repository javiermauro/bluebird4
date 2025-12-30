#!/bin/bash
# BLUEBIRD Notifier Watchdog Script
#
# Checks the notifier heartbeat in the database and restarts if stale.
# Includes crash loop detection and disk space monitoring.
#
# Intended to be run via cron every 5 minutes:
#   */5 * * * * /bin/bash "/Volumes/DOCK/BLUEBIRD 4.0/scripts/check_notifier.sh" >> /tmp/bluebird-watchdog.log 2>&1
#
# CRASH LOOP PROTECTION:
# - Tracks restart history in data/state/crash-loop-notifier.json
# - Pauses restarts after 3 restarts in 30 minutes
# - Manual clear: rm "/Volumes/DOCK/BLUEBIRD 4.0/data/state/crash-loop-notifier.json"
#
# DISK SPACE MONITORING:
# - Alerts when disk usage >= 90%
# - Once per day maximum (tracked in data/state/disk-alert.json)

set -e

# ----------------------------------------------------------------------------
# Path configuration (supports unattended launchd + relocations)
#
# Priority:
# 1) Source per-machine config: ~/Library/Application Support/BLUEBIRD/config.env
# 2) Env overrides: BLUEBIRD_PROJECT_DIR / BLUEBIRD_DB_PATH
# 3) Repo-relative default (when running from repo): <repo_root>
# 4) Last-resort hardcoded default (legacy)
# ----------------------------------------------------------------------------

CONFIG_ENV="$HOME/Library/Application Support/BLUEBIRD/config.env"
if [ -f "$CONFIG_ENV" ]; then
    # shellcheck disable=SC1090
    source "$CONFIG_ENV"
fi

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_GUESS="$(cd "${SCRIPT_DIR}/.." >/dev/null 2>&1 && pwd || true)"

PROJECT_DIR="${BLUEBIRD_PROJECT_DIR:-${PROJECT_DIR:-${REPO_GUESS:-/Volumes/DOCK/BLUEBIRD 4.0}}}"
DB_PATH="${BLUEBIRD_DB_PATH:-${DB_PATH:-${PROJECT_DIR}/data/bluebird.db}}"
STATE_DIR="${PROJECT_DIR}/data/state"
CRASH_STATE="${STATE_DIR}/crash-loop-notifier.json"
DISK_ALERT_STATE="${STATE_DIR}/disk-alert.json"
PENDING_ALERTS="${STATE_DIR}/pending-alerts.txt"
MAX_AGE_SECONDS=120  # 2 minutes (faster unattended recovery)
MAX_CRASHES=3
WINDOW_MINUTES=30
DISK_THRESHOLD=90
LOG_PREFIX="[$(date '+%Y-%m-%d %H:%M:%S')] [NOTIFIER]"

# Ensure state directory exists
mkdir -p "$STATE_DIR"

# ============================================================================
# DISK SPACE MONITORING
# ============================================================================
check_disk_space() {
    # Get disk usage percentage for the filesystem containing PROJECT_DIR
    DISK_USAGE=$(df -P "$PROJECT_DIR" | tail -1 | awk '{print $5}' | tr -d '%')

    if [ "$DISK_USAGE" -ge "$DISK_THRESHOLD" ]; then
        # Check if we already alerted today
        TODAY=$(date '+%Y-%m-%d')
        LAST_ALERT=""

        if [ -f "$DISK_ALERT_STATE" ]; then
            LAST_ALERT=$(python3 -c "import json; print(json.load(open('${DISK_ALERT_STATE}')).get('last_alert_date', ''))" 2>/dev/null || echo "")
        fi

        if [ "$LAST_ALERT" != "$TODAY" ]; then
            echo "${LOG_PREFIX} DISK ALERT: Usage at ${DISK_USAGE}% (threshold: ${DISK_THRESHOLD}%)"

            # Write alert to pending alerts
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] DISK SPACE CRITICAL: ${DISK_USAGE}% used on BLUEBIRD volume" >> "$PENDING_ALERTS"

            # Update last alert date
            echo "{\"last_alert_date\": \"${TODAY}\"}" > "$DISK_ALERT_STATE"
        fi
    fi
}

# Run disk check first
check_disk_space

# ============================================================================
# CRASH LOOP DETECTION
# ============================================================================
check_crash_loop() {
    local CURRENT_TIME
    CURRENT_TIME=$(date -u '+%Y-%m-%dT%H:%M:%S')

    # Read existing crash state (handle missing/corrupt file)
    local PAUSED="false"
    local HISTORY="[]"

    if [ -f "$CRASH_STATE" ]; then
        PAUSED=$(python3 -c "
import json
try:
    data = json.load(open('${CRASH_STATE}'))
    print(str(data.get('paused', False)).lower())
except:
    print('false')
" 2>/dev/null || echo "false")

        HISTORY=$(python3 -c "
import json
try:
    data = json.load(open('${CRASH_STATE}'))
    print(json.dumps(data.get('history', [])))
except:
    print('[]')
" 2>/dev/null || echo "[]")
    fi

    # If paused, log and exit without restarting
    if [ "$PAUSED" = "true" ]; then
        echo "${LOG_PREFIX} CRASH LOOP PAUSED: Too many restarts. Manual intervention required."
        echo "${LOG_PREFIX} To clear: rm \"${CRASH_STATE}\""
        exit 0
    fi

    # Add current timestamp and prune old entries
    python3 << EOF
import json
from datetime import datetime, timedelta

crash_state = "${CRASH_STATE}"
pending_alerts = "${PENDING_ALERTS}"
current_time = "${CURRENT_TIME}"
window_minutes = ${WINDOW_MINUTES}
max_crashes = ${MAX_CRASHES}

# Parse history
try:
    history = json.loads('${HISTORY}')
except:
    history = []

# Add current restart attempt
history.append(current_time)

# Prune entries older than window
cutoff = datetime.fromisoformat(current_time) - timedelta(minutes=window_minutes)
history = [ts for ts in history if datetime.fromisoformat(ts) > cutoff]

# Check if we've exceeded max crashes
if len(history) >= max_crashes:
    # CRASH LOOP DETECTED - pause and alert
    with open(crash_state, 'w') as f:
        json.dump({'paused': True, 'history': history}, f, indent=2)

    # Write pending alert
    with open(pending_alerts, 'a') as f:
        f.write(f"[{current_time}] NOTIFIER CRASH LOOP: {len(history)} restarts in {window_minutes} minutes. Auto-restart paused.\n")

    print(f"CRASH_LOOP_DETECTED:{len(history)}")
else:
    # Update history, continue with restart
    with open(crash_state, 'w') as f:
        json.dump({'paused': False, 'history': history}, f, indent=2)

    print(f"OK:{len(history)}")
EOF
}

# ============================================================================
# MAIN WATCHDOG LOGIC
# ============================================================================

# Check if database exists
if [ ! -f "$DB_PATH" ]; then
    echo "${LOG_PREFIX} ERROR: Database not found at $DB_PATH"
    exit 1
fi

# Query the heartbeat from database.
# Use a sqlite timeout to reduce false negatives when the DB is briefly locked.
HEARTBEAT_QUERY_OUT=$(sqlite3 -cmd ".timeout 5000" "$DB_PATH" "SELECT last_heartbeat FROM notifier_status WHERE id = 1" 2>&1) || SQLITE_RC=$?
SQLITE_RC=${SQLITE_RC:-0}
if [ "$SQLITE_RC" -ne 0 ]; then
    echo "${LOG_PREFIX} WARNING: sqlite heartbeat query failed (rc=${SQLITE_RC})."
    echo "${LOG_PREFIX} sqlite3 output: ${HEARTBEAT_QUERY_OUT}"

    # Fallback: if notifier process appears to be running, do not restart.
    if pgrep -f "src\\.notifications\\.notifier" >/dev/null 2>&1; then
        echo "${LOG_PREFIX} OK: notifier process detected (skipping restart despite DB read failure)"
        exit 0
    fi

    echo "${LOG_PREFIX} ALERT: notifier process not detected and DB unreadable - attempting restart"
    HEARTBEAT=""
else
    HEARTBEAT="$HEARTBEAT_QUERY_OUT"
fi

if [ -z "$HEARTBEAT" ]; then
    echo "${LOG_PREFIX} WARNING: No heartbeat record found in database"
    echo "${LOG_PREFIX} Notifier may have never started - attempting to start..."

    # Check crash loop before restarting
    CRASH_RESULT=$(check_crash_loop)
    if [[ "$CRASH_RESULT" == CRASH_LOOP_DETECTED* ]]; then
        echo "${LOG_PREFIX} CRASH LOOP DETECTED - NOT RESTARTING"
        exit 0
    fi

    cd "$PROJECT_DIR"
    # Propagate DB override to the notifier process (if configured)
    if [ -n "${BLUEBIRD_DB_PATH:-}" ]; then
        export BLUEBIRD_DB_PATH
    fi
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

    # Check crash loop before restarting
    CRASH_RESULT=$(check_crash_loop)
    if [[ "$CRASH_RESULT" == CRASH_LOOP_DETECTED* ]]; then
        echo "${LOG_PREFIX} CRASH LOOP DETECTED - NOT RESTARTING"
        exit 0
    fi

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
    # Propagate DB override to the notifier process (if configured)
    if [ -n "${BLUEBIRD_DB_PATH:-}" ]; then
        export BLUEBIRD_DB_PATH
    fi
    nohup caffeinate -i python3 -m src.notifications.notifier > /tmp/bluebird-notifier.log 2>&1 &
    NEW_PID=$!

    echo "${LOG_PREFIX} Restarted notifier with PID: $NEW_PID"
else
    echo "${LOG_PREFIX} OK: Notifier is healthy"
fi
