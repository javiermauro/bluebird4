#!/bin/bash
# BLUEBIRD Notifier Daemon
# Self-restarting wrapper that keeps the notifier running
# Uses caffeinate to prevent macOS from killing the process

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
LOG_FILE="/tmp/bluebird-notifier.log"
RESTART_DELAY=5

echo "$(date): BLUEBIRD Notifier Daemon starting..." >> "$LOG_FILE"
echo "$(date): Working directory: $SCRIPT_DIR" >> "$LOG_FILE"

cd "$SCRIPT_DIR" || exit 1

while true; do
    echo "$(date): Starting notifier with caffeinate..." >> "$LOG_FILE"

    # Run with caffeinate to prevent App Nap
    # -i = prevent idle sleep
    # -s = prevent system sleep
    caffeinate -is python3 src/notifications/notifier.py 2>&1 | tee -a "$LOG_FILE"

    EXIT_CODE=$?
    echo "$(date): Notifier exited with code $EXIT_CODE" >> "$LOG_FILE"

    # Check if we should stop (exit code 0 means clean shutdown)
    if [ $EXIT_CODE -eq 0 ]; then
        echo "$(date): Clean shutdown, not restarting" >> "$LOG_FILE"
        break
    fi

    # Check for stop signal file
    if [ -f "/tmp/bluebird-notifier-stop" ]; then
        echo "$(date): Stop signal received, exiting" >> "$LOG_FILE"
        rm -f "/tmp/bluebird-notifier-stop"
        break
    fi

    echo "$(date): Restarting in $RESTART_DELAY seconds..." >> "$LOG_FILE"
    sleep $RESTART_DELAY
done

echo "$(date): BLUEBIRD Notifier Daemon stopped" >> "$LOG_FILE"
