#!/bin/bash
# BLUEBIRD-LIVE Log Rotation Script
# WARNING: This is for LIVE TRADING instance
#
# Rotates log files when they exceed MAX_SIZE_MB
# Keeps KEEP_ROTATIONS old copies, compressed
#
# Cron (daily at 5 AM):
#   0 5 * * * /bin/bash "/Users/javierrodriguez/BLUEBIRD/bluebird-live/scripts/rotate_logs.sh" >> /tmp/bluebird-live-logrotate.log 2>&1

set -e

# Configuration
MAX_SIZE_MB=50
KEEP_ROTATIONS=3
LOG_PREFIX="[$(date '+%Y-%m-%d %H:%M:%S')] [LIVE-LOGROTATE]"

# LIVE log files to rotate
LOG_FILES=(
    "/tmp/bluebird-live-bot.log"
    "/tmp/bluebird-live-notifier.log"
    "/tmp/bluebird-live-watchdog.log"
    "/tmp/bluebird-live-backup.log"
    "/tmp/bluebird-live-cleanup.log"
)

rotate_log() {
    local logfile="$1"
    local basename=$(basename "$logfile")
    local dir=$(dirname "$logfile")
    
    if [ ! -f "$logfile" ]; then
        return
    fi
    
    # Get file size in MB
    local size_bytes=$(stat -f%z "$logfile" 2>/dev/null || stat -c%s "$logfile" 2>/dev/null)
    local size_mb=$((size_bytes / 1024 / 1024))
    
    if [ "$size_mb" -lt "$MAX_SIZE_MB" ]; then
        echo "${LOG_PREFIX} $basename: ${size_mb}MB (under ${MAX_SIZE_MB}MB limit)"
        return
    fi
    
    echo "${LOG_PREFIX} $basename: ${size_mb}MB exceeds limit, rotating..."
    
    # Remove oldest rotation
    if [ -f "${logfile}.${KEEP_ROTATIONS}.gz" ]; then
        rm -f "${logfile}.${KEEP_ROTATIONS}.gz"
    fi
    
    # Shift existing rotations
    for i in $(seq $((KEEP_ROTATIONS-1)) -1 1); do
        if [ -f "${logfile}.${i}.gz" ]; then
            mv "${logfile}.${i}.gz" "${logfile}.$((i+1)).gz"
        fi
    done
    
    # Rotate and compress current log
    cp "$logfile" "${logfile}.1"
    gzip -f "${logfile}.1"
    
    # Truncate current log (don't delete - processes may have it open)
    : > "$logfile"
    
    echo "${LOG_PREFIX} $basename: Rotated and compressed to ${logfile}.1.gz"
}

echo "${LOG_PREFIX} Starting log rotation check..."

for logfile in "${LOG_FILES[@]}"; do
    rotate_log "$logfile"
done

echo "${LOG_PREFIX} Log rotation complete"
