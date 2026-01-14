#!/bin/bash
# BLUEBIRD-LIVE Database Backup Script
# WARNING: This is for LIVE TRADING instance
#
# Creates daily backups of bluebird.db using SQLite's .backup command
# which is safe to run while the database is in use.
#
# Intended to be run via cron daily:
#   0 3 * * * /bin/bash "/Users/javierrodriguez/BLUEBIRD/bluebird-live/scripts/backup_db.sh" >> /tmp/bluebird-live-backup.log 2>&1
#

set -e

# Configuration - LIVE instance
PROJECT_DIR="/Users/javierrodriguez/BLUEBIRD/bluebird-live"
DB_PATH="${PROJECT_DIR}/data/bluebird.db"
BACKUP_DIR="${PROJECT_DIR}/data/backups"
KEEP_DAYS=7  # Number of daily backups to keep
LOG_PREFIX="[$(date '+%Y-%m-%d %H:%M:%S')] [LIVE-BACKUP]"

# Create backup directory if it doesn't exist
mkdir -p "$BACKUP_DIR"

# Check if database exists
if [ ! -f "$DB_PATH" ]; then
    echo "${LOG_PREFIX} ERROR: Database not found at $DB_PATH"
    exit 1
fi

# Generate backup filename with timestamp
BACKUP_DATE=$(date '+%Y%m%d')
BACKUP_FILE="${BACKUP_DIR}/bluebird-${BACKUP_DATE}.db"

# Check if today's backup already exists
if [ -f "$BACKUP_FILE" ]; then
    echo "${LOG_PREFIX} Backup for today already exists: $BACKUP_FILE"
    # Create an hourly backup instead if running multiple times
    BACKUP_FILE="${BACKUP_DIR}/bluebird-${BACKUP_DATE}-$(date '+%H%M').db"
    echo "${LOG_PREFIX} Creating additional backup: $BACKUP_FILE"
fi

# Get database size before backup
DB_SIZE=$(ls -lh "$DB_PATH" | awk '{print $5}')
echo "${LOG_PREFIX} Starting backup of $DB_PATH ($DB_SIZE)"

# Use SQLite's .backup command for safe online backup
sqlite3 "$DB_PATH" ".backup '$BACKUP_FILE'"

# Verify backup was created
if [ -f "$BACKUP_FILE" ]; then
    BACKUP_SIZE=$(ls -lh "$BACKUP_FILE" | awk '{print $5}')
    echo "${LOG_PREFIX} SUCCESS: Created $BACKUP_FILE ($BACKUP_SIZE)"

    # Verify backup integrity
    INTEGRITY=$(sqlite3 "$BACKUP_FILE" "PRAGMA integrity_check" 2>/dev/null || echo "FAILED")
    if [ "$INTEGRITY" = "ok" ]; then
        echo "${LOG_PREFIX} Backup integrity check: PASSED"
    else
        echo "${LOG_PREFIX} WARNING: Backup integrity check: $INTEGRITY"
    fi
else
    echo "${LOG_PREFIX} ERROR: Backup file was not created"
    exit 1
fi

# Cleanup old backups (keep only KEEP_DAYS worth)
echo "${LOG_PREFIX} Cleaning up old backups (keeping ${KEEP_DAYS} days)..."
DELETED=0
for old_backup in $(ls -t "${BACKUP_DIR}"/bluebird-*.db 2>/dev/null | tail -n +$((KEEP_DAYS + 1))); do
    echo "${LOG_PREFIX} Deleting old backup: $old_backup"
    rm -f "$old_backup"
    ((DELETED++)) || true
done

if [ "$DELETED" -gt 0 ]; then
    echo "${LOG_PREFIX} Deleted $DELETED old backup(s)"
fi

# List current backups
echo "${LOG_PREFIX} Current backups:"
ls -lh "${BACKUP_DIR}"/bluebird-*.db 2>/dev/null | while read line; do
    echo "${LOG_PREFIX}   $line"
done

echo "${LOG_PREFIX} Backup complete"
