#!/usr/bin/env python3
"""
BLUEBIRD Database Cleanup Script

Usage:
    python3 scripts/cleanup_db.py              # Dry run
    python3 scripts/cleanup_db.py --execute    # Actually clean

Cron (weekly Sunday 4 AM):
    0 4 * * 0 cd "/Volumes/DOCK/BLUEBIRD 4.0" && python3 scripts/cleanup_db.py --execute >> /tmp/bluebird-cleanup.log 2>&1
"""

import sqlite3
import os
import sys
from datetime import datetime, timedelta

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(PROJECT_DIR, "data", "bluebird.db")

# Retention periods
EQUITY_KEEP_DAYS = 90
SMS_KEEP_DAYS = 90

def get_stats(conn):
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
    stats = {}
    for (table,) in cursor.fetchall():
        cursor.execute(f"SELECT COUNT(*) FROM {table}")
        stats[table] = cursor.fetchone()[0]
    return stats

def cleanup_table(conn, table, ts_col, keep_days, execute):
    cutoff = (datetime.now() - timedelta(days=keep_days)).strftime('%Y-%m-%d')
    cursor = conn.cursor()
    cursor.execute(f"SELECT COUNT(*) FROM {table} WHERE {ts_col} < ?", (cutoff,))
    to_del = cursor.fetchone()[0]
    cursor.execute(f"SELECT COUNT(*) FROM {table}")
    total = cursor.fetchone()[0]
    
    if execute and to_del > 0:
        cursor.execute(f"DELETE FROM {table} WHERE {ts_col} < ?", (cutoff,))
        conn.commit()
    
    return {"table": table, "to_delete": to_del, "total": total, "cutoff": cutoff, "keep_days": keep_days}

def main():
    execute = "--execute" in sys.argv
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    print(f"\n{'='*50}")
    print(f"BLUEBIRD Database Cleanup - {ts}")
    print(f"{'='*50}")
    print(f"Mode: {'EXECUTE (will delete)' if execute else 'DRY RUN (preview)'}")
    print(f"Database: {DB_PATH}\n")
    
    if not os.path.exists(DB_PATH):
        print(f"ERROR: DB not found: {DB_PATH}")
        sys.exit(1)
    
    db_size = os.path.getsize(DB_PATH) / (1024*1024)
    print(f"Database size: {db_size:.2f} MB\n")
    
    conn = sqlite3.connect(DB_PATH)
    
    print("Row counts before cleanup:")
    for t, c in sorted(get_stats(conn).items()):
        print(f"  {t}: {c:,}")
    print()
    
    results = [
        cleanup_table(conn, "equity_snapshots", "timestamp", EQUITY_KEEP_DAYS, execute),
        cleanup_table(conn, "sms_history", "timestamp", SMS_KEEP_DAYS, execute),
    ]
    
    print("Cleanup summary:")
    print("-" * 50)
    total_del = 0
    for r in results:
        status = "DELETED" if execute and r['to_delete'] > 0 else "would delete"
        print(f"  {r['table']}: {r['to_delete']:,} rows {status} (keep {r['keep_days']} days)")
        total_del += r['to_delete']
    
    print("-" * 50)
    
    if execute and total_del > 0:
        print(f"\nTotal deleted: {total_del:,} rows")
        print("Running VACUUM to reclaim space...")
        conn.execute("VACUUM")
        new_size = os.path.getsize(DB_PATH) / (1024*1024)
        print(f"Size: {db_size:.2f} MB -> {new_size:.2f} MB (saved {db_size-new_size:.2f} MB)")
    elif not execute:
        print(f"\nWould delete: {total_del:,} rows total")
        print("Run with --execute to delete")
    
    conn.close()
    print()

if __name__ == "__main__":
    main()
