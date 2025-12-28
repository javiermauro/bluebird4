"""
BLUEBIRD 4.0 - SQLite Database Module

Persistent storage for:
- Trade history (all buys/sells with profit tracking)
- Equity snapshots (hourly/daily equity values)
- Order log (full order execution details)
- Grid state history
"""

import sqlite3
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from contextlib import contextmanager
import json

logger = logging.getLogger("BlueBirdDB")

# Database location - use project directory for persistence
DB_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DB_PATH = os.path.join(DB_DIR, "data", "bluebird.db")


def ensure_db_dir():
    """Ensure the data directory exists."""
    data_dir = os.path.dirname(DB_PATH)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        logger.info(f"Created data directory: {data_dir}")


@contextmanager
def get_db_connection():
    """Get a database connection with automatic cleanup."""
    ensure_db_dir()
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # Return rows as dicts
    try:
        yield conn
    finally:
        conn.close()


def _run_migrations(conn):
    """
    Run database migrations to add new columns.
    Safe to run multiple times (idempotent).
    """
    cursor = conn.cursor()

    # Get existing columns in trades table
    cursor.execute("PRAGMA table_info(trades)")
    existing_columns = {row[1] for row in cursor.fetchall()}

    # Fee audit columns migration (Dec 2025)
    fee_columns = [
        ("fee_rate", "REAL"),           # The rate applied (e.g., 0.0025)
        ("fee_type", "TEXT"),           # 'maker', 'taker', or 'maker_assumed'
        ("fee_tier", "TEXT"),           # 'Tier 1', 'Tier 2', etc.
        ("rolling_30d_volume", "REAL"), # Volume at time of fill
        ("fee_day_bucket", "TEXT"),     # '2025-12-10' (ET fee day)
        ("fee_conservative", "REAL"),   # Worst-case fee (taker rate)
    ]

    for col_name, col_type in fee_columns:
        if col_name not in existing_columns:
            try:
                cursor.execute(f"ALTER TABLE trades ADD COLUMN {col_name} {col_type}")
                logger.info(f"Migration: Added column trades.{col_name}")
            except sqlite3.OperationalError as e:
                if "duplicate column name" not in str(e).lower():
                    logger.warning(f"Migration error adding {col_name}: {e}")

    conn.commit()


def init_database():
    """Initialize the database with all required tables."""
    ensure_db_dir()

    with get_db_connection() as conn:
        cursor = conn.cursor()

        # Trades table - every buy/sell execution
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,  -- 'buy' or 'sell'
                quantity REAL NOT NULL,
                price REAL NOT NULL,
                total_value REAL NOT NULL,
                order_id TEXT,
                profit REAL DEFAULT 0,  -- Realized profit (for sells)
                fees REAL DEFAULT 0,
                source TEXT DEFAULT 'grid',  -- 'grid', 'manual', 'signal'
                notes TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Equity snapshots - track account value over time
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS equity_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                equity REAL NOT NULL,
                cash REAL,
                buying_power REAL,
                positions_value REAL,
                daily_pnl REAL,
                daily_pnl_pct REAL,
                source TEXT DEFAULT 'auto',  -- 'auto', 'manual', 'backfill'
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(timestamp)
            )
        """)

        # Orders table - full order details from Alpaca
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS orders (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                order_id TEXT UNIQUE NOT NULL,
                client_order_id TEXT,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                order_type TEXT,
                qty REAL,
                filled_qty REAL,
                filled_avg_price REAL,
                status TEXT,
                submitted_at TEXT,
                filled_at TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Daily summary table - aggregated daily stats
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS daily_summary (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT UNIQUE NOT NULL,
                starting_equity REAL,
                ending_equity REAL,
                high_equity REAL,
                low_equity REAL,
                total_trades INTEGER DEFAULT 0,
                total_buys INTEGER DEFAULT 0,
                total_sells INTEGER DEFAULT 0,
                realized_pnl REAL DEFAULT 0,
                unrealized_pnl REAL DEFAULT 0,
                fees REAL DEFAULT 0,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Grid snapshots - track grid state changes
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS grid_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                symbol TEXT NOT NULL,
                lower_price REAL,
                upper_price REAL,
                num_grids INTEGER,
                filled_levels INTEGER,
                total_levels INTEGER,
                total_profit REAL,
                completed_trades INTEGER,
                state_json TEXT,  -- Full grid state as JSON
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # ============ NOTIFICATION TABLES ============

        # SMS History - persistent record of all SMS sent
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sms_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                message_type TEXT NOT NULL,
                recipient TEXT NOT NULL,
                body_preview TEXT,
                twilio_sid TEXT,
                status TEXT DEFAULT 'sent',
                retry_count INTEGER DEFAULT 0,
                error_message TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Notified Trade IDs - prevent duplicate alerts
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS notified_trade_ids (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                order_id TEXT UNIQUE NOT NULL,
                notified_at TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Failed SMS Queue - retry later
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sms_queue (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                body TEXT NOT NULL,
                message_type TEXT,
                priority INTEGER DEFAULT 0,
                attempts INTEGER DEFAULT 0,
                max_attempts INTEGER DEFAULT 5,
                last_attempt TEXT,
                next_retry TEXT,
                error_message TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Notifier Status - heartbeat and state (single row table)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS notifier_status (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                pid INTEGER,
                started_at TEXT,
                last_heartbeat TEXT,
                last_sms_at TEXT,
                sms_today INTEGER DEFAULT 0,
                sms_today_date TEXT,
                last_overlay_mode TEXT,
                last_drawdown_alert REAL,
                api_failures INTEGER DEFAULT 0,
                status TEXT DEFAULT 'running'
            )
        """)

        # Bot Status - heartbeat and state for API server (single row table)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS bot_status (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                pid INTEGER,
                started_at TEXT,
                last_heartbeat TEXT,
                overlay_mode TEXT,
                active_symbols INTEGER DEFAULT 0,
                total_trades INTEGER DEFAULT 0,
                status TEXT DEFAULT 'running'
            )
        """)

        # Create indexes for faster queries
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_trades_timestamp ON trades(timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_equity_timestamp ON equity_snapshots(timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_orders_filled_at ON orders(filled_at)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_orders_symbol ON orders(symbol)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_daily_date ON daily_summary(date)")

        # Unique indexes for idempotency and upsert operations
        cursor.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_trades_unique ON trades(order_id, side, quantity, price)")
        cursor.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_daily_summary_date ON daily_summary(date)")

        # Notification indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_sms_history_timestamp ON sms_history(timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_sms_queue_next_retry ON sms_queue(next_retry)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_notified_trades_order_id ON notified_trade_ids(order_id)")

        conn.commit()
        logger.info(f"Database initialized at {DB_PATH}")

        # Run migrations after initial schema creation
        _run_migrations(conn)


# ============ TRADE FUNCTIONS ============

def record_trade(
    symbol: str,
    side: str,
    quantity: float,
    price: float,
    order_id: str = None,
    profit: float = 0,
    fees: float = 0,
    source: str = 'grid',
    notes: str = None,
    timestamp: datetime = None,
    # Fee audit fields (optional, for tier-correct fee tracking)
    fee_rate: float = None,
    fee_type: str = None,
    fee_tier: str = None,
    rolling_30d_volume: float = None,
    fee_day_bucket: str = None,
    fee_conservative: float = None,
) -> int:
    """
    Record a trade execution. Idempotent via unique constraint on order_id+side+qty+price.

    Fee audit fields:
        fee_rate: Applied fee rate (e.g., 0.0015 for maker)
        fee_type: 'maker', 'taker', or 'maker_assumed'
        fee_tier: 'Tier 1', 'Tier 2', etc.
        rolling_30d_volume: 30-day volume at time of fill (for tier calculation)
        fee_day_bucket: Fee day in ET (3am boundary), e.g., '2025-12-10'
        fee_conservative: Worst-case fee assuming taker rate

    Returns:
        Trade ID if inserted, -1 if already exists or skipped
    """
    # Guard: order_id must be present for idempotency
    if not order_id:
        logger.warning(f"record_trade called without order_id for {symbol} {side}, skipping")
        return -1

    if timestamp is None:
        timestamp = datetime.now()

    # Normalize side to lowercase for consistent grouping in daily summary
    side_normalized = side.lower()

    # Lazy import to avoid circular imports (db.py shouldn't depend on trading config at import time)
    try:
        from config_ultra import SYMBOL_PRECISION
        # Format: (price_decimals, qty_decimals)
        price_decimals, qty_decimals = SYMBOL_PRECISION.get(symbol, (2, 6))
    except ImportError:
        # Fallback if config not available
        price_decimals, qty_decimals = 2, 6

    # Round qty/price to symbol-specific precision to avoid float drift duplicates
    quantity_rounded = round(quantity, qty_decimals)
    price_rounded = round(price, price_decimals)
    total_value = quantity_rounded * price_rounded

    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT OR IGNORE INTO trades (
                timestamp, symbol, side, quantity, price, total_value,
                order_id, profit, fees, source, notes,
                fee_rate, fee_type, fee_tier, rolling_30d_volume, fee_day_bucket, fee_conservative
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            timestamp.isoformat(), symbol, side_normalized, quantity_rounded, price_rounded,
            total_value, order_id, profit, fees, source, notes,
            fee_rate, fee_type, fee_tier, rolling_30d_volume, fee_day_bucket, fee_conservative
        ))

        if cursor.rowcount == 0:
            logger.debug(f"Trade already exists for order {order_id}, skipping")
            return -1

        conn.commit()
        trade_id = cursor.lastrowid
        logger.debug(f"Recorded trade #{trade_id}: {side_normalized} {quantity_rounded} {symbol} @ ${price_rounded:.2f}")
        return trade_id


def update_trade_fees(
    trade_id: int = None,
    order_id: str = None,
    fees: float = None,
    fee_rate: float = None,
    fee_type: str = None,
    fee_tier: str = None,
    rolling_30d_volume: float = None,
    fee_day_bucket: str = None,
    fee_conservative: float = None,
) -> bool:
    """
    Update fee fields on an existing trade (for backfill).

    Args:
        trade_id: Trade ID to update (preferred)
        order_id: Order ID to match (if trade_id not provided)
        Other args: Fee fields to update

    Returns:
        True if updated, False if not found
    """
    if trade_id is None and order_id is None:
        logger.warning("update_trade_fees requires either trade_id or order_id")
        return False

    # Build SET clause dynamically for non-None fields
    updates = []
    params = []

    if fees is not None:
        updates.append("fees = ?")
        params.append(fees)
    if fee_rate is not None:
        updates.append("fee_rate = ?")
        params.append(fee_rate)
    if fee_type is not None:
        updates.append("fee_type = ?")
        params.append(fee_type)
    if fee_tier is not None:
        updates.append("fee_tier = ?")
        params.append(fee_tier)
    if rolling_30d_volume is not None:
        updates.append("rolling_30d_volume = ?")
        params.append(rolling_30d_volume)
    if fee_day_bucket is not None:
        updates.append("fee_day_bucket = ?")
        params.append(fee_day_bucket)
    if fee_conservative is not None:
        updates.append("fee_conservative = ?")
        params.append(fee_conservative)

    if not updates:
        return False

    # Build WHERE clause
    if trade_id is not None:
        where = "id = ?"
        params.append(trade_id)
    else:
        where = "order_id = ?"
        params.append(order_id)

    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(f"UPDATE trades SET {', '.join(updates)} WHERE {where}", params)
        conn.commit()

        if cursor.rowcount > 0:
            identifier = f"#{trade_id}" if trade_id else f"order={order_id}"
            logger.debug(f"Updated fees for trade {identifier}")
            return True
        return False


def get_fee_stats(days: int = None, since_date: str = None) -> Dict:
    """
    Get aggregated fee statistics.

    Args:
        days: Number of days to look back
        since_date: Start date string (ISO format), e.g., '2025-12-01'

    Returns:
        Dict with fee totals, uncertain count, by-tier breakdown
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()

        where_clauses = []
        params = []

        if days:
            start = (datetime.now() - timedelta(days=days)).isoformat()
            where_clauses.append("timestamp >= ?")
            params.append(start)

        if since_date:
            where_clauses.append("timestamp >= ?")
            params.append(since_date)

        where = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""

        # Total fee stats
        cursor.execute(f"""
            SELECT
                COALESCE(SUM(fees), 0) as total_fees_expected,
                COALESCE(SUM(fee_conservative), 0) as total_fees_conservative,
                COALESCE(SUM(total_value), 0) as total_notional,
                COUNT(*) as total_trades,
                SUM(CASE WHEN fee_type = 'maker_assumed' THEN 1 ELSE 0 END) as uncertain_count
            FROM trades {where}
        """, params)

        result = cursor.fetchone()

        # By tier breakdown
        cursor.execute(f"""
            SELECT
                fee_tier,
                COUNT(*) as trades,
                COALESCE(SUM(fees), 0) as fees_expected,
                COALESCE(SUM(fee_conservative), 0) as fees_conservative,
                COALESCE(SUM(total_value), 0) as notional
            FROM trades {where}
            GROUP BY fee_tier
            ORDER BY fee_tier
        """, params)

        by_tier = {}
        for row in cursor.fetchall():
            tier = row['fee_tier'] or 'Unknown'
            by_tier[tier] = {
                'trades': row['trades'],
                'fees_expected': row['fees_expected'] or 0,
                'fees_conservative': row['fees_conservative'] or 0,
                'notional': row['notional'] or 0,
            }

        return {
            'total_fees_expected': result['total_fees_expected'] or 0,
            'total_fees_conservative': result['total_fees_conservative'] or 0,
            'total_notional': result['total_notional'] or 0,
            'total_trades': result['total_trades'] or 0,
            'uncertain_count': result['uncertain_count'] or 0,
            'by_tier': by_tier,
        }


def get_rolling_30d_volume(as_of_timestamp: datetime = None) -> float:
    """
    Get rolling 30-day trading volume from trades table.

    Args:
        as_of_timestamp: Calculate volume as of this time (default: now)

    Returns:
        Rolling 30-day volume in USD
    """
    if as_of_timestamp is None:
        as_of_timestamp = datetime.now()

    start_time = as_of_timestamp - timedelta(days=30)

    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT COALESCE(SUM(total_value), 0) as volume
            FROM trades
            WHERE timestamp > ? AND timestamp <= ?
        """, (start_time.isoformat(), as_of_timestamp.isoformat()))

        result = cursor.fetchone()
        return float(result['volume']) if result else 0.0


def get_trades(
    symbol: str = None,
    side: str = None,
    days: int = None,
    start_date: datetime = None,
    end_date: datetime = None,
    limit: int = 1000
) -> List[Dict]:
    """Get trade history with optional filters."""
    with get_db_connection() as conn:
        cursor = conn.cursor()

        query = "SELECT * FROM trades WHERE 1=1"
        params = []

        if symbol:
            query += " AND symbol = ?"
            params.append(symbol)

        if side:
            query += " AND side = ?"
            params.append(side.lower())

        if days:
            start = (datetime.now() - timedelta(days=days)).isoformat()
            query += " AND timestamp >= ?"
            params.append(start)

        if start_date:
            query += " AND timestamp >= ?"
            params.append(start_date.isoformat())

        if end_date:
            query += " AND timestamp <= ?"
            params.append(end_date.isoformat())

        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        cursor.execute(query, params)
        return [dict(row) for row in cursor.fetchall()]


def get_trade_stats(days: int = None) -> Dict:
    """Get aggregated trade statistics."""
    with get_db_connection() as conn:
        cursor = conn.cursor()

        where_clause = ""
        params = []
        if days:
            start = (datetime.now() - timedelta(days=days)).isoformat()
            where_clause = "WHERE timestamp >= ?"
            params.append(start)

        # Total stats
        cursor.execute(f"""
            SELECT
                COUNT(*) as total_trades,
                SUM(CASE WHEN side = 'buy' THEN 1 ELSE 0 END) as total_buys,
                SUM(CASE WHEN side = 'sell' THEN 1 ELSE 0 END) as total_sells,
                SUM(total_value) as total_volume,
                SUM(profit) as total_profit,
                SUM(fees) as total_fees,
                AVG(price) as avg_price
            FROM trades {where_clause}
        """, params)

        stats = dict(cursor.fetchone())

        # By symbol
        cursor.execute(f"""
            SELECT symbol,
                   COUNT(*) as trades,
                   SUM(profit) as profit,
                   SUM(total_value) as volume
            FROM trades {where_clause}
            GROUP BY symbol
        """, params)

        stats['by_symbol'] = {row['symbol']: {
            'trades': row['trades'],
            'profit': row['profit'] or 0,
            'volume': row['volume'] or 0
        } for row in cursor.fetchall()}

        return stats


# ============ EQUITY FUNCTIONS ============

def record_equity_snapshot(
    equity: float,
    cash: float = None,
    buying_power: float = None,
    positions_value: float = None,
    daily_pnl: float = None,
    daily_pnl_pct: float = None,
    source: str = 'auto',
    timestamp: datetime = None
) -> int:
    """Record an equity snapshot."""
    if timestamp is None:
        timestamp = datetime.now()

    with get_db_connection() as conn:
        cursor = conn.cursor()
        try:
            cursor.execute("""
                INSERT OR REPLACE INTO equity_snapshots
                (timestamp, equity, cash, buying_power, positions_value, daily_pnl, daily_pnl_pct, source)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (timestamp.isoformat(), equity, cash, buying_power, positions_value,
                  daily_pnl, daily_pnl_pct, source))
            conn.commit()
            return cursor.lastrowid
        except sqlite3.IntegrityError:
            # Timestamp already exists, update instead
            cursor.execute("""
                UPDATE equity_snapshots
                SET equity = ?, cash = ?, buying_power = ?, positions_value = ?,
                    daily_pnl = ?, daily_pnl_pct = ?, source = ?
                WHERE timestamp = ?
            """, (equity, cash, buying_power, positions_value, daily_pnl, daily_pnl_pct,
                  source, timestamp.isoformat()))
            conn.commit()
            return -1


def get_equity_history(days: int = 30, interval: str = 'daily') -> List[Dict]:
    """
    Get equity history.

    Args:
        days: Number of days of history
        interval: 'hourly' or 'daily'
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()

        start = (datetime.now() - timedelta(days=days)).isoformat()

        if interval == 'daily':
            # Get one snapshot per day (latest of each day)
            cursor.execute("""
                SELECT date(timestamp) as date,
                       MAX(equity) as high,
                       MIN(equity) as low,
                       (SELECT equity FROM equity_snapshots e2
                        WHERE date(e2.timestamp) = date(e1.timestamp)
                        ORDER BY e2.timestamp DESC LIMIT 1) as close,
                       (SELECT equity FROM equity_snapshots e3
                        WHERE date(e3.timestamp) = date(e1.timestamp)
                        ORDER BY e3.timestamp ASC LIMIT 1) as open
                FROM equity_snapshots e1
                WHERE timestamp >= ?
                GROUP BY date(timestamp)
                ORDER BY date(timestamp) ASC
            """, (start,))
        else:
            # Get all snapshots
            cursor.execute("""
                SELECT timestamp, equity, cash, buying_power, positions_value,
                       daily_pnl, daily_pnl_pct
                FROM equity_snapshots
                WHERE timestamp >= ?
                ORDER BY timestamp ASC
            """, (start,))

        return [dict(row) for row in cursor.fetchall()]


def get_equity_range(days: int = 30) -> Dict:
    """Get equity statistics for a time range."""
    with get_db_connection() as conn:
        cursor = conn.cursor()

        start = (datetime.now() - timedelta(days=days)).isoformat()

        cursor.execute("""
            SELECT
                MIN(equity) as trough,
                MAX(equity) as peak,
                (SELECT equity FROM equity_snapshots WHERE timestamp >= ? ORDER BY timestamp ASC LIMIT 1) as starting,
                (SELECT equity FROM equity_snapshots ORDER BY timestamp DESC LIMIT 1) as current,
                (SELECT timestamp FROM equity_snapshots WHERE equity = (SELECT MIN(equity) FROM equity_snapshots WHERE timestamp >= ?)) as trough_date,
                (SELECT timestamp FROM equity_snapshots WHERE equity = (SELECT MAX(equity) FROM equity_snapshots WHERE timestamp >= ?)) as peak_date
            FROM equity_snapshots
            WHERE timestamp >= ?
        """, (start, start, start, start))

        result = dict(cursor.fetchone())

        # Calculate recovery percentage
        if result['starting'] and result['trough'] and result['starting'] > result['trough']:
            recovery_pct = ((result['current'] - result['trough']) / (result['starting'] - result['trough'])) * 100
        else:
            recovery_pct = 0 if result['current'] < result['starting'] else 100

        result['recovery_pct'] = round(recovery_pct, 2)

        if result['starting']:
            result['total_return_pct'] = round(((result['current'] - result['starting']) / result['starting']) * 100, 2)
        else:
            result['total_return_pct'] = 0

        return result


# ============ ORDER FUNCTIONS ============

def record_order(order_data: Dict) -> int:
    """Record an order from Alpaca."""
    with get_db_connection() as conn:
        cursor = conn.cursor()

        try:
            cursor.execute("""
                INSERT OR REPLACE INTO orders
                (order_id, client_order_id, symbol, side, order_type, qty,
                 filled_qty, filled_avg_price, status, submitted_at, filled_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                order_data.get('order_id') or order_data.get('id'),
                order_data.get('client_order_id'),
                order_data.get('symbol'),
                str(order_data.get('side', '')).lower().replace('orderside.', ''),
                str(order_data.get('order_type') or order_data.get('type', '')),
                order_data.get('qty'),
                order_data.get('filled_qty'),
                order_data.get('filled_avg_price'),
                str(order_data.get('status', '')),
                order_data.get('submitted_at'),
                order_data.get('filled_at')
            ))
            conn.commit()
            return cursor.lastrowid
        except Exception as e:
            logger.error(f"Failed to record order: {e}")
            return -1


def get_orders(days: int = 7, symbol: str = None, status: str = None, limit: int = 500) -> List[Dict]:
    """Get orders from database."""
    with get_db_connection() as conn:
        cursor = conn.cursor()

        query = "SELECT * FROM orders WHERE 1=1"
        params = []

        if days:
            start = (datetime.now() - timedelta(days=days)).isoformat()
            query += " AND (filled_at >= ? OR submitted_at >= ?)"
            params.extend([start, start])

        if symbol:
            query += " AND symbol = ?"
            params.append(symbol)

        if status:
            query += " AND status LIKE ?"
            params.append(f"%{status}%")

        query += " ORDER BY filled_at DESC NULLS LAST LIMIT ?"
        params.append(limit)

        cursor.execute(query, params)
        return [dict(row) for row in cursor.fetchall()]


def get_order_stats() -> Dict:
    """Get order statistics."""
    with get_db_connection() as conn:
        cursor = conn.cursor()

        cursor.execute("""
            SELECT
                COUNT(*) as total_orders,
                SUM(CASE WHEN side LIKE '%buy%' THEN 1 ELSE 0 END) as buys,
                SUM(CASE WHEN side LIKE '%sell%' THEN 1 ELSE 0 END) as sells,
                SUM(filled_qty * filled_avg_price) as total_volume,
                COUNT(DISTINCT symbol) as symbols_traded
            FROM orders
            WHERE status LIKE '%filled%'
        """)

        return dict(cursor.fetchone())


# ============ DAILY SUMMARY FUNCTIONS ============

def update_daily_summary(
    date: str = None,
    starting_equity: float = None,
    ending_equity: float = None,
    high_equity: float = None,
    low_equity: float = None,
    trades: int = None,
    buys: int = None,
    sells: int = None,
    realized_pnl: float = None
):
    """Update or create daily summary."""
    if date is None:
        date = datetime.now().strftime('%Y-%m-%d')

    with get_db_connection() as conn:
        cursor = conn.cursor()

        # Check if exists
        cursor.execute("SELECT * FROM daily_summary WHERE date = ?", (date,))
        existing = cursor.fetchone()

        if existing:
            # Update existing
            updates = []
            params = []

            if ending_equity is not None:
                updates.append("ending_equity = ?")
                params.append(ending_equity)
            if high_equity is not None:
                updates.append("high_equity = MAX(COALESCE(high_equity, 0), ?)")
                params.append(high_equity)
            if low_equity is not None:
                updates.append("low_equity = MIN(COALESCE(low_equity, 999999999), ?)")
                params.append(low_equity)
            if trades is not None:
                updates.append("total_trades = ?")
                params.append(trades)
            if buys is not None:
                updates.append("total_buys = ?")
                params.append(buys)
            if sells is not None:
                updates.append("total_sells = ?")
                params.append(sells)
            if realized_pnl is not None:
                updates.append("realized_pnl = ?")
                params.append(realized_pnl)

            updates.append("updated_at = ?")
            params.append(datetime.now().isoformat())
            params.append(date)

            if updates:
                cursor.execute(f"""
                    UPDATE daily_summary SET {', '.join(updates)} WHERE date = ?
                """, params)
        else:
            # Insert new
            cursor.execute("""
                INSERT INTO daily_summary
                (date, starting_equity, ending_equity, high_equity, low_equity,
                 total_trades, total_buys, total_sells, realized_pnl)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (date, starting_equity, ending_equity, high_equity, low_equity,
                  trades or 0, buys or 0, sells or 0, realized_pnl or 0))

        conn.commit()


def get_daily_summaries(days: int = 30) -> List[Dict]:
    """Get daily summaries."""
    with get_db_connection() as conn:
        cursor = conn.cursor()

        start = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')

        cursor.execute("""
            SELECT * FROM daily_summary
            WHERE date >= ?
            ORDER BY date DESC
        """, (start,))

        return [dict(row) for row in cursor.fetchall()]


def recompute_daily_summary(date_local: str = None) -> Dict:
    """
    Recompute daily summary from equity_snapshots and orders (filled) for given local date.
    Uses substr(timestamp, 1, 10) to match YYYY-MM-DD prefix (avoids SQLite date() parsing).
    Idempotent - can be called multiple times safely.

    Trade counts come from orders table (authoritative, synced with Alpaca).
    Realized P/L is derived from equity change (ending - starting).
    Fees are estimated at 0.25% of notional (Alpaca taker rate).

    Args:
        date_local: Date string in YYYY-MM-DD format. Defaults to today (Mac mini local time).

    Returns:
        Dict with date, trades count, realized_pnl, notional, and optional 'skipped' flag
    """
    if date_local is None:
        date_local = datetime.now().strftime('%Y-%m-%d')  # Mac mini local time

    with get_db_connection() as conn:
        cursor = conn.cursor()

        # Get equity stats for local day from equity_snapshots
        # Use substr() to match YYYY-MM-DD prefix (timestamps are stored in local ISO format)
        cursor.execute("""
            SELECT
                MIN(equity) as low_equity,
                MAX(equity) as high_equity,
                (SELECT equity FROM equity_snapshots
                 WHERE substr(timestamp, 1, 10) = ? ORDER BY timestamp ASC LIMIT 1) as starting_equity,
                (SELECT equity FROM equity_snapshots
                 WHERE substr(timestamp, 1, 10) = ? ORDER BY timestamp DESC LIMIT 1) as ending_equity
            FROM equity_snapshots
            WHERE substr(timestamp, 1, 10) = ?
        """, (date_local, date_local, date_local))
        equity_row = cursor.fetchone()

        # Get trade stats for local day from orders table (filled orders)
        # Note: orders table is authoritative - it syncs with Alpaca
        # We use filled_at for date grouping; fees estimated at 0.25% taker rate
        cursor.execute("""
            SELECT
                COUNT(*) as total_trades,
                SUM(CASE WHEN side = 'buy' THEN 1 ELSE 0 END) as total_buys,
                SUM(CASE WHEN side = 'sell' THEN 1 ELSE 0 END) as total_sells,
                COALESCE(SUM(filled_qty * filled_avg_price), 0) as notional,
                COALESCE(SUM(filled_qty * filled_avg_price * 0.0025), 0) as fees_estimated
            FROM orders
            WHERE status = 'filled' AND substr(COALESCE(filled_at, submitted_at), 1, 10) = ?
        """, (date_local,))
        trades_row = cursor.fetchone()

        # Guard: don't insert if no equity data for this date (would create NULL row)
        if equity_row['low_equity'] is None and equity_row['high_equity'] is None:
            logger.debug(f"No equity snapshots for {date_local}, skipping daily summary")
            return {'date': date_local, 'trades': 0, 'realized_pnl': 0, 'skipped': True}

        # Upsert daily_summary (requires unique index on date!)
        # Calculate realized P/L from equity change (ending - starting)
        # This is more accurate than summing individual trade profits
        realized_pnl = 0.0
        if equity_row['ending_equity'] and equity_row['starting_equity']:
            realized_pnl = equity_row['ending_equity'] - equity_row['starting_equity']

        cursor.execute("""
            INSERT INTO daily_summary
            (date, starting_equity, ending_equity, high_equity, low_equity,
             total_trades, total_buys, total_sells, realized_pnl, fees, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
            ON CONFLICT(date) DO UPDATE SET
                ending_equity = excluded.ending_equity,
                high_equity = MAX(daily_summary.high_equity, excluded.high_equity),
                low_equity = MIN(daily_summary.low_equity, excluded.low_equity),
                total_trades = excluded.total_trades,
                total_buys = excluded.total_buys,
                total_sells = excluded.total_sells,
                realized_pnl = excluded.realized_pnl,
                fees = excluded.fees,
                updated_at = datetime('now')
        """, (date_local,
              equity_row['starting_equity'], equity_row['ending_equity'],
              equity_row['high_equity'], equity_row['low_equity'],
              trades_row['total_trades'] or 0, trades_row['total_buys'] or 0,
              trades_row['total_sells'] or 0, realized_pnl,
              trades_row['fees_estimated'] or 0))

        conn.commit()
        logger.debug(f"Recomputed daily summary for {date_local}: {trades_row['total_trades']} trades, ${realized_pnl:.2f} P/L")
        return {
            'date': date_local,
            'trades': trades_row['total_trades'] or 0,
            'realized_pnl': realized_pnl,
            'notional': trades_row['notional'] or 0
        }


# ============ BACKFILL FUNCTIONS ============

def backfill_orders_from_alpaca(alpaca_orders: List[Dict]) -> int:
    """Backfill orders from Alpaca API response."""
    count = 0
    with get_db_connection() as conn:
        cursor = conn.cursor()

        for order in alpaca_orders:
            try:
                # Normalize symbol
                symbol = order.get('symbol', '')
                if '/' not in symbol:
                    symbol = symbol.replace('USD', '/USD')

                cursor.execute("""
                    INSERT OR IGNORE INTO orders
                    (order_id, client_order_id, symbol, side, order_type, qty,
                     filled_qty, filled_avg_price, status, submitted_at, filled_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    order.get('id'),
                    order.get('client_order_id'),
                    symbol,
                    str(order.get('side', '')).lower().replace('orderside.', ''),
                    str(order.get('type', '')),
                    order.get('qty'),
                    order.get('filled_qty'),
                    order.get('filled_avg_price'),
                    str(order.get('status', '')),
                    order.get('submitted_at'),
                    order.get('filled_at')
                ))
                if cursor.rowcount > 0:
                    count += 1
            except Exception as e:
                logger.warning(f"Failed to backfill order {order.get('id')}: {e}")

        conn.commit()

    logger.info(f"Backfilled {count} orders from Alpaca")
    return count


def backfill_equity_from_alpaca(equity_history: Dict) -> int:
    """Backfill equity snapshots from Alpaca portfolio history."""
    count = 0

    timestamps = equity_history.get('timestamps', [])
    equity_values = equity_history.get('equity', [])
    pnl_values = equity_history.get('profit_loss', [])
    pnl_pct_values = equity_history.get('profit_loss_pct', [])

    with get_db_connection() as conn:
        cursor = conn.cursor()

        for i, (ts, eq) in enumerate(zip(timestamps, equity_values)):
            if eq is None:
                continue

            try:
                # Parse timestamp
                if isinstance(ts, str):
                    timestamp = ts
                else:
                    timestamp = ts

                pnl = pnl_values[i] if i < len(pnl_values) else None
                pnl_pct = pnl_pct_values[i] if i < len(pnl_pct_values) else None

                cursor.execute("""
                    INSERT OR IGNORE INTO equity_snapshots
                    (timestamp, equity, daily_pnl, daily_pnl_pct, source)
                    VALUES (?, ?, ?, ?, 'backfill')
                """, (timestamp, eq, pnl, pnl_pct))

                if cursor.rowcount > 0:
                    count += 1
            except Exception as e:
                logger.warning(f"Failed to backfill equity snapshot: {e}")

        conn.commit()

    logger.info(f"Backfilled {count} equity snapshots from Alpaca")
    return count


# ============ RECONCILIATION FUNCTIONS ============

def reconcile_with_alpaca(alpaca_orders: List[Dict]) -> Dict:
    """
    Compare database orders with Alpaca orders to find discrepancies.

    Returns:
        Dict with reconciliation results:
        - matched: orders in both DB and Alpaca
        - missing_in_db: orders in Alpaca but not in DB
        - extra_in_db: orders in DB but not in Alpaca (shouldn't happen)
        - mismatched: orders with different data
    """
    results = {
        'matched': 0,
        'missing_in_db': [],
        'extra_in_db': [],
        'mismatched': [],
        'total_alpaca': len(alpaca_orders),
        'total_db': 0,
        'synced': True,
        'last_check': datetime.now().isoformat()
    }

    with get_db_connection() as conn:
        cursor = conn.cursor()

        # Get all DB order IDs
        cursor.execute("SELECT order_id, symbol, side, filled_qty, filled_avg_price, status FROM orders")
        db_orders = {row['order_id']: dict(row) for row in cursor.fetchall()}
        results['total_db'] = len(db_orders)

        # Compare each Alpaca order
        for alpaca_order in alpaca_orders:
            order_id = alpaca_order.get('id')

            if order_id in db_orders:
                # Check if data matches
                db_order = db_orders[order_id]

                # Compare key fields
                alpaca_qty = float(alpaca_order.get('filled_qty') or 0)
                db_qty = float(db_order.get('filled_qty') or 0)
                alpaca_price = float(alpaca_order.get('filled_avg_price') or 0)
                db_price = float(db_order.get('filled_avg_price') or 0)

                # Allow small floating point differences
                if abs(alpaca_qty - db_qty) > 0.0001 or abs(alpaca_price - db_price) > 0.01:
                    results['mismatched'].append({
                        'order_id': order_id,
                        'alpaca': {'qty': alpaca_qty, 'price': alpaca_price},
                        'db': {'qty': db_qty, 'price': db_price}
                    })
                    results['synced'] = False
                else:
                    results['matched'] += 1

                # Remove from db_orders to track extras
                del db_orders[order_id]
            else:
                # Missing in DB
                results['missing_in_db'].append({
                    'order_id': order_id,
                    'symbol': alpaca_order.get('symbol'),
                    'side': str(alpaca_order.get('side', '')),
                    'qty': alpaca_order.get('filled_qty'),
                    'price': alpaca_order.get('filled_avg_price'),
                    'filled_at': alpaca_order.get('filled_at')
                })
                results['synced'] = False

        # Any remaining db_orders are extra (in DB but not in Alpaca)
        for order_id, order in db_orders.items():
            results['extra_in_db'].append({
                'order_id': order_id,
                'symbol': order.get('symbol'),
                'side': order.get('side')
            })

        # Extra in DB doesn't necessarily mean out of sync (could be older than Alpaca's window)
        # Only flag as not synced if there are missing or mismatched

    return results


def sync_missing_orders(missing_orders: List[Dict]) -> int:
    """
    Add missing orders to the database.

    Args:
        missing_orders: List of orders from reconciliation that are missing in DB

    Returns:
        Number of orders added
    """
    count = 0

    with get_db_connection() as conn:
        cursor = conn.cursor()

        for order in missing_orders:
            try:
                # Normalize symbol
                symbol = order.get('symbol', '')
                if '/' not in symbol:
                    symbol = symbol.replace('USD', '/USD')

                cursor.execute("""
                    INSERT OR IGNORE INTO orders
                    (order_id, symbol, side, filled_qty, filled_avg_price, status, filled_at)
                    VALUES (?, ?, ?, ?, ?, 'filled', ?)
                """, (
                    order.get('order_id'),
                    symbol,
                    str(order.get('side', '')).lower().replace('orderside.', ''),
                    order.get('qty'),
                    order.get('price'),
                    order.get('filled_at')
                ))

                if cursor.rowcount > 0:
                    count += 1
                    logger.info(f"Synced missing order: {order.get('order_id')}")

            except Exception as e:
                logger.warning(f"Failed to sync order {order.get('order_id')}: {e}")

        conn.commit()

    return count


def get_reconciliation_status() -> Dict:
    """Get the last reconciliation status."""
    # This would be stored in a metadata table, but for now return basic stats
    with get_db_connection() as conn:
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) as count FROM orders WHERE status LIKE '%filled%'")
        filled_orders = cursor.fetchone()['count']

        cursor.execute("SELECT MAX(filled_at) as last_order FROM orders")
        last_order = cursor.fetchone()['last_order']

        return {
            'db_filled_orders': filled_orders,
            'last_order_time': last_order,
            'status': 'ok'
        }


# ============ GRID STATE PERSISTENCE ============

def save_grid_state(state_dict: Dict) -> int:
    """
    Save full grid strategy state to database.

    Args:
        state_dict: Output from GridTradingStrategy serialization containing:
            - grids: Dict[symbol, GridState.to_dict()]
            - pending_orders: Dict[order_id, PendingOrder.to_dict()]
            - applied_order_ids: Dict[order_id, applied_at]
            - open_limit_orders: Dict[key, OpenLimitOrder.to_dict()]

    Returns:
        Snapshot ID
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()

        # Store as single row with full JSON
        state_json = json.dumps(state_dict)
        timestamp = datetime.now().isoformat()

        cursor.execute("""
            INSERT INTO grid_snapshots
            (timestamp, symbol, state_json)
            VALUES (?, 'ALL', ?)
        """, (timestamp, state_json))

        conn.commit()
        snapshot_id = cursor.lastrowid
        logger.debug(f"Saved grid state snapshot #{snapshot_id}")
        return snapshot_id


def get_latest_grid_state() -> Optional[Dict]:
    """
    Load most recent grid state from database.

    Returns:
        State dict or None if no state exists
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()

        cursor.execute("""
            SELECT state_json, timestamp
            FROM grid_snapshots
            WHERE symbol = 'ALL' AND state_json IS NOT NULL
            ORDER BY timestamp DESC
            LIMIT 1
        """)

        row = cursor.fetchone()
        if row and row['state_json']:
            logger.info(f"Loaded grid state from {row['timestamp']}")
            return json.loads(row['state_json'])

        return None


def cleanup_old_grid_snapshots(keep_days: int = 7) -> int:
    """Delete grid snapshots older than N days."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cutoff = (datetime.now() - timedelta(days=keep_days)).isoformat()

        cursor.execute("""
            DELETE FROM grid_snapshots
            WHERE timestamp < ? AND symbol = 'ALL'
        """, (cutoff,))

        conn.commit()
        deleted = cursor.rowcount
        if deleted > 0:
            logger.info(f"Cleaned up {deleted} old grid snapshots")
        return deleted


# ============ UTILITY FUNCTIONS ============

def get_database_stats() -> Dict:
    """Get database statistics."""
    with get_db_connection() as conn:
        cursor = conn.cursor()

        stats = {}

        for table in ['trades', 'equity_snapshots', 'orders', 'daily_summary', 'grid_snapshots']:
            cursor.execute(f"SELECT COUNT(*) as count FROM {table}")
            stats[table] = cursor.fetchone()['count']

        # Get date ranges
        cursor.execute("SELECT MIN(timestamp) as oldest, MAX(timestamp) as newest FROM trades")
        row = cursor.fetchone()
        stats['trades_range'] = {'oldest': row['oldest'], 'newest': row['newest']}

        cursor.execute("SELECT MIN(timestamp) as oldest, MAX(timestamp) as newest FROM equity_snapshots")
        row = cursor.fetchone()
        stats['equity_range'] = {'oldest': row['oldest'], 'newest': row['newest']}

        # Database file size
        if os.path.exists(DB_PATH):
            stats['db_size_mb'] = round(os.path.getsize(DB_PATH) / (1024 * 1024), 2)
        else:
            stats['db_size_mb'] = 0

        return stats


# ============ NOTIFICATION FUNCTIONS ============

def record_sms(
    message_type: str,
    recipient: str,
    body_preview: str,
    twilio_sid: str = None,
    status: str = 'sent',
    retry_count: int = 0,
    error_message: str = None
) -> int:
    """Record an SMS in the history table."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO sms_history
            (timestamp, message_type, recipient, body_preview, twilio_sid, status, retry_count, error_message)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (datetime.now().isoformat(), message_type, recipient, body_preview[:100] if body_preview else None,
              twilio_sid, status, retry_count, error_message))
        conn.commit()
        return cursor.lastrowid


def is_trade_notified(order_id: str) -> bool:
    """Check if a trade has already been notified."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT 1 FROM notified_trade_ids WHERE order_id = ?", (order_id,))
        return cursor.fetchone() is not None


def mark_trade_notified(order_id: str) -> int:
    """Mark a trade as notified to prevent duplicates."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        try:
            cursor.execute("""
                INSERT OR IGNORE INTO notified_trade_ids (order_id, notified_at)
                VALUES (?, ?)
            """, (order_id, datetime.now().isoformat()))
            conn.commit()
            return cursor.lastrowid
        except Exception:
            return -1


def get_notified_trade_ids(limit: int = 1000) -> set:
    """Get all notified trade IDs (for migration from in-memory set)."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT order_id FROM notified_trade_ids
            ORDER BY notified_at DESC LIMIT ?
        """, (limit,))
        return {row['order_id'] for row in cursor.fetchall()}


def queue_failed_sms(
    body: str,
    message_type: str = None,
    priority: int = 0,
    error_message: str = None
) -> int:
    """Queue a failed SMS for later retry."""
    next_retry = (datetime.now() + timedelta(minutes=5)).isoformat()
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO sms_queue
            (body, message_type, priority, attempts, last_attempt, next_retry, error_message)
            VALUES (?, ?, ?, 1, ?, ?, ?)
        """, (body, message_type, priority, datetime.now().isoformat(), next_retry, error_message))
        conn.commit()
        return cursor.lastrowid


def get_queued_sms(limit: int = 10) -> List[Dict]:
    """Get SMS messages due for retry."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        now = datetime.now().isoformat()
        cursor.execute("""
            SELECT id, body, message_type, priority, attempts, max_attempts
            FROM sms_queue
            WHERE next_retry <= ? AND attempts < max_attempts
            ORDER BY priority DESC, created_at ASC
            LIMIT ?
        """, (now, limit))
        return [dict(row) for row in cursor.fetchall()]


def update_queued_sms(queue_id: int, success: bool, error_message: str = None):
    """Update a queued SMS after retry attempt."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        if success:
            # Remove from queue on success
            cursor.execute("DELETE FROM sms_queue WHERE id = ?", (queue_id,))
        else:
            # Increment attempts and set next retry with exponential backoff
            cursor.execute("SELECT attempts FROM sms_queue WHERE id = ?", (queue_id,))
            row = cursor.fetchone()
            if row:
                attempts = row['attempts'] + 1
                backoff_minutes = min(5 * (2 ** attempts), 60)  # Max 60 min backoff
                next_retry = (datetime.now() + timedelta(minutes=backoff_minutes)).isoformat()
                cursor.execute("""
                    UPDATE sms_queue
                    SET attempts = ?, last_attempt = ?, next_retry = ?, error_message = ?
                    WHERE id = ?
                """, (attempts, datetime.now().isoformat(), next_retry, error_message, queue_id))
        conn.commit()


def update_notifier_heartbeat(
    pid: int = None,
    status: str = 'running',
    overlay_mode: str = None,
    api_failures: int = None
):
    """Update the notifier status/heartbeat (upsert single row)."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        now = datetime.now().isoformat()
        today = datetime.now().strftime('%Y-%m-%d')

        # Check if row exists
        cursor.execute("SELECT sms_today, sms_today_date FROM notifier_status WHERE id = 1")
        row = cursor.fetchone()

        if row:
            # Reset daily counter if new day
            sms_today = row['sms_today'] if row['sms_today_date'] == today else 0

            updates = ["last_heartbeat = ?", "status = ?"]
            params = [now, status]

            if pid is not None:
                updates.append("pid = ?")
                params.append(pid)
            if overlay_mode is not None:
                updates.append("last_overlay_mode = ?")
                params.append(overlay_mode)
            if api_failures is not None:
                updates.append("api_failures = ?")
                params.append(api_failures)

            params.append(1)  # WHERE id = 1
            cursor.execute(f"UPDATE notifier_status SET {', '.join(updates)} WHERE id = ?", params)
        else:
            # Insert initial row
            cursor.execute("""
                INSERT INTO notifier_status
                (id, pid, started_at, last_heartbeat, sms_today, sms_today_date, status)
                VALUES (1, ?, ?, ?, 0, ?, ?)
            """, (pid, now, now, today, status))

        conn.commit()


def get_notifier_status() -> Optional[Dict]:
    """Get the current notifier status."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM notifier_status WHERE id = 1")
        row = cursor.fetchone()
        return dict(row) if row else None


def increment_sms_count():
    """Increment the daily SMS counter."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        today = datetime.now().strftime('%Y-%m-%d')
        now = datetime.now().isoformat()

        cursor.execute("SELECT sms_today_date FROM notifier_status WHERE id = 1")
        row = cursor.fetchone()

        if row:
            if row['sms_today_date'] == today:
                cursor.execute("""
                    UPDATE notifier_status
                    SET sms_today = sms_today + 1, last_sms_at = ?
                    WHERE id = 1
                """, (now,))
            else:
                # New day, reset counter
                cursor.execute("""
                    UPDATE notifier_status
                    SET sms_today = 1, sms_today_date = ?, last_sms_at = ?
                    WHERE id = 1
                """, (today, now))
        conn.commit()


def get_sms_history(limit: int = 20) -> List[Dict]:
    """Get recent SMS history."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT * FROM sms_history
            ORDER BY timestamp DESC
            LIMIT ?
        """, (limit,))
        return [dict(row) for row in cursor.fetchall()]


def cleanup_old_sms_records(keep_days: int = 30) -> int:
    """Delete old SMS history and resolved queue items."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cutoff = (datetime.now() - timedelta(days=keep_days)).isoformat()

        # Clean old history
        cursor.execute("DELETE FROM sms_history WHERE timestamp < ?", (cutoff,))
        deleted_history = cursor.rowcount

        # Clean old notified trade IDs (keep last 30 days)
        cursor.execute("DELETE FROM notified_trade_ids WHERE notified_at < ?", (cutoff,))
        deleted_trades = cursor.rowcount

        # Clean old failed queue items that exceeded max attempts
        cursor.execute("DELETE FROM sms_queue WHERE attempts >= max_attempts AND created_at < ?", (cutoff,))
        deleted_queue = cursor.rowcount

        conn.commit()
        total = deleted_history + deleted_trades + deleted_queue
        if total > 0:
            logger.info(f"Cleaned up {total} old notification records")
        return total


# ============================================================================
# Bot Status (Heartbeat for Watchdog)
# ============================================================================

def update_bot_heartbeat(
    pid: Optional[int] = None,
    overlay_mode: Optional[str] = None,
    active_symbols: Optional[int] = None,
    total_trades: Optional[int] = None,
    status: str = "running"
):
    """
    Update the bot heartbeat in the database.
    Called periodically by the API server to prove it's alive.
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()
        now = datetime.now().isoformat()

        # Check if row exists
        cursor.execute("SELECT id FROM bot_status WHERE id = 1")
        row = cursor.fetchone()

        if row:
            updates = ["last_heartbeat = ?", "status = ?"]
            params = [now, status]

            if pid is not None:
                updates.append("pid = ?")
                params.append(pid)
            if overlay_mode is not None:
                updates.append("overlay_mode = ?")
                params.append(overlay_mode)
            if active_symbols is not None:
                updates.append("active_symbols = ?")
                params.append(active_symbols)
            if total_trades is not None:
                updates.append("total_trades = ?")
                params.append(total_trades)

            params.append(1)  # WHERE id = 1
            cursor.execute(f"UPDATE bot_status SET {', '.join(updates)} WHERE id = ?", params)
        else:
            # Insert initial row
            cursor.execute("""
                INSERT INTO bot_status
                (id, pid, started_at, last_heartbeat, overlay_mode, active_symbols, total_trades, status)
                VALUES (1, ?, ?, ?, ?, ?, ?, ?)
            """, (pid, now, now, overlay_mode, active_symbols or 0, total_trades or 0, status))

        conn.commit()


def get_bot_status() -> Optional[Dict]:
    """Get the current bot status."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM bot_status WHERE id = 1")
        row = cursor.fetchone()
        return dict(row) if row else None


# Initialize database on import
init_database()
