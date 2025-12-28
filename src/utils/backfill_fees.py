"""
Backfill Fee Calculation Script

This script fetches all filled orders from Alpaca since Dec 1, 2025 and calculates
tier-correct fees for each trade. It's idempotent and safe to re-run.

Usage:
    python -m src.utils.backfill_fees

Steps:
    1. Fetch filled orders from Alpaca (bulk pagination)
    2. Upsert into orders table
    3. Normalize orders -> trades (create trade records if missing)
    4. Recompute fees for all trades chronologically using tier engine
"""

import logging
import time
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from zoneinfo import ZoneInfo

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

UTC = ZoneInfo("UTC")
ET = ZoneInfo("America/New_York")

# Symbols we trade
TRADED_SYMBOLS = ["BTCUSD", "SOLUSD", "LTCUSD", "AVAXUSD", "BTC/USD", "SOL/USD", "LTC/USD", "AVAX/USD"]


def normalize_symbol(symbol: str) -> str:
    """Normalize symbol to format with slash (e.g., BTC/USD)."""
    if '/' in symbol:
        return symbol
    # Insert slash before USD
    if symbol.endswith('USD'):
        return symbol[:-3] + '/' + 'USD'
    return symbol


def fetch_all_filled_orders(start_date: datetime, end_date: datetime) -> List[Dict]:
    """
    Fetch all filled crypto orders from Alpaca using bulk pagination.

    Args:
        start_date: Start of date range
        end_date: End of date range

    Returns:
        List of order dicts
    """
    from src.execution.alpaca_client import AlpacaClient
    from config_ultra import UltraConfig

    config = UltraConfig()
    client = AlpacaClient(config)
    all_orders = []

    logger.info(f"Fetching filled orders from {start_date.date()} to {end_date.date()}...")

    # Calculate total days for the range
    total_days = (end_date - start_date).days + 1

    try:
        # Fetch all orders for the period using get_order_history
        orders = client.get_order_history(days=total_days, status='closed')

        # Filter to our traded symbols and filled orders within date range
        for order in orders:
            symbol = order.get('symbol', '')
            status = str(order.get('status', '')).lower()
            filled_at = order.get('filled_at')

            # Check if within date range
            if filled_at:
                try:
                    if isinstance(filled_at, str):
                        fill_ts = datetime.fromisoformat(filled_at.replace('Z', '+00:00'))
                    else:
                        fill_ts = filled_at
                    if fill_ts.tzinfo is None:
                        fill_ts = fill_ts.replace(tzinfo=UTC)

                    if fill_ts < start_date or fill_ts > end_date:
                        continue
                except:
                    pass

            if symbol in TRADED_SYMBOLS and 'filled' in status:
                all_orders.append(order)

        logger.info(f"  Fetched {len(orders)} total orders, {len(all_orders)} matched our symbols")

    except Exception as e:
        logger.warning(f"  Error fetching orders: {e}")
        import traceback
        traceback.print_exc()

    logger.info(f"Total filled orders fetched: {len(all_orders)}")
    return all_orders


def upsert_order(order: Dict) -> bool:
    """
    Upsert an order into the orders table.

    Returns:
        True if inserted, False if already exists
    """
    from src.database import db as database

    order_id = order.get('id')
    if not order_id:
        return False

    try:
        with database.get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR IGNORE INTO orders (
                    order_id, client_order_id, symbol, side, order_type,
                    qty, filled_qty, filled_avg_price, status, submitted_at, filled_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                order_id,
                order.get('client_order_id'),
                normalize_symbol(order.get('symbol', '')),
                order.get('side'),
                order.get('type'),
                float(order.get('qty') or 0),
                float(order.get('filled_qty') or 0),
                float(order.get('filled_avg_price') or 0),
                str(order.get('status', '')),
                order.get('submitted_at'),
                order.get('filled_at'),
            ))
            conn.commit()
            return cursor.rowcount > 0
    except Exception as e:
        logger.warning(f"Error upserting order {order_id[:8]}: {e}")
        return False


def upsert_trade_from_order(order: Dict) -> Optional[int]:
    """
    Create a trade record from an order if it doesn't exist.

    Returns:
        Trade ID if created, None if already exists or error
    """
    from src.database import db as database

    order_id = order.get('id')
    filled_qty = float(order.get('filled_qty') or 0)
    filled_price = float(order.get('filled_avg_price') or 0)

    if not order_id or filled_qty <= 0 or filled_price <= 0:
        return None

    symbol = normalize_symbol(order.get('symbol', ''))
    side = str(order.get('side', '')).lower()
    filled_at = order.get('filled_at')

    # Parse timestamp
    if filled_at:
        if isinstance(filled_at, str):
            try:
                timestamp = datetime.fromisoformat(filled_at.replace('Z', '+00:00'))
            except:
                timestamp = datetime.now(UTC)
        else:
            timestamp = filled_at
    else:
        timestamp = datetime.now(UTC)

    try:
        trade_id = database.record_trade(
            symbol=symbol,
            side=side,
            quantity=filled_qty,
            price=filled_price,
            order_id=order_id,
            profit=0,  # Will be calculated during fee recomputation
            fees=0,    # Will be calculated during fee recomputation
            source='backfill',
            notes='backfill_from_alpaca',
            timestamp=timestamp,
        )
        return trade_id if trade_id > 0 else None
    except Exception as e:
        logger.debug(f"Trade already exists or error for order {order_id[:8]}: {e}")
        return None


def get_trades_since(start_date: datetime) -> List[Dict]:
    """Get all trades since a date, sorted chronologically."""
    from src.database import db as database

    with database.get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT * FROM trades
            WHERE timestamp >= ?
            ORDER BY timestamp ASC
        """, (start_date.isoformat(),))
        return [dict(row) for row in cursor.fetchall()]


def get_order_by_id(order_id: str) -> Optional[Dict]:
    """Get an order by its ID."""
    from src.database import db as database

    if not order_id:
        return None

    with database.get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM orders WHERE order_id = ?", (order_id,))
        row = cursor.fetchone()
        return dict(row) if row else None


def recompute_trade_fees(trade: Dict) -> Dict:
    """
    Recompute fees for a trade using the tier engine.

    Returns:
        Dict with fee calculation results
    """
    from src.utils.crypto_fee_tiers import (
        get_fee_tier,
        get_fee_day_bucket,
        determine_fee_type,
    )
    from src.database import db as database

    trade_id = trade['id']
    timestamp_str = trade['timestamp']
    total_value = float(trade.get('total_value') or 0)
    order_id = trade.get('order_id')

    # Parse timestamp
    try:
        if 'T' in timestamp_str:
            timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
        else:
            timestamp = datetime.fromisoformat(timestamp_str)
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=UTC)
    except:
        timestamp = datetime.now(UTC)

    # Get rolling 30-day volume at this point in time
    rolling_30d_volume = database.get_rolling_30d_volume(timestamp)

    # Get tier and rates
    tier = get_fee_tier(rolling_30d_volume)
    fee_day_bucket = get_fee_day_bucket(timestamp)

    # Determine fee type from order metadata
    order = get_order_by_id(order_id) if order_id else None
    order_type = order.get('order_type', 'limit') if order else 'limit'
    fee_type = determine_fee_type(order_type)

    # Calculate fees
    if fee_type in ('maker', 'maker_assumed'):
        fee_rate = tier['maker']
    else:
        fee_rate = tier['taker']

    fees = total_value * fee_rate
    fees_conservative = total_value * tier['taker']

    return {
        'trade_id': trade_id,
        'fees': round(fees, 6),
        'fee_conservative': round(fees_conservative, 6),
        'fee_rate': fee_rate,
        'fee_type': fee_type,
        'fee_tier': tier['name'],
        'rolling_30d_volume': rolling_30d_volume,
        'fee_day_bucket': fee_day_bucket,
    }


def update_trade_with_fees(fee_result: Dict) -> bool:
    """Update a trade record with calculated fees."""
    from src.database import db as database

    return database.update_trade_fees(
        trade_id=fee_result['trade_id'],
        fees=fee_result['fees'],
        fee_conservative=fee_result['fee_conservative'],
        fee_rate=fee_result['fee_rate'],
        fee_type=fee_result['fee_type'],
        fee_tier=fee_result['fee_tier'],
        rolling_30d_volume=fee_result['rolling_30d_volume'],
        fee_day_bucket=fee_result['fee_day_bucket'],
    )


def run_backfill(start_date: datetime = None, end_date: datetime = None, dry_run: bool = False):
    """
    Run the full backfill process.

    Args:
        start_date: Start date (default: Dec 1, 2025)
        end_date: End date (default: now)
        dry_run: If True, don't write to database
    """
    from src.database import db as database

    # Initialize database (runs migrations)
    database.init_database()

    if start_date is None:
        start_date = datetime(2025, 12, 1, tzinfo=UTC)
    if end_date is None:
        end_date = datetime.now(UTC)

    logger.info("=" * 60)
    logger.info("BLUEBIRD Fee Backfill")
    logger.info("=" * 60)
    logger.info(f"Date range: {start_date.date()} to {end_date.date()}")
    logger.info(f"Dry run: {dry_run}")
    logger.info("")

    # Step 1: Fetch orders from Alpaca
    logger.info("Step 1: Fetching filled orders from Alpaca...")
    orders = fetch_all_filled_orders(start_date, end_date)
    logger.info(f"  Fetched {len(orders)} filled orders")
    logger.info("")

    # Step 2: Upsert orders into database
    logger.info("Step 2: Upserting orders into database...")
    orders_inserted = 0
    for order in orders:
        if not dry_run:
            if upsert_order(order):
                orders_inserted += 1
    logger.info(f"  Inserted {orders_inserted} new orders (existing: {len(orders) - orders_inserted})")
    logger.info("")

    # Step 3: Create trade records from orders
    logger.info("Step 3: Creating trade records from orders...")
    trades_created = 0
    for order in orders:
        if not dry_run:
            if upsert_trade_from_order(order):
                trades_created += 1
    logger.info(f"  Created {trades_created} new trades (existing: {len(orders) - trades_created})")
    logger.info("")

    # Step 4: Recompute fees for all trades
    logger.info("Step 4: Recomputing fees for all trades...")
    trades = get_trades_since(start_date)
    logger.info(f"  Processing {len(trades)} trades chronologically...")

    fees_updated = 0
    total_fees_expected = 0.0
    total_fees_conservative = 0.0
    uncertain_count = 0

    for i, trade in enumerate(trades):
        fee_result = recompute_trade_fees(trade)

        if not dry_run:
            if update_trade_with_fees(fee_result):
                fees_updated += 1

        total_fees_expected += fee_result['fees']
        total_fees_conservative += fee_result['fee_conservative']
        if fee_result['fee_type'] == 'maker_assumed':
            uncertain_count += 1

        # Progress logging every 100 trades
        if (i + 1) % 100 == 0:
            pct = ((i + 1) / len(trades)) * 100
            logger.info(f"    Progress: {i + 1}/{len(trades)} ({pct:.1f}%)")

    logger.info(f"  Updated fees for {fees_updated} trades")
    logger.info("")

    # Summary
    logger.info("=" * 60)
    logger.info("BACKFILL COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Orders fetched:           {len(orders)}")
    logger.info(f"Orders inserted:          {orders_inserted}")
    logger.info(f"Trades created:           {trades_created}")
    logger.info(f"Fees updated:             {fees_updated}")
    logger.info(f"Total fees (expected):    ${total_fees_expected:,.2f}")
    logger.info(f"Total fees (conservative): ${total_fees_conservative:,.2f}")
    logger.info(f"Uncertain classifications: {uncertain_count}")
    logger.info("")

    return {
        'orders_fetched': len(orders),
        'orders_inserted': orders_inserted,
        'trades_created': trades_created,
        'fees_updated': fees_updated,
        'total_fees_expected': total_fees_expected,
        'total_fees_conservative': total_fees_conservative,
        'uncertain_count': uncertain_count,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Backfill fee calculations for BLUEBIRD trades')
    parser.add_argument('--dry-run', action='store_true', help='Simulate without writing to database')
    parser.add_argument('--start-date', type=str, help='Start date (YYYY-MM-DD), default: 2025-12-01')
    parser.add_argument('--end-date', type=str, help='End date (YYYY-MM-DD), default: today')

    args = parser.parse_args()

    start = None
    end = None

    if args.start_date:
        start = datetime.strptime(args.start_date, '%Y-%m-%d').replace(tzinfo=UTC)
    if args.end_date:
        end = datetime.strptime(args.end_date, '%Y-%m-%d').replace(tzinfo=UTC)

    run_backfill(start_date=start, end_date=end, dry_run=args.dry_run)
