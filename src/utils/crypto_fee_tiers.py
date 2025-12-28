"""
Alpaca Crypto Fee Tier Engine

Implements volume-based fee tier calculation per Alpaca's fee schedule.
Tier recalculation happens at 3am ET daily (per AlpacaCryptoLLCFeeDisclosure.pdf).

Usage:
    from src.utils.crypto_fee_tiers import get_fee_tier, get_fee_rates_for_fill

    # Get current tier based on 30-day volume
    tier = get_fee_tier(45000.00)
    # {'tier': 1, 'name': 'Tier 1', 'maker': 0.0015, 'taker': 0.0025}

    # Get rates for a specific fill timestamp
    rates = get_fee_rates_for_fill(fill_timestamp, rolling_30d_volume)
"""

from datetime import datetime, timedelta
from typing import Optional
from zoneinfo import ZoneInfo
import logging

logger = logging.getLogger(__name__)

# Timezone for fee day bucketing (3am ET boundary)
ET = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")

# Alpaca Crypto Fee Tiers (from docs.alpaca.markets/docs/crypto-fees)
# Format: (min_volume, max_volume, maker_rate, taker_rate, tier_name)
FEE_TIERS = [
    (0,           100_000,      0.0015, 0.0025, "Tier 1"),
    (100_000,     500_000,      0.0012, 0.0022, "Tier 2"),
    (500_000,     1_000_000,    0.0010, 0.0020, "Tier 3"),
    (1_000_000,   10_000_000,   0.0008, 0.0018, "Tier 4"),
    (10_000_000,  25_000_000,   0.0005, 0.0015, "Tier 5"),
    (25_000_000,  50_000_000,   0.0002, 0.0013, "Tier 6"),
    (50_000_000,  100_000_000,  0.0002, 0.0012, "Tier 7"),
    (100_000_000, float('inf'), 0.0000, 0.0010, "Tier 8"),
]


def get_fee_tier(volume_30d_usd: float) -> dict:
    """
    Get fee tier info based on 30-day trading volume.

    Args:
        volume_30d_usd: Rolling 30-day crypto trading volume in USD

    Returns:
        dict with: tier (int), name (str), maker (float), taker (float),
                   min_volume (float), max_volume (float)
    """
    volume = max(0, volume_30d_usd)  # Ensure non-negative

    for i, (min_vol, max_vol, maker, taker, name) in enumerate(FEE_TIERS, start=1):
        if min_vol <= volume < max_vol:
            return {
                'tier': i,
                'name': name,
                'maker': maker,
                'taker': taker,
                'min_volume': min_vol,
                'max_volume': max_vol if max_vol != float('inf') else None,
            }

    # Fallback to Tier 1 (should never reach here)
    return {
        'tier': 1,
        'name': 'Tier 1',
        'maker': 0.0015,
        'taker': 0.0025,
        'min_volume': 0,
        'max_volume': 100_000,
    }


def get_next_tier_info(volume_30d_usd: float) -> Optional[dict]:
    """
    Get info about the next tier (for progression display).

    Returns:
        dict with: next_tier_at (float), progress_pct (float), next_tier_name (str)
        or None if already at highest tier
    """
    volume = max(0, volume_30d_usd)

    for i, (min_vol, max_vol, _, _, name) in enumerate(FEE_TIERS, start=1):
        if min_vol <= volume < max_vol:
            if max_vol == float('inf'):
                return None  # Already at highest tier

            progress = ((volume - min_vol) / (max_vol - min_vol)) * 100
            next_tier_name = FEE_TIERS[i][4] if i < len(FEE_TIERS) else None

            return {
                'next_tier_at': max_vol,
                'progress_pct': round(progress, 1),
                'next_tier_name': next_tier_name,
                'volume_needed': max_vol - volume,
            }

    return None


def get_fee_day_bucket(ts: datetime) -> str:
    """
    Convert timestamp to fee-day bucket using 3am ET boundary.

    Alpaca crypto fees use a "fee day" that runs from 03:00:00 ET to 02:59:59 ET next day.
    This means trades between midnight and 3am ET belong to the previous fee day.

    Args:
        ts: Timestamp (timezone-aware or naive UTC)

    Returns:
        Fee day as 'YYYY-MM-DD' string
    """
    # Ensure timestamp is timezone-aware
    if ts.tzinfo is None:
        # Assume naive timestamps are UTC (common for DB storage)
        ts = ts.replace(tzinfo=UTC)

    # Convert to ET
    et_time = ts.astimezone(ET)

    # If before 3am ET, use previous day
    if et_time.hour < 3:
        et_time = et_time - timedelta(days=1)

    return et_time.strftime('%Y-%m-%d')


def get_fee_rates_for_fill(
    fill_timestamp: datetime,
    rolling_30d_volume: float,
    fee_type: str = 'taker'
) -> dict:
    """
    Get the applicable fee rate for a specific fill.

    Args:
        fill_timestamp: When the fill occurred
        rolling_30d_volume: 30-day volume at time of fill
        fee_type: 'maker', 'taker', or 'maker_assumed'

    Returns:
        dict with: rate (float), tier (dict), fee_day_bucket (str)
    """
    tier = get_fee_tier(rolling_30d_volume)
    fee_day = get_fee_day_bucket(fill_timestamp)

    # Determine rate based on fee type
    if fee_type == 'maker':
        rate = tier['maker']
    elif fee_type == 'maker_assumed':
        rate = tier['maker']  # Use maker for expected, caller should also compute conservative
    else:  # 'taker' or unknown
        rate = tier['taker']

    return {
        'rate': rate,
        'tier': tier,
        'tier_name': tier['name'],
        'fee_day_bucket': fee_day,
        'maker_rate': tier['maker'],
        'taker_rate': tier['taker'],
    }


def calculate_fee(
    notional_usd: float,
    fee_type: str,
    rolling_30d_volume: float,
    fill_timestamp: Optional[datetime] = None
) -> dict:
    """
    Calculate fee for a trade.

    Args:
        notional_usd: Trade value (qty * price)
        fee_type: 'maker', 'taker', or 'maker_assumed'
        rolling_30d_volume: 30-day volume at time of fill
        fill_timestamp: Optional timestamp for fee day bucket

    Returns:
        dict with: fee_expected (float), fee_conservative (float),
                   rate (float), conservative_rate (float), tier (dict)
    """
    tier = get_fee_tier(rolling_30d_volume)

    # Determine rates
    if fee_type == 'maker':
        expected_rate = tier['maker']
        conservative_rate = tier['maker']  # No uncertainty for definitive maker
    elif fee_type == 'taker':
        expected_rate = tier['taker']
        conservative_rate = tier['taker']  # No uncertainty for definitive taker
    else:  # 'maker_assumed' - limit orders where we can't prove maker/taker
        expected_rate = tier['maker']
        conservative_rate = tier['taker']  # Conservative uses taker rate

    fee_expected = notional_usd * expected_rate
    fee_conservative = notional_usd * conservative_rate

    result = {
        'fee_expected': round(fee_expected, 6),
        'fee_conservative': round(fee_conservative, 6),
        'rate': expected_rate,
        'conservative_rate': conservative_rate,
        'tier': tier,
        'tier_name': tier['name'],
        'fee_type': fee_type,
        'is_uncertain': fee_type == 'maker_assumed',
    }

    if fill_timestamp:
        result['fee_day_bucket'] = get_fee_day_bucket(fill_timestamp)

    return result


def get_rolling_30d_volume_from_trades(
    as_of_timestamp: datetime,
    db_connection=None
) -> float:
    """
    Calculate rolling 30-day volume from trades table.

    This should be called after backfill normalizes orders → trades.
    Uses trades.total_value which is already the notional (qty * price).

    Args:
        as_of_timestamp: Calculate volume as of this time
        db_connection: Optional database connection (if None, imports database module)

    Returns:
        Rolling 30-day trading volume in USD
    """
    # Ensure timestamp is timezone-aware
    if as_of_timestamp.tzinfo is None:
        as_of_timestamp = as_of_timestamp.replace(tzinfo=UTC)

    # Calculate 30-day window
    start_time = as_of_timestamp - timedelta(days=30)

    # Format for SQL query (ISO format)
    start_str = start_time.strftime('%Y-%m-%dT%H:%M:%S')
    end_str = as_of_timestamp.strftime('%Y-%m-%dT%H:%M:%S')

    try:
        if db_connection is None:
            from src.database import db as database
            conn = database.get_connection()
        else:
            conn = db_connection

        cursor = conn.cursor()

        # Sum total_value from trades in the 30-day window
        # trades.total_value is already notional (qty * price)
        cursor.execute("""
            SELECT COALESCE(SUM(total_value), 0) as volume
            FROM trades
            WHERE timestamp > ? AND timestamp <= ?
        """, (start_str, end_str))

        result = cursor.fetchone()
        volume = float(result[0]) if result else 0.0

        if db_connection is None:
            conn.close()

        return volume

    except Exception as e:
        logger.warning(f"Error calculating rolling 30d volume: {e}")
        return 0.0


def determine_fee_type(order_type: str, existing_fee_type: Optional[str] = None) -> str:
    """
    Determine fee type for an order.

    Args:
        order_type: Order type from Alpaca ('market', 'limit', etc.)
        existing_fee_type: Pre-determined fee type if available

    Returns:
        'maker', 'taker', or 'maker_assumed'
    """
    # If already determined (e.g., from pending order metadata), use it
    if existing_fee_type in ('maker', 'taker'):
        return existing_fee_type

    # Market orders are always taker
    if order_type and order_type.lower() == 'market':
        return 'taker'

    # Limit orders - we assume maker since grid uses resting limit orders
    # but we can't prove they weren't marketable at submission
    return 'maker_assumed'


# Convenience function for getting fallback rates from config
def get_fallback_rates() -> dict:
    """
    Get fallback rates from config_ultra (Tier 1).
    Used when tier calculation fails or before backfill.
    """
    try:
        import config_ultra
        return {
            'maker': getattr(config_ultra, 'MAKER_FEE_PCT', 0.0015),
            'taker': getattr(config_ultra, 'TAKER_FEE_PCT', 0.0025),
        }
    except ImportError:
        return {'maker': 0.0015, 'taker': 0.0025}
