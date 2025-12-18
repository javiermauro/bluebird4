"""
SMS Message Templates

Format bot data into concise SMS messages.
"""

from typing import Dict, Any, List
from datetime import datetime


def format_trade_alert(trade: Dict[str, Any]) -> str:
    """Format a trade execution alert."""
    symbol = trade.get('symbol', 'UNKNOWN')
    side = trade.get('side', '').upper()
    if 'BUY' in side:
        side = 'BUY'
    elif 'SELL' in side:
        side = 'SELL'

    qty = trade.get('filled_qty', trade.get('qty', 0))
    price = trade.get('filled_avg_price', trade.get('price', 0))
    profit = trade.get('profit', None)

    # Clean symbol
    symbol = symbol.replace('USD', '/USD')

    msg = f"BLUEBIRD: {side} {symbol}\n"
    msg += f"{qty:.4f} @ ${price:,.2f}"

    # Only show profit if it passes sanity check
    # Profit > 20% of sale value is likely a calculation bug
    if profit is not None and side == 'SELL':
        sale_value = qty * price
        max_reasonable_profit = sale_value * 0.20  # 20% max
        if abs(profit) <= max_reasonable_profit and profit != 0:
            sign = '+' if profit >= 0 else ''
            msg += f"\nProfit: {sign}${profit:.2f}"
        # else: skip showing bogus profit

    return msg


def format_daily_summary(stats: Dict[str, Any], starting_equity: float) -> str:
    """Format daily summary message."""
    account = stats.get('account', {})
    equity = account.get('equity', 0)
    daily_pnl = equity - starting_equity
    daily_pnl_pct = (daily_pnl / starting_equity * 100) if starting_equity > 0 else 0

    # Positions summary
    positions = stats.get('positions', [])
    total_unrealized = sum(p.get('unrealized_pl', 0) for p in positions)

    # Grid performance
    grid = stats.get('grid', {})
    total_realized = sum(
        g.get('performance', {}).get('total_profit', 0)
        for g in grid.get('summaries', {}).values()
    )

    # Order stats
    orders = stats.get('orders', {}).get('stats', {})
    buys = orders.get('by_side', {}).get('buy', 0)
    sells = orders.get('by_side', {}).get('sell', 0)

    # Risk
    risk = stats.get('risk', {})
    drawdown = risk.get('drawdown_pct', 0)

    # Format message
    sign = '+' if daily_pnl >= 0 else ''
    msg = f"BLUEBIRD Daily Summary\n"
    msg += f"Equity: ${equity:,.2f} ({sign}{daily_pnl_pct:.2f}%)\n"
    msg += f"Realized: ${total_realized:,.2f}\n"
    msg += f"Unrealized: ${total_unrealized:+,.2f}\n"
    msg += f"Trades: {buys + sells} ({buys}B/{sells}S)\n"
    msg += f"Drawdown: {drawdown:.2f}%"

    return msg


def format_quick_update(stats: Dict[str, Any], starting_equity: float) -> str:
    """Format a quick status update (shorter than daily summary)."""
    account = stats.get('account', {})
    equity = account.get('equity', 0)
    daily_pnl = equity - starting_equity
    daily_pnl_pct = (daily_pnl / starting_equity * 100) if starting_equity > 0 else 0

    # Positions
    positions = stats.get('positions', [])
    pos_str = ' | '.join([
        f"{p['symbol'].replace('USD', '')}: ${p['unrealized_pl']:+.0f}"
        for p in sorted(positions, key=lambda x: -x.get('unrealized_pl', 0))
    ])

    # Grid realized
    grid = stats.get('grid', {})
    total_realized = sum(
        g.get('performance', {}).get('total_profit', 0)
        for g in grid.get('summaries', {}).values()
    )

    sign = '+' if daily_pnl >= 0 else ''
    msg = f"BLUEBIRD Update\n"
    msg += f"${equity:,.2f} ({sign}${daily_pnl:.0f} / {sign}{daily_pnl_pct:.2f}%)\n"
    msg += f"{pos_str}\n"
    msg += f"Realized: ${total_realized:,.2f}"

    return msg


def format_risk_alert(risk: Dict[str, Any], reason: str = "") -> str:
    """Format a risk/circuit breaker alert."""
    drawdown = risk.get('drawdown_pct', 0)
    daily_pnl = risk.get('daily_pnl_pct', 0)
    halted = risk.get('trading_halted', False)
    halt_reason = risk.get('halt_reason', reason)

    msg = "BLUEBIRD ALERT\n"

    if halted:
        msg += "Trading HALTED\n"
        msg += f"Reason: {halt_reason}\n"
    else:
        msg += f"Risk Warning\n"
        msg += f"Reason: {reason}\n"

    msg += f"Drawdown: {drawdown:.2f}%\n"
    msg += f"Daily P&L: {daily_pnl:+.2f}%"

    return msg


def format_stop_loss_alert(symbol: str, trigger_info: Dict[str, Any]) -> str:
    """Format a stop-loss trigger alert."""
    msg = f"BLUEBIRD STOP-LOSS\n"
    msg += f"{symbol} position closed\n"

    if trigger_info:
        price = trigger_info.get('price', 0)
        loss = trigger_info.get('loss', 0)
        msg += f"Price: ${price:,.2f}\n"
        msg += f"Loss: ${loss:,.2f}"

    return msg


def format_grid_rebalance_alert(symbol: str, old_range: Dict, new_range: Dict) -> str:
    """Format a grid rebalance alert."""
    direction = "UP" if new_range.get('lower', 0) > old_range.get('lower', 0) else "DOWN"

    msg = f"BLUEBIRD Grid Rebalance\n"
    msg += f"{symbol} rebalanced {direction}\n"
    msg += f"Old: ${old_range.get('lower', 0):,.2f}-${old_range.get('upper', 0):,.2f}\n"
    msg += f"New: ${new_range.get('lower', 0):,.2f}-${new_range.get('upper', 0):,.2f}"

    return msg


def format_startup_message(config_summary: str) -> str:
    """Format startup notification."""
    return f"BLUEBIRD Notifier Started\n{config_summary}"


def format_shutdown_message(reason: str = "Manual shutdown") -> str:
    """Format shutdown notification."""
    return f"BLUEBIRD Notifier Stopped\n{reason}"


def format_risk_off_entry(overlay_status: Dict[str, Any]) -> str:
    """
    Format RISK_OFF entry alert.

    Sent when the system enters crash protection mode.
    """
    reasons = overlay_status.get('trigger_reasons', [])
    reasons_str = ', '.join(reasons) if reasons else 'Multi-signal trigger'

    telemetry = overlay_status.get('telemetry', {})
    avoided = telemetry.get('avoided_buys_notional', 0)

    msg = "BLUEBIRD RISK_OFF\n"
    msg += "Crash protection ACTIVE\n\n"
    msg += f"Triggers: {reasons_str}\n"
    msg += "Actions:\n"
    msg += "- All buys BLOCKED\n"
    msg += "- Buy limits CANCELLED\n"
    msg += "- Rebalance-down BLOCKED\n"
    msg += "- Sells still allowed\n\n"

    if avoided > 0:
        msg += f"(${avoided:,.0f} in buys avoided so far)"

    return msg


def format_risk_off_exit(overlay_status: Dict[str, Any], to_mode: str) -> str:
    """
    Format RISK_OFF exit alert.

    Sent when exiting crash protection (to RECOVERY or NORMAL).
    """
    telemetry = overlay_status.get('telemetry', {})
    avoided = telemetry.get('avoided_buys_notional', 0)
    cancelled = telemetry.get('cancelled_limits_notional', 0)
    duration = overlay_status.get('mode_duration_minutes', 0)

    if to_mode == "RECOVERY":
        msg = "BLUEBIRD RECOVERY\n"
        msg += "Entering recovery mode\n\n"

        recovery = overlay_status.get('recovery', {})
        if recovery:
            mult = recovery.get('stage_multiplier', 0.25)
            msg += f"Position size: {mult:.0%} (ramping up)\n"

        msg += "Buys allowed (reduced size)\n"
        msg += "Rebalance-down still blocked\n\n"
    else:  # NORMAL
        msg = "BLUEBIRD NORMAL\n"
        msg += "Full trading resumed\n\n"
        msg += "All restrictions lifted\n\n"

    msg += f"RISK_OFF duration: {duration:.0f} min\n"

    total_protected = avoided + cancelled
    if total_protected > 0:
        msg += f"Protected ${total_protected:,.0f} from crash exposure"

    return msg


def format_recovery_to_normal(overlay_status: Dict[str, Any]) -> str:
    """
    Format RECOVERY -> NORMAL transition alert.

    Sent when fully exiting recovery and returning to normal trading.
    """
    telemetry = overlay_status.get('telemetry', {})
    avoided = telemetry.get('avoided_buys_notional', 0)
    cancelled = telemetry.get('cancelled_limits_notional', 0)

    msg = "BLUEBIRD NORMAL\n"
    msg += "Full trading resumed\n\n"
    msg += "Recovery complete\n"
    msg += "All restrictions lifted\n\n"

    total_protected = avoided + cancelled
    if total_protected > 0:
        msg += f"Total protected: ${total_protected:,.0f}"

    return msg
