"""
BLUEBIRD ULTRA API Server

FastAPI server with WebSocket for real-time dashboard updates.
Now includes full Ultra system state broadcasting.
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import logging
import json
import asyncio
import os
import numpy as np
from datetime import datetime
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


def convert_numpy_types(obj):
    """
    Recursively convert numpy types to native Python types for JSON serialization.
    Handles numpy.float64, numpy.int64, numpy.bool_, numpy.ndarray, etc.
    """
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


logger = logging.getLogger("API")

app = FastAPI(title="BlueBird ULTRA API", version="4.0")

# Serve the dashboard static files
DASHBOARD_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "dashboard", "dist")
if os.path.exists(DASHBOARD_PATH):
    app.mount("/assets", StaticFiles(directory=os.path.join(DASHBOARD_PATH, "assets")), name="assets")

# Enable CORS for Frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ConnectionManager:
    """Manages WebSocket connections for real-time updates."""
    
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"Client connected. Total connections: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info(f"Client disconnected. Total connections: {len(self.active_connections)}")

    async def broadcast(self, message: dict):
        """Broadcast message to all connected clients."""
        disconnected = []
        for connection in self.active_connections:
            try:
                # Add timeout to prevent slow clients from freezing all broadcasts
                await asyncio.wait_for(connection.send_json(message), timeout=5.0)
            except asyncio.TimeoutError:
                logger.warning("Broadcast timeout, disconnecting slow client")
                disconnected.append(connection)
            except Exception as e:
                logger.error(f"Error broadcasting: {e}")
                disconnected.append(connection)

        # Clean up disconnected clients
        for conn in disconnected:
            self.disconnect(conn)


manager = ConnectionManager()


# Shared state (updated by the trading loop)
system_state = {
    "status": "initializing",
    "price": 0.0,
    "timestamp": "",
    "symbol": "BTC/USD",
    "market": {
        "high": 0,
        "low": 0,
        "volume": 0,
        "change": 0
    },
    "account": {
        "equity": 0.0,
        "buying_power": 0.0,
        "balance": 0.0
    },
    "positions": [],
    "logs": [],
    
    # Multi-asset state
    "multi_asset": {
        "symbols": [],
        "signals": {},
        "confidences": {},
        "active_symbol": ""
    },
    
    # AI system state (new)
    "ai": {
        "prediction": None,
        "confidence": 0,
        "signal": "HOLD",
        "reasoning": [],
        "features": {
            "rsi": 50,
            "macd": 0,
            "macd_hist": 0,
            "bb_position": 0.5,
            "atr_pct": 0,
            "volume_ratio": 1.0,
            "momentum": 0,
            "adx": 0,
            "price": 0,
            "trend": "NEUTRAL"
        },
        "thresholds": {
            "buy_threshold": 0.65,
            "sell_threshold": 0.35,
            "min_confidence": 70
        },
        "multi_timeframe": {
            "1min": "NEUTRAL",
            "5min": "NEUTRAL",
            "15min": "NEUTRAL"
        },
        "feature_importance": {
            "rsi": 0.35,
            "volume": 0.25,
            "macd": 0.18,
            "bollinger": 0.12,
            "price_action": 0.10
        }
    },
    
    # Ultra system state (legacy compatibility)
    "ultra": {
        "regime": "AI_ADAPTIVE",
        "strategy": "WAIT",
        "confidence": 0.0,
        "signal": "HOLD",
        "should_trade": False,
        "trade_reason": "",
        "metrics": {
            "adx": 0,
            "rsi": 50,
            "atr_pct": 0,
            "volume_ratio": 1.0,
            "trend_score": 0
        },
        "time_filter": {
            "score": 0.5,
            "window_name": "NEUTRAL",
            "is_weekend": False
        },
        "kelly": {
            "kelly_fraction": 0.25,
            "win_rate": 0.5,
            "sample_size": 0
        }
    },
    
    # Last trade info (for dashboard)
    "last_trade": None,

    # Order confirmation tracking (new)
    "orders": {
        "confirmed": [],  # List of Alpaca-verified orders
        "stats": {
            "total_confirmed": 0,
            "by_symbol": {},
            "by_side": {"buy": 0, "sell": 0},
            "total_volume": 0.0
        },
        "reconciliation": {
            "synced": True,
            "last_check": None,
            "matched": 0,
            "total": 0,
            "discrepancies": []
        }
    },

    # Notifier state (for dashboard)
    "notifier": {
        "running": False,
        "pid": None,
        "last_sms_sent": None,
        "sms_count_today": 0,
        "quiet_hours_active": False,
        "uptime": None
    }
}


@app.get("/")
async def root():
    return {
        "message": "BlueBird ULTRA API is running",
        "version": "4.0",
        "system": "Regime-Adaptive Trading"
    }


@app.get("/stats")
async def get_stats():
    """Get current system state."""
    # Convert numpy types to native Python types for JSON serialization
    return convert_numpy_types(system_state)


@app.get("/health")
async def health_check():
    """Health check endpoint with stream health monitoring."""
    # Check if grid trading mode is active
    try:
        from config_ultra import UltraConfig
        config = UltraConfig()
        is_grid = getattr(config, 'USE_GRID_TRADING', False)
    except:
        is_grid = False

    # Calculate stream health from last_update timestamp
    last_update_str = system_state.get("timestamp")
    stream_status = "unknown"
    seconds_since_bar = 0

    if last_update_str:
        try:
            # Parse timestamp (format: "2025-12-11 05:16:00+00:00")
            last_update = datetime.fromisoformat(str(last_update_str).replace('+00:00', '').replace('Z', ''))
            seconds_since_bar = (datetime.utcnow() - last_update).total_seconds()

            if seconds_since_bar < 90:
                stream_status = "healthy"
            elif seconds_since_bar < 180:
                stream_status = "degraded"
            else:
                stream_status = "stale"
        except Exception:
            stream_status = "unknown"

    return {
        "status": "healthy",
        "connections": len(manager.active_connections),
        "regime": "GRID_TRADING" if is_grid else system_state["ultra"]["regime"],
        "last_update": system_state["timestamp"],
        "stream_health": {
            "status": stream_status,
            "seconds_since_bar": int(seconds_since_bar)
        }
    }


@app.get("/api/services")
async def get_services_status():
    """Get status of all BLUEBIRD services (bot, notifier, etc.)."""
    try:
        from src.utils.process_lock import get_all_services_status
        services = get_all_services_status()

        # Add notifier status from existing check
        notifier_status = get_notifier_status()

        return {
            "services": services,
            "notifier": notifier_status,
            "api_server": {
                "running": True,
                "pid": os.getpid(),
                "connections": len(manager.active_connections)
            }
        }
    except ImportError:
        return {
            "error": "Process lock module not available",
            "api_server": {
                "running": True,
                "pid": os.getpid(),
                "connections": len(manager.active_connections)
            }
        }


class SettingsUpdate(BaseModel):
    scalping_mode: bool = None
    max_positions: int = None
    risk_per_trade: float = None


# Import config for settings updates
try:
    from config_ultra import UltraConfig as Config
except ImportError:
    from config import Config


@app.post("/api/settings")
async def update_settings(settings: SettingsUpdate):
    """Update trading settings."""
    logger.info(f"Updating settings: {settings}")
    
    updates = {}
    
    if settings.scalping_mode is not None:
        # Legacy support
        updates['scalping_mode'] = settings.scalping_mode
    
    if settings.max_positions is not None:
        Config.MAX_POSITIONS = settings.max_positions
        updates['max_positions'] = settings.max_positions
    
    if settings.risk_per_trade is not None:
        Config.MAX_RISK_PER_TRADE = settings.risk_per_trade
        updates['risk_per_trade'] = settings.risk_per_trade
    
    return {"status": "updated", "changes": updates}


@app.get("/api/settings")
async def get_settings():
    """Get current trading settings."""
    return {
        "max_positions": getattr(Config, 'MAX_POSITIONS', 2),
        "risk_per_trade": getattr(Config, 'MAX_RISK_PER_TRADE', 0.02),
        "kelly_fraction": getattr(Config, 'KELLY_FRACTION', 0.25),
        "symbol": getattr(Config, 'SYMBOL', 'BTC/USD'),
        "timeframe": getattr(Config, 'TIMEFRAME', '5Min')
    }


@app.get("/api/regime")
async def get_regime():
    """Get current market regime analysis."""
    return convert_numpy_types(system_state["ultra"])


@app.get("/api/positions")
async def get_positions():
    """
    Get current positions from Alpaca.

    Returns positions either from system_state (if WebSocket is broadcasting)
    or directly from Alpaca API.
    """
    # If system_state has positions, use them
    if system_state.get("positions"):
        return {"positions": system_state["positions"], "source": "websocket"}

    # Otherwise fetch directly from Alpaca
    try:
        from config_ultra import UltraConfig
        from src.execution.alpaca_client import AlpacaClient

        config = UltraConfig()
        client = AlpacaClient(config)
        positions = client.get_positions()

        positions_data = []
        for p in positions:
            positions_data.append({
                "symbol": p.symbol,
                "qty": float(p.qty),
                "avg_entry_price": float(p.avg_entry_price),
                "current_price": float(p.current_price),
                "unrealized_pl": float(p.unrealized_pl),
                "unrealized_plpc": float(p.unrealized_plpc),
                "market_value": float(p.market_value)
            })

        return {"positions": positions_data, "source": "alpaca"}
    except Exception as e:
        logger.error(f"Failed to fetch positions from Alpaca: {e}")
        return {"positions": [], "source": "error", "error": str(e)}


# === ORDER CONFIRMATION ENDPOINTS ===

@app.get("/api/orders")
async def get_orders(limit: int = 50, symbol: str = None):
    """
    Get confirmed orders from Alpaca.

    Args:
        limit: Maximum number of orders to return (default 50)
        symbol: Optional filter by symbol (e.g., 'BTCUSD')

    Returns:
        List of confirmed orders, most recent first
    """
    orders = system_state["orders"]["confirmed"]

    if symbol:
        # Normalize symbol for comparison
        symbol_normalized = symbol.replace('/', '').upper()
        orders = [o for o in orders if o.get('symbol', '').replace('/', '').upper() == symbol_normalized]

    # Sort by filled_at (most recent first)
    orders = sorted(orders, key=lambda x: x.get('filled_at', ''), reverse=True)

    return {
        "orders": orders[:limit],
        "total": len(orders),
        "limit": limit
    }


@app.get("/api/orders/stats")
async def get_order_stats():
    """
    Get order statistics and summary.

    Returns:
        Order stats including totals by symbol, side, and volume
    """
    # If system_state has data, use it
    stats = system_state["orders"]["stats"]
    if stats.get("total_confirmed", 0) > 0:
        return stats

    # Otherwise fetch directly from Alpaca
    try:
        from config_ultra import UltraConfig
        from src.execution.alpaca_client import AlpacaClient
        from datetime import datetime, timedelta

        config = UltraConfig()
        client = AlpacaClient(config)

        # Get orders from last 7 days
        orders = client.get_order_history(days=7, status='filled')

        by_symbol = {}
        by_side = {"buy": 0, "sell": 0}
        total_volume = 0.0

        for order in orders:
            symbol = order.get('symbol', 'UNKNOWN')
            side = order.get('side', '').lower()
            qty = float(order.get('filled_qty', 0))
            price = float(order.get('filled_avg_price', 0))
            volume = qty * price

            by_symbol[symbol] = by_symbol.get(symbol, 0) + 1
            if 'buy' in side:
                by_side['buy'] += 1
            elif 'sell' in side:
                by_side['sell'] += 1
            total_volume += volume

        return {
            "total_confirmed": len(orders),
            "by_symbol": by_symbol,
            "by_side": by_side,
            "total_volume": round(total_volume, 2)
        }
    except Exception as e:
        logger.error(f"Error fetching order stats: {e}")
        return stats


@app.get("/api/orders/reconciliation")
async def get_reconciliation_status():
    """
    Get reconciliation status between bot and Alpaca.

    Returns:
        Sync status, matched count, and any discrepancies
    """
    return convert_numpy_types(system_state["orders"]["reconciliation"])


@app.get("/api/grid/status")
async def get_grid_status():
    """
    Get current grid trading status using GridTradingStrategy for accurate profit calculation.

    Returns accurate data even when WebSocket is not connected.
    """
    grid_data = {"grids": {}, "total_trades": 0, "total_profit": 0}

    try:
        # Use GridTradingStrategy to get accurate profit estimates
        from src.strategy.grid_trading import GridTradingStrategy

        gs = GridTradingStrategy()
        if gs.load_state():
            total_trades = 0
            total_profit = 0.0
            summaries = {}

            for symbol in gs.grids.keys():
                summary = gs.get_grid_summary(symbol)
                perf = summary.get('performance', {})

                completed = perf.get('completed_trades', 0)
                profit = perf.get('total_profit', 0)
                total_trades += completed
                total_profit += profit

                summaries[symbol] = {
                    'is_active': summary.get('is_active', False),
                    'completed_trades': completed,
                    'total_profit': profit,
                    'levels': summary.get('levels', {}),
                    'total_buys': perf.get('total_buys', 0),
                    'total_sells': perf.get('total_sells', 0),
                    'range': summary.get('range', {})
                }

            grid_data = {
                'active': any(s.get('is_active') for s in summaries.values()),
                'total_trades': total_trades,
                'total_profit': round(total_profit, 2),
                'summaries': summaries
            }
    except Exception as e:
        grid_data['error'] = str(e)
        logger.error(f"Error getting grid status: {e}")

    # Also include order stats from Alpaca for verification
    alpaca_stats = system_state.get("orders", {}).get("stats", {})
    if alpaca_stats.get("total_confirmed", 0) == 0:
        # Fetch directly if system_state is empty
        try:
            from config_ultra import UltraConfig
            from src.execution.alpaca_client import AlpacaClient

            config = UltraConfig()
            client = AlpacaClient(config)
            orders = client.get_order_history(days=7, status='filled')

            by_symbol = {}
            by_side = {"buy": 0, "sell": 0}
            total_volume = 0.0

            for order in orders:
                symbol = order.get('symbol', 'UNKNOWN')
                side = order.get('side', '').lower()
                qty = float(order.get('filled_qty', 0))
                price = float(order.get('filled_avg_price', 0))
                volume = qty * price

                by_symbol[symbol] = by_symbol.get(symbol, 0) + 1
                if 'buy' in side:
                    by_side['buy'] += 1
                elif 'sell' in side:
                    by_side['sell'] += 1
                total_volume += volume

            alpaca_stats = {
                "total_confirmed": len(orders),
                "by_symbol": by_symbol,
                "by_side": by_side,
                "total_volume": round(total_volume, 2)
            }
        except Exception as e:
            alpaca_stats = {"error": str(e)}

    grid_data['alpaca_orders'] = alpaca_stats

    return grid_data


@app.get("/api/regime/smart-grid")
async def get_smart_grid_regime():
    """
    Get current Smart Grid regime detection status.

    Returns regime state for each symbol, including:
    - Current regime (RANGING, TRENDING_UP, TRENDING_DOWN, etc.)
    - ADX value
    - Whether grid is paused for this symbol
    - Allow buy/sell flags
    """
    # Get regime data from last WebSocket broadcast
    regime_data = system_state.get("regime", {})

    if not regime_data:
        # Return from stored broadcast data if available
        return {
            "status": "initializing",
            "message": "Regime detection initializing - waiting for bar data (need 50+ bars)",
            "all_regimes": {},
            "all_paused": {}
        }

    return {
        "status": "active",
        "current_symbol": regime_data.get("current", "UNKNOWN"),
        "adx": regime_data.get("adx", 0),
        "allow_buy": regime_data.get("allow_buy", True),
        "allow_sell": regime_data.get("allow_sell", True),
        "is_strong_trend": regime_data.get("is_strong_trend", False),
        "paused": regime_data.get("paused", False),
        "reason": regime_data.get("reason", ""),
        "strategy_hint": regime_data.get("strategy_hint", "WAIT"),
        "confidence": regime_data.get("confidence", 0),
        "all_regimes": regime_data.get("all_regimes", {}),
        "all_paused": regime_data.get("all_paused", {})
    }


@app.post("/api/orders/reconcile")
async def trigger_reconciliation():
    """
    Trigger a reconciliation check with Alpaca.

    Note: This is a placeholder - actual reconciliation happens in the bot.
    """
    return {
        "message": "Reconciliation triggered",
        "status": "pending",
        "note": "Check /api/orders/reconciliation for results"
    }


# === NOTIFIER CONTROL ENDPOINTS ===

PID_FILE = "/tmp/bluebird-notifier.pid"
SMS_COUNT_FILE = "/tmp/bluebird-notifier-count.json"


def get_notifier_status() -> dict:
    """Check if the notifier process is running."""
    import os
    import json
    from datetime import datetime

    result = {
        "running": False,
        "pid": None,
        "uptime": None,
        "last_sms_sent": None,
        "sms_count_today": 0,
        "quiet_hours_active": False
    }

    # Check PID file
    if os.path.exists(PID_FILE):
        try:
            with open(PID_FILE, 'r') as f:
                data = json.load(f)
                pid = data.get('pid')
                started_at = data.get('started_at')

            # Check if process is actually running
            if pid:
                try:
                    os.kill(pid, 0)  # Signal 0 just checks if process exists
                    result["running"] = True
                    result["pid"] = pid

                    # Calculate uptime
                    if started_at:
                        start_time = datetime.fromisoformat(started_at)
                        uptime_seconds = (datetime.now() - start_time).total_seconds()
                        hours, remainder = divmod(int(uptime_seconds), 3600)
                        minutes, _ = divmod(remainder, 60)
                        result["uptime"] = f"{hours}h {minutes}m"
                except (ProcessLookupError, PermissionError):
                    # Process not running, clean up PID file
                    os.remove(PID_FILE)
        except (json.JSONDecodeError, FileNotFoundError):
            pass

    # Check SMS count file
    if os.path.exists(SMS_COUNT_FILE):
        try:
            with open(SMS_COUNT_FILE, 'r') as f:
                count_data = json.load(f)
                result["sms_count_today"] = count_data.get("count", 0)
                result["last_sms_sent"] = count_data.get("last_sent")
        except:
            pass

    # Check quiet hours
    current_hour = datetime.now().hour
    result["quiet_hours_active"] = current_hour >= 23 or current_hour < 7

    return result


@app.get("/api/notifier/status")
async def notifier_status():
    """Get SMS notifier service status."""
    status = get_notifier_status()
    # Update system state for dashboard
    system_state["notifier"].update(status)
    return status


@app.post("/api/notifier/start")
async def start_notifier():
    """Start the SMS notification service."""
    import subprocess
    import os

    status = get_notifier_status()
    if status["running"]:
        return {"status": "already_running", "pid": status["pid"]}

    # Start the notifier as a background process
    # Use caffeinate to prevent macOS App Nap from suspending the process
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    notifier_script = os.path.join(project_root, "src", "notifications", "notifier.py")

    try:
        # caffeinate -i prevents idle sleep and App Nap
        # Use -- to separate caffeinate options from the command
        process = subprocess.Popen(
            ["caffeinate", "-i", "--", "python3", notifier_script],
            cwd=project_root,
            stdout=open("/tmp/bluebird-notifier.log", "a"),
            stderr=subprocess.STDOUT,
            start_new_session=True  # Detach from parent
        )

        # Give it a moment to start
        await asyncio.sleep(1)

        new_status = get_notifier_status()
        system_state["notifier"].update(new_status)

        return {
            "status": "started",
            "pid": process.pid,
            "message": "Notifier started successfully"
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.post("/api/notifier/stop")
async def stop_notifier():
    """Stop the SMS notification service gracefully."""
    import os
    import signal

    status = get_notifier_status()
    if not status["running"]:
        return {"status": "not_running"}

    pid = status["pid"]
    try:
        os.kill(pid, signal.SIGTERM)
        # Wait a moment for graceful shutdown
        await asyncio.sleep(1)

        new_status = get_notifier_status()
        system_state["notifier"].update(new_status)

        return {
            "status": "stopped",
            "message": f"Notifier (PID {pid}) stopped"
        }
    except ProcessLookupError:
        return {"status": "not_running", "message": "Process already stopped"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.post("/api/notifier/test")
async def test_notifier():
    """Send a test SMS message."""
    try:
        import sys
        import os
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        sys.path.insert(0, project_root)

        from src.notifications.notifier import NotificationService
        from src.notifications.config import NotificationConfig
        from src.notifications import templates

        config = NotificationConfig()
        service = NotificationService(config)

        # Fetch current stats for the test message
        stats = system_state

        if config.is_configured():
            message = templates.format_quick_update(
                stats,
                stats.get('risk', {}).get('starting_equity', 89309)
            )
            success = service.send_sms(message, force=True, sms_type="test")
            return {
                "status": "sent" if success else "failed",
                "message": message if success else "Failed to send SMS"
            }
        else:
            return {
                "status": "not_configured",
                "message": "Twilio credentials not configured in .env"
            }
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.get("/api/notifier/history")
async def get_notifier_history():
    """Get SMS notification history (last 20 messages)."""
    try:
        from src.notifications.config import load_sms_history
        history = load_sms_history()
        # Return in reverse order (newest first)
        return {"history": list(reversed(history))}
    except Exception as e:
        return {"history": [], "error": str(e)}


@app.get("/api/notifier/config")
async def get_notifier_config():
    """Get notification settings."""
    try:
        from src.notifications.config import load_settings, mask_phone, NotificationConfig
        settings = load_settings()

        # Get the phone number (masked)
        config = NotificationConfig()
        settings["phone_number"] = mask_phone(config.notify_phone_number)

        return settings
    except Exception as e:
        return {"error": str(e)}


@app.post("/api/notifier/config")
async def update_notifier_config(request: Request):
    """Update notification settings."""
    try:
        from src.notifications.config import load_settings, save_settings

        data = await request.json()
        key = data.get("key")
        value = data.get("value")

        if not key:
            return {"status": "error", "message": "Missing 'key' parameter"}

        # Load current settings
        settings = load_settings()

        # Update the specific setting
        if key in settings:
            settings[key] = value
            save_settings(settings)
            return {"status": "updated", "config": settings}
        else:
            return {"status": "error", "message": f"Unknown setting: {key}"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


# === NOTIFIER WATCHDOG ===
# Auto-restarts notifier if it stops detecting trades

WATCHDOG_INTERVAL = 300  # Check every 5 minutes
WATCHDOG_STATE_FILE = "/tmp/bluebird-watchdog.json"

watchdog_state = {
    "enabled": True,
    "last_check": None,
    "last_restart": None,
    "restart_count": 0,
    "last_known_trade_count": 0,
    "status": "initializing"
}


def save_watchdog_state():
    """Save watchdog state to disk."""
    try:
        with open(WATCHDOG_STATE_FILE, 'w') as f:
            json.dump(watchdog_state, f, indent=2, default=str)
    except Exception as e:
        logger.error(f"Failed to save watchdog state: {e}")


def check_notifier_health() -> dict:
    """
    Check if notifier is healthy by comparing:
    - Number of confirmed trades vs last SMS sent time
    - Only counts trades that happened AFTER the notifier started
    - If trades happened but no SMS in 15+ minutes, notifier is stale
    """
    result = {
        "healthy": True,
        "reason": "OK",
        "trades_since_last_sms": 0,
        "minutes_since_last_sms": 0
    }

    try:
        # Get notifier status
        notifier = get_notifier_status()

        # If notifier not running, it's unhealthy
        if not notifier.get("running"):
            result["healthy"] = False
            result["reason"] = "Notifier not running"
            return result

        # Get notifier start time - only consider trades AFTER this time
        notifier_started = notifier.get("started_at")
        notifier_start_time = None
        if notifier_started:
            try:
                notifier_start_time = datetime.fromisoformat(notifier_started.replace('Z', '+00:00'))
                if notifier_start_time.tzinfo:
                    notifier_start_time = notifier_start_time.replace(tzinfo=None)
            except:
                pass

        # Get last SMS time
        last_sms = notifier.get("last_sms_sent")
        if not last_sms:
            # No SMS ever sent - but only flag unhealthy if notifier has been running 5+ min
            # and there are trades AFTER notifier started
            if notifier_start_time:
                minutes_running = (datetime.now() - notifier_start_time).total_seconds() / 60
                if minutes_running < 5:
                    result["reason"] = f"Notifier just started ({int(minutes_running)} min ago)"
                    return result
            return result

        # Parse last SMS time
        try:
            last_sms_time = datetime.fromisoformat(last_sms.replace('Z', '+00:00'))
            if last_sms_time.tzinfo:
                last_sms_time = last_sms_time.replace(tzinfo=None)
        except:
            last_sms_time = datetime.now()

        minutes_since_sms = (datetime.now() - last_sms_time).total_seconds() / 60
        result["minutes_since_last_sms"] = round(minutes_since_sms, 1)

        # Count trades since last SMS that ALSO happened after notifier started
        # This prevents flagging trades that happened during restart before notifier was ready
        confirmed_orders = system_state.get("orders", {}).get("confirmed", [])
        trades_since_sms = 0

        # Use the later of: last_sms_time or notifier_start_time
        # This way we only count trades the notifier could have seen
        cutoff_time = last_sms_time
        if notifier_start_time and notifier_start_time > cutoff_time:
            cutoff_time = notifier_start_time

        for order in confirmed_orders:
            filled_at = order.get("filled_at", "")
            if filled_at:
                try:
                    trade_time = datetime.fromisoformat(filled_at.replace('Z', '+00:00'))
                    if trade_time.tzinfo:
                        trade_time = trade_time.replace(tzinfo=None)
                    # Only count if trade happened AFTER the cutoff
                    if trade_time > cutoff_time:
                        trades_since_sms += 1
                except:
                    pass

        result["trades_since_last_sms"] = trades_since_sms

        # If trades happened 15+ minutes ago but no SMS, notifier is stale
        # (Increased from 10 to 15 min to reduce false positives)
        if trades_since_sms > 0 and minutes_since_sms > 15:
            result["healthy"] = False
            result["reason"] = f"{trades_since_sms} trades in last {int(minutes_since_sms)} min with no SMS"

        return result

    except Exception as e:
        result["healthy"] = False
        result["reason"] = f"Health check error: {str(e)}"
        return result


async def restart_notifier_internal():
    """Internal function to restart the notifier."""
    import subprocess
    import signal

    logger.warning("WATCHDOG: Restarting notifier...")

    # Stop existing notifier
    status = get_notifier_status()
    if status.get("running") and status.get("pid"):
        try:
            os.kill(status["pid"], signal.SIGKILL)
            await asyncio.sleep(2)
        except:
            pass

    # Start new notifier with caffeinate to prevent App Nap
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    notifier_script = os.path.join(project_root, "src", "notifications", "notifier.py")

    try:
        # caffeinate -i prevents idle sleep and App Nap
        # Use -- to separate caffeinate options from the command
        process = subprocess.Popen(
            ["caffeinate", "-i", "--", "python3", notifier_script],
            cwd=project_root,
            stdout=open("/tmp/bluebird-notifier.log", "a"),
            stderr=subprocess.STDOUT,
            start_new_session=True
        )
        await asyncio.sleep(2)

        new_status = get_notifier_status()
        system_state["notifier"].update(new_status)

        logger.info(f"WATCHDOG: Notifier restarted with PID {process.pid}")
        return True
    except Exception as e:
        logger.error(f"WATCHDOG: Failed to restart notifier: {e}")
        return False


async def watchdog_loop():
    """Background task that monitors notifier health."""
    global watchdog_state

    logger.info("WATCHDOG: Started - monitoring notifier health every 5 minutes")
    watchdog_state["status"] = "running"

    # Wait for system to stabilize
    await asyncio.sleep(60)

    while True:
        try:
            if not watchdog_state["enabled"]:
                watchdog_state["status"] = "disabled"
                await asyncio.sleep(WATCHDOG_INTERVAL)
                continue

            watchdog_state["last_check"] = datetime.now().isoformat()
            health = check_notifier_health()

            if not health["healthy"]:
                logger.warning(f"WATCHDOG: Notifier unhealthy - {health['reason']}")

                # Restart the notifier
                success = await restart_notifier_internal()

                if success:
                    watchdog_state["last_restart"] = datetime.now().isoformat()
                    watchdog_state["restart_count"] += 1
                    watchdog_state["status"] = f"restarted ({health['reason']})"
                    logger.info("WATCHDOG: Notifier restart successful")
                else:
                    watchdog_state["status"] = "restart_failed"
            else:
                watchdog_state["status"] = "healthy"

            save_watchdog_state()

        except Exception as e:
            logger.error(f"WATCHDOG: Error in loop: {e}")
            watchdog_state["status"] = f"error: {str(e)}"

        await asyncio.sleep(WATCHDOG_INTERVAL)


@app.get("/api/watchdog/status")
async def get_watchdog_status():
    """Get watchdog health monitor status."""
    health = check_notifier_health()
    return {
        "watchdog": watchdog_state,
        "notifier_health": health
    }


@app.post("/api/watchdog/toggle")
async def toggle_watchdog():
    """Enable or disable the watchdog."""
    watchdog_state["enabled"] = not watchdog_state["enabled"]
    save_watchdog_state()
    return {
        "enabled": watchdog_state["enabled"],
        "message": f"Watchdog {'enabled' if watchdog_state['enabled'] else 'disabled'}"
    }


@app.post("/api/watchdog/restart-notifier")
async def manual_restart_notifier():
    """Manually trigger a notifier restart via watchdog."""
    success = await restart_notifier_internal()
    if success:
        watchdog_state["last_restart"] = datetime.now().isoformat()
        watchdog_state["restart_count"] += 1
        save_watchdog_state()
    return {
        "status": "restarted" if success else "failed",
        "watchdog": watchdog_state
    }


# === CIRCUIT BREAKER RESET ===

@app.get("/api/risk/status")
async def get_risk_status():
    """Get current circuit breaker status and file state."""
    import json
    file_state = {}
    try:
        circuit_file = "/tmp/bluebird-circuit-breaker.json"
        if os.path.exists(circuit_file):
            with open(circuit_file, 'r') as f:
                file_state = json.load(f)
    except Exception as e:
        file_state = {"error": str(e)}

    return {
        "from_api": system_state.get("risk", {}),
        "from_file": file_state,
        "message": "Use POST /api/risk/reset to clear circuit breakers"
    }


@app.post("/api/risk/reset")
async def reset_circuit_breakers(reset_type: str = "all"):
    """
    Manually reset circuit breakers to resume trading.

    Args:
        reset_type: 'all', 'drawdown', 'daily', or symbol name (e.g., 'BTC/USD')

    This is the ONLY way to resume trading after a circuit breaker triggers.
    Bot restarts will NOT clear these flags (they persist to disk).
    """
    import json

    circuit_file = "/tmp/bluebird-circuit-breaker.json"

    # Read current state
    current_state = {}
    if os.path.exists(circuit_file):
        try:
            with open(circuit_file, 'r') as f:
                current_state = json.load(f)
        except:
            pass

    # Track what we're resetting
    reset_items = []

    if reset_type == "all":
        if current_state.get("max_drawdown_hit"):
            reset_items.append("max_drawdown")
        if current_state.get("daily_limit_hit"):
            reset_items.append("daily_limit")
        if current_state.get("stop_losses_triggered"):
            reset_items.extend(list(current_state["stop_losses_triggered"].keys()))

        # Clear everything
        new_state = {
            "max_drawdown_hit": False,
            "max_drawdown_triggered_at": None,
            "daily_limit_hit": False,
            "daily_limit_date": None,
            "stop_losses_triggered": {},
            "last_updated": datetime.now().isoformat(),
            "reset_at": datetime.now().isoformat(),
            "reset_by": "API call to /api/risk/reset"
        }

    elif reset_type == "drawdown":
        reset_items.append("max_drawdown")
        new_state = current_state.copy()
        new_state["max_drawdown_hit"] = False
        new_state["max_drawdown_triggered_at"] = None
        new_state["last_updated"] = datetime.now().isoformat()

    elif reset_type == "daily":
        reset_items.append("daily_limit")
        new_state = current_state.copy()
        new_state["daily_limit_hit"] = False
        new_state["daily_limit_date"] = None
        new_state["last_updated"] = datetime.now().isoformat()

    else:
        # Assume it's a symbol for stop-loss reset
        stop_losses = current_state.get("stop_losses_triggered", {})
        if reset_type in stop_losses:
            reset_items.append(reset_type)
            del stop_losses[reset_type]
            new_state = current_state.copy()
            new_state["stop_losses_triggered"] = stop_losses
            new_state["last_updated"] = datetime.now().isoformat()
        else:
            return {
                "success": False,
                "message": f"Unknown reset type or symbol: {reset_type}",
                "valid_types": ["all", "drawdown", "daily"] + list(stop_losses.keys())
            }

    # Save the cleared state
    try:
        with open(circuit_file, 'w') as f:
            json.dump(new_state, f, indent=2)
    except Exception as e:
        return {"success": False, "message": f"Failed to save: {e}"}

    logger.warning(f"CIRCUIT BREAKERS RESET via API: {reset_items}")

    return {
        "success": True,
        "reset_items": reset_items,
        "message": f"Reset: {', '.join(reset_items)}. Bot needs restart to pick up changes.",
        "note": "Restart the bot for changes to take effect"
    }


@app.get("/api/orders/history")
async def get_alpaca_history(days: int = 7):
    """
    Get order history directly from Alpaca API.

    Args:
        days: Number of days of history to fetch (default 7)

    Returns:
        Raw order history from Alpaca
    """
    try:
        from src.execution.alpaca_client import AlpacaClient
        from config_ultra import UltraConfig

        config = UltraConfig()
        client = AlpacaClient(config)
        orders = client.get_order_history(days=days, status='all')

        return {
            "orders": orders,
            "total": len(orders),
            "days": days,
            "source": "alpaca"
        }
    except Exception as e:
        return {
            "error": str(e),
            "orders": [],
            "total": 0
        }


# =========================================
# WINDFALL PROFIT-TAKING API ENDPOINTS
# =========================================

@app.get("/api/windfall/stats")
async def get_windfall_stats():
    """
    Get windfall profit-taking statistics from system state.

    Returns:
        Windfall stats including total captures, profit, and recent transactions
    """
    # Try system_state first (updated during bar processing)
    windfall_data = system_state.get("windfall", None)

    if windfall_data and windfall_data.get("enabled") is not None:
        return windfall_data

    # If no live data yet, read from config and log file
    try:
        from config_ultra import WINDFALL_PROFIT_CONFIG
        import json

        log_file = "/tmp/bluebird-windfall-log.json"
        stats = {"total_captures": 0, "total_profit": 0.0, "transactions": []}

        if os.path.exists(log_file):
            try:
                with open(log_file, 'r') as f:
                    stats = json.load(f)
            except:
                pass

        return {
            "enabled": WINDFALL_PROFIT_CONFIG.get("enabled", False),
            "total_captures": stats.get("total_captures", 0),
            "total_profit": stats.get("total_profit", 0.0),
            "transactions": stats.get("transactions", [])[-10:],
            "config": {
                "soft_threshold_pct": WINDFALL_PROFIT_CONFIG.get("soft_threshold_pct", 4.0),
                "hard_threshold_pct": WINDFALL_PROFIT_CONFIG.get("hard_threshold_pct", 6.0),
                "rsi_threshold": WINDFALL_PROFIT_CONFIG.get("rsi_threshold", 70),
                "sell_portion": WINDFALL_PROFIT_CONFIG.get("sell_portion", 0.70),
                "cooldown_minutes": WINDFALL_PROFIT_CONFIG.get("cooldown_minutes", 30)
            },
            "active_cooldowns": {}
        }
    except ImportError:
        return {
            "enabled": False,
            "total_captures": 0,
            "total_profit": 0.0,
            "transactions": [],
            "config": {},
            "active_cooldowns": {},
            "note": "WINDFALL_PROFIT_CONFIG not found in config"
        }


@app.get("/api/windfall/transactions")
async def get_windfall_transactions(limit: int = 50):
    """
    Get recent windfall transactions.

    Args:
        limit: Maximum number of transactions to return (default 50)

    Returns:
        List of windfall transactions with total count
    """
    windfall_data = system_state.get("windfall", {"transactions": []})
    transactions = windfall_data.get("transactions", [])

    return {
        "transactions": transactions[-limit:],
        "total": len(transactions),
        "total_profit": windfall_data.get("total_profit", 0.0)
    }


@app.get("/api/windfall/log")
async def get_windfall_log():
    """
    Get windfall log from persistent file for weekly review.

    Returns:
        Full windfall log from file
    """
    import json
    log_file = "/tmp/bluebird-windfall-log.json"

    try:
        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                data = json.load(f)
            return {
                "success": True,
                "data": data,
                "file": log_file
            }
        else:
            return {
                "success": True,
                "data": {
                    "total_captures": 0,
                    "total_profit": 0.0,
                    "transactions": []
                },
                "file": log_file,
                "note": "No windfall log file yet - will be created on first windfall capture"
            }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "file": log_file
        }


# =========================================
# ACCOUNT HISTORY & REALIZED P/L ENDPOINTS
# =========================================

@app.get("/api/history/equity")
async def get_equity_history(period: str = "1M"):
    """
    Get portfolio equity history - uses database first, falls back to Alpaca.

    Args:
        period: Time period - 1W, 1M, 3M, 1A (default 1M)

    Returns:
        Equity curve data with peak, trough, recovery metrics
    """
    try:
        from src.execution.alpaca_client import AlpacaClient
        from config_ultra import UltraConfig
        from src.database import db as database

        # Map period to days
        period_days = {'1W': 7, '1M': 30, '3M': 90, '1A': 365, 'all': 9999}
        days = period_days.get(period, 30)

        config = UltraConfig()
        client = AlpacaClient(config)

        # Get current account data
        account = client.trading_client.get_account()
        current_equity = float(account.equity)
        starting = 100000.0

        # Try database first (more reliable, persisted data)
        try:
            db_stats = database.get_equity_range(days=days)
            db_history = database.get_equity_history(days=days, interval='daily')

            if db_history and len(db_history) > 1:
                # Use database data
                dates = [h.get('date', '')[:10] for h in db_history]
                equity = [h.get('close', 0) for h in db_history]

                # Add current day if not present
                today = datetime.now().strftime('%Y-%m-%d')
                if dates and dates[-1] != today:
                    dates.append(today)
                    equity.append(current_equity)
                elif dates:
                    equity[-1] = current_equity

                # Calculate recovery from database stats
                trough = db_stats.get('trough', current_equity)
                peak = db_stats.get('peak', starting)

                if starting > trough:
                    recovery_pct = ((current_equity - trough) / (starting - trough)) * 100
                else:
                    recovery_pct = 100.0 if current_equity >= starting else 0.0

                total_return_pct = ((current_equity - starting) / starting) * 100

                return {
                    "dates": dates,
                    "equity": equity,
                    "peak": peak,
                    "peak_date": db_stats.get('peak_date', '')[:10] if db_stats.get('peak_date') else None,
                    "trough": trough,
                    "trough_date": db_stats.get('trough_date', '')[:10] if db_stats.get('trough_date') else None,
                    "current": current_equity,
                    "starting": starting,
                    "recovery_pct": round(recovery_pct, 2),
                    "total_return_pct": round(total_return_pct, 2),
                    "source": "database"
                }
        except Exception as db_err:
            logger.warning(f"Database query failed, falling back to Alpaca: {db_err}")

        # Fallback to Alpaca portfolio history
        try:
            history = client.get_portfolio_history(period=period, timeframe="1D")

            if history.get('equity') and len(history['equity']) > 0:
                # Use Alpaca's real equity history
                timestamps = history.get('timestamps', [])
                equity_values = history.get('equity', [])

                # Convert timestamps to dates
                dates = []
                equity = []
                for ts, eq in zip(timestamps, equity_values):
                    if eq is not None:
                        # Handle both string and datetime timestamps
                        if isinstance(ts, str):
                            date = ts[:10]
                        else:
                            date = ts.strftime('%Y-%m-%d') if hasattr(ts, 'strftime') else str(ts)[:10]
                        dates.append(date)
                        equity.append(float(eq))

                if dates and equity:
                    # Find peak and trough
                    peak = max(equity)
                    trough = min(equity)
                    peak_date = dates[equity.index(peak)]
                    trough_date = dates[equity.index(trough)]

                    # Add/update current day
                    today = datetime.now().strftime('%Y-%m-%d')
                    if dates[-1] != today:
                        dates.append(today)
                        equity.append(current_equity)
                    else:
                        equity[-1] = current_equity

                    # Update peak/trough with current
                    if current_equity > peak:
                        peak = current_equity
                        peak_date = today
                    if current_equity < trough:
                        trough = current_equity
                        trough_date = today

                    # Recovery percentage from lowest point
                    if starting > trough:
                        recovery_pct = ((current_equity - trough) / (starting - trough)) * 100
                    else:
                        recovery_pct = 100.0 if current_equity >= starting else 0.0

                    total_return_pct = ((current_equity - starting) / starting) * 100

                    return {
                        "dates": dates,
                        "equity": equity,
                        "peak": peak,
                        "peak_date": peak_date,
                        "trough": trough,
                        "trough_date": trough_date,
                        "current": current_equity,
                        "starting": starting,
                        "recovery_pct": round(recovery_pct, 2),
                        "total_return_pct": round(total_return_pct, 2)
                    }
        except Exception as hist_err:
            logger.warning(f"Portfolio history unavailable: {hist_err}")

        # Fallback: Single point with current equity
        today = datetime.now().strftime('%Y-%m-%d')

        # Known historical trough from trading records
        # Lowest point was around $86,500 on Dec 5, 2025
        known_trough = 86500.0
        known_trough_date = "2025-12-05"

        # Use either known trough or current if lower
        trough = min(current_equity, known_trough)
        trough_date = today if current_equity < known_trough else known_trough_date
        peak = max(current_equity, starting)

        # Calculate recovery from the KNOWN trough towards break-even
        # recovery = (current - trough) / (starting - trough) * 100
        if starting > trough:
            recovery_pct = ((current_equity - trough) / (starting - trough)) * 100
        else:
            recovery_pct = 100.0 if current_equity >= starting else 0.0

        total_return_pct = ((current_equity - starting) / starting) * 100

        return {
            "dates": [today],
            "equity": [current_equity],
            "peak": peak,
            "peak_date": today if current_equity >= starting else None,
            "trough": trough,
            "trough_date": trough_date,
            "current": current_equity,
            "starting": starting,
            "recovery_pct": round(recovery_pct, 2),
            "total_return_pct": round(total_return_pct, 2)
        }

    except Exception as e:
        logger.error(f"Failed to get equity history: {e}")
        import traceback
        traceback.print_exc()
        return {
            "error": str(e),
            "dates": [],
            "equity": [],
            "peak": 0,
            "trough": 0,
            "current": 0,
            "starting": 100000,
            "recovery_pct": 0,
            "total_return_pct": 0
        }


@app.get("/api/history/realized-pnl")
async def get_realized_pnl(days: int = 30):
    """
    Calculate realized P/L from filled orders by matching buy/sell pairs.

    Args:
        days: Number of days of history (default 30)

    Returns:
        Realized P/L breakdown by day and symbol, with metrics
    """
    from collections import defaultdict

    try:
        from src.execution.alpaca_client import AlpacaClient
        from config_ultra import UltraConfig

        config = UltraConfig()
        client = AlpacaClient(config)
        orders = client.get_all_filled_orders(days=days)

        if not orders:
            return {
                "total": 0,
                "by_day": [],
                "by_symbol": {},
                "metrics": {
                    "win_rate": 0,
                    "total_trades": 0,
                    "avg_per_trade": 0,
                    "best_day": {"date": None, "pnl": 0},
                    "worst_day": {"date": None, "pnl": 0}
                }
            }

        # Group orders by symbol
        by_symbol = defaultdict(list)
        for order in orders:
            raw_symbol = order.get('symbol', '')
            # Only add slash if not already present (BTCUSD -> BTC/USD, BTC/USD stays as is)
            symbol = raw_symbol if '/' in raw_symbol else raw_symbol.replace('USD', '/USD')
            by_symbol[symbol].append(order)

        # Calculate realized P/L per symbol
        symbol_pnl = {}
        total_realized = 0
        total_wins = 0
        total_trades = 0

        for symbol, symbol_orders in by_symbol.items():
            buys = [o for o in symbol_orders if o.get('side') == 'buy']
            sells = [o for o in symbol_orders if o.get('side') == 'sell']

            # Calculate average buy price
            total_buy_qty = sum(o.get('qty', 0) for o in buys)
            total_buy_value = sum(o.get('qty', 0) * o.get('price', 0) for o in buys)
            avg_buy = total_buy_value / total_buy_qty if total_buy_qty > 0 else 0

            # Calculate average sell price
            total_sell_qty = sum(o.get('qty', 0) for o in sells)
            total_sell_value = sum(o.get('qty', 0) * o.get('price', 0) for o in sells)
            avg_sell = total_sell_value / total_sell_qty if total_sell_qty > 0 else 0

            # Realized P/L on matched quantity
            matched_qty = min(total_buy_qty, total_sell_qty)
            pnl = matched_qty * (avg_sell - avg_buy)

            symbol_pnl[symbol] = {
                'pnl': round(pnl, 2),
                'trades': len(buys) + len(sells),
                'buys': len(buys),
                'sells': len(sells),
                'avg_buy': round(avg_buy, 2),
                'avg_sell': round(avg_sell, 2),
                'matched_qty': round(matched_qty, 6),
                'win': pnl > 0
            }

            total_realized += pnl
            if pnl > 0:
                total_wins += 1
            total_trades += 1

        # Group by day for daily P/L
        daily_pnl = defaultdict(lambda: {'pnl': 0, 'trades': 0, 'buys': 0, 'sells': 0})
        for order in orders:
            filled_at = order.get('filled_at', '')
            if filled_at:
                date = filled_at[:10]  # YYYY-MM-DD
                side = order.get('side', '')
                daily_pnl[date]['trades'] += 1
                if side == 'buy':
                    daily_pnl[date]['buys'] += 1
                else:
                    daily_pnl[date]['sells'] += 1

        # For daily P/L, we need to estimate based on that day's trades
        # This is an approximation - actual P/L requires matching specific orders
        by_day = []
        for date, stats in sorted(daily_pnl.items(), reverse=True):
            by_day.append({
                'date': date,
                'trades': stats['trades'],
                'buys': stats['buys'],
                'sells': stats['sells']
            })

        # Find best and worst days (by trade count as proxy)
        best_day = max(by_day, key=lambda x: x['trades']) if by_day else {'date': None, 'pnl': 0}
        worst_day = min(by_day, key=lambda x: x['trades']) if by_day else {'date': None, 'pnl': 0}

        win_rate = (total_wins / total_trades * 100) if total_trades > 0 else 0

        return {
            "total": round(total_realized, 2),
            "by_day": by_day[:30],  # Last 30 days
            "by_symbol": symbol_pnl,
            "metrics": {
                "win_rate": round(win_rate, 1),
                "total_trades": total_trades,
                "total_orders": len(orders),
                "avg_per_symbol": round(total_realized / total_trades, 2) if total_trades else 0,
                "best_day": best_day,
                "worst_day": worst_day
            }
        }

    except Exception as e:
        logger.error(f"Failed to calculate realized P/L: {e}")
        return {
            "error": str(e),
            "total": 0,
            "by_day": [],
            "by_symbol": {},
            "metrics": {}
        }


@app.get("/api/history/trades")
async def get_trade_history(symbol: str = None, days: int = 7, limit: int = 100):
    """
    Get detailed trade history from Alpaca with P/L for sells.

    Args:
        symbol: Optional filter by symbol (e.g., 'BTC/USD')
        days: Number of days of history (default 7)
        limit: Maximum trades to return (default 100)

    Returns:
        List of trades with details and realized P/L for sells
    """
    from collections import defaultdict

    try:
        from src.execution.alpaca_client import AlpacaClient
        from config_ultra import UltraConfig

        config = UltraConfig()
        client = AlpacaClient(config)
        orders = client.get_all_filled_orders(days=days)

        # First pass: calculate cost basis for P/L calculation
        # Sort chronologically to track running cost basis
        sorted_orders = sorted(orders, key=lambda x: x.get('filled_at', ''))

        # Track cost basis by symbol
        cost_basis = defaultdict(lambda: {'qty': 0, 'cost': 0})

        # Calculate P/L for each sell
        order_pnl = {}  # order_id -> pnl

        for order in sorted_orders:
            sym = order.get('symbol', '')
            side = order.get('side', '')
            qty = order.get('qty', 0)
            price = order.get('price', 0)
            order_id = order.get('id', '')

            if side == 'buy':
                # Add to cost basis
                cost_basis[sym]['qty'] += qty
                cost_basis[sym]['cost'] += qty * price
                order_pnl[order_id] = None  # Buys have no realized P/L
            elif side == 'sell':
                # Calculate realized P/L
                if cost_basis[sym]['qty'] > 0:
                    avg_cost = cost_basis[sym]['cost'] / cost_basis[sym]['qty']
                    pnl = (price - avg_cost) * qty
                    order_pnl[order_id] = round(pnl, 2)
                    # Reduce cost basis
                    cost_basis[sym]['qty'] -= qty
                    cost_basis[sym]['cost'] -= avg_cost * qty
                else:
                    # Sell without prior buy (short or data issue)
                    order_pnl[order_id] = None

        # Filter by symbol if specified
        if symbol:
            symbol_normalized = symbol.replace('/', '').upper()
            orders = [o for o in orders if o.get('symbol', '').upper() == symbol_normalized]

        # Sort by filled_at (most recent first) for display
        orders = sorted(orders, key=lambda x: x.get('filled_at', ''), reverse=True)

        # Format trades for frontend
        trades = []
        for order in orders[:limit]:
            filled_at = order.get('filled_at', '')
            order_id = order.get('id', '')
            side = order.get('side', '')
            pnl = order_pnl.get(order_id)

            raw_sym = order.get('symbol', '')
            symbol = raw_sym if '/' in raw_sym else raw_sym.replace('USD', '/USD')
            trades.append({
                'timestamp': filled_at,
                'time': filled_at[11:19] if len(filled_at) > 19 else '',
                'date': filled_at[:10] if len(filled_at) > 10 else '',
                'symbol': symbol,
                'side': side,
                'qty': order.get('qty', 0),
                'price': order.get('price', 0),
                'value': order.get('value', 0),
                'pnl': pnl,  # Realized P/L for sells, None for buys
                'order_id': order_id
            })

        return {
            "trades": trades,
            "total": len(orders),
            "limit": limit,
            "days": days,
            "symbol_filter": symbol
        }

    except Exception as e:
        logger.error(f"Failed to get trade history: {e}")
        import traceback
        traceback.print_exc()
        return {
            "error": str(e),
            "trades": [],
            "total": 0
        }


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates."""
    await manager.connect(websocket)
    try:
        while True:
            try:
                # Add timeout to prevent indefinite blocking that can exhaust thread pool
                data = await asyncio.wait_for(websocket.receive_text(), timeout=60.0)
                # Could handle commands here in the future
            except asyncio.TimeoutError:
                # Send ping to keep connection alive and verify client is still there
                try:
                    await websocket.send_json({"type": "ping"})
                except Exception:
                    break  # Client disconnected
    except WebSocketDisconnect:
        manager.disconnect(websocket)


# Import bot runner - Grid Trading is now the default
try:
    from config_ultra import UltraConfig
    config = UltraConfig()
    USE_GRID = getattr(config, 'USE_GRID_TRADING', True)
except:
    USE_GRID = True

if USE_GRID:
    try:
        from src.execution.bot_grid import run_grid_bot
        BOT_RUNNER = run_grid_bot
        print("GRID TRADING BOT loaded")
        print("  Strategy: Buy dips, sell rips - no predictions needed")
        print("  Why: Model predicts SIDEWAYS 90%+ = perfect for grids")
    except ImportError as e:
        print(f"Grid bot not available: {e}, falling back to multi-asset")
        from src.execution.bot_multi import run_multi_bot
        BOT_RUNNER = run_multi_bot
else:
    try:
        from src.execution.bot_multi import run_multi_bot
        BOT_RUNNER = run_multi_bot
        print("Multi-Asset Bot loaded (BTC/USD + ETH/USD)")
    except ImportError as e:
        print(f"Multi-asset not available: {e}, using single-asset")
        from src.execution.bot import run_bot
        BOT_RUNNER = run_bot


@app.on_event("startup")
async def startup_event():
    """Start the trading bot on server startup."""
    logger.info("=" * 50)
    logger.info("BlueBird ULTRA API Starting...")
    logger.info("=" * 50)

    # Log lock status (lock is acquired in main.py before server starts)
    try:
        from src.utils.process_lock import is_bot_running, ProcessLock
        if is_bot_running():
            info = ProcessLock.get_service_info("bluebird-bot")
            logger.info(f"Process lock held (PID: {info.get('pid') if info else 'unknown'})")
    except ImportError:
        logger.warning("Process lock module not available")

    try:
        asyncio.create_task(BOT_RUNNER(broadcast_update, broadcast_log))
        logger.info("Trading bot task created successfully")
    except Exception as e:
        logger.error(f"Failed to create bot task: {e}", exc_info=True)

    # Start the notifier watchdog
    try:
        asyncio.create_task(watchdog_loop())
        logger.info("Notifier watchdog started")
    except Exception as e:
        logger.error(f"Failed to start watchdog: {e}", exc_info=True)


@app.on_event("shutdown")
async def shutdown_event():
    """Clean up on server shutdown."""
    logger.info("BlueBird API shutting down...")

    # Release process lock if held
    try:
        from src.utils.process_lock import ProcessLock
        lock = ProcessLock("bluebird-bot", auto_release=False)
        if lock.is_locked():
            lock.release()
            logger.info("Process lock released")
    except Exception as e:
        logger.error(f"Error releasing lock on shutdown: {e}")


async def broadcast_update(data: dict):
    """
    Update state and broadcast to all connected clients.
    
    Expected data structure:
    {
        "status": "running",
        "price": 68000.00,
        "timestamp": "2024-01-01T12:00:00",
        "market": {...},
        "account": {...},
        "positions": [...],
        "ultra": {
            "regime": "TRENDING_UP",
            "strategy": "TREND_FOLLOW",
            "confidence": 0.75,
            "signal": "BUY",
            "should_trade": True,
            "metrics": {...},
            "time_filter": {...},
            "kelly": {...}
        }
    }
    """
    # Convert numpy types to native Python types for JSON serialization
    clean_data = convert_numpy_types(data)

    # Update local state
    system_state.update(clean_data)

    # Broadcast to all WebSocket clients
    await manager.broadcast({
        "type": "update",
        "data": clean_data
    })


async def broadcast_log(message: str):
    """Broadcast log message to all clients."""
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "message": message
    }

    system_state["logs"].append(log_entry)

    # Keep logs limited to last 100
    if len(system_state["logs"]) > 100:
        system_state["logs"].pop(0)

    await manager.broadcast({
        "type": "log",
        "data": log_entry
    })


# ============ DATABASE ENDPOINTS ============

@app.get("/api/db/trades")
async def get_db_trades(
    symbol: str = None,
    side: str = None,
    days: int = 30,
    limit: int = 500
):
    """Get trades from database."""
    from src.database import db as database
    try:
        trades = database.get_trades(symbol=symbol, side=side, days=days, limit=limit)
        stats = database.get_trade_stats(days=days)
        return {
            "trades": trades,
            "stats": stats,
            "count": len(trades)
        }
    except Exception as e:
        return {"error": str(e), "trades": [], "stats": {}}


@app.get("/api/db/equity")
async def get_db_equity(days: int = 30, interval: str = "daily"):
    """Get equity history from database."""
    from src.database import db as database
    try:
        history = database.get_equity_history(days=days, interval=interval)
        stats = database.get_equity_range(days=days)
        return {
            "history": history,
            "stats": stats,
            "count": len(history)
        }
    except Exception as e:
        return {"error": str(e), "history": [], "stats": {}}


@app.get("/api/db/orders")
async def get_db_orders(days: int = 30, symbol: str = None, limit: int = 500):
    """Get orders from database."""
    from src.database import db as database
    try:
        orders = database.get_orders(days=days, symbol=symbol, limit=limit)
        stats = database.get_order_stats()
        return {
            "orders": orders,
            "stats": stats,
            "count": len(orders)
        }
    except Exception as e:
        return {"error": str(e), "orders": [], "stats": {}}


@app.get("/api/db/daily")
async def get_db_daily_summaries(days: int = 30):
    """Get daily summaries from database."""
    from src.database import db as database
    try:
        summaries = database.get_daily_summaries(days=days)
        return {
            "summaries": summaries,
            "count": len(summaries)
        }
    except Exception as e:
        return {"error": str(e), "summaries": []}


@app.get("/api/db/stats")
async def get_db_stats():
    """Get database statistics."""
    from src.database import db as database
    try:
        return database.get_database_stats()
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/db/reconcile")
async def reconcile_database():
    """
    Compare database with Alpaca to verify data integrity.
    Automatically syncs any missing orders.
    """
    from src.database import db as database
    from src.execution.alpaca_client import AlpacaClient
    from config_ultra import UltraConfig

    try:
        config = UltraConfig()
        client = AlpacaClient(config)

        # Get recent orders from Alpaca (last 30 days)
        alpaca_orders = client.get_order_history(days=30, status='closed')
        filled_orders = [o for o in alpaca_orders if 'filled' in str(o.get('status', '')).lower()]

        # Reconcile with database
        results = database.reconcile_with_alpaca(filled_orders)

        # Auto-sync missing orders
        if results['missing_in_db']:
            synced = database.sync_missing_orders(results['missing_in_db'])
            results['auto_synced'] = synced
            # Re-check after sync
            if synced > 0:
                results['synced'] = True
                results['missing_in_db'] = []

        return {
            "status": "synced" if results['synced'] else "discrepancies_found",
            "matched": results['matched'],
            "total_alpaca": results['total_alpaca'],
            "total_db": results['total_db'],
            "missing_in_db": len(results.get('missing_in_db', [])),
            "mismatched": len(results.get('mismatched', [])),
            "auto_synced": results.get('auto_synced', 0),
            "last_check": results['last_check'],
            "details": {
                "missing": results.get('missing_in_db', [])[:10],  # First 10 only
                "mismatched": results.get('mismatched', [])[:10]
            }
        }

    except Exception as e:
        logger.error(f"Reconciliation failed: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e), "status": "error"}


@app.post("/api/db/backfill")
async def backfill_database():
    """
    Backfill database with all available history from Alpaca.
    This pulls:
    - All orders (up to 500, last 30+ days)
    - Portfolio equity history (all available)
    """
    from src.database import db as database
    from src.execution.alpaca_client import AlpacaClient
    from config_ultra import UltraConfig

    results = {
        "orders_backfilled": 0,
        "equity_backfilled": 0,
        "errors": []
    }

    try:
        config = UltraConfig()
        client = AlpacaClient(config)

        # Backfill orders - get all filled orders
        logger.info("Backfilling orders from Alpaca...")
        try:
            orders = client.get_order_history(days=90, status='closed')
            results["orders_backfilled"] = database.backfill_orders_from_alpaca(orders)
            logger.info(f"Backfilled {results['orders_backfilled']} orders")
        except Exception as e:
            results["errors"].append(f"Orders backfill failed: {str(e)}")
            logger.error(f"Orders backfill failed: {e}")

        # Backfill equity - try different periods
        logger.info("Backfilling equity history from Alpaca...")
        for period in ['all', '1A', '3M', '1M']:
            try:
                history = client.get_portfolio_history(period=period, timeframe='1D')
                if history.get('equity') and len(history['equity']) > 0:
                    count = database.backfill_equity_from_alpaca(history)
                    results["equity_backfilled"] = count
                    logger.info(f"Backfilled {count} equity snapshots from period {period}")
                    break
            except Exception as e:
                logger.warning(f"Equity backfill for period {period} failed: {e}")
                continue

        # Get final stats
        results["db_stats"] = database.get_database_stats()

        return results

    except Exception as e:
        logger.error(f"Backfill failed: {e}")
        import traceback
        traceback.print_exc()
        results["errors"].append(str(e))
        return results


# Serve React dashboard (catch-all route - must be LAST)
@app.get("/dashboard")
@app.get("/dashboard/{full_path:path}")
async def serve_dashboard(full_path: str = ""):
    """Serve React SPA dashboard."""
    if os.path.exists(DASHBOARD_PATH):
        index_path = os.path.join(DASHBOARD_PATH, "index.html")
        if os.path.exists(index_path):
            return FileResponse(index_path)
    return {"error": "Dashboard not built. Run: cd dashboard && npm run build"}


# For running directly
if __name__ == "__main__":
    import sys
    import uvicorn

    # Acquire process lock to ensure only one bot runs
    from src.utils.process_lock import acquire_bot_lock, is_bot_running, ProcessLock

    # Check if already running
    if is_bot_running():
        info = ProcessLock.get_service_info("bluebird-bot")
        print("\n" + "=" * 60)
        print(" ERROR: BLUEBIRD is already running!")
        print("=" * 60)
        if info:
            print(f"  PID: {info.get('pid')}")
            print(f"  Started: {info.get('started_at')}")
            print(f"  Uptime: {info.get('uptime_str', 'unknown')}")
        print("\nTo stop the existing instance:")
        print(f"  kill {info.get('pid') if info else '<pid>'}")
        print("=" * 60 + "\n")
        sys.exit(1)

    # Acquire lock (will exit if fails)
    lock = acquire_bot_lock()
    print(f"Process lock acquired (PID: {os.getpid()})")

    uvicorn.run(app, host="0.0.0.0", port=8000)
