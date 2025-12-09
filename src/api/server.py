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
from datetime import datetime
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
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
                await connection.send_json(message)
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
    return system_state


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "connections": len(manager.active_connections),
        "regime": system_state["ultra"]["regime"],
        "last_update": system_state["timestamp"]
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
    return system_state["ultra"]


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
    return system_state["orders"]["stats"]


@app.get("/api/orders/reconciliation")
async def get_reconciliation_status():
    """
    Get reconciliation status between bot and Alpaca.

    Returns:
        Sync status, matched count, and any discrepancies
    """
    return system_state["orders"]["reconciliation"]


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
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    notifier_script = os.path.join(project_root, "src", "notifications", "notifier.py")

    try:
        process = subprocess.Popen(
            ["python3", notifier_script],
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
    - If trades happened but no SMS in 10+ minutes, notifier is stale
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

        # Get last SMS time
        last_sms = notifier.get("last_sms_sent")
        if not last_sms:
            # No SMS ever sent, check if there are trades
            if len(system_state.get("orders", {}).get("confirmed", [])) > 0:
                result["healthy"] = False
                result["reason"] = "Trades exist but no SMS ever sent"
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

        # Count trades since last SMS
        confirmed_orders = system_state.get("orders", {}).get("confirmed", [])
        trades_since_sms = 0

        for order in confirmed_orders:
            filled_at = order.get("filled_at", "")
            if filled_at:
                try:
                    trade_time = datetime.fromisoformat(filled_at.replace('Z', '+00:00'))
                    if trade_time.tzinfo:
                        trade_time = trade_time.replace(tzinfo=None)
                    if trade_time > last_sms_time:
                        trades_since_sms += 1
                except:
                    pass

        result["trades_since_last_sms"] = trades_since_sms

        # If trades happened 10+ minutes ago but no SMS, notifier is stale
        if trades_since_sms > 0 and minutes_since_sms > 10:
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

    # Start new notifier
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    notifier_script = os.path.join(project_root, "src", "notifications", "notifier.py")

    try:
        process = subprocess.Popen(
            ["python3", notifier_script],
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


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates."""
    await manager.connect(websocket)
    try:
        while True:
            # Keep connection alive
            data = await websocket.receive_text()
            # Could handle commands here in the future
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
    # Update local state
    system_state.update(data)
    
    # Broadcast to all WebSocket clients
    await manager.broadcast({
        "type": "update",
        "data": data
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
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
