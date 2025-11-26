"""
BLUEBIRD ULTRA API Server

FastAPI server with WebSocket for real-time dashboard updates.
Now includes full Ultra system state broadcasting.
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
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
    "last_trade": None
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


# Import bot runner - Try multi-asset first, fall back to single
try:
    from src.execution.bot_multi import run_multi_bot
    BOT_RUNNER = run_multi_bot
    print("✅ Multi-Asset Bot loaded (BTC/USD + ETH/USD)")
except ImportError as e:
    print(f"⚠️ Multi-asset not available: {e}, using single-asset")
    from src.execution.bot import run_bot
    BOT_RUNNER = run_bot


@app.on_event("startup")
async def startup_event():
    """Start the trading bot on server startup."""
    logger.info("=" * 50)
    logger.info("BlueBird ULTRA API Starting...")
    logger.info("=" * 50)
    
    try:
        asyncio.create_task(BOT_RUNNER(broadcast_update, broadcast_log))
        logger.info("Trading bot task created successfully")
    except Exception as e:
        logger.error(f"Failed to create bot task: {e}", exc_info=True)


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
