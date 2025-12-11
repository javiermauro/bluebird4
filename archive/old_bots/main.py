#!/usr/bin/env python3
"""
BLUEBIRD 4.0 Trading Bot - Main Entry Point

This script starts the FastAPI server which runs the trading bot.
It ensures only one instance can run at a time using process locking.
"""

import sys
import uvicorn
import logging

# Setup logging before importing other modules
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


def check_port_available(port: int) -> bool:
    """Check if the port is available for binding."""
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(('0.0.0.0', port))
            return True
        except OSError:
            return False


def main():
    """Main entry point with single-instance protection."""
    PORT = 8000

    # Import here to avoid circular imports
    from src.utils.process_lock import acquire_bot_lock, is_bot_running

    # Check if already running (quick check before acquiring lock)
    if is_bot_running():
        from src.utils.process_lock import ProcessLock
        info = ProcessLock.get_service_info("bluebird-bot")
        print("\n" + "=" * 60)
        print(" BLUEBIRD is already running!")
        print("=" * 60)
        if info:
            print(f"  PID: {info.get('pid')}")
            print(f"  Started: {info.get('started_at')}")
            print(f"  Uptime: {info.get('uptime_str', 'unknown')}")
        print("\nTo stop the existing instance:")
        print("  - Press Ctrl+C in its terminal")
        print(f"  - Or run: kill {info.get('pid') if info else '<pid>'}")
        print("=" * 60 + "\n")
        sys.exit(1)

    # Check if port is available
    if not check_port_available(PORT):
        print("\n" + "=" * 60)
        print(f" ERROR: Port {PORT} is already in use!")
        print("=" * 60)
        print("\nAnother service is using this port.")
        print(f"Check with: lsof -i :{PORT}")
        print("=" * 60 + "\n")
        sys.exit(1)

    # Acquire the process lock (will exit if another instance is running)
    lock = acquire_bot_lock()

    print("\n" + "=" * 60)
    print(" BLUEBIRD 4.0 - Intelligent Grid Trading")
    print("=" * 60)
    print(f"  Server: http://localhost:{PORT}")
    print(f"  Dashboard: http://localhost:{PORT}/dashboard")
    print(f"  API Docs: http://localhost:{PORT}/docs")
    print("=" * 60)
    print("  Press Ctrl+C to stop")
    print("=" * 60 + "\n")

    try:
        # Run the FastAPI app
        # reload=False for stability (reload would restart the bot)
        uvicorn.run(
            "src.api.server:app",
            host="0.0.0.0",
            port=PORT,
            reload=False,
            log_level="info"
        )
    except KeyboardInterrupt:
        logger.info("Shutting down gracefully...")
    except Exception as e:
        logger.error(f"Server error: {e}")
        raise
    finally:
        # Release lock (also handled by atexit, but explicit is better)
        lock.release()
        logger.info("BLUEBIRD stopped")


if __name__ == "__main__":
    main()
