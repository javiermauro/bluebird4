#!/usr/bin/env python3
"""
BLUEBIRD 4.0 - Unified Startup Script

Starts all BLUEBIRD services:
  - Trading Bot (FastAPI server on port 8000)
  - Dashboard (Vite dev server on port 5173)
  - SMS Notifier (optional, polls bot API)

Usage:
    python start.py              # Start bot + dashboard
    python start.py --all        # Start bot + dashboard + notifier
    python start.py --no-dashboard  # Start bot only
    python start.py --status     # Show status of all services
    python start.py --stop       # Stop all services
"""

import os
import sys
import time
import signal
import subprocess
import argparse
import logging
from pathlib import Path
from typing import Optional, List, Dict

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger('BLUEBIRD')

# Project paths
PROJECT_ROOT = Path(__file__).parent.absolute()
DASHBOARD_DIR = PROJECT_ROOT / "dashboard"

# Add project to path
sys.path.insert(0, str(PROJECT_ROOT))


class ServiceManager:
    """Manages all BLUEBIRD services."""

    def __init__(self):
        self.processes: Dict[str, subprocess.Popen] = {}
        self.running = False

    def check_port(self, port: int) -> bool:
        """Check if a port is available."""
        import socket
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(('0.0.0.0', port))
                return True
            except OSError:
                return False

    def wait_for_port(self, port: int, timeout: int = 30) -> bool:
        """Wait for a port to become available (service started)."""
        import socket
        start = time.time()
        while time.time() - start < timeout:
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.settimeout(1)
                    s.connect(('localhost', port))
                    return True
            except (socket.error, socket.timeout):
                time.sleep(0.5)
        return False

    def start_bot(self) -> bool:
        """Start the grid trading bot."""
        logger.info("Starting Grid Trading Bot...")

        if not self.check_port(8000):
            logger.error("Port 8000 is already in use!")
            return False

        # Run the grid bot server with caffeinate to prevent sleep
        proc = subprocess.Popen(
            ["caffeinate", "-i", sys.executable, "-m", "src.api.server"],
            cwd=str(PROJECT_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
        self.processes['bot'] = proc

        # Wait for bot to start
        if self.wait_for_port(8000, timeout=30):
            logger.info("Grid Trading Bot started on http://localhost:8000")
            return True
        else:
            logger.error("Grid Trading Bot failed to start")
            return False

    def start_dashboard(self) -> bool:
        """Start the dashboard dev server."""
        logger.info("Starting Dashboard...")

        if not DASHBOARD_DIR.exists():
            logger.error(f"Dashboard directory not found: {DASHBOARD_DIR}")
            return False

        # Check if node_modules exists
        if not (DASHBOARD_DIR / "node_modules").exists():
            logger.info("Installing dashboard dependencies...")
            result = subprocess.run(
                ["npm", "install"],
                cwd=str(DASHBOARD_DIR),
                capture_output=True,
                text=True
            )
            if result.returncode != 0:
                logger.error(f"npm install failed: {result.stderr}")
                return False

        proc = subprocess.Popen(
            ["npm", "run", "dev"],
            cwd=str(DASHBOARD_DIR),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
        self.processes['dashboard'] = proc

        # Wait for dashboard to start
        if self.wait_for_port(5173, timeout=30):
            logger.info("Dashboard started on http://localhost:5173")
            return True
        else:
            logger.error("Dashboard failed to start")
            return False

    def start_notifier(self) -> bool:
        """Start the SMS notifier."""
        logger.info("Starting SMS Notifier...")

        # Use caffeinate to prevent macOS from killing the process due to App Nap
        # -i flag prevents idle sleep
        proc = subprocess.Popen(
            ["caffeinate", "-i", sys.executable, "src/notifications/notifier.py"],
            cwd=str(PROJECT_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
        self.processes['notifier'] = proc

        # Give it a moment to start
        time.sleep(2)

        if proc.poll() is None:
            logger.info("SMS Notifier started")
            return True
        else:
            logger.error("SMS Notifier failed to start")
            return False

    def stop_all(self) -> None:
        """Stop all managed services."""
        logger.info("Stopping all services...")

        for name, proc in self.processes.items():
            if proc and proc.poll() is None:
                logger.info(f"Stopping {name}...")
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()

        # Also kill any orphaned processes
        self._kill_orphans()

        self.processes.clear()
        logger.info("All services stopped")

    def _kill_orphans(self) -> None:
        """Kill any orphaned BLUEBIRD processes."""
        import signal as sig

        # Try to kill by lock files first
        try:
            from src.utils.process_lock import ProcessLock
            for service in ["bluebird-bot", "bluebird-notifier"]:
                ProcessLock.kill_service(service)
        except:
            pass

        # Also try pkill
        subprocess.run(["pkill", "-f", "notifier.py"], capture_output=True)
        subprocess.run(["pkill", "-f", "vite.*dashboard"], capture_output=True)

    def get_status(self) -> Dict[str, dict]:
        """Get status of all services."""
        status = {}

        # Check bot
        try:
            import requests
            resp = requests.get("http://localhost:8000/health", timeout=2)
            status['bot'] = {
                'running': True,
                'url': 'http://localhost:8000',
                'api_docs': 'http://localhost:8000/docs'
            }
        except:
            status['bot'] = {'running': False}

        # Check dashboard
        try:
            import requests
            resp = requests.get("http://localhost:5173", timeout=2)
            status['dashboard'] = {
                'running': True,
                'url': 'http://localhost:5173'
            }
        except:
            status['dashboard'] = {'running': False}

        # Check notifier via lock file
        try:
            from src.utils.process_lock import ProcessLock
            info = ProcessLock.get_service_info("bluebird-notifier")
            if info:
                status['notifier'] = {
                    'running': True,
                    'pid': info.get('pid'),
                    'uptime': info.get('uptime_str')
                }
            else:
                status['notifier'] = {'running': False}
        except:
            status['notifier'] = {'running': False}

        return status

    def print_status(self) -> None:
        """Print status of all services."""
        status = self.get_status()

        print("\n" + "=" * 60)
        print(" BLUEBIRD 4.0 - Service Status")
        print("=" * 60)

        for service, info in status.items():
            if info['running']:
                print(f"  {service.upper():12} ✓ RUNNING")
                for key, val in info.items():
                    if key != 'running':
                        print(f"               {key}: {val}")
            else:
                print(f"  {service.upper():12} ✗ stopped")

        print("=" * 60 + "\n")

    def run(self, start_dashboard: bool = True, start_notifier: bool = False) -> None:
        """Run all services and wait."""
        self.running = True

        # Register signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        print("\n" + "=" * 60)
        print(" BLUEBIRD 4.0 - Grid Trading Bot")
        print("=" * 60)

        # Start bot first
        if not self.start_bot():
            self.stop_all()
            sys.exit(1)

        # Start dashboard
        if start_dashboard:
            if not self.start_dashboard():
                logger.warning("Dashboard failed to start, continuing without it")

        # Start notifier
        if start_notifier:
            if not self.start_notifier():
                logger.warning("Notifier failed to start, continuing without it")

        # Print URLs
        print("\n" + "-" * 60)
        print(" Services Running:")
        print("-" * 60)
        print(f"  Grid Bot:   http://localhost:8000")
        print(f"  API Docs:   http://localhost:8000/docs")
        if start_dashboard:
            print(f"  Dashboard:  http://localhost:5173")
        if start_notifier:
            print(f"  Notifier:   Running (check logs)")
        print("-" * 60)
        print("  Press Ctrl+C to stop all services")
        print("=" * 60 + "\n")

        # Monitor processes
        try:
            while self.running:
                # Check if any critical process died
                if 'bot' in self.processes:
                    if self.processes['bot'].poll() is not None:
                        logger.error("Trading Bot died unexpectedly!")
                        break

                time.sleep(1)
        except KeyboardInterrupt:
            pass

        self.stop_all()

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info("Shutdown signal received")
        self.running = False


def main():
    parser = argparse.ArgumentParser(
        description='BLUEBIRD 4.0 - Unified Startup Script',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python start.py              Start bot + dashboard
  python start.py --all        Start bot + dashboard + notifier
  python start.py --no-dashboard  Start bot only
  python start.py --status     Show status of all services
  python start.py --stop       Stop all services
        """
    )
    parser.add_argument('--all', action='store_true',
                        help='Start all services including notifier')
    parser.add_argument('--no-dashboard', action='store_true',
                        help='Skip starting the dashboard')
    parser.add_argument('--notifier', action='store_true',
                        help='Include SMS notifier')
    parser.add_argument('--status', action='store_true',
                        help='Show status of all services')
    parser.add_argument('--stop', action='store_true',
                        help='Stop all running services')

    args = parser.parse_args()

    manager = ServiceManager()

    if args.status:
        manager.print_status()
        return

    if args.stop:
        manager.stop_all()
        # Clean up lock files
        try:
            import shutil
            lock_dir = Path("/tmp/bluebird")
            if lock_dir.exists():
                for f in lock_dir.glob("*.lock"):
                    f.unlink()
                for f in lock_dir.glob("*.pid"):
                    f.unlink()
        except:
            pass
        print("All services stopped.")
        return

    # Determine what to start
    start_dashboard = not args.no_dashboard
    start_notifier = args.all or args.notifier

    manager.run(start_dashboard=start_dashboard, start_notifier=start_notifier)


if __name__ == "__main__":
    main()
