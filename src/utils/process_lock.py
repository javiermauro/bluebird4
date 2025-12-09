"""
Process Lock Manager for BLUEBIRD Trading Bot

Ensures only one instance of each service runs at a time using:
- Atomic file locking (fcntl.lockf)
- PID file validation
- Stale lock detection and cleanup

Usage:
    from src.utils.process_lock import ProcessLock

    lock = ProcessLock("bluebird-bot")
    if not lock.acquire():
        print("Another instance is running!")
        sys.exit(1)

    # ... run your service ...

    lock.release()  # Called automatically on normal exit
"""

import os
import sys
import json
import fcntl
import atexit
import signal
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

# Lock file directory
LOCK_DIR = Path("/tmp/bluebird")
LOCK_DIR.mkdir(exist_ok=True)


class ProcessLock:
    """
    Atomic process lock using file locking.

    Prevents multiple instances of a service from running simultaneously.
    Uses fcntl.lockf for atomic locking (works on macOS and Linux).
    """

    def __init__(self, service_name: str, auto_release: bool = True):
        """
        Initialize process lock.

        Args:
            service_name: Unique name for this service (e.g., "bot", "notifier", "api")
            auto_release: If True, automatically release lock on exit
        """
        self.service_name = service_name
        self.lock_file = LOCK_DIR / f"{service_name}.lock"
        self.pid_file = LOCK_DIR / f"{service_name}.pid"
        self.lock_fd: Optional[int] = None
        self.locked = False
        self.auto_release = auto_release
        self.started_at: Optional[datetime] = None

    def acquire(self, timeout: float = 0) -> bool:
        """
        Attempt to acquire the process lock.

        Args:
            timeout: Not used (kept for API compatibility). Lock is non-blocking.

        Returns:
            True if lock acquired, False if another instance is running
        """
        try:
            # Check for stale lock first
            if self._is_stale_lock():
                logger.info(f"Cleaning up stale lock for {self.service_name}")
                self._cleanup_stale_lock()

            # Open lock file (create if doesn't exist)
            self.lock_fd = os.open(
                str(self.lock_file),
                os.O_RDWR | os.O_CREAT,
                0o644
            )

            # Try to acquire exclusive lock (non-blocking)
            try:
                fcntl.lockf(self.lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except (IOError, OSError) as e:
                # Lock is held by another process
                os.close(self.lock_fd)
                self.lock_fd = None

                # Get info about the running instance
                running_info = self.get_running_instance_info()
                if running_info:
                    logger.error(
                        f"Another {self.service_name} instance is already running "
                        f"(PID: {running_info.get('pid')}, started: {running_info.get('started_at')})"
                    )
                else:
                    logger.error(f"Another {self.service_name} instance is already running")
                return False

            # Lock acquired - write PID file
            self.locked = True
            self.started_at = datetime.now()
            self._write_pid_file()

            # Register cleanup handlers
            if self.auto_release:
                atexit.register(self.release)
                signal.signal(signal.SIGTERM, self._signal_handler)
                signal.signal(signal.SIGINT, self._signal_handler)

            logger.info(f"Process lock acquired for {self.service_name} (PID: {os.getpid()})")
            return True

        except Exception as e:
            logger.error(f"Failed to acquire lock for {self.service_name}: {e}")
            if self.lock_fd is not None:
                os.close(self.lock_fd)
                self.lock_fd = None
            return False

    def release(self) -> None:
        """Release the process lock and clean up files."""
        if not self.locked:
            return

        try:
            # Release the lock
            if self.lock_fd is not None:
                try:
                    fcntl.lockf(self.lock_fd, fcntl.LOCK_UN)
                except:
                    pass
                try:
                    os.close(self.lock_fd)
                except:
                    pass
                self.lock_fd = None

            # Remove PID file
            self._remove_pid_file()

            # Remove lock file
            try:
                self.lock_file.unlink(missing_ok=True)
            except:
                pass

            self.locked = False
            logger.info(f"Process lock released for {self.service_name}")

        except Exception as e:
            logger.error(f"Error releasing lock for {self.service_name}: {e}")

    def _write_pid_file(self) -> None:
        """Write PID and metadata to file."""
        try:
            pid_data = {
                "pid": os.getpid(),
                "service": self.service_name,
                "started_at": self.started_at.isoformat() if self.started_at else None,
                "hostname": os.uname().nodename,
                "python_version": sys.version.split()[0]
            }
            with open(self.pid_file, 'w') as f:
                json.dump(pid_data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to write PID file: {e}")

    def _remove_pid_file(self) -> None:
        """Remove PID file."""
        try:
            self.pid_file.unlink(missing_ok=True)
        except Exception as e:
            logger.error(f"Failed to remove PID file: {e}")

    def _is_stale_lock(self) -> bool:
        """Check if existing lock is stale (process no longer exists)."""
        if not self.pid_file.exists():
            return False

        try:
            with open(self.pid_file, 'r') as f:
                data = json.load(f)
                pid = data.get('pid')

            if pid is None:
                return True

            # Check if process is still running
            try:
                os.kill(pid, 0)  # Signal 0 just checks if process exists
                return False  # Process exists, lock is valid
            except ProcessLookupError:
                return True  # Process doesn't exist, lock is stale
            except PermissionError:
                return False  # Process exists but we can't signal it

        except (json.JSONDecodeError, FileNotFoundError, KeyError):
            return True  # Corrupted or missing file

    def _cleanup_stale_lock(self) -> None:
        """Clean up stale lock and PID files."""
        try:
            self.pid_file.unlink(missing_ok=True)
            self.lock_file.unlink(missing_ok=True)
            logger.info(f"Cleaned up stale lock files for {self.service_name}")
        except Exception as e:
            logger.error(f"Failed to cleanup stale lock: {e}")

    def _signal_handler(self, signum, frame) -> None:
        """Handle termination signals gracefully."""
        logger.info(f"Received signal {signum}, releasing lock...")
        self.release()
        sys.exit(0)

    def get_running_instance_info(self) -> Optional[Dict[str, Any]]:
        """Get information about the currently running instance."""
        try:
            if self.pid_file.exists():
                with open(self.pid_file, 'r') as f:
                    return json.load(f)
        except:
            pass
        return None

    def is_locked(self) -> bool:
        """Check if this lock instance holds the lock."""
        return self.locked

    @classmethod
    def is_service_running(cls, service_name: str) -> bool:
        """
        Check if a service is currently running.

        Args:
            service_name: Name of the service to check

        Returns:
            True if service is running, False otherwise
        """
        pid_file = LOCK_DIR / f"{service_name}.pid"

        if not pid_file.exists():
            return False

        try:
            with open(pid_file, 'r') as f:
                data = json.load(f)
                pid = data.get('pid')

            if pid is None:
                return False

            # Check if process is still running
            os.kill(pid, 0)
            return True
        except (ProcessLookupError, PermissionError, json.JSONDecodeError, FileNotFoundError):
            return False

    @classmethod
    def get_service_info(cls, service_name: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a running service.

        Args:
            service_name: Name of the service

        Returns:
            Dict with service info or None if not running
        """
        if not cls.is_service_running(service_name):
            return None

        pid_file = LOCK_DIR / f"{service_name}.pid"
        try:
            with open(pid_file, 'r') as f:
                data = json.load(f)

            # Add uptime calculation
            if data.get('started_at'):
                started = datetime.fromisoformat(data['started_at'])
                uptime = datetime.now() - started
                data['uptime_seconds'] = uptime.total_seconds()
                data['uptime_str'] = str(uptime).split('.')[0]  # Remove microseconds

            return data
        except:
            return None

    @classmethod
    def kill_service(cls, service_name: str, force: bool = False) -> bool:
        """
        Kill a running service.

        Args:
            service_name: Name of the service to kill
            force: If True, use SIGKILL instead of SIGTERM

        Returns:
            True if service was killed, False otherwise
        """
        pid_file = LOCK_DIR / f"{service_name}.pid"

        if not pid_file.exists():
            return False

        try:
            with open(pid_file, 'r') as f:
                data = json.load(f)
                pid = data.get('pid')

            if pid is None:
                return False

            sig = signal.SIGKILL if force else signal.SIGTERM
            os.kill(pid, sig)
            logger.info(f"Sent signal {sig} to {service_name} (PID: {pid})")
            return True
        except ProcessLookupError:
            # Process already dead, clean up
            try:
                pid_file.unlink(missing_ok=True)
                (LOCK_DIR / f"{service_name}.lock").unlink(missing_ok=True)
            except:
                pass
            return False
        except Exception as e:
            logger.error(f"Failed to kill {service_name}: {e}")
            return False


# Convenience functions for the main services
def acquire_bot_lock() -> ProcessLock:
    """Acquire lock for the trading bot."""
    lock = ProcessLock("bluebird-bot")
    if not lock.acquire():
        print("\n" + "=" * 60)
        print("ERROR: Another BLUEBIRD trading bot is already running!")
        print("=" * 60)
        info = lock.get_running_instance_info()
        if info:
            print(f"  PID: {info.get('pid')}")
            print(f"  Started: {info.get('started_at')}")
        print("\nTo stop the existing instance:")
        print("  1. Press Ctrl+C in the terminal running the bot")
        print("  2. Or run: kill", info.get('pid') if info else '<pid>')
        print("=" * 60 + "\n")
        sys.exit(1)
    return lock


def acquire_notifier_lock() -> ProcessLock:
    """Acquire lock for the SMS notifier."""
    lock = ProcessLock("bluebird-notifier")
    if not lock.acquire():
        print("\n" + "=" * 60)
        print("ERROR: Another BLUEBIRD notifier is already running!")
        print("=" * 60)
        info = lock.get_running_instance_info()
        if info:
            print(f"  PID: {info.get('pid')}")
            print(f"  Started: {info.get('started_at')}")
        print("=" * 60 + "\n")
        sys.exit(1)
    return lock


def is_bot_running() -> bool:
    """Check if the trading bot is running."""
    return ProcessLock.is_service_running("bluebird-bot")


def is_notifier_running() -> bool:
    """Check if the notifier is running."""
    return ProcessLock.is_service_running("bluebird-notifier")


def get_all_services_status() -> Dict[str, Any]:
    """Get status of all BLUEBIRD services."""
    services = ["bluebird-bot", "bluebird-notifier"]
    status = {}

    for service in services:
        info = ProcessLock.get_service_info(service)
        status[service] = {
            "running": info is not None,
            "info": info
        }

    return status
