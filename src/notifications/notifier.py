"""
BLUEBIRD 4.0 - SMS Notification Service

Standalone service that polls the bot's /stats API and sends SMS alerts via Twilio.
100% independent - the trading bot doesn't know this exists.

Usage:
    python src/notifications/notifier.py

Requires .env with:
    TWILIO_ACCOUNT_SID=ACxxxx
    TWILIO_AUTH_TOKEN=xxxx
    TWILIO_PHONE_NUMBER=+1234567890
    NOTIFY_PHONE_NUMBER=+1987654321
"""

import os
import sys
import time
import signal
import logging
import requests
import json
import platform
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Set

# Disable macOS App Nap to prevent the process from being suspended
# This is critical for long-running background services
if platform.system() == 'Darwin':
    try:
        import subprocess
        # Set the process to be exempt from App Nap
        subprocess.run(
            ['defaults', 'write', 'NSGlobalDomain', 'NSAppSleepDisabled', '-bool', 'YES'],
            capture_output=True
        )
    except Exception:
        pass  # Non-critical, continue anyway

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.notifications.config import NotificationConfig
from src.notifications import templates

# PID and state file paths (for dashboard integration)
PID_FILE = "/tmp/bluebird-notifier.pid"
SMS_COUNT_FILE = "/tmp/bluebird-notifier-count.json"
LOG_FILE = "/tmp/bluebird-notifier.log"
LAST_STARTUP_FILE = "/tmp/bluebird-notifier-startup.json"

# Minimum time between startup SMS notifications (in seconds)
# Prevents SMS spam when launchd restarts the service frequently
STARTUP_SMS_COOLDOWN = 3600  # 1 hour

# Configure logging - both console and file
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_FILE)
    ]
)
logger = logging.getLogger('Notifier')


class NotificationService:
    """
    Polls the trading bot's /stats API and sends SMS notifications via Twilio.
    """

    def __init__(self, config: Optional[NotificationConfig] = None):
        self.config = config or NotificationConfig()
        self.running = False
        self.started_at: Optional[datetime] = None

        # State tracking
        self.last_order_count = 0
        self.last_trade_ids: Set[str] = set()
        self.last_risk_halted = False
        self.last_drawdown_alert_sent = False
        self.last_daily_summary_date: Optional[datetime] = None
        self.starting_equity: Optional[float] = None

        # Watchdog state - detect stale data (bot stream disconnected)
        self.last_bot_update: Optional[datetime] = None
        self.stale_alert_sent = False
        self.stale_threshold_minutes = 5  # Alert if no update for 5 minutes

        # SMS counter (persisted to disk)
        self.sms_count_today = 0
        self.sms_count_date: Optional[str] = None
        self._load_sms_count()

        # Twilio client (lazy load)
        self._twilio_client = None

        # Register signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _write_pid_file(self) -> None:
        """Write PID file for process tracking."""
        self.started_at = datetime.now()
        pid_data = {
            "pid": os.getpid(),
            "started_at": self.started_at.isoformat()
        }
        try:
            with open(PID_FILE, 'w') as f:
                json.dump(pid_data, f)
            logger.info(f"PID file written: {PID_FILE} (PID: {os.getpid()})")
        except Exception as e:
            logger.error(f"Failed to write PID file: {e}")

    def _remove_pid_file(self) -> None:
        """Remove PID file on shutdown."""
        try:
            if os.path.exists(PID_FILE):
                os.remove(PID_FILE)
                logger.info("PID file removed")
        except Exception as e:
            logger.error(f"Failed to remove PID file: {e}")

    def _should_send_startup_sms(self) -> bool:
        """Check if enough time has passed since last startup SMS."""
        try:
            if os.path.exists(LAST_STARTUP_FILE):
                with open(LAST_STARTUP_FILE, 'r') as f:
                    data = json.load(f)
                    last_startup = datetime.fromisoformat(data.get('timestamp', '2000-01-01'))
                    elapsed = (datetime.now() - last_startup).total_seconds()
                    if elapsed < STARTUP_SMS_COOLDOWN:
                        logger.info(f"Startup SMS suppressed (last sent {elapsed:.0f}s ago, cooldown is {STARTUP_SMS_COOLDOWN}s)")
                        return False
        except Exception as e:
            logger.debug(f"Could not check startup file: {e}")
        return True

    def _record_startup_sms(self) -> None:
        """Record that a startup SMS was sent."""
        try:
            with open(LAST_STARTUP_FILE, 'w') as f:
                json.dump({'timestamp': datetime.now().isoformat()}, f)
        except Exception as e:
            logger.debug(f"Could not write startup file: {e}")

    def _load_sms_count(self) -> None:
        """Load SMS count from disk."""
        try:
            if os.path.exists(SMS_COUNT_FILE):
                with open(SMS_COUNT_FILE, 'r') as f:
                    data = json.load(f)
                    # Reset count if it's a new day
                    today = datetime.now().strftime('%Y-%m-%d')
                    if data.get('date') == today:
                        self.sms_count_today = data.get('count', 0)
                        self.sms_count_date = today
                    else:
                        self.sms_count_today = 0
                        self.sms_count_date = today
        except Exception as e:
            logger.debug(f"Could not load SMS count: {e}")
            self.sms_count_today = 0
            self.sms_count_date = datetime.now().strftime('%Y-%m-%d')

    def _save_sms_count(self, last_sent: Optional[str] = None) -> None:
        """Save SMS count to disk for dashboard."""
        try:
            today = datetime.now().strftime('%Y-%m-%d')
            # Reset if new day
            if self.sms_count_date != today:
                self.sms_count_today = 0
                self.sms_count_date = today

            data = {
                "count": self.sms_count_today,
                "date": today,
                "last_sent": last_sent or datetime.now().isoformat()
            }
            with open(SMS_COUNT_FILE, 'w') as f:
                json.dump(data, f)
        except Exception as e:
            logger.debug(f"Could not save SMS count: {e}")

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        logger.info("Shutdown signal received")
        self.running = False

    @property
    def twilio_client(self):
        """Lazy-load Twilio client."""
        if self._twilio_client is None:
            try:
                from twilio.rest import Client
                self._twilio_client = Client(
                    self.config.twilio_account_sid,
                    self.config.twilio_auth_token
                )
            except ImportError:
                logger.error("Twilio package not installed. Run: pip install twilio")
                raise
        return self._twilio_client

    def send_sms(self, message: str, force: bool = False, sms_type: str = "alert") -> bool:
        """
        Send an SMS message via Twilio.

        Args:
            message: The message content
            force: If True, ignore quiet hours
            sms_type: Type of SMS (startup, trade, alert, summary, test)

        Returns:
            True if sent successfully, False otherwise
        """
        from src.notifications.config import add_sms_to_history

        # Check quiet hours
        current_hour = datetime.now().hour
        if not force and self.config.is_quiet_hours(current_hour):
            logger.debug(f"Quiet hours ({current_hour}:00) - message suppressed")
            return False

        # Check configuration
        if not self.config.is_configured():
            logger.warning("Twilio not configured - SMS not sent")
            logger.debug(f"Message would have been: {message}")
            return False

        try:
            result = self.twilio_client.messages.create(
                body=message,
                from_=self.config.twilio_phone_number,
                to=self.config.notify_phone_number
            )
            logger.info(f"SMS sent: {result.sid}")

            # Track SMS count
            self.sms_count_today += 1
            self._save_sms_count(datetime.now().isoformat())

            # Add to SMS history
            add_sms_to_history(
                sms_type=sms_type,
                preview=message,
                status="sent",
                recipient=self.config.notify_phone_number
            )

            return True
        except Exception as e:
            logger.error(f"Failed to send SMS: {e}")
            # Track failed SMS in history too
            add_sms_to_history(
                sms_type=sms_type,
                preview=message,
                status="failed",
                recipient=self.config.notify_phone_number
            )
            return False

    def fetch_stats(self) -> Optional[Dict[str, Any]]:
        """Fetch current stats from the bot's API."""
        try:
            response = requests.get(
                f"{self.config.bot_api_url}/stats",
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.ConnectionError:
            logger.warning("Cannot connect to bot API - is the bot running?")
            return None
        except Exception as e:
            logger.error(f"Failed to fetch stats: {e}")
            return None

    def check_for_new_trades(self, stats: Dict[str, Any]) -> None:
        """Check for new trade executions and send alerts."""
        if not self.config.alert_on_trade:
            return

        orders = stats.get('orders', {})
        confirmed = orders.get('confirmed', [])
        stats_info = orders.get('stats', {})
        current_count = stats_info.get('total_confirmed', 0)

        # First run - initialize state
        if self.last_order_count == 0:
            self.last_order_count = current_count
            self.last_trade_ids = {o.get('order_id', '') for o in confirmed}
            return

        # Check for new trades
        if current_count > self.last_order_count:
            # Find new orders
            current_ids = {o.get('order_id', '') for o in confirmed}
            new_ids = current_ids - self.last_trade_ids

            for order in confirmed:
                if order.get('order_id', '') in new_ids:
                    message = templates.format_trade_alert(order)
                    logger.info(f"New trade detected: {order.get('symbol')} {order.get('side')}")
                    self.send_sms(message, sms_type="trade")

            self.last_trade_ids = current_ids
            self.last_order_count = current_count

    def check_risk_alerts(self, stats: Dict[str, Any]) -> None:
        """Check for risk conditions and send alerts."""
        if not self.config.alert_on_risk:
            return

        risk = stats.get('risk', {})

        # Check circuit breaker / trading halted
        halted = risk.get('trading_halted', False)
        if halted and not self.last_risk_halted:
            message = templates.format_risk_alert(
                risk,
                reason=risk.get('halt_reason', 'Circuit breaker triggered')
            )
            logger.warning("Trading halted - sending alert")
            self.send_sms(message, force=True, sms_type="alert")  # Force send even in quiet hours
        self.last_risk_halted = halted

        # Check drawdown threshold
        drawdown = risk.get('drawdown_pct', 0)
        if drawdown >= self.config.drawdown_alert_threshold:
            if not self.last_drawdown_alert_sent:
                message = templates.format_risk_alert(
                    risk,
                    reason=f"Drawdown exceeded {self.config.drawdown_alert_threshold}%"
                )
                logger.warning(f"Drawdown alert: {drawdown:.2f}%")
                self.send_sms(message, sms_type="alert")
                self.last_drawdown_alert_sent = True
        else:
            # Reset flag when drawdown recovers
            self.last_drawdown_alert_sent = False

    def check_daily_summary(self, stats: Dict[str, Any]) -> None:
        """Check if it's time to send daily summary."""
        if not self.config.daily_summary_enabled:
            return

        now = datetime.now()
        current_hour = now.hour

        # Check if it's summary time and we haven't sent today
        if current_hour == self.config.daily_summary_hour:
            today = now.date()
            if self.last_daily_summary_date != today:
                if self.starting_equity:
                    message = templates.format_daily_summary(stats, self.starting_equity)
                    logger.info("Sending daily summary")
                    if self.send_sms(message, sms_type="summary"):
                        self.last_daily_summary_date = today

    def check_stale_data(self, stats: Dict[str, Any]) -> None:
        """
        Watchdog check - alert if bot data is stale (stream disconnected).

        This catches the scenario where the bot process is running but
        the Alpaca WebSocket stream silently disconnected.
        """
        # Get last_update from health endpoint
        try:
            response = requests.get(f"{self.config.bot_api_url}/health", timeout=5)
            if response.status_code == 200:
                health = response.json()
                last_update_str = health.get('last_update')

                if last_update_str:
                    # Parse the timestamp (format: "2025-12-11 05:16:00+00:00" or "...Z")
                    clean_ts = last_update_str.replace('+00:00', '').replace('Z', '')
                    last_update = datetime.fromisoformat(clean_ts)
                    now = datetime.utcnow()

                    minutes_stale = (now - last_update).total_seconds() / 60

                    if minutes_stale >= self.stale_threshold_minutes:
                        if not self.stale_alert_sent:
                            message = (
                                f"⚠️ BLUEBIRD WATCHDOG ALERT\n\n"
                                f"Bot data is STALE!\n"
                                f"Last update: {int(minutes_stale)} min ago\n"
                                f"Stream may be disconnected.\n\n"
                                f"Action: Check bot logs and restart if needed:\n"
                                f"python3 start.py --stop && python3 start.py"
                            )
                            logger.warning(f"Stale data detected: {minutes_stale:.1f} minutes old")
                            self.send_sms(message, force=True, sms_type="alert")
                            self.stale_alert_sent = True
                    else:
                        # Data is fresh - reset alert flag
                        if self.stale_alert_sent:
                            logger.info("Bot data is fresh again - resetting stale alert")
                            self.send_sms(
                                f"✅ BLUEBIRD RECOVERED\n\nBot is receiving data again.",
                                force=True, sms_type="alert"
                            )
                        self.stale_alert_sent = False
                        self.last_bot_update = last_update

        except Exception as e:
            logger.error(f"Failed to check bot health: {e}")

    def initialize_starting_equity(self, stats: Dict[str, Any]) -> None:
        """Initialize starting equity from stats or risk data."""
        if self.starting_equity is None:
            # Try to get from risk data first (more accurate)
            risk = stats.get('risk', {})
            starting = risk.get('starting_equity', 0)

            if starting > 0:
                self.starting_equity = starting
            else:
                # Fall back to current equity
                account = stats.get('account', {})
                self.starting_equity = account.get('equity', 0)

            logger.info(f"Starting equity initialized: ${self.starting_equity:,.2f}")

    def run(self) -> None:
        """Main loop - poll stats and send notifications."""
        logger.info("=" * 50)
        logger.info("BLUEBIRD Notification Service Starting")
        logger.info("=" * 50)
        logger.info(str(self.config))

        # Write PID file for dashboard tracking
        self._write_pid_file()

        if not self.config.is_configured():
            logger.error("Twilio not configured! Add credentials to .env:")
            logger.error("  TWILIO_ACCOUNT_SID=ACxxxx")
            logger.error("  TWILIO_AUTH_TOKEN=xxxx")
            logger.error("  TWILIO_PHONE_NUMBER=+1234567890")
            logger.error("  NOTIFY_PHONE_NUMBER=+1987654321")
            logger.error("")
            logger.error("Running in dry-run mode (no SMS will be sent)")

        self.running = True

        # Send startup notification (with cooldown to prevent spam on restarts)
        if self.config.is_configured() and self._should_send_startup_sms():
            self.send_sms(templates.format_startup_message(
                f"Polling every {self.config.poll_interval}s\n"
                f"Quiet hours: {self.config.quiet_hours_start}:00-{self.config.quiet_hours_end}:00"
            ), sms_type="startup")
            self._record_startup_sms()

        while self.running:
            try:
                # Fetch current stats
                stats = self.fetch_stats()

                if stats:
                    # Initialize starting equity on first successful fetch
                    self.initialize_starting_equity(stats)

                    # Run all checks
                    self.check_for_new_trades(stats)
                    self.check_risk_alerts(stats)
                    self.check_daily_summary(stats)
                    self.check_stale_data(stats)

                # Wait for next poll
                time.sleep(self.config.poll_interval)

            except KeyboardInterrupt:
                logger.info("Interrupted by user")
                break
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                time.sleep(self.config.poll_interval)

        logger.info("Notification service stopped")

        # Clean up PID file
        self._remove_pid_file()

        # Send shutdown notification
        if self.config.is_configured():
            self.send_sms(templates.format_shutdown_message(), force=True, sms_type="startup")


def main():
    """Entry point for the notification service."""
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='BLUEBIRD SMS Notification Service')
    parser.add_argument('--test', action='store_true', help='Send a test SMS and exit')
    parser.add_argument('--dry-run', action='store_true', help='Run without sending SMS')
    args = parser.parse_args()

    # Check for single-instance (skip for test mode)
    if not args.test:
        try:
            from src.utils.process_lock import acquire_notifier_lock, is_notifier_running, ProcessLock

            if is_notifier_running():
                info = ProcessLock.get_service_info("bluebird-notifier")
                print("\n" + "=" * 60)
                print(" BLUEBIRD Notifier is already running!")
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

            # Acquire the process lock
            lock = acquire_notifier_lock()
            logger.info(f"Process lock acquired (PID: {os.getpid()})")
        except ImportError:
            logger.warning("Process lock module not available, running without lock")
            lock = None

    config = NotificationConfig()

    if args.test:
        # Test mode - send a test message and exit
        service = NotificationService(config)
        stats = service.fetch_stats()
        if stats:
            message = templates.format_quick_update(
                stats,
                stats.get('risk', {}).get('starting_equity', 89309)
            )
            print("Test message:")
            print("-" * 40)
            print(message)
            print("-" * 40)
            if config.is_configured() and not args.dry_run:
                if service.send_sms(message, force=True):
                    print("SMS sent successfully!")
                else:
                    print("Failed to send SMS")
            else:
                print("(Dry run - SMS not sent)")
        else:
            print("Could not fetch stats from bot API")
        return

    # Normal operation
    try:
        service = NotificationService(config)
        service.run()
    finally:
        # Release lock on exit
        if 'lock' in dir() and lock is not None:
            lock.release()


if __name__ == '__main__':
    main()