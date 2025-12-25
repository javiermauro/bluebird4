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
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

# Persistent state directory (survives reboot, unlike /tmp)
STATE_DIR = os.path.join(PROJECT_ROOT, "data", "state")

from src.notifications.config import NotificationConfig
from src.notifications import templates
from src.database.db import (
    record_sms, is_trade_notified, mark_trade_notified, get_notified_trade_ids,
    queue_failed_sms, get_queued_sms, update_queued_sms,
    update_notifier_heartbeat, get_notifier_status, increment_sms_count,
    get_sms_history, cleanup_old_sms_records
)

# PID and state file paths (for dashboard integration)
PID_FILE = "/tmp/bluebird-notifier.pid"
SMS_COUNT_FILE = "/tmp/bluebird-notifier-count.json"
LOG_FILE = "/tmp/bluebird-notifier.log"
LAST_STARTUP_FILE = os.path.join(STATE_DIR, "notifier-startup.json")  # Persists across reboot

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
        self.last_trade_ids: Set[str] = get_notified_trade_ids()  # Load from DB
        self.last_risk_halted = False
        self.last_drawdown_alert_sent = False
        self.last_daily_summary_date: Optional[datetime] = None
        self.starting_equity: Optional[float] = None

        # Risk overlay state tracking (load from DB if exists)
        db_status = get_notifier_status()
        self.last_risk_overlay_mode: Optional[str] = db_status.get('last_overlay_mode') if db_status else None
        self.risk_overlay_alert_sent = False

        # Watchdog state - detect stale data (bot stream disconnected)
        self.last_bot_update: Optional[datetime] = None
        self.stale_alert_sent = False
        self.stale_threshold_minutes = 5  # Alert if no update for 5 minutes

        # API resilience tracking
        self.api_failure_count = db_status.get('api_failures', 0) if db_status else 0
        self.api_backoff_multiplier = 1
        self.circuit_breaker_open = False
        self.circuit_breaker_opened_at: Optional[datetime] = None

        # SMS counter (loaded from DB via get_notifier_status)
        self.sms_count_today = db_status.get('sms_today', 0) if db_status else 0
        self.sms_count_date: Optional[str] = db_status.get('sms_today_date') if db_status else None

        logger.info(f"Loaded {len(self.last_trade_ids)} notified trade IDs from database")

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

    # Note: _load_sms_count() and _save_sms_count() removed - now using database via
    # get_notifier_status() and increment_sms_count()

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

    def send_sms(self, message: str, force: bool = False, sms_type: str = "alert", max_retries: int = 3) -> bool:
        """
        Send an SMS message via Twilio with retry logic.

        Args:
            message: The message content
            force: If True, ignore quiet hours
            sms_type: Type of SMS (startup, trade, alert, summary, test)
            max_retries: Number of retry attempts on failure

        Returns:
            True if sent successfully, False otherwise
        """
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

        last_error = None
        for attempt in range(max_retries):
            try:
                result = self.twilio_client.messages.create(
                    body=message,
                    from_=self.config.twilio_phone_number,
                    to=self.config.notify_phone_number
                )
                logger.info(f"SMS sent: {result.sid}")

                # Track SMS count in database
                self.sms_count_today += 1
                increment_sms_count()

                # Record in database history
                record_sms(
                    message_type=sms_type,
                    recipient=self.config.notify_phone_number,
                    body_preview=message[:100] if message else "",
                    twilio_sid=result.sid,
                    status="sent",
                    retry_count=attempt
                )

                return True

            except Exception as e:
                last_error = str(e)
                wait_time = (2 ** attempt) * 5  # 5s, 10s, 20s
                logger.warning(f"SMS attempt {attempt + 1}/{max_retries} failed: {e}")

                if attempt < max_retries - 1:
                    logger.info(f"Retrying in {wait_time}s...")
                    time.sleep(wait_time)

        # All retries failed - queue for later
        logger.error(f"SMS failed after {max_retries} attempts, queuing for retry")
        queue_failed_sms(
            body=message,
            message_type=sms_type,
            priority=1 if sms_type == "alert" else 0,
            error_message=last_error
        )

        # Record failed SMS in history
        record_sms(
            message_type=sms_type,
            recipient=self.config.notify_phone_number,
            body_preview=message[:100] if message else "",
            status="queued",
            retry_count=max_retries,
            error_message=last_error
        )

        return False

    def _retry_queued_sms(self) -> None:
        """Retry any queued SMS messages that are due for retry."""
        queued = get_queued_sms(limit=5)
        if not queued:
            return

        logger.info(f"Retrying {len(queued)} queued SMS messages")
        for item in queued:
            try:
                result = self.twilio_client.messages.create(
                    body=item['body'],
                    from_=self.config.twilio_phone_number,
                    to=self.config.notify_phone_number
                )
                logger.info(f"Queued SMS sent successfully: {result.sid}")
                update_queued_sms(item['id'], success=True)

                # Record success in history
                record_sms(
                    message_type=item.get('message_type', 'retry'),
                    recipient=self.config.notify_phone_number,
                    body_preview=item['body'][:100],
                    twilio_sid=result.sid,
                    status="sent",
                    retry_count=item['attempts']
                )

                increment_sms_count()

            except Exception as e:
                logger.warning(f"Queued SMS retry failed: {e}")
                update_queued_sms(item['id'], success=False, error_message=str(e))

    def fetch_stats(self) -> Optional[Dict[str, Any]]:
        """Fetch current stats from the bot's API with exponential backoff and circuit breaker."""
        # Circuit breaker check
        if self.circuit_breaker_open:
            if self.circuit_breaker_opened_at:
                elapsed = (datetime.now() - self.circuit_breaker_opened_at).total_seconds()
                if elapsed < 300:  # 5 minute cooldown
                    logger.debug(f"Circuit breaker open, {300 - elapsed:.0f}s until retry")
                    return None
                else:
                    logger.info("Circuit breaker cooldown expired, attempting reconnection")
                    self.circuit_breaker_open = False

        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                response = requests.get(
                    f"{self.config.bot_api_url}/stats",
                    timeout=10
                )
                response.raise_for_status()

                # Success - reset failure tracking
                if self.api_failure_count > 0:
                    logger.info(f"API reconnected after {self.api_failure_count} failures")
                self.api_failure_count = 0
                self.api_backoff_multiplier = 1
                update_notifier_heartbeat(api_failures=0)

                return response.json()

            except requests.exceptions.ConnectionError:
                self.api_failure_count += 1
                if attempt < max_attempts - 1:
                    wait = min(self.api_backoff_multiplier * 10, 60)
                    logger.warning(f"Cannot connect to bot API (attempt {attempt + 1}/{max_attempts}), retrying in {wait}s")
                    time.sleep(wait)
                    self.api_backoff_multiplier *= 2
                else:
                    logger.warning(f"Cannot connect to bot API after {max_attempts} attempts")

            except Exception as e:
                self.api_failure_count += 1
                logger.error(f"Failed to fetch stats: {e}")
                break

        # Track failures in database
        update_notifier_heartbeat(api_failures=self.api_failure_count)

        # Circuit breaker logic - open after 5 consecutive failures
        if self.api_failure_count >= 5 and not self.circuit_breaker_open:
            self.circuit_breaker_open = True
            self.circuit_breaker_opened_at = datetime.now()
            logger.error(f"Circuit breaker OPEN after {self.api_failure_count} failures - bot may be down")

            # Send alert that bot may be unresponsive
            self.send_sms(
                f"⚠️ BLUEBIRD ALERT\n\n"
                f"Bot API unresponsive!\n"
                f"Failed {self.api_failure_count} consecutive attempts.\n\n"
                f"Check bot status:\n"
                f"  curl http://localhost:8000/health\n\n"
                f"Restart if needed:\n"
                f"  python3 start.py --stop && python3 start.py",
                force=True,
                sms_type="alert"
            )

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
            # Mark all current trades as notified (they existed before we started)
            for order in confirmed:
                order_id = order.get('order_id', '')
                if order_id and not is_trade_notified(order_id):
                    mark_trade_notified(order_id)
            self.last_trade_ids = {o.get('order_id', '') for o in confirmed}
            return

        # Check for new trades
        if current_count > self.last_order_count:
            for order in confirmed:
                order_id = order.get('order_id', '')
                if not order_id:
                    continue

                # Check database to prevent duplicates (survives restart)
                if not is_trade_notified(order_id):
                    # Only notify about recent trades (within last 5 minutes)
                    # This prevents SMS spam when bot restarts and reconciles old trades
                    filled_at = order.get('filled_at', '')
                    if filled_at:
                        try:
                            trade_time = datetime.fromisoformat(filled_at.replace('Z', '+00:00'))
                            if trade_time.tzinfo:
                                trade_time = trade_time.replace(tzinfo=None)
                            age_seconds = (datetime.now() - trade_time).total_seconds()
                            if age_seconds > 300:  # 5 minutes
                                logger.debug(f"Skipping old trade {order_id}: filled {age_seconds:.0f}s ago")
                                mark_trade_notified(order_id)  # Mark as notified to avoid re-checking
                                self.last_trade_ids.add(order_id)
                                continue
                        except Exception as e:
                            logger.warning(f"Could not parse filled_at timestamp: {e}")

                    message = templates.format_trade_alert(order)
                    logger.info(f"New trade detected: {order.get('symbol')} {order.get('side')}")
                    self.send_sms(message, sms_type="trade")

                    # Mark as notified in database
                    mark_trade_notified(order_id)
                    self.last_trade_ids.add(order_id)

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

    def check_risk_overlay(self, stats: Dict[str, Any]) -> None:
        """
        Check for risk overlay mode transitions and send alerts.

        Monitors RISK_OFF entry/exit to notify user of crash protection status.
        """
        # Try to get risk overlay status from stats (broadcast data)
        overlay = stats.get('risk_overlay')

        if not overlay:
            # Fall back to API endpoint
            try:
                response = requests.get(
                    f"{self.config.bot_api_url}/api/risk/overlay",
                    timeout=5
                )
                if response.status_code == 200:
                    overlay = response.json()
            except Exception as e:
                logger.debug(f"Could not fetch risk overlay status: {e}")
                return

        if not overlay:
            return

        current_mode = overlay.get('mode', 'NORMAL')

        # First run - just initialize state
        if self.last_risk_overlay_mode is None:
            self.last_risk_overlay_mode = current_mode
            update_notifier_heartbeat(overlay_mode=current_mode)  # Persist to DB
            logger.info(f"Risk overlay initialized: {current_mode}")
            return

        # Check for mode transitions
        if current_mode != self.last_risk_overlay_mode:
            logger.warning(f"Risk overlay mode changed: {self.last_risk_overlay_mode} -> {current_mode}")

            # Entering RISK_OFF - crash protection activated
            if current_mode == "RISK_OFF":
                message = templates.format_risk_off_entry(overlay)
                logger.warning("RISK_OFF entered - sending alert")
                self.send_sms(message, force=True, sms_type="alert")  # Force send even in quiet hours

            # Exiting RISK_OFF -> RECOVERY
            elif self.last_risk_overlay_mode == "RISK_OFF" and current_mode == "RECOVERY":
                message = templates.format_risk_off_exit(overlay, "RECOVERY")
                logger.info("Entering RECOVERY from RISK_OFF")
                self.send_sms(message, force=True, sms_type="alert")

            # Exiting RISK_OFF -> NORMAL (direct, skipped recovery)
            elif self.last_risk_overlay_mode == "RISK_OFF" and current_mode == "NORMAL":
                message = templates.format_risk_off_exit(overlay, "NORMAL")
                logger.info("Returning to NORMAL from RISK_OFF")
                self.send_sms(message, force=True, sms_type="alert")

            # Exiting RECOVERY -> NORMAL
            elif self.last_risk_overlay_mode == "RECOVERY" and current_mode == "NORMAL":
                message = templates.format_recovery_to_normal(overlay)
                logger.info("Recovery complete - returning to NORMAL")
                self.send_sms(message, sms_type="alert")

            # Relapse: RECOVERY -> RISK_OFF
            elif self.last_risk_overlay_mode == "RECOVERY" and current_mode == "RISK_OFF":
                message = templates.format_risk_off_entry(overlay)
                message = message.replace("Crash protection ACTIVE", "RELAPSE - back to RISK_OFF")
                logger.warning("Relapse during RECOVERY - back to RISK_OFF")
                self.send_sms(message, force=True, sms_type="alert")

            self.last_risk_overlay_mode = current_mode
            update_notifier_heartbeat(overlay_mode=current_mode)  # Persist to DB

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

        # Initialize heartbeat in database
        update_notifier_heartbeat(pid=os.getpid(), status='running')

        # Send startup notification (with cooldown to prevent spam on restarts)
        if self.config.is_configured() and self._should_send_startup_sms():
            self.send_sms(templates.format_startup_message(
                f"Polling every {self.config.poll_interval}s\n"
                f"Quiet hours: {self.config.quiet_hours_start}:00-{self.config.quiet_hours_end}:00"
            ), sms_type="startup")
            self._record_startup_sms()

        # Track cycles for periodic tasks
        cycle_count = 0

        while self.running:
            try:
                cycle_count += 1

                # Update heartbeat in database every cycle
                update_notifier_heartbeat(pid=os.getpid(), status='running')

                # Fetch current stats
                stats = self.fetch_stats()

                if stats:
                    # Initialize starting equity on first successful fetch
                    self.initialize_starting_equity(stats)

                    # Run all checks
                    self.check_for_new_trades(stats)
                    self.check_risk_alerts(stats)
                    self.check_risk_overlay(stats)  # Risk overlay mode transitions
                    self.check_daily_summary(stats)
                    self.check_stale_data(stats)

                # Retry any queued SMS (every cycle)
                if self.config.is_configured():
                    self._retry_queued_sms()

                # Periodic cleanup (every 60 cycles = ~1 hour at 60s poll)
                if cycle_count % 60 == 0:
                    cleanup_old_sms_records(keep_days=30)

                # Wait for next poll
                time.sleep(self.config.poll_interval)

            except KeyboardInterrupt:
                logger.info("Interrupted by user")
                break
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                time.sleep(self.config.poll_interval)

        logger.info("Notification service stopped")

        # Update status in database
        update_notifier_heartbeat(status='stopped')

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