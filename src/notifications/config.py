"""
Notification Configuration

Load Twilio credentials from .env and define notification settings.
"""

import os
import json
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, List
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Project root for persistent state files (survives reboot, unlike /tmp)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
STATE_DIR = os.path.join(PROJECT_ROOT, "data", "state")

# File paths for persistent storage
SMS_HISTORY_FILE = os.path.join(STATE_DIR, "notifier-history.json")  # Legacy, now using DB
SETTINGS_FILE = os.path.join(STATE_DIR, "notifier-settings.json")

# Default notification settings (can be modified via dashboard)
DEFAULT_SETTINGS = {
    "quiet_hours_enabled": True,
    "quiet_hours_start": 23,
    "quiet_hours_end": 7,
    "notify_on_trade": True,
    "notify_on_alert": True,
    "notify_daily_summary": True,
    "notify_on_startup": True
}


def load_settings() -> Dict[str, Any]:
    """Load notification settings from file, or return defaults."""
    try:
        if os.path.exists(SETTINGS_FILE):
            with open(SETTINGS_FILE, 'r') as f:
                saved = json.load(f)
                # Merge with defaults to ensure all keys exist
                return {**DEFAULT_SETTINGS, **saved}
    except Exception:
        pass
    return DEFAULT_SETTINGS.copy()


def save_settings(settings: Dict[str, Any]) -> bool:
    """Save notification settings to file."""
    try:
        with open(SETTINGS_FILE, 'w') as f:
            json.dump(settings, f, indent=2)
        return True
    except Exception:
        return False


def load_sms_history() -> List[Dict[str, Any]]:
    """Load SMS history from file."""
    try:
        if os.path.exists(SMS_HISTORY_FILE):
            with open(SMS_HISTORY_FILE, 'r') as f:
                return json.load(f)
    except Exception:
        pass
    return []


def save_sms_history(history: List[Dict[str, Any]]) -> bool:
    """Save SMS history to file (keeps last 20 entries)."""
    try:
        # Keep only last 20 entries
        history = history[-20:]
        with open(SMS_HISTORY_FILE, 'w') as f:
            json.dump(history, f, indent=2)
        return True
    except Exception:
        return False


def add_sms_to_history(sms_type: str, preview: str, status: str, recipient: str) -> Dict[str, Any]:
    """Add a new SMS entry to history."""
    from datetime import datetime
    import uuid

    entry = {
        "id": f"sms_{uuid.uuid4().hex[:8]}",
        "timestamp": datetime.now().isoformat(),
        "type": sms_type,  # startup, trade, alert, summary, test
        "preview": preview[:50] + "..." if len(preview) > 50 else preview,
        "status": status,  # sent, failed
        "recipient": mask_phone(recipient)
    }

    history = load_sms_history()
    history.append(entry)
    save_sms_history(history)

    return entry


def mask_phone(phone: str) -> str:
    """Mask phone number for privacy (show last 4 digits)."""
    if not phone or len(phone) < 4:
        return "••••"
    return f"+1••••••{phone[-4:]}"


@dataclass
class NotificationConfig:
    """Configuration for the notification service."""

    # Twilio credentials (from .env)
    twilio_account_sid: str = os.getenv("TWILIO_ACCOUNT_SID", "")
    twilio_auth_token: str = os.getenv("TWILIO_AUTH_TOKEN", "")
    twilio_phone_number: str = os.getenv("TWILIO_PHONE_NUMBER", "")
    notify_phone_number: str = os.getenv("NOTIFY_PHONE_NUMBER", "")

    # Bot API endpoint
    bot_api_url: str = os.getenv("BOT_API_URL", "http://localhost:8000")

    # Polling settings
    poll_interval: int = 60  # seconds between polls

    # Quiet hours (no SMS during these hours)
    quiet_hours_start: int = 23  # 11 PM
    quiet_hours_end: int = 7     # 7 AM

    # Daily summary
    daily_summary_hour: int = 8  # 8 AM local time
    daily_summary_enabled: bool = True

    # Alert thresholds
    drawdown_alert_threshold: float = 2.0  # Alert if drawdown > 2%

    # Alert toggles
    alert_on_trade: bool = True
    alert_on_risk: bool = True
    alert_on_circuit_breaker: bool = True

    # Starting equity for P&L calculation (will be fetched on startup)
    starting_equity: Optional[float] = None

    def is_configured(self) -> bool:
        """Check if Twilio credentials are configured."""
        return all([
            self.twilio_account_sid,
            self.twilio_auth_token,
            self.twilio_phone_number,
            self.notify_phone_number
        ])

    def is_quiet_hours(self, hour: int) -> bool:
        """Check if current hour is within quiet hours."""
        if self.quiet_hours_start > self.quiet_hours_end:
            # Spans midnight (e.g., 23:00 to 07:00)
            return hour >= self.quiet_hours_start or hour < self.quiet_hours_end
        else:
            # Same day range
            return self.quiet_hours_start <= hour < self.quiet_hours_end

    def __str__(self) -> str:
        return f"""NotificationConfig:
  Twilio configured: {self.is_configured()}
  Bot API: {self.bot_api_url}
  Poll interval: {self.poll_interval}s
  Quiet hours: {self.quiet_hours_start}:00 - {self.quiet_hours_end}:00
  Daily summary: {self.daily_summary_hour}:00 (enabled: {self.daily_summary_enabled})
  Drawdown threshold: {self.drawdown_alert_threshold}%
  Alert on trade: {self.alert_on_trade}
  Alert on risk: {self.alert_on_risk}"""
