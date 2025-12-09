"""
BLUEBIRD 4.0 - SMS Notification Module

100% independent notification service that polls the bot's /stats API
and sends SMS alerts via Twilio.
"""

from .notifier import NotificationService
from .config import NotificationConfig

__all__ = ['NotificationService', 'NotificationConfig']
