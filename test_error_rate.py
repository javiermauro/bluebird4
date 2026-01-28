#!/usr/bin/env python3
"""
Test script for Phase 6: Error Rate Monitoring

Creates test errors in the bot log to verify the error rate monitoring works.
"""

import logging
import time
from pathlib import Path

# Configure logging to write to bot log
LOG_FILE = "/tmp/bluebird-live-bot.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE)
    ]
)

def inject_test_errors(count: int = 15):
    """Inject test errors into the bot log."""
    test_logger = logging.getLogger("TestErrorRate")

    print(f"Injecting {count} test errors into {LOG_FILE}...")

    for i in range(count):
        test_logger.error(f"TEST ERROR INJECTION {i+1}/{count}: Simulated error for monitoring test")
        time.sleep(0.1)  # Small delay to spread timestamps

    print(f"✓ Injected {count} errors")
    print(f"\nNext steps:")
    print(f"1. Wait 60 seconds for notifier poll cycle")
    print(f"2. Check notifier log: tail -20 /tmp/bluebird-live-notifier.log")
    print(f"3. Should see: 'High error rate: {count} errors/hr'")
    print(f"4. Check SMS was sent")
    print(f"\nTo clean up test errors:")
    print(f"  grep -v 'TEST ERROR INJECTION' {LOG_FILE} > /tmp/temp.log && mv /tmp/temp.log {LOG_FILE}")

if __name__ == "__main__":
    inject_test_errors(15)
