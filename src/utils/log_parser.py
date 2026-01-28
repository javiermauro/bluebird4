"""Log parsing utilities for error rate monitoring."""

import re
from datetime import datetime, timedelta
from typing import List, Dict
from pathlib import Path

# Format: TIMESTAMP - LOGGER_NAME - LEVEL - MESSAGE
LOG_PATTERN = re.compile(
    r'^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) - ([\w.]+) - (ERROR|CRITICAL) - (.+)$'
)

# Known benign errors to exclude
BENIGN_PATTERNS = [
    r'connection limit exceeded',  # Alpaca WebSocket (not our fault)
]

def parse_log_errors(
    log_path: Path,
    since: datetime,
    max_errors: int = 100
) -> List[Dict[str, str]]:
    """
    Parse ERROR/CRITICAL entries from log since timestamp.

    Returns:
        List of {timestamp, logger, message} dicts
    """
    errors = []

    if not log_path.exists():
        return errors

    try:
        with open(log_path, 'r') as f:
            for line in f:
                match = LOG_PATTERN.match(line.strip())
                if not match:
                    continue

                timestamp_str, logger, level, message = match.groups()

                # Parse timestamp
                try:
                    ts = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S,%f')
                except ValueError:
                    continue

                # Filter by time window
                if ts < since:
                    continue

                # Skip benign patterns
                if any(re.search(p, message, re.IGNORECASE) for p in BENIGN_PATTERNS):
                    continue

                errors.append({
                    'timestamp': ts.isoformat(),
                    'logger': logger,
                    'message': message[:200]  # Truncate for state file
                })

                # Cap memory usage
                if len(errors) >= max_errors:
                    break

    except Exception:
        # Graceful failure - return partial results
        pass

    return errors
