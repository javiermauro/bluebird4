"""
Atomic file I/O utilities for BLUEBIRD 4.0.

Provides safe JSON writing that prevents corruption from partial writes or power loss.
Uses write-to-temp + fsync + atomic rename pattern.

CRITICAL: Write failures must NEVER crash the trading loop - log loudly and continue.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Union

logger = logging.getLogger(__name__)


def atomic_write_json(
    path: Union[str, Path],
    data: Dict[str, Any],
    indent: int = 2
) -> bool:
    """
    Write JSON data atomically using temp file + rename.

    This prevents corruption from:
    - Power loss during write
    - Partial writes
    - Concurrent read during write

    Args:
        path: Target file path (will be created if doesn't exist)
        data: Dictionary to serialize as JSON
        indent: JSON indentation (default 2)

    Returns:
        True if write succeeded, False if failed (trading continues either way)

    Note:
        On failure, logs error but does NOT raise exception.
        The trading loop must continue even if state persistence fails.
    """
    path = Path(path)
    tmp_path = path.with_suffix('.json.tmp')

    try:
        # Ensure parent directory exists
        path.parent.mkdir(parents=True, exist_ok=True)

        # Write to temp file in same directory (required for atomic rename)
        with open(tmp_path, 'w') as f:
            json.dump(data, f, indent=indent)
            f.flush()
            os.fsync(f.fileno())  # Force write to disk

        # Atomic rename (on POSIX systems)
        os.replace(tmp_path, path)

        return True

    except Exception as e:
        logger.error(f"[ATOMIC_IO] State write failed for {path}: {e}")

        # Best-effort cleanup of temp file
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except Exception as cleanup_err:
            logger.warning(f"[ATOMIC_IO] Failed to cleanup temp file {tmp_path}: {cleanup_err}")

        # DO NOT RAISE - trading must continue
        return False


def safe_read_json(
    path: Union[str, Path],
    default: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Safely read JSON file with fallback to default on any error.

    Args:
        path: File path to read
        default: Default value if file missing or corrupt (default: empty dict)

    Returns:
        Parsed JSON dict, or default if any error occurs

    Note:
        On error, logs warning but does NOT raise exception.
    """
    if default is None:
        default = {}

    path = Path(path)

    try:
        if not path.exists():
            return default

        with open(path, 'r') as f:
            return json.load(f)

    except json.JSONDecodeError as e:
        logger.warning(f"[ATOMIC_IO] Corrupt JSON in {path}, using default: {e}")
        return default

    except Exception as e:
        logger.warning(f"[ATOMIC_IO] Failed to read {path}, using default: {e}")
        return default
