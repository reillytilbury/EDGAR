"""Shared I/O utilities for progress_report.

Provides a robust JSONL reader that gracefully handles partial or
malformed lines, making it safe to call against an in-progress log.
"""

import json
import logging


def load_generation_log(log_path: str) -> list[dict]:
    """Load JSONL generation log; skip empty or malformed lines.

    Safe to call against a log that is being actively written to.
    JSONL writes one complete json.dumps(...) + newline atomically per
    program, so a partial write (OS crash) at most corrupts the last line,
    which this function handles gracefully via the try/except.

    Args:
        log_path: Path to the JSONL generation log file.

    Returns:
        List of parsed record dicts; incomplete/corrupt lines are skipped.
    """
    records = []
    with open(log_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                logging.warning("Skipping malformed line in %s", log_path)
    return records
