"""Manages the real-time status file for an EDGAR run.

This module provides utilities for atomically writing and reading a `status.json`
file, which tracks the live state of an EDGAR experiment. The atomic write
mechanism (`.tmp` file + `os.replace`) ensures that external readers, such as
the dashboard, never encounter a partially written or corrupted status file,
guaranteeing data integrity during live monitoring.

The `status.json` file adheres to the following schema:
    {
        "state":         "starting" | "running" | "complete" | "failed",
        "current_gen":   int | None,   # The current generation number, or None if not started
        "current_stage": str | None,   # e.g. "generate_models (32/48)" or "score (23/48)"
        "n_gens":        int,          # Total number of generations configured
        "started_at":    float,        # Unix epoch seconds when the run started
        "updated_at":    float,        # Unix epoch seconds when the status was last updated
        "error":         str | None,   # Error message if the run failed
        "last_metrics":  dict | None,  # last completed gen's metrics row (see io/metrics.py)
    }
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Literal

State = Literal["starting", "running", "complete", "failed"]

STATUS_FILENAME = "status.json"


def write_status(
    out_dir: str | Path,
    state: State,
    n_gens: int,
    current_gen: int | None = None,
    started_at: float | None = None,
    error: str | None = None,
    current_stage: str | None = None,
    last_metrics: dict | None = None,
) -> None:
    """Atomically writes the current run status to a `status.json` file.

    This function creates or updates the `status.json` file within the specified
    output directory. It uses an atomic write operation to ensure that the file
    is never in a corrupted or partially written state, which is critical for
    reliable live monitoring by the dashboard. The `updated_at` timestamp is
    automatically set to the current time.

    ``current_stage`` and ``last_metrics`` are optional live-progress fields
    populated by ``edgar.io.metrics``. Existing callers that pass only the
    coarser fields remain backward-compatible.

    Args:
        out_dir: The directory where `status.json` should be written.
        state: The current state of the run (e.g., "running", "complete").
        n_gens: The total number of generations configured for the run.
        current_gen: The current generation number being executed. Defaults to None.
        started_at: The Unix epoch timestamp when the run initially started. If None,
            the current time is used.
        error: An optional error message if the run has failed. Defaults to None.
        current_stage: An optional string describing the current stage (e.g., "spawn").
        last_metrics: An optional dictionary containing the last completed generation's metrics.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "state": state,
        "current_gen": current_gen,
        "current_stage": current_stage,
        "n_gens": int(n_gens),
        "started_at": float(started_at) if started_at is not None else time.time(),
        "updated_at": time.time(),
        "error": error,
        "last_metrics": last_metrics,
    }
    _atomic_write_json(out_dir / STATUS_FILENAME, payload)


def read_status(run_dir: str | Path) -> dict | None:
    """Reads the `status.json` file from a given run directory.

    This function attempts to load the status information. It is robust to the
    absence of the file (for legacy runs) and JSON decoding errors, returning
    `None` in such cases to prevent crashes.

    Args:
        run_dir: The directory from which to read `status.json`.

    Returns:
        A dictionary containing the run status, or None if the file is absent
        or corrupted.
    """
    path = Path(run_dir) / STATUS_FILENAME
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except json.JSONDecodeError:
        return None


def _atomic_write_json(path: Path, payload: dict) -> None:
    """Writes a dictionary payload to a JSON file atomically.

    This private helper function ensures that a JSON file is written
    atomically by first writing to a temporary file with a unique suffix
    (using `_tmp_path`) and then replacing the original file. This prevents
    readers from observing a partially written or corrupted file.
    In case of any error during the write or replace operation, the temporary
    file is cleaned up.

    Args:
        path: The target path for the JSON file.
        payload: The dictionary to be serialized to JSON.
    """
    tmp = _tmp_path(path)
    try:
        with open(tmp, "w") as f:
            json.dump(payload, f, indent=2)
        os.replace(tmp, path)
    except Exception:
        # Clean up the tmp file on any failure so it doesn't accumulate.
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def atomic_write_text(path: str | Path, text: str) -> None:
    """Atomically writes arbitrary string content to a file.

    This function provides the same atomic write semantics as `_atomic_write_json`
    but is designed for any string payload, such as JSONL files or log files.
    It uses a temporary file with a per-process and per-thread unique suffix to
    prevent conflicts between concurrent writers. Readers will continue to see
    the previous content of the file until the `os.replace` operation atomically
    swaps the inode, ensuring data consistency.

    Args:
        path: The target path for the text file.
        text: The string content to write to the file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp_path(path)
    try:
        with open(tmp, "w") as f:
            f.write(text)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _tmp_path(path: Path) -> Path:
    """Generates a temporary file path unique to the current process and thread.

    This helper is used to create temporary files for atomic write operations.
    By including the process ID (pid) and thread identifier in the suffix, it
    ensures that parallel writers in the same or different processes do not
    collide on their intermediate temporary files, preventing race conditions.

    Args:
        path: The original target path for which a temporary path is needed.

    Returns:
        A `Path` object representing a unique temporary file path.
    """
    return path.with_suffix(path.suffix + f".tmp.{os.getpid()}.{threading.get_ident()}")
