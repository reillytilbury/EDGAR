"""
status.py

Single-file run state for the live dashboard. Written atomically (.tmp + os.replace)
so a polling reader can never observe a torn file.

Schema (status.json):
    {
        "state":       "starting" | "running" | "complete" | "failed",
        "current_gen": int | None,
        "n_gens":      int,
        "started_at":  float (unix epoch seconds),
        "updated_at":  float (unix epoch seconds),
        "error":       str | None
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
) -> None:
    """Atomically write status.json into out_dir."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "state": state,
        "current_gen": current_gen,
        "n_gens": int(n_gens),
        "started_at": float(started_at) if started_at is not None else time.time(),
        "updated_at": time.time(),
        "error": error,
    }
    _atomic_write_json(out_dir / STATUS_FILENAME, payload)


def read_status(run_dir: str | Path) -> dict | None:
    """Read status.json; returns None if absent (legacy runs)."""
    path = Path(run_dir) / STATUS_FILENAME
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except json.JSONDecodeError:
        return None


def _atomic_write_json(path: Path, payload: dict) -> None:
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
    """Atomically write text to path. Same semantics as _atomic_write_json but
    for arbitrary string payloads (e.g. JSONL files).

    Uses a per-pid+thread tmp suffix so concurrent writers never collide on the
    rename step. The reader continues to see the previous content until the
    os.replace flips the inode atomically.
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
    """Return a tmp path unique to this (pid, thread) so parallel writers don't
    stomp on each other's intermediate state."""
    return path.with_suffix(path.suffix + f".tmp.{os.getpid()}.{threading.get_ident()}")
