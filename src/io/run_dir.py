"""
Run output directory creation.

Creates timestamped directories for run outputs in the format:
  <save_path>/MM-DD/HH-MM-SS/

Each run gets its own leaf directory. Month-day grouping allows browsing by date.

Example usage:
--------------
    # Create timestamped run directory
    run_dir = make_run_dir("/home/user/runs")
    # Returns: /home/user/runs/03-15/10-30-45/

    # Save run outputs
    spec.save_record(run_dir)
    population.save(run_dir / "population.jsonl")
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def make_run_dir(save_path: str) -> Path:
    """
    Create a timestamped run directory at <save_path>/MM-DD/HH-MM-SS/.

    Args:
        save_path: Base output directory (e.g., "/home/user/runs")

    Returns:
        Path to newly created (or existing) run directory
    """
    base = Path(save_path).resolve()
    now = pd.Timestamp.now()
    run = base / now.strftime("%m-%d") / now.strftime("%H-%M-%S")
    run.mkdir(parents=True, exist_ok=True)
    return run
