"""
Project output path creation.

Creates timestamped directories for project results in the format:
  <save_path>/MM-DD/HH-MM-SS/

Each project gets its own leaf directory. Month-day grouping allows browsing by date.

Example usage:
--------------
    # Create timestamped project output path
    output_dir = make_output_dir("/home/user/results")
    # Returns: /home/user/results/03-15/10-30-45/

    # Save project outputs
    spec.save_task_spec(output_dir)
    population.save(output_dir / "population.jsonl")
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd

def make_output_dir(save_path: str) -> Path:
    """
    Create a timestamped project output path at <save_path>/MM-DD/HH-MM-SS/.

    Args:
        save_path: Base output directory (e.g., "/home/user/results")

    Returns:
        Path to newly created (or existing) project output path
    """
    base = Path(save_path).resolve()
    now = pd.Timestamp.now()
    output_dir = base / now.strftime("%m-%d") / now.strftime("%H-%M-%S")
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir
