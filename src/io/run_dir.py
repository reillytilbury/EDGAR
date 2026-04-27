"""Output directory for one run: <save_path>/MM-DD/HH-MM-SS/."""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def make_run_dir(save_path: str) -> Path:
    base = Path(save_path).resolve()
    now = pd.Timestamp.now()
    run = base / now.strftime("%m-%d") / now.strftime("%H-%M-%S")
    run.mkdir(parents=True, exist_ok=True)
    return run
