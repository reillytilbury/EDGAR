#!/usr/bin/env python3
"""
cleanup_population_data.py — Remove massive "data" fields from existing population.jsonl logs.

This script scans a directory for 'population.jsonl' files, parses each program entry,
removes the raw 'data' dictionary if present, and atomically overwrites the file.
This is necessary as on some projects the data field can be extremely large.
The current code does not write `data` to the population.jsonl file, but older versions did.
"""

import os
import json
import tempfile
from pathlib import Path


def cleanup_file(path: Path) -> None:
    original_size = path.stat().st_size
    print(f"\nProcessing {path}")
    print(f"  Original size: {original_size / 1024 / 1024:.2f} MB")

    # Create a temporary file in the same directory to ensure atomic rename
    temp_dir = path.parent
    fd, temp_path_str = tempfile.mkstemp(
        dir=temp_dir, prefix="pop_cleanup_", suffix=".jsonl"
    )
    temp_path = Path(temp_path_str)

    try:
        lines_processed = 0
        with open(path, "r") as fin, os.fdopen(fd, "w") as fout:
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"  Warning: Skipped corrupted line: {e}")
                    continue

                if "data" in d:
                    d.pop("data")
                if "eval_fingerprint" in d:
                    d.pop("eval_fingerprint")

                fout.write(json.dumps(d) + "\n")
                lines_processed += 1

        # Replace original file atomically
        temp_path.replace(path)
        new_size = path.stat().st_size
        reduction = (
            (1.0 - (new_size / original_size)) * 100.0 if original_size > 0 else 0.0
        )
        print(f"  Success: Processed {lines_processed} programs.")
        print(f"  New size: {new_size / 1024:.2f} KB ({reduction:.3f}% size reduction)")

    except Exception as e:
        print(f"  Error processing {path}: {e}")
        if temp_path.exists():
            temp_path.unlink()


def main() -> None:
    target_dir = (
        Path("/home/rajah/projects/trial_variability/run_output").expanduser().resolve()
    )
    if not target_dir.exists():
        print(f"Error: Target directory {target_dir} does not exist.")
        return

    print(f"Scanning for 'population.jsonl' in {target_dir}...")
    files = list(target_dir.glob("**/population.jsonl"))

    if not files:
        print("No 'population.jsonl' files found.")
        return

    print(f"Found {len(files)} files to clean up.")
    for f in files:
        cleanup_file(f)


if __name__ == "__main__":
    main()
