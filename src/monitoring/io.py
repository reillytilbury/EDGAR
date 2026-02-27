"""Shared I/O utilities for monitoring.

Provides a robust JSONL reader that gracefully handles partial or
malformed lines, making it safe to call against an in-progress log.
"""

import html as html_module
import json
import logging
import os


def escape(text) -> str:
    """HTML-escape a value, returning <em>N/A</em> for None."""
    if text is None:
        return "<em>N/A</em>"
    return html_module.escape(str(text))


def resolve_image_path(raw_path: str | None, image_base_dir: str | None) -> str | None:
    """Resolve an image path to an absolute existing path, or None if missing."""
    if not raw_path:
        return None
    if not os.path.isabs(raw_path) and image_base_dir:
        raw_path = os.path.join(image_base_dir, raw_path)
    if os.path.isfile(raw_path):
        return raw_path
    return None


def record_key(rec: dict) -> tuple:
    """Extract the canonical (iteration, island, batch) identity tuple from a record."""
    return (rec.get("iteration_number"), rec.get("birth_island"), rec.get("batch_index"))


def parse_parent_key(parent_id) -> tuple | None:
    """Convert a stored [iteration, island, batch] parent_id to a key tuple, or None.

    Returns None if parent_id is None or not a valid 3-element list/tuple.
    """
    if isinstance(parent_id, (list, tuple)) and len(parent_id) == 3:
        return (int(parent_id[0]), int(parent_id[1]), int(parent_id[2]))
    return None


def build_record_entry(rec: dict) -> dict:
    """Extract the common sidebar fields shared by all monitoring views."""
    return {
        "iteration": rec.get("iteration_number"),
        "island": rec.get("birth_island"),
        "batch": rec.get("batch_index"),
        "train_loss": rec.get("train_loss"),
        "initial_loss": rec.get("initial_loss"),
        "n_params": rec.get("n_params"),
        "complexity_penalty": rec.get("complexity_penalty"),
        "mode": rec.get("mode"),
        "llm_name": rec.get("llm_name"),
        "temperature": rec.get("temperature"),
        "model_code": rec.get("model_code_numpy"),
        "param_est_code": rec.get("param_est_code"),
        "model_prompt": rec.get("model_prompt"),
        "image_prompt_path": rec.get("image_prompt_path"),
        "model_llm_response": rec.get("model_llm_response"),
        "param_est_prompt": rec.get("param_est_prompt"),
        "param_est_llm_response": rec.get("param_est_llm_response"),
        "train_fit_image_path": rec.get("train_fit_image_path"),
        "test_fit_image_path": rec.get("test_fit_image_path"),
    }


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
