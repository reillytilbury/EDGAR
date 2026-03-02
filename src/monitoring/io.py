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


def make_program_id(iteration, island, batch) -> str | None:
    """Format a canonical raw program id like ``6_6_3``."""
    if iteration is None or island is None or batch is None:
        return None
    return f"{iteration}_{island}_{batch}"


def record_id(rec: dict) -> str | None:
    """Return the canonical raw id for a record, or None if incomplete."""
    return make_program_id(*record_key(rec))


def island_label(island_idx: int) -> str:
    """Map island index to a letter label (0->A, 1->B, ..., 25->Z, 26->AA, ...)."""
    if island_idx < 0:
        return "S"
    letters = []
    idx = island_idx
    while True:
        idx, rem = divmod(idx, 26)
        letters.append(chr(ord("A") + rem))
        if idx == 0:
            break
        idx -= 1
    return "".join(reversed(letters))


def assign_display_labels(records: list[dict]) -> dict[str, str]:
    """Assign compact display labels like A12 and S1, mutating records in place."""
    label_map = {}

    for rec in records:
        is_seed = rec.get("is_seed", False) or rec.get("iteration_number") == -1
        if not is_seed:
            continue
        rec_id = record_id(rec)
        if rec_id is None:
            continue
        batch_idx = rec.get("batch_index")
        label = f"S{int(batch_idx) + 1}" if batch_idx is not None else "S?"
        rec["display_label"] = label
        label_map[rec_id] = label

    island_indices = sorted({
        rec.get("birth_island")
        for rec in records
        if rec.get("birth_island", -1) >= 0 and rec.get("iteration_number", -1) >= 0
    })
    for island_idx in island_indices:
        island_records = [
            rec for rec in records
            if rec.get("birth_island") == island_idx and rec.get("iteration_number", -1) >= 0
        ]
        island_records.sort(key=lambda r: (r.get("iteration_number", 0), r.get("batch_index", 0)))
        for i, rec in enumerate(island_records, start=1):
            rec_id = record_id(rec)
            if rec_id is None:
                continue
            label = f"{island_label(island_idx)}{i}"
            rec["display_label"] = label
            label_map[rec_id] = label

    return label_map


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
        "program_id": record_id(rec),
        "display_label": rec.get("display_label"),
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
        "removal_reason": rec.get("removal_reason"),
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
