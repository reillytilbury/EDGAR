"""Shared I/O utilities for monitoring.

Provides a robust JSONL reader that gracefully handles partial or
malformed lines, making it safe to call against an in-progress log.
"""

import ast
import html as html_module
import json
import logging
import os
import re


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
        if isinstance(rec.get("birth_island"), int) and rec.get("birth_island") >= 0 and rec.get("iteration_number", -1) >= 0
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


def build_removal_details(
    rec: dict,
    label_map: dict[str, str] | None = None,
) -> dict:
    """Normalize and enrich removal details for sidebar rendering."""
    removal_reason = rec.get("removal_reason")
    if not isinstance(removal_reason, dict):
        return {}

    raw_details = removal_reason.get("details")
    if isinstance(raw_details, dict):
        normalized = dict(raw_details)
    elif raw_details is None:
        normalized = {}
    else:
        normalized = {"value": raw_details}

    if normalized.get("iteration") is None and rec.get("iteration_number") is not None:
        normalized["iteration"] = rec.get("iteration_number")

    kept_uid = normalized.get("kept_uid")
    kept_key = parse_parent_key(kept_uid)
    if kept_key is not None:
        kept_program_id = make_program_id(*kept_key)
        normalized.setdefault("kept_program_id", kept_program_id)
        if label_map and kept_program_id in label_map:
            normalized.setdefault("kept_label", label_map[kept_program_id])

    reference_island = normalized.get("reference_island")
    if isinstance(reference_island, int):
        normalized.setdefault("reference_island_label", island_label(reference_island))

    preferred_order = [
        "iteration",
        "kept_label",
        "kept_program_id",
        "kept_uid",
        "reference_island_label",
        "reference_island",
        "kept_loss",
        "removed_loss",
        "match_rule",
        "tie_breaker",
        "cosine_similarity",
        "cosine_tol",
        "capacity",
        "min_wise_population_size",
        "removed_class",
        "value",
    ]
    ordered = {}
    for key in preferred_order:
        if key in normalized:
            ordered[key] = normalized[key]
    for key, value in normalized.items():
        if key not in ordered:
            ordered[key] = value
    return ordered


def build_removal_reason(
    rec: dict,
    label_map: dict[str, str] | None = None,
) -> dict | None:
    """Normalize removal metadata for sidebar rendering."""
    removal_reason = rec.get("removal_reason")
    if not isinstance(removal_reason, dict):
        return removal_reason

    normalized = dict(removal_reason)
    normalized["details"] = build_removal_details(rec, label_map)
    return normalized


def build_record_entry(
    rec: dict,
    label_map: dict[str, str] | None = None,
) -> dict:
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
        "removal_reason": build_removal_reason(rec, label_map),
    }


def backfill_removal_iterations_from_engine_log(
    records: list[dict], engine_log_path: str | None
) -> None:
    """Populate missing removal iteration fields from hypothesis_engine.log.

    Older runs may have ``removal_reason`` metadata without
    ``details.iteration``. The structured engine log preserves the active
    iteration around removal events, so we can recover that for report
    rendering without guessing from the program creation record.
    """
    if not engine_log_path or not os.path.isfile(engine_log_path):
        return

    iteration_re = re.compile(r"^Iteration\s+(-?\d+)(?:\s|$)")
    removed_uid_re = re.compile(r"removed_uid=(\([^)]*\))")
    rule_re = re.compile(r"rule=([^,]+)")
    removal_event_prefixes = (
        "DEDUP_WITHIN_ISLAND_REMOVE:",
        "DEDUP_CROSS_ISLAND_REMOVE:",
        "PRUNE_REMOVE:",
    )

    removal_events_by_uid: dict[tuple[int, int, int], list[dict]] = {}
    current_iteration: int | None = None

    with open(engine_log_path, "r") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue

            match = iteration_re.match(line)
            if match:
                current_iteration = int(match.group(1))
                continue

            if current_iteration is None or not line.startswith(removal_event_prefixes):
                continue

            uid_match = removed_uid_re.search(line)
            if not uid_match:
                continue

            try:
                uid = ast.literal_eval(uid_match.group(1))
            except (SyntaxError, ValueError):
                continue

            if (
                isinstance(uid, tuple)
                and len(uid) == 3
                and all(isinstance(part, int) for part in uid)
            ):
                event_type = line.split(":", 1)[0]
                rule_match = rule_re.search(line)
                rule = rule_match.group(1).strip() if rule_match else None
                removal_events_by_uid.setdefault(uid, []).append({
                    "iteration": current_iteration,
                    "event_type": event_type,
                    "rule": rule,
                })

    if not removal_events_by_uid:
        return

    for rec in records:
        removal_reason = rec.get("removal_reason")
        if not isinstance(removal_reason, dict):
            continue

        details = removal_reason.get("details")
        if isinstance(details, dict) and details.get("iteration") is not None:
            continue

        uid = record_key(rec)
        matches = removal_events_by_uid.get(uid)
        if not matches:
            continue

        target_event_type = removal_reason.get("event_type")
        target_rule = removal_reason.get("rule")
        chosen_match = next(
            (
                match
                for match in matches
                if match["event_type"] == target_event_type and match["rule"] == target_rule
            ),
            matches[0],
        )
        removal_iteration = chosen_match["iteration"]

        if not isinstance(details, dict):
            details = {}
        else:
            details = dict(details)
        details["iteration"] = removal_iteration
        removal_reason["details"] = details


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
