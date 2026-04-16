import json
import logging
import os

import numpy as np
import pandas as pd


def _append_generation_record(filepath, record):
    """Append a single generation record as a JSON line."""
    with open(filepath, 'a') as f:
        f.write(json.dumps(record, default=str) + '\n')


def _drop_nonfinite_train_loss_rows(df: pd.DataFrame, context: str) -> tuple[pd.DataFrame, int]:
    """
    Remove rows whose train_loss is NaN/Inf/non-numeric.

    Returns:
        (clean_df, n_removed)
    """
    if df is None or len(df) == 0:
        return df, 0
    if 'train_loss' not in df.columns:
        return df, 0

    train_loss_num = pd.to_numeric(df['train_loss'], errors='coerce')
    finite_mask = np.isfinite(train_loss_num.to_numpy(dtype=float))
    n_removed = int((~finite_mask).sum())
    if n_removed > 0:
        logging.info(
            "%s: dropped %d programs with non-finite train_loss.",
            context,
            n_removed,
        )
        print(f"{context}: dropped {n_removed} programs with non-finite train_loss.", flush=True)
    clean_df = df.loc[finite_mask].reset_index(drop=True)
    return clean_df, n_removed


def _drop_nonfinite_train_loss_from_islands(islands: list[pd.DataFrame], context: str) -> list[pd.DataFrame]:
    """
    Apply non-finite train_loss filtering to every island.
    """
    cleaned = []
    total_removed = 0
    for island_idx, island_df in enumerate(islands):
        island_clean, removed = _drop_nonfinite_train_loss_rows(
            island_df,
            context=f"{context} (island={island_idx})",
        )
        cleaned.append(island_clean)
        total_removed += removed
    if total_removed > 0:
        logging.info("%s: total dropped non-finite-loss programs=%d", context, total_removed)
        print(f"{context}: total dropped non-finite-loss programs={total_removed}", flush=True)
    return cleaned


def _update_generation_log_records(filepath, updates_by_key):
    """Patch existing JSONL records in-place by candidate UID."""
    if not updates_by_key or not os.path.isfile(filepath):
        return

    normalized_updates = {
        (int(key[0]), int(key[1]), int(key[2])): value
        for key, value in updates_by_key.items()
    }

    with open(filepath, 'r') as f:
        lines = f.readlines()
    with open(filepath, 'w') as f:
        for line in lines:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            key = (rec['iteration_number'], rec['birth_island'], rec['batch_index'])
            if key in normalized_updates:
                rec.update(normalized_updates[key])
            f.write(json.dumps(rec, default=str) + '\n')


def _apply_removal_reasons_to_log(filepath, removal_events):
    """Batch-update the JSONL log with removal_reason fields.

    Reads the log, adds ``removal_reason`` to any record whose UID matches
    a :class:`RemovalEvent`, then rewrites the file.  Records that already
    carry a ``removal_reason`` are left unchanged.

    Args:
        filepath: Path to program_generation_log.jsonl
        removal_events: List of ``RemovalEvent`` objects from dedup / prune.
    """
    if not removal_events or not os.path.isfile(filepath):
        return

    # Build lookup: (iteration, birth_island, batch_index) -> removal dict
    reason_lookup = {}
    for evt in removal_events:
        key = (int(evt.uid[0]), int(evt.uid[1]), int(evt.uid[2]))
        details = dict(evt.details or {})
        details.setdefault("iteration", evt.iteration)
        reason_lookup[key] = {
            "category": evt.category,
            "event_type": evt.event_type,
            "island_id": evt.island_id,
            "rule": evt.rule,
            "details": details,
        }

    # Non-atomic read/overwrite. If log corruption is observed, switch to
    # temp file + os.replace() for atomic writes.
    with open(filepath, 'r') as f:
        lines = f.readlines()
    with open(filepath, 'w') as f:
        for line in lines:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            key = (rec['iteration_number'], rec['birth_island'], rec['batch_index'])
            if key in reason_lookup and 'removal_reason' not in rec:
                rec['removal_reason'] = reason_lookup[key]
            f.write(json.dumps(rec, default=str) + '\n')


def _update_generation_log_test_losses_and_mark_winner(filepath, islands):
    """Update JSONL records in-place with test_loss values and mark the winner.

    Args:
        filepath: Path to the JSONL generation log
        islands: List of island dataframes with test_loss values
    """
    if not os.path.isfile(filepath):
        return
    # Build lookup: (iteration, island, batch) -> test_loss
    test_loss_lookup = {}
    for island_idx, island_df in enumerate(islands):
        for _, row in island_df.iterrows():
            key = (int(row['iteration_number']), int(row['birth_island']), int(row['batch_index']))
            tl = row.get('test_loss')
            if tl is not None and not (isinstance(tl, float) and np.isinf(tl)):
                test_loss_lookup[key] = float(tl)

    # from test_loss_lookup, find the best test loss (i.e. smallest) and corresponding key 
    best_test_loss = float('inf')
    best_key = None
    for key, tl in test_loss_lookup.items():
        if tl < best_test_loss:
            best_test_loss = tl
            best_key = key
    logging.info(f"Best test loss found: {best_test_loss:.6g} for program {best_key}.")

    # compare best_key to winner_id if provided
    winner_id = best_key    

    # Read, update, rewrite
    with open(filepath, 'r') as f:
        lines = f.readlines()
    with open(filepath, 'w') as f:
        for line in lines:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            key = (rec['iteration_number'], rec['birth_island'], rec['batch_index'])
            if key in test_loss_lookup:
                rec['test_loss'] = test_loss_lookup[key]
            # Mark the winner
            rec['is_winner'] = (winner_id is not None and key == winner_id)
            f.write(json.dumps(rec, default=str) + '\n')


