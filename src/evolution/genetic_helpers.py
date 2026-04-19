"""
genetic_helpers.py

Evolution config, removal logging, and the three population operations
(dedup, prune, migrate) that the engine loop calls each iteration.

All operate on list[pd.DataFrame] for now — the engine still uses DataFrames.
Once hypothesis_engine is migrated to Program/Population/Island these
functions will shrink further or disappear.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import logging

import jax.numpy as jnp
import numpy as np
import pandas as pd
import yaml

from .. import utils


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class EvolutionConfig:
    """Parameters that govern island population dynamics."""
    capacity_per_island: int = 12
    n_migrants: int = 2
    overlap_threshold: int = 6
    loss_tolerance: float = 0.01
    cosine_tolerance: float = 0.99
    min_wise_programs: int = 0
    large_lm_name: str = ""
    loss_type: str = "train_loss"

    @classmethod
    def from_yaml(cls, yaml_path: Optional[Path] = None) -> "EvolutionConfig":
        if yaml_path is None:
            yaml_path = Path(__file__).parent.parent / "config" / "experiment.yaml"
        try:
            with open(yaml_path) as f:
                data = yaml.safe_load(f)
        except FileNotFoundError:
            logging.warning("EvolutionConfig: %s not found, using defaults.", yaml_path)
            return cls()
        return cls.from_dict(data.get("evolution", {}))

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "EvolutionConfig":
        return cls(
            capacity_per_island=d.get("capacity_per_island", 12),
            n_migrants=d.get("n_migrants", 2),
            overlap_threshold=d.get("overlap_threshold", 6),
            loss_tolerance=d.get("loss_tolerance", 0.01),
            cosine_tolerance=d.get("cosine_tolerance", 0.99),
            min_wise_programs=d.get("min_wise_programs", 0),
            large_lm_name=d.get("large_lm_name", ""),
            loss_type=d.get("loss_type", "train_loss"),
        )


# ---------------------------------------------------------------------------
# Removal event (used by monitoring / JSONL log)
# ---------------------------------------------------------------------------

@dataclass
class RemovalEvent:
    """Records why a program was removed from an island."""
    uid: Tuple[int, int, int]      # (iteration, birth_island, batch_index)
    category: str                   # "deduplication" or "pruning"
    event_type: str
    island_id: int
    iteration: int
    rule: str
    details: Optional[Dict[str, Any]] = None


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _row_uid(row) -> tuple:
    return (row.get("iteration_number"), row.get("birth_island"), row.get("batch_index"))


def _row_loss(row, loss_type: str) -> float:
    v = row.get(loss_type, float("inf")) if hasattr(row, "get") else row[loss_type]
    return float(v) if v is not None else float("inf")


def compare_programs(
    p1, p2,
    mode: str = "complicated",
    loss_tol: float = 0.01,
    cosine_tol: float = 0.99,
    loss_type: str = "train_loss",
    return_details: bool = False,
):
    """Return True (or details dict) if two DataFrame rows are behavioural duplicates."""
    def _result(is_dup, logic, **details):
        payload = {"is_duplicate": is_dup, "logic": logic, "details": details}
        return payload if return_details else is_dup

    if p1.get("program_code_string", "") == p2.get("program_code_string", ""):
        return _result(True, "same_code")

    params1, params2 = p1.get("params"), p2.get("params")
    if params1 is not None and params2 is not None:
        if utils.params_signature(params1) != utils.params_signature(params2):
            return _result(False, "different_param_signature") if return_details else False

    l1, l2 = _row_loss(p1, loss_type), _row_loss(p2, loss_type)
    if np.isfinite(l1) and np.isfinite(l2):
        if abs(l1 - l2) / max(abs(l1), abs(l2), 1e-6) > loss_tol:
            return _result(False, "loss_too_different") if return_details else False

    if mode == "simple":
        return _result(False, "simple_mode_different_code")

    e1, e2 = p1.get("evaluation_matrix"), p2.get("evaluation_matrix")
    if e1 is None or e2 is None:
        return _result(False, "missing_evaluation_matrix")

    e1, e2 = jnp.array(e1), jnp.array(e2)
    if e1.ndim == 1:
        e1, e2 = e1.reshape(1, -1), e2.reshape(1, -1)
    if e1.shape != e2.shape:
        return _result(False, "shape_mismatch", shape1=tuple(e1.shape), shape2=tuple(e2.shape))

    n1 = jnp.linalg.norm(e1, axis=1, keepdims=True)
    n2 = jnp.linalg.norm(e2, axis=1, keepdims=True)
    n1 = jnp.where(n1 == 0, 1.0, n1)
    n2 = jnp.where(n2 == 0, 1.0, n2)
    cosine = float(jnp.mean(jnp.sum((e1 / n1) * (e2 / n2), axis=1)))

    if cosine >= cosine_tol:
        return _result(True, "behavioral_cosine_similarity", cosine_similarity=cosine)
    return _result(False, "cosine_below_threshold", cosine_similarity=cosine)


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

def remove_duplicates(
    df: pd.DataFrame,
    mode: str = "complicated",
    loss_tol: float = 0.01,
    cosine_tol: float = 0.99,
    loss_type: str = "train_loss",
    iteration: int | None = None,
    island_id: int | None = None,
) -> Tuple[pd.DataFrame, List[RemovalEvent]]:
    """Remove duplicate rows from a single island DataFrame."""
    events: List[RemovalEvent] = []
    if len(df) == 0:
        return df.copy(), events

    to_remove: set[int] = set()
    for i in range(len(df)):
        if i in to_remove:
            continue
        for j in range(i + 1, len(df)):
            if j in to_remove:
                continue
            cmp = compare_programs(df.iloc[i], df.iloc[j], mode, loss_tol, cosine_tol,
                                   loss_type, return_details=True)
            if not cmp["is_duplicate"]:
                continue
            li, lj = _row_loss(df.iloc[i], loss_type), _row_loss(df.iloc[j], loss_type)
            loser_idx = j if li <= lj else i
            if loser_idx in to_remove:
                continue
            to_remove.add(loser_idx)
            loser = df.iloc[loser_idx]
            survivor = df.iloc[j if loser_idx == i else i]
            events.append(RemovalEvent(
                uid=_row_uid(loser),
                category="deduplication",
                event_type="DEDUP_WITHIN_ISLAND_REMOVE",
                island_id=island_id if island_id is not None else -1,
                iteration=iteration if iteration is not None else -1,
                rule=cmp["logic"],
                details={
                    "kept_uid": list(_row_uid(survivor)),
                    "kept_loss": round(_row_loss(survivor, loss_type), 6),
                    "removed_loss": round(_row_loss(loser, loss_type), 6),
                    **(cmp.get("details") or {}),
                },
            ))

    keep = [i for i in range(len(df)) if i not in to_remove]
    return df.iloc[keep].reset_index(drop=True), events


def perform_island_deduplication(
    islands: List[pd.DataFrame],
    mode: str = "complicated",
    loss_tol: float = 0.01,
    cosine_tol: float = 0.99,
    loss_type: str = "train_loss",
    iteration: int | None = None,
    overlap_threshold: int = 6,
) -> Tuple[List[pd.DataFrame], List[RemovalEvent]]:
    """Deduplicate within each island, then across islands."""
    all_events: List[RemovalEvent] = []

    # Within-island
    deduped = []
    for island_id, df in enumerate(islands):
        clean, events = remove_duplicates(df, mode, loss_tol, cosine_tol, loss_type,
                                          iteration, island_id)
        deduped.append(clean)
        all_events.extend(events)

    # Cross-island: remove higher-loss duplicates from the higher-indexed island
    n = len(deduped)
    drop: List[set] = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if len(deduped[i]) - len(drop[i]) < overlap_threshold:
                continue
            if len(deduped[j]) - len(drop[j]) < overlap_threshold:
                continue
            for idx_j in range(len(deduped[j])):
                if idx_j in drop[j]:
                    continue
                pj = deduped[j].iloc[idx_j]
                for idx_i in range(len(deduped[i])):
                    if idx_i in drop[i]:
                        continue
                    pi = deduped[i].iloc[idx_i]
                    cmp = compare_programs(pi, pj, mode, loss_tol, cosine_tol, loss_type,
                                           return_details=True)
                    if not cmp["is_duplicate"]:
                        continue
                    li, lj = _row_loss(pi, loss_type), _row_loss(pj, loss_type)
                    if li <= lj:
                        loser_island, loser_idx, loser, survivor = j, idx_j, pj, pi
                    else:
                        loser_island, loser_idx, loser, survivor = i, idx_i, pi, pj
                    if loser_idx in drop[loser_island]:
                        if loser_island == j:
                            break
                        continue
                    drop[loser_island].add(loser_idx)
                    all_events.append(RemovalEvent(
                        uid=_row_uid(loser),
                        category="deduplication",
                        event_type="DEDUP_CROSS_ISLAND_REMOVE",
                        island_id=loser_island,
                        iteration=iteration if iteration is not None else -1,
                        rule="cross_island_keep_lower_loss",
                        details={
                            "kept_uid": list(_row_uid(survivor)),
                            "kept_loss": round(_row_loss(survivor, loss_type), 6),
                            "removed_loss": round(_row_loss(loser, loss_type), 6),
                            **(cmp.get("details") or {}),
                        },
                    ))
                    if loser_island == j:
                        break

    for island_id, bad in enumerate(drop):
        if bad:
            keep = [k for k in range(len(deduped[island_id])) if k not in bad]
            deduped[island_id] = deduped[island_id].iloc[keep].reset_index(drop=True)

    return deduped, all_events


# ---------------------------------------------------------------------------
# Pruning
# ---------------------------------------------------------------------------

def perform_population_pruning(
    islands: List[pd.DataFrame],
    critical_population_size: int = 12,
    large_lm_name: str = "",
    min_wise_population_size: int = 0,
    iteration: int | None = None,
    loss_type: str = "train_loss",
    # legacy alias
    max_population: int = None,
) -> Tuple[List[pd.DataFrame], List[RemovalEvent]]:
    """Prune each island to critical_population_size, optionally preserving wise programs."""
    if max_population is not None:
        critical_population_size = max_population
    assert min_wise_population_size <= critical_population_size

    all_events: List[RemovalEvent] = []
    pruned = []
    for island_id, df in enumerate(islands):
        if len(df) <= critical_population_size:
            pruned.append(df.copy())
            continue

        if "llm_name" in df.columns and large_lm_name and min_wise_population_size > 0:
            wise = df[df["llm_name"] == large_lm_name].nsmallest(min_wise_population_size, loss_type)
            rest = df[~df.index.isin(wise.index)]
            vacancies = critical_population_size - len(wise)
            kept = pd.concat([wise, rest.nsmallest(vacancies, loss_type)])
        else:
            kept = df.nsmallest(critical_population_size, loss_type)

        removed = df[~df.index.isin(kept.index)]
        for _, row in removed.iterrows():
            all_events.append(RemovalEvent(
                uid=_row_uid(row),
                category="pruning",
                event_type="PRUNE_REMOVE",
                island_id=island_id,
                iteration=iteration if iteration is not None else -1,
                rule="capacity_keep_lowest_loss",
                details={"removed_loss": round(_row_loss(row, loss_type), 6),
                         "capacity": critical_population_size},
            ))
        pruned.append(kept.reset_index(drop=True))

    return pruned, all_events


# ---------------------------------------------------------------------------
# Migration
# ---------------------------------------------------------------------------

def perform_probabilistic_migration(
    islands: List[pd.DataFrame],
    n_migrants: int,
    destination_islands: List[int],
    temperature: float = 1.0,
    iteration: int | None = None,
) -> List[pd.DataFrame]:
    """Send n_migrants rows from each island to its destination, selected by loss-weighted sampling."""
    n_islands = len(islands)
    if destination_islands is None:
        destination_islands = [(i + 1) % n_islands for i in range(n_islands)]

    temp = max(temperature, 1e-3)

    # Compute per-island migration probabilities
    probs = []
    for df in islands:
        n = len(df)
        if n == 0:
            probs.append(np.array([], dtype=float))
            continue
        losses = pd.to_numeric(df["train_loss"], errors="coerce").to_numpy(dtype=float)
        finite = np.isfinite(losses)
        if finite.any():
            worst = float(np.max(losses[finite]))
            losses = np.where(finite, losses, worst + float(np.max(losses[finite]) - np.min(losses[finite])) + 1.0)
        relative = losses - losses.min()
        logits = -(relative / (np.std(relative) + 1e-6)) / temp
        logits -= logits.max()
        p = np.exp(logits)
        s = p.sum()
        probs.append(p / s if np.isfinite(s) and s > 0 else np.full(n, 1.0 / n))

    # Select migrants and dispatch
    migrants = []
    for src_id, df in enumerate(islands):
        n = len(df)
        p = probs[src_id]
        k = min(n_migrants, int(np.sum(p > 0)), n)
        if k <= 0:
            migrants.append(df.iloc[0:0])
            continue
        idx = np.random.choice(n, size=k, replace=False, p=p)
        migrants.append(df.iloc[idx].reset_index(drop=True))

    for src_id, batch in enumerate(migrants):
        dest_id = destination_islands[src_id]
        islands[dest_id] = pd.concat([islands[dest_id], batch], ignore_index=True)

    return islands
