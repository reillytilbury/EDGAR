"""data.py — translate a run_dir into JSON-safe DTOs for the dashboard.

This module provides functions to load and transform raw EDGAR run data
(e.g., `population.jsonl`, `island_census.jsonl`, `status.json`) from a
specified run directory into JSON-safe Data Transfer Objects (DTOs). These DTOs
are then consumed by the EDGAR dashboard's HTTP API, providing real-time
monitoring and post-hoc analysis capabilities.

Key functionalities include:
-   **Run Discovery:** Identifying EDGAR runs within specified root directories.
-   **Data Loading & Caching:** Efficiently loading large data files with
    memoization based on file modification times to optimize dashboard
    performance during live runs.
-   **Run Summary & Live State:** Generating high-level summaries and detailed
    live state information for ongoing or completed experiments.
-   **Program Details:** Providing comprehensive information for individual
    evolved programs, including code, losses, parameters, and lineage.
-   **Family Tree Data:** Preparing data structures for visualizing the
    evolutionary lineage of programs.
-   **JSON Sanitization:** Recursively converting complex Python and NumPy
    types into JSON-compatible values, handling `NaN` and `inf` appropriately.
-   **Legacy Tolerance:** Implicitly treating runs predating the `status.json`
    convention as having a 'complete' status.
"""

from __future__ import annotations

import json
import math
import textwrap
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from ..evolution.island import load_island_census
from ..evolution.population import Population
from ..evolution.program import NotValidated, Program
from ..scoring.utils import _safe_loss as _scoring_safe_loss
from ..io.metrics import METRICS_FILENAME, read_metrics
from ..io.status import read_status
from ..llm.prompt_schema import PromptSchema


# ── Population cache (path, mtime) → Population ──
_POP_CACHE: dict[str, tuple[float, Population]] = {}
_CENSUS_CACHE: dict[str, tuple[float, list[list[set[int]]]]] = {}
_METRICS_CACHE: dict[str, tuple[float, list[dict]]] = {}


def _load_population(run_dir: Path) -> Population | None:
    """Loads the population data for a given run directory, utilizing a cache.

    The cache stores the Population object along with its file's modification
    time (`mtime`). If the file's `mtime` has not changed since the last
    load, the cached object is returned. This prevents expensive re-parsing
    during repeated polls from the dashboard during a live run.

    Args:
        run_dir: The path to the run directory (e.g., `program_databases/YYYY-MM-DD/HH-MM-SS/`).

    Returns:
        A `Population` object if `population.jsonl` exists and can be loaded,
        otherwise None. Returns cached population in case of mid-write race.
    """
    path = run_dir / "population.jsonl"
    if not path.exists():
        return None
    try:
        mtime = path.stat().st_mtime
    except FileNotFoundError:
        return None
    key = str(path)
    cached = _POP_CACHE.get(key)
    if cached is not None and cached[0] == mtime:
        return cached[1]
    try:
        pop = Population.load(str(path))
    except (json.JSONDecodeError, ValueError):
        # mid-write race (shouldn't happen with atomic writes, but defensive)
        return cached[1] if cached else None
    _POP_CACHE[key] = (mtime, pop)
    return pop


def _reconstruct_model_prompt(run_dir: Path, pop: Population, idx: int) -> str:
    """Reconstructs the exact prompt string shown to the LLM for a given program.

    This function uses the `PromptSchema` and configuration from `task_spec.yaml`,
    along with the program's parents and any injected "ideas", to rebuild
    the prompt. This allows for transparent inspection of LLM inputs.

    Args:
        run_dir: The path to the run directory.
        pop: The `Population` object containing all programs.
        idx: The index of the program for which to reconstruct the prompt.

    Returns:
        The reconstructed prompt string, or an error message if reconstruction fails.
        Returns a specific message for seed programs as no LLM prompt was used.
    """
    spec_doc = _load_task_spec(run_dir)
    if not spec_doc:
        return ""

    schemas = spec_doc.get("prompt_schemas") or {}
    model_schema_dict = schemas.get("model")
    if not model_schema_dict:
        return ""

    try:
        prompt_schema = PromptSchema(**model_schema_dict)
    except Exception:
        return "(error: could not parse PromptSchema from task_spec.yaml)"

    p = pop[idx]
    if p.birth.generation < 0:
        return "Seed program: no LLM prompt was used."

    # Flatten config for build_prompt (evolution + llms + scoring)
    flat_config = {
        **(spec_doc.get("evolution") or {}),
        **(spec_doc.get("llms") or {}),
        **(spec_doc.get("scoring") or {}),
        "ideas-injection-point": "\n".join(getattr(p.birth, "ideas", []) or []),
    }

    # Retrieve parents. Same sort logic as generate._resolve_parents
    parents = [pop[i] for i in p.birth.parent_indices if 0 <= i < len(pop)]

    def _loss(prog: Program) -> float:
        return _scoring_safe_loss(prog.program_losses.discover.final)

    parents = sorted(parents, key=_loss, reverse=True)

    mode = p.birth.mode or "explore"

    try:
        return prompt_schema.build_prompt(
            mode=mode,
            parent_programs=parents,
            config=flat_config,
        )
    except Exception as e:
        return f"(error: prompt reconstruction failed: {e})"


def _load_census(run_dir: Path) -> list[list[set[int]]]:
    """Loads the island census data for a given run directory, utilizing a cache.

    The cache stores the census list along with its file's modification
    time (`mtime`). If the file's `mtime` has not changed since the last
    load, the cached object is returned. This prevents expensive re-parsing
    during repeated polls from the dashboard during a live run.

    The JSON shape of the census is `census[generation][island_idx] -> list[int]`,
    where each top-level entry is a generation snapshot.

    Args:
        run_dir: The path to the run directory.

    Returns:
        A list of lists of sets of program indices representing the island
        census, or an empty list if `island_census.jsonl` does not exist
        or cannot be loaded. Returns cached census in case of mid-write race.
    """
    path = run_dir / "island_census.jsonl"
    if not path.exists():
        return []
    try:
        mtime = path.stat().st_mtime
    except FileNotFoundError:
        return []
    key = str(path)
    cached = _CENSUS_CACHE.get(key)
    if cached is not None and cached[0] == mtime:
        return cached[1]
    try:
        census = load_island_census(str(path))
    except (json.JSONDecodeError, ValueError):
        return cached[1] if cached else []
    _CENSUS_CACHE[key] = (mtime, census)
    return census


def _load_metrics(run_dir: Path) -> list[dict]:
    """Loads the metrics data for a given run directory, utilizing a cache.

    The cache stores the metrics list along with its file's modification
    time (`mtime`). If the file's `mtime` has not changed since the last
    load, the cached object is returned. This prevents expensive re-parsing
    during repeated polls from the dashboard during a live run.

    Args:
        run_dir: The path to the run directory.

    Returns:
        A list of dictionaries, where each dictionary represents metrics
        for a generation, or an empty list if `metrics.jsonl` does not exist
        or cannot be loaded. Returns cached metrics in case of mid-write race.
    """
    path = run_dir / METRICS_FILENAME
    if not path.exists():
        return []
    try:
        mtime = path.stat().st_mtime
    except FileNotFoundError:
        return []
    key = str(path)
    cached = _METRICS_CACHE.get(key)
    if cached is not None and cached[0] == mtime:
        return cached[1]
    rows = read_metrics(run_dir)
    _METRICS_CACHE[key] = (mtime, rows)
    return rows


def _summarise_metrics(rows: list[dict]) -> dict:
    """Computes cumulative totals from a list of generational metrics rows.

    This function efficiently reduces the detailed metrics from each generation
    into a single summary dictionary, including totals for LLM calls, token
    usage, latency, and scoring outcomes.

    Args:
        rows: A list of dictionaries, where each dictionary contains metrics
              for a specific generation.

    Returns:
        A dictionary containing cumulative totals for various metrics across
        all generations.
    """
    totals = {
        "in_tokens": 0,
        "out_tokens": 0,
        "n_llm_calls": 0,
        "n_llm_retried": 0,
        "llm_seconds": 0.0,
        "score_seconds": 0.0,
        "n_scored": 0,
        "n_ok": 0,
        "n_timeout": 0,
        "n_inf": 0,
        "n_banned": 0,
        "by_role": {},
    }
    for r in rows:
        for role, st in (r.get("llm_calls") or {}).items():
            totals["in_tokens"] += st.get("in_tokens_total", 0) or 0
            totals["out_tokens"] += st.get("out_tokens_total", 0) or 0
            totals["n_llm_calls"] += st.get("n", 0) or 0
            totals["n_llm_retried"] += st.get("retried", 0) or 0
            mean = (st.get("latency_ms") or {}).get("mean") or 0
            totals["llm_seconds"] += (mean * (st.get("n", 0) or 0)) / 1000.0
            by_role = totals["by_role"].setdefault(
                role,
                {
                    "in_tokens": 0,
                    "out_tokens": 0,
                    "n": 0,
                    "retried": 0,
                    "seconds": 0.0,
                },
            )
            by_role["in_tokens"] += st.get("in_tokens_total", 0) or 0
            by_role["out_tokens"] += st.get("out_tokens_total", 0) or 0
            by_role["n"] += st.get("n", 0) or 0
            by_role["retried"] += st.get("retried", 0) or 0
            by_role["seconds"] += (mean * (st.get("n", 0) or 0)) / 1000.0
        sc = r.get("scoring") or {}
        totals["n_scored"] += sc.get("n", 0) or 0
        totals["n_ok"] += sc.get("ok", 0) or 0
        totals["n_timeout"] += sc.get("timeout", 0) or 0
        totals["n_inf"] += sc.get("inf", 0) or 0
        totals["n_banned"] += sc.get("banned", 0) or 0
        mean = (sc.get("latency_ms") or {}).get("mean") or 0
        totals["score_seconds"] += (mean * (sc.get("n", 0) or 0)) / 1000.0
    return totals


def _load_task_spec(run_dir: Path) -> dict:
    """Loads the `task_spec.yaml` file from a run directory.

    This function attempts to load the `task_spec.yaml` file, which contains
    the configuration and callable references for an EDGAR run. It uses
    `yaml.safe_load` first and falls back to `yaml.unsafe_load` to tolerate
    Python-object tags (e.g., `!python/object/new:edgar.llm.llm_calling.CyclingModel`)
    that might be present in files written by the same machine (e.g., during
    `fake-LLM` test runs).

    Args:
        run_dir: The path to the run directory.

    Returns:
        A dictionary representing the contents of `task_spec.yaml`, or an
        empty dictionary if the file does not exist or cannot be parsed.
    """
    path = run_dir / "task_spec.yaml"
    if not path.exists():
        return {}
    try:
        with open(path) as f:
            return yaml.safe_load(f) or {}
    except yaml.YAMLError:
        try:
            with open(path) as f:
                return yaml.unsafe_load(f) or {}
        except Exception:
            return {}


def _read_log_tail(run_dir: Path, max_lines: int = 200) -> list[str]:
    """Reads the tail of the `run.log` file for a given run directory.

    Args:
        run_dir: The path to the run directory.
        max_lines: The maximum number of lines to read from the end of the log file.

    Returns:
        A list of strings, where each string is a line from the end of the
        `run.log` file. Returns an empty list if the file does not exist
        or cannot be read.
    """
    path = run_dir / "run.log"
    if not path.exists():
        return []
    try:
        with open(path) as f:
            lines = f.readlines()
        return [ln.rstrip("\n") for ln in lines[-max_lines:]]
    except OSError:
        return []


# ── JSON sanitiser ──


def _clean(v: Any) -> Any:
    """Recursively converts input values into JSON-safe types.

    This function handles common Python types, NumPy scalars and arrays,
    converting them to their JSON-compatible counterparts. It specifically
    converts `None`, `NotValidated`, `NaN`, and `inf` float values to `None`.

    Args:
        v: The value to clean. Can be any Python object.

    Returns:
        A JSON-safe representation of the input value.
    """
    if v is None:
        return None
    if isinstance(v, NotValidated):
        return None
    if isinstance(v, (bool, int, str)):
        return v
    if isinstance(v, float):
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    if isinstance(v, np.generic):
        return _clean(v.item())
    if isinstance(v, np.ndarray):
        return _clean(v.tolist())
    if isinstance(v, (list, tuple, set, frozenset)):
        return [_clean(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _clean(val) for k, val in v.items()}
    return str(v)


# ── Run discovery ──


def list_runs(roots: list[Path]) -> list[dict]:
    """Scans specified root directories for EDGAR runs and returns a summary list.

    An "EDGAR run" is identified as any directory containing a `task_spec.yaml`
    file. Conventionally, these are organized under
    `program_databases/<task_name>/YYYY-MM-DD/HH-MM-SS/`.

    Args:
        roots: A list of `Path` objects representing the directories to scan for runs.

    Returns:
        A list of dictionaries, where each dictionary is a compact summary
        ("run card") for an EDGAR run, sorted by start time in reverse
        chronological order (newest first).
    """
    out: list[dict] = []
    seen: set[str] = set()
    for root in roots:
        root = Path(root)
        if not root.exists():
            continue
        for spec in root.glob("**/task_spec.yaml"):
            run_dir = spec.parent
            key = str(run_dir.resolve())
            if key in seen:
                continue
            seen.add(key)
            out.append(_run_card(run_dir))
    out.sort(key=lambda d: d.get("started_at_ts") or 0, reverse=True)
    return out


def _run_card(run_dir: Path) -> dict:
    """Generates a compact summary card for a specific EDGAR run.

    This function provides a lightweight overview suitable for run selection
    interfaces. It avoids loading the full `Population` unless basic status
    information is missing, making it efficient for displaying many runs.

    Args:
        run_dir: The path to the run directory.

    Returns:
        A dictionary containing a compact summary of the run, including
        `run_id`, `task_name`, `started_at`, `status`, `current_gen`,
        `n_programs`, and `best_loss`.
    """
    spec = _load_task_spec(run_dir)
    status_doc = read_status(run_dir) or {"state": "complete"}
    derived, is_stale = _derived_state(status_doc)
    started_at_ts = status_doc.get("started_at")
    if not started_at_ts:
        try:
            started_at_ts = run_dir.stat().st_ctime
        except OSError:
            started_at_ts = None

    # Fast best-effort program/loss counts via the cached Population only if
    # population.jsonl exists. Loading is O(size) but the per-mtime cache
    # makes repeated polls cheap.
    pop = _load_population(run_dir)
    n_programs = len(pop) if pop else 0
    best_loss = _best_loss(pop, split="validate") or _best_loss(pop, split="discover")

    return {
        "run_id": _run_id(run_dir),
        "run_dir": str(run_dir),
        "task_name": spec.get("task_name") or run_dir.name,
        "started_at_ts": started_at_ts,
        "started_at": _format_ts(started_at_ts),
        "status": derived,
        "is_stale": is_stale,
        "current_gen": status_doc.get("current_gen"),
        "n_generations": (spec.get("evolution") or {}).get("n_generations"),
        "n_islands": (spec.get("evolution") or {}).get("n_islands"),
        "n_programs": n_programs,
        "best_loss": best_loss,
    }


def _run_id(run_dir: Path) -> str:
    """Generates a stable, URL-safe identifier for a run.

    The ID is constructed from the last two parts of the run directory's path
    (e.g., `YYYY-MM-DD_HH-MM-SS`). If the path structure doesn't conform
    to this, the directory name itself is used as a fallback.

    Args:
        run_dir: The path to the run directory.

    Returns:
        A string representing the stable, URL-safe run ID.
    """
    parts = run_dir.parts
    if len(parts) >= 2:
        return f"{parts[-2]}_{parts[-1]}"
    return run_dir.name


def resolve_run_dir(run_id: str, roots: list[Path]) -> Path | None:
    """Resolves a run ID back to its on-disk run directory.

    This function is the inverse of `_run_id`, searching through specified
    root directories to find the `Path` corresponding to a given run ID.

    Args:
        run_id: The stable, URL-safe identifier of the run.
        roots: A list of `Path` objects representing the directories to scan.

    Returns:
        The `Path` object to the run directory if found, otherwise `None`.
    """
    for root in roots:
        root = Path(root)
        if not root.exists():
            continue
        for spec in root.glob("**/task_spec.yaml"):
            if _run_id(spec.parent) == run_id:
                return spec.parent
    return None


# ── Summary ──

STALE_THRESHOLD_S = 180.0
"""Threshold in seconds beyond which a 'running' or 'starting' run is considered 'stale' (failed)."""


def _derived_state(status_doc: dict | None) -> tuple[str, bool]:
    """Determines the derived state and staleness of a run.

    A run is considered "stale" if its `status.json` hasn't been updated
    within `STALE_THRESHOLD_S` seconds while its raw state is 'running' or
    'starting'. This helps to identify runs that terminated abnormally
    without explicitly updating their status to 'failed'. Stale runs are
    reported as 'failed' for dashboard UI purposes, but their `status.json`
    remains untouched.

    Args:
        status_doc: The dictionary loaded from `status.json`, or `None` if not found.

    Returns:
        A tuple containing:
        - The derived state (`str`): 'complete', 'running', 'starting', or 'failed' (if stale).
        - `is_stale` (`bool`): True if the run is considered stale, False otherwise.
    """
    if status_doc is None:
        return "complete", False
    state = status_doc.get("state", "complete")
    updated_at = status_doc.get("updated_at") or status_doc.get("started_at")
    if state in ("running", "starting") and updated_at:
        try:
            if (time.time() - float(updated_at)) > STALE_THRESHOLD_S:
                return "failed", True
        except (TypeError, ValueError):
            pass
    return state, False


def load_run_summary(run_dir: Path) -> dict:
    """Loads and aggregates a comprehensive summary of an EDGAR run.

    This function combines information from `task_spec.yaml`, `status.json`,
    `population.jsonl`, `island_census.jsonl`, and `metrics.jsonl` to
    provide a detailed overview of the experiment, including configuration,
    status, and overall performance statistics.

    Args:
        run_dir: The path to the run directory.

    Returns:
        A dictionary containing a comprehensive summary of the EDGAR run.
    """
    spec = _load_task_spec(run_dir)
    status_doc = read_status(run_dir) or {"state": "complete"}
    derived, is_stale = _derived_state(status_doc)
    pop = _load_population(run_dir)
    census = _load_census(run_dir)

    evolution = spec.get("evolution") or {}
    llms = spec.get("llms") or {}
    scoring = spec.get("scoring") or {}
    schemas = spec.get("prompt_schemas") or {}
    model_schema = schemas.get("model") or {}

    started_at_ts = status_doc.get("started_at")
    elapsed_s = (time.time() - started_at_ts) if started_at_ts else None

    alive_idxs = _alive_set(census)
    best_discover = _best_loss(pop, "discover")
    best_validate = _best_loss(pop, "validate")
    discover_n = sum(1 for p in (pop or []) if _finite(p.program_losses.discover.final))

    return {
        "run_id": _run_id(run_dir),
        "run_dir": str(run_dir),
        "task_name": spec.get("task_name") or run_dir.name,
        "git_sha": spec.get("git_sha"),
        "git_dirty": spec.get("git_dirty"),
        "created_at": spec.get("created_at"),
        "started_at_ts": started_at_ts,
        "started_at": _format_ts(started_at_ts),
        "elapsed_s": elapsed_s,
        "data_path": (spec.get("io") or {}).get("data_path"),
        "n_generations": evolution.get("n_generations"),
        "n_islands": evolution.get("n_islands"),
        "batch_size": evolution.get("batch_size"),
        "num_parents": llms.get("num_parents"),
        "llms": {
            "model": _llm_name(llms.get("model_llm")),
            "param_est": _llm_name(llms.get("param_est_llm")),
            "jax_translator": _llm_name(llms.get("jax_model_translator_llm")),
            "latex": _llm_name(llms.get("jax_model_translator_llm")),
        },
        "scoring": {
            "param_penalty_weight": scoring.get("param_penalty_weight"),
            "timeout_s": scoring.get("timeout_s"),
        },
        "project_params": _clean(spec.get("project_params") or {}),
        "prompt": {
            "base": model_schema.get("base", ""),
            "code_guidelines": model_schema.get("code_guidelines", ""),
            "explore": model_schema.get("explore", ""),
            "exploit": model_schema.get("exploit", ""),
        },
        "status": derived,
        "raw_status": status_doc.get("state", "complete"),
        "is_stale": is_stale,
        "current_gen": status_doc.get("current_gen"),
        "error": status_doc.get("error")
        or ("run appears stalled (no status update for >60s)" if is_stale else None),
        "n_programs": len(pop) if pop else 0,
        "n_alive": len(alive_idxs),
        "n_scored_discover": discover_n,
        "best_discover_loss": best_discover,
        "best_validate_loss": best_validate,
        "totals": _summarise_metrics(_load_metrics(run_dir)),
    }


# ── Live state ──


def load_live_state(run_dir: Path) -> dict:
    """Loads and aggregates the live, real-time state of an EDGAR run.

    This function provides up-to-the-minute information necessary for live
    monitoring of an active experiment, including current generation,
    elapsed time, estimated time to completion (ETA), island populations,
    best programs per generation, overall best program, and LLM success rates.

    Args:
        run_dir: The path to the run directory.

    Returns:
        A dictionary containing the live state of the EDGAR run.
    """
    spec = _load_task_spec(run_dir)
    status_doc = read_status(run_dir) or {"state": "complete"}
    derived, is_stale = _derived_state(status_doc)
    pop = _load_population(run_dir)
    census = _load_census(run_dir)
    metrics_rows = _load_metrics(run_dir)
    totals = _summarise_metrics(metrics_rows)

    evolution = spec.get("evolution", {})
    scoring = spec.get("scoring", {})
    n_gens = evolution.get("n_generations") or 0
    n_islands = evolution.get("n_islands") or 0
    n_param_ests = scoring.get("n_param_ests", 1)
    started_at_ts = status_doc.get("started_at")
    elapsed_s = (time.time() - started_at_ts) if started_at_ts else 0.0
    current_gen = status_doc.get("current_gen")

    eta_s = _estimate_eta(current_gen, n_gens, elapsed_s, derived)

    alive_idxs = _alive_set(census)
    islands = _islands_payload(pop, census, n_islands, alive_idxs)
    best_per_gen = _best_per_gen(pop)
    best = _best_program(pop)
    success_rates = _success_rates_latest_gen(pop, n_param_ests)

    return {
        "status": derived,
        "raw_status": status_doc.get("state", "complete"),
        "is_stale": is_stale,
        "current_gen": current_gen,
        "current_stage": status_doc.get("current_stage"),
        "n_gens": n_gens,
        "elapsed_s": elapsed_s,
        "eta_s": eta_s,
        "started_at": _format_ts(started_at_ts),
        "n_islands": n_islands,
        "n_programs": len(pop) if pop else 0,
        "n_alive": len(alive_idxs),
        "islands": islands,
        "best_per_gen": best_per_gen,
        "best": best,
        "success_rates": success_rates,
        "metrics": metrics_rows,
        "totals": totals,
        "last_metrics": status_doc.get("last_metrics"),
        "recent_log": _read_log_tail(run_dir, max_lines=60),
        "error": status_doc.get("error")
        or ("run appears stalled (no status update for >60s)" if is_stale else None),
    }


def _islands_payload(
    pop: Population | None,
    census: list[list[set[int]]],
    n_islands: int,
    alive_idxs: set[int],
) -> list[dict]:
    """Prepares a payload containing information about each island.

    This function organizes programs by their originating island, providing
    details such as the number of 'alive' programs and a list of program
    cards for all programs ever born on that island.

    Args:
        pop: The `Population` object, or None if not available.
        census: The island census data.
        n_islands: The total number of islands configured for the run.
        alive_idxs: A set of indices of programs currently considered 'alive'.

    Returns:
        A list of dictionaries, where each dictionary represents an island
        and contains its index, the number of alive programs, and a list of
        program cards.
    """
    if not pop:
        return []
    by_island: dict[int, list[Program]] = {i: [] for i in range(n_islands)}
    for p in pop:
        if p.birth.island in by_island:
            by_island[p.birth.island].append(p)
        elif p.birth.island == -1:
            continue  # seeds: surfaced separately in the UI if needed
    rows = []
    for island_idx, progs in sorted(by_island.items()):
        progs_sorted = sorted(progs, key=lambda x: (x.birth.generation, x.idx))
        rows.append(
            {
                "idx": island_idx,
                "size_alive": sum(1 for p in progs_sorted if p.idx in alive_idxs),
                "programs": [_program_card(p, alive_idxs) for p in progs_sorted],
            }
        )
    return rows


def _program_card(p: Program, alive_idxs: set[int]) -> dict:
    """Generates a compact summary card for a single program.

    Args:
        p: The `Program` object.
        alive_idxs: A set of indices of programs currently considered 'alive'.

    Returns:
        A dictionary containing key details of the program, suitable for
        display in lists or tables, including losses, rank, parentage, and status.
    """
    return {
        "idx": p.idx,
        "name": p.name or f"P{p.idx}",
        "gen": p.birth.generation,
        "island": p.birth.island,
        "mode": p.birth.mode,
        "llm": p.birth.llm_name,
        "n_params": p.n_params,
        "loss_discover": _safe_loss(p.program_losses.discover.final),
        "loss_discover_init": _safe_loss(p.program_losses.discover.init),
        "loss_validate": _safe_loss(p.program_losses.validate.final),
        "rank": p.rank,
        "parents": list(p.birth.parent_indices),
        "alive": p.idx in alive_idxs,
        "status": p.status,
        "has_image": bool(p.image_path),
    }


def _alive_set(census: list[list[set[int]]]) -> set[int]:
    """Determines the set of program indices that are currently 'alive'.

    This is derived from the latest generation's island census. Programs are
    considered alive if they are present in any island's population in the
    most recent census snapshot.

    Args:
        census: The island census data, a list of generational snapshots.

    Returns:
        A set of integer indices of programs that are currently alive.
    """
    if not census:
        return set()
    last = census[-1]
    out: set[int] = set()
    for island in last:
        out |= island if isinstance(island, set) else set(island)
    return out


def _best_per_gen(pop: Population | None) -> list[dict]:
    """Identifies the best program (lowest discover loss) for each generation.

    Args:
        pop: The `Population` object, or None if not available.

    Returns:
        A list of dictionaries, where each dictionary contains the generation
        number and the best discover loss for that generation.
    """
    if not pop:
        return []
    by_gen: dict[int, float] = {}
    for p in pop:
        v = p.program_losses.discover.final
        if not _finite(v):
            continue
        cur = by_gen.get(p.birth.generation)
        if cur is None or v < cur:
            by_gen[p.birth.generation] = float(v)
    return [{"gen": g, "loss": l} for g, l in sorted(by_gen.items())]


def _best_program(pop: Population | None) -> dict | None:
    """Identifies the overall best program (lowest discover loss) across all generations.

    Args:
        pop: The `Population` object, or None if not available.

    Returns:
        A dictionary containing details of the overall best program,
        or None if no finite-loss programs are found.
    """
    if not pop:
        return None
    candidates = [p for p in pop if _finite(p.program_losses.discover.final)]
    if not candidates:
        return None
    p = min(candidates, key=lambda x: x.program_losses.discover.final)
    return {
        "idx": p.idx,
        "name": p.name or f"P{p.idx}",
        "loss": float(p.program_losses.discover.final),
        "gen": p.birth.generation,
        "island": p.birth.island,
        "n_params": p.n_params,
    }


def _success_rates_latest_gen(
    pop: Population | None, n_param_ests: int = 1
) -> dict | None:
    """Calculates success rates for various stages (model generation, parameter
    estimator generation, JAX translation, scoring) among programs born in the
    latest generation.

    Args:
        pop: The `Population` object, or None if not available.
        n_param_ests: The configured number of parameter estimators expected
                      per program. Defaults to 1.

    Returns:
        A dictionary containing success rates for the latest generation,
        or None if no programs were born in the latest generation.
    """
    if not pop:
        return None
    gens = sorted({p.birth.generation for p in pop if p.birth.generation >= 0})
    if not gens:
        return None
    last_gen = gens[-1]
    born = [p for p in pop if p.birth.generation == last_gen]
    n = len(born)
    if n == 0:
        return None

    # Calculate the percentage of successfully generated parameter estimators out of maximum possible (n * n_param_ests)
    total_expected = n * n_param_ests
    total_generated = sum(
        len(p.code.param_est) for p in born if isinstance(p.code.param_est, list)
    )
    param_est_rate = total_generated / total_expected if total_expected > 0 else 0.0

    return {
        "gen": last_gen,
        "n": n,
        "model": sum(1 for p in born if p.code.model is not None) / n,
        "param_est": param_est_rate,
        "jax": sum(1 for p in born if p.code.model_jax is not None) / n,
        "scored": sum(1 for p in born if _finite(p.program_losses.discover.final)) / n,
    }


def _estimate_eta(
    current_gen: int | None,
    n_gens: int,
    elapsed_s: float,
    state: str | None,
) -> float | None:
    """Estimates the remaining time until an EDGAR run completes.

    The ETA is calculated based on the average time spent per completed
    generation and the number of generations remaining.

    Args:
        current_gen: The index of the current generation (0-indexed).
        n_gens: The total number of generations configured for the run.
        elapsed_s: The total elapsed time in seconds since the run started.
        state: The current status of the run (e.g., 'running', 'starting', 'complete').

    Returns:
        The estimated time remaining in seconds, or None if the ETA cannot be
        calculated (e.g., run is not active, or insufficient data).
    """
    if state not in ("running", "starting"):
        return None
    if current_gen is None or current_gen < 0:
        return None
    completed = current_gen + 1
    if completed <= 0 or n_gens <= 0:
        return None
    per_gen = elapsed_s / completed
    remaining = max(n_gens - completed, 0)
    return per_gen * remaining


# ── Program list / detail ──


def load_program_list(run_dir: Path) -> list[dict]:
    """Loads a list of summary cards for all programs in a run.

    The list is sorted by rank (ascending), then by validation loss
    (ascending), and finally by program index. Programs without a rank are
    sorted last.

    Args:
        run_dir: The path to the run directory.

    Returns:
        A list of dictionaries, where each dictionary is a program card
        (`_program_card`) for a program in the population.
    """
    pop = _load_population(run_dir)
    if not pop:
        return []
    census = _load_census(run_dir)
    alive_idxs = _alive_set(census)
    items = [_program_card(p, alive_idxs) for p in pop]
    items.sort(key=_sort_key)
    return items


def _sort_key(card: dict) -> tuple:
    """Provides a sorting key for program cards.

    Programs are sorted primarily by their rank (ascending), with programs
    that have no rank (None) placed at the end. Secondary sorting is by
    validation loss (ascending), or discover loss if validation loss is not
    available. Finally, programs are sorted by their index (ascending) as a tie-breaker.

    Args:
        card: A program summary dictionary (program card).

    Returns:
        A tuple suitable for sorting, prioritizing rank, then loss, then index.
    """
    rank = card.get("rank")
    loss = (
        card.get("loss_validate")
        if card.get("loss_validate") is not None
        else card.get("loss_discover")
    )
    return (
        rank if rank is not None else 10**9,
        loss if loss is not None else 10**9,
        card.get("idx", 0),
    )


def load_program_detail(run_dir: Path, idx: int) -> dict | None:
    """Loads comprehensive detailed information for a specific program.

    This includes its full code, losses, parameters, sample-wise loss summaries,
    optimization trajectories, image URLs, and lineage (parents and children).

    Args:
        run_dir: The path to the run directory.
        idx: The index of the program to retrieve details for.

    Returns:
        A dictionary containing all available details for the specified program,
        or None if the program does not exist or the population cannot be loaded.
    """
    pop = _load_population(run_dir)
    if not pop or idx < 0 or idx >= len(pop):
        return None
    census = _load_census(run_dir)
    alive_idxs = _alive_set(census)
    p = pop[idx]

    base = _program_card(p, alive_idxs)

    children = [c.idx for c in pop if idx in c.birth.parent_indices]

    def _std(lst):
        if not lst:
            return None
        valid = [
            float(x)
            for x in lst
            if x is not None
            and not isinstance(x, NotValidated)
            and math.isfinite(float(x))
        ]
        if not valid:
            return None
        return float(np.std(valid))

    def _min(lst):
        if not lst:
            return None
        valid = [
            float(x)
            for x in lst
            if x is not None
            and not isinstance(x, NotValidated)
            and math.isfinite(float(x))
        ]
        if not valid:
            return None
        return float(np.min(valid))

    def _max(lst):
        if not lst:
            return None
        valid = [
            float(x)
            for x in lst
            if x is not None
            and not isinstance(x, NotValidated)
            and math.isfinite(float(x))
        ]
        if not valid:
            return None
        return float(np.max(valid))

    sample_losses_summary = None
    if p.sample_losses is not None:
        arr = np.asarray(p.sample_losses, dtype=float)
        if arr.size:
            finite_arr = arr[np.isfinite(arr)]
            if finite_arr.size:
                sample_losses_summary = {
                    "n": int(arr.size),
                    "n_finite": int(finite_arr.size),
                    "min": float(finite_arr.min()),
                    "median": float(np.median(finite_arr)),
                    "mean": float(finite_arr.mean()),
                    "max": float(finite_arr.max()),
                }

    trajectories_summary = None
    if p.program_losses.discover.trajectories is not None:
        trajectories = p.program_losses.discover.trajectories
        trajectories_summary = []
        # trajectories.shape[0] refers to the number of parallel optimizations run for a program
        for i in range(trajectories.shape[0]):
            traj = trajectories[i]
            trajectories_summary.append(
                {
                    "idx": i,
                    "init": float(traj[0]),
                    "final": float(traj[-1]),
                }
            )

    fingerprint_shape = None
    if p.eval_fingerprint is not None:
        try:
            fingerprint_shape = list(np.asarray(p.eval_fingerprint).shape)
        except Exception:
            fingerprint_shape = None

    params_clean = _clean(p.params) if p.params is not None else None

    return {
        **base,
        "reconstructed_prompt": _reconstruct_model_prompt(run_dir, pop, idx),
        "code": {
            "model": p.code.model or "",
            "param_est": p.param_est_code,
            "model_jax": p.code.model_jax or "",
        },
        "losses": {
            "discover": {
                "init": _safe_loss(p.program_losses.discover.init),
                "final": _safe_loss(p.program_losses.discover.final),
            },
            "validate": {
                "init": _safe_loss(p.program_losses.validate.init),
                "final": _safe_loss(p.program_losses.validate.final),
            },
        },
        "params": params_clean,
        "sample_losses_summary": sample_losses_summary,
        "fingerprint_shape": fingerprint_shape,
        "image_path": p.image_path,
        "image_url": _image_url_for(p),
        "fit_image_url": _fit_image_url_for(p),
        "trajectory_image_url": _trajectory_image_url_for(p),
        "parents_detail": [
            {
                "idx": parent_idx,
                "name": pop[parent_idx].name or f"P{parent_idx}",
                "loss_discover": _safe_loss(
                    pop[parent_idx].program_losses.discover.final
                ),
                "gen": pop[parent_idx].birth.generation,
                "island": pop[parent_idx].birth.island,
            }
            for parent_idx in p.birth.parent_indices
            if 0 <= parent_idx < len(pop)
        ],
        "children": children,
        "children_detail": [
            {
                "idx": c.idx,
                "name": c.name or f"P{c.idx}",
                "loss_discover": _safe_loss(c.program_losses.discover.final),
                "gen": c.birth.generation,
                "island": c.birth.island,
            }
            for c in pop
            if idx in c.birth.parent_indices
        ],
    }


def _image_url_for(p: Program) -> str | None:
    """Generates a relative URL for the image feedback plot of a program.

    The URL adheres to the convention `image/gen_NNN/island_NNN/batch_NNN`.
    This image is typically used for LLM image-based feedback.

    Args:
        p: The `Program` object.

    Returns:
        A string representing the relative URL for the image feedback plot,
        or None if the program's generation or island information is missing.
    """
    if p.birth.generation is None or p.birth.island is None:
        return None
    if p.birth.generation < 0 or p.birth.island < 0:
        return None
    return f"image/gen_{p.birth.generation:03d}/island_{p.birth.island:03d}/batch_{p.birth.batch_index:03d}"


def _fit_image_url_for(p: Program) -> str | None:
    """Generates a relative URL for the model fit image of a program.

    The URL adheres to the convention `fit_image/{program_idx}`.
    This image displays how well the program's model fits the data.

    Args:
        p: The `Program` object.

    Returns:
        A string representing the relative URL for the model fit image,
        or None if the program's index or `fit_image_path` is missing.
    """
    if p.idx is None or p.fit_image_path is None:
        return None
    return f"fit_image/{p.idx}"


def _trajectory_image_url_for(p: Program) -> str | None:
    """Generates a relative URL for the optimization trajectory image of a program.

    The URL adheres to the convention `trajectory_image/{program_idx}`.
    This image visualizes the loss curves during parameter optimization.

    Args:
        p: The `Program` object.

    Returns:
        A string representing the relative URL for the trajectory image,
        or None if the program's index or `trajectory_image_path` is missing.
    """
    if p.idx is None or getattr(p, "trajectory_image_path", None) is None:
        return None
    return f"trajectory_image/{p.idx}"


# ── Family Tree ──


def load_family_tree_data(run_dir: Path) -> dict:
    """Loads and formats data for visualizing the evolutionary family tree of programs.

    This function prepares data structures compatible with Plotly for rendering
    a hierarchical graph of programs, showing parent-child relationships,
    generational layout, and node attributes (color, size, symbol) based on
    their performance and status.

    Args:
        run_dir: The path to the run directory.

    Returns:
        A dictionary containing all necessary data for rendering the family tree,
        including node positions, edges, IDs, labels, hover text, colors,
        symbols, sizes, and a parent map for interactive highlighting.
    """
    pop = _load_population(run_dir)
    if not pop:
        return {}
    census = _load_census(run_dir)
    alive_idxs = _alive_set(census)

    pos = _layout_by_generation(pop)
    edge_x, edge_y = _build_edges(pop, pos)
    nodes = _build_nodes(pop, pos, alive_idxs)
    parent_map = _build_parent_map(pop)
    pos_map = {str(idx): list(xy) for idx, xy in pos.items()}

    return {
        "parent_map": parent_map,
        "pos_map": pos_map,
        "edge_x": edge_x,
        "edge_y": edge_y,
        "node_x": nodes["x"],
        "node_y": nodes["y"],
        "node_ids": nodes["ids"],
        "node_labels": nodes["labels"],
        "node_hover": nodes["hover"],
        "node_colors": nodes["colors"],
        "node_symbols": nodes["symbols"],
        "node_sizes": nodes["sizes"],
    }


def _layout_by_generation(pop: Population) -> dict[int, tuple[float, float]]:
    """Calculates a hierarchical layout for programs based on their generation.

    Programs from the same generation are placed on the same horizontal level
    (y-coordinate = -generation), and then evenly distributed horizontally.

    Args:
        pop: The `Population` object containing all programs.

    Returns:
        A dictionary mapping program indices to their (x, y) coordinates
        in the layout.
    """
    by_gen: dict[int, list[int]] = {}
    for i in range(len(pop)):
        p = pop[i]
        by_gen.setdefault(p.birth.generation, []).append(p.idx)

    pos = {}
    for gen, ids in by_gen.items():
        n = len(ids)
        for i, idx in enumerate(sorted(ids)):
            x = (i - (n - 1) / 2.0) if n > 1 else 0.0
            pos[idx] = (x, -gen)
    return pos


def _build_edges(pop: Population, pos: dict) -> tuple[list, list]:
    """Constructs the x and y coordinates for drawing edges (parent-child links)
    in the family tree visualization.

    Args:
        pop: The `Population` object containing all programs.
        pos: A dictionary mapping program indices to their (x, y) coordinates.

    Returns:
        A tuple containing two lists:
        - `edge_x`: List of x-coordinates for the edges, with `None` separating segments.
        - `edge_y`: List of y-coordinates for the edges, with `None` separating segments.
    """
    edge_x, edge_y = [], []
    for i in range(len(pop)):
        p = pop[i]
        for parent_idx in p.birth.parent_indices:
            if p.idx not in pos or parent_idx not in pos:
                continue
            x_c, y_c = pos[p.idx]
            x_p, y_p = pos[parent_idx]
            edge_x.extend([x_p, x_c, None])
            edge_y.extend([y_p, y_c, None])
    return edge_x, edge_y


def _build_nodes(pop: Population, pos: dict, alive_idxs: set[int]) -> dict:
    """Prepares display properties for each node (program) in the family tree.

    This includes their positions, IDs, labels, hover text, colors (based on loss),
    symbols (based on rank/seed status), and sizes.

    Args:
        pop: The `Population` object containing all programs.
        pos: A dictionary mapping program indices to their (x, y) coordinates.
        alive_idxs: A set of indices of programs currently considered 'alive'.

    Returns:
        A dictionary containing lists of x, y, ids, labels, hover text, colors,
        symbols, and sizes, structured for Plotly visualization.
    """
    x, y, ids, labels, hover = [], [], [], [], []
    colors, symbols, sizes = [], [], []
    for i in range(len(pop)):
        p = pop[i]
        if p.idx not in pos:
            continue
        nx, ny = pos[p.idx]
        x.append(nx)
        y.append(ny)
        ids.append(p.idx)
        label = p.name or f"P{p.idx}"
        labels.append(_wrap_label(label))
        hover_label = f"★ {label}" if p.rank == 1 else label
        final_loss = p.program_losses.discover.final
        loss_val = f"{final_loss:.4f}" if _finite(final_loss) else "N/A"
        hover.append(f"{hover_label}<br>loss: {loss_val}")
        colors.append(_node_colour(final_loss))
        if p.rank == 1:
            symbols.append("star")
            sizes.append(24)
        elif p.birth.generation == -1:
            symbols.append("square")
            sizes.append(20)
        else:
            symbols.append("circle")
            sizes.append(16)
    return {
        "x": x,
        "y": y,
        "ids": ids,
        "labels": labels,
        "hover": hover,
        "colors": colors,
        "symbols": symbols,
        "sizes": sizes,
    }


def _wrap_label(label: str, width: int = 15) -> str:
    """Wraps a string with `<br>` tags for multi-line display in Plotly.

    Short numeric IDs are not wrapped.

    Args:
        label: The string label to wrap.
        width: The maximum desired line width before wrapping.

    Returns:
        The wrapped string, with `<br>` tags replacing newlines.
    """
    if not label:
        return ""
    # Avoid wrapping short numeric IDs like "P0" or "P123"
    if len(label) <= width:
        return label
    return "<br>".join(textwrap.wrap(label, width=width, break_long_words=True))


def _node_colour(loss) -> str:
    """Determines the color of a node based on its loss value.

    This provides a visual indicator of program performance in the family tree.

    Args:
        loss: The loss value of the program.

    Returns:
        A hexadecimal string representing the color.
    """
    if not _finite(loss):
        return "#52525b"  # Grey for non-finite losses
    if loss < 30:
        return "#34d399"  # Green for good performance
    if loss < 50:
        return "#fbbf24"  # Yellow for moderate performance
    return "#fb7185"  # Red for poorer performance


def _build_parent_map(pop: Population) -> dict[str, list[int]]:
    """Builds a map from child program indices to their parent program indices.

    This map is used in the JavaScript frontend for ancestor highlighting
    in the family tree visualization.

    Args:
        pop: The `Population` object containing all programs.

    Returns:
        A dictionary where keys are string representations of child program
        indices and values are lists of integer indices of their parents.
    """
    parent_map = {}
    for i in range(len(pop)):
        p = pop[i]
        parents = list(p.birth.parent_indices)
        if parents:
            parent_map[str(p.idx)] = parents
    return parent_map


# ── helpers ──


def _safe_loss(v: Any) -> float | None:
    """Safely converts a value to a float loss, handling non-finite values and `NotValidated`.

    This helper is used within the dashboard to ensure consistent display
    of loss values in the UI, mapping `None`, `NotValidated`, `NaN`, and `inf`
    to `None`. This differs from `_scoring_safe_loss` which maps to `float("inf")`
    for algorithmic use.

    Args:
        v: The value to convert.

    Returns:
        A finite float representation of the loss, or None if the value is
        `None`, `NotValidated`, `NaN`, or `inf`.
    """
    if v is None or isinstance(v, NotValidated):
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(f):
        return None
    return f


def _finite(v: Any) -> bool:
    """Checks if a value can be considered a finite float.

    This helper is used to filter out programs with non-finite or invalid
    loss values before further processing or display.

    Args:
        v: The value to check.

    Returns:
        True if the value is a finite float, False otherwise.
    """
    if v is None or isinstance(v, NotValidated):
        return False
    try:
        f = float(v)
    except (TypeError, ValueError):
        return False
    return math.isfinite(f)


def _best_loss(pop: Population | None, split: str) -> float | None:
    """Finds the best (minimum) finite loss for a specified split across all programs.

    Args:
        pop: The `Population` object, or None if not available.
        split: The loss split to consider ('discover' or 'validate').

    Returns:
        The minimum finite loss value for the specified split, or None if no
        finite losses are found.
    """
    if not pop:
        return None
    losses = []
    for p in pop:
        v = (
            p.program_losses.validate.final
            if split == "validate"
            else p.program_losses.discover.final
        )
        if _finite(v):
            losses.append(float(v))
    return min(losses) if losses else None


def _llm_name(v: Any) -> str | None:
    """Coerces an LLM configuration value into a displayable string name.

    This function handles both string-based LLM names (for real runs) and
    `CyclingModel` objects (from the fake-LLM test runner), and lists of strings,
    ensuring a clean string representation for the dashboard UI.

    Args:
        v: The LLM configuration value, which can be a string, a list of strings,
           or a `CyclingModel` object.

    Returns:
        A string representing the LLM's name, or None if the input is None.
    """
    if v is None:
        return None
    if isinstance(v, str):
        return v
    if isinstance(v, list) and all(isinstance(x, str) for x in v):
        return ",".join([x for x in v])
    name = getattr(v, "model_name", None)
    if name:
        return str(name)
    return type(v).__name__


def _format_ts(ts: float | None) -> str | None:
    """Formats a Unix timestamp into an ISO 8601 string, truncated to seconds.

    Args:
        ts: The Unix timestamp (float) or None.

    Returns:
        An ISO 8601 formatted string (e.g., 'YYYY-MM-DDTHH:MM:SS'),
        or None if the input timestamp is invalid.
    """
    if ts is None:
        return None
    try:
        return datetime.fromtimestamp(float(ts)).isoformat(timespec="seconds")
    except (OSError, ValueError, TypeError):
        return None