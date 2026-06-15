"""data.py — translate a run_dir into JSON-safe DTOs for the dashboard.

All functions take a Path to a run directory (the timestamped folder under
program_databases/) and return plain dicts/lists ready to ship over HTTP.

Caching: Population.load is the only expensive call (24 MB JSONL on the largest
runs). We memoise per (path, mtime) so repeated polls during a live run only
re-parse when a new generation lands.

Legacy tolerance: runs predating the status.json convention are treated as
status='complete' implicitly.
"""

from __future__ import annotations

import json
import math
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from ..evolution.island import load_island_census
from ..evolution.population import Population
from ..evolution.program import NotValidated, Program
from ..io.metrics import METRICS_FILENAME, read_metrics
from ..io.status import read_status


# ── Population cache (path, mtime) → Population ──
_POP_CACHE: dict[str, tuple[float, Population]] = {}
_CENSUS_CACHE: dict[str, tuple[float, list[list[set[int]]]]] = {}
_METRICS_CACHE: dict[str, tuple[float, list[dict]]] = {}


def _load_population(run_dir: Path) -> Population | None:
    """Cached Population.load. Returns None if file doesn't exist yet."""
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


def _load_census(run_dir: Path) -> list[list[set[int]]]:
    """Cached load_island_census. [] if not yet written.

    NOTE: as currently saved the JSON shape is
        census[generation][island_idx] -> list[int]
    (see save_island_census), so each top-level entry is a *generation snapshot*.
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
    """Cached metrics.jsonl loader. [] if not yet written."""
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
    """Cumulative totals across all generation rows. Cheap reduction."""
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
        mean = (sc.get("latency_ms") or {}).get("mean") or 0
        totals["score_seconds"] += (mean * (sc.get("n", 0) or 0)) / 1000.0
    return totals


def _load_task_spec(run_dir: Path) -> dict:
    """Load task_spec.yaml. Trust boundary is local-only: this file was written
    by the same machine running the dashboard, so we tolerate Python-object
    tags (e.g. CyclingModel objects from the fake-LLM test runner) via
    unsafe_load if safe_load fails.
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
    """Recursively convert to JSON-safe values. NaN/inf -> None, numpy -> python."""
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
    """Scan roots for runs and return a summary list, newest first.

    A "run" is any directory containing a task_spec.yaml file. Conventionally
    these are organised as program_databases/MM-DD/HH-MM-SS/.
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
    """Compact summary card for the run picker. Cheap — does NOT load population
    on every call unless cheap stat info is missing.
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
    """Stable URL-safe id for a run: MM-DD_HH-MM-SS or the dir name as a fallback."""
    parts = run_dir.parts
    if len(parts) >= 2:
        return f"{parts[-2]}_{parts[-1]}"
    return run_dir.name


def resolve_run_dir(run_id: str, roots: list[Path]) -> Path | None:
    """Inverse of _run_id: find the on-disk run directory for an id."""
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


def _derived_state(status_doc: dict | None) -> tuple[str, bool]:
    """Return (state, is_stale).

    A run is "stale" if status.json hasn't been touched in STALE_THRESHOLD_S
    seconds while still reading 'running' or 'starting'. This catches runs
    that died abnormally (SIGKILL, OOM, scoring-subprocess-induced abort)
    where the runner never got to flip status to 'failed' itself.

    We surface stale runs as 'failed' for UI purposes while leaving the raw
    status.json file untouched.
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
    spec = _load_task_spec(run_dir)
    status_doc = read_status(run_dir) or {"state": "complete"}
    derived, is_stale = _derived_state(status_doc)
    pop = _load_population(run_dir)
    census = _load_census(run_dir)
    metrics_rows = _load_metrics(run_dir)
    totals = _summarise_metrics(metrics_rows)

    evolution = spec.get("evolution") or {}
    n_gens = evolution.get("n_generations") or 0
    n_islands = evolution.get("n_islands") or 0
    started_at_ts = status_doc.get("started_at")
    elapsed_s = (time.time() - started_at_ts) if started_at_ts else 0.0
    current_gen = status_doc.get("current_gen")

    eta_s = _estimate_eta(current_gen, n_gens, elapsed_s, derived)

    alive_idxs = _alive_set(census)
    islands = _islands_payload(pop, census, n_islands, alive_idxs)
    best_per_gen = _best_per_gen(pop)
    best = _best_program(pop)
    success_rates = _success_rates_latest_gen(pop)

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
    """One row per island, with all programs ever born on that island."""
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
        "has_image": bool(p.image_path),
    }


def _alive_set(census: list[list[set[int]]]) -> set[int]:
    if not census:
        return set()
    last = census[-1]
    out: set[int] = set()
    for island in last:
        out |= island if isinstance(island, set) else set(island)
    return out


def _best_per_gen(pop: Population | None) -> list[dict]:
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


def _success_rates_latest_gen(pop: Population | None) -> dict | None:
    """Per-stage success rates among programs born in the last generation."""
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
    return {
        "gen": last_gen,
        "n": n,
        "model": sum(1 for p in born if p.code.model is not None) / n,
        "param_est": sum(1 for p in born if p.code.param_est is not None) / n,
        "jax": sum(1 for p in born if p.code.model_jax is not None) / n,
        "scored": sum(1 for p in born if _finite(p.program_losses.discover.final)) / n,
    }


def _estimate_eta(
    current_gen: int | None,
    n_gens: int,
    elapsed_s: float,
    state: str | None,
) -> float | None:
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
    pop = _load_population(run_dir)
    if not pop:
        return []
    census = _load_census(run_dir)
    alive_idxs = _alive_set(census)
    items = [_program_card(p, alive_idxs) for p in pop]
    items.sort(key=_sort_key)
    return items


def _sort_key(card: dict) -> tuple:
    """Rank ascending; programs with no rank sorted last; then by validate loss."""
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
    pop = _load_population(run_dir)
    if not pop or idx < 0 or idx >= len(pop):
        return None
    census = _load_census(run_dir)
    alive_idxs = _alive_set(census)
    p = pop[idx]

    base = _program_card(p, alive_idxs)

    children = [c.idx for c in pop if idx in c.birth.parent_indices]

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

    fingerprint_shape = None
    if p.eval_fingerprint is not None:
        try:
            fingerprint_shape = list(np.asarray(p.eval_fingerprint).shape)
        except Exception:
            fingerprint_shape = None

    params_clean = _clean(p.params) if p.params is not None else None

    return {
        **base,
        "code": {
            "model": p.code.model or "",
            "param_est": p.code.param_est or "",
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
    """Convention from edgar/llm/generate.py: per gen/island/batch."""
    if p.birth.generation is None or p.birth.island is None:
        return None
    if p.birth.generation < 0 or p.birth.island < 0:
        return None
    return f"image/gen_{p.birth.generation:03d}/island_{p.birth.island:03d}/batch_{p.birth.batch_index:03d}"


def _fit_image_url_for(p: Program) -> str | None:
    """Convention from edgar/io/plotting.py: P{idx:04d}.png."""
    if p.idx is None or p.fit_image_path is None:
        return None
    return f"fit_image/{p.idx}"


# ── helpers ──


def _safe_loss(v: Any) -> float | None:
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
    if v is None or isinstance(v, NotValidated):
        return False
    try:
        f = float(v)
    except (TypeError, ValueError):
        return False
    return math.isfinite(f)


def _best_loss(pop: Population | None, split: str) -> float | None:
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
    """Coerce an LLM field from task_spec.yaml to a display string.

    Real runs save the LLM name as a string. The fake-LLM test runner
    pickles a CyclingModel into the yaml; if so, surface its repr instead of
    leaking a Python object into the JSON response.
    If llm is a list of strings, join the strings with a comma in between
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
    if ts is None:
        return None
    try:
        return datetime.fromtimestamp(float(ts)).isoformat(timespec="seconds")
    except (OSError, ValueError, TypeError):
        return None
