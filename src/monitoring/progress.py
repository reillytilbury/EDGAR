"""Loss progress + GD-effect visualizations from a Population.

write_progress() emits two HTML files:
- loss_progress.html — perplexity P(L) over generations, one trace per island
- gd_effect.html — perplexity of init loss vs final loss; above-diagonal = GD helped
"""
from __future__ import annotations

import json
from html import escape
from pathlib import Path

from ..evolution.population import Population
from ..evolution.program import Program
from .data import (
    _COLOUR_MODE_JS,
    _SIDEBAR_CSS,
    _SIDEBAR_JS,
    build_sidebar_data,
    compute_alive_set,
    islands_of,
    island_colour,
    loss_str,
    perplexity,
    reference_loss,
    render,
    split_seeds_and_progs,
)


# ── loss progress ──

def _loss_traces_per_island(
    progs: list[Program],
    islands: list[int],
    L_0: float,
) -> list[dict]:
    n_islands = len(islands)
    traces = []
    for island_idx in islands:
        island_progs = [p for p in progs if p.birth.island == island_idx]
        if not island_progs:
            continue
        x_vals, y_vals, custom, hover = [], [], [], []
        jitter = (island_idx - (n_islands - 1) / 2.0) * 0.05
        for p in island_progs:
            x_vals.append(p.birth.generation + jitter)
            perp = perplexity(p.program_losses.discover.final, L_0)
            y_vals.append(perp)
            custom.append(p.idx)
            label = p.name or f"P{p.idx}"
            perp_s = f"{perp:.4f}" if perp is not None else "N/A"
            hover.append(
                f"<b>{label}</b><br>P(L): {perp_s}"
                f"<br>train loss: {loss_str(p.program_losses.discover.final)}"
                f"<br>mode: {p.birth.mode or ''}"
            )
        traces.append({
            "island_idx": island_idx,
            "x": x_vals,
            "y_with_penalty": y_vals,
            "y_without_penalty": y_vals,
            "custom": custom,
            "hover": hover,
            "colour": island_colour(island_idx),
        })
    return traces


def _loss_seed_data(seeds: list[Program], L_0: float) -> dict:
    x, y, custom, hover = [], [], [], []
    for s in seeds:
        perp = perplexity(s.program_losses.discover.final, L_0)
        x.append(-1)
        y.append(perp)
        custom.append(s.idx)
        label = s.name or f"S{s.birth.batch_index}"
        perp_s = f"{perp:.4f}" if perp is not None else "N/A"
        hover.append(
            f"<b>{label}</b><br>P(L): {perp_s}"
            f"<br>train loss: {loss_str(s.program_losses.discover.final)}"
        )
    return {"x": x, "y": y, "custom": custom, "hover": hover}


def _lineage_edges(
    population: Population,
    pos: dict[int, tuple[float, float]],
) -> tuple[list[dict], dict[str, list[int]], dict[str, list[int]]]:
    """Build {edge_segs, edge_map, parent_map} for ancestor highlighting."""
    edge_segs: list[dict] = []
    edge_map: dict[str, list[int]] = {}
    parent_map: dict[str, list[int]] = {}

    for i in range(len(population)):
        p = population[i]
        parents = list(p.birth.parent_indices)
        if not parents:
            continue
        parent_map[str(p.idx)] = parents
        edge_map[str(p.idx)] = []
        for parent_idx in parents:
            if p.idx not in pos or parent_idx not in pos:
                continue
            cx, cy = pos[p.idx]
            px, py = pos[parent_idx]
            edge_map[str(p.idx)].append(len(edge_segs))
            edge_segs.append({"x": [cx, px], "y": [cy, py]})
    return edge_segs, edge_map, parent_map


def _axis_range(values: list[float], pad_min: float, pad_frac: float, clamp_zero: bool = False) -> str:
    finite = [v for v in values if v is not None]
    if not finite:
        return "null"
    pad = max(pad_min, (max(finite) - min(finite)) * pad_frac)
    lo = min(finite) - pad
    hi = max(finite) + pad
    if clamp_zero:
        lo = max(0.0, lo)
    return json.dumps([lo, hi])


def _render_loss_progress(
    population: Population,
    sidebar_data: dict,
    out_dir: Path,
    task_name: str,
) -> Path:
    seeds, progs = split_seeds_and_progs(population)
    L_0 = reference_loss(seeds)
    islands = islands_of(progs)

    traces = _loss_traces_per_island(progs, islands, L_0)
    seed = _loss_seed_data(seeds, L_0)

    pos: dict[int, tuple[float, float]] = {}
    for t in traces:
        for x, y, idx in zip(t["x"], t["y_with_penalty"], t["custom"]):
            pos[idx] = (x, y)
    for x, y, idx in zip(seed["x"], seed["y"], seed["custom"]):
        pos[idx] = (x, y)
    edge_segs, edge_map, parent_map = _lineage_edges(population, pos)

    all_x = [xv for t in traces for xv in t["x"]] + seed["x"]
    all_y = [yv for t in traces for yv in t["y_with_penalty"] if yv is not None] \
        + [yv for yv in seed["y"] if yv is not None]

    title = f"{task_name} — Loss Progress"
    html = render("loss_progress.html", {
        "__TITLE_HTML__": escape(title),
        "__TITLE_JSON__": json.dumps(title),
        "__SIDEBAR_CSS__": _SIDEBAR_CSS,
        "__SIDEBAR_JS__": _SIDEBAR_JS,
        "__COLOUR_MODE_JS__": _COLOUR_MODE_JS,
        "__SIDEBAR_DATA__": json.dumps(sidebar_data, default=str),
        "__TRACES__": json.dumps(traces),
        "__ISLANDS__": json.dumps(islands),
        "__SEED_X__": json.dumps(seed["x"]),
        "__SEED_Y_PENALTY__": json.dumps(seed["y"]),
        "__SEED_Y_RAW__": json.dumps(seed["y"]),
        "__SEED_CUSTOM__": json.dumps(seed["custom"]),
        "__SEED_HOVER__": json.dumps(seed["hover"]),
        "__EDGE_SEGS_PENALTY__": json.dumps(edge_segs),
        "__EDGE_SEGS_RAW__": json.dumps(edge_segs),
        "__PARENT_MAP__": json.dumps(parent_map),
        "__EDGE_MAP__": json.dumps(edge_map),
        "__X_RANGE__": _axis_range(all_x, pad_min=0.5, pad_frac=0.04),
        "__Y_RANGE__": _axis_range(all_y, pad_min=0.05, pad_frac=0.05, clamp_zero=True),
    })

    out_path = out_dir / "loss_progress.html"
    out_path.write_text(html)
    return out_path


# ── GD effect ──

def _gd_traces_per_island(
    progs: list[Program],
    islands: list[int],
    L_0: float,
) -> tuple[list[dict], list[float]]:
    traces = []
    all_perp: list[float] = []
    for island_idx in islands:
        island_progs = [
            p for p in progs
            if p.birth.island == island_idx
            and p.program_losses.discover.init is not None
            and p.program_losses.discover.final is not None
        ]
        if not island_progs:
            continue
        x_vals, y_vals, custom, hover = [], [], [], []
        for p in island_progs:
            p_init = perplexity(p.program_losses.discover.init, L_0)
            p_final = perplexity(p.program_losses.discover.final, L_0)
            delta = p.program_losses.discover.init - p.program_losses.discover.final
            x_vals.append(p_init)
            y_vals.append(p_final)
            custom.append(p.idx)
            label = p.name or f"P{p.idx}"
            hover.append(
                f"<b>{label}</b>"
                f"<br>P(initial): {p_init:.4f}"
                f"<br>P(after GD): {p_final:.4f}"
                f"<br>raw improvement: {delta:.4f}"
                f"<br>mode: {p.birth.mode or ''}"
            )
            all_perp.extend(v for v in (p_init, p_final) if v is not None)
        traces.append({
            "island_idx": island_idx,
            "x": x_vals,
            "y": y_vals,
            "custom": custom,
            "hover": hover,
            "colour": island_colour(island_idx),
        })
    return traces, all_perp


def _gd_seed_data(seeds: list[Program], L_0: float) -> tuple[dict, list[float]]:
    x, y, custom, hover = [], [], [], []
    all_perp: list[float] = []
    for s in seeds:
        if s.program_losses.discover.init is None or s.program_losses.discover.final is None:
            continue
        p_init = perplexity(s.program_losses.discover.init, L_0)
        p_final = perplexity(s.program_losses.discover.final, L_0)
        delta = s.program_losses.discover.init - s.program_losses.discover.final
        x.append(p_init)
        y.append(p_final)
        custom.append(s.idx)
        label = s.name or f"S{s.birth.batch_index}"
        hover.append(
            f"<b>{label}</b>"
            f"<br>P(initial): {p_init:.4f}"
            f"<br>P(after GD): {p_final:.4f}"
            f"<br>raw improvement: {delta:.4f}"
        )
        all_perp.extend(v for v in (p_init, p_final) if v is not None)
    return {"x": x, "y": y, "custom": custom, "hover": hover}, all_perp


def _render_gd_effect(
    population: Population,
    sidebar_data: dict,
    out_dir: Path,
    task_name: str,
) -> Path:
    seeds, progs = split_seeds_and_progs(population)
    L_0 = reference_loss(seeds)
    islands = islands_of(progs)

    traces, perp_a = _gd_traces_per_island(progs, islands, L_0)
    seed, perp_b = _gd_seed_data(seeds, L_0)
    all_perp = perp_a + perp_b

    diag_min = min(all_perp) if all_perp else 0.0
    diag_max = max(all_perp) if all_perp else 2.0

    title = f"{task_name} — GD Effect"
    html = render("gd_effect.html", {
        "__TITLE_HTML__": escape(title),
        "__TITLE_JSON__": json.dumps(title),
        "__SIDEBAR_CSS__": _SIDEBAR_CSS,
        "__SIDEBAR_JS__": _SIDEBAR_JS,
        "__COLOUR_MODE_JS__": _COLOUR_MODE_JS,
        "__SIDEBAR_DATA__": json.dumps(sidebar_data, default=str),
        "__TRACES__": json.dumps(traces),
        "__SEED_X__": json.dumps(seed["x"]),
        "__SEED_Y__": json.dumps(seed["y"]),
        "__SEED_CUSTOM__": json.dumps(seed["custom"]),
        "__SEED_HOVER__": json.dumps(seed["hover"]),
        "__DIAG_RANGE__": json.dumps([diag_min, diag_max]),
    })

    out_path = out_dir / "gd_effect.html"
    out_path.write_text(html)
    return out_path


# ── public API ──

def write_progress(
    population: Population,
    census: list[list[set[int]]],
    out_dir: Path,
    task_name: str = "EDGAR",
) -> tuple[Path, Path]:
    """Write loss_progress.html and gd_effect.html into out_dir."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    alive = compute_alive_set(census)
    sidebar_data = build_sidebar_data(population, alive)
    return (
        _render_loss_progress(population, sidebar_data, out_dir, task_name),
        _render_gd_effect(population, sidebar_data, out_dir, task_name),
    )
