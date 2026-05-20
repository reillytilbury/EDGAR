"""Family tree visualization: parent-child graph rendered from a Population.

Reads from:
- Population (program.idx, birth.parent_indices, birth.generation, name, losses, ...)
- census (which programs are alive at end of run, for the dead-vs-alive coloring)

Layout: hierarchical, y = generation (negated so older sits on top).
Each generation's programs are placed left-to-right by global idx.
"""
from __future__ import annotations

import json
import math
from html import escape
from pathlib import Path

from ..evolution.population import Population
from .data import (
    build_sidebar_data,
    compute_alive_set,
    is_seed,
    loss_str,
    render,
)


def _layout_by_generation(population: Population) -> dict[int, tuple[float, float]]:
    """Hierarchical layout: y = -generation, x = horizontal slot per generation."""
    by_gen: dict[int, list[int]] = {}
    for i in range(len(population)):
        p = population[i]
        by_gen.setdefault(p.birth.generation, []).append(p.idx)

    pos = {}
    for gen, ids in by_gen.items():
        n = len(ids)
        for i, idx in enumerate(sorted(ids)):
            x = (i - (n - 1) / 2.0) if n > 1 else 0.0
            pos[idx] = (x, -gen)
    return pos


def _node_colour(loss) -> str:
    if loss is None or not math.isfinite(loss):
        return "#cccccc"
    if loss < 1.0:
        return "#28a745"
    if loss < 10.0:
        return "#ffc107"
    return "#dc3545"


def _build_edges(population: Population, pos: dict) -> tuple[list, list]:
    edge_x, edge_y = [], []
    for i in range(len(population)):
        p = population[i]
        for parent_idx in p.birth.parent_indices:
            if p.idx not in pos or parent_idx not in pos:
                continue
            x_c, y_c = pos[p.idx]
            x_p, y_p = pos[parent_idx]
            edge_x.extend([x_p, x_c, None])
            edge_y.extend([y_p, y_c, None])
    return edge_x, edge_y


def _build_nodes(population: Population, pos: dict) -> dict:
    """Per-node display arrays for Plotly."""
    x, y, ids, labels, hover = [], [], [], [], []
    colors, symbols, sizes = [], [], []
    for i in range(len(population)):
        p = population[i]
        if p.idx not in pos:
            continue
        nx, ny = pos[p.idx]
        x.append(nx)
        y.append(ny)
        ids.append(str(p.idx))
        label = p.name or f"P{p.idx}"
        labels.append(label)
        hover_label = f"★ {label}" if p.rank == 1 else label
        hover.append(f"{hover_label}<br>loss: {loss_str(p.program_losses.discover.final)}")
        colors.append(_node_colour(p.program_losses.discover.final))
        if p.rank == 1:
            symbols.append("star")
            sizes.append(24)
        elif is_seed(p):
            symbols.append("square")
            sizes.append(20)
        else:
            symbols.append("circle")
            sizes.append(16)
    return {
        "x": x, "y": y, "ids": ids, "labels": labels, "hover": hover,
        "colors": colors, "symbols": symbols, "sizes": sizes,
    }


def _build_parent_map(population: Population) -> dict[str, list[int]]:
    """{str(child_idx): [parent_idx, ...]} for ancestor highlighting in the JS."""
    parent_map = {}
    for i in range(len(population)):
        p = population[i]
        parents = list(p.birth.parent_indices)
        if parents:
            parent_map[str(p.idx)] = parents
    return parent_map


def write_family_tree(
    population: Population,
    census: list[list[set[int]]],
    out_dir: Path,
    task_name: str = "EDGAR",
    param_penalty_weight: float = 0.0,
) -> Path:
    """Render family_tree.html into out_dir."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    alive = compute_alive_set(census)
    sidebar_data = build_sidebar_data(population, alive, param_penalty_weight)

    pos = _layout_by_generation(population)
    edge_x, edge_y = _build_edges(population, pos)
    nodes = _build_nodes(population, pos)
    parent_map = _build_parent_map(population)
    pos_map = {str(idx): list(xy) for idx, xy in pos.items()}

    title = f"{task_name} — Family Tree"
    html = render("family_tree.html", {
        "__TITLE_HTML__": escape(title),
        "__TITLE_JSON__": json.dumps(title),
        "__SIDEBAR_DATA__": json.dumps(sidebar_data, default=str),
        "__PARENT_MAP__": json.dumps(parent_map),
        "__POS_MAP__": json.dumps(pos_map),
        "__EDGE_X__": json.dumps(edge_x),
        "__EDGE_Y__": json.dumps(edge_y),
        "__NODE_X__": json.dumps(nodes["x"]),
        "__NODE_Y__": json.dumps(nodes["y"]),
        "__NODE_IDS__": json.dumps(nodes["ids"]),
        "__NODE_LABELS__": json.dumps(nodes["labels"]),
        "__NODE_HOVER__": json.dumps(nodes["hover"]),
        "__NODE_COLORS__": json.dumps(nodes["colors"]),
        "__NODE_SYMBOLS__": json.dumps(nodes["symbols"]),
        "__NODE_SIZES__": json.dumps(nodes["sizes"]),
    })

    out_path = out_dir / "family_tree.html"
    out_path.write_text(html)
    return out_path


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m src.monitoring.family_tree <path/to/population.jsonl>")
        sys.exit(1)

    pop_path = Path(sys.argv[1])
    population = Population.load(str(pop_path))
    out_dir = pop_path.parent
    out_path = write_family_tree(population, census=[], out_dir=out_dir)
    print(f"Written: {out_path}")
