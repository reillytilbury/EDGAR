"""Shared monitoring helpers: data extraction from Population, template rendering.

Public surface:
- build_sidebar_data(population, alive) -> dict[str, dict]
- compute_alive_set(census) -> set[int]
- perplexity(loss, ref_loss) -> float | None
- island_colour(island_idx) -> str
- is_seed(program) -> bool
- render(template_name, substitutions) -> str
- _SIDEBAR_CSS, _SIDEBAR_JS, _COLOUR_MODE_JS  (preloaded static assets)
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from ..evolution.population import Population
from ..evolution.program import Program

_TEMPLATES_DIR = Path(__file__).parent / "templates"

_SIDEBAR_CSS = (_TEMPLATES_DIR / "sidebar.css").read_text()
_SIDEBAR_JS = (_TEMPLATES_DIR / "sidebar.js").read_text()
_COLOUR_MODE_JS = (_TEMPLATES_DIR / "colour_mode.js").read_text()

_ISLAND_COLOURS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
]


def render(template_name: str, substitutions: dict[str, str]) -> str:
    template = (_TEMPLATES_DIR / template_name).read_text()
    for marker, value in substitutions.items():
        template = template.replace(marker, value)
    return template


def island_colour(island_idx: int) -> str:
    return _ISLAND_COLOURS[island_idx % len(_ISLAND_COLOURS)]


def perplexity(loss, ref_loss):
    """exp(-(loss - ref_loss)). None for None inputs, 0.0 on overflow."""
    if loss is None or ref_loss is None:
        return None
    try:
        return math.exp(-(loss - ref_loss))
    except OverflowError:
        return 0.0


def is_seed(program: Program) -> bool:
    return program.birth.generation == -1


def compute_alive_set(census: list[list[set[int]]]) -> set[int]:
    """Programs alive at end of run (union of each island's final-iteration set)."""
    alive: set[int] = set()
    for island in census:
        if island:
            alive |= island[-1]
    return alive


def build_sidebar_data(
    population: Population,
    alive: set[int],
) -> dict[str, dict[str, Any]]:
    """Build the click-to-inspect data dict, keyed by str(program.idx)."""
    sidebar = {}
    for i in range(len(population)):
        p = population[i]
        parents = list(p.birth.parent_indices)
        sidebar[str(p.idx)] = {
            "program_id": p.idx,
            "display_label": p.name or f"P{p.idx}",
            "is_seed": is_seed(p),
            "is_dead": p.idx not in alive,
            "is_extinct": p.idx not in alive,
            "iteration": p.birth.generation,
            "island": p.birth.island,
            "island_label": (
                f"Island {p.birth.island}" if p.birth.island >= 0 else "Seed"
            ),
            "batch": p.birth.batch_index,
            "mode": p.birth.mode,
            "temperature": p.birth.temperature,
            "llm_name": p.birth.llm_name,
            "parent1_id": parents[0] if len(parents) > 0 else None,
            "parent2_id": parents[1] if len(parents) > 1 else None,
            "n_params": p.n_params,
            "initial_loss": p.program_losses.discover.init,
            "train_loss": p.program_losses.discover.final,
            "test_loss": p.program_losses.validate.final,
            "model_code": p.code.model,
            "model_code_jax": p.code_jax.model,
            "param_est_code": p.code.param_est,
        }
    return sidebar


def split_seeds_and_progs(population: Population) -> tuple[list[Program], list[Program]]:
    seeds, progs = [], []
    for i in range(len(population)):
        (seeds if is_seed(population[i]) else progs).append(population[i])
    return seeds, progs


def reference_loss(seeds: list[Program]) -> float:
    """Reference loss for perplexity normalization: min seed train loss."""
    losses = [s.program_losses.discover.final for s in seeds if s.program_losses.discover.final is not None]
    return min(losses) if losses else 0.0


def islands_of(programs: list[Program]) -> list[int]:
    return sorted({p.birth.island for p in programs if p.birth.island >= 0})


def loss_str(loss) -> str:
    return f"{loss:.4f}" if loss is not None and math.isfinite(loss) else "N/A"
