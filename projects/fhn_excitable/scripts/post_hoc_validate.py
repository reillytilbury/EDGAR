"""Post-hoc validation for a stopped/finished run.

Reuses edgar's real ``score`` function on the validate split, so the numbers
match what a completed ``edgar run`` would have produced. Writes results back
to population.jsonl so the dashboard picks them up on refresh.

Usage:
    python projects/fhn_excitable/scripts/post_hoc_validate.py <run_dir>
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# JAX runtime guards, same as edgar.run
import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")

from edgar.io.task_spec import TaskSpec   # noqa: E402
from edgar.io.config import Config   # noqa: E402
from edgar.evolution.population import Population   # noqa: E402
from edgar.evolution.island import load_island_census   # noqa: E402
from edgar.scoring.scoring import score, rank   # noqa: E402


def main(run_dir: Path) -> int:
    task_spec_path = run_dir / "task_spec.yaml"
    if not task_spec_path.exists():
        raise SystemExit(f"task_spec.yaml not found at {task_spec_path}")

    print(f"[post_hoc_validate] loading Config from {task_spec_path}")
    cfg = Config.from_taskspec(task_spec_path)
    spec = TaskSpec.from_config(cfg)

    pop_path = run_dir / "population.jsonl"
    print(f"[post_hoc_validate] loading Population from {pop_path}")
    population = Population.load(str(pop_path))
    print(f"[post_hoc_validate] {len(population)} programs loaded")

    census_path = run_dir / "island_census.jsonl"
    if census_path.exists():
        census = load_island_census(str(census_path))
        if census:
            # census entries are lists of islands (each island a list of program idxs).
            last = census[-1]
            islands = [set(island) for island in last]
            alive = set().union(*islands)
            print(f"[post_hoc_validate] loaded {len(islands)} islands from census "
                  f"(most recent snapshot); {len(alive)} alive programs")
        else:
            islands = [set(range(len(population)))]
            print("[post_hoc_validate] empty census — treating all programs as alive")
    else:
        islands = [set(range(len(population)))]
        print("[post_hoc_validate] no census — treating all programs as alive")

    # Load validate data — task_spec exposes the loader via load_data_fn.
    data_path = cfg.io.data_path or ""
    project_params = dict(cfg.project_params or {})
    (_, _), (Xv_tr, Xv_te), _ = spec.load_data_fn(data_path, **project_params)
    print(f"[post_hoc_validate] X_validate loaded: train {Xv_tr['y'].shape}, "
          f"test {Xv_te['y'].shape}")

    # Mark alive programs eligible for validation
    population.prepare_validation_scoring(islands)

    n_to_validate = sum(
        1 for p in population if p.program_losses.validate.final is None
    )
    print(f"[post_hoc_validate] {n_to_validate} programs to validate")

    # Reuse edgar's real scoring path (no re-fit — score() with existing params
    # only computes losses using the trained params).
    score(
        population, (Xv_tr, Xv_te), None,
        config=spec.scoring, loss_fn=spec.loss_fn, split="validate",
        apply_model_fn=spec.apply_model_fn,
        rollout_fn=spec.rollout_fn,
    )

    rank(population)

    # Save back so the dashboard sees updated validate.final + ranks.
    population.save(str(pop_path))
    print(f"[post_hoc_validate] wrote updated population back to {pop_path}")

    # Quick summary of top-10 by validate.final
    validated = [
        (p.program_losses.validate.final, p.idx, p.name)
        for p in population
        if isinstance(p.program_losses.validate.final, (int, float))
        and p.program_losses.validate.final == p.program_losses.validate.final
    ]
    validated.sort()
    print()
    print("=== Top-10 by validate.final ===")
    for rank_i, (vf, idx, name) in enumerate(validated[:10], start=1):
        print(f"  rank {rank_i}: idx={idx:<3} val_final={vf:+.4f}  {name[:60]}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {sys.argv[0]} <run_dir>")
    main(Path(sys.argv[1]).resolve())
