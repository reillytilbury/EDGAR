"""
Generate sample plots (feedback and fit comparison) for a given project.
Useful for verifying that the project's plot_fn handles the data and
program structures correctly.

Usage:
    uv run python scripts/generate_sample_plots.py <project>
    uv run python scripts/generate_sample_plots.py orientation_tuning
"""

import sys
import os
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
_xla_flags = os.environ.get("XLA_FLAGS", "")
if "--xla_gpu_enable_command_buffer=" not in _xla_flags:
    os.environ["XLA_FLAGS"] = (_xla_flags + " --xla_gpu_enable_command_buffer=").strip()

from edgar.io.config import Config
from edgar.io.task_spec import TaskSpec
from edgar.evolution.program import Program, BirthCertificate
from edgar.evolution.population import Population
from edgar.scoring.scoring import score
from edgar.scoring.utils import _safe_loss
from edgar.io.plotting import generate_feedback_image, generate_program_fits, generate_trajectory_image

repo_root = Path(__file__).parent.parent

def generate_samples(project: str):
    print(f"--- Generating sample plots for project: {project} ---")

    # 1. Load project spec
    config_path = repo_root / "projects" / project / "config.yaml"
    if not config_path.exists():
        print(f"Error: Config not found at {config_path}")
        return

    config = Config.from_yaml(config_path)

    # Set a custom save path so output_dir is predictable
    sample_dir = repo_root / "sample_plots" / project
    config.io.save_path = str(sample_dir)

    spec = TaskSpec.from_config(config)
    output_dir = spec.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # 2. Load real data
    print("Loading project data...")
    X_discover, _, X_eval = spec.load_data_fn(
        data_path=spec.io["data_path"], **spec.project_params
    )
    # feedback_image and program_fits usually take the 'test' part of the discovery split
    data = X_discover[1]

    # 3. Score seed programs
    if not spec.seed_programs:
        print("Error: No seed programs found in project to use as samples.")
        return

    population = Population()
    for seed_p in spec.seed_programs:
        if not seed_p.code.model_jax:
            seed_p.code.model_jax = (
                seed_p.code.model
                .replace("import numpy as np", "import jax.numpy as jnp")
                .replace("np.", "jnp.")
            )
        population.add(seed_p)

    print("Scoring seed programs...")
    score(population, X_discover, X_eval, spec.scoring, spec.loss_fn, split="discover")

    programs = [population[i] for i in range(len(population))]

    # 4. Generate Feedback Image (using first 2 as parents)
    print("Generating feedback image...")
    def _loss(p: Program) -> float:
            return _safe_loss(p.program_losses.discover.final)

    parents = sorted(programs, key=_loss, reverse=True)
    current = Program(
        birth=BirthCertificate(generation=1, island=0, batch_index=0),
        name="Candidate Model",
    )
    current.idx = 99
    generate_feedback_image(spec, data, parents, current)
    if current.image_path:
        print(f"  [OK] Feedback image saved to: {current.image_path}")
    else:
        print("  [FAIL] Feedback image generation failed (check plot_fn).")

    # 5. Generate Program Fits
    print("Generating program fit comparisons...")
    generate_program_fits(spec, data, programs)

    fit_count = sum(1 for p in programs if p.fit_image_path)
    print(
        f"  [OK] {fit_count}/{len(programs)} fit images generated in {output_dir}/image_fits/"
    )

    # 6. Generate trajectories plots
    generate_trajectory_image(spec, programs)
    fit_count = sum(1 for p in programs if p.trajectory_image_path)
    print(
        f"  [OK] {fit_count}/{len(programs)} trajectory images generated in {output_dir}/image_trajectories/"
    )


    print(f"\nDone. View plots in: {output_dir}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <project>")
        sys.exit(1)

    generate_samples(sys.argv[1])
