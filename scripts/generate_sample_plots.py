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
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from edgar.io.config import Config
from edgar.io.task_spec import TaskSpec
from edgar.evolution.program import Program, BirthCertificate, Code, Losses, LossPair
from edgar.io.plotting import generate_feedback_image, generate_program_fits


def generate_samples(project: str):
    print(f"--- Generating sample plots for project: {project} ---")

    # 1. Load project spec
    config_path = Path(f"projects/{project}/config.yaml")
    if not config_path.exists():
        print(f"Error: Config not found at {config_path}")
        return

    config = Config.from_yaml(config_path)
    
    # Set a custom save path so output_dir is predictable
    sample_dir = Path("sample_plots") / project
    config.io.save_path = str(sample_dir)
    
    spec = TaskSpec.from_config(config)
    output_dir = spec.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # 2. Load real data
    print("Loading project data...")
    X_discover, _, _ = spec.load_data_fn(
        data_path=spec.io["data_path"], **spec.project_params
    )
    # feedback_image and program_fits usually take the 'test' part of the discovery split
    data = X_discover[1]
    n_samples = next(iter(data.values())).shape[0]

    # 3. Use seed programs as basis for samples
    if not spec.seed_programs:
        print("Error: No seed programs found in project to use as samples.")
        return

    programs = []
    for i, seed_p in enumerate(spec.seed_programs):
        # Create a copy with necessary fields for plotting
        p = Program(
            birth=seed_p.birth,
            name=seed_p.name,
            code=seed_p.code,
        )
        p.idx = i
        
        # Ensure model_jax is set for compile_model()
        if not p.code.model_jax:
            p.code.model_jax = p.code.model

        # Initialize with dummy parameters and losses to demonstrate plotting
        model_fn = p.compile_model()
        default_params = getattr(model_fn, "DEFAULT_PARAMS", {})

        # Mock initial and final parameters (per-sample)
        p.params_init = {
            k: np.full(n_samples, v) if isinstance(v, (int, float)) else np.repeat(np.asarray(v)[np.newaxis, ...], n_samples, axis=0)
            for k, v in default_params.items()
        }
        p.params = {
            k: v * 2 for k, v in p.params_init.items()
        }

        # Mock losses
        p.sample_losses_init = np.random.uniform(0.5, 1.0, size=n_samples)
        p.sample_losses = p.sample_losses_init * 0.8
        p.program_losses = Losses(
            discover=LossPair(init=float(np.mean(p.sample_losses_init)), final=float(np.mean(p.sample_losses)))
        )

        programs.append(p)

    # 4. Generate Feedback Image (using first 2 as parents)
    print("Generating feedback image...")
    parents = programs[:2]
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
    print(f"  [OK] {fit_count}/{len(programs)} fit images generated in {output_dir}/image_fits/")

    print(f"\nDone. View plots in: {output_dir}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <project>")
        sys.exit(1)

    generate_samples(sys.argv[1])
