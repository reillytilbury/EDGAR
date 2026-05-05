"""
CLI for EDGAR project scaffolding, validation, and running experiments.

Commands
--------

init-project
    Create a new project scaffold:

        edgar init-project my_task

    Creates:
        projects/my_task/config.yaml
        projects/my_task/seed_programs/model1.py, model2.py
        projects/my_task/seed_programs/param_est1.py, param_est2.py
        projects/my_task/data_loader/load_data.py
        projects/my_task/image_feedback/plot.py

    Each file contains stub functions with docstrings. Fill in the implementations.
    Existing files are overwritten.

validate
    Check that all required files and functions exist for a project:

        edgar validate my_task

run
    Run an experiment from a config.yaml or a task_spec.yaml saved from a previous run:

        edgar run projects/my_task/config.yaml
        edgar run runs/05-01/14-32-10/task_spec.yaml

    Control logging verbosity (default: compact):

        edgar run projects/my_task/config.yaml --log-level code
        edgar run projects/my_task/config.yaml --log-level prompts

    Override config values at the command line using --section.key=value:

        edgar run projects/my_task/config.yaml --evolution.n_generations=20
        edgar run projects/my_task/config.yaml --io.data_path=/data/new.npy --llms.model_llm=gemini-2.5-pro

    Valid sections: io, evolution, llms, scoring, project_params.
    Values are parsed as Python literals (int, float, bool) where possible.
"""

import argparse
from pathlib import Path
from textwrap import dedent

SPEC_TEMPLATE_DATA_LOADER = dedent(
    '''\
    from __future__ import annotations

    import numpy as np
    import jax.numpy as jnp


    def _to_jax(d):
        return {k: jnp.array(v) if k != "_sample_indices" else v for k, v in d.items()}


    def load_data(
        data_path: str,
        n_eval_samples: int = 10,
        **kwargs,  # Additional params from project_params in config.yaml
    ):
        """
        Load and preprocess data, then split into discover / validate / eval sets.

        Returns
        -------
        (X_discover, X_validate, X_eval)

        X_discover = (X_disc_train, X_disc_test)
            X_disc_train: dict of JAX arrays, shape (n_samples//2, n_trials//2) — seen by the LLM loop.
            X_disc_test:  dict of JAX arrays, shape (n_samples//2, n_trials//2) — held-out test within discovery.

        X_validate = (X_val_train, X_val_test)
            X_val_train: dict of JAX arrays, shape (n_samples//2, n_trials//2) — never seen during discovery.
            X_val_test:  dict of JAX arrays, shape (n_samples//2, n_trials//2) — final held-out evaluation.

        X_eval
            Small fingerprint subset from discover train, used for deduplication.
            Dict of JAX arrays plus '_sample_indices' (numpy int array of positions within disc_idx).
        """
        raise NotImplementedError


    def loss_fn(model_output, data):
        """
        Compute per-sample loss between model output and data.

        Args:
            model_output: JAX array of model predictions, shape (n_trials,) or (n_samples, n_trials).
            data: dict of JAX arrays for this split, e.g. data['response'].

        Returns:
            JAX array of per-sample losses, shape (n_samples,).
        """
        raise NotImplementedError
    '''
)

SPEC_TEMPLATE_MODEL = dedent(
    '''\
    import numpy as np


    def model(data, params):
        """
        Model function. Called once per sample.

        Args:
            data (dict): Data dict for one sample, e.g. data['stimulus'] shape (n_trials,).
            params (dict): Parameter dict with keys matching DEFAULT_PARAMS.

        Returns:
            np.ndarray: Predictions, shape (n_trials,).
        """
        raise NotImplementedError


    model.DEFAULT_PARAMS = {
        # "param_name": initial_value,
    }
    '''
)

SPEC_TEMPLATE_PARAM_EST = dedent(
    '''\
    import numpy as np


    def parameter_estimator(data):
        """
        Estimate model parameters for a single sample.

        Args:
            data (dict): Data dict for one sample, e.g. data['stimulus'] and data['response'],
                         each shape (n_trials,).

        Returns:
            dict: Estimated parameters with keys matching model.DEFAULT_PARAMS.
        """
        raise NotImplementedError
    '''
)

SPEC_TEMPLATE_PLOT = dedent(
    '''\
    import numpy as np
    import jax.numpy as jnp
    import matplotlib.pyplot as plt


    def plot_model_fits(data, parent_programs, save_path=""):
        """
        Optional. Plot model predictions vs data.

        Args:
            data: X_disc_train dict of JAX arrays, shape (n_samples, n_trials).
            parent_programs: list of Program objects. Each has:
                - .compile() -> (model_fn, param_est_fn)
                - .params: dict of per-sample params, each value shape (n_samples, ...)
                - .sample_losses: per-sample losses, shape (n_samples,), or None
                - .program_losses.discover.final: scalar overall loss
            save_path: file path (not directory) to save the figure.
        """
        pass
    '''
)


def _find_project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _find_collection_dir(project_root: Path) -> Path:
    projects_dir = project_root / "projects"
    experiments_dir = project_root / "experiments"
    if projects_dir.exists():
        return projects_dir
    if experiments_dir.exists():
        return experiments_dir
    return projects_dir


def _task_dir(task: str) -> Path:
    root = _find_project_root()
    collection = _find_collection_dir(root)
    return collection / task


def init_project(task: str) -> int:
    task_path = _task_dir(task)
    task_path.mkdir(parents=True, exist_ok=True)

    # Create subdirectories
    seed_programs_dir = task_path / "seed_programs"
    data_loader_dir = task_path / "data_loader"
    image_feedback_dir = task_path / "image_feedback"

    seed_programs_dir.mkdir(exist_ok=True)
    data_loader_dir.mkdir(exist_ok=True)
    image_feedback_dir.mkdir(exist_ok=True)

    # Seed program files
    model1_path = seed_programs_dir / "model1.py"
    model2_path = seed_programs_dir / "model2.py"
    param_est1_path = seed_programs_dir / "param_est1.py"
    param_est2_path = seed_programs_dir / "param_est2.py"

    # Data loader files
    load_data_path = data_loader_dir / "load_data.py"

    # Image feedback files
    plot_path = image_feedback_dir / "plot.py"

    # Config file
    config_path = task_path / "config.yaml"

    # Write seed program files
    model1_path.write_text(SPEC_TEMPLATE_MODEL, encoding="utf-8")
    model2_path.write_text(SPEC_TEMPLATE_MODEL, encoding="utf-8")
    param_est1_path.write_text(SPEC_TEMPLATE_PARAM_EST, encoding="utf-8")
    param_est2_path.write_text(SPEC_TEMPLATE_PARAM_EST, encoding="utf-8")

    # Write data loader file
    load_data_path.write_text(SPEC_TEMPLATE_DATA_LOADER, encoding="utf-8")

    # Write image feedback file
    plot_path.write_text(SPEC_TEMPLATE_PLOT, encoding="utf-8")

    # Write config
    config_text = dedent(
        f"""\
        io:
          data_path: /path/to/data.npy
          save_path: program_databases

        evolution:
          n_generations: 12
          n_islands: 8
          batch_size: 6
          critical_population_size: 12
          n_migrants: 2

        llms:
          k_max: 2
          model_llm: gemini-2.5-flash
          param_est_llm: gemini-2.5-flash
          jax_translator_llm: gemini-2.5-flash-lite

        scoring:
          param_penalty_weight: 0.01
        """
    )
    config_path.write_text(config_text, encoding="utf-8")

    print(f"Created project structure for '{task}':")
    print(f"  seed_programs/: model1.py, model2.py, param_est1.py, param_est2.py")
    print(f"  data_loader/: load_data.py")
    print(f"  image_feedback/: plot.py")
    print(f"  config.yaml")
    print(f"\nNext: fill in the functions in each file")
    return 0


def validate_project(task: str) -> int:
    from .llm.code_loading import load_function_from_source

    task_path = _task_dir(task)
    if not task_path.exists():
        print(f"Validation failed:\n- Task directory not found: {task_path}")
        return 1

    required_files = [
        task_path / "seed_programs" / "model1.py",
        task_path / "seed_programs" / "model2.py",
        task_path / "seed_programs" / "param_est1.py",
        task_path / "seed_programs" / "param_est2.py",
        task_path / "data_loader" / "load_data.py",
        task_path / "image_feedback" / "plot.py",
        task_path / "config.yaml",
    ]

    required_fns = [
        (task_path / "seed_programs" / "model1.py",      "model"),
        (task_path / "seed_programs" / "model2.py",      "model"),
        (task_path / "seed_programs" / "param_est1.py",  "parameter_estimator"),
        (task_path / "seed_programs" / "param_est2.py",  "parameter_estimator"),
        (task_path / "data_loader"   / "load_data.py",   "load_data"),
        (task_path / "data_loader"   / "load_data.py",   "loss_fn"),
        (task_path / "image_feedback"/ "plot.py",        "plot_model_fits"),
    ]

    errors = [f"Missing file: {f}" for f in required_files if not f.exists()]

    for path, fn_name in required_fns:
        if not path.exists():
            continue  # already reported as missing
        if load_function_from_source(path.read_text(), fn_name) is None:
            errors.append(f"Missing function '{fn_name}' in {path.relative_to(task_path)}")

    if errors:
        print("Validation failed:")
        for err in errors:
            print(f"  - {err}")
        return 1

    print(f"Validation passed for project '{task}'.")
    print(f"  ✓ seed_programs/model1.py, model2.py  (model)")
    print(f"  ✓ seed_programs/param_est1.py, param_est2.py  (parameter_estimator)")
    print(f"  ✓ data_loader/load_data.py  (load_data, loss_fn)")
    print(f"  ✓ image_feedback/plot.py  (plot_model_fits)")
    print(f"  ✓ config.yaml")
    return 0


def _apply_overrides(spec, overrides: list[str]) -> None:
    """
    Apply dotted key=value overrides to a TaskSpec in-place.

    Each override must be of the form section.key=value, where section is one of
    io, evolution, llms, scoring, project_params. Values are parsed as Python
    literals where possible (int, float, bool), otherwise kept as strings.

    Example:
        _apply_overrides(spec, ["evolution.n_generations=1", "io.data_path=/data/foo.npy"])
    """
    sections = {"io", "evolution", "llms", "scoring", "project_params"}
    for override in overrides:
        if not override.startswith("--"):
            continue
        override = override[2:]
        if "=" not in override or "." not in override:
            raise ValueError(f"Override must be --section.key=value, got: --{override}")
        dotted, value_str = override.split("=", 1)
        section, key = dotted.split(".", 1)
        if section not in sections:
            raise ValueError(f"Unknown section '{section}'. Must be one of {sections}")
        try:
            import ast
            value = ast.literal_eval(value_str)
        except (ValueError, SyntaxError):
            value = value_str
        getattr(spec, section)[key] = value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="EDGAR project scaffold, validation, and run CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p_init = sub.add_parser("init-project", help="Create a new project with seed programs, data loader, and config")
    p_init.add_argument("task", type=str, help="Project name (folder under projects/)")

    p_validate = sub.add_parser("validate", help="Validate project structure and required files")
    p_validate.add_argument("task", type=str, help="Project name (folder under projects/)")

    p_run = sub.add_parser("run", help="Run an EDGAR experiment from a config.yaml or task_spec.yaml")
    p_run.add_argument("config", type=str, help="Path to config.yaml or task_spec.yaml")
    p_run.add_argument(
        "--log-level",
        choices=["compact", "code", "prompts"],
        default="compact",
        help="Logging verbosity: compact (default), code, or prompts",
    )

    return parser


def run_cli(argv=None) -> int:
    parser = build_parser()
    args, overrides = parser.parse_known_args(argv)

    if args.command == "init-project":
        return init_project(args.task)
    if args.command == "validate":
        return validate_project(args.task)
    if args.command == "run":
        import asyncio
        from .io.task_spec import TaskSpec
        from .run import run
        spec = TaskSpec.from_config(Path(args.config))
        if overrides:
            _apply_overrides(spec, overrides)
        asyncio.run(run(spec, log_level=args.log_level))
        return 0
    parser.error("Unknown command")
    return 2


if __name__ == "__main__":
    raise SystemExit(run_cli())
