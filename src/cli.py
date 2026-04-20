"""
CLI for EDGAR project scaffolding and validation.

Example usage:

    # Create a new project scaffold
    edgar init-project my_task

    This creates:
    - projects/my_task/config.yaml (configuration file)
    - projects/my_task/seed_programs/model1.py (first seed model)
    - projects/my_task/seed_programs/model2.py (second seed model)
    - projects/my_task/seed_programs/param_est1.py (first parameter estimator)
    - projects/my_task/seed_programs/param_est2.py (second parameter estimator)
    - projects/my_task/data_loader/load_data.py (data loading functions)
    - projects/my_task/image_feedback/plot.py (plotting function)

    Each file contains stub functions with docstrings. Fill in the implementations.

    # Validate an existing project
    edgar validate my_task

    Checks that all required files exist and config.yaml has the correct task name.

    # Overwrite existing project (use with caution)
    edgar init-project my_task --force

    # Enable verbose logging (DEBUG level)
    edgar --verbose init-project my_task
    edgar --verbose validate my_task
"""

import argparse
import ast
from pathlib import Path
from textwrap import dedent
import sys

SPEC_TEMPLATE_DATA_LOADER = dedent(
    '''\
    import numpy as np
    from typing import Dict, Tuple

    def load_and_process_data(
        data_path: str,
        **kwargs  # Additional params specified in config.yaml
    ) -> Dict[str, np.ndarray]:
        """
        Load and preprocess data. Return a dict of arrays.
        All values must share the same last dimension (n_trials).

        Example:
            return {
                'stimulus': np.array(...),  # shape (n_samples, ..., n_trials)
                'response': np.array(...),  # shape (n_samples, ..., n_trials)
            }
        """
        raise NotImplementedError

    def train_test_split(
        data: Dict[str, np.ndarray],
        **kwargs  # Additional params specified in config.yaml
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return (train_sample_indices, train_trial_indices).
        These define which samples/trials form the training set;
        the remainder form the test set.
        """
        raise NotImplementedError

    def loss_fn(y_pred: np.ndarray, y_true: np.ndarray, params=None) -> float:
        """
        Compute loss. y_true is typically data['response'].
        Return scalar loss (per-sample losses will be aggregated).
        """
        raise NotImplementedError

    def build_evaluation_points(
        data: Dict[str, np.ndarray],
        **kwargs
    ) -> Dict[str, np.ndarray]:
        """
        Optional. Return a dict of arrays for model evaluation/plotting.
        Default: return data unchanged.
        """
        return data
    '''
)

SPEC_TEMPLATE_MODEL = dedent(
    '''\
    import numpy as np

    def model(data: dict, params: dict) -> np.ndarray:
        """
        Model function. Receives data dict and params dict.
        data: dict of named arrays, e.g. data['stimulus'], data['response']
        params: dict of named parameters matching DEFAULT_PARAMS keys
        Returns: model predictions (same shape as data['response'] or similar)
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

    def parameter_estimator(data: dict) -> dict:
        """
        Parameter estimator. Receives single-sample data (no sample axis).
        Returns: dict matching model.DEFAULT_PARAMS keys
        """
        raise NotImplementedError
    '''
)

SPEC_TEMPLATE_PLOT = dedent(
    '''\
    import numpy as np

    def plot_model_fits(data, programs_list, X_eval, save_path="", labels=None):
        """
        Optional. Plot model predictions vs data.
        data: original data dict
        programs_list: list of Program objects with compiled model/estimator
        X_eval: evaluation points dict (from build_evaluation_points)
        save_path: directory to save plots
        labels: optional display labels for programs
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


def init_project(task: str, force: bool = False) -> int:
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

    # Check if files exist
    files_to_create = [model1_path, model2_path, param_est1_path, param_est2_path, load_data_path, plot_path, config_path]
    existing = [f for f in files_to_create if f.exists()]
    if existing and not force:
        print(f"Error: Files already exist. Use --force to overwrite:")
        for f in existing:
            print(f"  - {f}")
        return 1

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
        task:
          name: {task}
          data_path: /path/to/data.npy

        evolution:
          n_iterations: 12
          n_islands: 8
          batch_size: 6
          critical_population_size: 12
          n_migrants: 2

        llms:
          k_max: 2
          model_llm: gemini-2.5-flash
          param_est_llm: gemini-2.0-flash
          jax_translator_llm: gemini-2.0-flash-lite

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
    task_path = _task_dir(task)
    config_path = task_path / "config.yaml"
    required_files = [
        task_path / "seed_programs" / "model1.py",
        task_path / "seed_programs" / "model2.py",
        task_path / "seed_programs" / "param_est1.py",
        task_path / "seed_programs" / "param_est2.py",
        task_path / "data_loader" / "load_data.py",
        task_path / "image_feedback" / "plot.py",
        config_path,
    ]

    errors = []
    if not task_path.exists():
        errors.append(f"Task directory not found: {task_path}")

    for f in required_files:
        if not f.exists():
            errors.append(f"Missing: {f}")

    if errors:
        print("Validation failed:")
        for err in errors:
            print(f"- {err}")
        return 1

    # Validate config has task name
    config_text = config_path.read_text(encoding="utf-8")
    if f"name: {task}" not in config_text:
        print("Validation failed:")
        print(f"- config.yaml must have 'name: {task}' under 'task:'")
        return 1

    print(f"Validation passed for project '{task}'.")
    print(f"  ✓ seed_programs/model1.py, model2.py")
    print(f"  ✓ seed_programs/param_est1.py, param_est2.py")
    print(f"  ✓ data_loader/load_data.py")
    print(f"  ✓ image_feedback/plot.py")
    print(f"  ✓ config.yaml")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="EDGAR project scaffold and validation CLI")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging (DEBUG level instead of INFO)")
    sub = parser.add_subparsers(dest="command", required=True)

    p_init = sub.add_parser("init-project", help="Create a new project with seed programs, data loader, and config")
    p_init.add_argument("task", type=str, help="Project name (folder under projects/)")
    p_init.add_argument("--force", action="store_true", help="Overwrite files if they exist")

    p_validate = sub.add_parser("validate", help="Validate project structure and required files")
    p_validate.add_argument("task", type=str, help="Project name (folder under projects/)")
    return parser


def run_cli(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "init-project":
        return init_project(args.task, force=args.force)
    if args.command == "validate":
        return validate_project(args.task)
    parser.error("Unknown command")
    return 2


if __name__ == "__main__":
    raise SystemExit(run_cli())
