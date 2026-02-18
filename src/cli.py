import argparse
import ast
from pathlib import Path
from textwrap import dedent
import sys


REQUIRED_SPEC_FUNCTIONS = [
    "load_and_process_data",
    "train_test_split",
    "model_v1",
    "param_est_v1",
    "model_v2",
    "param_est_v2",
]

FORBIDDEN_CONFIG_KEYS = {
    "load_and_process_data_fn",
    "create_train_test_sample_split_fn",
    "create_train_test_trial_split_fn",
    "seed_programs",
    "use_image_feedback",
}


SPEC_TEMPLATE = dedent(
    '''\
    """
    Welcome to the Model Discovery Engine! Fill in the components below to start building your model.

    NECESSARY COMPONENTS:

    Loading:
    - load_and_process_data(data_path, *preprocess_params) -> [X, Y]
    - train_test_split(X) -> [train_samples, train_trials]

    Seed Programs:
    - model_v1(X, *params) and param_est_v1(X, Y)
    - model_v2(X, *params) and param_est_v2(X, Y)

    OPTIONAL COMPONENTS:
    - plot_model_fits(X, Y, model_list, params_list)
    """
    import numpy as np
    from typing import Tuple
    from src.data_structures import Inputs, Outputs


    # ========================
    # 1. DATA
    # ========================

    def load_and_process_data(
        data_path: str,
        # ---- ALL SUBSEQUENT PARAMS MUST BE SPECIFIED IN THE CONFIG FILE ----
        random_seed: int = 42,
    ) -> Tuple[Inputs, Outputs]:
        """
        Load and preprocess data and return canonical Inputs/Outputs.
        """
        raise NotImplementedError


    def train_test_split(
        X: Inputs,
        # -- ALL SUBSEQUENT PARAMS MUST BE SPECIFIED IN THE CONFIG FILE ---
        random_seed: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return train sample indices and train trial indices.
        """
        raise NotImplementedError


    # ========================
    # 2. SEED MODELS
    # ========================

    def model_v1(X):
        raise NotImplementedError


    def param_est_v1(X, Y):
        raise NotImplementedError


    def model_v2(X):
        raise NotImplementedError


    def param_est_v2(X, Y):
        raise NotImplementedError


    # ========================
    # 3. DIAGNOSTICS
    # ========================

    def plot_model_fits(X, Y, programs_list, save_path=""):
        raise NotImplementedError


    # ========================
    # 4. OPTIONAL PROJECT-SPECIFIC HELPERS
    # ========================
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


def init_experiment(task: str, force: bool = False) -> int:
    task_path = _task_dir(task)
    task_path.mkdir(parents=True, exist_ok=True)

    spec_path = task_path / "spec.py"
    config_path = task_path / "config.yaml"

    if spec_path.exists() and not force:
        print(f"Error: {spec_path} already exists. Use --force to overwrite.")
        return 1
    if config_path.exists() and not force:
        print(f"Error: {config_path} already exists. Use --force to overwrite.")
        return 1

    spec_path.write_text(SPEC_TEMPLATE, encoding="utf-8")

    config_text = dedent(
        f"""\
        task: {task}

        data_processing_params:
          data_path: /path/to/data.npy

        inputs:
          - name: x
            description: "Input variable"

        experiment_params:
          use_param_estimator: true
        """
    )
    config_path.write_text(config_text, encoding="utf-8")

    print(f"Created {spec_path}")
    print(f"Created {config_path}")
    print(f"Next: fill in required functions in {spec_path}")
    return 0


def validate_experiment(task: str) -> int:
    task_path = _task_dir(task)
    spec_path = task_path / "spec.py"
    config_path = task_path / "config.yaml"
    errors = []

    if not task_path.exists():
        errors.append(f"Task directory not found: {task_path}")
    if not spec_path.exists():
        errors.append(f"Missing spec file: {spec_path}")
    if not config_path.exists():
        errors.append(f"Missing config file: {config_path}")

    if errors:
        print("Validation failed:")
        for err in errors:
            print(f"- {err}")
        return 1

    # Validate required functions in spec.py (syntax + contract)
    try:
        tree = ast.parse(spec_path.read_text(encoding="utf-8"), filename=str(spec_path))
    except SyntaxError as e:
        print("Validation failed:")
        print(f"- spec.py has syntax error: {e}")
        return 1

    func_names = {node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}
    missing = [name for name in REQUIRED_SPEC_FUNCTIONS if name not in func_names]
    if missing:
        print("Validation failed:")
        print("- spec.py is missing required functions:")
        for name in missing:
            print(f"  - {name}")
        return 1

    # Validate config task name
    task_in_config = None
    forbidden_found = []
    for line in config_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if ":" in stripped and not stripped.startswith("#"):
            key = stripped.split(":", 1)[0].strip()
            if key in FORBIDDEN_CONFIG_KEYS:
                forbidden_found.append(key)
        if stripped.startswith("task:"):
            value = stripped.split(":", 1)[1].strip()
            if "#" in value:
                value = value.split("#", 1)[0].strip()
            task_in_config = value
    if forbidden_found:
        print("Validation failed:")
        print("- config.yaml contains deprecated wiring keys. Remove these and use fixed spec naming:")
        for key in sorted(set(forbidden_found)):
            print(f"  - {key}")
        return 1

    if task_in_config != task:
        print("Validation failed:")
        print(f"- config.yaml task mismatch: expected '{task}', found '{task_in_config}'")
        return 1

    print(f"Validation passed for task '{task}'.")
    print(f"- spec: {spec_path}")
    print(f"- config: {config_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="EDGAR project scaffold and validation CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p_init = sub.add_parser("init-project", help="Create a new task scaffold with spec.py + config.yaml")
    p_init.add_argument("task", type=str, help="Task name (folder under projects/ or experiments/)")
    p_init.add_argument("--force", action="store_true", help="Overwrite spec.py/config.yaml if they exist")

    p_validate = sub.add_parser("validate", help="Validate task contract in spec.py + config.yaml")
    p_validate.add_argument("task", type=str, help="Task name (folder under projects/ or experiments/)")
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "init-project":
        return init_experiment(args.task, force=args.force)
    if args.command == "validate":
        return validate_experiment(args.task)
    parser.error("Unknown command")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
