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

resume
    Resume a crashed or interrupted run from its output directory. Picks up at
    the next unfinished generation and writes back into the same directory:

        edgar resume program_databases/05-26/14-54-15/

    Control logging verbosity (default: compact):

        edgar run projects/my_task/config.yaml --log-level code
        edgar run projects/my_task/config.yaml --log-level prompts

    Override config values at the command line using --section.key=value:

        edgar run projects/my_task/config.yaml --evolution.n_generations=20
        edgar run projects/my_task/config.yaml --io.data_path=/data/new.npy --llms.model_llm=gemini-2.5-pro

    Valid sections: io, evolution, llms, scoring, project_params.
    Values are parsed as Python literals (int, float, bool) where possible.

test
    Run a test run which overrides project config values to have n_generations=1, n_islands=2, batch_size=2, etc. This is useful for quickly checking that the pipeline runs end-to-end with real LLM calls:
        edgar test projects/my_task/config.yaml

    Output is saved to test_output/

test-fake
    Run a test run with fake LLM responses (no real API calls). This is useful for end-to-end testing of the pipeline without incurring API costs or waiting for LLM responses:
        edgar test-fake

    Output is saved to test_output/
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

        Args:
            data_path (str): The path to the raw data file.
            n_eval_samples (int): Number of samples to use for the evaluation fingerprint.
            **kwargs: Additional parameters passed from `project_params` in `config.yaml`.

        Returns:
            tuple: A tuple containing (X_discover, X_validate, X_eval).

            X_discover (tuple):
                Contains `X_disc_train` and `X_disc_test`.
                `X_disc_train` (dict): Dictionary of JAX arrays, shape (n_samples//2, n_trials//2)
                    — seen by the LLM loop for model discovery.
                `X_disc_test` (dict): Dictionary of JAX arrays, shape (n_samples//2, n_trials//2)
                    — held-out test set used within the discovery phase.
            X_validate (tuple):
                Contains `X_val_train` and `X_val_test`.
                `X_val_train` (dict): Dictionary of JAX arrays, shape (n_samples//2, n_trials//2)
                    — never seen during the discovery phase, used for validation.
                `X_val_test` (dict): Dictionary of JAX arrays, shape (n_samples//2, n_trials//2)
                    — final held-out evaluation set.
            X_eval (dict):
                A small subset of data from `X_disc_train` used for generating model
                fingerprints for deduplication. Contains JAX arrays plus `_sample_indices`
                (numpy int array of positions within `disc_idx`).
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
    """
    Finds the root directory of the EDGAR project.

    Returns:
        Path: The absolute path to the EDGAR project root directory.
    """
    return Path(__file__).resolve().parent.parent


def _find_collection_dir(project_root: Path) -> Path:
    """
    Determines the directory where projects are stored (either 'projects/' or 'experiments/').

    Prioritizes 'projects/' if it exists, otherwise 'experiments/', and defaults to 'projects/'
    if neither exist.

    Args:
        project_root (Path): The root directory of the EDGAR project.

    Returns:
        Path: The path to the project collection directory.
    """
    projects_dir = project_root / "projects"
    experiments_dir = project_root / "experiments"
    if projects_dir.exists():
        return projects_dir
    if experiments_dir.exists():
        return experiments_dir
    return projects_dir


def _task_dir(task: str) -> Path:
    """
    Constructs the absolute path for a given EDGAR task directory.

    Args:
        task (str): The name of the EDGAR task.

    Returns:
        Path: The absolute path to the task's directory within the project collection.
    """
    root = _find_project_root()
    collection = _find_collection_dir(root)
    return collection / task


def init_project(task: str) -> int:
    """
    Initializes a new EDGAR project with a predefined directory structure and template files.

    This command scaffolds a new project under `projects/` (or `experiments/` if that exists)
    by creating directories for seed programs, data loader, and image feedback, along with
    template Python files and a default `config.yaml`. Existing files will be overwritten.

    Args:
        task (str): The name of the project to initialize. This will be the name of the
                    directory created under `projects/` (e.g., `projects/my_task`).

    Returns:
        int: Exit code (0 for success, non-zero for failure).
    """
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
        """\
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
          num_parents: 2
          # default_provider: google (Gemini) or anthropic (Claude). Sets the default model
          # per role; override individual roles below to mix or pick specific models.
          default_provider: google
          # model_llm: gemini-2.5-flash
          # param_est_llm: gemini-2.5-flash
          # jax_model_translator_llm: gemini-2.5-flash-lite

        scoring:
          param_penalty_weight: 0.01
        """
    )
    config_path.write_text(config_text, encoding="utf-8")

    print(f"Created project structure for '{task}':")
    print("  seed_programs/: model1.py, model2.py, param_est1.py, param_est2.py")
    print("  data_loader/: load_data.py")
    print("  image_feedback/: plot.py")
    print("  config.yaml")
    print("\nNext: fill in the functions in each file")
    return 0


def validate_project(task: str) -> int:
    """
    Validates an EDGAR project by checking for the existence of required files and functions.

    Ensures that the project directory exists and contains all necessary files (e.g.,
    `model1.py`, `load_data.py`, `config.yaml`) and that these files define the expected
    functions (`model`, `parameter_estimator`, `load_data`, `loss_fn`, `plot_model_fits`).
    This helps to ensure a correct setup before an experiment is run.

    Args:
        task (str): The name of the project to validate.

    Returns:
        int: Exit code (0 for successful validation, 1 for failure).
    """
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
        (task_path / "seed_programs" / "model1.py", "model"),
        (task_path / "seed_programs" / "model2.py", "model"),
        (task_path / "seed_programs" / "param_est1.py", "parameter_estimator"),
        (task_path / "seed_programs" / "param_est2.py", "parameter_estimator"),
        (task_path / "data_loader" / "load_data.py", "load_data"),
        (task_path / "data_loader" / "load_data.py", "loss_fn"),
        (task_path / "image_feedback" / "plot.py", "plot_model_fits"),
    ]

    errors = [f"Missing file: {f}" for f in required_files if not f.exists()]

    for path, fn_name in required_fns:
        if not path.exists():
            continue  # already reported as missing
        if load_function_from_source(path.read_text(), fn_name) is None:
            errors.append(
                f"Missing function '{fn_name}' in {path.relative_to(task_path)}"
            )

    if errors:
        print("Validation failed:")
        for err in errors:
            print(f"  - {err}")
        return 1

    print(f"Validation passed for project '{task}'.")
    print("  ✓ seed_programs/model1.py, model2.py  (model)")
    print("  ✓ seed_programs/param_est1.py, param_est2.py  (parameter_estimator)")
    print("  ✓ data_loader/load_data.py  (load_data, loss_fn)")
    print("  ✓ image_feedback/plot.py  (plot_model_fits)")
    print("  ✓ config.yaml")
    return 0


def _apply_overrides(spec, overrides: list[str]) -> None:
    """
    Apply dotted key=value overrides to a TaskSpec in-place.

    Each override must be of the form section.key=value, where section is one of
    io, evolution, llms, scoring, project_params, run. Values are parsed as Python
    literals where possible (int, float, bool), otherwise kept as strings.

    Example:
        _apply_overrides(spec, ["evolution.n_generations=1", "io.data_path=/data/foo.npy"])

    Args:
        spec: The TaskSpec object to apply overrides to. This object is modified in-place.
        overrides (list[str]): A list of string overrides in the format "--section.key=value".

    Raises:
        ValueError: If an override is not in the correct format or specifies an unknown section.
    """
    sections = {"io", "evolution", "llms", "scoring", "project_params", "run"}
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


TEST_OVERRIDES = [
    "--io.save_path=./test_output",
    "--evolution.n_generations=1",
    "--evolution.n_islands=2",
    "--evolution.batch_size=2",
    "--evolution.num_parents=2",
    "--evolution.topology=[1, 0]",
    "--scoring.gradient_descent.max_iter=100",
    "--scoring.timeout_s=120",
    "--llms.model_llm=gemini-2.5-flash",
    "--llms.param_est_llm=gemini-2.5-flash",
    "--llms.jax_model_translator_llm=gemini-2.5-flash-lite",
    # "--llms.log_raw_llm_response=True",
]


def _build_and_run(config_path: str, overrides: list[str], log_level: str) -> None:
    """
    Builds a `TaskSpec` from a configuration and runs an EDGAR experiment.

    This function initializes a `Config` object from the provided `config_path`
    (either a `config.yaml` or a saved `task_spec.yaml`), then creates a `TaskSpec`.
    Any command-line `overrides` are applied to the `TaskSpec` before
    the main asynchronous `run` function is invoked.

    Args:
        config_path (str): Path to the `config.yaml` or `task_spec.yaml` file.
        overrides (list[str]): A list of command-line override strings
                                (e.g., `"--evolution.n_generations=20"`).
        log_level (str): The desired logging verbosity ('compact', 'code', or 'prompts').
    """
    import asyncio
    from .io.config import Config
    from .io.task_spec import TaskSpec
    from .run import run

    path = Path(config_path)
    config = (
        Config.from_taskspec(path)
        if path.name == "task_spec.yaml"
        else Config.from_yaml(path)
    )
    spec = TaskSpec.from_config(config)
    if overrides:
        _apply_overrides(spec, overrides)
    asyncio.run(run(spec, log_level=log_level))


def _build_and_resume(run_dir: str, log_level: str) -> int:
    """Resume a crashed/interrupted run from its output directory.

    Rebuilds the TaskSpec from the run's frozen task_spec.yaml and continues
    the evolution loop at the next unfinished generation. Output is written
    back into the same directory.
    """
    import asyncio
    from .io.config import Config
    from .io.task_spec import TaskSpec
    from .io.status import read_status
    from .run import run

    run_path = Path(run_dir).expanduser().resolve()
    task_spec_path = run_path / "task_spec.yaml"
    if not task_spec_path.exists():
        print(f"error: not a run directory (no task_spec.yaml): {run_path}")
        return 1

    status = read_status(run_path) or {}
    state = status.get("state")
    if state == "complete":
        print(f"error: run is already complete (status.state={state!r}): {run_path}")
        return 1
    if state not in (None, "starting", "running", "failed"):
        print(f"error: unrecognised status.state={state!r}; refusing to resume.")
        return 1

    config = Config.from_taskspec(task_spec_path)
    spec = TaskSpec.from_config(config)
    print(f"Resuming run at: {run_path}")
    print(
        f"  previous status: state={state!r} current_gen={status.get('current_gen')!r}"
    )
    asyncio.run(run(spec, log_level=log_level, resume_from=run_path))
    return 0


def _run_test_fake() -> None:
    """
    Runs a scaled-down EDGAR experiment with mocked LLM responses.

    This function is used for end-to-end testing of the EDGAR pipeline without
    making actual API calls to Large Language Models. It directly invokes the
    `run_test_fake` function from the system tests.
    """
    import sys

    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from tests.system.fake_runner import run_test_fake

    run_test_fake()


def _run_dashboard(target: str | None, port: int, host: str, no_open: bool) -> int:
    """Start the dashboard server, optionally opening a browser window.

    target may be:
        - None: scan ./program_databases for runs and let the user pick.
        - A path to a single run directory (contains task_spec.yaml).
        - A path to program_databases/ itself.

    Args:
        target (str | None): The target for the dashboard. Can be a path to a
                             specific run directory, a `program_databases/` root,
                             or None to scan the default location.
        port (int): The starting port number to try for the dashboard server.
        host (str): The host address for the dashboard server.
        no_open (bool): If True, prevents the automatic opening of a browser window.

    Returns:
        int: Exit code (0 for success, 1 for failure).
    """
    import webbrowser
    import uvicorn

    project_root = _find_project_root()
    pdb_default = project_root / "program_databases"

    roots: list[Path] = []
    default_run_dir: Path | None = None

    if target:
        target_path = Path(target).expanduser().resolve()
        if not target_path.exists():
            print(f"error: path does not exist: {target_path}")
            return 1
        if (target_path / "task_spec.yaml").exists():
            default_run_dir = target_path
            roots.append(target_path.parent.parent)  # program_databases/
            roots.append(target_path)  # also accept the run dir directly
        else:
            roots.append(target_path)
    else:
        roots.append(pdb_default)

    # always include the canonical program_databases/
    if pdb_default.exists() and pdb_default not in roots:
        roots.append(pdb_default)

    import socket

    def _find_free_port(start: int) -> int:
        for p in range(start, start + 10):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                if s.connect_ex((host, p)) != 0:
                    return p
        raise RuntimeError(f"No free port found in range {start}–{start + 10}")

    port = _find_free_port(port)

    from .dashboard.server import build_app

    app = build_app(roots, default_run_dir=default_run_dir)

    url = f"http://{host}:{port}/"
    if default_run_dir is not None:
        from .dashboard import data as dd

        url += f"#/inspect?run={dd._run_id(default_run_dir)}"
    print(f"EDGAR dashboard running at  {url}")
    print(f"  roots: {[str(r) for r in roots]}")
    try:
        import pydantic_ai  # noqa: F401
    except ModuleNotFoundError:
        import sys

        print(
            f"  warning: 'pydantic_ai' is not installed in {sys.executable!r}; "
            "the LaTeX tab will return 503. This is likely due to running the "
            "dashboard from the wrong environment. Activate the 'edgar' conda env, "
            "`pip install -e .` from the repo root, or use the prefix `uv run` and restart."
        )
    if not no_open:
        try:
            webbrowser.open(url)
        except Exception:
            pass
    uvicorn.run(app, host=host, port=port, log_level="warning")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """
    Builds and returns the ArgumentParser for the EDGAR command-line interface.

    This function defines all the available commands (`init-project`, `validate`, `run`,
    `test`, `test-fake`, `dashboard`, `launch-gcp`) and their respective arguments, help
    messages, and argument parsing logic.

    Returns:
        argparse.ArgumentParser: The configured argument parser.
    """
    parser = argparse.ArgumentParser(
        description="EDGAR project scaffold, validation, and run CLI"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_init = sub.add_parser(
        "init-project",
        help="Create a new project with seed programs, data loader, and config",
    )
    p_init.add_argument("task", type=str, help="Project name (folder under projects/)")

    p_validate = sub.add_parser(
        "validate", help="Validate project structure and required files"
    )
    p_validate.add_argument(
        "task", type=str, help="Project name (folder under projects/)"
    )

    # Helper for defining run args since run and test share most args
    def _add_run_args(p, help_str: str) -> None:
        p = sub.add_parser(p, help=help_str)
        p.add_argument("config", type=str, help="Path to config.yaml or task_spec.yaml")
        p.add_argument(
            "--log-level",
            choices=["compact", "code", "prompts"],
            default="compact",
            help="Logging verbosity: compact (default), code, or prompts",
        )

    _add_run_args("run", "Run an EDGAR experiment from a config.yaml or task_spec.yaml")
    _add_run_args("test", "Run a small test experiment with reduced evolution settings")

    p_resume = sub.add_parser(
        "resume",
        help="Resume a crashed/interrupted run from its output directory",
    )
    p_resume.add_argument(
        "run_dir",
        type=str,
        help="Path to a run directory (containing task_spec.yaml + population.jsonl + island_census.jsonl)",
    )
    p_resume.add_argument(
        "--log-level",
        choices=["compact", "code", "prompts"],
        default="compact",
        help="Logging verbosity: compact (default), code, or prompts",
    )

    sub.add_parser(
        "test-fake",
        help="Run a small end-to-end pipeline with fake LLM responses (no real API calls)",
    )

    p_dash = sub.add_parser(
        "dashboard",
        help="Launch the live + inspect dashboard for an EDGAR run",
    )
    p_dash.add_argument(
        "target",
        type=str,
        nargs="?",
        default=None,
        help="Path to a run dir (containing task_spec.yaml) or to program_databases/. "
        "Omit to scan ./program_databases.",
    )
    p_dash.add_argument("--port", type=int, default=8765)
    p_dash.add_argument("--host", type=str, default="127.0.0.1")
    p_dash.add_argument(
        "--no-open", action="store_true", help="don't auto-open the browser"
    )

    p_gcp = sub.add_parser(
        "launch-gcp",
        help="Launch a multi-run sweep on GCP (one GPU VM per run) from a launch spec",
    )
    p_gcp.add_argument(
        "spec",
        type=str,
        help="Path to a GCP launch spec YAML (see projects/gcp_launch.example.yaml)",
    )
    p_gcp.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the gcloud/gsutil commands and startup script without executing",
    )
    p_gcp.add_argument(
        "--teardown",
        action="store_true",
        help="Delete this user's EDGAR VMs instead of launching",
    )
    p_gcp.add_argument(
        "--fetch",
        action="store_true",
        help="Download results from the bucket to program_databases/ instead of launching",
    )

    return parser


def run_cli(argv=None) -> int:
    """
    Main entry point for the EDGAR command-line interface.

    Parses command-line arguments and dispatches to the appropriate function
    (`init_project`, `validate_project`, `_build_and_run`, `_run_test_fake`,
    or `_run_dashboard`) based on the subcommand provided. It handles known
    and unknown arguments, passing overrides to the run commands.

    Args:
        argv (list[str], optional): A list of command-line arguments to parse.
                                    If None, `sys.argv` is used. Defaults to None.

    Returns:
        int: The exit code of the executed command.
    """
    parser = build_parser()
    args, overrides = parser.parse_known_args(argv)

    if args.command == "init-project":
        return init_project(args.task)
    if args.command == "validate":
        return validate_project(args.task)
    if args.command == "run":
        print("Running experiment...")
        _build_and_run(args.config, overrides, args.log_level)
        return 0
    if args.command == "test":
        print("Running test run with real LLM calls...")
        _build_and_run(args.config, TEST_OVERRIDES + overrides, args.log_level)
        return 0
    if args.command == "resume":
        print("Resuming experiment...")
        return _build_and_resume(args.run_dir, args.log_level)

    if args.command == "test-fake":
        print("Running test run with fake LLM calls...")
        _run_test_fake()
        return 0
    if args.command == "dashboard":
        return _run_dashboard(
            target=args.target,
            port=args.port,
            host=args.host,
            no_open=args.no_open,
        )
    if args.command == "launch-gcp":
        from .cloud.launch_gcp import launch_gcp

        return launch_gcp(
            args.spec,
            teardown=args.teardown,
            dry_run=args.dry_run,
            fetch=args.fetch,
        )
    parser.error("Unknown command")
    return 2


if __name__ == "__main__":
    raise SystemExit(run_cli())
