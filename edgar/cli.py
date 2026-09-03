"""
CLI for EDGAR project scaffolding, validation, and running experiments.

This module provides command-line interface (CLI) commands to manage EDGAR projects,
including initialization, validation, and execution of evolutionary experiments.
It also includes commands for resuming interrupted runs, running scaled-down tests,
and launching a web-based dashboard for monitoring and analysis.

Commands
--------
init-project
    Creates a new project scaffold:

        edgar init-project my_task

    This command creates the following structure within the `projects/my_task`
    directory (or `experiments/my_task` if `experiments/` exists):
    - `config.yaml`: Main configuration file for the EDGAR run.
    - `seed_programs/model1.py`, `model2.py`: Placeholder files for initial
      model programs.
    - `seed_programs/param_est1.py`, `param_est2.py`: Placeholder files for
      initial parameter estimator programs.
    - `data_loader/load_data.py`: Placeholder for data loading and loss function
      definitions.
    - `image_feedback/plot.py`: Placeholder for plotting functions used for
      LLM feedback and program visualization.

    Each generated file contains stub functions with detailed Google-style
    docstrings, guiding the user to fill in the implementations.
    Existing files will be overwritten by this command.

validate
    Checks that all required files and functions exist for a project:

        edgar validate my_task

    This command verifies the presence of essential project files and ensures
    that the expected functions (e.g., `model`, `parameter_estimator`,
    `load_data`, `loss_fn`, `plot_model_fits`) are defined within them.
    It helps to ensure a correct project setup before an EDGAR experiment is run.

run
    Runs an EDGAR experiment from a `config.yaml` or a previously saved `task_spec.yaml`:

        edgar run projects/my_task/config.yaml
        edgar run program_databases/my_task/2026-05-01/14-32-10/task_spec.yaml

    Logging verbosity can be controlled (default: `compact`):

        edgar run projects/my_task/config.yaml --log-level code
        edgar run projects/my_task/config.yaml --log-level prompts

    Configuration values can be overridden at the command line using `--section.key=value`
    syntax:

        edgar run projects/my_task/config.yaml --evolution.n_generations=20
        edgar run projects/my_task/config.yaml --io.data_path=/data/new.npy --llms.model_llm=gemini-2.5-pro

    Valid top-level sections for overrides include `io`, `evolution`, `llms`,
    `scoring`, `project_params`, and `run`. Keys can be dotted to access nested
    configuration parameters (e.g., `scoring.gradient_descent.max_iter`).
    Values are parsed as Python literals (int, float, bool) where possible;
    otherwise, they are treated as strings.

resume
    Resumes a crashed or interrupted EDGAR run from its output directory:

        edgar resume program_databases/my_task/2026-05-26/14-54-15/

    The command automatically picks up the evolution loop at the next unfinished
    generation and continues writing output into the same directory.
    Logging verbosity can also be controlled with `--log-level`.

test
    Runs a scaled-down test experiment with real LLM calls:
        edgar test projects/my_task/config.yaml

    This command applies a set of predefined overrides (e.g., `n_generations=1`,
    `n_islands=2`, `batch_size=2`, `max_iter=100`, `n_param_ests=1`) to
    reduce the computational burden. It is useful for quickly checking that the
    EDGAR pipeline runs end-to-end with actual LLM interactions.
    Output is saved to `./test_output/`.

test-fake
    Runs an end-to-end pipeline using mocked LLM responses:
        edgar test-fake

    This command enables testing the EDGAR pipeline without requiring LLM API access
    or incurring API costs, by utilizing a fake runner that provides simulated
    LLM responses.
    Output is saved to `./test_output/`.

dashboard
    Launches a web-based dashboard for real-time monitoring and post-hoc
    analysis of EDGAR runs:
        edgar dashboard [<run_dir>]

    The dashboard can target a specific run directory (containing `task_spec.yaml`),
    a `program_databases/` root, or scan the default location (`./program_databases`).
    It allows specifying `port` and `host`, and a `--no-open` flag to prevent
    automatic browser launch.
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
        """Loads and preprocesses data, then splits it into discovery, validation, and evaluation sets.

        This function is responsible for loading raw experimental data, performing any
        necessary preprocessing, and then dividing the data into distinct sets:
        `discover` (used during the evolutionary algorithm's search phase), `validate`
        (used for final model selection), and `eval` (a small subset for model
        fingerprinting).

        Args:
            data_path (str): The path to the raw data file.
            n_eval_samples (int): The number of samples to use for generating the
                evaluation fingerprint, which is used for deduplication.
            **kwargs: Additional parameters passed from the `project_params`
                section of `config.yaml`.

        Returns:
            tuple: A tuple containing (X_discover, X_validate, X_eval).

                - **X_discover** (tuple): Contains `X_disc_train` and `X_disc_test`.
                    - `X_disc_train` (dict): A dictionary of JAX arrays, typically
                      with shape `(n_samples_discover_train, n_trials)`. This data
                      is exposed to the LLM during the model discovery phase.
                    - `X_disc_test` (dict): A dictionary of JAX arrays, typically
                      with shape `(n_samples_discover_test, n_trials)`. This is a
                      held-out test set used within the discovery phase for scoring
                      and feedback.
                - **X_validate** (tuple): Contains `X_val_train` and `X_val_test`.
                    - `X_val_train` (dict): A dictionary of JAX arrays, typically
                      with shape `(n_samples_validate_train, n_trials)`. This data
                      is never seen during the discovery phase and is reserved for
                      final model validation.
                    - `X_val_test` (dict): A dictionary of JAX arrays, typically
                      with shape `(n_samples_validate_test, n_trials)`. This is a
                      final held-out evaluation set for robust performance assessment.
                - **X_eval** (dict): A small subset of data from `X_disc_train` used
                  for generating model fingerprints for deduplication. It contains JAX
                  arrays plus `_sample_indices` (a NumPy integer array indicating
                  positions within `disc_idx`).
        """
        raise NotImplementedError


    def loss_fn(model_output, data):
        """Computes the per-sample loss between model predictions and actual data.

        This function calculates the loss for each individual sample, which can then
        be aggregated to determine the overall fitness of a model. The loss should be
        designed such that lower values indicate a better fit.

        Args:
            model_output: A JAX array of model predictions. Its shape can be
                          `(n_trials,)` for a single sample or `(n_samples, n_trials)`
                          if batched.
            data (dict): A dictionary of JAX arrays containing the ground truth data
                         for the current split, e.g., `data['response']`.

        Returns:
            JAX array: An array of per-sample losses, typically with shape
            `(n_samples,)`.
        """
        raise NotImplementedError
    '''
)

SPEC_TEMPLATE_MODEL = dedent(
    '''\
    import numpy as np


    def model(data, params):
        """Evaluates the model for a single sample given data and parameters.

        This function represents the core mathematical or algorithmic model.
        It takes a single sample of data and a set of parameters, and
        produces predictions.

        Args:
            data (dict): A dictionary containing data for one sample,
                         e.g., `data['stimulus']` with shape `(n_trials,)`.
            params (dict): A dictionary of model parameters, with keys
                           matching those defined in `model.DEFAULT_PARAMS`.

        Returns:
            np.ndarray: The model's predictions for the given sample,
                        typically with shape `(n_trials,)`.
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
        """Estimates model parameters for a single sample.

        This function takes a single sample of data and attempts to estimate
        the optimal parameters for the `model` function. This is often used
        to provide good initial parameter guesses for gradient descent optimization.

        Args:
            data (dict): A dictionary containing data for one sample,
                         e.g., `data['stimulus']` and `data['response']`,
                         each with shape `(n_trials,)`.

        Returns:
            dict: A dictionary of estimated parameters, with keys matching
                  those defined in `model.DEFAULT_PARAMS`.
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
        """Optional. Generates plots comparing model predictions to data for feedback.

        This function is used to create visual feedback for the LLM during the
        evolutionary process and for visualizing program performance in the dashboard.
        It should generate plots that help in understanding how well the parent programs
        fit the provided data.

        Args:
            data (dict): A dictionary of JAX arrays, typically the `X_disc_train`
                         data with shape `(n_samples, n_trials)`.
            parent_programs (list): A list of `Program` objects. Each `Program`
                                    object provides access to its compiled model,
                                    optimized parameters, sample-wise losses,
                                    and overall scalar loss. Key attributes include:
                                    - `.compile()`: Returns a tuple `(model_fn, param_est_fn)`.
                                    - `.params`: A dictionary of per-sample parameters,
                                                 where each value has shape `(n_samples, ...)`.
                                    - `.sample_losses`: Per-sample losses, with shape
                                                        `(n_samples,)`, or `None`.
                                    - `.program_losses.discover.final`: The scalar
                                                                        overall loss.
            save_path (str): The file path (not a directory) where the generated
                             figure should be saved.
        """
        pass
    '''
)


def _find_project_root() -> Path:
    """Finds the root directory of the EDGAR project.

    The project root is determined by navigating up from the current file's
    location until the `edgar/` directory is found.

    Returns:
        Path: The absolute path to the EDGAR project root directory.
    """
    return Path(__file__).resolve().parent.parent


def _find_collection_dir(project_root: Path) -> Path:
    """Determines the directory where EDGAR projects are stored.

    This function prioritizes a directory named `projects/`. If `projects/`
    does not exist, it checks for `experiments/`. If neither exist, it
    defaults to creating `projects/`.

    Args:
        project_root (Path): The root directory of the EDGAR project.

    Returns:
        Path: The path to the project collection directory (e.g., `projects/` or `experiments/`).
    """
    projects_dir = project_root / "projects"
    experiments_dir = project_root / "experiments"
    if projects_dir.exists():
        return projects_dir
    if experiments_dir.exists():
        return experiments_dir
    return projects_dir


def _task_dir(task: str) -> Path:
    """Constructs the absolute path for a given EDGAR task directory.

    Args:
        task (str): The name of the EDGAR task (e.g., "my_task").

    Returns:
        Path: The absolute path to the task's directory within the project
              collection (e.g., `projects/my_task/`).
    """
    root = _find_project_root()
    collection = _find_collection_dir(root)
    return collection / task


def init_project(task: str) -> int:
    """Initializes a new EDGAR project with a predefined directory structure and template files.

    This command scaffolds a new project under `projects/` (or `experiments/` if that exists)
    by creating directories for seed programs, data loader, and image feedback, along with
    template Python files and a default `config.yaml`. Existing files will be overwritten.

    Args:
        task (str): The name of the project to initialize. This will be the name of the
                    directory created under `projects/` (e.g., `projects/my_task`).

    Returns:
        int: Exit code (0 for success).
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
    """Validates an EDGAR project by checking for the existence of required files and functions.

    This function ensures that the specified project directory exists and contains all
    necessary files (e.g., `model1.py`, `load_data.py`, `config.yaml`). It also
    verifies that these files define the expected functions (`model`,
    `parameter_estimator`, `load_data`, `loss_fn`, `plot_model_fits`).
    This validation step is crucial to ensure a correct setup before an
    experiment is run.

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


def _apply_overrides(config, overrides: list[str]) -> None:
    """Applies command-line overrides to a configuration object in-place.

    Each override must be in the format `--section.key=value`, where `section` is
    a top-level configuration category (e.g., `io`, `evolution`, `llms`, `scoring`,
    `project_params`, `run`). The `key` can be dotted to specify nested
    configuration parameters (e.g., `scoring.gradient_descent.max_iter`).
    Values are parsed as Python literals (integers, floats, booleans) if possible;
    otherwise, they are treated as strings.

    Example:
        `_apply_overrides(config, ["evolution.n_generations=1", "io.data_path=/data/foo.npy"])`

    Args:
        config: The `Config` object to which overrides will be applied. This
                object is modified directly.
        overrides (list[str]): A list of string overrides, each in the format
                               `--section.key=value`.

    Raises:
        ValueError: If an override string is not in the correct format or
                    specifies an unknown configuration section.
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
        if section == "llms" and key == "default_provider":
            raise ValueError(
                "cannot override 'llms.default_provider': it only fills unset roles at "
                "config load, so it has no effect applied as an override. Set the role "
                "models directly (e.g. llms.model_llm=...) or change default_provider in "
                "config.yaml."
            )
        try:
            import ast

            value = ast.literal_eval(value_str)
        except (ValueError, SyntaxError):
            value = value_str

        if section == "project_params":
            config.project_params[key] = value
        else:
            *nested, leaf = key.split(".")
            sub_model = getattr(config, section)
            for part in nested:
                sub_model = getattr(sub_model, part)
            setattr(sub_model, leaf, value)


TEST_OVERRIDES = [
    "--io.save_path=./test_output",
    "--evolution.n_generations=1",
    "--evolution.n_islands=2",
    "--evolution.batch_size=2",
    "--llms.num_parents=2",
    "--evolution.topology=[1, 0]",
    "--scoring.gradient_descent.max_iter=100",
    "--scoring.timeout_s=120",
    "--scoring.n_param_ests=1",
    "--llms.model_llm=gemini-2.5-flash",
    "--llms.param_est_llm=gemini-2.5-flash",
    "--llms.jax_model_translator_llm=gemini-2.5-flash-lite",
    # "--llms.log_raw_llm_response=True",
]


def _build_and_run(config_path: str, overrides: list[str], log_level: str) -> None:
    """Builds a `TaskSpec` from a configuration and runs an EDGAR experiment.

    This function serves as a central point for initiating an EDGAR run. It
    first loads the experiment configuration from the specified `config_path`
    (which can be either a `config.yaml` for a new run or a `task_spec.yaml`
    from a previous run). It then applies any command-line overrides to the
    configuration before constructing a `TaskSpec` object. Finally, it
    invokes the main asynchronous `run` function to execute the evolutionary
    experiment.

    Args:
        config_path (str): The file path to either a `config.yaml` for a new
                           experiment or a `task_spec.yaml` from a previously
                           saved run.
        overrides (list[str]): A list of command-line override strings,
                               e.g., `"--evolution.n_generations=20"`. These
                               overrides modify the loaded configuration.
        log_level (str): The desired logging verbosity for the run. Valid
                         options are 'compact', 'code', or 'prompts'.
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
    if overrides:
        _apply_overrides(config, overrides)
    spec = TaskSpec.from_config(config)
    asyncio.run(run(spec, log_level=log_level))


def _build_and_resume(run_dir: str, log_level: str) -> int:
    """Resumes a crashed or interrupted EDGAR run from its output directory.

    This function reconstructs the `TaskSpec` from the `task_spec.yaml` file
    saved in the specified `run_dir` and continues the evolutionary loop.
    It identifies the last completed generation and restarts from the next
    unfinished one, writing all subsequent output back into the same
    directory. It performs checks to ensure the run is in a resumable state.

    Args:
        run_dir (str): The path to the output directory of the run to be resumed.
                       This directory must contain a `task_spec.yaml`.
        log_level (str): The desired logging verbosity for the resumed run.
                         Valid options are 'compact', 'code', or 'prompts'.

    Returns:
        int: Exit code (0 for success, 1 for failure due to unresumable state).
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
    """Runs a scaled-down EDGAR experiment with mocked LLM responses.

    This function is specifically designed for end-to-end testing of the EDGAR
    pipeline without incurring API costs or waiting for real LLM responses.
    It achieves this by invoking a fake runner from the system tests that
    simulates LLM interactions. Output is saved to `./test_output/`.
    """
    import sys

    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from tests.system.fake_runner import run_test_fake

    run_test_fake()


def _run_dashboard(target: str | None, port: int, host: str, no_open: bool) -> int:
    """Starts the EDGAR dashboard server, optionally opening a browser window.

    The dashboard can operate in several modes depending on the `target`
    parameter:
    - If `target` is `None`, the dashboard scans the default `./program_databases`
      directory for available runs, allowing the user to select one.
    - If `target` is a path to a single run directory (which must contain a
      `task_spec.yaml`), the dashboard will open directly to inspect that run.
    - If `target` is a path to a `program_databases/` root directory, the
      dashboard will scan this root for runs.

    Args:
        target (str | None): The target for the dashboard. Can be a path to a
                             specific run directory, a `program_databases/` root,
                             or `None` to scan the default location.
        port (int): The starting port number to attempt for the dashboard server.
                    The function will try consecutive ports if the specified one
                    is already in use.
        host (str): The host address for the dashboard server (e.g., "127.0.0.1").
        no_open (bool): If `True`, the function prevents the automatic opening
                        of a web browser window to the dashboard URL.

    Returns:
        int: Exit code (0 for success, 1 for failure, e.g., if the target path
             does not exist).
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
    """Builds and returns the ArgumentParser for the EDGAR command-line interface.

    This function defines all the available commands and their respective arguments,
    help messages, and argument parsing logic for the EDGAR CLI. This includes
    commands for project management (`init-project`, `validate`), experiment
    execution (`run`, `test`, `resume`, `test-fake`), monitoring (`dashboard`),
    and cloud deployment (`launch-gcp`).

    Returns:
        argparse.ArgumentParser: The configured argument parser instance.
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
    """Main entry point for the EDGAR command-line interface.

    This function parses command-line arguments and dispatches to the
    appropriate subcommand function based on user input. It supports
    project initialization, validation, running experiments, resuming
    interrupted runs, testing with fake or real LLM calls, launching
    the dashboard, and deploying to GCP.

    Args:
        argv (list[str], optional): A list of command-line arguments to parse.
                                    If `None`, `sys.argv` is used. Defaults to `None`.

    Returns:
        int: The exit code of the executed command (0 for success, non-zero for
             various failure conditions).
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
"""