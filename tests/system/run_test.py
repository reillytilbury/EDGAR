"""
System test runner for hypothesis_engine_fake.

Mirrors run.py but uses hypothesis_engine_fake with fake LLM responses
that cycle through Program1, Program2, ProgramSolution from tests/system/programs.py.
"""

import asyncio
import logging
import yaml
from pathlib import Path
import importlib
import os
import argparse
import inspect
import numpy as np
import pandas as pd

from tests.system import hypothesis_engine_fake

# JAX/XLA runtime guards to reduce GPU OOM frequency during large program sweeps.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
_xla_flags = os.environ.get("XLA_FLAGS", "")
if "--xla_gpu_enable_command_buffer=" not in _xla_flags:
    os.environ["XLA_FLAGS"] = (_xla_flags + " --xla_gpu_enable_command_buffer=").strip()

from src import utils
from src.prompt_manager import PromptManager
from src.data_summary import save_data_summary
from tests.system.programs import setup_fake_engine


def deep_merge(base: dict, override: dict) -> dict:
    """
    Recursively merge two dictionaries, with values in the override dictionary
    taking precedence over those in the base dictionary.

    - If both values for a key are dictionaries, they are merged recursively.
    - For other types, the value in the override dictionary replaces the value in the base dictionary.

    Args:
        base (dict): The base dictionary to be merged.
        override (dict): The dictionary whose values will override those in the base dictionary.

    Returns:
        dict: A new dictionary containing the merged values.
    """
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config_with_defaults(config_path: Path, project_root: Path) -> dict:
    """
    Load an experiment-specific configuration file and merge it with a default configuration file.

    - The default configuration file is located at `projects/config_default.yaml` relative to the project root.
    - Values in the experiment-specific configuration file override those in the default configuration file.

    Args:
        config_path (Path): The path to the experiment-specific configuration file.
        project_root (Path): The root directory of the project, used to locate the default configuration file.

    Returns:
        dict: A dictionary containing the merged configuration.

    Raises:
        ValueError: If the experiment-specific configuration file does not exist.
    """
    default_config_path = project_root / "projects" / "config_default.yaml"
    if not default_config_path.exists():
        raise FileNotFoundError(f"Default config file not found: {default_config_path}")

    with open(default_config_path) as f:
        default_config = yaml.safe_load(f) or {}

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path) as f:
        experiment_config = yaml.safe_load(f) or {}

    return deep_merge(default_config, experiment_config)


def _build_load_and_process_data_fn(spec_load_and_process_data_fn):
    """Wrap spec.load_and_process_data to validate the 2x2 output container."""

    def _wrapped_load_and_process_data_fn(**kwargs):
        data = spec_load_and_process_data_fn(**kwargs)
        data_arr = np.asarray(data, dtype=object)
        if data_arr.shape != (2, 2):
            raise ValueError(
                "load_and_process_data must return a 2x2 container "
                "[[data_train_train, data_train_test], [data_test_train, data_test_test]]."
            )
        for split_data in data_arr.reshape(-1):
            if not isinstance(split_data, dict):
                raise ValueError(
                    "Each split returned by load_and_process_data must be a dict[str, np.ndarray]."
                )
            utils.validate_data(split_data)
        return data_arr

    return _wrapped_load_and_process_data_fn


def _build_loss_fn(raw_loss_fn):
    """Validate loss signature for the engine contract: loss_fn(y_pred, y_true)"""
    if raw_loss_fn is None:
        return None

    sig = inspect.signature(raw_loss_fn)
    params = list(sig.parameters.values())
    has_varargs = any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in params)
    n_positional = sum(
        p.kind
        in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        for p in params
    )

    if has_varargs or n_positional not in (2, 3):
        raise ValueError("loss_fn must use signature loss_fn(y_pred, y_true).")

    if n_positional == 2:

        def _wrapped_loss_fn(y_pred, y_true, params=None):
            return raw_loss_fn(y_pred, y_true)

        return _wrapped_loss_fn

    def _wrapped_loss_fn(y_pred, y_true, params=None):
        return raw_loss_fn(y_pred, y_true, params)

    return _wrapped_loss_fn


def _broadcast_model_loss(loss_value, n_samples: int):
    if loss_value is None:
        return None
    loss_arr = np.asarray(loss_value)
    if loss_arr.size == 0:
        return None
    flat = loss_arr.reshape(-1)
    if flat.size == 1:
        return np.full(n_samples, float(flat[0]))
    if flat.size != n_samples:
        raise ValueError(
            f"Loss array size mismatch: got {flat.size}, expected {n_samples}."
        )
    return flat


def _programs_df_to_programs_list(
    programs_df, n_samples: int, params_col: str, loss_col: str | None
):
    programs_list = []
    if programs_df is None or len(programs_df) == 0:
        return programs_list

    for _, row in programs_df.iterrows():
        model = row.get("program")
        params = row.get(params_col)
        if model is None or params is None:
            continue
        program_dict = {
            "model": model,
            "params": utils.broadcast_params(params, n_samples),
        }
        if loss_col is not None and loss_col in row.index:
            losses = _broadcast_model_loss(row[loss_col], n_samples=n_samples)
            if losses is not None:
                program_dict["losses"] = losses
        programs_list.append(program_dict)
    return programs_list


def _build_plot_model_fits_fn(spec_plot_fn):
    """Build a stable plotting interface for hypothesis_engine."""
    if not callable(spec_plot_fn):
        return None

    sig = inspect.signature(spec_plot_fn)
    accepts_var_kwargs = any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
    )
    allowed_kwargs = set(sig.parameters.keys())

    def _wrapped_plot_fn(
        *,
        data,
        eval_grid,
        save_path: str,
        programs_df=None,
        programs_list=None,
        params_col: str = "params",
        loss_col: str | None = "train_loss",
        **kwargs,
    ):
        if save_path is None or save_path == "":
            return

        if programs_list is None:
            n_samples = utils.data_n_samples(data)
            programs_list_local = _programs_df_to_programs_list(
                programs_df=programs_df,
                n_samples=n_samples,
                params_col=params_col,
                loss_col=loss_col,
            )
        else:
            programs_list_local = programs_list

        if len(programs_list_local) == 0:
            return

        labels = kwargs.get("labels")
        n_programs = len(programs_list_local)
        if labels is None:
            kwargs["labels"] = tuple(f"model_{i + 1}" for i in range(n_programs))
        elif len(labels) < n_programs:
            padded = list(labels) + [
                f"model_{i + 1}" for i in range(len(labels), n_programs)
            ]
            kwargs["labels"] = tuple(padded)

        call_kwargs = {
            "data": data,
            "programs_list": programs_list_local,
            "eval_grid": eval_grid,
            "save_path": save_path,
            **kwargs,
        }
        if not accepts_var_kwargs:
            call_kwargs = {k: v for k, v in call_kwargs.items() if k in allowed_kwargs}
        spec_plot_fn(**call_kwargs)

    return _wrapped_plot_fn


async def _run_many(config_path: str = None, output_dir: str | None = None):
    """
    Run hypothesis_engine_fake with fake LLM responses.

    Args:
        config_path (str): Path to config.yaml. Defaults to tests/system/config.yaml.
        output_dir (str | None): Base output directory override. Defaults to program_databases/.
    """
    project_root = Path(__file__).parent.parent.parent  # EDGAR root
    if config_path is None:
        config_path = Path(__file__).parent / "config.yaml"
    config_path = Path(config_path)
    if not config_path.is_absolute():
        config_path = project_root / config_path

    print(f"Using config file: {config_path}")

    config = load_config_with_defaults(config_path, project_root)

    params = config.get("experiment_params", {})
    data_processing_params = config.get("data_processing_params", {})

    # Load the test spec directly instead of discovering via task name.
    spec_module = importlib.import_module("tests.system.spec")

    models = [getattr(spec_module, "model_v1"), getattr(spec_module, "model_v2")]
    param_estimators = [
        getattr(spec_module, "param_est_v1"),
        getattr(spec_module, "param_est_v2"),
    ]

    spec_load_and_process_data_fn = getattr(spec_module, "load_and_process_data")
    load_and_process_data_fn = _build_load_and_process_data_fn(
        spec_load_and_process_data_fn
    )

    spec_loss_fn = getattr(spec_module, "loss_fn", None)
    if spec_loss_fn is None or not callable(spec_loss_fn):
        raise ValueError(
            "tests.system.spec must define callable loss_fn(Y_pred, Y_true)."
        )

    spec_plot_fn = getattr(spec_module, "plot_model_fits", None)
    plot_model_fits_fn = (
        _build_plot_model_fits_fn(spec_plot_fn) if callable(spec_plot_fn) else None
    )

    loss_fn = _build_loss_fn(spec_loss_fn)

    prompt_manager = PromptManager(config=config)

    def _ring_topology(n_islands: int) -> list[int]:
        if n_islands <= 0:
            return []
        return list(range(1, n_islands)) + [0]

    def _topology_invalid(topology, n_islands: int) -> bool:
        if not isinstance(topology, (list, tuple)) or len(topology) != n_islands:
            return True
        for dest in topology:
            if not isinstance(dest, (int, np.integer)):
                return True
            if dest < 0 or dest >= n_islands:
                return True
        return False

    n_islands = int(params.get("n_islands", 0) or 0)
    if n_islands > 0:
        explore_top = params.get("exploration_topology")
        exploit_top = params.get("exploitation_topology")
        if _topology_invalid(explore_top, n_islands) or _topology_invalid(
            exploit_top, n_islands
        ):
            ring = _ring_topology(n_islands)
            msg = (
                f"Topology mismatch for n_islands={n_islands}. "
                f"Defaulting exploration/exploitation topology to ring: {ring}"
            )
            logging.warning(msg)
            print(f"Warning: {msg}")
            params["exploration_topology"] = ring
            params["exploitation_topology"] = ring

    random_seed = params.get("random_seed", 42)
    load_kwargs = dict(data_processing_params)
    if (
        "random_seed" in inspect.signature(spec_load_and_process_data_fn).parameters
        and "random_seed" not in load_kwargs
    ):
        load_kwargs["random_seed"] = random_seed

    data = load_and_process_data_fn(**load_kwargs)
    data_train_train = data[0, 0]

    data_eval = utils.build_evaluation_points(
        data=data_train_train,
        eval_keys=params.get("eval_keys"),
        x_min=params.get("x_min"),
        x_max=params.get("x_max"),
        n_bins=params.get("n_bins", 100),
    )

    if output_dir is not None:
        base_dir = str(output_dir)
    else:
        base_dir = os.path.join(os.getcwd(), "program_databases")
    print("Base directory:", base_dir)
    os.makedirs(base_dir, exist_ok=True)
    date_stamp = pd.Timestamp.now().strftime("%m-%d")
    time_stamp = pd.Timestamp.now().strftime("%H-%M-%S")
    full_dir_tuple = (base_dir, date_stamp, time_stamp)
    full_dir = os.path.join(*full_dir_tuple)
    os.makedirs(full_dir, exist_ok=True)
    print("Created folder:", full_dir)

    save_data_summary(
        data=data,
        output_dir=full_dir,
        random_seed=random_seed,
    )

    # Configure fake LLM responses before calling the engine.
    setup_fake_engine(params, spec_module.model_v1_jax, spec_module.model_v2_jax)

    def _require_llm(name: str):
        value = params.get(name)
        if value is None:
            raise ValueError(f"Missing required experiment_params.{name} in config.")
        return value

    await hypothesis_engine_fake.hypothesis_engine(
        n_iterations=params["n_iterations"],
        time_limit=params["time_limit"],
        param_penalty_weight=params["param_penalty_weight"],
        exploration_topology=params["exploration_topology"],
        exploitation_topology=params["exploitation_topology"],
        exploit_point=params["exploit_point"],
        k_max=params["k_max"],
        n_islands=params["n_islands"],
        batch_size=params["batch_size"],
        max_iter=params["max_iter"],
        critical_population_size=params["critical_population_size"],
        min_wise_population_size=params["min_wise_population_size"],
        n_migrants=params["n_migrants"],
        fit_params=params["fit_params"],
        use_param_estimator=params.get("use_param_estimator", True),
        learning_rate=params["learning_rate"],
        FAILED_PROGRAM_COST=params["FAILED_PROGRAM_COST"],
        model_llm=_require_llm("model_llm"),
        param_est_llm=_require_llm("param_est_llm"),
        jax_translator_llm=_require_llm("jax_translator_llm"),
        use_chat_mode=params.get("use_chat_mode", False),
        chat_token_limit=params.get("chat_token_limit", 50000),
        param_estimator_refinement_rounds=params.get(
            "param_estimator_refinement_rounds", 0
        ),
        numpy_programs=models,
        param_estimators=param_estimators,
        X=data,
        eval_grid=data_eval,
        plot_model_fits=plot_model_fits_fn,
        prompt_manager=prompt_manager,
        grad_descent_batch_size=params.get("grad_descent_batch_size", None),
        swear_words=params.get("swear_words"),
        open_family_tree=params.get("open_family_tree", False),
        loss_fn=loss_fn,
        random_seed=random_seed,
        full_dir_tuple=full_dir_tuple,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run system test for hypothesis_engine_fake"
    )
    parser.add_argument("--config", type=str, default=None, help="Path to config.yaml")
    args = parser.parse_args()
    asyncio.run(_run_many(config_path=args.config))
