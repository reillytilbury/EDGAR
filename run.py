import asyncio
import yaml
from pathlib import Path
import importlib
import os, argparse
import inspect
import numpy as np
from src import hypothesis_engine, utils
from src.prompt_manager import PromptManager
from src.data_summary import save_data_summary


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

    # Load experiment-specific config
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path) as f:
        experiment_config = yaml.safe_load(f) or {}

    # Merge default and project-specific configs, with project-specific values taking precedence
    return deep_merge(default_config, experiment_config)


def _build_load_and_process_data_fn(spec_load_and_process_data_fn):
    """
    Wrap spec.load_and_process_data so hypothesis_engine always receives
    a validated data dict (dict[str, np.ndarray]).
    """
    def _wrapped_load_and_process_data_fn(**kwargs):
        data = spec_load_and_process_data_fn(**kwargs)
        if not isinstance(data, dict):
            raise ValueError(
                "load_and_process_data must return a dict[str, np.ndarray]."
            )
        utils.validate_data(data)
        return data

    return _wrapped_load_and_process_data_fn


def _build_train_test_split_fn(spec_train_test_split_fn):
    """
    Wrap spec.train_test_split so hypothesis_engine gets fully materialized
    sample+trial train/test indices.
    """
    def _wrapped_train_test_split_fn(data: dict, random_seed: int):
        train_samples_raw, train_trials_raw = spec_train_test_split_fn(
            data,
            random_seed,
        )
        n_samples = utils.data_n_samples(data)
        n_trials = utils.data_n_trials(data)
        train_samples = np.asarray(train_samples_raw).reshape(-1).astype(np.int64, copy=False)
        train_trials = np.asarray(train_trials_raw).reshape(-1).astype(np.int64, copy=False)
        test_samples = np.setdiff1d(np.arange(n_samples, dtype=np.int64), train_samples, assume_unique=False)
        test_trials = np.setdiff1d(np.arange(n_trials, dtype=np.int64), train_trials, assume_unique=False)
        return train_samples, test_samples, train_trials, test_trials

    return _wrapped_train_test_split_fn


def _build_loss_fn(raw_loss_fn):
    """
    Validate loss signature for the engine contract:
      loss_fn(model_output, data)
    """
    if raw_loss_fn is None:
        return None

    sig = inspect.signature(raw_loss_fn)
    params = list(sig.parameters.values())
    has_varargs = any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in params)
    n_positional = sum(
        p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        for p in params
    )

    if has_varargs or n_positional != 2:
        raise ValueError(
            "loss_fn must use signature loss_fn(model_output, data)."
        )

    return raw_loss_fn


def _zscore_data(data: dict, skip_keys: list | None = None, eps: float = 1e-12) -> dict:
    """
    Z-score each array in the data dict across the last dimension (trials).

    For each key, z-scores independently along the trial axis. Arrays are expected
    to have shape (..., n_trials).

    Args:
        data: Data dict where all arrays share the same last dimension.
        skip_keys: Keys to skip (e.g., categorical features).
        eps: Small constant to avoid division by zero.
    """
    skip = set(skip_keys or [])
    result = {}
    for key, arr in data.items():
        if key in skip:
            result[key] = arr
        else:
            arr = np.asarray(arr)
            mu = arr.mean(axis=-1, keepdims=True)
            sd = arr.std(axis=-1, keepdims=True)
            result[key] = (arr - mu) / (sd + eps)
    return result


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


def _programs_df_to_programs_list(programs_df, n_samples: int, params_col: str, loss_col: str | None):
    programs_list = []
    if programs_df is None or len(programs_df) == 0:
        return programs_list

    for _, row in programs_df.iterrows():
        model = row.get("program")
        params = row.get(params_col)
        if model is None or params is None:
            continue
        program_dict = {"model": model, "params": utils.broadcast_params(params, n_samples)}
        if loss_col is not None and loss_col in row.index:
            losses = _broadcast_model_loss(row[loss_col], n_samples=n_samples)
            if losses is not None:
                program_dict["losses"] = losses
        programs_list.append(program_dict)
    return programs_list


def _build_plot_model_fits_fn(spec_plot_fn):
    """
    Build a stable plotting interface for hypothesis_engine.

    Engine-facing contract:
      plot_fn(data, programs_list, X_eval, save_path, labels=None)
    """
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
        X_eval,
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
            kwargs["labels"] = tuple(f"model_{i+1}" for i in range(n_programs))
        elif len(labels) < n_programs:
            padded = list(labels) + [f"model_{i+1}" for i in range(len(labels), n_programs)]
            kwargs["labels"] = tuple(padded)

        call_kwargs = {
            "data": data,
            "programs_list": programs_list_local,
            "X_eval": X_eval,
            "save_path": save_path,
            **kwargs,
        }
        if not accepts_var_kwargs:
            call_kwargs = {k: v for k, v in call_kwargs.items() if k in allowed_kwargs}
        spec_plot_fn(**call_kwargs)

    return _wrapped_plot_fn


async def _run_many(test_mode: bool = False, config_path: str = "config.yaml"):
    """
    Asynchronously runs multiple experiments based on the provided configuration and parameters.
    This function is designed to handle the setup, configuration, and execution of experiments
    using a hypothesis engine. It supports both standard and test modes, allowing for flexible
    execution depending on the use case.
    Args:
        test_mode (bool, optional):
            If True, runs the function in test mode with reduced parameters for quick validation.
            Defaults to False.
        config_path (str, optional):
            Path to the configuration file. Can be an absolute path or relative to the project root.
            Defaults to "config.yaml".
    """
    # Resolve config directory (relative to project root)
    project_root = Path(__file__).parent
    config_path = Path(config_path)
    if not config_path.is_absolute():
        config_path = project_root / config_path

    print(f"Using config file: {config_path}")

    # Load config with DEFAULT fallbacks
    config = load_config_with_defaults(config_path, project_root)

    # Extract hyperparameters
    params = config.get('experiment_params', {})
    data_processing_params = config.get('data_processing_params', {})
    task_name = config.get('task')
    if not task_name:
        raise ValueError("Config must specify `task` so the reader can load projects.<task>.spec")

    # Auto-load experiment spec module by fixed naming convention.
    spec_module_path = f"projects.{task_name}.spec"
    try:
        spec_module = importlib.import_module(spec_module_path)
    except ModuleNotFoundError as e:
        raise ValueError(
            f"Could not import {spec_module_path}. Expected file projects/{task_name}/spec.py with required functions."
        ) from e

    models = [getattr(spec_module, 'model_v1'), getattr(spec_module, 'model_v2')]
    param_estimators = [getattr(spec_module, 'param_est_v1'), getattr(spec_module, 'param_est_v2')]

    # assert that we have exactly 2 of each
    if len(models) != 2:
        raise ValueError("There must be exactly 2 model seeds.")
    if len(param_estimators) != 2:
        raise ValueError("There must be exactly 2 parameter estimator seeds.")

    # Data extraction/splits: require unified split API from spec.
    spec_load_and_process_data_fn = getattr(spec_module, 'load_and_process_data')
    spec_train_test_split_fn = getattr(spec_module, 'train_test_split', None)
    if not callable(spec_train_test_split_fn):
        raise ValueError(
            f"{spec_module_path} must define callable train_test_split(X, random_seed)."
        )
    load_and_process_data_fn = _build_load_and_process_data_fn(spec_load_and_process_data_fn)
    train_test_split_fn = _build_train_test_split_fn(spec_train_test_split_fn)

    # Loss function: required in spec
    spec_loss_fn = getattr(spec_module, "loss_fn", None)
    if spec_loss_fn is None or not callable(spec_loss_fn):
        raise ValueError(f"{spec_module_path} must define callable loss_fn(model_output, data).")

    # Image diagnostics are enabled automatically if spec defines plot_model_fits.
    spec_plot_fn = getattr(spec_module, "plot_model_fits", None)
    plot_model_fits_fn = _build_plot_model_fits_fn(spec_plot_fn) if callable(spec_plot_fn) else None

    raw_loss_fn = spec_loss_fn
    loss_fn = _build_loss_fn(raw_loss_fn)

    # Initialize prompt manager with merged config (includes DEFAULT prompts)
    prompt_manager = PromptManager(config=config)

    # Extract input names from config (for multi-input support)
    # These stay in data_config for the data loading function to use
    inputs_config = config.get('inputs', [])
    if inputs_config:
        config['input_names'] = [p['name'] for p in inputs_config]

    # Z-score config
    zscore_skip_keys = config.get('zscore_skip_keys', None)
    penalty_denominator = params.get('penalty_denominator', 1)

    if test_mode:
        params['num_runs'] = 1
        params['n_iterations'] = 1
        params['time_limit'] = 10  # seconds
        params['n_islands'] = 2
        params['k_max'] = 2
        params['batch_size'] = 2
        params['exploration_topology'] = [1, 0]
        params['exploitation_topology'] = [1, 0]
        params['max_iter'] = 100 # --- USE WHEN JUST CHECKING THAT THE SCRIPT RUNS ---

    for i in range(params['num_runs']):
        random_seed = params.get('random_seed', 42)
        data = load_and_process_data_fn(**data_processing_params)
        train_samples, test_samples, train_trials, test_trials = train_test_split_fn(
            data,
            random_seed,
        )
        train_samples = np.asarray(train_samples).reshape(-1).astype(np.int64, copy=False)
        test_samples = np.asarray(test_samples).reshape(-1).astype(np.int64, copy=False)
        train_trials = np.asarray(train_trials).reshape(-1).astype(np.int64, copy=False)
        test_trials = np.asarray(test_trials).reshape(-1).astype(np.int64, copy=False)

        # Create 2x2 nested holdout splits
        data_train_train = utils.slice_data_trials(utils.slice_data_samples(data, train_samples), train_trials)
        data_train_test = utils.slice_data_trials(utils.slice_data_samples(data, train_samples), test_trials)
        data_test_train = utils.slice_data_trials(utils.slice_data_samples(data, test_samples), train_trials)
        data_test_test = utils.slice_data_trials(utils.slice_data_samples(data, test_samples), test_trials)

        # Z-score each split independently
        data_train_train = _zscore_data(data_train_train, skip_keys=zscore_skip_keys)
        data_train_test = _zscore_data(data_train_test, skip_keys=zscore_skip_keys)
        data_test_train = _zscore_data(data_test_train, skip_keys=zscore_skip_keys)
        data_test_test = _zscore_data(data_test_test, skip_keys=zscore_skip_keys)

        X = np.empty((2, 2), dtype=object)
        X[0, 0] = data_train_train
        X[0, 1] = data_train_test
        X[1, 0] = data_test_train
        X[1, 1] = data_test_test

        X_eval = utils.build_evaluation_points(
            data=X[0, 0],
            eval_keys=params.get('eval_keys'),
            x_min=params.get('x_min'),
            x_max=params.get('x_max'),
            n_bins=params.get('n_bins', 100),
        )

        print("running with standard params")
        def _require_llm(name: str):
            value = params.get(name)
            if value is None:
                raise ValueError(f"Missing required experiment_params.{name} in config.")
            return value
        full_dir = await hypothesis_engine.hypothesis_engine(
            n_iterations=params['n_iterations'],
            time_limit=params['time_limit'],
            param_penalty_weight=params['param_penalty_weight'],
            penalty_denominator=penalty_denominator,
            exploration_topology=params['exploration_topology'],
            exploitation_topology=params['exploitation_topology'],
            exploit_point=params['exploit_point'],
            k_max=params['k_max'],
            n_islands=params['n_islands'],
            batch_size=params['batch_size'],
            max_iter=params['max_iter'],
            critical_population_size=params['critical_population_size'],
            min_wise_population_size=params['min_wise_population_size'],
            n_migrants=params['n_migrants'],
            fit_params=params['fit_params'],
            use_param_estimator=params.get('use_param_estimator', True),
            learning_rate=params['learning_rate'],
            FAILED_PROGRAM_COST=params['FAILED_PROGRAM_COST'],
            model_llm=_require_llm('model_llm'),
            param_est_llm=_require_llm('param_est_llm'),
            jax_translator_llm=_require_llm('jax_translator_llm'),
            linked_llm=_require_llm('linked_llm'),
            use_chat_mode=params.get('use_chat_mode', False),  # Default to legacy mode
            chat_token_limit=params.get('chat_token_limit', 50000),  # Max tokens per chat before auto-reset
            param_estimator_refinement_rounds=params.get('param_estimator_refinement_rounds', 0),
            numpy_programs=models,
            param_estimators=param_estimators,
            X=X,
            X_eval=X_eval,
            plot_model_fits=plot_model_fits_fn,
            prompt_manager=prompt_manager,
            trial_batch_size=params.get('trial_batch_size', None),
            swear_words=params.get('swear_words'),
            open_family_tree=params.get('open_family_tree', False),
            loss_fn=loss_fn,
            use_linked_prompt=params.get('use_linked_prompt', False),
            random_seed=random_seed,
        )
        # Save split/data summary from preprocessed run-level data.
        save_data_summary(
            data=data,
            training_samples=train_samples,
            test_samples=test_samples,
            train_trials=train_trials,
            test_trials=test_trials,
            output_dir=full_dir,
            random_seed=random_seed,
            train_test_split_fn=spec_train_test_split_fn,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Hypothesis Engine")
    parser.add_argument('--test_mode', action='store_true', help='Run in test mode with reduced iterations and time limit')
    parser.add_argument('--config', type=str, help='Path to experiment specific config file (relative to project root)', default="config.yaml")
    args = parser.parse_args()
    asyncio.run(_run_many(test_mode=args.test_mode, config_path=args.config))
