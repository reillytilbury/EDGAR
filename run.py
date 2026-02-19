import asyncio
import yaml
from pathlib import Path
import importlib
import os, argparse
from src import hypothesis_engine 
from src.prompt_manager import PromptManager


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

async def _run_many(test_mode: bool = False, config_path: str = "config.yaml"):
    """
    _run_many(test_mode: bool = False, config_path: str = "config.yaml")
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
    Raises:
        ValueError: 
            - If the configuration file does not specify a `task`.
            - If the required `spec` module for the task cannot be imported.
            - If there are not exactly two numpy function seeds or two parameter estimator seeds.
            - If the `train_test_split` function is not callable.
            - If the `loss_fn` specified in the configuration is not callable.
    Details:
        - Resolves the configuration file path relative to the project root if not absolute.
        - Loads the configuration file and merges it with default values.
        - Dynamically imports the task-specific `spec` module based on the `task` specified in the configuration.
        - Validates the presence and callability of required functions in the `spec` module:
            - `model_v1`, `model_v2`: Numpy function seeds.
            - `param_est_v1`, `param_est_v2`: Parameter estimator seeds.
            - `load_and_process_data`: Data loading and processing function.
            - `train_test_split`: Function for splitting data into training and testing sets.
            - `plot_model_fits` (optional): Function for generating diagnostic plots.
            - `loss_fn` (optional): Loss function for the hypothesis engine.
        - Initializes a `PromptManager` for managing prompts during the experiment.
        - Supports multi-input configurations by extracting input names from the configuration.
        - Configures and runs the hypothesis engine with the specified parameters.
    Test Mode:
        When `test_mode` is enabled, the function overrides certain parameters to ensure quick execution:
            - `num_runs`: 1
            - `n_iterations`: 1
            - `time_limit`: 10 seconds
            - `n_islands`: 2
            - `k_max`: 2
            - `batch_size`: 2
            - `exploration_topology`: [1, 0]
            - `exploitation_topology`: [1, 0]
            - `max_iter`: 100
    Execution:
        For each run (based on `num_runs`), the hypothesis engine is invoked with the specified 
        parameters, including:
            - Iteration limits, time limits, and topology configurations.
            - Data processing and splitting functions.
            - Numpy programs and parameter estimators.
            - Loss function and prompt manager.
            - Additional configurations such as random seeds, token limits, and refinement rounds.
    Returns:
        None
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
    load_and_process_data_fn = getattr(spec_module, 'load_and_process_data')
    train_test_split_fn = getattr(spec_module, 'train_test_split', None)
    if not callable(train_test_split_fn):
        raise ValueError(
            f"{spec_module_path} must define callable train_test_split(X, random_seed)."
        )
    # Loss function: required in spec
    spec_loss_fn = getattr(spec_module, "loss_fn", None)
    if spec_loss_fn is None or not callable(spec_loss_fn):
        raise ValueError(f"{spec_module_path} must define callable loss_fn(y_true, y_pred).")

    # Image diagnostics are enabled automatically if spec defines plot_model_fits.
    spec_plot_fn = getattr(spec_module, "plot_model_fits", None)
    use_image_feedback = callable(spec_plot_fn)
    plot_model_fits_fn = spec_plot_fn if use_image_feedback else None

    # Initialize prompt manager with merged config (includes DEFAULT prompts)
    prompt_manager = PromptManager(config=config)

    # Extract input names from config (for multi-input support)
    # These stay in data_config for the data loading function to use
    inputs_config = config.get('inputs', [])
    if inputs_config:
        config['input_names'] = [p['name'] for p in inputs_config]
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
        print("running with standard params")
        await hypothesis_engine.hypothesis_engine(
            n_iterations=params['n_iterations'],
            time_limit=params['time_limit'],
            use_image_feedback=use_image_feedback,
            use_large_every=params['use_large_every'],
            param_penalty_weight=params['param_penalty_weight'],
            exploration_topology=params['exploration_topology'],
            exploit_point=params['exploit_point'],
            k_max=params['k_max'],
            n_islands=params['n_islands'],
            batch_size=params['batch_size'],
            max_iter=params['max_iter'],
            critical_population_size=params['critical_population_size'],
            min_wise_population_size=params['min_wise_population_size'],
            n_migrants=params['n_migrants'],
            fit_params=params['fit_params'],
            learning_rate=params['learning_rate'],
            x_min=params.get('x_min'),
            x_max=params.get('x_max'),
            n_bins=params.get('n_bins', 100),
            FAILED_PROGRAM_COST=params['FAILED_PROGRAM_COST'],
            tiny_lm_name=params['tiny_lm_name'],
            little_lm_name=params['little_lm_name'],
            large_lm_name=params['large_lm_name'],
            use_chat_mode=params.get('use_chat_mode', False),  # Default to legacy mode
            chat_token_limit=params.get('chat_token_limit', 50000),  # Max tokens per chat before auto-reset
            param_estimator_refinement_rounds=params.get('param_estimator_refinement_rounds', 0),
            numpy_programs=models,
            param_estimators=param_estimators,
            load_and_process_data_fn=load_and_process_data_fn,
            train_test_split_fn=train_test_split_fn,
            data_processing_params=data_processing_params,
            plot_model_fits_fn=plot_model_fits_fn,
            prompt_manager=prompt_manager,
            use_large_model_for_param_estimators=params.get('use_large_model_for_param_estimators', False),
            trial_batch_size=params.get('trial_batch_size', None),
            swear_words=params.get('swear_words'),
            loss_fn=spec_loss_fn,
            random_seed=params.get('random_seed', 42),
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Hypothesis Engine")
    parser.add_argument('--test_mode', action='store_true', help='Run in test mode with reduced iterations and time limit')
    parser.add_argument('--config', type=str, help='Path to experiment specific config file (relative to project root)', default="config.yaml")
    args = parser.parse_args()
    asyncio.run(_run_many(test_mode=args.test_mode, config_path=args.config))
