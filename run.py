import asyncio
import yaml
from pathlib import Path
import importlib
import os, argparse
from src import hypothesis_engine 
from src.prompt_manager import PromptManager

FORBIDDEN_WIRING_KEYS = {
    "load_and_process_data_fn",
    "create_train_test_split_fn",
    "create_train_test_sample_split_fn",
    "create_train_test_trial_split_fn",
    "seed_programs",
}

FORBIDDEN_EXPERIMENT_PARAM_KEYS = {
    "use_image_feedback",
}


def deep_merge(base: dict, override: dict) -> dict:
    """
    Recursively merge override into base. 
    Values in override take precedence over base.
    For nested dicts, merge recursively. For other types, override replaces base.
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
    Load experiment config and merge with DEFAULT config.
    Experiment-specific values override DEFAULT values.
    """
    candidate_defaults = [
        project_root / "projects" / "base" / "base_config.yaml",
        project_root / "projects" / "base" / "config.yaml",
    ]
    default_config = {}
    loaded_default_path = None
    for candidate in candidate_defaults:
        if candidate.exists():
            with open(candidate) as f:
                default_config = yaml.safe_load(f) or {}
            loaded_default_path = candidate
            break
    if loaded_default_path is not None:
        print(f"Loaded default config from: {loaded_default_path}")
    else:
        print("Warning: shared default config not found under projects/base")
    
    # Load experiment-specific config
    if not config_path.exists():
        raise ValueError(f"Config file not found: {config_path}")
    
    with open(config_path) as f:
        experiment_config = yaml.safe_load(f) or {}
    
    # Merge: experiment overrides default
    config = deep_merge(default_config, experiment_config)
    
    return config


async def _run_many(test_mode: bool = False, config_path: str = "config.yaml"):
    # Resolve config directory (relative to project root)
    project_root = Path(__file__).parent
    config_path = Path(config_path)
    if not config_path.is_absolute():
        config_path = project_root / config_path
    
    print(f"Using config file: {config_path}")
    
    # Load config with DEFAULT fallbacks
    config = load_config_with_defaults(config_path, project_root)
    
    forbidden_present = [k for k in FORBIDDEN_WIRING_KEYS if k in config]
    if forbidden_present:
        raise ValueError(
            "Config uses deprecated wiring keys. Remove these keys and rely on spec naming conventions: "
            + ", ".join(sorted(forbidden_present))
        )

    # Extract hyperparameters
    params = config.get('experiment_params', {})
    data_processing_params = config.get('data_processing_params', {})
    task_name = config.get('task')
    if not task_name:
        raise ValueError("Config must specify `task` so the reader can load projects.<task>.spec")
    forbidden_exp_params = [k for k in FORBIDDEN_EXPERIMENT_PARAM_KEYS if k in params]
    if forbidden_exp_params:
        raise ValueError(
            "Config uses deprecated experiment_params keys. These are auto-derived now: "
            + ", ".join(sorted(forbidden_exp_params))
        )
    
    # Auto-load experiment spec module by fixed naming convention.
    spec_module_path = f"projects.{task_name}.spec"
    try:
        spec_module = importlib.import_module(spec_module_path)
    except ModuleNotFoundError as e:
        raise ValueError(
            f"Could not import {spec_module_path}. Expected file projects/{task_name}/spec.py with required functions."
        ) from e

    numpy_programs = [getattr(spec_module, 'model_v1'), getattr(spec_module, 'model_v2')]
    param_estimators = [getattr(spec_module, 'param_est_v1'), getattr(spec_module, 'param_est_v2')]

    # assert that we have exactly 2 of each
    if len(numpy_programs) != 2:
        raise ValueError("There must be exactly 2 numpy function seeds.")
    if len(param_estimators) != 2:  
        raise ValueError("There must be exactly 2 parameter estimator seeds.")

    # Data extraction/splits: require unified split API from spec.
    load_and_process_data_fn = getattr(spec_module, 'load_and_process_data')
    train_test_split_fn = getattr(spec_module, 'train_test_split', None)
    if not callable(train_test_split_fn):
        raise ValueError(
            f"{spec_module_path} must define callable train_test_split(X, random_seed)."
        )

    # Image diagnostics are enabled automatically if spec defines plot_model_fits.
    spec_plot_fn = getattr(spec_module, "plot_model_fits", None)
    use_image_feedback = callable(spec_plot_fn)

    diagnostics_module = spec_module if use_image_feedback else None

    loss_fn_path = config.pop('loss_fn', None)
    if loss_fn_path:
        module_path, function_name = loss_fn_path.rsplit('.', 1)
        loss_module = importlib.import_module(module_path)
        loss_fn = getattr(loss_module, function_name)
    else:
        loss_fn = None

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
            tol=params['tol'],
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
            numpy_programs=numpy_programs,
            param_estimators=param_estimators,
            load_and_process_data_fn=load_and_process_data_fn,
            train_test_split_fn=train_test_split_fn,
            data_processing_params=data_processing_params,
            diagnostics_module=diagnostics_module,
            prompt_manager=prompt_manager,
            use_large_model_for_param_estimators=params.get('use_large_model_for_param_estimators', False),
            trial_batch_size=params.get('trial_batch_size', None),
            swear_words=params.get('swear_words'),
            loss_fn=loss_fn,
            random_seed=params.get('random_seed', 42),
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Hypothesis Engine")
    parser.add_argument('--test_mode', action='store_true', help='Run in test mode with reduced iterations and time limit')
    parser.add_argument('--config', type=str, help='Path to experiment specific config file (relative to project root)', default="config.yaml")
    args = parser.parse_args()
    asyncio.run(_run_many(test_mode=args.test_mode, config_path=args.config))
