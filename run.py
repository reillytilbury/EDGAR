import asyncio
import yaml
from pathlib import Path
import importlib
import os, argparse
from src import hypothesis_engine 
from src.diagnostics_manager import load_diagnostics
from src.prompt_manager import PromptManager


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
    default_config_path = project_root / "experiments" / "DEFAULT" / "config.yaml"
    
    # Load DEFAULT config first
    if default_config_path.exists():
        with open(default_config_path) as f:
            default_config = yaml.safe_load(f) or {}
        print(f"Loaded default config from: {default_config_path}")
    else:
        default_config = {}
        print(f"Warning: DEFAULT config not found at {default_config_path}")
    
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
    config_path = project_root / config_path
    
    print(f"Using config file: {config_path}")
    
    # Load config with DEFAULT fallbacks
    config = load_config_with_defaults(config_path, project_root)
    
    # Extract hyperparameters
    params = config.get('experiment_params', {})
    seed_programs = config.get('seed_programs', {})
    data_processing_params = config.get('data_processing_params', {})
    
    # Load experiment-specific diagnostics (returns None if not configured or file missing)
    diagnostics_path = config.get('diagnostics_path', None)
    diagnostics_module = load_diagnostics(diagnostics_path)

    # Dynamically load seed programs module
    module_path = seed_programs.get('module')
    if not module_path:
        raise ValueError("seed_programs.module not specified in config.yaml")
    
    seed_module = importlib.import_module(module_path)
    
    # Get function seeds
    function_seed_names = seed_programs.get('function_seeds', [])
    numpy_programs = [getattr(seed_module, name) for name in function_seed_names]

    jax_function_seed_names = seed_programs.get('jax_function_seeds', [])
    if jax_function_seed_names:
        jax_programs = [getattr(seed_module, name) for name in jax_function_seed_names]
    else:
        jax_programs = None
    
    # Get parameter estimator seeds
    param_estimator_names = seed_programs.get('parameter_estimator_seeds', [])
    param_estimators = [getattr(seed_module, name) for name in param_estimator_names]

    # assert that we have exactly 2 of each
    if len(numpy_programs) != 2:
        raise ValueError("There must be exactly 2 numpy function seeds.")
    if jax_programs is not None and len(jax_programs) != 2:
        raise ValueError("There must be exactly 2 jax function seeds when provided.")
    if len(param_estimators) != 2:  
        raise ValueError("There must be exactly 2 parameter estimator seeds.")

    # Dynamically load data extraction function
    load_and_process_data_fn_path = config.pop('load_and_process_data_fn', None)
    if load_and_process_data_fn_path:
        # Parse module path and function name (e.g., 'experiments.orientation_tuning.data_parser.load_and_process_data')
        module_path, function_name = load_and_process_data_fn_path.rsplit('.', 1)
        data_module = importlib.import_module(module_path)
        load_and_process_data_fn = getattr(data_module, function_name)
    else:
        raise ValueError("load_and_process_data_fn must be specified in config.yaml")

    create_train_test_sample_split_fn_path = config.pop('create_train_test_sample_split_fn', None)
    if create_train_test_sample_split_fn_path:
        module_path, function_name = create_train_test_sample_split_fn_path.rsplit('.', 1)
        data_module = importlib.import_module(module_path)
        create_train_test_sample_split_fn = getattr(data_module, function_name)
    else:
        create_train_test_sample_split_fn = None

    create_train_test_trial_split_fn_path = config.pop('create_train_test_trial_split_fn', None)
    if create_train_test_trial_split_fn_path:
        module_path, function_name = create_train_test_trial_split_fn_path.rsplit('.', 1)
        data_module = importlib.import_module(module_path)
        create_train_test_trial_split_fn = getattr(data_module, function_name)
    else:
        create_train_test_trial_split_fn = None

    sample_loss_fn_path = config.pop('sample_loss_fn', None)
    if sample_loss_fn_path:
        module_path, function_name = sample_loss_fn_path.rsplit('.', 1)
        loss_module = importlib.import_module(module_path)
        sample_loss_fn = getattr(loss_module, function_name)
    else:
        sample_loss_fn = None

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
            use_image_feedback=params['use_image_feedback'],
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
            FAILED_PROGRAM_COST=params['FAILED_PROGRAM_COST'],
            tiny_lm_name=params['tiny_lm_name'],
            little_lm_name=params['little_lm_name'],
            large_lm_name=params['large_lm_name'],
            use_chat_mode=params.get('use_chat_mode', False),  # Default to legacy mode
            chat_token_limit=params.get('chat_token_limit', 50000),  # Max tokens per chat before auto-reset
            param_estimator_refinement_rounds=params.get('param_estimator_refinement_rounds', 0),
            numpy_programs=numpy_programs,
            jax_programs=jax_programs,
            param_estimators=param_estimators,
            load_and_process_data_fn=load_and_process_data_fn,
            create_train_test_sample_split_fn=create_train_test_sample_split_fn,
            create_train_test_trial_split_fn=create_train_test_trial_split_fn,
            training_sample_ratio=params.get('training_sample_ratio', 0.5),
            data_processing_params=data_processing_params,
            diagnostics_module=diagnostics_module,
            prompt_manager=prompt_manager,
            use_large_model_for_param_estimators=params.get('use_large_model_for_param_estimators', False),
            trial_batch_size=params.get('trial_batch_size', None),
            swear_words=params.get('swear_words'),
            sample_loss_fn=sample_loss_fn,
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Hypothesis Engine")
    parser.add_argument('--test_mode', action='store_true', help='Run in test mode with reduced iterations and time limit')
    parser.add_argument('--config', type=str, help='Path to experiment specific config file (relative to project root)', default="config.yaml")
    args = parser.parse_args()
    asyncio.run(_run_many(test_mode=args.test_mode, config_path=args.config))
