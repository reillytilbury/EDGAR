import asyncio
import yaml
from pathlib import Path
import importlib
import os, argparse
from src import hypothesis_engine 
from src.diagnostics_manager import load_diagnostics

async def _run_many(test_mode: bool = False, config_dir: str = "config"):
    # Resolve config directory (relative to project root)
    project_root = Path(__file__).parent
    config_path = project_root / config_dir
    
    if not config_path.exists():
        raise ValueError(f"Config directory not found: {config_path}")
    
    print(f"Using config directory: {config_path}")
    
    # Load experiment configuration
    experiment_config_path = config_path / "experiment.yaml"
    with open(experiment_config_path) as f:
        experiment_config = yaml.safe_load(f)
    
    # Extract experiment parameters
    params = experiment_config.get('experiment_params', {})
    seed_programs = experiment_config.get('seed_programs', {})
    
    # Load experiment-specific diagnostics (returns None if not configured or file missing)
    diagnostics_path = experiment_config.get('diagnostics_path', None)
    diagnostics_module = load_diagnostics(diagnostics_path)

    # Dynamically load seed programs module
    module_path = seed_programs.get('module')
    if not module_path:
        raise ValueError("seed_programs.module not specified in experiment.yaml")
    
    seed_module = importlib.import_module(module_path)
    
    # Get function seeds
    function_seed_names = seed_programs.get('function_seeds', [])
    numpy_programs = [getattr(seed_module, name) for name in function_seed_names]

    jax_function_seed_names = seed_programs.get('jax_function_seeds', [])
    jax_programs = [getattr(seed_module, name) for name in jax_function_seed_names]
    
    # Get parameter estimator seeds
    param_estimator_names = seed_programs.get('parameter_estimator_seeds', [])
    param_estimators = [getattr(seed_module, name) for name in param_estimator_names]

    # assert that we have exactly 2 of each
    if len(numpy_programs) != 2:
        raise ValueError("There must be exactly 2 numpy function seeds.")
    if len(jax_programs) != 2:
        raise ValueError("There must be exactly 2 jax function seeds.")
    if len(param_estimators) != 2:  
        raise ValueError("There must be exactly 2 parameter estimator seeds.")

    data_config_path = config_path / "data.yaml"
    with open(data_config_path) as f:
        data_config = yaml.safe_load(f)

    # Path to prompts.yaml in the config directory
    prompts_config_path = config_path / "prompts.yaml"
    if not prompts_config_path.exists():
        raise ValueError(f"prompts.yaml not found in config directory: {prompts_config_path}")

    # Dynamically load data extraction function
    load_and_process_data_fn_path = data_config.pop('load_and_process_data_fn', None)
    if load_and_process_data_fn_path:
        # Parse module path and function name (e.g., 'experiments.orientation_tuning.data_parser.load_and_process_data')
        module_path, function_name = load_and_process_data_fn_path.rsplit('.', 1)
        data_module = importlib.import_module(module_path)
        load_and_process_data_fn = getattr(data_module, function_name)
    else:
        raise ValueError("load_and_process_data_fn must be specified in data.yaml")
    
    # Extract predictor names from config (for multi-predictor support)
    # These stay in data_config for the data loading function to use
    predictors_config = data_config.get('predictors', [])
    if predictors_config:
        data_config['predictor_names'] = [p['name'] for p in predictors_config]

    if test_mode:
        params['num_runs'] = 1
        params['n_iterations'] = 1
        params['time_limit'] = 10  # seconds
        params['n_islands'] = 2
        params['k_max'] = 2
        params['batch_size'] = 2
        params['max_iter'] = 100
        params['exploration_topology'] = [1, 0]
        params['exploitation_topology'] = [1, 0]

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
            FAILED_PROGRAM_COST=params['FAILED_PROGRAM_COST'],
            tiny_lm_name=params['tiny_lm_name'],
            little_lm_name=params['little_lm_name'],
            large_lm_name=params['large_lm_name'],
            numpy_programs=numpy_programs,
            jax_programs=jax_programs,
            param_estimators=param_estimators,
            load_and_process_data_fn=load_and_process_data_fn,
            data_config=data_config,
            diagnostics_module=diagnostics_module,
            prompts_config_path=prompts_config_path,
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Hypothesis Engine")
    parser.add_argument('--test_mode', action='store_true', help='Run in test mode with reduced iterations and time limit')
    parser.add_argument('--config', type=str, help='Path to experiment specific config directory')
    args = parser.parse_args()
    asyncio.run(_run_many(test_mode=args.test_mode, config_dir=args.config))