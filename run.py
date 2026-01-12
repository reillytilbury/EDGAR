import asyncio
import yaml
from pathlib import Path
import importlib
from src import hypothesis_engine 

async def _run_many():
    # Load experiment configuration
    experiment_config_path = Path(__file__).parent / "config" / "experiment.yaml"
    with open(experiment_config_path) as f:
        experiment_config = yaml.safe_load(f)
    
    # Extract experiment parameters
    params = experiment_config.get('experiment_params', {})
    seed_programs = experiment_config.get('seed_programs', {})    

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

    data_config_path = Path(__file__).parent / "config" / "data.yaml"
    with open(data_config_path) as f:
        data_config = yaml.safe_load(f)

    params['data_path'] = data_config.get('data_path', '')

    for i in range(4):
        print("running with standard params")
        await hypothesis_engine.hypothesis_engine(
            n_iterations=params['n_iterations'],
            time_limit=params['time_limit'],
            use_image_feedback=params['use_image_feedback'],
            use_large_every=params['use_large_every'],
            param_penalty_weight=params['param_penalty_weight'],
            exploration_topology=params['exploration_topology'],
            exploit_point=params['exploit_point'],
            data_path=params['data_path'], 
            k_max=params['k_max'],
            n_islands=params['n_islands'],
            batch_size=params['batch_size'],
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
            param_estimators=param_estimators
        )

if __name__ == "__main__":
    asyncio.run(_run_many())