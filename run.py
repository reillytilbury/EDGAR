import asyncio
import yaml
from pathlib import Path
import hypothesis_engine 

async def _run_many():
    # Load experiment configuration
    config_path = Path(__file__).parent / "config" / "experiment.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Extract experiment parameters
    params = config.get('experiment_params', {})
    
    for i in range(4):
        print("running with standard params")
        await hypothesis_engine.main(
            n_iterations=params['n_iterations'],
            time_limit=params['time_limit'],
            use_image_feedback=params['use_image_feedback'],
            use_large_every=params['use_large_every'],
            param_penalty_weight=params['param_penalty_weight'],
            exploration_topology=params['exploration_topology'],
            exploit_point=params['exploit_point'],
        )

if __name__ == "__main__":
    asyncio.run(_run_many())