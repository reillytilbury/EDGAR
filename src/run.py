"""
Main runner. Translates the pseudocode directly into real code.
"""
from __future__ import annotations

import asyncio
import argparse
from pathlib import Path

from .io.task_spec import TaskSpec
from .evolution.population import Population
from .evolution.island import (
    seed,
    spawn,
    deduplicate,
    prune,
    migrate,
    save_island_census,
)
from .llm.generate import (
    generate_model_code,
    generate_param_est_code,
    translate_to_jax,
)
from .scoring.scoring import score
from .paths import create_run_paths


async def run(spec: TaskSpec) -> Path:
    paths = create_run_paths()
    spec.save_record(paths.full_dir)

    X_discover, X_validate, X_eval = spec.load_data_fn(**spec.data_processing_params)
    population = Population()
    islands = seed(population, spec.seed_programs, spec.evolution["n_islands"])
    score(population, islands, X_discover, X_eval, spec.scoring)

    census = []

    for i in range(spec.evolution["n_iterations"]):
        mode, temperature, llms = spec.schedule(i)
        prompt_schemas = spec.prompt_schemas
        spawn(population, islands, mode, temperature,
              batch_size=spec.evolution["batch_size"],
              k_max=spec.llms["k_max"])

        await generate_model_code(population, prompt_schemas.model, llms.model, mode, temperature)
        await generate_param_est_code(population, prompt_schemas.param_est, llms.param_est)
        await translate_to_jax(population, prompt_schemas.jax, llms.jax)

        score(population, islands, X_discover, X_eval, spec.scoring)

        islands = deduplicate(islands, population)
        islands = prune(islands, population, spec.evolution)
        islands = migrate(islands, population, spec.evolution, mode, temperature)
        census.append([set(island) for island in islands])

    score(population, islands, X_validate, X_eval, spec.scoring)

    population.save(str(paths.full_dir / "population.jsonl"))
    save_island_census(census, str(paths.full_dir / "census.json"))

    return paths.full_dir



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run EDGAR")
    parser.add_argument("config", type=str, help="Path to task config.yaml")
    args = parser.parse_args()

    spec = TaskSpec.from_config(Path(args.config))
    asyncio.run(run(spec))
