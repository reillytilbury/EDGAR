"""
Main runner. Translates the pseudocode directly into real code.
"""
from __future__ import annotations

# JAX/XLA runtime guards — must be set before any import that loads JAX.
# Reduces GPU OOM during the spawn-subprocess scoring sweeps.
import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
_xla_flags = os.environ.get("XLA_FLAGS", "")
if "--xla_gpu_enable_command_buffer=" not in _xla_flags:
    os.environ["XLA_FLAGS"] = (_xla_flags + " --xla_gpu_enable_command_buffer=").strip()

import asyncio
import argparse
from pathlib import Path

from .io.task_spec import TaskSpec
from .io.logging import open_log, log_generation
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
    generate_models,
    generate_param_ests,
    translate_programs,
)
from .scoring.scoring import score


async def run(spec: TaskSpec, log_level: str = "compact") -> str:
    os.makedirs(spec.output_dir, exist_ok=True)
    spec.save_task_spec(spec.output_dir)
    log = open_log(spec.output_dir, log_level)

    X_discover, X_validate, X_eval = spec.load_data_fn(data_path=spec.io["data_path"], **spec.project_params)
    population = Population()
    islands = seed(population, spec.seed_programs, spec.evolution["n_islands"])
    await translate_programs(population, spec.prompt_schemas.jax, spec.llms["jax_translator_llm"])
    score(population, X_discover, X_eval, spec.scoring, spec.loss_fn, split="discover")

    census = []

    for gen in range(spec.evolution["n_generations"]):
        print(f"Generation {gen} / {spec.evolution['n_generations']}")
        mode, temperature, llms = spec.schedule(gen)
        spawn(population, islands, gen, mode, temperature,
              batch_size=spec.evolution["batch_size"],
              k_max=spec.llms["k_max"])

        await generate_models(population, spec.prompt_schemas.model, llms.model, mode, temperature,
                              spec=spec, data=X_discover[1]) # use test data of X_discover for plotting 
        await generate_param_ests(population, spec.prompt_schemas.param_est, llms.param_est, spec.flat_config)
        await translate_programs(population, spec.prompt_schemas.jax, llms.jax)

        score(population, X_discover, X_eval, spec.scoring, spec.loss_fn, split="discover")

        deduplicate(islands, population, spec.evolution)
        prune(islands, population, spec.evolution)
        migrate(islands, population, spec.evolution, temperature)
        census.append([set(island) for island in islands])
        log_generation(log, gen, population, islands, spec)

    population.prepare_validation_scoring(islands)
    score(population, X_validate, None, spec.scoring, spec.loss_fn, split="validate")

    population.save(os.path.join(spec.output_dir, "population.jsonl"))
    save_island_census(census, os.path.join(spec.output_dir, "island_census.jsonl"))
    log.file.close()

    return spec.output_dir



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run EDGAR")
    parser.add_argument("config", type=str, help="Path to task config.yaml or task_spec.yaml")
    args = parser.parse_args()

    from .io.config import Config
    path = Path(args.config)
    if path.name == "task_spec.yaml":
        config = Config.from_taskspec(path)
    else:
        config = Config.from_yaml(path)
    spec = TaskSpec.from_config(config)
    asyncio.run(run(spec))
