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
from .io.output_dirs import make_output_dir
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


async def run(spec: TaskSpec) -> Path:
    output_dir = make_output_dir(spec.io["save_path"])
    spec.save_task_spec(output_dir)

    X_discover, X_validate, X_eval = spec.load_data_fn(data_path=spec.io["data_path"], **spec.project_params)
    population = Population()
    islands = seed(population, spec.seed_programs, spec.evolution["n_islands"])
    await translate_to_jax(population, spec.prompt_schemas.jax, spec.llms["jax_translator_llm"])
    score(population, X_discover, X_eval, spec.scoring, spec.loss_fn, split="discover")

    census = []

    for i in range(spec.evolution["n_iterations"]):
        mode, temperature, llms = spec.schedule(i)
        spawn(population, islands, i, mode, temperature,
              batch_size=spec.evolution["batch_size"],
              k_max=spec.llms["k_max"])

        await generate_model_code(population, spec.prompt_schemas.model, llms.model, mode, temperature)
        await generate_param_est_code(population, spec.prompt_schemas.param_est, llms.param_est)
        await translate_to_jax(population, spec.prompt_schemas.jax, llms.jax)

        score(population, X_discover, X_eval, spec.scoring, spec.loss_fn, split="discover")

        deduplicate(islands, population, spec.evolution)
        prune(islands, population, spec.evolution)
        migrate(islands, population, spec.evolution, temperature)
        census.append([set(island) for island in islands])

    population.prepare_validation_scoring(islands)
    score(population, X_validate, None, spec.scoring, spec.loss_fn, split="validate")

    population.save(os.path.join(output_dir, "population.jsonl"))
    save_island_census(census, os.path.join(output_dir, "island_census.jsonl"))

    return output_dir



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run EDGAR")
    parser.add_argument("config", type=str, help="Path to task config.yaml")
    args = parser.parse_args()

    spec = TaskSpec.from_config(Path(args.config))
    asyncio.run(run(spec))
