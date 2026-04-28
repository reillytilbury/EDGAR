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
from .io.output_dirs import make_output_dir, make_run_output_dir
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
        prompt_schemas = spec.prompt_schemas
        spawn(population, islands, mode, temperature,
              batch_size=spec.evolution["batch_size"],
              k_max=spec.llms["k_max"])

        await generate_model_code(population, prompt_schemas.model, llms.model, mode, temperature)
        await generate_param_est_code(population, prompt_schemas.param_est, llms.param_est)
        await translate_to_jax(population, prompt_schemas.jax, llms.jax)

        score(population, X_discover, X_eval, spec.scoring, spec.loss_fn, split="discover")

        islands = deduplicate(islands, population)
        islands = prune(islands, population, spec.evolution)
        islands = migrate(islands, population, spec.evolution, mode, temperature)
        census.append([set(island) for island in islands])

    score(population, X_validate, None, spec.scoring, spec.loss_fn, split="validate")

    population.save(str(output_dir / "population.jsonl"))
    save_island_census(census, str(output_dir / "census.json"))

    return output_dir



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run EDGAR")
    parser.add_argument("config", type=str, help="Path to task config.yaml")
    args = parser.parse_args()

    spec = TaskSpec.from_config(Path(args.config))
    asyncio.run(run(spec))
