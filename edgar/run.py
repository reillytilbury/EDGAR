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
import traceback
import sys
from pathlib import Path

from .io.task_spec import TaskSpec
from .io.logging import open_log, log_generation, close_log, print_and_log
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
from .io.config import RetryConfig
from .scoring.scoring import rank, score
from .monitoring.family_tree import write_family_tree


async def run(spec: TaskSpec, log_level: str = "compact") -> str:
    os.makedirs(spec.output_dir, exist_ok=True)
    spec.save(spec.output_dir)
    log = open_log(spec.output_dir, log_level)

    X_discover, X_validate, X_eval = spec.load_data_fn(
        data_path=spec.io["data_path"], **spec.project_params
    )
    retry_config = RetryConfig(**spec.llms.get("retry", {}))
    config = {**spec.flat_config, "retry_config": retry_config}

    population = Population()
    census = []

    try:
        islands = seed(population, spec.seed_programs, spec.evolution["n_islands"])
        await translate_programs(
            population,
            spec.prompt_schemas.jax_model,
            spec.llms["jax_model_translator_llm"],
            retry_config=retry_config,
            max_tokens=config.get("max_tokens"),
        )
        score(population, X_discover, X_eval, spec.scoring, spec.loss_fn, split="discover")

        for gen in range(spec.evolution["n_generations"]):
            print_and_log(log, f"Generation {gen} / {spec.evolution['n_generations']}")
            mode, temperature, llms = spec.schedule(gen)
            spawn(
                population,
                islands,
                gen,
                mode,
                temperature,
                batch_size=spec.evolution["batch_size"],
                num_parents=spec.llms["num_parents"],
                rng=spec.rng,
            )

            await generate_models(
                population,
                spec.prompt_schemas.model,
                llms.model,
                mode,
                temperature,
                config=config,
                spec=spec,
                data=X_discover[1],
            )  # use test data of X_discover for plotting
            await generate_param_ests(
                population, spec.prompt_schemas.param_est, llms.param_est, config,
            )
            await translate_programs(
                population,
                spec.prompt_schemas.jax_model,
                llms.model_jax,
                retry_config=retry_config,
            )

            score(
                population, X_discover, X_eval, spec.scoring, spec.loss_fn, split="discover"
            )

            deduplicate(islands, population, spec.evolution)
            prune(islands, population, spec.evolution)
            migrate(islands, population, spec.evolution, temperature, rng=spec.rng)
            census.append([set(island) for island in islands])
            log_generation(log, gen, population, islands, spec)

        population.prepare_validation_scoring(islands)
        score(population, X_validate, None, spec.scoring, spec.loss_fn, split="validate")
        rank(population)

        print_and_log(log, f"***** Run complete. Output directory: {spec.output_dir} *****")

    finally: #runs whether or not an exception is raised, ensuring that results are saved
        exc_info = sys.exc_info()
        if exc_info[0] is not None:
            print_and_log(log, f"***** Run failed with exception:\n{''.join(traceback.format_exception(*exc_info))}***** Output directory: {spec.output_dir} *****")
        population.save(os.path.join(spec.output_dir, "population.jsonl"))
        save_island_census(census, os.path.join(spec.output_dir, "island_census.jsonl"))
        close_log(log)
        try: 
            write_family_tree(population, census, spec.output_dir, param_penalty_weight=spec.scoring.get("param_penalty_weight"))
        except Exception:
            pass  # don't mask the original exception

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run EDGAR")
    parser.add_argument(
        "config", type=str, help="Path to task config.yaml or task_spec.yaml"
    )
    args = parser.parse_args()

    from .io.config import Config

    path = Path(args.config)
    if path.name == "task_spec.yaml":
        config = Config.from_taskspec(path)
    else:
        config = Config.from_yaml(path)
    spec = TaskSpec.from_config(config)
    asyncio.run(run(spec))
