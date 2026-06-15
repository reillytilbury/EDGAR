"""
Main runner. Translates the pseudocode directly into real code.
"""

# ruff: noqa: E402
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
import time
import traceback
import sys
from pathlib import Path

from .io.task_spec import TaskSpec
from .io.logging import open_log, log_generation, close_log, print_and_log, gen_banner
from .io.status import write_status
from .io.metrics import RunMetrics, timed
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
from .io.plotting import generate_program_fits


# Stage-timed aliases. Each wraps a pipeline call in stage_timer via timed() so
# the run loop below reads as clean pseudocode rather than nested `with` blocks
# (see PR #40 review). Behaviour is identical: same stage names, n_items progress
# counters, and quiet flags. Pass the per-stage item count as `n_items=...`.
t_seed = timed("seed", quiet=True)(seed)
t_translate_seeds = timed("translate_seeds")(translate_programs)
t_score_seeds = timed("score_seeds")(score)
t_fits_seeds = timed("generate_program_fits_seeds", quiet=True)(generate_program_fits)

t_spawn = timed("spawn", quiet=True)(spawn)
t_generate_models = timed("generate_models")(generate_models)
t_generate_param_ests = timed("generate_param_ests")(generate_param_ests)
t_translate_programs = timed("translate_programs")(translate_programs)
t_score = timed("score")(score)
t_fits = timed("generate_program_fits", quiet=True)(generate_program_fits)
t_deduplicate = timed("deduplicate", quiet=True)(deduplicate)
t_prune = timed("prune", quiet=True)(prune)
t_migrate = timed("migrate", quiet=True)(migrate)

t_score_validate = timed("score_validate")(score)


async def run(spec: TaskSpec, log_level: str = "compact") -> str:
    os.makedirs(spec.output_dir, exist_ok=True)
    spec.save(spec.output_dir)
    log = open_log(spec.output_dir, log_level)

    n_gens = spec.evolution["n_generations"]
    started_at = time.time()
    write_status(
        spec.output_dir, state="starting", n_gens=n_gens, started_at=started_at
    )
    pop_path = os.path.join(spec.output_dir, "population.jsonl")
    census_path = os.path.join(spec.output_dir, "island_census.jsonl")

    X_discover, X_validate, X_eval = spec.load_data_fn(
        data_path=spec.io["data_path"], **spec.project_params
    )
    retry_config = RetryConfig(**spec.llms.get("retry", {}))
    config = {**spec.flat_config, "retry_config": retry_config}

    n_islands = spec.evolution["n_islands"]
    batch_size = spec.evolution["batch_size"]

    population = Population()
    census = []

    with RunMetrics(
        output_dir=Path(spec.output_dir),
        run_log=log,
        n_gens=n_gens,
        started_at=started_at,
    ) as metrics:
        try:
            print_and_log(log, f"Run started. Output: {spec.output_dir}")
            print_and_log(
                log,
                f"Config: n_gens={n_gens} n_islands={n_islands} "
                f"batch_size={batch_size} "
                f"num_parents={spec.llms['num_parents']} "
                f"critical_pop={spec.evolution['critical_population_size']}",
            )

            # ── Seed phase ──
            metrics.start_generation(-1)
            islands = t_seed(population, spec.seed_programs, n_islands)
            await t_translate_seeds(
                population,
                spec.prompt_schemas.jax_model,
                spec.llms["jax_model_translator_llm"],
                retry_config=retry_config,
                max_tokens=config.get("max_tokens"),
                n_items=len(spec.seed_programs),
            )
            t_score_seeds(
                population,
                X_discover,
                X_eval,
                spec.scoring,
                spec.loss_fn,
                split="discover",
                n_items=len(spec.seed_programs),
            )
            t_fits_seeds(spec, X_discover[1], population)
            population.save(pop_path)
            save_island_census(census, census_path)
            metrics.finish_generation()
            write_status(
                spec.output_dir,
                state="running",
                n_gens=n_gens,
                current_gen=-1,
                started_at=started_at,
                current_stage=None,
                last_metrics=metrics._gen_rows[-1] if metrics._gen_rows else None,
            )

            # ── Evolution loop ──
            for gen in range(n_gens):
                metrics.start_generation(gen)
                mode, temperature, llms = spec.schedule(gen)
                n_spawn = n_islands * batch_size
                gen_banner(log, gen, n_gens, mode, temperature, llms, n_spawn=n_spawn)

                t_spawn(
                    population,
                    islands,
                    gen,
                    mode,
                    temperature,
                    batch_size=batch_size,
                    num_parents=spec.llms["num_parents"],
                    rng=spec.rng,
                )

                await t_generate_models(
                    population,
                    spec.prompt_schemas.model,
                    llms.model[gen % len(llms.model)]
                    if isinstance(llms.model, list)
                    else llms.model,
                    mode,
                    temperature,
                    config=config,
                    spec=spec,
                    data=X_discover[1],
                    n_items=n_spawn,
                )
                await t_generate_param_ests(
                    population,
                    spec.prompt_schemas.param_est,
                    llms.param_est,
                    config,
                    n_items=n_spawn,
                )
                await t_translate_programs(
                    population,
                    spec.prompt_schemas.jax_model,
                    llms.model_jax,
                    retry_config=retry_config,
                    max_tokens=config.get("max_tokens"),
                    n_items=n_spawn,
                )
                t_score(
                    population,
                    X_discover,
                    X_eval,
                    spec.scoring,
                    spec.loss_fn,
                    split="discover",
                    n_items=n_spawn,
                )
                t_fits(spec, X_discover[1], population)

                t_deduplicate(islands, population, spec.evolution)
                t_prune(islands, population, spec.evolution)
                t_migrate(
                    islands,
                    population,
                    spec.evolution,
                    temperature,
                    rng=spec.rng,
                )
                census.append([set(island) for island in islands])

                log_generation(log, gen, population, islands, spec, metrics=metrics)

                # Per-generation persistence for the live dashboard. Atomic writes
                # protect a polling reader from observing torn files.
                population.save(pop_path)
                save_island_census(census, census_path)
                metrics.finish_generation()
                write_status(
                    spec.output_dir,
                    state="running",
                    n_gens=n_gens,
                    current_gen=gen,
                    started_at=started_at,
                    current_stage=None,
                    last_metrics=metrics._gen_rows[-1] if metrics._gen_rows else None,
                )

            # ── Validation ──
            metrics.start_generation(n_gens)
            population.prepare_validation_scoring(islands)
            t_score_validate(
                population,
                X_validate,
                None,
                spec.scoring,
                spec.loss_fn,
                split="validate",
            )
            rank(population)
            metrics.finish_generation()

            print_and_log(
                log, f"***** Run complete. Output directory: {spec.output_dir} *****"
            )

        finally:  # runs whether or not an exception is raised
            exc_info = sys.exc_info()
            failed = exc_info[0] is not None
            if failed:
                print_and_log(
                    log,
                    f"***** Run failed with exception:\n"
                    f"{''.join(traceback.format_exception(*exc_info))}"
                    f"***** Output directory: {spec.output_dir} *****",
                )
            population.save(pop_path)
            save_island_census(census, census_path)
            write_status(
                spec.output_dir,
                state="failed" if failed else "complete",
                n_gens=n_gens,
                current_gen=(len(census) - 1) if census else None,
                started_at=started_at,
                error=(f"{exc_info[0].__name__}: {exc_info[1]}" if failed else None),
                current_stage=None,
                last_metrics=metrics._gen_rows[-1] if metrics._gen_rows else None,
            )
            close_log(log)

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
