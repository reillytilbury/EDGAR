"""Orchestrates the entire EDGAR evolutionary experiment.

This module serves as the main entry point for running an EDGAR experiment.
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
from .io.status import write_status, read_status
from .io.metrics import RunMetrics, timed, read_metrics
from .evolution.population import Population
from .evolution.island import (
    seed,
    spawn,
    deduplicate,
    prune,
    migrate,
    save_island_census,
    load_island_census,
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


async def run(
    spec: TaskSpec,
    log_level: str = "compact",
    resume_from: str | Path | None = None,
) -> None:
    """Orchestrates and executes the entire EDGAR evolutionary experiment.

    This function manages the full lifecycle of an EDGAR run, from initialization
    and data loading to the generational evolutionary loop, LLM interactions, scoring, and final
    validation. It ensures logging, status tracking, and persistence of results for
    real-time dashboard monitoring and post-hoc analysis.

    The core algorithm follows these steps:
    1.  **Initialization**: Sets up the run environment, creates the output directory, and
        saves the `TaskSpec` for reproducibility. Initializes structured logging and real-time
        status tracking.
    2.  **Data Loading**: Loads the scientific problem data into `X_discover`, `X_validate`,
        and `X_eval` splits using the `spec.load_data_fn`.
    3.  **Seed Phase**:
        *   Seed programs are loaded and optionally translated.
        *   Seed programs are scored to establish a baseline.
    4.  **Generational Loop**: For each generation:
        *   New program variants are `spawn`ed from the current population.
        *   LLMs generate new model architectures (`generate_models`) and parameter
            estimation logic (`generate_param_ests`).
        *   Programs are `translate`d to JAX.
        *   Programs are `score`ed and `rank`ed.
        *   The population is `deduplicate`d, `prune`d, and survivors `migrate` between islands.
    5.  **Finalization**:
        *   The final population and island census are saved.
        *   Programs are `rank`ed based on their final validation losses.
    6.  **Error Handling**: A `finally` block ensures that the `Population`, `census`, and
        `status.json` are always saved, even if an exception occurs during the run. Any exceptions
        are captured, logged with their traceback, and the run status is updated to 'failed'.

    Resuming Runs:
        If `resume_from` is provided:
              - The population and census are loaded from the specified directory.
              - The run starts from the next generation (max(population.birth.gen) + 1).
              - run.log is opened in append mode with a RESUMED banner.
              - started_at is preserved from the original status.json so total
                wall time across resumes is recoverable from gen timings.

            Caveats:
              - ``spec.rng`` state is lost. With a fixed run.random_seed the
                resumed spawning/migration draws differ from a continuous run.
                LLM responses are non-deterministic anyway.
              - The original task_spec.yaml is reused as-is (chmod read-only).

    Args:
        spec: A `TaskSpec` object containing all configuration, data, and callable functions
            required for the experiment.
        log_level: The verbosity level for logging messages to the console and `run.log`.
            Can be 'compact', 'code', or 'prompts'.
        resume_from: Optional path to a previous run's output directory to resume from.

    Returns:
        None. The function's primary effects are side effects: creating output files, updating
        status, and logging.
    """
    resume = resume_from is not None
    if resume:
        resume_from = Path(resume_from).resolve()
        _prepare_resume(spec, resume_from)
    os.makedirs(spec.output_dir, exist_ok=True)
    if not resume:
        spec.save(spec.output_dir)
    log = open_log(spec.output_dir, log_level, append=resume)

    n_gens = spec.evolution["n_generations"]
    pop_path = os.path.join(spec.output_dir, "population.jsonl")
    census_path = os.path.join(spec.output_dir, "island_census.jsonl")

    if resume:
        prior_status = read_status(spec.output_dir) or {}
        started_at = float(prior_status.get("started_at", time.time()))
    else:
        started_at = time.time()
        write_status(
            spec.output_dir, state="starting", n_gens=n_gens, started_at=started_at
        )

    X_discover, X_validate, X_eval = spec.load_data_fn(
        data_path=spec.io["data_path"], **spec.project_params
    )
    retry_config = RetryConfig(**spec.llms.get("retry", {}))
    config = {**spec.flat_config, "retry_config": retry_config}

    n_islands = spec.evolution["n_islands"]
    batch_size = spec.evolution["batch_size"]

    if resume:
        population = Population.load(pop_path)
        census = load_island_census(census_path)
        _validate_resume_state(population, census, n_gens)
        n_dropped = _drop_trailing_unscored(population)
        islands = [set(s) for s in census[-1]]
        start_gen = len(census)
    else:
        population = Population()
        census = []
        islands = None  # populated in seed phase below
        start_gen = 0
        n_dropped = 0

    with RunMetrics(
        output_dir=Path(spec.output_dir),
        run_log=log,
        n_gens=n_gens,
        started_at=started_at,
    ) as metrics:
        try:
            if resume:
                # Restore historical per-gen metrics so the dashboard's
                # last_metrics reflects the full timeline, not just the resume.
                metrics._gen_rows.extend(read_metrics(Path(spec.output_dir)))
                print_and_log(log, f"Run RESUMED from {spec.output_dir}")
                if n_dropped:
                    print_and_log(
                        log,
                        f"Resume: dropped {n_dropped} trailing unscored programs "
                        f"(stale spawn shells from the crashed generation).",
                    )
                print_and_log(
                    log,
                    f"Loaded population={len(population)} programs, "
                    f"census={len(census)} completed gens. Resuming at gen={start_gen}.",
                )
            else:
                print_and_log(log, f"Run started. Output: {spec.output_dir}")
            print_and_log(
                log,
                f"Config: n_gens={n_gens} n_islands={n_islands} "
                f"batch_size={batch_size} "
                f"num_parents={spec.llms['num_parents']} "
                f"critical_pop={spec.evolution['critical_population_size']}",
            )

            if not resume:
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
            for gen in range(start_gen, n_gens):
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


def _prepare_resume(spec: TaskSpec, run_dir: Path) -> None:
    """Restamp ``spec`` so its output_dir resolves to ``run_dir``.

    ``spec.output_dir`` is computed as ``os.path.join(io["save_path"],
    creation_timestamp)``. Setting save_path to "" and creation_timestamp to
    the absolute run_dir produces output_dir == str(run_dir).
    """
    if not run_dir.exists():
        raise FileNotFoundError(f"resume_from directory does not exist: {run_dir}")
    if not (run_dir / "task_spec.yaml").exists():
        raise FileNotFoundError(
            f"resume_from is missing task_spec.yaml: {run_dir}. "
            "This doesn't look like an EDGAR run directory."
        )
    spec.io["save_path"] = ""
    spec.creation_timestamp = str(run_dir)


def _drop_trailing_unscored(population: Population) -> int:
    """Drop programs from the end of population whose discover.final loss is None.

    These are stale spawn shells from a crashed generation: spawn() added them
    but score() never got the chance to set their loss. Dropping them keeps
    population.jsonl free of ghost entries after resume. Pruned-but-completed
    programs from earlier gens are preserved because they have a non-None
    final loss (or inf if scoring failed cleanly).

    Returns the number of programs dropped.
    """
    progs = population._programs
    n_before = len(progs)
    while progs and progs[-1].program_losses.discover.final is None:
        progs.pop()
    return n_before - len(progs)


def _validate_resume_state(population: Population, census: list, n_gens: int) -> None:
    """Refuse to resume if on-disk state is incomplete, inconsistent, or done."""
    if len(population) == 0:
        raise ValueError(
            "Cannot resume: population.jsonl is empty. The original run did "
            "not complete its seed phase. Re-run from scratch instead."
        )
    if len(census) == 0:
        raise ValueError(
            "Cannot resume: island_census.jsonl has no completed generations "
            "(seed phase saves an empty census). The original run crashed "
            "before gen 0 finished. Re-run from scratch instead."
        )
    if len(census) >= n_gens:
        raise ValueError(
            f"Cannot resume: census already has {len(census)} completed gens "
            f"(target n_generations={n_gens}). The original run reached the "
            "validation phase; nothing to resume."
        )
    max_idx = max((idx for island in census[-1] for idx in island), default=-1)
    if max_idx >= len(population):
        raise ValueError(
            f"Corrupt run dir: census references program idx {max_idx} but "
            f"population only has {len(population)} programs."
        )


if __name__ == "__main__":
    """Main execution block for running EDGAR from the command line.

    This block parses command-line arguments to obtain the configuration file path.
    It then loads the `Config` (either from a `config.yaml` or a previously saved
    `task_spec.yaml`), initializes a `TaskSpec` object from this configuration,
    and finally runs the asynchronous EDGAR experiment using `asyncio.run()`.
    """
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
