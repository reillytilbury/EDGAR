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
from .io.status import write_status, read_status
from .io.metrics import RunMetrics, stage_timer, read_metrics
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


async def run(
    spec: TaskSpec,
    log_level: str = "compact",
    resume_from: str | Path | None = None,
) -> str:
    """Run an EDGAR experiment, optionally resuming a crashed run.

    Args:
        spec: TaskSpec built from a config or task_spec.yaml.
        log_level: Logging verbosity (compact/code/prompts).
        resume_from: Path to a run directory (containing population.jsonl +
            island_census.jsonl + task_spec.yaml). When set:
              - The seed phase is skipped; state is loaded from disk.
              - Output is written back into ``resume_from`` (spec is restamped).
              - The loop continues at ``len(census)``.
              - run.log is opened in append mode with a RESUMED banner.
              - started_at is preserved from the original status.json so total
                wall time across resumes is recoverable from gen timings.

            Caveats:
              - ``spec.rng`` state is lost. With a fixed run.random_seed the
                resumed spawning/migration draws differ from a continuous run.
                LLM responses are non-deterministic anyway.
              - The original task_spec.yaml is reused as-is (chmod read-only).
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
                with stage_timer(metrics, "seed", quiet=True):
                    islands = seed(population, spec.seed_programs, n_islands)
                with stage_timer(
                    metrics, "translate_seeds", n_items=len(spec.seed_programs)
                ):
                    await translate_programs(
                        population,
                        spec.prompt_schemas.jax_model,
                        spec.llms["jax_model_translator_llm"],
                        retry_config=retry_config,
                        max_tokens=config.get("max_tokens"),
                    )
                with stage_timer(
                    metrics, "score_seeds", n_items=len(spec.seed_programs)
                ):
                    score(
                        population,
                        X_discover,
                        X_eval,
                        spec.scoring,
                        spec.loss_fn,
                        split="discover",
                    )
                with stage_timer(metrics, "generate_program_fits_seeds", quiet=True):
                    generate_program_fits(spec, X_discover[1], population)
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

                with stage_timer(metrics, "spawn", quiet=True):
                    spawn(
                        population,
                        islands,
                        gen,
                        mode,
                        temperature,
                        batch_size=batch_size,
                        num_parents=spec.llms["num_parents"],
                        rng=spec.rng,
                    )

                with stage_timer(metrics, "generate_models", n_items=n_spawn):
                    await generate_models(
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
                    )
                with stage_timer(metrics, "generate_param_ests", n_items=n_spawn):
                    await generate_param_ests(
                        population,
                        spec.prompt_schemas.param_est,
                        llms.param_est,
                        config,
                    )
                with stage_timer(metrics, "translate_programs", n_items=n_spawn):
                    await translate_programs(
                        population,
                        spec.prompt_schemas.jax_model,
                        llms.model_jax,
                        retry_config=retry_config,
                        max_tokens=config.get("max_tokens"),
                    )
                with stage_timer(metrics, "score", n_items=n_spawn):
                    score(
                        population,
                        X_discover,
                        X_eval,
                        spec.scoring,
                        spec.loss_fn,
                        split="discover",
                    )
                with stage_timer(metrics, "generate_program_fits", quiet=True):
                    generate_program_fits(spec, X_discover[1], population)

                with stage_timer(metrics, "deduplicate", quiet=True):
                    deduplicate(islands, population, spec.evolution)
                with stage_timer(metrics, "prune", quiet=True):
                    prune(islands, population, spec.evolution)
                with stage_timer(metrics, "migrate", quiet=True):
                    migrate(
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
            with stage_timer(metrics, "score_validate"):
                score(
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
