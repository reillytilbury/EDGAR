#%%
"""Cell-based walkthrough of an EDGAR run on `orientation_tuning`.

This script breaks `edgar.run.run(spec)` into ~24 inspectable `# %%` cells so
you can step through the pipeline, see intermediate state, and understand each
transition without the full evolution loop being a single black-box call.

Two modes of use — same source file:

    # End-to-end as a regular script (will hit the Anthropic API):
    python tutorials/walkthrough_orientation_tuning.py

    # Cell-by-cell in Cursor / VSCode "Interactive Python":
    # Open this file, place your cursor inside a cell, hit Shift+Enter.
    # State persists between cells, so you can inspect `population`,
    # `islands`, etc. at any checkpoint.

Prerequisites
─────────────
- `conda activate edgar` (or use the explicit interpreter path, if using uv this is `.venv/bin/python` in the repo root).
- `ANTHROPIC_API_KEY` set in `.env` at the repo root. This tutorial uses
  `claude-haiku-4-5` for all three LLM roles. Gemini can be used through instead setting `GOOGLE_API_KEY` and choosing gemini models such as `gemini-2.5-flash-lite`
- Data file at `data/gratings_drifting_GT1_2019_04_12_1.npy`.

Tutorial-time overrides applied in cell A6 keep this fast (~3-5 min, ~$0.01):
    n_generations=1, n_islands=2, batch_size=2, all LLM roles → claude-haiku-4-5,
    gradient_descent.max_iter=50.
"""

# ─────────────────────────────────────────────────────────────────────────
# Module-level preamble (runs even in spawn children — no side effects)
# ─────────────────────────────────────────────────────────────────────────
import asyncio
import os
import sys
from pathlib import Path
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

# Cursor / Jupyter cells run inside an already-active asyncio event loop, so the
# `asyncio.run(...)` calls in later cells would raise "cannot be called from a
# running event loop". `nest_asyncio.apply()` patches asyncio to allow nesting.
# In CLI script mode no loop is active and this is a harmless no-op. The import
# is wrapped because nest_asyncio is only needed for the interactive use case;
# someone running this as a pure script doesn't need to install it.
try:
    import nest_asyncio
    nest_asyncio.apply()
except ImportError:
    pass

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)  # so relative paths in the config (e.g. 'data/...') resolve

from dotenv import load_dotenv  # noqa: E402

load_dotenv(REPO_ROOT / ".env")

# EDGAR imports. Grouped by layer so the dependency direction is visible:
# config → spec → state → llm → evolution → scoring → io.
from edgar.io.config import Config, RetryConfig  # noqa: E402
from edgar.io.task_spec import TaskSpec  # noqa: E402
from edgar.evolution.population import Population  # noqa: E402
from edgar.evolution.island import (  # noqa: E402
    seed,
    spawn,
    deduplicate,
    prune,
    migrate,
    save_island_census,
)
from edgar.llm.generate import (  # noqa: E402
    generate_models,
    generate_param_ests,
    translate_programs,
)
from edgar.scoring.scoring import rank, score  # noqa: E402

# %%
# ─────────────────────────────────────────────────────────────────────────
# Cell A2: Sanity check — environment and API key
# ─────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"REPO_ROOT = {REPO_ROOT}")
    print(f"ANTHROPIC_API_KEY set: {bool(os.getenv('ANTHROPIC_API_KEY'))}")


# %%
# ─────────────────────────────────────────────────────────────────────────
# Cell A3: Load the project config
# ─────────────────────────────────────────────────────────────────────────
# Config.from_yaml does three things:
#   1. Read projects/<task>/config.yaml.
#   2. Merge it on top of projects/config_default.yaml (project overrides defaults).
#   3. Merge prompts in projects/<task>/prompts.yaml on top of prompt_defaults.yaml.
# The result is a fully-validated, fully-defaulted Config object — no surprises
# downstream from missing keys.
if __name__ == "__main__":
    config_path = REPO_ROOT / "projects" / "orientation_tuning" / "config.yaml"
    config = Config.from_yaml(config_path)

    print(f"task_name:       {config.task_name}")
    print(f"data_path:       {config.io.data_path}")
    print(f"n_generations:   {config.evolution.n_generations}  (default)")
    print(f"n_islands:       {config.evolution.n_islands}  (default)")
    print(f"model_llm:       {config.llms.model_llm}")
    print(f"project_params:  {config.project_params}")


# %%
# ─────────────────────────────────────────────────────────────────────────
# Cell A4: Build a TaskSpec from the Config
# ─────────────────────────────────────────────────────────────────────────
# TaskSpec is the "frozen bundle" that has both the config dicts AND the
# resolved project callables (load_data_fn, loss_fn, plot_fn) AND the seed
# programs loaded from disk. Everything past this point reads from `spec`.
# See edgar/io/task_spec.py for per-field docs.
if __name__ == "__main__":
    spec = TaskSpec.from_config(config)

    print(f"task_name:       {spec.task_name}")
    print(f"git_sha:         {spec.git_sha[:12]}...   (dirty: {spec.git_dirty})")
    print(f"seed programs:   {len(spec.seed_programs)}")
    print(f"output_dir:      {spec.output_dir}")


# %%
# ─────────────────────────────────────────────────────────────────────────
# Cell A5: Inspect the spec's generation schedule
# ─────────────────────────────────────────────────────────────────────────
# `spec.schedule(generation)` is what `run.py` calls every generation to pick
# the current mode ("explore" / "exploit"), temperature, and per-role LLMs.
# We probe it at gen=0 and the last generation so the range is concrete.
if __name__ == "__main__":
    n_gen = spec.evolution["n_generations"]
    mode0, temp0, llms0 = spec.schedule(0)
    mode_last, temp_last, llms_last = spec.schedule(n_gen - 1)

    print(f"schedule(0):        mode={mode0:<8}  temp={temp0:.4f}")
    print(f"schedule({n_gen - 1}):       mode={mode_last:<8}  temp={temp_last:.4f}")
    print()
    print("Temperature emitted is the Gemini-scale [1.37, 2.0]. For Anthropic,")
    print("`call_llm` rescales by /2 at the call boundary (see edgar/llm/llm_calling.py).")


# %%
# ─────────────────────────────────────────────────────────────────────────
# Cell A6: Tutorial-time overrides — make it fast and Anthropic-only
# ─────────────────────────────────────────────────────────────────────────
# TaskSpec stores config sections as plain mutable dicts so you can poke at
# them in-flight. We override the production defaults to:
#   - 1 generation (instead of 12)
#   - 2 islands × 2 batch (instead of 8 × 6) → 4 LLM calls per role per gen
#   - all three LLM roles → claude-haiku-4-5 (cheap; Gemini quota is small)
#   - 50 GD iterations (instead of 1000) so scoring is fast on real data
if __name__ == "__main__":
    print("Before overrides:")
    print(f"  n_generations={spec.evolution['n_generations']}, "
          f"n_islands={spec.evolution['n_islands']}, "
          f"batch_size={spec.evolution['batch_size']}")
    print(f"  model_llm={spec.llms['model_llm']}")
    print(f"  gradient_descent.max_iter={spec.scoring['gradient_descent']['max_iter']}")

    spec.evolution["n_generations"] = 1
    spec.evolution["n_islands"] = 2
    spec.evolution["batch_size"] = 2
    spec.evolution["topology"] = [1, 0]  # must match n_islands; here a 2-cycle
    spec.evolution["n_migrants"] = 1
    spec.evolution["critical_population_size"] = 6

    spec.llms["model_llm"] = "claude-haiku-4-5"
    spec.llms["param_est_llm"] = "claude-haiku-4-5"
    spec.llms["jax_model_translator_llm"] = "claude-haiku-4-5"

    spec.scoring["timeout_s"] = 60.0
    spec.scoring["gradient_descent"]["max_iter"] = 50

    print("\nAfter overrides:")
    print(f"  n_generations={spec.evolution['n_generations']}, "
          f"n_islands={spec.evolution['n_islands']}, "
          f"batch_size={spec.evolution['batch_size']}")
    print(f"  model_llm={spec.llms['model_llm']}")
    print(f"  gradient_descent.max_iter={spec.scoring['gradient_descent']['max_iter']}")


# %%
# ─────────────────────────────────────────────────────────────────────────
# Cell A7: Create output directory and persist the (overridden) task spec
# ─────────────────────────────────────────────────────────────────────────
# `spec.output_dir` is `<save_path>/YYYY-MM-DD/HH-MM-SS/`. Saving the spec there
# first means even if the tutorial crashes you'll have the exact config that
# produced the partial run. The saved file is chmod'd read-only (see
# TaskSpec.save).
if __name__ == "__main__":
    os.makedirs(spec.output_dir, exist_ok=True)
    saved_path = spec.save(spec.output_dir)
    print(f"Output dir:     {spec.output_dir}")
    print(f"Saved spec at:  {saved_path}")


# %%
# ─────────────────────────────────────────────────────────────────────────
# Cell B8: Load the data
# ─────────────────────────────────────────────────────────────────────────
# `spec.load_data_fn` was resolved from
# `projects/orientation_tuning/data_loader/load_data.py`. It returns a triple
# of dataset groups: discover (used for evolutionary search), validate (held
# out, only used at the end to rank survivors), and eval (a small per-program
# fingerprint used for dedup, not for scoring).
# Each `_train` / `_test` dict has 'stimulus' and 'response' keys.
if __name__ == "__main__":
    X_discover, X_validate, X_eval = spec.load_data_fn(
        data_path=spec.io["data_path"], **spec.project_params,
    )

    print("X_discover[0] (train) stimulus shape:", X_discover[0]["stimulus"].shape)
    print("X_discover[1] (test)  stimulus shape:", X_discover[1]["stimulus"].shape)
    print("X_validate[0] (train) stimulus shape:", X_validate[0]["stimulus"].shape)
    print("X_eval (fingerprint)  stimulus shape:", X_eval["stimulus"].shape)


# %%
# ─────────────────────────────────────────────────────────────────────────
# Cell B9: Build retry config + flat config
# ─────────────────────────────────────────────────────────────────────────
# `RetryConfig` controls how `call_llm` reacts to transient HTTP errors
# (e.g. 503 → retry with backoff; 400 → fail fast). The `flat_config` merges
# evolution + llms + scoring sections so prompts can look up variables by
# name without knowing which section they live in (see TaskSpec.flat_config).
if __name__ == "__main__":
    retry_config = RetryConfig(**spec.llms.get("retry", {}))
    config_dict = {**spec.flat_config, "retry_config": retry_config}

    print(f"retry: max_retries={retry_config.max_retries}, "
          f"backoff_multiplier={retry_config.backoff_multiplier}")
    print(f"flat_config has {len(config_dict)} keys (merged from evolution + llms + scoring).")


# %%
# ─────────────────────────────────────────────────────────────────────────
# Cell B10: Initialize Population and seed islands
# ─────────────────────────────────────────────────────────────────────────
# A `Population` is a flat list of all Programs ever created (alive or dead);
# `islands` is a list of `set[int]` of population indices, one set per island.
# Programs are referenced by index everywhere downstream — programs themselves
# never move, only their references.
# `seed(...)` adds the hand-written seed programs to the population and gives
# every island the full seed set as a starting point.
if __name__ == "__main__":
    population = Population()
    islands = seed(population, spec.seed_programs, spec.evolution["n_islands"])
    census = []  # will collect islands snapshot per generation for the family tree

    print(f"Total programs in population: {len(population)}")
    print(f"Number of islands:             {len(islands)}")
    for i, isl in enumerate(islands):
        print(f"  island {i}: program indices = {sorted(isl)}")


# %%
# ─────────────────────────────────────────────────────────────────────────
# Cell C11: Translate seed programs from numpy to JAX
# ─────────────────────────────────────────────────────────────────────────
# Seeds ship with hand-written numpy `model()` and `parameter_estimator()`
# code. JAX is needed for the gradient-descent fitting used by `score()`.
# `translate_programs` fills `program.code.model_jax` for every program that
# doesn't already have it. This is the first LLM call of the tutorial.
if __name__ == "__main__":
    asyncio.run(translate_programs(
        population,
        spec.prompt_schemas.jax_model,
        spec.llms["jax_model_translator_llm"],
        retry_config=retry_config,
        max_tokens=config_dict.get("max_tokens"),
    ))

    # Inspect one translation to see what came back from the LLM.
    first_seed = population[0]
    print(f"Seed program 0 has model_jax: {first_seed.code.model_jax is not None}")
    print("First 500 chars of JAX-translated model code:")
    print("─" * 60)
    print((first_seed.code.model_jax or "")[:500])
    print("─" * 60)


# %%
# ─────────────────────────────────────────────────────────────────────────
# Cell C12: Score the seed programs on the discover split
# ─────────────────────────────────────────────────────────────────────────
# `score()` runs each program in a fresh subprocess with a wall-clock timeout
# (so LLM-generated code can't hang the whole pipeline). It calls the
# parameter_estimator on each training sample, then JAX-grad-descents on the
# parameters using `loss_fn`, then evaluates on test data. Mutates the program
# in place: program_losses.discover.{init, final}, eval_fingerprint, n_params.
if __name__ == "__main__":
    score(population, X_discover, X_eval, spec.scoring, spec.loss_fn, split="discover")

    print(f"{'idx':>4} {'name':<30} {'init':>10} {'final':>10}  n_params")
    print("─" * 70)
    for p in population._programs:
        name = (p.name or "(unset)")[:30]
        init_ = p.program_losses.discover.init
        final_ = p.program_losses.discover.final
        init_str = f"{init_:.2f}" if isinstance(init_, (int, float)) else str(init_)
        final_str = f"{final_:.2f}" if isinstance(final_, (int, float)) else str(final_)
        print(f"{p.idx:>4} {name:<30} {init_str:>10} {final_str:>10}  {p.n_params}")


# %%
# ─────────────────────────────────────────────────────────────────────────
# Cell D13: Get the schedule values for generation 0
# ─────────────────────────────────────────────────────────────────────────
# At gen=0 the schedule emits mode="explore" (first half of generations) and
# temperature=2.0 (Gemini scale; the Anthropic call_llm guard will rescale to
# 1.0 at the actual API call). The `llms` tuple contains the model names for
# each of the three LLM roles.
if __name__ == "__main__":
    generation = 0
    mode, temperature, llms = spec.schedule(generation)

    print(f"generation:    {generation}")
    print(f"mode:          {mode}")
    print(f"temperature:   {temperature:.4f}  (call_llm will rescale to "
          f"{temperature / 2.0:.4f} for Anthropic)")
    print(f"llms.model:        {llms.model}")
    print(f"llms.param_est:    {llms.param_est}")
    print(f"llms.model_jax:    {llms.model_jax}")


# %%
# ─────────────────────────────────────────────────────────────────────────
# Cell D14: Spawn new program "shells" for this generation
# ─────────────────────────────────────────────────────────────────────────
# `spawn(...)` samples parents from each island (uniformly here) and creates
# empty Program objects with just a BirthCertificate (lineage metadata) — no
# code yet. The next three cells fill in the code via LLM calls.
if __name__ == "__main__":
    n_before = len(population)
    sizes_before = [len(isl) for isl in islands]

    spawn(
        population, islands, generation, mode, temperature,
        batch_size=spec.evolution["batch_size"],
        num_parents=spec.llms["num_parents"],
        rng=spec.rng,
    )

    n_added = len(population) - n_before
    sizes_after = [len(isl) for isl in islands]

    print(f"Programs added to population: {n_added}")
    print(f"Island sizes:  before={sizes_before}  after={sizes_after}")
    print("\nNew shells (no code yet):")
    for p in population._programs[n_before:]:
        parents = p.birth.parent_indices
        print(f"  idx={p.idx}  island={p.birth.island}  batch={p.birth.batch_index}  parents={parents}")


# %%
# ─────────────────────────────────────────────────────────────────────────
# Cell D15: Generate `model()` code via the LLM
# ─────────────────────────────────────────────────────────────────────────
# `generate_models` builds a prompt per shell (using the parent programs as
# in-context examples), calls Claude, parses the structured response, and
# fills in `program.code.model`, `program.name`, `program.birth.llm_name`.
# When `image_feedback/plot.py` exists for the project (it does for
# orientation_tuning), this also renders a per-program PNG of parent fits.
if __name__ == "__main__":
    asyncio.run(generate_models(
        population,
        spec.prompt_schemas.model,
        llms.model,  # single string after override; list cycling not exercised
        mode,
        temperature,
        config=config_dict,
        spec=spec,
        data=X_discover[1],  # test-split data is used for the image-feedback render
    ))

    # Show what Claude wrote for one of the new programs.
    new_program = population._programs[n_before]
    print(f"Generated name:  {new_program.name}")
    print(f"Generated by:    {new_program.birth.llm_name}")
    print(f"Model code ({len(new_program.code.model or '')} chars), first 600:")
    print("─" * 60)
    print((new_program.code.model or "")[:600])
    print("─" * 60)


# %%
# ─────────────────────────────────────────────────────────────────────────
# Cell D16: Generate `parameter_estimator()` code via the LLM
# ─────────────────────────────────────────────────────────────────────────
# Once we have `model()` code, the LLM is asked to write a fast, statistical
# initial estimator for the model's free parameters. This is what feeds the
# `init` loss reported by `score()`; gradient descent then drives it to
# `final`. Failures here are caught — the program just uses default_params.
if __name__ == "__main__":
    asyncio.run(generate_param_ests(
        population,
        spec.prompt_schemas.param_est,
        llms.param_est,
        config_dict,
    ))

    print(f"Param-estimator code ({len(new_program.code.param_est or '')} chars):")
    print("─" * 60)
    print(new_program.code.param_est or "(empty)")
    print("─" * 60)


# %%
# ─────────────────────────────────────────────────────────────────────────
# Cell D17: Translate the new programs' models to JAX
# ─────────────────────────────────────────────────────────────────────────
# Same as cell C11 but only translates programs without `model_jax` set —
# i.e. the freshly-spawned ones. Seeds already have their JAX code.
if __name__ == "__main__":
    asyncio.run(translate_programs(
        population,
        spec.prompt_schemas.jax_model,
        llms.model_jax,
        retry_config=retry_config,
        max_tokens=config_dict.get("max_tokens"),
    ))

    print(f"new_program.code.model_jax populated: {new_program.code.model_jax is not None}")


# %%
# ─────────────────────────────────────────────────────────────────────────
# Cell D18: Score the new programs on the discover split
# ─────────────────────────────────────────────────────────────────────────
# Same `score()` call as C12; only programs that don't yet have a discover
# loss are scored. Programs whose generated code fails to compile, hangs, or
# returns NaN get a loss of inf — they're not crashes, just unusable.
if __name__ == "__main__":
    score(population, X_discover, X_eval, spec.scoring, spec.loss_fn, split="discover")

    print(f"{'idx':>4} {'name':<30} {'init':>10} {'final':>10}  n_params")
    print("─" * 70)
    for p in population._programs:
        name = (p.name or "(unset)")[:30]
        init_ = p.program_losses.discover.init
        final_ = p.program_losses.discover.final
        init_str = f"{init_:.2f}" if isinstance(init_, (int, float)) else str(init_)
        final_str = f"{final_:.2f}" if isinstance(final_, (int, float)) else str(final_)
        print(f"{p.idx:>4} {name:<30} {init_str:>10} {final_str:>10}  {p.n_params}")


# %%
# ─────────────────────────────────────────────────────────────────────────
# Cell D19: Deduplicate programs across islands
# ─────────────────────────────────────────────────────────────────────────
# `deduplicate(...)` removes programs that produce statistically identical
# outputs (within-island and between-island), keeping the lower-loss copy.
# This stops the population from collapsing onto trivial variations.
# Tolerances come from `evolution` config (or defaults).
if __name__ == "__main__":
    sizes_before = [len(isl) for isl in islands]
    deduplicate(islands, population, spec.evolution)
    sizes_after = [len(isl) for isl in islands]
    print(f"Island sizes after dedup:  before={sizes_before}  after={sizes_after}")


# %%
# ─────────────────────────────────────────────────────────────────────────
# Cell D20: Prune each island to the best `critical_population_size` programs
# ─────────────────────────────────────────────────────────────────────────
# `prune(...)` cuts each island down to (critical_population_size − n_migrants)
# programs, ordered by discover.final loss. This is what makes the search
# selective: bad programs get dropped, good ones survive to seed the next gen.
if __name__ == "__main__":
    sizes_before = [len(isl) for isl in islands]
    prune(islands, population, spec.evolution)
    sizes_after = [len(isl) for isl in islands]
    print(f"Island sizes after prune:  before={sizes_before}  after={sizes_after}")
    print(f"(target = critical_population_size − n_migrants = "
          f"{spec.evolution['critical_population_size']} − {spec.evolution['n_migrants']} = "
          f"{spec.evolution['critical_population_size'] - spec.evolution['n_migrants']})")


# %%
# ─────────────────────────────────────────────────────────────────────────
# Cell D21: Migrate programs between islands
# ─────────────────────────────────────────────────────────────────────────
# `migrate(...)` Boltzmann-samples `n_migrants` programs from each island
# (biased toward low loss; the bias strength is governed by `temperature`)
# and copies them into the destination island indicated by `topology[i]`.
# Migration is what lets islands share their good discoveries.
if __name__ == "__main__":
    sizes_before = [len(isl) for isl in islands]
    migrate(islands, population, spec.evolution, temperature, rng=spec.rng)
    sizes_after = [len(isl) for isl in islands]
    print(f"Island sizes after migrate:  before={sizes_before}  after={sizes_after}")
    print(f"Topology: {spec.evolution['topology']}  (island i sends migrants to island topology[i])")

    # Append a snapshot to the census so the family tree can show the run's history.
    census.append([set(island) for island in islands])


# %%
# ─────────────────────────────────────────────────────────────────────────
# Cell E22: Validation scoring — score surviving programs on held-out data
# ─────────────────────────────────────────────────────────────────────────
# In a full run, this happens once at the very end (after all generations).
# We do it here for completeness so the family-tree HTML has validate losses
# to display. `prepare_validation_scoring` flips program_losses.validate from
# the `NotValidated()` sentinel to an unset state, signaling that score()
# should fill it in.
if __name__ == "__main__":
    population.prepare_validation_scoring(islands)
    # X_eval=None so the discover-derived fingerprint isn't overwritten by the
    # validate-split data.
    score(population, X_validate, None, spec.scoring, spec.loss_fn, split="validate")

    print(f"{'idx':>4} {'name':<30} {'discover.final':>15} {'validate.final':>15}")
    print("─" * 80)
    for p in population._programs:
        name = (p.name or "(unset)")[:30]
        df = p.program_losses.discover.final
        vf = p.program_losses.validate.final
        df_s = f"{df:.2f}" if isinstance(df, (int, float)) else str(df)
        vf_s = f"{vf:.2f}" if isinstance(vf, (int, float)) else str(vf)
        print(f"{p.idx:>4} {name:<30} {df_s:>15} {vf_s:>15}")


# %%
# ─────────────────────────────────────────────────────────────────────────
# Cell E23: Rank the validated programs
# ─────────────────────────────────────────────────────────────────────────
# `rank()` assigns 1..N to programs in ascending validate.final order so the
# output JSONL is sortable. Only programs that survived to the end get ranked.
if __name__ == "__main__":
    rank(population)
    print("Top of ranking (lowest validate.final loss):")
    ranked = sorted(
        [p for p in population._programs if p.rank is not None],
        key=lambda p: p.rank,
    )
    for p in ranked[:5]:
        name = (p.name or "(unset)")[:40]
        print(f"  rank={p.rank}  idx={p.idx}  loss={p.program_losses.validate.final:.3f}  {name}")


# %%
# ─────────────────────────────────────────────────────────────────────────
# Cell E24: Save population, census, family tree
# ─────────────────────────────────────────────────────────────────────────
# These are the three artifacts a full run leaves behind:
#   - population.jsonl: one Program per line (code, losses, lineage, etc.)
#   - island_census.jsonl: per-island, per-generation roster
#   - family_tree.html: standalone interactive HTML for browsing the run
# In `edgar/run.py` these live inside the `finally:` so they get written even
# on crash. We mirror that intent here but with a plain straight-line write.
if __name__ == "__main__":
    population.save(os.path.join(spec.output_dir, "population.jsonl"))
    save_island_census(census, os.path.join(spec.output_dir, "island_census.jsonl"))
    write_family_tree(
        population, census, spec.output_dir,
        param_penalty_weight=spec.scoring.get("param_penalty_weight"),
    )

    print(f"Saved to: {spec.output_dir}")
    print("Files:")
    for f in sorted(Path(spec.output_dir).iterdir()):
        size = f.stat().st_size
        print(f"  {f.name:<25} {size:>10} bytes")

    print("\nTo view the family tree (Mac):")
    print(f"  open {spec.output_dir}/family_tree.html")


# %%
# ─────────────────────────────────────────────────────────────────────────
# Cell E25: Where to go from here
# ─────────────────────────────────────────────────────────────────────────
# - For the automated equivalent of this tutorial, read `edgar/run.py`. The
#   `run(spec)` function does everything you just stepped through, looped
#   over `n_generations`, with logging and exception-safe output saves.
# - To advance to generation 1 by hand, re-execute cells D13–D21 with
#   `generation = 1`. State persists across cells.
# - To re-run the whole tutorial from scratch, restart the Python kernel
#   (or `python tutorials/walkthrough_orientation_tuning.py` for a fresh run).
# - To compare LLMs, change `model_llm` / `param_est_llm` / `jax_model_translator_llm`
#   in cell A6.
if __name__ == "__main__":
    print("Walkthrough complete.")