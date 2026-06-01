"""tutorials/inspect_outputs.py

Post-hoc walkthrough: how to inspect and query the outputs of a completed
EDGAR pipeline run. Companion to `tutorials/walkthrough_orientation_tuning.py`
(which teaches the run-side); this one teaches the read-side.

How to use:
    - In Cursor / VS Code / Jupyter: open this file and execute each `# %%`
      cell with Shift+Enter to see intermediate state between steps.
    - As a script: `python tutorials/inspect_outputs.py` runs every cell
      top-to-bottom in a few seconds (no LLM calls, no data loading).

The script anchors on a real run that finished this morning:
    program_databases/05-24/09-25-00/

To inspect a different run, change `RUN_DIR` at the top of cell 1.

What you'll learn (one concept per cell):
    1.  Setup + paths
    2.  Files written by a run
    3.  task_spec.yaml — the frozen, reproducible record of what was run
    4.  Reconstructing a TaskSpec from a saved run
    5.  population.jsonl — the raw on-disk JSON record
    6.  Population.load — the class API
    7.  Population summary statistics
    8.  Useful queries (best program, by generation, lineage)
    9.  Inspecting a single program's code + fitted params
    10. Finding failed / NaN programs
    11. island_census.jsonl — who was on which island when
    12. Monitoring helpers (family_tree + progress reports)
    13. Reading run.log
    14. Where to go next
"""

# %%
import json
import os
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

RUN_DIR = REPO_ROOT / "program_databases" / "06-01" / "15-18-49"

print(f"REPO_ROOT = {REPO_ROOT}")
print(f"RUN_DIR   = {RUN_DIR}")
print(f"exists?     {RUN_DIR.exists()}")


# %% [markdown]
# # 2. What files does a run write?
#
# Every successful run produces the same five-ish artifacts in
# `<save_path>/MM-DD/HH-MM-SS/`. Knowing what each one is up-front makes the
# rest of this walkthrough easier to follow.

# %%
print(f"Contents of {RUN_DIR}:")
for entry in sorted(RUN_DIR.iterdir()):
    kind = "dir " if entry.is_dir() else "file"
    size = "" if entry.is_dir() else f"  ({entry.stat().st_size:>8} bytes)"
    print(f"  [{kind}] {entry.name}{size}")

print("""
Cheat sheet:
  task_spec.yaml      — frozen config + git sha + seed programs. The "what was run".
  population.jsonl    — one Program per line (birth, code, losses, fitted params, ...).
  island_census.jsonl — per-generation, per-island membership (despite the .jsonl
                        extension this is a single JSON document, not line-delimited).
  family_tree.html    — interactive parent-child graph of every Program.
  run.log             — human-readable per-generation summary.
  image_feedback/     — model-fit plots used as image input to the LLM (if plot_fn set).
""")


# %% [markdown]
# # 3. task_spec.yaml — the frozen record
#
# This file is written read-only by `TaskSpec.save()` so it can't drift from
# the run that produced it. Read it as plain YAML first: it's just a dump of
# every config knob, the git sha, the resolved LLM names, the seed program
# source, and the full prompt schemas.

# %%
import yaml  # noqa: E402

with open(RUN_DIR / "task_spec.yaml") as f:
    spec_dict = yaml.safe_load(f)

print("Top-level keys in task_spec.yaml:")
for k in spec_dict:
    print(f"  - {k}")

print()
print(f"task_name:  {spec_dict['task_name']}")
print(f"git_sha:    {spec_dict['git_sha']}")
print(f"git_dirty:  {spec_dict['git_dirty']}  "
      f"(True = uncommitted edits at run time, sha alone is insufficient to reproduce)")
print(f"created_at: {spec_dict['created_at']}")
print()
print(f"data file:  {spec_dict['io']['data_path']}")
print(f"n_generations × n_islands × batch_size = "
      f"{spec_dict['evolution']['n_generations']} × "
      f"{spec_dict['evolution']['n_islands']} × "
      f"{spec_dict['evolution']['batch_size']}")
print(f"LLMs used:  model={spec_dict['llms']['model_llm']}  "
      f"param_est={spec_dict['llms']['param_est_llm']}  "
      f"jax={spec_dict['llms']['jax_model_translator_llm']}")
print(f"seeds:      {len(spec_dict['seed_programs'])} seed programs "
      f"(full source code embedded in the yaml)")


# %% [markdown]
# # 4. Reconstructing a TaskSpec from a saved run
#
# `Config.from_taskspec` + `TaskSpec.from_config` is the path the runner uses
# when you point it at a `task_spec.yaml` instead of a fresh `config.yaml`.
# We don't actually re-run anything here, but rebuilding the spec object is
# the right starting point for any post-hoc analysis that needs access to the
# resolved callables (loss_fn, load_data_fn, plot_fn).
#
# Note: `TaskSpec.from_config` does NOT load the data file — it just resolves
# project callables — so this works even when `data_path` is not on disk.

# %%
from edgar.io.config import Config  # noqa: E402
from edgar.io.task_spec import TaskSpec  # noqa: E402

config = Config.from_taskspec(RUN_DIR / "task_spec.yaml")
spec = TaskSpec.from_config(config)

print(f"Reconstructed TaskSpec:")
print(f"  task_name:    {spec.task_name}")
print(f"  output_dir:   {spec.output_dir}  "
      f"(NOTE: this is a *fresh* timestamp, not the original)")
print(f"  seed count:   {len(spec.seed_programs)}")
print(f"  loss_fn:      {spec.loss_fn.__module__}.{spec.loss_fn.__name__}")
print(f"  plot_fn:      "
      f"{spec.plot_fn.__name__ if spec.plot_fn else None}")
print(f"  schedule(0):  {spec.schedule(0)[:2]}  (mode, temperature)")


# %% [markdown]
# # 5. population.jsonl — raw JSON structure
#
# Each line is one `Program` serialised with `dataclasses.asdict`. See
# `edgar/evolution/program.py` for the dataclass definition. The keys you'll
# care about most:
#
#   - `birth.{generation, island, mode, parent_indices, llm_name}` — lineage.
#     Seeds have `generation=-1, island=-1, mode="seed"`; evolved programs
#     get `mode="explore"` or `"exploit"` from `schedule(gen)`.
#   - `code.{model, param_est, model_jax}` — the three source strings.
#   - `program_losses.{discover, validate}.{init, final}` — pre-/post-GD loss
#     on the two splits. `init` is loss with the parameter_estimator's guess;
#     `final` is loss after Adam gradient descent (max_iter steps).
#   - `n_params` — number of fitted parameters (drives the complexity penalty).
#   - `rank` — set at the end of the run by `scoring.rank()`. `rank=1` is best.
#   - `eval_fingerprint` — model outputs on the `X_eval` subset; used for
#     dedup (cosine similarity between programs).
#   - `params` — the fitted parameter dict after gradient descent.
#   - `sample_losses` — per-sample loss vector (no complexity penalty).

# %%
with open(RUN_DIR / "population.jsonl") as f:
    raw_lines = f.readlines()

print(f"population.jsonl has {len(raw_lines)} lines (= {len(raw_lines)} Programs)\n")

first = json.loads(raw_lines[0])
print("Keys of one Program (line 0):")
for k, v in first.items():
    sample = v if not isinstance(v, (list, dict)) else f"{type(v).__name__} of length {len(v)}"
    print(f"  {k:18s} = {sample}")


# %% [markdown]
# # 6. Loading via the Population class API
#
# `Population.load` rebuilds typed `Program` objects from the JSONL file. This
# is the path you'll use for any non-trivial analysis. Indexing by global idx
# is stable: `population[i]` is always the program with `idx == i`.

# %%
from edgar.evolution.population import Population  # noqa: E402

population = Population.load(str(RUN_DIR / "population.jsonl"))

print(f"len(population) = {len(population)}\n")
for p in population:
    print(f"  #{p.idx}  {p.name!r:55s}  "
          f"gen={p.birth.generation:>2}  island={p.birth.island:>2}  "
          f"mode={p.birth.mode!r}")


# %% [markdown]
# # 7. Population summary statistics
#
# Once everything is in typed form, summary stats are one-liners. Two of the
# 6 programs are seeds (gen=-1); the other 4 came from generation 0 (this
# run was a 1-generation smoke). One of them produced NaN losses — we'll dig
# into that in cell 10.

# %%
mode_counts = Counter(p.birth.mode for p in population)
gen_counts = Counter(p.birth.generation for p in population)
island_counts = Counter(p.birth.island for p in population)

print(f"By birth.mode:        {dict(mode_counts)}")
print(f"By generation:        {dict(sorted(gen_counts.items()))}")
print(f"By island:            {dict(sorted(island_counts.items()))}")
print(f"Distinct LLM authors: "
      f"{sorted({p.birth.llm_name for p in population if p.birth.llm_name})}")
print(f"Param counts:         min={min(p.n_params for p in population)} "
      f"max={max(p.n_params for p in population)}")


# %% [markdown]
# # 8. Useful queries
#
# A handful of patterns that come up constantly. Note: a few programs may
# have `None` or `NaN` losses — guard with `is not None` and `math.isfinite`
# before sorting or comparing.

# %%
import math  # noqa: E402

def is_finite(x):
    return x is not None and isinstance(x, (int, float)) and math.isfinite(x)


print("--- Q1: lowest discover.final loss ---")
finite = [p for p in population if is_finite(p.program_losses.discover.final)]
best_discover = min(finite, key=lambda p: p.program_losses.discover.final)
print(f"  #{best_discover.idx}  {best_discover.name!r}  "
      f"loss={best_discover.program_losses.discover.final:.4f}")

print("\n--- Q2: lowest validate.final loss (held-out data) ---")
finite_val = [
    p for p in population if is_finite(p.program_losses.validate.final)
]
best_validate = min(finite_val, key=lambda p: p.program_losses.validate.final)
print(f"  #{best_validate.idx}  {best_validate.name!r}  "
      f"loss={best_validate.program_losses.validate.final:.4f}  "
      f"(this program's rank is {best_validate.rank})")

print("\n--- Q3: losses by generation (seed = gen -1) ---")
for gen in sorted({p.birth.generation for p in population}):
    bucket = [
        p for p in population
        if p.birth.generation == gen
        and is_finite(p.program_losses.discover.final)
    ]
    if not bucket:
        print(f"  gen {gen:>2}: (no finite-loss programs)")
        continue
    losses = sorted(p.program_losses.discover.final for p in bucket)
    print(f"  gen {gen:>2}: n={len(bucket)}  "
          f"best={losses[0]:.2f}  median={losses[len(losses)//2]:.2f}  "
          f"worst={losses[-1]:.2f}")

print("\n--- Q4: most-fertile seed (most descendants by parent_indices) ---")
descendants = Counter()
for p in population:
    for parent_idx in p.birth.parent_indices:
        descendants[parent_idx] += 1
for seed_idx in (p.idx for p in population if p.birth.mode == "seed"):
    name = population[seed_idx].name
    print(f"  #{seed_idx} {name!r}: {descendants[seed_idx]} children")

print("\n--- Q5: parents of the best validate program ---")
for parent_idx in best_validate.birth.parent_indices:
    parent = population[parent_idx]
    print(f"  parent #{parent_idx}  {parent.name!r}  "
          f"discover.final={parent.program_losses.discover.final}")
if not best_validate.birth.parent_indices:
    print(f"  (no parents — this is a seed)")


# %% [markdown]
# # 9. Inspecting a single program in depth
#
# Once you have a program of interest (e.g. the rank-1 winner) you can pull
# out everything: source, fitted parameters, and per-sample losses.

# %%
ranked = [p for p in population if p.rank is not None]
winner = min(ranked, key=lambda p: p.rank)

print(f"Winner is #{winner.idx} (rank {winner.rank}): {winner.name!r}")
print(f"  birth:           {winner.birth}")
print(f"  n_params:        {winner.n_params}")
print(f"  fitted params:   {winner.params}")
print(f"  discover losses: init={winner.program_losses.discover.init:.4f}  "
      f"final={winner.program_losses.discover.final:.4f}")
print(f"  validate losses: init={winner.program_losses.validate.init:.4f}  "
      f"final={winner.program_losses.validate.final:.4f}")
print(f"  sample_losses:   shape={winner.sample_losses.shape}  "
      f"mean={winner.sample_losses.mean():.4f}")
print(f"  eval fingerprint shape: {winner.eval_fingerprint.shape}")

print("\n--- model source ---")
print(winner.code.model)
print("--- param_est source ---")
print(winner.code.param_est)
print("--- (jax-translated model source is at winner.code.model_jax) ---")


# %% [markdown]
# # 10. Failed / NaN programs
#
# When a generated program crashes, hits the scoring timeout, or produces a
# NaN gradient, scoring assigns sentinel values rather than aborting the run.
# `None` means "never scored at all" (e.g. the LLM didn't return code or the
# subprocess died); `nan` means "scored but the loss was non-finite".
# Triaging these is how you find prompt or sandbox issues.

# %%
print("Status of every program on discover.final:")
for p in population:
    v = p.program_losses.discover.final
    if v is None:
        status = "NEVER SCORED"
    elif isinstance(v, float) and not math.isfinite(v):
        status = f"NON-FINITE ({v})"
    else:
        status = f"loss = {v:.4f}"
    print(f"  #{p.idx}  {p.name!r:55s}  {status}")

failed = [
    p for p in population
    if p.program_losses.discover.final is None
    or (isinstance(p.program_losses.discover.final, float)
        and not math.isfinite(p.program_losses.discover.final))
]
print(f"\n{len(failed)} program(s) failed scoring. "
      f"Look at their code/run.log to diagnose.")


# %% [markdown]
# # 11. island_census.jsonl — who was on which island when
#
# Despite the `.jsonl` extension this is a single JSON document. The shape is
# `census[generation][island_idx]` -> list of program indices alive on that
# island at the END of that generation (post-prune, post-deduplicate,
# post-migrate). Use `load_island_census` to get back `list[list[set[int]]]`.

# %%
from edgar.evolution.island import load_island_census  # noqa: E402

census = load_island_census(str(RUN_DIR / "island_census.jsonl"))
n_gens = len(census)
n_islands = len(census[0]) if census else 0
print(f"Census: {n_gens} generation(s) × {n_islands} island(s)\n")

for g, gen in enumerate(census):
    print(f"  end of gen {g}:")
    for i, island in enumerate(gen):
        ids = sorted(island)
        print(f"    island {i}  size={len(ids)}  members={ids}")

print()
print("Programs alive at the end of the run "
      "(union of every island in the last generation):")
alive_at_end = set()
if census:
    for island in census[-1]:
        alive_at_end |= island
print(f"  {sorted(alive_at_end)}")

dead_at_end = {p.idx for p in population} - alive_at_end
if dead_at_end:
    print(f"  pruned at some point: {sorted(dead_at_end)}")


# %% [markdown]
# # 12. Monitoring helpers
#
# `edgar/monitoring/` writes three standalone HTML reports straight from a
# `Population` + `census`. They have no run-time dependency, so you can
# regenerate them post-hoc whenever you've tweaked the visualisation code or
# inherited a run that's missing one of them.

# %%
import tempfile  # noqa: E402

from edgar import monitoring  # noqa: E402
from edgar.monitoring.family_tree import write_family_tree  # noqa: E402
from edgar.monitoring.progress import write_progress  # noqa: E402

print("Public monitoring helpers:")
for name in monitoring.__all__:
    print(f"  monitoring.{name}")

# Write to a tempdir rather than mutating RUN_DIR itself, so the canonical
# run output stays pristine. Replace with any path of your choice if you
# want to keep the regenerated HTML around.
regen_dir = Path(tempfile.mkdtemp(prefix="edgar_regen_html_"))

family_tree_path = write_family_tree(
    population, census, regen_dir,
    task_name=spec.task_name,
    param_penalty_weight=spec.scoring.get("param_penalty_weight", 0.0),
)
loss_progress_path, gd_effect_path = write_progress(
    population, census, regen_dir,
    task_name=spec.task_name,
    param_penalty_weight=spec.scoring.get("param_penalty_weight", 0.0),
)

print(f"\nRegenerated into {regen_dir}/:")
print(f"  family_tree.html   ({family_tree_path.stat().st_size} bytes)")
print(f"  loss_progress.html ({loss_progress_path.stat().st_size} bytes)")
print(f"  gd_effect.html     ({gd_effect_path.stat().st_size} bytes)")
print(f"\nTo open the original family tree in a browser:")
print(f"  open {RUN_DIR / 'family_tree.html'}")


# %% [markdown]
# # 13. The run log
#
# `run.log` is a human-readable per-generation summary written by
# `edgar.io.logging.log_generation`. At the default `compact` level you get one
# block per generation: timings, success rates for each LLM stage, the global
# best loss so far, and the best program on each island. Buffered warnings
# get appended to whichever generation they fired during. Note this is only
# if the run was launched via `edgar.run.run()` e.g by running `edgar run ...` 
# from the command line.

# %%
with open(RUN_DIR / "run.log") as f:
    print(f.read())


# %% [markdown]
# # 14. Where to go next
#
# - For the **run-side** counterpart that builds and executes a pipeline cell
#   by cell, see `tutorials/walkthrough_orientation_tuning.py`.
# - To compare two runs, load both populations and diff their best programs:
#   `Population.load(run_a / "population.jsonl")` and similarly for run B.
# - To re-execute a saved run with the same settings:
#   `python -m edgar.cli run program_databases/<MM-DD>/<HH-MM-SS>/task_spec.yaml`
# - To regenerate any visualisation, see cell 12 — every helper takes
#   (population, census, out_dir) and writes a standalone HTML file.

# %%
print("Done.")
