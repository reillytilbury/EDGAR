---
name: Work thread
description: Ongoing work, long-term goals, system context, and architecture notes for EDGAR refactor
type: project
---

## System context

### Working style
- User has ADHD — be decisive, give one clear recommendation rather than listing options, keep responses short
- Don't add complexity beyond what's asked
- One thing at a time

---

## Architecture

- Consolidate `Program` dataclass as single candidate representation (currently ~4 variants)
- Make `Population` the persistence layer (replace JSONL)
- Clean config routing: modules only receive their subsection (`config["scoring"]`, `config["llms"]`, etc.)
- Create coherent program/island saving structure (currently programs_database is messy)

### Naming
- Sample/trial split overloaded: `train_sample`/`test_sample` (discovery vs eval) conflicts with `train`/`test` (GD splits)
- Standard terminology doesn't fit this structure. Parked for later clarification.

---

## TODO (priority order)

### Tests
- [-] Tests for `src/evolution` (island operations, population, program): worth an integration test with an example evolution?
- [ ] Tests for `src/io` (config, task_spec)
- [-] Tests for `src/llm`: done for fake LLM, need some for real LLM calls
- [-] Integration test for `run.py` (small end-to-end run): work in progress
- [ ] Wire tests to GitHub Actions CI with status badges in README

### Evolution
- [x] **Temperature warping in `migrate`** — raw temperature from `schedule()` is in [1, 2] but `boltzmann_sample` expects something in [0, 1]. Correct transform is `T_warped = (T - 1.0) ** 4`, confirmed in old `hypothesis_engine.py` on the `main` remote branch.

### Core scoring
- [ ] Separate input/target keys in data — prevent models from cheating by accessing target values. Add `input_keys` and `target_keys` to config, split data before passing to model_fn
- [ ] Support variable-length trials per sample — currently assumes rectangular data (all samples have same n_trials); would need list/dict of variable-length arrays instead
- [ ] Rethink `_eval_fingerprint` in scoring.py — design smell (indices smuggled in data dict), not urgent

### LLM
- [ ] Consider unifying `generate_models`, `generate_param_ests`, `translate_programs` into a single `generate(population, spec, prompt_type, ...)` — not worth it until there's a 4th generation type

### Documentation
- [x] Update README.md to reflect refactored architecture
- [ ] Add documentation describing the refactors (what changed and why)

---

## Completed

### Data loading
- [x] Change signature of all project `load_data` functions to output `X_discover`, `X_validate`, `X_eval` using shared split logic; remove per-project `train_test_split` functions
- [x] Ensure all data is in JAX format — `_to_jax()` helper added to all six `load_data` functions
- [x] Integrate Dabin's drop trials plan
- [x] Need a way of saving hyperparameters/config in metadata-only format + code used to load + seed islands

### LLM
- [x] Integrate image prompt into LLMs: `image_path` on Program, images saved at `output_dir/image_feedback/gen_NNN/island_NNN/batch_NNN/image.png`, bytes passed to `call_llm`

### CLI + logging
- [x] Implement logging (`src/io/logging.py`) with `compact`, `code`, `prompts` levels
- [x] Add `--log-level` flag to `edgar run`
- [x] `edgar run` accepts both `config.yaml` and `task_spec.yaml`; supports `--section.key=value` CLI overrides
- [x] `edgar validate` checks actual functions exist, not just files

### Utils removal
- [x] Remove `src/utils.py` — deleted along with `src/io/__init__.py`; all project plotters and data loaders updated to use Program API directly

### Bug fixes (found by Dabin during first run)
- [x] API key not loaded: added `load_dotenv()` to `llm_calling.py` so `.env` is read; renamed `.env` key from `GOOGLE_API_KEY` to `GEMINI_API_KEY` (pydantic-ai name)
- [x] Seed models not appearing in prompts: `PromptSchema.build_prompt` uses `getattr(p, "model_code")` etc., but `Program` had no such attributes. Added properties `descriptive_name`, `loss_discover`, `model_code`, `param_est_code` to `Program`.
- [x] `TaskSpec.from_config` accepted both config.yaml and task_spec.yaml via filename branch: split into `Config.from_yaml` / `Config.from_taskspec` in `config.py`; `TaskSpec.from_config` now takes a `Config` object only. CLI and `run.py` __main__ do the routing.

---

## Reference

### Data naming conventions
- `X_discover`: Data train + test seen by the LLM model discovery loop
- `X_validate`: Data train + test samples LLM model discovery loop never sees
- `X_eval`: Small data (prob part of X_discover) that is used to evaluate programs on for deduplication and fingerprinting

### Main running code — target pseudocode

```
INPUTS:
    spec  TaskSpec

STATE:
    population  Population
    islands     list[set[int]]

DATA:
    X_discover  (train, test)
    X_validate  (train, test)
    X_eval      dict


X_discover, X_validate, X_eval = load_and_split(spec)
population = Population()
islands = seed(spec, population, X_discover, X_eval)

for gen in range(spec.n_generations):
    mode, temperature, llms = spec.schedule(i)
    prompt_schemas = spec.prompt_schemas(mode)
    spawn(population, islands, mode, temperature)

    generate_model_code(population, prompt_schemas.model, llms.model, mode, temperature)
    generate_param_est_code(population, prompt_schemas.param_est, llms.param_est)
    translate_to_jax(population, prompt_schemas.jax, llms.jax)

    score(population, islands, X_discover, X_eval, spec.scoring)

    islands = deduplicate(islands, population)
    islands = prune(islands, population, spec.evolution)
    islands = migrate(islands, population, spec.evolution, mode, temperature)
    log_census(islands)

score(population, islands, X_validate, X_eval, spec.scoring)
save(population, islands)
```

## Project Output Format

Each run saves a self-contained directory with three files:

```
task_spec.yaml
population.jsonl
island_records.json
```

- **task_spec.yaml** — Run setup, metadata, and status. Includes task name, config path, merged config sections (io, evolution, llms, scoring, project_params), prompt schemas, seed program source, git SHA + dirty flag, start/end time, run status (completed / failed / interrupted), and any error summary.

- **population.jsonl** — Main scientific output. One JSON object per Program with birth metadata (generation, island, batch, mode, temperature, parent indices, LLM), numpy and JAX code for model and parameter estimator, model name, discover/validate loss (init/final), parameter count, and eval fingerprint if saved.

- **island_records.json** — Island membership history. For each iteration, the set of program IDs active on each island:
```json
[
  [[0, 1, 2], [0, 1, 3]],  // generation 0
  [[0, 4],    [1, 5]]       // generation 1
]
```

### Logging and image feedback output

Each run should have one global log file:

```text
run.log
```

Logging is a human-readable execution trace, not algorithm state. Algorithm state remains in:

```text
task_spec.yaml
population.jsonl
island_records.json
```

#### CLI verbosity

Logging verbosity should be controlled from the CLI, e.g.

```bash
--log-level compact
--log-level code
--log-level prompts
```

#### `compact`

Default logging level. Log one summary per generation:

- generation number
- mode: explore/exploit
- temperature
- LLMs used
- number of programs spawned
- model generation success rate
- parameter-estimator success rate
- JAX translation success rate
- scoring success rate
- per-island size
- per-island best program id/loss/name
- global best discover loss
- elapsed time

#### `code`

Includes everything in `compact`, plus generated code for each program:

- model code
- parameter-estimator code
- JAX model code
- JAX parameter-estimator code
- parse/compile/translation failures

#### `prompts`

Includes everything in `code`, plus full LLM prompts:

- model prompts
- parameter-estimator prompts
- JAX translation prompts
- image prompt paths, if present

#### Image feedback artifacts

Prompt images should be saved separately from logs:

```text
image_feedback/
  gen_000/
    island_000/
      batch_000/
        prompt_image.png
```

Use zero-padded indices so file browsers sort correctly.

Path should correspond to program birth metadata:

```python
program.birth.generation
program.birth.island
program.birth.batch_index
```

#### Design rule

```text
population.jsonl      = program state
island_records.json   = island history
task_spec.yaml        = run setup + metadata
run.log               = readable execution trace
image_feedback/       = prompt image artifacts
```