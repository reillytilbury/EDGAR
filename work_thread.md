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

## TODO

### Data loading
- [x] Change signature of all project `load_data` functions to output `X_discover`, `X_validate`, `X_eval` using shared split logic; remove per-project `train_test_split` functions
- [ ] Ensure all data is in JAX format
- [x] Integrate Dabin's drop trials plan
- [x] Need a way of saving hyperparameters/config in metadata-only format + code used to load + seed islands

### Core scoring
- [ ] Rethink `_eval_fingerprint` in scoring.py — current approach needs refinement
- [ ] Support variable-length trials per sample — currently assumes rectangular data (all samples have same n_trials); would need list/dict of variable-length arrays instead
- [ ] Separate input/target keys in data — prevent models from cheating by accessing target values. Add `input_keys` and `target_keys` to config, split data before passing to model_fn

### LLM
- [ ] Integrate image prompt into LLMs: add `image_dir` field to Program object, save images to disk

### CLI + logging
- [ ] Implement actual logging (monitoring currently just computes metrics) — see spec below
- [ ] Add CLI verbosity flag (`--log-level compact/code/prompts`) — see spec below
- [ ] **cli.py**: better validation — look for actual functions instead of just files

### Utils removal
- [ ] Remove `src/utils.py` — migrate helpers to appropriate modules (`src/io`, `src/scoring`, etc.) and update all project `load_data.py` and `image_feedback/plot.py` imports

### Tests
- [ ] Tests for `src/evolution` (island operations, population, program)
- [ ] Tests for `src/io` (config, task_spec, output_dirs)
- [ ] Tests for `src/llm`
- [ ] Integration test for `run.py` (small end-to-end run)
- [ ] Wire tests to GitHub Actions CI with status badges in README

### Documentation
- [ ] Update README.md to reflect refactored architecture
- [ ] Add documentation describing the refactors (what changed and why)

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

for i in range(spec.n_iterations):
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
  [[0, 1, 2], [0, 1, 3]],  // iteration 0
  [[0, 4],    [1, 5]]       // iteration 1
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