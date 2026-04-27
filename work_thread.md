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

## Refactoring issues

### src/llm/candidates.py
- Outdated: references old prompt formatting system, wrong import paths
- Needs rewrite once prompt system is finalized

### src/monitoring/
- **family_tree.py, progress_monitor.py**: Should accept `Population` object instead of JSONL file path. Population becomes single source of truth.
  - Eliminates separate generation JSONL
  - Reconstructs prompts post-hoc via `build_prompt` + parent_ids + mode
  - Sidebar shows `model_code`/`param_est_code` instead of raw LLM responses

### src/hypothesis_engine.py
- Still uses old DataFrame-based island API
- Not migrated to new `island.py` or `Program`/`Population` abstractions

### Architecture
- Consolidate `Program` dataclass as single candidate representation (currently ~4 variants)
- Make `Population` the persistence layer (replace JSONL)
- Clean config routing: modules only receive their subsection (`config["scoring"]`, `config["llms"]`, etc.)

### Naming
- Sample/trial split overloaded: `train_sample`/`test_sample` (discovery vs eval) conflicts with `train`/`test` (GD splits)
- Standard terminology doesn't fit this structure. Parked for later clarification.

---

## Outstanding Issues

### Initial config + data loading
- Loss doesn't belong in `load_data.py`
- Need to integrate Dabin's drop trials plan
- Need to introduce eval points for function fingerprinting
- **cli.py**: need better validation — look for actual functions instead of just files
- Logging verbosity should be set as CLI command
- **Data summary** still mentions trials even though we want to move towards data structs where concepts of trials don't exist
- **Data summary** is very messy
- **paths** module: not sure what it does, also quite messy, unclear if needed
- Need a way of saving hyperparameters/config in metadata-only format + code used to load + seed islands

### Monitoring + logging
- Monitoring needs to read from population records
- `diagnostic.py` is dead code
- `logging.py` is populating JSON, so also dead code?
- `logging.py` is not actually logging, just populating json. We should actually introduce some logging code
- Remaining code in this section: `family_tree`, `io`, `progress_monitor` (all confusing + unstructured)

### LLM
- Need test

### Evolution
- Need tests for island operations

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
