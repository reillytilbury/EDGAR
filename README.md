# EDGAR: Equation Discovery with Graphical AI Reasoning

EDGAR is an evolutionary framework for discovering scientific equations using LLM-generated programs and parameter estimators.

Each run evolves a population of candidate models across multiple islands. The LLM generates numpy model code and parameter estimators; JAX-translated versions are then optimised via gradient descent. Programs are selected, pruned, and migrated between islands over many generations.

---

## Quickstart

### 1. Install

```bash
conda create -n edgar python=3.13 -y
conda activate edgar
pip install -r requirements.txt
```

### 2. Set API key

Add your Gemini API key to `.env` in the project root:

```bash
echo "GEMINI_API_KEY=your_key_here" > .env
```

The key is loaded automatically at runtime via `python-dotenv`. You can also export it directly:

```bash
export GEMINI_API_KEY=your_key_here
```

### 3. Run an experiment

```bash
edgar run projects/orientation_tuning/config.yaml
```

Control logging verbosity (default: `compact`):

```bash
edgar run projects/orientation_tuning/config.yaml --log-level code
edgar run projects/orientation_tuning/config.yaml --log-level prompts
```

Override config values on the command line:

```bash
edgar run projects/orientation_tuning/config.yaml --evolution.n_generations=5
edgar run projects/orientation_tuning/config.yaml --llms.model_llm=gemini-2.5-pro
```

Reproduce a previous run from its saved `task_spec.yaml`:

```bash
edgar run program_databases/05-06/14-32-10/task_spec.yaml
```

---

## Project Layout

```text
EDGAR-gamma/
├── projects/
│   ├── config_default.yaml          # Base defaults for all projects
│   ├── prompt_defaults.yaml         # Base prompt schemas
│   └── <task_name>/
│       ├── config.yaml              # Task-specific config overrides
│       ├── prompts.yaml             # (optional) Task-specific prompt overrides
│       ├── seed_programs/
│       │   ├── model1.py, model2.py
│       │   └── param_est1.py, param_est2.py
│       ├── data_loader/
│       │   └── load_data.py         # Must define: load_data(), loss_fn()
│       └── image_feedback/
│           └── plot.py              # Optional: plot_model_fits()
├── src/
│   ├── run.py                       # Main entry point
│   ├── cli.py                       # edgar CLI
│   ├── evolution/
│   │   ├── program.py               # Program dataclass
│   │   ├── population.py            # Population (append-only, JSONL persistence)
│   │   └── island.py                # seed, spawn, prune, deduplicate, migrate
│   ├── io/
│   │   ├── config.py                # Config class: from_yaml / from_taskspec
│   │   ├── task_spec.py             # TaskSpec: frozen run bundle
│   │   └── logging.py              # open_log, log_generation
│   ├── llm/
│   │   ├── generate.py              # generate_models, generate_param_ests, translate_programs
│   │   ├── llm_calling.py           # call_llm (pydantic-ai wrapper)
│   │   └── prompt_schema.py         # PromptSchema.build_prompt
│   └── scoring/
│       └── scoring.py               # score() with subprocess timeout
├── program_databases/               # Run outputs (gitignored)
├── .env                             # API keys (gitignored)
└── tests/
```

---

## Run Output

Each run writes to `program_databases/MM-DD/HH-MM-SS/`:

```text
program_databases/
└── MM-DD/
    └── HH-MM-SS/
        ├── task_spec.yaml          # Full config + git SHA + prompt schemas + seed code. Read-only.
        ├── population.jsonl        # All Programs — code, losses, params, lineage. Main scientific output.
        ├── island_census.jsonl     # Island membership at the end of each generation.
        ├── run.log                 # Human-readable execution trace.
        └── image_feedback/         # Only present if plot_fn is defined.
            └── gen_000/
                └── island_000/
                    └── batch_000/
                        └── image.png
```

---

## Setting Up a New Project

### 1. Scaffold

```bash
edgar init-project my_task
```

This creates:

```
projects/my_task/
├── config.yaml
├── seed_programs/model1.py, model2.py, param_est1.py, param_est2.py
├── data_loader/load_data.py
└── image_feedback/plot.py
```

Each file contains a stub with a docstring. Fill in the implementations.

### 2. Fill in `data_loader/load_data.py`

Must define two callables:

**`load_data(data_path, **kwargs) -> (X_discover, X_validate, X_eval)`**

Returns three splits:
- `X_discover = (X_disc_train, X_disc_test)` — seen by the LLM discovery loop
- `X_validate = (X_val_train, X_val_test)` — never seen during discovery
- `X_eval` — small fingerprint subset (dict of JAX arrays + `_sample_indices`)

All arrays should be JAX arrays. Data shape convention: `(n_samples, n_trials)` per key.

**`loss_fn(model_output, data) -> JAX array of shape (n_samples,)`**

Per-sample loss between model predictions and data.

### 3. Fill in seed programs

`model*.py` must define `def model(data, params):`
- `data`: dict of JAX arrays, one sample, e.g. `data['stimulus']` shape `(n_trials,)`
- `params`: dict of named scalars/arrays
- Returns predictions shape `(n_trials,)`
- Must have `model.DEFAULT_PARAMS = {"param_name": initial_value, ...}`

`param_est*.py` must define `def parameter_estimator(data):`
- Returns a parameter dict with the same keys as `model.DEFAULT_PARAMS`
- Keep it simple (no scipy.optimize / curve_fit)

### 4. Configure `config.yaml`

Override only what differs from `projects/config_default.yaml`. Minimum:

```yaml
io:
  data_path: /path/to/data.npy
```

Common overrides:

```yaml
project_params:
  my_threshold: 0.5   # kwargs passed to load_data()

evolution:
  n_generations: 20

llms:
  model_llm: gemini-2.5-pro
```

### 5. Validate

```bash
edgar validate my_task
```

### 6. Run

```bash
edgar run projects/my_task/config.yaml
```

---

## Architecture

### Config and TaskSpec

`Config` holds the plain-dict settings from YAML. `TaskSpec` wraps a `Config` plus loaded callables, seed programs, and git metadata.

```python
from src.io.config import Config
from src.io.task_spec import TaskSpec

# New run:
spec = TaskSpec.from_config(Config.from_yaml("projects/my_task/config.yaml"))

# Reproduce a previous run:
spec = TaskSpec.from_config(Config.from_taskspec("program_databases/05-06/14-32-10/task_spec.yaml"))
```

### Evolution loop (pseudocode)

```
X_discover, X_validate, X_eval = load_data(spec)
population = Population()
islands = seed(population, spec.seed_programs, n_islands)
translate_to_jax(population)
score(population, X_discover, X_eval, split="discover")

for gen in range(n_generations):
    mode, temperature, llms = spec.schedule(gen)
    spawn(population, islands, gen, mode, temperature)

    generate_models(population, prompt_schemas.model, llms.model, ...)
    generate_param_ests(population, prompt_schemas.param_est, llms.param_est)
    translate_to_jax(population, prompt_schemas.jax, llms.jax)

    score(population, X_discover, X_eval, split="discover")
    deduplicate(islands, population)
    prune(islands, population)
    migrate(islands, population, temperature)

score(population, X_validate, split="validate")
population.save("population.jsonl")
```

### Key classes

| Class | File | Purpose |
|-------|------|---------|
| `Config` | `src/io/config.py` | Plain-dict config bundle. `from_yaml` / `from_taskspec`. |
| `TaskSpec` | `src/io/task_spec.py` | Frozen run bundle. Adds callables, seeds, git state. |
| `Program` | `src/evolution/program.py` | One evolved candidate: code, losses, params, lineage. |
| `Population` | `src/evolution/population.py` | Append-only list with JSONL save/load. |
| `PromptSchema` | `src/llm/prompt_schema.py` | Prompt template. `build_prompt(mode, parents, config)`. |

---

## Configuration Reference

All defaults live in `projects/config_default.yaml`. Override any key in `projects/<task>/config.yaml`.

### `io`
| Key | Default | Description |
|-----|---------|-------------|
| `data_path` | — | Path to task data (required) |
| `save_path` | `program_databases` | Output base directory |

### `evolution`
| Key | Default | Description |
|-----|---------|-------------|
| `n_generations` | 12 | Total generations |
| `n_islands` | 8 | Number of independent island populations |
| `batch_size` | 6 | LLM calls per island per generation |
| `critical_population_size` | 12 | Max programs per island after pruning |
| `n_migrants` | 2 | Programs exchanged between islands per generation |
| `topology` | `[1,2,...,7,0]` | Ring topology: topology[i] is island i's migration target |
| `exploit_point` | 0.5 | Fraction of generations before switching explore→exploit |

### `llms`
| Key | Default | Description |
|-----|---------|-------------|
| `model_llm` | `gemini-2.5-flash` | LLM for model code. String or list (cycled by generation). |
| `param_est_llm` | `gemini-2.0-flash` | LLM for parameter estimator code |
| `jax_translator_llm` | `gemini-2.0-flash-lite` | LLM for JAX translation |
| `k_max` | 2 | Number of parent programs shown per prompt |
| `max_lines` | 50 | Max lines allowed in a parameter estimator response |
| `swear_words` | `['lstsq', 'scipy.optimize', ...]` | Banned fragments in generated code |

### `scoring`
| Key | Default | Description |
|-----|---------|-------------|
| `param_penalty_weight` | 0.01 | Complexity penalty per parameter |
| `timeout_s` | 120.0 | Wall-clock timeout per scoring run (seconds) |
| `gradient_descent.max_iter` | 1000 | Max gradient descent iterations |
| `gradient_descent.learning_rate` | 0.003 | Gradient descent learning rate |

### `project_params`
Arbitrary kwargs passed directly to `load_data()`. Use for task-specific settings like thresholds or random seeds.

---

## Logging Levels

| Level | Contents |
|-------|----------|
| `compact` | One summary per generation: mode, temperature, LLMs, success rates, per-island best, global best, elapsed time |
| `code` | Everything in compact + generated code for each program |
| `prompts` | Everything in code + full LLM prompts and image paths |

---

## Implementation notes

### Scoring subprocesses

Each program is scored in a fresh subprocess (`scoring.py`). This is intentional: JAX-compiled code from LLM output can hang, OOM, or segfault in ways that are unrecoverable in-process. The subprocess is killed after `scoring.timeout_s` seconds, making timeouts reliable regardless of what the generated code does.

### LLM failure handling

`generate_models`, `generate_param_ests`, and `translate_programs` all use `asyncio.gather(..., return_exceptions=True)`. Failed LLM calls are caught and silently dropped — the corresponding program just keeps `None` code fields and gets filtered out at scoring time. Check `run.log` at `code` or `prompts` verbosity to see failures.

---

## Remaining work

### Tests (highest priority)
- Tests for `src/evolution` — island operations, population, program
- Tests for `src/io` — config, task_spec
- Tests for `src/llm`
- Integration test for `run.py` (small end-to-end run)
- Wire tests to GitHub Actions CI

### Core scoring
- **Separate input/target keys** — currently the full data dict (including target values) is passed to `model(data, params)`, so models can cheat by reading the target directly. Need `input_keys` / `target_keys` in config to split before passing.
- **Variable-length trials** — assumes rectangular data (all samples same `n_trials`); would need list/dict of variable-length arrays.
- **`_eval_fingerprint` design** — sample indices are smuggled into the data dict via `_sample_indices`. Flagged as a design smell; not urgent.

### LLM
- Consider unifying `generate_models`, `generate_param_ests`, `translate_programs` into a single `generate(population, spec, prompt_type, ...)` — deferred until there's a fourth generation type.

---

## Tests

```bash
python -m pytest tests -q
```

Currently only scoring tests exist (`src/scoring/tests/test_scoring.py`).
