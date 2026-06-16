# EDGAR: Equation Discovery with Graphical AI Reasoning

[![Tests](https://github.com/reillytilbury/EDGAR/actions/workflows/test.yml/badge.svg)](https://github.com/reillytilbury/EDGAR/actions/workflows/test.yml)

EDGAR is an evolutionary framework for discovering scientific equations using LLM-generated programs and parameter estimators.

Each run evolves a population of candidate models across multiple islands. The LLM generates numpy model code and parameter estimators; JAX-translated versions are then optimised via gradient descent. Programs are selected, pruned, and migrated between islands over many generations.

---

## Prerequisites

The recommended way to manage dependencies and environments is [uv](https://docs.astral.sh/uv/).

To install `uv` on Linux or macOS:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

---

## Quickstart

### 1. Install
#### uv (recommended)
Run the command
```bash
uv sync
```
from the repo root which will automatically setup the environment there.
Now any commands will be run in this environment when using the prefix `uv run`, e.g
```bash
uv run edgar test projects/synthetic_data/config.yaml
```

#### Conda + pip
```bash
conda create -n edgar python=3.13 -y
conda activate edgar
pip install -e .
```
This installs the `edgar` package in editable mode.
`import edgar` then works from any cwd / IDE cell without `sys.path` hacks.

To verify your environment is setup correctly run the script
```bash
bash scripts/check_env.sh
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
You can do the same for an `ANTHROPIC_API_KEY` if using an anthropic model.

Verify your API key is configured correctly by running
```bash
bash scripts/check_api_keys.sh
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
edgar projects/orientation_tuning/config.yaml --llms.model_llm=gemini-2.5-pro
```

Run a quick test with reduced settings (1 generation, 2 islands, batch size 2) to verify the pipeline is wired correctly

```bash
edgar test projects/orientation_tuning/config.yaml
```

Reproduce a previous run from its saved `task_spec.yaml`:

```bash
edgar run program_databases/05-06/14-32-10/task_spec.yaml
```

To launch the dashboard to view in progress and finished experiments:
```bash
edgar dashboard
```
By default this allows access to data saved in `program_databases`. If the data you require is saved elsewhere, do
```bash
edgar dashboard {data_directory}
```

A run which failed can be resume via, for example:
```bash
edgar resume program_databases/mm-dd/hh-mm-ss
```
---

## Run Output

By default, each run writes to `program_databases/MM-DD/HH-MM-SS/`:

```text
program_databases/
└── MM-DD/
    └── HH-MM-SS/
        ├── task_spec.yaml          # Full config + git SHA + prompt schemas + seed code. Read-only.
        ├── population.jsonl        # All Programs — code, losses, params, lineage. Main scientific output.
        ├── island_census.jsonl     # Island membership at the end of each generation.
        ├── metrics.jsonl           # Timing, token and retry statistics for the various parts of the algorithm.
        ├── status.json             # Overall status of the run, read by the dashboard.
        ├── run.log                 # Human-readable execution trace.
        └── image_feedback/         # Only present if plot_fn is defined, image_feedback prompt shown to LLM.
            └── gen_000/
                └── island_000/
                    └── batch_000/
                        └── image.png
        └── image_fits/             # Only present if plot_fn is defined, plots each program before and after parameter optimization.
            └── P0000.png 
```

---

## Setting Up a New Project

### 1. Scaffold

```bash
edgar init-project my_task
```

This creates:

```text
projects/my_task/
├── config.yaml
├── seed_programs/
│   ├── model1.py
│   ├── model2.py
│   ├── param_est1.py
│   └── param_est2.py
├── data_loader/
│   └── load_data.py
└── image_feedback/
    └── plot.py
```

Each file contains a stub with a docstring. Fill in the implementations.

### 2. Fill in `data_loader/load_data.py`

Must define two callables:

**`load_data(data_path, **kwargs) -> (X_discover, X_validate, X_eval)`**

Returns three splits:
- `X_discover = (X_disc_train, X_disc_test)` — seen by the LLM discovery loop.
- `X_validate = (X_val_train, X_val_test)` — never seen during discovery.
- `X_eval` (dict) — small subset of `X_disc_train` used for generating model fingerprints.

`X_eval` must be a dictionary containing:
- Feature/response JAX arrays (same keys as other splits).
- `_sample_indices`: a NumPy array of integer indices indicating which samples from `X_disc_train` are included in `X_eval`.

All data arrays should be JAX arrays. Data shape convention: `(n_samples, n_trials)` per key.

**`loss_fn(model_output, data) -> JAX array of shape (n_samples,)`**

Per-sample loss between model predictions (`model_output`) and data (`data`).

### 3. Fill in seed programs

`model*.py` must define `def model(data, params):`
- `data`: dict of JAX arrays for one sample, e.g. `data['stimulus']` shape `(n_trials,)`.
- `params`: dict of named scalars/arrays.
- Returns predictions shape `(n_trials,)`.
- Must have `model.DEFAULT_PARAMS = {"param_name": initial_value, ...}`

`param_est*.py` must define `def parameter_estimator(data):`
- Returns a parameter dict with the same keys as `model.DEFAULT_PARAMS`.
- Keep it simple (no `scipy.optimize` or `curve_fit`).

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

### 5. Customise prompts

Create `projects/<task>/prompts.yaml` to override the defaults in `projects/prompt_defaults.yaml`. The two files are **deep-merged**, so you only need to include the fields you want to change — everything else is inherited.

String fields (`base`, `explore`, `code_guidelines`, etc.) are replaced entirely when specified. **List fields (`config_vars`, `parent_vars`) are also replaced entirely** — if you add a new variable, you must re-list all of them.

`explore` and `exploit` can be set to `null` (or omitted entirely) if you don't need a mode-specific section — the JAX translator and parameter estimator prompts typically leave both as `null`.

Example — override only the `base` and `code_guidelines` for model generation:

```yaml
model:
  base: |
    You are an AI scientist modelling orientation tuning in visual cortex.
    Below are {k_max} neuron models sorted from worst to best.
    Create a new model with lower loss than all of them.
  code_guidelines: |
    * Model signature: def model(data, params):
    * data has keys "stimulus" (radians) and "response".
    * Clip free parameters to biologically plausible ranges at the top of the function.
```

All other `model` fields (explore, exploit, docstring_guidelines, parent_detail_template, config_vars, parent_vars) are inherited from the defaults unchanged.

### 6. Fill in `image_feedback/plot.py` (Optional)

Must define:

**`plot_model_fits(data, parent_programs, save_path="")`**

- `data`: `X_disc_train` dictionary of JAX arrays.
- `parent_programs`: list of `Program` objects to visualize.
- `save_path`: file path to save the generated figure.

If this file/function is left as a `pass` stub, no images will be generated or provided as LLM feedback.

### 7. Validate

```bash
edgar validate my_task
```

### 8. Run

```bash
edgar run projects/my_task/config.yaml
```

---
