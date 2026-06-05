# EDGAR: Equation Discovery with Graphical AI Reasoning

[![Tests](https://github.com/reillytilbury/EDGAR/actions/workflows/test.yml/badge.svg)](https://github.com/reillytilbury/EDGAR/actions/workflows/test.yml)

EDGAR is an evolutionary framework for discovering scientific equations using LLM-generated programs and parameter estimators.

Each run evolves a population of candidate models across multiple islands. The LLM generates numpy model code and parameter estimators; JAX-translated versions are then optimised via gradient descent. Programs are selected, pruned, and migrated between islands over many generations.

---

## Quickstart

### 1. Install
#### uv (recommended)
Run the command
```bash
uv sync
```
which will automatically setup the environment in the root folder.
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
This installs the `edgar` package in editable mode (deps are sourced from `requirements.txt` via `pyproject.toml`).
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

---

### 4. Dev setup

After cloning, install dev dependencies and register the pre-commit hook:

```bash
uv sync --group dev
pre-commit install
```

Verify your environment is correctly set up:
```bash
bash scripts/check_env.sh   # checks edgar imports + fake pipeline
bash scripts/check_api_keys.sh  # checks all LLM API keys work
```

When making a `git commit`, do the following
```bash
git add -u
make commit-check
# Returns status of pre-commit, files may need to be modified
git add -u
git commit -m 'a commit message'
```

Upon pushing to remote the following tests are run, and status displayed on github:
- All unit and integration pytests in tests except those with live llm calls.
- Pings google and anthropic LLMs to check they can be called.

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
python -m edgar.cli init-project my_task
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

### 5. Customise prompts (optional)

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

### 6. Validate

```bash
python -m edgar.cli validate my_task
```

### 7. Run

```bash
python -m edgar.cli run projects/my_task/config.yaml
```

---