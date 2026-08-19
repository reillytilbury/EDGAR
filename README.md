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
First clone the repository and checkout the `main` branch
```bash
git clone https://github.com/reillytilbury/EDGAR.git
git checkout main
``` 

To run the code, it is easiest to setup an environment which can be done in the following ways:
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

If need to setup a Google AI API key you can do so [here](https://aistudio.google.com/).

Add your Google (Gemini) API key to `.env` in the project root:

```bash
echo "GOOGLE_API_KEY=your_key_here" > .env
```

The key is loaded automatically at runtime via `python-dotenv`. You can also export it directly:

```bash
export GOOGLE_API_KEY=your_key_here
```
You can do the same for an `ANTHROPIC_API_KEY` if using an anthropic model.

To switch a project between providers, set `llms.provider` (`google` or `anthropic`) in its
`config.yaml` — this picks a sensible default model for each role. You can still override
individual roles (`model_llm`, `param_est_llm`, `jax_model_translator_llm`) with a specific
model of either provider; the API called is inferred from the model-name prefix.

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
edgar run program_databases/my_task/2026-05-06/14-32-10/task_spec.yaml
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
edgar resume program_databases/my_task/yyyy-mm-dd/hh-mm-ss
```
---

## Important directories
```text
- edgar/ #the code used to perform an edgar run
- projects/ #where the user should specify the configuration for their project
  # see "Setting up a new project"
- edgar-experimental/ #experimental features (for inspecting output of edgar runs)
- scripts/ #some useful scripts for debugging your project configuration
```

## Run Output

By default, each run writes to `program_databases/PROJECT_NAME/YYYY-MM-DD/HH-MM-SS/`:

```text
program_databases/
└── PROJECT_NAME/
    └── YYYY-MM-DD/
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

> **EXPERIMENTAL (work in progress): edgar-analyzer agent.**
  To in order to help with analyzing results we have configured a subagent for use with claude code or gemini cli in `agents/output`. See `agents/output/README.md` for further information on how to use this.
  The idea is to be able to use natural language to query results, e.g: **YOU:** Summarize the top models from the most recent run and identify common features which led to an improved score. 
  Additions to the instructions and tools in the mcp server are very welcome.

---

## GCP Cloud Runs

Run EDGAR on Google Cloud with `edgar launch-gcp` — one GPU VM per run, each building its
environment with `uv sync --frozen`, running `edgar run`, syncing results to a Cloud Storage
bucket, and self-deleting. API keys are stored in Secret Manager (never in the bucket) and
fetched by each VM's service account; no keys are sent for GCP itself.

```bash
# one-time
gcloud auth login && gcloud config set project <PROJECT_ID>
gcloud services enable compute.googleapis.com storage.googleapis.com secretmanager.googleapis.com
gcloud storage buckets create gs://<BUCKET> --location=<REGION>

# each launch
cp projects/gcp_launch.example.yaml gcp_launch.yaml     # gitignored; edit the gcp: block
uv run edgar launch-gcp gcp_launch.yaml --dry-run        # inspect; runs nothing
uv run edgar launch-gcp gcp_launch.yaml                  # launch
uv run edgar launch-gcp gcp_launch.yaml --fetch          # pull results -> program_databases/
```

See **[docs/source/gcp_cloud_runs.md](docs/source/gcp_cloud_runs.md)** for the full guide:
how it works (architecture, secrets, provenance), writing specs, monitoring sweeps, storage
cost and cleanup, and troubleshooting (GPU quota, spot stockouts, image families).

---

## Setting Up a New Project

> **EXPERIMENTAL (work in progress): start with the `data-loader-helper` agent.** The hardest part of a new project
> is deciding *what one sample and one trial are* — get it wrong and the loader trains and
> validates cleanly while silently testing a different claim. This repo ships an interactive
> helper that interviews you about your data and intended equation, works out the (sample, trial)
> mapping and the train/test/discover/validate splits, then writes `data_loader/load_data.py` for
> you. Run it before doing the manual steps below:
>
> ```bash
> claude '/data-loader-helper'
> ```
>
> **Not using Claude Code?** The agent is just a prompt — `.claude/skills/data-loader-helper/SKILL.md`
> (plus `questionnaire.md` and `design_log_template.md` in the same folder). Paste `SKILL.md` in as a
> system/instruction prompt to whatever assistant you use (Codex, Cursor, ChatGPT, etc.) and it will
> run the same interview, or just read it yourself as a design guide for the steps below.

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

### 8. Visualise the split (recommended)

Before launching a run, render the train/test/discover/validate partition and eyeball it — a
split that tests the wrong claim usually *looks* wrong the moment you plot it. This standalone
script works for **any** project with a `load_data` (you don't need the data-loader-helper
agent): it reads the loader, has Claude generate a project-tailored `plot_split`, runs the real
`load_data`, and renders the figure to `test_output/plot_split_test/`.

```bash
uv run python scripts/plot_data_split_prompt.py my_task   # or a path to config.yaml
```

It makes one real Anthropic API call to generate the plotting code (needs `ANTHROPIC_API_KEY`).

### 9. Run

```bash
edgar run projects/my_task/config.yaml
```

---
