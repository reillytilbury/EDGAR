# EDGAR-gamma: Equation Discovery with Graphical AI Reasoning

EDGAR-gamma is an evolutionary framework for discovering equations with LLM-generated programs and parameter estimators.

## Lightweight Quickstart (5 minutes)

### 1) Install

```bash
conda create -n edgar python=3.13 -y
conda activate edgar
pip install -r requirements.txt
```

### 2) Set API key

```bash
export GOOGLE_API_KEY=your_gemini_api_key
```

### 3) Run the shortest end-to-end experiment

```bash
python -m run --config experiments/synthetic_data/override_config.yaml --test_mode
```

### 4) Run a full example

```bash
python -m run --config experiments/orientation_tuning/override_config.yaml
```

Run outputs are written to `program_databases/MM-DD/HH-MM-SS/`.
Most useful artifacts:
- `combined/programs_db.csv`
- `combined/top_model_fits.png`
- `combined/train_vs_test_loss.png`
- `hypothesis_engine.log`

## New Experiment TODO (Minimal)

If you are setting up a new task, use this as a checklist first. Then use the detailed sections below for examples.

- [ ] Create `experiments/your_task_name/`.
- [ ] Add `experiments/your_task_name/override_config.yaml`.
- [ ] In `override_config.yaml`, define required keys: `task`, `model_name`, `load_and_process_data_fn`, `create_train_test_sample_split_fn`, `create_train_test_trial_split_fn`, `seed_programs.module`, `seed_programs.function_seeds` (2 names), `seed_programs.parameter_estimator_seeds` (2 names), `data_processing_params`, `inputs`, and `experiment_params` (or rely on defaults from `experiments/DEFAULT/config.yaml`).
- [ ] Add `experiments/your_task_name/seed_programs.py` with exactly 2 model functions (`def ...(X, ...)`) and 2 parameter estimator functions (`def ...(X, y)`).
- [ ] Add `experiments/your_task_name/data_parser.py` with `load_and_process_data(...)`, `create_train_test_sample_split(...)`, and `create_train_test_trial_split(...)`.
- [ ] Add `experiments/your_task_name/diagnostics.py` with `select_evaluation_points(...)` and `plot_model_fits(...)` (required if you want diagnostics/image feedback).
- [ ] Run:

```bash
python -m run --config experiments/your_task_name/override_config.yaml
```

## How Configuration Works

`run.py` merges:
- Base defaults: `experiments/DEFAULT/config.yaml`
- Your experiment override: `--config experiments/<task>/override_config.yaml`

This repository does not include a root `config.yaml`, so always pass `--config`.

## Project Layout

```text
EDGAR-gamma/
├── experiments/
│   ├── DEFAULT/
│   │   └── config.yaml
│   ├── orientation_tuning/
│   │   ├── override_config.yaml
│   │   ├── seed_programs.py
│   │   ├── data_parser.py
│   │   └── diagnostics.py
│   └── ...
├── src/
│   ├── hypothesis_engine.py
│   ├── diagnostics_manager.py
│   ├── prompt_manager.py
│   └── ...
├── run.py
└── tests/
```

## Create a New Experiment

Use this checklist to get from zero to a runnable experiment quickly.

### 1) Create a task folder

```bash
mkdir -p experiments/your_task_name
```

### 2) Add `seed_programs.py`

Minimum requirement: exactly 2 model seeds and exactly 2 parameter estimator seeds.

```python
import numpy as np

def neuron_model_1(X, a=1.0, b=0.0):
    x = X[0]
    return a * np.cos(x) + b


def neuron_model_2(X, a=1.0, b=0.0, c=0.1):
    x = X[0]
    return a * np.cos(x - c) + b


def parameter_estimator_1(X, y):
    return np.array([np.std(y), np.mean(y)])


def parameter_estimator_2(X, y):
    return np.array([np.std(y), np.mean(y), 0.0])
```

Seed rules:
- Model signature is `def ...(X, ...)`
- `X` shape is `(n_features, n_trials)`
- Access features by index (`X[0]`, `X[1]`, ...)
- Keep estimators simple heuristics (avoid optimize/curve_fit-style solvers)
- Give all free parameters default values

### 3) Add `data_parser.py`

Return a dict with model-ready inputs and outputs. Canonical shape is:
- `inputs`: `(n_samples, n_features, n_trials)`
- `outputs`: `(n_samples, n_targets, n_trials)`

### Features vs Targets (with examples)

- `n_features` is how many input variables each model uses per trial.
- `n_targets` is how many outputs the model predicts per trial.
- In model code, `X` has shape `(n_features, n_trials)`, so each feature is `X[i]`.

Examples:
- Orientation tuning: 1 feature (`theta`), 1 target (single-cell firing rate).
    - Shapes:
        - `X`: `(1, n_trials)`
        - model output: `(n_trials,)`
- Grid-cell tuning with position input: 2 features (`x`, `y`), 1 target (single-cell firing rate).
    - Shapes:
        - `X`: `(2, n_trials)`
        - model output: `(n_trials,)`
- Multi-cell prediction: 2 features (`x`, `y`), `N` targets (firing rates for `N` cells).
    - Shapes:
        - `X`: `(2, n_trials)`
        - model output: `(N, n_trials)` (or equivalent canonical outputs tensor with `n_targets=N`)

Current limitation:
- Data is represented as a plain `dict[str, np.ndarray]` where all values share the same last dimension (n_trials).

```python
import numpy as np
import jax
import jax.numpy as jnp


def load_and_process_data(data_path, **kwargs):
    data = np.load(data_path, allow_pickle=True).item()

    # Example arrays (adapt to your dataset)
    # response: (n_samples, n_trials)
    # stimulus: (n_trials,)
    response = data["response"]
    stimulus = data["stimulus"]

    n_samples, n_trials = response.shape
    # Broadcast stimulus to (n_samples, n_trials)
    stimulus_tiled = np.tile(stimulus.reshape(1, -1), (n_samples, 1))

    return {'stimulus': stimulus_tiled, 'response': response}


def create_train_test_sample_split(n_samples, training_sample_ratio=0.5, random_seed=0):
    key = jax.random.PRNGKey(random_seed)
    n_train = int(n_samples * training_sample_ratio)
    idx = jax.random.permutation(key, jnp.arange(n_samples))
    return idx[:n_train], idx[n_train:]


def create_train_test_trial_split(n_trials, random_seed=0):
    rng = np.random.default_rng(random_seed)
    idx = rng.permutation(n_trials)
    n_train = n_trials // 2
    return idx[:n_train], idx[n_train:]
```

### 4) Add `diagnostics.py` (recommended)

Required functions:
- `select_evaluation_points(inputs, n_points=100, random_seed=0, **kwargs)`
- `plot_model_fits(plot_data, ...)`

Fastest path: copy and adapt `experiments/DEFAULT/diagnostics.py`.

### 5) Add `override_config.yaml`

```yaml
task: your_task_name
load_and_process_data_fn: experiments.your_task_name.data_parser.load_and_process_data
create_train_test_sample_split_fn: experiments.your_task_name.data_parser.create_train_test_sample_split
create_train_test_trial_split_fn: experiments.your_task_name.data_parser.create_train_test_trial_split

diagnostics_path: experiments/your_task_name

seed_programs:
  module: experiments.your_task_name.seed_programs
  function_seeds:
    - neuron_model_1
    - neuron_model_2
  parameter_estimator_seeds:
    - parameter_estimator_1
    - parameter_estimator_2

data_processing_params:
  data_path: /path/to/data.npy

inputs:
  - name: stimulus
    description: "Primary input variable"

experiment_params:
  num_runs: 1
  n_iterations: 12
  time_limit: 60
  n_islands: 8
  batch_size: 6
  max_iter: 1000
  critical_population_size: 12
  min_wise_population_size: 0
  n_migrants: 2
  fit_params: true
  tol: 1e-6
  exploit_point: 0.5
  learning_rate: 3e-3
  param_penalty_weight: 0.01
  FAILED_PROGRAM_COST: .inf
  exploration_topology: [1, 2, 3, 4, 5, 6, 7, 0]
  exploitation_topology: [1, 2, 3, 4, 5, 6, 7, 0]
  model_llm: gemini-2.5-flash
  param_est_llm: gemini-2.0-flash
  jax_translator_llm: gemini-2.0-flash-lite
```

Then run:

```bash
python -m run --config experiments/your_task_name/override_config.yaml
```

## Useful Commands

Run all tests:

```bash
python -m pytest tests -q
```

Run one module:

```bash
python -m pytest tests/test_orientation_tuning_seed_loss_regression.py -q
```

## Common Failure Modes

- `Config file not found`: you forgot `--config .../override_config.yaml`
- `There must be exactly 2 ... seeds`: seed list lengths are not exactly 2
- `FAILED_PROGRAM_COST` for everything: seed code or data parser is broken
- `NonConcreteBooleanIndexError`: JAX-incompatible code (boolean indexing/control flow)
- Missing diagnostics plots: `diagnostics_path` invalid or diagnostics module missing required functions

## Citation

```bibtex
@article{edgar-gamma,
  title={EDGAR-gamma: Equation Discovery with Graphical AI Reasoning},
  author={Your Name},
  year={2026}
}
```

## License

[]

## Contributing

[]
