# EDGAR-gamma: Evolutionary Discovery of Generative AI-assisted Research

A framework for automated equation discovery using LLMs within an evolutionary algorithm.

## Overview

EDGAR-gamma uses an island-based genetic algorithm to evolve neuroscience models, leveraging large language models (LLMs) to generate candidate programs and parameter estimators. The system explores the hypothesis space through multiple populations (islands) that exchange successful models via migration.

## Installation

```bash
# Create conda environment
conda create -n edgar python=3.13
conda activate edgar

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

```bash
# Run with test mode (shorter iterations for testing)
python -m run --test_mode

# Run full experiment
python -m run
```

Results are saved to `program_databases/MM-DD/HH-MM-SS/` with:
- Individual island results in `island_{i}/`
- Combined best programs in `combined/`
- Diagnostic plots in `image_feedback/`
- Iteration snapshots in `iteration_updates/`

## Project Structure

```
EDGAR-gamma/
├── config/                    # Configuration files
│   ├── experiment.yaml        # Experiment parameters and seed programs
│   ├── data.yaml             # Data loading configuration
│   └── prompts.yaml          # LLM prompt templates
├── experiments/              # Task-specific implementations
│   └── orientation_tuning/   # Example task
│       ├── seed_programs.py  # Initial model implementations
│       └── data_parser.py    # Data loading functions
├── src/                      # Core framework code
│   ├── hypothesis_engine.py  # Main evolution loop
│   ├── genetic_helpers.py    # Island operations (migration, pruning)
│   ├── data_structures.py    # Inputs class for multi-input support
│   ├── prompt_manager.py     # LLM prompt generation
│   ├── llm_helper.py         # LLM API interactions
│   ├── diagnostic.py         # Visualization tools
│   └── utils.py              # Utility functions
└── run.py                    # Entry point
```

## Setting Up a New Experiment

### 1. Create Task Directory

```bash
mkdir -p experiments/your_task_name
```

### 2. Implement Seed Programs (`experiments/your_task_name/seed_programs.py`)

You need to provide 2 seed models with corresponding parameter estimators (NumPy only):

```python
import numpy as np

# NumPy version (used for parameter estimation)
def neuron_model_1(X, amplitude=1.0, baseline=0.0):
    """
    Simple model description.
    
    Args:
        X: Input array with shape (n_features, n_trials).
           X[0] is the primary stimulus (e.g., orientation angles).
        amplitude: Response amplitude
        baseline: Baseline firing rate
    
    Returns:
        Predicted firing rate, shape (n_trials,)
    """
    theta = X[0]  # Extract first input
    return amplitude * np.cos(theta) + baseline

# Parameter estimator
def parameter_estimator_1(X, response):
    """
    Estimate parameters from data.
    
    Args:
        X: Input array with shape (n_features, n_trials).
           X[0] is the primary stimulus.
        response: Observed responses for a single cell, shape (n_trials,)
    
    Returns:
        np.ndarray: Estimated parameters [amplitude, baseline]
    """
    theta = X[0]  # Extract first input
    baseline = np.mean(response)
    amplitude = np.std(response)
    return np.array([amplitude, baseline])

# Implement neuron_model_2, parameter_estimator_2
# ...
```

**Important constraints:**
- **Function signature**: Models must accept `X` as first argument with shape `(n_features, n_trials)`
- **Input access**: Use index-based access like `theta = X[0]`, `contrast = X[1]`
- Parameter estimators must be simple heuristics (no scipy.optimize, curve_fit, etc.)
- Seed JAX versions are generated automatically via the JAX translator prompt when optimization starts
- All parameters must have default values

### 3. Implement Data Loader (`experiments/your_task_name/data_parser.py`)

```python
import numpy as np
import jax.numpy as jnp

def load_and_process_data(data_path, conc_thresh=0.55, activity_thresh=0.4):
    """
    Load and preprocess data for the task.
    
    Args:
        data_path: Path to data file
        conc_thresh: Concentration threshold for cell selection
        activity_thresh: Activity threshold for cell selection
    
    Returns:
        dict with keys:
            - 'response': jnp.ndarray of shape (n_cells, n_trials)
            - 'angles': jnp.ndarray of shape (n_cells, n_trials)
            - 'good_cells': np.ndarray of selected cell indices
            - 'n_good_cells': int, number of selected cells
    """
    # Load your data
    data = np.load(data_path, allow_pickle=True).item()
    
    # Process data
    response = data['response']  # (n_cells, n_trials)
    stimuli = data['stimuli']    # (n_trials,) or (n_cells, n_trials)
    
    # Filter cells based on your criteria
    good_cells = np.where((response.mean(axis=1) > activity_thresh))[0]
    
    # Create Inputs object (supports multiple Inputs)
    n_good_cells = len(good_cells)
    n_trials = response.shape[1]
    
    # Single input example: expand stimuli to (n_cells, 1, n_trials)
    if stimuli.ndim == 1:
        stimuli_expanded = np.tile(stimuli.reshape(1, 1, -1), (n_good_cells, 1, 1))
    else:
        stimuli_expanded = stimuli[good_cells][:, np.newaxis, :]
    
    inputs = Inputs.from_array(
        stimuli_expanded,
        names=input_names or ['theta']
    )
    
    # Return with both new and legacy formats
    return {
        'response': jnp.array(response[good_cells]),
        'inputs': inputs,  # New format
        'angles': jnp.array(stimuli_expanded[:, 0, :]),  # Deprecated, for backward compat
        'good_cells': good_cells,
        'n_good_cells': n_good_cells
    }
```

#### Multi-Input Support

The framework supports models with multiple input variables (e.g., orientation + contrast):

```python
# In data_parser.py - create multi-input data
theta = data['orientation']  # (n_trials,)
contrast = data['contrast']  # (n_trials,)

# Stack into (n_cells, n_features, n_trials)
inputs_array = np.stack([
    np.tile(theta, (n_cells, 1)),
    np.tile(contrast, (n_cells, 1))
], axis=1)

inputs = Inputs.from_array(
    inputs_array,
    names=['theta', 'contrast']
)
```

```python
# In seed_programs.py - access multiple inputs
def neuron_model_multi(X, theta_pref=0.0, contrast_gain=1.0, baseline=0.0):
    theta = X[0]     # First nput: orientation
    contrast = X[1]  # Second input: contrast
    
    tuning = np.cos(theta - theta_pref)
    return baseline + contrast_gain * contrast * tuning
```


### 4. Configure `config/experiment.yaml`

```yaml
task: your_task_name  # Must match directory name

seed_programs:
  module: experiments.your_task_name.seed_programs
  function_seeds:
    - neuron_model_1
    - neuron_model_2
  parameter_estimator_seeds:
    - parameter_estimator_1
    - parameter_estimator_2

experiment_params:
  n_iterations: 9           # Number of evolution cycles
  time_limit: 60            # Max runtime in minutes
  k_max: 2                  # Number of parent models for crossover
  n_islands: 8              # Number of parallel populations
  batch_size: 6             # Programs generated per island per iteration
  max_iter: 1000            # Max optimization iterations per program
  critical_population_size: 12  # Max programs per island
  min_wise_population_size: 0   # Reserved slots for large LLM programs
  n_migrants: 2             # Programs migrating between islands
  fit_params: true          # Whether to optimize parameters
  tol: 1e-6                 # Optimization tolerance
  exploit_point: 0.5        # Fraction of iterations in explore mode
  param_penalty_weight: 0.01  # Complexity penalty
  FAILED_PROGRAM_COST: inf  # Loss for failed programs
  use_image_feedback: true  # Generate diagnostic plots for LLM
  use_param_estimator: true # Use parameter estimators vs random init
  
  # Migration topology (which island each island sends migrants to)
  exploration_topology: [1, 2, 3, 4, 5, 6, 7, 0]  # Ring topology
  exploitation_topology: [1, 2, 3, 4, 5, 6, 7, 0]
  
  # LLM configuration
  tiny_lm_name: gemini-2.0-flash-lite   # For JAX translation
  little_lm_name: gemini-2.0-flash      # For most generations
  large_lm_name: gemini-2.5-flash       # For periodic deep search
  use_large_every: 3                     # Use large LLM every N iterations
  
  # Training parameters
  training_ratio: 0.5       # Fraction of cells for training (rest for test)
  conc_thresh: 0.55         # Cell selection threshold
  activity_thresh: 0.4      # Cell selection threshold
```

### 5. Configure `config/data.yaml`

```yaml
task: your_task_name
load_and_process_data_fn: experiments.your_task_name.data_parser.load_and_process_data
data_path: /path/to/your/data.npy

# Cell selection thresholds
activity_threshold: 0.4
conc_threshold: 0.55

# Define inputs (names used in models)
inputs:
  - name: theta
    description: "Stimulus orientation angle in radians"
  # Add more inputs as needed:
  # - name: contrast
  #   description: "Stimulus contrast level"
```

### 6. Customize Prompts in `config/prompts.yaml`

The prompts control how LLMs generate code. Key sections:

- `program_prompt`: Instructions for generating neuron models
  - `base`: Core instructions
  - `explore`: Creative exploration mode
  - `exploit`: Refinement mode
  - `image_analysis`: Instructions for interpreting diagnostic plots
  - `code_guidelines`: Code constraints and JAX compatibility rules
  - `function_signature`: Required function signature format (X with shape `(n_features, n_trials)`)
  - `docstring_guidelines`: Documentation format

- `parameter_estimator`: Instructions for parameter estimation functions
  - Includes `function_signature` for the `(X, spike_counts)` format
- `jax_translator_prompt`: Instructions for NumPy → JAX conversion

**Important**: Maintain the variable placeholders like `{k}`, `{next_version}`, `{max_lines}`, etc.

## Running Experiments

### Basic Usage

```bash
# Standard run
python -m run

# Test mode (reduced iterations for quick validation)
python -m run --test_mode
```

### Environment Variables

Create a `.env` file with your API keys:

```bash
GOOGLE_API_KEY=your_gemini_api_key
```

### Monitoring Progress

The system provides real-time feedback:
- Progress bars show iteration completion
- Console output displays loss values and success rates
- Logs are written to `program_databases/MM-DD/HH-MM-SS/hypothesis_engine.log`

### Analyzing Results

After completion, check:

1. **Best programs**: `program_databases/MM-DD/HH-MM-SS/combined/programs_db.csv`
   - Sorted by test loss
   - Contains code, parameters, and genealogy

2. **Diagnostic plots**: `program_databases/MM-DD/HH-MM-SS/combined/top_model_fits.png`
   - Visual comparison of top models vs data

3. **Learning curves**: `program_databases/MM-DD/HH-MM-SS/combined/train_vs_test_loss.png`
   - Track overfitting and convergence

4. **Per-iteration snapshots**: `program_databases/MM-DD/HH-MM-SS/iteration_updates/iteration_{i}/`
   - Monitor evolution progress

## Key Parameters Explained

### Evolution Parameters

- **n_iterations**: More iterations = better solutions but longer runtime
- **n_islands**: More islands = better exploration but more computation
- **batch_size**: Programs per island per iteration (higher = more diverse but slower)
- **critical_population_size**: Island capacity (higher = more diversity, slower migration)
- **n_migrants**: Programs exchanged between islands (higher = faster convergence, less diversity)
- **exploit_point**: Fraction of iterations in explore mode (0.5 = half explore, half exploit)

### LLM Strategy

- **use_large_every**: Use expensive large LLM every N iterations for breakthrough ideas
- **temperature**: Dynamically adjusted, controls creativity vs refinement

### Optimization

- **max_iter**: Gradient descent iterations (higher = better fit but slower)
- **tol**: Convergence threshold (lower = more precise but slower)
- **fit_params**: Set false to only use parameter estimator without gradient descent

## Troubleshooting

### Common Issues

**"NonConcreteBooleanIndexError"**
- Your seed programs or LLM-generated code uses boolean indexing
- Fix: Replace `array[condition]` with `jnp.where(condition, true_val, false_val)`
- Update prompts to emphasize JAX constraints

**"Module not found"**
- Check that `task` name in YAML matches directory name exactly
- Ensure `__init__.py` files exist if using nested packages

**Programs all fail with FAILED_PROGRAM_COST**
- Seed programs may have bugs
- Check logs in `hypothesis_engine.log` for specific errors
- Validate seed programs can run on your data manually

**Low success rate**
- LLM may be generating invalid code
- Review prompts in `config/prompts.yaml`
- Check that `code_guidelines` are clear and comprehensive
- Reduce `batch_size` to focus on quality over quantity

**Out of memory**
- Reduce `n_islands`, `batch_size`, or number of cells
- Use smaller models (gemini-2.0-flash-lite)

## Advanced Features

### Custom Loss Functions

Edit `src/loss_functions.py` to add your loss function, then reference it in `src/hypothesis_engine.py`.

### Custom Migration Topologies

Modify `exploration_topology` and `exploitation_topology` in `experiment.yaml`:
- Ring: `[1, 2, 3, ..., 0]`
- Star: `[0, 0, 0, ...]` (all migrate to island 0)
- Random: `[randint(0, n_islands-1) for _ in range(n_islands)]`

### Validation Testing

```bash
# Validate configuration files
python -m pytest config/test.py

# Run specific tests
python -m pytest tests.py::test_function_name
```

## Citation

If you use EDGAR-gamma in your research, please cite:

```bibtex
@article{edgar-gamma,
  title={EDGAR-gamma: Evolutionary Discovery of Generative AI-assisted Research},
  author={Your Name},
  year={2026}
}
```

## License

[]

## Contributing

[]
