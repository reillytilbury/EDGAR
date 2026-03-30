# Data Structure Refactoring: Inputs/Outputs → Plain Dict

## Aim

The codebase assumed all problems have a clear (independent, dependent) variable structure, encoded as separate `Inputs` and `Outputs` dataclasses — both 3D tensors of shape `(n_samples, n_features/n_targets, n_trials)`. This failed for problems where:

1. There is no clear input/output distinction (e.g. fitting a probability distribution).
2. Features are categorically different and shouldn't be stacked (e.g. source neuron activity `(n_source, n_trials)` vs scalar stimulus identity `(n_trials,)`).
3. The model only uses a subset of available features.

The fix: replace `Inputs`/`Outputs` with a plain `dict[str, np.ndarray]` where all values share a common last dimension (n_trials). This is the simplest structure that supports arbitrary problem types while remaining a native JAX pytree (so `jax.vmap` and `jax.jit` work with no registration).

## Data structure

```python
# Orientation tuning
X = {'stimulus': angles, 'response': firing_rates}
# angles.shape = (n_samples, n_trials), firing_rates.shape = (n_samples, n_trials)

# Trial-to-trial variability (heterogeneous features)
X = {
    'source': source_activity,   # (n_samples, n_source, n_trials)
    'stimulus': stim_ids,        # (n_samples, n_trials)
    'target': target_activity,   # (n_samples, n_target, n_trials)
}
```

**Constraint**: every array must share the same last dimension (n_trials). Validated by `utils.validate_data()`.

## Signature changes

| Component | Before | After |
|-----------|--------|-------|
| Model | `model(X, params)` where X is `(n_features, n_trials)` | `model(data, params)` where data is a dict |
| Loss function | `loss_fn(y_pred, y_true)` | `loss_fn(model_output, data)` |
| Param estimator | `param_estimator(X, Y)` | `param_estimator(data)` |
| Plot function | `plot_model_fits(X=, Y=, ...)` | `plot_model_fits(data=, ...)` |

## What changed

### New: dict helpers in `src/utils.py`
`validate_data`, `data_n_trials`, `data_n_samples`, `slice_data_samples`, `slice_data_trials`, `get_data_sample`, `data_as_jax`, `data_as_numpy`.

### Deleted: `src/data_structures.py`
`Inputs`, `Outputs`, `ensure_inputs`, `ensure_outputs` — all removed.

### Project specs (all 5 migrated)
Each spec's `load_and_process_data()` now returns a dict. Models access keys by name (e.g. `data['stimulus']`). Loss functions extract their comparison target from the dict.

### `run.py`
- Expects dict from spec, calls `validate_data()`.
- Builds `D = np.empty((2, 2), dtype=object)` where each cell is a data dict (replaces separate `X[2,2]` and `Y[2,2]`).
- `_zscore_data()` replaces `_zscore_trials()`, supports `zscore_skip_keys`.
- `build_evaluation_points()` now returns a data dict; configured via `eval_keys` in config.

### `src/hypothesis_engine.py`
- `objective()` takes `data=[data_train, data_test]` instead of separate `x, y`.
- `loss_single_sample` calls `model(data_i, params)` then `loss_fn(output, data_i)`.
- `jax.vmap(loss_single_sample, in_axes=(0, 0))` works because dicts are native JAX pytrees.
- Penalty denominator is now user-specified (`penalty_denominator` in config, default 1).
- All helper functions (`generate_new_model`, `generate_new_parameter_estimator`, `_run_translation_check_on_eval`, etc.) updated to pass data dicts.

### `src/utils.py` existing functions
`vmap_over_samples`, `call_model`, `check_jax_translation`, `build_evaluation_points`, `compute_evaluation_matrix` — all updated for data dict interface.

### Config and prompts
- Added `eval_keys` to all project configs.
- Updated model/estimator signature guidelines in all YAML prompt templates.
- `src/data_summary.py` rewritten to report per-key shape/dtype info.

## New config options

| Option | Default | Purpose |
|--------|---------|---------|
| `eval_keys` | all keys | Which data dict keys to create evaluation grids for |
| `penalty_denominator` | 1 | Denominator for parameter count penalty (replaces auto-computed `n_in * n_out`) |
| `zscore_skip_keys` | none | Data dict keys to exclude from z-scoring |
