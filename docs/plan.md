# Refactoring Data Structures: From Inputs/Outputs to Plain Dict

## Context

The current codebase assumes all problems have a clear (independent, dependent) variable structure, encoded as separate `Inputs` and `Outputs` dataclasses — both 3D tensors of shape `(n_samples, n_features/n_targets, n_trials)`. This fails for problems where:

1. There is no clear input/output distinction (e.g., fitting a probability distribution)
2. Features are categorically different and shouldn't be stacked (e.g., source neuron activity `(n_source, n_trials)` vs scalar stimulus identity `(n_trials,)`)
3. The model only uses a subset of available features

The goal is to replace `Inputs`/`Outputs` with a plain `dict[str, np.ndarray]` where all values share a common last dimension (n_trials).

---

## Data Structure: Plain `dict[str, np.ndarray]`

X is a plain Python dict mapping string keys to numpy/jax arrays. All arrays must share the same last dimension (n_trials). Different arrays can have different shapes otherwise.

Example — orientation tuning:
```python
X = {'stimulus': angles, 'response': firing_rates}
# angles.shape = (n_samples, n_trials), firing_rates.shape = (n_samples, n_trials)
```

Example — trial-to-trial variability:
```python
X = {
    'source': source_activity,   # (n_samples, n_source, n_trials)
    'stimulus': stim_ids,        # (n_samples, n_trials)
    'speed': speed,              # (n_samples, n_trials)
    'target': target_activity,   # (n_samples, n_target, n_trials)
}
```

**Validation**: Add a `validate_data(X)` helper in `utils.py` that checks all values share the same last dim. Call it at the start of the pipeline and at the start of the objective. Fail loudly with a clear message if violated.

**Helper functions** in `utils.py` for common operations:
- `slice_data_samples(X, indices)` → `{k: v[indices] for k, v in X.items()}`
- `slice_data_trials(X, indices)` → `{k: v[..., indices] for k, v in X.items()}`
- `get_data_sample(X, idx)` → `{k: v[idx] for k, v in X.items()}` (removes sample axis)
- `data_as_jax(X)` → `{k: jnp.asarray(v) for k, v in X.items()}`
- `data_as_numpy(X)` → `{k: np.asarray(v) for k, v in X.items()}`
- `data_n_trials(X)` → last dim of first value
- `data_n_samples(X)` → first dim of first value

A plain dict is already a valid JAX pytree, so `jax.vmap` and `jax.jit` work natively with no registration needed.

---

## Key Design Decisions

**1. No roles, no input/output distinction.**
The dict is role-agnostic. The model and loss function (both user-defined) decide how to interpret the keys. This supports problems with no clear input/output structure.

**2. Loss function: `loss_fn(model_output, data)`**
- Old: `loss_fn(y_pred, y_true)` — assumed a target always exists.
- New: `loss_fn(model_output, data)` — receives model output and the full per-sample data dict.
  - Tuning curve: `loss_fn` extracts `data['response']`, computes MSE against `model_output`.
  - Probability distribution: `loss_fn` returns `-model_output` (neg log-likelihood).

**3. Model interface: `model(data, params)`**
- Old: `model(X, params)` where `X` is `(n_features, n_trials)`.
- New: `model(data, params)` where `data` is a dict for one sample. Model accesses keys by name.

**4. Param estimator: `param_estimator(data)`**
- Old: `param_estimator(X, Y)` — separate arrays.
- New: `param_estimator(data)` — single dict for one sample.

**5. Sample axis for vmap.**
All arrays must have a sample axis (dim 0) by the time they enter the objective. If a spec produces `stimulus: (n_trials,)`, `run.py` broadcasts it to `(n_samples, n_trials)` during slicing. This makes `jax.vmap(..., in_axes=0)` work uniformly.

**6. Penalty: user-specified `penalty_denominator` in config.**
Defaults to 1. Set per-problem in the spec config.

**7. Z-scoring: `zscore_skip_keys` config option.**
List of keys to skip. Default: z-score everything.

---

## Step-by-Step Plan

### Step 1: Add dict helper functions to `src/utils.py`

Add the helper functions listed above: `validate_data`, `slice_data_samples`, `slice_data_trials`, `get_data_sample`, `data_as_jax`, `data_as_numpy`, `data_n_trials`, `data_n_samples`.

### Step 2: Migrate project specs (all 5, clean break)

For each project (`synthetic_data`, `orientation_tuning`, `grid_cells`, `place_cells`, `peer_pred`):

- **`load_and_process_data()`** → return `dict` instead of `(Inputs, Outputs)`
  - e.g., `return {'stimulus': angles, 'response': firing_rates}`
- **`train_test_split(X, seed)`** → use `data_n_samples(X)`, `data_n_trials(X)` from utils
- **`loss_fn(model_output, data)`** → extract comparison target from dict if needed
  - e.g., `mse(model_output, data['response'])`
- **`model_v1(data, params)`** → access `data['stimulus']` instead of `X[0]`
- **`param_est_v1(data)`** → single dict arg instead of `(X, Y)`
- **`plot_model_fits()`** → accept dict

### Step 3: Update `run.py` — data pipeline

- `_build_load_and_process_data_fn`: expect `dict` from spec. Remove `Inputs`/`Outputs` handling. Call `validate_data(X)`.
- Replace `inputs[train_samples][:, :, train_trials]` with `slice_data_samples(slice_data_trials(X, train_trials), train_samples)` (or in whichever order gives correct semantics — sample slice first, then trial slice).
- `_zscore_trials` → `_zscore_data(X, skip_keys=None)`: iterate keys, z-score each unless in skip list.
- Build `D = np.empty((2, 2), dtype=object)` where each cell is a data dict. Replaces both `X[2,2]` and `Y[2,2]`.
- `_build_train_test_split_fn`: pass dict to spec's train_test_split.
- Read `zscore_skip_keys` and `penalty_denominator` from config, pass to engine.

### Step 4: Update `hypothesis_engine.py` — objective function

- New signature: `objective(model, param_estimator, data, loss_fn, ...)` where `data = [data_train, data_test]`, each a dict.
- Remove `ensure_inputs`/`ensure_outputs`. Call `validate_data` at entry.
- `loss_single_sample(params, data_i)`:
  ```python
  model_output = model(data_i, params)
  return loss_fn(model_output, data_i)
  ```
- `jax.vmap(loss_single_sample, in_axes=(0, 0))` — dict is a native JAX pytree, 0 maps over sample axis of every leaf.
- `compute_initial_params`: call `param_estimator(get_data_sample(data, i))`.
- Penalty: `param_penalty_weight * n_params / penalty_denominator`.
- Trial batching: `slice_data_trials(data_train, slice(start, end))`.
- Remove `n_features`, `n_targets` extraction and shape broadcasting logic.

### Step 5: Update `hypothesis_engine.py` — main function

- Accept `D` (2x2 object array of dicts) instead of separate `X` and `Y`.
- `n_training_samples = data_n_samples(D[0, 0])`
- Wire `D[0]` to `_call_objective` and `D[1]` to test evaluation.
- Update `_run_translation_check_on_eval`, plotting, and diagnostic calls.

### Step 6: Update `src/utils.py` — existing functions

- `vmap_over_samples`: accept dict for data arg.
- `build_evaluation_points`: make spec-defined (optional `build_eval_points(X)` in spec). Provide simple default.
- `compute_evaluation_matrix`: use dict-aware model calling.
- `check_jax_translation`: update to use dict.

### Step 7: Update `src/data_summary.py`

- Accept dict, report per-key shape/dtype info.

### Step 8: Update prompt system

- `prompt_manager.py` and config YAML: describe dict keys and shapes to the LLM.
- Update code guidelines to show new model/param_est/loss_fn signatures.

### Step 9: Clean up

- Delete `Inputs`, `Outputs`, `ensure_inputs`, `ensure_outputs` from `data_structures.py` (or delete the file entirely if empty).
- Remove all remaining references to the old classes throughout the codebase.

---

## Files Modified (summary)

| File | Change |
|------|--------|
| `src/utils.py` | Add dict helper functions, update existing helpers |
| `src/data_structures.py` | Delete `Inputs`/`Outputs` (end of migration) |
| `run.py` | Dict-based pipeline, z-scoring, D[2,2] |
| `src/hypothesis_engine.py` | `objective()` and `hypothesis_engine()` accept dicts |
| `src/data_summary.py` | Accept dict |
| `projects/synthetic_data/spec.py` | Return dict, new signatures |
| `projects/orientation_tuning/spec.py` | Return dict, new signatures |
| `projects/grid_cells/spec.py` | Return dict, new signatures |
| `projects/place_cells/spec.py` | Return dict, new signatures |
| `projects/peer_pred/spec.py` | Return dict, new signatures |
| `src/prompt_manager.py` | Update data descriptions |
| Config YAMLs | Add `penalty_denominator`, `zscore_skip_keys` |

---

## Verification Plan

1. **Unit test dict helpers**: `validate_data`, slicing, conversion — quick sanity checks.
2. **Run `synthetic_data` end-to-end** (test mode): simplest project, validates core pipeline.
3. **Run `orientation_tuning` end-to-end** (test mode): validates real-data project.
4. **Check JAX vmap**: small test verifying `jax.vmap` over a dict pytree maps over sample axis correctly.
5. **Test z-scoring with `zscore_skip_keys`**: mixed continuous/categorical keys.
6. **Run remaining projects** end-to-end.
