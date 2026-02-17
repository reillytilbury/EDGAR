import inspect
import re
import os
import logging
import asyncio
import numpy as np
import jax, jax.numpy as jnp
import timeout_decorator
import jaxopt, optax
import pandas as pd
from pathlib import Path
from . import utils, llm_helper
from . import genetic_helpers_v2 as genetic_helpers  # Using v2 with compatibility API
from .data_structures import Inputs, Outputs, ensure_inputs, ensure_outputs
from .diagnostics_manager import ModelFitPlotData, plot_train_vs_test_loss as plot_train_vs_test_loss_shared
import experiments.orientation_tuning.seed_programs # delete this once we read seed_programs from experiment.yaml
from tqdm import tqdm
from google import genai
from dotenv import load_dotenv
import warnings
import time
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message="The behavior of DataFrame concatenation with empty or all-NA entries is deprecated.*"
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(relativeCreated)dms | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

print(jax.default_backend())    # should print "gpu"
print(jax.devices())


def _enforce_single_feature_code_access(code_string: str, stimuli, code_label: str) -> str:
    """
    For single-feature datasets, rewrite out-of-bounds feature access (e.g., X[1]) to X[0].
    """
    if code_string is None:
        return code_string

    x_arr = np.asarray(stimuli)
    n_features = 1 if x_arr.ndim == 2 else int(x_arr.shape[1])
    if n_features != 1:
        return code_string

    pattern = r"\bX\s*\[\s*[1-9]\d*\s*\]"
    rewritten, n_rewrites = re.subn(pattern, "X[0]", code_string)
    if n_rewrites > 0:
        logging.info(
            f"{code_label}: rewrote {n_rewrites} out-of-bounds feature accesses to X[0] "
            "for single-feature data."
        )
    return rewritten


def normalize_loaded_data(data_dict: dict) -> tuple[np.ndarray, np.ndarray]:
    """
    Normalize loaded data to canonical 3D tensors.

    Returns:
        inputs_3d: shape (n_samples, n_features, n_trials)
        outputs_3d: shape (n_samples, n_targets, n_trials)
    """
    if 'inputs' in data_dict:
        inputs_obj = ensure_inputs(data_dict['inputs'])
    elif 'trials' in data_dict:
        inputs_obj = ensure_inputs(data_dict['trials'])
    else:
        raise ValueError("Loaded data must contain either 'inputs' or legacy 'trials'.")

    if 'outputs' in data_dict:
        outputs_obj = ensure_outputs(data_dict['outputs'])
    elif 'response' in data_dict:
        outputs_obj = ensure_outputs(data_dict['response'])
    else:
        raise ValueError("Loaded data must contain either 'outputs' or legacy 'response'.")

    inputs_3d = np.asarray(inputs_obj.to_tensor())
    outputs_3d = np.asarray(outputs_obj.to_tensor())

    if inputs_3d.ndim != 3:
        raise ValueError(f"Canonical inputs must be 3D, got shape {inputs_3d.shape}.")
    if outputs_3d.ndim != 3:
        raise ValueError(f"Canonical outputs must be 3D, got shape {outputs_3d.shape}.")
    if inputs_3d.shape[0] != outputs_3d.shape[0]:
        raise ValueError(
            "Inputs/outputs sample-count mismatch: "
            f"{inputs_3d.shape[0]} != {outputs_3d.shape[0]}."
        )
    return inputs_3d, outputs_3d


def _call_trial_split(split_fn, n_trials_x, n_trials_y, random_seed):
    """
    Call the trial split function, supporting both legacy and generalized signatures.

    Legacy signature: split_fn(n_trials, random_seed) -> (train_idx, test_idx)
        Only valid when n_trials_x == n_trials_y.
        Returns the same indices for both x and y.

    Generalized signature: split_fn(n_trials_x, n_trials_y, random_seed) ->
        (x_train_idx, x_test_idx, y_train_idx, y_test_idx)
        Supports mismatched trial counts.

    Returns:
        (x_train_idx, x_test_idx, y_train_idx, y_test_idx)
    """
    if split_fn is None:
        raise ValueError("Trial split function is None. Please provide a valid function for splitting trials.")

    # Try the generalized 3-positional-arg signature first:
    # split_fn(n_trials_x, n_trials_y, random_seed) -> 4-tuple
    try:
        result = split_fn(n_trials_x, n_trials_y, random_seed)
        if isinstance(result, tuple) and len(result) == 4:
            return result
    except TypeError:
        pass

    # Fall back to legacy 2-arg signature: split_fn(n_trials, random_seed)
    if n_trials_x != n_trials_y:
        raise ValueError(
            f"Trial split function has legacy signature (n_trials, random_seed) "
            f"but n_trials_x={n_trials_x} != n_trials_y={n_trials_y}. "
            f"Provide a generalized trial split function that accepts "
            f"(n_trials_x, n_trials_y, random_seed) and returns 4 index arrays: "
            f"(x_train_idx, x_test_idx, y_train_idx, y_test_idx)."
        )

    # Try common legacy calling conventions
    legacy_attempts = [
        lambda: split_fn(n_trials_x, random_seed),
        lambda: split_fn(n_trials_x, random_seed=random_seed),
        lambda: split_fn(n_trials_x),
    ]
    last_exc = None
    for attempt in legacy_attempts:
        try:
            train_idx, test_idx = attempt()
            return train_idx, test_idx, train_idx, test_idx
        except TypeError as exc:
            last_exc = exc
    raise TypeError(
        f"Unable to call trial split function {split_fn} with any supported signature."
    ) from last_exc


def scalar_outputs_view(outputs_3d: np.ndarray) -> np.ndarray:
    """
    Return 2D scalar-output view (n_samples, n_trials) from canonical outputs.
    """
    arr = np.asarray(outputs_3d)
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D outputs tensor, got shape {arr.shape}.")
    if arr.shape[1] != 1:
        raise ValueError(
            f"Scalar output view requires n_targets=1, got n_targets={arr.shape[1]}."
        )
    return arr[:, 0, :]


def save_data_summary(
    response: np.ndarray,
    inputs: np.ndarray,
    training_samples: jnp.ndarray,
    test_samples: jnp.ndarray,
    output_dir: str,
    random_seed: int = 0,
    training_sample_ratio: float = 0.5,
    create_train_test_sample_split_fn=None,
    create_train_test_trial_split_fn=None,
) -> pd.DataFrame:
    """
    Save a summary of the realized sample/trial splits and matrix sizes to CSV.

    This uses the exact sample indices provided and computes trial split indices by
    invoking the trial split function used by objective().
    """
    response_arr = np.asarray(response)
    if response_arr.ndim == 2:
        n_total_samples, n_trials_y = response_arr.shape
        n_targets = 1
    elif response_arr.ndim == 3:
        n_total_samples, n_targets, n_trials_y = response_arr.shape
    else:
        raise ValueError(
            f"Response/outputs must be 2D or 3D, got shape {response_arr.shape}"
        )

    inputs_arr = np.asarray(inputs)
    if inputs_arr.ndim != 3:
        raise ValueError(f"Inputs must be 3D in canonical pipeline, got {inputs_arr.shape}")
    n_trials_x = inputs_arr.shape[2]

    training_samples_np = np.asarray(training_samples).reshape(-1)
    test_samples_np = np.asarray(test_samples).reshape(-1)
    n_train_samples = int(training_samples_np.size)
    n_test_samples = int(test_samples_np.size)

    # Determine inputs shape and features
    n_features = inputs_arr.shape[1]
    inputs_shape_str = f"({inputs_arr.shape[0]}, {inputs_arr.shape[1]}, {inputs_arr.shape[2]})"

    def _split_stats(train_idx, test_idx, n_total):
        train_arr = np.asarray(train_idx).reshape(-1)
        test_arr = np.asarray(test_idx).reshape(-1)
        train_unique = np.unique(train_arr)
        test_unique = np.unique(test_arr)
        overlap = np.intersect1d(train_unique, test_unique)
        coverage = np.union1d(train_unique, test_unique)
        return {
            "disjoint": bool(overlap.size == 0),
            "cover_all": bool(coverage.size == n_total),
            "n_overlap": int(overlap.size),
            "n_uncovered": int(max(0, n_total - coverage.size)),
            "train_has_duplicates": bool(train_unique.size != train_arr.size),
            "test_has_duplicates": bool(test_unique.size != test_arr.size),
            "train_first10": train_arr[:10].tolist(),
            "test_first10": test_arr[:10].tolist(),
        }

    def _describe_fn(fn):
        if fn is None:
            return "None"
        module = getattr(fn, "__module__", "<unknown_module>")
        name = getattr(fn, "__qualname__", getattr(fn, "__name__", repr(fn)))
        return f"{module}.{name}"

    # Trial split is the one used inside objective()
    try:
        x_train_trial_idx, x_test_trial_idx, y_train_trial_idx, y_test_trial_idx = \
            _call_trial_split(create_train_test_trial_split_fn, n_trials_x, n_trials_y, random_seed)
        trial_split_error = None
    except Exception as exc:
        x_train_trial_idx = np.arange(n_trials_x, dtype=np.int32)
        x_test_trial_idx = x_train_trial_idx
        y_train_trial_idx = np.arange(n_trials_y, dtype=np.int32)
        y_test_trial_idx = y_train_trial_idx
        trial_split_error = str(exc)

    x_train_trial_idx_np = np.asarray(x_train_trial_idx).reshape(-1)
    x_test_trial_idx_np = np.asarray(x_test_trial_idx).reshape(-1)
    y_train_trial_idx_np = np.asarray(y_train_trial_idx).reshape(-1)
    y_test_trial_idx_np = np.asarray(y_test_trial_idx).reshape(-1)
    n_training_trials_x = int(x_train_trial_idx_np.size)
    n_test_trials_x = int(x_test_trial_idx_np.size)
    n_training_trials_y = int(y_train_trial_idx_np.size)
    n_test_trials_y = int(y_test_trial_idx_np.size)

    sample_stats = _split_stats(training_samples_np, test_samples_np, n_total_samples)
    # Use input trial indices for trial split stats (representative when matched)
    trial_stats = _split_stats(x_train_trial_idx_np, x_test_trial_idx_np, n_trials_x)

    sample_split_method = (
        f"fn={_describe_fn(create_train_test_sample_split_fn)}; "
        f"training_sample_ratio={training_sample_ratio}; random_seed={random_seed}; "
        f"disjoint={sample_stats['disjoint']}; cover_all={sample_stats['cover_all']}; "
        f"overlap={sample_stats['n_overlap']}; uncovered={sample_stats['n_uncovered']}; "
        f"train_has_duplicates={sample_stats['train_has_duplicates']}; "
        f"test_has_duplicates={sample_stats['test_has_duplicates']}; "
        f"train_first10={sample_stats['train_first10']}; "
        f"test_first10={sample_stats['test_first10']}"
    )

    trial_split_method = (
        f"fn={_describe_fn(create_train_test_trial_split_fn)}; "
        f"random_seed={random_seed}; "
        f"n_trials_x={n_trials_x}; n_trials_y={n_trials_y}; "
        f"disjoint={trial_stats['disjoint']}; cover_all={trial_stats['cover_all']}; "
        f"overlap={trial_stats['n_overlap']}; uncovered={trial_stats['n_uncovered']}; "
        f"train_has_duplicates={trial_stats['train_has_duplicates']}; "
        f"test_has_duplicates={trial_stats['test_has_duplicates']}; "
        f"train_first10={trial_stats['train_first10']}; "
        f"test_first10={trial_stats['test_first10']}"
    )
    if trial_split_error is not None:
        trial_split_method += f"; error={trial_split_error}"

    # Helper to calculate size in bytes
    def calc_size(shape, dtype):
        n_elements = np.prod(shape)
        bytes_per_element = np.dtype(dtype).itemsize
        return n_elements * bytes_per_element

    def format_size(size_bytes):
        if size_bytes >= 1e9:
            return f"{size_bytes / 1e9:.2f} GB"
        if size_bytes >= 1e6:
            return f"{size_bytes / 1e6:.2f} MB"
        if size_bytes >= 1e3:
            return f"{size_bytes / 1e3:.2f} KB"
        return f"{size_bytes} B"

    # Build summary rows
    rows = []

    # === SAMPLE SPLIT SUMMARY ===
    rows.append({
        'category': 'SAMPLE_SPLIT',
        'matrix_name': 'total_samples',
        'description': 'Total number of cells/samples in dataset',
        'shape': f"({n_total_samples},)",
        'dtype': '-',
        'size_bytes': '-',
        'size_human': '-',
        'n_elements': n_total_samples
    })
    rows.append({
        'category': 'SAMPLE_SPLIT',
        'matrix_name': 'training_samples',
        'description': 'Samples used for training (held-in)',
        'shape': f"({n_train_samples},)",
        'dtype': str(training_samples_np.dtype),
        'size_bytes': calc_size((n_train_samples,), training_samples_np.dtype),
        'size_human': format_size(calc_size((n_train_samples,), training_samples_np.dtype)),
        'n_elements': n_train_samples
    })
    rows.append({
        'category': 'SAMPLE_SPLIT',
        'matrix_name': 'test_samples',
        'description': 'Samples used for testing (held-out)',
        'shape': f"({n_test_samples},)",
        'dtype': str(test_samples_np.dtype),
        'size_bytes': calc_size((n_test_samples,), test_samples_np.dtype),
        'size_human': format_size(calc_size((n_test_samples,), test_samples_np.dtype)),
        'n_elements': n_test_samples
    })
    rows.append({
        'category': 'SAMPLE_SPLIT',
        'matrix_name': 'sample_split_method',
        'description': sample_split_method,
        'shape': '-',
        'dtype': '-',
        'size_bytes': '-',
        'size_human': '-',
        'n_elements': '-'
    })

    # === TRIAL SPLIT SUMMARY (within objective function) ===
    rows.append({
        'category': 'TRIAL_SPLIT',
        'matrix_name': 'total_trials_x',
        'description': 'Total number of input trials per sample',
        'shape': f"({n_trials_x},)",
        'dtype': '-',
        'size_bytes': '-',
        'size_human': '-',
        'n_elements': n_trials_x
    })
    rows.append({
        'category': 'TRIAL_SPLIT',
        'matrix_name': 'total_trials_y',
        'description': 'Total number of output trials per sample',
        'shape': f"({n_trials_y},)",
        'dtype': '-',
        'size_bytes': '-',
        'size_human': '-',
        'n_elements': n_trials_y
    })
    rows.append({
        'category': 'TRIAL_SPLIT',
        'matrix_name': 'training_trials_x',
        'description': 'Input trials used for param fitting in objective()',
        'shape': f"({n_training_trials_x},)",
        'dtype': str(x_train_trial_idx_np.dtype),
        'size_bytes': calc_size((n_training_trials_x,), x_train_trial_idx_np.dtype),
        'size_human': format_size(calc_size((n_training_trials_x,), x_train_trial_idx_np.dtype)),
        'n_elements': n_training_trials_x
    })
    rows.append({
        'category': 'TRIAL_SPLIT',
        'matrix_name': 'test_trials_x',
        'description': 'Input trials used for loss evaluation in objective()',
        'shape': f"({n_test_trials_x},)",
        'dtype': str(x_test_trial_idx_np.dtype),
        'size_bytes': calc_size((n_test_trials_x,), x_test_trial_idx_np.dtype),
        'size_human': format_size(calc_size((n_test_trials_x,), x_test_trial_idx_np.dtype)),
        'n_elements': n_test_trials_x
    })
    rows.append({
        'category': 'TRIAL_SPLIT',
        'matrix_name': 'training_trials_y',
        'description': 'Output trials used for param fitting in objective()',
        'shape': f"({n_training_trials_y},)",
        'dtype': str(y_train_trial_idx_np.dtype),
        'size_bytes': calc_size((n_training_trials_y,), y_train_trial_idx_np.dtype),
        'size_human': format_size(calc_size((n_training_trials_y,), y_train_trial_idx_np.dtype)),
        'n_elements': n_training_trials_y
    })
    rows.append({
        'category': 'TRIAL_SPLIT',
        'matrix_name': 'test_trials_y',
        'description': 'Output trials used for loss evaluation in objective()',
        'shape': f"({n_test_trials_y},)",
        'dtype': str(y_test_trial_idx_np.dtype),
        'size_bytes': calc_size((n_test_trials_y,), y_test_trial_idx_np.dtype),
        'size_human': format_size(calc_size((n_test_trials_y,), y_test_trial_idx_np.dtype)),
        'n_elements': n_test_trials_y
    })
    rows.append({
        'category': 'TRIAL_SPLIT',
        'matrix_name': 'trial_split_method',
        'description': trial_split_method,
        'shape': '-',
        'dtype': '-',
        'size_bytes': '-',
        'size_human': '-',
        'n_elements': '-'
    })
    
    # === DATA MATRICES ===
    # Response matrices
    response_dtype = response_arr.dtype
    rows.append({
        'category': 'DATA_MATRIX',
        'matrix_name': 'outputs (full)',
        'description': 'All samples, all targets, all trials',
        'shape': str(response_arr.shape),
        'dtype': str(response_dtype),
        'size_bytes': calc_size(response_arr.shape, response_dtype),
        'size_human': format_size(calc_size(response_arr.shape, response_dtype)),
        'n_elements': np.prod(response_arr.shape)
    })
    
    response_train_shape = (n_train_samples, n_targets, n_trials_y)
    rows.append({
        'category': 'DATA_MATRIX',
        'matrix_name': 'outputs_train',
        'description': 'Training samples, all targets, all output trials',
        'shape': str(response_train_shape),
        'dtype': str(response_dtype),
        'size_bytes': calc_size(response_train_shape, response_dtype),
        'size_human': format_size(calc_size(response_train_shape, response_dtype)),
        'n_elements': np.prod(response_train_shape)
    })

    response_test_shape = (n_test_samples, n_targets, n_trials_y)
    rows.append({
        'category': 'DATA_MATRIX',
        'matrix_name': 'outputs_test',
        'description': 'Test samples, all targets, all output trials',
        'shape': str(response_test_shape),
        'dtype': str(response_dtype),
        'size_bytes': calc_size(response_test_shape, response_dtype),
        'size_human': format_size(calc_size(response_test_shape, response_dtype)),
        'n_elements': np.prod(response_test_shape)
    })
    
    # Input matrices
    inputs_dtype = inputs_arr.dtype
    rows.append({
        'category': 'DATA_MATRIX',
        'matrix_name': 'inputs (full)',
        'description': f'All samples, {n_features} features, all trials',
        'shape': inputs_shape_str,
        'dtype': str(inputs_dtype),
        'size_bytes': calc_size(inputs_arr.shape, inputs_dtype),
        'size_human': format_size(calc_size(inputs_arr.shape, inputs_dtype)),
        'n_elements': np.prod(inputs_arr.shape)
    })
    
    inputs_train_shape = (n_train_samples, n_features, n_trials_x)
    inputs_test_shape = (n_test_samples, n_features, n_trials_x)
    
    rows.append({
        'category': 'DATA_MATRIX',
        'matrix_name': 'inputs_train',
        'description': f'Training samples, {n_features} features, all trials',
        'shape': str(inputs_train_shape),
        'dtype': str(inputs_dtype),
        'size_bytes': calc_size(inputs_train_shape, inputs_dtype),
        'size_human': format_size(calc_size(inputs_train_shape, inputs_dtype)),
        'n_elements': np.prod(inputs_train_shape)
    })
    
    rows.append({
        'category': 'DATA_MATRIX',
        'matrix_name': 'inputs_test',
        'description': f'Test samples, {n_features} features, all trials',
        'shape': str(inputs_test_shape),
        'dtype': str(inputs_dtype),
        'size_bytes': calc_size(inputs_test_shape, inputs_dtype),
        'size_human': format_size(calc_size(inputs_test_shape, inputs_dtype)),
        'n_elements': np.prod(inputs_test_shape)
    })
    
    # === OBJECTIVE FUNCTION SUB-MATRICES (within training samples) ===
    # These are created inside objective() for the training samples
    x_train_shape = (n_train_samples, n_features, n_training_trials_x)
    x_test_shape = (n_train_samples, n_features, n_test_trials_x)

    y_train_shape = (n_train_samples, n_targets, n_training_trials_y)
    y_test_shape = (n_train_samples, n_targets, n_test_trials_y)
    
    rows.append({
        'category': 'OBJECTIVE_SUBMATRIX',
        'matrix_name': 'x_train (in objective)',
        'description': 'Training samples, training trials (param fitting)',
        'shape': str(x_train_shape),
        'dtype': str(inputs_dtype),
        'size_bytes': calc_size(x_train_shape, inputs_dtype),
        'size_human': format_size(calc_size(x_train_shape, inputs_dtype)),
        'n_elements': np.prod(x_train_shape)
    })
    
    rows.append({
        'category': 'OBJECTIVE_SUBMATRIX',
        'matrix_name': 'y_train (in objective)',
        'description': 'Training samples, training trials (param fitting)',
        'shape': str(y_train_shape),
        'dtype': str(response_dtype),
        'size_bytes': calc_size(y_train_shape, response_dtype),
        'size_human': format_size(calc_size(y_train_shape, response_dtype)),
        'n_elements': np.prod(y_train_shape)
    })
    
    rows.append({
        'category': 'OBJECTIVE_SUBMATRIX',
        'matrix_name': 'x_test (in objective)',
        'description': 'Training samples, test trials (loss evaluation)',
        'shape': str(x_test_shape),
        'dtype': str(inputs_dtype),
        'size_bytes': calc_size(x_test_shape, inputs_dtype),
        'size_human': format_size(calc_size(x_test_shape, inputs_dtype)),
        'n_elements': np.prod(x_test_shape)
    })
    
    rows.append({
        'category': 'OBJECTIVE_SUBMATRIX',
        'matrix_name': 'y_test (in objective)',
        'description': 'Training samples, test trials (loss evaluation)',
        'shape': str(y_test_shape),
        'dtype': str(response_dtype),
        'size_bytes': calc_size(y_test_shape, response_dtype),
        'size_human': format_size(calc_size(y_test_shape, response_dtype)),
        'n_elements': np.prod(y_test_shape)
    })
    
    # === FEATURE INFO ===
    rows.append({
        'category': 'FEATURES',
        'matrix_name': 'n_features',
        'description': 'Number of input features per sample',
        'shape': '-',
        'dtype': '-',
        'size_bytes': '-',
        'size_human': '-',
        'n_elements': n_features
    })
    rows.append({
        'category': 'FEATURES',
        'matrix_name': 'n_targets',
        'description': 'Number of output targets per sample',
        'shape': '-',
        'dtype': '-',
        'size_bytes': '-',
        'size_human': '-',
        'n_elements': n_targets
    })
    
    # Create DataFrame and save
    df = pd.DataFrame(rows)
    csv_path = os.path.join(output_dir, 'data_summary.csv')
    df.to_csv(csv_path, index=False)
    
    # Also print a summary
    print("\n" + "=" * 70)
    print("DATA SUMMARY")
    print("=" * 70)
    print(
        f"Sample Split: {n_train_samples}/{n_total_samples} train, "
        f"{n_test_samples}/{n_total_samples} test "
        f"(disjoint={sample_stats['disjoint']}, cover_all={sample_stats['cover_all']})"
    )
    print(
        f"Input Trial Split:  {n_training_trials_x}/{n_trials_x} train, "
        f"{n_test_trials_x}/{n_trials_x} test (per sample, in objective; "
        f"seed={random_seed}, disjoint={trial_stats['disjoint']}, "
        f"cover_all={trial_stats['cover_all']})"
    )
    print(
        f"Output Trial Split:  {n_training_trials_y}/{n_trials_y} train, "
        f"{n_test_trials_y}/{n_trials_y} test (per sample, in objective; "
        f"seed={random_seed}, disjoint={trial_stats['disjoint']}, "
        f"cover_all={trial_stats['cover_all']})"
    )
    print(f"Features:     {n_features} per sample")
    print(f"Targets:      {n_targets} per sample")
    print(f"Data Types:   outputs={response_dtype}, inputs={inputs_dtype}")
    total_size = sum(
        r['size_bytes']
        for r in rows
        if isinstance(r['size_bytes'], (int, float, np.integer, np.floating))
    )
    print(f"Total Data:   {format_size(total_size)}")
    print(f"Saved to:     {csv_path}")
    print("=" * 70 + "\n")
    
    logging.info(f"Data summary saved to {csv_path}")
    
    return df


def compute_initial_params(param_estimator, model, x, y) -> jnp.ndarray:
    """
    Compute initial parameters for the model using the provided parameter estimator. 
    The parameter estimator is written in numpy, but the model is written in JAX. 
    So the data x and y will be numpy arrays, but the output will be a JAX array.
    
    Args:
        param_estimator (function): Function to estimate initial parameters for the model.
                                    Signature: param_estimator(X, response) -> params
                                    where X has shape (n_features, n_trials) for a single sample,
                                    and response has shape (n_trials,) for scalar or (n_targets, n_trials) for vectorized.
        model (function): The model which predicts neural activity from stimuli and free parameters.
                                 Signature: model(X, *params) -> activity
                                 where X has shape (n_features, n_trials) for a single sample.
        x (np.ndarray): Input data, shape (n_samples, n_features, n_trials).
        y (np.ndarray): Response data, shape (n_samples, n_trials) for scalar
                        or (n_samples, n_targets, n_trials) for vectorized.
    Returns:
        jnp.ndarray: The estimated parameters for each sample, shape (n_samples, n_params).
                     If the parameter estimation fails, returns an array of default parameters based on the model's signature.
                     If this also fails, returns None.
    """
    @timeout_decorator.timeout(5, use_signals=True)
    def _safe_estimate(pe, xi, yi):
        return pe(xi, yi)

    def _estimator_response_arg(yi):
        """
        Keep canonical 2D target format internally, but adapt scalar target for
        legacy estimators that expect y shape (n_trials,).
        """
        yi_arr = np.asarray(yi)
        if yi_arr.ndim == 2 and yi_arr.shape[0] == 1:
            return yi_arr[0]
        return yi_arr

    try:
        # any call taking >5s will raise timeout_decorator.TimeoutError
        # xi has shape (n_features, n_trials)
        # yi has shape (n_trials,) for scalar or (n_targets, n_trials) for vectorized
        return jnp.array([
            _safe_estimate(param_estimator, x[i], _estimator_response_arg(y[i]))
            for i in range(y.shape[0])
        ])
    except timeout_decorator.TimeoutError:
        logging.warning("param_estimator timed out, falling back to defaults")
    except Exception as e:
        logging.info(f"Error during parameter estimation: {e}")

    # If parameter estimation fails, compute default parameters based on the model's signature
    params = compute_default_params(model)
    if params is not None:
        # default params is a 2D array with shape (1, n_params), so we need to repeat it for each sample
        n_samples = y.shape[0]
        return jnp.repeat(params, n_samples, axis=0)
    else:
        logging.info("Error: Unable to compute default parameters for the neuron model.")
        return None

def compute_default_params(model) -> jnp.ndarray:
    """
    Compute default parameters for the model based on its signature.
    Args:
        model (function): The model which predicts neural activity from stimuli and free parameters.
                                 Signature: model(X, *params) -> activity
                                 where X has shape (n_features, n_trials) for a single sample.
    Returns:
        jnp.ndarray: The default parameters for the model, shape (1, n_params).
                     If the parameter estimation fails, returns None.
    """
    try:
        sig = inspect.signature(model)
        # First parameter is the input (X or theta), skip it
        all_param_names = list(sig.parameters.keys())
        # The first param is the input (could be named 'X', 'theta', or anything)
        # All subsequent params are the model parameters to fit
        param_names = all_param_names[1:] if all_param_names else []
        defaults = [sig.parameters[n].default if sig.parameters[n].default is not inspect._empty else 0.0 for n in param_names]
        default_arr = jnp.array(defaults, dtype=np.float32)
        return default_arr.reshape(1, -1)  # reshape to (1, n_params)
    except Exception as e:
        logging.info(f"Error while generating default parameters: {e}")
        return None
        return None    


def validate_model_output(
    output: jnp.ndarray,
    expected_n_trials: int,
    expected_n_targets: int = 1,
    allow_1d_for_single_target: bool = True,
) -> tuple[bool, str]:
    """
    Validate that model output has the expected shape.
    
    For scalar outputs (n_targets=1):
        - Expected shape: (n_trials,) - 1D array
        - If allow_1d_for_single_target=True, also accepts 2D (1, n_trials)
    
    For vectorized outputs (n_targets>1):
        - Expected shape: (n_targets, n_trials) - 2D array
    
    Args:
        output: The model output to validate
        expected_n_trials: Expected number of trials (last dimension)
        expected_n_targets: Expected number of targets. Default 1 (scalar output).
        allow_1d_for_single_target: If True and n_targets=1, accept both 1D and 2D output.
    
    Returns:
        tuple[bool, str]: (is_valid, error_message)
            - is_valid: True if output shape is correct
            - error_message: Empty string if valid, otherwise describes the issue
    """
    if expected_n_targets == 1:
        # Scalar output: prefer 1D array of shape (n_trials,)
        if output.ndim == 1:
            if output.shape[0] != expected_n_trials:
                return False, f"Model output n_trials={output.shape[0]} does not match expected {expected_n_trials}"
            return True, ""
        elif output.ndim == 2 and allow_1d_for_single_target:
            # Also accept 2D (1, n_trials) for single target
            if output.shape[0] != 1:
                return False, f"For n_targets=1, 2D output should have shape (1, n_trials), got {output.shape}"
            if output.shape[1] != expected_n_trials:
                return False, f"Model output n_trials={output.shape[1]} does not match expected {expected_n_trials}"
            return True, ""
        else:
            return False, f"Scalar model output should be 1D (n_trials,), got {output.ndim}D with shape {output.shape}"
    else:
        # Vectorized output: expect 2D array of shape (n_targets, n_trials)
        if output.ndim != 2:
            return False, f"Vectorized model output should be 2D (n_targets, n_trials), got {output.ndim}D with shape {output.shape}"
        if output.shape[0] != expected_n_targets:
            return False, f"Model output n_targets={output.shape[0]} does not match expected {expected_n_targets}"
        if output.shape[1] != expected_n_trials:
            return False, f"Model output n_trials={output.shape[1]} does not match expected {expected_n_trials}"
        return True, ""


def validate_model_execution(
    model,
    x_data: jnp.ndarray,
    initial_params: jnp.ndarray,
    n_samples: int,
    expected_n_targets: int = 1,
    n_validation_samples: int = 10,
) -> tuple[bool, str]:
    """
    Validate that a model can execute correctly and produces expected output shapes.
    
    Tests the model on a random subset of samples to verify:
    1. Model runs without exceptions
    2. Model is compatible with JAX JIT and tracing
    3. Output shape matches expected (n_trials,) for scalar or (n_targets, n_trials) for vectorized
    
    Args:
        model: The model function to validate
        x_data: Input data of shape (n_samples, n_features, n_trials)
        initial_params: Initial parameters of shape (n_samples, n_params)
        n_samples: Number of samples
        expected_n_targets: Expected number of output targets (1 for scalar, >1 for vectorized)
        n_validation_samples: Number of random samples to test (default 10)
    
    Returns:
        tuple[bool, str]: (is_valid, error_message)
    """
    try:
        model_jit = jax.jit(model)
        test_n_trials = x_data.shape[2]
        
        for sample_idx in np.random.choice(n_samples, size=min(n_validation_samples, n_samples), replace=False):
            # Validate with concrete values: x_data[sample_idx] is (n_features, n_trials)
            output = model_jit(x_data[sample_idx], *initial_params[sample_idx])
            
            is_valid, error_msg = validate_model_output(output, test_n_trials, expected_n_targets)
            if not is_valid:
                return False, error_msg
            
            # Validate with abstract tracer values
            jax.eval_shape(model_jit, x_data[sample_idx], *initial_params[sample_idx])
        
        return True, ""
    except Exception as e:
        return False, f"Model failed to run or is incompatible with JAX tracing: {e}"


def objective_legacy(model, param_estimator, x, y, create_train_test_trial_split_fn=None,
              loss_fn=None,
              param_penalty_weight=0.1, fit_params=True, random_seed=0,
              FAILED_PROGRAM_COST=jnp.inf, tol=1e-2, max_iter=1_000, learning_rate=3e-3,
              use_param_estimator=True, trial_batch_size=None) -> tuple[float, jnp.ndarray, float, jnp.ndarray]:
    """
    LEGACY: Calculate the loss of the model for scalar (single-target) outputs.
    
    This is the original objective function preserved for backward compatibility.
    For new code, use objective() which handles both scalar and vectorized outputs.
    
    The loss is calculated as the mean over samples and trials of the loss function provided.
    
    Args:
        model (function): The model which predicts neural activity from inputs
                                and free parameters (for a single sample).
                                Signature: model(X, *params) -> activity
                                where X has shape (n_features, n_trials) for a single sample.
        param_estimator (function): Function to estimate initial parameters for the model.
                                Signature: param_estimator(X, response) -> params
                                where X has shape (n_features, n_trials) for a single sample.
        loss_fn (function): Per-sample loss function.
                                Signature: loss_fn(model, x_i, y_i, params) -> scalar.
                                Defaults to MSE over all outputs and trials.
        x: Input data. Can be:
           - 2D array (n_samples, n_trials) - will be auto-expanded to (n_samples, 1, n_trials)
           - 3D array (n_samples, n_features, n_trials)
           - Inputs object
        y (jnp.ndarray): Response data, shape (n_samples, n_trials). MUST be 2D for legacy.
        param_penalty_weight (float): Weight for the penalty on the number of parameters. Default is 0.1.
        fit_params (bool): Whether to fit the parameters of the model. Default is True.
        random_seed (int or None): Random seed for reproducibility. Default is 0.
        FAILED_PROGRAM_COST (float): Cost assigned to failed models. Default is np.inf.
        tol (float): Tolerance for optimization convergence. Default is 1e-2.
        max_iter (int): Maximum number of iterations for optimization. Default is 1_000.
        use_param_estimator (bool): Whether to use the parameter estimator to compute initial parameters. Default is True.
        trial_batch_size (int | None): Number of trials to process per mini-batch
                          to avoid GPU OOM. If None, uses all trials in one batch.

    Returns:
        tuple[
            - float: The cross-validated loss of the model with data fit by the parameter estimator,
            - jnp.ndarray: The parameters fit by the parameter estimator.
            - float: The average loss (MSE on test set) across all samples. 
                     Returns FAILED_PROGRAM_COST if the model fails for ANY cell.
            - jnp.ndarray: The parameters for each sample (n_samples, n_params).
    """
    t_start = time.time()
    
    # Normalize x to Inputs format: (n_samples, n_features, n_trials)
    x_inputs = ensure_inputs(x)
    x_data = x_inputs.to_tensor()  # shape: (n_samples, n_features, n_trials)
    
    n_samples, n_features, n_trials_x = x_data.shape
    n_trials_y = y.shape[1]  # y is 2D: (n_samples, n_trials_y)

    # Split trials for inputs and outputs (may use different indices when mismatched)
    x_train_trial_idx, x_test_trial_idx, y_train_trial_idx, y_test_trial_idx = \
        _call_trial_split(create_train_test_trial_split_fn, n_trials_x, n_trials_y, random_seed)

    # Split inputs and response using their respective trial indices
    x_train = x_data[:, :, x_train_trial_idx]    # (n_samples, n_features, n_train_trials_x)
    y_train = y[:, y_train_trial_idx]             # (n_samples, n_train_trials_y)
    x_test = x_data[:, :, x_test_trial_idx]       # (n_samples, n_test_trials_x)
    y_test = y[:, y_test_trial_idx]                # (n_samples, n_test_trials_y)

    # Perform initial param calc. x must be numpy array of shape (n_samples, n_features, n_trials)
    if use_param_estimator:
        initial_params = compute_initial_params(param_estimator, model, np.asarray(x_train), np.asarray(y_train))
    else:
        initial_params = compute_default_params(model)
        # if initial_params not none, reshape from (1, n_params) to (n_samples, n_params)
        if initial_params is not None:
            n_params = initial_params.shape[1]
            initial_params = jnp.repeat(initial_params, n_samples, axis=0)
    
    # Fail immediately if initial_params is None or not a JAX array
    if initial_params is None or not isinstance(initial_params, jnp.ndarray):
        logging.info("Error: initial_params should be a JAX array.")
        return FAILED_PROGRAM_COST, jnp.zeros((n_samples, 0)), FAILED_PROGRAM_COST, jnp.zeros((n_samples, 0))
    if initial_params.ndim != 2 or initial_params.shape[0] != n_samples:
        logging.info(f"Error: initial_params should be a 2D array with shape ({n_samples}, n_params).")
        return FAILED_PROGRAM_COST, jnp.zeros((n_samples, 0)), FAILED_PROGRAM_COST, jnp.zeros((n_samples, 0))

    # Fail immediately if fit_params is True and non-numeric params
    n_params = initial_params.shape[1]
    all_numeric = (initial_params.dtype.kind in 'biufc' and 
                  jnp.all(jnp.isfinite(initial_params)))
    if fit_params and not all_numeric:
        logging.info("Error: Cannot fit non-numeric parameters.")
        return FAILED_PROGRAM_COST, jnp.zeros((n_samples, n_params)), FAILED_PROGRAM_COST, jnp.zeros((n_samples, n_params))

    # Fail immediately if model doesn't run
    # x_data[sample_idx] has shape (n_features, n_trials)
    try:
        # Check compatibility with JAX's tracing mechanism
        model_jit = jax.jit(model)
        test_n_trials = x_data.shape[2]  # full n_trials for validation
        for sample_idx in np.random.choice(n_samples, size=min(10, n_samples), replace=False):
            # Validate with concrete values: x_data[sample_idx] is (n_features, n_trials)
            output = model_jit(x_data[sample_idx], *initial_params[sample_idx])
            is_valid, error_msg = validate_model_output(
                output=output,
                expected_n_trials=test_n_trials,
                expected_n_targets=1,
                allow_1d_for_single_target=True,
            )
            if not is_valid:
                logging.info(f"Error: {error_msg}")
                logging.info(
                    f"Model output shape: {output.shape}, "
                    f"expected: ({test_n_trials},) or (1, {test_n_trials})"
                )
                return FAILED_PROGRAM_COST, initial_params, FAILED_PROGRAM_COST, initial_params
            # Validate with abstract tracer values
            jax.eval_shape(model_jit, x_data[sample_idx], *initial_params[sample_idx])
    except Exception as e:
        logging.info(f"Model failed to run or is incompatible with JAX tracing: {e}")
        return FAILED_PROGRAM_COST, initial_params, FAILED_PROGRAM_COST, initial_params

    # Per-sample loss function
    loss_single_cell = lambda params, x_i, y_i: loss_fn(model, x_i, y_i, params)

    # Vectorize over samples
    loss_total = jax.vmap(loss_single_cell, in_axes=(0, 0, 0), out_axes=0)

    # Mini-batched loss and gradient computation to avoid GPU OOM
    n_train_trials_x = x_train.shape[2]
    n_train_trials_y = y_train.shape[1]  # y is 2D in legacy
    trials_matched = (n_train_trials_x == n_train_trials_y)

    # JIT-compiled loss for a single batch
    @jax.jit
    def loss_single_batch(params_2d, x_batch, y_batch):
        """Compute sum of losses for one batch (JIT-compiled)."""
        batch_losses = loss_total(params_2d, x_batch, y_batch)  # (n_samples,)
        return jnp.sum(batch_losses)

    # Combined loss and gradient computation
    loss_and_grad_single_batch = jax.jit(jax.value_and_grad(loss_single_batch))

    if trials_matched and trial_batch_size is not None:
        n_train_trials = n_train_trials_x
        def loss_and_grad_batched(params):
            """Compute loss and gradient by accumulating over trial batches."""
            params_2d = params.reshape(-1, n_params)
            total_loss = 0.0
            total_grad = jnp.zeros_like(params)

            for start_idx in range(0, n_train_trials, trial_batch_size):
                end_idx = min(start_idx + trial_batch_size, n_train_trials)
                batch_weight = (end_idx - start_idx) / n_train_trials
                x_batch = x_train[:, :, start_idx:end_idx]
                y_batch = y_train[:, start_idx:end_idx]

                batch_loss, batch_grad = loss_and_grad_single_batch(params_2d, x_batch, y_batch)

                total_loss += batch_loss * batch_weight
                total_grad += batch_grad.reshape(-1) * batch_weight

            return total_loss / n_samples, total_grad / n_samples
    else:
        # No trial batching: either trials are mismatched or trial_batch_size is None
        def loss_and_grad_batched(params):
            """Compute loss and gradient over full data."""
            params_2d = params.reshape(-1, n_params)
            loss, grad = loss_and_grad_single_batch(params_2d, x_train, y_train)
            return loss / n_samples, grad.reshape(-1) / n_samples

    if fit_params:
        # 1.  build adam
        beta1, beta2  = 0.9, 0.999
        lr = float(learning_rate)
        opt = optax.adam(lr, b1=beta1, b2=beta2, eps=1e-8)
        opt_state = opt.init(initial_params.reshape(-1))

        if trial_batch_size is None:
            # define the loss function wrt params. This will have input shape n_cells * n_params (note that params is flattened) and output shape (1,)
            loss_param = lambda params: jnp.mean(loss_total(params.reshape(-1, n_params), x_train, y_train))
            loss_param_and_grad = jax.value_and_grad(loss_param)

            # solver = jaxopt.ScipyMinimize(
            #     fun=loss_param_and_grad,
            #     value_and_grad=True,
            #     method='L-BFGS-B',
            #     maxiter=max_iter,
            #     tol=tol,
            #     jit=True)
            # try:
            #     result = solver.run(initial_params.reshape(-1))
            #     params = jnp.asarray(result.params).reshape(n_cells, n_params)
            #     print(f"Optimization success: {result.state.success}, iterations: {result.state.iter_num}")
            # except Exception as e:
            #     params = initial_params
            #     logging.info(f"Error during optimization: {e}")

            
            # 2. jit single step
            @jax.jit
            def train_step(params, opt_state):
                loss, grad = loss_param_and_grad(params)
                updates, opt_state = opt.update(grad, opt_state, params)
                params = optax.apply_updates(params, updates)
                return params, opt_state, loss

            # 3.  iterate
            print_every = 50
            params = initial_params.reshape(-1)  # Flatten params for the optimizer
            initial_loss = loss_param(params)
            best_loss, best_params = initial_loss.copy(), params.copy()
            for step in range(1, max_iter + 1):
                params, opt_state, loss_val = train_step(params, opt_state)
                if jnp.isnan(loss_val) or jnp.isinf(loss_val) or jnp.any(jnp.isnan(params)) or jnp.any(jnp.isinf(params)):
                    logging.info(f"Loss is NaN or Inf at step {step}. Stopping optimization.")
                    print(f"Final loss: {loss_val:.4f} at step {step}")
                    break
                if loss_val < best_loss:
                    best_loss = loss_val.copy()
                    best_params = params.copy()
                if step % print_every == 0:
                    print(f"step {step:4d}  loss {loss_val:.4f}")
            params = best_params.reshape(n_samples, n_params)
            print(f"params optimized. Loss: {best_loss:.4f}")

        else:             
            # 2. Define update step (NOT jit-compiled because loss_and_grad_batched has Python loop)
            def train_step(params, opt_state):
                loss, grad = loss_and_grad_batched(params)
                updates, new_opt_state = opt.update(grad, opt_state, params)
                new_params = optax.apply_updates(params, updates)
                return new_params, new_opt_state, loss
        
            # 3.  iterate
            print_every = 50
            params = initial_params.reshape(-1)  # Flatten params for the optimizer
            initial_loss, _ = loss_and_grad_batched(params)
        
            # Early exit for catastrophically bad programs (loss > 1e10 suggests garbage outputs)
            CATASTROPHIC_LOSS_THRESHOLD = 1e6
            if initial_loss > CATASTROPHIC_LOSS_THRESHOLD:
                print(f"Initial loss {initial_loss:.2e} exceeds threshold {CATASTROPHIC_LOSS_THRESHOLD:.0e}. Skipping optimization.")
                logging.info(f"Skipping optimization: initial loss {initial_loss:.2e} > {CATASTROPHIC_LOSS_THRESHOLD:.0e}")
                return FAILED_PROGRAM_COST, initial_params, FAILED_PROGRAM_COST, initial_params
            
            best_loss, best_params = initial_loss.copy(), params.copy()
            for step in range(1, max_iter + 1):
                params, opt_state, loss_val = train_step(params, opt_state)
                if jnp.isnan(loss_val) or jnp.isinf(loss_val) or jnp.any(jnp.isnan(params)) or jnp.any(jnp.isinf(params)):
                    logging.info(f"Loss is NaN or Inf at step {step}. Stopping optimization.")
                    print(f"Final loss: {loss_val:.4f} at step {step}")
                    break
                # Also exit early if loss explodes during training
                if loss_val > CATASTROPHIC_LOSS_THRESHOLD:
                    logging.info(f"Loss exploded to {loss_val:.2e} at step {step}. Stopping optimization.")
                    print(f"Loss exploded to {loss_val:.2e}. Stopping optimization.")
                    return FAILED_PROGRAM_COST, initial_params, FAILED_PROGRAM_COST, initial_params
                if loss_val < best_loss:
                    best_loss = loss_val.copy()
                    best_params = params.copy()
                if step % print_every == 0:
                    print(f"step {step:4d}  loss {loss_val:.4f}")
            params = best_params.reshape(n_samples, n_params)
            print(f"params optimized. Loss: {best_loss:.4f}")
    else:
        params = compute_initial_params(param_estimator, model, np.asarray(x_train), np.asarray(y_train))
        if params is None or not isinstance(params, jnp.ndarray):
            logging.info("Error: params should be a JAX array.")
            return FAILED_PROGRAM_COST, jnp.zeros((n_samples, n_params))

    # Compute the final loss on the test set for the initial and optimized parameters
    if trials_matched and trial_batch_size is not None:
        def eval_loss_batched(params_2d, x_eval, y_eval):
            """Compute loss by iterating over trial batches (matched trials)."""
            n_eval_trials = x_eval.shape[2]
            weighted_sum = 0.0
            for start_idx in range(0, n_eval_trials, trial_batch_size):
                end_idx = min(start_idx + trial_batch_size, n_eval_trials)
                batch_size = end_idx - start_idx
                x_batch = x_eval[:, :, start_idx:end_idx]
                y_batch = y_eval[:, start_idx:end_idx]
                batch_losses = loss_total(params_2d, x_batch, y_batch)
                weighted_sum += jnp.nansum(batch_losses) * (batch_size / n_eval_trials)
            return weighted_sum / n_samples
        initial_loss = eval_loss_batched(initial_params, x_test, y_test) + param_penalty_weight * n_params
    else:
        initial_loss = jnp.nanmean(loss_total(initial_params, x_test, y_test)) + param_penalty_weight * n_params
    # print number of nans in initial_loss
    n_nans = jnp.sum(jnp.isnan(initial_loss))
    if n_nans > 0:
        print(f"Warning: initial loss contains {n_nans} NaNs. This may indicate a problem with the model or data.")
    initial_loss = jnp.nan_to_num(initial_loss, nan=FAILED_PROGRAM_COST, posinf=FAILED_PROGRAM_COST, neginf=FAILED_PROGRAM_COST)

    if trials_matched and trial_batch_size is not None:
        final_loss = eval_loss_batched(params, x_test, y_test) + param_penalty_weight * n_params
    else:
        final_loss = jnp.nanmean(loss_total(params, x_test, y_test)) + param_penalty_weight * n_params
    # print number of nans in final_loss
    n_nans = jnp.sum(jnp.isnan(final_loss))
    if n_nans > 0:
        print(f"Warning: final loss contains {n_nans} NaNs. This may indicate a problem with the model or data.")
    final_loss = jnp.nan_to_num(final_loss, nan=FAILED_PROGRAM_COST, posinf=FAILED_PROGRAM_COST, neginf=FAILED_PROGRAM_COST)
    # Round final losses to 2 decimal places
    # final_loss = jnp.round(final_loss, 2)
    t_end = time.time()
    print(f"Time taken for optimization: {t_end - t_start:.4f} seconds")
    return float(initial_loss), initial_params, float(final_loss), params


def objective_vectorized(model, param_estimator, x, y, create_train_test_trial_split_fn=None,
                        loss_fn=None,
                         param_penalty_weight=0.1, fit_params=True, random_seed=0,
                         FAILED_PROGRAM_COST=jnp.inf, tol=1e-2, max_iter=1_000, learning_rate=3e-3,
                         use_param_estimator=True, trial_batch_size=None) -> tuple[float, jnp.ndarray, float, jnp.ndarray]:
    """
    Calculate the loss of a model that predicts multiple targets (vectorized outputs).
    
    This function handles models where the output has shape (n_targets, n_trials) for
    each sample, allowing prediction of multiple target cells simultaneously.
    
    The loss is computed as a weighted sum of per-target MSE:
        loss = sum_t(weight[t] * mean((pred[t] - true[t])^2))
    
    Args:
        model (function): The model which predicts neural activity from inputs
                          and free parameters (for a single sample).
                          Signature: model(X, *params) -> activity
                          where X has shape (n_features, n_trials) for a single sample.
                          Output shape: (n_targets, n_trials) for vectorized.
                          For n_targets=1, output can be (n_trials,) and will be auto-expanded.
        param_estimator (function): Function to estimate initial parameters for the model.
                          Signature: param_estimator(X, response) -> params
                          where X has shape (n_features, n_trials) for a single sample,
                          and response has shape (n_trials,) for scalar or (n_targets, n_trials) for vectorized.
        x: Input data. Can be:
           - 2D array (n_samples, n_trials) - will be auto-expanded to (n_samples, 1, n_trials)
           - 3D array (n_samples, n_features, n_trials_x)
           - Inputs object
        y: Output/response data. Can be:
           - 2D array (n_samples, n_trials) - auto-expanded to (n_samples, 1, n_trials)
           - 3D array (n_samples, n_targets, n_trials_y)
           - Outputs object
           n_trials_x and n_trials_y may differ when a custom loss_fn is provided.
        loss_fn (function): Per-sample loss function with full control over loss computation.
                          Defaults to MSE averaged over all targets and trials.
                          Signature: loss_fn(model, x_i, y_i, params) -> scalar
                          where x_i has shape (n_features, n_trials_x),
                          y_i has shape (n_targets, n_trials_y),
                          and params is a 1D array of shape (n_params,).
                          Must be JAX-compatible (supports jit, vmap, grad).
        param_penalty_weight (float): Weight for the penalty on the number of parameters. Default is 0.1.
        fit_params (bool): Whether to fit the parameters of the model. Default is True.
        random_seed (int or None): Random seed for reproducibility. Default is 0.
        FAILED_PROGRAM_COST (float): Cost assigned to failed models. Default is np.inf.
        tol (float): Tolerance for optimization convergence. Default is 1e-2.
        max_iter (int): Maximum number of iterations for optimization. Default is 1_000.
        use_param_estimator (bool): Whether to use the parameter estimator to compute initial parameters. Default is True.
        trial_batch_size (int | None): Number of trials to process per mini-batch
                          to avoid GPU OOM. If None, uses all trials in one batch.

    Returns:
        tuple[
            - float: The cross-validated loss of the model with initial parameters,
            - jnp.ndarray: The initial parameters (n_samples, n_params).
            - float: The average loss on test set after optimization.
                     Returns FAILED_PROGRAM_COST if the model fails.
            - jnp.ndarray: The optimized parameters (n_samples, n_params).
    """
    t_start = time.time()
    
    # Normalize inputs to Inputs format: (n_samples, n_features, n_trials)
    x_inputs = ensure_inputs(x)
    x_data = x_inputs.to_tensor()  # shape: (n_samples, n_features, n_trials)
    
    # Normalize outputs to Outputs format: (n_samples, n_targets, n_trials)
    y_outputs = ensure_outputs(y)
    y_data = y_outputs.data  # shape: (n_samples, n_targets, n_trials)
    
    n_samples, n_features, n_trials_x = x_data.shape
    n_trials_y = y_data.shape[2]
    n_targets = y_outputs.n_targets

    # Split trials for inputs and outputs (may use different indices when mismatched)
    x_train_trial_idx, x_test_trial_idx, y_train_trial_idx, y_test_trial_idx = \
        _call_trial_split(create_train_test_trial_split_fn, n_trials_x, n_trials_y, random_seed)

    # Split inputs and outputs using their respective trial indices
    x_train = x_data[:, :, x_train_trial_idx]  # (n_samples, n_features, n_train_trials_x)
    y_train = y_data[:, :, y_train_trial_idx]   # (n_samples, n_targets, n_train_trials_y)
    x_test = x_data[:, :, x_test_trial_idx]     # (n_samples, n_features, n_test_trials_x)
    y_test = y_data[:, :, y_test_trial_idx]      # (n_samples, n_targets, n_test_trials_y)
    
    # Compute initial parameters
    # param_estimator receives y as (n_targets, n_trials) for each sample
    if use_param_estimator:
        initial_params = compute_initial_params(param_estimator, model, np.asarray(x_train), np.asarray(y_train))
    else:
        initial_params = compute_default_params(model)
        if initial_params is not None:
            initial_params = jnp.repeat(initial_params, n_samples, axis=0)
    
    # Fail immediately if initial_params is None or not a JAX array
    if initial_params is None or not isinstance(initial_params, jnp.ndarray):
        logging.info("Error: initial_params should be a JAX array.")
        return FAILED_PROGRAM_COST, jnp.zeros((n_samples, 0)), FAILED_PROGRAM_COST, jnp.zeros((n_samples, 0))
    if initial_params.ndim != 2 or initial_params.shape[0] != n_samples:
        logging.info(f"Error: initial_params should be a 2D array with shape ({n_samples}, n_params).")
        return FAILED_PROGRAM_COST, jnp.zeros((n_samples, 0)), FAILED_PROGRAM_COST, jnp.zeros((n_samples, 0))
    
    # Fail immediately if fit_params is True and non-numeric params
    n_params = initial_params.shape[1]
    all_numeric = (initial_params.dtype.kind in 'biufc' and 
                   jnp.all(jnp.isfinite(initial_params)))
    if fit_params and not all_numeric:
        logging.info("Error: Cannot fit non-numeric parameters.")
        return FAILED_PROGRAM_COST, jnp.zeros((n_samples, n_params)), FAILED_PROGRAM_COST, jnp.zeros((n_samples, n_params))
    
    # Validate model execution and output shape
    is_valid, error_msg = validate_model_execution(
        model, x_data, initial_params, n_samples,
        expected_n_targets=n_targets,
        n_validation_samples=10
    )
    if not is_valid:
        logging.info(f"Model validation failed: {error_msg}")
        return FAILED_PROGRAM_COST, initial_params, FAILED_PROGRAM_COST, initial_params

    # Per-sample loss function
    def loss_single_sample(params, x_i, y_i):
        return loss_fn(model, x_i, y_i, params)

    # Vectorize over samples
    # params: (n_samples, n_params), x: (n_samples, n_features, n_trials_x), y: (n_samples, n_targets, n_trials_y)
    # Output: (n_samples,)
    loss_total = jax.vmap(loss_single_sample, in_axes=(0, 0, 0), out_axes=0)

    # Mini-batched loss and gradient computation to avoid GPU OOM
    n_train_trials_x = x_train.shape[2]
    n_train_trials_y = y_train.shape[2]
    trials_matched = (n_train_trials_x == n_train_trials_y)
    effective_trial_batch_size = n_train_trials_x if trial_batch_size is None else int(trial_batch_size)
    # Scalar single-feature full-batch fast path only when using default loss
    use_scalar_single_feature_fullbatch = (
        trial_batch_size is None and n_targets == 1 and n_features == 1
        and trials_matched 
    )
    
    @jax.jit
    def loss_single_batch(params_2d, x_batch, y_batch):
        """Compute sum of losses for one batch (JIT-compiled)."""
        batch_losses = loss_total(params_2d, x_batch, y_batch)  # (n_samples,)
        return jnp.sum(batch_losses)

    # Combined loss and gradient computation - more efficient than separate calls
    loss_and_grad_single_batch = jax.jit(jax.value_and_grad(loss_single_batch))

    # JIT-compiled eval (nansum to ignore NaN losses from bad model outputs)
    @jax.jit
    def eval_single_batch(params_2d, x_batch, y_batch):
        """Compute nansum of losses for one batch (JIT-compiled, no grad)."""
        batch_losses = loss_total(params_2d, x_batch, y_batch)
        return jnp.nansum(batch_losses)

    if trials_matched:
        # Standard path: batch over trial dimension with same indices for x and y
        n_train_trials = n_train_trials_x  # same as n_train_trials_y
        def loss_and_grad_batched(params):
            """Compute loss and gradient by accumulating over trial batches."""
            params_2d = params.reshape(-1, n_params)
            total_loss = 0.0
            total_grad = jnp.zeros_like(params)

            for start_idx in range(0, n_train_trials, effective_trial_batch_size):
                end_idx = min(start_idx + effective_trial_batch_size, n_train_trials)
                batch_weight = (end_idx - start_idx) / n_train_trials
                x_batch = x_train[:, :, start_idx:end_idx]
                y_batch = y_train[:, :, start_idx:end_idx]

                batch_loss, batch_grad = loss_and_grad_single_batch(params_2d, x_batch, y_batch)

                total_loss += batch_loss * batch_weight
                total_grad += batch_grad.reshape(-1) * batch_weight

            return total_loss / n_samples, total_grad / n_samples
    else:
        # Mismatched trials: no trial mini-batching. Pass full x_train and y_train.
        # The loss_fn handles the shape relationship internally.
        def loss_and_grad_batched(params):
            """Compute loss and gradient over full data (no trial batching)."""
            params_2d = params.reshape(-1, n_params)
            loss, grad = loss_and_grad_single_batch(params_2d, x_train, y_train)
            return loss / n_samples, grad.reshape(-1) / n_samples
    
    if fit_params:
        # Adam optimizer with learning rate schedule
        # Ensure learning_rate is a Python float (not JAX array) for optax
        learning_rate = float(learning_rate)
        opt = optax.adam(learning_rate, b1=0.9, b2=0.999, eps=1e-8)
        opt_state = opt.init(initial_params.reshape(-1))
        
        if use_scalar_single_feature_fullbatch:
            # Match legacy scalar full-batch path for speed/consistency when no batching is requested.
            loss_param = lambda params: jnp.mean(loss_total(params.reshape(-1, n_params), x_train, y_train))
            loss_param_and_grad = jax.value_and_grad(loss_param)

            @jax.jit
            def train_step(params, opt_state):
                loss, grad = loss_param_and_grad(params)
                updates, opt_state = opt.update(grad, opt_state, params)
                params = optax.apply_updates(params, updates)
                return params, opt_state, loss

            print_every = 50
            params = initial_params.reshape(-1)
            initial_loss = loss_param(params)
            best_loss, best_params = initial_loss.copy(), params.copy()
            for step in range(1, max_iter + 1):
                params, opt_state, loss_val = train_step(params, opt_state)
                if jnp.isnan(loss_val) or jnp.isinf(loss_val) or jnp.any(jnp.isnan(params)) or jnp.any(jnp.isinf(params)):
                    logging.info(f"Loss is NaN or Inf at step {step}. Stopping optimization.")
                    print(f"Final loss: {loss_val:.4f} at step {step}")
                    break
                if loss_val < best_loss:
                    best_loss = loss_val.copy()
                    best_params = params.copy()
                if step % print_every == 0:
                    print(f"step {step:4d}  loss {loss_val:.4f}")
            params = best_params.reshape(n_samples, n_params)
            print(f"params optimized. Loss: {best_loss:.4f}")
        else:
            def train_step(params, opt_state):
                loss, grad = loss_and_grad_batched(params)
                updates, new_opt_state = opt.update(grad, opt_state, params)
                new_params = optax.apply_updates(params, updates)
                return new_params, new_opt_state, loss
            
            print_every = 50
            params = initial_params.reshape(-1)
            initial_loss, _ = loss_and_grad_batched(params)
            
            CATASTROPHIC_LOSS_THRESHOLD = 1e6
            if initial_loss > CATASTROPHIC_LOSS_THRESHOLD:
                print(f"Initial loss {initial_loss:.2e} exceeds threshold. Skipping optimization.")
                logging.info(f"Skipping optimization: initial loss {initial_loss:.2e} > {CATASTROPHIC_LOSS_THRESHOLD:.0e}")
                return FAILED_PROGRAM_COST, initial_params, FAILED_PROGRAM_COST, initial_params
            
            best_loss, best_params = initial_loss.copy(), params.copy()
            for step in range(1, max_iter + 1):
                params, opt_state, loss_val = train_step(params, opt_state)
                if jnp.isnan(loss_val) or jnp.isinf(loss_val) or jnp.any(jnp.isnan(params)) or jnp.any(jnp.isinf(params)):
                    logging.info(f"Loss is NaN or Inf at step {step}. Stopping optimization.")
                    print(f"Final loss: {loss_val:.4f} at step {step}")
                    break
                if loss_val > CATASTROPHIC_LOSS_THRESHOLD:
                    logging.info(f"Loss exploded to {loss_val:.2e} at step {step}. Stopping optimization.")
                    print(f"Loss exploded to {loss_val:.2e}. Stopping optimization.")
                    return FAILED_PROGRAM_COST, initial_params, FAILED_PROGRAM_COST, initial_params
                if loss_val < best_loss:
                    best_loss = loss_val.copy()
                    best_params = params.copy()
                if step % print_every == 0:
                    print(f"step {step:4d}  loss {loss_val:.4f}")
            params = best_params.reshape(n_samples, n_params)
            print(f"params optimized. Loss: {best_loss:.4f}")
    else:
        params = compute_initial_params(param_estimator, model, np.asarray(x_train), np.asarray(y_train))
        if params is None or not isinstance(params, jnp.ndarray):
            logging.info("Error: params should be a JAX array.")
            return FAILED_PROGRAM_COST, jnp.zeros((n_samples, n_params)), FAILED_PROGRAM_COST, jnp.zeros((n_samples, n_params))
    
    # Compute final loss on test set
    if trials_matched:
        def eval_loss_batched(params_2d, x_eval, y_eval):
            """Compute loss by iterating over trial batches (matched trials)."""
            n_eval_trials = x_eval.shape[2]
            weighted_sum = 0.0
            for start_idx in range(0, n_eval_trials, effective_trial_batch_size):
                end_idx = min(start_idx + effective_trial_batch_size, n_eval_trials)
                batch_size = end_idx - start_idx
                x_batch = x_eval[:, :, start_idx:end_idx]
                y_batch = y_eval[:, :, start_idx:end_idx]
                weighted_sum += eval_single_batch(params_2d, x_batch, y_batch) * (batch_size / n_eval_trials)
            return weighted_sum / n_samples
    else:
        def eval_loss_batched(params_2d, x_eval, y_eval):
            """Compute loss over full data (mismatched trials, no batching)."""
            return eval_single_batch(params_2d, x_eval, y_eval) / n_samples
    
    if use_scalar_single_feature_fullbatch:
        initial_loss = jnp.nanmean(loss_total(initial_params, x_test, y_test)) + param_penalty_weight * n_params
    else:
        initial_loss = eval_loss_batched(initial_params, x_test, y_test) + param_penalty_weight * n_params
    n_nans = jnp.sum(jnp.isnan(initial_loss))
    if n_nans > 0:
        print(f"Warning: initial loss contains {n_nans} NaNs.")
    initial_loss = jnp.nan_to_num(initial_loss, nan=FAILED_PROGRAM_COST, posinf=FAILED_PROGRAM_COST, neginf=FAILED_PROGRAM_COST)
    
    if use_scalar_single_feature_fullbatch:
        final_loss = jnp.nanmean(loss_total(params, x_test, y_test)) + param_penalty_weight * n_params
    else:
        final_loss = eval_loss_batched(params, x_test, y_test) + param_penalty_weight * n_params
    n_nans = jnp.sum(jnp.isnan(final_loss))
    if n_nans > 0:
        print(f"Warning: final loss contains {n_nans} NaNs.")
    final_loss = jnp.nan_to_num(final_loss, nan=FAILED_PROGRAM_COST, posinf=FAILED_PROGRAM_COST, neginf=FAILED_PROGRAM_COST)
    
    t_end = time.time()
    print(f"Time taken for optimization: {t_end - t_start:.4f} seconds")
    return float(initial_loss), initial_params, float(final_loss), params


def objective(model, param_estimator, x, y, create_train_test_trial_split_fn=None,
              loss_fn=None,
              param_penalty_weight=0.1, fit_params=True, random_seed=0,
              FAILED_PROGRAM_COST=jnp.inf, tol=1e-2, max_iter=1_000, learning_rate=3e-3,
              use_param_estimator=True, trial_batch_size=None) -> tuple[float, jnp.ndarray, float, jnp.ndarray]:
    """
    Calculate the loss of the model. Always uses vectorized Outputs representation.
    
    This is the main entry point for model evaluation. All outputs are normalized
    to Outputs objects with shape (n_samples, n_targets, n_trials), even for
    single-target (scalar) cases where n_targets=1.
    
    Args:
        model (function): The model which predicts neural activity from inputs
                          and free parameters (for a single sample).
                          Signature: model(X, *params) -> activity
                          where X has shape (n_features, n_trials) for a single sample.
                          Output shape: (n_trials,) for scalar, (n_targets, n_trials) for vectorized.
        param_estimator (function): Function to estimate initial parameters for the model.
                          Signature: param_estimator(X, response) -> params
                          where X has shape (n_features, n_trials) for a single sample.
        x: Input data. Can be:
           - 2D array (n_samples, n_trials) - will be auto-expanded to (n_samples, 1, n_trials)
           - 3D array (n_samples, n_features, n_trials)
           - Inputs object
        y: Output/response data. Always normalized to Outputs object. Can be:
           - 2D array (n_samples, n_trials) - auto-expanded to (n_samples, 1, n_trials)
           - 3D array (n_samples, n_targets, n_trials)
           - Outputs object
        loss_fn (function): Per-sample loss function.
                          Signature: loss_fn(model, x_i, y_i, params) -> scalar.
                          Defaults to MSE over all outputs and trials.
        param_penalty_weight (float): Weight for the penalty on the number of parameters. Default is 0.1.
        fit_params (bool): Whether to fit the parameters of the model. Default is True.
        random_seed (int or None): Random seed for reproducibility. Default is 0.
        FAILED_PROGRAM_COST (float): Cost assigned to failed models. Default is np.inf.
        tol (float): Tolerance for optimization convergence. Default is 1e-2.
        max_iter (int): Maximum number of iterations for optimization. Default is 1_000.
        use_param_estimator (bool): Whether to use the parameter estimator to compute initial parameters. Default is True.
        trial_batch_size (int): Number of trials to process per mini-batch to avoid GPU OOM. Default is 5000.

    Returns:
        tuple[
            - float: The cross-validated loss with initial parameters,
            - jnp.ndarray: The initial parameters.
            - float: The average loss on test set after optimization.
            - jnp.ndarray: The optimized parameters for each sample (n_samples, n_params).
    """
    # Normalize y once to canonical Outputs format and always use vectorized path.
    y_outputs = ensure_outputs(y)
    n_targets = y_outputs.n_targets
    
    return objective_vectorized(
        model=model,
        param_estimator=param_estimator,
        x=x,
        y=y_outputs,
        create_train_test_trial_split_fn=create_train_test_trial_split_fn,
        loss_fn=loss_fn,
        param_penalty_weight=param_penalty_weight,
        fit_params=fit_params,
        random_seed=random_seed,
        FAILED_PROGRAM_COST=FAILED_PROGRAM_COST,
        tol=tol,
        max_iter=max_iter,
        learning_rate=learning_rate,
        use_param_estimator=use_param_estimator,
        trial_batch_size=trial_batch_size,
    )

def evaluate_param_estimator_loss(model, param_estimator, x, y,
                                  create_train_test_trial_split_fn=None,
                                  loss_fn=None,
                                  param_penalty_weight=0.1, random_seed=0,
                                  trial_batch_size=None, FAILED_PROGRAM_COST=jnp.inf):
    """
    Evaluate parameter estimator loss without gradient descent.

    Returns:
        (loss, params) where loss is computed using the objective pipeline with
        fit_params=False and params are the initial parameters from the estimator.
    """
    if create_train_test_trial_split_fn is None:
        def _default_split(n_trials, random_seed=0):
            idx = jnp.arange(n_trials)
            return idx, idx
        split_fn = _default_split
    else:
        split_fn = create_train_test_trial_split_fn

    try:
        initial_loss, initial_params, _, _ = objective(
            model=model,
            param_estimator=param_estimator,
            x=x,
            y=y,
            create_train_test_trial_split_fn=split_fn,
            loss_fn=loss_fn,
            param_penalty_weight=param_penalty_weight,
            fit_params=False,
            random_seed=random_seed,
            FAILED_PROGRAM_COST=FAILED_PROGRAM_COST,
            use_param_estimator=True,
            trial_batch_size=trial_batch_size,
        )
        return float(initial_loss), initial_params
    except Exception as e:
        logging.info(f"Error evaluating parameter estimator loss: {e}")
        return float(FAILED_PROGRAM_COST), None


def _infer_n_features(inputs):
    """Infer feature count from (n_samples, n_trials) or (n_samples, n_features, n_trials)."""
    x_arr = jnp.asarray(inputs)
    if x_arr.ndim == 2:
        return 1
    if x_arr.ndim == 3:
        return int(x_arr.shape[1])
    raise ValueError(f"Expected 2D or 3D inputs, got shape {x_arr.shape}.")


def build_evaluation_points(inputs,
                            n_points=100,
                            random_seed=0):
    """
    Build explicit evaluation points from training inputs.

    - Single-input data: uniform grid across observed input range, broadcast per sample.
    - Multi-input data: sample trial columns from observed inputs.
    """
    x_arr = jnp.asarray(inputs)
    n_features = _infer_n_features(x_arr)
    n_samples = int(x_arr.shape[0])

    if n_features == 1:
        x_min = float(jnp.min(x_arr))
        x_max = float(jnp.max(x_arr))
        if x_max <= x_min:
            x_max = x_min + 1e-6
        grid = jnp.linspace(x_min, x_max, n_points)
        return jnp.broadcast_to(grid, (n_samples, n_points))

    if x_arr.ndim != 3:
        raise ValueError(
            f"Expected 3D input for multi-feature eval points, got shape {x_arr.shape}."
        )

    n_trials = int(x_arr.shape[2])
    n_eval = min(int(n_points), n_trials)
    rng = np.random.default_rng(random_seed)
    trial_idx = rng.choice(n_trials, size=n_eval, replace=False)
    return x_arr[:, :, trial_idx]


def select_evaluation_points(inputs,
                             diagnostics_module=None,
                             n_points=100,
                             random_seed=0):
    """
    Select evaluation points for model diagnostics.

    If a diagnostics module provides `select_evaluation_points`, delegate to it.
    Otherwise, use a generic fallback based on observed input ranges/trials.
    """
    if diagnostics_module is not None and hasattr(diagnostics_module, "select_evaluation_points"):
        selector = diagnostics_module.select_evaluation_points
        try:
            return selector(inputs=inputs, n_points=n_points, random_seed=random_seed)
        except TypeError:
            # Backward compatibility with alternate arg naming.
            return selector(inputs=inputs, n_evaluation_points=n_points, random_seed=random_seed)

    return build_evaluation_points(
        inputs=inputs,
        n_points=n_points,
        random_seed=random_seed,
    )


def compute_evaluation_matrix(program,
                              params,
                              eval_points):
    """
    Compute model evaluations used for logging/comparison.
    """
    if eval_points is None:
        raise ValueError("eval_points must be provided.")

    params_arr = jnp.asarray(params)
    n_samples = params_arr.shape[0]
    program_vmap = utils.vmap_over_cells(program)
    eval_arr = jnp.asarray(eval_points)

    if eval_arr.ndim == 1:
        eval_arr = jnp.broadcast_to(eval_arr, (n_samples, eval_arr.shape[0]))

    if eval_arr.ndim == 2:
        if eval_arr.shape[0] != n_samples:
            raise ValueError(
                f"eval_points first dimension must match n_samples={n_samples}, got {eval_arr.shape}."
            )
        # Backward compatibility: some models expect 1D, others (1, n_trials).
        try:
            return program_vmap(eval_arr, params_arr)
        except Exception:
            return program_vmap(eval_arr[:, jnp.newaxis, :], params_arr)

    if eval_arr.ndim != 3:
        raise ValueError(
            f"eval_points must be 1D, 2D, or 3D, got shape {eval_arr.shape}."
        )
    if eval_arr.shape[0] != n_samples:
        raise ValueError(
            f"eval_points first dimension must match n_samples={n_samples}, got {eval_arr.shape}."
        )

    try:
        return program_vmap(eval_arr, params_arr)
    except Exception:
        if eval_arr.shape[1] == 1:
            return program_vmap(eval_arr[:, 0, :], params_arr)
        raise


def _validate_model_fit_plot_data(plot_data: ModelFitPlotData) -> ModelFitPlotData:
    """Validate `prepare_model_fit_plot_data` outputs and fail early on schema drift."""
    required_keys = (
        "sample_selection",
        "inputs_full",
        "inputs_plot",
        "observed_outputs",
        "trial_predictions",
        "model_loss_dict",
        "n_grid_side",
        "n_models",
        "n_samples",
        "n_trials_x",
        "n_trials_y",
        "input_idx",
    )
    missing = [k for k in required_keys if k not in plot_data]
    if missing:
        raise ValueError(f"plot_data missing required keys: {missing}")

    n_samples = int(plot_data["n_samples"])
    n_models = int(plot_data["n_models"])
    n_trials_x = int(plot_data["n_trials_x"])
    n_trials_y = int(plot_data["n_trials_y"])
    n_grid_side = int(plot_data["n_grid_side"])

    sample_selection = np.asarray(plot_data["sample_selection"])
    inputs_full = jnp.asarray(plot_data["inputs_full"])
    inputs_plot = jnp.asarray(plot_data["inputs_plot"])
    observed_outputs = jnp.asarray(plot_data["observed_outputs"])
    trial_predictions = jnp.asarray(plot_data["trial_predictions"])
    model_loss_dict = plot_data["model_loss_dict"]

    if sample_selection.ndim != 1 or sample_selection.shape[0] != n_samples:
        raise ValueError(
            f"plot_data['sample_selection'] must have shape ({n_samples},), got {sample_selection.shape}."
        )
    if inputs_plot.shape != (n_samples, n_trials_x):
        raise ValueError(
            f"plot_data['inputs_plot'] must have shape ({n_samples}, {n_trials_x}), got {inputs_plot.shape}."
        )
    if inputs_full.ndim != 3 or inputs_full.shape[0] != n_samples or inputs_full.shape[2] != n_trials_x:
        raise ValueError(
            f"plot_data['inputs_full'] must have shape (n_samples, n_features, n_trials_x) with "
            f"n_samples={n_samples}, n_trials_x={n_trials_x}, got {inputs_full.shape}."
        )
    if observed_outputs.shape != (n_samples, n_trials_y):
        raise ValueError(
            f"plot_data['observed_outputs'] must have shape ({n_samples}, {n_trials_y}), got {observed_outputs.shape}."
        )
    if n_trials_x == n_trials_y and trial_predictions is None:
        raise ValueError(
            "plot_data['trial_predictions'] should not be None when n_trials_x == n_trials_y."
        )
    if n_trials_x != n_trials_y and trial_predictions is not None:
        raise ValueError(
            "plot_data['trial_predictions'] should be None when n_trials_x != n_trials_y."
        )
    if trial_predictions is not None and trial_predictions.shape != (n_models, n_samples, n_trials_y):
        raise ValueError(
            f"plot_data['trial_predictions'] must have shape ({n_models}, {n_samples}, {n_trials_y}), got {trial_predictions.shape}."
        )
    if model_loss_dict.keys() != set(range(n_models)):
        raise ValueError(
            f"plot_data['model_loss_dict'] keys must be integers from 0 to n_models-1 ({n_models}), got {model_loss_dict.keys()}."
        )
    if n_grid_side * n_grid_side != n_samples:
        raise ValueError(
            f"plot_data['n_grid_side']={n_grid_side} is inconsistent with n_samples={n_samples}."
        )
    return plot_data


def prepare_model_fit_plot_data(programs_df,
                                inputs,
                                response,
                                sample_selection,
                                loss_fn,
                                input_idx=0) -> ModelFitPlotData:
    """
    Compute canonical plotting tensors for diagnostics `plot_model_fits(plot_data=...)`.

    Returned `plot_data` schema:
    - `sample_selection`: `(n_samples,)` original sample ids selected for plotting.
    - `inputs_full`: `(n_samples, n_features, n_trials_x)` full input tensor.
    - `inputs_plot`: `(n_samples, n_trials_x)` input values used for x-axis plotting.
    - `observed_outputs`: `(n_samples, n_trials_y)` observed targets/outputs.
    - `trial_predictions`: `(n_models, n_samples, n_trials_y)` model predictions on observed trials.
    - `model_loss_dict`: dict mapping model index to overall loss scalar. The value is a list of per-sample losses that can be averaged or plotted separately.
    - `n_grid_side`: subplot side length (`sqrt(n_samples)`).
    - `n_models`: number of candidate models in `programs_df`.
    - `n_samples`: number of plotted samples.
    - `n_trials_x`: number of observed trials per plotted sample (inputs).
    - `n_trials_y`: number of observed trials per plotted sample (outputs).
    - `input_idx`: input feature index used to build `inputs_plot`.

    Why both `inputs_full` and `inputs_plot`:
    - `inputs_full` is required for diagnostics that need the complete feature
      vector per trial (e.g., 2D spatial plots).
    - `inputs_plot` is a 1D projection used by line/scatter diagnostics that
      plot one feature on the x-axis.
    """
    sample_selection = np.asarray(sample_selection)
    if sample_selection.size == 0:
        raise ValueError("sample_selection must not be empty.")
    n_side = int(np.sqrt(sample_selection.size))
    if n_side * n_side != sample_selection.size:
        raise ValueError("sample_selection size must be a square number (e.g., 1,4,9).")

    x_arr = jnp.asarray(inputs)
    y_arr = jnp.asarray(response)
    n_features = _infer_n_features(x_arr)
    if x_arr.ndim == 2 and input_idx != 0:
        raise ValueError("input_idx must be 0 for 2D inputs.")
    if x_arr.ndim == 3 and (input_idx < 0 or input_idx >= n_features):
        raise ValueError(f"input_idx ({input_idx}) must be in range [0, {n_features}).")

    models = programs_df['program'].tolist()
    params_all = [jnp.asarray(p)[sample_selection] for p in programs_df['params'].tolist()]
    observed_outputs = y_arr[sample_selection]

    if x_arr.ndim == 2:
        inputs_full = x_arr[sample_selection][:, jnp.newaxis, :]
        inputs_plot = x_arr[sample_selection]
    else:
        inputs_full = x_arr[sample_selection]
        inputs_plot = x_arr[sample_selection][:, input_idx, :]

    n_models = len(models)
    n_samples = int(inputs_full.shape[0])
    n_trials_x = int(inputs_full.shape[2])
    n_trials_y = int(observed_outputs.shape[-1])

    def _as_trial_vector(arr, expected_len, name):
        """
        Normalize model/loss outputs to a 1D trial vector.

        Accepts scalar (broadcast), (n_trials,), (1, n_trials), or (n_trials, 1).
        """
        vec = jnp.asarray(arr)
        vec = jnp.squeeze(vec)
        if vec.ndim == 0:
            return jnp.broadcast_to(vec, (expected_len,))
        if vec.ndim != 1:
            raise ValueError(
                f"{name} must reduce to 1D shape ({expected_len},), got shape {jnp.asarray(arr).shape}."
            )
        if vec.shape[0] != expected_len:
            raise ValueError(
                f"{name} has length {vec.shape[0]}, expected {expected_len}."
            )
        return vec

    if n_trials_x == n_trials_y:
        trial_predictions = jnp.zeros((n_models, n_samples, n_trials_x))
    else:
        trial_predictions = None 

    model_loss_dict = {}
    for i, model in enumerate(models):
        model_loss_dict[i] = []        
        for c in range(n_samples):
            x = inputs_full[c]
            y = observed_outputs[c]
            params = params_all[i][c]
            model_loss_dict[i].append(loss_fn(model, x, y, params))

            # if n_trials_x == n_trials_y, calculate the per trial prediction for diagnostics 
            if n_trials_x == n_trials_y:
                y_pred_raw = model(x, *params)
                y_pred = _as_trial_vector(y_pred_raw, n_trials_x, "model prediction")
                trial_predictions = trial_predictions.at[i, c].set(y_pred)
    
    plot_data: ModelFitPlotData = {
        'sample_selection': sample_selection,
        'inputs_full': inputs_full,
        'inputs_plot': inputs_plot,
        'observed_outputs': observed_outputs,
        'trial_predictions': trial_predictions,
        'model_loss_dict': model_loss_dict,
        'n_grid_side': n_side,
        'n_models': n_models,
        'n_samples': n_samples,
        'n_trials_x': n_trials_x,
        'n_trials_y': n_trials_y,
        'input_idx': int(input_idx),
    }
    return _validate_model_fit_plot_data(plot_data)


def _call_with_supported_kwargs(func, kwargs):
    """Call a function with only supported keyword args unless it accepts **kwargs."""
    sig = inspect.signature(func)
    accepts_kwargs = any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
    )
    if accepts_kwargs:
        return func(**kwargs)
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return func(**filtered_kwargs)


def prepare_and_plot_model_fits(diagnostics_module,
                                programs_df,
                                loss_fn,
                                inputs,
                                response,
                                sample_selection,
                                **plot_kwargs):
    """
    Prepare `plot_data` and render model-fit diagnostics.

    This wrapper intentionally owns the `diagnostics_module is None` guard so
    the hypothesis engine can run in configurations where image diagnostics are
    disabled. When diagnostics are enabled, it computes canonical `plot_data`
    via `prepare_model_fit_plot_data(...)` and forwards it to
    `diagnostics_module.plot_model_fits(...)`.
    """
    if diagnostics_module is None:
        return

    plot_fn = diagnostics_module.plot_model_fits
    plot_data = prepare_model_fit_plot_data(
        programs_df=programs_df,
        inputs=inputs,
        response=response,
        sample_selection=sample_selection,
        loss_fn=loss_fn,
        input_idx=plot_kwargs.get('input_idx', 0),
    )
    kwargs = dict(
        plot_data=plot_data,
        **plot_kwargs,
    )
    _call_with_supported_kwargs(plot_fn, kwargs)


async def generate_new_model(current_island, llm_name, client, 
                                    spike_matrix, stimuli, prompt_manager,
                                    loss_fn=None,
                                    mode='explore', k_max=2, temp=1, 
                                    thinking_budget=1, img_dir=None, diagnostics_module=None,
                                    island_chat_manager=None, island_id: int = None,
                                    batch_id: int = 0,
                                    use_large_model: bool = True):
    k = min(k_max, len(current_island))
    random_programs = current_island.sample(k, replace=False).reset_index(drop=True)
    random_programs = random_programs.sort_values(by='train_loss', ascending=False).reset_index(drop=True)
    # save parent1_id and parent2_id. These are strings of the form "(iteration_number)_(birth_island)_(batch_index)"
    parent1_id = (random_programs['iteration_number'][0], 
                  random_programs['birth_island'][0], 
                  random_programs['batch_index'][0])
    parent2_id = (random_programs['iteration_number'][1],
                  random_programs['birth_island'][1], 
                  random_programs['batch_index'][1])
    use_image = img_dir is not None
    use_chat_mode = island_chat_manager is not None and island_id is not None
    model_name = prompt_manager.get_model_name()
    
    # Use appropriate prompt function based on mode
    if use_chat_mode:
        program_prompt = prompt_manager.get_program_prompt(random_programs, mode=mode, use_image=use_image)
    else:
        program_prompt = prompt_manager.get_program_prompt_legacy(random_programs, mode=mode, use_image=use_image)

    if use_image and diagnostics_module is not None:
        try:
            sup_title = "".join([f"{model_name}_v{i+1}: Loss = {random_programs['train_loss'][i]:.2f} \n" for i in range(min(3, len(random_programs)))])
            prepare_and_plot_model_fits(
                diagnostics_module=diagnostics_module,
                programs_df=random_programs,
                loss_fn=loss_fn,
                inputs=stimuli,
                response=spike_matrix,
                sample_selection=np.random.choice(spike_matrix.shape[0], size=9, replace=False),
                save_path=img_dir,
                labels=[f'{model_name}_v_1', f'{model_name}_v_2'],
                colours=['tab:green', 'tab:red'],
                dpi=384 * 3 / 20,
                title=sup_title,
                legend_fontsize=20,
                line_alpha=0.9,
                line_width=4,
            )
            
            img_path = Path(img_dir)
            with img_path.open("rb") as f:
                img_bytes = f.read()
        except Exception as e:
            logging.info(f"Error generating image for neuron model prompt: {e}")
            img_bytes = None
            # if we can't generate an image, we will just use the text prompt without image
            use_image = False
    else:
        img_bytes = None
    
    # Use chat-based or legacy LLM call
    if island_chat_manager is not None and island_id is not None:
        llm_output = await island_chat_manager.ask_island(
            island_id, program_prompt,
            batch_id=batch_id,
            mode=mode, 
            use_large_model=use_large_model,
            png_img=img_bytes
        )
    else:
        # Legacy: independent query
        llm_output = await llm_helper.call_llm_async(program_prompt, model_name=llm_name, client=client, temperature=temp, 
                                                thinking_budget=thinking_budget, img_bytes=img_bytes)
    
    code_string = utils.extract_code_block(llm_output)
    if code_string is None:
        return None, None, (parent1_id, parent2_id)
    code_string = code_string.replace(f'def {model_name}_v{k+1}(', f'def {model_name}(')
    code_string = _enforce_single_feature_code_access(
        code_string, stimuli=stimuli, code_label="Model generation"
    )
    
    return code_string, program_prompt, (parent1_id, parent2_id)

async def generate_new_parameter_estimator(current_island, 
                                           model_code_string: str,
                                           model_fn,
                                           llm_name, client, 
                                           spike_matrix, stimuli, prompt_manager,
                                           mode='explore', k_max=1, temp=1,
                                           param_estimator_max_lines=100, img_dir=None,
                                           swear_words=None,
                                           refine_rounds: int = 0,
                                           param_penalty_weight: float = 0.1,
                                           create_train_test_trial_split_fn=None,
                                           random_seed: int = 0,
                                           island_chat_manager=None, island_id: int = None,
                                           batch_id: int = 0,
                                           diagnostics_module=None,
                                           use_large_model: bool = False):                                           
    if model_code_string is None:
        logging.info("No model code string provided, skipping parameter estimator generation.")
        return None, None
    k = min(k_max, len(current_island))
    random_programs = current_island.sample(k, replace=False).reset_index(drop=True)
    # sort from worst to best (loss descending)
    random_programs = random_programs.sort_values(by='train_loss', ascending=False).reset_index(drop=True)
    use_image = img_dir is not None
    use_chat_mode = island_chat_manager is not None and island_id is not None
    
    # Use appropriate prompt function based on mode
    if use_chat_mode:
        prompt = prompt_manager.get_parameter_estimator_prompt(random_programs,
                                                        model_code_string=model_code_string,
                                                        max_lines=param_estimator_max_lines,
                                                        use_image=use_image)
    else:
        prompt = prompt_manager.get_parameter_estimator_prompt_legacy(random_programs,
                                                        model_code_string=model_code_string,
                                                        max_lines=param_estimator_max_lines,
                                                        use_image=use_image)
    
    random_programs_crude = random_programs.copy()
    random_programs_crude['params'] = random_programs['initial_params']
    # now try generating an image from the random programs
    if use_image and diagnostics_module is not None:
        try:
            sup_title = "".join([f"model_v{i+1}: Loss = {random_programs['train_loss'][i]:.2f} \n" for i in range(min(3, len(random_programs)))])
            prepare_and_plot_model_fits(
                diagnostics_module=diagnostics_module,
                programs_df=random_programs_crude,
                loss_fn=loss_fn,
                inputs=stimuli,
                response=spike_matrix,
                sample_selection=np.random.choice(spike_matrix.shape[0], size=4, replace=False),
                save_path=img_dir,
                labels=['v_1', 'v_2'],
                colours=['tab:green', 'tab:red'],
                dpi=384 * 2 / 20,
                title=sup_title,
                legend_fontsize=20,
                line_alpha=0.9,
                line_width=4,
            )
            img_path = Path(img_dir)
            with img_path.open("rb") as f:
                img_bytes = f.read()
        except Exception as e:
            logging.info(f"Error generating image for parameter estimator prompt: {e}")
            img_bytes = None
            # if we can't generate an image, we will just use the text prompt without image
            use_image = False
    else:
        img_bytes = None
    
    # Use chat-based or legacy LLM call
    if island_chat_manager is not None and island_id is not None:
        llm_output = await island_chat_manager.ask_island(
            island_id, prompt,
            batch_id=batch_id,
            mode=mode,
            use_large_model=use_large_model,
            png_img=img_bytes
        )
    else:
        # Legacy: independent query
        llm_output = await llm_helper.call_llm_async(prompt, model_name=llm_name, client=client, temperature=temp,
                                                thinking_budget=0.25, img_bytes=img_bytes)
    # extract the code block from the LLM output
    code_string = utils.extract_code_block(llm_output)
    if code_string is None:
        logging.info("No code block found in the LLM output for parameter estimator, skipping.")
        return None, None
    contains_swear_word = any(word in code_string for word in swear_words)
    if contains_swear_word:
        # find the word that is in the code_string
        swear_word = next((word for word in swear_words if word in code_string), None)
        logging.info(f"Parameter estimator code contains swear word: {swear_word}, skipping.")
        logging.info(f"Code string was:\n{code_string}")
        return None, None
    code_string = code_string.replace(f'def parameter_estimator_v{k+1}(', 'def parameter_estimator(')
    code_string = _enforce_single_feature_code_access(
        code_string, stimuli=stimuli, code_label="Parameter estimator generation"
    )
    func = utils.str_to_func(code_string, 'parameter_estimator')

    if func is None:
        logging.info("Failed to parse parameter estimator code, skipping.")
        return None, None

    if refine_rounds <= 0 or model_fn is None:
        return code_string, func

    best_code = code_string
    best_func = func
    best_loss = float(jnp.inf)

    current_code = code_string
    current_func = func
    current_loss, current_params = evaluate_param_estimator_loss(
        model=model_fn,
        param_estimator=current_func,
        x=stimuli,
        y=spike_matrix,
        create_train_test_trial_split_fn=create_train_test_trial_split_fn,
        param_penalty_weight=param_penalty_weight,
        random_seed=random_seed,
    )

    if current_loss < best_loss:
        best_loss = current_loss
        best_code = current_code
        best_func = current_func

    for r in range(refine_rounds):
        img_bytes = None
        refine_img_path = None
        if diagnostics_module is not None and img_dir is not None and current_params is not None:
            try:
                base_path = Path(img_dir)
                refine_img_path = base_path.with_name(f"{base_path.stem}_refine_{r+1}{base_path.suffix}")

                programs_df = pd.DataFrame({
                    'program': [model_fn],
                    'params': [current_params],
                })
                n_cells = spike_matrix.shape[0]
                n_cells_img = min(4, n_cells)
                sample_selection = np.random.choice(n_cells, size=n_cells_img, replace=False)
                prepare_and_plot_model_fits(
                    diagnostics_module=diagnostics_module,
                    programs_df=programs_df,
                    loss_fn=loss_fn,
                    inputs=stimuli,
                    response=spike_matrix,
                    sample_selection=sample_selection,
                    save_path=str(refine_img_path),
                    labels=[f"refine_{r+1}"],
                    colours=['tab:green'],
                    dpi=100.0,
                    title=f"Param estimator refinement {r+1}/{refine_rounds} (no-GD loss={current_loss:.4f})",
                )
                with refine_img_path.open("rb") as f:
                    img_bytes = f.read()
            except Exception as e:
                logging.info(f"Error generating refinement image: {e}")
                img_bytes = None

        # Build refinement prompt using current estimator as the only parent
        refinement_df = pd.DataFrame({
            'train_loss': [current_loss],
            'program_code_string': [model_code_string],
            'parameter_estimator_code_string': [current_code],
        })

        refine_header = (
            f"Refinement round {r+1}/{refine_rounds}.\n"
            f"Current no-GD loss: {current_loss:.4f}.\n"
            "Improve the parameter estimator without using gradient descent or external optimizers.\n"
            "Return only the updated parameter_estimator code.\n"
        )

        if use_chat_mode:
            refine_prompt = prompt_manager.get_parameter_estimator_prompt(
                refinement_df,
                model_code_string=model_code_string,
                max_lines=param_estimator_max_lines,
                use_image=img_bytes is not None,
            )
        else:
            refine_prompt = prompt_manager.get_parameter_estimator_prompt_legacy(
                refinement_df,
                model_code_string=model_code_string,
                max_lines=param_estimator_max_lines,
                use_image=img_bytes is not None,
            )
        refine_prompt = refine_header + "\n" + refine_prompt

        # Call LLM for refinement
        if island_chat_manager is not None and island_id is not None:
            llm_output = await island_chat_manager.ask_island(
                island_id, refine_prompt,
                batch_id=batch_id,
                mode=mode,
                use_large_model=use_large_model,
                png_img=img_bytes
            )
        else:
            llm_output = await llm_helper.call_llm_async(
                refine_prompt,
                model_name=llm_name,
                client=client,
                temperature=temp,
                thinking_budget=0.25,
                img_bytes=img_bytes
            )

        new_code = utils.extract_code_block(llm_output)
        if new_code is None:
            logging.info("No code block found in refinement output; keeping current estimator.")
            continue
        if any(word in new_code for word in swear_words):
            logging.info("Refinement code contains banned words; skipping.")
            continue

        new_code = re.sub(r"def\s+parameter_estimator_v\d+\s*\(", "def parameter_estimator(", new_code)
        new_func = utils.str_to_func(new_code, 'parameter_estimator')
        if new_func is None:
            logging.info("Failed to parse refined parameter estimator; keeping current.")
            continue

        new_loss, new_params = evaluate_param_estimator_loss(
            model=model_fn,
            param_estimator=new_func,
            x=stimuli,
            y=spike_matrix,
            create_train_test_trial_split_fn=create_train_test_trial_split_fn,
            param_penalty_weight=param_penalty_weight,
            random_seed=random_seed,
        )

        current_code = new_code
        current_func = new_func
        current_loss = new_loss
        current_params = new_params

        if new_loss < best_loss:
            best_loss = new_loss
            best_code = new_code
            best_func = new_func

    return best_code, best_func

async def not_used_yet_generate_new_parameter_estimator_from_image_feedback(image_prompt: str,
                                                               image_dir: str,
                                                               model_name='gemini-2.0-flash',
                                                               swear_words=None,
                                                               max_lines=100,
                                                               temp=1,
                                                               client=None) -> tuple[str, callable]:
    """ Generates a new parameter estimator from an image feedback prompt.
    Args:
        image_prompt (str): The prompt string for the AI to generate a new parameter estimator.
        image_dir (str): Directory where the image is stored.
        swear_words (list): List of words that should not be present in the generated code.
        max_lines (int): Maximum number of lines for the generated code.
        client: The genai client to use for LLM calls.
    Returns:
        tuple[str, callable]: The generated parameter estimator code string and the function object.
    """
    if image_prompt is None or image_dir is None:
        logging.info("No image prompt or image directory provided for parameter estimator generation.")
        return None, None
    # load image as bytes
    image_path = Path(image_dir)
    if not image_path.exists():
        logging.info(f"Image path {image_path} does not exist, skipping parameter estimator generation from image feedback.")
        return None, None
    with image_path.open("rb") as f:
        img_bytes = f.read()
    # call the LLM with the image prompt and image bytes
    llm_output = await llm_helper.call_llm_async(image_prompt, model_name=model_name, client=client, temperature=temp, img_bytes=img_bytes)
    code_string = utils.extract_code_block(llm_output) # extract the code block from the LLM output
    if code_string is None:
        logging.info("No code block found in the LLM output for parameter estimator from image feedback, skipping.")
        return None, None
    # check for swear words
    contains_swear_word = any(word in code_string for word in swear_words)
    if contains_swear_word:
        swear_word = next((word for word in swear_words if word in code_string), None)
        logging.info(f"Parameter estimator code contains swear word: {swear_word}, skipping.")
        return None, None
    # extract the function from the code string
    func = utils.str_to_func(code_string, 'parameter_estimator')
    return code_string, func

async def translate_to_jax(code_string: str, client, prompt_manager, llm_name='gemini-2.0-flash-lite') -> tuple[str, callable]:
    """
    Translates a neuron model code string to JAX format.
    Args:
        code_string (str): The neuron model code string to translate.
        client: The LLM client.
        prompt_manager: PromptManager instance for generating prompts.
        llm_name (str): The LLM model name to use.
    Returns:
        callable: The translated JAX function.
    """
    if code_string is None:
        logging.info("No neuron model code string provided for translation.")
        return None, None
    
    prompt = prompt_manager.get_jax_translator_prompt(code_string)
    # print(f"Translating neuron model to JAX with prompt:\n{prompt}")
    if prompt is None:
        return None, None
    
    jax_code_string = await llm_helper.call_llm_async(prompt, client=client, model_name=llm_name, temperature=0)
    jax_code_string = utils.extract_code_block(jax_code_string)
    model_name = prompt_manager.get_model_name()
    func = utils.str_to_func(jax_code_string, model_name)
    return jax_code_string, func


def _prepare_seed_translation_check_data(inputs, response, sample_idx=0, max_trials=32):
    if inputs.ndim == 2:
        x_full = np.asarray(inputs[sample_idx])[None, :]
    else:
        x_full = np.asarray(inputs[sample_idx])

    if response.ndim == 3:
        y_full = np.asarray(response[sample_idx, 0])
    else:
        y_full = np.asarray(response[sample_idx])

    n_trials = x_full.shape[-1]
    if max_trials is not None and n_trials > max_trials:
        trial_idx = np.linspace(0, n_trials - 1, num=max_trials, dtype=int)
        x_check = x_full[..., trial_idx]
    else:
        trial_idx = None
        x_check = x_full

    return x_full, y_full, x_check, trial_idx


def _check_jax_translation(np_func, jax_func, param_estimator, inputs, response,
                           max_trials=32, rtol=1e-4, atol=1e-4) -> None:
    def _coerce_trial_vector(pred, n_trials, label):
        """
        Normalize scalar-model predictions to shape (n_trials,).

        Accepts:
        - (n_trials,)
        - (1, n_trials)
        """
        arr = np.asarray(pred)
        if arr.ndim == 1 and arr.shape[0] == n_trials:
            return arr
        if arr.ndim == 2 and arr.shape == (1, n_trials):
            return arr[0]
        raise ValueError(
            f"{label} output has unsupported shape {arr.shape}; "
            f"expected one of ({n_trials},), (1, {n_trials})."
        )

    x_full, y_full, x_check, _ = _prepare_seed_translation_check_data(
        inputs, response, sample_idx=0, max_trials=max_trials
    )
    try:
        params = param_estimator(x_full, y_full)
    except Exception as e:
        n_features = int(np.asarray(x_full).shape[0])
        raise ValueError(
            f"Parameter estimator failed during translation check: {e}. "
            f"Input has {n_features} feature(s); ensure code only accesses valid X[i] indices."
        ) from e
    params = np.asarray(params).reshape(-1)

    try:
        np_pred = np.asarray(np_func(x_check, *params))
        jax_pred = np.asarray(jax_func(jnp.asarray(x_check), *params))
    except Exception as e:
        n_features = int(np.asarray(x_check).shape[0])
        raise ValueError(
            f"Model execution failed during translation check: {e}. "
            f"Input has {n_features} feature(s); ensure code only accesses valid X[i] indices."
        ) from e

    n_trials = int(np.asarray(x_check).shape[-1])
    np_pred = _coerce_trial_vector(np_pred, n_trials, "NumPy model")
    jax_pred = _coerce_trial_vector(jax_pred, n_trials, "JAX model")

    if np_pred.shape != jax_pred.shape:
        raise ValueError(
            f"JAX translation shape mismatch: numpy={np_pred.shape}, jax={jax_pred.shape}."
        )

    if not np.allclose(np_pred, jax_pred, rtol=rtol, atol=atol):
        max_abs_diff = float(np.max(np.abs(np_pred - jax_pred)))
        raise ValueError(
            "JAX translation failed numeric check for seed program. "
            f"max_abs_diff={max_abs_diff:.6g}, rtol={rtol}, atol={atol}."
        )

async def hypothesis_engine(n_iterations=9, time_limit=60, k_max=2, n_islands=8, batch_size=6, 
                critical_population_size=12, min_wise_population_size=0, 
                n_migrants=2, fit_params=True, tol=1e-6, exploit_point=0.5,
                param_penalty_weight=0.01, FAILED_PROGRAM_COST=np.inf,
                use_image_feedback=True, use_param_estimator=True,
                use_chat_mode=False,  # If True, use persistent chat sessions per island (expensive)
                chat_token_limit=50000,  # Max tokens per chat before auto-summarize and reset. 0 = unlimited
                param_estimator_refinement_rounds=0,
                exploration_topology = [1, 2, 3, 4, 5, 6, 7, 0],
                exploitation_topology = [1, 2, 3, 4, 5, 6, 7, 0],
                tiny_lm_name = 'gemini-2.0-flash-lite',
                little_lm_name = 'gemini-2.0-flash',
                large_lm_name = 'gemini-2.5-flash',
                use_large_every = 3,
                training_sample_ratio = 0.5, 
                max_iter = 1_000,
                learning_rate = 3e-3,
                use_large_model_for_param_estimators=False,
                numpy_programs = None,
                jax_programs = None,
                param_estimators = None,
                load_and_process_data_fn = None,
                create_train_test_sample_split_fn = None,
                create_train_test_trial_split_fn = None,
                data_processing_params = None,
                diagnostics_module = None,
                prompt_manager = None,
                log_best_loss = True,
                trial_batch_size = None,
                swear_words = None,
                loss_fn = None,
                random_seed = 0, # consider setting up a seed_manager to make behaviours more robustly reproducible.
                ):
    """ 
    Main function to run the hypothesis engine.
    
    Args:
        data_processing_params: Dict containing all data loading parameters. This is passed
                     directly to load_and_process_data_fn, which extracts whatever
                     parameters it needs. This allows different experiments to have
                     different parameter sets without changing hypothesis_engine.
        log_best_loss: If True, logs the best loss at each iteration to a CSV file
                       for live monitoring. The file is saved to the experiment
                       output directory as 'best_loss_log.csv'.
    """
    if data_processing_params is None:
        data_processing_params = {}

    # Set default loss_fn if none provided
    if loss_fn is None:
        raise ValueError("loss_fn must be provided. This is unexpected to be None since we expect to use the default MSE loss specifed in the DEFAULT config")
    # load api keys
    load_dotenv()
    client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

    # Initialize IslandChatManager if using chat mode
    island_chat_manager = None
    if use_chat_mode:
        # Create IslandChatManager with mode-aware system instructions
        island_chat_manager = llm_helper.IslandChatManager(
            client=client,
            get_system_instruction=prompt_manager.get_system_instruction,
            small_model_name=little_lm_name,
            large_model_name=large_lm_name,
            explore_temperature=1.5,  # Higher temperature for creative exploration
            exploit_temperature=0.7,  # Lower temperature for focused exploitation
            thinking_budget_fraction=1.0,
            chat_token_limit=chat_token_limit,
            batch_size=batch_size
        )
        logging.info(f"Initialized IslandChatManager with models {little_lm_name} / {large_lm_name}")
        print(f"Chat mode enabled: using persistent chat sessions per (island, batch) pair")
        print(f"  - Total chats: {n_islands * batch_size} ({n_islands} islands × {batch_size} batches)")
        print(f"  - Explore: T=1.5, Exploit: T=0.7")
        print(f"  - Small model: {little_lm_name}, Large model: {large_lm_name}")
        print(f"  - Token limit per chat: {chat_token_limit} (0 = unlimited)")
    else:
        logging.info("Chat mode disabled: using independent LLM queries")
        print("Chat mode disabled: using independent LLM queries")

    # raise error if numpy_programs or param_estimators are invalid
    if numpy_programs is None or len(numpy_programs) != 2:
        raise ValueError("numpy_programs must be a list of 2 functions.")
    if param_estimators is None or len(param_estimators) != 2:
        raise ValueError("param_estimators must be a list of 2 functions.")
    if jax_programs is not None and len(jax_programs) != 2:
        raise ValueError("jax_programs must be a list of 2 functions when provided.")

    data_dict = load_and_process_data_fn(**data_processing_params)
    inputs, outputs = normalize_loaded_data(data_dict)

    n_good_samples, n_features, n_trials_x = inputs.shape
    n_trials_y = outputs.shape[2]
    n_targets = outputs.shape[1]
    if n_targets != 1:
        raise ValueError(
            "Current hypothesis_engine evolutionary loop assumes scalar targets "
            f"(n_targets=1) for prompting/diagnostics, got n_targets={n_targets}."
        )
    response = scalar_outputs_view(outputs)  # 2D view for scalar-only boundaries

    sample_split_random_seed = 42
    training_samples, test_samples = create_train_test_sample_split_fn(n_good_samples, training_sample_ratio, random_seed = sample_split_random_seed) # use 42 to keep orientation_gratings result
    inputs_train, inputs_test = inputs[training_samples, :], inputs[test_samples, :]
    outputs_train, outputs_test = outputs[training_samples, :], outputs[test_samples, :]
    response_train, response_test = response[training_samples, :], response[test_samples, :]
    # Use run-level seed for objective() trial split so reporting and runtime align.
    if n_trials_x == n_trials_y:
        print(f"Loaded {n_good_samples} samples, {n_trials_x} trials per sample.")
    else:
        print(f"Loaded {n_good_samples} samples, {n_trials_x} input trials and {n_trials_y} output trials per sample.")
    print(f"Using {len(training_samples)} samples for training and {len(test_samples)} samples for testing.")

    jax_program_code_strings = None
    if jax_programs is None:
        logging.info("No JAX seed programs provided; translating NumPy seeds to JAX via LLM.")
        model_name = prompt_manager.get_model_name()
        seed_code_strings = [
            utils.format_function_source(program, f'{model_name}_v{i+1}', 'import numpy as np')
            for i, program in enumerate(numpy_programs)
        ]
        translation_tasks = [
            translate_to_jax(code_string, client, prompt_manager, tiny_lm_name)
            for code_string in seed_code_strings
        ]
        jax_results = await asyncio.gather(*translation_tasks)

        jax_programs = []
        jax_program_code_strings = []
        for i, (jax_code_string, jax_func) in enumerate(jax_results):
            if jax_func is None or jax_code_string is None:
                raise ValueError(f"JAX translation failed for seed program {i + 1}.")
            _check_jax_translation(
                numpy_programs[i],
                jax_func,
                param_estimators[i],
                inputs_train,
                outputs_train,
            )
            jax_programs.append(jax_func)
            jax_program_code_strings.append(jax_code_string)

    # create a dataframe to store the programs in each island
    islands = []
    for _ in range(n_islands):
        islands.append(pd.DataFrame(columns=['program_code_string', 'program', 'parameter_estimator_code_string', 'parameter_estimator',
                                             'iteration_number', 'birth_island', 'batch_index', 'train_loss', 'test_loss', 'params',
                                             'initial_loss', 'initial_params', 'llm_name', 'parent1_id', 'parent2_id', 'evaluation_matrix']))
    initial_programs = pd.DataFrame([])

    # wherever you run “python script.py” from…
    base_dir = os.path.join(os.getcwd(), 'program_databases')
    print("Base directory:", base_dir)
    os.makedirs(base_dir, exist_ok=True)
    date_stamp = pd.Timestamp.now().strftime("%m-%d")
    time_stamp = pd.Timestamp.now().strftime("%H-%M-%S")
    full_dir = os.path.join(base_dir, date_stamp, time_stamp)
    os.makedirs(full_dir, exist_ok=True)
    print("Created folder:", full_dir)
    # create a directory for image diagnostics
    image_feedback_dir = os.path.join(full_dir, 'image_feedback')
    os.makedirs(image_feedback_dir, exist_ok=True)
    print("Created image feedback folder:", image_feedback_dir)

    # Save data summary CSV for inspection
    save_data_summary(
        response=outputs,
        inputs=inputs,
        training_samples=training_samples,
        test_samples=test_samples,
        output_dir=full_dir,
        random_seed=random_seed,
        create_train_test_sample_split_fn=create_train_test_sample_split_fn,
        create_train_test_trial_split_fn=create_train_test_trial_split_fn,
    )

    # census[i] = [generation, island, batch_index, llm_name, loss, time, parent1_id, parent2_id, evaluation_matrix, n_free_params]
    census = []
    
    # Initialize best loss tracking for live monitoring
    best_loss_log = []  # List of dicts: {iteration, timestamp, best_train_loss, best_island, ...}
    best_loss_path = os.path.join(full_dir, 'best_loss_log.csv') if log_best_loss else None
    evaluation_points_train = select_evaluation_points(
        inputs_train,
        diagnostics_module=diagnostics_module,
        n_points=100,
        random_seed=random_seed,
    )
    
    # store and compute loss of 2 initial programs
    t_start = time.time()
    seed_losses = np.zeros(2)
    model_name = prompt_manager.get_model_name()
    for i in range(2):
        # get the program, parameter estimator, and jax program
        program_num = numpy_programs[i]
        param_est = param_estimators[i]
        program_jax = jax_programs[i]
        # score the initial program
        loss_init, params_init, loss, params = objective(program_jax, param_est,
                                        x=inputs_train, y=outputs_train,
                                        create_train_test_trial_split_fn=create_train_test_trial_split_fn,
                                        loss_fn=loss_fn,
                                        fit_params=fit_params, param_penalty_weight=param_penalty_weight, tol=tol, learning_rate=learning_rate,
                                        use_param_estimator=use_param_estimator, max_iter=max_iter, trial_batch_size=trial_batch_size,
                                        random_seed=random_seed)
        print(f"Initial program {i + 1} loss before parameter fitting: {loss_init:.2f} and loss after fitting: {loss:.2f}")

        seed_losses[i] = loss
        # format strings
        program_code_string = utils.format_function_source(
            program_num, f'{model_name}_v{i+1}', 'import numpy as np'
        )
        parameter_estimator_code_string = utils.format_function_source(
            param_est, f'parameter_estimator_v{i+1}', 'import numpy as np'
        )
        if jax_program_code_strings is not None:
            program_jax_code_string = jax_program_code_strings[i]
        else:
            program_jax_code_string = utils.format_function_source(
                program_jax, f'{model_name}_v{i+1}', 'import jax.numpy as jnp'
            )
        y_eval = compute_evaluation_matrix(
            program_jax,
            params,
            eval_points=evaluation_points_train,
        )

        new_program_df = pd.DataFrame({'program_code_string': program_code_string,
                                    'program': program_jax,
                                    'parameter_estimator_code_string': parameter_estimator_code_string,
                                    'parameter_estimator': param_est,
                                    'iteration_number': -1,
                                    'birth_island': -1,  # Birth island is set to a special value for initial programs
                                    'batch_index': i,
                                    'train_loss': loss, 
                                    'test_loss': None,  # all test losses will be computed at the end
                                    'llm_name': None,
                                    'params': [params],
                                    'initial_loss': loss_init,
                                    'initial_params': [params_init],
                                    'parent1_id': None,
                                    'parent2_id': None,
                                    'evaluation_matrix': [y_eval]})
        initial_programs = pd.concat([initial_programs, new_program_df], ignore_index=True)
        print(f"Initial program {i + 1} loss: {loss:.2f}")
        census.append([-1, -1, i, None, loss, time.time() - t_start, None, None, y_eval, params.shape[1]])

    # seed each island with the initial programs
    for i in range(n_islands):
        islands[i] = pd.concat([islands[i], initial_programs], ignore_index=True)

    # Reset logging configuration
    log_file = os.path.join(full_dir, 'hypothesis_engine.log')
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(filename=log_file, level=logging.INFO, format='%(message)s')
    
    # Log the IslandChatManager configuration (including system instruction)
    if island_chat_manager is not None:
        island_chat_manager.log_configuration()
    
    if diagnostics_module is not None:
        prepare_and_plot_model_fits(
            diagnostics_module=diagnostics_module,
            programs_df=initial_programs,
            loss_fn=loss_fn,
            inputs=inputs_train,
            response=response_train,
            sample_selection=np.random.choice(len(inputs_train), size=9, replace=False),
            save_path=os.path.join(image_feedback_dir, 'initial_programs.png'),
            labels=['seed_1', 'seed_2'],
            colours=['tab:green', 'tab:red'],
            dpi=100.0,
            title="Seed Programs",
            legend_fontsize=20,
            line_alpha=0.9,
            line_width=4,
        )

    # -----------------------------
    # HYPOTHESIS ENGINE
    # -----------------------------
    for i in tqdm(range(n_iterations), desc="Hypothesis Engine Iterations"):
        # check if time limit is reached
        if time.time() - t_start > time_limit * 60:
            logging.info(f"Time limit of {time_limit} minutes reached. Stopping iterations.")
            break
        
        # Reset per-iteration token tracking (if using chat mode)
        if island_chat_manager is not None:
            island_chat_manager.start_iteration()
        
        logging.info(f"Iteration {i}")
        if use_large_every > 0 and i % use_large_every == 0:
            llm_name = large_lm_name
            logging.info(f"Using large LLM: {llm_name}")
        else:
            llm_name = little_lm_name
            logging.info(f"Using little LLM: {llm_name}")
        use_large_model = (llm_name == large_lm_name)  # Track whether using large model
        mode = 'explore' if i < n_iterations * exploit_point else 'exploit'
        temperature = 1 + np.exp(-i / n_iterations)
        model_image_dirs = np.empty((n_islands, batch_size), dtype=object)
        # param_est_image_dirs = np.empty((n_islands, batch_size), dtype=object)
        for island_idx in range(n_islands):
            for j in range(batch_size):
                if use_image_feedback:
                    model_image_dirs[island_idx, j] = os.path.join(image_feedback_dir, f'iter_{i}_island_{island_idx}_batch_{j}.png')
                    # param_est_image_dirs[island_idx, j] = os.path.join(image_feedback_dir, f'iter_{i}_island_{island_idx}_batch_{j}_param_est.png')
                else:
                    model_image_dirs[island_idx, j] = None
                    # param_est_image_dirs[island_idx, j] = None
        # generate new programs
        model_generation_tasks = [generate_new_model(islands[island_idx], 
                                                                   llm_name=llm_name, 
                                                                   client=client, 
                                                                   mode=mode, 
                                                                   k_max=k_max, 
                                                                   temp=temperature,
                                                                   spike_matrix=response_train, 
                                                                   stimuli=inputs_train,
                                                                   prompt_manager=prompt_manager,
                                                                   loss_fn=loss_fn,
                                                                   img_dir=model_image_dirs[island_idx, j],
                                                                   diagnostics_module=diagnostics_module,
                                                                   island_chat_manager=island_chat_manager,
                                                                   island_id=island_idx,
                                                                   batch_id=j,
                                                                   use_large_model=use_large_model) 
                                         for island_idx in range(n_islands) for j in range(batch_size)]
        logging.info(f"Generating {n_islands * batch_size} new programs... LLM Model: {llm_name}, mode: {mode}, temperature: {temperature:.2f}")
        print(f"Generating {n_islands * batch_size} new programs... LLM Model: {llm_name}, mode: {mode}, temperature: {temperature:.2f}")
        model_results = await asyncio.gather(*model_generation_tasks)
        model_code_strings = [result[0] for result in model_results]
        model_prompts = [result[1] for result in model_results]
        parent_ids = [result[2] for result in model_results]
        
        # convert to jax
        model_function_translation_tasks = [translate_to_jax(code_string, client, prompt_manager, tiny_lm_name) for code_string in model_code_strings]
        jax_results = await asyncio.gather(*model_function_translation_tasks)
        model_results = [(model_code_strings[j], model_prompts[j], jax_results[j][0], jax_results[j][1]) for j in range(n_islands * batch_size)]
        
        # build parameter‑estimator tasks
        param_estimation_tasks = [
            generate_new_parameter_estimator(
                current_island=islands[island_idx],
                model_code_string=model_code_strings[island_idx * batch_size + j],
                model_fn=model_results[island_idx * batch_size + j][3],
                llm_name=little_lm_name,
                client=client,
                spike_matrix=response_train,
                stimuli=inputs_train,
                prompt_manager=prompt_manager,
                mode=mode,
                k_max=2,
                temp=temperature,
                param_estimator_max_lines=100,
                img_dir=os.path.join(image_feedback_dir, f'iter_{i}_island_{island_idx}_batch_{j}_param_estimator.png') if use_large_model_for_param_estimators else None,
                refine_rounds=param_estimator_refinement_rounds,
                param_penalty_weight=param_penalty_weight,
                create_train_test_trial_split_fn=create_train_test_trial_split_fn,
                random_seed=random_seed,
                swear_words=swear_words,
                island_chat_manager=island_chat_manager,
                island_id=island_idx,
                batch_id=j,
                diagnostics_module=diagnostics_module,
                use_large_model=use_large_model_for_param_estimators,
            )
            for island_idx in range(n_islands)
            for j in range(batch_size)
        ]

        logging.info(
            f"Generating {n_islands * batch_size} parameter estimators "
            f"(LLM={little_lm_name}, mode={mode}, T={temperature:.2f})"
        )
        logging.info(f"Generating {n_islands * batch_size} new parameter estimators... Model: {little_lm_name}, mode: {mode}, temperature: {temperature:.2f}")
        param_est_results = await asyncio.gather(*param_estimation_tasks)
        # combine results
        island_results = [[model_results[island_idx * batch_size + j] + param_est_results[island_idx * batch_size + j] for j in range(batch_size)] for island_idx in range(n_islands)]

        # now loop through the results and compute losses
        success_rate = 0.0
        for island_idx, j in np.ndindex(n_islands, batch_size):
            logging.info(f"id={i},{island_idx},{j}")
            model_code_string, prompt, model_code_string_jax, model_new, param_est_code_string, param_est_new = island_results[island_idx][j]
            parent1_id, parent2_id = parent_ids[island_idx * batch_size + j]
            if model_new is None or param_est_new is None:
                logging.info(f"Skipping island {island_idx}, batch {j} due to LLM generation failure.")
                logging.info('-' * 50)
                continue

            model_name = prompt_manager.get_model_name()
            model_np = utils.str_to_func(model_code_string, model_name)
            if model_np is None:
                logging.info(
                    f"Skipping island {island_idx}, batch {j}: failed to parse NumPy model."
                )
                logging.info('-' * 50)
                continue
            try:
                _check_jax_translation(
                    model_np,
                    model_new,
                    param_est_new,
                    inputs_train,
                    outputs_train,
                )
            except Exception as e:
                logging.info(
                    f"Skipping island {island_idx}, batch {j}: JAX translation check failed: {e}"
                )
                logging.info('-' * 50)
                continue
            
            initial_loss, initial_params, loss, optimized_params = objective(model_new, param_est_new,
                                                                                x=inputs_train, y=outputs_train,
                                                                                create_train_test_trial_split_fn=create_train_test_trial_split_fn,
                                                                                loss_fn=loss_fn,
                                                                                param_penalty_weight=param_penalty_weight,
                                                                                fit_params=fit_params, tol=tol,
                                                                                use_param_estimator=use_param_estimator,
                                                                                max_iter=max_iter, trial_batch_size=trial_batch_size,
                                                                                random_seed=random_seed)
            if loss == FAILED_PROGRAM_COST:
                logging.info('-' * 50)
                continue

            y_eval = compute_evaluation_matrix(
                model_new,
                optimized_params,
                eval_points=evaluation_points_train,
            )
            logging.info(f"Prompt: \n{prompt}\n")
            logging.info(f"Loss: {loss:.2f}\n")
            logging.info(f"Model: \n{model_code_string}\n")
            logging.info(f"Model (JAX): \n{model_code_string_jax}\n")
            logging.info(f"Parameter Estimator: \n{param_est_code_string}\n")


            # plot the fits of the neuron model and parameter estimator if using image feedback
            if use_image_feedback and diagnostics_module is not None:
                prepare_and_plot_model_fits(
                    diagnostics_module=diagnostics_module,
                    programs_df=pd.DataFrame({'program': [model_new, model_new], 'params': [initial_params, optimized_params]}),
                    loss_fn=loss_fn,
                    inputs=inputs_train,
                    response=response_train,
                    sample_selection=np.random.choice(len(inputs_train), size=4, replace=False),
                    colours=['tab:green', 'tab:red'],
                    labels=['Param Estimator', 'Gradient Descent'],
                    line_alpha=1.0,
                    line_width=5.0,
                    point_alpha=0.2,
                    point_size=120,
                    legend_fontsize=20,
                    title=f"Updated Parameter Estimator and Gradient Descent Fit \n"
                    f"Initial Loss: {initial_loss:.2f}, Final Loss: {loss:.2f}",
                    save_path=os.path.join(image_feedback_dir, f'iter_{i}_island_{island_idx}_batch_{j}_updated_param_est.png')
                )
            
            param_names = [n for n in inspect.signature(model_new).parameters if n != "theta"]
            if optimized_params.shape[1] == len(param_names):
                df = pd.DataFrame(np.array(optimized_params)[:10], columns=param_names)
                logging.info(f"Optimized Parameters for 10 samples:\n{df}\n")
            t_added = time.time() - t_start
            new_program_df = pd.DataFrame({'program_code_string': model_code_string,
                                        'program': model_new,
                                        'parameter_estimator_code_string': param_est_code_string,
                                        'parameter_estimator': param_est_new,
                                        'iteration_number': i,
                                        'birth_island': island_idx,
                                        'batch_index': j,
                                        'train_loss': loss,
                                        'test_loss': None,  # will be filled later
                                        'llm_name': llm_name,
                                        'params': [optimized_params],
                                        'initial_loss': initial_loss,
                                        'initial_params': [initial_params],
                                        'parent1_id': [parent1_id],
                                        'parent2_id': [parent2_id],
                                        'evaluation_matrix': [y_eval]
                                        })
            
            islands[island_idx] = pd.concat([islands[island_idx], new_program_df], ignore_index=True)
            census.append([i, island_idx, j, llm_name, loss, t_added, parent1_id, parent2_id, y_eval, optimized_params.shape[1]])
            success_rate += 1 / (n_islands * batch_size)
            print(f"iteration {i}, island {island_idx}, batch {j}, loss: {loss:.2f}", flush=True)
            print('-' * 50, flush=True)
            logging.info("-" * 50)
        print("Success rate:", success_rate, flush=True)

        # sort each island by loss
        for island_idx in range(n_islands):
            islands[island_idx] = islands[island_idx].sort_values(by='train_loss').reset_index(drop=True)
        logging.info(f"Iteration {i} complete. The proportion of programs that successfully ran and received a loss is {success_rate:.2f}.")
        logging.info('-' * 50)
        # migrate and prune programs (better here for temperature to be in [0, 1] range)
        islands = genetic_helpers.perform_island_deduplication(islands, overlap_threshold=int(0.75 * critical_population_size))
        islands = genetic_helpers.perform_population_pruning(islands, critical_population_size=critical_population_size - n_migrants,
                                                min_wise_population_size=min_wise_population_size,)
        islands = genetic_helpers.perform_probabilistic_migration(islands, 
                                                                  n_migrants=n_migrants,
                                                                  destination_islands=exploration_topology if mode == 'explore' else exploitation_topology, 
                                                                  temperature=(temperature - 1.0)**4)

                                                             
        # save diagnostics
        iteration_dir = os.path.join(full_dir, 'iteration_updates', f'iteration_{i}')
        os.makedirs(iteration_dir, exist_ok=True)
        for island_idx in range(n_islands):
            pg_info = islands[island_idx][['iteration_number', 'birth_island', 'batch_index', 'train_loss']].to_string(index=False, header=False)
            print(f"Iter {i}, Island {island_idx} programs:\n{pg_info}\n")
            logging.info(f"Iter {i}, Island {island_idx} programs:\n{pg_info}\n")
        
            # Save plots of top programs
            if diagnostics_module is not None:
                top_df = islands[island_idx].sort_values(by='train_loss').head(3).reset_index(drop=True)
                top_df = top_df.sort_values(by='train_loss', ascending=False).reset_index(drop=True)
                sup_title = f"Iteration {i}, Island {island_idx}, Top {len(top_df)} Programs\n"
                sup_title += "\n".join([f"model {j+1}: iter {top_df['iteration_number'][j]}, birth island {top_df['birth_island'][j]}, batch {top_df['batch_index'][j]}, loss: {top_df['train_loss'][j]:.2f}" for j in range(len(top_df))])
                prepare_and_plot_model_fits(
                    diagnostics_module=diagnostics_module,
                    programs_df=top_df,
                    loss_fn=loss_fn,
                    inputs=inputs_train,
                    response=response_train,
                    sample_selection=np.random.choice(response_train.shape[0], size=9, replace=False),
                    title=sup_title,
                    save_path=os.path.join(iteration_dir, f'island_{island_idx}_top_programs.png'),
                    dpi=300.0,
                )
        
        if diagnostics_module is not None:
            all_programs = pd.concat([islands[idx] for idx in range(n_islands)], ignore_index=True)
            top_programs = all_programs.sort_values(by='train_loss').head(3).reset_index(drop=True)
            top_programs = top_programs.sort_values(by='train_loss', ascending=False).reset_index(drop=True)
            sup_title = f"Iteration {i}, Top 3 Programs Overall\n"
            sup_title += "\n".join([f"model {j+1}: iter {top_programs['iteration_number'][j]}, birth island {top_programs['birth_island'][j]}, batch {top_programs['batch_index'][j]}, loss: {top_programs['train_loss'][j]:.2f}" for j in range(len(top_programs))])
            prepare_and_plot_model_fits(
                diagnostics_module=diagnostics_module,
                programs_df=top_programs,
                loss_fn=loss_fn,
                inputs=inputs_train,
                response=response_train,
                sample_selection=np.random.choice(response_train.shape[0], size=9, replace=False),
                title=sup_title,
                save_path=os.path.join(iteration_dir, 'top_programs_overall.png'),
                dpi=300.0,
            )
        
        # save census
        census_path = os.path.join(iteration_dir, 'census.npy')
        census_np = np.array(census, dtype=object)
        np.save(census_path, census_np)
        
        # Log token usage summary for this iteration (if using chat mode)
        if island_chat_manager is not None:
            island_chat_manager.log_iteration_summary(i)
        
        # Log best loss across all islands for live monitoring
        if log_best_loss:
            all_programs = pd.concat([islands[idx] for idx in range(n_islands)], ignore_index=True)
            best_program = all_programs.loc[all_programs['train_loss'].idxmin()]
            best_loss_log.append({
                'iteration': i,
                'timestamp': pd.Timestamp.now().isoformat(),
                'best_train_loss': best_program['train_loss'],
                'best_island': best_program['birth_island'],
                'n_programs_total': len(all_programs),
                'elapsed_time': time.time() - t_start
            })
            # Write to CSV after each iteration for live monitoring
            pd.DataFrame(best_loss_log).to_csv(best_loss_path, index=False)

    # -----------------------------
    # now carry out the loss calculation on the test samples
    logging.info("Calculating loss on test set...")
    for island_idx in range(n_islands):
        logging.info(f"Island {island_idx} programs: {(len(islands[island_idx]))} programs to evaluate.")
        for j in range(len(islands[island_idx])):
            program = islands[island_idx].iloc[j]
            model = program['program']
            param_estimator = program['parameter_estimator']
            # compute the test loss
            _, _, test_loss, optimized_params = objective(model, param_estimator,
                                                          x=inputs_test, y=outputs_test,
                                                          create_train_test_trial_split_fn=create_train_test_trial_split_fn,
                                                          loss_fn=loss_fn,
                                                          fit_params=fit_params,
                                                          max_iter=max_iter,
                                                          param_penalty_weight=param_penalty_weight, tol=tol,
                                                          use_param_estimator=use_param_estimator,
                                                          trial_batch_size=trial_batch_size,
                                                          random_seed=random_seed,
                                                          )
            islands[island_idx].at[j, 'test_loss'] = test_loss
            islands[island_idx].at[j, 'params'] = optimized_params
            islands[island_idx].at[j, 'mean_loss'] = np.mean(test_loss)
            print(f"Test loss: {test_loss:.2f}")

    # group all islands together and save
    combined_dir = os.path.join(base_dir, date_stamp, time_stamp, 'combined')
    os.makedirs(combined_dir, exist_ok=True)
    combined_programs_dataframe = pd.concat(islands, ignore_index=True)
    combined_programs_dataframe = genetic_helpers.remove_duplicates(combined_programs_dataframe, mode='complicated', loss_tol=0.025, cosine_tol=0.99, loss_type='test_loss')
    # combined_programs_dataframe = combined_programs_dataframe.sort_values(by='test_loss').reset_index(drop=True)
    # sort by mean loss
    combined_programs_dataframe = combined_programs_dataframe.sort_values(by='mean_loss').reset_index(drop=True)
    # save the combined programs dataframe, reordering columns to have order:
    # iteration_number, birth_island, batch_index, train_loss, test_loss, program_code_string, parameter_estimator_code_string, program, parameter_estimator, params, parent1_id, parent2_id
    combined_programs_dataframe = combined_programs_dataframe[['iteration_number', 'birth_island', 'batch_index',
                                                                'train_loss', 'test_loss',
                                                                'program_code_string', 'parameter_estimator_code_string',
                                                                'program', 'parameter_estimator', 'params',
                                                                'parent1_id', 'parent2_id', 'llm_name']]
    combined_programs_dataframe.to_csv(os.path.join(combined_dir, 'programs_db.csv'), index=False)

    # save census npy array
    census_path = os.path.join(combined_dir, 'census.npy')
    census_np = np.array(census, dtype=object)
    np.save(census_path, census_np)

    # save island-specific results
    for island_id, island_df in enumerate(islands):
        island_dir = os.path.join(base_dir, date_stamp, time_stamp, f'island_{island_id}' if island_id < n_islands else 'meta_island')
        os.makedirs(island_dir, exist_ok=True)
        island_df.to_csv(os.path.join(island_dir, 'programs_db.csv'), index=False)

    # ---------------------------
    # save losses plot    
    if diagnostics_module is not None:
        plot_train_vs_test_loss_shared(
            programs_df=combined_programs_dataframe,
            island_labels=[f'Island {i}' for i in range(n_islands)] + ['garden_of_eden'],
            save_path=os.path.join(combined_dir, 'train_vs_test_loss.png'),
        )
    
    # ---------------------------
    df_list = [combined_programs_dataframe] + islands
    combined_dir = [os.path.join(base_dir, date_stamp, time_stamp, "combined")] 
    island_dirs = [os.path.join(base_dir, date_stamp, time_stamp, f'island_{i}') for i in range(n_islands)]
    df_dirs = combined_dir + island_dirs
    config_str = f"n_islands={n_islands}, batch_size={batch_size}, n_iterations={n_iterations},\n"
    config_str += f"llm_names={little_lm_name, large_lm_name}, fit_params={fit_params}, \n"
    config_str += f"critical_population_size={critical_population_size}.\n"

    if diagnostics_module is not None:
        for i, df in enumerate(df_list):
            df_sup = config_str
            df = df.head(3)
            df = df.sort_values(by='test_loss', ascending=False).reset_index(drop=True)
            df_sup += "".join([f"model {len(df) - i}: iter {df['iteration_number'][i]}, birth_island {df['birth_island'][i]}, batch {df['batch_index'][i]}, total loss {0.5 * (df['test_loss'][i] + df['train_loss'][i]):.2f}\n" for i in range(min(3, len(df)))])
            prepare_and_plot_model_fits(
                diagnostics_module=diagnostics_module,
                programs_df=df,
                loss_fn=loss_fn,
                inputs=inputs_test,
                response=response_test,
                sample_selection=np.random.choice(response_test.shape[0], size=9, replace=False),
                title=df_sup,
                save_path=os.path.join(df_dirs[i], 'top_model_fits.png'),
            )
            # Plot top models separately using the same plot_model_fits pathway.
            for j in range(min(3, len(df))):
                birth_island = df['birth_island'][j]
                iteration_number = df['iteration_number'][j]
                batch_index = df['batch_index'][j]
                model_df = df.iloc[[j]].copy().reset_index(drop=True)
                model_title = (
                    f"Island {birth_island}, Iteration {iteration_number}, "
                    f"Batch {batch_index}, loss: {df['test_loss'][j]:.2f}"
                )
                prepare_and_plot_model_fits(
                    diagnostics_module=diagnostics_module,
                    programs_df=model_df,
                    loss_fn=loss_fn,
                    inputs=inputs_test,
                    response=response_test,
                    sample_selection=np.random.choice(response_test.shape[0], size=9, replace=False),
                    labels=['model'],
                    colours=['tab:green'],
                    title=model_title,
                    save_path=os.path.join(df_dirs[i], f'top_model_fit_{min(3, len(df)) - j}.png')
                )
    
    # Log final token usage summary (if using chat mode)
    if island_chat_manager is not None:
        island_chat_manager.log_final_summary()
