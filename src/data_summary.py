import logging
import os

import numpy as np
import pandas as pd


def save_data_summary(
    response: np.ndarray,
    inputs: np.ndarray,
    training_samples: np.ndarray,
    test_samples: np.ndarray,
    x_train_trial_idx: np.ndarray,
    x_test_trial_idx: np.ndarray,
    output_dir: str,
    random_seed: int = 0,
    train_test_split_fn=None,
    y_train_trial_idx: np.ndarray | None = None,
    y_test_trial_idx: np.ndarray | None = None,
) -> pd.DataFrame:
    """
    Save a summary of the realized sample/trial splits and matrix sizes to CSV.

    This uses the exact sample/trial indices used by hypothesis_engine.
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

    if y_train_trial_idx is None:
        y_train_trial_idx = x_train_trial_idx
    if y_test_trial_idx is None:
        y_test_trial_idx = x_test_trial_idx

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
        f"fn={_describe_fn(train_test_split_fn)}; "
        f"random_seed={random_seed}; "
        f"disjoint={sample_stats['disjoint']}; cover_all={sample_stats['cover_all']}; "
        f"overlap={sample_stats['n_overlap']}; uncovered={sample_stats['n_uncovered']}; "
        f"train_has_duplicates={sample_stats['train_has_duplicates']}; "
        f"test_has_duplicates={sample_stats['test_has_duplicates']}; "
        f"train_first10={sample_stats['train_first10']}; "
        f"test_first10={sample_stats['test_first10']}"
    )

    trial_split_method = (
        f"fn={_describe_fn(train_test_split_fn)}; "
        f"random_seed={random_seed}; "
        f"n_trials_x={n_trials_x}; n_trials_y={n_trials_y}; "
        f"disjoint={trial_stats['disjoint']}; cover_all={trial_stats['cover_all']}; "
        f"overlap={trial_stats['n_overlap']}; uncovered={trial_stats['n_uncovered']}; "
        f"train_has_duplicates={trial_stats['train_has_duplicates']}; "
        f"test_has_duplicates={trial_stats['test_has_duplicates']}; "
        f"train_first10={trial_stats['train_first10']}; "
        f"test_first10={trial_stats['test_first10']}"
    )

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
