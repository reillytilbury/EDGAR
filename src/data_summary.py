import logging
import os

import numpy as np
import pandas as pd

from . import utils


def save_data_summary(
    data,
    output_dir: str,
    random_seed: int = 0,
    training_samples: np.ndarray | None = None,
) -> pd.DataFrame:
    """Save a summary of realized sample/trial splits and per-key shapes to CSV.

    Args:
        data: Pre-split 2x2 container
            ``[[train_train, train_test], [test_train, test_test]]``.
        output_dir (str): Directory to write ``data_summary.csv``.
        random_seed (int): Seed used for splitting.

    Returns:
        pd.DataFrame: Summary dataframe.
    """
    data_arr = np.asarray(data, dtype=object)
    if data_arr.shape != (2, 2):
        raise ValueError("Pre-split data must have shape (2, 2).")

    split_mode = "pre_split"
    data_train_train = data_arr[0, 0]
    data_train_test = data_arr[0, 1]
    data_test_train = data_arr[1, 0]
    data_test_test = data_arr[1, 1]
    for split_data in (data_train_train, data_train_test, data_test_train, data_test_test):
        utils.validate_data(split_data)

    n_train_samples = utils.data_n_samples(data_train_train)
    n_test_samples = utils.data_n_samples(data_test_test)
    n_train_trials = utils.data_n_trials(data_train_train)
    n_test_trials = utils.data_n_trials(data_test_test)
    n_samples = n_train_samples + n_test_samples
    n_trials = n_train_trials + n_test_trials

    training_samples_np = np.arange(n_train_samples, dtype=np.int64)
    test_samples_np = np.arange(n_train_samples, n_train_samples + n_test_samples, dtype=np.int64)
    train_trials_np = np.arange(n_train_trials, dtype=np.int64)
    test_trials_np = np.arange(n_train_trials, n_train_trials + n_test_trials, dtype=np.int64)
    data_blocks = [
        ("data_train_train", data_train_train),
        ("data_train_test", data_train_test),
        ("data_test_train", data_test_train),
        ("data_test_test", data_test_test),
    ]

    n_train_samples = int(training_samples_np.size)
    n_test_samples = int(test_samples_np.size)
    n_train_trials = int(train_trials_np.size)
    n_test_trials = int(test_trials_np.size)

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

    sample_stats = _split_stats(training_samples_np, test_samples_np, n_samples)
    trial_stats = _split_stats(train_trials_np, test_trials_np, n_trials)

    sample_split_method = (
        f"disjoint={sample_stats['disjoint']}; cover_all={sample_stats['cover_all']}; "
        f"overlap={sample_stats['n_overlap']}; uncovered={sample_stats['n_uncovered']}; "
        f"train_has_duplicates={sample_stats['train_has_duplicates']}; "
        f"test_has_duplicates={sample_stats['test_has_duplicates']}; "
        f"train_first10={sample_stats['train_first10']}; "
        f"test_first10={sample_stats['test_first10']}"
    )
    if split_mode == "pre_split":
        sample_split_method += "; note=indices shown are synthetic summaries derived from split sizes"

    trial_split_method = (
        f"disjoint={trial_stats['disjoint']}; cover_all={trial_stats['cover_all']}; "
        f"overlap={trial_stats['n_overlap']}; uncovered={trial_stats['n_uncovered']}; "
        f"train_has_duplicates={trial_stats['train_has_duplicates']}; "
        f"test_has_duplicates={trial_stats['test_has_duplicates']}; "
        f"train_first10={trial_stats['train_first10']}; "
        f"test_first10={trial_stats['test_first10']}"
    )

    def calc_size(shape, dtype):
        n_elements = int(np.prod(shape))
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

    rows = []

    # === SAMPLE SPLIT SUMMARY ===
    rows.append({
        'category': 'SAMPLE_SPLIT',
        'matrix_name': 'total_samples',
        'description': 'Total number of samples in dataset',
        'shape': f"({n_samples},)",
        'dtype': '-', 'size_bytes': '-', 'size_human': '-',
        'n_elements': n_samples,
    })
    rows.append({
        'category': 'SAMPLE_SPLIT',
        'matrix_name': 'training_samples',
        'description': 'Samples used for training (held-in)',
        'shape': f"({n_train_samples},)",
        'dtype': str(training_samples_np.dtype),
        'size_bytes': calc_size((n_train_samples,), training_samples_np.dtype),
        'size_human': format_size(calc_size((n_train_samples,), training_samples_np.dtype)),
        'n_elements': n_train_samples,
    })
    rows.append({
        'category': 'SAMPLE_SPLIT',
        'matrix_name': 'test_samples',
        'description': 'Samples used for testing (held-out)',
        'shape': f"({n_test_samples},)",
        'dtype': str(test_samples_np.dtype),
        'size_bytes': calc_size((n_test_samples,), test_samples_np.dtype),
        'size_human': format_size(calc_size((n_test_samples,), test_samples_np.dtype)),
        'n_elements': n_test_samples,
    })
    rows.append({
        'category': 'SAMPLE_SPLIT',
        'matrix_name': 'sample_split_method',
        'description': sample_split_method,
        'shape': '-', 'dtype': '-', 'size_bytes': '-', 'size_human': '-',
        'n_elements': '-',
    })

    # === TRIAL SPLIT SUMMARY ===
    rows.append({
        'category': 'TRIAL_SPLIT',
        'matrix_name': 'total_trials',
        'description': 'Total number of trials per sample (shared last dim)',
        'shape': f"({n_trials},)",
        'dtype': '-', 'size_bytes': '-', 'size_human': '-',
        'n_elements': n_trials,
    })
    rows.append({
        'category': 'TRIAL_SPLIT',
        'matrix_name': 'training_trials',
        'description': 'Trials used for param fitting in objective()',
        'shape': f"({n_train_trials},)",
        'dtype': str(train_trials_np.dtype),
        'size_bytes': calc_size((n_train_trials,), train_trials_np.dtype),
        'size_human': format_size(calc_size((n_train_trials,), train_trials_np.dtype)),
        'n_elements': n_train_trials,
    })
    rows.append({
        'category': 'TRIAL_SPLIT',
        'matrix_name': 'test_trials',
        'description': 'Trials used for loss evaluation in objective()',
        'shape': f"({n_test_trials},)",
        'dtype': str(test_trials_np.dtype),
        'size_bytes': calc_size((n_test_trials,), test_trials_np.dtype),
        'size_human': format_size(calc_size((n_test_trials,), test_trials_np.dtype)),
        'n_elements': n_test_trials,
    })
    rows.append({
        'category': 'TRIAL_SPLIT',
        'matrix_name': 'trial_split_method',
        'description': trial_split_method,
        'shape': '-', 'dtype': '-', 'size_bytes': '-', 'size_human': '-',
        'n_elements': '-',
    })

    # === PER-KEY DATA SUMMARY ===
    total_size = 0
    for block_name, block_data in data_blocks:
        for key, arr in block_data.items():
            arr_np = np.asarray(arr)
            key_size = calc_size(arr_np.shape, arr_np.dtype)
            total_size += key_size
            rows.append({
                'category': 'DATA_KEY',
                'matrix_name': f"{block_name}['{key}']",
                'description': f"Key '{key}' — {block_name}",
                'shape': str(arr_np.shape),
                'dtype': str(arr_np.dtype),
                'size_bytes': key_size,
                'size_human': format_size(key_size),
                'n_elements': int(np.prod(arr_np.shape)),
            })

    # Create DataFrame and save
    df = pd.DataFrame(rows)
    csv_path = os.path.join(output_dir, 'data_summary.csv')
    df.to_csv(csv_path, index=False)

    # Print summary
    print("\n" + "=" * 70)
    print("DATA SUMMARY")
    print("=" * 70)
    print(
        f"Sample Split: {n_train_samples}/{n_samples} train, "
        f"{n_test_samples}/{n_samples} test "
        f"(disjoint={sample_stats['disjoint']}, cover_all={sample_stats['cover_all']})"
    )
    print(
        f"Trial Split:  {n_train_trials}/{n_trials} train, "
        f"{n_test_trials}/{n_trials} test (per sample, in objective; "
        f"seed={random_seed}, disjoint={trial_stats['disjoint']}, "
        f"cover_all={trial_stats['cover_all']})"
    )
    print(f"Data blocks:  {[name for name, _ in data_blocks]}")
    for block_name, block_data in data_blocks:
        print(f"  {block_name}:")
        for key, arr in block_data.items():
            print(f"    '{key}': shape={np.asarray(arr).shape}, dtype={np.asarray(arr).dtype}")
    print(f"Total Data:   {format_size(total_size)}")
    print(f"Saved to:     {csv_path}")
    print("=" * 70 + "\n")

    logging.info(f"Data summary saved to {csv_path}")

    return df
