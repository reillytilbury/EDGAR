import logging
import os

import numpy as np
import pandas as pd

from . import utils


def save_data_summary(
    data,
    output_dir: str,
    random_seed: int = 0,
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

    data_blocks = [
        ("data_train_train", data_train_train),
        ("data_train_test", data_train_test),
        ("data_test_train", data_test_train),
        ("data_test_test", data_test_test),
    ]

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
