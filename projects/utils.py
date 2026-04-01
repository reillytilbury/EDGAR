import numpy as np

def zscore_data(data: dict, skip_keys: list | None = None, eps: float = 1e-12) -> dict:
    """
    Z-score each array in the data dict across the last dimension (trials).

    For each key, z-scores independently along the trial axis. Arrays are expected
    to have shape (..., n_trials).

    Args:
        data: Data dict where all arrays share the same last dimension.
        skip_keys: Keys to skip (e.g., categorical features).
        eps: Small constant to avoid division by zero.
    """
    skip = set(skip_keys or [])
    result = {}
    for key, arr in data.items():
        if key in skip:
            result[key] = arr
        else:
            arr = np.asarray(arr)
            mu = arr.mean(axis=-1, keepdims=True)
            sd = arr.std(axis=-1, keepdims=True)
            result[key] = (arr - mu) / (sd + eps)
    return result
