import numpy as np


def shuffle(*arrays: np.ndarray, axis: int = -1) -> list[np.ndarray]:
    """
    Shuffles multiple arrays along the specified axis.
    """
    if not arrays:
        return []

    n_trials = arrays[0].shape[axis]
    shuff_idx = np.random.permutation(n_trials)

    # Use take to shuffle along the specified axis
    return [
        np.take(arr, shuff_idx, axis=axis) if arr.ndim > 0 else arr for arr in arrays
    ]
