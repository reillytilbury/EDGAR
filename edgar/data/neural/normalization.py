import numpy as np


def by_vector_norm(response: np.ndarray, axis: int = 0) -> np.ndarray:
    """
    Normalizes neural responses to have a unit L2 norm along the specified axis.
    """
    return response / np.linalg.norm(response, axis=axis, keepdims=True)


def by_peak(response: np.ndarray, peak_values: np.ndarray) -> np.ndarray:
    """
    Normalizes neural responses by their peak values.
    """
    safe_peaks = np.where(peak_values == 0, 1e-10, peak_values)
    return response / safe_peaks[:, np.newaxis]
