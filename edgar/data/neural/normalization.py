import numpy as np


def by_vector_norm(response: np.ndarray, axis: int = 0) -> np.ndarray:
    """Normalizes neural responses to have a unit L2 norm along the specified axis.

    This function divides each vector (along the specified `axis`) in the `response`
    array by its L2 norm, effectively scaling it to have a magnitude of 1.

    Args:
        response: The input neural response data, typically a multi-dimensional array.
        axis: The axis along which to compute the L2 norm and normalize.
            Defaults to 0.

    Returns:
        A new `np.ndarray` with neural responses normalized to a unit L2 norm
        along the specified axis.
    """
    return response / np.linalg.norm(response, axis=axis, keepdims=True)


def by_peak(response: np.ndarray, peak_values: np.ndarray) -> np.ndarray:
    """Normalizes neural responses by their peak values.

    This function divides each response in the `response` array by its corresponding
    `peak_value`. To prevent division by zero, `peak_values` that are zero are
    replaced with a small epsilon (1e-10) before division.

    Args:
        response: The input neural response data, typically a multi-dimensional array.
        peak_values: A 1D array of peak values, where each element corresponds to
            the peak of a response in `response`.

    Returns:
        A new `np.ndarray` with neural responses normalized by their peak values.
    """
    safe_peaks = np.where(peak_values == 0, 1e-10, peak_values)
    return response / safe_peaks[:, np.newaxis]