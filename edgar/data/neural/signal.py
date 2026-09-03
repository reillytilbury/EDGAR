import numpy as np


def extract_stimulus_related(
    data: dict, n_pcs: int = 8, z_score: bool = False, spont_mean_removal: bool = False
) -> np.ndarray:
    """Extracts the stimulus-related response from neural data.

    This function preprocesses neural response data by optionally removing spontaneous
    activity, projecting out principal components, and z-scoring. It aims to isolate
    the signal component correlated with external stimuli.

    Args:
        data: A dictionary containing the neural response data. Expected keys are:
            - "sresp": (np.ndarray) Stimulus response data, typically with shape
              (n_neurons, n_trials).
            - "mean_spont": (np.ndarray, optional) Mean spontaneous activity
              per neuron, shape (n_neurons,). Required if `spont_mean_removal` is True.
            - "u_spont": (np.ndarray, optional) Principal components of spontaneous
              activity, shape (n_neurons, n_components). Required if `n_pcs > 0`.
        n_pcs: The number of principal components of spontaneous activity to project
            out from the stimulus response. If 0, no PCA projection is performed.
        z_score: If True, the stimulus response is z-scored along the second axis
            (per neuron) after other preprocessing steps.
        spont_mean_removal: If True, the mean spontaneous activity is subtracted from
            the stimulus response.

    Returns:
        A `np.ndarray` representing the preprocessed stimulus-related response,
        typically with the same shape as the input "sresp".
    """
    sresp = np.asarray(data["sresp"])

    if spont_mean_removal:
        mean_spont = np.asarray(data["mean_spont"])
        sresp = sresp - mean_spont[:, np.newaxis]

    if n_pcs > 0:
        u_spont = np.asarray(data["u_spont"])
        sresp = sresp - u_spont[:, :n_pcs] @ (u_spont[:, :n_pcs].T @ sresp)

    if z_score:
        sresp = (sresp - np.mean(sresp, axis=1, keepdims=True)) / np.std(
            sresp, axis=1, keepdims=True
        )

    return sresp


def _unbiased_fraction(R, min_repeats=2):
    """Computes the unbiased fraction of stimulus-related variance.

    This function implements the method described by Sahani & Linden (2003)
    to estimate the fraction of variance in neural responses that is attributable
    to the stimulus, unbiased by noise.

    The signal-to-noise ratio is estimated by:
    $$ S^2 = \\frac{1}{N} \\sum_{i=1}^{N} (\\mu_i - \\bar{f})^2 - \\frac{N-1}{N^2} \\sum_{i=1}^{N} \\frac{\\sigma_i^2}{R_s} $$
    $$ V^2 = \\frac{1}{N} \\sum_{i=1}^{N} \\frac{\\sigma_i^2}{R_s} $$
    where $\\mu_i$ is the mean response for stimulus $i$, $\\bar{f}$ is the
    overall mean response, $\\sigma_i^2$ is the variance of responses for stimulus $i$,
    $R_s$ is the number of repeats for stimulus $i$, and $N$ is the number of stimuli.
    The unbiased signal fraction is then $S^2 / (S^2 + V^2)$.

    Args:
        R: A `np.ndarray` of neural responses with shape
            (n_repeats, n_cells, n_angles), where `n_repeats` is the number of
            trials per stimulus angle, `n_cells` is the number of neurons, and
            `n_angles` is the number of stimulus angles.
        min_repeats: The minimum number of repeats required per angle. If `R` has
            fewer repeats than this, a ValueError is raised.

    Returns:
        A tuple containing:
        - `signal_fraction`: A `np.ndarray` of shape (n_cells,) representing the
          unbiased fraction of stimulus-related variance for each neuron, clipped
          to the range [0, 1].
        - `dict`: A dictionary containing intermediate computed values:
            - "S2": Signal variance (np.ndarray of shape (n_cells,)).
            - "V2": Noise variance (np.ndarray of shape (n_cells,)).
            - "mu_angles": Mean response per angle (np.ndarray of shape (n_cells, n_angles)).
            - "var_angles": Variance per angle (np.ndarray of shape (n_cells, n_angles)).

    Raises:
        ValueError: If the number of repeats per angle is less than `min_repeats`.
    """
    n_repeats, n_cells, n_angles = R.shape
    if n_repeats < min_repeats:
        raise ValueError(
            f"Need at least {min_repeats} repeats per angle, got {n_repeats}."
        )

    mu_angles = np.mean(R, axis=0)  # (n_cells, n_angles)
    var_angles = np.var(R, axis=0, ddof=1)  # (n_cells, n_angles)

    N = n_angles
    R_s = np.full(N, n_repeats, dtype=float)

    fbar_dot = np.mean(mu_angles, axis=1)
    term1 = np.mean((mu_angles - fbar_dot[:, None]) ** 2, axis=1)
    term2 = ((N - 1) / N**2) * np.sum(var_angles / R_s[None, :], axis=1)

    S2 = term1 - term2
    V2 = np.sum(var_angles / R_s[None, :], axis=1) / N

    signal_fraction = S2 / (S2 + V2)
    signal_fraction = np.clip(signal_fraction, 0, 1)

    return signal_fraction, {
        "S2": S2,
        "V2": V2,
        "mu_angles": mu_angles,
        "var_angles": var_angles,
    }


def binned_mean(
    x: np.ndarray,
    y: np.ndarray,
    x_grid: np.ndarray,
    return_indices: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Bins `y` values based on their corresponding `x` values and a specified `x_grid`.

    For each point in `x_grid`, this function defines a bin (interval) and calculates
    the mean of all `y` values whose corresponding `x` falls into that bin. This
    is useful for discretizing continuous data and computing average responses
    within defined ranges.

    Args:
        x: A `np.ndarray` representing the independent variable for binning.
            The last dimension is used for binning.
        y: A `np.ndarray` representing the dependent variable(s) to be averaged.
            The last dimension should correspond to the `x` values.
        x_grid: A `np.ndarray` of sorted values defining the centers of the bins.
            The bins are constructed such that each `x_grid` point is the center
            of an interval whose edges are halfway between adjacent `x_grid` points.
        return_indices: If True, in addition to the binned means, the function also
            returns a `np.ndarray` of the bin indices for each `x` value.

    Returns:
        A `np.ndarray` representing the mean of `y` for each bin defined by `x_grid`.
        The shape will be `y.shape[:-1] + (x_grid.size,)`.
        If `return_indices` is True, returns a tuple `(y_mean, bin_idx)`, where
        `bin_idx` is a `np.ndarray` of the same shape as `x`, containing the
        assigned bin index for each element in `x`.
    """
    if x_grid.size == 0:
        return (x_grid, np.array([])) if return_indices else x_grid
    if x_grid.size == 1:
        res = np.mean(y, axis=-1, keepdims=True)
        return (res, np.zeros(x.shape, dtype=int)) if return_indices else res

    edges = np.empty(x_grid.size + 1)
    edges[1:-1] = 0.5 * (x_grid[:-1] + x_grid[1:])
    edges[0] = x_grid[0] - 0.5 * (x_grid[1] - x_grid[0])
    edges[-1] = x_grid[-1] + 0.5 * (x_grid[-1] - x_grid[-2])

    bin_idx = np.digitize(x, edges) - 1
    bin_idx = np.clip(bin_idx, 0, x_grid.size - 1)

    y_shape = list(y.shape)
    y_shape[-1] = x_grid.size
    y_mean = np.zeros(y_shape)

    for i in range(x_grid.size):
        mask = bin_idx == i
        if np.any(mask):
            y_mean[..., i] = np.mean(y[..., mask], axis=-1)

    if return_indices:
        return y_mean, bin_idx
    return y_mean