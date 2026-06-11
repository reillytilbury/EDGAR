import numpy as np


def extract_stimulus_related(
    data: dict, n_pcs: int = 8, z_score: bool = False, spont_mean_removal: bool = False
) -> np.ndarray:
    """
    Extracts the stimulus-related response from the data.
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
    """
    Compute unbiased fraction of stimulus-related variance (Sahani & Linden, 2003)
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
    """
    Bin y by proximity to each x_grid point and return per-bin means.
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
