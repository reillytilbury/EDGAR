"""
Welcome to the Model Discovery Engine! Fill in the components below to start building your model.

NECESSARY COMPONENTS:

Loading:
- load_and_process_data(data_path, *preprocess_params) -> [X, Y]
- train_test_split(X) -> [train_samples, train_trials]

Seed Programs:
- model_v1(X, *params) and param_est_v1(X, Y)
- model_v2(X, *params) and param_est_v2(X, Y)

OPTIONAL COMPONENTS:
- plot_model_fits(X, Y, model_list, params_list)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from typing import Dict, Any, Optional, List, Tuple

from src.data_structures import Inputs, Outputs


# ========================
# 1. DATA
# ========================

def load_and_process_data(
    data_path: str,
    # ---- ALL SUBSEQUENT PARAMS MUST BE SPECIFIED IN THE CONFIG FILE ----
    time_start: float = 27826,
    time_end: float = 31223,
    time_bin_ms: int = 10,
    min_spikes: int = 100,
    speed_threshold: float = 2.5,
    max_trials: int = 5000,
) -> Tuple[Inputs, Outputs]:
    """
    Load and preprocess grid-cell data, returning canonical Inputs/Outputs.

    Alignment strategy (moved from long comments in previous parser version):
    1. Behavioral variables (x, y, optionally others) are sampled over time and indexed by a common time vector.
    2. Spike trains are event-time lists per neuron; they are not inherently aligned to behavior samples.
    3. Define a fixed time discretization (e.g., 10 ms bins) over the analysis interval.
    4. Bin each neuron's spike times into those bins to obtain firing-rate time series.
    5. Bin/interpolate behavior into the same time bins so neural and behavioral signals share one time axis.
    6. Remove low-speed periods to exclude immobility-related bins.
    7. Keep neurons with sufficient spike counts and build canonical tensors.

    Parameters
    ----------
    data_path : str
        Path to a `.npz` file with keys `t`, `x`, `y`, and spike module key (e.g. `spikes_mod1`).
    time_start : float
        Analysis window start time in seconds.
    time_end : float
        Analysis window end time in seconds.
    time_bin_ms : int
        Bin size in milliseconds for spike-rate conversion.
    min_spikes : int
        Minimum spikes in the time window for a neuron to be retained.
    speed_threshold : float
        Minimum speed (cm/s) for bins to be retained.
    max_trials : int
        Maximum bins retained after preprocessing (uniform subsampling if exceeded).

    Returns
    -------
    X : Inputs
        Input tensor of shape `(n_samples, 2, n_trials)` for `[x, y]`.
    Y : Outputs
        Output tensor of shape `(n_samples, 1, n_trials)` for firing rate.
    """
    spatial_bin_cm = 3.0
    smoothing_sigma = 1.5
    wall_val = 0.75
    module_key = "spikes_mod1"
    input_names = ["x", "y"]

    data = np.load(data_path, allow_pickle=True)
    t_raw = np.asarray(data["t"], dtype=float)

    features_raw: Dict[str, np.ndarray] = {}
    for feat_name in input_names:
        if feat_name not in data.files:
            raise KeyError(f"Feature '{feat_name}' not found in file. Available: {list(data.files)}")
        features_raw[feat_name] = np.asarray(data[feat_name], dtype=float)

    spike_times_dict = data[module_key].item()
    n_neurons_raw = len(spike_times_dict)

    time_bin_s = time_bin_ms / 1000.0
    n_time_bins = int(np.ceil((time_end - time_start) / time_bin_s))
    bin_edges = np.linspace(time_start, time_end, n_time_bins + 1)

    firing_rates = np.zeros((n_neurons_raw, n_time_bins), dtype=float)
    total_spikes_per_neuron = np.zeros(n_neurons_raw, dtype=float)

    for neuron_idx, (_, spike_times) in enumerate(spike_times_dict.items()):
        spikes = np.asarray(spike_times, dtype=float)
        spikes_in_window = spikes[(spikes >= time_start) & (spikes < time_end)]
        total_spikes_per_neuron[neuron_idx] = len(spikes_in_window)
        spike_counts, _ = np.histogram(spikes_in_window, bins=bin_edges)
        firing_rates[neuron_idx] = spike_counts / time_bin_s

    bin_indices = np.digitize(t_raw, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, n_time_bins - 1)
    counts_per_bin = np.bincount(bin_indices, minlength=n_time_bins)

    features = {}
    for feat_name, feat_raw in features_raw.items():
        sums_per_bin = np.bincount(bin_indices, weights=feat_raw, minlength=n_time_bins)
        with np.errstate(invalid="ignore"):
            features[feat_name] = np.where(counts_per_bin > 0, sums_per_bin / counts_per_bin, np.nan)

    # Speed filter
    arena_half_width_cm = wall_val * 100
    dx = np.diff(features["x"], prepend=features["x"][0]) * arena_half_width_cm
    dy = np.diff(features["y"], prepend=features["y"][0]) * arena_half_width_cm
    speed = np.sqrt(dx**2 + dy**2) / time_bin_s
    speed[0] = speed[1] if len(speed) > 1 else 0
    keep_speed = speed >= speed_threshold
    firing_rates = firing_rates[:, keep_speed]
    features = {name: arr[keep_speed] for name, arr in features.items()}

    # Spike-count filter
    good_neurons = total_spikes_per_neuron >= min_spikes
    firing_rates = firing_rates[good_neurons]
    n_cells = firing_rates.shape[0]

    # Normalize positions to approximately [-1, 1]
    features["x"] = features["x"] / wall_val
    features["y"] = features["y"] / wall_val

    if max_trials is not None and firing_rates.shape[1] > max_trials:
        keep_idx = np.linspace(0, firing_rates.shape[1] - 1, max_trials).astype(int)
        firing_rates = firing_rates[:, keep_idx]
        features = {name: arr[keep_idx] for name, arr in features.items()}

    # Compute and lightly smooth rate maps (kept for consistency with the original workflow)
    n_spatial_bins = int(np.ceil((2 * wall_val * 100) / spatial_bin_cm))
    _ = _compute_rate_maps(features["x"], features["y"], firing_rates, n_spatial_bins, smoothing_sigma)

    inputs_data = np.stack([
        np.tile(features["x"], (n_cells, 1)),
        np.tile(features["y"], (n_cells, 1)),
    ], axis=1)

    X = Inputs(data=inputs_data, names=["x", "y"])
    Y = Outputs.from_array(firing_rates, names=["firing_rate"])
    return X, Y


def train_test_split(
    X: Inputs,
    # -- ALL SUBSEQUENT PARAMS MUST BE SPECIFIED IN THE CONFIG FILE ---
    random_seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sample and trial train splits (50/50 each).

    Parameters
    ----------
    X : Inputs
        Inputs object of shape `(n_samples, n_input_features, n_trials)`.
    random_seed : int
        RNG seed.

    Returns
    -------
    train_samples : np.ndarray
        Sample indices for training.
    train_trials : np.ndarray
        Trial indices for training.
    """
    n_samples, _, n_trials = X.shape
    assert n_samples >= 2, "Need at least 2 samples for model optimization/eval"
    assert n_trials >= 2, "Need at least 2 trials for parameter optimization/eval"

    rng = np.random.default_rng(random_seed)
    train_samples = rng.choice(np.arange(n_samples), n_samples // 2, replace=False)
    train_trials = rng.choice(np.arange(n_trials), n_trials // 2, replace=False)
    return train_samples, train_trials


# ========================
# 2. SEED MODELS
# ========================

def model_v1(X, lam=0.5, theta=0.0, phi_x=0.0, phi_y=0.0, baseline=0.0, amplitude=1.0):
    """
    Independent variable:
    X = [x, y]  # position (normalized to [-1, 1])

    Grid model as sum of 3 cosines 60 degrees apart.

    Args:
        X (np.ndarray): Shape (2, n_trials), with `x = X[0]`, `y = X[1]`.
        lam (float): Grid spacing parameter.
        theta (float): Grid orientation in radians (periodic modulo pi/3).
        phi_x (float): Phase offset along x.
        phi_y (float): Phase offset along y.
        baseline (float): Baseline firing rate.
        amplitude (float): Modulation amplitude.

    Returns:
        np.ndarray: Predicted firing rates, shape (n_trials,).
    """
    x = X[0]
    y = X[1]

    lam = np.clip(lam, 0.1, 2.0)
    theta = np.clip(theta, 0, np.pi / 3)
    baseline = np.clip(baseline, 0, None)
    amplitude = np.clip(amplitude, 0, None)

    q = 4.0 * np.pi / (np.sqrt(3.0) * lam)
    angles = theta + 2.0 * np.pi * np.arange(3) / 3.0
    ux = np.cos(angles)
    uy = np.sin(angles)

    dx = x - phi_x
    dy = y - phi_y
    proj = np.outer(ux, dx) + np.outer(uy, dy)
    s = np.sum(np.cos(q * proj), axis=0)

    return baseline + amplitude * s


def param_est_v1(X, Y):
    """
    Estimate parameters for `model_v1` using FFT/ring heuristics.

    Args:
        X (np.ndarray): Input array with shape (2, n_trials).
        Y (np.ndarray): Observed firing rates, shape (n_trials,).

    Returns:
        np.ndarray: Estimated [lam, theta, phi_x, phi_y, baseline, amplitude].
    """
    x = X[0]
    y = X[1]
    rm, occ = _make_ratemap(x, y, Y, nbins=60, sigma=1.5)
    F, py, px, rad, ang, cy, cx = _fft_peak_candidates(rm, occ)

    if len(py) < 6:
        baseline = max(0.0, np.percentile(rm, 5))
        return np.array([0.5, 0.0, 0.0, 0.0, baseline, 1.0])

    baseline = np.percentile(Y, 10)
    amplitude = max(0.0, np.percentile(Y, 95) - baseline)

    r0 = np.median(rad[:10])
    cand = np.where(np.abs(rad - r0) < 0.15 * max(r0, 1e-6))[0]
    if len(cand) == 0:
        cand = np.arange(min(len(rad), 10))

    theta = float(ang[cand[0]] % (np.pi / 3))

    arena = 2.0
    fx = (px[cand[0]] - cx) / arena
    fy = (py[cand[0]] - cy) / arena
    k_mag = 2 * np.pi * np.sqrt(fx * fx + fy * fy)
    q = k_mag if k_mag > 1e-6 else 4 * np.pi / (np.sqrt(3.0) * 0.5)
    lam = float(np.clip(4.0 * np.pi / (np.sqrt(3.0) * q), 0.1, 2.0))

    phase = np.angle(F[py[cand[0]], px[cand[0]]])
    phi_x = float(np.clip(-phase / (q + 1e-8), -1.0, 1.0))
    phi_y = 0.0

    return np.array([lam, theta, phi_x, phi_y, baseline, amplitude])


def model_v2(X, lam=0.5, theta=0.0, phi_x=0.0, phi_y=0.0, baseline=0.0, amplitude=1.0, sigma=0.08):
    """
    Independent variable:
    X = [x, y]  # position (normalized to [-1, 1])

    Grid model as Gaussian bumps on a rotated hexagonal lattice.

    Args:
        X (np.ndarray): Shape (2, n_trials), with `x = X[0]`, `y = X[1]`.
        lam (float): Lattice spacing.
        theta (float): Lattice orientation (radians, modulo pi/3).
        phi_x (float): Phase offset in x.
        phi_y (float): Phase offset in y.
        baseline (float): Baseline firing rate.
        amplitude (float): Peak amplitude per lattice point.
        sigma (float): Width of Gaussian bumps.

    Returns:
        np.ndarray: Predicted firing rates, shape (n_trials,).
    """
    x = X[0]
    y = X[1]

    lam = np.clip(lam, 0.1, 2.0)
    theta = np.clip(theta, 0, np.pi / 3)
    sigma = np.clip(sigma, 0.01, 0.5)
    baseline = np.clip(baseline, 0, None)
    amplitude = np.clip(amplitude, 0, None)

    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]])
    v1 = R @ np.array([lam, 0.0])
    v2 = R @ np.array([0.5 * lam, 0.5 * np.sqrt(3.0) * lam])

    dx = x - phi_x
    dy = y - phi_y

    extent = 2.0
    step = min(np.linalg.norm(v1), np.linalg.norm(v2))
    n_range = int(np.ceil((extent + 2.0 * lam) / max(step, 1e-6))) + 2

    inv2sig2 = 1.0 / (2.0 * sigma * sigma)
    r = np.full_like(x, baseline, dtype=float)

    for n in range(-n_range, n_range + 1):
        for m in range(-n_range, n_range + 1):
            cx, cy = n * v1 + m * v2
            ddx = dx - cx
            ddy = dy - cy
            r += amplitude * np.exp(-(ddx * ddx + ddy * ddy) * inv2sig2)

    return r


def param_est_v2(X, Y):
    """
    Estimate parameters for `model_v2` from a smoothed rate-map template.

    Args:
        X (np.ndarray): Input array with shape (2, n_trials).
        Y (np.ndarray): Observed firing rates, shape (n_trials,).

    Returns:
        np.ndarray: Estimated [lam, theta, phi_x, phi_y, baseline, amplitude, sigma].
    """
    p1 = param_est_v1(X, Y)
    lam, theta, phi_x, phi_y, baseline, amplitude = p1
    sigma = 0.08
    return np.array([lam, theta, phi_x, phi_y, baseline, amplitude, sigma])


# ========================
# 3. DIAGNOSTICS
# ========================

def plot_model_fits(
    X,
    Y,
    programs_list,
    n_bins=50,
    domain=(-1.0, 1.0),
    save_path="",
    # -- ALL SUBSEQUENT PARAMS MUST BE SPECIFIED IN THE CONFIG FILE ---
):
    """
    Plot data and model-predicted 2D rate maps for up to 9 random samples.

    Parameters
    ----------
    X : array-like or Inputs
        Input tensor with shape `(n_samples, n_features, n_trials)`.
    Y : array-like or Outputs
        Output tensor with shape `(n_samples, 1, n_trials)`.
    programs_list : list[dict]
        List of dictionaries with keys:
        - `'model'`: callable model function
        - `'params'`: `(n_samples, n_params)` parameter matrix
        - optionally `'losses'`: per-sample losses
    n_bins : int
        Number of bins per axis for each rate map.
    domain : tuple[float, float]
        Plotting domain for both x and y axes.
    save_path : str
        Output figure path.
    """
    if save_path == "":
        raise ValueError("Please provide a save_path for the plot")

    x_arr = _to_array3d(X)
    y_arr = _to_array3d(Y)

    if x_arr.shape[1] < 2:
        raise ValueError("Grid-cell diagnostics require at least 2 input features (x,y)")

    n_samples = x_arr.shape[0]
    n_show = min(9, n_samples)
    show_idx = np.random.default_rng(0).choice(n_samples, size=n_show, replace=False)

    n_models = len(programs_list)
    fig, axes = plt.subplots(n_show, 1 + n_models, figsize=(4 * (1 + n_models), 3 * n_show))
    axes = np.atleast_2d(axes)

    for row, s in enumerate(show_idx):
        x = x_arr[s, 0]
        y = x_arr[s, 1]
        y_obs = y_arr[s, 0]

        rm_obs = _bin_to_rate_map(x, y, y_obs, n_bins=n_bins, domain=domain)
        ax = axes[row, 0]
        im = ax.imshow(
            rm_obs.T,
            origin="lower",
            extent=[domain[0], domain[1], domain[0], domain[1]],
            cmap="viridis",
        )
        ax.set_title(f"Sample {s} data")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        for m_idx, program in enumerate(programs_list):
            model = program["model"]
            params = program["params"][s]
            y_pred = model(x_arr[s], *params)
            rm_pred = _bin_to_rate_map(x, y, y_pred, n_bins=n_bins, domain=domain)

            axm = axes[row, m_idx + 1]
            imm = axm.imshow(
                rm_pred.T,
                origin="lower",
                extent=[domain[0], domain[1], domain[0], domain[1]],
                cmap="viridis",
            )
            label = f"Model {m_idx + 1}"
            if "losses" in program:
                label += f" (loss={program['losses'][s]:.2f})"
            axm.set_title(label)
            fig.colorbar(imm, ax=axm, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ========================
# 4. OPTIONAL PROJECT-SPECIFIC HELPERS
# ========================

def _to_array3d(obj) -> np.ndarray:
    if hasattr(obj, "to_tensor"):
        return np.asarray(obj.to_tensor())
    arr = np.asarray(obj)
    if arr.ndim == 2:
        return arr[:, np.newaxis, :]
    return arr


def _compute_rate_maps(
    x: np.ndarray,
    y: np.ndarray,
    firing_rates: np.ndarray,
    n_bins: int,
    smoothing_sigma: float,
) -> np.ndarray:
    occupancy, _, _ = np.histogram2d(x, y, bins=n_bins, range=[[-1, 1], [-1, 1]])
    occupancy_s = gaussian_filter(occupancy, sigma=smoothing_sigma)
    bin_x = np.clip(((x + 1.0) / 2.0 * n_bins).astype(int), 0, n_bins - 1)
    bin_y = np.clip(((y + 1.0) / 2.0 * n_bins).astype(int), 0, n_bins - 1)

    n_cells = firing_rates.shape[0]
    n_trials = firing_rates.shape[1]
    rate_maps = np.zeros((n_cells, n_bins, n_bins), dtype=float)
    for c in range(n_cells):
        spike_map = np.zeros((n_bins, n_bins), dtype=float)
        for t in range(n_trials):
            spike_map[bin_x[t], bin_y[t]] += firing_rates[c, t]
        spike_map_s = gaussian_filter(spike_map, sigma=smoothing_sigma)
        rate_maps[c] = spike_map_s / (occupancy_s + 1e-6)
    return rate_maps


def _make_ratemap(x: np.ndarray, y: np.ndarray, values: np.ndarray, nbins=60, sigma=1.5):
    heat, _, _ = np.histogram2d(x, y, bins=nbins, range=[[-1, 1], [-1, 1]], weights=values)
    occ, _, _ = np.histogram2d(x, y, bins=nbins, range=[[-1, 1], [-1, 1]])
    rm = np.divide(heat, occ, out=np.zeros_like(heat), where=occ > 0).T
    return gaussian_filter(rm, sigma=sigma), occ.T


def _fft_peak_candidates(rm: np.ndarray, occ: Optional[np.ndarray] = None):
    H, W = rm.shape
    if occ is not None and np.any(occ > 0):
        thr = np.percentile(occ[occ > 0], 20)
        mask = (occ >= thr).astype(float)
    else:
        mask = np.ones_like(rm)

    wy = np.hanning(H)
    wx = np.hanning(W)
    window = wy[:, None] * wx[None, :]
    z = (rm - np.nanmean(rm[mask > 0])) * mask * window

    F = np.fft.fftshift(np.fft.fft2(z))
    M = np.abs(F)

    cy, cx = H // 2, W // 2
    yy, xx = np.indices((H, W))
    rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)

    rmin = 0.05 * np.max(rr)
    rmax = 0.45 * np.max(rr)
    band = (rr >= rmin) & (rr <= rmax)
    M_band = np.where(band, M, 0.0)

    flat_idx = np.argpartition(M_band.ravel(), -50)[-50:]
    py, px = np.unravel_index(flat_idx, (H, W))
    order = np.argsort(M_band[py, px])[::-1]
    py, px = py[order], px[order]

    dy = py - cy
    dx = px - cx
    ang = np.arctan2(dy, dx)
    rad = np.sqrt(dx * dx + dy * dy)
    return F, py, px, rad, ang, cy, cx


def _bin_to_rate_map(
    x: np.ndarray,
    y: np.ndarray,
    values: np.ndarray,
    n_bins: int = 50,
    domain: Tuple[float, float] = (-1.0, 1.0),
) -> np.ndarray:
    edges = np.linspace(domain[0], domain[1], n_bins + 1)
    occ, _, _ = np.histogram2d(x, y, bins=[edges, edges])
    weighted, _, _ = np.histogram2d(x, y, bins=[edges, edges], weights=values)
    return weighted / (occ + 1e-8)
