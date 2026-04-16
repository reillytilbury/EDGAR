"""
Welcome to the Model Discovery Engine! Fill in the components below to start building your model.

NECESSARY COMPONENTS:

Loading:
- load_and_process_data(data_path, *preprocess_params) -> dict[str, np.ndarray]
- train_test_split(X) -> [train_samples, train_trials]

Seed Programs:
- model_v1(data, params) and param_est_v1(data)
- model_v2(data, params) and param_est_v2(data)

Loss:
- loss_fn(model_output, data) -> loss values

OPTIONAL COMPONENTS:
- plot_model_fits(data, programs_list, eval_grid, save_path, labels)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from typing import Dict, Any, Optional, List, Tuple

from src import utils


# ========================
# 1. DATA
# ========================

def load_and_process_data(
    data_path: str,
    # ---- ALL SUBSEQUENT PARAMS MUST BE SPECIFIED IN THE CONFIG FILE ----
    random_seed: int = 42,
    time_start: float = 27826,
    time_end: float = 31223,
    time_bin_ms: int = 100,
    min_spikes: int = 200,
    speed_threshold: float = 2.5,
    max_trials: int = 5000,
    min_active_frac: float = 0.02,
    min_modulation: float = 1.0,
    min_spatial_reliability: float = 0.2,
    normalize_per_sample: bool = True,
    target_l2_norm: float = 1.0,
    min_l2_norm: float = 1e-6,
) -> list[list[dict[str, np.ndarray]]]:
    """
    Load and preprocess grid-cell data, returning a dict of arrays.

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
    2 x 2 list of dicts
        ``[[data_train_train, data_train_test], [data_test_train, data_test_test]]``.
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

    # Spike-count filter (raw window) + activity/modulation filters (post-speed)
    active_frac = np.mean(firing_rates > 0, axis=1)
    modulation = np.percentile(firing_rates, 95, axis=1) - np.percentile(firing_rates, 5, axis=1)
    good_neurons = (
        (total_spikes_per_neuron >= min_spikes)
        & (active_frac >= min_active_frac)
        & (modulation >= min_modulation)
    )
    firing_rates = firing_rates[good_neurons]
    n_cells = firing_rates.shape[0]

    # Normalize positions to approximately [-1, 1]
    features["x"] = features["x"] / wall_val
    features["y"] = features["y"] / wall_val

    if max_trials is not None and firing_rates.shape[1] > max_trials:
        keep_idx = np.linspace(0, firing_rates.shape[1] - 1, max_trials).astype(int)
        firing_rates = firing_rates[:, keep_idx]
        features = {name: arr[keep_idx] for name, arr in features.items()}

    # Spatial reliability filter: split trials into two halves and correlate rate maps
    if firing_rates.shape[1] >= 4:
        n_trials = firing_rates.shape[1]
        half = n_trials // 2
        idx_a = np.arange(0, half)
        idx_b = np.arange(half, n_trials)
        n_spatial_bins = int(np.ceil((2 * wall_val * 100) / spatial_bin_cm))
        rm_a = _compute_rate_maps(
            features["x"][idx_a],
            features["y"][idx_a],
            firing_rates[:, idx_a],
            n_spatial_bins,
            smoothing_sigma,
        )
        rm_b = _compute_rate_maps(
            features["x"][idx_b],
            features["y"][idx_b],
            firing_rates[:, idx_b],
            n_spatial_bins,
            smoothing_sigma,
        )
        a_flat = rm_a.reshape(rm_a.shape[0], -1)
        b_flat = rm_b.reshape(rm_b.shape[0], -1)
        a_mean = a_flat.mean(axis=1, keepdims=True)
        b_mean = b_flat.mean(axis=1, keepdims=True)
        a_centered = a_flat - a_mean
        b_centered = b_flat - b_mean
        denom = np.linalg.norm(a_centered, axis=1) * np.linalg.norm(b_centered, axis=1)
        corr = np.where(denom > 0, np.sum(a_centered * b_centered, axis=1) / denom, 0.0)
        good_reliability = corr >= min_spatial_reliability
        firing_rates = firing_rates[good_reliability]
        n_cells = firing_rates.shape[0]

    if normalize_per_sample and n_cells > 0:
        target = float(target_l2_norm)
        min_l2 = float(min_l2_norm)
        l2 = np.linalg.norm(firing_rates, axis=1, keepdims=True)
        scale = np.maximum(l2, min_l2)
        firing_rates = firing_rates * (target / scale)

    # Compute and lightly smooth rate maps (kept for consistency with the original workflow)
    n_spatial_bins = int(np.ceil((2 * wall_val * 100) / spatial_bin_cm))
    _ = _compute_rate_maps(features["x"], features["y"], firing_rates, n_spatial_bins, smoothing_sigma)

    data = {
        'pos_x': np.tile(features["x"], (n_cells, 1)),   # (n_samples, n_trials)
        'pos_y': np.tile(features["y"], (n_cells, 1)),   # (n_samples, n_trials)
        'response': firing_rates,                          # (n_samples, n_trials)
    }

    train_samples, train_trials = train_test_split(data, random_seed=random_seed)
    n_trials_final = utils.data_n_trials(data)
    test_samples = np.setdiff1d(np.arange(n_cells, dtype=np.int64), train_samples, assume_unique=False)
    test_trials = np.setdiff1d(np.arange(n_trials_final, dtype=np.int64), train_trials, assume_unique=False)

    data_train_train = utils.slice_data(data, train_samples, train_trials)
    data_train_test = utils.slice_data(data, train_samples, test_trials)
    data_test_train = utils.slice_data(data, test_samples, train_trials)
    data_test_test = utils.slice_data(data, test_samples, test_trials)

    return [[data_train_train, data_train_test], [data_test_train, data_test_test]]


def train_test_split(
    X: Dict[str, np.ndarray],
    # -- ALL SUBSEQUENT PARAMS MUST BE SPECIFIED IN THE CONFIG FILE ---
    random_seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sample and trial train splits (50/50 each).

    Parameters
    ----------
    X : dict[str, np.ndarray]
        Data dictionary. All arrays share the same first dimension (n_samples)
        and the same last dimension (n_trials).
    random_seed : int
        RNG seed.

    Returns
    -------
    train_samples : np.ndarray
        Sample indices for training.
    train_trials : np.ndarray
        Trial indices for training.
    """
    n_samples = utils.data_n_samples(X)
    n_trials = utils.data_n_trials(X)
    assert n_samples >= 2, "Need at least 2 samples for model optimization/eval"
    assert n_trials >= 2, "Need at least 2 trials for parameter optimization/eval"

    rng = np.random.default_rng(random_seed)
    train_samples = rng.choice(np.arange(n_samples), n_samples // 2, replace=False)

    # Split trials into contiguous blocks, then select blocks for train/test.
    n_blocks = 10
    block_size = max(1, n_trials // n_blocks)
    blocks = []
    for start in range(0, n_trials, block_size):
        end = min(start + block_size, n_trials)
        blocks.append(np.arange(start, end))
    blocks = np.array(blocks, dtype=object)

    perm = rng.permutation(len(blocks))
    n_train_blocks = len(blocks) // 2
    train_blocks = blocks[perm[:n_train_blocks]]
    train_trials = np.concatenate(train_blocks) if len(train_blocks) > 0 else np.array([], dtype=int)

    return train_samples, train_trials


# ========================
# 2. SEED MODELS
# ========================


def model_v1(
    data,
    params,
):
    """
    Hexagonal grid-cell model with isotropic Gaussian fields.

    This model defines firing rate as

        r(x) = baseline + amplitude * Sigma_{n,m} exp( -||x - c_{n,m}||^2 / (2*sigma^2) )

    where {c_{n,m}} form a hexagonal lattice with spacing `lam`,
    rotated by `theta`, and shifted by (phi_x, phi_y).

    Parameters
    ----------
    data : dict[str, np.ndarray]
        Data dictionary for a single sample (no sample axis) with keys:
        - 'pos_x': shape (n_trials,) -- x positions.
        - 'pos_y': shape (n_trials,) -- y positions.
    params : dict
        Parameter dictionary with keys:
        - lam: lattice spacing
        - theta: lattice orientation (radians)
        - phi_x, phi_y: spatial phase shifts
        - baseline: additive baseline firing rate
        - amplitude: scaling of summed Gaussian bumps
        - sigma: isotropic Gaussian width
    K_MAX = 10
        Fixed lattice truncation radius. Centers use indices n,m in [-K_MAX, K_MAX].

    Returns
    -------
    np.ndarray, shape (n_trials,)
        Predicted firing rates.
    """
    x = data['pos_x']
    y = data['pos_y']
    lam = params["lam"]
    theta = params["theta"]
    phi_x = params["phi_x"]
    phi_y = params["phi_y"]
    baseline = params["baseline"]
    amplitude = params["amplitude"]
    sigma = params["sigma"]

    # Rotated hexagonal basis vectors
    c, s = np.cos(theta), np.sin(theta)
    v1x, v1y = lam * c, lam * s
    v2x = 0.5 * lam * c - 0.5 * np.sqrt(3.0) * lam * s
    v2y = 0.5 * lam * s + 0.5 * np.sqrt(3.0) * lam * c

    K_MAX = 10
    ns = np.arange(-K_MAX, K_MAX + 1)
    ms = np.arange(-K_MAX, K_MAX + 1)
    nn, mm = np.meshgrid(ns, ms, indexing="ij")

    cx = nn * v1x + mm * v2x
    cy = nn * v1y + mm * v2y
    cx = cx.reshape(-1)[:, None]
    cy = cy.reshape(-1)[:, None]

    dx = (x - phi_x)[None, :] - cx
    dy = (y - phi_y)[None, :] - cy
    dist2 = dx * dx + dy * dy

    inv2sig2 = 1.0 / (2.0 * sigma * sigma + 1e-12)
    bumps = np.exp(-dist2 * inv2sig2)

    return baseline + amplitude * np.sum(bumps, axis=0)


model_v1.DEFAULT_PARAMS = {
    "lam": 0.6,
    "theta": 0.0,
    "phi_x": 0.0,
    "phi_y": 0.0,
    "baseline": 0.0,
    "amplitude": 1.0,
    "sigma": 0.12,
}


def param_est_v1(data):
    """
    Autocorr-based initializer for an isotropic hex-grid Gaussian-bump model.

    Intended model family (for context):
        r(x,y) = baseline + amplitude * Sigma_{n,m} exp(-||[x,y]-c_{n,m}||^2 / (2*sigma^2)),
    where {c_{n,m}} is a rotated hexagonal lattice with spacing `lam`, orientation `theta`,
    and global phase shift (phi_x, phi_y).

    What this estimator does (self-contained, robust-ish):
    1) Bin samples into a 2D rate map R over [-1,1]^2 and lightly smooth it.
    2) Compute 2D autocorrelation A of (masked, windowed, zero-mean) R via FFT.
       For grid structure, A has a central peak and a first ring of 6 peaks.
    3) Estimate:
       - `lam` from radius (in bins -> coordinate units) of the nearest non-central peak.
       - `theta` from the angle of that peak, reduced modulo pi/3.
    4) Estimate (phi_x, phi_y) by FFT cross-correlation between R and a hex-lattice
       template (built with the estimated lam/theta and a default sigma), taking the
       argmax shift.
    5) baseline/amplitude from response percentiles; sigma proportional to lam.

    Parameters
    ----------
    data : dict[str, np.ndarray]
        Data dictionary for a single sample (no sample axis) with keys:
        - 'pos_x': shape (n_trials,) -- x positions.
        - 'pos_y': shape (n_trials,) -- y positions.
        - 'response': shape (n_trials,) -- observed firing rates.

    Returns
    -------
    dict
        Parameter dictionary with keys
        ["lam", "theta", "phi_x", "phi_y", "baseline", "amplitude", "sigma"].
    """
    from scipy.ndimage import gaussian_filter

    x = np.asarray(data['pos_x'], float)
    y = np.asarray(data['pos_y'], float)
    yobs = np.asarray(data['response'], float)

    # ---------- fixed internal hyperparameters ----------
    nbins = 48
    smooth_sigma = 2.5
    occ_prct = 20          # occupancy mask threshold percentile
    peak_excl_frac = 0.08  # exclude small radii in autocorr
    topk = 120             # peak candidates in autocorr band
    # ----------------------------------------------------

    # Robust baseline/amplitude init
    baseline = float(np.percentile(yobs, 10))
    amplitude = float(max(1e-6, np.percentile(yobs, 95) - baseline))

    # --- build smoothed rate map R over [-1,1]^2 ---
    edges = np.linspace(-1.0, 1.0, nbins + 1)
    occ, _, _ = np.histogram2d(x, y, bins=[edges, edges])
    heat, _, _ = np.histogram2d(x, y, bins=[edges, edges], weights=yobs)
    R = heat / (occ + 1e-8)

    # Smooth in occupancy-normalized way (reduces sampling noise)
    Rs = gaussian_filter(R, smooth_sigma)
    Os = gaussian_filter(occ, smooth_sigma)
    Rm = Rs / (Os + 1e-8)

    # Occupancy mask (ignore sparse bins)
    if np.any(occ > 0):
        thr = np.percentile(occ[occ > 0], occ_prct)
        mask = (occ >= thr).astype(float)
    else:
        mask = np.ones_like(Rm)

    # --- autocorrelation via FFT of windowed, masked, zero-mean map ---
    wy = np.hanning(nbins)
    wx = np.hanning(nbins)
    win = wy[:, None] * wx[None, :]
    w = mask * win
    mu = np.sum(Rm * w) / (np.sum(w) + 1e-12)
    Z = (Rm - mu) * w

    F = np.fft.fft2(Z)
    A = np.fft.ifft2(np.abs(F) ** 2).real
    A = np.fft.fftshift(A)
    A = A / (np.max(A) + 1e-12)

    # --- pick nearest-ring peak in autocorr to estimate lam, theta ---
    cy, cx = nbins // 2, nbins // 2
    yy, xx = np.indices((nbins, nbins))
    dy = yy - cy
    dx = xx - cx
    rr = np.sqrt(dx * dx + dy * dy)

    r_excl = peak_excl_frac * np.max(rr)
    rmin = max(r_excl, 0.15 * np.max(rr))
    rmax = 0.6 * np.max(rr)
    band = (rr >= rmin) & (rr <= rmax)
    Ab = np.where(band, A, -np.inf)

    flat = Ab.ravel()
    k = min(topk, flat.size)
    idx = np.argpartition(flat, -k)[-k:]
    py, px = np.unravel_index(idx, (nbins, nbins))
    vals = Ab[py, px]
    order = np.argsort(vals)[::-1]
    py, px = py[order], px[order]

    if len(py) == 0 or not np.isfinite(vals).any():
        lam = 0.6
        theta = 0.0
    else:
        # pick the strongest peak in the annulus as the ring representative
        py0, px0 = py[0], px[0]
        r0 = float(rr[py0, px0])
        ang0 = float(np.arctan2(py0 - cy, px0 - cx))

        binw = 2.0 / nbins
        lam = float(np.clip(r0 * binw, 1.0, 1.5))
        theta = float(ang0 % (np.pi / 3))

    sigma = float(np.clip(0.20 * lam, 0.01, 0.6))

    # --- phase (phi_x,phi_y) via FFT cross-correlation with a small lattice template ---
    xs = (np.arange(nbins) + 0.5) * (2.0 / nbins) - 1.0
    ys = (np.arange(nbins) + 0.5) * (2.0 / nbins) - 1.0
    Xg, Yg = np.meshgrid(xs, ys, indexing="ij")

    c, s = np.cos(theta), np.sin(theta)
    v1 = np.array([lam * c, lam * s])
    v2 = np.array([0.5 * lam * c - 0.5 * np.sqrt(3.0) * lam * s,
                   0.5 * lam * s + 0.5 * np.sqrt(3.0) * lam * c])

    K = int(np.clip(np.ceil(2.0 / max(lam, 1e-3)), 2, 5))
    ns = np.arange(-K, K + 1)
    ms = np.arange(-K, K + 1)
    nn, mm = np.meshgrid(ns, ms, indexing="ij")
    centers = nn[..., None] * v1 + mm[..., None] * v2
    centers = centers.reshape(-1, 2)

    inv2 = 1.0 / (2.0 * sigma * sigma + 1e-12)
    dx0 = Xg[None, :, :] - centers[:, 0][:, None, None]
    dy0 = Yg[None, :, :] - centers[:, 1][:, None, None]
    T = np.exp(-(dx0 * dx0 + dy0 * dy0) * inv2).sum(axis=0)

    Rz = (Rm - Rm.mean()) / (Rm.std() + 1e-12)
    Tz = (T - T.mean()) / (T.std() + 1e-12)
    C = np.fft.ifft2(np.fft.fft2(Rz) * np.conj(np.fft.fft2(Tz))).real
    iy, ix = np.unravel_index(np.argmax(C), C.shape)

    sy = iy if iy <= nbins // 2 else iy - nbins
    sx = ix if ix <= nbins // 2 else ix - nbins
    binw = 2.0 / nbins
    shift_x = sx * binw
    shift_y = sy * binw
    phi_x = float(np.clip(-shift_x, -1.0, 1.0))
    phi_y = float(np.clip(-shift_y, -1.0, 1.0))

    return {
        "lam": float(lam),
        "theta": float(theta),
        "phi_x": float(phi_x),
        "phi_y": float(phi_y),
        "baseline": float(baseline),
        "amplitude": float(amplitude),
        "sigma": float(sigma),
    }


def model_v2(
    data,
    params,
):
    """
    Hexagonal grid-cell model with anisotropic (elliptical) Gaussian fields.

    This model generalizes model_v1 by allowing elliptical receptive fields
    aligned to the lattice frame. Firing rate is

        r(x) = baseline + amplitude * Sigma_{n,m}
               exp( - (u^2 / (2*sigma_par^2) + v^2 / (2*sigma_perp^2)) )

    where (u,v) are coordinates of x - c_{n,m} expressed in the rotated
    lattice frame (angle theta).

    Parameters
    ----------
    data : dict[str, np.ndarray]
        Data dictionary for a single sample (no sample axis) with keys:
        - 'pos_x': shape (n_trials,) -- x positions.
        - 'pos_y': shape (n_trials,) -- y positions.
    params : dict
        Parameter dictionary with keys:
        - lam: lattice spacing
        - theta: lattice orientation (radians)
        - phi_x, phi_y: spatial phase shift
        - baseline: additive baseline firing rate
        - amplitude: global scaling
        - sigma_par: Gaussian width parallel to lattice axis
        - sigma_perp: Gaussian width perpendicular to lattice axis
    K_MAX = 10
        Fixed lattice truncation radius.

    Returns
    -------
    np.ndarray, shape (n_trials,)
        Predicted firing rates.
    """
    x = data['pos_x']
    y = data['pos_y']
    lam = params["lam"]
    theta = params["theta"]
    phi_x = params["phi_x"]
    phi_y = params["phi_y"]
    baseline = params["baseline"]
    amplitude = params["amplitude"]
    sigma_par = params["sigma_par"]
    sigma_perp = params["sigma_perp"]

    c, s = np.cos(theta), np.sin(theta)
    v1x, v1y = lam * c, lam * s
    v2x = 0.5 * lam * c - 0.5 * np.sqrt(3.0) * lam * s
    v2y = 0.5 * lam * s + 0.5 * np.sqrt(3.0) * lam * c

    K_MAX = 10
    ns = np.arange(-K_MAX, K_MAX + 1)
    ms = np.arange(-K_MAX, K_MAX + 1)
    nn, mm = np.meshgrid(ns, ms, indexing="ij")

    cx = nn * v1x + mm * v2x
    cy = nn * v1y + mm * v2y
    cx = cx.reshape(-1)[:, None]
    cy = cy.reshape(-1)[:, None]

    dx = (x - phi_x)[None, :] - cx
    dy = (y - phi_y)[None, :] - cy

    # Rotate displacement into lattice frame
    u =  c * dx + s * dy
    v = -s * dx + c * dy

    inv2sp2 = 1.0 / (2.0 * sigma_par * sigma_par + 1e-12)
    inv2st2 = 1.0 / (2.0 * sigma_perp * sigma_perp + 1e-12)

    dist2 = u * u * inv2sp2 + v * v * inv2st2
    bumps = np.exp(-dist2)

    return baseline + amplitude * np.sum(bumps, axis=0)


model_v2.DEFAULT_PARAMS = {
    "lam": 0.6,
    "theta": 0.0,
    "phi_x": 0.0,
    "phi_y": 0.0,
    "baseline": 0.0,
    "amplitude": 1.0,
    "sigma_par": 0.14,
    "sigma_perp": 0.10,
}


def param_est_v2(data):
    """
    Autocorr-based initializer for an anisotropic hex-grid Gaussian-bump model.

    Intended model family (for context):
        r(x,y) = baseline + amplitude * Sigma exp(-(u^2/(2*sigma_par^2) + v^2/(2*sigma_perp^2))),
    where (u,v) are coordinates of (x,y) in the lattice-aligned frame (rotation `theta`).

    What this estimator does (self-contained):
    - Same autocorrelation + template-shift steps as v1 to get (lam, theta, phi_x, phi_y).
    - Then estimates anisotropy (sigma_par, sigma_perp) by computing a weighted second moment
      of the local neighborhood around a strong peak of the smoothed rate map, and
      rotating that covariance into the lattice frame.

    Parameters
    ----------
    data : dict[str, np.ndarray]
        Data dictionary for a single sample (no sample axis) with keys:
        - 'pos_x': shape (n_trials,) -- x positions.
        - 'pos_y': shape (n_trials,) -- y positions.
        - 'response': shape (n_trials,) -- observed firing rates.

    Returns
    -------
    dict
        Parameter dictionary with keys
        ["lam", "theta", "phi_x", "phi_y", "baseline", "amplitude", "sigma_par", "sigma_perp"].
    """
    from scipy.ndimage import gaussian_filter

    x = np.asarray(data['pos_x'], float)
    y = np.asarray(data['pos_y'], float)
    yobs = np.asarray(data['response'], float)

    # ---------- fixed internal hyperparameters ----------
    nbins = 48
    smooth_sigma = 2.5
    occ_prct = 20
    peak_excl_frac = 0.08
    topk = 120
    # ----------------------------------------------------

    baseline = float(np.percentile(yobs, 10))
    amplitude = float(max(1e-6, np.percentile(yobs, 95) - baseline))

    # --- rate map ---
    edges = np.linspace(-1.0, 1.0, nbins + 1)
    occ, _, _ = np.histogram2d(x, y, bins=[edges, edges])
    heat, _, _ = np.histogram2d(x, y, bins=[edges, edges], weights=yobs)
    R = heat / (occ + 1e-8)

    Rs = gaussian_filter(R, smooth_sigma)
    Os = gaussian_filter(occ, smooth_sigma)
    Rm = Rs / (Os + 1e-8)

    if np.any(occ > 0):
        thr = np.percentile(occ[occ > 0], occ_prct)
        mask = (occ >= thr).astype(float)
    else:
        mask = np.ones_like(Rm)

    # --- autocorr ---
    wy = np.hanning(nbins)
    wx = np.hanning(nbins)
    win = wy[:, None] * wx[None, :]
    w = mask * win
    mu = np.sum(Rm * w) / (np.sum(w) + 1e-12)
    Z = (Rm - mu) * w

    F = np.fft.fft2(Z)
    A = np.fft.ifft2(np.abs(F) ** 2).real
    A = np.fft.fftshift(A)
    A = A / (np.max(A) + 1e-12)

    cy, cx = nbins // 2, nbins // 2
    yy, xx = np.indices((nbins, nbins))
    dy = yy - cy
    dx = xx - cx
    rr = np.sqrt(dx * dx + dy * dy)

    r_excl = peak_excl_frac * np.max(rr)
    rmin = max(r_excl, 0.15 * np.max(rr))
    rmax = 0.6 * np.max(rr)
    band = (rr >= rmin) & (rr <= rmax)
    Ab = np.where(band, A, -np.inf)

    flat = Ab.ravel()
    k = min(topk, flat.size)
    idx = np.argpartition(flat, -k)[-k:]
    py, px = np.unravel_index(idx, (nbins, nbins))
    vals = Ab[py, px]
    order = np.argsort(vals)[::-1]
    py, px = py[order], px[order]

    if len(py) == 0 or not np.isfinite(vals).any():
        lam = 0.6
        theta = 0.0
    else:
        py0, px0 = py[0], px[0]
        r0 = float(rr[py0, px0])
        ang0 = float(np.arctan2(py0 - cy, px0 - cx))

        binw = 2.0 / nbins
        lam = float(np.clip(r0 * binw, 1.0, 1.5))
        theta = float(ang0 % (np.pi / 3))

    sigma0 = float(np.clip(0.20 * lam, 0.01, 0.6))

    # --- phase via FFT cross-correlation with a small lattice template ---
    xs = (np.arange(nbins) + 0.5) * (2.0 / nbins) - 1.0
    ys = (np.arange(nbins) + 0.5) * (2.0 / nbins) - 1.0
    Xg, Yg = np.meshgrid(xs, ys, indexing="ij")

    c, s = np.cos(theta), np.sin(theta)
    v1 = np.array([lam * c, lam * s])
    v2 = np.array([0.5 * lam * c - 0.5 * np.sqrt(3.0) * lam * s,
                   0.5 * lam * s + 0.5 * np.sqrt(3.0) * lam * c])

    K = int(np.clip(np.ceil(2.0 / max(lam, 1e-3)), 2, 5))
    ns = np.arange(-K, K + 1)
    ms = np.arange(-K, K + 1)
    nn, mm = np.meshgrid(ns, ms, indexing="ij")
    centers = nn[..., None] * v1 + mm[..., None] * v2
    centers = centers.reshape(-1, 2)

    inv2 = 1.0 / (2.0 * sigma0 * sigma0 + 1e-12)
    dx0 = Xg[None, :, :] - centers[:, 0][:, None, None]
    dy0 = Yg[None, :, :] - centers[:, 1][:, None, None]
    T = np.exp(-(dx0 * dx0 + dy0 * dy0) * inv2).sum(axis=0)

    Rz = (Rm - Rm.mean()) / (Rm.std() + 1e-12)
    Tz = (T - T.mean()) / (T.std() + 1e-12)
    Ccorr = np.fft.ifft2(np.fft.fft2(Rz) * np.conj(np.fft.fft2(Tz))).real
    iy, ix = np.unravel_index(np.argmax(Ccorr), Ccorr.shape)

    sy = iy if iy <= nbins // 2 else iy - nbins
    sx = ix if ix <= nbins // 2 else ix - nbins
    binw = 2.0 / nbins
    shift_x = sx * binw
    shift_y = sy * binw

    phi_x = float(np.clip(-shift_x, -1.0, 1.0))
    phi_y = float(np.clip(-shift_y, -1.0, 1.0))

    # --- anisotropy from local second moments around a strong peak ---
    # pick peak in smoothed map
    iy0, ix0 = np.unravel_index(np.argmax(Rm), Rm.shape)

    # local window size scaled by lam (in bins)
    lam_bins = max(1.0, lam / binw)
    rad = int(np.clip(np.round(0.6 * lam_bins), 3, 12))

    y0, y1 = max(0, iy0 - rad), min(nbins, iy0 + rad + 1)
    x0, x1 = max(0, ix0 - rad), min(nbins, ix0 + rad + 1)
    patch = Rm[y0:y1, x0:x1].copy()

    # weights: positive part above a floor
    floor = np.percentile(patch, 30)
    wgt = np.maximum(patch - floor, 0.0) + 1e-12
    wgt /= np.sum(wgt)

    # physical coordinates centered at the peak-bin center
    ys_loc = (np.arange(y0, y1) - (iy0 + 0.0)) * binw
    xs_loc = (np.arange(x0, x1) - (ix0 + 0.0)) * binw
    YY, XX = np.meshgrid(ys_loc, xs_loc, indexing="ij")  # note YY first

    mx = np.sum(wgt * XX)
    my = np.sum(wgt * YY)
    DX = XX - mx
    DY = YY - my

    Cxx = np.sum(wgt * DX * DX)
    Cyy = np.sum(wgt * DY * DY)
    Cxy = np.sum(wgt * DX * DY)

    # rotate covariance into lattice frame: [u;v] = [[c,s],[-s,c]] [x;y]
    c, s = np.cos(theta), np.sin(theta)
    Rmat = np.array([[c, s], [-s, c]])
    Cmat = np.array([[Cxx, Cxy], [Cxy, Cyy]])
    Cuv = Rmat @ Cmat @ Rmat.T

    sigma_par = float(np.clip(np.sqrt(max(Cuv[0, 0], 1e-8)), 0.01, 0.8))
    sigma_perp = float(np.clip(np.sqrt(max(Cuv[1, 1], 1e-8)), 0.01, 0.8))

    # avoid extreme anisotropy; shrink toward sigma0 for stability
    ratio = sigma_par / max(sigma_perp, 1e-8)
    if ratio > 2.5 or ratio < 0.4:
        sigma_par = 0.5 * sigma_par + 0.5 * sigma0
        sigma_perp = 0.5 * sigma_perp + 0.5 * sigma0
    else:
        sigma_par = 0.7 * sigma_par + 0.3 * sigma0
        sigma_perp = 0.7 * sigma_perp + 0.3 * sigma0

    return {
        "lam": float(lam),
        "theta": float(theta),
        "phi_x": float(phi_x),
        "phi_y": float(phi_y),
        "baseline": float(baseline),
        "amplitude": float(amplitude),
        "sigma_par": float(sigma_par),
        "sigma_perp": float(sigma_perp),
    }


# ========================
# 3. LOSS
# ========================

def loss_fn(model_output, data):
    """
    Elementwise squared-error loss.

    Parameters
    ----------
    model_output : np.ndarray
        Model predictions.
    data : dict[str, np.ndarray]
        Data dictionary; the comparison target is extracted from data['response'].

    Returns
    -------
    np.ndarray
        Elementwise squared errors.
    """
    return (data['response'] - model_output) ** 2


# ========================
# 4. DIAGNOSTICS
# ========================

def plot_model_fits(
    data,
    programs_list,
    eval_grid,
    save_path="",
    labels=("model_v1", "model_v2"),
    n_bins: int = 40,
    smoothing_sigma: float = 1.0,
    max_show: int = 6,
    show_colorbar: bool = False,
    # -- ALL SUBSEQUENT PARAMS MUST BE SPECIFIED IN THE CONFIG FILE ---
):
    """
    Plot data and model-predicted 2D rate maps for up to 6 random samples.

    Parameters
    ----------
    data : dict[str, np.ndarray]
        Data dictionary with keys 'pos_x', 'pos_y', 'response'.
        Each array has shape (n_samples, n_trials).
    programs_list : list[dict]
        List of dictionaries with keys:
        - `'model'`: callable model function
        - `'params'`: batched parameter pytree
        - optionally `'losses'`: per-sample losses
    eval_grid : dict[str, np.ndarray]
        Evaluation grid dictionary with keys 'pos_x', 'pos_y'.
        Each array has shape (n_eval_points,).
    save_path : str
        Output figure path.
    labels : tuple of str
        Labels for each model in programs_list.
    """
    if save_path == "":
        raise ValueError("Please provide a save_path for the plot")

    pos_x = np.asarray(data['pos_x'])
    pos_y = np.asarray(data['pos_y'])
    response = np.asarray(data['response'])
    eval_pos_x = np.asarray(eval_grid['pos_x']).reshape(-1)
    eval_pos_y = np.asarray(eval_grid['pos_y']).reshape(-1)

    n_samples = pos_x.shape[0]
    n_show = min(max_show, n_samples)
    # Intentionally unseeded so displayed samples vary across calls/runs.
    show_idx = np.random.default_rng().choice(n_samples, size=n_show, replace=False)

    n_models = len(programs_list)
    fig, axes = plt.subplots(n_show, 1 + n_models, figsize=(4 * (1 + n_models), 3 * n_show))
    axes = np.atleast_2d(axes)

    # Normalize params shape per program to avoid indexing mismatches.
    params_by_model = [
        utils.broadcast_params(program["params"], n_samples)
        for program in programs_list
    ]

    for row, s in enumerate(show_idx):
        x = pos_x[s]
        y = pos_y[s]
        y_obs = response[s]
        x_domain = (float(np.min(eval_pos_x)), float(np.max(eval_pos_x)))
        y_domain = (float(np.min(eval_pos_y)), float(np.max(eval_pos_y)))

        rm_obs = _bin_to_rate_map(
            x, y, y_obs, n_bins=n_bins, x_domain=x_domain, y_domain=y_domain,
            smoothing_sigma=smoothing_sigma
        )
        ax = axes[row, 0]
        im = ax.imshow(
            rm_obs.T,
            origin="lower",
            extent=[x_domain[0], x_domain[1], y_domain[0], y_domain[1]],
            cmap="viridis",
        )
        ax.set_title(f"Sample {s} data")
        if show_colorbar:
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        for m_idx, program in enumerate(programs_list):
            model = program["model"]
            params = utils.slice_params(params_by_model[m_idx], s)
            sample_data = {'pos_x': pos_x[s], 'pos_y': pos_y[s]}
            y_pred = utils.call_model(model, sample_data, params)
            rm_pred = _bin_to_rate_map(
                x, y, y_pred, n_bins=n_bins, x_domain=x_domain, y_domain=y_domain,
                smoothing_sigma=smoothing_sigma
            )

            axm = axes[row, m_idx + 1]
            imm = axm.imshow(
                rm_pred.T,
                origin="lower",
                extent=[x_domain[0], x_domain[1], y_domain[0], y_domain[1]],
                cmap="viridis",
            )
            label = labels[m_idx] if labels is not None and m_idx < len(labels) else f"Model {m_idx + 1}"
            if "losses" in program:
                label += f" (loss={program['losses'][s]:.2f})"
            axm.set_title(label)
            if show_colorbar:
                fig.colorbar(imm, ax=axm, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(save_path, dpi=100.0, bbox_inches="tight")
    plt.close(fig)


# ========================
# 5. OPTIONAL PROJECT-SPECIFIC HELPERS
# ========================

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
    x_domain: Tuple[float, float] = (-1.0, 1.0),
    y_domain: Tuple[float, float] = (-1.0, 1.0),
    smoothing_sigma: float = 1.0,
) -> np.ndarray:
    edges_x = np.linspace(x_domain[0], x_domain[1], n_bins + 1)
    edges_y = np.linspace(y_domain[0], y_domain[1], n_bins + 1)
    occ, _, _ = np.histogram2d(x, y, bins=[edges_x, edges_y])
    weighted, _, _ = np.histogram2d(x, y, bins=[edges_x, edges_y], weights=values)
    if smoothing_sigma and smoothing_sigma > 0:
        occ = gaussian_filter(occ, sigma=smoothing_sigma)
        weighted = gaussian_filter(weighted, sigma=smoothing_sigma)
    return weighted / (occ + 1e-8)
