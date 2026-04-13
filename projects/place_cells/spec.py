"""
Welcome to the Model Discovery Engine! Fill in the components below to start building your model.

NECESSARY COMPONENTS:

Loading:
- load_and_process_data(data_path, *preprocess_params) -> dict[str, np.ndarray]
- train_test_split(X, random_seed) -> [train_samples, train_trials]

Seed Programs:
- model_v1(data, params) and param_est_v1(data)
- model_v2(data, params) and param_est_v2(data)

Loss:
- loss_fn(model_output, data) -> loss values

OPTIONAL COMPONENTS:
- plot_model_fits(data, programs_list, eval_grid, save_path, labels)
"""

import time
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter

from src import utils


# ========================
# 1. DATA
# ========================

def load_and_process_data(
    data_path: str,
    # ---- ALL SUBSEQUENT PARAMS MUST BE SPECIFIED IN THE CONFIG FILE ----
    random_seed: int = 42,
    time_start: Optional[float] = None,
    time_end: Optional[float] = None,
    time_bin_ms: int = 20,
    input_names: Optional[List[str]] = None,
    min_spikes: int = 50,
    speed_threshold: float = 2.5,
    max_trials: Optional[int] = 8000,
    zscore_response: bool = True,
) -> list[list[dict[str, np.ndarray]]]:
    """
    Load and preprocess place-cell data and return a dict of arrays.

    This function converts raw position/spike-time recordings into:
    - 'pos_x': x-position array of shape (n_samples, n_trials)
    - 'pos_y': y-position array of shape (n_samples, n_trials)
    - 'response': firing-rate array of shape (n_samples, n_trials)

    All arrays share the same first dimension (n_samples = n_cells) and last
    dimension (n_trials = n_time_bins).

    Parameters
    ----------
    data_path : str
        Path to a `.npz` or `.npy` file containing position and spike-time data.
    time_start, time_end : float or None
        Time window (seconds) to extract. If `None`, the full available range is used.
    time_bin_ms : int
        Temporal bin size (ms) used to convert spike times to firing rates.
    input_names : list[str] or None
        Ordered list of input variable names to load (default `['x', 'y']`).
    min_spikes : int
        Minimum spikes required for a neuron to be retained.
    speed_threshold : float
        Minimum speed (cm/s) required for trial bins to be retained.
    max_trials : int or None
        If provided, subsamples the number of time bins to this value.
    zscore_response : bool
        Whether to z-score response per cell across trials.

    Returns
    -------
    2 x 2 list of dicts
        ``[[data_train_train, data_train_test], [data_test_train, data_test_test]]``.
    """
    # Hardcoded defaults for simplified API.
    spatial_bin_cm = 3.0
    smoothing_sigma = 1.5
    wall_val = 0.75
    spike_key = None
    time_key = None
    position_time_key = None
    drop_nan_bins = True
    filter_place_cells = True
    place_filter_kwargs = None

    clock_time_start = time.time()
    if input_names is None:
        input_names = ["x", "y"]

    data_dict = _load_data_file(data_path)

    # Resolve keys
    time_key = _find_key(data_dict, time_key, ["t", "time", "timestamps", "pos_t", "position_t"])
    t_raw = np.asarray(data_dict[time_key], dtype=float)

    if position_time_key is None:
        position_time_key = time_key
    t_pos = np.asarray(data_dict[position_time_key], dtype=float) if position_time_key in data_dict else t_raw

    spike_key = _find_key(
        data_dict,
        spike_key,
        ["spikes", "spike_times", "spikes_mod1", "spikes_mod2", "spike_times_dict"],
    )
    spike_times_dict = _to_spike_dict(data_dict[spike_key])
    n_neurons_raw = len(spike_times_dict)

    features_raw: Dict[str, np.ndarray] = {}
    for feat in input_names:
        if feat not in data_dict:
            raise KeyError(f"Input '{feat}' requested but not available. Available: {list(data_dict.keys())}")
        features_raw[feat] = np.asarray(data_dict[feat], dtype=float)

    if time_start is None:
        time_start = float(np.nanmin(t_pos))
    if time_end is None:
        time_end = float(np.nanmax(t_pos))
    if time_end <= time_start:
        raise ValueError(f"time_end ({time_end}) must be > time_start ({time_start}).")

    arena_size_cm = 2 * wall_val * 100
    n_spatial_bins = int(np.ceil(arena_size_cm / spatial_bin_cm))
    print(
        f"{time.time() - clock_time_start:.3f}s : Spatial binning: "
        f"{n_spatial_bins}x{n_spatial_bins} bins of {spatial_bin_cm:.1f}cm for {arena_size_cm:.0f}cm arena"
    )

    # 1) time-window selection for position streams
    time_mask = (t_pos >= time_start) & (t_pos < time_end)
    t_pos = t_pos[time_mask]
    for name in list(features_raw.keys()):
        features_raw[name] = features_raw[name][time_mask]

    # 2) common time bins
    time_bin_s = time_bin_ms / 1000.0
    n_time_bins = int(np.ceil((time_end - time_start) / time_bin_s))
    bin_edges = np.linspace(time_start, time_end, n_time_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # 3) spike-times -> binned rates
    firing_rates = np.zeros((n_neurons_raw, n_time_bins), dtype=float)
    total_spikes = np.zeros(n_neurons_raw, dtype=float)
    for neuron_idx, (_, spike_times) in enumerate(spike_times_dict.items()):
        spike_times = np.asarray(spike_times, dtype=float)
        spikes_in_window = spike_times[(spike_times >= time_start) & (spike_times < time_end)]
        total_spikes[neuron_idx] = len(spikes_in_window)
        spike_counts, _ = np.histogram(spikes_in_window, bins=bin_edges)
        firing_rates[neuron_idx] = spike_counts / time_bin_s

    # 4) bin behavioral features by time bin
    bin_idx = np.digitize(t_pos, bin_edges) - 1
    bin_idx = np.clip(bin_idx, 0, n_time_bins - 1)
    counts = np.bincount(bin_idx, minlength=n_time_bins)

    features: Dict[str, np.ndarray] = {}
    for feat_name, feat_raw in features_raw.items():
        sums = np.bincount(bin_idx, weights=feat_raw, minlength=n_time_bins)
        with np.errstate(invalid="ignore"):
            features[feat_name] = np.where(counts > 0, sums / counts, np.nan)

    # 5) normalize positions to [-1, 1] (assume source data may be in meters).
    for axis in ("x", "y"):
        if axis in features and np.nanmax(np.abs(features[axis])) > 1.5:
            features[axis] = features[axis] / max(wall_val, 1e-6)

    # 6) drop NaN bins
    if drop_nan_bins:
        valid = np.ones(n_time_bins, dtype=bool)
        for arr in features.values():
            valid &= ~np.isnan(arr)
        if not np.all(valid):
            firing_rates = firing_rates[:, valid]
            bin_centers = bin_centers[valid]
            features = {name: arr[valid] for name, arr in features.items()}
            n_time_bins = len(bin_centers)

    # 7) speed filtering
    if "x" in features and "y" in features and speed_threshold > 0:
        arena_half_width_cm = wall_val * 100
        dx = np.diff(features["x"], prepend=features["x"][0]) * arena_half_width_cm
        dy = np.diff(features["y"], prepend=features["y"][0]) * arena_half_width_cm
        speed = np.sqrt(dx * dx + dy * dy) / time_bin_s
        speed[0] = speed[1] if len(speed) > 1 else 0.0
        keep = speed >= speed_threshold
        firing_rates = firing_rates[:, keep]
        bin_centers = bin_centers[keep]
        features = {name: arr[keep] for name, arr in features.items()}
        n_time_bins = len(bin_centers)

    # 8) cap number of trials
    if max_trials is not None and n_time_bins > max_trials:
        keep_idx = np.linspace(0, n_time_bins - 1, max_trials).astype(int)
        firing_rates = firing_rates[:, keep_idx]
        bin_centers = bin_centers[keep_idx]
        features = {name: arr[keep_idx] for name, arr in features.items()}
        n_time_bins = len(bin_centers)

    # 9) min-spike filter
    keep_cells = total_spikes >= min_spikes
    firing_rates = firing_rates[keep_cells]
    n_cells = firing_rates.shape[0]
    print(
        f"{time.time() - clock_time_start:.3f}s : Loaded {n_cells} neurons "
        f"(from {n_neurons_raw} total) with >= {min_spikes} spikes"
    )

    # 10) compute ratemaps
    x = features["x"]
    y = features["y"]
    occupancy, _, _ = np.histogram2d(x, y, bins=n_spatial_bins, range=[[-1, 1], [-1, 1]])
    occupancy_s = gaussian_filter(occupancy, sigma=smoothing_sigma) if smoothing_sigma is not None else occupancy

    bin_x = np.clip(((x + 1.0) / 2.0 * n_spatial_bins).astype(int), 0, n_spatial_bins - 1)
    bin_y = np.clip(((y + 1.0) / 2.0 * n_spatial_bins).astype(int), 0, n_spatial_bins - 1)
    rate_maps = np.zeros((n_cells, n_spatial_bins, n_spatial_bins), dtype=float)

    for c in range(n_cells):
        spike_map = np.zeros((n_spatial_bins, n_spatial_bins), dtype=float)
        for t in range(n_time_bins):
            spike_map[bin_x[t], bin_y[t]] += firing_rates[c, t]
        if smoothing_sigma is None:
            rate_maps[c] = spike_map / (occupancy + 1e-6)
        else:
            spike_map_s = gaussian_filter(spike_map, sigma=smoothing_sigma)
            rate_maps[c] = spike_map_s / (occupancy_s + 1e-6)

    # 11) optional place-cell filtering
    if filter_place_cells:
        filter_kwargs = place_filter_kwargs or {}
        keep_idx, _ = _place_cell_filter_indices(
            response=firing_rates,
            rate_maps=rate_maps,
            x=x,
            y=y,
            min_spatial_info=float(filter_kwargs.get("min_spatial_info", 0.3)),
            min_peak_rate=float(filter_kwargs.get("min_peak_rate", 1.0)),
            min_mean_rate=float(filter_kwargs.get("min_mean_rate", 0.05)),
            verbose=bool(filter_kwargs.get("verbose", True)),
        )
        if len(keep_idx) > 0:
            firing_rates = firing_rates[keep_idx]
            rate_maps = rate_maps[keep_idx]

    # 12) normalize outputs
    n_cells = firing_rates.shape[0]
    response = firing_rates
    if zscore_response:
        response = (response - response.mean(axis=1, keepdims=True)) / (response.std(axis=1, keepdims=True) + 1e-6)
    _ = rate_maps

    # 13) build output dict: each array has shape (n_samples, n_trials)
    # where n_samples = n_cells and n_trials = n_time_bins.
    # pos_x and pos_y are tiled across cells so every sample shares positions.
    data = {
        "pos_x": np.tile(features["x"], (n_cells, 1)),   # (n_cells, n_trials)
        "pos_y": np.tile(features["y"], (n_cells, 1)),   # (n_cells, n_trials)
        "response": response,                              # (n_cells, n_trials)
    }

    train_samples, train_trials = train_test_split(data, random_seed=random_seed)
    test_samples = np.setdiff1d(np.arange(n_cells, dtype=np.int64), train_samples, assume_unique=False)
    test_trials = np.setdiff1d(np.arange(n_time_bins, dtype=np.int64), train_trials, assume_unique=False)

    data_train_train = utils.slice_data(data, train_samples, train_trials)
    data_train_test = utils.slice_data(data, train_samples, test_trials)
    data_test_train = utils.slice_data(data, test_samples, train_trials)
    data_test_test = utils.slice_data(data, test_samples, test_trials)

    skip_keys = ["pos_x", "pos_y"]
    data_train_train = utils.zscore_data(data_train_train, skip_keys=skip_keys)
    data_train_test = utils.zscore_data(data_train_test, skip_keys=skip_keys)
    data_test_train = utils.zscore_data(data_test_train, skip_keys=skip_keys)
    data_test_test = utils.zscore_data(data_test_test, skip_keys=skip_keys)

    return [[data_train_train, data_train_test], [data_test_train, data_test_test]]


def train_test_split(
    X: Dict[str, np.ndarray],
    # -- ALL SUBSEQUENT PARAMS MUST BE SPECIFIED IN THE CONFIG FILE ---
    random_seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Define train splits for samples and trials.

    This mirrors the orientation-tuning contract:
    - random half of samples for model optimization,
    - random half of trials for parameter optimization.

    Parameters
    ----------
    X : dict[str, np.ndarray]
        Data dictionary. Each value has shape (n_samples, ..., n_trials).
    random_seed : int
        Seed used for reproducible splitting.

    Returns
    -------
    train_samples : np.ndarray
        Sample indices of length `n_samples // 2`.
    train_trials : np.ndarray
        Trial indices of length `n_trials // 2`.
    """
    n_samples = utils.data_n_samples(X)
    n_trials = utils.data_n_trials(X)
    assert n_samples >= 2, "Need at least 2 samples for model optimization/eval"
    assert n_trials >= 2, "Need at least 2 trials for parameter optimization/eval"

    rng = np.random.default_rng(random_seed)
    train_samples = rng.choice(np.arange(n_samples), n_samples // 2, replace=False)
    train_trials = rng.choice(np.arange(n_trials), n_trials // 2, replace=False)
    return train_samples, train_trials


# ========================
# 2. SEED MODELS
# ========================

def model_v1(data, params):
    """
    Independent variable:
    data['pos_x'], data['pos_y']  # 2D position (normalized approximately to [-1, 1])

    A simple place-cell model with an isotropic 2D Gaussian firing field:
    r(x, y) = baseline + amplitude * exp(-0.5 * d^2 / sigma^2)
    where d^2 = (x - x0)^2 + (y - y0)^2.

    Args:
        data (dict): Single-sample data dictionary with keys:
            - 'pos_x': x-position array of shape (n_trials,).
            - 'pos_y': y-position array of shape (n_trials,).
        params (dict): Parameter dictionary with keys:
            - x0: Place-field center x-coordinate.
            - y0: Place-field center y-coordinate.
            - sigma: Isotropic spatial width of the field.
            - amplitude: Peak response above baseline.
            - baseline: Baseline firing rate.

    Returns:
        np.ndarray: Predicted firing rate for each trial, shape (n_trials,).
    """
    x = data["pos_x"]
    y = data["pos_y"]

    x0 = np.clip(params["x0"], -1.0, 1.0)
    y0 = np.clip(params["y0"], -1.0, 1.0)
    sigma = np.clip(params["sigma"], 0.05, 1.0)
    amplitude = np.clip(params["amplitude"], 0.0, 50.0)
    baseline = np.clip(params["baseline"], 0.0, 20.0)

    dx = x - x0
    dy = y - y0
    dist2 = dx * dx + dy * dy
    return baseline + amplitude * np.exp(-0.5 * dist2 / (sigma ** 2))


model_v1.DEFAULT_PARAMS = {
    "x0": 0.0,
    "y0": 0.0,
    "sigma": 0.25,
    "amplitude": 1.0,
    "baseline": 0.0,
}


def param_est_v1(data):
    """
    Estimate parameters for `model_v1` from observed responses.

    The estimator uses weighted moments of position with non-negative response
    weights to recover center and width, with percentile-based baseline.

    Args:
        data (dict): Single-sample data dictionary with keys:
            - 'pos_x': x-position array of shape (n_trials,).
            - 'pos_y': y-position array of shape (n_trials,).
            - 'response': observed firing rates of shape (n_trials,).

    Returns:
        dict: Estimated parameters with keys
              {"x0", "y0", "sigma", "amplitude", "baseline"}.
    """
    x = data["pos_x"]
    y = data["pos_y"]
    firing_rates = np.asarray(data["response"])

    baseline = np.percentile(firing_rates, 10)
    weights = np.clip(firing_rates - baseline, 0.0, None)
    wsum = np.sum(weights) + 1e-8

    x0 = np.sum(x * weights) / wsum
    y0 = np.sum(y * weights) / wsum

    dx = x - x0
    dy = y - y0
    var = np.sum(weights * (dx * dx + dy * dy)) / wsum
    sigma = np.sqrt(np.clip(var / 2.0, 1e-6, None))

    amplitude = np.max(firing_rates) - baseline
    return {
        "x0": float(x0),
        "y0": float(y0),
        "sigma": float(sigma),
        "amplitude": float(amplitude),
        "baseline": float(baseline),
    }


def model_v2(data, params):
    """
    Independent variable:
    data['pos_x'], data['pos_y']  # 2D position (normalized approximately to [-1, 1])

    A place-cell model with an elliptical, rotated Gaussian field:
    r(x, y) = baseline + amplitude * exp(-0.5 * (xr^2/sigma_x^2 + yr^2/sigma_y^2))
    where (xr, yr) are coordinates rotated by angle theta about (x0, y0).

    Args:
        data (dict): Single-sample data dictionary with keys:
            - 'pos_x': x-position array of shape (n_trials,).
            - 'pos_y': y-position array of shape (n_trials,).
        params (dict): Parameter dictionary with keys:
            - x0: Place-field center x-coordinate.
            - y0: Place-field center y-coordinate.
            - sigma_x: Width along major/minor rotated x-axis.
            - sigma_y: Width along major/minor rotated y-axis.
            - theta: Field orientation (radians).
            - amplitude: Peak response above baseline.
            - baseline: Baseline firing rate.

    Returns:
        np.ndarray: Predicted firing rate for each trial, shape (n_trials,).
    """
    x = data["pos_x"]
    y = data["pos_y"]

    x0 = np.clip(params["x0"], -1.0, 1.0)
    y0 = np.clip(params["y0"], -1.0, 1.0)
    sigma_x = np.clip(params["sigma_x"], 0.05, 1.0)
    sigma_y = np.clip(params["sigma_y"], 0.05, 1.0)
    theta = np.clip(params["theta"], 0.0, np.pi)
    amplitude = np.clip(params["amplitude"], 0.0, 50.0)
    baseline = np.clip(params["baseline"], 0.0, 20.0)

    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    dx = x - x0
    dy = y - y0
    xr = cos_t * dx + sin_t * dy
    yr = -sin_t * dx + cos_t * dy

    dist2 = (xr * xr) / (sigma_x ** 2) + (yr * yr) / (sigma_y ** 2)
    return baseline + amplitude * np.exp(-0.5 * dist2)


model_v2.DEFAULT_PARAMS = {
    "x0": 0.0,
    "y0": 0.0,
    "sigma_x": 0.3,
    "sigma_y": 0.2,
    "theta": 0.0,
    "amplitude": 1.0,
    "baseline": 0.0,
}


def param_est_v2(data):
    """
    Estimate parameters for `model_v2` from observed responses.

    Uses weighted covariance of positions to infer ellipse axes and orientation.

    Args:
        data (dict): Single-sample data dictionary with keys:
            - 'pos_x': x-position array of shape (n_trials,).
            - 'pos_y': y-position array of shape (n_trials,).
            - 'response': observed firing rates of shape (n_trials,).

    Returns:
        dict: Estimated parameters with keys
              {"x0", "y0", "sigma_x", "sigma_y", "theta", "amplitude", "baseline"}.
    """
    x = data["pos_x"]
    y = data["pos_y"]
    firing_rates = np.asarray(data["response"])

    baseline = np.percentile(firing_rates, 10)
    weights = np.clip(firing_rates - baseline, 0.0, None)
    wsum = np.sum(weights) + 1e-8

    x0 = np.sum(x * weights) / wsum
    y0 = np.sum(y * weights) / wsum

    dx = x - x0
    dy = y - y0
    cov_xx = np.sum(weights * dx * dx) / wsum
    cov_yy = np.sum(weights * dy * dy) / wsum
    cov_xy = np.sum(weights * dx * dy) / wsum

    cov = np.array([[cov_xx, cov_xy], [cov_xy, cov_yy]], dtype=float)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    sigma_x = np.sqrt(np.clip(eigvals[0], 1e-6, None))
    sigma_y = np.sqrt(np.clip(eigvals[1], 1e-6, None))
    theta = np.arctan2(eigvecs[1, 0], eigvecs[0, 0])

    amplitude = np.max(firing_rates) - baseline
    return {
        "x0": float(x0),
        "y0": float(y0),
        "sigma_x": float(sigma_x),
        "sigma_y": float(sigma_y),
        "theta": float(theta),
        "amplitude": float(amplitude),
        "baseline": float(baseline),
    }


# ========================
# 3. LOSS
# ========================

def loss_fn(model_output, data):
    """
    Elementwise squared-error loss.

    Args:
        model_output (np.ndarray): Predicted firing rates.
        data (dict): Data dictionary; the comparison target is data['response'].

    Returns:
        np.ndarray: Elementwise squared errors.
    """
    return (data["response"] - model_output) ** 2


# ========================
# 4. DIAGNOSTICS
# ========================

def plot_model_fits(
    data,
    programs_list,
    eval_grid,
    save_path="",
    labels=("model_v1", "model_v2"),
    # -- ALL SUBSEQUENT PARAMS MUST BE SPECIFIED IN THE CONFIG FILE ---
):
    """
    Plot observed and predicted place-cell rate maps for up to 9 random samples.

    Parameters
    ----------
    data : dict[str, np.ndarray]
        Data dictionary with keys 'pos_x', 'pos_y', 'response'.
        Each value has shape (n_samples, n_trials).
    programs_list : list[dict]
        List of model dictionaries. Each dictionary should contain:
        - 'model': callable model function with signature model(data, params),
        - 'params': batched parameter pytree,
        - optionally 'losses': per-sample losses.
    eval_grid : dict[str, np.ndarray]
        Evaluation grid dict with keys 'pos_x', 'pos_y'.
        Each value has shape (n_eval_points,).
    save_path : str
        Output path for the saved figure.
    labels : tuple[str, ...]
        Labels for each model.

    Returns
    -------
    None
    """
    if save_path == "":
        raise ValueError("Please provide a save_path for the plot")

    # Build 3D arrays for compatibility with existing plotting logic.
    # pos_x and pos_y -> stack into (n_samples, 2, n_trials) input array.
    pos_x = np.asarray(data["pos_x"])
    pos_y = np.asarray(data["pos_y"])
    response = np.asarray(data["response"])
    x_arr = np.stack([pos_x, pos_y], axis=1)  # (n_samples, 2, n_trials)
    y_arr = response[:, np.newaxis, :]          # (n_samples, 1, n_trials)

    pos_x_eval = np.asarray(eval_grid["pos_x"]).reshape(-1)
    pos_y_eval = np.asarray(eval_grid["pos_y"]).reshape(-1)

    if x_arr.shape[1] < 2:
        raise ValueError("Place-cell diagnostics require at least 2 input features: x and y")

    n_samples = x_arr.shape[0]
    n_show = min(9, n_samples)
    # Intentionally unseeded so displayed samples vary across calls/runs.
    rng = np.random.default_rng()
    show_idx = rng.choice(n_samples, size=n_show, replace=False)

    n_models = len(programs_list)
    fig, axes = plt.subplots(n_show, 1 + n_models, figsize=(4 * (1 + n_models), 3 * n_show))
    axes = np.atleast_2d(axes)

    params_by_model = [
        utils.broadcast_params(program["params"], n_samples)
        for program in programs_list
    ]

    for row, sample_idx in enumerate(show_idx):
        x = x_arr[sample_idx, 0]
        y = x_arr[sample_idx, 1]
        y_obs = y_arr[sample_idx, 0]
        n_bins = len(pos_x_eval)
        x_domain = (float(np.min(pos_x_eval)), float(np.max(pos_x_eval)))
        y_domain = (float(np.min(pos_y_eval)), float(np.max(pos_y_eval)))

        rm_obs = _bin_to_rate_map(
            x, y, y_obs, n_bins=n_bins, x_domain=x_domain, y_domain=y_domain
        )
        ax = axes[row, 0]
        im = ax.imshow(
            rm_obs.T,
            origin="lower",
            extent=[x_domain[0], x_domain[1], y_domain[0], y_domain[1]],
            cmap="viridis",
        )
        ax.set_title(f"Sample {sample_idx} data")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        for m_idx, program in enumerate(programs_list):
            model = program["model"]
            params = utils.slice_params(params_by_model[m_idx], sample_idx)
            # Build single-sample data dict for model call
            sample_data = {
                "pos_x": x_arr[sample_idx, 0],
                "pos_y": x_arr[sample_idx, 1],
            }
            y_pred = utils.call_model(model, sample_data, params)
            rm_pred = _bin_to_rate_map(
                x, y, y_pred, n_bins=n_bins, x_domain=x_domain, y_domain=y_domain
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
                label += f", loss={program['losses'][sample_idx]:.2f}"
            axm.set_title(label)
            fig.colorbar(imm, ax=axm, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ========================
# 5. OPTIONAL PROJECT-SPECIFIC HELPERS
# ========================

def _load_data_file(data_path: str) -> Dict[str, Any]:
    data_loaded = np.load(data_path, allow_pickle=True)
    if isinstance(data_loaded, np.lib.npyio.NpzFile):
        data_dict = {k: data_loaded[k] for k in data_loaded.files}
        data_loaded.close()
        return data_dict
    if isinstance(data_loaded, np.ndarray) and data_loaded.shape == ():
        item = data_loaded.item()
        if isinstance(item, dict):
            return item
    if isinstance(data_loaded, dict):
        return data_loaded
    raise ValueError("Unsupported data format. Expected .npz or .npy containing a dict.")


def _find_key(data_dict: Dict[str, Any], preferred: Optional[str], candidates: List[str]) -> str:
    if preferred and preferred in data_dict:
        return preferred
    for key in candidates:
        if key in data_dict:
            return key
    raise KeyError(f"None of {candidates} found. Available keys: {list(data_dict.keys())}")


def _to_spike_dict(spike_obj: Any) -> Dict[int, np.ndarray]:
    if isinstance(spike_obj, dict):
        return spike_obj
    if isinstance(spike_obj, (list, tuple)):
        return {i: np.asarray(spikes) for i, spikes in enumerate(spike_obj)}
    if isinstance(spike_obj, np.ndarray):
        if spike_obj.dtype == object:
            if spike_obj.shape == ():
                item = spike_obj.item()
                if isinstance(item, dict):
                    return item
                if isinstance(item, (list, tuple)):
                    return {i: np.asarray(spikes) for i, spikes in enumerate(item)}
            return {i: np.asarray(spikes) for i, spikes in enumerate(spike_obj)}
        if spike_obj.ndim == 2:
            return {i: spike_obj[i] for i in range(spike_obj.shape[0])}
    raise ValueError("Could not interpret spike data.")


def _place_cell_filter_indices(
    response: np.ndarray,
    rate_maps: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    min_spatial_info: float,
    min_peak_rate: float,
    min_mean_rate: float,
    verbose: bool = True,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    n_cells = response.shape[0]
    n_bins = rate_maps.shape[1]

    occupancy, _, _ = np.histogram2d(x, y, bins=n_bins, range=[[-1, 1], [-1, 1]])
    p = occupancy / (np.sum(occupancy) + 1e-6)

    spatial_info = np.zeros(n_cells)
    peak_rate = np.zeros(n_cells)
    mean_rate = response.mean(axis=1)

    for c in range(n_cells):
        r = rate_maps[c]
        r_bar = np.sum(p * r)
        if r_bar <= 1e-8:
            spatial_info[c] = 0.0
        else:
            ratio = r / (r_bar + 1e-8)
            spatial_info[c] = np.nansum(p * ratio * np.log2(ratio + 1e-12))
        peak_rate[c] = np.nanmax(r)

    keep = (spatial_info >= min_spatial_info) & (peak_rate >= min_peak_rate) & (mean_rate >= min_mean_rate)
    indices = np.where(keep)[0]

    if verbose:
        print(
            f"Place cell filter: spatial_info>={min_spatial_info}, peak_rate>={min_peak_rate}, "
            f"mean_rate>={min_mean_rate} -> {len(indices)}/{n_cells} cells"
        )

    info = {
        "spatial_info": spatial_info,
        "peak_rate": peak_rate,
        "mean_rate": mean_rate,
        "place_cell_indices": indices,
    }
    return indices, info


def _bin_to_rate_map(
    x: np.ndarray,
    y: np.ndarray,
    values: np.ndarray,
    n_bins: int = 50,
    x_domain: Tuple[float, float] = (-1.0, 1.0),
    y_domain: Tuple[float, float] = (-1.0, 1.0),
) -> np.ndarray:
    edges_x = np.linspace(x_domain[0], x_domain[1], n_bins + 1)
    edges_y = np.linspace(y_domain[0], y_domain[1], n_bins + 1)
    occ, _, _ = np.histogram2d(x, y, bins=[edges_x, edges_y])
    weighted, _, _ = np.histogram2d(x, y, bins=[edges_x, edges_y], weights=values)
    return weighted / (occ + 1e-8)
