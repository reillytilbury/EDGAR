from __future__ import annotations

import numpy as np
import jax.numpy as jnp
from scipy.ndimage import gaussian_filter
from typing import Any, Dict, List, Optional, Tuple


def load_data(
    data_path: str,
    time_bin_ms: int = 20,
    speed_threshold: float = 2.5,
    min_spikes: int = 50,
    max_trials: Optional[int] = 8000,
    zscore_response: bool = True,
    spatial_bin_cm: float = 3.0,
    smoothing_sigma: float = 1.5,
    wall_val: float = 0.75,
    min_spatial_info: float = 0.3,
    min_peak_rate: float = 1.0,
    min_mean_rate: float = 0.05,
    input_names: Optional[List[str]] = None,
    time_start: Optional[float] = None,
    time_end: Optional[float] = None,
    random_seed: int = 42,
    n_eval_trials: int = 100,
) -> Tuple[Tuple[Dict, Dict], Tuple[Dict, Dict], Dict]:
    """
    Load and preprocess place-cell data.

    Returns
    -------
    X_discover, X_validate, X_eval
        X_discover = (train, test) dicts split by trials for use in the LLM loop.
        X_validate = (train, test) dicts held out for final evaluation.
        X_eval = small fixed trial subset from X_discover for fingerprinting.
    """
    if input_names is None:
        input_names = ["x", "y"]

    data_dict = _load_data_file(data_path)

    time_key = _find_key(data_dict, None, ["t", "time", "timestamps", "pos_t", "position_t"])
    t_raw = np.asarray(data_dict[time_key], dtype=float)
    t_pos = t_raw

    spike_key = _find_key(data_dict, None,
                          ["spikes", "spike_times", "spikes_mod1", "spikes_mod2", "spike_times_dict"])
    spike_times_dict = _to_spike_dict(data_dict[spike_key])
    n_neurons_raw = len(spike_times_dict)

    features_raw: Dict[str, np.ndarray] = {}
    for feat in input_names:
        if feat not in data_dict:
            raise KeyError(f"Input '{feat}' not found. Available: {list(data_dict.keys())}")
        features_raw[feat] = np.asarray(data_dict[feat], dtype=float)

    if time_start is None:
        time_start = float(np.nanmin(t_pos))
    if time_end is None:
        time_end = float(np.nanmax(t_pos))

    n_spatial_bins = int(np.ceil(2 * wall_val * 100 / spatial_bin_cm))

    time_mask = (t_pos >= time_start) & (t_pos < time_end)
    t_pos = t_pos[time_mask]
    for name in list(features_raw.keys()):
        features_raw[name] = features_raw[name][time_mask]

    time_bin_s = time_bin_ms / 1000.0
    n_time_bins = int(np.ceil((time_end - time_start) / time_bin_s))
    bin_edges = np.linspace(time_start, time_end, n_time_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    firing_rates = np.zeros((n_neurons_raw, n_time_bins), dtype=float)
    total_spikes = np.zeros(n_neurons_raw, dtype=float)
    for neuron_idx, (_, spike_times) in enumerate(spike_times_dict.items()):
        spikes = np.asarray(spike_times, dtype=float)
        spikes_in_window = spikes[(spikes >= time_start) & (spikes < time_end)]
        total_spikes[neuron_idx] = len(spikes_in_window)
        spike_counts, _ = np.histogram(spikes_in_window, bins=bin_edges)
        firing_rates[neuron_idx] = spike_counts / time_bin_s

    bin_idx = np.digitize(t_pos, bin_edges) - 1
    bin_idx = np.clip(bin_idx, 0, n_time_bins - 1)
    counts = np.bincount(bin_idx, minlength=n_time_bins)

    features: Dict[str, np.ndarray] = {}
    for feat_name, feat_raw in features_raw.items():
        sums = np.bincount(bin_idx, weights=feat_raw, minlength=n_time_bins)
        with np.errstate(invalid="ignore"):
            features[feat_name] = np.where(counts > 0, sums / counts, np.nan)

    for axis in ("x", "y"):
        if axis in features and np.nanmax(np.abs(features[axis])) > 1.5:
            features[axis] = features[axis] / max(wall_val, 1e-6)

    valid = np.ones(n_time_bins, dtype=bool)
    for arr in features.values():
        valid &= ~np.isnan(arr)
    if not np.all(valid):
        firing_rates = firing_rates[:, valid]
        bin_centers = bin_centers[valid]
        features = {name: arr[valid] for name, arr in features.items()}
        n_time_bins = len(bin_centers)

    if "x" in features and "y" in features and speed_threshold > 0:
        arena_half_width_cm = wall_val * 100
        dx = np.diff(features["x"], prepend=features["x"][0]) * arena_half_width_cm
        dy = np.diff(features["y"], prepend=features["y"][0]) * arena_half_width_cm
        speed = np.sqrt(dx * dx + dy * dy) / time_bin_s
        speed[0] = speed[1] if len(speed) > 1 else 0.0
        keep = speed >= speed_threshold
        firing_rates = firing_rates[:, keep]
        features = {name: arr[keep] for name, arr in features.items()}
        n_time_bins = firing_rates.shape[1]

    if max_trials is not None and n_time_bins > max_trials:
        keep_idx = np.linspace(0, n_time_bins - 1, max_trials).astype(int)
        firing_rates = firing_rates[:, keep_idx]
        features = {name: arr[keep_idx] for name, arr in features.items()}
        n_time_bins = max_trials

    firing_rates = firing_rates[total_spikes >= min_spikes]
    n_cells = firing_rates.shape[0]

    x = features["x"]
    y = features["y"]
    occupancy, _, _ = np.histogram2d(x, y, bins=n_spatial_bins, range=[[-1, 1], [-1, 1]])
    occupancy_s = gaussian_filter(occupancy, sigma=smoothing_sigma)
    bin_x = np.clip(((x + 1.0) / 2.0 * n_spatial_bins).astype(int), 0, n_spatial_bins - 1)
    bin_y = np.clip(((y + 1.0) / 2.0 * n_spatial_bins).astype(int), 0, n_spatial_bins - 1)
    rate_maps = np.zeros((n_cells, n_spatial_bins, n_spatial_bins), dtype=float)
    for c in range(n_cells):
        spike_map = np.zeros((n_spatial_bins, n_spatial_bins), dtype=float)
        for t in range(n_time_bins):
            spike_map[bin_x[t], bin_y[t]] += firing_rates[c, t]
        spike_map_s = gaussian_filter(spike_map, sigma=smoothing_sigma)
        rate_maps[c] = spike_map_s / (occupancy_s + 1e-6)

    keep_idx, _ = _place_cell_filter_indices(
        response=firing_rates, rate_maps=rate_maps, x=x, y=y,
        min_spatial_info=min_spatial_info, min_peak_rate=min_peak_rate,
        min_mean_rate=min_mean_rate, verbose=True,
    )
    if len(keep_idx) > 0:
        firing_rates = firing_rates[keep_idx]

    n_cells = firing_rates.shape[0]
    response = firing_rates
    if zscore_response:
        response = (response - response.mean(axis=1, keepdims=True)) / (response.std(axis=1, keepdims=True) + 1e-6)

    X = {
        "pos_x": np.tile(x, (n_cells, 1)),
        "pos_y": np.tile(y, (n_cells, 1)),
        "response": response,
    }

    return _split(X, random_seed, n_eval_trials)


def loss_fn(model_output, data):
    return jnp.mean((data["response"] - model_output) ** 2)


# ── internal helpers ──

def _split(
    X: Dict[str, np.ndarray],
    seed: int,
    n_eval_trials: int,
) -> Tuple[Tuple[Dict, Dict], Tuple[Dict, Dict], Dict]:
    n_samples = next(iter(X.values())).shape[0]
    n_trials = next(iter(X.values())).shape[-1]
    rng = np.random.default_rng(seed)

    perm_s = rng.permutation(n_samples)
    disc_idx = np.sort(perm_s[:n_samples // 2])
    val_idx = np.sort(perm_s[n_samples // 2:])

    perm_t = rng.permutation(n_trials)
    train_trials = np.sort(perm_t[:n_trials // 2])
    test_trials = np.sort(perm_t[n_trials // 2:])

    def _sel(sidx, tidx):
        return {k: v[sidx][..., tidx] for k, v in X.items()}

    X_disc_train = _sel(disc_idx, train_trials)
    X_disc_test = _sel(disc_idx, test_trials)
    X_val_train = _sel(val_idx, train_trials)
    X_val_test = _sel(val_idx, test_trials)

    n_eval = min(n_eval_trials, len(train_trials))
    eval_trials = np.sort(rng.choice(train_trials, n_eval, replace=False))
    X_eval = _sel(disc_idx, eval_trials)

    return (X_disc_train, X_disc_test), (X_val_train, X_val_test), X_eval


def _load_data_file(data_path: str) -> Dict[str, Any]:
    data_loaded = np.load(data_path, allow_pickle=True)
    if isinstance(data_loaded, np.lib.npyio.NpzFile):
        result = {k: data_loaded[k] for k in data_loaded.files}
        data_loaded.close()
        return result
    if isinstance(data_loaded, np.ndarray) and data_loaded.shape == ():
        item = data_loaded.item()
        if isinstance(item, dict):
            return item
    if isinstance(data_loaded, dict):
        return data_loaded
    raise ValueError("Unsupported data format.")


def _find_key(data_dict: Dict[str, Any], preferred: Optional[str], candidates: List[str]) -> str:
    if preferred and preferred in data_dict:
        return preferred
    for key in candidates:
        if key in data_dict:
            return key
    raise KeyError(f"None of {candidates} found. Available: {list(data_dict.keys())}")


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
        print(f"Place cell filter: {len(indices)}/{n_cells} cells retained")

    return indices, {"spatial_info": spatial_info, "peak_rate": peak_rate, "mean_rate": mean_rate}
