from __future__ import annotations

import numpy as np
import jax.numpy as jnp
from scipy.ndimage import gaussian_filter


def _to_jax(d):
    return {k: jnp.array(v) if k != "_sample_indices" else v for k, v in d.items()}


def load_data(
    data_path: str,
    time_start: float = 27826,
    time_end: float = 31223,
    time_bin_ms: int = 100,
    min_spikes: int = 200,
    speed_threshold: float = 2.5,
    max_trials: int = 20000,
    min_active_frac: float = 0.02,
    min_modulation: float = 1.0,
    min_spatial_reliability: float = 0.5,
    normalize_per_sample: bool = True,
    target_l2_norm: float = 100.0,
    min_l2_norm: float = 1.0e-06,
    n_trial_blocks: int = 10,
    random_seed: int = 42,
    spatial_bin_cm: float = 3.0,
    smoothing_sigma: float = 1.5,
    wall_val: float = 0.75,
    module_key: str = "spikes_mod1",
    input_names: list | None = None,
):
    """
    Load and preprocess grid-cell data.

    Returns
    -------
    X_discover, X_validate, X_eval
        Samples split 50/50, trials split by temporal blocks within discover/validate.

        X_discover = (train, test)
            X_disc_train: (n_cells//2, n_train_trials) - used by LLM loop.
            X_disc_test:  (n_cells//2, n_test_trials)  - test within discovery phase.

        X_validate = (train, test)
            X_val_train: (n_cells//2, n_train_trials) - held out, unseen by LLM.
            X_val_test:  (n_cells//2, n_test_trials)  - held out final evaluation.

        X_eval
            Single-cell fingerprint subset from discover train trials.
            Shape: (1, n_train_trials).
    """
    if input_names is None:
        input_names = ["x", "y"]

    data = np.load(data_path, allow_pickle=True)
    t_raw = np.asarray(data["t"], dtype=float)

    features_raw = {}
    for feat_name in input_names:
        if feat_name not in data.files:
            raise KeyError(f"Feature '{feat_name}' not found.")
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

    bin_indices = np.clip(np.digitize(t_raw, bin_edges) - 1, 0, n_time_bins - 1)
    counts_per_bin = np.bincount(bin_indices, minlength=n_time_bins)

    features = {}
    for feat_name, feat_raw in features_raw.items():
        sums = np.bincount(bin_indices, weights=feat_raw, minlength=n_time_bins)
        with np.errstate(invalid="ignore"):
            features[feat_name] = np.where(counts_per_bin > 0, sums / counts_per_bin, np.nan)

    arena_half_width_cm = wall_val * 100
    dx = np.diff(features["x"], prepend=features["x"][0]) * arena_half_width_cm
    dy = np.diff(features["y"], prepend=features["y"][0]) * arena_half_width_cm
    speed = np.sqrt(dx**2 + dy**2) / time_bin_s
    speed[0] = speed[1] if len(speed) > 1 else 0
    keep_speed = speed >= speed_threshold
    firing_rates = firing_rates[:, keep_speed]
    features = {name: arr[keep_speed] for name, arr in features.items()}

    active_frac = np.mean(firing_rates > 0, axis=1)
    modulation = np.percentile(firing_rates, 95, axis=1) - np.percentile(firing_rates, 5, axis=1)
    good_neurons = (
        (total_spikes_per_neuron >= min_spikes)
        & (active_frac >= min_active_frac)
        & (modulation >= min_modulation)
    )
    firing_rates = firing_rates[good_neurons]
    n_cells = firing_rates.shape[0]

    features["x"] = features["x"] / wall_val
    features["y"] = features["y"] / wall_val

    if max_trials is not None and firing_rates.shape[1] > max_trials:
        keep_idx = np.linspace(0, firing_rates.shape[1] - 1, max_trials).astype(int)
        firing_rates = firing_rates[:, keep_idx]
        features = {name: arr[keep_idx] for name, arr in features.items()}

    if firing_rates.shape[1] >= 4:
        n_trials = firing_rates.shape[1]
        half = n_trials // 2
        n_spatial_bins = int(np.ceil((2 * wall_val * 100) / spatial_bin_cm))
        rm_a = _compute_rate_maps(features["x"][:half], features["y"][:half],
                                  firing_rates[:, :half], n_spatial_bins, smoothing_sigma)
        rm_b = _compute_rate_maps(features["x"][half:], features["y"][half:],
                                  firing_rates[:, half:], n_spatial_bins, smoothing_sigma)
        a_flat = rm_a.reshape(rm_a.shape[0], -1)
        b_flat = rm_b.reshape(rm_b.shape[0], -1)
        a_mean = a_flat.mean(axis=1, keepdims=True)
        b_mean = b_flat.mean(axis=1, keepdims=True)
        denom = np.linalg.norm(a_flat - a_mean, axis=1) * np.linalg.norm(b_flat - b_mean, axis=1)
        corr = np.where(denom > 0,
                        np.sum((a_flat - a_mean) * (b_flat - b_mean), axis=1) / denom, 0.0)
        firing_rates = firing_rates[corr >= min_spatial_reliability]
        n_cells = firing_rates.shape[0]

    if normalize_per_sample and n_cells > 0:
        l2 = np.linalg.norm(firing_rates, axis=1, keepdims=True)
        scale = np.maximum(l2, float(min_l2_norm))
        firing_rates = firing_rates * (float(target_l2_norm) / scale)

    pos_x = np.tile(features["x"], (n_cells, 1))
    pos_y = np.tile(features["y"], (n_cells, 1))
    n_samples, n_trials = firing_rates.shape

    # ── split samples 50/50 into discover / validate ──
    rng = np.random.default_rng(random_seed)

    perm_s   = rng.permutation(n_samples)
    disc_idx = np.sort(perm_s[:n_samples // 2])
    val_idx  = np.sort(perm_s[n_samples // 2:])

    # ── block-based trial split to preserve temporal structure ──
    block_size    = max(1, n_trials // n_trial_blocks)
    blocks        = [np.arange(start, min(start + block_size, n_trials))
                     for start in range(0, n_trials, block_size)]
    blocks_arr    = np.array(blocks, dtype=object)
    perm_b        = rng.permutation(len(blocks_arr))
    n_train_blocks = len(blocks_arr) // 2
    train_trials  = np.sort(np.concatenate(blocks_arr[perm_b[:n_train_blocks]]).astype(int))
    test_trials   = np.sort(np.concatenate(blocks_arr[perm_b[n_train_blocks:]]).astype(int))

    X_disc_train = {'pos_x': pos_x[disc_idx][:, train_trials], 'pos_y': pos_y[disc_idx][:, train_trials], 'response': firing_rates[disc_idx][:, train_trials]}
    X_disc_test  = {'pos_x': pos_x[disc_idx][:, test_trials],  'pos_y': pos_y[disc_idx][:, test_trials],  'response': firing_rates[disc_idx][:, test_trials]}
    X_val_train  = {'pos_x': pos_x[val_idx][:, train_trials],  'pos_y': pos_y[val_idx][:, train_trials],  'response': firing_rates[val_idx][:, train_trials]}
    X_val_test   = {'pos_x': pos_x[val_idx][:, test_trials],   'pos_y': pos_y[val_idx][:, test_trials],   'response': firing_rates[val_idx][:, test_trials]}

    # ── build X_eval: single cell from discover train for fingerprinting ──
    eval_pos     = rng.choice(len(disc_idx), 1, replace=False)
    X_eval = {
        'pos_x':            pos_x[disc_idx][eval_pos][:, train_trials],
        'pos_y':            pos_y[disc_idx][eval_pos][:, train_trials],
        'response':         firing_rates[disc_idx][eval_pos][:, train_trials],
        '_sample_indices':  eval_pos,
    }

    return (_to_jax(X_disc_train), _to_jax(X_disc_test)), (_to_jax(X_val_train), _to_jax(X_val_test)), _to_jax(X_eval)


def loss_fn(model_output, data):
    return jnp.mean((data['response'] - model_output) ** 2, axis=-1)


# ── internal helpers ──

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
        rate_maps[c] = gaussian_filter(spike_map, sigma=smoothing_sigma) / (occupancy_s + 1e-6)
    return rate_maps
