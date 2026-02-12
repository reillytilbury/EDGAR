"""
Data parser for place cell experiments.

This module loads and processes place cell data from typical hippocampal
recordings where the animal's 2D position is tracked alongside spike times.
Place cells fire in localized spatial fields, so we compute spatial rate maps
and optionally filter cells using spatial information and peak rate criteria.
"""

import time
from typing import Dict, Any, Optional, List, Tuple

import numpy as np
import jax.numpy as jnp
from scipy.ndimage import gaussian_filter

from src.data_structures import Inputs, Outputs


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
    raise ValueError(
        "Unsupported data format. Expected .npz with named arrays or .npy containing a dict."
    )


def _find_key(data_dict: Dict[str, Any], preferred: Optional[str], candidates: List[str]) -> str:
    if preferred and preferred in data_dict:
        return preferred
    for key in candidates:
        if key in data_dict:
            return key
    raise KeyError(f"None of the candidate keys found. Candidates: {candidates}. Available: {list(data_dict.keys())}")


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
        # Fallback: treat as 2D array (n_cells, n_spikes) if rectangular
        if spike_obj.ndim == 2:
            return {i: spike_obj[i] for i in range(spike_obj.shape[0])}
    raise ValueError("Could not interpret spike data. Expected dict or list-like of spike times.")


def load_and_process_data(
    data_path: str,
    time_start: Optional[float] = None,
    time_end: Optional[float] = None,
    spatial_bin_cm: float = 3.0,
    time_bin_ms: int = 20,
    smoothing_sigma: float = 1.5,
    wall_val: float = 0.75,
    input_names: Optional[List[str]] = None,
    spike_key: Optional[str] = None,
    time_key: Optional[str] = None,
    position_time_key: Optional[str] = None,
    min_spikes: int = 50,
    speed_threshold: float = 2.5,
    max_trials: Optional[int] = 8000,
    filter_place_cells: bool = True,
    place_filter_kwargs: Optional[Dict[str, Any]] = None,
    positions_normalized: bool = True,
    position_unit: str = 'normalized',  # 'normalized', 'm', or 'cm'
    drop_nan_bins: bool = True,
    zscore_response: bool = True,
    zscore_rate_maps: bool = True,
    **kwargs,
) -> Dict[str, Any]:
    """
    Load and preprocess place cell data from a .npz or .npy file.

    Parameters
    ----------
    data_path : str
        Path to the data file.
    time_start, time_end : float or None
        Start/end time in seconds for data extraction. If None, uses full session.
    spatial_bin_cm : float
        Spatial bin size in cm for rate map computation.
    time_bin_ms : int
        Temporal bin size in ms for spike binning.
    smoothing_sigma : float
        Gaussian smoothing sigma (in bins) for rate maps. None to disable smoothing.
    wall_val : float
        Arena half-width in meters (0.75m for 1.5m x 1.5m arena).
    input_names : list of str, optional
        Names for input variables. Defaults to ['x', 'y'].
    spike_key : str, optional
        Key for spike times in the data file. If None, searches common keys.
    time_key : str, optional
        Key for the global time vector. If None, searches common keys.
    position_time_key : str, optional
        Key for the position time vector. If None, uses time_key.
    min_spikes : int
        Minimum number of spikes within the time window for a cell to be included.
    speed_threshold : float
        Minimum speed (cm/s) to include a time bin.
    max_trials : int or None
        If set, subsample time bins to this max size (evenly spaced) to avoid OOM.
    filter_place_cells : bool
        Whether to apply place-cell filtering by spatial information and peak rate.
    place_filter_kwargs : dict, optional
        Parameters for place_cell_filter.
    positions_normalized : bool
        Whether positions are already normalized to [-1, 1].
    position_unit : str
        Unit of positions if not normalized: 'm' or 'cm'.
    drop_nan_bins : bool
        Drop time bins with missing (NaN) position values.
    zscore_response : bool
        Z-score responses per cell.
    zscore_rate_maps : bool
        Z-score rate maps per cell.

    Returns
    -------
    data_dict : dict
        Dictionary containing:
          - 'response': (LEGACY) Firing rate array (n_cells, n_trials)
          - 'outputs': Outputs object (n_cells, 1, n_trials)
          - 'inputs': Inputs object (n_cells, n_features, n_trials)
          - 'trials': Deprecated alias for inputs.data
          - 'rate_maps': (n_cells, n_bins, n_bins)
          - 'position_data': dict with position/time metadata
          - 'place_filter_info': dict of filtering diagnostics (or None)
    """
    clock_time_start = time.time()

    if input_names is None:
        input_names = ['x', 'y']

    data_dict = _load_data_file(data_path)

    # Resolve time vector
    time_key = _find_key(
        data_dict,
        time_key,
        candidates=['t', 'time', 'timestamps', 'pos_t', 'position_t'],
    )
    t_raw = np.asarray(data_dict[time_key]).astype(float)

    # Resolve position time vector (can be separate)
    if position_time_key is None:
        position_time_key = time_key
    if position_time_key in data_dict:
        t_pos = np.asarray(data_dict[position_time_key]).astype(float)
    else:
        t_pos = t_raw

    # Load features
    features_raw: Dict[str, np.ndarray] = {}
    for feat_name in input_names:
        if feat_name in data_dict:
            features_raw[feat_name] = np.asarray(data_dict[feat_name]).astype(float)
        else:
            raise KeyError(
                f"Input '{feat_name}' requested but not available. Available keys: {list(data_dict.keys())}"
            )

    # Resolve spikes
    spike_key = _find_key(
        data_dict,
        spike_key,
        candidates=['spikes', 'spike_times', 'spikes_mod1', 'spikes_mod2', 'spike_times_dict'],
    )
    spike_times_dict = _to_spike_dict(data_dict[spike_key])
    n_neurons_raw = len(spike_times_dict)

    if time_start is None:
        time_start = float(np.nanmin(t_pos))
    if time_end is None:
        time_end = float(np.nanmax(t_pos))
    if time_end <= time_start:
        raise ValueError(f"time_end ({time_end}) must be greater than time_start ({time_start}).")

    arena_size_cm = 2 * wall_val * 100
    n_spatial_bins = int(np.ceil(arena_size_cm / spatial_bin_cm))
    print(
        f"{time.time() - clock_time_start:.3f}s : Spatial binning: "
        f"{n_spatial_bins}x{n_spatial_bins} bins of {spatial_bin_cm:.1f}cm for {arena_size_cm:.0f}cm arena"
    )

    # ---------------------------------------------------------------------
    # Step 1: Align to time window
    # ---------------------------------------------------------------------
    time_mask = (t_pos >= time_start) & (t_pos < time_end)
    t_pos = t_pos[time_mask]
    for name in list(features_raw.keys()):
        features_raw[name] = features_raw[name][time_mask]

    # ---------------------------------------------------------------------
    # Step 2: Define common time discretisation
    # ---------------------------------------------------------------------
    time_bin_s = time_bin_ms / 1000.0
    n_time_bins = int(np.ceil((time_end - time_start) / time_bin_s))
    bin_edges = np.linspace(time_start, time_end, n_time_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    print(
        f"{time.time() - clock_time_start:.3f}s : Time discretisation: "
        f"{n_time_bins} bins of {time_bin_ms} ms from {time_start}s to {time_end}s"
    )

    # ---------------------------------------------------------------------
    # Step 3: Convert spike times into binned firing rates
    # ---------------------------------------------------------------------
    firing_rates = np.zeros((n_neurons_raw, n_time_bins))
    total_spikes_per_neuron = np.zeros(n_neurons_raw)

    for neuron_idx, (neuron_id, spike_times) in enumerate(spike_times_dict.items()):
        spike_times = np.asarray(spike_times, dtype=float)
        spikes_in_window = spike_times[(spike_times >= time_start) & (spike_times < time_end)]
        total_spikes_per_neuron[neuron_idx] = len(spikes_in_window)
        spike_counts, _ = np.histogram(spikes_in_window, bins=bin_edges)
        firing_rates[neuron_idx] = spike_counts / time_bin_s

    print(f"{time.time() - clock_time_start:.3f}s : Spike binning complete")

    # ---------------------------------------------------------------------
    # Step 4: Bin behavioral features
    # ---------------------------------------------------------------------
    bin_indices = np.digitize(t_pos, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, n_time_bins - 1)
    counts_per_bin = np.bincount(bin_indices, minlength=n_time_bins)

    features: Dict[str, np.ndarray] = {}
    for feat_name, feat_raw in features_raw.items():
        sums_per_bin = np.bincount(bin_indices, weights=feat_raw, minlength=n_time_bins)
        with np.errstate(invalid='ignore'):
            features[feat_name] = np.where(counts_per_bin > 0, sums_per_bin / counts_per_bin, np.nan)

    print(f"{time.time() - clock_time_start:.3f}s : Binned behavioral features computed")

    # ---------------------------------------------------------------------
    # Step 5: Normalize positions if required
    # ---------------------------------------------------------------------
    if positions_normalized:
        # Assume features already in [-1, 1]
        pass
    else:
        if position_unit not in ('m', 'cm'):
            raise ValueError("position_unit must be 'm' or 'cm' when positions_normalized=False")
        scale = wall_val if position_unit == 'm' else wall_val * 100.0
        for axis in ('x', 'y'):
            if axis in features:
                features[axis] = features[axis] / scale

    # ---------------------------------------------------------------------
    # Step 6: Drop NaN bins if requested
    # ---------------------------------------------------------------------
    if drop_nan_bins:
        valid_mask = np.ones(n_time_bins, dtype=bool)
        for arr in features.values():
            valid_mask &= ~np.isnan(arr)
        if not np.all(valid_mask):
            firing_rates = firing_rates[:, valid_mask]
            bin_centers = bin_centers[valid_mask]
            features = {name: arr[valid_mask] for name, arr in features.items()}
            n_time_bins = len(bin_centers)

    # ---------------------------------------------------------------------
    # Step 7: Exclude low-speed periods
    # ---------------------------------------------------------------------
    speed = None
    if 'x' in features and 'y' in features and speed_threshold > 0:
        arena_half_width_cm = wall_val * 100
        dx = np.diff(features['x'], prepend=features['x'][0]) * arena_half_width_cm
        dy = np.diff(features['y'], prepend=features['y'][0]) * arena_half_width_cm
        speed = np.sqrt(dx**2 + dy**2) / time_bin_s
        speed[0] = speed[1] if len(speed) > 1 else 0
        speed_mask = speed >= speed_threshold
        n_excluded = np.sum(~speed_mask)
        print(f"Speed filtering: excluding {n_excluded}/{n_time_bins} bins below {speed_threshold} cm/s")
        firing_rates = firing_rates[:, speed_mask]
        bin_centers = bin_centers[speed_mask]
        features = {name: arr[speed_mask] for name, arr in features.items()}
        if speed is not None:
            speed = speed[speed_mask]
        n_time_bins = len(bin_centers)

    # ---------------------------------------------------------------------
    # Step 8: Subsample time bins if requested
    # ---------------------------------------------------------------------
    if max_trials is not None and n_time_bins > max_trials:
        keep_idx = np.linspace(0, n_time_bins - 1, max_trials).astype(int)
        firing_rates = firing_rates[:, keep_idx]
        bin_centers = bin_centers[keep_idx]
        features = {name: arr[keep_idx] for name, arr in features.items()}
        if speed is not None:
            speed = speed[keep_idx]
        n_time_bins = len(bin_centers)
        print(f"Subsampled to {n_time_bins} time bins for memory safety")

    # ---------------------------------------------------------------------
    # Step 9: Filter neurons by minimum spike count
    # ---------------------------------------------------------------------
    good_neurons = total_spikes_per_neuron >= min_spikes
    firing_rates = firing_rates[good_neurons]
    n_cells = firing_rates.shape[0]
    print(
        f"{time.time() - clock_time_start:.3f}s : Loaded {n_cells} neurons "
        f"(from {n_neurons_raw} total) with >= {min_spikes} spikes"
    )

    # ---------------------------------------------------------------------
    # Step 10: Build Inputs object
    # ---------------------------------------------------------------------
    input_arrays = []
    for name in input_names:
        if name not in features:
            raise ValueError(f"Input '{name}' requested but not available. Available: {list(features.keys())}")
        input_arrays.append(np.tile(features[name], (n_cells, 1)))
    inputs_data = np.stack(input_arrays, axis=1)
    inputs = Inputs(data=inputs_data, names=input_names)

    # ---------------------------------------------------------------------
    # Step 11: Compute rate maps
    # ---------------------------------------------------------------------
    x_norm = features['x']
    y_norm = features['y']

    occupancy, x_edges, y_edges = np.histogram2d(
        x_norm, y_norm, bins=n_spatial_bins, range=[[-1, 1], [-1, 1]]
    )

    if smoothing_sigma is not None:
        occupancy_smooth = gaussian_filter(occupancy, sigma=smoothing_sigma)
    else:
        occupancy_smooth = occupancy

    rate_maps = np.zeros((n_cells, n_spatial_bins, n_spatial_bins))
    bin_x = np.clip(((x_norm + 1) / 2 * n_spatial_bins).astype(int), 0, n_spatial_bins - 1)
    bin_y = np.clip(((y_norm + 1) / 2 * n_spatial_bins).astype(int), 0, n_spatial_bins - 1)

    for c in range(n_cells):
        spike_map = np.zeros((n_spatial_bins, n_spatial_bins))
        for t_idx in range(n_time_bins):
            spike_map[bin_x[t_idx], bin_y[t_idx]] += firing_rates[c, t_idx]
        if smoothing_sigma is None:
            rate_maps[c] = spike_map / (occupancy + 1e-6)
        else:
            spike_map_smooth = gaussian_filter(spike_map, sigma=smoothing_sigma)
            rate_maps[c] = spike_map_smooth / (occupancy_smooth + 1e-6)

    # ---------------------------------------------------------------------
    # Step 12: Optional place cell filtering
    # ---------------------------------------------------------------------
    place_filter_info = None
    if filter_place_cells:
        filter_kwargs = place_filter_kwargs or {}
        filter_kwargs.setdefault('wall_val', wall_val)
        filter_kwargs.setdefault('time_bin_ms', time_bin_ms)

        position_data_for_filter = {
            **features,
            't': bin_centers,
            'n_spatial_bins': n_spatial_bins,
            'time_bin_ms': time_bin_ms,
            'x_edges': x_edges,
            'y_edges': y_edges,
        }

        place_cell_indices, place_filter_info = place_cell_filter(
            response=firing_rates,
            inputs=inputs,
            rate_maps=rate_maps,
            position_data=position_data_for_filter,
            **filter_kwargs,
        )

        if len(place_cell_indices) > 0:
            firing_rates = firing_rates[place_cell_indices]
            rate_maps = rate_maps[place_cell_indices]
            inputs_data = inputs_data[place_cell_indices]
            inputs = Inputs(data=inputs_data, names=input_names)
            n_cells = len(place_cell_indices)
            print(f"Place cell filtering: kept {n_cells} place cells")
        else:
            print("Warning: No place cells identified, returning all cells")

    # ---------------------------------------------------------------------
    # Step 13: Normalize response and rate maps
    # ---------------------------------------------------------------------
    response = firing_rates
    if zscore_response:
        response = (response - response.mean(axis=1, keepdims=True)) / (
            response.std(axis=1, keepdims=True) + 1e-6
        )
    if zscore_rate_maps:
        rate_maps = (rate_maps - rate_maps.mean(axis=(1, 2), keepdims=True)) / (
            rate_maps.std(axis=(1, 2), keepdims=True) + 1e-6
        )

    outputs = Outputs.from_array(response, names=['firing_rate'])

    position_data = {
        **features,
        't': bin_centers,
        'n_spatial_bins': n_spatial_bins,
        'time_bin_ms': time_bin_ms,
        'x_edges': x_edges,
        'y_edges': y_edges,
    }
    if speed is not None:
        position_data['speed_cm_s'] = speed

    return {
        'response': response,
        'outputs': outputs,
        'inputs': inputs,
        'trials': inputs_data,
        'rate_maps': rate_maps,
        'position_data': position_data,
        'place_filter_info': place_filter_info,
        'smoothing_sigma': smoothing_sigma,
    }


def place_cell_filter(
    response: np.ndarray,
    inputs: Inputs,
    rate_maps: np.ndarray,
    position_data: Dict[str, np.ndarray],
    min_spatial_info: float = 0.3,
    min_peak_rate: float = 1.0,
    min_mean_rate: float = 0.05,
    wall_val: float = 0.75,
    time_bin_ms: int = 20,
    verbose: bool = True,
    **kwargs,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Place cell identification based on spatial information and firing statistics.

    Parameters
    ----------
    response : np.ndarray
        Firing rates (n_cells, n_time_bins).
    inputs : Inputs
        Inputs object with position data.
    rate_maps : np.ndarray
        Rate maps (n_cells, n_bins, n_bins).
    position_data : dict
        Position/time metadata.
    min_spatial_info : float
        Minimum spatial information (bits/spike).
    min_peak_rate : float
        Minimum peak firing rate (Hz) in the rate map.
    min_mean_rate : float
        Minimum mean firing rate (Hz).

    Returns
    -------
    place_cell_indices : np.ndarray
        Indices of cells passing the filters.
    filter_info : dict
        Diagnostics for filtering.
    """
    n_cells = response.shape[0]
    n_bins = rate_maps.shape[1]

    # Occupancy probability
    x = position_data.get('x')
    y = position_data.get('y')
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
            ratio = (r / (r_bar + 1e-8))
            spatial_info[c] = np.nansum(p * ratio * np.log2(ratio + 1e-12))
        peak_rate[c] = np.nanmax(r)

    spatial_mask = spatial_info >= min_spatial_info
    peak_mask = peak_rate >= min_peak_rate
    mean_mask = mean_rate >= min_mean_rate

    keep_mask = spatial_mask & peak_mask & mean_mask
    indices = np.where(keep_mask)[0]

    if verbose:
        print(
            f"Place cell filter: spatial_info>={min_spatial_info}, peak_rate>={min_peak_rate}, "
            f"mean_rate>={min_mean_rate} -> {len(indices)}/{n_cells} cells"
        )

    filter_info = {
        'spatial_info': spatial_info,
        'peak_rate': peak_rate,
        'mean_rate': mean_rate,
        'spatial_info_threshold': min_spatial_info,
        'peak_rate_threshold': min_peak_rate,
        'mean_rate_threshold': min_mean_rate,
        'spatial_mask': spatial_mask,
        'peak_mask': peak_mask,
        'mean_mask': mean_mask,
        'place_cell_indices': indices,
    }

    return indices, filter_info
