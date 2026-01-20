"""
Data parser for grid cell experiments.

This module loads and processes grid cell data from the Toroidal topology dataset.
Grid cells fire in hexagonal patterns as animals navigate through 2D environments.
"""

import numpy as np
import jax.numpy as jnp
from scipy.ndimage import gaussian_filter
from typing import Dict, Any, Optional, List, Tuple

from src.data_structures import Predictors


def load_and_process_data(
    data_path: str,
    time_start: float = 27826,
    time_end: float = 31223,
    n_bins: int = 65,
    smoothing_sigma: float = 1.5,
    wall_val: float = 0.75,
    predictor_names: Optional[List[str]] = None,
    module_key: str = 'spikes_mod1',
    min_spikes: int = 100,
    **kwargs  # Accept additional config params (e.g., task, predictors) without error
) -> Dict[str, Any]:
    """
    Load and preprocess grid cell data from .npz file.
    
    Parameters
    ----------
    data_path : str
        Path to the .npz file containing grid cell data.
    time_start : float
        Start time in seconds for data extraction.
    time_end : float
        End time in seconds for data extraction.
    n_bins : int
        Number of spatial bins per dimension for rate map computation.
    smoothing_sigma : float
        Gaussian smoothing sigma for rate maps.
    wall_val : float
        Arena half-width value for position normalization.
    predictor_names : list of str, optional
        Names for the predictor variables. Defaults to ['x', 'y'].
    module_key : str
        Key in the npz file for spike data (e.g., 'spikes_mod1', 'spikes_mod2').
    min_spikes : int
        Minimum number of spikes for a cell to be included.
    
    Returns
    -------
    data_dict : dict
        Dictionary containing:
          - 'response': Firing rates at each position. (n_cells, n_trials)
          - 'predictors': Predictors object with x, y positions. (n_cells, n_features, n_trials)
          - 'rate_maps': 2D rate maps for visualization. (n_cells, n_bins, n_bins)
          - 'position_data': Dict with raw x, y, t arrays for diagnostics.
    """
    if predictor_names is None:
        predictor_names = ['x', 'y']
    
    # Load data
    data = np.load(data_path, allow_pickle=True)
    x_raw, y_raw, t_raw = data['x'], data['y'], data['t']
    
    # Filter to time window
    time_mask = (t_raw >= time_start) & (t_raw <= time_end)
    x = x_raw[time_mask]
    y = y_raw[time_mask]
    t = t_raw[time_mask]
    
    # Process spike times
    spike_times_dict = data[module_key].item()
    
    # Convert time to integer indices (matching notebook approach)
    # Round spike times and t to 2 decimal places, then convert to int
    t_int = (np.round(t, 2) * 100).astype(int) - int(time_start * 100)
    
    spike_times_list = []
    for neuron, times in spike_times_dict.items():
        times_filtered = times[(times >= time_start) & (times <= time_end)]
        times_rounded = np.round(times_filtered, 2)
        times_int = (times_rounded * 100).astype(int) - int(time_start * 100)
        # Keep only spikes within the valid time range
        times_int = times_int[(times_int >= t_int[0]) & (times_int <= t_int[-1])]
        spike_times_list.append(times_int)
    
    # Normalize positions to [-1, 1]
    x_norm = x / wall_val
    y_norm = y / wall_val
    
    # Filter cells by minimum spike count
    n_cells_raw = len(spike_times_list)
    good_cells = [i for i, spikes in enumerate(spike_times_list) if len(spikes) >= min_spikes]
    spike_times_list = [spike_times_list[i] for i in good_cells]
    n_cells = len(spike_times_list)
    
    print(f"Loaded {n_cells} cells (from {n_cells_raw} total) with >= {min_spikes} spikes")
    print(f"Time window: {time_start}s to {time_end}s ({len(t)} timepoints)")
    
    # Compute occupancy map
    occupancy, x_edges, y_edges = np.histogram2d(
        x_norm, y_norm, bins=n_bins, range=[[-1, 1], [-1, 1]]
    )
    
    # Compute spike count maps and rate maps for each cell
    spike_maps = np.zeros((n_cells, n_bins, n_bins))
    for c, spikes in enumerate(spike_times_list):
        x_spikes = x_norm[spikes]
        y_spikes = y_norm[spikes]
        spike_maps[c], _, _ = np.histogram2d(
            x_spikes, y_spikes, bins=n_bins, range=[[-1, 1], [-1, 1]]
        )
    
    # Smooth and compute rate maps
    occupancy_smooth = gaussian_filter(occupancy, sigma=smoothing_sigma)
    spike_maps_smooth = np.array([
        gaussian_filter(spike_maps[c], sigma=smoothing_sigma) 
        for c in range(n_cells)
    ])
    rate_maps = spike_maps_smooth / (occupancy_smooth + 1e-6)
    
    # Create trial-based representation
    # Each timepoint is a "trial" with x, y predictors and firing rate response
    n_trials = len(t)
    
    # Build response matrix: for each cell, firing rate at each timepoint
    # Use the rate map to look up firing rate at each position
    bin_x = np.clip(((x_norm + 1) / 2 * n_bins).astype(int), 0, n_bins - 1)
    bin_y = np.clip(((y_norm + 1) / 2 * n_bins).astype(int), 0, n_bins - 1)
    
    response = np.zeros((n_cells, n_trials))
    for c in range(n_cells):
        response[c] = rate_maps[c, bin_x, bin_y]
    
    # Create predictors: x and y position at each timepoint
    # Shape: (n_cells, n_features, n_trials) where n_features = 2
    # Note: same x, y for all cells (broadcast)
    x_trials = np.tile(x_norm, (n_cells, 1))  # (n_cells, n_trials)
    y_trials = np.tile(y_norm, (n_cells, 1))  # (n_cells, n_trials)
    
    # Stack to create (n_cells, 2, n_trials)
    predictors_data = np.stack([x_trials, y_trials], axis=1)
    predictors = Predictors(data=predictors_data, names=predictor_names)
    
    return {
        "response": response,
        "predictors": predictors,
        "trials": predictors_data,  # For backward compatibility
        "rate_maps": rate_maps,
        "position_data": {
            "x": x_norm,
            "y": y_norm, 
            "t": t,
            "spike_times": spike_times_list,
            "n_bins": n_bins,
            "x_edges": x_edges,
            "y_edges": y_edges,
        }
    }


def compute_rate_map(
    x: np.ndarray,
    y: np.ndarray,
    spike_times: np.ndarray,
    n_bins: int = 65,
    sigma: float = 1.5,
    extent: Tuple[float, float, float, float] = (-1, 1, -1, 1),
) -> np.ndarray:
    """
    Compute a smoothed 2D rate map for a single cell.
    
    Parameters
    ----------
    x, y : np.ndarray
        Position coordinates (n_timepoints,).
    spike_times : np.ndarray
        Integer indices of spike times into x, y arrays.
    n_bins : int
        Number of spatial bins per dimension.
    sigma : float
        Gaussian smoothing sigma.
    extent : tuple
        (xmin, xmax, ymin, ymax) for histogram range.
    
    Returns
    -------
    rate_map : np.ndarray
        Smoothed rate map of shape (n_bins, n_bins).
    """
    xmin, xmax, ymin, ymax = extent
    
    # Occupancy
    occupancy, _, _ = np.histogram2d(
        x, y, bins=n_bins, range=[[xmin, xmax], [ymin, ymax]]
    )
    
    # Spike counts
    x_spikes = x[spike_times]
    y_spikes = y[spike_times]
    spike_counts, _, _ = np.histogram2d(
        x_spikes, y_spikes, bins=n_bins, range=[[xmin, xmax], [ymin, ymax]]
    )
    
    # Smooth and divide
    occupancy_smooth = gaussian_filter(occupancy, sigma=sigma)
    spike_counts_smooth = gaussian_filter(spike_counts, sigma=sigma)
    
    rate_map = spike_counts_smooth / (occupancy_smooth + 1e-6)
    return rate_map


def split_train_test(
    response: np.ndarray,
    predictors: Predictors,
    test_fraction: float = 0.2,
    random_seed: int = 42,
) -> Dict[str, Any]:
    """
    Split data into training and test sets.
    
    Parameters
    ----------
    response : np.ndarray
        Response matrix of shape (n_cells, n_trials).
    predictors : Predictors
        Predictors object with shape (n_cells, n_features, n_trials).
    test_fraction : float
        Fraction of trials to use for testing.
    random_seed : int
        Random seed for reproducibility.
    
    Returns
    -------
    split_dict : dict
        Dictionary with train/test splits.
    """
    np.random.seed(random_seed)
    n_trials = response.shape[1]
    n_test = int(n_trials * test_fraction)
    
    # Random permutation of trial indices
    perm = np.random.permutation(n_trials)
    test_idx = perm[:n_test]
    train_idx = perm[n_test:]
    
    return {
        "response_train": response[:, train_idx],
        "response_test": response[:, test_idx],
        "predictors_train": predictors.slice_trials(train_idx),
        "predictors_test": predictors.slice_trials(test_idx),
        "train_idx": train_idx,
        "test_idx": test_idx,
    }
