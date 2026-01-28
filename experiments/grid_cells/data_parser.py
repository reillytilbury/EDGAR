"""
Data parser for grid cell experiments.

This module loads and processes grid cell data from the Toroidal topology dataset.
Grid cells fire in hexagonal patterns as animals navigate through 2D environments.
"""

import numpy as np
import jax.numpy as jnp
from scipy.ndimage import gaussian_filter
from typing import Dict, Any, Optional, List, Tuple
import time 

from src.data_structures import Predictors

# How to align spike trains with behavioural time and position

# Goal:
# Convert irregular spike-time data from many neurons into a time-aligned population activity representation that can be analysed jointly with the animal’s position and behaviour.

# 1. Understand the raw data representations
# - Behavioural variables (x, y, z, azimuth) are sampled continuously over time and indexed by a common time vector t.
# - Neural activity (spikes_mod1, spikes_mod2) is stored as event times:
# - Each neuron has its own list of spike times.
# - Different neurons have different numbers of spikes because they fire at different rates.
# - There is no one-to-one correspondence between spikes and behavioural samples.

# 2. Define a common time discretisation
# Choose a fixed temporal resolution (e.g. 10 ms).
# Partition the full recording duration into consecutive, non-overlapping time bins.
# These bins define the shared time axis onto which all neurons and behavioural variables will be aligned.

# 3. Convert spike times into binned firing rates
# For each neuron:
# Count how many spikes fall into each time bin.
# This produces:
# - One firing-rate time series per neuron,
# - All defined on the same time grid.
# Stack these time series to form a population activity matrix:
# - Rows (or columns): neurons,
# - Columns (or rows): time bins.

# 4. Align behavioural variables to the same time bins
# For each time bin:
# - Assign a representative time (e.g. the bin centre).
# - Interpolate behavioural variables (x, y, etc.) to these bin times.
# - Behavioural and neural data are now expressed on a common temporal basis.

# 5. Exclude behaviourally irrelevant time periods
# - Compute the animal’s instantaneous speed from the binned position data.
# - Exclude time bins in which the animal is moving too slowly (e.g. below 2.5 cm/s).
# - This step removes immobility and sleep periods that would otherwise contaminate spatial analyses.

# 6. Prepare the data for population-level analysis
# - Optionally smooth firing-rate time series to reduce noise.
# - Normalize firing rates across neurons (e.g. z-scoring) to prevent high-rate neurons from dominating.

# 7. Spatial binning for rate map visualization
# - The arena is 1.5m × 1.5m (150cm × 150cm).
# - For rate map computation, divide the arena into spatial bins of size 3cm × 3cm.
# - This gives 150/3 = 50 bins per dimension (50 × 50 = 2500 spatial bins total).
# - Rate maps show average firing rate in each spatial bin, used for visualization and grid score computation.


def load_and_process_data(
    data_path: str,
    time_start: float = 27826,
    time_end: float = 31223,
    spatial_bin_cm: float = 3.0,
    time_bin_ms: int = 10,
    smoothing_sigma: float = 1.5,
    wall_val: float = 0.75,
    predictor_names: Optional[List[str]] = None,
    module_key: str = 'spikes_mod1',
    min_spikes: int = 100,
    speed_threshold: float = 2.5,
    max_trials: int = 5000,  # Subsample to avoid GPU OOM
    filter_grid_cells: bool = True,  # Apply grid cell filtering
    grid_filter_kwargs: Optional[Dict[str, Any]] = None,  # Parameters for grid_cell_filter
    **kwargs  # Accept additional config params (e.g., task, predictors) without error
) -> Dict[str, Any]:
    """
    Load and preprocess grid cell data from .npz file.
    Strategy : apply time and spatial binning first. And then filter for speed and min spikes.
    
    Parameters
    ----------
    data_path : str
        Path to the .npz file containing grid cell data.
    time_start : float
        Start time in seconds for data extraction.
    time_end : float
        End time in seconds for data extraction.
    spatial_bin_cm : float
        Size of spatial bins in cm for rate map computation. Default 3.0 cm.
        Arena is 150cm x 150cm, so 3cm bins give 50x50 = 2500 spatial bins.
    time_bin_ms : int
        Time bin size in milliseconds for temporal binning.
    smoothing_sigma : float
        Gaussian smoothing sigma for rate maps (in bins).
    wall_val : float
        Arena half-width in meters (0.75m for 1.5m x 1.5m arena).
    predictor_names : list of str, optional
        Names for the predictor variables. Defaults to ['x', 'y'].
    module_key : str
        Key in the npz file for spike data (e.g., 'spikes_mod1', 'spikes_mod2').
    min_spikes : int
        Minimum number of spikes for a cell to be included.
    speed_threshold : float
        Minimum speed threshold for including data points (in cm/s).
    
    Returns
    -------
    data_dict : dict
        Dictionary containing:
          - 'response': Firing rate at each position. (n_cells, n_trials) where n_trials = n_time_bins after filtering
          - 'predictors': Predictors object with x, y, z, azimuth positions. (n_cells, n_features, n_trials) where n_features = len(predictor_names) and n_trials = n_time_bins after filtering
          - 'position_data': Dict with raw x, y, t arrays.
    """
    # take note of time for logging purposes 
    clock_time_start = time.time()    
    if predictor_names is None:
        predictor_names = ['x', 'y']
    
    # Compute number of spatial bins from bin size
    # Arena is 2 * wall_val meters = 2 * wall_val * 100 cm
    arena_size_cm = 2 * wall_val * 100  # 150 cm for default wall_val=0.75
    n_spatial_bins = int(np.ceil(arena_size_cm / spatial_bin_cm))
    time_taken = time.time() - clock_time_start
    print(f"{time_taken:.3f}s : Spatial binning: {n_spatial_bins}x{n_spatial_bins} bins of {spatial_bin_cm:.1f}cm for {arena_size_cm:.0f}cm arena")
    
    # =========================================================================
    # Step 1: Load raw data
    # =========================================================================
    data = np.load(data_path, allow_pickle=True)
    t_raw = data['t']
    
    # Define all possible feature names (excluding 't' which is always required)
    # Add new features here as they become available in data files
    # KNOWN_FEATURES = ['x', 'y', 'z', 'azimuth']
    KNOWN_FEATURES = ['x', 'y']
    
    # Load all available features into a dictionary 
    features_raw = {}
    for feat_name in KNOWN_FEATURES:
        if feat_name in data.files:
            features_raw[feat_name] =  data[feat_name]

    time_taken = time.time() - clock_time_start    
    print(f"{time_taken:.3f}s : Available features in data file: {list(features_raw.keys())}")
    # Load spike times for each neuron
    spike_times_dict = data[module_key].item()
    n_neurons_raw = len(spike_times_dict)
    
    # =========================================================================
    # Step 2: Define common time discretisation
    # =========================================================================
    time_bin_s = time_bin_ms / 1000.0
    n_time_bins = int(np.ceil((time_end - time_start) / time_bin_s))
    # Bin edges and centers
    bin_edges = np.linspace(time_start, time_end, n_time_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    time_taken = time.time() - clock_time_start
    print(f"{time_taken:.3f}s : Time discretisation: {n_time_bins} bins of {time_bin_ms} ms from {time_start}s to {time_end}s")
    
    # =========================================================================
    # Step 3: Convert spike times into binned firing rates (vectorized)
    # =========================================================================
    firing_rates = np.zeros((n_neurons_raw, n_time_bins))
    total_spikes_per_neuron = np.zeros(n_neurons_raw)
    
    for neuron_idx, (neuron_id, spike_times) in enumerate(spike_times_dict.items()):
        # Filter spikes to time window
        spikes_in_window = spike_times[(spike_times >= time_start) & (spike_times < time_end)]
        total_spikes_per_neuron[neuron_idx] = len(spikes_in_window)
        
        # Bin the spikes using histogram 
        spike_counts, _ = np.histogram(spikes_in_window, bins=bin_edges)
        # Convert to firing rate (Hz)
        firing_rates[neuron_idx] = spike_counts / time_bin_s

    time_taken = time.time() - clock_time_start
    print(f"{time_taken:.3f}s : Step 3 Spike binning complete")
    
    # =========================================================================
    # Step 4: Calculate the binned mean of behavioural variables (vectorized)
    # =========================================================================
    # Use np.digitize for O(n) instead of O(n_time_bins * n) complexity
    bin_indices = np.digitize(t_raw, bin_edges) - 1  # digitize returns 1-indexed
    bin_indices = np.clip(bin_indices, 0, n_time_bins - 1)
    
    # Count samples per bin
    counts_per_bin = np.bincount(bin_indices, minlength=n_time_bins)
    
    features = {}
    for feat_name, feat_raw in features_raw.items():
        # Sum values per bin using bincount with weights
        sums_per_bin = np.bincount(bin_indices, weights=feat_raw, minlength=n_time_bins)
        # Compute mean, handling empty bins
        with np.errstate(invalid='ignore'):
            features[feat_name] = np.where(counts_per_bin > 0, 
                                           sums_per_bin / counts_per_bin, 
                                           np.nan)

    time_taken = time.time() - clock_time_start
    print(f"{time_taken:.3f}s : Step 4 Binned behavioural features computed")

    # =========================================================================
    # Step 5: Exclude low-speed periods
    # =========================================================================
    if 'x' in features and 'y' in features and speed_threshold > 0:
        # Compute instantaneous speed from binned positions
        # Positions are normalized to [-1, 1], scale by arena half-width to get cm
        # Arena is 1.5m x 1.5m, so half-width = 75 cm
        arena_half_width_cm = wall_val * 100  # wall_val is in meters, convert to cm
        
        dx = np.diff(features['x'], prepend=features['x'][0]) * arena_half_width_cm
        dy = np.diff(features['y'], prepend=features['y'][0]) * arena_half_width_cm
        speed = np.sqrt(dx**2 + dy**2) / time_bin_s  # cm/s
        speed[0] = speed[1] if len(speed) > 1 else 0  # First bin has no valid speed
        
        speed_mask = speed >= speed_threshold
        n_excluded = np.sum(~speed_mask)
        print(f"Speed filtering: excluding {n_excluded}/{n_time_bins} bins below {speed_threshold} cm/s")
        
        # Apply mask
        bin_centers = bin_centers[speed_mask]
        firing_rates = firing_rates[:, speed_mask]
        features = {name: arr[speed_mask] for name, arr in features.items()}
        n_time_bins = len(bin_centers)

    time_taken = time.time() - clock_time_start
    print(f"{time_taken:.3f}s : Step 5 Low-speed periods excluded")

    # =========================================================================
    # Step 6: Filter neurons by minimum spike count
    # =========================================================================
    good_neurons = total_spikes_per_neuron >= min_spikes
    firing_rates = firing_rates[good_neurons]
    n_cells = firing_rates.shape[0]
    
    time_taken = time.time() - clock_time_start
    print(f"{time_taken:.3f}s : Step 6 : Loaded {n_cells} neurons (from {n_neurons_raw} total) with >= {min_spikes} spikes")
    print(f"Final data shape: {n_cells} neurons x {n_time_bins} time bins")
    
    # =========================================================================
    # Step 7: Normalize positions and prepare predictors
    # =========================================================================
    # Normalize x, y to [-1, 1]
    features['x'] = features['x'] / wall_val
    features['y'] = features['y'] / wall_val
    
    
    # Response: firing rates (n_cells, n_time_bins)
    response = firing_rates
    
    # Build predictors array
    predictor_arrays = []
    for name in predictor_names:
        if name not in features:
            raise ValueError(f"Predictor '{name}' requested but not available. Available: {list(features.keys())}")
        # Tile to match (n_cells, n_time_bins)
        predictor_arrays.append(np.tile(features[name], (n_cells, 1)))
    
    predictors_data = np.stack(predictor_arrays, axis=1)
    predictors = Predictors(data=predictors_data, names=predictor_names)
    
    # =========================================================================
    # Compute rate maps using compute_rate_map 
    # =========================================================================
    x_norm = features['x']
    y_norm = features['y']
    
    occupancy, x_edges, y_edges = np.histogram2d(
        x_norm, y_norm, bins=n_spatial_bins, range=[[-1, 1], [-1, 1]]
    )
    
    # Weight occupancy by time spent (all bins equal for now)
    if smoothing_sigma is not None:
        occupancy_smooth = gaussian_filter(occupancy, sigma=smoothing_sigma)
    
    # Compute rate maps by averaging firing rates at each spatial bin
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
    
    # =========================================================================
    # Optional: Apply grid cell filtering
    # =========================================================================
    grid_filter_info = None
    if filter_grid_cells:
        filter_kwargs = grid_filter_kwargs or {}
        # Pass through relevant parameters
        filter_kwargs.setdefault('wall_val', wall_val)
        filter_kwargs.setdefault('time_bin_ms', time_bin_ms)
        
        position_data_for_filter = {
            **features,
            "t": bin_centers,
            "n_spatial_bins": n_spatial_bins,
            "time_bin_ms": time_bin_ms,
            "x_edges": x_edges,
            "y_edges": y_edges,
        }
        
        grid_cell_indices, grid_filter_info = grid_cell_filter(
            response=response,
            predictors=predictors,
            rate_maps=rate_maps,
            position_data=position_data_for_filter,
            **filter_kwargs
        )
        
        if len(grid_cell_indices) > 0:
            # Filter to only grid cells
            response = response[grid_cell_indices]
            rate_maps = rate_maps[grid_cell_indices]
            # Rebuild predictors for filtered cells
            predictors_data = predictors_data[grid_cell_indices]
            predictors = Predictors(data=predictors_data, names=predictor_names)
            n_cells = len(grid_cell_indices)
            print(f"Grid cell filtering: kept {n_cells} grid cells")
        else:
            print("Warning: No grid cells identified, returning all cells")
    
    return {
        "response": response, # Firing rates (n_cells, n_trials)
        "predictors": predictors, # Predictors object
        "trials": predictors_data,  # (n_cells, n_features, n_trials)
        "rate_maps": rate_maps, # (n_cells, n_spatial_bins, n_spatial_bins)
        "position_data": {
            **features,
            "t": bin_centers,
            "n_spatial_bins": n_spatial_bins,
            "time_bin_ms": time_bin_ms,
            "x_edges": x_edges,
            "y_edges": y_edges,
        },
        "grid_filter_info": grid_filter_info,  # None if filter_grid_cells=False, 
        "smoothing_sigma": smoothing_sigma,
    }

def grid_cell_filter(
    response: np.ndarray,
    predictors: Predictors,
    rate_maps: np.ndarray,
    position_data: Dict[str, np.ndarray],
    # Quality control parameters
    min_firing_rate_hz: float = 0.05,
    max_firing_rate_hz: float = 10.0,
    refractory_violation_threshold: float = 0.01,
    # Spatial analysis parameters
    coarse_bin_cm: float = 10.0,
    autocorr_min_lag_cm: float = 30.0,
    autocorr_max_lag_cm: float = 100.0,
    # Clustering parameters
    umap_n_neighbors: int = 15,
    umap_min_dist: float = 0.1,
    umap_n_components: int = 2,
    dbscan_eps: float = 0.7,
    dbscan_min_samples: int = 5,
    # Head direction filtering
    hd_tuning_threshold: float = 0.3,
    # Other
    wall_val: float = 0.75,
    time_bin_ms: int = 10,
    random_seed: int = 42,
    verbose: bool = True,
    **kwargs
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Grid-cell identification and module filtering procedure.

    This module implements the grid-cell filtering pipeline used in
    Gardner et al. (Nature, 2022), in which grid cells are identified
    via network-level clustering of spatially periodic firing patterns,
    rather than by applying a fixed grid-score threshold to individual
    cells.

    Parameters
    ----------
    response : np.ndarray
        Firing rates matrix of shape (n_cells, n_time_bins).
    predictors : Predictors
        Predictors object with position data.
    rate_maps : np.ndarray
        Fine-grained rate maps of shape (n_cells, n_spatial_bins, n_spatial_bins).
    position_data : dict
        Dictionary containing position and timing information.
    min_firing_rate_hz : float
        Minimum mean firing rate for inclusion (default 0.05 Hz).
    max_firing_rate_hz : float
        Maximum mean firing rate for inclusion (default 10 Hz).
    coarse_bin_cm : float
        Size of coarse spatial bins for autocorrelogram (default 10 cm).
    autocorr_min_lag_cm : float
        Minimum spatial lag to include in autocorrelogram (default 30 cm).
    autocorr_max_lag_cm : float
        Maximum spatial lag to include in autocorrelogram (default 100 cm).
    umap_n_neighbors : int
        Number of neighbors for UMAP embedding.
    umap_min_dist : float
        Minimum distance for UMAP embedding.
    dbscan_eps : float
        DBSCAN epsilon parameter.
    dbscan_min_samples : int
        DBSCAN minimum samples parameter.
    hd_tuning_threshold : float
        Mean vector length threshold for head direction exclusion (default 0.3).
    wall_val : float
        Arena half-width in meters.
    verbose : bool
        Whether to print progress information.

    Returns
    -------
    grid_cell_indices : np.ndarray
        Indices of cells identified as grid cells.
    filter_info : dict
        Dictionary containing intermediate results for diagnostics:
        - 'coarse_rate_maps': Coarse rate maps used for autocorrelation
        - 'autocorrelograms': Spatial autocorrelograms
        - 'umap_embedding': UMAP embedding coordinates
        - 'cluster_labels': DBSCAN cluster assignments
        - 'module_assignments': Grid module assignment for each grid cell
        - 'firing_rate_mask': Boolean mask for firing rate filter
        - 'hd_tuning_mask': Boolean mask for head direction filter
    """
    try:
        from umap import UMAP
        from sklearn.cluster import DBSCAN
    except ImportError:
        raise ImportError("grid_cell_filter requires umap-learn and scikit-learn. "
                          "Install with: pip install umap-learn scikit-learn")
    
    n_cells = response.shape[0]
    n_time_bins = response.shape[1]
    arena_size_cm = 2 * wall_val * 100  # 150 cm for default
    time_bin_s = time_bin_ms / 1000.0
    
    if verbose:
        print(f"Grid cell filtering: {n_cells} cells, {n_time_bins} time bins")
    
    # =========================================================================
    # Step 1: Initial unit selection - firing rate filter
    # =========================================================================
    mean_firing_rates = response.mean(axis=1)
    firing_rate_mask = (mean_firing_rates >= min_firing_rate_hz) & (mean_firing_rates <= max_firing_rate_hz)
    
    if verbose:
        n_pass_fr = firing_rate_mask.sum()
        print(f"Step 1: Firing rate filter ({min_firing_rate_hz}-{max_firing_rate_hz} Hz): "
              f"{n_pass_fr}/{n_cells} cells pass")
    
    # =========================================================================
    # Step 2: Construction of coarse spatial rate maps (10 cm × 10 cm bins)
    # =========================================================================
    n_coarse_bins = int(np.ceil(arena_size_cm / coarse_bin_cm))
    
    # Get positions from predictors (first cell, all same)
    x_norm = predictors.data[0, 0, :]  # Normalized x in [-1, 1]
    y_norm = predictors.data[0, 1, :]  # Normalized y in [-1, 1]
    
    # Compute coarse rate maps without smoothing
    coarse_rate_maps = np.zeros((n_cells, n_coarse_bins, n_coarse_bins))
    coarse_occupancy = np.zeros((n_coarse_bins, n_coarse_bins))
    
    # Bin positions
    bin_x = np.clip(((x_norm + 1) / 2 * n_coarse_bins).astype(int), 0, n_coarse_bins - 1)
    bin_y = np.clip(((y_norm + 1) / 2 * n_coarse_bins).astype(int), 0, n_coarse_bins - 1)
    
    # Compute occupancy
    for t_idx in range(n_time_bins):
        coarse_occupancy[bin_x[t_idx], bin_y[t_idx]] += 1
    
    # Compute rate maps (no smoothing for coarse maps)
    for c in range(n_cells):
        spike_map = np.zeros((n_coarse_bins, n_coarse_bins))
        for t_idx in range(n_time_bins):
            spike_map[bin_x[t_idx], bin_y[t_idx]] += response[c, t_idx]
        coarse_rate_maps[c] = spike_map / (coarse_occupancy + 1e-6)
    
    if verbose:
        print(f"Step 2: Coarse rate maps: {n_coarse_bins}x{n_coarse_bins} bins of {coarse_bin_cm} cm")
    
    # =========================================================================
    # Step 3: Spatial autocorrelogram computation
    # =========================================================================
    autocorrelograms = []
    autocorr_size = 2 * n_coarse_bins - 1
    
    # Compute spatial lag distances for masking
    lag_indices = np.arange(autocorr_size) - (n_coarse_bins - 1)
    lag_x, lag_y = np.meshgrid(lag_indices, lag_indices)
    lag_distance_cm = np.sqrt(lag_x**2 + lag_y**2) * coarse_bin_cm
    
    # Mask for valid lags (exclude center, keep intermediate range)
    lag_mask = (lag_distance_cm >= autocorr_min_lag_cm) & (lag_distance_cm <= autocorr_max_lag_cm)
    
    for c in range(n_cells):
        # Compute 2D autocorrelation using FFT
        rate_map = coarse_rate_maps[c]
        rate_map_centered = rate_map - rate_map.mean()
        
        # Zero-pad for full autocorrelation
        padded = np.zeros((2 * n_coarse_bins - 1, 2 * n_coarse_bins - 1))
        padded[:n_coarse_bins, :n_coarse_bins] = rate_map_centered
        
        # FFT-based autocorrelation
        fft = np.fft.fft2(padded)
        autocorr = np.fft.ifft2(fft * np.conj(fft)).real
        autocorr = np.fft.fftshift(autocorr)
        
        # Normalize
        autocorr = autocorr / (autocorr.max() + 1e-6)
        
        # Apply lag mask (zero out center and distant lags)
        autocorr_masked = autocorr * lag_mask
        
        autocorrelograms.append(autocorr_masked)
    
    autocorrelograms = np.array(autocorrelograms)
    
    if verbose:
        print(f"Step 3: Autocorrelograms computed, lag range: {autocorr_min_lag_cm}-{autocorr_max_lag_cm} cm")
    
    # =========================================================================
    # Step 4: Embedding of autocorrelogram structure (UMAP)
    # =========================================================================
    # Flatten and z-score autocorrelograms
    autocorr_flat = autocorrelograms.reshape(n_cells, -1)
    
    # Z-score each cell's autocorrelogram
    autocorr_zscore = np.zeros_like(autocorr_flat)
    for c in range(n_cells):
        if autocorr_flat[c].std() > 1e-6:
            autocorr_zscore[c] = (autocorr_flat[c] - autocorr_flat[c].mean()) / autocorr_flat[c].std()
        else:
            autocorr_zscore[c] = 0
    
    # Only embed cells that pass firing rate filter
    cells_to_embed = np.where(firing_rate_mask)[0]
    
    if len(cells_to_embed) < umap_n_neighbors + 1:
        if verbose:
            print(f"Warning: Not enough cells ({len(cells_to_embed)}) for UMAP embedding")
        return np.array([], dtype=int), {
            'coarse_rate_maps': coarse_rate_maps,
            'autocorrelograms': autocorrelograms,
            'firing_rate_mask': firing_rate_mask,
        }
    
    # UMAP embedding
    umap_model = UMAP(
        n_neighbors=min(umap_n_neighbors, len(cells_to_embed) - 1),
        min_dist=umap_min_dist,
        n_components=umap_n_components,
        random_state=random_seed,
    )
    embedding = umap_model.fit_transform(autocorr_zscore[cells_to_embed])
    
    # Store full embedding (NaN for excluded cells)
    umap_embedding = np.full((n_cells, umap_n_components), np.nan)
    umap_embedding[cells_to_embed] = embedding
    
    if verbose:
        print(f"Step 4: UMAP embedding computed for {len(cells_to_embed)} cells")
    
    # =========================================================================
    # Step 5: Clustering into grid modules (DBSCAN)
    # =========================================================================
    dbscan = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples)
    cluster_labels_subset = dbscan.fit_predict(embedding)
    
    # Map back to full cell indices
    cluster_labels = np.full(n_cells, -1)  # -1 = not clustered
    cluster_labels[cells_to_embed] = cluster_labels_subset
    
    unique_labels = np.unique(cluster_labels_subset)
    unique_labels = unique_labels[unique_labels >= 0]  # Exclude noise (-1)
    
    if verbose:
        n_noise = (cluster_labels_subset == -1).sum()
        print(f"Step 5: DBSCAN found {len(unique_labels)} clusters, {n_noise} noise points")
    
    # Find the largest cluster (likely non-grid cells) and exclude it
    if len(unique_labels) > 0:
        cluster_sizes = [(label, (cluster_labels == label).sum()) for label in unique_labels]
        cluster_sizes.sort(key=lambda x: x[1], reverse=True)
        
        if verbose:
            print(f"  Cluster sizes: {cluster_sizes}")
        
        # Identify grid module clusters (excluding the largest if it's much bigger)
        if len(cluster_sizes) > 1:
            largest_label, largest_size = cluster_sizes[0]
            second_size = cluster_sizes[1][1]
            
            # If largest cluster is > 2x the second, it's likely non-grid cells
            if largest_size > 2 * second_size:
                grid_module_labels = [label for label, size in cluster_sizes[1:]]
                if verbose:
                    print(f"  Excluding largest cluster {largest_label} (size {largest_size}) as non-grid")
            else:
                grid_module_labels = [label for label, size in cluster_sizes]
        else:
            grid_module_labels = [cluster_sizes[0][0]]
    else:
        grid_module_labels = []
    
    # Cells in grid modules
    grid_module_mask = np.isin(cluster_labels, grid_module_labels)
    
    # =========================================================================
    # Step 6: Module assignment
    # =========================================================================
    module_assignments = np.full(n_cells, -1)
    for i, label in enumerate(grid_module_labels):
        module_assignments[cluster_labels == label] = i
    
    if verbose:
        n_grid_candidates = grid_module_mask.sum()
        print(f"Step 6: {n_grid_candidates} grid cell candidates in {len(grid_module_labels)} modules")
    
    # =========================================================================
    # Step 7: Exclusion of conjunctive grid × head-direction cells
    # =========================================================================
    hd_tuning_mask = np.ones(n_cells, dtype=bool)  # True = keep (not HD tuned)
    
    # Check if azimuth data is available
    if 'azimuth' in position_data and position_data['azimuth'] is not None:
        azimuth = position_data['azimuth']
        
        # Compute mean vector length for each cell
        for c in range(n_cells):
            # Weight angles by firing rate
            weights = response[c]
            if weights.sum() > 0:
                # Circular mean
                mean_x = np.sum(weights * np.cos(azimuth)) / weights.sum()
                mean_y = np.sum(weights * np.sin(azimuth)) / weights.sum()
                mean_vector_length = np.sqrt(mean_x**2 + mean_y**2)
                
                if mean_vector_length > hd_tuning_threshold:
                    hd_tuning_mask[c] = False
        
        n_hd_excluded = (~hd_tuning_mask).sum()
        if verbose:
            print(f"Step 7: Excluding {n_hd_excluded} conjunctive HD cells (MVL > {hd_tuning_threshold})")
    else:
        if verbose:
            print(f"Step 7: No azimuth data available, skipping HD filtering")
    
    # =========================================================================
    # Final grid cell selection
    # =========================================================================
    grid_cell_mask = firing_rate_mask & grid_module_mask & hd_tuning_mask
    grid_cell_indices = np.where(grid_cell_mask)[0]
    
    if verbose:
        print(f"\nFinal result: {len(grid_cell_indices)} grid cells identified")
    
    # Compile diagnostic information
    filter_info = {
        'grid_cell_indices': grid_cell_indices,  # Original indices of grid cells
        'coarse_rate_maps': coarse_rate_maps,
        'autocorrelograms': autocorrelograms,
        'umap_embedding': umap_embedding,
        'cluster_labels': cluster_labels,
        'grid_module_labels': grid_module_labels,
        'module_assignments': module_assignments,
        'firing_rate_mask': firing_rate_mask,
        'grid_module_mask': grid_module_mask,
        'hd_tuning_mask': hd_tuning_mask,
        'mean_firing_rates': mean_firing_rates,
    }
    
    return grid_cell_indices, filter_info


def compute_rate_map(
    x: np.ndarray,
    y: np.ndarray,
    firing_rates: np.ndarray,
    spatial_bin_cm: float = 3.0,
    sigma: Optional[float] = 1.5,
    extent: Tuple[float, float, float, float] = (-1, 1, -1, 1),
) -> np.ndarray:
    """
    Compute a smoothed 2D rate map for one cell.
    
    Parameters
    ----------
    x, y : np.ndarray
        Position coordinates (n_time_bins,).
    firing_rates : np.ndarray
        Firing rates (n_time_bins,).
    spatial_bin_cm : float
        Size of spatial bins in centimeters.
    sigma : Optional[float]
        Gaussian smoothing sigma. If None, no smoothing is applied.
    extent : tuple
        (xmin, xmax, ymin, ymax) for histogram range.
    
    Returns
    -------
    rate_map : np.ndarray
        rate map of shape (n_bins, n_bins).
    """
    xmin, xmax, ymin, ymax = extent
    
    # Compute number of spatial bins from bin size
    arena_size_cm = (xmax - xmin) * 100  # Convert meters to cm
    n_spatial_bins = int(np.ceil(arena_size_cm / spatial_bin_cm))
    
    # Occupancy - global to all cells
    occupancy, _, _ = np.histogram2d(
        x, y, bins=n_spatial_bins, range=[[xmin, xmax], [ymin, ymax]]
    )
    if sigma is not None:
        occupancy = gaussian_filter(occupancy, sigma=sigma)

    rate_map = np.zeros((n_spatial_bins, n_spatial_bins))
    bin_x = np.clip(((x + 1) / 2 * n_spatial_bins).astype(int), 0, n_spatial_bins - 1)
    bin_y = np.clip(((y + 1) / 2 * n_spatial_bins).astype(int), 0, n_spatial_bins - 1)

    spike_map = np.zeros((n_spatial_bins, n_spatial_bins))
    for t_idx in range(firing_rates.shape[0]):
        spike_map[bin_x[t_idx], bin_y[t_idx]] += firing_rates[t_idx]

    if sigma is None:
        rate_map = spike_map / (occupancy + 1e-6)
    else:
        spike_map_smooth = gaussian_filter(spike_map, sigma=sigma)
        rate_map = spike_map_smooth / (occupancy + 1e-6)
    
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
