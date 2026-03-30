"""
Test gradient optimization for Gaussian grid cell model (grid_model_2).

This test assesses whether gradient optimization of parameters for the Gaussian
model improves model performance, and produces diagnostic visualizations.

Usage:
    python -m tests.test_grid_cells_gradient_optimization --cell_idx 45 --output_dir figures/gradopt_tests
    python -m tests.test_grid_cells_gradient_optimization --cell_idx 45 --show  # Interactive mode
    python -m tests.test_grid_cells_gradient_optimization --cell_idx 45 46 47 48  # Multiple cells
    python -m tests.test_grid_cells_gradient_optimization --cell_idx 40-50  # Range of cells
    python -m tests.test_grid_cells_gradient_optimization --cell_idx 45 --perturbation_test 0.1  # 10% perturbation
    python -m tests.test_grid_cells_gradient_optimization --cell_idx 45 --perturbation_test 0.1 --perturbation_seed 42  # Reproducible
    python -m tests.test_grid_cells_gradient_optimization --cell_idx 45 --landscape_params lam theta  # Custom loss landscape
"""

import argparse
import logging
import os
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
from scipy import ndimage, signal
from scipy.ndimage import label, find_objects, maximum_filter

from experiments.grid_cells.data_parser import compute_rate_map, load_and_process_data
from src.loss_functions import quadratic_loss

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Parameter name to index mapping for grid_model_2
# Parameters: [lam, theta, phi_x, phi_y, baseline, amplitude, sigma]
PARAM_NAMES = ['lam', 'theta', 'phi_x', 'phi_y', 'baseline', 'amplitude', 'sigma']
PARAM_INDEX = {name: idx for idx, name in enumerate(PARAM_NAMES)}

# Default ranges for each parameter in loss landscape
PARAM_RANGES = {
    'lam': (0.1, 1.5),
    'theta': (0.0, 1.05),  # ~pi/3
    'phi_x': (-0.5, 0.5),
    'phi_y': (-0.5, 0.5),
    'baseline': (-1.0, 3.0),
    'amplitude': (-0.1, 5.0),
    'sigma': (-0.1, 2.0),
}


# =============================================================================
# MODEL DEFINITIONS
# =============================================================================

def grid_model_2(X, lam=0.5, theta=0.0, phi_x=0.0, phi_y=0.0, baseline=0.0,
                 amplitude=1.0, sigma=0.08):
    """
    Grid cell model using Gaussian bumps on a hexagonal lattice.
    
    Args:
        X (np.ndarray): Predictor array (n_features, n_trials).
                        X[0] is x, X[1] is y (normalized to [-1, 1]).
        lam (float): Lattice spacing.
        theta (float): Orientation of the lattice in radians.
        phi_x, phi_y (float): Phase offsets.
        baseline (float): Baseline firing rate.
        amplitude (float): Peak amplitude per field.
        sigma (float): Width of each Gaussian field.
    
    Returns:
        np.ndarray: Predicted firing rate, shape (n_trials,).
    """
    x = X[0]
    y = X[1]
    
    # Clip parameters
    lam = np.clip(lam, 0.1, 2.0)
    theta = np.clip(theta, 0, np.pi / 3)
    sigma = np.clip(sigma, 0.01, 0.5)
    baseline = np.clip(baseline, 0, None)
    amplitude = np.clip(amplitude, 0, None)
    
    # Hexagonal lattice basis vectors
    v1 = np.array([lam, 0.0])
    v2 = np.array([0.5 * lam, 0.5 * np.sqrt(3.0) * lam])
    
    # Rotate basis by theta
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]])
    v1 = R @ v1
    v2 = R @ v2
    
    # Determine range of lattice points to sum over
    extent = 2.0
    margin = 2.0
    step = min(np.linalg.norm(v1), np.linalg.norm(v2))
    n_range = int(np.ceil((extent + margin * lam) / step)) + 2
    
    # Shifted positions
    dx = x - phi_x
    dy = y - phi_y
    
    # Sum Gaussian bumps
    r = np.full_like(x, baseline, dtype=float)
    inv2sig2 = 1.0 / (2.0 * sigma * sigma)
    
    for n in range(-n_range, n_range + 1):
        for m in range(-n_range, n_range + 1):
            cx, cy = n * v1 + m * v2
            ddx = dx - cx
            ddy = dy - cy
            r = r + amplitude * np.exp(-(ddx * ddx + ddy * ddy) * inv2sig2)
    
    return r


def grid_model_2_jax(X, lam=0.5, theta=0.0, phi_x=0.0, phi_y=0.0, baseline=0.0,
                     amplitude=1.0, sigma=0.08):
    """
    JAX-compatible version of grid_model_2.
    Uses a fixed range of lattice points for JIT compatibility.
    """
    x = X[0]
    y = X[1]
    
    lam = jnp.clip(lam, 0.1, 2.0)
    theta = jnp.clip(theta, 0, jnp.pi / 3)
    sigma = jnp.clip(sigma, 0.01, 0.5)
    baseline = jnp.clip(baseline, 0, None)
    amplitude = jnp.clip(amplitude, 0, None)
    
    # Hexagonal lattice basis vectors
    v1_x = lam * jnp.cos(theta)
    v1_y = lam * jnp.sin(theta)
    v2_x = 0.5 * lam * jnp.cos(theta) - 0.5 * jnp.sqrt(3.0) * lam * jnp.sin(theta)
    v2_y = 0.5 * lam * jnp.sin(theta) + 0.5 * jnp.sqrt(3.0) * lam * jnp.cos(theta)
    
    dx = x - phi_x
    dy = y - phi_y
    
    inv2sig2 = 1.0 / (2.0 * sigma * sigma)
    
    # Fixed lattice range for JIT compatibility
    n_range = 5
    n_vals = jnp.arange(-n_range, n_range + 1)
    m_vals = jnp.arange(-n_range, n_range + 1)
    N, M = jnp.meshgrid(n_vals, m_vals, indexing='ij')
    N = N.ravel()
    M = M.ravel()
    
    # Lattice centers
    cx = N * v1_x + M * v2_x
    cy = N * v1_y + M * v2_y
    
    # Distances from each lattice point
    ddx = dx[None, :] - cx[:, None]
    ddy = dy[None, :] - cy[:, None]
    
    # Sum of Gaussians
    bumps = amplitude * jnp.exp(-(ddx * ddx + ddy * ddy) * inv2sig2)
    r = baseline + jnp.sum(bumps, axis=0)
    
    return r


# =============================================================================
# PARAMETER ESTIMATION
# =============================================================================

def compute_rate_maps_from_data(x_pos, y_pos, firing_rates, n_spatial_bins=50,
                                 extent_min=-1.0, extent_max=1.0, smoothing_sigma=1.5):
    """
    Compute raw and smoothed rate maps from position and firing rate data.
    
    Args:
        x_pos: X positions of the animal.
        y_pos: Y positions of the animal.
        firing_rates: Firing rates at each position.
        n_spatial_bins: Number of spatial bins.
        extent_min, extent_max: Spatial extent.
        smoothing_sigma: Gaussian smoothing sigma for smoothed map.
    
    Returns:
        rate_map_raw: Raw (unsmoothed) rate map.
        rate_map_smooth: Smoothed rate map.
    """
    # Compute bin indices
    bin_x = np.clip(((x_pos - extent_min) / (extent_max - extent_min) * n_spatial_bins).astype(int), 
                    0, n_spatial_bins - 1)
    bin_y = np.clip(((y_pos - extent_min) / (extent_max - extent_min) * n_spatial_bins).astype(int), 
                    0, n_spatial_bins - 1)
    
    # Compute occupancy and spike maps
    occupancy, _, _ = np.histogram2d(x_pos, y_pos, bins=n_spatial_bins, 
                                      range=[[extent_min, extent_max], [extent_min, extent_max]])
    spike_map = np.zeros((n_spatial_bins, n_spatial_bins))
    for i in range(len(firing_rates)):
        spike_map[bin_x[i], bin_y[i]] += firing_rates[i]
    
    # Raw rate map
    rate_map_raw = spike_map / (occupancy + 1e-10)
    
    # Smoothed rate map
    spike_map_smooth = ndimage.gaussian_filter(spike_map, sigma=smoothing_sigma)
    occupancy_smooth = ndimage.gaussian_filter(occupancy, sigma=smoothing_sigma)
    rate_map_smooth = spike_map_smooth / (occupancy_smooth + 1e-10)
    
    return rate_map_raw, rate_map_smooth


def estimate_parameters_from_rate_map(rate_map_raw, rate_map_smooth, 
                                       extent_min=-0.75, extent_max=0.75):
    """
    Estimate grid cell model parameters from rate maps.
    
    Uses smoothed rate map for geometrical parameters (lambda, theta, phi).
    Uses raw rate map for amplitude, baseline, and sigma.
    
    Args:
        rate_map_raw: Raw (unsmoothed) rate map.
        rate_map_smooth: Smoothed rate map for geometry estimation.
        extent_min, extent_max: Spatial extent of the rate map.
    
    Returns:
        params: Array [lam, theta, phi_x, phi_y, baseline, amplitude, sigma]
    """
    # Work with transposed data (matching notebook convention)
    data_smooth = rate_map_smooth.T
    data_raw = rate_map_raw.T
    
    num_bins = data_smooth.shape[0]
    bin_size = (extent_max - extent_min) / num_bins
    
    # =========================================================================
    # 1. Find peaks in smoothed rate map for geometry
    # =========================================================================
    data_max = maximum_filter(data_smooth, size=5, mode='constant', cval=-np.inf)
    peaks = (data_smooth == data_max) & (data_smooth > 0)
    
    labeled, _ = label(peaks)
    slices = find_objects(labeled)
    
    x_peaks, y_peaks = [], []
    for s in slices:
        if s is None:
            continue
        dy, dx = s
        x_peaks.append(extent_min + ((dx.start + dx.stop - 1) / 2) * bin_size)
        y_peaks.append(extent_min + ((dy.start + dy.stop - 1) / 2) * bin_size)
    
    if len(x_peaks) == 0:
        logger.warning("No peaks found in rate map. Using default parameters.")
        return np.array([0.5, 0.0, 0.0, 0.0, 0.0, 1.0, 0.08])
    
    # =========================================================================
    # 2. Estimate lambda (grid spacing) - normalized to [-1, 1] extent
    # =========================================================================
    # Lambda is mean distance from peaks to origin, normalized
    lam = np.mean([np.sqrt(x**2 + y**2) for x, y in zip(x_peaks, y_peaks) 
                   if not (x == 0 and y == 0)])
    lam = lam / abs(extent_max)  # Normalize to [-1, 1]
    
    # =========================================================================
    # 3. Estimate phase (phi_x, phi_y) - central peak location
    # =========================================================================
    central_peak = min(zip(x_peaks, y_peaks), key=lambda p: np.sqrt(p[0]**2 + p[1]**2))
    phi_x, phi_y = central_peak
    
    # =========================================================================
    # 4. Estimate theta (orientation) from peak pairs
    # =========================================================================
    six_closest = sorted([(x, y) for x, y in zip(x_peaks, y_peaks) if (x, y) != central_peak],
                         key=lambda p: np.sqrt((p[0] - phi_x)**2 + (p[1] - phi_y)**2))[:6]
    
    # Find pairs of peaks that are 180 degrees apart
    pairs_of_peaks = []
    for i in range(len(six_closest)):
        for j in range(i + 1, len(six_closest)):
            p1, p2 = six_closest[i], six_closest[j]
            vec1 = np.array([p1[0] - phi_x, p1[1] - phi_y])
            vec2 = np.array([p2[0] - phi_x, p2[1] - phi_y])
            norm_prod = np.linalg.norm(vec1) * np.linalg.norm(vec2)
            if norm_prod > 0:
                cos_angle = np.dot(vec1, vec2) / norm_prod
                if np.isclose(cos_angle, -1.0, atol=0.2):
                    pairs_of_peaks.append((p1, p2))
    
    # Find orientation from the pair with smallest slope
    min_m = None
    for (x1, y1), (x2, y2) in pairs_of_peaks:
        if abs(x1 - x2) > 1e-6:
            A = np.array([[x1, 1], [x2, 1]])
            b = np.array([y1, y2])
            m, c = np.linalg.lstsq(A, b, rcond=None)[0]
            if min_m is None or abs(m) < abs(min_m):
                min_m = m
    
    theta = np.arctan(min_m) if min_m is not None else 0.0
    
    # =========================================================================
    # 5. Estimate baseline and amplitude from raw rate map
    # =========================================================================
    baseline = np.quantile(data_raw, 0.02)
    
    # Amplitude is mean peak value minus baseline
    peak_values = []
    for x_p, y_p in zip(x_peaks, y_peaks):
        x_bin = int((x_p - extent_min) / bin_size)
        y_bin = int((y_p - extent_min) / bin_size)
        x_bin = np.clip(x_bin, 0, num_bins - 1)
        y_bin = np.clip(y_bin, 0, num_bins - 1)
        peak_values.append(data_raw[y_bin, x_bin])
    amplitude = np.mean(peak_values) - baseline if peak_values else 1.0
    
    # =========================================================================
    # 6. Estimate sigma (bump width) from FWHM
    # =========================================================================
    sigma_list = []
    for x_p, y_p in zip(x_peaks, y_peaks):
        peak_bin_x = int((x_p - extent_min) / bin_size)
        peak_bin_y = int((y_p - extent_min) / bin_size)
        peak_bin_x = np.clip(peak_bin_x, 0, num_bins - 1)
        peak_bin_y = np.clip(peak_bin_y, 0, num_bins - 1)
        
        peak_value = data_raw[peak_bin_y, peak_bin_x]
        half_max = (peak_value + baseline) / 2.0
        
        # Search left and right for half-max
        left_bin = peak_bin_x
        while left_bin > 0 and data_raw[peak_bin_y, left_bin] > half_max:
            left_bin -= 1
        
        right_bin = peak_bin_x
        while right_bin < num_bins - 1 and data_raw[peak_bin_y, right_bin] > half_max:
            right_bin += 1
        
        fwhm = (right_bin - left_bin) * bin_size
        sigma_list.append(fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0))))
    
    sigma = np.mean(sigma_list) / abs(extent_max) if sigma_list else 0.08
    
    return np.array([lam, theta, phi_x, phi_y, baseline, amplitude, sigma])


# =============================================================================
# GRADIENT OPTIMIZATION
# =============================================================================

def objective_legacy(model, initial_params, loss_func, x, y,
                     param_penalty_weight=0.1, fit_params=True, random_seed=0,
                     FAILED_PROGRAM_COST=jnp.inf, tol=1e-2, max_iter=1000,
                     trial_batch_size=None, verbose=True):
    """
    Gradient-based optimization of model parameters.
    
    Optimized version that JIT-compiles the entire training step for speed.
    
    Args:
        model: JAX-compatible model function.
        initial_params: Initial parameters, shape (n_samples, n_params).
        loss_func: Loss function.
        x: Input data (positions).
        y: Response data (firing rates).
        param_penalty_weight: Penalty weight for number of parameters.
        fit_params: Whether to optimize parameters.
        random_seed: Random seed for train/test split.
        FAILED_PROGRAM_COST: Cost for failed optimizations.
        tol: Convergence tolerance.
        max_iter: Maximum optimization iterations.
        trial_batch_size: Batch size (None = use all trials at once, fastest).
        verbose: Whether to print progress.
    
    Returns:
        initial_loss: Loss before optimization.
        initial_params: Initial parameters.
        final_loss: Loss after optimization.
        final_params: Optimized parameters.
        training_trials_idx: Indices of training trials.
        test_trials_idx: Indices of test trials.
        loss_history: Dictionary with training and test loss history.
    """
    t_start = time.time()
    
    # Normalize x to Inputs format
    x_inputs = ensure_inputs(x)
    x_data = x_inputs.to_tensor()
    
    n_samples, n_features, n_trials = x_data.shape
    
    # Train/test split - alternating chunks
    n_trial_splits = 10
    trials_per_split = n_trials // n_trial_splits
    split_indices = [jnp.arange(i * trials_per_split, (i + 1) * trials_per_split) 
                     for i in range(n_trial_splits)]
    training_trials_idx = jnp.concatenate([split_indices[i] for i in range(n_trial_splits) if i % 2 == 1])
    test_trials_idx = jnp.concatenate([split_indices[i] for i in range(n_trial_splits) if i % 2 == 0])
    
    x_train = x_data[:, :, training_trials_idx]
    y_train = y[:, training_trials_idx]
    x_test = x_data[:, :, test_trials_idx]
    y_test = y[:, test_trials_idx]
    
    # Validate initial params
    if initial_params is None or not isinstance(initial_params, jnp.ndarray):
        logger.error("initial_params should be a JAX array")
        return FAILED_PROGRAM_COST, jnp.zeros((n_samples, 0)), FAILED_PROGRAM_COST, jnp.zeros((n_samples, 0)), training_trials_idx, test_trials_idx, {}
    
    if initial_params.ndim != 2 or initial_params.shape[0] != n_samples:
        logger.error(f"initial_params should have shape ({n_samples}, n_params)")
        return FAILED_PROGRAM_COST, jnp.zeros((n_samples, 0)), FAILED_PROGRAM_COST, jnp.zeros((n_samples, 0)), training_trials_idx, test_trials_idx, {}
    
    n_params = initial_params.shape[1]
    
    # Validate model (only check first sample to save time)
    try:
        model_jit = jax.jit(model)
        output = model_jit(x_data[0], *initial_params[0])
        if output.ndim != 1 or output.shape[0] != n_trials:
            logger.error(f"Model output shape mismatch")
            return FAILED_PROGRAM_COST, initial_params, FAILED_PROGRAM_COST, initial_params, training_trials_idx, test_trials_idx, {}
    except Exception as e:
        logger.error(f"Model validation failed: {e}")
        return FAILED_PROGRAM_COST, initial_params, FAILED_PROGRAM_COST, initial_params, training_trials_idx, test_trials_idx, {}
    
    # Loss function setup - fully JIT-compiled
    # vmap over cells: each cell has its own params and data
    def loss_single_cell(params, x_i, y_i):
        preds = model(x_i, *params)
        return jnp.mean(loss_func(preds, y_i))
    
    loss_all_cells = jax.vmap(loss_single_cell, in_axes=(0, 0, 0))
    
    @jax.jit
    def compute_total_loss(params_2d, x_data, y_data):
        """Compute mean loss across all cells."""
        cell_losses = loss_all_cells(params_2d, x_data, y_data)
        return jnp.mean(cell_losses)
    
    # JIT-compiled gradient function
    grad_fn = jax.jit(jax.grad(compute_total_loss))
    
    # Track loss history
    loss_history = {'train': [], 'test': [], 'steps': []}
    
    if fit_params:
        # Adam optimizer with learning rate schedule
        peak_lr = 0.001
        schedule = optax.warmup_cosine_decay_schedule(
            init_value=peak_lr * 0.1,
            peak_value=peak_lr,
            # warmup_steps=50,
            warmup_steps=max_iter // 10,
            decay_steps=max_iter,
            end_value=peak_lr * 0.01
        )
        opt = optax.adam(schedule, b1=0.9, b2=0.999, eps=1e-8)
        opt_state = opt.init(initial_params)
        
        # Fully JIT-compiled train step
        @jax.jit
        def train_step(params, opt_state):
            loss = compute_total_loss(params, x_train, y_train)
            grad = grad_fn(params, x_train, y_train)
            updates, new_opt_state = opt.update(grad, opt_state, params)
            new_params = optax.apply_updates(params, updates)
            return new_params, new_opt_state, loss
        
        params = initial_params
        
        # Warmup JIT compilation
        if verbose:
            print("Compiling JIT functions...")
        _ = train_step(params, opt_state)
        initial_train_loss = float(compute_total_loss(params, x_train, y_train))
        if verbose:
            print(f"JIT compilation complete. Initial loss: {initial_train_loss:.4f}")
        
        # Catastrophic loss threshold
        CATASTROPHIC_LOSS_THRESHOLD = 1e6
        if initial_train_loss > CATASTROPHIC_LOSS_THRESHOLD:
            if verbose:
                print(f"Initial loss {initial_train_loss:.2e} exceeds threshold. Skipping optimization.")
            return FAILED_PROGRAM_COST, initial_params, FAILED_PROGRAM_COST, initial_params, training_trials_idx, test_trials_idx, loss_history
        
        best_loss = initial_train_loss
        best_params = params
        print_every = 100
        log_every = 50  # Reduced frequency for test loss evaluation
        
        for step in range(1, max_iter + 1):
            params, opt_state, loss_val = train_step(params, opt_state)
            loss_val_float = float(loss_val)
            
            if jnp.isnan(loss_val) or jnp.isinf(loss_val):
                if verbose:
                    print(f"Loss is NaN/Inf at step {step}. Stopping.")
                break
            
            if loss_val_float > CATASTROPHIC_LOSS_THRESHOLD:
                if verbose:
                    print(f"Loss exploded to {loss_val_float:.2e}. Stopping.")
                return FAILED_PROGRAM_COST, initial_params, FAILED_PROGRAM_COST, initial_params, training_trials_idx, test_trials_idx, loss_history
            
            if loss_val_float < best_loss:
                best_loss = loss_val_float
                best_params = params
            
            # Log less frequently to reduce overhead
            if step % log_every == 0:
                test_loss = float(compute_total_loss(params, x_test, y_test))
                loss_history['train'].append(loss_val_float)
                loss_history['test'].append(test_loss)
                loss_history['steps'].append(step)
            
            if verbose and step % print_every == 0:
                print(f"step {step:4d}  loss {loss_val_float:.4f}")
        
        params = best_params
        if verbose:
            print(f"Optimization complete. Best loss: {best_loss:.4f}")
    else:
        params = initial_params
    
    # Final evaluation
    initial_loss = float(compute_total_loss(initial_params, x_test, y_test)) + param_penalty_weight * n_params
    final_loss = float(compute_total_loss(params, x_test, y_test)) + param_penalty_weight * n_params
    
    initial_loss = float(jnp.nan_to_num(initial_loss, nan=FAILED_PROGRAM_COST))
    final_loss = float(jnp.nan_to_num(final_loss, nan=FAILED_PROGRAM_COST))
    
    t_end = time.time()
    if verbose:
        print(f"Time taken: {t_end - t_start:.2f} seconds")
    
    return initial_loss, initial_params, final_loss, params, training_trials_idx, test_trials_idx, loss_history


# =============================================================================
# LOSS MAP COMPUTATION
# =============================================================================

def compute_loss_map(model, params, x_pos, y_pos, firing_rate, smoothing_sigma=0.0,
                     trials_idx=None, loss_func=None):
    """
    Compute loss map over spatial bins.
    """
    if loss_func is None:
        loss_func = quadratic_loss
    
    # Filter data
    x_filtered = x_pos[trials_idx] if trials_idx is not None else x_pos
    y_filtered = y_pos[trials_idx] if trials_idx is not None else y_pos
    fr_filtered = firing_rate[trials_idx] if trials_idx is not None else firing_rate
    
    # Compute predictions
    X_filtered = jnp.zeros((2, len(x_filtered)))
    X_filtered = X_filtered.at[0].set(x_filtered)
    X_filtered = X_filtered.at[1].set(y_filtered)
    preds_filtered = model(X_filtered, *params)
    
    # Compute rate maps
    real_rate_map = compute_rate_map(x_filtered, y_filtered, fr_filtered, 
                                      spatial_bin_cm=3.0, sigma=smoothing_sigma)
    pred_rate_map = compute_rate_map(x_filtered, y_filtered, preds_filtered,
                                      spatial_bin_cm=3.0, sigma=smoothing_sigma)
    
    loss_map = loss_func(pred_rate_map, real_rate_map)
    
    # Count map
    num_bins = real_rate_map.shape[0]
    count_map = np.zeros_like(real_rate_map)
    bin_edges = np.linspace(-1, 1, num_bins + 1)
    for xi, yi in zip(x_filtered, y_filtered):
        x_bin = np.digitize(xi, bin_edges) - 1
        y_bin = np.digitize(yi, bin_edges) - 1
        if 0 <= x_bin < num_bins and 0 <= y_bin < num_bins:
            count_map[x_bin, y_bin] += 1
    
    return loss_map, count_map


def compute_loss_landscape(model, base_params, X, y_true, loss_func,
                           amp_range=(0.0, 5.0), sigma_range=(0.0, 1.0), n_grid=50):
    """
    Compute loss landscape over amplitude and sigma using vectorized JAX.
    """
    amp_values = jnp.linspace(amp_range[0], amp_range[1], n_grid)
    sigma_values = jnp.linspace(sigma_range[0], sigma_range[1], n_grid)
    
    AMP, SIG = jnp.meshgrid(amp_values, sigma_values, indexing='ij')
    
    # Build all parameter combinations
    all_params = jnp.broadcast_to(base_params, (n_grid, n_grid, len(base_params))).copy()
    all_params = all_params.at[:, :, 5].set(AMP)  # amplitude is index 5
    all_params = all_params.at[:, :, 6].set(SIG)  # sigma is index 6
    all_params_flat = all_params.reshape(-1, len(base_params))
    
    X_jax = jnp.array(X)
    y_jax = jnp.array(y_true)
    
    @jax.jit
    def compute_loss_single(p):
        preds = grid_model_2_jax(X_jax, *p)
        return jnp.mean(loss_func(preds, y_jax))
    
    compute_loss_batched = jax.vmap(compute_loss_single)
    losses_flat = compute_loss_batched(all_params_flat)
    
    loss_landscape = losses_flat.reshape(n_grid, n_grid)
    
    return loss_landscape, amp_values, sigma_values


# =============================================================================
# VISUALIZATION
# =============================================================================

def create_diagnostic_plot(cell_idx, rate_map_raw, rate_map_smooth, params_init, params_opt,
                           loss_init, loss_opt, loss_history, loss_map_init_train, loss_map_opt_train,
                           loss_map_init_test, loss_map_opt_test, count_map_train, count_map_test,
                           X, x_pos, y_pos, firing_rate, training_trials_idx, test_trials_idx, perturbation_scale=None,
                           extent_min=-0.75, extent_max=0.75, smoothing_sigma=0.0):
    """
    Create comprehensive diagnostic visualization.
    """
    fig = plt.figure(figsize=(24, 20))
    
    # Compute predictions for visualization
    preds_init = grid_model_2(X, *params_init[0])
    preds_opt = grid_model_2(X, *params_opt[0])
    
    pred_rate_map_init = compute_rate_map(x_pos, y_pos, preds_init, spatial_bin_cm=3.0, sigma=smoothing_sigma)
    pred_rate_map_opt = compute_rate_map(x_pos, y_pos, preds_opt, spatial_bin_cm=3.0, sigma=smoothing_sigma)
    
    # Compute global color scale for loss maps
    all_losses = [loss_map_init_train, loss_map_opt_train, loss_map_init_test, loss_map_opt_test]
    vmin_global = np.nanmin([np.nanmin(lm) for lm in all_losses])
    vmax_global = np.nanmax([np.nanmax(lm) for lm in all_losses])
    
    # Row 1: Rate maps comparison
    ax1 = plt.subplot(5, 4, 1)
    plt.imshow(rate_map_raw.T, origin='lower', extent=[extent_min, extent_max, extent_min, extent_max], cmap='viridis')
    plt.axhline(0, color='white', linestyle='--', alpha=0.5)
    plt.axvline(0, color='white', linestyle='--', alpha=0.5)
    plt.colorbar(label='Firing Rate')
    plt.title(f'Real Rate Map (Raw) - Cell {cell_idx}', fontweight='bold')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    
    ax2 = plt.subplot(5, 4, 2)
    plt.imshow(rate_map_smooth.T, origin='lower', extent=[extent_min, extent_max, extent_min, extent_max], cmap='viridis')
    plt.axhline(0, color='white', linestyle='--', alpha=0.5)
    plt.axvline(0, color='white', linestyle='--', alpha=0.5)
    plt.colorbar(label='Firing Rate')
    plt.title(f'Real Rate Map (Smoothed) - Cell {cell_idx}', fontweight='bold')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    
    ax3 = plt.subplot(5, 4, 3)
    plt.imshow(pred_rate_map_init.T, origin='lower', extent=[extent_min, extent_max, extent_min, extent_max], cmap='viridis')
    plt.axhline(0, color='white', linestyle='--', alpha=0.5)
    plt.axvline(0, color='white', linestyle='--', alpha=0.5)
    plt.colorbar(label='Firing Rate')
    plt.title(f'Initial Params\nLoss: {loss_init:.4f}', fontweight='bold')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    
    ax4 = plt.subplot(5, 4, 4)
    plt.imshow(pred_rate_map_opt.T, origin='lower', extent=[extent_min, extent_max, extent_min, extent_max], cmap='viridis')
    plt.axhline(0, color='white', linestyle='--', alpha=0.5)
    plt.axvline(0, color='white', linestyle='--', alpha=0.5)
    plt.colorbar(label='Firing Rate')
    plt.title(f'Optimized Params\nLoss: {loss_opt:.4f}', fontweight='bold')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    
    # Row 2: Coverage maps
    ax5 = plt.subplot(5, 4, 5)
    plt.imshow(count_map_train.T, origin='lower', extent=[-1, 1, -1, 1], cmap='Greys')
    plt.colorbar(label='Count')
    plt.title(f'Train Coverage (n={len(training_trials_idx)})')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    
    ax6 = plt.subplot(5, 4, 6)
    plt.imshow(count_map_test.T, origin='lower', extent=[-1, 1, -1, 1], cmap='Greys')
    plt.colorbar(label='Count')
    plt.title(f'Test Coverage (n={len(test_trials_idx)})')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    
    # Row 2: Loss maps - Initial
    ax7 = plt.subplot(5, 4, 7)
    plt.imshow(loss_map_init_train.T, origin='lower', extent=[-1, 1, -1, 1],
               cmap='Reds', vmin=vmin_global, vmax=vmax_global)
    plt.colorbar(label='MSE Loss')
    plt.title(f'Initial - Train\nMean: {np.nanmean(loss_map_init_train):.4f}')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    
    ax8 = plt.subplot(5, 4, 8)
    plt.imshow(loss_map_init_test.T, origin='lower', extent=[-1, 1, -1, 1],
               cmap='Reds', vmin=vmin_global, vmax=vmax_global)
    plt.colorbar(label='MSE Loss')
    plt.title(f'Initial - Test\nMean: {np.nanmean(loss_map_init_test):.4f}')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    
    # Row 3: Loss maps - Optimized
    ax9 = plt.subplot(5, 4, 9)
    plt.imshow(loss_map_opt_train.T, origin='lower', extent=[-1, 1, -1, 1],
               cmap='Reds', vmin=vmin_global, vmax=vmax_global)
    plt.colorbar(label='MSE Loss')
    plt.title(f'Optimized - Train\nMean: {np.nanmean(loss_map_opt_train):.4f}')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    
    ax10 = plt.subplot(5, 4, 10)
    plt.imshow(loss_map_opt_test.T, origin='lower', extent=[-1, 1, -1, 1],
               cmap='Reds', vmin=vmin_global, vmax=vmax_global)
    plt.colorbar(label='MSE Loss')
    plt.title(f'Optimized - Test\nMean: {np.nanmean(loss_map_opt_test):.4f}')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    
    # Row 3: Training curves
    ax11 = plt.subplot(5, 4, 11)
    if loss_history['steps']:
        plt.plot(loss_history['steps'], loss_history['train'], 'b-', label='Train', linewidth=2)
        plt.plot(loss_history['steps'], loss_history['test'], 'r-', label='Test', linewidth=2)
        plt.xlabel('Optimization Step')
        plt.ylabel('Loss')
        plt.title('Training Curves')
        plt.legend()
        plt.grid(True, alpha=0.3)
    else:
        plt.text(0.5, 0.5, 'No training history', ha='center', va='center')
        plt.title('Training Curves')
    
    # Row 3: Loss distribution
    ax12 = plt.subplot(5, 4, 12)
    losses_init = loss_map_init_test[~np.isnan(loss_map_init_test)].flatten()
    losses_opt = loss_map_opt_test[~np.isnan(loss_map_opt_test)].flatten()
    plt.hist(losses_init, bins=40, alpha=0.5, color='red', label='Initial', edgecolor='black')
    plt.hist(losses_opt, bins=40, alpha=0.5, color='green', label='Optimized', edgecolor='black')
    plt.xlabel('MSE Loss')
    plt.ylabel('Frequency')
    plt.title('Test Loss Distribution')
    plt.legend()
    plt.axvline(np.nanmean(losses_init), color='darkred', linestyle='--', linewidth=2)
    plt.axvline(np.nanmean(losses_opt), color='darkgreen', linestyle='--', linewidth=2)
    
    # Row 4: Parameter comparison
    ax13 = plt.subplot(5, 4, 13)
    param_names = ['λ', 'θ', 'φx', 'φy', 'base', 'amp', 'σ']
    x_pos_bar = np.arange(len(param_names))
    width = 0.35
    plt.bar(x_pos_bar - width/2, params_init[0], width, label='Initial', alpha=0.8)
    plt.bar(x_pos_bar + width/2, params_opt[0], width, label='Optimized', alpha=0.8)
    plt.xticks(x_pos_bar, param_names)
    plt.ylabel('Parameter Value')
    plt.title('Parameter Comparison')
    plt.legend()
    
    # Row 4: Summary statistics
    ax14 = plt.subplot(5, 4, 14)
    plt.axis('off')

    if perturbation_scale is not None:
        perturbation_test_text = f" (PERTURBATION TEST {perturbation_scale}) "
    else:
        perturbation_test_text = ""

    summary_text = f"""
GRADIENT OPTIMIZATION SUMMARY {perturbation_test_text}
{'='*45}

Cell Index: {cell_idx}

PARAMETERS:
  Initial:   [{', '.join([f'{p:.3f}' for p in params_init[0]])}]
  Optimized: [{', '.join([f'{p:.3f}' for p in params_opt[0]])}]

LOSS (Test Set):
  Initial:    {loss_init:.4f}
  Optimized:  {loss_opt:.4f}
  Change:     {loss_opt - loss_init:.4f} ({(loss_opt / loss_init - 1) * 100:+.1f}%)

SPATIAL LOSS (Mean):
  Train Initial:    {np.nanmean(loss_map_init_train):.4f}
  Train Optimized:  {np.nanmean(loss_map_opt_train):.4f}
  Test Initial:     {np.nanmean(loss_map_init_test):.4f}
  Test Optimized:   {np.nanmean(loss_map_opt_test):.4f}

IMPROVEMENT: {'YES' if loss_opt < loss_init else 'NO'}
"""
    plt.text(0.05, 0.95, summary_text, fontsize=10, verticalalignment='top',
             fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    return fig


def create_loss_landscape_plot(cell_idx, loss_landscape, param1_values, param2_values,
                                params_init, params_opt, X, firing_rate, grid_model,
                                param1_name='amplitude', param2_name='sigma',
                                param1_idx=5, param2_idx=6):
    """
    Create 3D loss landscape visualization.
    
    Args:
        cell_idx: Cell index for title.
        loss_landscape: 2D array of loss values.
        param1_values: Values for x-axis parameter.
        param2_values: Values for y-axis parameter.
        params_init: Initial parameters.
        params_opt: Optimized parameters.
        X: Input positions.
        firing_rate: True firing rates.
        grid_model: Model function.
        param1_name: Name of first parameter (x-axis).
        param2_name: Name of second parameter (y-axis).
        param1_idx: Index of first parameter in params array.
        param2_idx: Index of second parameter in params array.
    """
    fig = plt.figure(figsize=(14, 6))
    
    # 2D heatmap
    ax1 = fig.add_subplot(1, 2, 1)
    im = ax1.imshow(loss_landscape.T, origin='lower',
                    extent=[param1_values[0], param1_values[-1], param2_values[0], param2_values[-1]],
                    aspect='auto', cmap='viridis')
    plt.colorbar(im, ax=ax1, label='MSE Loss')
    ax1.scatter(params_init[0][param1_idx], params_init[0][param2_idx], color='red', marker='x', s=200, 
                linewidths=3, label='Initial', zorder=5)
    ax1.scatter(params_opt[0][param1_idx], params_opt[0][param2_idx], color='lime', marker='o', s=200,
                linewidths=3, label='Optimized', zorder=5)
    
    # Find minimum
    min_idx = np.unravel_index(np.argmin(loss_landscape), loss_landscape.shape)
    min_p1 = param1_values[min_idx[0]]
    min_p2 = param2_values[min_idx[1]]
    ax1.scatter(min_p1, min_p2, color='cyan', marker='^', s=200,
                linewidths=3, label='Minimum', zorder=5)
    
    ax1.set_xlabel(param1_name)
    ax1.set_ylabel(param2_name)
    ax1.set_title(f'Loss Landscape (Cell {cell_idx})\nMin at {param1_name}={min_p1:.3f}, {param2_name}={min_p2:.3f}')
    ax1.legend()
    
    # 3D surface
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    P1, P2 = np.meshgrid(param1_values, param2_values)
    ax2.plot_surface(P1, P2, loss_landscape.T, cmap='viridis', edgecolor='none', alpha=0.7)
    ax2.set_xlabel(param1_name)
    ax2.set_ylabel(param2_name)
    ax2.set_zlabel('MSE Loss')
    ax2.set_title(f'3D Loss Landscape (Cell {cell_idx})')
    
    # Mark points
    loss_init = float(jnp.mean(quadratic_loss(firing_rate, grid_model(X, *params_init[0]))))
    loss_opt = float(jnp.mean(quadratic_loss(firing_rate, grid_model(X, *params_opt[0]))))
    ax2.scatter(params_init[0][param1_idx], params_init[0][param2_idx], loss_init, 
                color='red', marker='x', s=100, label='Initial')
    ax2.scatter(params_opt[0][param1_idx], params_opt[0][param2_idx], loss_opt,
                color='lime', marker='o', s=100, label='Optimized')
    ax2.legend()
    
    plt.tight_layout()
    return fig


# =============================================================================
# MAIN TEST FUNCTION
# =============================================================================

def run_gradient_optimization_test(cell_idx, data_path, output_dir=None, show_plots=False,
                                    max_iter=1000, smoothing_sigma=1.5):
    """
    Run complete gradient optimization test for a single cell.
    
    Args:
        cell_idx: Index of cell to analyze.
        data_path: Path to data file.
        output_dir: Directory to save figures (optional).
        show_plots: Whether to keep figures open for interactive display.
        max_iter: Maximum optimization iterations.
        smoothing_sigma: Sigma for rate map smoothing.
    
    Returns:
        results: Dictionary with test results.
    """
    # Use the batch version with a single cell
    results_list = run_gradient_optimization_batch(
        cell_indices=[cell_idx],
        data_path=data_path,
        output_dir=output_dir,
        show_plots=show_plots,
        max_iter=max_iter,
        smoothing_sigma=smoothing_sigma
    )
    return results_list[0] if results_list else None


def run_gradient_optimization_batch(cell_indices, data_path, output_dir=None, show_plots=False,
                                     max_iter=1000, smoothing_sigma=1.5,
                                     perturbation_scale=None, perturbation_seed=None,
                                     landscape_params=None):
    """
    Run gradient optimization test for multiple cells with parallelized computation.
    
    Data loading is done once. Initial parameter estimation is done iteratively.
    Gradient optimization is done in parallel across all cells.
    
    Args:
        cell_indices: List of cell indices to analyze.
        data_path: Path to data file.
        output_dir: Directory to save figures (optional).
        show_plots: Whether to keep figures open for interactive display.
        max_iter: Maximum optimization iterations.
        smoothing_sigma: Sigma for rate map smoothing.
        perturbation_scale: If provided, add Gaussian noise to initial parameters
                           with std = perturbation_scale * |param_value|.
                           Tests robustness of gradient descent to initialization.
        perturbation_seed: Random seed for perturbation (for reproducibility).
        landscape_params: Tuple of two parameter names to vary in loss landscape.
                         Default is ('amplitude', 'sigma').
    
    Returns:
        results_list: List of dictionaries with test results for each cell.
    """
    # Set default landscape parameters
    if landscape_params is None:
        landscape_params = ('amplitude', 'sigma')
    
    # Validate landscape params
    param1_name, param2_name = landscape_params
    if param1_name not in PARAM_INDEX:
        raise ValueError(f"Unknown parameter '{param1_name}'. Valid: {PARAM_NAMES}")
    if param2_name not in PARAM_INDEX:
        raise ValueError(f"Unknown parameter '{param2_name}'. Valid: {PARAM_NAMES}")
    
    param1_idx = PARAM_INDEX[param1_name]
    param2_idx = PARAM_INDEX[param2_name]
    param1_range = PARAM_RANGES[param1_name]
    param2_range = PARAM_RANGES[param2_name]
    
    logger.info(f"Loss landscape: {param1_name} vs {param2_name}")
    
    n_cells = len(cell_indices)
    logger.info(f"Running batch gradient optimization for {n_cells} cells: {cell_indices}")
    
    # =========================================================================
    # 1. Load data ONCE for all cells
    # =========================================================================
    logger.info("Loading data...")
    t_start = time.time()
    result = load_and_process_data(data_path, filter_grid_cells=True)
    inputs = result['inputs']
    outputs = result['outputs']
    
    total_cells = outputs.shape[0]
    
    # Validate cell indices
    valid_indices = []
    for idx in cell_indices:
        if idx >= total_cells:
            logger.warning(f"cell_idx {idx} out of range (max: {total_cells - 1}). Skipping.")
        else:
            valid_indices.append(idx)
    
    if not valid_indices:
        logger.error("No valid cell indices provided")
        return []
    
    n_valid = len(valid_indices)
    logger.info(f"Data loaded in {time.time() - t_start:.2f}s. Processing {n_valid} valid cells.")
    
    # =========================================================================
    # 2. Compute rate maps and estimate initial parameters (iteratively - as requested)
    # =========================================================================
    logger.info("Computing rate maps and estimating initial parameters...")
    t_start = time.time()
    
    rate_maps_raw = {}
    rate_maps_smooth = {}
    initial_params_list = []
    
    for i, cell_idx in enumerate(valid_indices):
        x_pos = inputs['x'][cell_idx]
        y_pos = inputs['y'][cell_idx]
        firing_rate = outputs['firing_rate'][cell_idx]
        
        # Compute rate maps
        rate_map_raw, rate_map_smooth = compute_rate_maps_from_data(
            x_pos, y_pos, firing_rate, 
            n_spatial_bins=50, 
            smoothing_sigma=smoothing_sigma
        )
        rate_maps_raw[cell_idx] = rate_map_raw
        rate_maps_smooth[cell_idx] = rate_map_smooth
        
        # Estimate initial parameters
        params_guessed = estimate_parameters_from_rate_map(
            rate_map_raw, rate_map_smooth,
            extent_min=-0.75, extent_max=0.75
        )
        initial_params_list.append(params_guessed)
        
        if (i + 1) % 10 == 0 or i == n_valid - 1:
            logger.info(f"  Parameter estimation: {i+1}/{n_valid} cells done")
    
    logger.info(f"Parameter estimation completed in {time.time() - t_start:.2f}s")
    
    # =========================================================================
    # 3. Run PARALLEL gradient optimization for all cells
    # =========================================================================
    logger.info(f"Running parallel gradient optimization for {n_valid} cells...")
    t_start = time.time()
    
    # Stack all initial parameters
    all_initial_params = jnp.array(initial_params_list)  # Shape: (n_valid, n_params)
    
    # Apply perturbation if requested
    params_before_perturbation = None
    if perturbation_scale is not None and perturbation_scale > 0:
        logger.info(f"Applying perturbation with scale {perturbation_scale}")
        params_before_perturbation = np.array(all_initial_params)  # Save original
        
        # Set random seed if provided
        rng = np.random.default_rng(perturbation_seed)
        
        # Perturbation: add Gaussian noise proportional to parameter magnitude
        # For each parameter, noise_std = perturbation_scale * |param_value|
        # Use minimum noise std of 0.01 to avoid zero noise for near-zero params
        param_magnitudes = np.maximum(np.abs(all_initial_params), 0.01)
        noise = rng.normal(0, perturbation_scale, all_initial_params.shape) * param_magnitudes
        
        all_initial_params = jnp.array(np.array(all_initial_params) + noise)
        
        # Log perturbation statistics
        perturbation_pct = 100 * np.abs(noise) / param_magnitudes
        logger.info(f"  Mean perturbation: {np.mean(perturbation_pct):.1f}% of parameter magnitude")
        logger.info(f"  Max perturbation:  {np.max(perturbation_pct):.1f}% of parameter magnitude")
    
    # Stack all inputs and outputs for valid cells
    x_batch = jnp.array(inputs.data[valid_indices])  # Shape: (n_valid, n_features, n_trials)
    y_batch = jnp.array(outputs['firing_rate'][valid_indices])  # Shape: (n_valid, n_trials)
    
    # Run batched optimization
    loss_init, params_init, loss_opt, params_opt, training_trials_idx, test_trials_idx, loss_history = \
        objective_legacy(
            grid_model_2_jax,
            initial_params=all_initial_params,
            loss_func=quadratic_loss,
            x=x_batch,
            y=y_batch,
            fit_params=True,
            param_penalty_weight=0.001,
            max_iter=max_iter,
            verbose=True
        )
    
    logger.info(f"Parallel optimization completed in {time.time() - t_start:.2f}s")
    
    # Compute per-cell losses using raw position data
    logger.info("Computing per-cell losses...")
    per_cell_losses_init = []
    per_cell_losses_opt = []
    
    # Get test indices from objective_legacy return
    # These are the indices for the test split
    
    for i, cell_idx in enumerate(valid_indices):
        x_pos = np.array(inputs['x'][cell_idx])
        y_pos_data = np.array(inputs['y'][cell_idx])
        firing_rate = np.array(outputs['firing_rate'][cell_idx])
        
        # Get test subset using the same indices from optimization
        x_test = x_pos[test_trials_idx]
        y_test = y_pos_data[test_trials_idx]
        fr_test = firing_rate[test_trials_idx]
        
        # Construct X matrix for model
        X_test = jnp.zeros((2, len(x_test)))
        X_test = X_test.at[0].set(x_test)
        X_test = X_test.at[1].set(y_test)
        
        # Compute predictions and losses
        preds_init = grid_model_2_jax(X_test, *params_init[i])
        preds_opt = grid_model_2_jax(X_test, *params_opt[i])
        
        loss_i_init = float(jnp.mean(quadratic_loss(preds_init, fr_test)))
        loss_i_opt = float(jnp.mean(quadratic_loss(preds_opt, fr_test)))
        
        per_cell_losses_init.append(loss_i_init)
        per_cell_losses_opt.append(loss_i_opt)
    
    # =========================================================================
    # 4. Compute loss maps (can be parallelized per cell but kept simple for now)
    # =========================================================================
    logger.info("Computing loss maps...")
    t_start = time.time()
    
    loss_maps_data = {}
    smoothing_sigma_map = 0.0
    
    for i, cell_idx in enumerate(valid_indices):
        x_pos = inputs['x'][cell_idx]
        y_pos = inputs['y'][cell_idx]
        firing_rate = outputs['firing_rate'][cell_idx]
        
        loss_map_init_train, count_map_train = compute_loss_map(
            grid_model_2, params_init[i], x_pos, y_pos, firing_rate,
            smoothing_sigma_map, training_trials_idx, quadratic_loss
        )
        loss_map_init_test, count_map_test = compute_loss_map(
            grid_model_2, params_init[i], x_pos, y_pos, firing_rate,
            smoothing_sigma_map, test_trials_idx, quadratic_loss
        )
        loss_map_opt_train, _ = compute_loss_map(
            grid_model_2, params_opt[i], x_pos, y_pos, firing_rate,
            smoothing_sigma_map, training_trials_idx, quadratic_loss
        )
        loss_map_opt_test, _ = compute_loss_map(
            grid_model_2, params_opt[i], x_pos, y_pos, firing_rate,
            smoothing_sigma_map, test_trials_idx, quadratic_loss
        )
        
        loss_maps_data[cell_idx] = {
            'loss_map_init_train': loss_map_init_train,
            'loss_map_init_test': loss_map_init_test,
            'loss_map_opt_train': loss_map_opt_train,
            'loss_map_opt_test': loss_map_opt_test,
            'count_map_train': count_map_train,
            'count_map_test': count_map_test,
        }
        
        if (i + 1) % 10 == 0 or i == n_valid - 1:
            logger.info(f"  Loss maps: {i+1}/{n_valid} cells done")
    
    logger.info(f"Loss map computation completed in {time.time() - t_start:.2f}s")
    
    # =========================================================================
    # 5. Compute loss landscapes in parallel using vmap
    # =========================================================================
    logger.info(f"Computing loss landscapes ({param1_name} vs {param2_name})...")
    t_start = time.time()
    
    n_grid = 100
    param1_values = jnp.linspace(param1_range[0], param1_range[1], n_grid)
    param2_values = jnp.linspace(param2_range[0], param2_range[1], n_grid)
    P1, P2 = jnp.meshgrid(param1_values, param2_values, indexing='ij')
    
    loss_landscapes = {}
    
    # Vectorized loss landscape computation for all cells
    for i, cell_idx in enumerate(valid_indices):
        x_pos = inputs['x'][cell_idx]
        y_pos = inputs['y'][cell_idx]
        firing_rate = outputs['firing_rate'][cell_idx]
        
        X = np.zeros((2, len(x_pos)))
        X[0] = x_pos
        X[1] = y_pos
        
        base_params = params_init[i]
        
        # Build parameter grid with configurable parameters
        all_params = jnp.broadcast_to(base_params, (n_grid, n_grid, len(base_params))).copy()
        all_params = all_params.at[:, :, param1_idx].set(P1)
        all_params = all_params.at[:, :, param2_idx].set(P2)
        all_params_flat = all_params.reshape(-1, len(base_params))
        
        X_jax = jnp.array(X)
        y_jax = jnp.array(firing_rate)
        
        @jax.jit
        def compute_loss_single(p):
            preds = grid_model_2_jax(X_jax, *p)
            return jnp.mean(quadratic_loss(preds, y_jax))
        
        compute_loss_batched = jax.vmap(compute_loss_single)
        losses_flat = compute_loss_batched(all_params_flat)
        loss_landscape = losses_flat.reshape(n_grid, n_grid)
        
        loss_landscapes[cell_idx] = loss_landscape
        
        if (i + 1) % 10 == 0 or i == n_valid - 1:
            logger.info(f"  Loss landscapes: {i+1}/{n_valid} cells done")
    
    logger.info(f"Loss landscape computation completed in {time.time() - t_start:.2f}s")
    
    # =========================================================================
    # 6. Create visualizations and compile results
    # =========================================================================
    logger.info("Creating visualizations...")
    t_start = time.time()
    
    results_list = []
    
    for i, cell_idx in enumerate(valid_indices):
        x_pos = inputs['x'][cell_idx]
        y_pos = inputs['y'][cell_idx]
        firing_rate = outputs['firing_rate'][cell_idx]
        
        X = np.zeros((2, len(x_pos)))
        X[0] = x_pos
        X[1] = y_pos
        
        loss_i_init = per_cell_losses_init[i]
        loss_i_opt = per_cell_losses_opt[i]
        
        # Main diagnostic plot
        fig_main = create_diagnostic_plot(
            cell_idx, 
            rate_maps_raw[cell_idx], 
            rate_maps_smooth[cell_idx], 
            params_init[i:i+1], 
            params_opt[i:i+1],
            loss_i_init, loss_i_opt, loss_history,
            loss_maps_data[cell_idx]['loss_map_init_train'],
            loss_maps_data[cell_idx]['loss_map_opt_train'],
            loss_maps_data[cell_idx]['loss_map_init_test'],
            loss_maps_data[cell_idx]['loss_map_opt_test'],
            loss_maps_data[cell_idx]['count_map_train'],
            loss_maps_data[cell_idx]['count_map_test'],
            X, x_pos, y_pos, firing_rate, training_trials_idx, test_trials_idx,
            perturbation_scale=perturbation_scale,
            extent_min=-0.75, extent_max=0.75, smoothing_sigma=0.0
        )
        
        # Loss landscape plot
        fig_landscape = create_loss_landscape_plot(
            cell_idx, loss_landscapes[cell_idx], param1_values, param2_values,
            params_init[i:i+1], params_opt[i:i+1], X, firing_rate, grid_model_2,
            param1_name=param1_name, param2_name=param2_name,
            param1_idx=param1_idx, param2_idx=param2_idx
        )
        
        # Save figures
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Add suffix for perturbation test
            suffix = '_perturbed' if perturbation_scale is not None else ''
            if perturbation_scale is not None and perturbation_seed is not None:
                suffix = f'_perturbed_seed{perturbation_seed}'
            
            # Include param names in loss landscape filename
            landscape_suffix = f'_{param1_name}_vs_{param2_name}'
            
            fig_main.savefig(output_path / f'cell_{cell_idx}_diagnostic{suffix}.png', dpi=150, bbox_inches='tight')
            fig_landscape.savefig(output_path / f'cell_{cell_idx}_loss_landscape{landscape_suffix}{suffix}.png', dpi=150, bbox_inches='tight')
        
        if not show_plots:
            plt.close(fig_main)
            plt.close(fig_landscape)
        
        # Compile results
        results = {
            'cell_idx': cell_idx,
            'params_init': np.array(params_init[i]),
            'params_opt': np.array(params_opt[i]),
            'params_before_perturbation': params_before_perturbation[i] if params_before_perturbation is not None else None,
            'perturbation_scale': perturbation_scale,
            'perturbation_seed': perturbation_seed,
            'loss_init': loss_i_init,
            'loss_opt': loss_i_opt,
            'improvement': loss_i_init - loss_i_opt,
            'improvement_pct': (1 - loss_i_opt / loss_i_init) * 100 if loss_i_init > 0 else 0,
            'loss_history': loss_history,
            'training_trials': len(training_trials_idx),
            'test_trials': len(test_trials_idx),
        }
        results_list.append(results)
    
    if output_dir:
        logger.info(f"Figures saved to {output_dir}")
    
    logger.info(f"Visualization completed in {time.time() - t_start:.2f}s")
    
    return results_list


# =============================================================================
# CLI ENTRY POINT
# =============================================================================

def parse_cell_indices(cell_idx_args):
    """
    Parse cell index arguments supporting multiple formats:
    - Single values: 45
    - Multiple values: 45 46 47
    - Ranges: 40-50
    - Mixed: 45 48-52 55
    
    Returns:
        List of cell indices (sorted, unique)
    """
    indices = set()
    for arg in cell_idx_args:
        if '-' in arg and not arg.startswith('-'):
            # Range format: "40-50"
            parts = arg.split('-')
            if len(parts) == 2:
                try:
                    start, end = int(parts[0]), int(parts[1])
                    indices.update(range(start, end + 1))
                except ValueError:
                    raise ValueError(f"Invalid range format: {arg}")
        else:
            # Single value
            try:
                indices.add(int(arg))
            except ValueError:
                raise ValueError(f"Invalid cell index: {arg}")
    return sorted(indices)


def main():
    parser = argparse.ArgumentParser(
        description='Test gradient optimization for grid cell model'
    )
    parser.add_argument(
        '--cell_idx', type=str, nargs='+', default=['45'],
        help='Cell index/indices to analyze. Supports: single (45), multiple (45 46 47), ranges (40-50), or mixed (45 48-52 55)'
    )
    parser.add_argument(
        '--data_path', type=str,
        default='/home/dabin/data/Toroidal_topology_grid_cell_data/rat_q_grid_modules_1_2.npz',
        help='Path to data file'
    )
    parser.add_argument(
        '--output_dir', type=str, default=None,
        help='Directory to save output figures'
    )
    parser.add_argument(
        '--show', action='store_true',
        help='Show plots interactively'
    )
    parser.add_argument(
        '--max_iter', type=int, default=1000,
        help='Maximum optimization iterations (default: 1000)'
    )
    parser.add_argument(
        '--smoothing_sigma', type=float, default=1.5,
        help='Smoothing sigma for rate maps (default: 1.5)'
    )
    parser.add_argument(
        '--perturbation_test', type=float, default=None,
        help='Perturbation scale for initial parameters (e.g., 0.1 for 10%% Gaussian noise). '
             'Tests robustness of gradient descent to initialization.'
    )
    parser.add_argument(
        '--perturbation_seed', type=int, default=None,
        help='Random seed for perturbation (for reproducibility)'
    )
    parser.add_argument(
        '--landscape_params', type=str, nargs=2, default=['amplitude', 'sigma'],
        metavar=('PARAM1', 'PARAM2'),
        help=f'Two parameters to vary in loss landscape. Default: amplitude sigma. '
             f'Valid: {", ".join(PARAM_NAMES)}'
    )
    
    args = parser.parse_args()
    
    # Parse cell indices
    cell_indices = parse_cell_indices(args.cell_idx)
    n_cells = len(cell_indices)
    
    print(f"\n{'='*60}")
    print(f"GRADIENT OPTIMIZATION TEST (BATCH MODE)")
    print(f"{'='*60}")
    print(f"Testing {n_cells} cell(s): {cell_indices}")
    print(f"Loss landscape: {args.landscape_params[0]} vs {args.landscape_params[1]}")
    if args.perturbation_test:
        print(f"Perturbation test: scale={args.perturbation_test}, seed={args.perturbation_seed}")
    print(f"{'='*60}\n")
    
    # Run batch optimization (parallelized)
    try:
        all_results = run_gradient_optimization_batch(
            cell_indices=cell_indices,
            data_path=args.data_path,
            output_dir=args.output_dir,
            show_plots=args.show,
            max_iter=args.max_iter,
            smoothing_sigma=args.smoothing_sigma,
            perturbation_scale=args.perturbation_test,
            perturbation_seed=args.perturbation_seed,
            landscape_params=tuple(args.landscape_params)
        )
    except Exception as e:
        logger.error(f"Batch optimization failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    successful = len([r for r in all_results if 'error' not in r])
    improved = len([r for r in all_results if 'error' not in r and r['improvement'] > 0])
    
    # Print summary
    print("\n" + "=" * 60)
    print("TEST RESULTS SUMMARY")
    print("=" * 60)
    
    if n_cells == 1:
        # Single cell output
        r = all_results[0]
        if 'error' not in r:
            print(f"Cell Index:        {r['cell_idx']}")
            print(f"Initial Loss:      {r['loss_init']:.4f}")
            print(f"Optimized Loss:    {r['loss_opt']:.4f}")
            print(f"Improvement:       {r['improvement']:.4f} ({r['improvement_pct']:.1f}%)")
            print(f"Train/Test Trials: {r['training_trials']}/{r['test_trials']}")
        else:
            print(f"Cell {r['cell_idx']} FAILED: {r['error']}")
    else:
        # Multi-cell summary table
        print(f"\n{'Cell':<8} {'Init Loss':<12} {'Opt Loss':<12} {'Improvement':<15} {'Status':<10}")
        print("-" * 60)
        
        for r in all_results:
            if 'error' not in r:
                status = "✓ Better" if r['improvement'] > 0 else "✗ Worse"
                print(f"{r['cell_idx']:<8} {r['loss_init']:<12.4f} {r['loss_opt']:<12.4f} "
                      f"{r['improvement']:+.4f} ({r['improvement_pct']:+.1f}%)  {status}")
            else:
                print(f"{r['cell_idx']:<8} {'FAILED':<12} {'-':<12} {'-':<15} ✗ Error")
        
        print("-" * 60)
        print(f"\nTotal cells: {n_cells}")
        print(f"Successful:  {successful}/{n_cells} ({100*successful/n_cells:.1f}%)")
        print(f"Improved:    {improved}/{successful} ({100*improved/successful:.1f}% of successful)" if successful > 0 else "")
        
        # Aggregate statistics for successful runs
        valid_results = [r for r in all_results if 'error' not in r]
        if valid_results:
            mean_init = np.mean([r['loss_init'] for r in valid_results])
            mean_opt = np.mean([r['loss_opt'] for r in valid_results])
            mean_improvement = np.mean([r['improvement'] for r in valid_results])
            print(f"\nMean Initial Loss:   {mean_init:.4f}")
            print(f"Mean Optimized Loss: {mean_opt:.4f}")
            print(f"Mean Improvement:    {mean_improvement:+.4f}")
    
    print("=" * 60)
    
    # Show plots for multiple cells if requested
    if args.show and n_cells > 1:
        print("\nShowing plots...")
        plt.show()
    
    # Return success/failure
    return 0 if improved > 0 else 1


if __name__ == '__main__':
    exit(main())
