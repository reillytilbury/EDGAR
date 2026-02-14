"""
Diagnostic plotting functions for grid cell experiments.

This module provides visualization functions for 2D grid cell rate maps,
model fits, and learning curves specific to spatial navigation data.

Key visualizations:
- Rate map grids (with spatial binning info)
- Grid cell filtering diagnostics (UMAP, autocorrelograms, module assignments)
- Speed and occupancy distributions
- Model fits with ground truth rate maps
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Set non-GUI backend before importing pyplot
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import jax.numpy as jnp
from typing import Optional, Callable, Sequence, Tuple, Dict, Any

from src import utils
from src.diagnostics_manager import ModelFitPlotData


def select_evaluation_points(inputs: jnp.ndarray,
                             n_points: int = 100,
                             random_seed: int = 0,
                             **kwargs) -> jnp.ndarray:
    """
    Select evaluation points for grid-cell diagnostics.

    Purpose:
    - Define experiment-specific evaluation-point selection outside hypothesis_engine.
    - Preserve realistic joint `(x, y)` trajectories when evaluating models.

    Strategy:
    - For 3D spatial inputs `(n_samples, 2, n_trials)`, sample `n_points` trial
      columns from observed trajectories using a seeded RNG.
    - For 2D inputs, sample trial indices directly as a fallback.
    """
    x_arr = jnp.asarray(inputs)
    n_trials = int(x_arr.shape[-1])
    n_eval = min(int(n_points), n_trials)
    rng = np.random.default_rng(random_seed)
    trial_idx = rng.choice(n_trials, size=n_eval, replace=False)

    if x_arr.ndim == 3:
        return x_arr[:, :, trial_idx]
    if x_arr.ndim == 2:
        return x_arr[:, trial_idx]
    raise ValueError(f"Expected 2D or 3D inputs, got shape {x_arr.shape}.")
# =============================================================================
# Data loading diagnostics
# =============================================================================

def plot_data_summary(data_dict: Dict[str, Any],
                      n_cells_to_show: int = 9,
                      save_path: Optional[str] = None,
                      dpi: float = 100.0):
    """
    Comprehensive summary plot of loaded grid cell data.
    
    Creates a multi-panel figure showing:
    - Sample rate maps
    - Occupancy map
    - Speed distribution (if available)
    - Firing rate distribution
    
    Args:
        data_dict: Dictionary returned by load_and_process_data.
        n_cells_to_show: Number of sample cells to display.
        save_path: Path to save figure.
    """
    response = data_dict['response']
    rate_maps = data_dict['rate_maps']
    position_data = data_dict.get('position_data', {})
    grid_filter_info = data_dict.get('grid_filter_info', None)
    
    n_cells = response.shape[0]
    n_time_bins = response.shape[1]
    n_spatial_bins = rate_maps.shape[1] if len(rate_maps) > 0 else 50
    
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 4, figure=fig, height_ratios=[1.2, 1, 1])
    
    # --- Row 0: Sample rate maps ---
    n_show = min(n_cells_to_show, n_cells, 4)
    for i in range(n_show):
        ax = fig.add_subplot(gs[0, i])
        im = ax.imshow(rate_maps[i].T, origin='lower', extent=[-1, 1, -1, 1],
                       cmap='viridis', aspect='equal')
        ax.set_title(f'Cell {i}', fontsize=10)
        ax.set_xlabel('X (norm)')
        ax.set_ylabel('Y (norm)')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Hz')
    
    # --- Row 1, Col 0-1: Occupancy map ---
    ax_occ = fig.add_subplot(gs[1, 0:2])
    if 'x' in position_data and 'y' in position_data:
        x_norm = position_data['x']
        y_norm = position_data['y']
        occupancy, _, _ = np.histogram2d(x_norm, y_norm, bins=n_spatial_bins, 
                                          range=[[-1, 1], [-1, 1]])
        time_bin_ms = position_data.get('time_bin_ms', 10)
        occupancy_seconds = occupancy * (time_bin_ms / 1000)
        
        im = ax_occ.imshow(occupancy_seconds.T, origin='lower', extent=[-1, 1, -1, 1],
                           cmap='hot', aspect='equal')
        ax_occ.set_title(f'Occupancy ({n_time_bins} time bins)', fontsize=12)
        ax_occ.set_xlabel('X (normalized)')
        ax_occ.set_ylabel('Y (normalized)')
        plt.colorbar(im, ax=ax_occ, label='Time (s)')
    else:
        ax_occ.text(0.5, 0.5, 'No position data', ha='center', va='center')
    
    # --- Row 1, Col 2-3: Firing rate distribution ---
    ax_fr = fig.add_subplot(gs[1, 2:4])
    mean_fr = response.mean(axis=1)
    ax_fr.hist(mean_fr, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
    ax_fr.axvline(np.median(mean_fr), color='red', linestyle='--', 
                  label=f'Median: {np.median(mean_fr):.2f} Hz')
    ax_fr.set_xlabel('Mean Firing Rate (Hz)')
    ax_fr.set_ylabel('Count')
    ax_fr.set_title(f'Firing Rate Distribution (n={n_cells} cells)', fontsize=12)
    ax_fr.legend()
    
    # --- Row 2, Col 0-1: Trajectory ---
    ax_traj = fig.add_subplot(gs[2, 0:2])
    if 'x' in position_data and 'y' in position_data:
        x_norm = position_data['x']
        y_norm = position_data['y']
        # Subsample for faster plotting
        step = max(1, len(x_norm) // 5000)
        ax_traj.plot(x_norm[::step], y_norm[::step], 'k-', alpha=0.3, linewidth=0.5)
        ax_traj.scatter(x_norm[0], y_norm[0], c='green', s=50, zorder=5, label='Start')
        ax_traj.scatter(x_norm[-1], y_norm[-1], c='red', s=50, zorder=5, label='End')
        ax_traj.set_xlim(-1.05, 1.05)
        ax_traj.set_ylim(-1.05, 1.05)
        ax_traj.set_aspect('equal')
        ax_traj.set_title(f'Animal Trajectory ({n_time_bins} bins)', fontsize=12)
        ax_traj.set_xlabel('X (normalized)')
        ax_traj.set_ylabel('Y (normalized)')
        ax_traj.legend(loc='upper right')
    else:
        ax_traj.text(0.5, 0.5, 'No trajectory data', ha='center', va='center')
    
    # --- Row 2, Col 2-3: Summary statistics ---
    ax_stats = fig.add_subplot(gs[2, 2:4])
    ax_stats.axis('off')
    
    stats_text = [
        f"Data Summary",
        f"─" * 30,
        f"Number of cells: {n_cells}",
        f"Number of time bins: {n_time_bins}",
        f"Spatial bins: {n_spatial_bins} × {n_spatial_bins}",
        f"Time bin size: {position_data.get('time_bin_ms', 'N/A')} ms",
        f"",
        f"Firing rates:",
        f"  Mean: {mean_fr.mean():.3f} Hz",
        f"  Median: {np.median(mean_fr):.3f} Hz",
        f"  Max: {mean_fr.max():.3f} Hz",
        f"  Min: {mean_fr.min():.3f} Hz",
    ]
    
    if grid_filter_info is not None:
        stats_text.extend([
            f"",
            f"Grid cell filtering applied:",
            f"  Cells identified: {len(data_dict['response'])}",
        ])
    
    ax_stats.text(0.1, 0.95, '\n'.join(stats_text), transform=ax_stats.transAxes,
                  fontsize=11, verticalalignment='top', fontfamily='monospace',
                  bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    else:
        plt.show()
    plt.close(fig)


def plot_grid_filter_diagnostics(grid_filter_info: Dict[str, Any],
                                  save_path: Optional[str] = None,
                                  dpi: float = 100.0):
    """
    Plot diagnostics for grid cell filtering pipeline.
    
    Shows:
    - UMAP embedding with cluster labels
    - Sample autocorrelograms for grid vs non-grid cells
    - Module assignments
    - Filtering statistics
    
    Args:
        grid_filter_info: Dictionary from load_and_process_data['grid_filter_info'].
        save_path: Path to save figure.
    """
    if grid_filter_info is None:
        print("No grid filter info available (filter_grid_cells=False)")
        return
    
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 3, figure=fig)
    
    # --- UMAP embedding with cluster labels ---
    ax_umap = fig.add_subplot(gs[0, 0:2])
    embedding = grid_filter_info.get('umap_embedding')
    cluster_labels = grid_filter_info.get('cluster_labels')
    
    if embedding is not None and cluster_labels is not None:
        unique_labels = np.unique(cluster_labels)
        cmap = plt.cm.get_cmap('tab10')
        
        for label in unique_labels:
            mask = cluster_labels == label
            if label == -1:
                ax_umap.scatter(embedding[mask, 0], embedding[mask, 1],
                               c='gray', s=20, alpha=0.5, label='Noise')
            else:
                ax_umap.scatter(embedding[mask, 0], embedding[mask, 1],
                               c=[cmap(label % 10)], s=30, alpha=0.7,
                               label=f'Cluster {label} (n={mask.sum()})')
        
        ax_umap.set_xlabel('UMAP 1')
        ax_umap.set_ylabel('UMAP 2')
        ax_umap.set_title('UMAP Embedding of Autocorrelograms')
        ax_umap.legend(loc='upper right', fontsize=8)
    else:
        ax_umap.text(0.5, 0.5, 'No UMAP data', ha='center', va='center')
    
    # --- Filtering statistics ---
    ax_stats = fig.add_subplot(gs[0, 2])
    ax_stats.axis('off')
    
    fr_mask = grid_filter_info.get('firing_rate_mask')
    hd_mask = grid_filter_info.get('hd_tuning_mask')
    grid_indices = grid_filter_info.get('grid_cell_indices', np.array([]))
    
    n_total = len(fr_mask) if fr_mask is not None else 0
    n_pass_fr = fr_mask.sum() if fr_mask is not None else 0
    n_pass_hd = hd_mask.sum() if hd_mask is not None else 0
    n_grid = len(grid_indices)
    
    stats_text = [
        "Grid Cell Filtering Summary",
        "─" * 30,
        f"Total cells: {n_total}",
        f"Pass FR filter: {n_pass_fr}",
        f"Pass HD filter: {n_pass_hd}",
        f"Final grid cells: {n_grid}",
        "",
        "Pipeline:",
        "1. Firing rate filter",
        "2. Coarse rate maps (10cm)",
        "3. Autocorrelograms",
        "4. UMAP embedding",
        "5. DBSCAN clustering",
        "6. Exclude non-grid cluster",
        "7. HD tuning exclusion",
    ]
    
    ax_stats.text(0.05, 0.95, '\n'.join(stats_text), transform=ax_stats.transAxes,
                  fontsize=10, verticalalignment='top', fontfamily='monospace',
                  bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    # --- Sample autocorrelograms ---
    autocorrs = grid_filter_info.get('autocorrelograms')
    if autocorrs is not None and len(autocorrs) > 0:
        # Show up to 6 autocorrelograms
        n_show = min(6, len(autocorrs))
        for i in range(n_show):
            row = 1
            col = i % 3
            if i >= 3:
                continue  # Only show first row of autocorrs
            
            ax = fig.add_subplot(gs[1, col])
            autocorr = autocorrs[i]
            
            # Reshape if flattened
            side = int(np.sqrt(len(autocorr)))
            if side * side == len(autocorr):
                autocorr = autocorr.reshape(side, side)
            
            im = ax.imshow(autocorr, origin='lower', cmap='RdBu_r', aspect='equal')
            ax.set_title(f'Autocorr Cell {i}', fontsize=10)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    else:
        plt.show()
    plt.close(fig)


def plot_rate_map_comparison(rate_maps: np.ndarray,
                             position_data: Dict[str, Any],
                             cell_indices: Optional[Sequence[int]] = None,
                             n_cols: int = 4,
                             title: str = 'Rate Maps',
                             save_path: Optional[str] = None,
                             dpi: float = 100.0):
    """
    Plot rate maps with spatial scale information.
    
    Args:
        rate_maps: Rate maps of shape (n_cells, n_bins, n_bins).
        position_data: Dictionary with spatial binning info.
        cell_indices: Which cells to plot. If None, plots first 16.
        n_cols: Number of columns in the grid.
        title: Plot title.
        save_path: Path to save figure.
    """
    if cell_indices is None:
        cell_indices = list(range(min(16, len(rate_maps))))
    
    n_cells = len(cell_indices)
    n_rows = int(np.ceil(n_cells / n_cols))
    
    n_spatial_bins = position_data.get('n_spatial_bins', rate_maps.shape[1])
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.5 * n_cols, 3 * n_rows))
    axes = np.atleast_2d(axes)
    
    # Compute global color limits for consistency
    vmin = np.percentile(rate_maps[cell_indices], 1)
    vmax = np.percentile(rate_maps[cell_indices], 99)
    
    for idx, cell_idx in enumerate(cell_indices):
        row, col = divmod(idx, n_cols)
        ax = axes[row, col]
        
        rate_map = rate_maps[cell_idx]
        peak_rate = rate_map.max()
        
        im = ax.imshow(rate_map.T, origin='lower', extent=[-1, 1, -1, 1],
                       cmap='viridis', aspect='equal', vmin=vmin, vmax=vmax)
        ax.set_title(f'Cell {cell_idx}\nPeak: {peak_rate:.1f} Hz', fontsize=9)
        ax.set_xticks([-1, 0, 1])
        ax.set_yticks([-1, 0, 1])
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # Hide unused subplots
    for idx in range(n_cells, n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axes[row, col].axis('off')
    
    plt.suptitle(f'{title} ({n_spatial_bins}×{n_spatial_bins} bins)', fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    else:
        plt.show()
    plt.close(fig)


# =============================================================================
# Model fitting diagnostics
# =============================================================================

def _ensure_input_format(x_cell: jnp.ndarray) -> jnp.ndarray:
    """Convert 1D/2D stimulus array to 2D input format (n_features, n_trials)."""
    if x_cell.ndim == 1:
        return x_cell.reshape(1, -1)  # (n_trials,) -> (1, n_trials)
    return x_cell  # already (n_features, n_trials)


def compute_evaluation_matrix(program: callable,
                              params: jnp.ndarray,
                              eval_points: np.ndarray,
                              n_features: int = 2,
                              **kwargs) -> jnp.ndarray:
    """
    Computes the evaluation matrix for a given program and parameters.
    
    Evaluate the model at explicit eval_points.

    Args:
        program (callable): The neuron model function with signature:
                           program(X, *params) -> (n_trials,)
                           where X has shape (n_features, n_trials).
        params (jnp.ndarray): The parameters for the neuron model. Shape: (n_samples, n_params)
        eval_points (np.ndarray): Evaluation points with shape (n_samples, n_features, n_eval).
        n_features (int): Total number of inputs in the model. Default is 2.
    Returns:
        jnp.ndarray: The evaluation matrix of shape (n_samples, n_eval).
    """
    if n_features != eval_points.shape[1]:
        raise ValueError(
            f"eval_points must have shape (n_samples, n_features, n_trials). "
            f"Got {eval_points.shape} with n_features={n_features}."
        )
    if eval_points.shape[0] != params.shape[0]:
        raise ValueError(
            f"eval_points first dimension must match params ({params.shape[0]}), got {eval_points.shape}."
        )
    X_eval = jnp.asarray(eval_points)

    # vmap over samples
    program_vmap = utils.vmap_over_cells(program)
    y_eval = program_vmap(X_eval, params)
    return y_eval

def plot_model_fits_old(programs_df: pd.DataFrame, loss_function: Callable,
                    inputs: jnp.ndarray, response: jnp.ndarray,
                    sample_selection: Sequence[int],
                    rate_maps: Optional[np.ndarray] = None,
                    n_eval: int = 50,
                    colours: list = ["#FDC91E", "#15AC15", '#EB2B2C'],
                    labels: Optional[list] = None,
                    title: str = '',
                    line_width=4.0,
                    line_alpha=1.0,
                    point_alpha=0.1,
                    point_size: int = 80,
                    legend_fontsize: int = 12,
                    dpi: float = 100.0,
                    save_path: Optional[str] = None,
                    input_idx: int = 0):
    """
    Plot 2D rate map fits for grid cell models.
    
    For grid cells, we show actual rate maps vs model predictions in 2D.
    Uses precomputed rate maps from data_parser when available.
    
    Args:
        programs_df: DataFrame with 'program' and 'params' columns.
        loss_function: Loss function (y_est, y_true) -> loss.
        inputs: Input data of shape (n_cells, n_features, n_trials) where n_features=2 (x, y).
        response: Response data of shape (n_cells, n_trials).
        rate_maps: Precomputed rate maps of shape (n_cells, n_bins, n_bins). If None,
            rate maps are computed from inputs, response.
        sample_selection: Indices of cells to plot.
        n_eval: Number of evaluation points per dimension for model prediction grid.
        save_path: Path to save figure.
        input_idx: Ignored for grid cells (uses both x and y).
    """
    assert len(programs_df) <= 3, f"programs_df must have at most 3 rows, got {len(programs_df)}"
    assert len(sample_selection) > 0, "sample_selection must not be empty"
    
    n_cells_plot = len(sample_selection)
    n_models = len(programs_df)
    
    models = programs_df['program'].tolist()
    params = programs_df['params'].tolist()
    sample_idx = jnp.array(sample_selection)
    
    # Subset params and data for selected cells
    params = [p[sample_idx] for p in params]
    response_subset = response[sample_idx]
    inputs_subset = inputs[sample_idx]  # (n_cells_plot, n_features, n_trials)
    
    # Subset rate maps if provided
    rate_maps_subset = rate_maps[sample_idx] if rate_maps is not None else None
    
    if labels is None:
        labels = [f'Model {i+1}' for i in range(n_models)]
    
    # Create evaluation grid
    eval_pts = np.linspace(-1, 1, n_eval)
    X_grid, Y_grid = np.meshgrid(eval_pts, eval_pts, indexing='xy')
    X_eval = jnp.stack([X_grid.ravel(), Y_grid.ravel()], axis=0)  # (2, n_eval^2)
    
    # Figure layout: rows = cells, cols = [data, model1, model2, ...]
    n_cols = 1 + n_models
    fig, axes = plt.subplots(n_cells_plot, n_cols, figsize=(4 * n_cols, 4 * n_cells_plot))
    if n_cells_plot == 1:
        axes = axes.reshape(1, -1)
    
    for c_idx, c in enumerate(range(n_cells_plot)):
        cell_idx = sample_selection[c_idx]
        
        # Use precomputed rate map if available, otherwise compute from data
        if rate_maps_subset is not None:
            data_rate_map = rate_maps_subset[c_idx]
            # Resize to n_eval if needed
            if data_rate_map.shape[0] != n_eval:
                from scipy.ndimage import zoom
                zoom_factor = n_eval / data_rate_map.shape[0]
                data_rate_map = zoom(data_rate_map, zoom_factor, order=1)
        else:
            x_pos = np.array(inputs_subset[c_idx, 0, :])
            y_pos = np.array(inputs_subset[c_idx, 1, :])
            response_cell = np.array(response_subset[c_idx])
            data_rate_map = _bin_to_rate_map(x_pos, y_pos, response_cell, n_bins=n_eval)
        
        peak_rate = data_rate_map.max()
        
        # Plot actual data rate map
        ax = axes[c_idx, 0]
        im = ax.imshow(data_rate_map.T, origin='lower', extent=[-1, 1, -1, 1],
                       cmap='viridis', aspect='equal')
        ax.set_title(f'Cell {cell_idx} - Data\nPeak: {peak_rate:.1f} Hz', fontsize=10)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        # Plot model predictions
        for m_idx, model in enumerate(models):
            params_c = params[m_idx][c]
            
            # Evaluate model on grid
            model_output = model(X_eval, *params_c)  # (n_eval^2,)
            model_map = np.array(model_output).reshape(n_eval, n_eval)
            
            # Compute loss
            X_cell = inputs_subset[c, :, :]  # (n_features, n_trials)
            pred = model(X_cell, *params_c)
            loss_val = float(jnp.mean(loss_function(pred, response_subset[c])))
            
            ax = axes[c_idx, m_idx + 1]
            im = ax.imshow(model_map.T, origin='lower', extent=[-1, 1, -1, 1],
                           cmap='viridis', aspect='equal')
            ax.set_title(f'{labels[m_idx]} (loss: {loss_val:.3f})')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    if save_path:
        plt.savefig(save_path, dpi=dpi)
    else:
        plt.show()
    plt.close(fig)


def plot_model_fits(plot_data: ModelFitPlotData,
                    n_eval: int = 50,
                    colours: list = ["#FDC91E", "#15AC15", '#EB2B2C'],
                    labels: Optional[list] = None,
                    title: str = '',
                    line_width=4.0,
                    line_alpha=1.0,
                    point_alpha=0.1,
                    point_size: int = 80,
                    legend_fontsize: int = 12,
                    dpi: float = 100.0,
                    save_path: Optional[str] = None,
                    smoothing_sigma: float = 1.5,
                    plot_raw_rates: bool = True):
    """
    Plot 2D rate map fits for grid cell models with detailed diagnostics.
    
    Shows for each cell:
    - Unsmoothed actual rate map (if plot_raw_rates=True)
    - Smoothed actual rate map
    - For each model: predicted rate map and overlay (red=smoothed actual, green=pred)
    
    This creates a layout with 2 + 2*n_models columns:
    [Unsmoothed Actual | Smoothed Actual | Model1 Pred | Model1 Overlay | Model2 Pred | Model2 Overlay | ...]
    
    Args:
        plot_data: Precomputed plotting tensors from
            hypothesis_engine.prepare_model_fit_plot_data(...).
        n_eval: Number of bins for rate map computation (default 50).
        colours: Colors for each model (unused, kept for signature compatibility).
        labels: Labels for each model. If None, uses 'Model 1', 'Model 2', etc.
        title: Plot title.
        line_width: Unused, kept for signature compatibility.
        line_alpha: Unused, kept for signature compatibility.
        point_alpha: Unused, kept for signature compatibility.
        point_size: Unused, kept for signature compatibility.
        legend_fontsize: Unused, kept for signature compatibility.
        dpi: Figure DPI for saving.
        save_path: Path to save figure. If None, displays the figure.
        smoothing_sigma: Gaussian smoothing sigma for smoothed rate maps.
        plot_raw_rates: Whether to plot the unsmoothed actual rate maps.
    """
    sample_selection = np.asarray(plot_data['sample_selection'])
    inputs_subset = jnp.asarray(plot_data['stimuli_3d'])        # (n_cells, n_features, n_trials)
    response_subset = jnp.asarray(plot_data['spike_matrix'])    # (n_cells, n_trials)
    trial_predictions = jnp.asarray(plot_data['trial_predictions'])  # (n_models, n_cells, n_trials)
    point_losses = jnp.asarray(plot_data['point_losses'])       # (n_models, n_cells, n_trials)

    n_cells_plot = int(plot_data['n_cells'])
    n_models = int(plot_data['n_models'])
    if inputs_subset.ndim != 3 or inputs_subset.shape[1] < 2:
        raise ValueError(
            f"Grid-cell plot_model_fits requires 2D position inputs; got stimuli_3d shape {inputs_subset.shape}."
        )
    if sample_selection.shape[0] != n_cells_plot:
        raise ValueError(
            f"sample_selection length ({sample_selection.shape[0]}) must equal n_cells ({n_cells_plot})."
        )
    
    if labels is None:
        labels = [f'Model {i+1}' for i in range(n_models)]
    
    # Figure layout: rows = cells, cols = [unsmoothed, smoothed, (pred, overlay) per model]
    n_cols = 2 + 2 * n_models if plot_raw_rates else 1 + 2 * n_models
    fig, axes = plt.subplots(n_cells_plot, n_cols, figsize=(4 * n_cols, 4 * n_cells_plot))
    if n_cells_plot == 1:
        axes = axes.reshape(1, -1)
    
    for c_idx in range(n_cells_plot):
        cell_idx = sample_selection[c_idx]
        
        # Get position data for this cell
        x_pos = np.array(inputs_subset[c_idx, 0, :])
        y_pos = np.array(inputs_subset[c_idx, 1, :])
        response_cell = np.array(response_subset[c_idx])
        
        # Compute unsmoothed and smoothed actual rate maps
        unsmoothed_rate_map = _bin_to_rate_map(x_pos, y_pos, response_cell, 
                                                n_bins=n_eval, smoothing_sigma=0.0)
        smoothed_rate_map = _bin_to_rate_map(x_pos, y_pos, response_cell, 
                                              n_bins=n_eval, smoothing_sigma=smoothing_sigma)
        
        if plot_raw_rates:
            # Column 0: Unsmoothed actual rate map
            ax = axes[c_idx, 0]
            im = ax.imshow(unsmoothed_rate_map.T, origin='lower', extent=[-1, 1, -1, 1],
                        cmap='viridis', aspect='equal')
            ax.set_title(f'Unsmoothed Actual (cell {cell_idx})')
            ax.set_xlabel('X Position (normalized)')
            ax.set_ylabel('Y Position (normalized)')
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        # Column 1: Smoothed actual rate map
        ax = axes[c_idx, 1 if plot_raw_rates else 0]
        im = ax.imshow(smoothed_rate_map.T, origin='lower', extent=[-1, 1, -1, 1],
                       cmap='viridis', aspect='equal')
        ax.set_title(f'Smoothed Actual (sigma={smoothing_sigma}, cell {cell_idx})')
        ax.set_xlabel('X Position (normalized)')
        ax.set_ylabel('Y Position (normalized)')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        # For each model: prediction and overlay
        for m_idx in range(n_models):
            pred_np = np.asarray(trial_predictions[m_idx, c_idx])
            loss_val = float(jnp.mean(point_losses[m_idx, c_idx]))
            
            # Bin model predictions into rate map (no smoothing for model predictions)
            pred_rate_map = _bin_to_rate_map(x_pos, y_pos, pred_np, 
                                             n_bins=n_eval, smoothing_sigma=0.0)
            
            # Column for model prediction
            pred_col = (2 if plot_raw_rates else 1) + m_idx * 2
            ax = axes[c_idx, pred_col]
            im = ax.imshow(pred_rate_map.T, origin='lower', extent=[-1, 1, -1, 1],
                           cmap='viridis', aspect='equal')
            ax.set_title(f'Predicted | Loss: {loss_val:.4f}')
            ax.set_xlabel('X Position (normalized)')
            ax.set_ylabel('Y Position (normalized)')
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            
            # Column for overlay (red=smoothed actual, green=pred)
            overlay_col = (2 if plot_raw_rates else 1) + m_idx * 2 + 1
            ax = axes[c_idx, overlay_col]
            
            # Normalize both maps to shared scale for overlay
            vmin = 0.0
            vmax = max(float(np.nanmax(smoothed_rate_map)), float(np.nanmax(pred_rate_map)))
            vmax = max(vmax, 1e-8)  # avoid divide-by-zero
            
            def scale_shared(m):
                m = np.asarray(m, dtype=float)
                m = (m - vmin) / (vmax - vmin)
                return np.clip(m, 0.0, 1.0)
            
            actual_s = scale_shared(smoothed_rate_map).T
            pred_s = scale_shared(pred_rate_map).T
            
            # Create RGB overlay: red=actual, green=pred
            overlay = np.zeros((*actual_s.shape, 3))
            overlay[..., 0] = actual_s  # red channel
            overlay[..., 1] = pred_s    # green channel
            
            ax.imshow(overlay, origin='lower', extent=[-1, 1, -1, 1])
            ax.set_title(f'Overlay (red=smoothed actual, green=pred), vmax={vmax:.2f}')
            ax.set_xlabel('X Position (normalized)')
            ax.set_ylabel('Y Position (normalized)')
    
    if title:
        plt.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95] if title else None)
    
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    else:
        plt.show()
    plt.close(fig)

def plot_rate_maps(rate_maps: np.ndarray, 
                   sample_selection: Optional[Sequence[int]] = None,
                   title: str = 'Rate Maps',
                   position_data: Optional[Dict[str, Any]] = None,
                   show_peak_rate: bool = True,
                   save_path: Optional[str] = None,
                   dpi: float = 100.0):
    """
    Plot 2D rate maps for selected cells.
    
    Args:
        rate_maps: Rate maps of shape (n_cells, n_bins, n_bins).
        sample_selection: Indices of cells to plot. If None, plots first 16.
        title: Plot title.
        position_data: Optional dict with spatial binning info.
        show_peak_rate: Whether to display peak firing rate in title.
        save_path: Path to save figure.
    """
    if sample_selection is None:
        sample_selection = list(range(min(16, len(rate_maps))))
    
    n_cells = len(sample_selection)
    n_cols = int(np.ceil(np.sqrt(n_cells)))
    n_rows = int(np.ceil(n_cells / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows))
    axes = np.atleast_2d(axes)
    
    # Consistent color scale across cells
    vmin = np.percentile([rate_maps[i] for i in sample_selection], 1)
    vmax = np.percentile([rate_maps[i] for i in sample_selection], 99)
    
    for idx, cell_idx in enumerate(sample_selection):
        row, col = divmod(idx, n_cols)
        ax = axes[row, col]
        
        rate_map = rate_maps[cell_idx]
        peak_rate = rate_map.max()
        
        im = ax.imshow(rate_map.T, origin='lower', extent=[-1, 1, -1, 1],
                       cmap='viridis', aspect='equal', vmin=vmin, vmax=vmax)
        
        if show_peak_rate:
            ax.set_title(f'Cell {cell_idx}\nPeak: {peak_rate:.1f} Hz', fontsize=9)
        else:
            ax.set_title(f'Cell {cell_idx}', fontsize=9)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # Hide unused subplots
    for idx in range(n_cells, n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axes[row, col].axis('off')
    
    # Add spatial bin info to title if available
    n_spatial_bins = rate_maps.shape[1] if len(rate_maps) > 0 else 0
    full_title = f'{title} ({n_spatial_bins}×{n_spatial_bins} bins)'
    
    plt.suptitle(full_title, fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    else:
        plt.show()
    plt.close(fig)


def _bin_to_rate_map(x: np.ndarray, y: np.ndarray, values: np.ndarray,
                     n_bins: int = 50, extent: Tuple[float, float, float, float] = (-1, 1, -1, 1),
                     smoothing_sigma: float = 1.5
                     ) -> np.ndarray:
    """
    Bin scattered data into a 2D rate map with Gaussian smoothing.
    
    Args:
        x, y: Position coordinates.
        values: Values at each position (e.g., firing rates).
        n_bins: Number of bins per dimension.
        extent: (xmin, xmax, ymin, ymax).
        smoothing_sigma: Gaussian smoothing sigma in bins.
    
    Returns:
        rate_map: 2D binned average of values, smoothed.
    """
    from scipy.ndimage import gaussian_filter
    
    xmin, xmax, ymin, ymax = extent
    
    # Bin indices
    bin_x = np.clip(((x - xmin) / (xmax - xmin) * n_bins).astype(int), 0, n_bins - 1)
    bin_y = np.clip(((y - ymin) / (ymax - ymin) * n_bins).astype(int), 0, n_bins - 1)
    
    # Sum and count per bin
    spike_map = np.zeros((n_bins, n_bins))
    occupancy = np.zeros((n_bins, n_bins))
    
    for i in range(len(x)):
        spike_map[bin_x[i], bin_y[i]] += values[i]
        occupancy[bin_x[i], bin_y[i]] += 1
    
    # Smooth both spike map and occupancy before dividing
    spike_map_smooth = gaussian_filter(spike_map, sigma=smoothing_sigma)
    occupancy_smooth = gaussian_filter(occupancy, sigma=smoothing_sigma)
    
    # Compute rate map as smoothed spikes / smoothed occupancy
    rate_map = spike_map_smooth / (occupancy_smooth + 1e-6)
    
    return rate_map


# =============================================================================
# Convenience functions
# =============================================================================

def run_data_diagnostics(data_dict: Dict[str, Any],
                         output_dir: str = '.',
                         prefix: str = 'diagnostic',
                         dpi: float = 100.0):
    """
    Run all data diagnostics and save plots.
    
    Args:
        data_dict: Dictionary returned by load_and_process_data.
        output_dir: Directory to save plots.
        prefix: Prefix for output file names.
        dpi: Figure resolution.
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Data summary
    print(f"Generating data summary plot...")
    plot_data_summary(
        data_dict,
        save_path=os.path.join(output_dir, f'{prefix}_data_summary.png'),
        dpi=dpi
    )
    
    # 2. Grid filter diagnostics (if available)
    grid_filter_info = data_dict.get('grid_filter_info')
    if grid_filter_info is not None:
        print(f"Generating grid filter diagnostics...")
        plot_grid_filter_diagnostics(
            grid_filter_info,
            save_path=os.path.join(output_dir, f'{prefix}_grid_filter.png'),
            dpi=dpi
        )
    
    # 3. Rate maps for all cells (in batches)
    rate_maps = data_dict.get('rate_maps')
    if rate_maps is not None and len(rate_maps) > 0:
        n_cells = len(rate_maps)
        batch_size = 16
        
        for batch_idx, start_idx in enumerate(range(0, n_cells, batch_size)):
            end_idx = min(start_idx + batch_size, n_cells)
            cell_indices = list(range(start_idx, end_idx))
            
            print(f"Generating rate maps for cells {start_idx}-{end_idx-1}...")
            plot_rate_maps(
                rate_maps,
                sample_selection=cell_indices,
                title=f'Rate Maps (Cells {start_idx}-{end_idx-1})',
                position_data=data_dict.get('position_data'),
                save_path=os.path.join(output_dir, f'{prefix}_rate_maps_batch{batch_idx}.png'),
                dpi=dpi
            )
    
    print(f"Diagnostics saved to {output_dir}/")


def quick_data_check(data_dict: Dict[str, Any]) -> None:
    """
    Print a quick summary of loaded data for verification.
    
    Args:
        data_dict: Dictionary returned by load_and_process_data.
    """
    response = data_dict.get('response')
    inputs = data_dict.get('inputs')
    rate_maps = data_dict.get('rate_maps')
    position_data = data_dict.get('position_data', {})
    grid_filter_info = data_dict.get('grid_filter_info')
    
    print("=" * 50)
    print("Data Loading Summary")
    print("=" * 50)
    
    if response is not None:
        n_cells, n_time_bins = response.shape
        print(f"Response: {n_cells} cells × {n_time_bins} time bins")
        print(f"  Mean FR: {response.mean():.3f} Hz")
        print(f"  Max FR:  {response.max():.3f} Hz")
    
    if inputs is not None:
        print(f"Inputs: {inputs.data.shape}")
        print(f"  Names: {inputs.names}")
    
    if rate_maps is not None:
        print(f"Rate maps: {rate_maps.shape}")
        print(f"  Spatial bins: {rate_maps.shape[1]}×{rate_maps.shape[2]}")
    
    if position_data:
        print(f"Position data keys: {list(position_data.keys())}")
        if 'time_bin_ms' in position_data:
            print(f"  Time bin: {position_data['time_bin_ms']} ms")
        if 'n_spatial_bins' in position_data:
            print(f"  Spatial bins: {position_data['n_spatial_bins']}×{position_data['n_spatial_bins']}")
    
    if grid_filter_info is not None:
        print(f"Grid cell filtering: Applied")
        if 'grid_cell_indices' in grid_filter_info:
            print(f"  Grid cells identified: {len(grid_filter_info['grid_cell_indices'])}")
    else:
        print(f"Grid cell filtering: Not applied")
    
    print("=" * 50)
