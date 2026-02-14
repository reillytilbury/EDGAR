"""
Diagnostic plotting functions for place cell experiments.

This module provides visualization functions for 2D place cell rate maps,
model fits, and learning curves specific to spatial navigation data.

Key visualizations:
- Rate map grids (with spatial binning info)
- Place cell filtering diagnostics (spatial information, peak rates)
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
    Select evaluation points for place-cell diagnostics.

    Purpose:
    - Keep evaluation-point policy experiment-local rather than hardcoded in engine.
    - Evaluate models on realistic `(x, y)` positions sampled from observed data.

    Strategy:
    - For 3D spatial inputs `(n_samples, 2, n_trials)`, sample `n_points` trial
      columns from observed trajectories with a seeded RNG.
    - For 2D inputs, sample trials directly as a fallback.
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
    Comprehensive summary plot of loaded place cell data.
    
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
    place_filter_info = data_dict.get('place_filter_info', None)
    
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
        f"-" * 30,
        f"Number of cells: {n_cells}",
        f"Number of time bins: {n_time_bins}",
        f"Spatial bins: {n_spatial_bins} x {n_spatial_bins}",
        f"Time bin size: {position_data.get('time_bin_ms', 'N/A')} ms",
        f"",
        f"Firing rates:",
        f"  Mean: {mean_fr.mean():.3f} Hz",
        f"  Median: {np.median(mean_fr):.3f} Hz",
        f"  Max: {mean_fr.max():.3f} Hz",
        f"  Min: {mean_fr.min():.3f} Hz",
    ]
    
    if place_filter_info is not None:
        stats_text.extend([
            f"",
            f"Place cell filtering applied:",
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


def plot_place_filter_diagnostics(place_filter_info: Dict[str, Any],
                                  save_path: Optional[str] = None,
                                  dpi: float = 100.0):
    """
    Plot diagnostics for place cell filtering.

    Shows:
    - Spatial information distribution
    - Peak rate distribution
    - Mean rate distribution
    - Spatial info vs peak rate scatter

    Args:
        place_filter_info: Dictionary from load_and_process_data['place_filter_info'].
        save_path: Path to save figure.
    """
    if place_filter_info is None:
        print("No place filter info available (filter_place_cells=False)")
        return

    spatial_info = place_filter_info.get('spatial_info')
    peak_rate = place_filter_info.get('peak_rate')
    mean_rate = place_filter_info.get('mean_rate')

    if spatial_info is None or peak_rate is None or mean_rate is None:
        print("Place filter info is missing required fields.")
        return

    info_thresh = place_filter_info.get('spatial_info_threshold', None)
    peak_thresh = place_filter_info.get('peak_rate_threshold', None)
    mean_thresh = place_filter_info.get('mean_rate_threshold', None)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Spatial information histogram
    ax = axes[0, 0]
    ax.hist(spatial_info, bins=30, color='steelblue', edgecolor='black', alpha=0.7)
    if info_thresh is not None:
        ax.axvline(info_thresh, color='red', linestyle='--', label=f'Threshold {info_thresh:.2f}')
    ax.set_xlabel('Spatial Information (bits/spike)')
    ax.set_ylabel('Count')
    ax.set_title('Spatial Information Distribution')
    ax.legend()

    # Peak rate histogram
    ax = axes[0, 1]
    ax.hist(peak_rate, bins=30, color='darkorange', edgecolor='black', alpha=0.7)
    if peak_thresh is not None:
        ax.axvline(peak_thresh, color='red', linestyle='--', label=f'Threshold {peak_thresh:.2f} Hz')
    ax.set_xlabel('Peak Rate (Hz)')
    ax.set_ylabel('Count')
    ax.set_title('Peak Rate Distribution')
    ax.legend()

    # Mean rate histogram
    ax = axes[1, 0]
    ax.hist(mean_rate, bins=30, color='seagreen', edgecolor='black', alpha=0.7)
    if mean_thresh is not None:
        ax.axvline(mean_thresh, color='red', linestyle='--', label=f'Threshold {mean_thresh:.2f} Hz')
    ax.set_xlabel('Mean Rate (Hz)')
    ax.set_ylabel('Count')
    ax.set_title('Mean Rate Distribution')
    ax.legend()

    # Scatter: spatial info vs peak rate
    ax = axes[1, 1]
    ax.scatter(spatial_info, peak_rate, s=20, alpha=0.6, color='purple')
    if info_thresh is not None:
        ax.axvline(info_thresh, color='red', linestyle='--')
    if peak_thresh is not None:
        ax.axhline(peak_thresh, color='red', linestyle='--')
    ax.set_xlabel('Spatial Information (bits/spike)')
    ax.set_ylabel('Peak Rate (Hz)')
    ax.set_title('Spatial Info vs Peak Rate')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    else:
        plt.show()
    plt.close(fig)


def compute_evaluation_matrix(program: callable,
                              params: jnp.ndarray,
                              eval_points: np.ndarray,
                              n_features: int = 2) -> jnp.ndarray:
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
                    save_path: Optional[str] = None):
    """
    Plot 2D rate map fits for place cell models.
    
    For place cells, we show actual rate maps vs model predictions in 2D.
    Uses precomputed rate maps from data_parser when available.
    
    Args:
        plot_data: Precomputed plotting tensors from
            hypothesis_engine.prepare_model_fit_plot_data(...).
        n_eval: Number of evaluation points per dimension for model prediction grid.
        save_path: Path to save figure.
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
            f"Place-cell plot_model_fits requires 2D position inputs; got stimuli_3d shape {inputs_subset.shape}."
        )
    if sample_selection.shape[0] != n_cells_plot:
        raise ValueError(
            f"sample_selection length ({sample_selection.shape[0]}) must equal n_cells ({n_cells_plot})."
        )
    
    if labels is None:
        labels = [f'Model {i+1}' for i in range(n_models)]
    
    # Figure layout: rows = cells, cols = [data, model1, model2, ...]
    n_cols = 1 + n_models
    fig, axes = plt.subplots(n_cells_plot, n_cols, figsize=(4 * n_cols, 4 * n_cells_plot))
    if n_cells_plot == 1:
        axes = axes.reshape(1, -1)
    
    for c_idx in range(n_cells_plot):
        cell_idx = sample_selection[c_idx]

        x_pos = np.asarray(inputs_subset[c_idx, 0, :])
        y_pos = np.asarray(inputs_subset[c_idx, 1, :])
        response_cell = np.asarray(response_subset[c_idx])
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
        
        # Plot model predictions from precomputed trial predictions
        for m_idx in range(n_models):
            pred_trials = np.asarray(trial_predictions[m_idx, c_idx])
            model_map = _bin_to_rate_map(x_pos, y_pos, pred_trials, n_bins=n_eval)
            loss_val = float(jnp.mean(point_losses[m_idx, c_idx]))
            
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


def plot_single_model_fit(model: Callable, loss_function: Callable,
                          x: jnp.ndarray, y: jnp.ndarray, params: jnp.ndarray,
                          rate_maps: Optional[np.ndarray] = None,
                          n_eval: int = 50,
                          dpi: float = 100.0, title: str = '',
                          save_path: Optional[str] = None,
                          input_idx: int = 0):
    """
    Plot fit of a single place cell model.
    
    Args:
        model: Place cell model function.
        loss_function: Loss function.
        x: Input data of shape (n_cells, n_features, n_trials).
        y: Response data of shape (n_cells, n_trials).
        params: Parameters of shape (n_cells, n_params).
        rate_maps: Precomputed rate maps of shape (n_cells, n_bins, n_bins). Optional.
        n_eval: Grid resolution for model evaluation.
        save_path: Path to save figure.
    """
    n_cells = y.shape[0]
    n_row_cols = int(np.ceil(np.sqrt(n_cells)))
    
    # Create evaluation grid
    eval_pts = np.linspace(-1, 1, n_eval)
    X_grid, Y_grid = np.meshgrid(eval_pts, eval_pts, indexing='xy')
    X_eval = jnp.stack([X_grid.ravel(), Y_grid.ravel()], axis=0)  # (2, n_eval^2)
    
    fig, axes = plt.subplots(n_row_cols, n_row_cols * 2, 
                             figsize=(4 * n_row_cols * 2, 4 * n_row_cols))
    axes = np.atleast_2d(axes)
    
    for c in range(n_cells):
        row = c // n_row_cols
        col = (c % n_row_cols) * 2
        
        # Use precomputed rate map if available
        if rate_maps is not None:
            data_rate_map = rate_maps[c]
            if data_rate_map.shape[0] != n_eval:
                from scipy.ndimage import zoom
                zoom_factor = n_eval / data_rate_map.shape[0]
                data_rate_map = zoom(data_rate_map, zoom_factor, order=1)
        else:
            x_pos = np.array(x[c, 0, :])
            y_pos = np.array(x[c, 1, :])
            response = np.array(y[c])
            data_rate_map = _bin_to_rate_map(x_pos, y_pos, response, n_bins=n_eval)
        
        peak_rate = data_rate_map.max()
        
        ax_data = axes[row, col]
        im = ax_data.imshow(data_rate_map.T, origin='lower', extent=[-1, 1, -1, 1],
                            cmap='viridis', aspect='equal')
        ax_data.set_title(f'Cell {c} - Data\nPeak: {peak_rate:.1f} Hz', fontsize=9)
        plt.colorbar(im, ax=ax_data, fraction=0.046, pad=0.04)
        
        # Model prediction
        params_c = params[c]
        model_output = model(X_eval, *params_c)
        model_map = np.array(model_output).reshape(n_eval, n_eval)
        
        # Compute loss
        X_cell = x[c]
        pred = model(X_cell, *params_c)
        loss_val = float(jnp.mean(loss_function(pred, y[c])))
        
        ax_model = axes[row, col + 1]
        im = ax_model.imshow(model_map.T, origin='lower', extent=[-1, 1, -1, 1],
                             cmap='viridis', aspect='equal')
        ax_model.set_title(f'Model (loss: {loss_val:.3f})', fontsize=9)
        plt.colorbar(im, ax=ax_model, fraction=0.046, pad=0.04)
    
    # Hide unused subplots
    for idx in range(n_cells, n_row_cols * n_row_cols):
        row = idx // n_row_cols
        col = (idx % n_row_cols) * 2
        if row < axes.shape[0] and col + 1 < axes.shape[1]:
            axes[row, col].axis('off')
            axes[row, col + 1].axis('off')
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    if save_path:
        plt.savefig(save_path, dpi=dpi)
    else:
        plt.show()
    plt.close(fig)


def plot_train_vs_test_loss(programs_df: pd.DataFrame,
                            island_labels: list,
                            save_path: Optional[str] = None,
                            loss_thresh: float = 1000.0):
    """
    Plot train vs test loss for each program in the DataFrame.
    
    Args:
        programs_df: DataFrame with 'train_loss' and 'test_loss' columns.
        island_labels: Labels for each island.
        save_path: Path to save plot.
    """
    if 'train_loss' not in programs_df.columns or 'test_loss' not in programs_df.columns:
        raise ValueError("DataFrame must contain 'train_loss' and 'test_loss' columns.")
    
    train_loss = programs_df['train_loss'].to_numpy()
    test_loss = programs_df['test_loss'].to_numpy()
    birth_island = programs_df['birth_island'].to_numpy()
    
    # Filter out invalid values
    train_loss = np.nan_to_num(train_loss, nan=np.inf)
    test_loss = np.nan_to_num(test_loss, nan=np.inf)
    
    mask = (train_loss < loss_thresh) & (test_loss < loss_thresh)
    train_loss = train_loss[mask]
    test_loss = test_loss[mask]
    birth_island = birth_island[mask]
    
    # Guard against empty arrays after filtering
    if len(train_loss) == 0:
        print(f"Warning: No valid programs to plot (all losses >= {loss_thresh}). Skipping plot.")
        plt.figure(figsize=(10, 10))
        plt.text(0.5, 0.5, f'No valid programs\n(all losses >= {loss_thresh})', 
                 ha='center', va='center', fontsize=14, transform=plt.gca().transAxes)
        plt.xlabel('Train Loss')
        plt.ylabel('Test Loss')
        plt.title('Train vs Test Loss')
        if save_path:
            plt.savefig(save_path)
        plt.close()
        return
    
    cmap = plt.get_cmap('tab10')
    
    plt.figure(figsize=(10, 10))
    for island_id in np.unique(birth_island):
        island_mask = (birth_island == island_id)
        plt.scatter(train_loss[island_mask], test_loss[island_mask],
                    label=island_labels[int(island_id)], color=cmap(int(island_id)), alpha=1.0)
    
    plt.xlabel('Train Loss')
    plt.ylabel('Test Loss')
    lim_min = 0.9 * min(np.min(train_loss), np.min(test_loss))
    lim_max = 1.1 * max(np.median(train_loss), np.median(test_loss))
    plt.xlim(lim_min, lim_max)
    plt.ylim(lim_min, lim_max)
    plt.plot([0, 100], [0, 100], color='black', linestyle='--', alpha=0.5)
    plt.title('Train vs Test Loss')
    plt.legend()
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()


def plot_losses(loss: np.ndarray, true_model_loss: Optional[float] = None,
                island_labels: Optional[list] = None,
                alpha: float = 0.5, dpi: float = 100.0, y_lims: Optional[tuple] = None,
                title: str = 'Learning Curve', legend_font_size: int = 6,
                save_path: Optional[str] = None):
    """
    Plot losses over iterations for all islands.
    
    Args:
        loss: (n_iter, n_islands) array of loss lists.
        true_model_loss: Optional ground truth loss.
        island_labels: Labels for each island.
        save_path: Path to save figure.
    """
    n_iter, n_islands = loss.shape
    island_min = np.full((n_iter, n_islands), np.inf)
    
    for iter_id, island_id in np.ndindex(n_iter, n_islands):
        island_min[iter_id, island_id] = np.nanmin(np.array(loss[iter_id, island_id]))
    
    global_min = np.nanmin(island_min, axis=1)
    
    if island_labels is None:
        island_labels = [f'Island {i}' for i in range(n_islands)]
    
    plt.figure(figsize=(10, 5))
    cmap = plt.get_cmap('tab10')
    
    for iter_id, island_id in np.ndindex(n_iter, n_islands):
        y_vals = loss[iter_id, island_id]
        x_vals = np.ones(len(y_vals)) * (n_islands * iter_id + island_id)
        if iter_id == 0:
            plt.scatter(x_vals, y_vals, label=island_labels[island_id] if alpha > 0.0 else None,
                        alpha=alpha, color=cmap(island_id))
        else:
            plt.scatter(x_vals, y_vals, alpha=alpha, color=cmap(island_id))
    
    # Plot minimum loss per island
    for island_id in range(n_islands):
        plt.plot(np.arange(n_iter) * n_islands + island_id, island_min[:, island_id],
                 label=island_labels[island_id], color=cmap(island_id), linewidth=1,
                 linestyle='--', alpha=0.25)
    
    # Global minimum
    global_min_repeated = np.repeat(global_min[:, np.newaxis], n_islands, axis=1).reshape(-1)
    plt.plot(np.arange(n_islands * n_iter), global_min_repeated,
             label='Global min loss', color='black', linewidth=2, linestyle='-', alpha=1.0)
    
    if true_model_loss is not None:
        plt.axhline(y=true_model_loss, color='black', linestyle='--', alpha=0.5,
                    label='True model loss')
    
    # Iteration separators
    for i in range(n_iter):
        plt.axvline(x=n_islands * i - 0.5, color='grey', linestyle='--', alpha=0.5)
    
    if y_lims is None:
        y_lims = (0.99 * np.nanmin(island_min), 1.01 * np.nanmax(island_min))
    plt.ylim(y_lims)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.xticks(np.arange(n_iter) * n_islands + n_islands / 2,
               [f'Iter {i}' for i in range(n_iter)], rotation=45)
    plt.title(title)
    plt.legend(loc='upper right', fontsize=legend_font_size)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=dpi)
    else:
        plt.show()
    plt.close()


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
    full_title = f'{title} ({n_spatial_bins}x{n_spatial_bins} bins)'
    
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
    
    # 2. Place filter diagnostics (if available)
    place_filter_info = data_dict.get('place_filter_info')
    if place_filter_info is not None:
        print(f"Generating place filter diagnostics...")
        plot_place_filter_diagnostics(
            place_filter_info,
            save_path=os.path.join(output_dir, f'{prefix}_place_filter.png'),
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
    place_filter_info = data_dict.get('place_filter_info')
    
    print("=" * 50)
    print("Data Loading Summary")
    print("=" * 50)
    
    if response is not None:
        n_cells, n_time_bins = response.shape
        print(f"Response: {n_cells} cells x {n_time_bins} time bins")
        print(f"  Mean FR: {response.mean():.3f} Hz")
        print(f"  Max FR:  {response.max():.3f} Hz")
    
    if inputs is not None:
        print(f"Inputs: {inputs.data.shape}")
        print(f"  Names: {inputs.names}")
    
    if rate_maps is not None:
        print(f"Rate maps: {rate_maps.shape}")
        print(f"  Spatial bins: {rate_maps.shape[1]}x{rate_maps.shape[2]}")
    
    if position_data:
        print(f"Position data keys: {list(position_data.keys())}")
        if 'time_bin_ms' in position_data:
            print(f"  Time bin: {position_data['time_bin_ms']} ms")
        if 'n_spatial_bins' in position_data:
            print(f"  Spatial bins: {position_data['n_spatial_bins']}x{position_data['n_spatial_bins']}")
    
    if place_filter_info is not None:
        print(f"Place cell filtering: Applied")
        if 'place_cell_indices' in place_filter_info:
            print(f"  Place cells identified: {len(place_filter_info['place_cell_indices'])}")
    else:
        print(f"Place cell filtering: Not applied")
    
    print("=" * 50)
