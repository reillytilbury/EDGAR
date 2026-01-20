"""
Diagnostic plotting functions for grid cell experiments.

This module provides visualization functions for 2D grid cell rate maps,
model fits, and learning curves specific to spatial navigation data.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Set non-GUI backend before importing pyplot
import matplotlib.pyplot as plt
import jax.numpy as jnp
from typing import Optional, Callable, Sequence, Tuple


def _ensure_predictor_format(x_cell: jnp.ndarray) -> jnp.ndarray:
    """Convert 1D/2D stimulus array to 2D predictor format (n_features, n_trials)."""
    if x_cell.ndim == 1:
        return x_cell.reshape(1, -1)  # (n_trials,) -> (1, n_trials)
    return x_cell  # already (n_features, n_trials)


def plot_model_fits(programs_df: pd.DataFrame, loss_function: Callable,
                    x: jnp.ndarray, y: jnp.ndarray,
                    sample_selection: Sequence[int],
                    n_eval: int = 50, n_mean: int = 50,
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
                    predictor_idx: int = 0):
    """
    Plot 2D rate map fits for grid cell models.
    
    For grid cells, we show actual rate maps vs model predictions in 2D.
    
    Args:
        programs_df: DataFrame with 'program' and 'params' columns.
        loss_function: Loss function (y_est, y_true) -> loss.
        x: Predictor data of shape (n_cells, n_features, n_trials) where n_features=2 (x, y).
        y: Response data of shape (n_cells, n_trials).
        sample_selection: Indices of cells to plot.
        n_eval: Number of evaluation points per dimension for model prediction grid.
        save_path: Path to save figure.
        predictor_idx: Ignored for grid cells (uses both x and y).
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
    y_subset = y[sample_idx]
    x_subset = x[sample_idx]  # (n_cells_plot, n_features, n_trials)
    
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
        # Compute "data" rate map from binned responses
        x_pos = np.array(x_subset[c, 0, :])  # x positions
        y_pos = np.array(x_subset[c, 1, :])  # y positions
        response = np.array(y_subset[c])
        
        # Bin the actual data into a rate map for visualization
        data_rate_map = _bin_to_rate_map(x_pos, y_pos, response, n_bins=n_eval)
        
        # Plot actual data rate map
        ax = axes[c_idx, 0]
        im = ax.imshow(data_rate_map.T, origin='lower', extent=[-1, 1, -1, 1],
                       cmap='viridis', aspect='equal')
        ax.set_title(f'Cell {sample_selection[c]} - Data')
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
            X_cell = x_subset[c]  # (2, n_trials)
            pred = model(X_cell, *params_c)
            loss_val = float(jnp.mean(loss_function(pred, y_subset[c])))
            
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
                          n_eval: int = 50, n_mean: int = 50,
                          dpi: float = 100.0, title: str = '',
                          save_path: Optional[str] = None,
                          predictor_idx: int = 0):
    """
    Plot fit of a single grid cell model.
    
    Args:
        model: Grid cell model function.
        loss_function: Loss function.
        x: Predictor data of shape (n_cells, n_features, n_trials).
        y: Response data of shape (n_cells, n_trials).
        params: Parameters of shape (n_cells, n_params).
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
        
        x_pos = np.array(x[c, 0, :])
        y_pos = np.array(x[c, 1, :])
        response = np.array(y[c])
        
        # Actual data
        data_rate_map = _bin_to_rate_map(x_pos, y_pos, response, n_bins=n_eval)
        
        ax_data = axes[row, col]
        im = ax_data.imshow(data_rate_map.T, origin='lower', extent=[-1, 1, -1, 1],
                            cmap='viridis', aspect='equal')
        ax_data.set_title(f'Cell {c} - Data')
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
        ax_model.set_title(f'Model (loss: {loss_val:.3f})')
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
                            save_path: Optional[str] = None):
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
    
    mask = (train_loss < 100) & (test_loss < 100)
    train_loss = train_loss[mask]
    test_loss = test_loss[mask]
    birth_island = birth_island[mask]
    
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
                   save_path: Optional[str] = None,
                   dpi: float = 100.0):
    """
    Plot 2D rate maps for selected cells.
    
    Args:
        rate_maps: Rate maps of shape (n_cells, n_bins, n_bins).
        sample_selection: Indices of cells to plot. If None, plots first 16.
        title: Plot title.
        save_path: Path to save figure.
    """
    if sample_selection is None:
        sample_selection = list(range(min(16, len(rate_maps))))
    
    n_cells = len(sample_selection)
    n_cols = int(np.ceil(np.sqrt(n_cells)))
    n_rows = int(np.ceil(n_cells / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows))
    axes = np.atleast_2d(axes)
    
    for idx, cell_idx in enumerate(sample_selection):
        row, col = divmod(idx, n_cols)
        ax = axes[row, col]
        
        im = ax.imshow(rate_maps[cell_idx].T, origin='lower', extent=[-1, 1, -1, 1],
                       cmap='viridis', aspect='equal')
        ax.set_title(f'Cell {cell_idx}')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # Hide unused subplots
    for idx in range(n_cells, n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axes[row, col].axis('off')
    
    plt.suptitle(title, fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    if save_path:
        plt.savefig(save_path, dpi=dpi)
    else:
        plt.show()
    plt.close(fig)


def _bin_to_rate_map(x: np.ndarray, y: np.ndarray, values: np.ndarray,
                     n_bins: int = 50, extent: Tuple[float, float, float, float] = (-1, 1, -1, 1)
                     ) -> np.ndarray:
    """
    Bin scattered data into a 2D rate map.
    
    Args:
        x, y: Position coordinates.
        values: Values at each position (e.g., firing rates).
        n_bins: Number of bins per dimension.
        extent: (xmin, xmax, ymin, ymax).
    
    Returns:
        rate_map: 2D binned average of values.
    """
    xmin, xmax, ymin, ymax = extent
    
    # Bin indices
    bin_x = np.clip(((x - xmin) / (xmax - xmin) * n_bins).astype(int), 0, n_bins - 1)
    bin_y = np.clip(((y - ymin) / (ymax - ymin) * n_bins).astype(int), 0, n_bins - 1)
    
    # Sum and count per bin
    rate_map = np.zeros((n_bins, n_bins))
    counts = np.zeros((n_bins, n_bins))
    
    for i in range(len(x)):
        rate_map[bin_x[i], bin_y[i]] += values[i]
        counts[bin_x[i], bin_y[i]] += 1
    
    # Average
    rate_map = rate_map / (counts + 1e-6)
    return rate_map
