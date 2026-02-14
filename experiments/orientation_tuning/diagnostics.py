import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Set non-GUI backend before importing pyplot
import matplotlib.pyplot as plt
import jax.numpy as jnp
from typing import Optional, Callable

from src import utils
from src.diagnostics_manager import ModelFitPlotData


def _ensure_input_format(x_cell: jnp.ndarray) -> jnp.ndarray:
    """Convert 1D stimulus array to 2D input format (n_features, n_trials)."""
    if x_cell.ndim == 1:
        return x_cell.reshape(1, -1)  # (n_trials,) -> (1, n_trials)
    return x_cell  # already (n_features, n_trials)


def select_evaluation_points(inputs: jnp.ndarray,
                             n_points: int = 100,
                             random_seed: int = 0,
                             input_idx: int = 0,
                             **kwargs) -> jnp.ndarray:
    """
    Select evaluation points for orientation-tuning diagnostics.

    Purpose:
    - Centralize experiment-specific evaluation-point policy outside hypothesis_engine.
    - Provide explicit, reproducible points used to compute `evaluation_matrix`.

    Strategy:
    - Single-input data `(n_samples, n_trials)`: create a shared linear grid spanning
      the observed input range and broadcast to all samples.
    - Multi-input data `(n_samples, n_features, n_trials)`: vary `input_idx` across
      its observed range while fixing other features at each sample's trial-mean.
    """
    x_arr = jnp.asarray(inputs)
    n_samples = int(x_arr.shape[0])

    if x_arr.ndim == 2:
        x_min = float(jnp.min(x_arr))
        x_max = float(jnp.max(x_arr))
        if x_max <= x_min:
            x_max = x_min + 1e-6
        grid = jnp.linspace(x_min, x_max, n_points)
        return jnp.broadcast_to(grid, (n_samples, n_points))

    if x_arr.ndim != 3:
        raise ValueError(f"Expected 2D or 3D inputs, got shape {x_arr.shape}.")

    n_features = int(x_arr.shape[1])
    if input_idx < 0 or input_idx >= n_features:
        raise ValueError(f"input_idx ({input_idx}) must be in range [0, {n_features}).")

    feature_vals = x_arr[:, input_idx, :]
    f_min = float(jnp.min(feature_vals))
    f_max = float(jnp.max(feature_vals))
    if f_max <= f_min:
        f_max = f_min + 1e-6
    sweep = jnp.linspace(f_min, f_max, n_points)

    base = jnp.mean(x_arr, axis=2, keepdims=True)
    base = jnp.repeat(base, n_points, axis=2)
    sweep_broadcast = jnp.broadcast_to(sweep, (n_samples, n_points))
    return base.at[:, input_idx, :].set(sweep_broadcast)


def compute_evaluation_matrix(program: callable,
                              params: jnp.ndarray,
                              eval_points: jnp.ndarray,
                              **kwargs) -> jnp.ndarray:
    """
    Computes the evaluation matrix for a given program and parameters.
    
    Evaluate the model on explicit evaluation points.
    
    Args:
        program (callable): The neuron model function.
        params (jnp.ndarray): The parameters for the neuron model. Shape: (n_samples, n_params)
        eval_points: Explicit points with shape:
            - (n_samples, n_eval) for single-input models
            - (n_samples, n_features, n_eval) for multi-input models
    Returns:
        jnp.ndarray: The evaluation matrix of shape (n_samples, n_eval).
    """
    eval_arr = jnp.asarray(eval_points)
    if eval_arr.ndim == 1:
        eval_arr = jnp.broadcast_to(eval_arr, (params.shape[0], eval_arr.shape[0]))

    program_vmap = utils.vmap_over_cells(program)

    if eval_arr.ndim == 2:
        try:
            return program_vmap(eval_arr, params)
        except Exception:
            return program_vmap(eval_arr[:, jnp.newaxis, :], params)
    if eval_arr.ndim == 3:
        try:
            return program_vmap(eval_arr, params)
        except Exception:
            if eval_arr.shape[1] == 1:
                return program_vmap(eval_arr[:, 0, :], params)
            raise
    raise ValueError(f"eval_points must be 1D, 2D, or 3D. Got shape {eval_arr.shape}.")

def plot_model_fits(plot_data: ModelFitPlotData,
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
    Plot orientation-tuning fits using precomputed plotting data.

    This function is visualization-only. Model evaluation, loss computation,
    and curve preparation are done upstream in
    `hypothesis_engine.prepare_model_fit_plot_data(...)`.
    """
    stimuli_1d = jnp.asarray(plot_data['stimuli_1d'])
    spike_matrix = jnp.asarray(plot_data['spike_matrix'])
    point_losses = jnp.asarray(plot_data['point_losses'])
    x_values_mean = jnp.asarray(plot_data['x_values_mean'])
    binned_mean = jnp.asarray(plot_data['binned_mean'])
    x_values_eval = jnp.asarray(plot_data['x_values_eval'])
    model_outputs = jnp.asarray(plot_data['model_outputs'])
    n_models = int(plot_data['n_models'])
    n_cells = int(plot_data['n_cells'])
    n_row_cols = int(plot_data['n_row_cols'])
    sample_selection = np.asarray(plot_data['sample_selection'])

    if labels is None:
        labels = [f'model {i + 1}' for i in range(n_models)]
    if len(colours) < n_models:
        raise ValueError(f"Need at least {n_models} colours, got {len(colours)}.")

    fig, ax = plt.subplots(n_row_cols, n_row_cols, figsize=(20, 20))
    if n_cells == 1:
        ax = np.array([[ax]])

    for c in range(n_cells):
        row, col = divmod(c, n_row_cols)
        # Scatter plot of data points (x=stimulus, y=response) for sample c
        ax[row, col].scatter(stimuli_1d[c], spike_matrix[c], c='black', alpha=point_alpha, s=point_size)

        # Plot running mean for sample c
        ax[row, col].plot(x_values_mean, binned_mean[c], 
                          label='Mean', color="#3BD1FF", linewidth=line_width * 1.35)

        # Plot model fits to sample c
        for i in range(n_models):
            ax[row, col].plot(x_values_eval, model_outputs[i, c], 
                              label=labels[i] + f' (loss: {jnp.mean(point_losses[i, c]):.2f})',
                              color=colours[i], 
                              alpha=line_alpha, 
                              linewidth=line_width)
        model_max = jnp.max(model_outputs[:, c])
        mean_max = jnp.max(binned_mean[c])

        # Set axis properties
        ax[row, col].set_ylim(0, max(model_max, mean_max) * 2)
        ax[row, col].set_title(f'Sample {sample_selection[c]}', fontsize=16)
        ax[row, col].legend(loc='upper right', fontsize=legend_fontsize)
        if row == n_row_cols - 1:
            ax[row, col].set_xlabel('Theta (radians)', fontsize=20)
        if col == 0:
            ax[row, col].set_ylabel('Firing Rate', fontsize=20)

    plt.suptitle(title, fontsize=25)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path, dpi=dpi) if save_path else plt.show()
    plt.close(fig)

def plot_losses(loss: np.ndarray, true_model_loss: Optional[float] = None, 
                island_labels: Optional[list] = None,
                alpha: float = 0.5, dpi: float = 100.0, y_lims: Optional[tuple] = None,
                title: str = 'Learning Curve', legend_font_size: int = 6,
                save_path: Optional[str] = None):
    """
    Plot losss of arrays over iterations.
    Args:
        loss: (n_iter, n_islands) array of lists of losses for each island at each iteration.
        true_model_loss: float true model loss for simulated data.
        island_labels: (list) labels for each island. If not provided, will use default labels.
        save_path: (str) where to save the data. If not provided, will show the data but not save it.
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
        cmap_idx = island_id # colour by island_id
        if iter_id == 0:
            plt.scatter(x_vals, y_vals, label=island_labels[island_id] if alpha>0.0 else None,
                        alpha=alpha, color=cmap(cmap_idx))
        else:
            plt.scatter(x_vals, y_vals, alpha=alpha, color=cmap(cmap_idx))

    # plot the minimum loss for each island at each iteration
    for island_id in range(n_islands):
        plt.plot(np.arange(n_iter) * n_islands + island_id, island_min[:, island_id],
                 label=island_labels[island_id], color=cmap(island_id), linewidth=1, linestyle='--', alpha=0.25)
        
    # plot min loss across all islands at each iteration in black
    # the x axis has n_islands * n_iter points, so we need to create an array of that length
    # global min is only of length n_iter, so we need to repeat it for each island
    global_min = np.repeat(global_min[:, np.newaxis], n_islands, axis=1).reshape(-1)
    plt.plot(np.arange(n_islands * n_iter), global_min,
             label='Global min loss', color='black', linewidth=2, linestyle='-', alpha=1.0)
    
    # plot the true model loss
    if true_model_loss is not None:
        plt.axhline(y=true_model_loss, color='black', linestyle='--', alpha=0.5, label='True model loss')
    
    # put dashed verical lines at the end of each iteration
    for i in range(n_iter):
        plt.axvline(x=n_islands * i - 0.5, color='grey', linestyle='--', alpha=0.5)

    # make the plot look nice
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

    # save or plot the figure
    if save_path:
        plt.savefig(save_path, dpi=dpi)
        # plt.show()
    else:
        plt.show()
    plt.close()

def plot_train_vs_test_loss(programs_df: pd.DataFrame, 
                            island_labels: list,
                            save_path: Optional[str] = None):
    """
    Plot train vs test loss for each program in the DataFrame.
    Args:
        programs_df: DataFrame containing the programs and their losses. 
            It should have columns 'train_loss' and 'test_loss'.
        save_path: Path to save the plot. If None, will show the plot instead.
    """
    if 'train_loss' not in programs_df.columns or 'test_loss' not in programs_df.columns:
        raise ValueError("DataFrame must contain 'train_loss' and 'test_loss' columns.")
    
    # define variables
    train_loss = programs_df['train_loss'].to_numpy()
    test_loss = programs_df['test_loss'].to_numpy()
    birth_island = programs_df['birth_island'].to_numpy()

    # turn nan to num
    train_loss = np.nan_to_num(train_loss, nan=np.inf)
    test_loss = np.nan_to_num(test_loss, nan=np.inf)

    # only take loss < 100
    mask = (train_loss < 100) & (test_loss < 100)
    train_loss = train_loss[mask]
    test_loss = test_loss[mask]
    birth_island = birth_island[mask]
    cmap = plt.get_cmap('tab10')

    # plot the train vs test loss
    plt.figure(figsize=(10, 10))
    for island_id in np.unique(birth_island):
        island_mask = (birth_island == island_id)
        plt.scatter(train_loss[island_mask], test_loss[island_mask], 
                    label=island_labels[island_id], color=cmap(island_id), alpha=1.0)
    plt.xlabel('Train Loss')
    plt.ylabel('Test Loss')
    plt.xlim(0.9 * min(np.min(train_loss), np.min(test_loss)), 
             1.1 * max(np.median(train_loss), np.median(test_loss)))
    plt.ylim(0.9 * min(np.min(train_loss), np.min(test_loss)),
             1.1 * max(np.median(train_loss), np.median(test_loss)))
    plt.plot([0, 100], [0, 100], color='black', linestyle='--', alpha=0.5)  # diagonal line
    plt.title('Train vs Test Loss')
    plt.legend()
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()
