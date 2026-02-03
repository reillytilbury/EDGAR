import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Set non-GUI backend before importing pyplot
import matplotlib.pyplot as plt
import jax.numpy as jnp
from typing import Optional, Callable, Sequence

from src import utils


def _ensure_input_format(x_cell: jnp.ndarray) -> jnp.ndarray:
    """Convert 1D stimulus array to 2D input format (n_features, n_trials)."""
    if x_cell.ndim == 1:
        return x_cell.reshape(1, -1)  # (n_trials,) -> (1, n_trials)
    return x_cell  # already (n_features, n_trials)


def compute_evaluation_matrix(program: callable, params: jnp.ndarray, n_evaluation_points: int = 100, eval_points : Optional[np.ndarray] = None, 
                               input_idx: int = 0, n_features: int = 1) -> jnp.ndarray:
    """
    Computes the evaluation matrix for a given program and parameters.
    
    Creates a uniform grid of evaluation points and evaluates the model at those points
    for all samples. 
    
    For backward compatibility:
    - When n_features=1: passes X as 1D array (n_trials,) - original behavior
    - When n_features>1: passes X as 2D array (n_features, n_trials) - new behavior
    
    Args:
        program (callable): The neuron model function.
        params (jnp.ndarray): The parameters for the neuron model. Shape: (n_samples, n_params)
        n_evaluation_points (int): Number of points to evaluate the model at.
        eval_points (np.ndarray, optional): Custom evaluation points. If None, uses linspace(0, 2π).
        input_idx (int): Index of the input to vary for evaluation (for n_features > 1).
        n_features (int): Total number of inputs in the model. Default is 1.
    Returns:
        jnp.ndarray: The evaluation matrix of shape (n_samples, n_evaluation_points).
    """
    # Create evaluation points
    if eval_points is not None:
        trials = jnp.asarray(eval_points)
        n_evaluation_points = trials.shape[0]
    else:
        trials = jnp.linspace(0, 2 * jnp.pi, n_evaluation_points)
    
    # vmap over samples
    program_vmap = utils.vmap_over_cells(program)
    
    if n_features == 1:
        # Original 1D behavior: pass trials directly (shape: n_eval,)
        # This is what orientation tuning models expect
        y_eval = program_vmap(trials, params)
    else:
        # New 2D behavior for multi-input models (e.g., grid cells)
        # Create X with zeros for all inputs, then set the evaluation input
        n_samples = params.shape[0]
        X_eval = jnp.zeros((n_samples, n_features, n_evaluation_points))
        trials_broadcast = jnp.broadcast_to(trials, (n_samples, n_evaluation_points))
        X_eval = X_eval.at[:, input_idx, :].set(trials_broadcast)
        y_eval = program_vmap(X_eval, params)
    
    return y_eval

def plot_model_fits(programs_df: pd.DataFrame, loss_function: Callable, 
                    inputs: jnp.ndarray, response: jnp.ndarray, 
                    sample_selection: Sequence[int],
                    n_eval: int = 100, n_mean: int = 50,
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
    plot fits of all models in programs_df over a subset of cells in x and y, along with the running mean.
    Args:
        programs_df:
            - must have columns 'program' and 'params'. 
            - must have n_rows <= 3
            - 'program': callable (written in JAX): (X: jnp.ndarray, *params) -> jnp.ndarray
                         where X has shape (n_features, n_trials)
            - 'params': jnp.ndarray (n_cells, n_params)
        loss_function: 
            - callable (written in JAX): (y_est: jnp.ndarray, y_true: jnp.ndarray) -> jnp.ndarray
        inputs: Input data. Can be:
           - 2D array (n_cells, n_trials) - will use first axis as theta
           - 3D array (n_cells, n_features, n_trials)
        response: (n_cells x n_trials) - jnp.ndarray
        input_idx (int): Index of the input to use for plotting (x-axis). Default is 0.
                             Must be 0 if inputs is 2D.
    
    Raises:
        ValueError: If input_idx != 0 when inputs is 2D, or if input_idx is out of range.
    """
    assert len(programs_df) <= 3, f"programs_df must have at most 3 rows, but has {len(programs_df)} rows."
    assert len(sample_selection) > 0, "sample_selection must not be empty."
    assert len(sample_selection) == int(np.sqrt(len(sample_selection)))**2, \
        f"sample_selection must be a square number, but has {len(sample_selection)} elements."

    # Early validation of input_idx
    x_arr = jnp.asarray(inputs)
    y = jnp.asarray(response)
    if x_arr.ndim == 2:
        if input_idx != 0:
            raise ValueError(
                f"input_idx must be 0 for 2D input (single input), got {input_idx}."
            )
    else:
        n_features = x_arr.shape[1]
        if input_idx < 0 or input_idx >= n_features:
            raise ValueError(
                f"input_idx ({input_idx}) must be in range [0, {n_features}). "
                f"Got n_features={n_features}."
            )

    # define frequently used variables
    models = programs_df['program'].tolist()
    params = programs_df['params'].tolist()
    sample_idx = jnp.array(sample_selection)
    params = [p[sample_idx] for p in params]
    spike_matrix = y[sample_idx]
    
    # Handle both 2D (n_samples, n_trials) and 3D (n_samples, n_features, n_trials) input
    if x_arr.ndim == 2:
        # 2D input: (n_samples, n_trials) - expand to (n_samples, 1, n_trials)
        stimuli_3d = x_arr[sample_idx][:, jnp.newaxis, :]
        stimuli_1d = x_arr[sample_idx]  # for plotting (single input)
    else:
        # 3D input: (n_cells, n_features, n_trials)
        stimuli_3d = x_arr[sample_idx]
        stimuli_1d = x_arr[sample_idx][:, input_idx, :]  # use specified input for plotting
    
    n_cells, n_features, n_trials = stimuli_3d.shape
    n_models = len(models)
    if labels is None:
        labels = [f'model {i + 1}' for i in range(n_models)]

    # define figure and axes, ensuring ax is 2D even if n_cells == 1
    n_row_cols = int(np.sqrt(n_cells))
    fig, ax = plt.subplots(n_row_cols, n_row_cols, figsize=(20, 20))
    if n_cells == 1:
        ax = np.array([[ax]])  # Ensure ax is 2D for single plot

    # Calculate loss for each model, cell and trial
    point_losses = jnp.zeros((n_models, n_cells, n_trials))
    for i, model in enumerate(models):
        for c in range(n_cells):
            params_ic = params[i][c]
            X_cell = stimuli_3d[c]  # (n_features, n_trials)
            predicted_response = model(X_cell, *params_ic)
            point_losses = point_losses.at[i, c].set(loss_function(predicted_response, spike_matrix[c]))
    
    # compute running mean (using first input for binning)
    x_values_mean = jnp.linspace(0, 2 * jnp.pi, n_mean, endpoint=False) + 0.5 * (2 * jnp.pi / n_mean)  # Shift to center bins
    binned_mean = jnp.zeros((n_cells, n_mean))
    for c in range(n_cells):
        bin_idx = jnp.clip(((stimuli_1d[c] * n_mean) / (2 * jnp.pi)).astype(jnp.int32), 0, n_mean - 1)
        sums = jnp.bincount(bin_idx, weights=spike_matrix[c], minlength=n_mean)
        counts = jnp.bincount(bin_idx, minlength=n_mean)
        binned_mean = binned_mean.at[c].set((sums + 1e-6) / (counts + 1e-6))  # Avoid division by zero

    # compute cell outputs at evaluation points
    x_values_eval = jnp.linspace(0, 2 * jnp.pi, n_eval, endpoint=False)
    X_eval = x_values_eval.reshape(1, -1)  # (1, n_eval) - single input format
    model_outputs = jnp.zeros((n_models, n_cells, n_eval))
    for i, model in enumerate(models):
        for c in range(n_cells):
            params_ic = params[i][c]
            model_outputs = model_outputs.at[i, c].set(model(X_eval, *params_ic))

    for c in range(n_cells):
        row, col = divmod(c, n_row_cols)
        # Scatter plot of data points (x=stimulus, y=response) for sample c
        ax[row, col].scatter(stimuli_1d[c], spike_matrix[c], c='black', alpha=point_alpha, s=point_size)

        # Plot running mean for sample c
        ax[row, col].plot(x_values_mean, binned_mean[c], 
                          label='Mean', color="#3BD1FF", linewidth=line_width * 1.35)

        # Plot model fits to sample c
        for i, model in enumerate(models):
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
    
def plot_single_model_fit(model: Callable, loss_function: Callable, 
                          x: jnp.ndarray, y: jnp.ndarray, params: jnp.ndarray, 
                          n_eval: int = 100, n_mean: int = 50,
                          dpi: float = 100.0, title: str = '', 
                          save_path: Optional[str] = None,
                          input_idx: int = 0):
    """
    Plots the fit of a single model to a selection of cells in x and y, along with the running mean.
    Args:
        model: callable (written in JAX): (X: jnp.ndarray, *params) -> jnp.ndarray
               where X has shape (n_features, n_trials)
        loss_function: callable (written in JAX): (y_est: jnp.ndarray, y_true: jnp.ndarray) -> jnp.ndarray
        x: Input data. Can be:
           - 2D array (n_cells, n_trials) - will use as single input
           - 3D array (n_cells, n_features, n_trials)
        y: (n_cells x n_trials) - jnp.ndarray
        params: (n_cells x n_params) - jnp.ndarray
        input_idx (int): Index of the input to use for plotting (x-axis). Default is 0.
                             Must be 0 if x is 2D.
    
    Raises:
        ValueError: If input_idx != 0 when x is 2D, or if input_idx is out of range.
    """
    n_cells = y.shape[0]
    assert n_cells == int(np.sqrt(n_cells))**2, f"n_cells must be a square number, but got {n_cells} cells."
    
    # Early validation of input_idx
    x_arr = jnp.asarray(x)
    if x_arr.ndim == 2:
        if input_idx != 0:
            raise ValueError(
                f"input_idx must be 0 for 2D input (single input), got {input_idx}."
            )
    else:
        n_inputs = x_arr.shape[1]
        if input_idx < 0 or input_idx >= n_inputs:
            raise ValueError(
                f"input_idx ({input_idx}) must be in range [0, {n_inputs}). "
                f"Got n_features={n_inputs}."
            )
    
    # Handle both 2D and 3D input
    if x_arr.ndim == 2:
        # 2D input: (n_cells, n_trials) - expand to (n_cells, 1, n_trials)
        x_3d = x_arr[:, jnp.newaxis, :]
        x_1d = x_arr  # for plotting (single input)
    else:
        # 3D input: (n_cells, n_features, n_trials)
        x_3d = x_arr
        x_1d = x_arr[:, input_idx, :]  # use specified input for plotting
    
    n_cells, n_features, n_trials = x_3d.shape

    # Calculate loss for each cell and trial
    point_losses = jnp.zeros((n_cells, n_trials))
    for c in range(n_cells):
        params_c = params[c]
        X_cell = x_3d[c]  # (n_features, n_trials)
        predicted_response = model(X_cell, *params_c)
        point_losses = point_losses.at[c].set(loss_function(predicted_response, y[c]))

    # compute running mean (using first input for binning)
    x_values_mean = jnp.linspace(0, 2 * jnp.pi, n_mean, endpoint=False) + 0.5 * (2 * jnp.pi / n_mean)  # Shift to center bins
    binned_mean = jnp.zeros((n_cells, n_mean))
    for c in range(n_cells):
        bin_idx = jnp.clip(((x_1d[c] * n_mean) / (2 * jnp.pi)).astype(jnp.int32), 0, n_mean - 1)
        sums = jnp.bincount(bin_idx, weights=y[c], minlength=n_mean)
        counts = jnp.bincount(bin_idx, minlength=n_mean)
        binned_mean = binned_mean.at[c].set((sums + 1e-6) / (counts + 1e-6))  # Avoid division by zero

    # compute cell outputs at evaluation points
    x_values_eval = jnp.linspace(0, 2 * jnp.pi, n_eval, endpoint=False)
    X_eval = x_values_eval.reshape(1, -1)  # (1, n_eval) - single input format
    model_output = jnp.zeros((n_cells, n_eval))
    for c in range(n_cells):
        params_c = params[c]
        model_output = model_output.at[c].set(model(X_eval, *params_c))

    n_row_cols = int(np.sqrt(n_cells))
    fig, ax = plt.subplots(n_row_cols, n_row_cols, figsize=(20, 20))
    if n_cells == 1:
        ax = np.array([[ax]])

    for c in range(n_cells):
        row, col = divmod(c, n_row_cols)

        # data scatter
        vmin, vmax = np.percentile(point_losses[c], [1,99])
        sc = ax[row, col].scatter(x_1d[c], y[c], c=point_losses[c], cmap='viridis', vmin=vmin, vmax=vmax, alpha=0.5)
        plt.colorbar(sc, ax=ax[row, col], label='Loss')

        # running mean
        ax[row, col].plot(x_values_mean, binned_mean[c], label='Mean', color='cyan', linewidth=4.0)

        # model fit
        ax[row, col].plot(x_values_eval, model_output[c], label='Model', color='red', alpha=1, linewidth=3.0)

        # Set axis properties
        ax[row, col].set_ylabel('Firing Rate')
        ax[row, col].set_title(f'Cell {c}. Loss: {jnp.mean(point_losses[c]):.2f}')
        if c == 0:
            ax[row, col].legend(loc='upper right')

    # make each line of the sup title the colour of the model
    plt.suptitle(title)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    if save_path:
        plt.savefig(save_path, dpi=dpi)
    else:
        plt.show()
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