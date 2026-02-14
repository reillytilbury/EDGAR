
import numpy as np
import pandas as pd
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import jax.numpy as jnp
from typing import Optional, Callable, Sequence
from src import utils

# def plot_model_fits(x, y_true, 
#                     y_pred_v1, y_pred_v2, 
#                     loss_v1, loss_v2,
#                     n_bins=10, n_rows_plot=3, n_cols_plot=3, save_path=''):
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
    Plot the true vs predicted values to visualize model fit.
    
    Args:
        programs_df (pd.DataFrame): DataFrame containing the programs and their parameters
        loss_function (Callable): function to compute the loss
        inputs (jnp.ndarray): input data
        response (jnp.ndarray): true output values
        sample_selection (Sequence[int]): indices of samples to plot
        n_eval (int): number of evaluation points
        n_mean (int): number of points to compute mean
        colours (list): list of colors for plotting
        labels (Optional[list]): list of labels for the programs
        title (str): title of the plot
        line_width (float): width of the lines
        line_alpha (float): alpha value for the lines
        point_alpha (float): alpha value for the points
        point_size (int): size of the points
        legend_fontsize (int): font size for the legend
        dpi (float): DPI for the plot
        save_path (Optional[str]): path to save the plot
        input_idx (int): index of the input feature to plot
        n_bins (int): number of bins for plotting
        n_rows_plot (int): number of rows in the plot grid
        n_cols_plot (int): number of columns in the plot grid
        save_path (str): path to save the plot
    """
    x_arr = jnp.asarray(inputs)
    y = jnp.asarray(response)
    sample_idx = jnp.array(sample_selection)

    # define frequently used variables
    models = programs_df['program'].tolist()
    params = programs_df['params'].tolist()
    sample_idx = jnp.array(sample_selection)
    params = [p[sample_idx] for p in params]

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
            ax[row, col].set_xlabel('Input', fontsize=20)
        if col == 0:
            ax[row, col].set_ylabel('Output', fontsize=20)

    plt.suptitle(title, fontsize=25)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path, dpi=dpi) if save_path else plt.show()
    plt.close(fig)

    # # Subsample the data for plotting and compute losses
    # response_sample = y[sample_idx]
    # y_pred_v1_sample = y_pred_v1[sample_idx]
    # y_pred_v2_sample = y_pred_v2[sample_idx]
    # v1_loss = np.mean((response_sample - y_pred_v1_sample) ** 2, axis=1)
    # v2_loss = np.mean((response_sample - y_pred_v2_sample) ** 2, axis=1)

    # # Evaluate fit by binning the data and plotting mean true vs mean predicted values in each bin
    # bins = np.linspace(x.min(), x.max(), n_bins + 1)
    # bin_centers = (bins[:-1] + bins[1:]) / 2

    # # Interpolate predictions for smoother curves
    # x_interp = np.linspace(x.min(), x.max(), 100)

    # # Plot the results
    # fig, axes = plt.subplots(n_rows_plot, n_cols_plot, figsize=(15, 10))
    # for i, ax in enumerate(axes.flatten()):
    #     # Scatter plot for observed data
    #     ax.scatter(x, y_true_sample[i], label='Observed (with noise)', alpha=0.2, s=10, c='black')

    #     # Binned mean for true data
    #     mean_true = np.array([y_true_sample[i][(x >= bins[j]) & (x < bins[j+1])].mean() for j in range(n_bins)])
    #     ax.plot(bin_centers, mean_true, label='Binned observed mean', color='blue', alpha=0.7, linewidth=4, marker='o')

    #     # Interpolated predictions
    #     y_pred_v1_interp = np.interp(x_interp, x, y_pred_v1_sample[i])
    #     y_pred_v2_interp = np.interp(x_interp, x, y_pred_v2_sample[i])
    #     ax.plot(x_interp, y_pred_v1_interp, label=f'v1. loss = {v1_loss[i]:.2f}', color='green', alpha=0.7, linewidth=4)
    #     ax.plot(x_interp, y_pred_v2_interp, label=f'v2. loss = {v2_loss[i]:.2f}', color='red', alpha=0.7, linewidth=4)

    #     ax.set_title(f'Sample {sample_indices[i]}')
    #     ax.set_xlabel('Input (x)')
    #     ax.set_ylabel('Output (y)')
    #     ax.legend()

    # plt.tight_layout()
    # plt.suptitle(f'Model Fit Comparison (v1 loss = {loss_v1:.2f}, v2 loss = {loss_v2:.2f})', y=1.02)
    # plt.savefig(save_path)
    # plt.close()

    #     programs_df:
    #         - must have columns 'program' and 'params'. 
    #         - must have n_rows <= 3
    #         - 'program': callable (written in JAX): (X: jnp.ndarray, *params) -> jnp.ndarray
    #                      where X has shape (n_features, n_trials)
    #         - 'params': jnp.ndarray (n_cells, n_params)
    #     loss_function: 
    #         - callable (written in JAX): (y_est: jnp.ndarray, y_true: jnp.ndarray) -> jnp.ndarray
    #     inputs: Input data. Can be:
    #        - 2D array (n_cells, n_trials) - will use first axis as theta
    #        - 3D array (n_cells, n_features, n_trials)
    #     response: (n_cells x n_trials) - jnp.ndarray
    #     input_idx (int): Index of the input to use for plotting (x-axis). Default is 0.
    #                          Must be 0 if inputs is 2D.
    # Raises:
    #     ValueError: If input_idx != 0 when inputs is 2D, or if input_idx is out of range.
    # """
    # assert len(programs_df) <= 3, f"programs_df must have at most 3 rows, but has {len(programs_df)} rows."
    # assert len(sample_selection) > 0, "sample_selection must not be empty."
    # assert len(sample_selection) == int(np.sqrt(len(sample_selection)))**2, \
    #     f"sample_selection must be a square number, but has {len(sample_selection)} elements."

    # # Early validation of input_idx
    # x_arr = jnp.asarray(inputs)
    # y = jnp.asarray(response)
    # if x_arr.ndim == 2:
    #     if input_idx != 0:
    #         raise ValueError(
    #             f"input_idx must be 0 for 2D input (single input), got {input_idx}."
    #         )
    # else:
    #     n_features = x_arr.shape[1]
    #     if input_idx < 0 or input_idx >= n_features:
    #         raise ValueError(
    #             f"input_idx ({input_idx}) must be in range [0, {n_features}). "
    #             f"Got n_features={n_features}."
    #         )

    # models = programs_df['program'].tolist()
    # params = programs_df['params'].tolist()
    # sample_idx = jnp.array(sample_selection)
    # params = [p[sample_idx] for p in params]
    # spike_matrix = y[sample_idx]

    # if x_arr.ndim == 2:
    #     stimuli_3d = x_arr[sample_idx][:, jnp.newaxis, :]
    #     stimuli_1d = x_arr[sample_idx]
    # else:
    #     stimuli_3d = x_arr[sample_idx]
    #     stimuli_1d = x_arr[sample_idx][:, input_idx, :]

    # n_cells, n_features, n_trials = stimuli_3d.shape
    # n_models = len(models)
    # if labels is None:
    #     labels = [f'model {i + 1}' for i in range(n_models)]

    # n_row_cols = int(np.sqrt(n_cells))
    # fig, ax = plt.subplots(n_row_cols, n_row_cols, figsize=(20, 20))
    # if n_cells == 1:
    #     ax = np.array([[ax]])

    # point_losses = jnp.zeros((n_models, n_cells, n_trials))
    # for i, model in enumerate(models):
    #     for c in range(n_cells):
    #         params_ic = params[i][c]
    #         X_cell = stimuli_3d[c]
    #         predicted_response = model(X_cell, *params_ic)
    #         point_losses = point_losses.at[i, c].set(loss_function(predicted_response, spike_matrix[c]))

    # x_values_mean = jnp.linspace(0, 2 * jnp.pi, n_mean, endpoint=False) + 0.5 * (2 * jnp.pi / n_mean)
    # binned_mean = jnp.zeros((n_cells, n_mean))
    # for c in range(n_cells):
    #     bin_idx = jnp.clip(((stimuli_1d[c] * n_mean) / (2 * jnp.pi)).astype(jnp.int32), 0, n_mean - 1)
    #     sums = jnp.bincount(bin_idx, weights=spike_matrix[c], minlength=n_mean)
    #     counts = jnp.bincount(bin_idx, minlength=n_mean)
    #     binned_mean = binned_mean.at[c].set((sums + 1e-6) / (counts + 1e-6))

    # x_values_eval = jnp.linspace(0, 2 * jnp.pi, n_eval, endpoint=False)
    # X_eval = x_values_eval.reshape(1, -1)
    # model_outputs = jnp.zeros((n_models, n_cells, n_eval))
    # for i, model in enumerate(models):
    #     for c in range(n_cells):
    #         params_ic = params[i][c]
    #         model_outputs = model_outputs.at[i, c].set(model(X_eval, *params_ic))

    # for c in range(n_cells):
    #     row, col = divmod(c, n_row_cols)
    #     ax[row, col].scatter(stimuli_1d[c], spike_matrix[c], c='black', alpha=point_alpha, s=point_size)
    #     ax[row, col].plot(x_values_mean, binned_mean[c], 
    #                       label='Mean', color="#3BD1FF", linewidth=line_width * 1.35)
    #     for i, model in enumerate(models):
    #         ax[row, col].plot(x_values_eval, model_outputs[i, c], 
    #                           label=labels[i] + f' (loss: {jnp.mean(point_losses[i, c]):.2f})',
    #                           color=colours[i], 
    #                           alpha=line_alpha, 
    #                           linewidth=line_width)
    #     model_max = jnp.max(model_outputs[:, c])
    #     mean_max = jnp.max(binned_mean[c])
    #     ax[row, col].set_ylim(0, max(model_max, mean_max) * 2)
    #     ax[row, col].set_title(f'Sample {sample_selection[c]}', fontsize=16)
    #     ax[row, col].legend(loc='upper right', fontsize=legend_fontsize)
    #     if row == n_row_cols - 1:
    #         ax[row, col].set_xlabel('Input (x)', fontsize=20)
    #     if col == 0:
    #         ax[row, col].set_ylabel('Output (y)', fontsize=20)

def compute_evaluation_matrix(program: callable, optimized_params, eval_points : np.ndarray, **kwargs):
    """
    Compute the evaluation matrix for the new model across a range of parameter values.
    
    Args:
        program (function): the new model function to evaluate
        optimized_params (dict): dictionary of optimized parameter values for the new model
        eval_points (np.ndarray): array of input values to evaluate the model on
    Returns:
        evaluation_matrix (dict): a dictionary containing the evaluation results across parameter values
    """
    trials = eval_points[:, 0, :]  # shape (n_trials, n_features)
    n_evaluation_points = trials.shape[1]

    # vmap over samples
    program_vmap = utils.vmap_over_cells(program)
    
    n_samples = optimized_params.shape[0]
    X_eval = jnp.zeros((n_samples, n_evaluation_points))
    trials_broadcast = jnp.broadcast_to(trials, (n_samples, n_evaluation_points)) # shape (n_samples, n_evaluation_points)
    y_eval = program_vmap(trials_broadcast, optimized_params)
    
    return y_eval

# CR dkwon : fix this 
plot_single_model_fit = plot_model_fits
plot_train_vs_test_loss = plot_model_fits