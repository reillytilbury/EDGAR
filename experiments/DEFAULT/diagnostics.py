import numpy as np
import matplotlib.pyplot as plt


def select_evaluation_points(inputs: np.ndarray,
                             n_points: int = 100,
                             random_seed: int = 0,
                             **kwargs) -> np.ndarray:
    """
    Select evaluation points for default/template diagnostics.

    Purpose:
    - Provide a clear extension point for experiment-specific evaluation policy.
    - Return explicit evaluation points that downstream code can pass to model evaluation.

    Strategy:
    - If `inputs` is 1D or 2D, return a linear grid across observed min/max and
      broadcast per sample.
    - If `inputs` is 3D, sample trial columns with a seeded RNG.
    """
    x_arr = np.asarray(inputs)
    if x_arr.ndim <= 2:
        if x_arr.ndim == 1:
            n_samples = 1
        else:
            n_samples = x_arr.shape[0]
        x_min = float(np.min(x_arr))
        x_max = float(np.max(x_arr))
        if x_max <= x_min:
            x_max = x_min + 1e-6
        grid = np.linspace(x_min, x_max, n_points)
        return np.broadcast_to(grid, (n_samples, n_points))

    if x_arr.ndim == 3:
        n_trials = x_arr.shape[2]
        n_eval = min(int(n_points), int(n_trials))
        rng = np.random.default_rng(random_seed)
        trial_idx = rng.choice(n_trials, size=n_eval, replace=False)
        return x_arr[:, :, trial_idx]

    raise ValueError(f"Expected 1D, 2D, or 3D inputs, got shape {x_arr.shape}.")

def plot_model_fits(x, y_true, 
                    y_pred_v1, y_pred_v2, 
                    loss_v1, loss_v2,
                    n_bins=10, n_rows_plot=3, n_cols_plot=3, save_path=''):
    """
    Plot the true vs predicted values to visualize model fit.
    
    Args:
        x (n_trials,): input data
        y_true (n_samples, n_trials): true output values
        y_pred_v1 (n_samples, n_trials): predicted output values from model version 1
        y_pred_v2 (n_samples, n_trials): predicted output values from model version 2
        loss_v1 (float,): loss value for model version 1
        loss_v2 (float,): loss value for model version 2
        n_bins (int): number of bins for plotting
        n_rows_plot (int): number of rows in the plot grid
        n_cols_plot (int): number of columns in the plot grid
        save_path (str): path to save the plot
    """
    # Subsample the data for plotting and compute losses
    n_samples_to_plot = n_rows_plot * n_cols_plot
    sample_indices = np.random.choice(y_true.shape[0], n_samples_to_plot, replace=False)
    y_true_sample = y_true[sample_indices]
    y_pred_v1_sample = y_pred_v1[sample_indices]
    y_pred_v2_sample = y_pred_v2[sample_indices]
    v1_loss = np.mean((y_true_sample - y_pred_v1_sample) ** 2, axis=1)
    v2_loss = np.mean((y_true_sample - y_pred_v2_sample) ** 2, axis=1)

    # Evaluate fit by binning the data and plotting mean true vs mean predicted values in each bin
    bins = np.linspace(x.min(), x.max(), n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    # Interpolate predictions for smoother curves
    x_interp = np.linspace(x.min(), x.max(), 100)

    # Plot the results
    fig, axes = plt.subplots(n_rows_plot, n_cols_plot, figsize=(15, 10))
    for i, ax in enumerate(axes.flatten()):
        # Scatter plot for observed data
        ax.scatter(x, y_true_sample[i], label='Observed (with noise)', alpha=0.2, s=10, c='black')

        # Binned mean for true data
        mean_true = np.array([y_true_sample[i][(x >= bins[j]) & (x < bins[j+1])].mean() for j in range(n_bins)])
        ax.plot(bin_centers, mean_true, label='Binned observed mean', color='blue', alpha=0.7, linewidth=4, marker='o')

        # Interpolated predictions
        y_pred_v1_interp = np.interp(x_interp, x, y_pred_v1_sample[i])
        y_pred_v2_interp = np.interp(x_interp, x, y_pred_v2_sample[i])
        ax.plot(x_interp, y_pred_v1_interp, label=f'v1. loss = {v1_loss[i]:.2f}', color='green', alpha=0.7, linewidth=4)
        ax.plot(x_interp, y_pred_v2_interp, label=f'v2. loss = {v2_loss[i]:.2f}', color='red', alpha=0.7, linewidth=4)

        ax.set_title(f'Sample {sample_indices[i]}')
        ax.set_xlabel('Input (x)')
        ax.set_ylabel('Output (y)')
        ax.legend()

    plt.tight_layout()
    plt.suptitle(f'Model Fit Comparison (v1 loss = {loss_v1:.2f}, v2 loss = {loss_v2:.2f})', y=1.02)
    plt.savefig(save_path)
    plt.close()
