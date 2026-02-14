import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import jax.numpy as jnp
from typing import Optional, Callable

from src.diagnostics_manager import ModelFitPlotData


def select_evaluation_points(inputs: np.ndarray,
                             n_points: int = 100,
                             random_seed: int = 0,
                             **kwargs) -> np.ndarray:
    """
    Select evaluation points for default/template diagnostics.

    Strategy:
    - 1D/2D inputs: linear grid over observed range, broadcast by sample.
    - 3D inputs: sample observed trial columns with a seeded RNG.
    """
    x_arr = np.asarray(inputs)
    if x_arr.ndim <= 2:
        n_samples = 1 if x_arr.ndim == 1 else x_arr.shape[0]
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


def plot_model_fits(plot_data: ModelFitPlotData,
                    colours: list = ["#15AC15", "#EB2B2C", "#FDC91E"],
                    labels: Optional[list] = None,
                    title: str = '',
                    line_width: float = 3.0,
                    line_alpha: float = 0.95,
                    point_alpha: float = 0.15,
                    point_size: int = 24,
                    legend_fontsize: int = 9,
                    dpi: float = 100.0,
                    save_path: Optional[str] = None):
    """
    Plot model fits from precomputed `plot_data`.
    """
    sample_selection = np.asarray(plot_data['sample_selection'])
    stimuli_1d = np.asarray(plot_data['stimuli_1d'])
    spike_matrix = np.asarray(plot_data['spike_matrix'])
    point_losses = np.asarray(plot_data['point_losses'])
    x_values_mean = np.asarray(plot_data['x_values_mean'])
    binned_mean = np.asarray(plot_data['binned_mean'])
    x_values_eval = np.asarray(plot_data['x_values_eval'])
    model_outputs = np.asarray(plot_data['model_outputs'])
    n_models = int(plot_data['n_models'])
    n_cells = int(plot_data['n_cells'])
    n_row_cols = int(plot_data['n_row_cols'])

    if labels is None:
        labels = [f"model {i + 1}" for i in range(n_models)]
    if len(colours) < n_models:
        repeats = int(np.ceil(n_models / max(1, len(colours))))
        colours = (colours * repeats)[:n_models]

    fig, axes = plt.subplots(n_row_cols, n_row_cols, figsize=(4.5 * n_row_cols, 3.8 * n_row_cols))
    axes = np.array([[axes]]) if n_cells == 1 else axes

    for c in range(n_cells):
        row, col = divmod(c, n_row_cols)
        ax = axes[row, col]
        ax.scatter(stimuli_1d[c], spike_matrix[c], c='black', alpha=point_alpha, s=point_size)
        ax.plot(x_values_mean, binned_mean[c], color='#2E86DE', linewidth=line_width, label='Observed mean')
        for i in range(n_models):
            loss_val = float(np.mean(point_losses[i, c]))
            ax.plot(
                x_values_eval,
                model_outputs[i, c],
                color=colours[i],
                alpha=line_alpha,
                linewidth=line_width,
                label=f"{labels[i]} (loss={loss_val:.2f})",
            )
        ax.set_title(f"Sample {sample_selection[c]}")
        ax.set_xlabel("Input")
        ax.set_ylabel("Output")
        ax.legend(fontsize=legend_fontsize)

    if title:
        plt.suptitle(title)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95] if title else None)
    if save_path:
        plt.savefig(save_path, dpi=dpi)
    else:
        plt.show()
    plt.close(fig)
