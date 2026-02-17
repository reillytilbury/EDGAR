
import numpy as np
import pandas as pd
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import jax.numpy as jnp
from typing import Optional, Callable
from src import utils
from src.diagnostics_manager import ModelFitPlotData


def select_evaluation_points(inputs: jnp.ndarray,
                             n_points: int = 100,
                             random_seed: int = 0,
                             input_idx: int = 0,
                             **kwargs) -> jnp.ndarray:
    """
    Select evaluation points for synthetic-data diagnostics.

    Purpose:
    - Make evaluation-point selection explicit and experiment-local.
    - Provide reproducible points for model comparison and cached evaluations.

    Strategy:
    - Single-input `(n_samples, n_trials)`: linear grid over observed input range.
    - Multi-input `(n_samples, n_features, n_trials)`: sample trial columns from
      observed data using a seeded RNG (preserves feature correlations).
    """
    x_arr = jnp.asarray(inputs)
    if x_arr.ndim == 2:
        n_samples = int(x_arr.shape[0])
        x_min = float(jnp.min(x_arr))
        x_max = float(jnp.max(x_arr))
        if x_max <= x_min:
            x_max = x_min + 1e-6
        grid = jnp.linspace(x_min, x_max, n_points)
        return jnp.broadcast_to(grid, (n_samples, n_points))

    if x_arr.ndim != 3:
        raise ValueError(f"Expected 2D or 3D inputs, got shape {x_arr.shape}.")

    n_trials = int(x_arr.shape[2])
    if n_trials < n_points:
        raise ValueError(f"Requested n_points ({n_points}) exceeds available trials ({n_trials}).")
    n_eval = min(int(n_points), n_trials)
    rng = np.random.default_rng(random_seed)
    trial_idx = rng.choice(n_trials, size=n_eval, replace=False)
    return x_arr[:, :, trial_idx]

def plot_model_fits(plot_data: ModelFitPlotData,
                    colours: list = ["#FDC91E", "#15AC15", '#EB2B2C'],
                    labels: Optional[list] = None,
                    title: str = '',
                    n_mean: int = 50,
                    line_width=4.0,
                    line_alpha=1.0,
                    point_alpha=0.1,
                    point_size: int = 80,
                    legend_fontsize: int = 8,
                    dpi: float = 100.0,
                    save_path: Optional[str] = None):
    """
    Plot model fits for synthetic data.
    """
    sample_idx = np.asarray(plot_data['sample_selection'])
    inputs_plot = jnp.asarray(plot_data['inputs_plot'])
    observed_outputs = jnp.asarray(plot_data['observed_outputs'])
    trial_predictions = jnp.asarray(plot_data['trial_predictions'])
    model_loss_dict = plot_data['model_loss_dict'] # dict of model_name -> loss array (n_samples,)
    n_models = int(plot_data['n_models'])
    n_samples = int(plot_data['n_samples'])
    n_grid_side = int(plot_data['n_grid_side'])

    if labels is None:
        labels = [f'model {i + 1}' for i in range(n_models)]

    if len(colours) < n_models:
        repeats = int(np.ceil(n_models / max(len(colours), 1)))
        colours = (colours * repeats)[:n_models]

    fig, axes = plt.subplots(n_grid_side, n_grid_side, figsize=(n_grid_side * 5, n_grid_side * 5))
    axes = np.array([[axes]]) if n_samples == 1 else axes

    x_min = float(jnp.min(inputs_plot))
    x_max = float(jnp.max(inputs_plot))
    if x_max <= x_min:
        x_max = x_min + 1e-6
    x_values_mean = jnp.linspace(x_min, x_max, n_mean)
    binned_mean = jnp.zeros((n_samples, n_mean))
    denom = max(x_max - x_min, 1e-6)
    for c in range(n_samples):
        bin_idx = jnp.clip(
            (((inputs_plot[c] - x_min) / denom) * n_mean).astype(jnp.int32),
            0,
            n_mean - 1,
        )
        sums = jnp.bincount(bin_idx, weights=observed_outputs[c], minlength=n_mean)
        counts = jnp.bincount(bin_idx, minlength=n_mean)
        binned_mean = binned_mean.at[c].set((sums + 1e-6) / (counts + 1e-6))

    for c in range(n_samples):
        row, col = divmod(c, n_grid_side)
        ax = axes[row, col]
        ax.scatter(np.asarray(inputs_plot[c]), np.asarray(observed_outputs[c]), c='black', alpha=point_alpha, s=point_size)
        ax.plot(np.asarray(x_values_mean), np.asarray(binned_mean[c]), color='blue', alpha=0.8, linewidth=line_width, label='Binned observed mean')
        sort_idx = jnp.argsort(inputs_plot[c])
        x_sorted = np.asarray(inputs_plot[c][sort_idx])
        for i in range(n_models):
            loss_val = float(model_loss_dict[i][c])
            ax.plot(
                x_sorted,
                np.asarray(trial_predictions[i, c][sort_idx]),
                color=colours[i],
                alpha=line_alpha,
                linewidth=line_width,
                label=f"{labels[i]} (loss={loss_val:.2f})",
            )
        ax.set_title(f"Sample {sample_idx[c]}")
        ax.set_xlabel("Input (x)")
        ax.set_ylabel("Output (y)")
        ax.legend(fontsize=legend_fontsize)

    plt.tight_layout()
    if title:
        plt.suptitle(title, y=1.02)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    if save_path:
        plt.savefig(save_path, dpi=dpi)
    else:
        plt.show()
    plt.close()
