
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
    n_eval = min(int(n_points), n_trials)
    rng = np.random.default_rng(random_seed)
    trial_idx = rng.choice(n_trials, size=n_eval, replace=False)
    return x_arr[:, :, trial_idx]

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
    Plot model fits for synthetic data.
    """
    sample_idx = np.asarray(plot_data['sample_selection'])
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

    if labels is None:
        labels = [f'model {i + 1}' for i in range(n_models)]

    if len(colours) < n_models:
        repeats = int(np.ceil(n_models / max(len(colours), 1)))
        colours = (colours * repeats)[:n_models]

    fig, axes = plt.subplots(n_row_cols, n_row_cols, figsize=(15, 10))
    axes = np.array([[axes]]) if n_cells == 1 else axes

    for c in range(n_cells):
        row, col = divmod(c, n_row_cols)
        ax = axes[row, col]
        ax.scatter(np.asarray(stimuli_1d[c]), np.asarray(spike_matrix[c]), c='black', alpha=point_alpha, s=point_size)
        ax.plot(np.asarray(x_values_mean), np.asarray(binned_mean[c]), color='blue', alpha=0.8, linewidth=line_width, label='Binned observed mean')
        for i in range(n_models):
            loss_val = float(jnp.mean(point_losses[i, c]))
            ax.plot(
                np.asarray(x_values_eval),
                np.asarray(model_outputs[i, c]),
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


def plot_train_vs_test_loss(programs_df: pd.DataFrame,
                            island_labels: list,
                            save_path: Optional[str] = None):
    """
    Plot train-vs-test loss scatter for synthetic-data experiments.
    """
    if 'train_loss' not in programs_df.columns or 'test_loss' not in programs_df.columns:
        raise ValueError("DataFrame must contain 'train_loss' and 'test_loss' columns.")

    train_loss = np.asarray(programs_df['train_loss'].to_numpy(), dtype=float)
    test_loss = np.asarray(programs_df['test_loss'].to_numpy(), dtype=float)
    train_loss = np.nan_to_num(train_loss, nan=np.inf)
    test_loss = np.nan_to_num(test_loss, nan=np.inf)

    mask = np.isfinite(train_loss) & np.isfinite(test_loss)
    train_loss = train_loss[mask]
    test_loss = test_loss[mask]

    plt.figure(figsize=(8, 8))
    plt.scatter(train_loss, test_loss, alpha=0.8, color='tab:blue', edgecolor='none')
    if train_loss.size > 0 and test_loss.size > 0:
        lo = min(float(np.min(train_loss)), float(np.min(test_loss)))
        hi = max(float(np.max(train_loss)), float(np.max(test_loss)))
        plt.plot([lo, hi], [lo, hi], 'k--', alpha=0.6)
        plt.xlim(lo * 0.95, hi * 1.05)
        plt.ylim(lo * 0.95, hi * 1.05)
    plt.xlabel('Train Loss')
    plt.ylabel('Test Loss')
    plt.title('Train vs Test Loss')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()
