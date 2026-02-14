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


def plot_single_model_fit(model: Callable,
                          loss_function: Callable,
                          x: jnp.ndarray,
                          y: jnp.ndarray,
                          params: jnp.ndarray,
                          n_eval: int = 100,
                          n_mean: int = 50,
                          dpi: float = 100.0,
                          title: str = '',
                          save_path: Optional[str] = None,
                          input_idx: int = 0):
    """
    Plot one model by assembling a minimal `plot_data` payload.
    """
    sample_selection = np.arange(y.shape[0])
    x_arr = jnp.asarray(x)
    y_arr = jnp.asarray(y)
    sample_idx = np.asarray(sample_selection)

    if x_arr.ndim == 2:
        if input_idx != 0:
            raise ValueError(f"input_idx must be 0 for 2D input, got {input_idx}.")
        stimuli_3d = x_arr[sample_idx][:, jnp.newaxis, :]
        stimuli_1d = x_arr[sample_idx]
    elif x_arr.ndim == 3:
        if input_idx < 0 or input_idx >= x_arr.shape[1]:
            raise ValueError(f"input_idx ({input_idx}) out of range for n_features={x_arr.shape[1]}.")
        stimuli_3d = x_arr[sample_idx]
        stimuli_1d = x_arr[sample_idx][:, input_idx, :]
    else:
        raise ValueError(f"Expected 2D or 3D inputs, got {x_arr.shape}.")

    spike_matrix = y_arr[sample_idx]
    params_sel = jnp.asarray(params)[sample_idx]
    n_cells, n_features, n_trials = stimuli_3d.shape
    n_row_cols = int(np.sqrt(n_cells))
    if n_row_cols * n_row_cols != n_cells:
        raise ValueError(f"n_cells must be a square number for plotting, got {n_cells}.")

    x_min = float(jnp.min(stimuli_1d))
    x_max = float(jnp.max(stimuli_1d))
    if x_max <= x_min:
        x_max = x_min + 1e-6
    x_values_mean = jnp.linspace(x_min, x_max, n_mean)
    x_values_eval = jnp.linspace(x_min, x_max, n_eval)

    trial_predictions = jnp.zeros((1, n_cells, n_trials))
    point_losses = jnp.zeros((1, n_cells, n_trials))
    for c in range(n_cells):
        pred = jnp.squeeze(jnp.asarray(model(stimuli_3d[c], *params_sel[c])))
        if pred.ndim == 0:
            pred = jnp.broadcast_to(pred, (n_trials,))
        trial_predictions = trial_predictions.at[0, c].set(pred)
        losses = jnp.squeeze(jnp.asarray(loss_function(pred, spike_matrix[c])))
        if losses.ndim == 0:
            losses = jnp.broadcast_to(losses, (n_trials,))
        point_losses = point_losses.at[0, c].set(losses)

    binned_mean = jnp.zeros((n_cells, n_mean))
    denom = max(x_max - x_min, 1e-6)
    for c in range(n_cells):
        bin_idx = jnp.clip((((stimuli_1d[c] - x_min) / denom) * n_mean).astype(jnp.int32), 0, n_mean - 1)
        sums = jnp.bincount(bin_idx, weights=spike_matrix[c], minlength=n_mean)
        counts = jnp.bincount(bin_idx, minlength=n_mean)
        binned_mean = binned_mean.at[c].set((sums + 1e-6) / (counts + 1e-6))

    model_outputs = jnp.zeros((1, n_cells, n_eval))
    for c in range(n_cells):
        x_eval = jnp.zeros((n_features, n_eval))
        x_eval = x_eval.at[input_idx, :].set(x_values_eval)
        y_eval = jnp.squeeze(jnp.asarray(model(x_eval, *params_sel[c])))
        if y_eval.ndim == 0:
            y_eval = jnp.broadcast_to(y_eval, (n_eval,))
        model_outputs = model_outputs.at[0, c].set(y_eval)

    plot_data: ModelFitPlotData = {
        'sample_selection': sample_idx,
        'stimuli_3d': stimuli_3d,
        'stimuli_1d': stimuli_1d,
        'spike_matrix': spike_matrix,
        'trial_predictions': trial_predictions,
        'point_losses': point_losses,
        'x_values_mean': x_values_mean,
        'binned_mean': binned_mean,
        'x_values_eval': x_values_eval,
        'model_outputs': model_outputs,
        'n_row_cols': n_row_cols,
        'n_models': 1,
        'n_cells': n_cells,
        'n_trials': n_trials,
        'n_eval': int(n_eval),
        'n_mean': int(n_mean),
        'input_idx': int(input_idx),
    }

    plot_model_fits(
        plot_data=plot_data,
        labels=['Model'],
        colours=["#15AC15"],
        title=title,
        dpi=dpi,
        save_path=save_path,
    )


def plot_train_vs_test_loss(programs_df: pd.DataFrame,
                            island_labels: list,
                            save_path: Optional[str] = None):
    """
    Plot train-vs-test loss scatter.
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
