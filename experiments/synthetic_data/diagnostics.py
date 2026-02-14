
import numpy as np
import pandas as pd
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import jax.numpy as jnp
from typing import Optional, Callable, Sequence
from src import utils


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

def plot_model_fits(programs_df: Optional[pd.DataFrame],
                    loss_function: Callable,
                    inputs: jnp.ndarray,
                    response: jnp.ndarray,
                    sample_selection: Sequence[int],
                    plot_data: Optional[dict] = None,
                    n_eval: int = 100,
                    n_mean: int = 50,
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
    Plot model fits for synthetic data.

    Preferred path: consume precomputed `plot_data` prepared by hypothesis_engine.
    Fallback path: compute plotting tensors from `programs_df`, `inputs`, and `response`.
    """
    if plot_data is None and programs_df is None:
        raise ValueError("Either plot_data or programs_df must be provided.")

    if plot_data is None:
        sample_idx = np.asarray(sample_selection)
        if sample_idx.size == 0:
            raise ValueError("sample_selection must not be empty.")
        x_arr = jnp.asarray(inputs)
        y_arr = jnp.asarray(response)
        if x_arr.ndim == 2:
            stimuli_3d = x_arr[sample_idx][:, jnp.newaxis, :]
            stimuli_1d = x_arr[sample_idx]
        elif x_arr.ndim == 3:
            if input_idx < 0 or input_idx >= x_arr.shape[1]:
                raise ValueError(f"input_idx ({input_idx}) out of range for n_features={x_arr.shape[1]}.")
            stimuli_3d = x_arr[sample_idx]
            stimuli_1d = x_arr[sample_idx][:, input_idx, :]
        else:
            raise ValueError(f"Expected 2D or 3D inputs, got {x_arr.shape}.")

        models = programs_df['program'].tolist()
        params_list = [jnp.asarray(p)[sample_idx] for p in programs_df['params'].tolist()]
        spike_matrix = y_arr[sample_idx]
        n_models = len(models)
        n_cells, n_features, n_trials = stimuli_3d.shape
        n_row_cols = int(np.sqrt(n_cells))

        x_min = float(jnp.min(stimuli_1d))
        x_max = float(jnp.max(stimuli_1d))
        if x_max <= x_min:
            x_max = x_min + 1e-6
        x_values_mean = jnp.linspace(x_min, x_max, n_mean)
        x_values_eval = jnp.linspace(x_min, x_max, n_eval)

        point_losses = jnp.zeros((n_models, n_cells, n_trials))
        for i, model in enumerate(models):
            for c in range(n_cells):
                pred = model(stimuli_3d[c], *params_list[i][c])
                point_losses = point_losses.at[i, c].set(loss_function(pred, spike_matrix[c]))

        binned_mean = jnp.zeros((n_cells, n_mean))
        denom = max(x_max - x_min, 1e-6)
        for c in range(n_cells):
            bin_idx = jnp.clip(
                (((stimuli_1d[c] - x_min) / denom) * n_mean).astype(jnp.int32),
                0,
                n_mean - 1,
            )
            sums = jnp.bincount(bin_idx, weights=spike_matrix[c], minlength=n_mean)
            counts = jnp.bincount(bin_idx, minlength=n_mean)
            binned_mean = binned_mean.at[c].set((sums + 1e-6) / (counts + 1e-6))

        model_outputs = jnp.zeros((n_models, n_cells, n_eval))
        for i, model in enumerate(models):
            for c in range(n_cells):
                x_eval = jnp.zeros((n_features, n_eval))
                x_eval = x_eval.at[input_idx, :].set(x_values_eval)
                model_outputs = model_outputs.at[i, c].set(model(x_eval, *params_list[i][c]))
    else:
        sample_idx = np.asarray(plot_data.get('sample_selection', sample_selection))
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
    Plot fit diagnostics for a single model by delegating to `plot_model_fits`.
    """
    programs_df = pd.DataFrame({'program': [model], 'params': [params]})
    sample_selection = np.arange(y.shape[0])
    plot_model_fits(
        programs_df=programs_df,
        loss_function=loss_function,
        inputs=x,
        response=y,
        sample_selection=sample_selection,
        n_eval=n_eval,
        n_mean=n_mean,
        labels=['Model'],
        colours=["#15AC15"],
        title=title,
        dpi=dpi,
        save_path=save_path,
        input_idx=input_idx,
    )


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
