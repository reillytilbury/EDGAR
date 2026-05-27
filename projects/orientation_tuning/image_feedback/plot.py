import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt


def plot_model_fits(data, parent_programs, save_path=""):
    """
    Plot observed and predicted orientation tuning curves for up to 9 random cells.

    Args:
        data: X_disc_train dict with keys 'stimulus', 'response' — shape (n_cells, n_trials).
        parent_programs: list of Program objects with .params and .compile() methods.
        save_path: file path to save the figure.
    """
    if not save_path:
        raise ValueError("Please provide a save path for the plot")

    stimulus = np.asarray(data['stimulus'])  # (n_cells, n_trials)
    response = np.asarray(data['response'])  # (n_cells, n_trials)
    n_cells  = stimulus.shape[0]
    colours  = ["tab:red", "tab:green", "tab:orange"]

    n_show         = min(9, n_cells)
    sample_indices = np.random.choice(n_cells, size=n_show, replace=False)
    model_fns      = [program.compile()[0] for program in parent_programs]

    fig, axes = plt.subplots(3, 3, figsize=(18, 18))

    for i, s in enumerate(sample_indices):
        ax       = axes[i // 3, i % 3]
        theta    = stimulus[s]
        y_obs    = response[s]
        x_grid   = np.linspace(theta.min(), theta.max(), 200)

        ax.scatter(theta, y_obs, s=10, c="black", alpha=0.2, label="Observed")
        ax.plot(x_grid, _binned_mean(theta, y_obs, x_grid), color="deepskyblue",
                linewidth=4, label="Binned mean", alpha=0.8)

        for j, (program, model_fn) in enumerate(zip(parent_programs, model_fns)):
            params_s = {k: np.asarray(v[s]) for k, v in program.params.items()}
            y_pred   = np.asarray(model_fn({'stimulus': jnp.asarray(x_grid)}, params_s)).flatten()
            sample_loss = program.sample_losses[s] if program.sample_losses is not None else None
            label    = f"{program.name} (loss={sample_loss:.3f})" if sample_loss is not None else f"{program.name}"
            ax.plot(x_grid, y_pred, color=colours[j % len(colours)], linewidth=3,
                    label=label, alpha=0.8)

        ax.set_xlim(float(theta.min()), float(theta.max()))
        ax.set_title(f"Cell {s}")
        ax.set_xlabel("stimulus (rad)")
        ax.set_ylabel("response")
        ax.legend(fontsize=12)

    for i in range(n_show, 9):
        axes[i // 3, i % 3].axis("off")

    title_parts = []
    for j, program in enumerate(parent_programs):
        loss = program.program_losses.discover.final
        title_parts.append(f"{program.name}: loss={loss:.3f}" if loss is not None else f"{program.name}: loss=n/a")
    plt.suptitle("Model Fits\n" + "  |  ".join(title_parts), fontsize=24)

    plt.tight_layout()
    plt.savefig(save_path, dpi=100.0, bbox_inches="tight")
    plt.close(fig)


def _binned_mean(x, y, x_grid):
    """Bin y by proximity to each x_grid point and return per-bin means."""
    if x_grid.size == 0:
        return x_grid
    if x_grid.size == 1:
        return np.array([float(np.mean(y))])

    edges = np.empty(x_grid.size + 1)
    edges[1:-1] = 0.5 * (x_grid[:-1] + x_grid[1:])
    edges[0]    = x_grid[0] - 0.5 * (x_grid[1] - x_grid[0])
    edges[-1]   = x_grid[-1] + 0.5 * (x_grid[-1] - x_grid[-2])

    bin_idx = np.digitize(x, edges) - 1
    y_mean  = np.full(x_grid.size, np.nan)
    for i in range(x_grid.size):
        vals = y[bin_idx == i]
        if vals.size > 0:
            y_mean[i] = float(np.mean(vals))

    valid = np.isfinite(y_mean)
    if np.any(valid):
        return np.interp(x_grid, x_grid[valid], y_mean[valid],
                         left=float(y_mean[valid][0]), right=float(y_mean[valid][-1]))
    return np.zeros_like(x_grid)
