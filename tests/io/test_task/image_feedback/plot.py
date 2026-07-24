import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt


def plot_model_fits(
    data,
    programs,
    rng: np.random.Generator,
    save_path="",
    losses=None,
    sample_losses=None,
    program_names=None,
    params=None,
):
    """
    Plot observed synthetic data and model predictions for up to 9 random samples.

    Args:
        data: X_disc_train dict with keys 'x', 'y' — shape (n_samples, n_trials).
        programs: list of Program objects with .params and .compile_model() methods.
        save_path: file path to save the figure.
        losses: list of loss values for each program, if None defaults to program.program_losses.discover.final
        sample_losses: list of arrays of shape (n_cells,) for each program, if None defaults to program.sample_losses
        program_names: optional list of names for the programs to use in the legend, if None defaults to program.name
        params: list of parameter dictionaries for each program, if None defaults to program.params
    """
    if not save_path:
        raise ValueError("Please provide a save path for the plot")

    # Default losses, sample_losses and program_names
    if losses is None:
        losses = [program.program_losses.discover.final for program in programs]
    if sample_losses is None:
        sample_losses = [
            program.sample_losses if program.sample_losses is not None else None
            for program in programs
        ]
    if program_names is None:
        program_names = [program.name for program in programs]
    if params is None:
        params = [program.params for program in programs]

    x_arr = np.asarray(data["x"])  # (n_samples, n_trials)
    y_arr = np.asarray(data["y"])  # (n_samples, n_trials)
    n_samples = x_arr.shape[0]
    colours = ["tab:red", "tab:green", "tab:orange"]
    n_show = min(9, n_samples)
    sample_indices = rng.choice(n_samples, size=n_show, replace=False)
    model_fns = [program.compile_model() for program in programs]

    fig, axes = plt.subplots(3, 3, figsize=(18, 18))

    for i, s in enumerate(sample_indices):
        ax = axes[i // 3, i % 3]
        x_obs = x_arr[s]
        y_obs = y_arr[s]
        x_grid = np.linspace(x_obs.min(), x_obs.max(), 200)

        ax.scatter(x_obs, y_obs, s=8, c="black", alpha=0.15, label="Observed")
        ax.plot(
            x_grid,
            _binned_mean(x_obs, y_obs, x_grid),
            color="deepskyblue",
            linewidth=3,
            label="Binned mean",
            alpha=0.8,
        )

        for j, (program, model_fn) in enumerate(zip(programs, model_fns)):
            params_s = {k: np.asarray(v[s]) for k, v in params[j].items()}
            y_pred = np.asarray(
                model_fn({"x": jnp.asarray(x_grid)}, params_s)
            ).flatten()
            s_loss = sample_losses[j][s] if sample_losses[j] is not None else None
            label = (
                f"{program_names[j]} (loss={s_loss:.3f})"
                if s_loss is not None
                else f"{program_names[j]}"
            )
            ax.plot(
                x_grid,
                y_pred,
                color=colours[j % len(colours)],
                linewidth=2.5,
                label=label,
                alpha=0.85,
            )

        ax.set_title(f"Sample {s}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.legend(fontsize=10)

    for i in range(n_show, 9):
        axes[i // 3, i % 3].axis("off")

    title_parts = []
    for j, program in enumerate(programs):
        title_parts.append(
            f"{program_names[j]}: loss={losses[j]:.3f}"
            if losses[j] is not None
            else f"{program_names[j]}: loss=n/a"
        )
    plt.suptitle("Model Fits\n" + "  |  ".join(title_parts), fontsize=16)

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
    edges[0] = x_grid[0] - 0.5 * (x_grid[1] - x_grid[0])
    edges[-1] = x_grid[-1] + 0.5 * (x_grid[-1] - x_grid[-2])

    bin_idx = np.digitize(x, edges) - 1
    y_mean = np.full(x_grid.size, np.nan)
    for i in range(x_grid.size):
        vals = y[bin_idx == i]
        if vals.size > 0:
            y_mean[i] = float(np.mean(vals))

    valid = np.isfinite(y_mean)
    if np.any(valid):
        return np.interp(
            x_grid,
            x_grid[valid],
            y_mean[valid],
            left=float(y_mean[valid][0]),
            right=float(y_mean[valid][-1]),
        )
    return np.zeros_like(x_grid)
