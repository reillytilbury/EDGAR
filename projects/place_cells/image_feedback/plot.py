import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt


def plot_model_fits(
    data,
    programs,
    save_path="",
    losses=None,
    sample_losses=None,
    program_names=None,
    params=None,
    n_bins: int = 50,
    max_show: int = 9,
):
    """
    Plot observed and predicted 2D rate maps for a random selection of cells.

    Args:
        data: X_disc_train dict with keys 'pos_x', 'pos_y', 'response' — shape (n_cells, n_trials).
        programs: list of Program objects with .params and .compile_model() methods.
        save_path: file path to save the figure.
        losses: list of loss values for each program, if None defaults to program.program_losses.discover.final
        sample_losses: list of arrays of shape (n_cells,) for each program, if None defaults to program.sample_losses
        program_names: optional list of names for the programs to use in the legend, if None defaults to program.name
        params: list of parameter dictionaries for each program, if None defaults to program.params
        n_bins: number of spatial bins per axis for rate maps.
        max_show: maximum number of cells to show.
    """
    if not save_path:
        raise ValueError("Please provide a save_path for the plot")

    # Default losses, sample_losses and program_names
    if losses is None:
        losses = [program.program_losses.discover.final for program in programs]
    if sample_losses is None:
        sample_losses = [program.sample_losses if program.sample_losses is not None else None for program in programs]
    if program_names is None:
        program_names = [program.name for program in programs]
    if params is None:
        params = [program.params for program in programs]

    pos_x    = np.asarray(data["pos_x"])    # (n_cells, n_trials)
    pos_y    = np.asarray(data["pos_y"])    # (n_cells, n_trials)
    response = np.asarray(data["response"]) # (n_cells, n_trials)
    n_cells  = pos_x.shape[0]
    n_models = len(programs)

    n_show         = min(max_show, n_cells)
    sample_indices = np.random.choice(n_cells, size=n_show, replace=False)
    model_fns      = [program.compile_model() for program in programs]

    x_domain = (float(pos_x.min()), float(pos_x.max()))
    y_domain = (float(pos_y.min()), float(pos_y.max()))

    fig, axes = plt.subplots(n_show, 1 + n_models, figsize=(4 * (1 + n_models), 3 * n_show))
    axes = np.atleast_2d(axes)

    for row, s in enumerate(sample_indices):
        x     = pos_x[s]
        y     = pos_y[s]
        y_obs = response[s]

        rm_obs = _bin_to_rate_map(x, y, y_obs, n_bins, x_domain, y_domain)
        im = axes[row, 0].imshow(rm_obs.T, origin="lower",
                                 extent=[*x_domain, *y_domain], cmap="viridis")
        axes[row, 0].set_title(f"Cell {s} — observed")
        fig.colorbar(im, ax=axes[row, 0], fraction=0.046, pad=0.04)

        for j, (program, model_fn) in enumerate(zip(programs, model_fns)):
            params_s = {k: np.asarray(v[s]) for k, v in params[j].items()}
            y_pred   = np.asarray(model_fn(
                {'pos_x': jnp.asarray(x), 'pos_y': jnp.asarray(y)}, params_s
            )).flatten()
            rm_pred  = _bin_to_rate_map(x, y, y_pred, n_bins, x_domain, y_domain)
            s_loss = sample_losses[j][s] if sample_losses[j] is not None else None
            title    = f"{program_names[j]} (loss={s_loss:.3f})" if s_loss is not None else program_names[j]
            im = axes[row, j + 1].imshow(rm_pred.T, origin="lower",
                                         extent=[*x_domain, *y_domain], cmap="viridis")
            axes[row, j + 1].set_title(title)
            fig.colorbar(im, ax=axes[row, j + 1], fraction=0.046, pad=0.04)

    title_parts = []
    for j, program in enumerate(programs):
        title_parts.append(f"{program_names[j]}: loss={losses[j]:.3f}" if losses[j] is not None else f"{program_names[j]}: loss=n/a")
    plt.suptitle("Rate Maps\n" + "  |  ".join(title_parts), fontsize=16)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _bin_to_rate_map(x, y, values, n_bins, x_domain, y_domain):
    """Bin values into a 2D rate map over (x, y) position."""
    edges_x        = np.linspace(x_domain[0], x_domain[1], n_bins + 1)
    edges_y        = np.linspace(y_domain[0], y_domain[1], n_bins + 1)
    occ, _, _      = np.histogram2d(x, y, bins=[edges_x, edges_y])
    weighted, _, _ = np.histogram2d(x, y, bins=[edges_x, edges_y], weights=np.asarray(values))
    return weighted / (occ + 1e-8)
