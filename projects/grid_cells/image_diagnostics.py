import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
from scipy.ndimage import gaussian_filter

SIGMA = 1.0
N_BINS = 65
HIST_RANGE = ((-1.0, 1.0), (-1.0, 1.0))
EPS = 1e-6


def plot_rate_maps(programs_df, loss_function, x, y, cell_selection, save_path, **kwargs):
    def _collapse_vector(arr):
        arr = np.asarray(arr)
        if arr.ndim <= 1:
            return arr
        return np.mean(arr, axis=-1)

    def _rate_map(pos_xy, spikes, bins=N_BINS, eps=EPS, sigma=SIGMA):
        occ, xedges, yedges = np.histogram2d(
            pos_xy[:, 0],
            pos_xy[:, 1],
            bins=bins,
            range=HIST_RANGE,
        )
        spk, _, _ = np.histogram2d(
            pos_xy[:, 0],
            pos_xy[:, 1],
            bins=[xedges, yedges],
            weights=spikes,
        )
        occ_s = gaussian_filter(occ, sigma=sigma)
        spk_s = gaussian_filter(spk, sigma=sigma)
        rate = spk_s / (occ_s + eps)
        return rate, xedges, yedges

    pos = np.asarray(x)
    if pos.ndim == 2 and pos.shape[0] == 2:
        pos_xy = pos.T
    elif pos.ndim == 2 and pos.shape[1] == 2:
        pos_xy = pos
    else:
        raise ValueError(f"Grid cell diagnostics require 2D positions. Got shape {pos.shape}.")

    models = programs_df["program"].tolist()
    params = programs_df["params"].tolist()
    n_models = len(models)
    labels = kwargs.get("labels", [f"model {i + 1}" for i in range(n_models)])
    bins = kwargs.get("bins", N_BINS)
    sigma = kwargs.get("smoothing_sigma", SIGMA)

    n_cells = len(cell_selection)
    if n_cells == 0 or n_models == 0:
        logging = kwargs.get("logger")
        if logging is not None:
            logging.info("Grid cell diagnostics skipped: no cells or models to plot.")
        return
    cell_cols = kwargs.get("cell_cols")
    if cell_cols is None:
        cell_cols = 2
    cell_cols = max(1, int(cell_cols))
    n_cell_rows = int(np.ceil(n_cells / cell_cols))
    n_cols = (n_models + 1) * cell_cols
    fig_w = 3.4 * n_cols
    fig_h = 3.2 * n_cell_rows
    fig, ax = plt.subplots(n_cell_rows, n_cols, figsize=(fig_w, fig_h), squeeze=False)
    cell_title_fontsize = kwargs.get("cell_title_fontsize", 12)
    model_title_fontsize = kwargs.get("model_title_fontsize", 11)

    for i, c in enumerate(cell_selection):
        cell_row = i % n_cell_rows
        cell_col = i // n_cell_rows
        base_col = cell_col * (n_models + 1)
        spikes_raw = np.asarray(y[c])
        spikes = _collapse_vector(spikes_raw)
        data_map, xedges, yedges = _rate_map(pos_xy, spikes, bins=bins, sigma=sigma)
        data_map = data_map / (np.nanmax(data_map) + 1e-6)
        ax[cell_row, base_col].imshow(
            data_map,
            origin="lower",
            extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
            cmap="viridis",
            aspect="equal",
        )
        ax[cell_row, base_col].set_title(f"Cell {c} | data", fontsize=cell_title_fontsize)
        ax[cell_row, base_col].set_xticks([])
        ax[cell_row, base_col].set_yticks([])

        for m, model in enumerate(models):
            params_c = params[m][c]
            pred_raw = np.asarray(model(pos_xy, *params_c))
            pred = _collapse_vector(pred_raw)
            xs = np.linspace(HIST_RANGE[0][0], HIST_RANGE[0][1], bins + 1)
            ys = np.linspace(HIST_RANGE[1][0], HIST_RANGE[1][1], bins + 1)
            xc = 0.5 * (xs[:-1] + xs[1:])
            yc = 0.5 * (ys[:-1] + ys[1:])
            grid_x, grid_y = np.meshgrid(xc, yc, indexing="xy")
            grid_pos = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)
            pred_grid_raw = np.asarray(model(grid_pos, *params_c))
            pred_grid = _collapse_vector(pred_grid_raw).reshape(bins, bins)
            pred_map = pred_grid
            xedges, yedges = xs, ys
            pred_map = pred_map / (np.nanmax(pred_map) + 1e-6)
            ax[cell_row, base_col + m + 1].imshow(
                pred_map,
                origin="lower",
                extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                cmap="viridis",
                aspect="equal",
            )
            if pred_raw.shape == spikes_raw.shape:
                loss_val = loss_function(jnp.asarray(pred_raw), jnp.asarray(spikes_raw))
            else:
                loss_val = loss_function(jnp.asarray(pred), jnp.asarray(spikes))
            cell_loss = float(jnp.mean(loss_val))
            ax[cell_row, base_col + m + 1].set_title(f"{labels[m]} | loss {cell_loss:.2f}", fontsize=model_title_fontsize)
            ax[cell_row, base_col + m + 1].set_xticks([])
            ax[cell_row, base_col + m + 1].set_yticks([])

    title = kwargs.get("title", "")
    if title:
        plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path, dpi=kwargs.get("dpi", 120)) if save_path else plt.show()
    plt.close(fig)
