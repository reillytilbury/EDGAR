import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp

SIGMA = 2.0
N_BINS = 24


def plot_rate_maps(programs_df, loss_function, x, y, cell_selection, save_path, **kwargs):
    def _collapse_vector(arr):
        arr = np.asarray(arr)
        if arr.ndim <= 1:
            return arr
        return np.mean(arr, axis=-1)

    def _rate_map(pos_xy, spikes, bins=N_BINS, eps=1e-6, sigma=SIGMA, beta=None):
        def _gaussian_kernel(sigma):
            if sigma <= 0:
                return None
            size = int(max(3, np.ceil(6 * sigma)))
            if size % 2 == 0:
                size += 1
            ax = np.arange(size) - size // 2
            xx, yy = np.meshgrid(ax, ax)
            kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
            kernel /= np.sum(kernel)
            return kernel

        def _fft_convolve2d(a, b):
            out_shape = (a.shape[0] + b.shape[0] - 1, a.shape[1] + b.shape[1] - 1)
            fa = np.fft.rfftn(a, s=out_shape)
            fb = np.fft.rfftn(b, s=out_shape)
            out = np.fft.irfftn(fa * fb, s=out_shape)
            start0 = (b.shape[0] - 1) // 2
            start1 = (b.shape[1] - 1) // 2
            return out[start0:start0 + a.shape[0], start1:start1 + a.shape[1]]

        occ, xedges, yedges = np.histogram2d(pos_xy[:, 0], pos_xy[:, 1], bins=bins)
        spk, _, _ = np.histogram2d(pos_xy[:, 0], pos_xy[:, 1], bins=[xedges, yedges], weights=spikes)
        kernel = _gaussian_kernel(sigma)
        if beta is None:
            nz = np.asarray(spikes)
            nz = nz[nz > 0]
            beta = float(np.min(nz)) if nz.size > 0 else 1e-3
        if kernel is None:
            rate = spk / (occ + beta**2 + eps)
        else:
            occ_s = _fft_convolve2d(occ, kernel)
            spk_s = _fft_convolve2d(spk, kernel)
            rate = spk_s / (occ_s + beta**2 + eps)
        return rate.T, xedges, yedges

    stim = np.asarray(x)
    if stim.ndim != 2:
        raise ValueError(f"Place cell diagnostics require 2D stimuli. Got shape {stim.shape}.")
    if stim.shape[0] < 2 and stim.shape[1] < 2:
        raise ValueError(f"Stimuli must include at least x,y. Got shape {stim.shape}.")
    if stim.shape[0] >= 2:
        stimuli_full = stim.T
    else:
        stimuli_full = stim
    pos_xy = stimuli_full[:, :2]
    if pos_xy.shape[1] != 2:
        raise ValueError(f"Place cell diagnostics require x,y in stimuli. Got shape {pos_xy.shape}.")

    models = programs_df["program"].tolist()
    params = programs_df["params"].tolist()
    n_models = len(models)
    labels = kwargs.get("labels", [f"model {i + 1}" for i in range(n_models)])
    bins = kwargs.get("bins", N_BINS)
    sigma = kwargs.get("smoothing_sigma", SIGMA)

    n_cells = len(cell_selection)
    has_hd_speed = stimuli_full.shape[1] >= 4
    feature_cols = 3 if has_hd_speed else 1
    cell_cols = kwargs.get("cell_cols")
    if cell_cols is None:
        cell_cols = 1 if has_hd_speed else 2
    cell_cols = max(1, int(cell_cols))
    n_cell_rows = int(np.ceil(n_cells / cell_cols))
    n_cols = (n_models + 1) * feature_cols * cell_cols
    fig_w = 3.0 * n_cols
    fig_h = 3.0 * n_cell_rows
    fig, ax = plt.subplots(n_cell_rows, n_cols, figsize=(fig_w, fig_h), squeeze=False)
    cell_title_fontsize = kwargs.get("cell_title_fontsize", 12)
    model_title_fontsize = kwargs.get("model_title_fontsize", 11)

    def _tuning_curve(values, rates, n_bins, vmin, vmax, min_count=5):
        bins = np.linspace(vmin, vmax, n_bins + 1)
        idx = np.digitize(values, bins) - 1
        idx = np.clip(idx, 0, n_bins - 1)
        sums = np.bincount(idx, weights=rates, minlength=n_bins).astype(float)
        counts = np.bincount(idx, minlength=n_bins).astype(float)
        with np.errstate(invalid="ignore", divide="ignore"):
            curve = sums / counts
        curve[counts < min_count] = np.nan
        centers = 0.5 * (bins[:-1] + bins[1:])
        return centers, curve

    for i, c in enumerate(cell_selection):
        cell_row = i % n_cell_rows
        cell_col = i // n_cell_rows
        base_col = cell_col * (n_models + 1) * feature_cols
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

        if has_hd_speed:
            hd = stimuli_full[:, 2]
            speed = stimuli_full[:, 3]
            hd_centers, hd_curve = _tuning_curve(hd, spikes, n_bins=24, vmin=-np.pi, vmax=np.pi)
            sp_max = np.percentile(speed, 99.0)
            sp_centers, sp_curve = _tuning_curve(speed, spikes, n_bins=24, vmin=0.0, vmax=sp_max)
            ax[cell_row, base_col + 1].plot(hd_centers, hd_curve, color="tab:blue", lw=1.5)
            ax[cell_row, base_col + 1].set_title("HD data", fontsize=model_title_fontsize)
            ax[cell_row, base_col + 1].set_xticks([])
            ax[cell_row, base_col + 1].set_yticks([])
            ax[cell_row, base_col + 2].plot(sp_centers, sp_curve, color="tab:orange", lw=1.5)
            ax[cell_row, base_col + 2].set_title("Speed data", fontsize=model_title_fontsize)
            ax[cell_row, base_col + 2].set_xticks([])
            ax[cell_row, base_col + 2].set_yticks([])

        for m, model in enumerate(models):
            params_c = params[m][c]
            pred_raw = np.asarray(model(stimuli_full, *params_c))
            pred = _collapse_vector(pred_raw)
            pred_map, xedges, yedges = _rate_map(pos_xy, pred, bins=bins, sigma=sigma)
            pred_map = pred_map / (np.nanmax(pred_map) + 1e-6)
            model_col = base_col + feature_cols * (m + 1)
            ax[cell_row, model_col].imshow(
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
            ax[cell_row, model_col].set_title(f"{labels[m]} | loss {cell_loss:.2f}", fontsize=model_title_fontsize)
            ax[cell_row, model_col].set_xticks([])
            ax[cell_row, model_col].set_yticks([])
            if has_hd_speed:
                hd_centers, hd_curve = _tuning_curve(hd, pred, n_bins=24, vmin=-np.pi, vmax=np.pi)
                sp_centers, sp_curve = _tuning_curve(speed, pred, n_bins=24, vmin=0.0, vmax=sp_max)
                ax[cell_row, model_col + 1].plot(hd_centers, hd_curve, color="tab:blue", lw=1.5)
                ax[cell_row, model_col + 1].set_title("HD", fontsize=model_title_fontsize)
                ax[cell_row, model_col + 1].set_xticks([])
                ax[cell_row, model_col + 1].set_yticks([])
                ax[cell_row, model_col + 2].plot(sp_centers, sp_curve, color="tab:orange", lw=1.5)
                ax[cell_row, model_col + 2].set_title("Speed", fontsize=model_title_fontsize)
                ax[cell_row, model_col + 2].set_xticks([])
                ax[cell_row, model_col + 2].set_yticks([])

    title = kwargs.get("title", "")
    if title:
        plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path, dpi=kwargs.get("dpi", 120)) if save_path else plt.show()
    plt.close(fig)
