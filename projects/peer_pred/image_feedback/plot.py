import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from src import utils


def plot_model_fits(
    data,
    programs_list,
    data_eval,
    save_path="",
    labels=None,
    title_prefix: str | None = None,
):
    """
    Plot observed activity and model predictions for random target cells.

    Shows target cells in a 3x3 grid over a random contiguous time block.
    """
    if save_path == "":
        raise ValueError("Please provide a save path for the plot")

    def _to_array3d(obj) -> np.ndarray:
        if hasattr(obj, "to_tensor"):
            arr = np.asarray(obj.to_tensor())
        else:
            arr = np.asarray(obj)
        if arr.ndim == 2:
            return arr[:, np.newaxis, :]
        return arr

    x_arr = _to_array3d(data["source"])
    y_arr = _to_array3d(data["target"])
    n_samples, n_features, n_trials = x_arr.shape
    _, n_targets, _ = y_arr.shape

    sample_idx = 0
    x = x_arr[sample_idx]  # (n_source, n_time)
    y = y_arr[sample_idx]  # (n_target, n_time)

    block_len = 360
    if n_trials <= block_len:
        sl = slice(0, n_trials)
    else:
        rng = np.random.default_rng()
        start = block_len * rng.integers(0, max(1, n_trials // block_len))
        sl = slice(start, min(start + block_len, n_trials))

    fig, axes = plt.subplots(
        3,
        3,
        figsize=(21, 14),
        gridspec_kw={"width_ratios": [1.25, 1.25, 1.0], "height_ratios": [1.0, 1.0, 0.8]},
    )

    # Precompute predictions and overall losses.
    preds_by_model = []
    model_losses = []
    for program in programs_list:
        model = program["model"]
        params = utils.slice_params(
            utils.broadcast_params(program["params"], n_samples), sample_idx
        )
        sample_data = {"source": x}
        y_pred = utils.call_model(model, sample_data, params)
        y_pred = np.asarray(y_pred)
        if y_pred.ndim == 1:
            y_pred = y_pred[None, :]
        preds_by_model.append(y_pred)

        if "losses" in program:
            try:
                model_losses.append(float(np.asarray(program["losses"])[sample_idx]))
            except Exception:
                model_losses.append(float(np.mean((y_pred - y) ** 2)))
        else:
            try:
                sample_data_with_target = {"source": x, "target": y}
                loss_vals = (sample_data_with_target["target"] - y_pred) ** 2
                model_losses.append(float(np.mean(loss_vals)))
            except Exception:
                model_losses.append(float(np.mean((y_pred - y) ** 2)))

    def _model_label(j: int) -> str:
        if labels is not None and j < len(labels):
            return str(labels[j])
        return f"Model v{j+1}"

    def _pc1_order(cell_by_trial: np.ndarray) -> np.ndarray:
        arr = np.asarray(cell_by_trial, dtype=float)
        if arr.ndim != 2:
            return np.arange(0, dtype=int)
        n_cells, n_t = arr.shape
        if n_cells <= 1:
            return np.arange(n_cells, dtype=int)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        try:
            arr_centered = arr - arr.mean(axis=1, keepdims=True)
            u, s, _ = np.linalg.svd(arr_centered, full_matrices=False)
            if u.shape[0] == n_cells and s.size > 0:
                pc1_scores = u[:, 0] * s[0]
                return np.argsort(pc1_scores)
        except Exception:
            pass
        return np.argsort(np.nanargmax(arr, axis=1))

    def _positive_vmax(arr: np.ndarray, pct: float = 99.0) -> float:
        pos = np.clip(np.asarray(arr, dtype=float), 0.0, None)
        vmax = float(np.nanpercentile(pos, pct))
        if not np.isfinite(vmax) or vmax <= 1e-12:
            vmax = 1.0
        return vmax

    def _obs_gray_rgb(obs: np.ndarray) -> np.ndarray:
        arr = np.nan_to_num(np.asarray(obs, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
        vmax = _positive_vmax(arr)
        norm = np.clip(np.clip(arr, 0.0, None) / vmax, 0.0, 1.0)
        gray = 1.0 - norm
        return np.repeat(gray[:, :, None], 3, axis=2)

    def _pred_color_rgb(pred: np.ndarray, color: str) -> np.ndarray:
        arr = np.nan_to_num(np.asarray(pred, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
        vmax = _positive_vmax(arr)
        norm = np.clip(np.clip(arr, 0.0, None) / vmax, 0.0, 1.0)
        rgb = np.ones((arr.shape[0], arr.shape[1], 3), dtype=float)
        if color == "red":
            rgb[:, :, 1] = 1.0 - norm
            rgb[:, :, 2] = 1.0 - norm
        else:  # blue
            rgb[:, :, 0] = 1.0 - norm
            rgb[:, :, 1] = 1.0 - norm
        return rgb

    def _residual_rgb(residual: np.ndarray, cmap_name: str = "BrBG"):
        arr = np.nan_to_num(np.asarray(residual, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
        rv = float(np.nanpercentile(np.abs(arr), 99))
        if not np.isfinite(rv) or rv <= 1e-12:
            rv = 1.0
        norm = mcolors.TwoSlopeNorm(vmin=-rv, vcenter=0.0, vmax=rv)
        cmap = plt.get_cmap(cmap_name)
        rgb = cmap(norm(arr))[..., :3]
        return rgb, norm, cmap

    y_block = y[:, sl]
    order = _pc1_order(y_block)
    plot_stride = 4
    order_plot = order[::plot_stride] if order.size > 0 else order
    if order_plot.size == 0 and order.size > 0:
        order_plot = order[:1]
    y_sorted = y_block[order_plot, :]
    pred_blocks = [np.asarray(pred[:, sl], dtype=float) for pred in preds_by_model]

    # Left column: observed stacked on prediction.
    for row in range(2):
        ax = axes[row, 0]
        if row >= len(pred_blocks):
            ax.axis("off")
            continue
        pred_sorted = pred_blocks[row][order_plot, :]
        panel = np.concatenate(
            [_obs_gray_rgb(y_sorted), _pred_color_rgb(pred_sorted, "red" if row == 0 else "blue")],
            axis=0,
        )
        ax.imshow(panel, aspect="auto", interpolation="none")
        n_cells_plot = y_sorted.shape[0]
        ax.axhline(n_cells_plot - 0.5, color="white", linewidth=1.0, alpha=0.95)
        ax.set_yticks([n_cells_plot / 2.0, n_cells_plot + n_cells_plot / 2.0])
        if row == 0:
            ax.set_yticklabels(["Y_obs (gray)", "pred v1 (red)"])
        else:
            ax.set_yticklabels(["Y_obs (gray)", "pred v2 (blue)"])
        ax.set_title(f"{_model_label(row)}: observed + prediction")
        if row == 1:
            ax.set_xlabel("time (sec)")
        else:
            ax.set_xticks([])

    # Middle column: observed stacked on residual.
    residual_cmap = "BrBG"
    for row in range(2):
        ax = axes[row, 1]
        if row >= len(pred_blocks):
            ax.axis("off")
            continue
        residual = y_sorted - pred_blocks[row][order_plot, :]
        residual_rgb, residual_norm, cmap = _residual_rgb(residual, cmap_name=residual_cmap)
        panel = np.concatenate([_obs_gray_rgb(y_sorted), residual_rgb], axis=0)
        ax.imshow(panel, aspect="auto", interpolation="none")
        n_cells_plot = y_sorted.shape[0]
        ax.axhline(n_cells_plot - 0.5, color="white", linewidth=1.0, alpha=0.95)
        ax.set_yticks([n_cells_plot / 2.0, n_cells_plot + n_cells_plot / 2.0])
        if row == 0:
            ax.set_yticklabels(["Y_obs (gray)", "resid (Y_obs - v1)"])
            ax.set_title("Observed + residual (Y_obs - v1)")
        else:
            ax.set_yticklabels(["Y_obs (gray)", "resid (Y_obs - v2)"])
            ax.set_title("Observed + residual (Y_obs - v2)")
        if row == 1:
            ax.set_xlabel("time (sec)")
        else:
            ax.set_xticks([])
        mappable = plt.cm.ScalarMappable(norm=residual_norm, cmap=cmap)
        mappable.set_array([])
        cbar = fig.colorbar(mappable, ax=ax, fraction=0.046, pad=0.02)
        cbar.set_label("residual value")

    # Right-top: per-cell MSE scatter (v1 vs v2).
    ax_scatter = axes[0, 2]
    if len(pred_blocks) >= 2:
        loss_v1 = np.mean((pred_blocks[0] - y_block) ** 2, axis=1)
        loss_v2 = np.mean((pred_blocks[1] - y_block) ** 2, axis=1)
        finite = np.isfinite(loss_v1) & np.isfinite(loss_v2)
        x_loss = loss_v1[finite]
        y_loss = loss_v2[finite]
        if x_loss.size > 0:
            ax_scatter.scatter(x_loss, y_loss, s=14, alpha=0.65, color="black")
            lo = float(min(np.min(x_loss), np.min(y_loss)))
            hi = float(max(np.max(x_loss), np.max(y_loss)))
            if np.isclose(lo, hi):
                pad = max(1e-6, abs(lo) * 0.05 + 1e-6)
                lo -= pad
                hi += pad
            ax_scatter.plot([lo, hi], [lo, hi], linestyle="--", color="gray", linewidth=1.5)
            ax_scatter.set_xlim(lo, hi)
            ax_scatter.set_ylim(lo, hi)
        ax_scatter.set_xlabel("v1 per-cell MSE")
        ax_scatter.set_ylabel("v2 per-cell MSE")
        ax_scatter.set_title("Per-cell loss comparison")
    else:
        ax_scatter.text(0.5, 0.5, "need two models", ha="center", va="center")
        ax_scatter.set_axis_off()

    # Right-bottom: population mean and residual means.
    ax_trace = axes[1, 2]
    if len(pred_blocks) >= 2:
        t = np.arange(y_block.shape[1])
        pop_mean = np.mean(y_block, axis=0)
        resid1_mean = np.mean(y_block - pred_blocks[0], axis=0)
        resid2_mean = np.mean(y_block - pred_blocks[1], axis=0)
        ax_trace.plot(t, pop_mean, color="black", linewidth=2, label="Population mean (Y_obs)")
        ax_trace.plot(t, resid1_mean, color="red", linewidth=2, alpha=0.9, label="Residual mean (Y_obs-v1)")
        ax_trace.plot(t, resid2_mean, color="blue", linewidth=2, alpha=0.9, label="Residual mean (Y_obs-v2)")
        ax_trace.axhline(0.0, color="gray", linestyle="--", linewidth=1.0, alpha=0.7)
        ax_trace.set_title("Population mean and residual means")
        ax_trace.set_xlabel("trial index")
        ax_trace.set_ylabel("z-scored activity")
        ax_trace.legend(fontsize=8)
    else:
        ax_trace.text(0.5, 0.5, "need two models", ha="center", va="center")
        ax_trace.set_axis_off()

    # Bottom row: three random single-cell traces (observed, v1, v2).
    if len(pred_blocks) >= 2 and y_block.shape[0] > 0:
        rng_cells = np.random.default_rng()
        n_available_cells = y_block.shape[0]
        n_show = min(3, n_available_cells)
        selected = rng_cells.choice(n_available_cells, size=n_show, replace=False)
        t = np.arange(y_block.shape[1])
        for col in range(3):
            ax = axes[2, col]
            if col >= n_show:
                ax.axis("off")
                continue
            cell_idx = int(selected[col])
            y_true = y_block[cell_idx]
            y_v1 = pred_blocks[0][cell_idx]
            y_v2 = pred_blocks[1][cell_idx]
            mse_v1 = float(np.mean((y_true - y_v1) ** 2))
            mse_v2 = float(np.mean((y_true - y_v2) ** 2))
            ax.plot(t, y_true, color="black", linewidth=1.8, label="Y_obs")
            ax.plot(t, y_v1, color="red", linewidth=1.6, alpha=0.9, label="v1")
            ax.plot(t, y_v2, color="blue", linewidth=1.6, alpha=0.9, label="v2")
            ax.axhline(0.0, color="gray", linestyle="--", linewidth=1.0, alpha=0.6)
            ax.set_title(f"Random cell {cell_idx} | MSE v1={mse_v1:.2f}, v2={mse_v2:.2f}")
            ax.set_xlabel("trial index")
            if col == 0:
                ax.set_ylabel("z-scored activity")
            if col == 2:
                ax.legend(fontsize=8, loc="upper right")
    else:
        for col in range(3):
            ax = axes[2, col]
            ax.text(0.5, 0.5, "need two models", ha="center", va="center")
            ax.set_axis_off()

    title_parts = []
    if model_losses:
        title_parts.extend(
            f"{_model_label(j)} loss={model_losses[j]:.2f}"
            for j in range(min(len(model_losses), len(programs_list)))
        )
    title_text = " | ".join(title_parts)
    if title_prefix:
        title_text = f"{title_prefix} | {title_text}" if title_text else str(title_prefix)
    if title_text:
        plt.suptitle(title_text, fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
