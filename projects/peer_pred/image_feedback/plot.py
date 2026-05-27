import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def plot_model_fits(
    data,
    parent_programs,
    save_path="",
    title_prefix: str | None = None,
):
    """
    Plot observed activity and model predictions for a random contiguous time block.

    Args:
        data: X_disc_train dict with keys 'source', 'target' — shape (n_samples, n_features, n_trials).
        parent_programs: list of Program objects with .params and .compile_model() methods.
        save_path: file path to save the figure.
        title_prefix: optional string prepended to the suptitle.
    """
    if not save_path:
        raise ValueError("Please provide a save path for the plot")

    def _to_array3d(obj) -> np.ndarray:
        arr = np.asarray(obj)
        if arr.ndim == 2:
            return arr[:, np.newaxis, :]
        return arr

    x_arr = _to_array3d(data["source"])  # (n_samples, n_source, n_trials)
    y_arr = _to_array3d(data["target"])  # (n_samples, n_target, n_trials)
    n_samples, _, n_trials = x_arr.shape

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
        3, 3, figsize=(21, 14),
        gridspec_kw={"width_ratios": [1.25, 1.25, 1.0], "height_ratios": [1.0, 1.0, 0.8]},
    )

    model_fns   = [program.compile_model() for program in parent_programs]
    preds_by_model = []
    model_losses   = []
    for j, (program, model_fn) in enumerate(zip(parent_programs, model_fns)):
        params_s = {k: np.asarray(v[sample_idx]) for k, v in program.params.items()}
        y_pred = np.asarray(model_fn({"source": x}, params_s))
        if y_pred.ndim == 1:
            y_pred = y_pred[None, :]
        preds_by_model.append(y_pred)
        sample_loss = program.sample_losses[sample_idx] if program.sample_losses is not None else None
        model_losses.append(sample_loss)

    def _pc1_order(cell_by_trial: np.ndarray) -> np.ndarray:
        arr = np.nan_to_num(np.asarray(cell_by_trial, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
        if arr.ndim != 2 or arr.shape[0] <= 1:
            return np.arange(max(arr.shape[0], 0), dtype=int)
        try:
            arr_centered = arr - arr.mean(axis=1, keepdims=True)
            u, s, _ = np.linalg.svd(arr_centered, full_matrices=False)
            if u.shape[0] == arr.shape[0] and s.size > 0:
                return np.argsort(u[:, 0] * s[0])
        except Exception:
            pass
        return np.argsort(np.nanargmax(arr, axis=1))

    def _positive_vmax(arr: np.ndarray, pct: float = 99.0) -> float:
        vmax = float(np.nanpercentile(np.clip(np.asarray(arr, dtype=float), 0.0, None), pct))
        return vmax if np.isfinite(vmax) and vmax > 1e-12 else 1.0

    def _obs_gray_rgb(obs: np.ndarray) -> np.ndarray:
        arr = np.nan_to_num(np.asarray(obs, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
        norm = np.clip(np.clip(arr, 0.0, None) / _positive_vmax(arr), 0.0, 1.0)
        gray = 1.0 - norm
        return np.repeat(gray[:, :, None], 3, axis=2)

    def _pred_color_rgb(pred: np.ndarray, color: str) -> np.ndarray:
        arr = np.nan_to_num(np.asarray(pred, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
        norm = np.clip(np.clip(arr, 0.0, None) / _positive_vmax(arr), 0.0, 1.0)
        rgb = np.ones((arr.shape[0], arr.shape[1], 3), dtype=float)
        if color == "red":
            rgb[:, :, 1] = 1.0 - norm
            rgb[:, :, 2] = 1.0 - norm
        else:
            rgb[:, :, 0] = 1.0 - norm
            rgb[:, :, 1] = 1.0 - norm
        return rgb

    def _residual_rgb(residual: np.ndarray, cmap_name: str = "BrBG"):
        arr = np.nan_to_num(np.asarray(residual, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
        rv = float(np.nanpercentile(np.abs(arr), 99))
        rv = rv if np.isfinite(rv) and rv > 1e-12 else 1.0
        norm = mcolors.TwoSlopeNorm(vmin=-rv, vcenter=0.0, vmax=rv)
        cmap = plt.get_cmap(cmap_name)
        return cmap(norm(arr))[..., :3], norm, cmap

    y_block     = y[:, sl]
    order       = _pc1_order(y_block)
    order_plot  = order[::4] if order.size > 0 else order
    if order_plot.size == 0 and order.size > 0:
        order_plot = order[:1]
    y_sorted    = y_block[order_plot, :]
    pred_blocks = [np.asarray(pred[:, sl], dtype=float) for pred in preds_by_model]

    colours = ["red", "blue"]
    for row in range(2):
        ax = axes[row, 0]
        if row >= len(pred_blocks):
            ax.axis("off")
            continue
        pred_sorted = pred_blocks[row][order_plot, :]
        panel = np.concatenate(
            [_obs_gray_rgb(y_sorted), _pred_color_rgb(pred_sorted, colours[row])], axis=0
        )
        ax.imshow(panel, aspect="auto", interpolation="none")
        n_cells_plot = y_sorted.shape[0]
        ax.axhline(n_cells_plot - 0.5, color="white", linewidth=1.0, alpha=0.95)
        ax.set_yticks([n_cells_plot / 2.0, n_cells_plot + n_cells_plot / 2.0])
        ax.set_yticklabels([f"Y_obs (gray)", f"pred {parent_programs[row].name} ({colours[row]})"])
        ax.set_title(f"{parent_programs[row].name}: observed + prediction")
        if row == 1:
            ax.set_xlabel("time (sec)")
        else:
            ax.set_xticks([])

    for row in range(2):
        ax = axes[row, 1]
        if row >= len(pred_blocks):
            ax.axis("off")
            continue
        residual = y_sorted - pred_blocks[row][order_plot, :]
        residual_rgb, residual_norm, cmap = _residual_rgb(residual)
        panel = np.concatenate([_obs_gray_rgb(y_sorted), residual_rgb], axis=0)
        ax.imshow(panel, aspect="auto", interpolation="none")
        n_cells_plot = y_sorted.shape[0]
        ax.axhline(n_cells_plot - 0.5, color="white", linewidth=1.0, alpha=0.95)
        ax.set_yticks([n_cells_plot / 2.0, n_cells_plot + n_cells_plot / 2.0])
        ax.set_yticklabels(["Y_obs (gray)", f"resid (Y_obs - {parent_programs[row].name})"])
        ax.set_title(f"Observed + residual (Y_obs - {parent_programs[row].name})")
        if row == 1:
            ax.set_xlabel("time (sec)")
        else:
            ax.set_xticks([])
        mappable = plt.cm.ScalarMappable(norm=residual_norm, cmap=cmap)
        mappable.set_array([])
        fig.colorbar(mappable, ax=ax, fraction=0.046, pad=0.02).set_label("residual value")

    ax_scatter = axes[0, 2]
    if len(pred_blocks) >= 2:
        loss_v1 = np.mean((pred_blocks[0] - y_block) ** 2, axis=1)
        loss_v2 = np.mean((pred_blocks[1] - y_block) ** 2, axis=1)
        finite  = np.isfinite(loss_v1) & np.isfinite(loss_v2)
        if finite.any():
            x_loss, y_loss = loss_v1[finite], loss_v2[finite]
            ax_scatter.scatter(x_loss, y_loss, s=14, alpha=0.65, color="black")
            lo = float(min(x_loss.min(), y_loss.min()))
            hi = float(max(x_loss.max(), y_loss.max()))
            if np.isclose(lo, hi):
                pad = max(1e-6, abs(lo) * 0.05 + 1e-6)
                lo -= pad; hi += pad
            ax_scatter.plot([lo, hi], [lo, hi], linestyle="--", color="gray", linewidth=1.5)
            ax_scatter.set_xlim(lo, hi); ax_scatter.set_ylim(lo, hi)
        ax_scatter.set_xlabel(f"{parent_programs[0].name} per-cell MSE")
        ax_scatter.set_ylabel(f"{parent_programs[1].name} per-cell MSE")
        ax_scatter.set_title("Per-cell loss comparison")
    else:
        ax_scatter.text(0.5, 0.5, "need two models", ha="center", va="center")
        ax_scatter.set_axis_off()

    ax_trace = axes[1, 2]
    if len(pred_blocks) >= 2:
        t = np.arange(y_block.shape[1])
        ax_trace.plot(t, np.mean(y_block, axis=0), color="black", linewidth=2, label="Population mean (Y_obs)")
        ax_trace.plot(t, np.mean(y_block - pred_blocks[0], axis=0), color="red", linewidth=2, alpha=0.9, label=f"Residual mean (Y_obs - {parent_programs[0].name})")
        ax_trace.plot(t, np.mean(y_block - pred_blocks[1], axis=0), color="blue", linewidth=2, alpha=0.9, label=f"Residual mean (Y_obs - {parent_programs[1].name})")
        ax_trace.axhline(0.0, color="gray", linestyle="--", linewidth=1.0, alpha=0.7)
        ax_trace.set_title("Population mean and residual means")
        ax_trace.set_xlabel("trial index")
        ax_trace.set_ylabel("z-scored activity")
        ax_trace.legend(fontsize=8)
    else:
        ax_trace.text(0.5, 0.5, "need two models", ha="center", va="center")
        ax_trace.set_axis_off()

    if len(pred_blocks) >= 2 and y_block.shape[0] > 0:
        n_show   = min(3, y_block.shape[0])
        selected = np.random.choice(y_block.shape[0], size=n_show, replace=False)
        t = np.arange(y_block.shape[1])
        for col in range(3):
            ax = axes[2, col]
            if col >= n_show:
                ax.axis("off")
                continue
            cell_idx = int(selected[col])
            y_true = y_block[cell_idx]
            ax.plot(t, y_true, color="black", linewidth=1.8, label="Y_obs")
            for j, (pred_block, colour) in enumerate(zip(pred_blocks, colours)):
                mse = float(np.mean((y_true - pred_block[cell_idx]) ** 2))
                ax.plot(t, pred_block[cell_idx], color=colour, linewidth=1.6, alpha=0.9,
                        label=f"{parent_programs[j].name} (mse={mse:.2f})")
            ax.axhline(0.0, color="gray", linestyle="--", linewidth=1.0, alpha=0.6)
            ax.set_title(f"Random cell {cell_idx}")
            ax.set_xlabel("trial index")
            if col == 0:
                ax.set_ylabel("z-scored activity")
            ax.legend(fontsize=8, loc="upper right")
    else:
        for col in range(3):
            axes[2, col].text(0.5, 0.5, "need two models", ha="center", va="center")
            axes[2, col].set_axis_off()

    title_parts = [title_prefix] if title_prefix else []
    for j, (program, loss) in enumerate(zip(parent_programs, model_losses)):
        loss_str = f"{loss:.3f}" if loss is not None else "n/a"
        title_parts.append(f"{program.name}: loss={loss_str}")
    if title_parts:
        plt.suptitle(" | ".join(title_parts), fontsize=14)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
