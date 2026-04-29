from __future__ import annotations

import matplotlib
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg

from projects.minimodel_discovery.data_loader.load_data import (
    _DATASET_CONTEXT,
    _DIAGNOSTIC_CACHE,
)
from projects.minimodel_discovery.seed_programs.param_est1 import (
    parameter_estimator as _param_est1,
    _normalized_to_pixel,
    _extract_patch,
    _compute_sta,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _gaussian_mask_np(
    height: int,
    width: int,
    x0: float,
    y0: float,
    sigma_x: float,
    sigma_y: float,
) -> np.ndarray:
    y = np.linspace(-1.0, 1.0, height, dtype=np.float32)
    x = np.linspace(-1.0, 1.0, width, dtype=np.float32)
    yy, xx = np.meshgrid(y, x, indexing="ij")
    sigma_x = np.clip(float(sigma_x), 0.03, 1.5)
    sigma_y = np.clip(float(sigma_y), 0.03, 1.5)
    x0 = np.clip(float(x0), -1.0, 1.0)
    y0 = np.clip(float(y0), -1.0, 1.0)
    mask = np.exp(-0.5 * (((xx - x0) / sigma_x) ** 2 + ((yy - y0) / sigma_y) ** 2))
    mask /= np.sum(mask) + 1e-8
    return mask.astype(np.float32)


def _whitened_stc_components(
    image: np.ndarray,
    response: np.ndarray,
    x0: float,
    y0: float,
    patch_radius: int = 4,
) -> tuple[np.ndarray, np.ndarray, float]:
    height, width, n_trials = image.shape
    center_y = _normalized_to_pixel(y0, height)
    center_x = _normalized_to_pixel(x0, width)
    patch_size = 2 * patch_radius + 1

    patches = np.stack(
        [
            _extract_patch(image[..., trial_idx], center_y, center_x, patch_size=patch_size).reshape(-1)
            for trial_idx in range(n_trials)
        ],
        axis=0,
    ).astype(np.float32)

    weights = np.clip(np.asarray(response, dtype=np.float32), a_min=0.0, a_max=None)
    if float(np.sum(weights)) <= 0.0 or patches.shape[0] < 4:
        empty = np.zeros((height, width), dtype=np.float32)
        return empty, empty, 0.0

    weights = weights / (np.sum(weights) + 1e-6)
    mean_patch = np.mean(patches, axis=0, keepdims=True)
    patches_centered = patches - mean_patch
    cov = (patches_centered.T @ patches_centered) / max(patches_centered.shape[0] - 1, 1)
    eps = 1e-3 * float(np.trace(cov) / max(cov.shape[0], 1) + 1e-6)
    eigvals, eigvecs = np.linalg.eigh(cov + eps * np.eye(cov.shape[0], dtype=np.float32))
    inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(np.maximum(eigvals, eps))) @ eigvecs.T
    white = patches_centered @ inv_sqrt

    weighted_mean = np.sum(weights[:, None] * white, axis=0, keepdims=True)
    centered_white = white - weighted_mean
    stc = (centered_white.T * weights) @ centered_white
    delta = stc - np.eye(stc.shape[0], dtype=np.float32)
    delta_vals, delta_vecs = np.linalg.eigh(delta)

    pos_patch = delta_vecs[:, -1].reshape(patch_size, patch_size)
    neg_patch = delta_vecs[:, 0].reshape(patch_size, patch_size)
    strength = float(max(abs(delta_vals[-1]), abs(delta_vals[0])))

    pos_canvas = np.zeros((height, width), dtype=np.float32)
    neg_canvas = np.zeros((height, width), dtype=np.float32)

    y_start = max(center_y - patch_radius, 0)
    y_stop = min(center_y + patch_radius + 1, height)
    x_start = max(center_x - patch_radius, 0)
    x_stop = min(center_x + patch_radius + 1, width)

    patch_y_start = patch_radius - (center_y - y_start)
    patch_y_stop = patch_y_start + (y_stop - y_start)
    patch_x_start = patch_radius - (center_x - x_start)
    patch_x_stop = patch_x_start + (x_stop - x_start)

    pos_canvas[y_start:y_stop, x_start:x_stop] = pos_patch[patch_y_start:patch_y_stop, patch_x_start:patch_x_stop]
    neg_canvas[y_start:y_stop, x_start:x_stop] = neg_patch[patch_y_start:patch_y_stop, patch_x_start:patch_x_stop]
    return pos_canvas, neg_canvas, strength


def _feve_for_sample(repeats: np.ndarray, pred: np.ndarray) -> tuple[float, float]:
    pred = np.asarray(pred, dtype=np.float32).reshape(-1)
    repeats = np.asarray(repeats, dtype=np.float32)
    valid_trials = np.any(np.isfinite(repeats), axis=0)
    if int(np.sum(valid_trials)) < 2:
        return float("nan"), float("nan")

    rep = repeats[:, valid_trials]
    pred = pred[valid_trials]
    total_var = float(np.nanvar(rep.reshape(-1), ddof=1))
    noise_var = float(np.nanmean(np.nanvar(rep, axis=0, ddof=1)))
    mse = float(np.nanmean((rep - pred[None, :]) ** 2))

    total_var = max(total_var, 1e-6)
    explainable = max(total_var - noise_var, 1e-6)
    fev = (total_var - noise_var) / total_var
    feve = 1.0 - (mse - noise_var) / explainable
    return float(fev), float(feve)


def _figure_to_rgb(fig) -> np.ndarray:
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    rgba = np.asarray(canvas.buffer_rgba())
    rgb = np.asarray(rgba[..., :3], dtype=np.uint8)
    plt.close(fig)
    return rgb


def _render_summary_panel(
    *,
    cell_id: int,
    fev: float,
    top_images: np.ndarray,
    sta: np.ndarray,
    stc_pos: np.ndarray,
    stc_neg: np.ndarray,
    gaussian_mask: np.ndarray,
) -> np.ndarray:
    fig, axes = plt.subplots(2, 4, figsize=(8.6, 4.6))
    title = f"cell {cell_id} | FEV={fev:.2f}"
    fig.suptitle(title, fontsize=12)

    for idx in range(4):
        ax = axes[0, idx]
        if idx < top_images.shape[0]:
            ax.imshow(top_images[idx], cmap="gray")
            ax.set_title(f"top {idx + 1}", fontsize=9)
        else:
            ax.axis("off")
        ax.set_xticks([])
        ax.set_yticks([])

    vmax_sta = float(np.percentile(np.abs(sta), 99))
    vmax_stc = float(np.percentile(np.abs(np.stack([stc_pos, stc_neg], axis=0)), 99))
    vmax_sta = max(vmax_sta, 1e-4)
    vmax_stc = max(vmax_stc, 1e-4)

    axes[1, 0].imshow(sta, cmap="RdBu_r", vmin=-vmax_sta, vmax=vmax_sta)
    axes[1, 0].set_title("STA", fontsize=9)
    axes[1, 1].imshow(stc_pos, cmap="RdBu_r", vmin=-vmax_stc, vmax=vmax_stc)
    axes[1, 1].set_title("STC+", fontsize=9)
    axes[1, 2].imshow(stc_neg, cmap="RdBu_r", vmin=-vmax_stc, vmax=vmax_stc)
    axes[1, 2].set_title("STC-", fontsize=9)
    axes[1, 3].imshow(gaussian_mask, cmap="viridis")
    axes[1, 3].set_title("Gaussian readout", fontsize=9)

    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])

    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.92])
    return _figure_to_rgb(fig)


def _get_anchor_indices(data: dict[str, np.ndarray], n_anchor: int) -> np.ndarray:
    cell_ids = np.asarray(data["cell_index"][:, 0], dtype=np.int64)
    fev_lookup = _DATASET_CONTEXT.get("fev_lookup", {})
    scores = np.asarray([fev_lookup.get(int(cell_id), 0.0) for cell_id in cell_ids], dtype=np.float32)
    order = np.argsort(scores)[::-1]
    take = _select_evenly_spaced_indices(order.size, min(n_anchor, order.size))
    return order[take]


def _select_evenly_spaced_indices(n_total: int, n_keep: int) -> np.ndarray:
    if n_total <= 0 or n_keep <= 0:
        return np.zeros((0,), dtype=np.int64)
    if n_keep >= n_total:
        return np.arange(n_total, dtype=np.int64)
    return np.unique(np.linspace(0, n_total - 1, n_keep).round().astype(np.int64))


def _build_anchor_cache(data: dict[str, np.ndarray]) -> dict[str, object]:
    cell_ids = tuple(np.asarray(data["cell_index"][:, 0], dtype=np.int64).tolist())
    cache_key = (cell_ids, next(iter(data.values())).shape[-1])
    if cache_key in _DIAGNOSTIC_CACHE:
        return _DIAGNOSTIC_CACHE[cache_key]

    fev_lookup = _DATASET_CONTEXT.get("fev_lookup", {})
    anchor_count = int(_DATASET_CONTEXT.get("anchor_cell_count", 8))
    anchor_indices = _get_anchor_indices(data, n_anchor=anchor_count)
    anchors = []

    for sample_idx in anchor_indices.tolist():
        sample = {k: v[int(sample_idx)] for k, v in data.items()}
        cell_id = int(np.asarray(sample["cell_index"]).reshape(-1)[0])
        fev = float(fev_lookup.get(cell_id, float("nan")))
        params = _param_est1(sample)
        sta = _compute_sta(np.asarray(sample["image"]), np.asarray(sample["response"]))
        stc_pos, stc_neg, _ = _whitened_stc_components(
            np.asarray(sample["image"]),
            np.asarray(sample["response"]),
            params["x0"],
            params["y0"],
        )
        gaussian_mask = _gaussian_mask_np(
            sta.shape[0],
            sta.shape[1],
            params["x0"],
            params["y0"],
            params["sigma_x"],
            params["sigma_y"],
        )
        order = np.argsort(np.asarray(sample["response"]))[::-1]
        top_idx = order[: min(4, order.size)]
        top_images = np.transpose(np.asarray(sample["image"])[..., top_idx], (2, 0, 1))

        summary_panel = _render_summary_panel(
            cell_id=cell_id,
            fev=fev,
            top_images=top_images,
            sta=sta,
            stc_pos=stc_pos,
            stc_neg=stc_neg,
            gaussian_mask=gaussian_mask,
        )
        anchors.append(
            {
                "sample_idx": int(sample_idx),
                "cell_id": cell_id,
                "fev": fev,
                "summary_panel": summary_panel,
            }
        )

    cache = {"anchors": anchors}
    _DIAGNOSTIC_CACHE[cache_key] = cache
    return cache


def _vectorize_prediction(pred: np.ndarray) -> np.ndarray:
    arr = np.asarray(pred, dtype=np.float32)
    if arr.ndim == 2 and arr.shape[0] == 1:
        return arr[0]
    return arr.reshape(-1)


def _top_eval_strip(images: np.ndarray, scores: np.ndarray, k: int = 3) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    order = np.argsort(scores)[-k:][::-1]
    chosen = images[order]
    separator = np.full((chosen.shape[1], 2), np.nan, dtype=np.float32)
    strips = []
    for idx, image in enumerate(chosen):
        strips.append(image)
        if idx + 1 < chosen.shape[0]:
            strips.append(separator)
    return np.concatenate(strips, axis=1)


def plot_model_fits(
    data,
    parent_programs,
    save_path="",
    title_prefix: str | None = None,
):
    if save_path == "":
        raise ValueError("plot_model_fits requires a non-empty save_path")

    cache = _build_anchor_cache(data)
    anchors = cache["anchors"][: min(4, len(cache["anchors"]))]
    if len(anchors) == 0:
        raise ValueError("No anchor neurons available for plotting.")

    n_models = len(parent_programs)
    model_fns = [program.compile()[0] for program in parent_programs]

    fig = plt.figure(figsize=(4.2 * (1 + n_models), 3.0 * len(anchors)))
    outer = fig.add_gridspec(len(anchors), 1 + n_models, wspace=0.25, hspace=0.35)

    colors = plt.get_cmap("tab10")(np.linspace(0.0, 1.0, max(n_models, 3)))

    for row_idx, anchor in enumerate(anchors):
        sample_idx = int(anchor["sample_idx"])
        cell_id = int(anchor["cell_id"])
        sample = {k: v[sample_idx] for k, v in data.items()}
        response = np.asarray(sample["response"], dtype=np.float32)
        repeats = np.asarray(sample["response_repeats"], dtype=np.float32)
        images = np.transpose(np.asarray(sample["image"], dtype=np.float32), (2, 0, 1))

        ax_summary = fig.add_subplot(outer[row_idx, 0])
        ax_summary.imshow(anchor["summary_panel"])
        ax_summary.set_xticks([])
        ax_summary.set_yticks([])
        ax_summary.set_title(f"static summary | cell {cell_id}", fontsize=10)

        order = np.argsort(response)[::-1]
        sorted_response = response[order]

        for model_idx, (program, model_fn) in enumerate(zip(parent_programs, model_fns)):
            label = f"model_{model_idx + 1}"
            params_s = {k: np.asarray(v[sample_idx]) for k, v in program.params.items()}
            pred = _vectorize_prediction(np.asarray(model_fn(sample, params_s)))
            pred_sorted = pred[order]

            sample_loss = program.sample_losses[sample_idx] if program.sample_losses is not None else float("nan")
            fev_value, feve_value = _feve_for_sample(repeats, pred)

            sub = outer[row_idx, 1 + model_idx].subgridspec(2, 1, height_ratios=[1.0, 1.35], hspace=0.08)
            ax_top = fig.add_subplot(sub[0, 0])
            ax_trace = fig.add_subplot(sub[1, 0])

            strip = _top_eval_strip(images, pred, k=min(3, images.shape[0]))
            vmax = float(np.nanpercentile(np.abs(strip), 99))
            vmax = max(vmax, 1e-4)
            ax_top.imshow(strip, cmap="gray", vmin=-vmax, vmax=vmax)
            ax_top.set_xticks([])
            ax_top.set_yticks([])
            ax_top.set_title(f"{label} | top predicted images", fontsize=9)

            ax_trace.plot(sorted_response, color="black", lw=1.6, label="observed")
            ax_trace.plot(pred_sorted, color=colors[model_idx], lw=1.6, label="predicted")
            ax_trace.set_xticks([])
            ax_trace.spines["top"].set_visible(False)
            ax_trace.spines["right"].set_visible(False)
            if row_idx == len(anchors) - 1:
                ax_trace.set_xlabel("trials sorted by observed response", fontsize=9)
            ax_trace.set_ylabel("resp", fontsize=9)
            if model_idx == n_models - 1:
                ax_trace.legend(fontsize=7, frameon=False, loc="upper right")

            text_lines = [f"loss={sample_loss:.2f}" if np.isfinite(sample_loss) else "loss=n/a"]
            if np.isfinite(fev_value):
                text_lines.append(f"FEV={fev_value:.2f}")
            if np.isfinite(feve_value):
                text_lines.append(f"FEVE={feve_value:.2f}")
            ax_trace.text(
                0.02,
                0.98,
                "\n".join(text_lines),
                transform=ax_trace.transAxes,
                va="top",
                ha="left",
                fontsize=8,
                bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
            )

    title_parts = [title_prefix] if title_prefix else []
    for j, program in enumerate(parent_programs):
        loss = program.program_losses.discover.final
        title_parts.append(f"Model {j + 1}: loss={loss:.3f}" if loss is not None else f"Model {j + 1}: loss=n/a")
    if title_parts:
        fig.suptitle(" | ".join([p for p in title_parts if p]), fontsize=13)
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.03, top=0.92, wspace=0.25, hspace=0.4)
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
