from __future__ import annotations

import importlib
import math
import os
import sys

import matplotlib
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.patches import Ellipse
from scipy.io import loadmat

from src import utils
from projects.minimodel_discovery.primitives import (
    gabor_bank16,
    local_gaussian_readout,
    make_gabor_kernels,
    pairwise_orientation_terms,
    quadrature_energy,
)
from projects.minimodel_discovery.data_loader.load_data import (
    _DATASET_CONTEXT,
    _DIAGNOSTIC_CACHE,
)
from projects.minimodel_discovery.seed_programs.param_est1 import (
    parameter_estimator as _param_est1,
    _normalized_to_pixel,
    _extract_patch,
    _compute_sta,
    _fit_gaussian_from_sta,
    _masked_sta_channel_weights,
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


def _teacher_imports(repo_path: str):
    if not repo_path or not os.path.isdir(repo_path):
        raise FileNotFoundError(f"minimodel repo path not found: {repo_path}")
    if repo_path not in sys.path:
        sys.path.insert(0, repo_path)
    torch = importlib.import_module("torch")
    model_builder = importlib.import_module("minimodel.model_builder")
    return torch, model_builder


def _get_teacher_state() -> dict[str, object]:
    if _DATASET_CONTEXT.get("teacher_state") is not None:
        return _DATASET_CONTEXT["teacher_state"]  # type: ignore[index]

    if not _DATASET_CONTEXT.get("use_teacher_diagnostics", False):
        teacher_state: dict[str, object] = {"available": False, "reason": "disabled in config"}
        _DATASET_CONTEXT["teacher_state"] = teacher_state
        return teacher_state

    repo_path = str(_DATASET_CONTEXT.get("minimodel_repo_path", ""))
    checkpoint_path = str(_DATASET_CONTEXT.get("teacher_checkpoint_path", ""))
    total_neurons = int(_DATASET_CONTEXT.get("teacher_total_neurons", 0))

    try:
        torch, model_builder = _teacher_imports(repo_path)
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"teacher checkpoint not found: {checkpoint_path}")

        preferred_device = str(_DATASET_CONTEXT.get("teacher_device", "cpu")).lower()
        if preferred_device.startswith("cuda") and torch.cuda.is_available():
            device = torch.device(preferred_device)
        else:
            device = torch.device("cpu")

        model, _ = model_builder.build_model(
            NN=total_neurons,
            n_layers=2,
            n_conv=16,
            n_conv_mid=320,
        )
        state = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state)
        model = model.to(device)
        model.eval()
        for param in model.parameters():
            param.requires_grad_(False)
        teacher_state = {
            "available": True,
            "torch": torch,
            "model": model,
            "device": device,
        }
    except Exception as exc:
        teacher_state = {"available": False, "reason": str(exc)}

    _DATASET_CONTEXT["teacher_state"] = teacher_state
    return teacher_state


def _make_gratings(height: int, width: int, n_thetas: int = 24) -> np.ndarray:
    thetas = np.linspace(0.0, 2.0 * np.pi, n_thetas, endpoint=False, dtype=np.float32)
    yy, xx = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    yy = yy - (height - 1) / 2.0
    xx = xx - (width - 1) / 2.0
    gratings = []
    for theta in thetas:
        carrier = (xx * np.cos(theta) + yy * np.sin(theta)) * (2.0 * np.pi / 24.0)
        grating = np.where(np.sin(carrier) >= 0.0, 1.0, -1.0).astype(np.float32)
        grating = grating - np.mean(grating)
        grating = grating / (np.std(grating) + 1e-6)
        gratings.append(grating)
    return np.stack(gratings, axis=0)


def _teacher_mei(teacher: dict[str, object], neuron_id: int, image_bank: np.ndarray) -> np.ndarray:
    torch = teacher["torch"]  # type: ignore[index]
    model = teacher["model"]  # type: ignore[index]
    device = teacher["device"]  # type: ignore[index]

    reference = torch.as_tensor(image_bank[: min(64, image_bank.shape[0])], dtype=torch.float32, device=device)
    reference = reference.unsqueeze(1)
    value_min = float(reference.min().item())
    value_max = float(reference.max().item())
    value_std = float(reference.std().item() + 1e-6)

    image = torch.randn((1, 1, reference.shape[-2], reference.shape[-1]), device=device) * value_std
    image = image.clamp_(value_min, value_max)
    image.requires_grad_(True)
    optimizer = torch.optim.Adam([image], lr=0.05)

    best_image = image.detach().clone()
    best_value = -float("inf")
    for _ in range(24):
        optimizer.zero_grad(set_to_none=True)
        pred = model(image)[0, neuron_id]
        tv = (image[:, :, 1:] - image[:, :, :-1]).abs().mean()
        tv = tv + (image[:, :, :, 1:] - image[:, :, :, :-1]).abs().mean()
        loss = -(pred - 1e-2 * tv)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            image.clamp_(value_min, value_max)
            value = float(pred.detach().item())
            if value > best_value:
                best_value = value
                best_image = image.detach().clone()

    return best_image[0, 0].detach().cpu().numpy().astype(np.float32)


def _teacher_local_gradient(teacher: dict[str, object], neuron_id: int, image_2d: np.ndarray) -> np.ndarray:
    torch = teacher["torch"]  # type: ignore[index]
    model = teacher["model"]  # type: ignore[index]
    device = teacher["device"]  # type: ignore[index]

    image = torch.as_tensor(image_2d[None, None], dtype=torch.float32, device=device)
    image.requires_grad_(True)
    pred = model(image)[0, neuron_id]
    pred.backward()
    grad = image.grad[0, 0].detach().cpu().numpy().astype(np.float32)
    return grad


def _teacher_grating_tuning(teacher: dict[str, object], neuron_id: int, shape: tuple[int, int]) -> np.ndarray:
    torch = teacher["torch"]  # type: ignore[index]
    model = teacher["model"]  # type: ignore[index]
    device = teacher["device"]  # type: ignore[index]

    gratings = _make_gratings(shape[0], shape[1], n_thetas=24)
    batch = torch.as_tensor(gratings[:, None], dtype=torch.float32, device=device)
    with torch.no_grad():
        responses = model(batch)[:, neuron_id].detach().cpu().numpy().astype(np.float32)
    return responses


def _render_teacher_panel(
    *,
    cell_id: int,
    teacher: dict[str, object],
    sample_image_bank: np.ndarray,
    top_image: np.ndarray,
) -> np.ndarray:
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.8))
    fig.suptitle(f"teacher diagnostics | cell {cell_id}", fontsize=12)

    if not teacher.get("available", False):
        for ax in axes.ravel():
            ax.axis("off")
        axes[0, 0].text(
            0.5,
            0.5,
            f"Teacher unavailable\n{teacher.get('reason', 'unknown')}",
            ha="center",
            va="center",
            fontsize=10,
            wrap=True,
        )
        fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.92])
        return _figure_to_rgb(fig)

    mei = _teacher_mei(teacher, cell_id, sample_image_bank)
    grad = _teacher_local_gradient(teacher, cell_id, top_image)
    tuning = _teacher_grating_tuning(teacher, cell_id, top_image.shape)
    model = teacher["model"]  # type: ignore[index]
    wc = np.abs(model.readout.Wc[cell_id].detach().cpu().numpy().reshape(-1))

    vmax_grad = max(float(np.percentile(np.abs(grad), 99)), 1e-4)
    axes[0, 0].imshow(mei, cmap="gray")
    axes[0, 0].set_title("MEI", fontsize=9)
    axes[0, 1].imshow(grad, cmap="RdBu_r", vmin=-vmax_grad, vmax=vmax_grad)
    axes[0, 1].set_title("local derivative", fontsize=9)
    axes[1, 0].plot(np.linspace(0.0, 2.0 * np.pi, tuning.size, endpoint=False), tuning, color="black", lw=2)
    axes[1, 0].set_title("square-grating tuning", fontsize=9)
    axes[1, 0].set_xticks([0.0, np.pi, 2.0 * np.pi])
    axes[1, 0].set_xticklabels(["0", "pi", "2pi"], fontsize=8)
    top_idx = np.argsort(wc)[-20:][::-1]
    axes[1, 1].bar(np.arange(top_idx.size), wc[top_idx], color="#4C78A8")
    axes[1, 1].set_title("|Wc| top 20", fontsize=9)

    for ax in axes.ravel():
        if ax is not axes[1, 0]:
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
    cache_key = (
        cell_ids,
        utils.data_n_trials(data),
        bool(_DATASET_CONTEXT.get("use_teacher_diagnostics", False)),
    )
    if cache_key in _DIAGNOSTIC_CACHE:
        return _DIAGNOSTIC_CACHE[cache_key]

    teacher = _get_teacher_state()
    fev_lookup = _DATASET_CONTEXT.get("fev_lookup", {})
    anchor_count = int(_DATASET_CONTEXT.get("anchor_cell_count", 8))
    anchor_indices = _get_anchor_indices(data, n_anchor=anchor_count)
    anchors = []

    for sample_idx in anchor_indices.tolist():
        sample = utils.get_data_sample(data, int(sample_idx))
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
        teacher_panel = _render_teacher_panel(
            cell_id=cell_id,
            teacher=teacher,
            sample_image_bank=np.transpose(np.asarray(sample["image"]), (2, 0, 1)),
            top_image=top_images[0],
        )
        anchors.append(
            {
                "sample_idx": int(sample_idx),
                "cell_id": cell_id,
                "fev": fev,
                "summary_panel": summary_panel,
                "teacher_panel": teacher_panel,
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
    programs_list,
    X_eval,
    save_path="",
    labels=None,
    title_prefix: str | None = None,
):
    if save_path == "":
        raise ValueError("plot_model_fits requires a non-empty save_path")

    cache = _build_anchor_cache(data)
    anchors = cache["anchors"][: min(4, len(cache["anchors"]))]
    if len(anchors) == 0:
        raise ValueError("No anchor neurons available for plotting.")

    n_models = len(programs_list)
    fig = plt.figure(figsize=(4.2 * (2 + n_models), 3.0 * len(anchors)))
    outer = fig.add_gridspec(len(anchors), 2 + n_models, wspace=0.25, hspace=0.35)

    colors = plt.get_cmap("tab10")(np.linspace(0.0, 1.0, max(n_models, 3)))
    model_losses_text = []

    for row_idx, anchor in enumerate(anchors):
        sample_idx = int(anchor["sample_idx"])
        cell_id = int(anchor["cell_id"])
        sample = utils.get_data_sample(data, sample_idx)
        eval_sample = utils.get_data_sample(X_eval, sample_idx)
        response = np.asarray(sample["response"], dtype=np.float32)
        repeats = np.asarray(sample["response_repeats"], dtype=np.float32)
        eval_images = np.transpose(np.asarray(eval_sample["image"], dtype=np.float32), (2, 0, 1))

        ax_summary = fig.add_subplot(outer[row_idx, 0])
        ax_summary.imshow(anchor["summary_panel"])
        ax_summary.set_xticks([])
        ax_summary.set_yticks([])
        ax_summary.set_title(f"static summary | cell {cell_id}", fontsize=10)

        ax_teacher = fig.add_subplot(outer[row_idx, 1])
        ax_teacher.imshow(anchor["teacher_panel"])
        ax_teacher.set_xticks([])
        ax_teacher.set_yticks([])
        ax_teacher.set_title("teacher diagnostics", fontsize=10)

        order = np.argsort(response)[::-1]
        sorted_response = response[order]

        for model_idx, program in enumerate(programs_list):
            label = labels[model_idx] if labels is not None and model_idx < len(labels) else f"model_{model_idx + 1}"
            params = utils.slice_params(utils.broadcast_params(program["params"], utils.data_n_samples(data)), sample_idx)
            pred = _vectorize_prediction(utils.call_model(program["model"], sample, params))
            pred_eval = _vectorize_prediction(utils.call_model(program["model"], eval_sample, params))
            pred_sorted = pred[order]

            loss_value = float("nan")
            if "losses" in program:
                try:
                    loss_value = float(np.asarray(program["losses"])[sample_idx])
                except Exception:
                    loss_value = float(np.mean(pred - response))
            fev_value, feve_value = _feve_for_sample(repeats, pred)

            if row_idx == 0:
                if np.isfinite(feve_value):
                    model_losses_text.append(f"{label}: loss={loss_value:.2f}, FEVE={feve_value:.2f}")
                else:
                    model_losses_text.append(f"{label}: loss={loss_value:.2f}, FEVE=n/a")

            sub = outer[row_idx, 2 + model_idx].subgridspec(2, 1, height_ratios=[1.0, 1.35], hspace=0.08)
            ax_top = fig.add_subplot(sub[0, 0])
            ax_trace = fig.add_subplot(sub[1, 0])

            strip = _top_eval_strip(eval_images, pred_eval, k=min(3, eval_images.shape[0]))
            vmax = float(np.nanpercentile(np.abs(strip), 99))
            vmax = max(vmax, 1e-4)
            ax_top.imshow(strip, cmap="gray", vmin=-vmax, vmax=vmax)
            ax_top.set_xticks([])
            ax_top.set_yticks([])
            ax_top.set_title(f"{label} | top predicted eval images", fontsize=9)

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

            text_lines = [f"loss={loss_value:.2f}"]
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
    title_parts.extend(model_losses_text[:n_models])
    if title_parts:
        fig.suptitle(" | ".join([part for part in title_parts if part]), fontsize=13)
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.03, top=0.92, wspace=0.25, hspace=0.4)
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
