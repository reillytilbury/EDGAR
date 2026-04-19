from __future__ import annotations

import numpy as np
import jax.numpy as jnp
from scipy.io import loadmat

from src import utils


_DATASET_CONTEXT: dict[str, object] = {}
_DIAGNOSTIC_CACHE: dict[tuple, dict[str, object]] = {}


def _select_evenly_spaced_indices(n_total: int, n_keep: int) -> np.ndarray:
    if n_total <= 0 or n_keep <= 0:
        return np.zeros((0,), dtype=np.int64)
    if n_keep >= n_total:
        return np.arange(n_total, dtype=np.int64)
    return np.unique(np.linspace(0, n_total - 1, n_keep).round().astype(np.int64))


def _normalize_responses(
    train_response: np.ndarray,
    test_repeats: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    scale = np.std(train_response, axis=1, keepdims=True)
    scale = np.maximum(scale, 1e-2)
    train_norm = train_response / scale
    test_repeats_norm = test_repeats / scale[:, None, :]
    test_mean_norm = np.nanmean(test_repeats_norm, axis=1)
    return train_norm.astype(np.float32), test_mean_norm.astype(np.float32), test_repeats_norm.astype(np.float32)


def _compute_fev_from_repeats(repeats: np.ndarray) -> np.ndarray:
    fev = np.zeros((repeats.shape[0],), dtype=np.float32)
    for idx in range(repeats.shape[0]):
        cell_repeats = repeats[idx]
        total_var = float(np.nanvar(cell_repeats.reshape(-1), ddof=1))
        noise_var = float(np.nanmean(np.nanvar(cell_repeats, axis=0, ddof=1)))
        denom = max(total_var, 1e-6)
        fev[idx] = np.float32((total_var - noise_var) / denom)
    return fev


def load_and_process_data(
    data_path: str,
    image_path: str,
    mouse_id: int = 4,
    min_fev: float = 0.15,
    max_cells: int = 64,
    max_train_images: int = 1024,
    anchor_cell_count: int = 8,
    minimodel_repo_path: str = "",
    teacher_checkpoint_path: str = "",
    use_teacher_diagnostics: bool = True,
    teacher_device: str = "cpu",
):
    raw = np.load(data_path, allow_pickle=True)
    stimulus_mat = loadmat(image_path, squeeze_me=True)
    images = np.transpose(stimulus_mat["img"], (2, 0, 1)).astype(np.float32)

    if int(mouse_id) == 5:
        images = images[:, :, 46:176]
    else:
        images = images[:, :, :130]
    images = (images - np.mean(images)) / (np.std(images) + 1e-6)

    train_response_all = np.asarray(raw["sp"], dtype=np.float32)
    train_stim_ids_all = np.asarray(raw["istim_sp"], dtype=np.int64)
    test_stim_ids = np.asarray(raw["istim_ss"], dtype=np.int64)
    test_repeats_all = np.asarray(raw["ss_all"], dtype=np.float32).transpose(2, 1, 0)

    fev_all = _compute_fev_from_repeats(test_repeats_all)
    valid = np.where(fev_all >= float(min_fev))[0]
    if valid.size == 0:
        valid = np.arange(train_response_all.shape[0], dtype=np.int64)
    order = valid[np.argsort(fev_all[valid])[::-1]]
    if max_cells is not None:
        order = order[: int(max_cells)]

    if max_train_images is not None and int(max_train_images) < train_stim_ids_all.size:
        train_keep = _select_evenly_spaced_indices(train_stim_ids_all.size, int(max_train_images))
    else:
        train_keep = np.arange(train_stim_ids_all.size, dtype=np.int64)

    train_response = train_response_all[order][:, train_keep]
    test_repeats = test_repeats_all[order]
    train_response, test_response, test_repeats = _normalize_responses(train_response, test_repeats)

    train_images = images[train_stim_ids_all[train_keep]]
    test_images = images[test_stim_ids]
    all_images = np.concatenate([train_images, test_images], axis=0)

    all_response = np.concatenate([train_response, test_response], axis=1)
    n_cells = all_response.shape[0]
    n_train = train_response.shape[1]
    n_test = test_response.shape[1]
    n_trials = n_train + n_test

    response_repeats = np.full((n_cells, test_repeats.shape[1], n_trials), np.nan, dtype=np.float32)
    response_repeats[:, :, n_train:] = test_repeats

    image_tensor = np.transpose(all_images, (1, 2, 0))[None, ...]
    image_tensor = np.broadcast_to(image_tensor, (n_cells,) + image_tensor.shape[1:])
    stimulus_id = np.broadcast_to(
        np.concatenate([train_stim_ids_all[train_keep], test_stim_ids])[None, :],
        (n_cells, n_trials),
    ).astype(np.int32)
    cell_index = np.broadcast_to(order[:, None], (n_cells, n_trials)).astype(np.int32)

    _DATASET_CONTEXT.clear()
    _DATASET_CONTEXT.update(
        {
            "selected_cell_ids": order.astype(np.int64),
            "fev_lookup": {int(cell_id): float(fev_all[cell_id]) for cell_id in range(fev_all.shape[0])},
            "n_train_trials": int(n_train),
            "n_test_trials": int(n_test),
            "anchor_cell_count": int(anchor_cell_count),
            "minimodel_repo_path": minimodel_repo_path,
            "teacher_checkpoint_path": teacher_checkpoint_path,
            "use_teacher_diagnostics": bool(use_teacher_diagnostics),
            "teacher_device": teacher_device,
            "teacher_total_neurons": int(train_response_all.shape[0]),
            "teacher_state": None,
        }
    )
    _DIAGNOSTIC_CACHE.clear()

    return {
        "image": image_tensor,
        "response": all_response.astype(np.float32),
        "response_repeats": response_repeats,
        "stimulus_id": stimulus_id,
        "cell_index": cell_index,
    }


def train_test_split(X, random_seed: int):
    n_samples = utils.data_n_samples(X)
    rng = np.random.default_rng(random_seed)
    train_samples = np.sort(rng.choice(np.arange(n_samples), n_samples // 2, replace=False))
    train_trials = np.arange(int(_DATASET_CONTEXT["n_train_trials"]), dtype=np.int64)
    return train_samples, train_trials


def loss_fn(model_output, data):
    y_true = jnp.asarray(data["response"], dtype=jnp.float32)
    y_pred = jnp.clip(jnp.asarray(model_output, dtype=jnp.float32), 1e-4, 1e6)
    return y_pred - y_true * jnp.log(y_pred)


def build_evaluation_points(
    data,
    random_seed: int = 0,
    n_eval_images: int = 256,
    source: str = "train_anchor",
):
    if source != "train_anchor":
        raise ValueError(f"Unsupported evaluation point source: {source}")

    n_trials = utils.data_n_trials(data)
    if n_trials == 0:
        raise ValueError("Cannot build evaluation points from empty trial set.")
    n_keep = min(int(n_eval_images), int(n_trials))
    rng = np.random.default_rng(random_seed)
    trial_idx = np.sort(rng.choice(np.arange(n_trials), size=n_keep, replace=False))
    return utils.slice_data_trials(data, trial_idx)
