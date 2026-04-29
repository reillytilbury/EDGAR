from __future__ import annotations

import numpy as np
import jax.numpy as jnp
from scipy.io import loadmat
from scipy.signal import butter, filtfilt
from typing import Tuple


def load_data(
    data_path: str = "",
    random_seed: int = 42,
    n_cells: int = 4000,
    downsample_factor: int = 4,
    var_thresh: float = 0.0001,
    zscore: bool = True,
    block_size: int = 180,
    time_split_mode: str = "interleave",
) -> Tuple[Tuple[dict, dict], Tuple[dict, dict], dict]:
    """
    Load spontaneous activity data and split into source/target populations.

    Returns
    -------
    X_discover, X_validate, X_eval
        X_discover = (train, test) dicts over time, using train-cell populations.
        X_validate = (train, test) dicts over time, using test-cell populations.
        X_eval = small fixed time subset from X_discover for fingerprinting.

    Data shape: {"source": (1, n_source, T), "target": (1, n_target, T)}
    The leading 1-dim sample axis lets scoring.py vmap over it once, so the
    model sees source/target as 2-D matrices (n_cells, T) as expected.
    """
    random_seed = int(random_seed)
    n_cells = int(n_cells)
    downsample_factor = int(downsample_factor)

    rng = np.random.default_rng(random_seed)
    spks, _, _, _ = _load_mat_spont(data_path, var_thresh=var_thresh,
                                     downsample_factor=downsample_factor)

    n_cells = min(n_cells, spks.shape[0])
    n_cells = n_cells - (n_cells % 4)
    if n_cells < 4:
        raise ValueError("Need at least 4 cells for source/target and train/test splits.")
    spks = _subsample_cells(spks, n_cells, rng)
    spks = _zscore_rows(spks)

    b, a = butter(N=3, Wn=0.15, btype="low")
    spks = filtfilt(b, a, spks, axis=1)

    cell_idx = rng.permutation(spks.shape[0])
    half = cell_idx.shape[0] // 2
    source_cells = cell_idx[:half]
    target_cells = cell_idx[half:half * 2]

    half_source = source_cells.size // 2
    half_target = target_cells.size // 2
    train_source = source_cells[:half_source]
    test_source = source_cells[half_source:]
    train_targets = target_cells[:half_target]
    test_targets = target_cells[half_target:]

    X_train = spks[train_source]
    X_test = spks[test_source]
    Y_train = spks[train_targets]
    Y_test = spks[test_targets]
    if zscore:
        X_train = _zscore_rows(X_train)
        X_test = _zscore_rows(X_test)
        Y_train = _zscore_rows(Y_train)
        Y_test = _zscore_rows(Y_test)

    T = X_train.shape[1]
    train_t, test_t = _make_time_split(T, block_size, time_split_mode)

    # Shape: (1, n_cells, T_*) — 1-dim sample axis for scoring.py vmap
    X_discover = (
        {"source": X_train[:, train_t][np.newaxis], "target": Y_train[:, train_t][np.newaxis]},
        {"source": X_train[:, test_t][np.newaxis], "target": Y_train[:, test_t][np.newaxis]},
    )
    X_validate = (
        {"source": X_test[:, train_t][np.newaxis], "target": Y_test[:, train_t][np.newaxis]},
        {"source": X_test[:, test_t][np.newaxis], "target": Y_test[:, test_t][np.newaxis]},
    )
    X_eval = {"source": X_train[:, train_t][np.newaxis], "target": Y_train[:, train_t][np.newaxis]}

    # Store which position each eval sample occupies for param matching in scoring (always 0 for this project)
    X_eval['_sample_indices'] = np.array([0])

    return X_discover, X_validate, X_eval


def loss_fn(model_output, data):
    return jnp.mean((data["target"] - model_output) ** 2, axis=-1)


# ── internal helpers ──

def _load_mat_spont(
    mat_path: str,
    var_thresh: float = 1e-4,
    downsample_factor: int = 4,
) -> tuple:
    mt = loadmat(mat_path)
    spks = mt["Fsp"]
    spks = spks[spks.var(axis=1) >= var_thresh]
    run_speed = mt["beh"]["runSpeed"][0, 0][:, 0]
    pupil_area = mt["beh"]["pupil"][0, 0][0, 0].item()[0][:, 0]
    pupil_com = mt["beh"]["pupil"][0, 0][0, 0].item()[1]

    spks = _downsample_mean_2d(spks, downsample_factor)
    run_speed = _downsample_mean_1d(run_speed, downsample_factor)
    pupil_area = _downsample_mean_1d(pupil_area, downsample_factor)
    pupil_com = _downsample_mean_2d(pupil_com.T, downsample_factor).T
    return spks, run_speed, pupil_area, pupil_com


def _downsample_mean_1d(x: np.ndarray, factor: int) -> np.ndarray:
    n = (len(x) // factor) * factor
    return x[:n].reshape(-1, factor).mean(axis=1)


def _downsample_mean_2d(spks: np.ndarray, factor: int) -> np.ndarray:
    n = (spks.shape[1] // factor) * factor
    return spks[:, :n].reshape(spks.shape[0], -1, factor).mean(axis=2)


def _zscore_rows(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    mu = X.mean(axis=1, keepdims=True)
    sd = X.std(axis=1, keepdims=True)
    return (X - mu) / (sd + eps)


def _subsample_cells(spks: np.ndarray, n_cells: int, rng: np.random.Generator) -> np.ndarray:
    return spks[rng.choice(spks.shape[0], size=n_cells, replace=False)]


def _make_time_split(T: int, block_size: int, mode: str) -> tuple:
    assert mode in {"interleave", "half"}
    if mode == "interleave":
        T = (T // block_size) * block_size
        n_blocks = T // block_size
        n_blocks = n_blocks - (n_blocks % 2)
        train_t = np.concatenate([b * block_size + np.arange(block_size) for b in range(0, n_blocks, 2)])
        test_t = np.concatenate([b * block_size + np.arange(block_size) for b in range(1, n_blocks, 2)])
    else:
        T = T - (T % 2)
        train_t = np.arange(0, T // 2)
        test_t = np.arange(T // 2, T)
    return train_t, test_t
