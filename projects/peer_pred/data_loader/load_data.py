import numpy as np
from scipy.io import loadmat
from scipy.signal import butter, filtfilt
from typing import Tuple

from src import utils

# Configuration constants from project config
RANDOM_SEED = 42
N_CELLS = 4000
DOWNSAMPLE_FACTOR = 4
VAR_THRESH = 0.0001
ZSCORE = True


def load_and_process_data(
    data_path: str = "",
    random_seed: int = RANDOM_SEED,
    n_cells: int = N_CELLS,
    downsample_factor: int = DOWNSAMPLE_FACTOR,
    var_thresh: float = VAR_THRESH,
    zscore: bool = ZSCORE,
) -> dict:
    """
    Load spontaneous activity data and split into source/target populations.

    Returns dict with:
        'source': shape (2, n_source, n_time) — sample 0 = train cells, 1 = test cells
        'target': shape (2, n_target, n_time) — sample 0 = train cells, 1 = test cells
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

    source = np.stack([X_train, X_test], axis=0)  # (2, n_source, T)
    target = np.stack([Y_train, Y_test], axis=0)  # (2, n_target, T)
    return {"source": source, "target": target}


def train_test_split(
    X: dict,
    random_seed: int,
    block_size: int = 180,
    mode: str = "interleave",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return train sample index (always 0) and train trial indices.
    """
    n_trials = utils.data_n_trials(X)
    assert utils.data_n_samples(X) == 2, "Expected exactly 2 samples."
    train_trials, _ = _make_time_split(n_trials, block_size, mode)
    return np.array([0]), train_trials


def loss_fn(model_output, data):
    return (data["target"] - model_output) ** 2


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
