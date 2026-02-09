import numpy as np
from scipy.io import loadmat

# CONFIG (caps)
SEED = 42
DOWNSAMPLE_FACTOR = 4      # average within this many original samples
N_CELLS = 4_000
BLOCK_SIZE = 60 * (4 // DOWNSAMPLE_FACTOR)           # in downsampled time bins
TRAIN_TEST_SPLIT = "interleave"  # {"interleave", "half"}
PLS_K = 16
EPS = 1e-12


def downsample_mean_1d(x: np.ndarray, factor: int) -> np.ndarray:
    """
    Downsample a 1D array by averaging in non-overlapping windows.

    The array is cropped to a multiple of `factor` before downsampling.
    """
    n = (len(x) // factor) * factor
    x = x[:n]
    return x.reshape(-1, factor).mean(axis=1)


def downsample_mean_2d(spks: np.ndarray, factor: int) -> np.ndarray:
    """
    Downsample a (cells x time) array by averaging across time in non-overlapping windows.

    The time axis is cropped to a multiple of `factor` before downsampling.
    """
    n = (spks.shape[1] // factor) * factor
    spks = spks[:, :n]
    return spks.reshape(spks.shape[0], -1, factor).mean(axis=2)


def zscore_rows(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Z-score each row of a 2D array independently.
    """
    mu = X.mean(axis=1, keepdims=True)
    sd = X.std(axis=1, keepdims=True)
    return (X - mu) / (sd + eps)


def subsample_cells(spks: np.ndarray, n_cells: int, rng: np.random.Generator) -> np.ndarray:
    """
    Subsample cells (rows) without replacement.
    """
    idx = rng.choice(spks.shape[0], size=n_cells, replace=False)
    return spks[idx]


def make_time_split(T: int, block_size: int, mode: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Split time indices into train/test by consecutive blocks.

    Modes:
      - "interleave": even-numbered blocks -> train, odd-numbered blocks -> test
      - "half": first half of time -> train, second half -> test
    """
    assert T >= 2 * block_size, "T must be at least 2 * block_size"
    assert mode in {"interleave", "half"}, "Invalid mode"

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


def split_cells(n_cells: int, rng: np.random.Generator, train_frac: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """
    Randomly split cell indices into two disjoint halves.
    """
    train_cells = rng.choice(n_cells, size=int(n_cells * train_frac), replace=False)
    test_cells = np.setdiff1d(np.arange(n_cells), train_cells)
    return train_cells, test_cells


def load_mat_spont(mat_path: str, var_thresh: float = 1e-4, downsample_factor: int = DOWNSAMPLE_FACTOR):
    """
    Load spikes and behavior from a .mat file with the expected structure.

    Returns:
        spks: (n_neurons, n_timepoints)
        run_speed: (n_timepoints,)
        pupil_area: (n_timepoints,)
        pupil_com: (n_timepoints, 2)
    """
    mt = loadmat(mat_path)
    spks = mt["Fsp"]  # neurons x timepoints
    cell_var = spks.var(axis=1)
    keep_cells = cell_var >= var_thresh
    spks = spks[keep_cells, :]

    run_speed = mt["beh"]["runSpeed"][0, 0][:, 0]
    pupil_area = mt["beh"]["pupil"][0, 0][0, 0].item()[0][:, 0]
    pupil_com = mt["beh"]["pupil"][0, 0][0, 0].item()[1]

    spks = downsample_mean_2d(spks, downsample_factor)
    run_speed = downsample_mean_1d(run_speed, downsample_factor)
    pupil_area = downsample_mean_1d(pupil_area, downsample_factor)
    pupil_com = downsample_mean_2d(pupil_com.T, downsample_factor).T

    return spks, run_speed, pupil_area, pupil_com


def make_XY(
    spks: np.ndarray,
    train_cells: np.ndarray,
    test_cells: np.ndarray,
    train_t: np.ndarray,
    test_t: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Construct train/test design matrices as (cells x time).

    Returns:
        X = [2 x n_source_cells x n_time]
        Y = [2 x n_target_cells x n_time]
    """
    X_train = spks[np.ix_(train_cells, train_t)]
    Y_train = spks[np.ix_(test_cells, train_t)]
    X_test = spks[np.ix_(train_cells, test_t)]
    Y_test = spks[np.ix_(test_cells, test_t)]
    return np.array([X_train, X_test]), np.array([Y_train, Y_test])


def load_data(
    data_path: str,
    seed: int = SEED,
    downsample_factor: int = DOWNSAMPLE_FACTOR,
    n_cells: int | None = N_CELLS,
    block_size: int = BLOCK_SIZE,
    train_test_split: str = TRAIN_TEST_SPLIT,
    train_frac: float = 0.5,
    var_thresh: float = 1e-4,
    eps: float = EPS,
    return_split: str = "train",
):
    rng = np.random.default_rng(seed)
    spks, run_speed, pupil_area, pupil_com = load_mat_spont(
        data_path,
        var_thresh=var_thresh,
        downsample_factor=downsample_factor,
    )

    if n_cells is None:
        n_cells = spks.shape[0]
    n_cells = int(min(n_cells, spks.shape[0]))
    spks = spks[:n_cells]

    T = spks.shape[1]
    train_t, test_t = make_time_split(T, block_size, train_test_split)
    train_cells, test_cells = split_cells(n_cells, rng, train_frac=train_frac)
    X, Y = make_XY(spks, train_cells, test_cells, train_t, test_t)

    for i in range(2):
        X[i] = zscore_rows(X[i], eps=eps)
        Y[i] = zscore_rows(Y[i], eps=eps)

    def _to_population_targets(y_mat: np.ndarray) -> np.ndarray:
        # y_mat: (n_target, n_time) -> (1, n_time, n_target)
        return y_mat.T[None, ...]

    if return_split == "train":
        return X[0], _to_population_targets(Y[0])
    if return_split == "test":
        return X[1], _to_population_targets(Y[1])
    if return_split == "all":
        X_all = np.concatenate([X[0], X[1]], axis=1)
        Y_all = np.concatenate([Y[0], Y[1]], axis=1)
        return X_all, _to_population_targets(Y_all)

    raise ValueError(f"Invalid return_split: {return_split}")
