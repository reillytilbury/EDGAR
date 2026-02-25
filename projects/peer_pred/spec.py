"""
Welcome to the Model Discovery Engine! Fill in the components below to start building your model.

NECESSARY COMPONENTS:

Loading:
- load_and_process_data(data_path, *preprocess_params) -> [X, Y]
- train_test_split(X) -> [train_samples, train_trials]

Seed Programs:
- model_v1(X, params) and param_est_v1(X, Y)
- model_v2(X, params) and param_est_v2(X, Y)

LOSS FUNCTION:
- loss_fn(Y_pred, Y_true) -> loss values

OPTIONAL COMPONENTS:
- plot_model_fits(X, Y, model_list, params_list)
"""
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple
from scipy.io import loadmat
from src.data_structures import Inputs, Outputs
from src import utils
from scipy.signal import medfilt
from skimage.restoration import denoise_tv_chambolle


# ========================
# 1. DATA
# ========================

def load_and_process_data(
    data_path: str = "/home/reilly/datasets/spontaneous_data/spont_M161025_MP030_20161120.mat",
    # ---- ALL SUBSEQUENT PARAMS MUST BE SPECIFIED IN THE CONFIG FILE ----
    random_seed: int = 42,
    n_cells: int = 4_000,
    downsample_factor: int = 4,
    var_thresh: float = 1e-4,
    zscore: bool = True,
) -> Tuple[Inputs, Outputs]:
    """
    Load and preprocess data and return canonical Inputs/Outputs.
    """
    random_seed = int(random_seed)
    n_cells = int(n_cells)
    downsample_factor = int(downsample_factor)
    var_thresh = float(var_thresh)

    rng = np.random.default_rng(random_seed)
    spks, _, _, _ = load_mat_spont(
        mat_path=data_path,
        var_thresh=var_thresh,
        downsample_factor=downsample_factor,
    )

    n_cells = min(n_cells, spks.shape[0])
    n_cells = n_cells - (n_cells % 4)
    if n_cells < 4:
        raise ValueError("Need at least 4 cells to form equal source/target and train/test splits.")
    spks = subsample_cells(spks, n_cells, rng)
    # clean up with median filter and TV denoising to make the models' lives easier
    for i in range(spks.shape[0]):
        spks[i] = preprocess_trace(spks[i], tv_weight=0.1, median_filter=True)
    if zscore:
        spks = zscore_rows(spks)

    # Source/target split for peer prediction, then split each into train/test cells.
    cell_idx = rng.permutation(spks.shape[0])
    half = cell_idx.shape[0] // 2
    source_cells = cell_idx[:half]
    target_cells = cell_idx[half:half * 2]

    # Split source and target populations into disjoint train/test cell sets.
    if source_cells.size < 2 or target_cells.size < 2:
        raise ValueError("Need at least 2 source and 2 target cells for train/test splits.")
    if source_cells.size % 2 != 0 or target_cells.size % 2 != 0:
        raise ValueError("Source and target cell counts must be even for equal splits.")

    half_source = source_cells.size // 2
    half_target = target_cells.size // 2
    train_source = source_cells[:half_source]
    test_source = source_cells[half_source:]
    train_targets = target_cells[:half_target]
    test_targets = target_cells[half_target:]

    X_train = spks[train_source]  # (n_source_train, T)
    X_test = spks[test_source]    # (n_source_test, T)
    Y_train = spks[train_targets] # (n_target_train, T)
    Y_test = spks[test_targets]   # (n_target_test, T)

    X = np.stack([X_train, X_test], axis=0)  # (2, n_source, T)
    Y = np.stack([Y_train, Y_test], axis=0)  # (2, n_target, T)
    return Inputs.from_array(X), Outputs.from_array(Y)


def train_test_split(
    X: Inputs,
    # -- ALL SUBSEQUENT PARAMS MUST BE SPECIFIED IN THE CONFIG FILE ---
    random_seed: int,
    block_size: int = 60,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return train sample indices and train trial indices.

    Sample 0 is the training population; sample 1 is held-out.
    """
    x_arr = np.asarray(X.to_tensor())
    _, _, T = x_arr.shape
    train_samples = np.array([0], dtype=np.int64)
    train_t, _ = make_time_split(T=T, block_size=block_size, mode="interleave")
    return np.asarray(train_samples, dtype=np.int64), np.asarray(train_t, dtype=np.int64)


# ========================
# 2. SEED MODELS
# ========================

def model_v1(X, params):
    """
    Linear peer-prediction model with a weight matrix.

    Equation: For each target cell c at timepoint t,
    Y[c, t] = sum_{s in source_cells} A[c, s] * X[s, t]

    Args:
        X (np.ndarray): Input array with shape (n_source_cells, n_time).
        params (dict): Parameter dictionary with keys:
            - A: Weight matrix of shape (n_target_cells, n_source_cells)

    Returns:
        np.ndarray: Predicted target activity, shape (n_target_cells, n_time).
    """
    weight_matrix_A = params["A"]
    return weight_matrix_A @ X


def param_est_v1(X, Y):
    """
    Fit a PLS weight matrix mapping source cells to target cells with rank 16.

    Args:
        X (np.ndarray): Input array with shape (n_source_cells, n_time).
        Y (np.ndarray): Target array with shape (n_target_cells, n_time).

    Returns:
        dict: Parameter dictionary with key {"A"}.
    """
    from sklearn.cross_decomposition import PLSRegression
    pls = PLSRegression(n_components=16, scale=False)
    pls.fit(X.T, Y.T)
    weight_matrix_A = pls.coef_.T  # (n_target_cells, n_source_cells)
    return {"A": weight_matrix_A}


def model_v2(X, params):
    """
    Linear peer-prediction with cell-specific quadratic terms.

    Equation: For each target cell c at timepoint t,
    Y[c, t] = q_c(sum_{s in source_cells} A[c, s] * X[s, t])
    where q_c(x) = a0_c + a1_c * x + a2_c * x^2 is a cell-specific quadratic function.

    Args:
        X (np.ndarray): Input array with shape (n_source_cells, n_time).
        params (dict): Parameter dictionary with keys:
            - A: Weight matrix of shape (n_target_cells, n_source_cells)
            - quadratic: Quadratic coefficients of shape (n_target_cells, 3)

    Returns:
        np.ndarray: Predicted target activity, shape (n_target_cells, n_time).
    """
    weight_matrix_A = params["A"]
    quadratic_coeffs = params["quadratic"]
    
    intercept = quadratic_coeffs[:, 0:1]
    linear_coef = quadratic_coeffs[:, 1:2]
    quadratic_coef = quadratic_coeffs[:, 2:3]
    
    Y_pred_linear = weight_matrix_A @ X  # (n_target, n_time)
    Y_pred = intercept + linear_coef * Y_pred_linear + quadratic_coef * (Y_pred_linear ** 2)
    return Y_pred


def param_est_v2(X, Y):
    """
    Fit parameters for model_v2 in two stages:
    1. Fit the linear weights A using PLS as in param_est_v1.
    2. Fit the quadratic coefficients for each target cell independently using least squares.

    Args:
        X (np.ndarray): Input array with shape (n_source_cells, n_time).
        Y (np.ndarray): Target array with shape (n_target_cells, n_time).

    Returns:
        dict: Parameter dictionary with keys {"A", "quadratic"}.
    """
    weight_matrix_A = param_est_v1(X, Y)["A"]
    n_target_cells = Y.shape[0]
    
    Y_pred_linear = weight_matrix_A @ X  # (n_target, n_time)

    quadratic_coeffs = np.zeros((n_target_cells, 3))
    for c in range(n_target_cells):
        X_c = np.stack(
            [np.ones_like(Y_pred_linear[c]), Y_pred_linear[c], Y_pred_linear[c] ** 2],
            axis=1,
        )  # (n_time, 3)
        coeffs, _, _, _ = np.linalg.lstsq(X_c, Y[c], rcond=None)
        quadratic_coeffs[c] = coeffs

    return {"A": weight_matrix_A, "quadratic": quadratic_coeffs}


# ========================
# 3. LOSS
# ========================

def loss_fn(Y_pred, Y_true):
    """
    Compute mean squared error between predicted and true target activity.

    Args:
        Y_pred (np.ndarray): Predicted target activity, shape (n_target_cells, n_time).
        Y_true (np.ndarray): True target activity, shape (n_target_cells, n_time).

    Returns:
        float: Mean squared error between predicted and true target activity.
    """
    return np.mean((Y_pred - Y_true) ** 2)


# ========================
# 4. DIAGNOSTICS
# ========================

def plot_model_fits(
    X,
    Y,
    programs_list,
    X_eval,
    save_path="",
    labels=None,
):
    """
    Plot observed activity and model predictions for random target cells.

    Shows 4 random target cells in a 2x2 grid over a random contiguous
    time block (default length 120).
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

    x_arr = _to_array3d(X)
    y_arr = _to_array3d(Y)
    n_samples, n_features, n_trials = x_arr.shape
    _, n_targets, _ = y_arr.shape

    sample_idx = 0
    x = x_arr[sample_idx]  # (n_features, n_trials)
    y = y_arr[sample_idx]  # (n_targets, n_trials)

    rng = np.random.default_rng()
    n_show = min(4, n_targets)
    cell_idx = rng.choice(n_targets, size=n_show, replace=False)

    block_len = min(120, n_trials)
    if n_trials == block_len:
        start = 0
    else:
        start = int(rng.integers(0, n_trials - block_len + 1))
    sl = slice(start, start + block_len)

    colours = ["red", "blue", "green", "purple", "orange"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.reshape(2, 2)

    # Precompute predictions and overall losses for the selected sample.
    preds_by_model = []
    model_losses = []
    for program in programs_list:
        model = program["model"]
        params = utils.slice_params(
            utils.broadcast_params(program["params"], n_samples), sample_idx
        )
        y_pred = utils.call_model(model, x, params)
        y_pred = np.asarray(y_pred)
        preds_by_model.append(y_pred)
        try:
            model_loss = float(loss_fn(y_pred, y))
        except Exception:
            model_loss = float(np.mean((y_pred - y) ** 2))
        model_losses.append(model_loss)

    def _model_label(j: int) -> str:
        if labels is not None and j < len(labels):
            return str(labels[j])
        return f"Model v{j+1}"

    for k in range(4):
        ax = axes[k // 2, k % 2]
        if k >= n_show:
            ax.axis("off")
            continue
        c = cell_idx[k]
        ax.plot(y[c, sl], color="black", linewidth=2, label="Observed")

        for j, y_pred in enumerate(preds_by_model):
            name = _model_label(j)
            cell_loss = float(np.mean((y_pred[c, sl] - y[c, sl]) ** 2))
            ax.plot(
                y_pred[c, sl],
                color=colours[j % len(colours)],
                linewidth=2,
                label=f"{name} (loss={cell_loss:.2f})",
                alpha=0.8,
            )

        ax.set_title(f"Target cell {int(c)}")
        ax.set_xlabel("time (s)")
        ax.set_ylabel("z-scored activity")
        ax.legend(fontsize=8)

    if model_losses:
        title_parts = [
            f"{_model_label(j)} loss={model_losses[j]:.2f}"
            for j in range(min(len(model_losses), len(programs_list)))
        ]
        plt.suptitle(" | ".join(title_parts), fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ========================
# 4. OPTIONAL PROJECT-SPECIFIC HELPERS
# ========================

def downsample_mean_1d(x: np.ndarray, factor: int) -> np.ndarray:
    """
    Downsample a 1D array by averaging in non-overlapping windows.
    """
    n = (len(x) // factor) * factor
    x = x[:n]
    return x.reshape(-1, factor).mean(axis=1)


def downsample_mean_2d(spks: np.ndarray, factor: int) -> np.ndarray:
    """
    Downsample a (cells x time) array by averaging across time in non-overlapping windows.
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


def load_mat_spont(mat_path: str, var_thresh: float = 1e-4, downsample_factor: int = 4) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load spikes and running speed from a .mat file with the expected structure.
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

def preprocess_trace(raw_trace, tv_weight=0.1, median_filter=True):
    # 1. Median Filter: Kills "salt and pepper" spikes (size 3 is subtle)
    if median_filter:
        trace = medfilt(raw_trace, kernel_size=3)
    else:
        trace = raw_trace
    
    # 2. TV Denoising: Flattens the "fuzz" while keeping the big jumps
    # weight: Higher = smoother/flatter. Start at 0.1 and tune.
    clean_trace = denoise_tv_chambolle(trace, weight=tv_weight)
    
    return clean_trace

# Example usage on one of your cells:
# cleaned = preprocess_trace(observed_cell_691, tv_weight=0.05)