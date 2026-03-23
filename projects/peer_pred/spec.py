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
- loss_fn(Y_pred, Y_true[, params]) -> loss values

OPTIONAL COMPONENTS:
- plot_model_fits(X, Y, model_list, params_list)
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from typing import Tuple
from scipy.io import loadmat
from src.data_structures import Inputs, Outputs
from src import utils
from scipy.signal import medfilt, butter, filtfilt
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
    X is the source population activity; Y is the target population activity.
    - Input has shape (2, n_source_cells, n_time) where sample 0 is a training 
    population and sample 1 is a held-out population for testing. 
    - Output has shape (2, n_target_cells, n_time) where sample 0 is the training 
    target population and sample 1 is the held-out target population for testing.
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
    spks = zscore_rows(spks)
    # add butterworth low-pass filter to smooth traces and make the task more learnable (optional)
    b, a = butter(N=3, Wn=0.15, btype="low")
    spks = filtfilt(b, a, spks, axis=1)

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
    if zscore:
        X_train = zscore_rows(X_train)
        X_test = zscore_rows(X_test)
        Y_train = zscore_rows(Y_train)
        Y_test = zscore_rows(Y_test)

    X = np.stack([X_train, X_test], axis=0)  # (2, n_source, T)
    Y = np.stack([Y_train, Y_test], axis=0)  # (2, n_target, T)
    return Inputs.from_array(X), Outputs.from_array(Y)


def train_test_split(
    X: Inputs,
    # -- ALL SUBSEQUENT PARAMS MUST BE SPECIFIED IN THE CONFIG FILE ---
    random_seed: int,
    block_size: int = 180,
    mode: str = "interleave",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return train sample indices and train trial indices.

    Sample 0 is the training population; sample 1 is held-out.
    """
    x_arr = np.asarray(X.to_tensor())
    n_samples, n_features, n_trials = x_arr.shape
    assert n_samples == 2, "Expected exactly 2 samples for train/test split."
    train_sample_idx = 0
    train_trials, _ = make_time_split(n_trials, block_size, mode)
    return np.array([train_sample_idx]), train_trials


# ========================
# 2. SEED MODELS
# ========================


def model_v1(X, params):
    """
    Lagged Linear Reduced Rank Regression.

    Equation: Y[t] = A @ [X[t]; X[t-1]; ... ; X[t-L]]

    Args:
        X (np.ndarray): Input array with shape (n_source_cells, n_time).
        params (dict): Parameter dictionary with key:
            - A: Weight matrix of shape (n_target_cells, n_source_cells * n_lags)

    Returns:
        np.ndarray: Predicted target activity, shape (n_target_cells, n_time).
    """
    weight_matrix_A = params["A"]
    
    n_source, n_time = X.shape
    n_features_total = weight_matrix_A.shape[1]
    
    # Infer n_lags from the shape of the weights to avoid integers in params
    n_lags = n_features_total // n_source
    
    # Construct the lagged stack: (n_source * n_lags, n_time)
    X_list = []
    for l in range(n_lags):
        # Shift data and pad with the first column to maintain shape (n_time)
        shifted = np.roll(X, l, axis=1)
        if l > 0:
            shifted[:, :l] = X[:, :1]
        X_list.append(shifted)
        
    X_stack = np.concatenate(X_list, axis=0) # Shape: (n_source * n_lags, n_time)
    
    return weight_matrix_A @ X_stack


def param_est_v1(X, Y):
    """
    Estimator for model_v1. Solves a regularized linear map A in closed form with a low-rank bottleneck:
    A = U[:, :rank] @ diag(S[:rank]) @ Vt[:rank, :], where U, S, Vt = SVD(Y X^T (X X^T + lambda I)^(-1))

    This approach allows us to keep the number of hyperparameters low and avoid putting integers in the params dict, 
    which can be tricky for some optimizers to handle. 

    Args:
        X (np.ndarray): Input array with shape (n_source_cells, n_time).
        Y (np.ndarray): Target array with shape (n_target_cells, n_time).
    Returns:
        dict: Parameter dictionary with key {"A"}.

    """
    def _coerce_xy(X_local, Y_local):
        X_local = np.asarray(X_local, dtype=np.float64)
        Y_local = np.asarray(Y_local, dtype=np.float64)
        if X_local.ndim == 1:
            X_local = X_local[None, :]
        elif X_local.ndim != 2:
            X_local = np.reshape(X_local, (X_local.shape[0], -1))
        if Y_local.ndim == 1:
            Y_local = Y_local[None, :]
        elif Y_local.ndim != 2:
            Y_local = np.reshape(Y_local, (Y_local.shape[0], -1))
        if Y_local.shape[1] != X_local.shape[1] and Y_local.shape[0] == X_local.shape[1]:
            Y_local = Y_local.T
        if Y_local.shape[1] != X_local.shape[1]:
            n_time_local = min(X_local.shape[1], Y_local.shape[1])
            X_local = X_local[:, :n_time_local]
            Y_local = Y_local[:, :n_time_local]
        X_local = np.nan_to_num(X_local, nan=0.0, posinf=0.0, neginf=0.0)
        Y_local = np.nan_to_num(Y_local, nan=0.0, posinf=0.0, neginf=0.0)
        return X_local, Y_local

    def _build_lag_stack(X_local, n_lags_local):
        cols = []
        n_lags_local = max(1, int(n_lags_local))
        for lag in range(n_lags_local):
            shifted = np.roll(X_local, lag, axis=1)
            if lag > 0:
                shifted[:, :lag] = X_local[:, :1]
            cols.append(shifted)
        return np.concatenate(cols, axis=0)

    def _stable_ridge_map(X_feat_local, Y_tgt_local, ridge_local):
        X_feat_local = np.asarray(X_feat_local, dtype=np.float64)
        Y_tgt_local = np.asarray(Y_tgt_local, dtype=np.float64)
        n_feat_local = X_feat_local.shape[0]
        xx_t_local = X_feat_local @ X_feat_local.T
        diag_scale_local = float(np.mean(np.diag(xx_t_local))) if n_feat_local > 0 else 1.0
        if not np.isfinite(diag_scale_local) or diag_scale_local <= 0.0:
            diag_scale_local = 1.0
        lam_local = float(max(ridge_local, 1e-8) * diag_scale_local)
        reg_local = lam_local * np.eye(n_feat_local, dtype=np.float64)
        rhs_local = Y_tgt_local @ X_feat_local.T
        try:
            a_t_local = np.linalg.solve((xx_t_local + reg_local).T, rhs_local.T)
            A_local = a_t_local.T
        except Exception:
            A_local = rhs_local @ np.linalg.pinv(xx_t_local + reg_local)
        return np.nan_to_num(A_local, nan=0.0, posinf=0.0, neginf=0.0)

    try:
        Xc, Yc = _coerce_xy(X, Y)
        n_source = Xc.shape[0]
        n_target = Yc.shape[0]
        n_lags = 2
        X_stack = _build_lag_stack(Xc, n_lags_local=n_lags)
        weight_matrix_A = _stable_ridge_map(X_stack, Yc, ridge_local=2e-2)
        if weight_matrix_A.shape != (n_target, n_source * n_lags):
            weight_matrix_A = np.resize(weight_matrix_A, (n_target, n_source * n_lags))
        weight_matrix_A = np.nan_to_num(weight_matrix_A, nan=0.0, posinf=0.0, neginf=0.0)
        return {"A": np.asarray(weight_matrix_A, dtype=np.float32)}
    except Exception:
        # Guaranteed finite fallback keeps seed estimator robust.
        Xc, Yc = _coerce_xy(X, Y)
        n_source = Xc.shape[0]
        n_target = Yc.shape[0]
        return {"A": np.zeros((n_target, n_source * 2), dtype=np.float32)}


def model_v2(X, params):
    """
    Linear peer-prediction model with a weight matrix. 
    
    No temporal lags, so it's a simpler model than a lagged regression, but this usually gives better held out performance (less overfitting).
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


def param_est_v2(X, Y):
    """
    Fast "quick-and-dirty" estimator for model_v2.
    Solves a regularized linear map A in closed form:
    A = Y X^T (X X^T + lambda I)^(-1)

    Args:
        X (np.ndarray): Input array with shape (n_source_cells, n_time).
        Y (np.ndarray): Target array with shape (n_target_cells, n_time).

    Returns:
        dict: Parameter dictionary with key {"A"}.
    """
    def _coerce_xy(X_local, Y_local):
        X_local = np.asarray(X_local, dtype=np.float64)
        Y_local = np.asarray(Y_local, dtype=np.float64)
        if X_local.ndim == 1:
            X_local = X_local[None, :]
        elif X_local.ndim != 2:
            X_local = np.reshape(X_local, (X_local.shape[0], -1))
        if Y_local.ndim == 1:
            Y_local = Y_local[None, :]
        elif Y_local.ndim != 2:
            Y_local = np.reshape(Y_local, (Y_local.shape[0], -1))
        if Y_local.shape[1] != X_local.shape[1] and Y_local.shape[0] == X_local.shape[1]:
            Y_local = Y_local.T
        if Y_local.shape[1] != X_local.shape[1]:
            n_time_local = min(X_local.shape[1], Y_local.shape[1])
            X_local = X_local[:, :n_time_local]
            Y_local = Y_local[:, :n_time_local]
        X_local = np.nan_to_num(X_local, nan=0.0, posinf=0.0, neginf=0.0)
        Y_local = np.nan_to_num(Y_local, nan=0.0, posinf=0.0, neginf=0.0)
        return X_local, Y_local

    def _stable_ridge_map(X_feat_local, Y_tgt_local, ridge_local):
        X_feat_local = np.asarray(X_feat_local, dtype=np.float64)
        Y_tgt_local = np.asarray(Y_tgt_local, dtype=np.float64)
        n_feat_local = X_feat_local.shape[0]
        xx_t_local = X_feat_local @ X_feat_local.T
        diag_scale_local = float(np.mean(np.diag(xx_t_local))) if n_feat_local > 0 else 1.0
        if not np.isfinite(diag_scale_local) or diag_scale_local <= 0.0:
            diag_scale_local = 1.0
        lam_local = float(max(ridge_local, 1e-8) * diag_scale_local)
        reg_local = lam_local * np.eye(n_feat_local, dtype=np.float64)
        rhs_local = Y_tgt_local @ X_feat_local.T
        try:
            a_t_local = np.linalg.solve((xx_t_local + reg_local).T, rhs_local.T)
            A_local = a_t_local.T
        except Exception:
            A_local = rhs_local @ np.linalg.pinv(xx_t_local + reg_local)
        return np.nan_to_num(A_local, nan=0.0, posinf=0.0, neginf=0.0)

    try:
        Xc, Yc = _coerce_xy(X, Y)
        n_source = Xc.shape[0]
        n_target = Yc.shape[0]
        weight_matrix_A = _stable_ridge_map(Xc, Yc, ridge_local=1e-2)
        if weight_matrix_A.shape != (n_target, n_source):
            weight_matrix_A = np.resize(weight_matrix_A, (n_target, n_source))
        weight_matrix_A = np.nan_to_num(weight_matrix_A, nan=0.0, posinf=0.0, neginf=0.0)
        return {"A": np.asarray(weight_matrix_A, dtype=np.float32)}
    except Exception:
        Xc, Yc = _coerce_xy(X, Y)
        n_source = Xc.shape[0]
        n_target = Yc.shape[0]
        return {"A": np.zeros((n_target, n_source), dtype=np.float32)}

# ========================
# 3. LOSS
# ========================

def loss_fn(Y_pred, Y_true, params=None):
    """
    Compute MSE plus optional parameter regularization.

    Args:
        Y_pred (np.ndarray): Predicted target activity, shape (n_target_cells, n_time).
        Y_true (np.ndarray): True target activity, shape (n_target_cells, n_time).
        params (dict | None): Optional model parameter dictionary. Supports both
            NumPy and JAX array leaves.

    Returns:
        float: Scalar loss value.
    """
    mse = np.mean((Y_pred - Y_true) ** 2)
    if params is None:
        return mse

    def _iter_leaves(obj):
        if isinstance(obj, dict):
            for v in obj.values():
                yield from _iter_leaves(v)
        elif isinstance(obj, (list, tuple)):
            for v in obj:
                yield from _iter_leaves(v)
        else:
            yield obj

    reg_weight = 1e4
    reg_sum = 0.0
    for leaf in _iter_leaves(params):
        try:
            reg_sum = reg_sum + np.mean(leaf * leaf)
        except Exception:
            # Ignore non-numeric leaves.
            continue
    return mse + reg_weight * reg_sum


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
    title_prefix: str | None = None,
):
    """
    Plot peer-prediction diagnostics in a 3x3 layout:
    - Left column: stacked observed+prediction rasters (v1 top, v2 bottom).
    - Middle column: stacked observed+residual rasters with residual colorbars.
    - Right column: per-cell loss scatter (top) and population/residual means (bottom).
    - Bottom row: three randomly chosen single-cell trace overlays (obs, v1, v2).
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
    n_samples, _, n_trials = x_arr.shape

    sample_idx = 0
    x = x_arr[sample_idx]  # (n_features, n_trials)
    y = y_arr[sample_idx]  # (n_targets, n_trials)

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
        y_pred = utils.call_model(model, x, params)
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
                model_losses.append(float(loss_fn(y_pred, y)))
            except Exception:
                model_losses.append(float(np.mean((y_pred - y) ** 2)))

    def _model_label(j: int) -> str:
        if labels is not None and j < len(labels):
            return str(labels[j])
        return f"Model v{j+1}"

    def _pc1_order(cell_by_trial: np.ndarray) -> np.ndarray:
        """Sort neurons by their score on PC1 of the observed activity block."""
        arr = np.asarray(cell_by_trial, dtype=float)
        if arr.ndim != 2:
            return np.arange(0, dtype=int)
        n_cells, n_t = arr.shape
        if n_cells <= 1:
            return np.arange(n_cells, dtype=int)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        try:
            # Center per-neuron time series and get first left singular vector scores.
            arr_centered = arr - arr.mean(axis=1, keepdims=True)
            u, s, _ = np.linalg.svd(arr_centered, full_matrices=False)
            if u.shape[0] == n_cells and s.size > 0:
                pc1_scores = u[:, 0] * s[0]
                return np.argsort(pc1_scores)
        except Exception:
            pass
        # Fallback keeps plotting robust if SVD fails.
        return np.argsort(np.nanargmax(arr, axis=1))

    def _positive_vmax(arr: np.ndarray, pct: float = 99.0) -> float:
        pos = np.clip(np.asarray(arr, dtype=float), 0.0, None)
        vmax = float(np.nanpercentile(pos, pct))
        if not np.isfinite(vmax) or vmax <= 1e-12:
            vmax = 1.0
        return vmax

    def _obs_gray_rgb(obs: np.ndarray) -> np.ndarray:
        """
        White=baseline (0), black=high positive activity.
        """
        arr = np.nan_to_num(np.asarray(obs, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
        vmax = _positive_vmax(arr)
        norm = np.clip(np.clip(arr, 0.0, None) / vmax, 0.0, 1.0)
        gray = 1.0 - norm
        return np.repeat(gray[:, :, None], 3, axis=2)

    def _pred_color_rgb(pred: np.ndarray, color: str) -> np.ndarray:
        """
        White=baseline (0), high positive activity maps to strong red/blue.
        """
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

    # Use same ordering for all raster panels, then display every 10th neuron.
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
            [ _obs_gray_rgb(y_sorted), _pred_color_rgb(pred_sorted, "red" if row == 0 else "blue") ],
            axis=0,
        )
        ax.imshow(panel, aspect="auto", interpolation="none")
        n_cells = y_sorted.shape[0]
        ax.axhline(n_cells - 0.5, color="white", linewidth=1.0, alpha=0.95)
        ax.set_yticks([n_cells / 2.0, n_cells + n_cells / 2.0])
        if row == 0:
            ax.set_yticklabels(["Y_obs (gray)", "pred v1 (red)"])
        else:
            ax.set_yticklabels(["Y_obs (gray)", "pred v2 (blue)"])
        ax.set_title(f"{_model_label(row)}: observed + prediction")
        if row == 1:
            ax.set_xlabel("time (sec)")
        else:
            ax.set_xticks([])

    # Middle column: observed stacked on residual, with residual colorbars.
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
        n_cells = y_sorted.shape[0]
        ax.axhline(n_cells - 0.5, color="white", linewidth=1.0, alpha=0.95)
        ax.set_yticks([n_cells / 2.0, n_cells + n_cells / 2.0])
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
