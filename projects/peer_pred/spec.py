"""
Welcome to the Model Discovery Engine! Fill in the components below to start building your model.

NECESSARY COMPONENTS:

Loading:
- load_and_process_data(data_path, *preprocess_params) -> dict[str, np.ndarray]
- train_test_split(X, random_seed) -> [train_samples, train_trials]

Seed Programs:
- model_v1(data, params) and param_est_v1(data)
- model_v2(data, params) and param_est_v2(data)

LOSS FUNCTION:
- loss_fn(Y_pred, Y_true) -> loss values

OPTIONAL COMPONENTS:
- plot_model_fits(data, programs_list, data_eval, save_path, labels)
"""
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple
from scipy.io import loadmat
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
    block_size: int = 180,
    mode: str = "interleave",
    zscore: bool = True,
) -> list[list[dict[str, np.ndarray]]]:
    """
    Load and preprocess data and return a dict of arrays.

    Returns a 2x2 split container:
    ``[[data_train_train, data_train_test], [data_test_train, data_test_test]]``.
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

    data = {
        "source": np.stack([X_train, X_test], axis=0),  # (2, n_source, T)
        "target": np.stack([Y_train, Y_test], axis=0),  # (2, n_target, T)
    }

    train_samples, train_trials = train_test_split(
        data,
        random_seed=random_seed,
        block_size=block_size,
        mode=mode,
    )
    _, test_trials = make_time_split(utils.data_n_trials(data), block_size=block_size, mode=mode)
    test_samples = np.setdiff1d(np.arange(utils.data_n_samples(data), dtype=np.int64), train_samples, assume_unique=False)

    data_train_train = utils.slice_data(data, train_samples, train_trials)
    data_train_test = utils.slice_data(data, train_samples, test_trials)
    data_test_train = utils.slice_data(data, test_samples, train_trials)
    data_test_test = utils.slice_data(data, test_samples, test_trials)

    data_train_train = utils.zscore_data(data_train_train)
    data_train_test = utils.zscore_data(data_train_test)
    data_test_train = utils.zscore_data(data_test_train)
    data_test_test = utils.zscore_data(data_test_test)

    return [[data_train_train, data_train_test], [data_test_train, data_test_test]]


def train_test_split(
    X: Dict[str, np.ndarray],
    # -- ALL SUBSEQUENT PARAMS MUST BE SPECIFIED IN THE CONFIG FILE ---
    random_seed: int,
    block_size: int = 180,
    mode: str = "interleave",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return train sample indices and train trial indices.

    Sample 0 is the training population; sample 1 is held-out.

    Parameters
    ----------
    X : dict[str, np.ndarray]
        Data dictionary. Each value has shape (n_samples, ..., n_trials).
    random_seed : int
        Seed used for reproducible splitting.
    block_size : int
        Size of contiguous time blocks for interleaved splitting.
    mode : str
        Splitting mode: 'interleave' or 'half'.

    Returns
    -------
    train_samples : np.ndarray
        Sample indices for training (just [0]).
    train_trials : np.ndarray
        Trial indices for training.
    """
    n_samples = utils.data_n_samples(X)
    n_trials = utils.data_n_trials(X)
    assert n_samples == 2, "Expected exactly 2 samples for train/test split."
    train_sample_idx = 0
    train_trials, _ = make_time_split(n_trials, block_size, mode)
    return np.array([train_sample_idx]), train_trials


# ========================
# 2. SEED MODELS
# ========================


def model_v1(X, params):
    """
    Linear peer-prediction model with a weight matrix. 
    
    No temporal lags, so it's a simpler model than a lagged regression, but this usually gives better held out performance (less overfitting).
    Equation: For each target cell c at timepoint t,
    Y[c, t] = sum_{s in source_cells} A[c, s] * data['source'][s, t]

    Args:
        data (dict): Single-sample data dictionary with keys:
            - 'source': source activity array of shape (n_source_cells, n_time).
        params (dict): Parameter dictionary with keys:
            - A: Weight matrix of shape (n_target_cells, n_source_cells)

    Returns:
        np.ndarray: Predicted target activity, shape (n_target_cells, n_time).
    """
    weight_matrix_A = params["A"]
    return weight_matrix_A @ data["source"]


def param_est_v1(X, Y):
    """
    Fast "quick-and-dirty" estimator for model_v2.
    Solves a regularized linear map A in closed form:
    A = Y X^T (X X^T + lambda I)^(-1)

    Args:
        data (dict): Single-sample data dictionary with keys:
            - 'source': source activity array of shape (n_source_cells, n_time).
            - 'target': target activity array of shape (n_target_cells, n_time).

    Returns:
        dict: Parameter dictionary with key {"A"}.
    """
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.linear_model import Ridge

    # HYPERPARAMS: Maybe add some cross-validated tuning?
    PLS_MAX_ITER = 500
    PLS_TOL = 1e-5
    N_COMPONENTS_PLS = 16
    ALPHA_RIDGE = 100.0

    # 1. PLS dimensionality reduction
    # X.T and Y.T are used because sklearn expects (n_samples, n_features)
    pls = PLSRegression(n_components=N_COMPONENTS_PLS, scale=False, max_iter=PLS_MAX_ITER, tol=PLS_TOL)
    Z = pls.fit_transform(X.T, Y.T)[0]  # Latent representation Z (n_time, k)

    # 2. Ridge regression from Latent Space -> Target
    # fit_intercept=False to ensure compatibility with the model equation Y = A @ X
    ridge = Ridge(alpha=ALPHA_RIDGE, fit_intercept=False)
    ridge.fit(Z, Y.T)

    # 3. Combine PLS rotations and Ridge weights into a single linear mapping matrix A
    # A (n_tgt, n_src) = Ridge_coef (n_tgt, k) @ PLS_rotations^T (k, n_src)
    weight_matrix_A = ridge.coef_ @ pls.x_rotations_.T

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
    1. Fit the linear weights A using PLS-Ridge (k=16, alpha=100).
    2. Fit the quadratic coefficients for each target cell independently using least squares.

    Args:
        X (np.ndarray): Input array with shape (n_source_cells, n_time).
        Y (np.ndarray): Target array with shape (n_target_cells, n_time).

    Returns:
        dict: Parameter dictionary with keys {"A", "quadratic"}.
    """
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.linear_model import Ridge

    # HYPERPARAMS: Maybe add some cross-validated tuning?
    PLS_MAX_ITER = 500
    PLS_TOL = 1e-5
    N_COMPONENTS_PLS = 16
    ALPHA_RIDGE = 100.0

    # Stage 1: Fit linear weights A using PLS-Ridge logic
    pls = PLSRegression(n_components=N_COMPONENTS_PLS, scale=False, max_iter=PLS_MAX_ITER, tol=PLS_TOL)
    Z = pls.fit_transform(X.T, Y.T)[0]
    ridge = Ridge(alpha=ALPHA_RIDGE, fit_intercept=False)
    ridge.fit(Z, Y.T)
    weight_matrix_A = ridge.coef_ @ pls.x_rotations_.T

    # Generate linear predictions to serve as input for Stage 2
    Y_pred_linear = weight_matrix_A @ X 
    n_target_cells = Y.shape[0]

    # Stage 2: Fit quadratic coefficients
    quadratic_coeffs = np.zeros((n_target_cells, 3))
    for c in range(n_target_cells):
        # Design matrix for quadratic: [1, x, x^2]
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
    Compute MSE plus optional parameter regularization.

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
    data,
    programs_list,
    data_eval,
    save_path="",
    labels=None,
    title_prefix: str | None = None,
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
    x = source_arr[sample_idx]  # (n_source, n_time)
    y = target_arr[sample_idx]  # (n_target, n_time)

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
        # Build single-sample data dict for model call
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
                model_losses.append(float(loss_fn(y_pred, sample_data_with_target)))
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
