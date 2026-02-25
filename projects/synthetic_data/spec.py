"""
Welcome to the Model Discovery Engine! Fill in the components below to start building your model.

NECESSARY COMPONENTS:

Loading:
- load_and_process_data(data_path, *preprocess_params) -> [X, Y]
- train_test_split(X) -> [train_samples, train_trials]

Seed Programs:
- model_v1(X, *params) and param_est_v1(X, Y)
- model_v2(X, *params) and param_est_v2(X, Y)

Loss:
- loss_fn(Y_pred, Y_true) -> loss values

OPTIONAL COMPONENTS (for enhanced diagnostics and visualization):

Plotting:
- plot_model_fits(X, Y, programs_list, X_eval, save_path, labels)
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple

from src.data_structures import Inputs, Outputs


# ========================
# 1. DATA
# ========================

def load_and_process_data(
    data_path: str = "",
    # ---- ALL SUBSEQUENT PARAMS MUST BE SPECIFIED IN THE CONFIG FILE ----
    SEED: int = 42,
    n_samples: int = 1000,
    n_trials: int = 2000,
    noise_std: float = 0.1,
) -> Tuple[Inputs, Outputs]:
    """
    Simulate synthetic single-input regression data and return canonical Inputs/Outputs.

    Data-generating process:
    y = (a*x^2 + b*x + c) * sin(k*x + phi_0) + epsilon,
    where epsilon ~ N(0, noise_std^2), and each sample has its own parameters.

    Parameters
    ----------
    data_path : str
        Unused for synthetic generation; included to keep a consistent API.
    SEED : int
        Random seed for reproducibility.
    n_samples : int
        Number of synthetic samples (different latent parameter sets).
    n_trials : int
        Number of trials/points per sample.
    noise_std : float
        Standard deviation of additive Gaussian noise.

    Returns
    -------
    X : Inputs
        Input tensor with shape (n_samples, 1, n_trials).
    Y : Outputs
        Output tensor with shape (n_samples, 1, n_trials).
    """
    rng = np.random.default_rng(SEED)

    a = rng.uniform(-1.0, 1.0, n_samples)
    b = rng.uniform(-1.0, 1.0, n_samples)
    c = rng.uniform(-1.0, 1.0, n_samples)
    k = rng.uniform(1.0, 5.0, n_samples)
    phi_0 = rng.uniform(0.0, 2 * np.pi, n_samples)

    x = rng.uniform(-1.0, 1.0, n_trials)
    noise = rng.normal(0.0, noise_std, (n_samples, n_trials))

    y = np.array([
        target_function(x, a[i], b[i], c[i], k[i], phi_0[i]) for i in range(n_samples)
    ]) + noise

    x_tiled = np.tile(x, (n_samples, 1))

    X = Inputs.from_array(x_tiled, names=["x"])
    Y = Outputs.from_array(y, names=["y"])
    return X, Y


def train_test_split(
    X: Inputs,
    # -- ALL SUBSEQUENT PARAMS MUST BE SPECIFIED IN THE CONFIG FILE ---
    random_seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Define train sample and train trial splits.

    Parameters
    ----------
    X : Inputs
        Inputs object of shape (n_samples, n_input_features, n_trials).
    random_seed : int
        Seed for reproducible random split.

    Returns
    -------
    train_samples : np.ndarray
        Sample indices of length n_samples // 2.
    train_trials : np.ndarray
        Trial indices of length n_trials // 2.
    """
    n_samples, _, n_trials = X.shape
    assert n_samples >= 2, "Need at least 2 samples for model optimization/eval"
    assert n_trials >= 2, "Need at least 2 trials for parameter optimization/eval"

    rng = np.random.default_rng(random_seed)
    train_samples = rng.choice(np.arange(n_samples), n_samples // 2, replace=False)
    train_trials = rng.choice(np.arange(n_trials), n_trials // 2, replace=False)
    return train_samples, train_trials


# ========================
# 2. SEED MODELS
# ========================
def model_v1(X, a=1.0, b=0.0):
    """
    Independent variable:
    X = [x]  # scalar input

    A RELU model:
    y = a * relu(x-b) = a * max(0, x-b)
    Args:
        X (np.ndarray): Input array with shape (1, n_trials)
        a (float): Scaling factor.
        b (float): Threshold for RELU.
    Returns:
        np.ndarray: Predicted output, shape (n_trials,).
    """
    x = X[0]
    return a * np.maximum(0, x - b)

def param_est_v1(X, Y):
    """
    Estimate parameters for model_v1 using a simple grid search.

    Args:
        X (np.ndarray): Input array with shape (1, n_trials) or (n_trials,).
        Y (np.ndarray): Target values with shape (n_trials,).

    Returns:
        np.ndarray: Estimated parameters [a, b].
    """
    y = np.asarray(Y)

    a_values = np.linspace(0.1, 5.0, 20)
    b_values = np.linspace(-1.0, 1.0, 20)

    best_loss = float("inf")
    best_params = (1.0, 0.0)

    for a in a_values:
        for b in b_values:
            y_pred = model_v1(X, a=a, b=b)
            loss = np.mean((y - y_pred) ** 2)
            if loss < best_loss:
                best_loss = loss
                best_params = (a, b)

    return np.array(best_params)

def model_v2(X, a=1.0, b=0.0):
    """
    Independent variable:
    X = [x]  # scalar input

    A simple linear model:
    y = a*x + b

    Args:
        X (np.ndarray): Input array with shape (1, n_trials) or (n_trials,).
        a (float): Linear slope.
        b (float): Linear intercept.

    Returns:
        np.ndarray: Predicted output, shape (n_trials,).
    """
    x = X[0]
    return a * x + b

def param_est_v2(X, Y):
    """
    Estimate parameters for model_v2 using least squares.

    Args:
        X (np.ndarray): Input array with shape (1, n_trials) or (n_trials,).
        Y (np.ndarray): Target values with shape (n_trials,).

    Returns:
        np.ndarray: Estimated parameters [a, b].
    """
    x = X[0]
    y = np.asarray(Y)

    A = np.vstack([x, np.ones(len(x))]).T
    a, b = np.linalg.lstsq(A, y, rcond=None)[0]
    return np.array([a, b])


# ========================
# 3. LOSS
# ========================

def loss_fn(Y_pred, Y_true):
    """
    Scaled squared error loss function.

    Args:
        Y_pred (jnp.ndarray): Predicted values, shape (1, n_trials)
        Y_true (jnp.ndarray): True values, shape (1, n_trials)

    Returns:
        jnp.ndarray: Scalar loss values for each trial, shape (1, n_trials).
    """
    return 10 * (Y_true - Y_pred) ** 2

# ========================
# 4. DIAGNOSTICS
# ========================

def plot_model_fits(
    X,
    Y,
    programs_list,
    X_eval,
    save_path="",
    labels=("model_v1", "model_v2"),
):
    """
    Plot observed synthetic data and model predictions for 9 random samples.

    Parameters
    ----------
    X : array-like or Inputs
        Input tensor with shape (n_samples, n_features, n_trials).
    Y : array-like or Outputs
        Output tensor with shape (n_samples, 1, n_trials).
    programs_list : list[dict]
        List of dictionaries with model metadata. Expected keys include:
        - 'model': callable model(X_one, *params)
        - 'params': array of shape (n_samples, n_params)
        - 'losses': array of shape (n_samples,)
    X_eval : array-like or Inputs
        Evaluation grid with shape (n_samples, n_features, n_eval_trials).
    save_path : str
        Output path. If empty, raises an error.
    """
    if save_path == "":
        raise ValueError("Please provide a save path for the plot")

    x_arr = _to_array3d(X)
    y_arr = _to_array3d(Y)
    x_eval_arr = _to_array3d(X_eval)
    n_samples = x_arr.shape[0]
    # diff colours depending on how many models we have
    if len(programs_list) == 1:
        colours = ["red"]
    elif len(programs_list) == 2:
        colours = ["green", "red"]
    else:
        colours = ["purple", "green", "red"]
    binned_colour = "deepskyblue"

    n_show = min(9, n_samples)
    # Intentionally unseeded so diagnostic samples vary across calls/runs.
    idx = np.random.default_rng().choice(n_samples, size=n_show, replace=False)

    fig, axes = plt.subplots(3, 3, figsize=(18, 18))
    axes = axes.reshape(3, 3)

    for i in range(9):
        ax = axes[i // 3, i % 3]
        if i >= n_show:
            ax.axis("off")
            continue

        s = idx[i]
        x = x_arr[s, 0]
        y_obs = y_arr[s, 0]
        x_eval = np.asarray(x_eval_arr[s, 0]).reshape(-1)

        y_mean = compute_binned_means_on_eval(x, y_obs, x_eval)
        ax.scatter(x, y_obs, s=10, c="black", alpha=0.15, label="Observed")
        ax.plot(x_eval, y_mean, color=binned_colour, linewidth=4, label="Binned mean", alpha=0.8)

        for j, program in enumerate(programs_list):
            model = program["model"]
            params = program["params"][s]
            y_pred = model(np.array([x_eval]), *params)

            label = labels[j] if labels is not None and j < len(labels) else f"Model {j+1}"
            if "losses" in program:
                label += f" (loss={program['losses'][s]:.2f})"
            ax.plot(
                x_eval,
                np.asarray(y_pred).flatten(),
                color=colours[j % len(colours)],
                linewidth=3,
                label=label,
                alpha=0.8,
            )

        ax.set_xlim((float(np.min(x_eval)), float(np.max(x_eval))))
        ax.set_title(f"Sample {s}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.legend(fontsize=12)

    mean_loss_parts = []
    for j, program in enumerate(programs_list):
        if "losses" in program and np.size(program["losses"]) > 0:
            mean_loss_parts.append(f"Model {j+1} Loss: {np.mean(program['losses']):.2f}")
        else:
            mean_loss_parts.append(f"Model {j+1} Loss: n/a")
    summary = "\n".join(mean_loss_parts) if mean_loss_parts else "n/a"
    plt.suptitle(f"Model Fits\n{summary}", fontsize=24)
    plt.savefig(save_path, dpi=100.0, bbox_inches="tight")
    plt.close(fig)

# ========================
# 5. OPTIONAL PROJECT-SPECIFIC HELPERS
# ========================

def target_function(x, a, b, c, k, phi_0):
    """
    Synthetic ground-truth function.

    f(x) = (a*x^2 + b*x + c) * sin(k*x + phi_0)
    """
    return (a * x**2 + b * x + c) * np.sin(k * x + phi_0)


def _to_array3d(obj) -> np.ndarray:
    """
    Convert Inputs/Outputs/ndarray-like objects to a 3D ndarray.
    """
    if hasattr(obj, "to_tensor"):
        return np.asarray(obj.to_tensor())
    arr = np.asarray(obj)
    if arr.ndim == 2:
        return arr[:, np.newaxis, :]
    return arr


def compute_binned_means(theta, y, n_bins=20, domain=(-1.0, 1.0)):
    """
    Compute binned means of y over theta for visualization.

    Returns
    -------
    x_eval : np.ndarray
        Bin centers.
    y_mean : np.ndarray
        Mean y per bin.
    """
    edges = np.linspace(domain[0], domain[1], n_bins + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])

    idx = np.digitize(theta, edges) - 1
    idx = np.clip(idx, 0, n_bins - 1)

    sums = np.bincount(idx, weights=y, minlength=n_bins)
    counts = np.bincount(idx, minlength=n_bins)
    mean = sums / (counts + 1e-8)
    return centres, mean


def compute_binned_means_on_eval(theta, y, x_eval):
    """
    Compute binned means of y at a provided evaluation grid.
    """
    x_eval = np.asarray(x_eval).reshape(-1)
    if x_eval.size == 0:
        return x_eval
    if x_eval.size == 1:
        return np.array([float(np.mean(y))])

    edges = np.empty(x_eval.size + 1, dtype=float)
    edges[1:-1] = 0.5 * (x_eval[:-1] + x_eval[1:])
    edges[0] = x_eval[0] - 0.5 * (x_eval[1] - x_eval[0])
    edges[-1] = x_eval[-1] + 0.5 * (x_eval[-1] - x_eval[-2])

    idx = np.digitize(theta, edges) - 1
    y_mean = np.full(x_eval.size, np.nan, dtype=float)
    for i in range(x_eval.size):
        vals = y[idx == i]
        if vals.size > 0:
            y_mean[i] = float(np.mean(vals))

    valid = np.isfinite(y_mean)
    if np.any(valid):
        y_mean = np.interp(
            x_eval,
            x_eval[valid],
            y_mean[valid],
            left=float(y_mean[valid][0]),
            right=float(y_mean[valid][-1]),
        )
    else:
        y_mean = np.zeros_like(x_eval, dtype=float)
    return y_mean
