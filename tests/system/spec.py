"""
Welcome to the Model Discovery Engine! Fill in the components below to start building your model.

NECESSARY COMPONENTS:

Loading:
- load_and_process_data(data_path, *preprocess_params) -> dict[str, np.ndarray]
- train_test_split(X) -> [train_samples, train_trials]

Seed Programs:
- model_v1(data, params) and param_est_v1(data)
- model_v2(data, params) and param_est_v2(data)

Loss:
- loss_fn(model_output, data) -> loss values

OPTIONAL COMPONENTS (for enhanced diagnostics and visualization):

Plotting:
- plot_model_fits(data, programs_list, eval_grid, save_path, labels)
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple
import warnings

from src import utils


# ========================
# 1. DATA
# ========================


def load_and_process_data(
    data_path: str = "",
    # ---- ALL SUBSEQUENT PARAMS MUST BE SPECIFIED IN THE CONFIG FILE ----
    SEED: int = 42,
    random_seed: int = 42,
    n_samples: int = 1000,
    n_trials: int = 2000,
    noise_std: float = 0.1,
) -> list[list[dict[str, np.ndarray]]]:
    """
    Simulate synthetic single-input regression data.

    Data-generating process:
    y = (a*x^2 + b*x + c) * sin(k*x + phi_0) + epsilon,
    where epsilon ~ N(0, noise_std^2), and each sample has its own parameters.
    For test purposes, use zero noise and generate a,b,c,k,phi_0 deterministically based on n_samples

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
    2 x 2 list of dicts
        ``[[data_train_train, data_train_test], [data_test_train, data_test_test]]``
        with the same split semantics as the legacy runner-managed split.
    """
    a = np.linspace(-1, 1, n_samples)
    b = np.linspace(-1, 1, n_samples)
    c = np.linspace(-1, 1, n_samples)
    k = np.full(n_samples, 4)
    phi_0 = np.zeros((n_samples,))

    x = np.linspace(-1.0, 1.0, n_trials)
    warnings.warn(
        "Using zero noise for synthetic data generation in load_and_process_data for testing"
    )

    y = np.array(
        [target_function(x, a[i], b[i], c[i], k[i], phi_0[i]) for i in range(n_samples)]
    )

    x_tiled = np.tile(x, (n_samples, 1))

    data = {"x": x_tiled, "y": y}

    train_samples, train_trials = train_test_split(data, random_seed=random_seed)
    test_samples = np.setdiff1d(
        np.arange(n_samples, dtype=np.int64), train_samples, assume_unique=False
    )
    test_trials = np.setdiff1d(
        np.arange(n_trials, dtype=np.int64), train_trials, assume_unique=False
    )

    data_train_train = utils.slice_data(data, train_samples, train_trials)
    data_train_test = utils.slice_data(data, train_samples, test_trials)
    data_test_train = utils.slice_data(data, test_samples, train_trials)
    data_test_test = utils.slice_data(data, test_samples, test_trials)

    return [[data_train_train, data_train_test], [data_test_train, data_test_test]]


def train_test_split(
    X: Dict[str, np.ndarray],
    # -- ALL SUBSEQUENT PARAMS MUST BE SPECIFIED IN THE CONFIG FILE ---
    random_seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Define train sample and train trial splits.

    Parameters
    ----------
    X : dict[str, np.ndarray]
        Data dict where all arrays share the same last dimension (n_trials).
    random_seed : int
        Seed for reproducible random split.

    Returns
    -------
    train_samples : np.ndarray
        Sample indices of length n_samples // 2.
    train_trials : np.ndarray
        Trial indices of length n_trials // 2.
    """
    n_samples = utils.data_n_samples(X)
    n_trials = utils.data_n_trials(X)
    assert n_samples >= 2, "Need at least 2 samples for model optimization/eval"
    assert n_trials >= 2, "Need at least 2 trials for parameter optimization/eval"

    train_samples = np.arange(n_samples // 2)  # First half of samples
    train_trials = np.arange(n_trials)[::2]  # Every other trial
    return train_samples, train_trials


# ========================
# 2. SEED MODELS
# ========================
def model_v1(data, params):
    """
    Data keys used:
    data['x']  # scalar input, shape (n_trials,)

    A RELU model:
    y = a * relu(x-b) = a * max(0, x-b)
    Args:
        data (dict): Data dict for one sample with key 'x', shape (n_trials,).
        params (dict): Parameter dictionary with keys:
            - a: Scaling factor.
            - b: Threshold for RELU.
    Returns:
        np.ndarray: Predicted output, shape (n_trials,).
    """
    x = data["x"]
    a = params["a"]
    b = params["b"]
    return a * np.maximum(0, x - b)


model_v1.DEFAULT_PARAMS = {"a": 1.0, "b": 0.0}


def param_est_v1(data):
    """
    Estimate parameters for model_v1 using a simple grid search.

    Args:
        data (dict): Data dict for one sample with keys 'x' and 'y'.

    Returns:
        dict: Estimated parameters with keys {"a", "b"}.
    """
    y = np.asarray(data["y"])

    a_values = np.linspace(0.1, 5.0, 20)
    b_values = np.linspace(-1.0, 1.0, 20)

    best_loss = float("inf")
    best_params = (1.0, 0.0)

    for a in a_values:
        for b in b_values:
            y_pred = model_v1(data, {"a": a, "b": b})
            loss = np.mean((y - y_pred) ** 2)
            if loss < best_loss:
                best_loss = loss
                best_params = (a, b)

    return {"a": float(best_params[0]), "b": float(best_params[1])}


def model_v2(data, params):
    """
    Data keys used:
    data['x']  # scalar input, shape (n_trials,)

    A simple linear model:
    y = a*x + b

    Args:
        data (dict): Data dict for one sample with key 'x', shape (n_trials,).
        params (dict): Parameter dictionary with keys:
            - a: Linear slope.
            - b: Linear intercept.

    Returns:
        np.ndarray: Predicted output, shape (n_trials,).
    """
    x = data["x"]
    a = params["a"]
    b = params["b"]
    return a * x + b


model_v2.DEFAULT_PARAMS = {"a": 1.0, "b": 0.0}


def param_est_v2(data):
    """
    Estimate parameters for model_v2 using least squares.

    Args:
        data (dict): Data dict for one sample with keys 'x' and 'y'.

    Returns:
        dict: Estimated parameters with keys {"a", "b"}.
    """
    x = data["x"]
    y = np.asarray(data["y"])

    A = np.vstack([x, np.ones(len(x))]).T
    a, b = np.linalg.lstsq(A, y, rcond=None)[0]
    return {"a": float(a), "b": float(b)}


# ========================
# 3. LOSS
# ========================


def loss_fn(model_output, data):
    """
    Scaled squared error loss function.

    Args:
        model_output (jnp.ndarray): Predicted values, shape (n_trials,).
        data (dict): Data dict for one sample. Uses key 'y' as target.

    Returns:
        jnp.ndarray: Scalar loss value.
    """
    y_true = data["y"]
    return 10 * (y_true - model_output) ** 2


# ========================
# 4. DIAGNOSTICS
# ========================


def plot_model_fits(
    data,
    programs_list,
    eval_grid,
    save_path="",
    labels=("model_v1", "model_v2"),
):
    """
    Plot observed synthetic data and model predictions for 9 random samples.

    Parameters
    ----------
    data : dict[str, np.ndarray]
        Data dict with keys 'x' and 'y', each shape (n_samples, n_trials).
    programs_list : list[dict]
        List of dictionaries with model metadata. Expected keys include:
        - 'model': callable model(data, params)
        - 'params': batched parameter pytree
        - 'losses': array of shape (n_samples,)
    eval_grid : dict[str, np.ndarray]
        Evaluation grid dict with key 'x', shape (n_eval_points,).
    save_path : str
        Output path. If empty, raises an error.
    """
    if save_path == "":
        raise ValueError("Please provide a save path for the plot")

    x_arr = data["x"]
    y_arr = data["y"]
    x_eval = np.asarray(eval_grid["x"]).reshape(-1)
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
    idx = np.arange(n_show)

    params_by_model = [
        utils.broadcast_params(program["params"], n_samples)
        for program in programs_list
    ]

    fig, axes = plt.subplots(3, 3, figsize=(18, 18))
    axes = axes.reshape(3, 3)

    for i in range(9):
        ax = axes[i // 3, i % 3]
        if i >= n_show:
            ax.axis("off")
            continue

        s = idx[i]
        x = x_arr[s]
        y_obs = y_arr[s]

        y_mean = compute_binned_means_on_eval(x, y_obs, x_eval)
        ax.scatter(x, y_obs, s=10, c="black", alpha=0.15, label="Observed")
        ax.plot(
            x_eval,
            y_mean,
            color=binned_colour,
            linewidth=4,
            label="Binned mean",
            alpha=0.8,
        )

        for j, program in enumerate(programs_list):
            model = program["model"]
            params = utils.slice_params(params_by_model[j], s)
            y_pred = utils.call_model(model, eval_grid, params)

            label = (
                labels[j]
                if labels is not None and j < len(labels)
                else f"Model {j + 1}"
            )
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
            mean_loss_parts.append(
                f"Model {j + 1} Loss: {np.mean(program['losses']):.2f}"
            )
        else:
            mean_loss_parts.append(f"Model {j + 1} Loss: n/a")
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
