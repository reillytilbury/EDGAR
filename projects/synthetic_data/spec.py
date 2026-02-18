"""
Welcome to the Model Discovery Engine! Fill in the components below to start building your model.

NECESSARY COMPONENTS:

Loading:
- load_and_process_data(data_path, *preprocess_params) -> [X, Y]
- train_test_split(X) -> [train_samples, train_trials]

Seed Programs:
- model_v1(X, *params) and param_est_v1(X, Y)
- model_v2(X, *params) and param_est_v2(X, Y)

OPTIONAL COMPONENTS:
- plot_model_fits(X, Y, model_list, params_list)
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


def param_est_v1(X, Y):
    """
    Estimate parameters for model_v1 using least squares.

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


def model_v2(X, a=1.0, b=0.0, lam=0.1):
    """
    Independent variable:
    X = [x]  # scalar input

    A nonlinear baseline model:
    y = (a*x + b) * exp(lam*x)

    Args:
        X (np.ndarray): Input array with shape (1, n_trials) or (n_trials,).
        a (float): Linear coefficient.
        b (float): Offset coefficient.
        lam (float): Exponential growth/decay parameter.

    Returns:
        np.ndarray: Predicted output, shape (n_trials,).
    """
    x = X[0]
    exp_term = np.exp(np.clip(lam * x, -10.0, 10.0))
    return (a * x + b) * exp_term


def param_est_v2(X, Y):
    """
    Estimate parameters for model_v2 via a polynomial approximation.

    Uses the local approximation exp(lam*x) ~ 1 + lam*x and fits
    y ≈ p2*x^2 + p1*x + p0 by least squares, then maps coefficients
    back to [a, b, lam].

    Args:
        X (np.ndarray): Input array with shape (1, n_trials) or (n_trials,).
        Y (np.ndarray): Target values with shape (n_trials,).

    Returns:
        np.ndarray: Estimated parameters [a, b, lam].
    """
    x = X[0]
    y = np.asarray(Y)

    A = np.vstack([x, x**2, np.ones(len(x))]).T
    p1, p2, p0 = np.linalg.lstsq(A, y, rcond=None)[0]

    b = p0
    if np.abs(p2) < 1e-8:
        a = p1
        lam = 0.0
    else:
        # Approximate solve for a and lam using p2 = a*lam and p1 = a + b*lam
        # Rearranged quadratic in a: a^2 - p1*a + b*p2 = 0
        disc = np.clip(p1**2 - 4.0 * b * p2, 0.0, None)
        a_cand_1 = 0.5 * (p1 + np.sqrt(disc))
        a_cand_2 = 0.5 * (p1 - np.sqrt(disc))
        a = a_cand_1 if np.abs(a_cand_1) > 1e-6 else a_cand_2
        if np.abs(a) < 1e-8:
            a = p1
            lam = 0.0
        else:
            lam = p2 / a

    return np.array([a, b, lam])


# ========================
# 3. DIAGNOSTICS
# ========================

def plot_model_fits(
    X,
    Y,
    programs_list,
    n_bins=50,
    domain=(-1.0, 1.0),
    save_path="",
    # -- ALL SUBSEQUENT PARAMS MUST BE SPECIFIED IN THE CONFIG FILE ---
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
        - optionally 'losses': array of shape (n_samples,)
    n_bins : int
        Number of bins for plotting binned means of observed data.
    domain : tuple[float, float]
        x-domain over which binned means are computed.
    save_path : str
        Output path. If empty, raises an error.
    """
    if save_path == "":
        raise ValueError("Please provide a save path for the plot")

    x_arr = _to_array3d(X)
    y_arr = _to_array3d(Y)
    n_samples = x_arr.shape[0]

    n_show = min(9, n_samples)
    idx = np.random.default_rng(0).choice(n_samples, size=n_show, replace=False)

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

        x_eval, y_mean = compute_binned_means(x, y_obs, n_bins=n_bins, domain=domain)
        ax.scatter(x, y_obs, s=8, c="black", alpha=0.15, label="Observed")
        ax.plot(x_eval, y_mean, color="blue", linewidth=2, label="Binned mean")

        for j, program in enumerate(programs_list):
            model = program["model"]
            params = program["params"][s]
            y_pred = model(np.array([x_eval]), *params)

            label = f"Model {j+1}"
            if "losses" in program:
                label += f" (loss={program['losses'][s]:.2f})"
            ax.plot(x_eval, np.asarray(y_pred).flatten(), linewidth=2, label=label)

        ax.set_xlim(domain)
        ax.set_title(f"Sample {s}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ========================
# 4. OPTIONAL PROJECT-SPECIFIC HELPERS
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
