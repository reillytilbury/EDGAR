"""
Welcome to the Model Discovery Engine! Fill in the components below to start building your model.

NECESSARY COMPONENTS:

Loading:
- load_and_process_data(data_path, *preprocess_params) -> [X, Y]
- train_test_split(X) -> [train_samples, train_trials]

Seed Programs:
- model_v1(X, params) and param_est_v1(X, Y)
- model_v2(X, params) and param_est_v2(X, Y)

Loss:
- loss_fn(Y_pred, Y_true) -> loss values

OPTIONAL COMPONENTS:
- plot_model_fits(X, Y, programs_list, X_eval, save_path, labels)
"""
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from typing import List, Tuple
from projects.synthetic_data.spec import compute_binned_means_on_eval
from src.data_structures import Inputs, Outputs
from src import utils


# ========================
# 1. DATA
# ========================

def load_and_process_data(
    data_path: str, 
    # ---- ALL SUBSEQUENT PARAMS MUST BE SPECIFIED IN THE CONFIG FILE ----
    activity_threshold: float = 0.1, 
    conc_threshold: float = 0.1,
) -> Tuple[Inputs, Outputs]:
    """
    Load and preprocess neural data and return data in the form of 
    Inputs and Outputs objects. 
    
    Parameters
    ----------
    data_path : str
        Path to the .npy file containing neural data.
    activity_threshold : float
        Threshold for activity to select good cells.
    conc_threshold : float
        Threshold for concentration to select good cells.
    # TODO: ADD back a way to create input and output variable names
    # Left this out as I don't know if this should be a list or a string. 
    # If a list, same number of entries as features? This gets unwieldy 
    # for high-dim problems...

    Returns
    -------
    X: Inputs object
    Y: Outputs object
        """
    # load and preprocess data
    neural_data = np.load(data_path, allow_pickle=True)
    neural_data = neural_data.item()
    # Use passed data extraction function or fall back to default
    response = extract_stimulus_related_response(neural_data, n_pcs=0)

    angles = neural_data['istim']
    n_trials = response.shape[1]
    n_trials_small = int(n_trials * activity_threshold)

    # filter 
    active = (response > 0).astype(np.float32)
    firing_probs = np.mean(active, axis=1)
    conc = np.abs(np.sum(np.exp(2j * angles)[np.newaxis, :] * response, axis=1) / np.sum(response, axis=1))
    good_cells = np.where((firing_probs > activity_threshold) & (conc > conc_threshold))[0]
    n_good_cells = len(good_cells)

    # update angles and response to be (n_cells_small, n_trials_small) and (n_cells_small, n_trials_small)
    response_cropped, angles_cropped = np.zeros((n_good_cells, n_trials_small)), np.zeros((n_good_cells, n_trials_small))
    for i, cell in enumerate(good_cells):
        active_trials = response[cell] > 0
        active_trials_idx = np.where(active_trials)[0][:n_trials_small]
        response_cropped[i] = response[cell, active_trials_idx]
        angles_cropped[i] = angles[active_trials_idx]

    response_cropped = normalize_response(response_cropped)
    
    # Create Inputs object with the angles as the first (and currently only) input
    # Shape: (n_cells, 1, n_trials)
    X = Inputs.from_array(angles_cropped)
    
    # Create Outputs object wrapping the response
    # Shape: (n_cells, 1, n_trials) - scalar output per cell
    Y = Outputs.from_array(response_cropped)

    return X, Y

def train_test_split(
        X: Inputs, 
        # -- ALL SUBSEQUENT PARAMS MUST BE SPECIFIED IN THE CONFIG FILE ---
        random_seed: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    This function defines a way to generate:
     - sample splits for model optimization (train samples) and evaluation (test samples),
     - trial splits for parameter optimization (train trials) and evaluation (test trials) conditional on a model.
    
    Parameters
    ----
    X: Inputs object of shape (n_samples, n_input_features, n_trials)
    # NB: Must have n_samples >= 2 & n_trials >= 2
    random_seed: int defining RNG seed.

    Returns 
    ----
    train_samples: np.ndarray of length n_samples//2
    train_trials: np.ndarray of length n_trials//2
    """
    n_samples, _, n_trials = X.shape
    assert n_samples >= 2, 'Need at least 2 samples for cross-validated model optimzation/eval'
    assert n_trials >= 2, 'Need at least 2 trials for cross-validated params optimzation/eval'

    # Match legacy behavior:
    # - sample split uses JAX permutation with seed=random_seed (config defaults to 42)
    # - trial split uses JAX permutation with fixed seed=0
    sample_key = jax.random.PRNGKey(random_seed)
    shuffled_samples = jax.random.permutation(sample_key, jnp.arange(n_samples))
    train_samples = np.asarray(shuffled_samples[: n_samples // 2], dtype=np.int64)

    trial_key = jax.random.PRNGKey(0)
    shuffled_trials = jax.random.permutation(trial_key, jnp.arange(n_trials))
    train_trials = np.asarray(shuffled_trials[: n_trials // 2], dtype=np.int64)

    return train_samples, train_trials

# ========================
# 2. SEED MODELS
# ========================

def model_v1(X, params):  # independent variable: X = [theta]
    """
    Independent variable:
    X = [theta]  # stimulus angle (radians)

    A simple neuron model that computes the response based on a Gaussian tuning curve.
    Args:
        X (np.ndarray): Input array with shape (1, n_trials).
                        X[0] is the stimulus angle theta in radians.
        params (dict): Parameter dictionary with keys:
            - theta_pref: Preferred direction of the neuron.
            - baseline: Baseline firing rate.
            - amplitude: Maximum firing rate above baseline.
            - tuning_width: Width of the tuning curve.
    Returns:
        np.ndarray: The firing rate of the neuron, shape (n_trials,).
    """
    theta = X[0]  # Extract theta from first input
    theta_pref = np.clip(params["theta_pref"], 0, 2 * np.pi)
    baseline = np.clip(params["baseline"], 0, None)
    amplitude = np.clip(params["amplitude"], 0, None)
    tuning_width = np.clip(params["tuning_width"], 0.01, None)

    circ_dist_rad = lambda theta1, theta2: np.abs(np.arctan2(np.sin(theta1 - theta2), np.cos(theta1 - theta2)))
    dist = circ_dist_rad(theta, theta_pref)
    return baseline + amplitude * np.exp(-0.5 * (dist / tuning_width) ** 2)


model_v1.DEFAULT_PARAMS = {
    "theta_pref": 0.0,
    "baseline": 0.0,
    "amplitude": 1.0,
    "tuning_width": 1.0,
}

def param_est_v1(X, Y):
    """
    Estimates the parameters of the gaussian neuron model. We do this by creating a binned tuning curve and picking out salient features.
    Args:
        X (np.ndarray): Input array with shape (1, n_trials).
                        X[0] is the stimulus angle theta in radians.
        Y (np.ndarray): Spike counts corresponding to each trial, shape (n_trials,).
    Returns:
        dict: Estimated parameters with keys {"theta_pref", "baseline", "amplitude", "tuning_width"}.
    """
    theta = X[0]  # Extract theta from first input
    n_bins = 20
    bin_idx = ((theta * n_bins) / (2 * np.pi)).astype(np.int32)
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)
    sums = np.bincount(bin_idx, weights=Y, minlength=n_bins)
    counts = np.bincount(bin_idx, minlength=n_bins)
    tuning_curve = np.zeros(n_bins, dtype=np.float32)
    tuning_curve[counts > 0] = sums[counts > 0] / counts[counts > 0]
    pref_idx = np.argmax(tuning_curve)
    theta_pref = pref_idx * (2 * np.pi / n_bins)
    baseline = np.min(tuning_curve)
    amplitude = np.max(tuning_curve) - baseline
    half_max = baseline + amplitude / 2.0
    indices = (np.arange(-5, 6) + pref_idx) % n_bins
    above_half_max = tuning_curve[indices] >= half_max
    full_width_half_max = 2 * np.pi * np.sum(above_half_max) / n_bins
    tuning_width = full_width_half_max / (2.0 * np.sqrt(2 * np.log(2)))
    return {
        "theta_pref": float(theta_pref),
        "baseline": float(baseline),
        "amplitude": float(amplitude),
        "tuning_width": float(tuning_width),
    }

def model_v2(X, params):  # independent variable: X = [theta]
    """
    Independent variable:
    X = [theta]  # stimulus angle (radians)

    A neuron model that computes the response based on a double peaked gaussian tuning curve, with peaks at theta_pref and (theta_pref + pi) % 2pi.
    Args:
        X (np.ndarray): Input array with shape (1, n_trials).
                        X[0] is the stimulus angle theta in radians.
        params (dict): Parameter dictionary with keys:
            - theta_pref: Preferred angle in radians.
            - baseline: Baseline firing rate.
            - amplitude_1: Amplitude of the first peak.
            - amplitude_2: Amplitude of the second peak.
            - tuning_width: Width of the tuning curves around preferred angles.
    Returns:
        np.ndarray: The response of the neuron model, shape (n_trials,).
    """
    theta = X[0]  # Extract theta from first input
    theta_pref = np.clip(params["theta_pref"], 0, 2 * np.pi)
    baseline = np.clip(params["baseline"], 0, None)
    amplitude_1 = np.clip(params["amplitude_1"], 0, None)
    amplitude_2 = np.clip(params["amplitude_2"], 0, None)
    tuning_width = np.clip(params["tuning_width"], 0.01, None)
    
    circ_dist_rad = lambda theta1, theta2: np.abs(np.arctan2(np.sin(theta1 - theta2), np.cos(theta1 - theta2)))
    dist_1 = circ_dist_rad(theta, theta_pref)
    dist_2 = circ_dist_rad(theta, (theta_pref + np.pi) % (2 * np.pi))
    return baseline + amplitude_1 * np.exp(-0.5 * (dist_1 / tuning_width) ** 2) + amplitude_2 * np.exp(-0.5 * (dist_2 / tuning_width) ** 2)


model_v2.DEFAULT_PARAMS = {
    "theta_pref": 0.0,
    "baseline": 0.0,
    "amplitude_1": 1.0,
    "amplitude_2": 0.0,
    "tuning_width": 1.0,
}

def param_est_v2(X, Y):
    """
    A parameter estimator for the double peaked neuron model. Creates a binned tuning curve from spike counts and estimates parameters using features from the tuning curve.
    Args:
        X (np.ndarray): Input array with shape (1, n_trials).
                        X[0] is the stimulus angle theta in radians.
        Y (np.ndarray): Spike counts corresponding to the angles, shape (n_trials,).
    Returns:
        dict: Estimated parameters with keys {"theta_pref", "baseline", "amplitude_1", "amplitude_2", "tuning_width"}.
    """
    theta = X[0]  # Extract theta from first input
    n_bins = 50
    bin_idx = ((theta * n_bins) / (2 * np.pi)).astype(np.int32)
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)
    sums = np.bincount(bin_idx, weights=Y, minlength=n_bins)
    counts = np.bincount(bin_idx, minlength=n_bins)
    def gaussian_kernel(sig: int) -> np.ndarray:
        x = np.arange(-int(3 * sig), int(3 * sig) + 1)
        k = np.exp(-0.5 * (x / sig) ** 2)
        return k / np.sum(k)
    k = gaussian_kernel(2)
    pad = len(k) // 2
    sums_padded = np.pad(sums, (pad, pad), mode='wrap')
    counts_padded = np.pad(counts, (pad, pad), mode='wrap')
    num_conv = np.convolve(sums_padded, k, mode='valid')
    den_conv = np.convolve(counts_padded, k, mode='valid')
    tuning_curve = num_conv / (den_conv + 1e-8)
    pref_idx = np.argmax(tuning_curve)
    theta_pref = pref_idx * (2 * np.pi / n_bins)
    baseline = np.min(tuning_curve)
    amplitude_1 = np.max(tuning_curve) - baseline
    amplitude_2 = tuning_curve[(pref_idx + n_bins // 2) % n_bins] - baseline
    half_max = baseline + amplitude_1 / 2.0
    indices = (np.arange(-5, 6) + pref_idx) % n_bins
    above_half_max = tuning_curve[indices] >= half_max
    full_width_half_max = 2 * np.pi * np.sum(above_half_max) / n_bins
    tuning_width = full_width_half_max / (2.0 * np.sqrt(2 * np.log(2)))
    return {
        "theta_pref": float(theta_pref),
        "baseline": float(baseline),
        "amplitude_1": float(amplitude_1),
        "amplitude_2": float(amplitude_2),
        "tuning_width": float(tuning_width),
    }


# ========================
# 3. LOSS
# ========================

def loss_fn(Y_pred, Y_true):
    """
    Elementwise squared-error loss.
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
        - 'model': callable model(X_one, params)
        - 'params': batched parameter pytree
        - 'losses': array of shape (n_samples,)
    X_eval : array-like or Inputs
        Evaluation grid with shape (n_samples, n_features, n_eval_trials).
    save_path : str
        Output path. If empty, raises an error.
    """
    if save_path == "":
        raise ValueError("Please provide a save path for the plot")
    x_eval_fine = np.linspace(0, 2 * np.pi, 200)

    x_arr = _to_array3d(X)
    y_arr = _to_array3d(Y)
    x_eval_arr = _to_array3d(X_eval)
    n_samples = x_arr.shape[0]
    # diff colours for 1, 2 or 3 models
    if len(programs_list) == 1:
        colours = ["tab:red"]
    elif len(programs_list) == 2:
        colours = ["tab:green", "tab:red"]
    else:
        colours = ["tab:orange", "tab:green", "tab:red"]
    binned_colour = "deepskyblue"

    n_show = min(9, n_samples)
    # Intentionally unseeded so diagnostic samples vary across calls/runs.
    idx = np.random.default_rng().choice(n_samples, size=n_show, replace=False)

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
        x = x_arr[s, 0]
        y_obs = y_arr[s, 0]
        x_eval = np.asarray(x_eval_arr[s, 0]).reshape(-1)

        y_mean = compute_binned_means_on_eval(x, y_obs, x_eval)
        ax.scatter(x, y_obs, s=10, c="black", alpha=0.2, label="Observed")
        ax.plot(x_eval, y_mean, color=binned_colour, linewidth=4, label="Binned mean", alpha=0.8)

        for j, program in enumerate(programs_list):
            model = program["model"]
            params = utils.slice_params(params_by_model[j], s)
            loss = program["losses"][s]
            y_pred = utils.call_model(model, np.array([x_eval_fine]), params)

            label = labels[j] + f" (loss={loss:.2f})"
            ax.plot(
                x_eval_fine,
                np.asarray(y_pred).flatten(),
                color=colours[j % len(colours)],
                linewidth=3,
                label=label,
                alpha=0.8,
            )

        ax.set_xlim((float(np.min(x_eval)), float(np.max(x_eval))))
        ax.set_ylim((0, np.max(y_mean) * 1.5))
        ax.set_title(f"Cell {s}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.legend(fontsize=12)

    mean_loss_parts = []
    for j, program in enumerate(programs_list):
        if "losses" in program and np.size(program["losses"]) > 0:
            mean_loss_parts.append(f"Model {j+1}: {np.mean(program['losses']):.2f}")
        else:
            mean_loss_parts.append(f"Model {j+1}: n/a")
    summary = "\n".join(mean_loss_parts) if mean_loss_parts else "n/a"
    plt.suptitle(f"Model Fits\n{summary}", fontsize=24)
    plt.savefig(save_path, dpi=100.0, bbox_inches="tight")
    plt.close(fig)


    
# ========================
# 5. OPTIONAL PROJECT-SPECIFIC HELPERS
# ========================

def extract_stimulus_related_response(data: dict, n_pcs: int = 8, z_score: bool = False, spont_mean_removal: bool = False) -> np.ndarray:
    """
    Extracts the stimulus-related response from the data. Copy pasted with small modifications from https://github.com/MouseLand/stringer-et-al-2019/blob/master/utils.py#L98
    Args:
        data (dict): The data dictionary containing the stimulus-related response and other information. Values expected to be convertible to JAX arrays.
        n_pcs (int): The number of spointaneous PCs to remove from the response.
        z_score (bool): Whether to z-score the response.
    Returns:
        stim_related_response (np.ndarray): The stimulus-related response matrix.
    """
    # Convert relevant data parts to JAX arrays explicitly if needed, JAX often handles np arrays implicitly
    sresp = np.asarray(data['sresp'])

    if spont_mean_removal:
        mean_spont = np.asarray(data['mean_spont'])
        sresp = sresp - mean_spont[:, np.newaxis]

    if n_pcs > 0:
        u_spont = np.asarray(data['u_spont'])
        sresp = sresp - u_spont[:, :n_pcs] @ (u_spont[:, :n_pcs].T @ sresp)

    if z_score:
        sresp = (sresp - np.mean(sresp, axis=1, keepdims=True)) / np.std(sresp, axis=1, keepdims=True)

    return sresp

def normalize_response(response: np.ndarray) -> np.ndarray:
    """
    Normalizes the response matrix
    
    Args:
        response (np.ndarray): The neural response matrix of shape (n_cells, n_trials).
    Returns:
        normalized_response (np.ndarray): The normalized response matrix.
    """
    normalized_response = 100 * response / np.linalg.norm(response, axis=1, keepdims=True)  # multiply the response by 100 and normalize
    return normalized_response


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
