"""
Welcome to the Model Discovery Engine! Fill in the components below to start building your model.

NECESSARY COMPONENTS:

Loading:
- load_and_process_data(data_path, *preprocess_params) -> [[d_train_train, d_train_test], [d_test_train, d_test_test]]

Seed Programs:
- model_v1(data, params) and param_est_v1(data)
- model_v2(data, params) and param_est_v2(data)
- params is a dict of named arrays/scalars (same keys for model + estimator)

LOSS FUNCTION:
- loss_fn(model_output, data) -> loss values

OPTIONAL COMPONENTS:
- plot_model_fits(data, programs_list, eval_grid, save_path, labels)
"""
import numpy as np
import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import matplotlib.pyplot as plt
import scipy
from typing import Dict
import optax
import src.utils as utils

# ========================
# 1. DATA
# ========================

def load_and_process_data(data_path : str, *args, **kwargs) -> list:
    X = np.load(data_path, allow_pickle=True)
    return X.tolist()

def load_and_process_data_true(
    data_path: str,
    # ---- ALL SUBSEQUENT PARAMS MUST BE SPECIFIED IN THE CONFIG FILE ----
    train_cells_random_seed: int = 42,
    test_cells_random_seed: int = 0,
    source_to_target_split_ratio: float = 0.5,
    conc_threshold: float = 0.55,
) -> list[list[Dict[str, np.ndarray]]]:
    """
    Load and preprocess neural data, split into train/test times and cells,
    and return a 2x2 container of data dicts.

    The sample split divides time into two independent halves : S1 and S2.
    Within each half we split cells into 4 groups using a random seed:
    source1, target1, source2, target2.

    For S1 (using train_cells_random_seed):
        X00 contains S1 stim responses of source1 and target1.
        X01 contains S1 stim responses of source2 and target2.

    For S2 (using test_cells_random_seed):
        X10 contains S2 stim responses of source1' and target1'.
        X11 contains S2 stim responses of source2' and target2'.

    If train_cells_random_seed == test_cells_random_seed, the cell split
    is the same across the two stimuli sets.

    Ensure that the number of cells in source1 and source2 are the same - and similarly for target1 and target2.

    Parameters
    ----------
    data_path : str
        Path to the .npy file containing neural data.
    train_cells_random_seed : int
        Random seed for reproducibility of cell splits for X00 and X01.
    test_cells_random_seed : int
        Random seed for reproducibility of cell splits for X10 and X11.
    source_to_target_split_ratio : float
        Fraction of cells assigned to source vs target (e.g. 0.5 means
        equal source and target populations).
    conc_threshold : float
        Orientation-selectivity threshold for cell filtering.

    Returns
    -------
    2x2 list of dicts:
        [[X00, X01],
         [X10, X11]]
        Each dict has keys 'stimulus', 'source', 'target',
        'source_tuning_params', 'target_tuning_params',
        'source_mean_pred', 'target_mean_pred'.
        'stimulus' has shape (n_time,), 'source' has shape
        (n_time, n_source_cells), 'target' has shape
        (n_time, n_target_cells).
    """
    neural_data = np.load(data_path, allow_pickle=True)
    neural_data = neural_data.item()
    response = np.asarray(neural_data['sresp'])  # (n_cells, n_trials)

    angles = neural_data['istim']
    assert max(angles) <= 2 * np.pi, "Expected angles to be in radians and between 0 and 2pi"
    n_cells_total, n_trials = response.shape

    if conc_threshold is not None:
        conc = np.abs(np.sum(np.exp(2j * angles)[np.newaxis, :] * response, axis=1) / np.sum(response, axis=1))
        good_cells = np.where(conc > conc_threshold)[0]
        print(f"Filtering for orientation selectivity. Kept {len(good_cells)} out of {n_cells_total} cells.")
        response = response[good_cells]

    # Normalise per cell (divide by std across all trials)
    eps = 1e-12
    response = response / (response.std(axis=1, keepdims=True) + eps)

    # --- Time split: first half S1, second half S2 ---
    half_trials = n_trials // 2
    S1_trials = np.arange(half_trials)
    S2_trials = np.arange(half_trials, n_trials)

    def _split_cells(seed):
        """Split cells into source1, target1, source2, target2 using given seed."""
        rng = np.random.default_rng(seed)
        n_cells = response.shape[0]
        
        # Ensure that the number of cells in source1 and source2 are the same - and similarly for target1 and target2.
        cell_idx = rng.permutation(n_cells)
        n_source = int(np.ceil(source_to_target_split_ratio * n_cells))
        source_cells = cell_idx[:n_source]
        target_cells = cell_idx[n_source:]

        if source_cells.size == 0 or target_cells.size == 0:
            raise ValueError(
                "source_to_target_split_ratio results in empty source or "
                "target cell population. Please adjust the ratio."
            )
        half_source = source_cells.size // 2
        half_target = target_cells.size // 2
        source1 = source_cells[:half_source]
        source2 = source_cells[half_source:][:half_source] # ensure source2 has the same number of cells as source1
        target1 = target_cells[:half_target]
        target2 = target_cells[half_target:][:half_target] # ensure target2 has the same number of cells as target1
        return source1, target1, source2, target2

    # Train cell split (for S1)
    source1, target1, source2, target2 = _split_cells(train_cells_random_seed)
    # Test cell split (for S2, potentially different)
    source1p, target1p, source2p, target2p = _split_cells(test_cells_random_seed)

    def _make_data_dict(source_cells, target_cells, trial_idx):
        """Build a data dict with shapes (1, n_time, n_cells) for source/target."""
        stimulus = angles[trial_idx]
        source_resp = response[source_cells][:, trial_idx]  # (n_source, n_time)
        target_resp = response[target_cells][:, trial_idx]  # (n_target, n_time)

        # Fit tuning parameters (per-cell, shape (n_cells, 11))
        source_tuning_params = fit_tuning_parameters_jax(stimulus, source_resp)
        target_tuning_params = fit_tuning_parameters_jax(stimulus, target_resp)

        # Compute mean predictions: (n_cells, n_time)
        stimuli_tiled_s = jnp.tile(stimulus[None, :], (source_tuning_params.shape[0], 1))
        source_pred = jax.vmap(
            lambda stim, params: single_cell_tuning_function(stim, *params),
            in_axes=(0, 0),
        )(stimuli_tiled_s, source_tuning_params)  # (n_source, n_time)

        stimuli_tiled_t = jnp.tile(stimulus[None, :], (target_tuning_params.shape[0], 1))
        target_pred = jax.vmap(
            lambda stim, params: single_cell_tuning_function(stim, *params),
            in_axes=(0, 0),
        )(stimuli_tiled_t, target_tuning_params)  # (n_target, n_time)

        # Add a leading size-1 sample axis so data_n_samples returns 1 and
        # slice_data_samples(data, 0) cleanly removes it, restoring the
        # 2D shapes that model_v1 and param_est_v1 expect.
        # Transpose responses and predictions: (n_cells, n_time) -> (n_time, n_cells)
        return {
            'stimulus': np.array(stimulus)[np.newaxis],                          # (1, n_time)
            'source': np.array(source_resp.T)[np.newaxis],                       # (1, n_time, n_source)
            'target': np.array(target_resp.T)[np.newaxis],                       # (1, n_time, n_target)
            'source_tuning_params': np.array(source_tuning_params)[np.newaxis],  # (1, n_source, 11)
            'target_tuning_params': np.array(target_tuning_params)[np.newaxis],  # (1, n_target, 11)
            'source_mean_pred': np.array(source_pred.T)[np.newaxis],             # (1, n_time, n_source)
            'target_mean_pred': np.array(target_pred.T)[np.newaxis],             # (1, n_time, n_target)
        }

    # X00 and X01 share trials (S1), differ in cells
    # X10 and X11 share trials (S2), differ in cells (different random seed)
    return [
        [_make_data_dict(source1, target1, S1_trials),    # X00
         _make_data_dict(source2, target2, S1_trials)],   # X01
        [_make_data_dict(source1p, target1p, S2_trials),  # X10
         _make_data_dict(source2p, target2p, S2_trials)], # X11
    ]

# ========================
# 2. SEED MODELS
# ========================

def model_v1(data, params):
    """ Gain Modulation + Additive Offset

    Equation : For each target cell c at timepoint t with stimulus angle theta,
        f(theta, t; cell_params) = multiplicative_gain(t) * g(theta(t) ; cell_params) + additive_offset(t)
    where g(theta(t); cell_params) is the tuning function prediction for each target cell.

    Args :
        data (dict) : Dictionary with keys
            - 'stimulus' : shape (n_time,)
            - 'source' : shape (n_time, n_source_cells)
            - 'source_tuning_params' : shape (n_source_cells, n_params)
            - 'target_tuning_params' : shape (n_target_cells, n_params)
            - 'source_mean_pred' : shape (n_time, n_source_cells)
            - 'target_mean_pred' : shape (n_time, n_target_cells)
        params (dict) : Parameter dictionary with keys:
            - multiplicative_gain : shape (n_time,)
            - additive_offset : shape (n_time,)

    Returns :
        jnp.ndarray : Predicted responses for the target cells with shape (n_time, n_target_cells)
    """
    g_target = data['target_mean_pred']  # shape (n_time, n_target)

    multiplicative_gain = params['multiplicative_gain']  # shape (n_time,)
    additive_offset = params['additive_offset']          # shape (n_time,)

    pred = multiplicative_gain[:, None] * g_target + additive_offset[:, None]  # shape (n_time, n_target)
    # clip to non-negative firing rates
    pred = jnp.clip(pred, a_min=0.0)
    return pred


def param_est_v1(data):
    """ Parameter estimator for model_v1. Estimates time-specific multiplicative
    gain and additive offset from the source cell responses.

    Args :
        data (dict) : Dictionary containing input and output arrays. Keys:
            - 'stimulus' : shape (n_time,)
            - 'source' : shape (n_time, n_source_cells)
            - 'target' : shape (n_time, n_target_cells)
            - 'source_tuning_params' : shape (n_source_cells, n_params)
            - 'target_tuning_params' : shape (n_target_cells, n_params)
            - 'source_mean_pred' : shape (n_time, n_source_cells)
            - 'target_mean_pred' : shape (n_time, n_target_cells)

    Returns :
        params (dict) : Estimated parameters. Keys:
            - multiplicative_gain : shape (n_time,)
            - additive_offset : shape (n_time,)
    """
    x = jnp.array(data['source'])           # shape (n_time, n_source)
    g_source = jnp.array(data['source_mean_pred'])  # shape (n_time, n_source)

    eps = 1e-8
    # Step 1 : Fit multiplicative gain per timepoint using least squares
    # For each t: multiplicative_gain[t] = dot(g_source[t,:], x[t,:]) / dot(g_source[t,:], g_source[t,:])
    multiplicative_gain = jnp.sum(g_source * x, axis=1) / (jnp.sum(g_source**2, axis=1) + eps)  # shape (n_time,)

    # Step 2 : Estimate additive offset from the mean residual across source cells
    residual = x - multiplicative_gain[:, None] * g_source  # shape (n_time, n_source)
    additive_offset = jnp.mean(residual, axis=1)  # shape (n_time,)

    params = {
        'multiplicative_gain': multiplicative_gain,
        'additive_offset': additive_offset,
    }
    return params

def model_v2(data, params):
    """ Gain Modulation + source to target coupling

    Equation : For each target cell c at timepoint t with stimulus angle theta,
        f(theta, t; cell_params) = multiplicative_gain(t) * g(theta(t) ; cell_params) + source_response(t) @ coupling_weight(c)
    where g(theta(t); cell_params) is the tuning function prediction,
    source_response(t) is the source cell responses at time t (shape n_source,),
    and coupling_weight(c) is the coupling from source cells to target cell c.

    Args :
        data (dict) : Dictionary with keys
            - 'stimulus' : shape (n_time,)
            - 'source' : shape (n_time, n_source_cells)
            - 'source_tuning_params' : shape (n_source_cells, n_params)
            - 'target_tuning_params' : shape (n_target_cells, n_params)
            - 'source_mean_pred' : shape (n_time, n_source_cells)
            - 'target_mean_pred' : shape (n_time, n_target_cells)
        params (dict) : Parameter dictionary with keys:
            - multiplicative_gain : shape (n_time,)
            - coupling_factor : shape (n_target_cells, n_source_cells)

    Returns :
        jnp.ndarray : Predicted responses for the target cells with shape (n_time, n_target_cells)
    """
    source_response = jnp.array(data['source'])       # shape (n_time, n_source)
    g_target = jnp.array(data['target_mean_pred'])     # shape (n_time, n_target)

    multiplicative_gain = params['multiplicative_gain']  # shape (n_time,)
    coupling_factor = params['coupling_factor']          # shape (n_target, n_source)

    # source_response @ coupling_factor.T: (n_time, n_source) @ (n_source, n_target) = (n_time, n_target)
    pred = multiplicative_gain[:, None] * g_target + source_response @ coupling_factor.T  # shape (n_time, n_target)

    # clip to non-negative firing rates
    pred = jnp.clip(pred, a_min=0.0)
    return pred

def param_est_v2(data):
    """Parameter estimator for model_v2. Estimates multiplicative gain from
    source responses and coupling factor by regressing target residuals
    against source responses.

    Args :
        data (dict) : Dictionary containing input and output arrays. Keys:
            - 'stimulus' : shape (n_time,)
            - 'source' : shape (n_time, n_source_cells)
            - 'target' : shape (n_time, n_target_cells)
            - 'source_tuning_params' : shape (n_source_cells, n_params)
            - 'target_tuning_params' : shape (n_target_cells, n_params)
            - 'source_mean_pred' : shape (n_time, n_source_cells)
            - 'target_mean_pred' : shape (n_time, n_target_cells)

    Returns :
        params (dict) : Estimated parameters. Keys:
            - multiplicative_gain : shape (n_time,)
            - coupling_factor : shape (n_target_cells, n_source_cells)
    """
    x = jnp.array(data['source'])                # shape (n_time, n_source)
    y = jnp.array(data['target'])                # shape (n_time, n_target)
    g_source = jnp.array(data['source_mean_pred'])  # shape (n_time, n_source)
    g_target = jnp.array(data['target_mean_pred'])  # shape (n_time, n_target)

    # Step 1 : Fit multiplicative gain per timepoint using least squares
    eps = 1e-8
    multiplicative_gain = jnp.sum(g_source * x, axis=1) / (jnp.sum(g_source**2, axis=1) + eps)  # shape (n_time,)

    # Step 2 : Fit coupling factor by regressing target residuals against source responses
    # residual = y - gain-only prediction, shape (n_time, n_target)
    residual = y - multiplicative_gain[:, None] * g_target

    # x.T @ residual: (n_source, n_time) @ (n_time, n_target) = (n_source, n_target)
    XtX = x.T @ x + eps * jnp.eye(x.shape[1])  # Add ridge regularization for stability
    coupling_factor = jnp.linalg.solve(XtX, x.T @ residual).T  # (n_target, n_source)

    params = {
        'multiplicative_gain': multiplicative_gain,
        'coupling_factor': coupling_factor,
    }
    return params

# ========================
# 3. LOSS
# ========================

def loss_fn(model_output, data):
    """
    Elementwise squared-error loss.

    Parameters
    ----------
    model_output : jnp.ndarray
        Predicted target-cell responses, shape (n_time, n_target_cells).
    data : dict
        Data dictionary; the comparison target is data['target'].
    """
    Y_true = data['target']
    return (Y_true - model_output) ** 2

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
    Plot observed target responses and overlaid model predictions for 9 random
    stimulus angles, with binned per-angle MSE shown underneath each panel.

    Parameters
    ----------
    data : dict[str, np.ndarray]
        Expected keys:
        - 'stimulus': array of shape (n_stims,)
        - 'target': array of shape (n_stims, n_target_cells)
        - 'target_tuning_params': array of shape (n_target_cells, n_params),
          where column 0 contains preferred angles
    programs_list : list[dict]
        List of model program dictionaries. Expected keys include:
        - 'model': callable model(data, params)
        - 'params': parameter pytree or parameter object for the model
        Optionally:
        - 'losses': array-like
    eval_grid : dict[str, np.ndarray]
        Included to preserve interface compatibility. Not used here.
    save_path : str
        Output path. If empty, raises an error.
    labels : tuple[str, ...]
        Labels for plotted models.
    """
    if save_path == "":
        raise ValueError("Please provide a save path for the plot")

    if len(programs_list) < 2:
        raise ValueError("programs_list must contain at least 2 models")

    # The engine passes a size-1 sample axis into project plotters, but the
    # seed models in this spec expect a single sample with no leading batch dim.
    plot_data = utils.slice_data_samples(data, 0)

    stims = np.asarray(plot_data["stimulus"]).reshape(-1)
    target = np.asarray(plot_data["target"])
    target_tuning_params = np.asarray(plot_data["target_tuning_params"])

    n_show = min(9, len(stims))
    random_angles_idx = np.random.default_rng().choice(len(stims), size=n_show, replace=False)

    preferred_angles = target_tuning_params[:, 0]
    preferred_angles_sorted_indices = np.argsort(preferred_angles)
    preferred_angles = preferred_angles[preferred_angles_sorted_indices]

    actual_response = target[:, preferred_angles_sorted_indices]

    # Compute predictions for each model on the full dataset
    predictions = []
    binned_mse_losses = []

    n_bins = 60
    bin_edges = np.linspace(0, 2 * np.pi, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_indices = np.digitize(preferred_angles, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    for program in programs_list[:2]:
        model = program["model"]
        params = program["params"]

        params_leaves = jax.tree_util.tree_leaves(params)
        plot_params = (
            utils.slice_params(params, 0)
            if any(np.asarray(leaf).ndim > 1 for leaf in params_leaves)
            else params
        )

        y_pred = model(plot_data, plot_params)
        y_pred = np.asarray(y_pred)[:, preferred_angles_sorted_indices]
        predictions.append(y_pred)

        resid = y_pred - actual_response  # (n_time, n_target_cells)
        squared_loss = resid ** 2

        # x-axis is the preferred angle of each cell. For a given time, bin cells by preferred
        # angle and average their squared residuals within each bin. Output shape (n_bins, n_time).
        binned_mse_loss = np.zeros((n_bins, squared_loss.shape[0]))  # (n_bins, n_time)
        for i in range(n_bins):
            mask = bin_indices == i  # mask over cells (axis 1 of squared_loss)
            if np.any(mask):
                binned_mse_loss[i] = np.mean(squared_loss[:, mask], axis=1)
            else:
                binned_mse_loss[i] = 0.0

        binned_mse_losses.append(binned_mse_loss)

    colours = ["tab:orange", "tab:blue"]
    actual_colour = "tab:grey"

    fig = plt.figure(figsize=(15, 15))
    outer_gs = fig.add_gridspec(3, 3, hspace=0.5, wspace=0.3)

    for i in range(9):
        row, col = divmod(i, 3)

        ax_slot = outer_gs[row, col]
        inner_gs = ax_slot.subgridspec(
            2,
            1,
            height_ratios=[4, 1],
            hspace=0.05,
        )

        ax1 = fig.add_subplot(inner_gs[0])
        ax2 = fig.add_subplot(inner_gs[1], sharex=ax1)

        if i >= n_show:
            ax1.axis("off")
            ax2.axis("off")
            continue

        angle_idx = random_angles_idx[i]
        random_angle = stims[angle_idx]

        # Top plot: actual vs two model predictions
        ax1.scatter(
            preferred_angles,
            actual_response[angle_idx, :],
            color=actual_colour,
            label="Actual Response",
            alpha=0.4,
            s=12,
        )

        for j, y_pred in enumerate(predictions):
            label = labels[j] if labels is not None and j < len(labels) else f"Model {j+1}"

            ax1.plot(
                preferred_angles,
                y_pred[angle_idx, :],
                color=colours[j],
                label=label,
                alpha=0.4,
                linewidth=1,
            )

        ax1.axvline(random_angle, color="gray", linestyle="--", label="Stimulus Angle")
        ax1.set_ylabel("Cell Response")
        ax1.set_title(f"Stimulus Angle = {random_angle:.2f} radians")
        ax1.legend()
        # ax1.legend(loc="upper left", fontsize=9)
        ax1.tick_params(axis="x", labelbottom=False)

        # Bottom plot: binned MSE for both models
        for j, binned_mse_loss in enumerate(binned_mse_losses):
            mse_label = (
                f"{labels[j]} binned MSE"
                if labels is not None and j < len(labels)
                else f"Model {j+1} binned MSE"
            )
            ax2.plot(
                bin_centers,
                binned_mse_loss[:, angle_idx],
                color=colours[j],
                alpha=0.9,
                linewidth=2,
                label=mse_label,
            )

        ax2.axvline(random_angle, color="gray", linestyle="--")
        ax2.set_xlabel("Preferred Angle of Target Cell (radians)")
        ax2.set_ylabel("Binned\nMSE", fontsize=9)
        ax2.legend(fontsize=8)

    mean_loss_parts = []
    for j, program in enumerate(programs_list[:2]):
        if "losses" in program and np.size(program["losses"]) > 0:
            mean_loss_parts.append(
                f"{labels[j] if labels is not None and j < len(labels) else f'Model {j+1}'} "
                f"Loss: {np.mean(program['losses']):.2f}"
            )
        else:
            mean_loss_parts.append(
                f"{labels[j] if labels is not None and j < len(labels) else f'Model {j+1}'} "
                f"Loss: n/a"
            )

    summary = "\n".join(mean_loss_parts)
    fig.suptitle(
        f"Observed vs Model Predictions for 9 Random Stimulus Angles\n{summary}",
        y=0.995,
        fontsize=16,
    )
    plt.savefig(save_path, dpi=100.0, bbox_inches="tight")
    plt.close(fig)

# ========================
# 4. OPTIONAL PROJECT-SPECIFIC HELPERS
# ========================

def single_cell_tuning_function(theta,
                theta_pref_1=0.0,
                baseline=0.0,
                amplitude_1=1.0,
                width_ccw_1=1.0,
                width_cw_1=1.0,
                exponent_1=2.0,
                theta_pref_2=jnp.pi,
                amplitude_2=0.0,
                width_ccw_2=1.0,
                width_cw_2=1.0,
                exponent_2=2.0):

    min_width = 5e-2
    eps = 1e-12
    min_exponent, max_exponent = 0.1, 5.0
    width_ccw_1, width_cw_1 = jnp.clip(width_ccw_1, min_width, None), jnp.clip(width_cw_1, min_width, None)
    width_ccw_2, width_cw_2 = jnp.clip(width_ccw_2, min_width, None), jnp.clip(width_cw_2, min_width, None)
    exponent_1, exponent_2 = jnp.clip(exponent_1, min_exponent, max_exponent), jnp.clip(exponent_2, min_exponent, max_exponent)
    baseline = jnp.clip(baseline, 0.0, None)
    amplitude_1, amplitude_2 = jnp.clip(amplitude_1, 0.0, None), jnp.clip(amplitude_2, 0.0, None)

    def _signed_circ_diff_rad(angle_radians, preferred_angle_radians):
        delta = angle_radians - preferred_angle_radians
        return jnp.arctan2(jnp.sin(delta), jnp.cos(delta))
        
    signed_diff_1 = _signed_circ_diff_rad(theta, theta_pref_1) + eps  # Add small epsilon to avoid log(0) issues
    width_1_effective = jnp.where(signed_diff_1 < 0, width_ccw_1, width_cw_1)
    width_1_effective = jnp.maximum(width_1_effective, 1e-6)
    peak1_component = amplitude_1 * jnp.exp(-0.5 * (jnp.abs(signed_diff_1) / width_1_effective) ** exponent_1)

    signed_diff_2 = _signed_circ_diff_rad(theta, theta_pref_2) + eps  # Add small epsilon to avoid log(0) issues
    width_2_effective = jnp.where(signed_diff_2 < 0, width_ccw_2, width_cw_2)
    width_2_effective = jnp.maximum(width_2_effective, 1e-6)
    peak2_component = amplitude_2 * jnp.exp(-0.5 * (jnp.abs(signed_diff_2) / width_2_effective) ** exponent_2)
    return baseline + peak1_component + peak2_component


def fit_tuning_parameters_jax(stims, spike_counts):
    """ Fit tuning parameters for a single cell using JAX. This is a helper function for load_and_process_data 
    and we will store the tuning parameters as part of the params dict that gets passed to the model."""

    def tuning_parameter_estimator_jax(stimuli, spike_counts):
        """JAX-compatible equivalent of parameter_estimator for jnp.array inputs."""
        n_bins = 256
        kernel_sigma = 2.5
        min_peak_amplitude = 0.5
        min_model_width = 1e-6
        default_width_value = 1.0
        min_exponent = 0.1
        max_exponent = 5.0
        default_exponent_value = 2.0
        min_second_peak_ratio = 0.1
        min_second_peak_separation = jnp.pi / 4

        stimuli = jnp.asarray(stimuli)
        spike_counts = jnp.asarray(spike_counts)

        bin_idx = ((stimuli * n_bins) / (2 * jnp.pi)).astype(jnp.int32)
        bin_idx = jnp.clip(bin_idx, 0, n_bins - 1)

        sums = jnp.zeros(n_bins, dtype=spike_counts.dtype).at[bin_idx].add(spike_counts)
        counts = jnp.zeros(n_bins, dtype=spike_counts.dtype).at[bin_idx].add(1.0)

        kernel_radius = int(3 * kernel_sigma)
        x_kernel = jnp.arange(-kernel_radius, kernel_radius + 1)
        kernel = jnp.exp(-0.5 * (x_kernel / kernel_sigma) ** 2)
        kernel = kernel / (jnp.sum(kernel) + 1e-8)

        pad = kernel.shape[0] // 2
        sums_padded = jnp.concatenate([sums[-pad:], sums, sums[:pad]])
        counts_padded = jnp.concatenate([counts[-pad:], counts, counts[:pad]])

        num_conv = jnp.convolve(sums_padded, kernel, mode='valid')
        den_conv = jnp.convolve(counts_padded, kernel, mode='valid')
        tuning_curve = jnp.where(den_conv > 1e-8, num_conv / jnp.maximum(den_conv, 1e-8), 0.0)

        angle_step = 2 * jnp.pi / n_bins
        baseline_est = jnp.maximum(0.0, jnp.min(tuning_curve))

        def _get_peak_params_simple_jax(peak_idx_val, peak_idx, bsl, tc, n_bns, ang_step,
                                        min_w, def_w, min_exp, max_exp, def_exp, min_amp_thresh):
            amp = peak_idx_val - bsl
            search_offsets = jnp.arange(1, n_bns // 2 + 1)
            sentinel = n_bns + 1

            target_half_val = bsl + amp / 2.0
            ccw_half_vals = tc[(peak_idx - search_offsets + n_bns) % n_bns]
            cw_half_vals = tc[(peak_idx + search_offsets) % n_bns]
            half_ccw_bins = jnp.min(jnp.where(ccw_half_vals <= target_half_val, search_offsets, sentinel))
            half_cw_bins = jnp.min(jnp.where(cw_half_vals <= target_half_val, search_offsets, sentinel))
            half_ccw_bins = jnp.where(half_ccw_bins == sentinel, 0, half_ccw_bins)
            half_cw_bins = jnp.where(half_cw_bins == sentinel, 0, half_cw_bins)

            sqrt_2_log_2 = jnp.sqrt(2.0 * jnp.log(2.0))
            width_ccw = jnp.where(half_ccw_bins > 0,
                                (half_ccw_bins.astype(tc.dtype) * ang_step) / sqrt_2_log_2,
                                def_w)
            width_cw = jnp.where(half_cw_bins > 0,
                                (half_cw_bins.astype(tc.dtype) * ang_step) / sqrt_2_log_2,
                                def_w)
            width_ccw = jnp.clip(width_ccw, min_w, jnp.pi)
            width_cw = jnp.clip(width_cw, min_w, jnp.pi)

            target_qtr_val = bsl + amp / 4.0
            ccw_qtr_vals = tc[(peak_idx - search_offsets + n_bns) % n_bns]
            cw_qtr_vals = tc[(peak_idx + search_offsets) % n_bns]
            qtr_ccw_bins = jnp.min(jnp.where(ccw_qtr_vals <= target_qtr_val, search_offsets, sentinel))
            qtr_cw_bins = jnp.min(jnp.where(cw_qtr_vals <= target_qtr_val, search_offsets, sentinel))
            qtr_ccw_bins = jnp.where(qtr_ccw_bins == sentinel, 0, qtr_ccw_bins)
            qtr_cw_bins = jnp.where(qtr_cw_bins == sentinel, 0, qtr_cw_bins)

            valid_ccw = (half_ccw_bins > 0) & (qtr_ccw_bins > half_ccw_bins)
            valid_cw = (half_cw_bins > 0) & (qtr_cw_bins > half_cw_bins)
            ratio_ccw = jnp.where(valid_ccw,
                                qtr_ccw_bins.astype(tc.dtype) / half_ccw_bins.astype(tc.dtype),
                                2.0)
            ratio_cw = jnp.where(valid_cw,
                                qtr_cw_bins.astype(tc.dtype) / half_cw_bins.astype(tc.dtype),
                                2.0)
            exponent_ccw = jnp.log(2.0) / jnp.log(ratio_ccw)
            exponent_cw = jnp.log(2.0) / jnp.log(ratio_cw)
            exponent_sum = jnp.where(valid_ccw, exponent_ccw, 0.0) + jnp.where(valid_cw, exponent_cw, 0.0)
            exponent_count = valid_ccw.astype(tc.dtype) + valid_cw.astype(tc.dtype)
            exponent = jnp.where(exponent_count > 0, exponent_sum / exponent_count, def_exp)
            exponent = jnp.clip(exponent, min_exp, max_exp)

            use_default_shape = amp < min_amp_thresh
            width_ccw = jnp.where(use_default_shape, def_w, width_ccw)
            width_cw = jnp.where(use_default_shape, def_w, width_cw)
            exponent = jnp.where(use_default_shape, def_exp, exponent)
            return amp, width_ccw, width_cw, exponent

        prev_vals = jnp.roll(tuning_curve, 1)
        next_vals = jnp.roll(tuning_curve, -1)
        is_local_max = (tuning_curve >= prev_vals) & (tuning_curve >= next_vals)
        local_maxima_vals = jnp.where(is_local_max, tuning_curve, -jnp.inf)
        sorted_indices = jnp.argsort(local_maxima_vals)[::-1]
        sorted_values = local_maxima_vals[sorted_indices]
        has_primary = jnp.any(is_local_max)

        default_theta_pref_1 = jnp.asarray(0.0, dtype=tuning_curve.dtype)
        default_amplitude_1 = jnp.asarray(default_width_value, dtype=tuning_curve.dtype)
        default_theta_pref_2 = jnp.asarray(jnp.pi, dtype=tuning_curve.dtype)
        default_amplitude_2 = jnp.asarray(0.0, dtype=tuning_curve.dtype)
        default_width = jnp.asarray(default_width_value, dtype=tuning_curve.dtype)
        default_exponent = jnp.asarray(default_exponent_value, dtype=tuning_curve.dtype)

        peak_1_idx = sorted_indices[0]
        peak_1_val = sorted_values[0]
        amplitude_1_calc, width_ccw_1_calc, width_cw_1_calc, exponent_1_calc = _get_peak_params_simple_jax(
            peak_1_val, peak_1_idx, baseline_est, tuning_curve, n_bins, angle_step,
            min_model_width, default_width_value, min_exponent, max_exponent,
            default_exponent_value, min_peak_amplitude,
        )
        theta_pref_1_calc = peak_1_idx.astype(tuning_curve.dtype) * angle_step

        theta_pref_1 = jnp.where(has_primary, theta_pref_1_calc, default_theta_pref_1)
        amplitude_1 = jnp.where(has_primary, amplitude_1_calc, default_amplitude_1)
        width_ccw_1 = jnp.where(has_primary, width_ccw_1_calc, default_width)
        width_cw_1 = jnp.where(has_primary, width_cw_1_calc, default_width)
        exponent_1 = jnp.where(has_primary, exponent_1_calc, default_exponent)

        secondary_indices = sorted_indices[1:]
        secondary_values = sorted_values[1:]
        secondary_amplitudes = secondary_values - baseline_est
        secondary_thetas = secondary_indices.astype(tuning_curve.dtype) * angle_step
        secondary_separation = jnp.abs(jnp.arctan2(jnp.sin(theta_pref_1 - secondary_thetas),
                                                jnp.cos(theta_pref_1 - secondary_thetas)))
        valid_secondary = (
            jnp.isfinite(secondary_values)
            & (secondary_amplitudes >= (amplitude_1 * min_second_peak_ratio))
            & (secondary_separation >= min_second_peak_separation)
        )
        has_secondary = has_primary & jnp.any(valid_secondary)
        first_secondary_pos = jnp.argmax(valid_secondary.astype(jnp.int32))
        peak_2_idx = secondary_indices[first_secondary_pos]
        peak_2_val = secondary_values[first_secondary_pos]

        amplitude_2_calc, width_ccw_2_calc, width_cw_2_calc, exponent_2_calc = _get_peak_params_simple_jax(
            peak_2_val, peak_2_idx, baseline_est, tuning_curve, n_bins, angle_step,
            min_model_width, default_width_value, min_exponent, max_exponent,
            default_exponent_value, min_peak_amplitude,
        )
        theta_pref_2_calc = peak_2_idx.astype(tuning_curve.dtype) * angle_step

        theta_pref_2 = jnp.where(has_secondary, theta_pref_2_calc, default_theta_pref_2)
        amplitude_2 = jnp.where(has_secondary, amplitude_2_calc, default_amplitude_2)
        width_ccw_2 = jnp.where(has_secondary, width_ccw_2_calc, default_width)
        width_cw_2 = jnp.where(has_secondary, width_cw_2_calc, default_width)
        exponent_2 = jnp.where(has_secondary, exponent_2_calc, default_exponent)

        return jnp.array([theta_pref_1, baseline_est, amplitude_1, width_ccw_1, width_cw_1,
                        exponent_1, theta_pref_2, amplitude_2, width_ccw_2, width_cw_2, exponent_2])

    def loss_fn(Y_pred, Y_true):
        """
        Elementwise squared-error loss.
        """
        return (Y_true - Y_pred) ** 2

    # Per-sample loss function
    def loss_single_sample(params, x_i, y_i):
        y_pred = single_cell_tuning_function(x_i, *params)
        if y_i.ndim == 1:
            y_i = y_i[None, :]
        if y_pred.ndim == 1:
            y_pred = y_pred[None, :]
        sample_loss = jnp.asarray(loss_fn(y_pred, y_i))
        if sample_loss.ndim == 0:
            return sample_loss
        return jnp.mean(sample_loss)

    # TODO : remember to revert this max_iter to 2000
    def _optimize_params(params, x, y, learning_rate=1e-3, max_iter=100):
        learning_rate_local = float(learning_rate)
        opt = optax.adam(learning_rate_local, b1=0.9, b2=0.999, eps=1e-8)
        # flat_params, unflatten = ravel_pytree(params)
        opt_state = opt.init(params)

        # Vectorize over samples
        # params: pytree (batched), x: (n_samples, n_features, n_trials_x), y: (n_samples, n_targets, n_trials_y)
        # Output: (n_samples,)
        loss_total = jax.vmap(loss_single_sample, in_axes=(0, 0, 0), out_axes=0)

        # if x.size != y.size & x.shape[0] == y.shape[1]:
        #     # tile x to match y's shape - y has shpae (n_features, n_trials) whereas x has shape (n_trials,) so we need to tile x by n_features times
        x = jnp.tile(x, (y.shape[0], 1))
        loss_param = lambda params: jnp.mean(loss_total(params, x, y))
        loss_param_and_grad = jax.value_and_grad(loss_param)

        @jax.jit
        def train_step(params, opt_state):
            loss, grad = loss_param_and_grad(params)
            updates, opt_state = opt.update(grad, opt_state, params)
            params = optax.apply_updates(params, updates)
            return params, opt_state, loss

        print_every = 200
        initial_loss = loss_param(params)

        best_loss, best_params = initial_loss.copy(), params.copy()
        for step in range(1, max_iter + 1):
            # _check_timeout()
            params, opt_state, loss_val = train_step(params, opt_state)
            # _check_timeout()
            if jnp.isnan(loss_val) or jnp.isinf(loss_val) or jnp.any(jnp.isnan(params)) or jnp.any(jnp.isinf(params)):
                logging.info(f"Loss is NaN or Inf at step {step}. Stopping optimization.")
                print(f"Final loss: {loss_val:.4f} at step {step}")
                break
            if loss_val < best_loss:
                best_loss = loss_val.copy()
                best_params = params.copy()
            if step % print_every == 0:
                print(f"step {step:4d}  loss {loss_val:.4f}")
        # params = unflatten(best_params)
        print(f"params optimized. Loss: {best_loss:.4f}")
        return best_params

    # Step 1 : For every source cell, fit a peaky tuning curve 
    params_init = jax.vmap(tuning_parameter_estimator_jax, in_axes=(None, 0))(stims, spike_counts) # shape (n_source, n_params)
    params = _optimize_params(params_init, stims, spike_counts)

    return params


def load_and_process_data_jacob(
    data_path: str, 
    # ---- ALL SUBSEQUENT PARAMS MUST BE SPECIFIED IN THE CONFIG FILE ----
    random_seed: int = 42,
    train_to_test_split_ratio: float = 0.5,
    conc_threshold: float = 0.55,
) -> Dict[str, np.ndarray]:
    """
    Load and preprocess neural data and return data in the form of 
    
    Parameters
    ----------
    data_path : str
        Path to the .npy file containing neural data.
    random_seed : int
        Random seed for reproducibility of source / target cell split 
    train_to_test_split_ratio : float
        Ratio of source cells to target cells (e.g. 0.7 means 70% of cells are in the training pouplation)     

    Returns
    -------
    data dict with keys 'stimulus', 'source', and 'target'.
    'stimulus' is a 1D array of angles (n_trials,)
    'source' is a 3D array of shape (2, n_source_cells, n_trials) where sample 0 is a training population and sample 1 is the held-out population for testing 
    'target' is a 3D array of shape (2, n_target_cells, n_trials) where sample 0 is a training population and sample 1 is the held-out population for testing
    """
    # data_path = "/home/dabin/data/jacob_gratings_202507/parsed/"
    # mouse = 'BZ015'
    # date = '2025-07-03'
    mouse = 'BZ016'
    date = '2025-06-24'
    exp_nums = [2, 3, 5] if mouse == 'BZ015' else [1]

    dataset_name = f'jacob_{mouse}_{date}'

    data_dirs = []
    metadata_dirs = []

    for n_exp in exp_nums:
        spks_path = f"{data_path}/{mouse}_{date}_{n_exp}"
        stims_path = f"{data_path}/{mouse}_{date}_{n_exp}"
        spks_file = f"{spks_path}/{mouse}_{date}_{n_exp}_dspikes.npy"
        stims_file = f"{stims_path}/{date}_{n_exp}_{mouse}_Block.mat"

        data_dirs.append(spks_file)
        metadata_dirs.append(stims_file)

    data_dirs, metadata_dirs = data_dirs, metadata_dirs
    responses = []
    for data_dir in data_dirs:
        response = np.load(data_dir).T
        responses.append(response)
    angles = []
    for metadata_dir in metadata_dirs:
        mat_data = scipy.io.loadmat(metadata_dir, simplify_cells=True)
        # in the single block case the first and last angles should be removed
        if 'BZ016' in metadata_dir:
            angles.append(np.array([entry['gratingOrient'] for entry in mat_data['block']['paramsValues']])[1:-1])
        else: 
            angles.append(np.array([entry['gratingOrient'] for entry in mat_data['block']['paramsValues']]))

    # remove responses where angle = 0
    for i in range(len(responses)):
        responses[i] = responses[i][:, angles[i] != 0]
        angles[i] = angles[i][angles[i] != 0]
        angles[i] = np.deg2rad(angles[i])

    # # for each repeat, reorder angles and responses
    # for i in range(len(responses)):
    #     responses[i] = responses[i][:, np.argsort(angles[i])]
    #     angles[i] = np.sort(angles[i])

    # now turn responses into an array and replace angles with any of its entries
    response = np.array(responses) # shape (n_blocks, n_cells, n_trials)
    n_blocks = response.shape[0] # n_repeats for BZ015 Which had 3 blocks, whereas BZ016 had 1 block but every trial was still repeated 3 times randomly. 
    angles = angles[0]

    response_flat = np.transpose(response, (1, 2, 0))  # n_cells x n_trials x n_blocks
    response_flat = response_flat.reshape(response_flat.shape[0], -1)  # n_cells x (n_trials*n_blocks)
    angles_flat = np.repeat(angles, n_blocks)  # now angles is (n_trials*n_blocks)
    response, angles = response_flat, angles_flat

    n_trials = response.shape[1]

    if conc_threshold is not None:
        conc = np.abs(np.sum(np.exp(2j * angles)[np.newaxis, :] * response, axis=1) / np.sum(response, axis=1))
        good_cells = np.where(conc > conc_threshold)[0]

        print(f"Filtering for orientation selectivity. Kept {len(good_cells)} out of {response.shape[0]} cells.")
        response = response[good_cells]

    rng = np.random.default_rng(random_seed)

    # Source/Target split for trial to trial variability 
    cell_idx = rng.permutation(response.shape[0])
    n_source_cells = int(train_to_test_split_ratio * response.shape[0])
    source_cells = cell_idx[:n_source_cells]
    target_cells = cell_idx[n_source_cells:]

    if source_cells.size == 0 or target_cells.size == 0:
        raise ValueError("Train to test split ratio results in empty source or target cell population. Please adjust the ratio.")    

    half_source = source_cells.size // 2 
    half_target = target_cells.size // 2 
    train_source = source_cells[:half_source]
    # make sure we handle odd number of cells by putting the extra cell in the training set
    test_source = source_cells[-half_source:]
    train_target = target_cells[:half_target]
    test_target = target_cells[-half_target:]

    # TODO : think about whether z_scoring should happen after the cell and trial split or before? Currently it's after 
    X_train = response[train_source]
    X_test = response[test_source]
    Y_train = response[train_target]
    Y_test = response[test_target]

    # rather than zscore, just divide by the std 
    eps = 1e-12
    X_train = X_train / (X_train.std(axis=1, keepdims=True) + eps)
    X_test = X_test / (X_test.std(axis=1, keepdims=True) + eps)
    Y_train = Y_train / (Y_train.std(axis=1, keepdims=True) + eps)
    Y_test = Y_test / (Y_test.std(axis=1, keepdims=True) + eps)

    source = np.stack([X_train, X_test], axis=0)  # (2, n_source, T)
    target = np.stack([Y_train, Y_test], axis=0)  # (2, n_target, T)
    return {"stimulus" : angles, "source": source, "target": target}
