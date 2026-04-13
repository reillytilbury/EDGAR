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
import scipy
from typing import Dict
import src.utils as utils

# ========================
# 1. DATA
# ========================

def zscore_rows(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    z-score each row of a 2D array
    """
    mean = X.mean(axis=1, keepdims=True)
    std = X.std(axis=1, keepdims=True)    
    return (X - mean) / (std + eps)

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

def load_and_process_data(
    data_path: str,
    # ---- ALL SUBSEQUENT PARAMS MUST BE SPECIFIED IN THE CONFIG FILE ----
    random_seed: int = 42,
    train_to_test_split_ratio: float = 0.5,
    conc_threshold: float = 0.55,
) -> list[list[Dict[str, np.ndarray]]]:
    """
    Load and preprocess neural data, split into train/test samples and trials,
    and return a 2x2 container of data dicts.

    The sample split divides cells into two independent halves (for
    source and target separately). The trial split randomly assigns half
    the trials for training and half for testing.

    Parameters
    ----------
    data_path : str
        Path to the .npy file containing neural data.
    random_seed : int
        Random seed for reproducibility of cell and trial splits.
    train_to_test_split_ratio : float
        Fraction of cells assigned to source vs target (e.g. 0.5 means
        equal source and target populations).
    conc_threshold : float
        Orientation-selectivity threshold for cell filtering.

    Returns
    -------
    2x2 list of dicts:
        [[data_train_train, data_train_test],
         [data_test_train, data_test_test]]
        Each dict has keys 'stimulus', 'source', 'target'.
        'stimulus' has shape (n_trials,), 'source' has shape
        (n_source_cells, n_trials), 'target' has shape
        (n_target_cells, n_trials).
    """
    neural_data = np.load(data_path, allow_pickle=True)
    neural_data = neural_data.item()
    response = np.asarray(neural_data['sresp'])

    angles = neural_data['istim']
    assert max(angles) <= 2 * np.pi, "Expected angles to be in radians and between 0 and 2pi"
    n_trials = response.shape[1]

    if conc_threshold is not None:
        conc = np.abs(np.sum(np.exp(2j * angles)[np.newaxis, :] * response, axis=1) / np.sum(response, axis=1))
        good_cells = np.where(conc > conc_threshold)[0]
        print(f"Filtering for orientation selectivity. Kept {len(good_cells)} out of {response.shape[0]} cells.")
        response = response[good_cells]

    rng = np.random.default_rng(random_seed)

    # --- Cell split: source vs target ---
    cell_idx = rng.permutation(response.shape[0])
    n_source_cells = int(train_to_test_split_ratio * response.shape[0])
    source_cells = cell_idx[:n_source_cells]
    target_cells = cell_idx[n_source_cells:]

    if source_cells.size == 0 or target_cells.size == 0:
        raise ValueError("Train to test split ratio results in empty source or target cell population. Please adjust the ratio.")

    # --- Sample split: each population halved into train/test samples ---
    half_source = source_cells.size // 2
    half_target = target_cells.size // 2
    train_source = source_cells[:half_source]
    test_source = source_cells[-half_source:]
    train_target = target_cells[:half_target]
    test_target = target_cells[-half_target:]

    # --- Trial split: random half for training, rest for testing ---
    train_trials = np.sort(rng.choice(n_trials, size=n_trials // 2, replace=False))
    test_trials = np.setdiff1d(np.arange(n_trials), train_trials)

    # Normalise per cell (divide by std)
    eps = 1e-12
    source_train = response[train_source]
    source_test = response[test_source]
    target_train = response[train_target]
    target_test = response[test_target]
    source_train = source_train / (source_train.std(axis=1, keepdims=True) + eps)
    source_test = source_test / (source_test.std(axis=1, keepdims=True) + eps)
    target_train = target_train / (target_train.std(axis=1, keepdims=True) + eps)
    target_test = target_test / (target_test.std(axis=1, keepdims=True) + eps)

    def _make_data_dict(source, target, trial_idx):
        return {
            'stimulus': angles[trial_idx],
            'source': source[:, trial_idx],
            'target': target[:, trial_idx],
        }

    return [
        [_make_data_dict(source_train, target_train, train_trials),
         _make_data_dict(source_train, target_train, test_trials)],
        [_make_data_dict(source_test, target_test, train_trials),
         _make_data_dict(source_test, target_test, test_trials)],
    ]

# ========================
# 2. SEED MODELS
# ========================

def model_v1(data, params):
    """ Gain Modulation + per cell modulation

    Equation : For each target cell c at timepoint t with stimulus angle theta, 
        f(theta, t; cell_params) = multiplicative_gain(t) * g(theta(t) ; cell_params) + additive_offset(t) * coupling_factor
    where g(theta(t); cell_params) is some tuning function. 

    Args : 
        data (dict) : Inputs object with keys 'stimulus', neural responses with shape (n_source_cells + 1 , n_time). You can get the neural responses only by removing the 'stimulus' feature from X. 
        params (dict) : Parameter dictionary with keys: 
            - source_tuning_params : parameters for the tuning function g(theta) (shape (n_source_cells, n_params))
            - source_coupling_factor : coupling_factor (shape (n_source_cells,)) 
            - target_tuning_params : parameters for the tuning function g(theta) (shape (n_target_cells, n_params))
            - target_coupling_factor : coupling_factor (shape (n_target_cells,)) 

    Returns : 
        jnp.ndarray : Predicted responses for the target cells with shape (n_target_cells, n_time)
    """
    stimuli = data['stimulus'] # shape (n_time)
    source_response = data['source'] # shape (n_source, n_time)

    source_tuning_params = params['source_tuning_params'] # shape (n_source_cells, n_params)
    source_coupling_factor = params['source_coupling_factor'] # shape (n_source_cells,)
    target_tuning_params = params['target_tuning_params'] # shape (n_target_cells, n_params)
    target_coupling_factor = params['target_coupling_factor'] # shape (n_target_cells,)    

    source_stimuli = jnp.tile(stimuli[None, :], (source_tuning_params.shape[0], 1)) # shape (n_source, n_time)
    g_source = jax.vmap(lambda stim, params: single_cell_tuning_function(stim, *params), in_axes=(0, 0))(source_stimuli, source_tuning_params) # shape (n_source, n_time)

    eps = 1e-8
    multiplicative_gain = jnp.sum(g_source * source_response, axis=0) / (jnp.sum(g_source**2, axis=0) + eps) # shape (n_time,)

    source_residual = source_response.T - (multiplicative_gain[:, None] * g_source.T) # shape (n_time, n_source)
    # we already know the source_coupling_factor
    additive_offset = jnp.sum(source_residual * source_coupling_factor[None, :], axis=1) / (jnp.sum(source_coupling_factor**2) + eps) # shape (n_time,)

    target_stimuli = jnp.tile(stimuli[None, :], (target_tuning_params.shape[0], 1)) # shape (n_target, n_time)
    g_target = jax.vmap(lambda stim, params: single_cell_tuning_function(stim, *params), in_axes=(0, 0))(target_stimuli, target_tuning_params).T # shape (n_time, n_target)
    pred = multiplicative_gain[:, None] * g_target + additive_offset[:, None] * target_coupling_factor[None, :]
    # clip to non-negative firing rates
    pred = jnp.clip(pred, a_min=0.0)
    return pred.T

def param_est_v1(data):
    """ Parameter estimator for model_v1. This function estimates params and from the data.
    Tuning_params : contain per cell level parameters for both source and target cells that are independent of trials

    Args :
        data (dict) : Dictionary containing input and output arrays. Keys:
            - 'stimulus' : shape (n_time,)
            - 'source' : shape (n_source_cells, n_time)
            - 'target' : shape (n_target_cells, n_time)
    
    Returns :
        params (dict) : Estimated tuning parameters for target cells. Keys:
            - source_tuning_params : shape (n_source_cells, n_params)
            - source_cell_coupling_factor : shape (n_source_cells,)
            - target_tuning_params : shape (n_target_cells, n_params)
            - target_cell_coupling_factor : shape (n_target_cells,)
    """
    # first sort the response of x and y by the stimulus angles 
    stims = data['stimulus'] # shape (n_time,)
    x = jnp.array(data['source']) # shape (n_source, n_time)
    y = jnp.array(data['target']) # shape (n_target, n_time)

    stims_idx = jnp.argsort(stims)
    stims = stims[stims_idx]
    x = x[:, stims_idx]
    y = y[:, stims_idx]

    # first calculate the tuning parameters from the "mean responses" which is the binned mean response of angle bins. 
    n_bins = 256
    bin_edges = jnp.linspace(0, 2 * jnp.pi, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_idx = jnp.digitize(stims, bin_edges) - 1
    bin_idx = jnp.clip(bin_idx, 0, n_bins - 1)
    
    x_binned_spike_counts = jnp.zeros((x.shape[0], n_bins)).at[:, bin_idx].add(x)
    x_binned_counts = jnp.zeros(n_bins).at[bin_idx].add(1.0)
    x_binned_spike_rate = jnp.zeros_like(x_binned_spike_counts).at[:, :].set(jnp.where(x_binned_counts > 0, x_binned_spike_counts / x_binned_counts, 0.0))

    y_binned_spike_counts = jnp.zeros((y.shape[0], n_bins)).at[:, bin_idx].add(y)
    y_binned_counts = jnp.zeros(n_bins).at[bin_idx].add(1.0)
    y_binned_spike_rate = jnp.zeros_like(y_binned_spike_counts).at[:, :].set(jnp.where(y_binned_counts > 0, y_binned_spike_counts / y_binned_counts, 0.0))

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

    def tuning_parameter_estimator_jax(theta, spike_counts):
        n_bins = 75

        # Bin angles into [0, n_bins-1]
        bin_idx = ((theta * n_bins) / (2 * jnp.pi)).astype(jnp.int32)
        bin_idx = jnp.clip(bin_idx, 0, n_bins - 1)

        # Weighted and unweighted bin counts
        sums = jnp.bincount(bin_idx, weights=spike_counts, length=n_bins)
        counts = jnp.bincount(bin_idx, length=n_bins)

        # Gaussian smoothing kernel
        k = jnp.exp(-0.5 * (jnp.arange(-5, 6) ** 2))
        k = k / jnp.sum(k)

        # Circular padding
        sums_padded = jnp.pad(sums, (5, 5), mode="wrap")
        counts_padded = jnp.pad(counts, (5, 5), mode="wrap")

        # Smoothed tuning curve
        smoothed_sums = jnp.convolve(sums_padded, k, mode="valid")
        smoothed_counts = jnp.convolve(counts_padded, k, mode="valid")
        tuning_curve = smoothed_sums / (smoothed_counts + 1e-8)

        baseline = jnp.min(tuning_curve)

        peak_1_idx = jnp.argmax(tuning_curve)
        theta_pref = peak_1_idx * (2 * jnp.pi / n_bins)

        amplitude_1 = tuning_curve[peak_1_idx] - baseline

        anti_pref_idx = (peak_1_idx + n_bins // 2) % n_bins
        amplitude_2 = jnp.maximum(0.0, tuning_curve[anti_pref_idx] - baseline)

        theta_pref_2_est = anti_pref_idx * (2 * jnp.pi / n_bins)
        angle_offset_2 = theta_pref_2_est - ((theta_pref + jnp.pi) % (2 * jnp.pi))

        tuning_width_1_left = jnp.pi / 8
        tuning_width_1_right = jnp.pi / 8
        tuning_width_2_left = jnp.pi / 8
        tuning_width_2_right = jnp.pi / 8
        exponent_1 = 2.0
        exponent_2 = 2.0

        return jnp.array([theta_pref, baseline, amplitude_1, amplitude_2, tuning_width_1_left, tuning_width_1_right, tuning_width_2_left, tuning_width_2_right, exponent_1, exponent_2, angle_offset_2,])

    # Step 1 : For every source cell, fit a peaky tuning curve 
    source_params_init = jax.vmap(tuning_parameter_estimator_jax, in_axes=(None, 0))(bin_centers, x_binned_spike_rate) # shape (n_source, n_params)
    # use gradient descent to optimise the tuning parameters 
    source_tuning_params = _optimize_params(source_params_init, stims, y) 

    target_params_init = jax.vmap(tuning_parameter_estimator_jax, in_axes=(None, 0))(bin_centers, y_binned_spike_rate) # shape (n_target, n_params)
    target_tuning_params = _optimize_params(target_params_init, stims, y) 

    # Step 2 : Fit the gain factor using leastsq 
    source_stimuli = jnp.tile(stims[None, :], (source_tuning_params.shape[0], 1)) # shape (n_source, n_time)
    g_source = jax.vmap(lambda stim, params: single_cell_tuning_function(stim, *params), in_axes=(0, 0))(source_stimuli, source_tuning_params) # shape (n_source, n_time)

    n_source, n_t = g_source.shape

    eps = 1e-8
    multiplicative_gain = jnp.sum(g_source * x, axis=0) / (jnp.sum(g_source**2, axis=0) + eps)

    # Step 3 : Fit a rank 1 model to the residual using SVD 
    residual = x.T - (multiplicative_gain[:, None] * g_source.T) # has shape (n_time, n_source)
    U, S, Vh = jnp.linalg.svd(residual, full_matrices=False)
    source_coupling_factor = Vh[0, :] # shape (n_source,)
    additive_offset = U[:, 0] * S[0] # shape (n_time,)

    # Step 4 : Fit the target cell coupling factor using the multiplicative_gain and additive offset
    n_target = target_tuning_params.shape[0]
    target_stims = jnp.tile(stims[None, :], (n_target, 1)) # shape (n_target, n_time)
    g_target = jax.vmap(lambda stim, params: single_cell_tuning_function(stim, *params), in_axes=(0, 0))(target_stims, target_tuning_params) # shape (n_target, n_time)

    multiplicative_only_pred = multiplicative_gain[:, None] * g_target.T # shape (n_time, n_target)
    residual_target = y.T - multiplicative_only_pred # shape (n_time, n_target
    target_coupling_factor = (residual_target.T @ additive_offset) / (additive_offset @ additive_offset) # shape (n_target,)

    params = {
        'source_tuning_params' : source_tuning_params,
        'source_coupling_factor' : source_coupling_factor,
        'target_tuning_params' : target_tuning_params,
        'target_coupling_factor' : target_coupling_factor
    }
    return params

def model_v2(data, params):
    """ Gain Modulation + source to target coupling 

    Equation : For each target cell c at timepoint t with stimulus angle theta, 
        f(theta, t; cell_params) = multiplicative_gain(t) * g(theta(t) ; cell_params) + source_cell_response(t) * coupling_weight
    where g(theta(t); cell_params) is some tuning function, source_cell_response(t) is the response of the source cell at time t (shape n_source,) and the coupling weight is the coupling factor for each target cell (shape n_source,).

    Args : 
        data (dict) : Inputs object with keys 'stimulus', neural responses with shape (n_source_cells + 1 , n_time). You can get the neural responses only by removing the 'stimulus' feature from X. 
        params (dict) : Parameter dictionary with keys: 
            - source_tuning_params : parameters for the tuning function g(theta) (shape (n_source_cells, n_params))
            - target_tuning_params : parameters for the tuning function g(theta) (shape (n_target_cells, n_params))
            - coupling_factor : coupling_factor (shape (n_target_cells, n_source_cells)) 

    Returns : 
        jnp.ndarray : Predicted responses for the target cells with shape (n_target_cells, n_time)
    """
    stimuli = jnp.array(data['stimulus']) # shape (n_time)
    source_response = jnp.array(data['source']) # shape (n_source, n_time)

    source_tuning_params = params['source_tuning_params'] # shape (n_source_cells, n_params)
    target_tuning_params = params['target_tuning_params'] # shape (n_target_cells, n_params)
    coupling_factor = params['target_coupling_factor'] # shape (n_target_cells, n_source_cells)    

    source_stimuli = jnp.tile(stimuli[None, :], (source_tuning_params.shape[0], 1)) # shape (n_source, n_time)
    g_source = jax.vmap(lambda stim, params: single_cell_tuning_function(stim, *params), in_axes=(0, 0))(source_stimuli, source_tuning_params) # shape (n_source, n_time)

    eps = 1e-8
    multiplicative_gain = jnp.sum(g_source * source_response, axis=0) / (jnp.sum(g_source**2, axis=0) + eps) # shape (n_time,)

    target_stimuli = jnp.tile(stimuli[None, :], (target_tuning_params.shape[0], 1)) # shape (n_target, n_time)
    g_target = jax.vmap(lambda stim, params: single_cell_tuning_function(stim, *params), in_axes=(0, 0))(target_stimuli, target_tuning_params).T # shape (n_time, n_target)
    pred = multiplicative_gain[:, None] * g_target + (coupling_factor @ source_response).T # shape (n_time, n_target)
    # clip to non-negative firing rates
    pred = jnp.clip(pred, a_min=0.0)
    return pred.T

def param_est_v2(data):
    """ Parameter estimator for model_v1. This function estimates params and from the data.
    Tuning_params : contain per cell level parameters for both source and target cells that are independent of trials

    Args :
        data (dict) : Dictionary containing input and output arrays. Keys:
            - 'stimulus' : shape (n_time,)
            - 'source' : shape (n_source_cells, n_time)
            - 'target' : shape (n_target_cells, n_time)
    
    Returns :
        params (dict) : Estimated tuning parameters for target cells. Keys:
            - source_tuning_params : shape (n_source_cells, n_params)
            - target_tuning_params : shape (n_target_cells, n_params)
            - coupling_factor : shape (n_target_cells, n_source_cells)
    """
    # first sort the response of x and y by the stimulus angles 
    stims = data['stimulus'] # shape (n_time,)
    x = jnp.array(data['source']) # shape (n_source, n_time)
    y = jnp.array(data['target']) # shape (n_target, n_time)

    stims_idx = jnp.argsort(stims)
    stims = stims[stims_idx]
    x = x[:, stims_idx]
    y = y[:, stims_idx]

    # first calculate the tuning parameters from the "mean responses" which is the binned mean response of angle bins. 
    n_bins = 256
    bin_edges = jnp.linspace(0, 2 * jnp.pi, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_idx = jnp.digitize(stims, bin_edges) - 1
    bin_idx = jnp.clip(bin_idx, 0, n_bins - 1)
    
    x_binned_spike_counts = jnp.zeros((x.shape[0], n_bins)).at[:, bin_idx].add(x)
    x_binned_counts = jnp.zeros(n_bins).at[bin_idx].add(1.0)
    x_binned_spike_rate = jnp.zeros_like(x_binned_spike_counts).at[:, :].set(jnp.where(x_binned_counts > 0, x_binned_spike_counts / x_binned_counts, 0.0))

    y_binned_spike_counts = jnp.zeros((y.shape[0], n_bins)).at[:, bin_idx].add(y)
    y_binned_counts = jnp.zeros(n_bins).at[bin_idx].add(1.0)
    y_binned_spike_rate = jnp.zeros_like(y_binned_spike_counts).at[:, :].set(jnp.where(y_binned_counts > 0, y_binned_spike_counts / y_binned_counts, 0.0))

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

    def tuning_parameter_estimator_jax(theta, spike_counts):
        n_bins = 75

        # Bin angles into [0, n_bins-1]
        bin_idx = ((theta * n_bins) / (2 * jnp.pi)).astype(jnp.int32)
        bin_idx = jnp.clip(bin_idx, 0, n_bins - 1)

        # Weighted and unweighted bin counts
        sums = jnp.bincount(bin_idx, weights=spike_counts, length=n_bins)
        counts = jnp.bincount(bin_idx, length=n_bins)

        # Gaussian smoothing kernel
        k = jnp.exp(-0.5 * (jnp.arange(-5, 6) ** 2))
        k = k / jnp.sum(k)

        # Circular padding
        sums_padded = jnp.pad(sums, (5, 5), mode="wrap")
        counts_padded = jnp.pad(counts, (5, 5), mode="wrap")

        # Smoothed tuning curve
        smoothed_sums = jnp.convolve(sums_padded, k, mode="valid")
        smoothed_counts = jnp.convolve(counts_padded, k, mode="valid")
        tuning_curve = smoothed_sums / (smoothed_counts + 1e-8)

        baseline = jnp.min(tuning_curve)

        peak_1_idx = jnp.argmax(tuning_curve)
        theta_pref = peak_1_idx * (2 * jnp.pi / n_bins)

        amplitude_1 = tuning_curve[peak_1_idx] - baseline

        anti_pref_idx = (peak_1_idx + n_bins // 2) % n_bins
        amplitude_2 = jnp.maximum(0.0, tuning_curve[anti_pref_idx] - baseline)

        theta_pref_2_est = anti_pref_idx * (2 * jnp.pi / n_bins)
        angle_offset_2 = theta_pref_2_est - ((theta_pref + jnp.pi) % (2 * jnp.pi))

        tuning_width_1_left = jnp.pi / 8
        tuning_width_1_right = jnp.pi / 8
        tuning_width_2_left = jnp.pi / 8
        tuning_width_2_right = jnp.pi / 8
        exponent_1 = 2.0
        exponent_2 = 2.0

        return jnp.array([theta_pref, baseline, amplitude_1, amplitude_2, tuning_width_1_left, tuning_width_1_right, tuning_width_2_left, tuning_width_2_right, exponent_1, exponent_2, angle_offset_2,])

    # Step 1 : For every source cell, fit a peaky tuning curve 
    source_params_init = jax.vmap(tuning_parameter_estimator_jax, in_axes=(None, 0))(bin_centers, x_binned_spike_rate) # shape (n_source, n_params)
    # use gradient descent to optimise the tuning parameters 
    source_tuning_params = _optimize_params(source_params_init, stims, x) 

    target_params_init = jax.vmap(tuning_parameter_estimator_jax, in_axes=(None, 0))(bin_centers, y_binned_spike_rate) # shape (n_target, n_params)
    target_tuning_params = _optimize_params(target_params_init, stims, y) 

    # Step 2 : Fit the gain factor using leastsq 
    source_stimuli = jnp.tile(stims[None, :], (source_tuning_params.shape[0], 1)) # shape (n_source, n_time)
    g_source = jax.vmap(lambda stim, params: single_cell_tuning_function(stim, *params), in_axes=(0, 0))(source_stimuli, source_tuning_params) # shape (n_source, n_time)

    n_source, n_t = g_source.shape

    eps = 1e-8
    multiplicative_gain = jnp.sum(g_source * x, axis=0) / (jnp.sum(g_source**2, axis=0) + eps)

    # Step 3 : Fit the coupling factor by regressing the residual against the source cell responses
    residual = x.T - (multiplicative_gain[:, None] * g_source.T) # has shape (n_time, n_source)    
    coupling_factor = jnp.linalg.lstsq(y, residual, rcond=None)[0].T # shape (n_target, n_source)

    params = {
        'source_tuning_params' : source_tuning_params,
        'target_tuning_params' : target_tuning_params,
        'coupling_factor' : coupling_factor
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
        Predicted target-cell responses, shape (n_target_cells, n_trials).
    data : dict
        Data dictionary; the comparison target is data['target'].
    """
    Y_true = data['target']
    return (Y_true - model_output) ** 2

# ========================
# 4. DIAGNOSTICS
# ========================

def plot_model_fits(data, programs_list, eval_grid, save_path="", labels=None):
    raise NotImplementedError


# ========================
# 4. OPTIONAL PROJECT-SPECIFIC HELPERS
# ========================
