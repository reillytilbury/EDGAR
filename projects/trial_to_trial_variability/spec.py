"""
Welcome to the Model Discovery Engine! Fill in the components below to start building your model.

NECESSARY COMPONENTS:

Loading:
- load_and_process_data(data_path, *preprocess_params) -> [X, Y]
- train_test_split(X) -> [train_samples, train_trials]

Seed Programs:
- model_v1(X, params) and param_est_v1(X, Y)
- model_v2(X, params) and param_est_v2(X, Y)
- params is a dict of named arrays/scalars (same keys for model + estimator)

LOSS FUNCTION:
- loss_fn(Y_pred, Y_true) -> loss values

OPTIONAL COMPONENTS:
- plot_model_fits(X, Y, model_list, params_list)
"""
import numpy as np
import scipy
from typing import Tuple
from src.data_structures import Inputs, Outputs


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
) -> Tuple[Inputs, Outputs]:
    """
    Load and preprocess neural data and return data in the form of 
    Inputs and Outputs objects. 
    
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
    ) -> Tuple[Inputs, Outputs]:
    """
    Load and preprocess neural data and return data in the form of 
    Inputs and Outputs objects. 
    
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
    # load and preprocess data
    neural_data = np.load(data_path, allow_pickle=True)
    neural_data = neural_data.item()
    response = np.asarray(neural_data['sresp'])

    angles = neural_data['istim'] # shape (n_trials)
    assert max(angles) <= 2 * np.pi, "Expected angles to be in radians and between 0 and 2pi"
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

def train_test_split(
    X: Dict[str, np.ndarray],
    # -- ALL SUBSEQUENT PARAMS MUST BE SPECIFIED IN THE CONFIG FILE ---
    random_seed: int = 42,
    expected_number_of_repeats = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return train sample indices and train trial indices.
    """    
    n_samples = utils.data_n_samples(X)
    n_trials = utils.data_n_trials(X)
    assert n_samples == 2, "Expected exactly 2 samples for train/test split."

    angles = X['stimulus']

    # For BZ015 this angle had 6 repeats for some reason. Remove the extra 3 repeats 
    # repeated_angle = 0.017453292519943295
    if expected_number_of_repeats is not None:
        assert (len(unique_angles) + 1) * 3 == n_trials, f"Expected each unique angle to have 3 repeated trials. But got {len(unique_angles)} unique angles and {n_trials} total trials."

    if expected_number_of_repeats == 3 :
        train_trials = []
        for angle in unique_angles:
            angle_trials = np.where(angles == angle)[0]
            # select the first 2 trials for training and the last trial for testing
            train_trials.extend(angle_trials[:2])
    else: 
        # randomly select 0.5 of the trials for training and 0.5 for testing
        rng = np.random.default_rng(random_seed)
        train_trials = rng.choice(n_trials, size=n_trials // 2, replace=False)

    return np.array([0]), train_trials

# ========================
# 2. SEED MODELS
# ========================

def model_v1(X, params):
    """ Gain Modulation + per cell modulation

    Equation : For each target cell c at timepoint t with stimulus angle theta, 
        f(theta, t; cell_params) = multiplicative_gain(t) * g(theta(t) ; cell_params) + additive_offset(t) * coupling_factor
    where g(theta(t); cell_params) is some tuning function. 

    Args : 
        X (dict) : Inputs object with keys 'stimulus', neural responses with shape (n_source_cells + 1 , n_time). You can get the neural responses only by removing the 'stimulus' feature from X. 
        params (dict) : Parameter dictionary with keys: 
            - source_tuning_params : parameters for the tuning function g(theta) (shape (n_source_cells, n_params))
            - source_coupling_factor : coupling_factor (shape (n_source_cells,)) 
            - target_tuning_params : parameters for the tuning function g(theta) (shape (n_target_cells, n_params))
            - target_coupling_factor : coupling_factor (shape (n_target_cells,)) 

    Returns : 
        jnp.ndarray : Predicted responses for the target cells with shape (n_target_cells, n_time)
    """
    stimuli = X['stimulus'] # shape (n_time)
    source_response = X['source'] # shape (n_source, n_time)

    source_tuning_params = params['source_tuning_params'] # shape (n_source_cells, n_params)
    source_coupling_factor = params['source_coupling_factor'] # shape (n_source_cells,)
    target_tuning_params = params['target_tuning_params'] # shape (n_target_cells, n_params)
    target_coupling_factor = params['target_coupling_factor'] # shape (n_target_cells,)    

    source_stimuli = jnp.tile(stimuli[None, :], (source_tuning_params.shape[0], 1)) # shape (n_source, n_time)
    g_source = jax.vmap(lambda stim, params: neuron_model(stim, *params), in_axes=(0, 0))(source_stimuli, source_tuning_params) # shape (n_source, n_time)

    eps = 1e-8
    multiplicative_gain = jnp.sum(g_source * source_response, axis=0) / (jnp.sum(g_source**2, axis=0) + eps) # shape (n_time,)

    source_residual = source_response.T - (multiplicative_gain[:, None] * g_source.T) # shape (n_time, n_source)
    # we already know the source_coupling_factor
    additive_offset = jnp.sum(source_residual * source_coupling_factor[None, :], axis=1) / (jnp.sum(source_coupling_factor**2) + eps) # shape (n_time,)

    target_stimuli = jnp.tile(stimuli[None, :], (target_tuning_params.shape[0], 1)) # shape (n_target, n_time)
    g_target = jax.vmap(lambda stim, params: neuron_model(stim, *params), in_axes=(0, 0))(target_stimuli, target_tuning_params).T # shape (n_time, n_target)
    pred = multiplicative_gain[:, None] * g_target + additive_offset[:, None] * target_coupling_factor[None, :]
    # clip to non-negative firing rates
    pred = jnp.clip(pred, a_min=0.0)
    return pred.T

def param_est_v1(X, Y):
    """ Parameter estimator for model_v1. This function estimates params and from the data.
    Tuning_params : contain per cell level parameters for both source and target cells that are independent of trials

    Args :
        X (jnp.ndarray) : Input array of Training responses of source cells with shape (n_source_cells + 1 , n_time), including an angle feature. You can get the neural responses only by removing the 'stimulus' feature from X.
        Y (jnp.ndarray) : Output array of Training responses of target cells with shape (n_target_cells + 1, n_time), including an angle feature. You can get the neural responses only by removing the 'stimulus' target from Y.
        neuron_model (function) : Tuning function to fit for each cell. Should take in stimulus and cell-specific parameters and output predicted response.
        parameter_estimator_jax (function) : Function to estimate tuning parameters for each cell given the stimulus and responses. Should return parameters in the correct order for the neuron_model. Should be JAX-compatible for use in gradient-based optimization.
    
    Returns :
        params (dict) : Estimated tuning parameters for target cells. Keys:
            - source_tuning_params : shape (n_source_cells, n_params)
            - source_cell_coupling_factor : shape (n_source_cells,)
            - target_tuning_params : shape (n_target_cells, n_params)
            - target_cell_coupling_factor : shape (n_target_cells,)
    """
    # first sort the response of x and y by the stimulus angles 
    stims = X['stimulus'][0] # shape (n_time,)
    x = jnp.array(X.remove_feature('stimulus')) # shape (n_source, n_time)
    y = jnp.array(Y.remove_target('stimulus')) # shape (n_target, n_time)

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

    # Step 1 : For every source cell, fit a peaky tuning curve 
    source_params_init = jax.vmap(parameter_estimator_jax, in_axes=(None, 0))(bin_centers, x_binned_spike_rate) # shape (n_source, n_params)
    # use gradient descent to optimise the tuning parameters 
    source_tuning_params = _optimize_params(source_params_init, stims, x) 

    target_params_init = jax.vmap(parameter_estimator_jax, in_axes=(None, 0))(bin_centers, y_binned_spike_rate) # shape (n_target, n_params)
    target_tuning_params = _optimize_params(target_params_init, stims, y) 

    # Step 2 : Fit the gain factor using leastsq 
    source_stimuli = jnp.tile(stims[None, :], (source_tuning_params.shape[0], 1)) # shape (n_source, n_time)
    g_source = jax.vmap(lambda stim, params: neuron_model(stim, *params), in_axes=(0, 0))(source_stimuli, source_tuning_params) # shape (n_source, n_time)

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
    g_target = jax.vmap(lambda stim, params: neuron_model(stim, *params), in_axes=(0, 0))(target_stims, target_tuning_params) # shape (n_target, n_time)

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

def model_v2(X, params):
    """ Gain Modulation + source to target coupling 

    Equation : For each target cell c at timepoint t with stimulus angle theta, 
        f(theta, t; cell_params) = multiplicative_gain(t) * g(theta(t) ; cell_params) + source_cell_response(t) * coupling_weight
    where g(theta(t); cell_params) is some tuning function, source_cell_response(t) is the response of the source cell at time t (shape n_source,) and the coupling weight is the coupling factor for each target cell (shape n_source,).

    Args : 
        X (dict) : Inputs object with keys 'stimulus', neural responses with shape (n_source_cells + 1 , n_time). You can get the neural responses only by removing the 'stimulus' feature from X. 
        params (dict) : Parameter dictionary with keys: 
            - source_tuning_params : parameters for the tuning function g(theta) (shape (n_source_cells, n_params))
            - target_tuning_params : parameters for the tuning function g(theta) (shape (n_target_cells, n_params))
            - coupling_factor : coupling_factor (shape (n_target_cells, n_source_cells)) 

    Returns : 
        jnp.ndarray : Predicted responses for the target cells with shape (n_target_cells, n_time)
    """
    stimuli = jnp.array(X['stimulus']) # shape (n_time)
    source_response = jnp.array(X.remove_feature('stimulus')) # shape (n_source, n_time)

    source_tuning_params = params['source_tuning_params'] # shape (n_source_cells, n_params)
    target_tuning_params = params['target_tuning_params'] # shape (n_target_cells, n_params)
    coupling_factor = params['target_coupling_factor'] # shape (n_target_cells, n_source_cells)    

    source_stimuli = jnp.tile(stimuli[None, :], (source_tuning_params.shape[0], 1)) # shape (n_source, n_time)
    g_source = jax.vmap(lambda stim, params: neuron_model(stim, *params), in_axes=(0, 0))(source_stimuli, source_tuning_params) # shape (n_source, n_time)

    eps = 1e-8
    multiplicative_gain = jnp.sum(g_source * source_response, axis=0) / (jnp.sum(g_source**2, axis=0) + eps) # shape (n_time,)

    target_stimuli = jnp.tile(stimuli[None, :], (target_tuning_params.shape[0], 1)) # shape (n_target, n_time)
    g_target = jax.vmap(lambda stim, params: neuron_model(stim, *params), in_axes=(0, 0))(target_stimuli, target_tuning_params).T # shape (n_time, n_target)
    pred = multiplicative_gain[:, None] * g_target + (coupling_factor @ source_response).T # shape (n_time, n_target)
    # clip to non-negative firing rates
    pred = jnp.clip(pred, a_min=0.0)
    return pred.T

def param_est_v2(X, Y):
    """ Parameter estimator for model_v1. This function estimates params and from the data.
    Tuning_params : contain per cell level parameters for both source and target cells that are independent of trials

    Args :
        stims (jnp.ndarray) : Stimulus angles with shape (n_time,)
        x (jnp.ndarray) : Training responses of source cells with shape (n_source_cells, n_time)
        y (jnp.ndarray) : Training responses of target cells with shape (n_target_cells, n_time)
        neuron_model (function) : Tuning function to fit for each cell. Should take in stimulus and cell-specific parameters and output predicted response.
        parameter_estimator_jax (function) : Function to estimate tuning parameters for each cell given the stimulus and responses. Should return parameters in the correct order for the neuron_model. Should be JAX-compatible for use in gradient-based optimization.
    
    Returns :
        params (dict) : Estimated tuning parameters for target cells. Keys:
            - source_tuning_params : shape (n_source_cells, n_params)
            - target_tuning_params : shape (n_target_cells, n_params)
            - coupling_factor : shape (n_target_cells, n_source_cells)
    """
    # first sort the response of x and y by the stimulus angles 
    stims = X['stimulus'][0] # shape (n_time,)
    x = jnp.array(X.remove_feature('stimulus')) # shape (n_source, n_time)
    y = jnp.array(Y.remove_target('stimulus')) # shape (n_target, n_time)

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

    # Step 1 : For every source cell, fit a peaky tuning curve 
    source_params_init = jax.vmap(parameter_estimator_jax, in_axes=(None, 0))(bin_centers, x_binned_spike_rate) # shape (n_source, n_params)
    # use gradient descent to optimise the tuning parameters 
    source_tuning_params = _optimize_params(source_params_init, stims, x) 

    target_params_init = jax.vmap(parameter_estimator_jax, in_axes=(None, 0))(bin_centers, y_binned_spike_rate) # shape (n_target, n_params)
    target_tuning_params = _optimize_params(target_params_init, stims, y) 

    # Step 2 : Fit the gain factor using leastsq 
    source_stimuli = jnp.tile(stims[None, :], (source_tuning_params.shape[0], 1)) # shape (n_source, n_time)
    g_source = jax.vmap(lambda stim, params: neuron_model(stim, *params), in_axes=(0, 0))(source_stimuli, source_tuning_params) # shape (n_source, n_time)

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

def loss_fn(Y_pred, Y_true):
    """
    Elementwise squared-error loss.
    """
    return (Y_true - Y_pred) ** 2


# ========================
# 4. DIAGNOSTICS
# ========================

def plot_model_fits(X, Y, programs_list, save_path=""):
    raise NotImplementedError


# ========================
# 4. OPTIONAL PROJECT-SPECIFIC HELPERS
# ========================
