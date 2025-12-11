
import numpy as np
import scipy.io as sc
import jax.numpy as jnp
import jax
from typing import Union, List, Tuple

def circular_distance_rad_np(angle1, angle2) -> np.ndarray:
    """Shortest distance between two angles (radians) on a circle.
    Args:
        angle1: First angle (radians). (float or np.ndarray)
        angle2: Second angle (radians). (float or np.ndarray)
    Returns:
        diff: absolute smalles distance between the two angles (radians). (float or np.ndarray)
    """
    diff = angle1 - angle2
    diff = np.mod(diff + np.pi, 2 * np.pi) - np.pi  # Normalize to [-pi, pi]
    return np.abs(diff)

def circular_distance_rad_jax(angle1, angle2) -> jnp.ndarray:
    """Shortest distance between two angles (radians) on a circle.
    Args:
        angle1: First angle (radians). (float or jnp.ndarray)
        angle2: Second angle (radians). (float or jnp.ndarray)
    Returns:
        diff: absolute smalles distance between the two angles (radians). (float or jnp.ndarray)
    """
    diff = angle1 - angle2
    diff = jnp.mod(diff + jnp.pi, 2 * jnp.pi) - jnp.pi  # Normalize to [-pi, pi] # Changed np to jnp
    return jnp.abs(diff) # Changed np to jnp

def extract_stimulus_related_response(data: dict, n_pcs: int = 8, z_score: bool = False, spont_mean_removal: bool = False) -> np.ndarray:
    """
    Extracts the stimulus-related response from the data. Copy pasted with small modifications from https://github.com/MouseLand/stringer-et-al-2019/blob/master/utils.py#L98
    Args:
        data (dict): The data dictionary containing the stimulus-related response and other information. Values expected to be convertible to JAX arrays.
        n_pcs (int): The number of spointaneous PCs to remove from the response.
        z_score (bool): Whether to z-score the response.
    Returns:
        stim_related_response (jnp.ndarray): The stimulus-related response matrix.
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

def unbiased_signal_fraction(R, min_repeats=2):
    """
    Compute unbiased fraction of stimulus-related variance (Sahani & Linden, 2003)
    using explicit repeats per angle (no binning).

    Parameters
    ----------
    R : array, shape (n_repeats, n_cells, n_angles)
        Neural responses: repeats × cells × unique angles.
    min_repeats : int
        Minimum repeats per stimulus (default: 2).

    Returns
    -------
    signal_fraction : array, shape (n_cells,)
        Fraction of total variance explained by the stimulus.
    components : dict
        Contains:
          - 'S2': stimulus-related variance per cell
          - 'V2': noise variance per cell
          - 'mu_angles': mean response per angle (n_cells × n_angles)
          - 'var_angles': within-angle variance (n_cells × n_angles)
    """

    n_repeats, n_cells, n_angles = R.shape
    if n_repeats < min_repeats:
        raise ValueError(f"Need at least {min_repeats} repeats per angle, got {n_repeats}.")

    # ----------------------------------------------------------------
    # Per-angle means and within-angle variances
    # ----------------------------------------------------------------
    mu_angles = np.mean(R, axis=0)        # (n_cells, n_angles)
    var_angles = np.var(R, axis=0, ddof=1)  # (n_cells, n_angles)

    # ----------------------------------------------------------------
    # Unbiased stimulus-related and noise variances
    # ----------------------------------------------------------------
    N = n_angles
    R_s = np.full(N, n_repeats, dtype=float)  # repeats per stimulus, constant

    # Global mean across all stimuli
    fbar_dot = np.mean(mu_angles, axis=1)  # (n_cells,)

    # Across-stimulus variance (stimulus-related term)
    term1 = np.mean((mu_angles - fbar_dot[:, None])**2, axis=1)

    # Bias correction term (Eq. from Sahani & Linden, 2003)
    term2 = ((N - 1) / N**2) * np.sum(var_angles / R_s[None, :], axis=1)

    S2 = term1 - term2  # unbiased stimulus-related variance
    V2 = np.sum(var_angles / R_s[None, :], axis=1) / N  # average noise variance

    # Fraction of total variance due to the stimulus
    signal_fraction = S2 / (S2 + V2)
    signal_fraction = np.clip(signal_fraction, 0, 1)

    return signal_fraction, {
        "S2": S2,
        "V2": V2,
        "mu_angles": mu_angles,
        "var_angles": var_angles,
    }

def load_data_binned(data_dir: Union[str, List[List[str]]],
              data_type: str = 'stringer',
              shuffle: bool = False,
              conc_thresh: float = 0.4, 
              activity_thresh: float = 0.0, 
              signal_fraction_thresh: float = 0.0,
              n_bins: int = 256, 
              min_repeats: int = 6) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Load and preprocess neural data from a specified directory.

    Parameters
    ----------
    data_dir : str
        Path to the .npy file containing neural data (if 'stringer' or 'ali') or [[data_paths], [metadata_paths]] (if 'jacob').
    data_type : str
        Type of data to load ('stringer' or 'jacob' or 'ali')
    shuffle : bool
        Whether to shuffle the repeats for each trial. Only relevant if we have exact repeats (i.e., Jacob's data).
    conc_thresh : float
        Concentration threshold for filtering neurons.
    activity_thresh : float
        Activity threshold for filtering neurons.
    signal_fraction_thresh : float
        Signal fraction threshold for filtering neurons.
    n_bins : int
        Number of bins for response averaging.
    min_repeats : int
        Minimum number of repeats for response averaging.

    Returns 
    -------
    response : jnp.ndarray
        Preprocessed neural response. (n_repeats, n_cells, n_bins)
    angles : jnp.ndarray
        Preprocessed stimulus angles. (n_bins,)
    """
    assert data_type in ['stringer', 'jacob', 'ali'], "data_type must be either 'stringer', 'jacob', or 'ali'"

    # load data matrix (n_cells, n_trials) and angles (n_trials,)
    if data_type == 'stringer':
        neural_data = np.load(data_dir, allow_pickle=True).item()
        response = extract_stimulus_related_response(neural_data, n_pcs=0)
        angles = neural_data['istim']
        if shuffle:
            # shuffle responses for each trial
            n_trials = angles.shape[0]
            perm = np.random.permutation(n_trials)
            response = response[:, perm]
            angles = angles[perm]

    elif data_type == 'jacob':
        assert isinstance(data_dir, list) and len(data_dir) == 2, "For 'jacob' data_type, data_dir must be a list of two lists: [[data_paths], [metadata_paths]]"
        data_dirs, metadata_dirs = data_dir
        responses = []
        for data_dir in data_dirs:
            response = np.load(data_dir).T
            responses.append(response)
        angles = []
        for metadata_dir in metadata_dirs:
            mat_data = sc.io.loadmat(metadata_dir, simplify_cells=True)
            # in the single block case the first and last angles should be removed
            if 'BZ016' in metadata_dir:
                angles.append(np.array([entry['gratingOrient'] for entry in mat_data['block']['paramsValues']])[1:-1])
            else: 
                angles.append(np.array([entry['gratingOrient'] for entry in mat_data['block']['paramsValues']]))
        # remove responses where angle = 1
        for i in range(len(responses)):
            responses[i] = responses[i][:, angles[i] != 1]
            angles[i] = angles[i][angles[i] != 1]
            angles[i] = np.deg2rad(angles[i])
        # for each repeat, reorder angles and responses
        for i in range(len(responses)):
            responses[i] = responses[i][:, np.argsort(angles[i])]
            angles[i] = np.sort(angles[i])
        # now turn responses into an array and replace angles with any of its entries
        response = np.array(responses)
        n_blocks = response.shape[0]
        angles = angles[0]
        # optionally shuffle repeats for each trial
        if shuffle:
            for trial in range(len(angles)):
                perm = np.random.permutation(n_blocks)
                response[:, :, trial] = response[perm, :, trial]
        # Jacob's data included 0 as well as 2pi, so shift any angles starting with 6.2831 to 2pi - small epsilon
        angles[angles >= 6.2831] = 2 * np.pi - 1e-5
        response_flat = np.transpose(response, (1, 2, 0))  # n_cells x n_trials x n_blocks
        response_flat = response_flat.reshape(response_flat.shape[0], -1)  # n_cells x (n_trials*n_blocks)
        angles_flat = np.repeat(angles, n_blocks)  # now angles is (n_trials*n_blocks)
        response, angles = response_flat, angles_flat
    
    else:  # 'ali' data
        # with open(data_dir, 'rb') as f:
        #     neural_data = pickle.load(f)
        # response = neural_data['resps'].T    # shape (n_cells, n_trials)
        # angles = neural_data['stims'].astype(float) 
        # angles = angles % 360  # ensure angles are in [0, 360)
        # angles = np.deg2rad(angles)  # convert to radians
        # angles = angles % (2 * np.pi)  # ensure angles are in [0, 2pi)
        # # set each cell's 1th percentile response to 0
        # response = response - np.percentile(response, 1, axis=1, keepdims=True)
        # response[response < 0] = 0
        angles = np.load(data_dir[0])
        angles = np.deg2rad(angles)  # convert to radians
        response = np.load(data_dir[1])
        response = response.mean(axis=-1)
        response = response.T  # shape (n_cells, n_trials)
        # optionally shuffle responses for each trial
        if shuffle:
            n_trials = angles.shape[0]
            perm = np.random.permutation(n_trials)
            response = response[:, perm]
            angles = angles[perm]

    # Activity, concentration filtering
    active = (response > 0).astype(np.float32)
    firing_probs = np.mean(active, axis=1)
    conc = np.abs(np.sum(np.exp(2j * angles)[np.newaxis, :] * response, axis=1) / np.sum(response, axis=1))
    good_cells = np.where((firing_probs > activity_thresh) & (conc > conc_thresh))[0]
    n_good_cells = len(good_cells)
    print(f"Selected {n_good_cells} / {response.shape[0]} cells with activity > {activity_thresh} and concentration > {conc_thresh}.")

    # Keep only good cells
    conc = conc[good_cells]
    firing_probs = firing_probs[good_cells]
    response = response[good_cells, :]

    # bin responses
    bin_edges = np.linspace(0, 2 * np.pi, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    # digitize angles
    bin_indices = np.digitize(angles, bin_edges) - 1
    # raise error if any bin has fewer than min_repeats
    min_bin_counts = np.min(np.bincount(bin_indices, minlength=n_bins))
    if min_bin_counts < min_repeats:
        raise ValueError(f"Not enough repeats in some bins. Minimum repeats: {min_bin_counts}, required: {min_repeats}")
    response_binned = np.zeros((min_repeats, n_good_cells, n_bins))
    for b in range(n_bins):
        relevant_indices = np.where(bin_indices == b)[0]
        n_responses = len(relevant_indices)
        pool_size = n_responses // min_repeats
        # take average of each pool
        mean_responses = []
        for r in range(min_repeats):
            # this choice of pool indices mixes blocks
            # pool_indices = relevant_indices[r * pool_size:(r + 1) * pool_size]
            # this choice of indices keeps blocks separate
            pool_indices = relevant_indices[r::min_repeats][:pool_size]
            mean_responses.append(np.mean(response[:, pool_indices], axis=1))
        response_binned[:, :, b] = np.array(mean_responses)
    angles = bin_centers
    response = response_binned

    # Convert to JAX arrays
    response, angles = jnp.asarray(response), jnp.asarray(angles)

    # Normalize responses so that for each cell and repeat, the RMS across bins is 1
    activity_norms = jnp.linalg.norm(response, axis=-1)
    normalization_factors = activity_norms / jnp.sqrt(n_bins)
    response = response / normalization_factors[:, :, None]
    # compute signal fraction for each cell
    signal_fraction = unbiased_signal_fraction(np.array(response))[0]
    reliable_cells = jnp.where(signal_fraction > signal_fraction_thresh)[0]
    n_reliable_cells = len(reliable_cells)
    print(f"Selected {n_reliable_cells} / {n_good_cells} cells with signal fraction > {signal_fraction_thresh}.")
    # keep only reliable cells
    response = response[:, reliable_cells, :]
    return response, angles

def load_data(data_path: str = '/home/reilly/Downloads/8279387/gratings_drifting_GT1_2019_04_12_1.npy', 
              conc_thresh: float = 0.55, activity_thresh: float = 0.4):
    """
    Loads and preprocesses neural data from the specified file path.
    Args:
        data_path (str): Path to the neural data file.
        conc_thresh (float): Concentration threshold for filtering cells.
        activity_thresh (float): Activity threshold for filtering cells.
    Returns:
        tuple: Processed response and angles as JAX arrays, split into training and testing sets.
    """
    # load and preprocess data
    neural_data = np.load(data_path, allow_pickle=True)
    neural_data = neural_data.item()
    response = extract_stimulus_related_response(neural_data, n_pcs=0)
    angles = neural_data['istim']
    n_trials = response.shape[1]
    n_trials_small = int(n_trials * activity_thresh)

    # filter 
    active = (response > 0).astype(np.float32)
    firing_probs = np.mean(active, axis=1)
    conc = np.abs(np.sum(np.exp(2j * angles)[np.newaxis, :] * response, axis=1) / np.sum(response, axis=1))
    good_cells = np.where((firing_probs > activity_thresh) & (conc > conc_thresh))[0]
    n_good_cells = len(good_cells)

    # update angles and response to be (n_cells_small, n_trials_small) and (n_cells_small, n_trials_small)
    response_cropped, angles_cropped = np.zeros((len(good_cells), n_trials_small)), np.zeros((len(good_cells), n_trials_small))
    for i, cell in enumerate(good_cells):
        active_trials = response[cell] > 0
        active_trials_idx = np.where(active_trials)[0][:n_trials_small]
        response_cropped[i] = response[cell, active_trials_idx]
        angles_cropped[i] = angles[active_trials_idx]
        
    # update response and angles to be the cropped versions and convert to JAX arrays, normalize and split into train/test
    response, angles = jnp.asarray(response_cropped), jnp.asarray(angles_cropped)
    response = 100 * response / jnp.linalg.norm(response, axis=1, keepdims=True)  # normalize response
    key = jax.random.PRNGKey(42)
    training_size = n_good_cells // 2
    shuffled_indices = jax.random.permutation(key, jnp.arange(n_good_cells))
    training_cells, test_cells = shuffled_indices[:training_size], shuffled_indices[training_size:]
    response_train, response_test = response[training_cells, :], response[test_cells, :]
    angles_train, angles_test = angles[training_cells, :], angles[test_cells, :]
    print(f"Selected {len(good_cells)} cells with activity > {activity_thresh} and concentration > {conc_thresh}.")
    print(f"Using {len(training_cells)} cells for training and {len(test_cells)} cells for testing.")
    return response_train, response_test, angles_train, angles_test

# -------------
# SEED PROGRAMS
# -------------

def neuron_model_gauss(theta, theta_pref=0.0, baseline=0.0, amplitude=1.0, tuning_width=1.0):
    """
    A simple neuron model that computes the response based on a Gaussian tuning curve.
    Args:
        theta (np.ndarray): The angle in radians.
        theta_pref (float): Preferred direction of the neuron.
        baseline (float): Baseline firing rate.
        amplitude (float): Maximum firing rate above baseline.
        tuning_width (float): Width of the tuning curve.
    Returns:
        np.ndarray: The firing rate of the neuron at angle theta.
    """
    theta_pref = np.clip(theta_pref, 0, 2 * np.pi)
    baseline = np.clip(baseline, 0, None)
    amplitude = np.clip(amplitude, 0, None)
    tuning_width = np.clip(tuning_width, 0.01, None)

    circ_dist_rad = lambda theta1, theta2: np.abs(np.arctan2(np.sin(theta1 - theta2), np.cos(theta1 - theta2)))
    dist = circ_dist_rad(theta, theta_pref)
    return baseline + amplitude * np.exp(-0.5 * (dist / tuning_width) ** 2)

def neuron_model_gauss_jax(theta, theta_pref=0.0, baseline=0.0, amplitude=1.0, tuning_width=1.0):
    theta_pref = jnp.clip(theta_pref, 0, 2 * jnp.pi)
    baseline = jnp.clip(baseline, 0, None)
    amplitude = jnp.clip(amplitude, 0, None)
    tuning_width = jnp.clip(tuning_width, 0.01, None)
    circ_dist_rad = lambda theta1, theta2: jnp.abs(jnp.arctan2(jnp.sin(theta1 - theta2), jnp.cos(theta1 - theta2)))
    dist = circ_dist_rad(theta, theta_pref)
    return baseline + amplitude * jnp.exp(-0.5 * (dist / tuning_width) ** 2)

def parameter_estimator_gauss(theta, spike_counts):
    """
    Estimates the parameters of the gaussian neuron model. We do this by creating a binned tuning curve and picking out salient features.
    Args:
        theta (np.ndarray): Angles in radians.
        spike_counts (np.ndarray): Spike counts corresponding to each angle.
    Returns:
        np.ndarray: Estimated parameters [theta_pref, baseline, amplitude, tuning_width].
    """
    n_bins = 20
    bin_idx = ((theta * n_bins) / (2 * np.pi)).astype(np.int32)
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)
    sums = np.bincount(bin_idx, weights=spike_counts, minlength=n_bins)
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
    return np.array([theta_pref, baseline, amplitude, tuning_width])

def neuron_model_double_gauss(theta, theta_pref=0.0, baseline=0.0, amplitude_1=1.0, amplitude_2=0.0, tuning_width=1.0):
    """
    A neuron model that computes the response based on a double peaked gaussian tuning curve, with peaks at theta_pref and (theta_pref + pi) % 2pi.
    Args:
        theta (np.ndarray): Input angles in radians.
        theta_pref (float): Preferred angle in radians.
        baseline (float): Baseline firing rate.
        amplitude_1 (float): Amplitude of the first peak.
        amplitude_2_ratio (float): Ratio of the second peak's amplitude to the first peak's amplitude.
        tuning_width (float): Width of the tuning curves around preferred angles.
    Returns:
        np.ndarray: The response of the neuron model.

    """
    theta_pref = np.clip(theta_pref, 0, 2 * np.pi)
    baseline = np.clip(baseline, 0, None)
    amplitude_1 = np.clip(amplitude_1, 0, None)
    amplitude_2 = np.clip(amplitude_2, 0, None)
    tuning_width = np.clip(tuning_width, 0.01, None)
    
    circ_dist_rad = lambda theta1, theta2: np.abs(np.arctan2(np.sin(theta1 - theta2), np.cos(theta1 - theta2)))
    dist_1 = circ_dist_rad(theta, theta_pref)
    dist_2 = circ_dist_rad(theta, (theta_pref + np.pi) % (2 * np.pi))
    return baseline + amplitude_1 * np.exp(-0.5 * (dist_1 / tuning_width) ** 2) + amplitude_2 * np.exp(-0.5 * (dist_2 / tuning_width) ** 2)

def neuron_model_double_gauss_jax(theta, theta_pref=0.0, baseline=0.0, amplitude_1=1.0, amplitude_2=0.0, tuning_width=1.0):
    theta_pref = jnp.clip(theta_pref, 0, 2 * jnp.pi)
    baseline = jnp.clip(baseline, 0, None)
    amplitude_1 = jnp.clip(amplitude_1, 0, None)
    amplitude_2 = jnp.clip(amplitude_2, 0, None)
    tuning_width = jnp.clip(tuning_width, 0.01, None)
    
    circ_dist_rad = lambda theta1, theta2: jnp.abs(jnp.arctan2(jnp.sin(theta1 - theta2), jnp.cos(theta1 - theta2)))
    dist_1 = circ_dist_rad(theta, theta_pref)
    dist_2 = circ_dist_rad(theta, (theta_pref + jnp.pi) % (2 * jnp.pi))
    return baseline + amplitude_1 * jnp.exp(-0.5 * (dist_1 / tuning_width) ** 2) + amplitude_2 * jnp.exp(-0.5 * (dist_2 / tuning_width) ** 2)

def parameter_estimator_double_gauss(theta, spike_counts):
    """
    A parameter estimator for the double peaked neuron model. Creates a binned tuning curve from spike counts and estimates parameters using features from the tuning curve.
    Args:
        theta (np.ndarray): Input angles in radians. (n_trials,)
        spike_counts (np.ndarray): Spike counts corresponding to the angles. (n_trials,)
    Returns:
        np.ndarray: Estimated parameters [theta_pref, baseline, amplitude_1, amplitude_2_ratio, tuning_width].
    """
    n_bins = 50
    bin_idx = ((theta * n_bins) / (2 * np.pi)).astype(np.int32)
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)
    sums = np.bincount(bin_idx, weights=spike_counts, minlength=n_bins)
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
    return np.array([theta_pref, baseline, amplitude_1, amplitude_2, tuning_width])

def neuron_model_von_mises(theta, theta_pref=0.0, baseline=0.0, amplitude=1.0, tuning_width=1.0):
    """
    A neuron model that computes the response based on a von Mises tuning curve.
    Args:
        theta (np.ndarray): Input angles in radians.
        theta_pref (float): Preferred angle in radians.
        baseline (float): Baseline firing rate.
        amplitude (float): Maximum firing rate above baseline.
        tuning_width (float): Concentration parameter of the von Mises distribution.
    Returns:
        np.ndarray: The firing rate of the neuron at angle theta.
    """
    return baseline + amplitude * np.exp(tuning_width * (np.cos(theta - theta_pref) - 1))

def neuron_model_von_mises_jax(theta, theta_pref=0.0, baseline=0.0, amplitude=1.0, kappa=1.0):
    """
    A JAX implementation of the neuron model that computes the response based on a von Mises tuning curve.
    Args:
        theta (jnp.ndarray): Input angles in radians.
        theta_pref (float): Preferred angle in radians.
        baseline (float): Baseline firing rate.
        amplitude (float): Maximum firing rate above baseline.
        kappa (float): Concentration parameter of the von Mises distribution.
    Returns:
        jnp.ndarray: The firing rate of the neuron at angle theta.
    """
    return baseline + amplitude * jnp.exp(kappa * (jnp.cos(theta - theta_pref) - 1))

def parameter_estimator_von_mises(theta, spike_counts):
    """
    A parameter estimator for the von Mises neuron model. Creates a binned tuning curve from spike counts and estimates parameters using features from the tuning curve.
    Args:
        theta (np.ndarray): Input angles in radians. (n_trials,)
        spike_counts (np.ndarray): Spike counts corresponding to the angles. (n_trials,)
    Returns:
        np.ndarray: Estimated parameters [theta_pref, baseline, amplitude, kappa].
    """
    n_bins = 50
    bin_idx = ((theta * n_bins) / (2 * np.pi)).astype(np.int32)
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)
    sums = np.bincount(bin_idx, weights=spike_counts, minlength=n_bins)
    counts = np.bincount(bin_idx, minlength=n_bins)
    tuning_curve = np.zeros(n_bins, dtype=np.float32)
    tuning_curve[counts > 0] = sums[counts > 0] / counts[counts > 0]
    pref_idx = np.argmax(tuning_curve)
    theta_pref = pref_idx * (2 * np.pi / n_bins)
    baseline = np.min(tuning_curve)
    amplitude = np.max(tuning_curve) - baseline
    kappa = 1.0 / (np.std(theta) + 1e-8)  # Simple estimate based on standard deviation
    return np.array([theta_pref, baseline, amplitude, kappa])

def neuron_model_double_von_mises(theta, theta_pref=0.0, baseline=0.0, amplitude1=1.0, amplitude2=0.0, kappa_1=1.0, kappa_2=1.0):
    """
    A neuron model that computes the response based on a double peaked von Mises tuning curve.
    Args:
        theta (np.ndarray): Input angles in radians.
        theta_pref (float): Preferred angle in radians.
        baseline (float): Baseline firing rate.
        amplitude1 (float): Amplitude of the first peak.
        amplitude2 (float): Amplitude of the second peak.
        kappa_1 (float): Concentration parameter of the first peak.
        kappa_2 (float): Concentration parameter of the second peak.
    Returns:
        np.ndarray: The firing rate of the neuron at angle theta.
    """
    rate = (baseline +
            amplitude1 * np.exp(kappa_1 * (np.cos(theta - theta_pref) - 1)) +
            amplitude2 * np.exp(kappa_2 * (np.cos(theta - (theta_pref + np.pi)) - 1)))
    return rate

def neuron_model_double_von_mises_jax(theta, theta_pref=0.0, baseline=0.0, amplitude1=1.0, amplitude2=0.0, kappa_1=1.0, kappa_2=1.0):
    """
    A JAX implementation of the neuron model that computes the response based on a double peaked von Mises tuning curve.
    Args:
        theta (jnp.ndarray): Input angles in radians.
        theta_pref (float): Preferred angle in radians.
        baseline (float): Baseline firing rate.
        amplitude1 (float): Amplitude of the first peak.
        amplitude2 (float): Amplitude of the second peak.
        kappa_1 (float): Concentration parameter of the first peak.
        kappa_2 (float): Concentration parameter of the second peak.
    Returns:
        jnp.ndarray: The firing rate of the neuron at angle theta.
    """
    kappa_1 = jnp.clip(kappa_1, 1e-8, None)  # Avoid division by zero
    kappa_2 = jnp.clip(kappa_2, 1e-8, None)  # Avoid division by zero
    # f(theta) = baseline + amplitude1 * exp(kappa_1 * (cos(theta - theta_pref) - 1)) + amplitude2 * exp(kappa_2 * (cos(theta - (theta_pref + pi)) - 1))
    rate = (baseline +
            amplitude1 * jnp.exp(kappa_1 * (jnp.cos(theta - theta_pref) - 1)) +
            amplitude2 * jnp.exp(kappa_2 * (jnp.cos(theta - (theta_pref + jnp.pi)) - 1)))
    return rate

def parameter_estimator_double_von_mises(theta, spike_counts):
    """
    A parameter estimator for the double peaked von Mises neuron model based on sample stats.
    Args:
        theta (np.ndarray): Input angles in radians. (n_trials,)
        spike_counts (np.ndarray): Spike counts corresponding to the angles. (n_trials,)
    Returns:
        np.ndarray: Estimated parameters [theta_pref, baseline, amplitude1, amplitude2, kappa1, kappa2].
    """
    # f(theta) = baseline + amplitude1 * exp(kappa * (cos(theta - theta_pref) - 1)) + amplitude2 * exp(kappa * (cos(theta - (theta_pref + pi)) - 1))
    n_bins = 50
    bin_idx = ((theta * n_bins) / (2 * np.pi)).astype(np.int32)
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)
    sums = np.bincount(bin_idx, weights=spike_counts, minlength=n_bins)
    counts = np.bincount(bin_idx, minlength=n_bins)
    tuning_curve = np.zeros(n_bins, dtype=np.float32)
    tuning_curve[counts > 0] = sums[counts > 0] / counts[counts > 0]
    pref_idx = np.argmax(tuning_curve)
    theta_pref = pref_idx * (2 * np.pi / n_bins)
    baseline = np.min(tuning_curve)
    amplitude1 = np.max(tuning_curve) - baseline
    amplitude2 = tuning_curve[(pref_idx + n_bins // 2) % n_bins] - baseline
    # estimate kappa by seeing how qucikly tuning curve goes from max to halfmax
    half_max = baseline + amplitude1 / 2.0
    indices = (np.arange(-5, 6) + pref_idx) % n_bins
    above_half_max = tuning_curve[indices] >= half_max
    full_width_half_max = 2 * np.pi * np.sum(above_half_max)
    tuning_width = full_width_half_max / (2.0 * np.sqrt(2 * np.log(2)))
    kappa1 = 1.0 / (tuning_width + 1e-8)  # Simple estimate based on tuning width
    # same for kappa2, but using the second peak
    half_max2 = baseline + amplitude2 / 2.0
    indices2 = (np.arange(-5, 6) + (pref_idx + n_bins // 2)) % n_bins
    above_half_max2 = tuning_curve[indices2] >= half_max2
    full_width_half_max2 = 2 * np.pi * np.sum(above_half_max2)
    tuning_width2 = full_width_half_max2 / (2.0 * np.sqrt(2 * np.log(2)))
    kappa2 = 1.0 / (tuning_width2 + 1e-8)  # Simple estimate based on tuning width
    return np.array([theta_pref, baseline, amplitude1, amplitude2, kappa1, kappa2])

def neuron_model_trivial(theta, baseline=0.0):
    """
    A trivial neuron model that returns a constant firing rate.
    Args:
        theta (np.ndarray): Input angles in radians
        baseline (float): Baseline firing rate.
    Returns:
        np.ndarray: The firing rate of the neuron, which is constant.
    """
    return baseline

def neuron_model_trivial_jax(theta, baseline=0.0):
    """
    A trivial neuron model that returns a constant firing rate.
    Args:
        theta (jnp.ndarray): Input angles in radians
        baseline (float): Baseline firing rate.
    Returns:
        jnp.ndarray: The firing rate of the neuron, which is constant.
    """
    return baseline * jnp.ones_like(theta)

def parameter_estimator_trivial(theta, spike_counts):
    """
    Parameter estimator for a trivial neuron model that estimates only the baseline firing rate.
    Args:
        theta (np.ndarray): Input angles in radians.
        spike_counts (np.ndarray): Spike counts corresponding to the angles.
    Returns:
        np.ndarray: Estimated parameters [baseline].
    """
    baseline = np.mean(spike_counts)
    return np.array([baseline])

def neuron_model_delta(theta, theta_pref=0.0, baseline=0.0, amplitude=1.0):
    """
    A neuron model that returns a delta function response at the preferred angle.
    Args:
        theta (np.ndarray): Input angles in radians.
        theta_pref (float): Preferred angle in radians.
        baseline (float): Baseline firing rate.
        amplitude (float): Amplitude of the response at the preferred angle.
    Returns:
        np.ndarray: The firing rate of the neuron, which is zero everywhere except at the preferred angle.
    """
    return baseline + amplitude * (theta == theta_pref).astype(np.float32)

def neuron_model_delta_jax(theta, theta_pref=0.0, baseline=0.0, amplitude=1.0):
    """
    A JAX implementation of the neuron model that returns a delta function response at the preferred angle.
    Args:
        theta (jnp.ndarray): Input angles in radians.
        theta_pref (float): Preferred angle in radians.
        baseline (float): Baseline firing rate.
        amplitude (float): Amplitude of the response at the preferred angle.
    Returns:
        jnp.ndarray: The firing rate of the neuron, which is zero everywhere except at the preferred angle.
    """
    return baseline + amplitude * (theta == theta_pref).astype(jnp.float32)

def parameter_estimator_delta(theta, spike_counts):
    """
    Parameter estimator for the delta neuron model that estimates the preferred angle, baseline, and amplitude.
    Args:
        theta (np.ndarray): Input angles in radians.
        spike_counts (np.ndarray): Spike counts corresponding to the angles.
    Returns:
        np.ndarray: Estimated parameters [theta_pref, baseline, amplitude].
    """
    max_idx = np.argmax(spike_counts)
    theta_pref = theta[max_idx]
    baseline = np.mean(spike_counts)
    amplitude = spike_counts[max_idx] - baseline
    return np.array([theta_pref, baseline, amplitude])