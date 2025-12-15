
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
        tuple: Processed response and angles as JAX arrays
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
    return angles, response

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

# IMAGE FEEDBACK FUNCTIONS
import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
from typing import Optional, Callable, Sequence

def plot_model_fits(programs: Sequence[dict], loss_function: Callable, 
                    x: jnp.ndarray, y: jnp.ndarray, 
                    unit_selection: Sequence[int],
                    n_eval: int = 100, n_mean: int = 50,
                    colours: list = ["#FDC91E", "#15AC15", '#EB2B2C'],
                    labels: Optional[list] = None, 
                    title: str = '',
                    line_width=4.0, 
                    line_alpha=1.0, 
                    point_alpha=0.1,
                    point_size: int = 80,
                    legend_fontsize: int = 12,
                    dpi: float = 100.0, 
                    save_path: Optional[str] = None):
    """
    plot fits of all models in the provided sequence over a subset of cells in x and y, along with the running mean.
    Args:
        programs:
            - sequence of dict-like objects with keys 'function' and 'params'. 
            - length <= 3
            - 'function': callable (written in JAX): (x: jnp.ndarray, *params) -> jnp.ndarray
            - 'params': jnp.ndarray (n_cells, n_params)
        loss_function: 
            - callable (written in JAX): (y_est: jnp.ndarray, y_true: jnp.ndarray) -> jnp.ndarray
        x: (n_cells x n_trials) - jnp.ndarray
        y: (n_cells x n_trials) - jnp.ndarray
    """
    if not programs:
        return
    assert len(programs) <= 3, f"programs must have at most 3 entries, but has {len(programs)}."
    assert len(unit_selection) > 0, "unit_selection must not be empty."
    assert len(unit_selection) == int(np.sqrt(len(unit_selection)))**2, \
        f"unit_selection must be a square number, but has {len(unit_selection)} elements."

    # define frequently used variables
    models = [entry['function'] for entry in programs]
    selection = np.array(unit_selection, dtype=int)
    params = [entry['params'][selection] for entry in programs]
    spike_matrix = y[selection]
    stimuli = x[selection]
    n_cells, n_trials = spike_matrix.shape
    n_models = len(models)
    if labels is None:
        labels = [f'model {i + 1}' for i in range(n_models)]

    # define figure and axes, ensuring ax is 2D even if n_cells == 1
    n_row_cols = int(np.sqrt(n_cells))
    fig, ax = plt.subplots(n_row_cols, n_row_cols, figsize=(20, 20))
    if n_cells == 1:
        ax = np.array([[ax]])  # Ensure ax is 2D for single plot

    # Calculate loss for each model, cell and trial
    point_losses = jnp.zeros((n_models, n_cells, n_trials))
    for i, model in enumerate(models):
        for c in range(n_cells):
            params_ic = params[i][c]
            predicted_response = model(stimuli[c], *params_ic)
            point_losses = point_losses.at[i, c].set(loss_function(predicted_response, spike_matrix[c]))
    
    # compute running mean
    x_values_mean = jnp.linspace(0, 2 * jnp.pi, n_mean, endpoint=False) + 0.5 * (2 * jnp.pi / n_mean)  # Shift to center bins
    binned_mean = jnp.zeros((n_cells, n_mean))
    for c in range(n_cells):
        bin_idx = jnp.clip(((stimuli[c] * n_mean) / (2 * jnp.pi)).astype(jnp.int32), 0, n_mean - 1)
        sums = jnp.bincount(bin_idx, weights=spike_matrix[c], minlength=n_mean)
        counts = jnp.bincount(bin_idx, minlength=n_mean)
        binned_mean = binned_mean.at[c].set((sums + 1e-6) / (counts + 1e-6))  # Avoid division by zero

    # compute cell outputs at evaluation points
    x_values_eval = jnp.linspace(0, 2 * jnp.pi, n_eval, endpoint=False)
    model_outputs = jnp.zeros((n_models, n_cells, n_eval))
    for i, model in enumerate(models):
        for c in range(n_cells):
            params_ic = params[i][c]
            model_outputs = model_outputs.at[i, c].set(model(x_values_eval, *params_ic))

    for c in range(n_cells):
        row, col = divmod(c, n_row_cols)
        # Scatter plot of data points (x=stimulus, y=response) for cell c
        ax[row, col].scatter(stimuli[c], spike_matrix[c], c='black', alpha=point_alpha, s=point_size)

        # Plot running mean for cell c
        ax[row, col].plot(x_values_mean, binned_mean[c], 
                          label='Mean', color="#3BD1FF", linewidth=line_width * 1.35)

        # Plot model fits to cell c
        for i, model in enumerate(models):
            ax[row, col].plot(x_values_eval, model_outputs[i, c], 
                              label=labels[i] + f' (loss: {jnp.mean(point_losses[i, c]):.2f})',
                              color=colours[i], 
                              alpha=line_alpha, 
                              linewidth=line_width)
        model_max = jnp.max(model_outputs[:, c])
        mean_max = jnp.max(binned_mean[c])

        # Set axis properties
        ax[row, col].set_ylim(0, max(model_max, mean_max) * 2)
        ax[row, col].set_title(f'Cell {selection[c]}', fontsize=16)
        ax[row, col].legend(loc='upper right', fontsize=legend_fontsize)
        if row == n_row_cols - 1:
            ax[row, col].set_xlabel('Theta (radians)', fontsize=20)
        if col == 0:
            ax[row, col].set_ylabel('Firing Rate', fontsize=20)

    plt.suptitle(title, fontsize=25)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path, dpi=dpi) if save_path else plt.show()
    plt.close(fig)
    
def plot_single_model_fit(model: Callable, loss_function: Callable, 
                          x: jnp.ndarray, y: jnp.ndarray, params: jnp.ndarray, 
                          n_eval: int = 100, n_mean: int = 50,
                          dpi: float = 100.0, title: str = '', 
                          save_path: Optional[str] = None):
    """
    Plots the fit of a single model to a selection of cells in x and y, along with the running mean.
    Args:
        model: callable (written in JAX): (x: jnp.ndarray, *params) -> jnp.ndarray
        loss_function: callable (written in JAX): (y_est: jnp.ndarray, y_true: jnp.ndarray) -> jnp.ndarray
        x: (n_cells x n_trials) - jnp.ndarray
        y: (n_cells x n_trials) - jnp.ndarray
        params: (n_cells x n_params) - jnp.ndarray
    """
    assert y.shape[0] == int(np.sqrt(y.shape[0]))**2, f"n_cells must be a square number, but got {y.shape[0]} cells."
    assert x.shape == y.shape, f"x and y must have the same shape, but got {x.shape} and {y.shape}."
    n_cells, n_trials = y.shape

    # Calculate loss for each cell and trial
    point_losses = jnp.zeros((n_cells, n_trials))
    for c in range(n_cells):
        params_c = params[c]
        predicted_response = model(x[c], *params_c)
        point_losses = point_losses.at[c].set(loss_function(predicted_response, y[c]))

    # compute running mean
    x_values_mean = jnp.linspace(0, 2 * jnp.pi, n_mean, endpoint=False) + 0.5 * (2 * jnp.pi / n_mean)  # Shift to center bins
    binned_mean = jnp.zeros((n_cells, n_mean))
    for c in range(n_cells):
        bin_idx = jnp.clip(((x[c] * n_mean) / (2 * jnp.pi)).astype(jnp.int32), 0, n_mean - 1)
        sums = jnp.bincount(bin_idx, weights=y[c], minlength=n_mean)
        counts = jnp.bincount(bin_idx, minlength=n_mean)
        binned_mean = binned_mean.at[c].set((sums + 1e-6) / (counts + 1e-6))  # Avoid division by zero

    # compute cell outputs at evaluation points
    x_values_eval = jnp.linspace(0, 2 * jnp.pi, n_eval, endpoint=False)
    model_output = jnp.zeros((n_cells, n_eval))
    for c in range(n_cells):
        params_c = params[c]
        model_output = model_output.at[c].set(model(x_values_eval, *params_c))

    n_row_cols = int(np.sqrt(n_cells))
    fig, ax = plt.subplots(n_row_cols, n_row_cols, figsize=(20, 20))
    if n_cells == 1:
        ax = np.array([[ax]])

    for c in range(n_cells):
        row, col = divmod(c, n_row_cols)

        # data scatter
        vmin, vmax = np.percentile(point_losses[c], [1,99])
        sc = ax[row, col].scatter(x[c], y[c], c=point_losses[c], cmap='viridis', vmin=vmin, vmax=vmax, alpha=0.5)
        plt.colorbar(sc, ax=ax[row, col], label='Loss')

        # running mean
        ax[row, col].plot(x_values_mean, binned_mean[c], label='Mean', color='cyan', linewidth=4.0)

        # model fit
        ax[row, col].plot(x_values_eval, model_output[c], label='Model', color='red', alpha=1, linewidth=3.0)

        # Set axis properties
        ax[row, col].set_ylabel('Firing Rate')
        ax[row, col].set_title(f'Cell {c}. Loss: {jnp.mean(point_losses[c]):.2f}')
        if c == 0:
            ax[row, col].legend(loc='upper right')

    # make each line of the sup title the colour of the model
    plt.suptitle(title)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    if save_path:
        plt.savefig(save_path, dpi=dpi)
    else:
        plt.show()
    plt.close(fig)

def plot_losses(loss: np.ndarray, true_model_loss: Optional[float] = None, 
                island_labels: Optional[list] = None,
                alpha: float = 0.5, dpi: float = 100.0, y_lims: Optional[tuple] = None,
                title: str = 'Learning Curve', legend_font_size: int = 6,
                save_path: Optional[str] = None):
    """
    Plot losss of arrays over iterations.
    Args:
        loss: (n_iter, n_islands) array of lists of losses for each island at each iteration.
        true_model_loss: float true model loss for simulated data.
        island_labels: (list) labels for each island. If not provided, will use default labels.
        save_path: (str) where to save the data. If not provided, will show the data but not save it.
    """

    n_iter, n_islands = loss.shape
    island_min = np.full((n_iter, n_islands), np.inf)
    for iter_id, island_id in np.ndindex(n_iter, n_islands):
        island_min[iter_id, island_id] = np.nanmin(np.array(loss[iter_id, island_id]))
    global_min = np.nanmin(island_min, axis=1)
    if island_labels is None:
        island_labels = [f'Island {i}' for i in range(n_islands)]
    
    plt.figure(figsize=(10, 5))
    cmap = plt.get_cmap('tab10')
    for iter_id, island_id in np.ndindex(n_iter, n_islands):
        y_vals = loss[iter_id, island_id]
        x_vals = np.ones(len(y_vals)) * (n_islands * iter_id + island_id)
        cmap_idx = island_id # colour by island_id
        if iter_id == 0:
            plt.scatter(x_vals, y_vals, label=island_labels[island_id] if alpha>0.0 else None,
                        alpha=alpha, color=cmap(cmap_idx))
        else:
            plt.scatter(x_vals, y_vals, alpha=alpha, color=cmap(cmap_idx))

    # plot the minimum loss for each island at each iteration
    for island_id in range(n_islands):
        plt.plot(np.arange(n_iter) * n_islands + island_id, island_min[:, island_id],
                 label=island_labels[island_id], color=cmap(island_id), linewidth=1, linestyle='--', alpha=0.25)
        
    # plot min loss across all islands at each iteration in black
    # the x axis has n_islands * n_iter points, so we need to create an array of that length
    # global min is only of length n_iter, so we need to repeat it for each island
    global_min = np.repeat(global_min[:, np.newaxis], n_islands, axis=1).reshape(-1)
    plt.plot(np.arange(n_islands * n_iter), global_min,
             label='Global min loss', color='black', linewidth=2, linestyle='-', alpha=1.0)
    
    # plot the true model loss
    if true_model_loss is not None:
        plt.axhline(y=true_model_loss, color='black', linestyle='--', alpha=0.5, label='True model loss')
    
    # put dashed verical lines at the end of each iteration
    for i in range(n_iter):
        plt.axvline(x=n_islands * i - 0.5, color='grey', linestyle='--', alpha=0.5)

    # make the plot look nice
    if y_lims is None:
        y_lims = (0.99 * np.nanmin(island_min), 1.01 * np.nanmax(island_min))
    plt.ylim(y_lims)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.xticks(np.arange(n_iter) * n_islands + n_islands / 2,
                [f'Iter {i}' for i in range(n_iter)], rotation=45)
    plt.title(title)
    plt.legend(loc='upper right', fontsize=legend_font_size)
    plt.tight_layout()

    # save or plot the figure
    if save_path:
        plt.savefig(save_path, dpi=dpi)
        # plt.show()
    else:
        plt.show()
    plt.close()

def plot_train_vs_test_loss(programs: Sequence[dict], 
                            island_labels: list,
                            save_path: Optional[str] = None):
    """
    Plot train vs test loss for each program in the provided sequence.
    Args:
        programs: Sequence of dict-like objects containing 'train_loss', 'test_loss', and 'birth_island'.
        save_path: Path to save the plot. If None, will show the plot instead.
    """
    if not programs:
        return
    
    train_loss = np.array([entry['train_loss'] for entry in programs], dtype=float)
    test_loss = np.array([entry.get('test_loss') if entry.get('test_loss') is not None else np.nan for entry in programs], dtype=float)
    birth_island = np.array([entry.get('birth_island', -1) for entry in programs], dtype=int)

    # turn nan to num
    train_loss = np.nan_to_num(train_loss, nan=np.inf)
    test_loss = np.nan_to_num(test_loss, nan=np.inf)

    # only take loss < 100
    mask = (train_loss < 100) & (test_loss < 100)
    train_loss = train_loss[mask]
    test_loss = test_loss[mask]
    birth_island = birth_island[mask]
    if train_loss.size == 0 or test_loss.size == 0:
        return
    cmap = plt.get_cmap('tab10')

    # plot the train vs test loss
    plt.figure(figsize=(10, 10))
    for island_id in np.unique(birth_island):
        island_mask = (birth_island == island_id)
        plt.scatter(train_loss[island_mask], test_loss[island_mask], 
                    label=island_labels[island_id], color=cmap(island_id), alpha=1.0)
    plt.xlabel('Train Loss')
    plt.ylabel('Test Loss')
    plt.xlim(0.9 * min(np.min(train_loss), np.min(test_loss)), 
             1.1 * max(np.median(train_loss), np.median(test_loss)))
    plt.ylim(0.9 * min(np.min(train_loss), np.min(test_loss)),
             1.1 * max(np.median(train_loss), np.median(test_loss)))
    plt.plot([0, 100], [0, 100], color='black', linestyle='--', alpha=0.5)  # diagonal line
    plt.title('Train vs Test Loss')
    plt.legend()
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()
