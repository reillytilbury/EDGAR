import numpy as np

def neuron_model_1(stimuli, theta_pref=0.0, baseline=0.0, amplitude=1.0, tuning_width=1.0):
    """
    A simple neuron model that computes the response based on a Gaussian tuning curve.
    Args:
        stimuli (np.ndarray): The stimulus (e.g., angle in radians).
        theta_pref (float): Preferred direction of the neuron.
        baseline (float): Baseline firing rate.
        amplitude (float): Maximum firing rate above baseline.
        tuning_width (float): Width of the tuning curve.
    Returns:
        np.ndarray: The firing rate of the neuron at each stimulus.
    """
    theta = stimuli
    theta_pref = np.clip(theta_pref, 0, 2 * np.pi)
    baseline = np.clip(baseline, 0, None)
    amplitude = np.clip(amplitude, 0, None)
    tuning_width = np.clip(tuning_width, 0.01, None)

    circ_dist_rad = lambda theta1, theta2: np.abs(np.arctan2(np.sin(theta1 - theta2), np.cos(theta1 - theta2)))
    dist = circ_dist_rad(theta, theta_pref)
    return baseline + amplitude * np.exp(-0.5 * (dist / tuning_width) ** 2)

def parameter_estimator_1(stimuli, spike_counts):
    """
    Estimates the parameters of the gaussian neuron model. We do this by creating a binned tuning curve and picking out salient features.
    Args:
        stimuli (np.ndarray): Stimuli (e.g., angles in radians).
        spike_counts (np.ndarray): Spike counts corresponding to each angle.
    Returns:
        np.ndarray: Estimated parameters [theta_pref, baseline, amplitude, tuning_width].
    """
    n_bins = 20
    theta = np.asarray(stimuli)
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

def neuron_model_2(stimuli, theta_pref=0.0, baseline=0.0, amplitude_1=1.0, amplitude_2=0.0, tuning_width=1.0):
    """
    A neuron model that computes the response based on a double peaked gaussian tuning curve, with peaks at theta_pref and (theta_pref + pi) % 2pi.
    Args:
        stimuli (np.ndarray): Input stimuli (e.g., angles in radians).
        theta_pref (float): Preferred angle in radians.
        baseline (float): Baseline firing rate.
        amplitude_1 (float): Amplitude of the first peak.
        amplitude_2_ratio (float): Ratio of the second peak's amplitude to the first peak's amplitude.
        tuning_width (float): Width of the tuning curves around preferred angles.
    Returns:
        np.ndarray: The response of the neuron model.

    """
    theta = stimuli
    theta_pref = np.clip(theta_pref, 0, 2 * np.pi)
    baseline = np.clip(baseline, 0, None)
    amplitude_1 = np.clip(amplitude_1, 0, None)
    amplitude_2 = np.clip(amplitude_2, 0, None)
    tuning_width = np.clip(tuning_width, 0.01, None)
    
    circ_dist_rad = lambda theta1, theta2: np.abs(np.arctan2(np.sin(theta1 - theta2), np.cos(theta1 - theta2)))
    dist_1 = circ_dist_rad(theta, theta_pref)
    dist_2 = circ_dist_rad(theta, (theta_pref + np.pi) % (2 * np.pi))
    return baseline + amplitude_1 * np.exp(-0.5 * (dist_1 / tuning_width) ** 2) + amplitude_2 * np.exp(-0.5 * (dist_2 / tuning_width) ** 2)

def parameter_estimator_2(stimuli, spike_counts):
    """
    A parameter estimator for the double peaked neuron model. Creates a binned tuning curve from spike counts and estimates parameters using features from the tuning curve.
    Args:
        stimuli (np.ndarray): Input stimuli (e.g., angles in radians). (n_trials,)
        spike_counts (np.ndarray): Spike counts corresponding to the angles. (n_trials,)
    Returns:
        np.ndarray: Estimated parameters [theta_pref, baseline, amplitude_1, amplitude_2_ratio, tuning_width].
    """
    n_bins = 50
    theta = np.asarray(stimuli)
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


def neuron_model_3(stimuli, theta_pref=0.0, baseline=0.0, amplitude=1.0, tuning_width=1.0):
    """
    A neuron model that computes the response based on a von Mises tuning curve.
    Args:
        stimuli (np.ndarray): Input stimuli (e.g., angles in radians).
        theta_pref (float): Preferred angle in radians.
        baseline (float): Baseline firing rate.
        amplitude (float): Maximum firing rate above baseline.
        tuning_width (float): Concentration parameter of the von Mises distribution.
    Returns:
        np.ndarray: The firing rate of the neuron at each stimulus.
    """
    theta = stimuli
    return baseline + amplitude * np.exp(tuning_width * (np.cos(theta - theta_pref) - 1))

def parameter_estimator_3(stimuli, spike_counts):
    """
    A parameter estimator for the von Mises neuron model. Creates a binned tuning curve from spike counts and estimates parameters using features from the tuning curve.
    Args:
        stimuli (np.ndarray): Input stimuli (e.g., angles in radians). (n_trials,)
        spike_counts (np.ndarray): Spike counts corresponding to the angles. (n_trials,)
    Returns:
        np.ndarray: Estimated parameters [theta_pref, baseline, amplitude, kappa].
    """
    n_bins = 50
    theta = np.asarray(stimuli)
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

def neuron_model_4(stimuli, theta_pref=0.0, baseline=0.0, amplitude1=1.0, amplitude2=0.0, kappa_1=1.0, kappa_2=1.0):
    """
    A neuron model that computes the response based on a double peaked von Mises tuning curve.
    Args:
        stimuli (np.ndarray): Input stimuli (e.g., angles in radians).
        theta_pref (float): Preferred angle in radians.
        baseline (float): Baseline firing rate.
        amplitude1 (float): Amplitude of the first peak.
        amplitude2 (float): Amplitude of the second peak.
        kappa_1 (float): Concentration parameter of the first peak.
        kappa_2 (float): Concentration parameter of the second peak.
    Returns:
        np.ndarray: The firing rate of the neuron at each stimulus.
    """
    theta = stimuli
    rate = (baseline +
            amplitude1 * np.exp(kappa_1 * (np.cos(theta - theta_pref) - 1)) +
            amplitude2 * np.exp(kappa_2 * (np.cos(theta - (theta_pref + np.pi)) - 1)))
    return rate


def parameter_estimator_4(stimuli, spike_counts):
    """
    A parameter estimator for the double peaked von Mises neuron model based on sample stats.
    Args:
        stimuli (np.ndarray): Input stimuli (e.g., angles in radians). (n_trials,)
        spike_counts (np.ndarray): Spike counts corresponding to the angles. (n_trials,)
    Returns:
        np.ndarray: Estimated parameters [theta_pref, baseline, amplitude1, amplitude2, kappa1, kappa2].
    """
    # f(theta) = baseline + amplitude1 * exp(kappa * (cos(theta - theta_pref) - 1)) + amplitude2 * exp(kappa * (cos(theta - (theta_pref + pi)) - 1))
    n_bins = 50
    theta = np.asarray(stimuli)
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


def neuron_model_trivial(stimuli, baseline=0.0):
    """
    A trivial neuron model that returns a constant firing rate.
    Args:
        stimuli (np.ndarray): Input stimuli
        baseline (float): Baseline firing rate.
    Returns:
        np.ndarray: The firing rate of the neuron, which is constant.
    """
    stimuli = np.asarray(stimuli)
    if stimuli.ndim == 0:
        return baseline
    return baseline * np.ones(stimuli.shape[0], dtype=stimuli.dtype)

def parameter_estimator_trivial(stimuli, spike_counts):
    """
    Parameter estimator for a trivial neuron model that estimates only the baseline firing rate.
    Args:
        stimuli (np.ndarray): Input stimuli.
        spike_counts (np.ndarray): Spike counts corresponding to the angles.
    Returns:
        np.ndarray: Estimated parameters [baseline].
    """
    baseline = np.mean(spike_counts)
    return np.array([baseline])

def neuron_model_delta(stimuli, theta_pref=0.0, baseline=0.0, amplitude=1.0):
    """
    A neuron model that returns a delta function response at the preferred angle.
    Args:
        stimuli (np.ndarray): Input stimuli (e.g., angles in radians).
        theta_pref (float): Preferred angle in radians.
        baseline (float): Baseline firing rate.
        amplitude (float): Amplitude of the response at the preferred angle.
    Returns:
        np.ndarray: The firing rate of the neuron, which is zero everywhere except at the preferred angle.
    """
    theta = stimuli
    return baseline + amplitude * (theta == theta_pref).astype(np.float32)

def parameter_estimator_delta(stimuli, spike_counts):
    """
    Parameter estimator for the delta neuron model that estimates the preferred angle, baseline, and amplitude.
    Args:
        stimuli (np.ndarray): Input stimuli (e.g., angles in radians).
        spike_counts (np.ndarray): Spike counts corresponding to the angles.
    Returns:
        np.ndarray: Estimated parameters [theta_pref, baseline, amplitude].
    """
    theta = np.asarray(stimuli)
    max_idx = np.argmax(spike_counts)
    theta_pref = theta[max_idx]
    baseline = np.mean(spike_counts)
    amplitude = spike_counts[max_idx] - baseline
    return np.array([theta_pref, baseline, amplitude])
