import numpy as np


def parameter_estimator(data):
    """
    Estimate double-peaked Gaussian tuning curve parameters from a smoothed
    binned tuning curve.

    Args:
        data (dict): Keys 'stimulus' (radians, shape n_trials) and
                     'response' (spike counts, shape n_trials).

    Returns:
        dict: {"theta_pref", "baseline", "amplitude_1", "amplitude_2", "tuning_width"}
    """
    theta = data['stimulus']
    Y = data['response']
    n_bins = 50
    bin_idx = ((theta * n_bins) / (2 * np.pi)).astype(np.int32)
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)
    sums = np.bincount(bin_idx, weights=Y, minlength=n_bins)
    counts = np.bincount(bin_idx, minlength=n_bins)

    # Smooth with a Gaussian kernel (circular padding)
    sig = 2
    x = np.arange(-int(3 * sig), int(3 * sig) + 1)
    k = np.exp(-0.5 * (x / sig) ** 2)
    k = k / np.sum(k)
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
