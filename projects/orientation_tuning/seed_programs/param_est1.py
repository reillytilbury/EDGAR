import numpy as np


def parameter_estimator(data):
    """
    Estimate Gaussian tuning curve parameters from a binned tuning curve.

    Args:
        data (dict): Keys 'stimulus' (radians, shape n_trials) and
                     'response' (spike counts, shape n_trials).

    Returns:
        dict: {"theta_pref", "baseline", "amplitude", "tuning_width"}
    """
    theta = data['stimulus']
    Y = data['response']
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
