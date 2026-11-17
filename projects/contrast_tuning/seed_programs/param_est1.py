import numpy as np


def parameter_estimator(data):
    """
    Estimate Gaussian x Naka-Rushton parameters. Direction parameters come from the binned
    tuning curve at the highest contrast. c50 comes from the ratio of the preferred-bin
    amplitude at the lowest vs highest contrast, assuming n = 2.

    Args:
        data (dict): Keys 'stimulus' (radians), 'contrast' (fraction), 'response',
                     each shape (n_trials,).

    Returns:
        dict: {"theta_pref", "baseline", "amplitude", "tuning_width", "c50", "n"}
    """
    theta, c, Y = data['stimulus'], data['contrast'], data['response']
    n_bins = 20
    bin_idx = np.clip(((theta * n_bins) / (2 * np.pi)).astype(np.int32), 0, n_bins - 1)

    def curve(mask):
        sums = np.bincount(bin_idx[mask], weights=Y[mask], minlength=n_bins)
        counts = np.bincount(bin_idx[mask], minlength=n_bins)
        return sums / np.maximum(counts, 1)

    c_hi, c_lo = np.max(c), np.min(c)
    tc_hi, tc_lo = curve(c == c_hi), curve(c == c_lo)
    pref_idx = np.argmax(tc_hi)
    baseline = np.min(tc_hi)
    a_hi = max(tc_hi[pref_idx] - baseline, 1e-3)
    a_lo = tc_lo[pref_idx] - baseline

    # Naka-Rushton with n=2: a_lo/a_hi = c_lo^2 (c_hi^2 + s) / (c_hi^2 (c_lo^2 + s)), s = c50^2
    lo2, hi2 = c_lo ** 2, c_hi ** 2
    r = np.clip(a_lo / a_hi, lo2 / hi2 + 0.01, 0.99)
    s = lo2 * hi2 * (1 - r) / (r * hi2 - lo2)
    amplitude = a_hi * (hi2 + s) / hi2

    half_max = baseline + a_hi / 2.0
    indices = (np.arange(-5, 6) + pref_idx) % n_bins
    fwhm = 2 * np.pi * np.sum(tc_hi[indices] >= half_max) / n_bins
    return {
        "theta_pref": float(pref_idx * (2 * np.pi / n_bins)),
        "baseline": float(baseline),
        "amplitude": float(amplitude),
        "tuning_width": float(fwhm / (2.0 * np.sqrt(2 * np.log(2)))),
        "c50": float(np.sqrt(s)),
        "n": 2.0,
    }
