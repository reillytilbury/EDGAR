import numpy as np


def parameter_estimator(data):
    """
    Estimate double-peaked Gaussian x Naka-Rushton parameters. Direction parameters come
    from the smoothed binned curve at the highest contrast. c50 comes from the ratio of the
    preferred-bin amplitude at the lowest vs highest contrast, assuming n = 2.

    Args:
        data (dict): Keys 'stimulus' (radians), 'contrast' (fraction), 'response',
                     each shape (n_trials,).

    Returns:
        dict: {"theta_pref", "baseline", "amplitude_1", "amplitude_2", "tuning_width", "c50", "n"}
    """
    theta, c, Y = data['stimulus'], data['contrast'], data['response']
    n_bins = 40
    bin_idx = np.clip(((theta * n_bins) / (2 * np.pi)).astype(np.int32), 0, n_bins - 1)
    x = np.arange(-6, 7)
    k = np.exp(-0.5 * (x / 2.0) ** 2)
    k = k / np.sum(k)

    def curve(mask):
        sums = np.pad(np.bincount(bin_idx[mask], weights=Y[mask], minlength=n_bins), 6, mode='wrap')
        counts = np.pad(np.bincount(bin_idx[mask], minlength=n_bins), 6, mode='wrap').astype(float)
        return np.convolve(sums, k, mode='valid') / (np.convolve(counts, k, mode='valid') + 1e-8)

    c_hi, c_lo = np.max(c), np.min(c)
    tc_hi, tc_lo = curve(c == c_hi), curve(c == c_lo)
    pref_idx = np.argmax(tc_hi)
    baseline = np.min(tc_hi)
    a_hi = max(tc_hi[pref_idx] - baseline, 1e-3)
    a_lo = tc_lo[pref_idx] - baseline

    lo2, hi2 = c_lo ** 2, c_hi ** 2
    r = np.clip(a_lo / a_hi, lo2 / hi2 + 0.01, 0.99)
    s = lo2 * hi2 * (1 - r) / (r * hi2 - lo2)
    scale = (hi2 + s) / hi2

    half_max = baseline + a_hi / 2.0
    indices = (np.arange(-5, 6) + pref_idx) % n_bins
    fwhm = 2 * np.pi * np.sum(tc_hi[indices] >= half_max) / n_bins
    a2_hi = max(tc_hi[(pref_idx + n_bins // 2) % n_bins] - baseline, 0.0)
    return {
        "theta_pref": float(pref_idx * (2 * np.pi / n_bins)),
        "baseline": float(baseline),
        "amplitude_1": float(a_hi * scale),
        "amplitude_2": float(a2_hi * scale),
        "tuning_width": float(fwhm / (2.0 * np.sqrt(2 * np.log(2)))),
        "c50": float(np.sqrt(s)),
        "n": 2.0,
    }
