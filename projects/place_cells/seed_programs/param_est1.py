import numpy as np


def parameter_estimator(data):
    """
    Estimate isotropic Gaussian place field parameters using weighted moments.

    Args:
        data (dict): Keys 'pos_x', 'pos_y', 'response' — each shape (n_trials,).

    Returns:
        dict: {"x0", "y0", "sigma", "amplitude", "baseline"}
    """
    x = data["pos_x"]
    y = data["pos_y"]
    firing_rates = np.asarray(data["response"])

    baseline = np.percentile(firing_rates, 10)
    weights = np.clip(firing_rates - baseline, 0.0, None)
    wsum = np.sum(weights) + 1e-8

    x0 = np.sum(x * weights) / wsum
    y0 = np.sum(y * weights) / wsum

    dx = x - x0
    dy = y - y0
    var = np.sum(weights * (dx * dx + dy * dy)) / wsum
    sigma = np.sqrt(np.clip(var / 2.0, 1e-6, None))

    amplitude = np.max(firing_rates) - baseline
    return {
        "x0": float(x0),
        "y0": float(y0),
        "sigma": float(sigma),
        "amplitude": float(amplitude),
        "baseline": float(baseline),
    }
