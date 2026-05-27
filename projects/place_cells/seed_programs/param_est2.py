import numpy as np


def parameter_estimator(data):
    """
    Estimate elliptical Gaussian place field parameters using weighted covariance.

    Args:
        data (dict): Keys 'pos_x', 'pos_y', 'response' — each shape (n_trials,).

    Returns:
        dict: {"x0", "y0", "sigma_x", "sigma_y", "theta", "amplitude", "baseline"}
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
    cov_xx = np.sum(weights * dx * dx) / wsum
    cov_yy = np.sum(weights * dy * dy) / wsum
    cov_xy = np.sum(weights * dx * dy) / wsum

    cov = np.array([[cov_xx, cov_xy], [cov_xy, cov_yy]], dtype=float)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    sigma_x = np.sqrt(np.clip(eigvals[0], 1e-6, None))
    sigma_y = np.sqrt(np.clip(eigvals[1], 1e-6, None))
    theta = np.arctan2(eigvecs[1, 0], eigvecs[0, 0])

    amplitude = np.max(firing_rates) - baseline
    return {
        "x0": float(x0),
        "y0": float(y0),
        "sigma_x": float(sigma_x),
        "sigma_y": float(sigma_y),
        "theta": float(theta),
        "amplitude": float(amplitude),
        "baseline": float(baseline),
    }
