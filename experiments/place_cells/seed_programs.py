"""
Seed programs for place cell experiments.

Place cells fire in localized spatial fields in the hippocampus.
We provide two simple parametric models:
1) Isotropic 2D Gaussian place field
2) Elliptical (rotated) 2D Gaussian place field
"""

import numpy as np


def place_model_1(
    X,
    x0=0.0,
    y0=0.0,
    sigma=0.25,
    amplitude=1.0,
    baseline=0.0,
):
    """
    Independent variable:
    X = [x, y]  # position (normalized to [-1, 1])

    Isotropic 2D Gaussian place field with equation
    f(x, y) = baseline + amplitude * exp(-0.5 * ((x - x0)^2 + (y - y0)^2) / sigma^2)

    Args:
        X (np.ndarray): Input array with shape (2, n_trials).
                        X[0] is x position, X[1] is y position (normalized to [-1, 1]).
        x0, y0 (float): Place field center.
        sigma (float): Field width (same for x and y).
        amplitude (float): Peak firing rate above baseline.
        baseline (float): Baseline firing rate.

    Returns:
        np.ndarray: Predicted firing rate, shape (n_trials,).
    """
    x = X[0]
    y = X[1]

    x0 = np.clip(x0, -1.0, 1.0)
    y0 = np.clip(y0, -1.0, 1.0)
    sigma = np.clip(sigma, 0.05, 1.0)
    amplitude = np.clip(amplitude, 0.0, 50.0)
    baseline = np.clip(baseline, 0.0, 20.0)

    dx = x - x0
    dy = y - y0
    dist2 = dx * dx + dy * dy
    return baseline + amplitude * np.exp(-0.5 * dist2 / (sigma ** 2))




def parameter_estimator_1(X, firing_rates):
    """
    Estimate parameters for the isotropic Gaussian place field.

    Strategy:
    - Baseline from low percentile of firing rates
    - Center from weighted mean of positions
    - Width from weighted variance
    - Amplitude from max minus baseline
    """
    x = X[0]
    y = X[1]
    firing_rates = np.asarray(firing_rates)

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

    return np.array([x0, y0, sigma, amplitude, baseline])


def place_model_2(
    X,
    x0=0.0,
    y0=0.0,
    sigma_x=0.3,
    sigma_y=0.2,
    theta=0.0,
    amplitude=1.0,
    baseline=0.0,
):
    """
    Independent variable:
    X = [x, y]  # position (normalized to [-1, 1])

    Elliptical (rotated) 2D Gaussian place field with equation
    f(x, y) = baseline + amplitude * exp(-0.5 * (xr^2 / sigma_x^2 + yr^2 / sigma_y^2))
    where
    xr = cos(theta) * (x - x0) + sin(theta) * (y - y0)
    yr = -sin(theta) * (x - x0) + cos(theta) * (y - y0)

    Args:
        X (np.ndarray): Input array with shape (2, n_trials).
                        X[0] is x position, X[1] is y position (normalized to [-1, 1]).
        x0, y0 (float): Place field center.
        sigma_x, sigma_y (float): Field widths along principal axes.
        theta (float): Rotation angle (radians).
        amplitude (float): Peak firing rate above baseline.
        baseline (float): Baseline firing rate.

    Returns:
        np.ndarray: Predicted firing rate, shape (n_trials,).
    """
    x = X[0]
    y = X[1]

    x0 = np.clip(x0, -1.0, 1.0)
    y0 = np.clip(y0, -1.0, 1.0)
    sigma_x = np.clip(sigma_x, 0.05, 1.0)
    sigma_y = np.clip(sigma_y, 0.05, 1.0)
    theta = np.clip(theta, 0.0, np.pi)
    amplitude = np.clip(amplitude, 0.0, 50.0)
    baseline = np.clip(baseline, 0.0, 20.0)

    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    dx = x - x0
    dy = y - y0

    xr = cos_t * dx + sin_t * dy
    yr = -sin_t * dx + cos_t * dy

    dist2 = (xr * xr) / (sigma_x ** 2) + (yr * yr) / (sigma_y ** 2)
    return baseline + amplitude * np.exp(-0.5 * dist2)




def parameter_estimator_2(X, firing_rates):
    """
    Estimate parameters for the elliptical Gaussian place field.

    Strategy:
    - Baseline from low percentile of firing rates
    - Center from weighted mean
    - Covariance from weighted positions to infer sigma_x, sigma_y, theta
    - Amplitude from max minus baseline
    """
    x = X[0]
    y = X[1]
    firing_rates = np.asarray(firing_rates)

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

    return np.array([x0, y0, sigma_x, sigma_y, theta, amplitude, baseline])
