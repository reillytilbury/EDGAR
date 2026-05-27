import numpy as np


def model(data, params):
    """
    Hexagonal grid-cell model with isotropic Gaussian fields.

    r(x) = baseline + amplitude * sum_{n,m} exp(-||x - c_{n,m}||^2 / (2*sigma^2))

    data keys: 'pos_x', 'pos_y'  # shape (n_trials,)
    params: lam, theta, phi_x, phi_y, baseline, amplitude, sigma
    """
    x = data['pos_x']
    y = data['pos_y']
    lam = params["lam"]
    theta = params["theta"]
    phi_x = params["phi_x"]
    phi_y = params["phi_y"]
    baseline = params["baseline"]
    amplitude = params["amplitude"]
    sigma = params["sigma"]

    c, s = np.cos(theta), np.sin(theta)
    v1x, v1y = lam * c, lam * s
    v2x = 0.5 * lam * c - 0.5 * np.sqrt(3.0) * lam * s
    v2y = 0.5 * lam * s + 0.5 * np.sqrt(3.0) * lam * c

    K_MAX = 10
    ns = np.arange(-K_MAX, K_MAX + 1)
    ms = np.arange(-K_MAX, K_MAX + 1)
    nn, mm = np.meshgrid(ns, ms, indexing="ij")

    cx = (nn * v1x + mm * v2x).reshape(-1)[:, None]
    cy = (nn * v1y + mm * v2y).reshape(-1)[:, None]

    dx = (x - phi_x)[None, :] - cx
    dy = (y - phi_y)[None, :] - cy
    bumps = np.exp(-(dx * dx + dy * dy) / (2.0 * sigma * sigma + 1e-12))

    return baseline + amplitude * np.sum(bumps, axis=0)


model.DEFAULT_PARAMS = {
    "lam": 0.6,
    "theta": 0.0,
    "phi_x": 0.0,
    "phi_y": 0.0,
    "baseline": 0.0,
    "amplitude": 1.0,
    "sigma": 0.12,
}
