import numpy as np


def model(data, params):
    """
    Isotropic 2D Gaussian place field.

    r(x, y) = baseline + amplitude * exp(-0.5 * d^2 / sigma^2)

    data keys: 'pos_x', 'pos_y'  # shape (n_trials,)
    params: x0, y0, sigma, amplitude, baseline
    """
    x = data["pos_x"]
    y = data["pos_y"]

    x0 = np.clip(params["x0"], -1.0, 1.0)
    y0 = np.clip(params["y0"], -1.0, 1.0)
    sigma = np.clip(params["sigma"], 0.05, 1.0)
    amplitude = np.clip(params["amplitude"], 0.0, 50.0)
    baseline = np.clip(params["baseline"], 0.0, 20.0)

    dx = x - x0
    dy = y - y0
    return baseline + amplitude * np.exp(-0.5 * (dx * dx + dy * dy) / (sigma ** 2))


model.DEFAULT_PARAMS = {
    "x0": 0.0,
    "y0": 0.0,
    "sigma": 0.25,
    "amplitude": 1.0,
    "baseline": 0.0,
}
