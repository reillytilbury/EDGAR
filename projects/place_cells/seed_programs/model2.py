import numpy as np


def model(data, params):
    """
    Elliptical rotated Gaussian place field.

    r(x, y) = baseline + amplitude * exp(-0.5 * (xr^2/sigma_x^2 + yr^2/sigma_y^2))
    where (xr, yr) are coordinates rotated by theta about (x0, y0).

    data keys: 'pos_x', 'pos_y'  # shape (n_trials,)
    params: x0, y0, sigma_x, sigma_y, theta, amplitude, baseline
    """
    x = data["pos_x"]
    y = data["pos_y"]

    x0 = np.clip(params["x0"], -1.0, 1.0)
    y0 = np.clip(params["y0"], -1.0, 1.0)
    sigma_x = np.clip(params["sigma_x"], 0.05, 1.0)
    sigma_y = np.clip(params["sigma_y"], 0.05, 1.0)
    theta = np.clip(params["theta"], 0.0, np.pi)
    amplitude = np.clip(params["amplitude"], 0.0, 50.0)
    baseline = np.clip(params["baseline"], 0.0, 20.0)

    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    dx = x - x0
    dy = y - y0
    xr = cos_t * dx + sin_t * dy
    yr = -sin_t * dx + cos_t * dy

    dist2 = (xr * xr) / (sigma_x ** 2) + (yr * yr) / (sigma_y ** 2)
    return baseline + amplitude * np.exp(-0.5 * dist2)


model.DEFAULT_PARAMS = {
    "x0": 0.0,
    "y0": 0.0,
    "sigma_x": 0.3,
    "sigma_y": 0.2,
    "theta": 0.0,
    "amplitude": 1.0,
    "baseline": 0.0,
}
