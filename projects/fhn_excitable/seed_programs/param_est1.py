import numpy as np


def parameter_estimator(data):
    """Persistence: log_sigma from first-difference residuals; s0 = y[0]."""
    y = np.asarray(data["y"], dtype=np.float64)
    residuals = y[1:] - y[:-1]
    sigma = max(float(residuals.std()), 1e-3)
    return {
        "log_sigma_obs": float(np.log(sigma)),
        "s0_y_last": float(y[0]),
    }
