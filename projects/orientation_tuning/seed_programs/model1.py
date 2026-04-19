import numpy as np


def model(data, params):
    """
    Gaussian orientation tuning curve.

    data['stimulus'] = theta  # stimulus angle (radians), shape (n_trials,)

    params:
        theta_pref: Preferred direction.
        baseline: Baseline firing rate.
        amplitude: Maximum firing rate above baseline.
        tuning_width: Width of the tuning curve.

    Returns:
        np.ndarray: Firing rate, shape (n_trials,).
    """
    theta = data['stimulus']
    theta_pref = np.clip(params["theta_pref"], 0, 2 * np.pi)
    baseline = np.clip(params["baseline"], 0, None)
    amplitude = np.clip(params["amplitude"], 0, None)
    tuning_width = np.clip(params["tuning_width"], 0.01, None)

    dist = np.abs(np.arctan2(np.sin(theta - theta_pref), np.cos(theta - theta_pref)))
    return baseline + amplitude * np.exp(-0.5 * (dist / tuning_width) ** 2)


model.DEFAULT_PARAMS = {
    "theta_pref": 0.0,
    "baseline": 0.0,
    "amplitude": 1.0,
    "tuning_width": 1.0,
}
